#!/usr/bin/env python3
"""Independent check of the clamped-phase ("on-device") learning rule for Potts hidden units (2026-11-20).

Small model (6x6 pixels, global Potts unit with 7 states and spring couplings Q, per-pixel Ising units, two sets of
per-patch Potts units: 4x4 every 2 px with 3 states, 3x3 every 3 px with 4 states), a stand-in visible energy, T = 0.02.

1. EXACT STATISTICS. The local rule of DESIGN.md 2.1, written from my own derivation in numpy, for the global unit
   AND the per-patch units, against jax.grad of latent_model.make_loss.
2. A DEVICE PROTOCOL. A numpy emulation that uses only what a device with clamped pixels has: the hidden state at each
   of S reads, the force on every clamp at each read and its window average, and the couplings of the state that is
   active at a read. No softmax, no probability, no derivative of a probability. Updates go to the active state only.
   Against jax.grad of make_loss(samples=S, jackknife=...) with the same draws.
3. BIAS. The mean over many keys of the plug-in, first-order and second-order jackknife gradients against the exact
   gradient, and the plug-in loss bias against -V / (2 S T^2).
4. NUDGES. [a_m(x + eps u) - a_m(x - eps u)] / (2 eps) = G_m . u for a large eps.

Usage (CPU, 4 min): sbatch jobs/run_cpu_neuronic.sh check_device_rule.py      (first run: job 4287868, 2026-11-20)
"""
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
import latent_model as lm  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

T = lm.TEMP
GRID, N, M, B = 6, 36, 7, 5
PATCHES = (("Wp", "bp", "Qp", 4, 2, 3), ("Wp2", "bp2", "Qp2", 3, 3, 4))   # keys, side, stride, states


class StubVisible:
    """E^v = sum_i (a_i x_i^2 / 2 + k_i x_i^4 / 4 - h_i x_i); theta = [a, k, h]. Only grad_lap is used by the loss."""

    def grad_lap(self, theta, X):
        a, k, h = theta[:N], theta[N:2 * N], theta[2 * N:]
        return X * a + X ** 3 * k - h, jnp.sum(a) + 3.0 * (X * X) @ k


def patch_index(G, side, stride):
    """Pixels of each patch, from the layout stated in PatchGeometry's docstring (written out independently)."""
    n_p = (G - side) // stride + 1
    idx = np.zeros((n_p * n_p, side * side), int)
    for pi in range(n_p):
        for pj in range(n_p):
            for u in range(side):
                for v in range(side):
                    idx[pi * n_p + pj, u * side + v] = (pi * stride + u) * G + pj * stride + v
    return idx


def build(seed=0):
    r = np.random.default_rng(seed)
    P = {"theta": np.concatenate([1.0 + 0.3 * r.random(N), 0.5 * r.random(N), 0.1 * r.standard_normal(N)]),
         "W": 0.03 * r.standard_normal((M, N)), "Q": 0.02 * r.standard_normal((M, N)),
         "b": 0.02 * r.standard_normal(M),
         "wl": 0.05 * r.standard_normal(N), "bl": 0.02 * r.standard_normal(N)}
    for kw, kb, kq, side, stride, S in PATCHES:
        n_p = (GRID - side) // stride + 1
        P[kw] = 0.03 * r.standard_normal((n_p * n_p, S, side * side))
        P[kq] = 0.02 * r.standard_normal((n_p * n_p, S, side * side))
        P[kb] = 0.02 * r.standard_normal((n_p * n_p, S))
    return P, 0.4 * r.standard_normal((B, N))


def units_of(P):
    """Every Potts unit as (keys, position in the patch set or None, its pixels, W, Q, b)."""
    us = [dict(keys=("W", "b", "Q"), pos=None, idx=np.arange(N), W=P["W"], Q=P["Q"], b=P["b"])]
    for kw, kb, kq, side, stride, _ in PATCHES:
        idx = patch_index(GRID, side, stride)
        for p in range(idx.shape[0]):
            us.append(dict(keys=(kw, kb, kq), pos=p, idx=idx[p], W=P[kw][p], Q=P[kq][p], b=P[kb][p]))
    return us


def zeros_like(P):
    return {k: np.zeros_like(v) for k, v in P.items()}


def add(out, u, gW, gb, gQ):
    kw, kb, kq = u["keys"]
    if u["pos"] is None:
        out[kw] += gW
        out[kb] += gb
        out[kq] += gQ
    else:
        out[kw][u["pos"]] += gW
        out[kb][u["pos"]] += gb
        out[kq][u["pos"]] += gQ


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def visible(P, x):
    a, k, h = P["theta"][:N], P["theta"][N:2 * N], P["theta"][2 * N:]
    return a * x + k * x ** 3 - h, a.sum() + 3.0 * (k * x * x).sum()


def visible_rule(x, psi):
    return np.concatenate([x * psi / T ** 2 - 1.0 / T, x ** 3 * psi / T ** 2 - 3.0 * x * x / T, -psi / T ** 2])


# ----------------------------------------------------------------------------------------------------------------
# 1. the local rule with exact statistics (formula form)
# ----------------------------------------------------------------------------------------------------------------
def potts_rule(u, x, p, psi):
    xu, pu = x[u["idx"]], psi[u["idx"]]
    G = u["W"] + 2.0 * u["Q"] * xu
    D = G - p @ G
    D2 = np.sum(D * D, axis=1)
    s = u["Q"].sum(axis=1)
    c = D2 - p @ D2 + 2.0 * T * (s - p @ s) - D @ pu
    R = 2.0 * D - pu
    return (p[:, None] * (np.outer(c, xu) / T ** 3 + R / T ** 2), p * c / T ** 3,
            p[:, None] * (np.outer(c, xu * xu) / T ** 3 + 2.0 * xu * R / T ** 2 + 2.0 / T))


def exact_rule(P, X):
    out, us, loss, vtot = zeros_like(P), units_of(P), 0.0, 0.0
    for x in X:
        psi, lap = visible(P, x)
        ps = []
        for u in us:
            xu = x[u["idx"]]
            p = softmax((u["b"] + u["W"] @ xu + u["Q"] @ (xu * xu)) / T)
            G = u["W"] + 2.0 * u["Q"] * xu
            gbar = p @ G
            V = p @ np.sum(G * G, axis=1) - gbar @ gbar
            psi[u["idx"]] -= gbar
            lap += -V / T - 2.0 * p @ u["Q"].sum(axis=1)
            vtot += V
            ps.append(p)
        wl = P["wl"]
        ml = np.tanh((P["bl"] + wl * x) / T)
        vl = 1.0 - ml * ml
        psi -= ml * wl
        lap += -np.sum(vl * wl * wl) / T
        vtot += np.sum(vl * wl * wl)
        loss += -lap / T + psi @ psi / (2.0 * T * T)
        for u, p in zip(us, ps):
            add(out, u, *potts_rule(u, x, p, psi))
        e = -2.0 * ml * vl * wl * wl - vl * wl * psi
        out["wl"] += (-ml * psi + 2.0 * vl * wl + x * e / T) / T ** 2
        out["bl"] += e / T ** 3
        out["theta"] += visible_rule(x, psi)
    return {k: v / len(X) for k, v in out.items()}, loss / len(X), vtot / len(X), ps


# ----------------------------------------------------------------------------------------------------------------
# 2. the device protocol (one clamped configuration, one averaging window of n reads)
# ----------------------------------------------------------------------------------------------------------------
def device_window(P, us, x, reads, spins):
    """reads[j] (n,): the state of Potts unit j at each read; spins (n, N): the per-pixel units (+-1) at each read."""
    n = spins.shape[0]
    gv, lap = visible(P, x)
    f = np.tile(gv, (n, 1))                    # force on the clamps at each read: grad_x of the joint energy
    acts = []
    for u, h in zip(us, reads):
        g = u["W"][h] + 2.0 * u["Q"][h] * x[u["idx"]]      # (n, d): what the active state pulls with
        f[:, u["idx"]] -= g
        acts.append(g)
    f -= spins * P["wl"]
    psi = f.mean(axis=0)                       # window-averaged clamp force
    out = zeros_like(P)
    for u, h, g in zip(us, reads, acts):
        xu, pu = x[u["idx"]], psi[u["idx"]]
        delta = g - g.mean(axis=0)             # fluctuation of the unit's force about its window average
        s_act = u["Q"].sum(axis=1)[h]          # the active state's total spring term
        phi = np.sum(delta * delta, axis=1) - delta @ pu + 2.0 * T * s_act    # one scalar per read
        dphi = phi - phi.mean()
        R = 2.0 * delta - pu
        gW, gb, gQ = np.zeros_like(u["W"]), np.zeros_like(u["b"]), np.zeros_like(u["Q"])
        for s in range(n):                     # only the active state's row changes
            m = h[s]
            gb[m] += dphi[s] / T ** 3 / n
            gW[m] += (xu * dphi[s] / T ** 3 + R[s] / T ** 2) / n
            gQ[m] += (xu * xu * dphi[s] / T ** 3 + 2.0 * xu * R[s] / T ** 2 + 2.0 / T) / n
        add(out, u, gW, gb, gQ)
        lap += -np.mean(np.sum(delta * delta, axis=1)) / T - 2.0 * np.mean(s_act)
    wl = P["wl"]
    mh = spins.mean(axis=0)
    v = 1.0 - mh * mh
    e = -2.0 * mh * v * wl * wl - v * wl * psi
    out["wl"] += (-mh * psi + 2.0 * v * wl + x * e / T) / T ** 2
    out["bl"] += e / T ** 3
    lap += -np.sum(v * wl * wl) / T
    out["theta"] += visible_rule(x, psi)
    return out, -lap / T + psi @ psi / (2.0 * T * T)


def comb(parts, weights):
    return ({k: sum(w * p[0][k] for p, w in zip(parts, weights)) for k in parts[0][0]},
            sum(w * p[1] for p, w in zip(parts, weights)))


def device_gradient(P, X, reads, spins, jackknife):
    """reads[j] (S, B), spins (S, B, N). jackknife 0: one window; 1: 2 full - mean of the two half windows."""
    us, S = units_of(P), spins.shape[0]
    out, loss = zeros_like(P), 0.0
    for b, x in enumerate(X):
        def win(lo, hi):
            return device_window(P, us, x, [r[lo:hi, b] for r in reads], spins[lo:hi, b])
        if jackknife:
            g, l = comb([win(0, S), win(0, S // 2), win(S // 2, S)], [2.0, -0.5, -0.5])
        else:
            g, l = win(0, S)
        for k in out:
            out[k] += g[k] / len(X)
        loss += l / len(X)
    return out, loss


def code_draws(Pj, Xj, key, S):
    """The hidden states make_loss_per_sample reads for `key`: its own fields and key use (latent_model.py 1439-1451)."""
    k_h, k_l = jr.split(key)
    A = lm.hidden_terms_quad(Pj["W"], Pj["Q"], Pj["b"], Xj, T, "potts")[2]
    reads = [np.asarray(jr.categorical(k_h, A / T, axis=-1, shape=(S,) + A.shape[:-1]))]
    Al = lm.local_terms(Pj["wl"], Pj["bl"], Xj, T)[2]
    up = np.asarray(jr.bernoulli(k_l, 0.5 * (1.0 + jnp.tanh(Al / T)), shape=(S,) + Al.shape))
    for i, (kw, kb, kq) in enumerate(lm.PATCH_SETS):
        if kw in Pj:
            Ap = lm.patch_terms(Pj[kw], Pj.get(kq), Pj[kb], Xj, T)[2]
            idx = np.asarray(jr.categorical(jr.fold_in(key, 1000 + i), Ap / T, axis=-1, shape=(S,) + Ap.shape[:-1]))
            reads += [idx[:, p, :] for p in range(idx.shape[1])]
    return reads, 2.0 * up.astype(float) - 1.0


def rel(a, b):
    return {k: float(np.linalg.norm(np.asarray(a[k]) - np.asarray(b[k])) / max(np.linalg.norm(np.asarray(b[k])), 1e-300))
            for k in b}


def show(title, d):
    print(f"  {title}: max over groups {max(d.values()):.2e}   [" + "  ".join(f"{k} {v:.1e}" for k, v in d.items()) + "]",
          flush=True)


def main():
    t0 = time.time()
    P, X = build()
    Pj, Xj = {k: jnp.asarray(v) for k, v in P.items()}, jnp.asarray(X)
    vis = StubVisible()
    print(f"T = {T}; {N} pixels, batch {B}; global Potts {M} states + Q, per-pixel units, patch sets "
          + ", ".join(f"{s}x{s} every {r} px with {S} states" for *_, s, r, S in PATCHES))

    # --- 1 ---------------------------------------------------------------------------------------------------------
    print("\n1. Local rule with exact statistics (my numpy rule, global AND patch units) vs jax.grad(make_loss)")
    mine, loss_mine, vtot, ps = exact_rule(P, X)
    exact_loss = lm.make_loss(vis, T, M, "potts", True)
    g_exact = jax.grad(exact_loss)(Pj, Xj)
    l_exact = float(exact_loss(Pj, Xj))
    print(f"  loss: mine {loss_mine:.10f}  code {l_exact:.10f}")
    show("relative difference per group", rel(mine, g_exact))
    A = P["b"] + X @ P["W"].T + (X * X) @ P["Q"].T
    pg = np.array([softmax(a / T) for a in A])
    print(f"  regime: global unit max p per sample {np.round(pg.max(1), 2)}, effective states "
          f"{np.round(1.0 / np.sum(pg * pg, axis=1), 2)}")

    # --- 2 ---------------------------------------------------------------------------------------------------------
    print("\n2. Device protocol (reads + clamp forces only, active-state updates) vs jax.grad(make_loss(samples=S))")
    for S, jk in ((8, 0), (8, 1), (64, 1)):
        worst, worst_l = {}, 0.0
        for seed in range(3):
            key = jr.PRNGKey(100 + seed)
            fn = lm.make_loss(vis, T, M, "potts", True, samples=S, jackknife=bool(jk))
            g_code = jax.grad(fn)(Pj, Xj, key)
            l_code = float(fn(Pj, Xj, key))
            reads, spins = code_draws(Pj, Xj, key, S)
            g_dev, l_dev = device_gradient(P, X, reads, spins, jk)
            r = rel(g_dev, g_code)
            worst = {k: max(worst.get(k, 0.0), v) for k, v in r.items()}
            worst_l = max(worst_l, abs(l_dev - l_code) / abs(l_code))
        show(f"S = {S}, {'jackknife' if jk else 'plug-in'}, 3 keys: relative difference", worst)
        print(f"    loss relative difference {worst_l:.1e}")
    r = rel(g_dev, g_exact)
    show("  for scale: one S = 64 jackknife gradient vs the EXACT gradient", r)

    # --- 3 ---------------------------------------------------------------------------------------------------------
    S, chunk, n_chunks = 8, 4000, 50
    print(f"\n3. Bias: mean over {chunk * n_chunks} keys at S = {S} against the exact gradient")
    flat_exact, unravel = ravel_pytree(g_exact)
    flat_exact = np.asarray(flat_exact)
    ests = {"plug-in": lm.make_loss(vis, T, M, "potts", True, samples=S),
            "jackknife 1": lm.make_loss(vis, T, M, "potts", True, samples=S, jackknife=True),
            "jackknife 2": lm.make_loss(vis, T, M, "potts", True, samples=S, jackknife=2)}
    fns = {}
    for name, fn in ests.items():
        vg = jax.value_and_grad(fn)

        def one(key, vg=vg):
            l, g = vg(Pj, Xj, key)
            return l, ravel_pytree(g)[0]
        fns[name] = jax.jit(jax.vmap(one))
    acc = {name: [0.0, 0.0, 0.0, 0.0] for name in ests}
    for c in range(n_chunks):
        keys = jr.split(jr.PRNGKey(7000 + c), chunk)
        for name in ests:
            l, g = fns[name](keys)
            l, g = np.asarray(l), np.asarray(g)
            a = acc[name]
            a[0] += g.sum(0)
            a[1] += (g * g).sum(0)
            a[2] += l.sum()
            a[3] += (l * l).sum()
    R = chunk * n_chunks
    print(f"  plug-in loss bias predicted by -V / (2 S T^2): {-vtot / (2 * S * T * T):.4f}  (V = {vtot:.6f})")
    for name in ests:
        a = acc[name]
        mean, se = a[0] / R, np.sqrt(np.maximum(a[1] / R - (a[0] / R) ** 2, 0.0) / R)
        lmean, lse = a[2] / R, np.sqrt(max(a[3] / R - (a[2] / R) ** 2, 0.0) / R)
        dm, ds = unravel(jnp.asarray(mean - flat_exact)), unravel(jnp.asarray(se))
        ok = se > 0
        chi2 = float(np.mean(((mean - flat_exact)[ok] / se[ok]) ** 2))
        print(f"  {name}: loss bias {lmean - l_exact:+.4f} +- {lse:.4f};  gradient chi2/dof of (mean - exact) = "
              f"{chi2:.2f} over {int(ok.sum())} elements (1 = unbiased within Monte Carlo error)")
        bias = {k: float(np.linalg.norm(np.asarray(dm[k])) / np.linalg.norm(np.asarray(g_exact[k]))) for k in g_exact}
        floor = {k: float(np.linalg.norm(np.asarray(ds[k])) / np.linalg.norm(np.asarray(g_exact[k]))) for k in g_exact}
        print("     |mean - exact| / |exact| (Monte Carlo floor): "
              + "  ".join(f"{k} {bias[k]:.3f} ({floor[k]:.3f})" for k in ("W", "b", "Q", "Wp", "bp", "Qp", "Wp2", "bp2",
                                                                         "Qp2", "wl", "bl", "theta")), flush=True)

    # --- 4 ---------------------------------------------------------------------------------------------------------
    print("\n4. Nudges: [a_m(x + eps u) - a_m(x - eps u)] / (2 eps) against G_m . u, eps = 0.5 (not small)")
    x, u_dir, eps = X[0], exact_rule(P, X[:1])[0]["theta"][2 * N:], 0.5          # some direction (here ∝ psi)
    a = lambda z: P["b"] + P["W"] @ z + P["Q"] @ (z * z)                           # noqa: E731
    G = P["W"] + 2.0 * P["Q"] * x
    d = (a(x + eps * u_dir) - a(x - eps * u_dir)) / (2 * eps)
    print(f"  max relative difference over the {M} states: {np.max(np.abs(d - G @ u_dir) / np.abs(G @ u_dir)):.1e}")
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
