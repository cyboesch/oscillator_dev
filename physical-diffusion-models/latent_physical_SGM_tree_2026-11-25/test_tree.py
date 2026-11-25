#!/usr/bin/env python3
"""Tests of tree_model.py (2026-11-25), CPU, float64 unless stated.

A. F_tree against a brute-force sum over every hidden configuration (4x4 grid, K 3, four 2x2 patches of 3 states).
B. tree_terms' gradient and Laplacian against autodiff of tree_free_energy (8x8, K 3, two patch sets, with and
   without Qp, with and without Q of the global unit).
C. J = 0 is today's model: the loss per sample and the force equal lm.make_loss_per_sample / lm.make_force
   (potts + quad + local + two patch sets) on 8x8.
D. The loss's gradient with respect to J against central differences.
E. Krotov's variant (the global unit without pixel couplings, W = Q = 0): finite, and F against brute force.
F. Mixed precision (float32 loss) against float64 at field strengths as in a trained model.

Usage (from this folder): sbatch jobs/run_cpu.sh test_tree.py
"""
import itertools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tree_model as tm  # noqa: E402
lm = tm.lm

T = lm.TEMP
FAIL = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def rand_model(rng, grid, K, specs, quad=True, quad_p=(True, True), j_rms=1.0, scale=0.06):
    """Random parameters with fields of a few T. specs: [(side, stride, S)]."""
    N = grid * grid
    p = {"W": scale * rng.standard_normal((K, N)) / np.sqrt(N) * 4, "b": T * rng.standard_normal(K)}
    if quad:
        p["Q"] = 0.3 * scale * rng.standard_normal((K, N)) / np.sqrt(N) * 4
    for (side, stride, S), (kw, kb, kq), kj, qp in zip(specs, lm.PATCH_SETS, tm.J_KEYS, quad_p):
        geom = lm.PatchGeometry(grid, side, stride)
        p[kw] = scale * rng.standard_normal((geom.P, S, geom.d)) / np.sqrt(geom.d) * 2
        p[kb] = T * rng.standard_normal((geom.P, S))
        if qp:
            p[kq] = 0.3 * scale * rng.standard_normal((geom.P, S, geom.d)) / np.sqrt(geom.d) * 2
        p[kj] = j_rms * T * rng.standard_normal((geom.P, K, S))
    return {k: jnp.asarray(v) for k, v in p.items()}


def brute_force_F(p, X):
    """F by explicit enumeration of (g, s_1 .. s_P) for one patch set."""
    W, b, Q = np.asarray(p["W"]), np.asarray(p["b"]), p.get("Q")
    Wp, bp, J = np.asarray(p["Wp"]), np.asarray(p["bp"]), np.asarray(p["J"])
    Qp = p.get("Qp")
    geom = lm.patch_geometry(X.shape[1], Wp.shape[0], Wp.shape[2])
    out = []
    for x in np.asarray(X):
        a_g = W @ x + b + (0 if Q is None else np.asarray(Q) @ (x * x))
        xp = x[geom.idx]                                               # (P, d)
        a_ps = np.einsum("psd,pd->ps", Wp, xp) + bp + (0 if Qp is None else np.einsum("psd,pd->ps", np.asarray(Qp), xp * xp))
        P, S = a_ps.shape
        vals = []
        for g in range(W.shape[0]):
            for s in itertools.product(range(S), repeat=P):
                vals.append((a_g[g] + sum(a_ps[q, s[q]] + J[q, g, s[q]] for q in range(P))) / T)
        vals = np.array(vals)
        out.append(-T * (vals.max() + np.log(np.exp(vals - vals.max()).sum())))
    return np.array(out)


def main():
    rng = np.random.default_rng(0)

    # A. brute force
    p = rand_model(rng, 4, 3, [(2, 2, 3)])
    X = jnp.asarray(rng.standard_normal((5, 16)))
    Ft = np.asarray(tm.tree_free_energy(p["W"], p["Q"], p["b"], tm.tree_sets(p), X, T))
    Fb = brute_force_F(p, X)
    check(np.max(np.abs(Ft - Fb)) < 1e-12, f"A. F_tree = brute-force sum over {3 * 3 ** 4} hidden configurations (max diff {np.max(np.abs(Ft - Fb)):.1e})")

    # B. gradient and Laplacian against autodiff
    for label, quad, quad_p in (("Q, Qp, Qp2", True, (True, True)), ("Q, no Qp, Qp2", True, (False, True)),
                                ("no Q, Qp, no Qp2", False, (True, False))):
        p = rand_model(rng, 8, 3, [(4, 2, 3), (2, 2, 4)], quad=quad, quad_p=quad_p)
        X = jnp.asarray(rng.standard_normal((6, 64)))
        sets = tm.tree_sets(p)
        F1 = lambda x: tm.tree_free_energy(p["W"], p.get("Q"), p["b"], sets, x[None, :], T)[0]   # noqa: E731
        g_ad = jax.vmap(jax.grad(F1))(X)
        lap_ad = jax.vmap(lambda x: jnp.trace(jax.hessian(F1)(x)))(X)
        g, lap, pi, qs, _ = tm.tree_terms(p["W"], p.get("Q"), p["b"], sets, X, T)
        eg = float(jnp.max(jnp.abs(g - g_ad)) / jnp.max(jnp.abs(g_ad)))
        el = float(jnp.max(jnp.abs(lap - lap_ad)) / jnp.max(jnp.abs(lap_ad)))
        check(eg < 1e-10 and el < 1e-10, f"B. ({label}) grad rel. err {eg:.1e}, Laplacian rel. err {el:.1e}")
        check(abs(float(jnp.sum(pi)) - 6) < 1e-10 and all(abs(float(jnp.sum(q)) - q.shape[0] * 6) < 1e-8 for q in qs),
              f"B. ({label}) posteriors normalised; effective global states {float(jnp.mean(jnp.exp(-jnp.sum(pi * jnp.log(pi + 1e-300), 1)))):.2f}")
        g2 = tm.tree_terms(p["W"], p.get("Q"), p["b"], sets, X, T, with_lap=False)[0]
        check(bool(jnp.all(g2 == g)), f"B. ({label}) with_lap=False gives the same gradient")

    # C. J = 0 = today's model
    grid, K = 8, 3
    N = grid * grid
    conn = lm.connectivity(grid, 1)
    for impl in ("dense", "stencil"):
        vis = lm.make_visible(conn, N, impl, ("lin",), sextic=False)
        theta0, _ = lm.paper_theta_layout(vis.P, N)
        theta0 = lm.quartic_initial_theta(theta0, N, vis.P)
        for s in lm.dropped_pair_slices(N, vis.P, ("lin",)):
            theta0 = theta0.at[s].set(0.0)
        p = rand_model(rng, grid, K, [(4, 2, 3), (2, 2, 4)])
        p0 = {k: v for k, v in p.items() if k not in tm.J_KEYS}
        p0.update(theta=theta0 + 0.05 * jnp.asarray(rng.standard_normal(theta0.shape)) * (theta0 != 0),
                  wl=jnp.asarray(0.05 * rng.standard_normal(N)), bl=jnp.asarray(T * rng.standard_normal(N)))
        X = jnp.asarray(rng.standard_normal((7, N)))
        ref = lm.make_loss_per_sample(vis, T, K, "potts", True)(p0, X)
        new = tm.make_tree_loss_per_sample(vis, T, True)(p0, X)
        pz = dict(p0, **{kj: jnp.zeros_like(p[kj]) for kj in ("J", "J2")})
        newz = tm.make_tree_loss_per_sample(vis, T, True)(pz, X)
        e = float(jnp.max(jnp.abs(new - ref) / jnp.abs(ref)))
        check(e < 1e-10 and bool(jnp.all(newz == new)), f"C. ({impl}) J absent or 0: loss per sample = lm.make_loss_per_sample (rel. {e:.1e})")
        fr = lm.make_force(vis, T, K, "potts", True)(p0, X)
        fn = tm.make_tree_force(vis, T, True)(p0, X)
        ef = float(jnp.max(jnp.abs(fn - fr)) / jnp.max(jnp.abs(fr)))
        check(ef < 1e-11, f"C. ({impl}) J absent: force = lm.make_force (rel. {ef:.1e})")
        pj = dict(p0, J=p["J"], J2=p["J2"])
        d = float(jnp.max(jnp.abs(tm.make_tree_loss_per_sample(vis, T, True)(pj, X) - ref) / jnp.abs(ref)))
        check(d > 1e-6, f"C. ({impl}) J of rms 1 T enters the loss (rel. change {d:.2g}; the toy model's global unit is nearly decided)")

    # D. gradient with respect to J against central differences (stencil model of C)
    loss = tm.make_tree_loss(vis, T, True)
    gJ = jax.grad(loss)(pj, X)
    for kj in ("J", "J2"):
        worst = 0.0
        for _ in range(3):
            v = jnp.asarray(rng.standard_normal(pj[kj].shape))
            h = 1e-5 * T
            fd = (loss(dict(pj, **{kj: pj[kj] + h * v}), X) - loss(dict(pj, **{kj: pj[kj] - h * v}), X)) / (2 * h)
            an = jnp.sum(gJ[kj] * v)
            worst = max(worst, float(abs(fd - an) / abs(an)))
        check(worst < 1e-6, f"D. dL/d{kj} against central differences (rel. {worst:.1e})")

    # E. Krotov's variant: the global unit sees only the patch states
    p = rand_model(rng, 4, 3, [(2, 2, 3)], j_rms=3.0)
    p = dict(p, W=jnp.zeros_like(p["W"]), Q=jnp.zeros_like(p["Q"]))
    X = jnp.asarray(rng.standard_normal((5, 16)))
    Ft = np.asarray(tm.tree_free_energy(p["W"], p["Q"], p["b"], tm.tree_sets(p), X, T))
    g, lap, pi, _, _ = tm.tree_terms(p["W"], p["Q"], p["b"], tm.tree_sets(p), X, T)
    check(np.max(np.abs(Ft - brute_force_F(p, X))) < 1e-12 and bool(jnp.all(jnp.isfinite(g)) & jnp.all(jnp.isfinite(lap))),
          f"E. W = Q = 0 (top unit coupled to the patches only): F = brute force, grad and Laplacian finite; "
          f"effective global states {float(jnp.mean(jnp.exp(-jnp.sum(pi * jnp.log(pi + 1e-300), 1)))):.2f} of 3")

    # F. mixed precision
    p = rand_model(rng, 8, 3, [(4, 2, 3), (2, 2, 4)], j_rms=2.0, scale=0.2)
    pf = dict(p, theta=p0["theta"], wl=p0["wl"], bl=p0["bl"])
    X = jnp.asarray(rng.standard_normal((64, N)))
    l64 = tm.make_tree_loss(vis, T, True)
    l32 = tm.make_tree_loss(vis, T, True, compute_dtype=jnp.float32)
    v64, g64 = jax.value_and_grad(l64)(pf, X)
    v32, g32 = jax.value_and_grad(l32)(pf, X)
    rel = {k: float(jnp.linalg.norm(g32[k] - g64[k]) / jnp.linalg.norm(g64[k])) for k in ("W", "Wp", "Wp2", "J", "J2", "theta")}
    check(abs(float(v32 - v64) / float(v64)) < 1e-4 and max(rel.values()) < 1e-3,
          f"F. float32 loss against float64: value rel. {abs(float(v32 - v64) / float(v64)):.1e}; gradients rel. "
          + ", ".join(f"{k} {v:.1e}" for k, v in rel.items()))

    print(f"test_tree: {'ALL PASSED' if not FAIL else str(len(FAIL)) + ' FAILED'}", flush=True)
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
