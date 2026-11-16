#!/usr/bin/env python3
"""Numerical checks of the formulas in DESIGN.md (fast hidden units averaged out of the pixel dynamics).

Each analytic expression is compared against JAX autodiff of the free energy
F(x) = E_v(x) - T sum_j log Z_j(a_j(x)), a_j = b_j + W_j . x:
  1. psi = grad F and Laplacian F (DESIGN §1), for Ising and continuous double-well hidden units;
  2. the gradients of J_T = E[-Lap F / T + |grad F|^2 / (2 T^2)] w.r.t. W, b and the visible parameters (§2);
  3. the reverse process (§3): doubling the whole joint energy with the hidden units at 2T gives 2F - |x|^2/2,
     while doubling (W, b) with the hidden units at T (the theta_rev arithmetic) does not;
  4. the CD-1 / score-matching relation for F (Hyvarinen 2007), with antithetic noise, for shrinking dt,
     at T = 1 and T = 0.25 (the latter tests the power of T in the scale T^2 dt / 2);
  5. checks 1-2 at full size (N = 784, K = 16, Ising).
The double-well partition function is a sum over a fixed grid; the cumulant identities hold exactly for that sum.
Read-only; CPU only. Writes out/diagnostics/latent_checks/<job id>/report.txt and checks.json.
Run under Slurm (cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_latent_checks_neuronic.sh).
"""

import json
import os
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import vmap  # noqa: E402
from jax.scipy.special import logsumexp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT_DIR = os.path.join(PKG, "out", "diagnostics", "latent_checks", os.environ.get("SLURM_JOB_ID", "manual"))
TOL_EXACT = 1e-8          # relative tolerance for identities that should hold to round-off
RESULTS = {}


def report(line=""):
    print(line, flush=True)
    with open(os.path.join(OUT_DIR, "report.txt"), "a") as f:
        f.write(line + "\n")


def rel_err(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def record(name, err, tol=TOL_EXACT):
    ok = err < tol
    RESULTS[name] = {"rel_err": err, "tol": tol, "pass": bool(ok)}
    report(f"  {'PASS' if ok else 'FAIL'}  {name:<62} rel err {err:.2e}  (tol {tol:.0e})")
    return ok


# ----------------------------------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------------------------------
def make_visible(N, key, pairs=True):
    """E_v = sum_i (k_i x^2/2 + q_i x^4/4 + beta_i x_i) + sum_{i<l} c_il x_i x_l (generic, linear in theta_v)."""
    k1, k2, k3, k4 = jr.split(key, 4)
    iu = np.triu_indices(N, 1) if pairs else (np.zeros(0, int), np.zeros(0, int))
    tv = {"k": 1.0 + 0.5 * jr.uniform(k1, (N,)), "q": 0.2 + 0.2 * jr.uniform(k2, (N,)),
          "beta": 0.1 * jr.normal(k3, (N,)), "c": 0.1 * jr.normal(k4, (iu[0].size,)) / np.sqrt(N)}
    iu = (jnp.asarray(iu[0]), jnp.asarray(iu[1]))

    def E_vis(x, tv):
        e = jnp.sum(0.5 * tv["k"] * x**2 + 0.25 * tv["q"] * x**4 + tv["beta"] * x)
        return e + jnp.sum(tv["c"] * x[iu[0]] * x[iu[1]])

    return tv, E_vis


YG = jnp.linspace(-3.0, 3.0, 3001)   # quadrature grid for the double-well hidden units


def hidden_model(kind, T, kappa=None, lam=None, energy_scale=1.0):
    """log Z(a) and the conditional cumulants (m, v, k3) of one hidden unit at temperature T.

    kind 'ising': y in {-1, +1}, U = 0.  kind 'dw': U(y) = energy_scale * (-kappa y^2/2 + lam y^4/4) on YG.
    """
    if kind == "ising":
        def logZ(a):
            return jnp.logaddexp(a / T, -a / T)                       # log(2 cosh(a/T))

        def cum(a):
            m = jnp.tanh(a / T)
            v = 1.0 - m**2
            return m, v, -2.0 * m * v
    else:
        U = energy_scale * (-0.5 * kappa * YG**2 + 0.25 * lam * YG**4)

        def logZ(a):
            return logsumexp(-(U - YG * a) / T)                       # grid sum (spacing is a constant)

        def cum(a):
            w = jax.nn.softmax(-(U - YG * a) / T)
            m = jnp.sum(w * YG)
            d = YG - m
            return m, jnp.sum(w * d**2), jnp.sum(w * d**3)
    return logZ, cum


def make_F(E_vis, logZ, T):
    def F(x, tv, W, b):
        a = b + W @ x
        return E_vis(x, tv) - T * jnp.sum(vmap(logZ)(a))
    return F


def make_J(F, T):
    """Autodiff ground truth: J_T = mean_x [-Lap F / T + |grad F|^2 / (2 T^2)]."""
    def per_x(x, tv, W, b):
        g = jax.grad(F)(x, tv, W, b)
        lap = jnp.trace(jax.jacfwd(jax.grad(F))(x, tv, W, b))
        return -lap / T + jnp.sum(g**2) / (2.0 * T**2)

    def J(tv, W, b, X):
        return jnp.mean(vmap(per_x, in_axes=(0, None, None, None))(X, tv, W, b))
    return J


def analytic_psi_D(E_vis, cum, T):
    """DESIGN §1: psi = grad E_v - sum_j m_j W_j ; Lap F = Lap E_v - (1/T) sum_j v_j |W_j|^2."""
    def f(x, tv, W, b):
        a = b + W @ x
        m, v, k3 = vmap(cum)(a)
        psi = jax.grad(E_vis)(x, tv) - W.T @ m
        lap = jnp.trace(jax.jacfwd(jax.grad(E_vis))(x, tv)) - jnp.sum(v * jnp.sum(W**2, axis=1)) / T
        return psi, lap, m, v, k3
    return f


def analytic_grads(E_vis, cum, T):
    """DESIGN §2: dJ/dW_ji, dJ/db_j (hidden) and dJ/dtheta_v (visible, with the total force psi)."""
    psiD = analytic_psi_D(E_vis, cum, T)

    def per_x(x, tv, W, b):
        psi, _, m, v, k3 = psiD(x, tv, W, b)
        Wn2 = jnp.sum(W**2, axis=1)
        Wpsi = W @ psi
        gW = (-m[:, None] * psi[None, :] + 2.0 * v[:, None] * W
              + (x[None, :] / T) * (k3 * Wn2 - v * Wpsi)[:, None]) / T**2
        gb = (k3 * Wn2 - v * Wpsi) / T**3
        # visible: E[-d(Lap E_v)/dtheta / T + (d grad E_v / dtheta)^T psi / T^2]
        lap_grad = jax.grad(lambda t: jnp.trace(jax.jacfwd(jax.grad(E_vis))(x, t)))(tv)
        _, vjp = jax.vjp(lambda t: jax.grad(E_vis)(x, t), tv)
        force_term = vjp(psi)[0]
        gv = jax.tree_util.tree_map(lambda l, f_: -l / T + f_ / T**2, lap_grad, force_term)
        return gW, gb, gv

    def grads(tv, W, b, X):
        gW, gb, gv = vmap(per_x, in_axes=(0, None, None, None))(X, tv, W, b)
        return gW.mean(0), gb.mean(0), jax.tree_util.tree_map(lambda g: g.mean(0), gv)
    return grads


def setup_case(N, K, T, key, pairs=True, x_scale=0.5, a_over_T=1.5):
    kv, kw, kb, kx = jr.split(key, 4)
    tv, E_vis = make_visible(N, kv, pairs=pairs)
    X = x_scale * jr.normal(kx, (6, N))
    W = jr.normal(kw, (K, N)) * (a_over_T * T / (x_scale * np.sqrt(N)))   # a/T of order a_over_T
    b = 0.5 * T * jr.normal(kb, (K,))
    return tv, E_vis, W, b, X


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------
def check_formulas(label, N, K, T, kind, key, pairs=True, dw=(None, None)):
    report(f"[{label}] N={N}, K={K}, T={T}, hidden={kind}, pair terms={'yes' if pairs else 'no'}")
    tv, E_vis, W, b, X = setup_case(N, K, T, key, pairs=pairs)
    logZ, cum = hidden_model(kind, T, *dw)
    F = make_F(E_vis, logZ, T)
    psiD = analytic_psi_D(E_vis, cum, T)
    g_auto = vmap(jax.grad(F), in_axes=(0, None, None, None))(X, tv, W, b)
    lap_auto = vmap(lambda x: jnp.trace(jax.jacfwd(jax.grad(F))(x, tv, W, b)))(X)
    psi, lap, m, v, k3 = vmap(psiD, in_axes=(0, None, None, None))(X, tv, W, b)
    a_over_T = np.asarray((b[None, :] + X @ W.T) / T)
    report(f"  a/T range [{a_over_T.min():.2f}, {a_over_T.max():.2f}]; m range [{float(m.min()):.2f}, {float(m.max()):.2f}]")
    record(f"{label}: psi = grad F", rel_err(psi, g_auto))
    record(f"{label}: Lap F", rel_err(lap, lap_auto))
    J = make_J(F, T)
    gv_auto, gW_auto, gb_auto = jax.grad(J, argnums=(0, 1, 2))(tv, W, b, X)
    gW, gb, gv = analytic_grads(E_vis, cum, T)(tv, W, b, X)
    record(f"{label}: dJ/dW", rel_err(gW, gW_auto))
    record(f"{label}: dJ/db", rel_err(gb, gb_auto))
    for name in ("k", "q", "beta") + (("c",) if pairs else ()):
        record(f"{label}: dJ/dtheta_v[{name}] (visible, total psi)", rel_err(gv[name], gv_auto[name]))


def check_reverse(label, N, K, T, kind, key, dw=(None, None)):
    report(f"[{label}] reverse process: N={N}, K={K}, T={T}, hidden={kind}")
    tv, E_vis, W, b, X = setup_case(N, K, T, key)
    logZ_T, _ = hidden_model(kind, T, *dw)
    F = make_F(E_vis, logZ_T, T)

    def target(x):                       # pixels must feel 2F - |x|^2/2
        return 2.0 * F(x, tv, W, b) - 0.5 * x @ x

    logZ_2T, _ = hidden_model(kind, 2.0 * T, *dw, energy_scale=2.0)   # hidden energy doubled, at 2T

    def physical(x):                     # joint energy doubled, hidden units at T_h = 2T
        a2 = 2.0 * (b + W @ x)
        return 2.0 * E_vis(x, tv) - 0.5 * x @ x - 2.0 * T * jnp.sum(vmap(logZ_2T)(a2))

    def naive(x):                        # theta_rev arithmetic applied to (W, b), hidden units at T
        a2 = 2.0 * (b + W @ x)
        return 2.0 * E_vis(x, tv) - 0.5 * x @ x - T * jnp.sum(vmap(logZ_T)(a2))

    f_t = vmap(jax.grad(target))(X)
    record(f"{label}: force of doubled energy at T_h = 2T", rel_err(vmap(jax.grad(physical))(X), f_t))
    e_naive = rel_err(vmap(jax.grad(naive))(X), f_t)
    RESULTS[f"{label}: force of naive doubling at T_h = T (should differ)"] = {"rel_err": e_naive}
    report(f"  info  {label}: force of naive doubling (W, b -> 2W, 2b at T_h = T): rel err {e_naive:.2e} (should be large)")
    d = np.asarray(vmap(physical)(X) - vmap(target)(X))
    record(f"{label}: free energy offset is constant", float(np.ptp(d) / max(np.max(np.abs(np.asarray(vmap(target)(X)))), 1e-300)))


def check_cd1(key, T):
    """Hyvarinen (2007): <d_theta F(x)> - <d_theta F(x')> = (T^2 dt / 2) d_theta J_T + O(dt^2) for any smooth F."""
    N, K = 6, 3
    report(f"[CD-1, T = {T}] N={N}, K={K}, Ising hidden units; x' = x - (dt/2) psi + sqrt(T dt) xi, antithetic xi")
    tv, E_vis, W, b, X = setup_case(N, K, T, key, x_scale=0.8)
    logZ, _ = hidden_model("ising", T)
    F = make_F(E_vis, logZ, T)
    J = make_J(F, T)
    _, gW_J, gb_J = jax.grad(J, argnums=(0, 1, 2))(tv, W, b, X)
    dF = jax.jit(vmap(jax.grad(F, argnums=(2, 3)), in_axes=(0, None, None, None)))
    psi = vmap(jax.grad(F), in_axes=(0, None, None, None))(X, tv, W, b)
    n_pairs = 100_000
    rows = []
    for dt in (1e-1, 3e-2, 1e-2, 3e-3):
        lhs_W, lhs_b = 0.0, 0.0
        for i in range(X.shape[0]):
            xi = jr.normal(jr.fold_in(key, i), (n_pairs, N))
            base = X[i] - 0.5 * dt * psi[i]
            xp = jnp.concatenate([base + np.sqrt(T * dt) * xi, base - np.sqrt(T * dt) * xi])
            gW_p, gb_p = dF(xp, tv, W, b)
            gW_0, gb_0 = jax.grad(F, argnums=(2, 3))(X[i], tv, W, b)
            lhs_W = lhs_W + (gW_0 - gW_p.mean(0)) / X.shape[0]
            lhs_b = lhs_b + (gb_0 - gb_p.mean(0)) / X.shape[0]
        scale = T**2 * dt / 2.0
        eW, eb = rel_err(lhs_W, scale * gW_J), rel_err(lhs_b, scale * gb_J)
        rows.append((dt, eW, eb))
        report(f"  dt {dt:7.0e}: rel err W {eW:.2e}, b {eb:.2e}")
    RESULTS[f"CD-1 relation, T = {T}"] = {"dt_relerr_W_relerr_b": rows}
    shrinks = all(rows[-1][c] < 0.2 * rows[0][c] and rows[-1][c] < 0.05 for c in (1, 2))
    record(f"CD-1, T = {T}: W and b errors shrink with dt, < 5% at the smallest", 0.0 if shrinks else 1.0, tol=0.5)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; jax {jax.__version__}; devices {jax.devices()}; "
           f"default float {jnp.zeros(1).dtype}")
    key = jr.PRNGKey(0)
    T = 0.02
    dw = (12 * T, 12 * T)   # double well with minima at +-1 and barrier 3T
    check_formulas("small Ising", 8, 3, T, "ising", jr.fold_in(key, 1))
    check_formulas("small double well", 8, 3, T, "dw", jr.fold_in(key, 2), dw=dw)
    check_formulas("small Ising, T = 1", 8, 3, 1.0, "ising", jr.fold_in(key, 3))
    check_reverse("reverse Ising", 8, 3, T, "ising", jr.fold_in(key, 4))
    check_reverse("reverse double well", 8, 3, T, "dw", jr.fold_in(key, 5), dw=dw)
    check_cd1(jr.fold_in(key, 6), 1.0)
    check_cd1(jr.fold_in(key, 8), 0.25)
    check_formulas("full size Ising", 784, 16, T, "ising", jr.fold_in(key, 7), pairs=False)
    n_fail = sum(1 for r in RESULTS.values() if r.get("pass") is False)
    report(f"\n{len(RESULTS)} checks, {n_fail} failed; total time {time.time() - t0:.0f} s")
    with open(os.path.join(OUT_DIR, "checks.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)


if __name__ == "__main__":
    main()
