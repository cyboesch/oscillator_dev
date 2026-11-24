#!/usr/bin/env python3
"""REVIEW COPY (2026-11-23): checks of the bond units (latent_model.bond_terms; train_latent --bond).
1. grad and Laplacian of F_bond against autodiff of bond_free_energy (float64, random parameters).
2. the training loss with bond units against the loss built from autodiff of the total free energy.
3. the sampler's drift (sample_latent.make_drift with BOND_P) against the training force, at a knot and between knots.
4. the reference / start: shapes, the centred start changes the force little.
CPU is enough. Prints PASS / FAIL per check and exits non-zero on a failure."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

T = lm.TEMP
rng = np.random.default_rng(0)
ok_all = True


def check(name, err, tol):
    global ok_all
    ok = bool(err < tol)
    ok_all &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {err:.3e} (tolerance {tol:g})", flush=True)


G, S, B = 8, 3, 5
n = G * G
NB = int(os.environ.get("BOND_CHECK_NB", "4"))        # 4: range 1; 12: range 2
shape = (NB, G, G, S)
wb = jnp.asarray(0.05 * rng.standard_normal(shape))
qb = jnp.asarray(0.05 * rng.standard_normal(shape))
bb = jnp.asarray(0.02 * rng.standard_normal(shape))
X = jnp.asarray(rng.standard_normal((B, n)))

# 1. bond term alone
g, lap, A, p = lm.bond_terms(wb, qb, bb, X, T)
f1 = lambda x: lm.bond_free_energy(wb, qb, bb, x[None], T)[0]   # noqa: E731
g_ad = jax.vmap(jax.grad(f1))(X)
lap_ad = jax.vmap(lambda x: jnp.trace(jax.hessian(f1)(x)))(X)
check("bond grad vs autodiff (relative)", float(jnp.max(jnp.abs(g - g_ad)) / jnp.max(jnp.abs(g_ad))), 1e-10)
check("bond Laplacian vs autodiff (relative)", float(jnp.max(jnp.abs(lap - lap_ad)) / jnp.max(jnp.abs(lap_ad))), 1e-10)
# the adjoint pair bond_diff / bond_fold
D = lm.bond_diff(X, NB)
Y = jnp.asarray(rng.standard_normal(D.shape)) * jnp.asarray(lm.bond_mask(G, NB))
check("bond_fold is the adjoint of bond_diff", float(abs(jnp.sum(D * Y) - jnp.sum(X * lm.bond_fold(Y)))), 1e-10)
check("bonds on the grid (G = 8: range 1: 2*8*7 + 2*7*7 = 210; range 2: + 2*8*6 + 2*6*6 + 4*7*6 = 546)",
      abs(float(lm.bond_mask(G, NB).sum()) - (210.0 if NB == 4 else 546.0)), 0.5)

# 2. the full training loss: visible (quartic, radius 1, lin) + global Potts with Q + per-pixel + one patch set + bonds
conn = lm.connectivity(G, 1)
vis = lm.make_visible(conn, n=n, impl="dense")
theta0, _ = lm.paper_theta_layout(conn.shape[0], n)
theta = lm.quartic_initial_theta(theta0, n, conn.shape[0]) + 0.01 * jnp.asarray(rng.standard_normal(theta0.shape))
K = 4
geom = lm.PatchGeometry(G, 4, 2)
params = {"theta": theta, "W": jnp.asarray(0.05 * rng.standard_normal((K, n))),
          "Q": jnp.asarray(0.01 * rng.standard_normal((K, n))), "b": jnp.asarray(0.02 * rng.standard_normal(K)),
          "wl": jnp.asarray(0.05 * rng.standard_normal(n)), "bl": jnp.asarray(0.02 * rng.standard_normal(n)),
          "Wp": jnp.asarray(0.05 * rng.standard_normal((geom.P, 3, geom.d))),
          "Qp": jnp.asarray(0.01 * rng.standard_normal((geom.P, 3, geom.d))),
          "bp": jnp.asarray(0.02 * rng.standard_normal((geom.P, 3))), "wb": wb, "qb": qb, "bb": bb}
loss_ps = lm.make_loss_per_sample(vis, T, K, "potts", True)
force = lm.make_force(vis, T, K, "potts", True)


def F_total(x):
    xb = x[None]
    f = vis.energy(params["theta"], xb)[0]
    A_ = xb @ params["W"].T + (xb * xb) @ params["Q"].T + params["b"]
    f = f - T * jax.scipy.special.logsumexp(A_ / T, axis=1)[0]
    Al = xb * params["wl"] + params["bl"]
    f = f - T * jnp.sum(jnp.logaddexp(Al / T, -Al / T))
    f = f + lm.patch_free_energy(params["Wp"], params["Qp"], params["bp"], xb, T)[0]
    return f + lm.bond_free_energy(wb, qb, bb, xb, T)[0]


gF = jax.vmap(jax.grad(F_total))(X)
lapF = jax.vmap(lambda x: jnp.trace(jax.hessian(F_total)(x)))(X)
loss_ad = -lapF / T + jnp.sum(gF * gF, axis=1) / (2 * T * T)
l_an = loss_ps(params, X)
check("full loss with bonds vs autodiff of the total free energy (relative)",
      float(jnp.max(jnp.abs(l_an - loss_ad) / jnp.abs(loss_ad))), 1e-9)
check("full force with bonds vs autodiff (relative)", float(jnp.max(jnp.abs(force(params, X) - gF)) / jnp.max(jnp.abs(gF))), 1e-9)
p_nb = {k: v for k, v in params.items() if k not in lm.BOND_KEYS}
check("bonds change the loss (sanity; should be far from 0)", 1.0 / (1e-30 + float(jnp.max(jnp.abs(loss_ps(p_nb, X) - l_an)))), 1e3)
# float32 compute path (--precision mixed) against float64
l32 = lm.make_loss_per_sample(vis, T, K, "potts", True, compute_dtype=jnp.float32)(params, X)
check("mixed-precision loss vs float64 (relative)", float(jnp.max(jnp.abs(l32 - l_an) / jnp.abs(l_an))), 2e-3)

# 3. the sampler's drift: x - drift_ode = grad F, with the histories interpolated in t
import sample_latent as sl  # noqa: E402
taus = np.array([5.0, 2.5, 0.0])
hist = {k: np.stack([np.asarray(v) * f for f in (0.5, 1.0, 1.5)]) for k, v in params.items()}
hist["theta"] = np.stack([np.asarray(theta)] * 3) * np.array([0.9, 1.0, 1.1])[:, None]
sl.BOND_P = tuple(hist[g] for g in lm.BOND_KEYS)
lm.PROBLEM_GEOMETRY["gauss2_8"] = (G, 1)
drift_ode, _ = sl.make_drift(hist["theta"], hist["W"], hist["b"], "ode", "gauss2_8", "potts", (hist["wl"], hist["bl"]),
                             hist["Q"], radius=1, taus=taus, patch=(hist["Wp"], hist["bp"], hist["Qp"], "potts"))
drift_sde, _ = sl.make_drift(hist["theta"], hist["W"], hist["b"], "sde", "gauss2_8", "potts", (hist["wl"], hist["bl"]),
                             hist["Q"], radius=1, taus=taus, patch=(hist["Wp"], hist["bp"], hist["Qp"], "potts"))
for t_fwd, w in ((2.5, None), (1.25, 0.5), (4.0, 0.4)):
    if w is None:
        pk = {k: jnp.asarray(hist[k][1]) for k in hist}
    else:
        i0 = 1 if t_fwd < 2.5 else 0
        a = (taus[i0] - t_fwd) / (taus[i0] - taus[i0 + 1])
        pk = {k: jnp.asarray(hist[k][i0] + a * (hist[k][i0 + 1] - hist[k][i0])) for k in hist}
    gref = force(pk, X)
    gs = jax.vmap(lambda x: x - drift_ode(lm.T_FORWARD - t_fwd, x, None))(X)
    gs2 = jax.vmap(lambda x: (x - drift_sde(lm.T_FORWARD - t_fwd, x, None)) / 2.0)(X)
    check(f"sampler ODE drift vs training force at forward t = {t_fwd} (relative)",
          float(jnp.max(jnp.abs(gs - gref)) / jnp.max(jnp.abs(gref))), 1e-9)
    check(f"sampler SDE drift vs training force at forward t = {t_fwd} (relative)",
          float(jnp.max(jnp.abs(gs2 - gref)) / jnp.max(jnp.abs(gref))), 1e-9)
sl.BOND_P = None
d0, _ = sl.make_drift(hist["theta"], hist["W"], hist["b"], "ode", "gauss2_8", "potts", (hist["wl"], hist["bl"]),
                      hist["Q"], radius=1, taus=taus, patch=(hist["Wp"], hist["bp"], hist["Qp"], "potts"))
gs0 = jax.vmap(lambda x: x - d0(2.5, x, None))(X)
check("without BOND_P the drift is the bond-free force (relative)",
      float(jnp.max(jnp.abs(gs0 - force({k: jnp.asarray(hist[k][1]) for k in hist if k not in lm.BOND_KEYS}, X)))
            / jnp.max(jnp.abs(gs0))), 1e-9)

# 4. reference and start
data = rng.standard_normal((2000, n)) * np.linspace(0.3, 1.5, n)[None]
q_ref, b_ref, v, M = lm.bond_reference(data, np.array([5.0, 2.0, 0.0]), T, 4, nb=NB)
w0, q0, b0 = lm.bond_start(q_ref[1], b_ref, 0.25)
print("reference shapes", q_ref.shape, b_ref.shape, v.shape, "start", w0.shape, q0.shape, b0.shape,
      "lr scales", [a.round(5).tolist() for a in lm.bond_lr_scales(q_ref, v, M, T)], flush=True)
check("start: sum over states of qb is zero", float(np.max(np.abs(q0.sum(-1)))), 1e-12)
Xd = jnp.asarray(np.exp(-2.0) * data[:64] + np.sqrt(T * (1 - np.exp(-4.0))) * rng.standard_normal((64, n)))
gb = lm.bond_terms(jnp.asarray(w0), jnp.asarray(q0), jnp.asarray(b0), Xd, T, with_lap=False)[0]
print(f"start force rms {float(jnp.sqrt(jnp.mean(gb ** 2))):.4g} (data scale e^-2 = {np.exp(-2.0):.3f})", flush=True)
print("ALL PASS" if ok_all else "SOME CHECKS FAILED", flush=True)
sys.exit(0 if ok_all else 1)
