#!/usr/bin/env python3
"""The locked sampler (../latent_physical_SGM/sample_latent.py, unchanged) for a tree run (2026-11-25).

A tree run is either a mirror folder below this folder's out/ written by train_J.py (links to a locked run's files
plus J_history.npz, the coupling tables J, J2 per slice) or a run trained by train_tree.py --tree (J, J2 in its
hidden_history.npz). This script imports sample_latent, points its run root at this
folder's out/ (so samples are written only here), and replaces two of its functions:

 - load_params: the original, plus the J history (kept here; the manifest records its file and sha256 as 'tree');
 - make_drift: for a run with J, the same drift with the global unit's and the patch units' forces replaced by the
   tree's force (tree_model.tree_force; J interpolated linearly in time like every other parameter). Everything
   else (visible force, per-pixel units, interpolation, solver, start states, Brownian keys, quench, corrector) is
   sample_latent's own code.

Check at every start: with J set to zero the new drift must equal sample_latent's drift (three times, random states);
the job stops if it does not.

Usage (GPU job, from this folder), the same arguments as sample_latent.py:
  sbatch jobs/run_gpu.sh sample_tree.py --run RUN__J-TAG --n 2000 --chunk 250 --quench 1
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tree_model as tm  # noqa: E402
lm = tm.lm
lm.OUT_ROOT = os.path.join(HERE, "out")          # runs = the mirrors here; nothing is written to the locked out/
import sample_latent as sl  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

_load_params, _make_drift = sl.load_params, sl.make_drift
TREE_J = None                                    # [J (n_slices, P, K, S) per patch set] of the run being sampled
CHECKED = False


def load_params(run, **kw):
    global TREE_J
    out = _load_params(run, **kw)
    run_dir, prov = out[3], out[4]
    jp = os.path.join(run_dir, "J_history.npz")          # train_J.py; a train_tree.py --tree run keeps J in
    if not os.path.exists(jp):                           # hidden_history.npz (members J, J2, J3)
        with np.load(os.path.join(run_dir, "hidden_history.npz")) as z:
            if "J" in z.files:
                jp = os.path.join(run_dir, "hidden_history.npz")
    if os.path.exists(jp):
        with np.load(jp) as z:
            TREE_J = [np.asarray(z[kj], np.float64) for kj in tm.J_KEYS if kj in z.files]
        prov["tree"] = {"J_file": os.path.basename(jp), "sha256": lm.sha256_file(jp),
                        "rms_J_over_T_last_slice": [float(np.sqrt(np.mean(J[-1] ** 2)) / lm.TEMP) for J in TREE_J]}
        print(f"tree run: J history {[J.shape for J in TREE_J]} from {jp}", flush=True)
    return out


def tree_drift(J_list, theta_hist, W_hist, b_hist, sampler, problem, hidden, local, quad, radius, neighbourhood, taus, patch):
    """sample_latent.make_drift for the locked recipe (Potts global unit with Q, per-pixel units, Potts patch sets)
    with the tree's force in place of the global unit's and the patch units' forces."""
    n_theta, K = theta_hist.shape[1], W_hist.shape[1]
    grid, r0 = lm.problem_geometry(problem)
    radius = r0 if radius is None else int(radius)
    n = grid * grid
    psets = [patch] if isinstance(patch, tuple) else list(patch)
    if len(J_list) != len(psets):
        raise ValueError(f"{len(J_list)} J tables for {len(psets)} patch sets")
    parts = [theta_hist, W_hist.reshape(W_hist.shape[0], -1), b_hist, quad.reshape(quad.shape[0], -1), local[0], local[1]]
    specs = []
    for ps, J in zip(psets, J_list):
        Wp, bp, Qp = np.asarray(ps[0]), np.asarray(ps[1]), None if ps[2] is None else np.asarray(ps[2])
        _, P, S, d = Wp.shape
        if J.shape != (Wp.shape[0], P, K, S):
            raise ValueError(f"J of shape {J.shape} for patch units {Wp.shape} and K = {K}")
        specs.append((sum(a.shape[1] for a in parts), P, S, d, Qp is not None))
        parts += [Wp.reshape(Wp.shape[0], -1), bp.reshape(bp.shape[0], -1)] + ([] if Qp is None else [Qp.reshape(Qp.shape[0], -1)])
        parts += [J.reshape(J.shape[0], -1)]
    stacked = np.concatenate(parts, axis=1)
    if sl.HISTORY_DTYPE != np.float64:
        stacked = stacked.astype(sl.HISTORY_DTYPE)
    taus = lm.forward_time_pts() if taus is None else np.asarray(taus, dtype=np.float64)
    interp = sl.interpolate_history(stacked, taus)
    conn = lm.connectivity(grid, radius, neighbourhood)
    _, unflatten = lm.paper_theta_layout(conn.shape[0], n)
    grad_E = jax.grad(sl.setup_energy_fn(jnp.asarray(conn), unflatten))
    th_lin = lm.theta_lin(n_theta, n)
    scale = 2.0 if sampler == "sde" else 1.0
    T = lm.TEMP

    def drift(t, x, args):
        p = interp(lm.T_FORWARD - t)
        d = -grad_E(x, scale * p[:n_theta] - th_lin)
        o = n_theta
        W = p[o:o + K * n].reshape(K, n)
        b = p[o + K * n:o + K * n + K]
        Q = p[o + K * n + K:o + 2 * K * n + K].reshape(K, n)
        o = o + 2 * K * n + K
        wl, bl = p[o:o + n], p[o + n:o + 2 * n]
        sets = []
        for o, P, S, dd, with_qp in specs:
            m = P * S * dd
            Wt, bt = p[o:o + m].reshape(P, S, dd), p[o + m:o + m + P * S].reshape(P, S)
            o2 = o + m + P * S
            Qt = p[o2:o2 + m].reshape(P, S, dd) if with_qp else None
            o2 = o2 + (m if with_qp else 0)
            sets.append((Wt, Qt, bt, p[o2:o2 + P * K * S].reshape(P, K, S)))
        d = d - scale * tm.tree_force(W, Q, b, sets, x[None], T)[0]
        return d + scale * jnp.tanh((wl * x + bl) / T) * wl

    def diffusion(t, x, args):
        return jnp.sqrt(2.0 * T) * jnp.eye(n)

    return drift, diffusion


def make_drift(theta_hist, W_hist, b_hist, sampler, problem="mnist28", hidden="ising", local=None, quad=None,
               radius=None, neighbourhood="chebyshev", taus=None, harmonic=None, patch=None, cond=None):
    global CHECKED
    if TREE_J is None:
        return _make_drift(theta_hist, W_hist, b_hist, sampler, problem, hidden, local, quad, radius, neighbourhood, taus,
                           harmonic=harmonic, patch=patch, cond=cond)
    if hidden != "potts" or quad is None or local is None or harmonic is not None or cond is not None or patch is None:
        raise SystemExit("sample_tree: a tree run must be the locked recipe (Potts global unit with Q, per-pixel units, "
                         "Potts patch sets; no harmonic modes, no --cond)")
    if not CHECKED:                               # J = 0 must give sample_latent's own drift
        ref, _ = _make_drift(theta_hist, W_hist, b_hist, sampler, problem, hidden, local, quad, radius, neighbourhood, taus,
                             harmonic=None, patch=patch, cond=None)
        zero, _ = tree_drift([np.zeros_like(J) for J in TREE_J], theta_hist, W_hist, b_hist, sampler, problem, hidden,
                             local, quad, radius, neighbourhood, taus, patch)
        worst = 0.0
        for i, t in enumerate((0.3, 2.6, 4.95)):
            x = jr.normal(jr.PRNGKey(i), (W_hist.shape[2],))
            a, b = ref(t, x, None), zero(t, x, None)
            worst = max(worst, float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a))))
        print(f"sample_tree check: tree drift with J = 0 against sample_latent's drift, max rel. diff {worst:.1e} "
              f"({'OK' if worst < 1e-9 else 'MISMATCH'})", flush=True)
        if not worst < 1e-9:
            raise SystemExit("sample_tree: the tree drift with J = 0 does not reproduce sample_latent's drift")
        del ref, zero                              # their device copies of the parameter history (64x64: 7 GB each)
        import gc
        gc.collect()
        CHECKED = True
    return tree_drift(TREE_J, theta_hist, W_hist, b_hist, sampler, problem, hidden, local, quad, radius, neighbourhood,
                      taus, patch)


sl.load_params, sl.make_drift = load_params, make_drift

if __name__ == "__main__":
    sl.main()
