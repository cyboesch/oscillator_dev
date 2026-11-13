#!/usr/bin/env python3
"""Generate samples from a trained run of train_latent.py, or from the paper checkpoint (--run paper).

Reverse SDE (DESIGN 3: in simulation use x - 2 grad F directly):
    dx = [x - 2 grad F(x; params(5 - t))] dt + sqrt(2T) dW
       = [-grad E^v_{theta_rev}(x) + 2 sum_j tanh(a_j(x)/T) W_j] dt + sqrt(2T) dW,   theta_rev = 2 theta_v - theta_lin
PF-ODE: dx/dt = x - grad F = -grad E^v_{theta_v - theta_lin}(x) + sum_j tanh(a_j/T) W_j.
The visible force uses the paper's energy_fn with autodiff (the paper's own sampler code path); the hidden force
is analytic. All parameters (theta_v, W, b) are interpolated linearly in t between the 100 knots, as in the paper,
and the solver is forced to step onto the knots. Solver settings are the paper's (SRA1 rtol 1e-4 atol 1e-6,
Brownian tolerance 1e-12; Tsit5 rtol 1e-7 atol 1e-9; dt0 1e-8).
The knots are the run's slice times: its time grid (train_latent.py --time-grid, config.json 'time_grid'; runs
without it and the paper checkpoint are 'uniform', t_k = 5 (1 - k/99), i.e. lm.forward_time_pts() as always),
checked against the 't' of the run's hidden_history.npz (load_params; a mismatch beyond 1e-12 is refused). Both
samplers, the PF-ODE and --quench use them; the samples' .json records 'time_grid' and the knots 'knots_t'.
Initial states and Brownian keys: latent_model.trajectory_set ('base' = the 500 trajectories of the ode_vs_sde
study; with --run paper, --set base reproduces job 4187940).

Writes out/latent_physical_SGM/<run>/samples/<sampler>_<set>_n<n>.npy (+ .json). Run under Slurm
(jobs/run_sample_latent_neuronic.sh).

--fast (opt-in; reverse SDE only; sample_fast.py): the same SDE, solver, tolerances, Brownian paths and parameter
interpolation, integrated knot interval by knot interval with the hidden-unit fields and forces mixed from the two
neighbouring knots, an analytic visible force, diagonal noise and no progress meter. With its defaults the drift
equals the one below up to rounding. Any rounding-level change of the drift (this one, or scaling the old drift by
1 + 1e-15) changes the step sequence of nearly every trajectory, and final states move by about 1e-4 (four-digit
Potts 784), rarely onto another digit (1 of 500 at K = 3072); FID, KID and multiclass rates are unchanged (jobs
4223298, 4223460; the old sampler run twice agrees to about 1e-15 in most trajectories, job 4223512). Options of the
fast path only:
--brownian-tol (default 1e-12, as above), --brownian cached (the same Brownian path, one tree descent per step instead
of two), --precision mixed (the two hidden-unit products in float32 at HIGHEST precision), --visible-impl,
--stem-suffix. It writes the same file names (plus <stem>_steps.npz), records its settings in the .json, and refuses
to overwrite an existing <stem>.npy. Without --fast nothing changes.

--init-dist (opt-in, 2026-11-12; default sampler, --sampler sde or ode): the start distribution of the reverse process
at t = T_FORWARD = 5. The default 'set' is the trajectory set's N(0, T I) states, as always (nothing changes, and the
manifest gets no new key). The forward distribution at t = 5 still carries e^-5 of the data (job 4229465: a mean
offset of 0.73 noise SDs along the mean image, 21% more variance along the top principal axis; KL 0.29 nats). Two
starts that remove this mismatch without retraining:
  'gauss'    the Gaussian with the exact mean and covariance of the forward distribution at t = 5,
             N(e^-5 mu, e^-10 C + T (1 - e^-10) I), mu and C those of the run's training data (data_stats; a fixed
             summary of the data, like any trained parameter, cached in out/latent_physical_SGM/data_stats/). The set's
             own draws are mapped onto it (gauss_start), so every trajectory keeps its noise and its Brownian key and
             the samples are paired with the normal ones. --init-rank r: only the top r principal axes get their extra
             variance (r = 0: the mean shift alone, 784 numbers).
  'equil:S'  the set's states after S time units of Langevin dynamics under the model's own free energy at t = 5
             (slice 0, frozen), dx = -grad F_0 dt + sqrt(2T) dW, whose stationary distribution is exp(-F_0 / T)
             (equilibrate): no data at all, only the model (prepare the thermal state of the initial Hamiltonian, then
             run the protocol). Then the usual reverse process with the usual keys.
Writes <stem>_init-gauss.npy, <stem>_init-gauss_r<r>.npy or <stem>_init-equil<S>.npy (+ .json with the settings under
'init_dist'); never the normal samples, and it refuses to overwrite. Not with --fast, --thin-knots or --oracle-start.

Harmonic hidden modes (train_latent.py --harmonic R, 2026-11-12; members V, c of hidden_history.npz): the drift adds
their force s sum_j (v_j . x + c_j) v_j (s = 2 for the SDE, 1 for the PF-ODE, as for every term), V and c interpolated
like every other parameter (make_drift harmonic=...), in sampling, --quench and --init-dist equil:S alike. load_params
refuses such a run unless its caller takes the modes (with_harmonic=True), so that no tool drops their force by
accident; --fast refuses them. The manifest records 'harmonic_modes': R (runs without modes get no new key).

Per-patch Potts units (train_latent.py --patch SIDE:STRIDE:STATES, 2026-11-13; members Wp, bp (and Qp) of
hidden_history.npz): the drift adds their mean force -s grad F_patch = s fold(gbar) (latent_model.patch_terms; s = 2 for
the SDE, 1 for the PF-ODE), Wp, bp, Qp interpolated like every other parameter (make_drift patch=...), in sampling,
--quench and --init-dist equil:S alike. load_params refuses such a run unless its caller takes the units
(with_patch=True); --fast refuses them. The manifest records 'patch_units': the run's spec (runs without them get no
new key).
"""

import argparse
import gc
import hashlib
import io
import json
import math
import os
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import vmap  # noqa: E402

from physical_diffusion_fns.helper_fns import interpolate_parameters  # noqa: E402
from physical_diffusion_fns.network_fns import setup_energy_fn, solve_ODE, solve_SDE  # noqa: E402


def load_params(run, with_harmonic=False, with_patch=False):
    """Returns (params_history (100, n_theta), W (100, K, N), b (100, K), out_dir, provenance, problem).
    provenance['time_grid'] is the run's time grid (config.json 'time_grid'; 'uniform' for runs without it and for
    the paper checkpoint), checked against the 't' of the run's hidden_history.npz (latent_model.checked_time_pts:
    a mismatch beyond 1e-12 is refused); slice_times(provenance) gives the knots.
    A run with harmonic hidden modes (train_latent.py --harmonic R; members V, c of hidden_history.npz) is refused
    unless with_harmonic=True, so that no caller drops their force by accident; with it, provenance['harmonic'] is
    (V (100, R, N), c (100, R)), for make_drift(harmonic=...), and the caller pops it like 'local' and 'quad'. For
    every other run with_harmonic changes nothing.
    Likewise a run with per-patch Potts units (train_latent.py --patch; members Wp, bp, Qp) is refused unless
    with_patch=True; with it, provenance['patch'] is (Wp (100, P, S, d), bp (100, P, S), Qp (100, P, S, d) or None),
    checked against config.json 'patch', for make_drift(patch=...), which the caller pops, and
    provenance['patch_units'] the run's spec. For every other run with_patch changes nothing."""
    if run == "paper":
        sha = lm.sha256_file(lm.PAPER_CKPT)
        if sha != lm.PAPER_CKPT_SHA256:
            raise RuntimeError(f"paper checkpoint SHA mismatch: {sha}")
        th = np.array(np.load(lm.PAPER_CKPT, mmap_mode="r"), dtype=np.float64)
        return (th, np.zeros((th.shape[0], 0, lm.N)), np.zeros((th.shape[0], 0)), os.path.join(lm.OUT_ROOT, "paper_K0"),
                {"params_sha256": sha, "time_grid": "uniform"}, "mnist28")
    run_dir = os.path.join(lm.OUT_ROOT, run)
    if not os.path.exists(os.path.join(run_dir, "DONE")):
        raise FileNotFoundError(f"{run_dir} has no DONE marker (training incomplete)")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    problem = cfg.get("problem", "mnist28")
    hidden = cfg.get("hidden", "ising")
    th = np.load(os.path.join(run_dir, "params_history.npy"))
    hid = np.load(os.path.join(run_dir, "hidden_history.npz"))
    grid = lm.config_time_grid(cfg)                       # 'uniform' unless trained with --time-grid
    lm.checked_time_pts(grid, hid["t"] if "t" in hid.files else None, run_dir)   # refuses other recorded slice times
    prov = {"params_sha256": lm.sha256_file(os.path.join(run_dir, "params_history.npy")), "hidden": hidden,
            "radius": cfg.get("radius"), "neighbourhood": cfg.get("neighbourhood", "chebyshev"), "time_grid": grid}
    if "wl" in hid.files:
        prov["local"] = (hid["wl"], hid["bl"])
    if "Q" in hid.files:
        prov["quad"] = hid["Q"]
    if "V" in hid.files:                                  # harmonic modes (train_latent.py --harmonic)
        if not with_harmonic:
            raise RuntimeError(f"{run_dir} has {hid['V'].shape[1]} harmonic hidden modes (train_latent.py --harmonic): "
                               f"this caller does not pass them to the drift (load_params(run, with_harmonic=True) and "
                               f"make_drift(harmonic=prov['harmonic']) do); the fast sampler (--fast) does not support "
                               f"them")
        prov["harmonic"] = (hid["V"], hid["c"])
    if "Wp" in hid.files:                                 # per-patch Potts units (train_latent.py --patch)
        spec = cfg.get("patch", "")
        if not with_patch:
            raise RuntimeError(f"{run_dir} has per-patch Potts units (train_latent.py --patch {spec}): this caller does "
                               f"not pass them to the drift (load_params(run, with_patch=True) and "
                               f"make_drift(patch=prov['patch']) do); the fast sampler (--fast) does not support them")
        side, stride, S = lm.parse_patch_spec(spec, lm.problem_geometry(problem)[0]) or (0, 0, 0)
        if not S:
            raise RuntimeError(f"{run_dir}: hidden_history.npz has patch units but config.json records no --patch spec")
        geom = lm.PatchGeometry(lm.problem_geometry(problem)[0], side, stride)
        Wp, bp = hid["Wp"], hid["bp"]
        Qp = hid["Qp"] if "Qp" in hid.files else None
        n_sl = th.shape[0]
        if (Wp.shape != (n_sl, geom.P, S, geom.d) or bp.shape != (n_sl, geom.P, S)
                or (Qp is not None and Qp.shape != Wp.shape)):
            raise RuntimeError(f"{run_dir}: patch units of shapes {Wp.shape}, {bp.shape}"
                               + ("" if Qp is None else f", {Qp.shape}") + f" do not match --patch {spec} "
                               f"({n_sl} slices, P {geom.P}, S {S}, d {geom.d})")
        prov["patch"] = (Wp, bp, Qp)
        prov["patch_units"] = lm.patch_spec_str((side, stride, S))
    return th, hid["W"], hid["b"], run_dir, prov, problem


def slice_times(prov):
    """The forward slice times (the knots) of a run loaded by load_params, in slice order (t = 5 first):
    latent_model.time_grid_pts of its time grid prov['time_grid'] (checked there; 'uniform' if absent, which gives
    lm.forward_time_pts() itself)."""
    return lm.time_grid_pts(prov.get("time_grid", "uniform"))


def thin_knot_keep(taus, spec):
    """--thin-knots 'LO:HI' -> (keep, LO, HI): the slice indices kept as knots, i.e. every slice except those with an
    odd index whose time lies strictly between LO and HI (taus: the run's slice times, slice order). The strict bounds
    keep slice 0 (t = 5) and slice 99 (t = 0) for LO = 0, HI = 5. On the uniform grid '0:1' drops slices 81-97 (odd),
    '3.4:5' slices 1-31 (odd): the spacing there doubles from 0.0505 to 0.101."""
    try:
        lo, hi = (float(v) for v in spec.split(":"))
    except ValueError:
        raise SystemExit(f"--thin-knots {spec!r}: expected 'LO:HI', e.g. '0:1'") from None
    taus = np.asarray(taus, dtype=np.float64)
    drop = (taus > lo) & (taus < hi) & (np.arange(len(taus)) % 2 == 1)
    drop[0] = drop[-1] = False
    if not drop.any():
        raise SystemExit(f"--thin-knots {spec!r}: no slice to drop in that range")
    return np.flatnonzero(~drop), lo, hi


def step_knots(taus=None):
    """The step_ts of the reverse-time solve: the 98 interior knots sort(T_FORWARD - taus)[1:-1] (taus: the forward
    slice times; default lm.forward_time_pts(), the uniform grid)."""
    taus = lm.forward_time_pts() if taus is None else np.asarray(taus, dtype=np.float64)
    return jnp.asarray(np.sort(lm.T_FORWARD - taus)[1:-1])


def make_drift(theta_hist, W_hist, b_hist, sampler, problem="mnist28", hidden="ising", local=None, quad=None,
               radius=None, neighbourhood="chebyshev", taus=None, harmonic=None, patch=None):
    """Drift and diffusion of the reverse SDE (sampler 'sde') or the PF-ODE ('ode') of a run's parameter history, the
    parameters interpolated linearly in forward time between the slice times taus (slice order, t = 5 first; default
    lm.forward_time_pts(), the uniform grid; a --time-grid run: slice_times(prov)).
    harmonic = (V (n_slices, R, N), c (n_slices, R)) (load_params(run, with_harmonic=True)'s prov['harmonic']; train_latent
    --harmonic R): the harmonic modes' force s sum_j (v_j . x + c_j) v_j (s = 2 for the SDE, 1 for the PF-ODE, as for
    every term), V and c interpolated like every other parameter. None (default): no modes, the drift as before.
    patch = (Wp (n_slices, P, S, d), bp (n_slices, P, S), Qp (n_slices, P, S, d) or None) (load_params(run,
    with_patch=True)'s prov['patch']; train_latent --patch): the per-patch Potts units' mean force -s grad F_patch
    (latent_model.patch_terms), Wp, bp, Qp interpolated like every other parameter, after the harmonic modes in the
    parameter table (the offsets of every other group are unchanged). None (default): the drift as before."""
    n_theta = theta_hist.shape[1]
    K = W_hist.shape[1]
    grid, r0 = lm.problem_geometry(problem)
    radius = r0 if radius is None else int(radius)       # the run's coupling radius (config "radius"), if not default
    n = grid * grid
    parts = [theta_hist, W_hist.reshape(W_hist.shape[0], -1), b_hist]
    if quad is not None:
        parts += [quad.reshape(quad.shape[0], -1)]
    if local is not None:
        parts += [local[0], local[1]]
    R = 0
    if harmonic is not None:                             # after every other group: the offsets above are unchanged
        V_h, c_h = np.asarray(harmonic[0]), np.asarray(harmonic[1])
        R = V_h.shape[1]
        if V_h.shape != (theta_hist.shape[0], R, n) or c_h.shape != (theta_hist.shape[0], R):
            raise ValueError(f"harmonic modes of shapes {V_h.shape}, {c_h.shape} for a history of "
                             f"{theta_hist.shape[0]} slices in dimension {n}")
        parts += [V_h.reshape(V_h.shape[0], -1), c_h]
    Pp = Sp = dp = 0
    with_qp = False
    if patch is not None:                                # after the modes: the offsets above are unchanged
        Wp_h, bp_h = np.asarray(patch[0]), np.asarray(patch[1])
        Qp_h = None if patch[2] is None else np.asarray(patch[2])
        if Wp_h.ndim != 4 or Wp_h.shape[0] != theta_hist.shape[0]:
            raise ValueError(f"patch units of shape {Wp_h.shape} for a history of {theta_hist.shape[0]} slices")
        _, Pp, Sp, dp = Wp_h.shape
        lm.patch_geometry(n, Pp, dp)                     # raises if the shapes describe no patch grid on n pixels
        if bp_h.shape != (theta_hist.shape[0], Pp, Sp) or (Qp_h is not None and Qp_h.shape != Wp_h.shape):
            raise ValueError(f"patch units of shapes {Wp_h.shape}, {bp_h.shape}"
                             + ("" if Qp_h is None else f", {Qp_h.shape}") + " do not fit together")
        with_qp = Qp_h is not None
        parts += [Wp_h.reshape(Wp_h.shape[0], -1), bp_h.reshape(bp_h.shape[0], -1)]
        if with_qp:
            parts += [Qp_h.reshape(Qp_h.shape[0], -1)]
    stacked = np.concatenate(parts, axis=1)
    taus = lm.forward_time_pts() if taus is None else np.asarray(taus, dtype=np.float64)
    if taus.shape != (stacked.shape[0],):
        raise ValueError(f"slice times of shape {taus.shape} for a history of {stacked.shape[0]} slices")
    interp = interpolate_parameters(jnp.asarray(stacked), jnp.asarray(taus))
    conn = lm.connectivity(grid, radius, neighbourhood)       # the run's neighbourhood (config, default chebyshev)
    _, unflatten = lm.paper_theta_layout(conn.shape[0], n)
    if 4 * n + 3 * conn.shape[0] != n_theta:
        raise ValueError(f"parameter count {n_theta} does not match problem {problem}")
    energy_fn = setup_energy_fn(jnp.asarray(conn), unflatten)
    grad_E = jax.grad(energy_fn)
    th_lin = lm.theta_lin(n_theta, n)
    scale = 2.0 if sampler == "sde" else 1.0

    def drift(t, x, args):
        p = interp(lm.T_FORWARD - t)
        th = p[:n_theta]
        d = -grad_E(x, scale * th - th_lin)
        if K > 0:
            W = p[n_theta:n_theta + K * n].reshape(K, n)
            b = p[n_theta + K * n:n_theta + K * n + K]
            if quad is not None:                       # a_j = b_j + W_j.x + Q_j.x^2; force sum_j m_j (W_j + 2 Q_j x)
                Q = p[n_theta + K * n + K:n_theta + 2 * K * n + K].reshape(K, n)
                a = W @ x + Q @ (x * x) + b
                mm = jnp.tanh(a / lm.TEMP) if hidden == "ising" else jax.nn.softmax(a / lm.TEMP)
                d = d + scale * (mm @ W + 2.0 * x * (mm @ Q))
            elif hidden == "ising":
                d = d + scale * (jnp.tanh((W @ x + b) / lm.TEMP) @ W)
            else:                                      # Potts: mean force sum_m p_m W_m, p = softmax(a / T)
                d = d + scale * (jax.nn.softmax((W @ x + b) / lm.TEMP) @ W)
        if local is not None:                          # one unit per pixel: force tanh(a_i / T) w_i on pixel i
            o = n_theta + K * n + K + (K * n if quad is not None else 0)
            wl, bl = p[o:o + n], p[o + n:o + 2 * n]
            d = d + scale * jnp.tanh((wl * x + bl) / lm.TEMP) * wl
        if R:                                          # harmonic modes: -grad F_h = sum_j (v_j . x + c_j) v_j
            o = n_theta + K * n + K + (K * n if quad is not None else 0) + (2 * n if local is not None else 0)
            Vt, ct = p[o:o + R * n].reshape(R, n), p[o + R * n:o + R * n + R]
            d = d + scale * ((Vt @ x + ct) @ Vt)
        if Pp:                                         # patch units: -grad F_patch = fold(sum_s pi_ps G_ps)
            o = (n_theta + K * n + K + (K * n if quad is not None else 0) + (2 * n if local is not None else 0)
                 + R * n + R)
            m = Pp * Sp * dp
            Wt, bt = p[o:o + m].reshape(Pp, Sp, dp), p[o + m:o + m + Pp * Sp].reshape(Pp, Sp)
            Qt = p[o + m + Pp * Sp:o + 2 * m + Pp * Sp].reshape(Pp, Sp, dp) if with_qp else None
            d = d - scale * lm.patch_terms(Wt, Qt, bt, x[None], lm.TEMP, with_lap=False)[0][0]
        return d

    def diffusion(t, x, args):
        return jnp.sqrt(2.0 * lm.TEMP) * jnp.eye(n)

    return drift, diffusion


# ---------------------------------------------------------------------------------------------------------------
# Start distributions of the reverse process (--init-dist, 2026-11-12)
# ---------------------------------------------------------------------------------------------------------------
DATA_STATS_DIR = os.path.join(lm.OUT_ROOT, "data_stats")   # data_stats' cache: <problem>[_cls<c>].npz
STATS_MU_TOL = 1e-9          # largest |difference| allowed between a cached mean and the data's (rounding only)
EQUIL_KEY_FOLD = 1           # equilibrate's Brownian key: jax.random.fold_in(trajectory key, EQUIL_KEY_FOLD)
INIT_KL_RANKS = (0, 1, 2, 5, 10, 20, 50, 100, 200)        # the ranks of the KL table a gauss start prints


def parse_init_dist(spec):
    """--init-dist SPEC -> (kind, S): ('set', None) the trajectory set's own states (the default, as always);
    ('gauss', None) the Gaussian with the exact mean and covariance of the forward distribution at t = T_FORWARD
    (gauss_start); ('equil', S) the set's states after S > 0 time units of Langevin dynamics under the model's own free
    energy at t = T_FORWARD (equilibrate). Anything else raises ValueError."""
    spec = str(spec).strip()
    if spec in ("set", "gauss"):
        return spec, None
    if spec.startswith("equil:"):
        try:
            s = float(spec[len("equil:"):])
        except ValueError:
            s = float("nan")
        if math.isfinite(s) and s > 0.0:
            return "equil", s
    raise ValueError(f"--init-dist {spec!r}: expected 'set', 'gauss' or 'equil:S' with a duration S > 0 in time units "
                     f"(e.g. 'equil:3')")


def init_suffix(kind, S=None, rank=None):
    """The samples' file-name suffix of a start distribution (parse_init_dist): '' for 'set', '_init-gauss' (full
    covariance), '_init-gauss_r<rank>' (--init-rank), '_init-equil<S>' (S as %g, e.g. _init-equil3)."""
    if kind == "set":
        return ""
    if kind == "gauss":
        return "_init-gauss" + ("" if rank is None else f"_r{int(rank)}")
    return f"_init-equil{S:g}"


def data_stats(problem, cls="none", cache_dir=DATA_STATS_DIR):
    """Mean and covariance of a problem's training images, latent_model.make_problem(problem)['data'] (all of them; or
    those of class `cls` for a run trained with train_latent.py --cls), for --init-dist gauss. The covariance is the
    population covariance C = (1/N) sum_i (x_i - mu)(x_i - mu)^T of the N images, i.e. that of the empirical
    distribution the forward process starts from, kept as its eigen-decomposition: eigenvalues lam (descending) and
    orthonormal eigenvectors U (columns), C = U diag(lam) U^T.

    Cached in <cache_dir>/<problem>[_cls<c>].npz (mu, lam, U, n_images, problem, cls, data_file_sha256, created,
    slurm_job). The file is computed once and written atomically (a temporary file, then os.replace); every caller
    then uses the file as it loads it from disk, so all jobs sample from the same numbers. A cached file is used only
    if it names the same problem and class, the sha256 of the data file the problem is built from
    (latent_model.problem_data_path, recorded by train_latent.py as config.json's data_sha256) matches, it has the same
    number of images, and its mean agrees with the data's to STATS_MU_TOL; otherwise it is recomputed and replaced.
    (The normalised images themselves are not hashed: make_problem computes them on the default device, and the GPU's
    and the CPU's rounding differ in the last digit.)

    Returns dict(mu, lam, U, n_images, problem, cls, data_file_sha256, created, slurm_job, path, file_sha256 (of the
    .npz bytes that were loaded), recomputed (bool))."""
    cls = str(cls)
    prob = lm.make_problem(problem, 1)            # the images do not depend on the coupling radius; 1 keeps it quick
    X = np.asarray(prob["data"], dtype=np.float64)
    if cls != "none":
        X = X[np.asarray(prob["labels"]) == int(cls)]
    file_sha = lm.sha256_file(lm.problem_data_path(problem))
    mu = X.mean(axis=0)
    tag = problem if cls == "none" else f"{problem}_cls{cls}"
    path = os.path.join(cache_dir, f"{tag}.npz")

    def load():
        try:
            with open(path, "rb") as f:
                raw = f.read()
            with np.load(io.BytesIO(raw)) as z:
                st = {k: z[k] for k in z.files}
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            return None
        need = ("mu", "lam", "U", "n_images", "problem", "cls", "data_file_sha256")
        if (any(k not in st for k in need) or str(st["problem"]) != problem or str(st["cls"]) != cls
                or str(st["data_file_sha256"]) != file_sha or int(st["n_images"]) != X.shape[0]
                or st["mu"].shape != mu.shape or st["U"].shape != (mu.size, mu.size)
                or not float(np.max(np.abs(st["mu"] - mu))) <= STATS_MU_TOL):
            return None
        out = {k: st[k] for k in ("mu", "lam", "U")}
        out.update({k: str(st[k]) for k in ("problem", "cls", "data_file_sha256", "created", "slurm_job") if k in st})
        out.update(n_images=int(st["n_images"]), path=path, file_sha256=hashlib.sha256(raw).hexdigest())
        return out

    st = load()
    recomputed = st is None
    if recomputed:
        Xc = X - mu
        lam, U = np.linalg.eigh((Xc.T @ Xc) / X.shape[0])
        order = np.argsort(lam)[::-1]
        os.makedirs(cache_dir, exist_ok=True)
        job = os.environ.get("SLURM_JOB_ID", "manual")
        tmp = os.path.join(cache_dir, f".{tag}.{job}.{os.getpid()}.tmp.npz")
        np.savez(tmp, mu=mu, lam=lam[order], U=U[:, order], n_images=X.shape[0], problem=problem, cls=cls,
                 data_file_sha256=file_sha, created=time.strftime("%Y-%m-%d %H:%M:%S"), slurm_job=job)
        os.replace(tmp, path)
        st = load()
        if st is None:
            raise RuntimeError(f"{path}: the statistics just written do not load or do not match the data")
    st["recomputed"] = recomputed
    return st


def _gauss_axes(stats, t):
    """(v0, e, delta): the default start's variance v0 = T s^2 (s = SIGMA_FORWARD), the decay e = e^{-t/s^2}, and the
    extra variance of the forward distribution at t along every principal axis of the data, delta_k = e^2 (lam_k - v0):
    its covariance is Sigma = e^2 C + v0 (1 - e^2) I = v0 I + U diag(delta) U^T."""
    s2 = lm.SIGMA_FORWARD ** 2
    v0 = lm.TEMP * s2
    e = math.exp(-t / s2)
    return v0, e, e * e * (np.asarray(stats["lam"], dtype=np.float64) - v0)


def gauss_start(x0_set, stats, rank=None, t=lm.T_FORWARD):
    """--init-dist gauss: the trajectory set's start states x0_set (N(0, v0 I) draws, v0 = T SIGMA_FORWARD^2) mapped
    onto the Gaussian with the exact mean and covariance of the forward distribution at forward time t,
        m = e mu,   Sigma = e^2 C + v0 (1 - e^2) I = v0 I + U diag(delta) U^T,   e = e^{-t / SIGMA_FORWARD^2}
    (mu, C = U diag(lam) U^T: data_stats; delta: _gauss_axes). With z = x0_set / sqrt(v0) the states are
    x = m + Sigma^(1/2) z, computed as
        x = m + x0_set + U diag(d) U^T z,   d_k = sqrt(v0 + delta_k) - sqrt(v0) = delta_k / (sqrt(v0 + delta_k) + sqrt(v0)),
    so every trajectory keeps its own noise z (and its Brownian key): the samples stay paired with the normal ones.
    Sigma^(1/2) is the symmetric square root, an exact factor of Sigma (as the Cholesky factor is), chosen because it
    moves every draw the least (it is the optimal-transport map from N(0, v0 I) to N(m, Sigma)) and makes the rank-r
    starts nested. rank r (--init-rank): only the top r principal axes get their extra variance, the others keep v0,
    x = m + x0_set + U_r diag(d_r) U_r^T z: a rank-r start differs from the full one only outside the kept axes;
    r = 0 is the mean shift alone (x = m + x0_set exactly), rank None (default) or r = n the full Gaussian. Returns the
    states, float64 (N, n)."""
    v0, e, delta = _gauss_axes(stats, t)
    r = delta.size if rank is None else int(rank)
    if not 0 <= r <= delta.size:
        raise ValueError(f"rank {r}: expected 0 <= rank <= {delta.size}")
    U = np.asarray(stats["U"], dtype=np.float64)[:, :r]
    d = delta[:r] / (np.sqrt(v0 + delta[:r]) + math.sqrt(v0))
    x0 = np.asarray(x0_set, dtype=np.float64)
    z = x0 / math.sqrt(v0)
    return e * np.asarray(stats["mu"], dtype=np.float64) + x0 + ((z @ U) * d) @ U.T


def gauss_kl(stats, rank=None, t=lm.T_FORWARD, shift=True):
    """(mean part, covariance part) of KL(N(m, Sigma) || start) in nats: how far a start is from the exact Gaussian
    N(m, Sigma) of gauss_start. start = N(m, Sigma_r), the rank-r start (rank None: Sigma itself, KL 0); with
    shift=False the start's mean is 0 instead of m, so rank=0, shift=False is the default start N(0, v0 I) (mnist28_d0123
    at t = 5: 0.266 + 0.0185 nats, as job 4229465)."""
    v0, e, delta = _gauss_axes(stats, t)
    r = delta.size if rank is None else int(rank)
    eps = delta[r:] / v0                         # Sigma_kk / v0 - 1 on the axes where the start keeps v0
    kl_cov = 0.5 * float(np.sum(eps - np.log1p(eps)))
    mu = np.asarray(stats["mu"], dtype=np.float64)
    return (0.0 if shift else 0.5 * e * e * float(mu @ mu) / v0), kl_cov


def start_moments(x, stats, t=lm.T_FORWARD):
    """A diagnostic of start states x (N, n): their mean offset along the data's mean image mu / |mu| in units of
    sqrt(v0) (v0 = T SIGMA_FORWARD^2, the default start's variance), and their variance along the top principal axis
    of the data relative to v0; with the exact values of the forward distribution at t (whose covariance is exactly
    Sigma; mnist28_d0123 at t = 5: 0.73 and 1.21; the trajectory set: 0 and 1 up to sampling error). Non-finite rows
    are left out."""
    v0, e, delta = _gauss_axes(stats, t)
    mu = np.asarray(stats["mu"], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    ok = np.all(np.isfinite(x), axis=1)
    xs = x[ok]
    return {"offset_along_mu_sd": float(np.mean(xs @ (mu / np.linalg.norm(mu)))) / math.sqrt(v0),
            "var_pc1_rel": float(np.var(xs @ np.asarray(stats["U"])[:, 0])) / v0, "n": int(ok.sum()),
            "exact_offset_along_mu_sd": e * float(np.linalg.norm(mu)) / math.sqrt(v0),
            "exact_var_pc1_rel": 1.0 + float(delta[0]) / v0}


def equilibrate(drift_ode, diffusion, x_start, keys, S, chunk, t_rev=0.0):
    """--init-dist equil:S: S time units of overdamped Langevin dynamics under the model's own free energy at reverse
    time t_rev (default 0: slice 0, forward time T_FORWARD), its parameters frozen there,
        dx = -grad F_0(x) ds + sqrt(2T) dW,    -grad F_0(x) = drift_ode(t_rev, x) - x
    (drift_ode: make_drift(..., 'ode'), whose drift is x - grad F, as do_quench uses it at the other end), whose
    stationary distribution is exp(-F_0 / T), the model's own distribution at t = 5: the thermal state of the initial
    Hamiltonian is prepared before the protocol runs. No data are used. The solver is the reverse SDE's (solve_SDE:
    SRA1, rtol RTOL_SDE, atol ATOL_SDE, dt0 DT0, a VirtualBrownianTree over [0, S] at tolerance BROWNIAN_TOL; no
    step_ts, the parameters do not change), in chunks of `chunk` trajectories (the last one padded, as sample does).
    Each trajectory's Brownian path comes from jax.random.fold_in(its key, EQUIL_KEY_FOLD): independent of the path of
    its reverse process, which keeps the key itself. diffusion: make_drift's (sqrt(2T) I). x_start (N, n) float64,
    keys (N, 2) uint32. Returns (the states after time S, info: S, solver settings and statistics, timings)."""
    def ldrift(s, x, args_):
        return drift_ode(t_rev, x, args_) - x

    fn = vmap(lambda y, k: solve_SDE(ldrift, diffusion, y, jr.fold_in(k, EQUIL_KEY_FOLD), 0.0, S, 2, lm.DT0,
                                     rtol=lm.RTOL_SDE, atol=lm.ATOL_SDE, brownian_tolerance=lm.BROWNIAN_TOL))
    x_start = np.asarray(x_start, dtype=np.float64)
    keys = np.asarray(keys)
    n = x_start.shape[0]
    x_eq = np.full_like(x_start, np.nan)
    steps, accepted, secs = np.zeros(n, np.int64), np.zeros(n, np.int64), []
    for c0 in range(0, n, chunk):
        t_c = time.time()
        idx = np.arange(c0, min(c0 + chunk, n))
        pad = np.concatenate([idx, np.repeat(idx[-1:], chunk - idx.size)])
        sol = fn(jnp.asarray(x_start[pad]), jnp.asarray(keys[pad]))
        x_eq[idx] = np.asarray(sol.ys[:, -1, :])[: idx.size]
        steps[idx] = np.asarray(sol.stats["num_steps"])[: idx.size]
        accepted[idx] = np.asarray(sol.stats["num_accepted_steps"])[: idx.size]
        secs.append(time.time() - t_c)
        print(f"equilibration chunk {c0}-{idx[-1]} done in {secs[-1]:.0f} s; accepted steps per trajectory mean "
              f"{accepted[idx].mean():.0f}", flush=True)
    ok = np.all(np.isfinite(x_eq), axis=1)
    disp = np.linalg.norm(x_eq[ok] - x_start[ok], axis=1)
    rejected = steps - accepted
    info = {"S": float(S), "t_rev": float(t_rev), "chunk": int(chunk), "solver": "SRA1 (solve_SDE)",
            "rtol": lm.RTOL_SDE, "atol": lm.ATOL_SDE, "dt0": lm.DT0, "brownian_tol": lm.BROWNIAN_TOL,
            "key": f"jax.random.fold_in(trajectory key, {EQUIL_KEY_FOLD})",
            "steps_accepted_mean": float(accepted.mean()), "steps_accepted_min": int(accepted.min()),
            "steps_accepted_max": int(accepted.max()), "steps_rejected_mean": float(rejected.mean()),
            "steps_rejected_max": int(rejected.max()), "chunk_seconds": secs,
            "median_displacement": float(np.median(disp)) if disp.size else float("nan"),
            "n_nonfinite": int(np.sum(~ok))}
    return x_eq, info


def init_start(args, kind, S, problem, run_dir, prov, hist, taus, diffusion, harmonic=None, patch=None):
    """--init-dist gauss / equil:S (main): the start states at reverse time 0 of the trajectory set's first args.n
    trajectories, whose Brownian keys stay theirs (so the samples are paired with the normal ones), and the manifest
    record (prov['init_dist']). hist = (theta_hist, W_hist, b_hist, local, quad) as make_drift takes them; diffusion:
    make_drift's; harmonic, patch: the run's harmonic modes and patch units for make_drift (None: none). gauss:
    data_stats of the run's training data (refused if the run's config.json records another data file hash),
    gauss_start with --init-rank, the KL table of gauss_kl. equil: equilibrate under the PF-ODE drift of the run (make_drift 'ode') at reverse time 0,
    i.e. slice 0; the data statistics only for the diagnostic start_moments (where the states sit before and after,
    against the exact forward distribution), never for the start itself."""
    grid, _ = lm.problem_geometry(problem)
    x0, keys = lm.trajectory_set(args.set, args.n, problem, grid * grid)
    if x0.shape[0] < args.n:
        raise SystemExit(f"set {args.set} has only {x0.shape[0]} trajectories for {problem}; --n {args.n} is too large")
    x0, keys = np.asarray(x0[: args.n], dtype=np.float64), np.asarray(keys[: args.n])
    cfg = {}
    if os.path.exists(os.path.join(run_dir, "config.json")):
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
    cls = str(cfg.get("cls", "none"))
    file_sha = lm.sha256_file(lm.problem_data_path(problem))
    same_data = cfg.get("data_sha256") in (None, file_sha)
    if kind == "gauss" and not same_data:
        raise SystemExit(f"--init-dist gauss: {run_dir} was trained on a data file with sha256 {cfg['data_sha256']}, "
                         f"but {lm.problem_data_path(problem)} now has {file_sha}")
    st = data_stats(problem, cls)
    rec = {"t": lm.T_FORWARD, "stats_file": st["path"], "stats_sha256": st["file_sha256"],
           "stats_recomputed": st["recomputed"], "data_file_sha256": st["data_file_sha256"],
           "n_images": st["n_images"], "cls": cls}
    if kind == "gauss":
        x_init = gauss_start(x0, st, args.init_rank)
        n = st["mu"].size
        r = n if args.init_rank is None else int(args.init_rank)
        kl_default, kl_used = gauss_kl(st, 0, shift=False), gauss_kl(st, args.init_rank)
        v0, e, delta = _gauss_axes(st, lm.T_FORWARD)
        table = {str(k): sum(gauss_kl(st, k)) for k in INIT_KL_RANKS if k < n}
        moments = start_moments(x_init, st)
        rec = {"kind": "gauss", "rank": "full" if args.init_rank is None else r,
               "numbers_used": n + r * n - r * (r - 1) // 2,     # the mean, and a symmetric rank-r correction of T I
               "factor": "symmetric square root of Sigma = e^-10 C + T (1 - e^-10) I (gauss_start)",
               "z": "x0_set / sqrt(T SIGMA_FORWARD^2): the trajectory set's own draws, keys unchanged", **rec,
               "mean_shift_sd": e * float(np.linalg.norm(st["mu"])) / math.sqrt(v0),
               "var_excess_top5": [float(v) for v in delta[:5] / v0],
               "kl_default_start_nats": list(kl_default), "kl_this_start_nats": list(kl_used),
               "kl_by_rank_nats": table, "start_moments": moments}
        print(f"--init-dist gauss, rank {rec['rank']} ({rec['numbers_used']} numbers): start at t = {lm.T_FORWARD:g} from "
              f"N(e^-t mu, e^-2t C + T (1 - e^-2t) I) of {problem} ({st['n_images']} images; {st['path']}, sha256 "
              f"{st['file_sha256'][:16]}, {'computed now' if st['recomputed'] else 'cached'}); mean shift "
              f"{rec['mean_shift_sd']:.3f} sd, extra variance on the top 5 axes "
              f"{np.array2string(delta[:5] / v0, precision=3)} x T", flush=True)
        print(f"  KL(exact Gaussian || start), nats: default start {kl_default[0]:.4f} (mean) + {kl_default[1]:.4f} "
              f"(cov); this start {kl_used[1]:.4f}; by rank " + ", ".join(f"{k}: {v:.4f}" for k, v in table.items()),
              flush=True)
        print(f"  start states: offset along mu {moments['offset_along_mu_sd']:.3f} sd (exact "
              f"{moments['exact_offset_along_mu_sd']:.3f}), variance along PC1 {moments['var_pc1_rel']:.3f} x T (exact "
              f"{moments['exact_var_pc1_rel']:.3f}), {moments['n']} states", flush=True)
        return x_init, rec
    theta_hist, W_hist, b_hist, local_p, quad_p = hist
    drift_ode, _ = make_drift(theta_hist, W_hist, b_hist, "ode", problem, prov.get("hidden", "ising"), local_p, quad_p,
                              prov.get("radius"), prov.get("neighbourhood", "chebyshev"), taus, harmonic=harmonic,
                              patch=patch)
    print(f"--init-dist equil:{S:g}: {S:g} time units of Langevin dynamics dx = -grad F(x; slice 0, t = "
          f"{lm.T_FORWARD:g}) ds + sqrt(2T) dW from the set's states (keys fold_in(key, {EQUIL_KEY_FOLD}))", flush=True)
    x_init, eq = equilibrate(drift_ode, diffusion, x0, keys, S, args.chunk)
    del drift_ode
    gc.collect()
    jax.clear_caches()                 # the PF-ODE drift's parameter table, before the reverse solve compiles
    before, after = start_moments(x0, st), start_moments(x_init, st)
    rec = {"kind": "equil", **eq, "energy": "the model's free energy at slice 0 (t = 5), parameters frozen",
           "dynamics": "dx = (drift_ode(0, x) - x) ds + sqrt(2T) dW = -grad F_0 ds + sqrt(2T) dW",
           "diagnostic_only": {**rec, "run_data_sha256_matches": bool(same_data), "start_moments_before": before,
                               "start_moments_after": after}}
    print(f"  equilibrated in {sum(eq['chunk_seconds']):.0f} s: accepted steps per trajectory mean "
          f"{eq['steps_accepted_mean']:.0f} (min {eq['steps_accepted_min']}, max {eq['steps_accepted_max']}), rejected "
          f"mean {eq['steps_rejected_mean']:.0f}; median displacement {eq['median_displacement']:.4f}; non-finite "
          f"{eq['n_nonfinite']}", flush=True)
    print(f"  diagnostic (data statistics, not used for the start): offset along mu {before['offset_along_mu_sd']:.3f} "
          f"-> {after['offset_along_mu_sd']:.3f} sd (exact forward distribution {after['exact_offset_along_mu_sd']:.3f}),"
          f" variance along PC1 {before['var_pc1_rel']:.3f} -> {after['var_pc1_rel']:.3f} x T (exact "
          f"{after['exact_var_pc1_rel']:.3f})", flush=True)
    return x_init, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under out/latent_physical_SGM, or 'paper'")
    ap.add_argument("--set", default="base", choices=["base", "ext"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=125)
    ap.add_argument("--sampler", default="sde", choices=["sde", "ode"])
    ap.add_argument("--snapshots", action="store_true",
                    help="also save the states at t = 0, 0.5, ..., 5 (interpolated; the steps are unchanged)")
    ap.add_argument("--quench", default="",
                    help="comma-separated durations s (e.g. 0.05,0.25,1): afterwards, relax every finite sample at zero "
                         "temperature for time s under the learned t = 0 energy, dx/ds = -grad F(x; params at t = 0) "
                         "(cooling the bath at the end); writes <stem>_q<s>.npy. If <stem>.npy exists, the saved samples "
                         "are quenched and not resampled.")
    ap.add_argument("--fast", action="store_true",
                    help="fast reverse-SDE path (sample_fast.py): knot intervals with shared hidden-unit products, "
                         "diagonal noise; same SDE, solver, tolerances and Brownian paths")
    ap.add_argument("--brownian-tol", type=float, default=None,
                    help=f"(--fast) VirtualBrownianTree tolerance, default {lm.BROWNIAN_TOL:g}")
    ap.add_argument("--precision", default="f64", choices=["f64", "mixed"],
                    help="(--fast) mixed: the two hidden-unit products in float32 at HIGHEST precision")
    ap.add_argument("--visible-impl", default="auto", choices=["auto", "stencil", "dense"],
                    help="(--fast) visible gradient: auto = stencil up to 24 lattice offsets, dense above")
    ap.add_argument("--brownian", default="tree", choices=["tree", "cached"],
                    help="(--fast) cached: the same Brownian tree, evaluated once per step instead of twice")
    ap.add_argument("--stem-suffix", default="",
                    help="(--fast) appended to the samples file name, e.g. _fast to keep them next to existing ones")
    ap.add_argument("--thin-knots", default="",
                    help="test of the knot spacing (default sampler only; thin_knot_keep): 'LO:HI' drops every slice "
                         "with an odd index whose time lies strictly between LO and HI, so the knot spacing there "
                         "doubles; the parameters are interpolated linearly across the gap and the solver no longer "
                         "steps onto the dropped knots. Writes <stem>_thin<LO>-<HI>.npy / .json (never the normal "
                         "samples; refuses to overwrite)")
    ap.add_argument("--oracle-start", type=int, default=None,
                    help="diagnostic (default sampler only): start the reverse process at slice K (forward time t_K) "
                         "from the exact forward distribution, x = e^-t x0 + sqrt(T (1 - e^-2t)) xi with x0 training "
                         "images (a fixed choice and noise, the same for every run and every K), instead of at t = 5. "
                         "For a perfect model the samples would not depend on K, so FID against K shows how much error "
                         "the part of the reverse process before t_K adds. Writes <stem>_oracle<K>.npy (never the "
                         "normal samples; refuses to overwrite)")
    ap.add_argument("--init-dist", default="set",
                    help="start distribution of the reverse process at t = 5 (default sampler only; parse_init_dist): "
                         "'set' (default) the trajectory set's N(0, T I) states, as always; 'gauss' the Gaussian with the "
                         "exact mean and covariance of the forward distribution at t = 5, N(e^-5 mu, e^-10 C + T (1 - "
                         "e^-10) I), mu and C those of the run's training data (data_stats, cached in "
                         "out/latent_physical_SGM/data_stats/), onto which the set's own draws are mapped, so every "
                         "trajectory keeps its noise and Brownian key (gauss_start); 'equil:S' the set's states after S "
                         "time units of Langevin dynamics dx = -grad F(x; slice 0) dt + sqrt(2T) dW under the model's own "
                         "free energy at t = 5 (equilibrate; no data). Writes <stem>_init-gauss[_r<r>].npy / "
                         "<stem>_init-equil<S>.npy (never the normal samples; refuses to overwrite)")
    ap.add_argument("--init-rank", type=int, default=None,
                    help="(--init-dist gauss) only the top r principal axes of the data get their extra variance, the "
                         "others keep T: 0 = the mean shift alone; default: the full covariance")
    args = ap.parse_args()
    quench = [float(s) for s in args.quench.split(",") if s.strip()]
    if args.oracle_start is not None and (args.fast or args.thin_knots):
        ap.error("--oracle-start works on the default sampler only, and not together with --thin-knots")
    if args.thin_knots and args.fast:
        ap.error("--thin-knots works on the default sampler only (the fast sampler needs all 100 knots)")
    try:
        init_kind, init_s = parse_init_dist(args.init_dist)
    except ValueError as err:
        ap.error(str(err))
    if args.init_rank is not None and (init_kind != "gauss" or args.init_rank < 0):
        ap.error("--init-rank needs --init-dist gauss and a rank r >= 0")
    if init_kind != "set" and (args.fast or args.thin_knots or args.oracle_start is not None):
        ap.error("--init-dist gauss / equil:S works on the default sampler only, and not together with --thin-knots "
                 "or --oracle-start")
    if args.fast:
        return main_fast(args, quench)
    if (args.brownian_tol is not None or args.precision != "f64" or args.visible_impl != "auto"
            or args.brownian != "tree" or args.stem_suffix):
        ap.error("--brownian-tol, --brownian, --precision, --visible-impl and --stem-suffix need --fast")

    theta_hist, W_hist, b_hist, run_dir, prov, problem = load_params(args.run, with_harmonic=True, with_patch=True)
    taus = slice_times(prov)                  # the run's knots ('uniform': lm.forward_time_pts(), as before)
    out_dir = os.path.join(run_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{args.sampler}_{args.set}_n{args.n}")
    t_start = time.time()
    print(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
          f"jax {jax.__version__}; devices {jax.devices()}; run {args.run}; K {W_hist.shape[1]}; {args.sampler}, "
          f"set {args.set}, n {args.n}; time grid {prov['time_grid']}"
          + (f"; harmonic modes {prov['harmonic'][0].shape[1]}" if "harmonic" in prov else "")
          + (f"; patch units {prov['patch_units']}" if "patch" in prov else ""), flush=True)

    local_p, quad_p = prov.pop("local", None), prov.pop("quad", None)
    harm_p = prov.pop("harmonic", None)       # harmonic modes (train_latent --harmonic): (V, c) histories, or None
    if harm_p is not None:                    # the manifest records their number (runs without modes: no new key)
        prov["harmonic_modes"] = int(harm_p[0].shape[1])
    patch_p = prov.pop("patch", None)         # patch units (train_latent --patch): (Wp, bp, Qp) histories, or None;
    #                                           the manifest records prov['patch_units'] (runs without them: no new key)
    if args.thin_knots:                       # --thin-knots: a subset of the slices as knots (a spacing test)
        keep, lo, hi = thin_knot_keep(taus, args.thin_knots)
        stem += f"_thin{lo:g}-{hi:g}"
        if os.path.exists(stem + ".npy"):
            raise SystemExit(f"{stem}.npy exists; --thin-knots does not overwrite samples")
        dropped = [k for k in range(len(taus)) if k not in set(keep.tolist())]
        theta_hist, W_hist, b_hist, taus = theta_hist[keep], W_hist[keep], b_hist[keep], taus[keep]
        local_p = None if local_p is None else (local_p[0][keep], local_p[1][keep])
        quad_p = None if quad_p is None else quad_p[keep]
        harm_p = None if harm_p is None else (harm_p[0][keep], harm_p[1][keep])
        patch_p = None if patch_p is None else tuple(None if a is None else a[keep] for a in patch_p)
        prov["thin_knots"] = {"spec": args.thin_knots, "dropped_slices": dropped}
        print(f"--thin-knots {args.thin_knots}: {len(dropped)} slices dropped ({dropped}); {len(keep)} knots; "
              f"writes {stem}.npy", flush=True)
    t0, x_init = 0.0, None
    if args.oracle_start is not None:         # --oracle-start: exact forward states at t_K, then the model
        ks = int(args.oracle_start)
        if not 1 <= ks <= len(taus) - 2:
            raise SystemExit(f"--oracle-start {ks}: a slice index in 1..{len(taus) - 2}")
        t_k = float(taus[ks])
        t0 = lm.T_FORWARD - t_k
        stem += f"_oracle{ks}"
        if os.path.exists(stem + ".npy"):
            raise SystemExit(f"{stem}.npy exists; --oracle-start does not overwrite samples")
        prob = lm.make_problem(problem, prov.get("radius"), prov.get("neighbourhood", "chebyshev"))
        data = np.asarray(prob["data"], dtype=np.float64)
        rng = np.random.default_rng(12345)
        pick = rng.choice(data.shape[0], size=args.n, replace=False)
        xi = rng.standard_normal((args.n, data.shape[1]))
        e = np.exp(-t_k)
        x_init = e * data[pick] + np.sqrt(lm.TEMP * (1.0 - e * e)) * xi
        cls = np.bincount(np.asarray(prob["labels"])[pick]).tolist()
        prov["oracle_start"] = {"slice": ks, "t": t_k, "reverse_t0": t0, "x0": "training images, rng 12345",
                                "x0_index": pick.tolist(), "x0_classes": cls}
        print(f"--oracle-start {ks}: exact forward states at t = {t_k:.4f} (reverse time {t0:.4f}) from {args.n} "
              f"training images (classes {cls}); writes {stem}.npy", flush=True)
    if init_kind != "set":                    # --init-dist: another start at t = 5 (init_start), never the normal samples
        n_dim = lm.problem_geometry(problem)[0] ** 2
        if args.init_rank is not None and args.init_rank > n_dim:
            raise SystemExit(f"--init-rank {args.init_rank}: at most the dimension {n_dim} of {problem}")
        stem += init_suffix(init_kind, init_s, args.init_rank)
        if os.path.exists(stem + ".npy"):
            raise SystemExit(f"{stem}.npy exists; --init-dist does not overwrite samples")
        print(f"--init-dist {args.init_dist}" + ("" if args.init_rank is None else f" --init-rank {args.init_rank}")
              + f": writes {stem}.npy", flush=True)
    drift, diffusion = make_drift(theta_hist, W_hist, b_hist, args.sampler, problem, prov.get("hidden", "ising"),
                                  local_p, quad_p, prov.get("radius"), prov.get("neighbourhood", "chebyshev"), taus,
                                  harmonic=harm_p, patch=patch_p)
    if init_kind != "set":
        x_init, prov["init_dist"] = init_start(args, init_kind, init_s, problem, run_dir, prov,
                                               (theta_hist, W_hist, b_hist, local_p, quad_p), taus, diffusion,
                                               harmonic=harm_p, patch=patch_p)
    if quench and os.path.exists(stem + ".npy"):
        x_final = np.load(stem + ".npy")
        print(f"quench only: loaded {stem}.npy ({x_final.shape[0]} samples)", flush=True)
    else:
        x_final = sample(args, drift, diffusion, problem, stem, prov, W_hist, t_start, taus, t0, x_init)
    if quench:
        drift_ode, _ = make_drift(theta_hist, W_hist, b_hist, "ode", problem, prov.get("hidden", "ising"),
                                  local_p, quad_p, prov.get("radius"), prov.get("neighbourhood", "chebyshev"), taus,
                                  harmonic=harm_p, patch=patch_p)
        do_quench(x_final, drift_ode, quench, stem, args, t_start)
    print(f"total {time.time() - t_start:.0f} s", flush=True)


def main_fast(args, quench):
    """--fast: the reverse SDE with sample_fast.FastSampler. Writes <run>/samples/sde_<set>_n<n><suffix>.npy (+ .json
    with the sampler settings and solver statistics, + _steps.npz). Refuses to overwrite an existing <stem>.npy; with
    --quench and an existing <stem>.npy only the quench runs (default-path code, as without --fast)."""
    import sample_fast
    if args.sampler != "sde" or args.snapshots:
        raise SystemExit("--fast samples the reverse SDE only (no --sampler ode, no --snapshots)")
    theta_hist, W_hist, b_hist, run_dir, prov, problem = load_params(args.run, with_harmonic=True, with_patch=True)
    if "harmonic" in prov:                    # train_latent --harmonic: FastSampler has no harmonic force
        raise SystemExit(f"--fast does not support runs with harmonic modes ({args.run} has "
                         f"{prov['harmonic'][0].shape[1]}; train_latent.py --harmonic); sample them with the default "
                         f"sampler (without --fast)")
    if "patch" in prov:                       # train_latent --patch: FastSampler has no patch force
        raise SystemExit(f"--fast does not support runs with patch units ({args.run} has {prov['patch_units']}; "
                         f"train_latent.py --patch); use the default sampler (without --fast)")
    taus = slice_times(prov)                  # the run's knots ('uniform': lm.forward_time_pts(), as before)
    out_dir = os.path.join(run_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{args.sampler}_{args.set}_n{args.n}{args.stem_suffix}")
    t_start = time.time()
    tol = lm.BROWNIAN_TOL if args.brownian_tol is None else args.brownian_tol
    print(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
          f"jax {jax.__version__}; devices {jax.devices()}; run {args.run}; K {W_hist.shape[1]}; fast sde, set "
          f"{args.set}, n {args.n}, chunk {args.chunk}, Brownian tolerance {tol:g} ({args.brownian}), precision "
          f"{args.precision}; time grid {prov['time_grid']}", flush=True)
    local_p, quad_p = prov.pop("local", None), prov.pop("quad", None)
    if quench and os.path.exists(stem + ".npy"):
        x_final = np.load(stem + ".npy")
        print(f"quench only: loaded {stem}.npy ({x_final.shape[0]} samples)", flush=True)
    else:
        if os.path.exists(stem + ".npy"):
            raise SystemExit(f"{stem}.npy exists; --fast does not overwrite samples (choose a --stem-suffix)")
        fs = sample_fast.FastSampler(theta_hist, W_hist, b_hist, problem=problem, hidden=prov.get("hidden", "ising"),
                                     local=local_p, quad=quad_p, radius=prov.get("radius"),
                                     neighbourhood=prov.get("neighbourhood", "chebyshev"), sampler="sde",
                                     precision=args.precision, visible_impl=args.visible_impl, brownian_tol=tol,
                                     brownian=args.brownian, taus=taus)
        x_final = sample_fast.sample(args, fs, problem, stem, prov, W_hist, t_start)
        if quench:                        # free the knot history on the GPU (7.7 GB at K = 6144) before the ODE table
            import gc
            del fs
            gc.collect()
            jax.clear_caches()
    if quench:
        drift_ode, _ = make_drift(theta_hist, W_hist, b_hist, "ode", problem, prov.get("hidden", "ising"),
                                  local_p, quad_p, prov.get("radius"), prov.get("neighbourhood", "chebyshev"), taus)
        do_quench(x_final, drift_ode, quench, stem, args, t_start)
    print(f"total {time.time() - t_start:.0f} s", flush=True)


def do_quench(x_final, drift_ode, durations, stem, args, t_start):
    """Zero-temperature relaxation under the learned t = 0 free energy: the PF-ODE drift at the end of the reverse
    process (t = T_FORWARD, i.e. forward time 0) is x - grad F, so dx/ds = drift_ode(T_FORWARD, x) - x = -grad F."""
    def qdrift(s, x, args_):
        return drift_ode(lm.T_FORWARD, x, args_) - x

    ok = np.all(np.isfinite(x_final), 1)
    idx_ok = np.flatnonzero(ok)
    for s in durations:
        t_q = time.time()
        fn = vmap(lambda y: solve_ODE(qdrift, y, 0.0, s, 2, lm.DT0, rtol=lm.RTOL_ODE, atol=lm.ATOL_ODE))
        xq = np.full_like(x_final, np.nan)
        for c0 in range(0, idx_ok.size, args.chunk):
            idx = idx_ok[c0:c0 + args.chunk]
            pad = np.concatenate([idx, np.repeat(idx[-1:], args.chunk - idx.size)])
            sol = fn(jnp.asarray(x_final[pad]))
            xq[idx] = np.asarray(sol.ys[:, -1, :])[: idx.size]
        name = f"{stem}_q{s:g}"
        np.save(name + ".npy", xq)
        d = np.linalg.norm(xq[ok] - x_final[ok], axis=1)
        info = {"source": os.path.basename(stem) + ".npy", "quench_duration": s, "n_quenched": int(ok.sum()),
                "median_displacement": float(np.median(d)), "max_abs_final": float(np.nanmax(np.abs(xq))),
                "seconds": time.time() - t_q, "slurm_job": os.environ.get("SLURM_JOB_ID", "manual"),
                "sha256_sample_latent": lm.sha256_file(os.path.join(HERE, "sample_latent.py"))}
        with open(name + ".json", "w") as f:
            json.dump(info, f, indent=2)
        print(f"quench s = {s:g}: {int(ok.sum())} samples, median displacement {np.median(d):.4f} (normalised units), "
              f"{time.time() - t_q:.0f} s; saved {name}.npy", flush=True)


def sample(args, drift, diffusion, problem, stem, prov, W_hist, t_start, taus=None, t0=0.0, x_init=None):
    """Solves the reverse SDE (or PF-ODE) for the trajectory set in chunks, stepping onto the knots of the slice times
    taus (default the uniform grid; the same taus as make_drift's), and writes <stem>.npy and <stem>.json.
    t0, x_init (--oracle-start): start at reverse time t0 from the states x_init (the knots after t0 only); the
    Brownian keys stay the trajectory set's. (--init-dist gauss / equil:S: x_init with t0 = 0, init_start.) Defaults:
    reverse time 0 from the trajectory set's states, as always."""
    taus = lm.forward_time_pts() if taus is None else np.asarray(taus, dtype=np.float64)
    knots = step_knots(taus)
    if t0 > 0.0:                                  # --oracle-start: only the knots after the start
        knots = knots[knots > t0 + 1e-9]
    n_save = 11 if args.snapshots else 2
    grid, _ = lm.problem_geometry(problem)
    x0, keys = lm.trajectory_set(args.set, args.n, problem, grid * grid)
    if x0.shape[0] < args.n:
        raise SystemExit(f"set {args.set} has only {x0.shape[0]} trajectories for {problem}; --n {args.n} is too large")
    if x_init is not None:
        x0 = np.asarray(x_init, dtype=np.asarray(x0).dtype)
    if args.sampler == "sde":
        fn = vmap(lambda y, k: solve_SDE(drift, diffusion, y, k, t0, lm.T_FORWARD, n_save, lm.DT0, rtol=lm.RTOL_SDE,
                                         atol=lm.ATOL_SDE, brownian_tolerance=lm.BROWNIAN_TOL, step_ts=knots))
    else:
        fn = vmap(lambda y, k: solve_ODE(drift, y, t0, lm.T_FORWARD, n_save, lm.DT0, rtol=lm.RTOL_ODE, atol=lm.ATOL_ODE,
                                         step_ts=knots))
    x_final = np.full_like(x0, np.nan)
    x_snap = np.full((x0.shape[0], n_save, x0.shape[1]), np.nan, dtype=np.float32) if args.snapshots else None
    chunk_times = []
    for c0 in range(0, args.n, args.chunk):
        t_c = time.time()
        idx = np.arange(c0, min(c0 + args.chunk, args.n))
        pad = np.concatenate([idx, np.repeat(idx[-1:], args.chunk - idx.size)])
        sol = fn(jnp.asarray(x0[pad]), jnp.asarray(keys[pad]))
        x_final[idx] = np.asarray(sol.ys[:, -1, :])[: idx.size]
        if x_snap is not None:
            x_snap[idx] = np.asarray(sol.ys, dtype=np.float32)[: idx.size]
        np.save(stem + ".partial.npy", x_final[: idx[-1] + 1])
        chunk_times.append(time.time() - t_c)
        print(f"chunk {c0}-{idx[-1]} done in {chunk_times[-1]:.0f} s", flush=True)
    np.save(stem + ".npy", x_final)
    if x_snap is not None:
        np.savez(stem + "_snapshots.npz", states=x_snap, t=np.linspace(0.0, lm.T_FORWARD, n_save))
    os.remove(stem + ".partial.npy")
    manifest = {
        "run": args.run, "problem": problem, "K": int(W_hist.shape[1]), "sampler": args.sampler, "set": args.set, "n": args.n,
        "chunk": args.chunk, "rtol": lm.RTOL_SDE if args.sampler == "sde" else lm.RTOL_ODE,
        "atol": lm.ATOL_SDE if args.sampler == "sde" else lm.ATOL_ODE, "brownian_tol": lm.BROWNIAN_TOL, "dt0": lm.DT0,
        "interp": "linear", "step_ts": "98 interior knots", "knots_t": [float(v) for v in taus],
        "chunk_seconds": chunk_times,
        "max_abs_final": float(np.nanmax(np.abs(x_final))), "n_nonfinite": int(np.sum(~np.isfinite(x_final).all(1))),
        "slurm_job": os.environ.get("SLURM_JOB_ID", "manual"), "jax": jax.__version__, **prov,
        "sha256_sample_latent": lm.sha256_file(os.path.join(HERE, "sample_latent.py")),
    }
    with open(stem + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    if problem == "mnist28" and args.run == "paper" and args.set == "base" and args.sampler == "sde" and os.path.exists(lm.UNSPLIT_4187940):
        ref = np.load(lm.UNSPLIT_4187940)[: args.n]
        d = np.max(np.abs(x_final - ref), axis=1)
        print(f"reproduction of job 4187940: median max|diff| {np.median(d):.2e}, max {np.max(d):.2e}; "
              f"{int(np.sum(d < 1e-3))}/{d.size} below 1e-3", flush=True)
    print(f"saved {stem}.npy; total {time.time() - t_start:.0f} s", flush=True)
    return x_final


if __name__ == "__main__":
    main()
