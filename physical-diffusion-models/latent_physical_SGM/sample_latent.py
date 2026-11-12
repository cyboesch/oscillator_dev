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
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import vmap  # noqa: E402

from physical_diffusion_fns.helper_fns import interpolate_parameters  # noqa: E402
from physical_diffusion_fns.network_fns import setup_energy_fn, solve_ODE, solve_SDE  # noqa: E402


def load_params(run):
    """Returns (params_history (100, n_theta), W (100, K, N), b (100, K), out_dir, provenance, problem).
    provenance['time_grid'] is the run's time grid (config.json 'time_grid'; 'uniform' for runs without it and for
    the paper checkpoint), checked against the 't' of the run's hidden_history.npz (latent_model.checked_time_pts:
    a mismatch beyond 1e-12 is refused); slice_times(provenance) gives the knots."""
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
               radius=None, neighbourhood="chebyshev", taus=None):
    """Drift and diffusion of the reverse SDE (sampler 'sde') or the PF-ODE ('ode') of a run's parameter history, the
    parameters interpolated linearly in forward time between the slice times taus (slice order, t = 5 first; default
    lm.forward_time_pts(), the uniform grid; a --time-grid run: slice_times(prov))."""
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
        return d

    def diffusion(t, x, args):
        return jnp.sqrt(2.0 * lm.TEMP) * jnp.eye(n)

    return drift, diffusion


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
    args = ap.parse_args()
    quench = [float(s) for s in args.quench.split(",") if s.strip()]
    if args.thin_knots and args.fast:
        ap.error("--thin-knots works on the default sampler only (the fast sampler needs all 100 knots)")
    if args.fast:
        return main_fast(args, quench)
    if (args.brownian_tol is not None or args.precision != "f64" or args.visible_impl != "auto"
            or args.brownian != "tree" or args.stem_suffix):
        ap.error("--brownian-tol, --brownian, --precision, --visible-impl and --stem-suffix need --fast")

    theta_hist, W_hist, b_hist, run_dir, prov, problem = load_params(args.run)
    taus = slice_times(prov)                  # the run's knots ('uniform': lm.forward_time_pts(), as before)
    out_dir = os.path.join(run_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{args.sampler}_{args.set}_n{args.n}")
    t_start = time.time()
    print(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
          f"jax {jax.__version__}; devices {jax.devices()}; run {args.run}; K {W_hist.shape[1]}; {args.sampler}, "
          f"set {args.set}, n {args.n}; time grid {prov['time_grid']}", flush=True)

    local_p, quad_p = prov.pop("local", None), prov.pop("quad", None)
    if args.thin_knots:                       # --thin-knots: a subset of the slices as knots (a spacing test)
        keep, lo, hi = thin_knot_keep(taus, args.thin_knots)
        stem += f"_thin{lo:g}-{hi:g}"
        if os.path.exists(stem + ".npy"):
            raise SystemExit(f"{stem}.npy exists; --thin-knots does not overwrite samples")
        dropped = [k for k in range(len(taus)) if k not in set(keep.tolist())]
        theta_hist, W_hist, b_hist, taus = theta_hist[keep], W_hist[keep], b_hist[keep], taus[keep]
        local_p = None if local_p is None else (local_p[0][keep], local_p[1][keep])
        quad_p = None if quad_p is None else quad_p[keep]
        prov["thin_knots"] = {"spec": args.thin_knots, "dropped_slices": dropped}
        print(f"--thin-knots {args.thin_knots}: {len(dropped)} slices dropped ({dropped}); {len(keep)} knots; "
              f"writes {stem}.npy", flush=True)
    drift, diffusion = make_drift(theta_hist, W_hist, b_hist, args.sampler, problem, prov.get("hidden", "ising"),
                                  local_p, quad_p, prov.get("radius"), prov.get("neighbourhood", "chebyshev"), taus)
    if quench and os.path.exists(stem + ".npy"):
        x_final = np.load(stem + ".npy")
        print(f"quench only: loaded {stem}.npy ({x_final.shape[0]} samples)", flush=True)
    else:
        x_final = sample(args, drift, diffusion, problem, stem, prov, W_hist, t_start, taus)
    if quench:
        drift_ode, _ = make_drift(theta_hist, W_hist, b_hist, "ode", problem, prov.get("hidden", "ising"),
                                  local_p, quad_p, prov.get("radius"), prov.get("neighbourhood", "chebyshev"), taus)
        do_quench(x_final, drift_ode, quench, stem, args, t_start)
    print(f"total {time.time() - t_start:.0f} s", flush=True)


def main_fast(args, quench):
    """--fast: the reverse SDE with sample_fast.FastSampler. Writes <run>/samples/sde_<set>_n<n><suffix>.npy (+ .json
    with the sampler settings and solver statistics, + _steps.npz). Refuses to overwrite an existing <stem>.npy; with
    --quench and an existing <stem>.npy only the quench runs (default-path code, as without --fast)."""
    import sample_fast
    if args.sampler != "sde" or args.snapshots:
        raise SystemExit("--fast samples the reverse SDE only (no --sampler ode, no --snapshots)")
    theta_hist, W_hist, b_hist, run_dir, prov, problem = load_params(args.run)
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


def sample(args, drift, diffusion, problem, stem, prov, W_hist, t_start, taus=None):
    """Solves the reverse SDE (or PF-ODE) for the trajectory set in chunks, stepping onto the knots of the slice times
    taus (default the uniform grid; the same taus as make_drift's), and writes <stem>.npy and <stem>.json."""
    taus = lm.forward_time_pts() if taus is None else np.asarray(taus, dtype=np.float64)
    knots = step_knots(taus)
    n_save = 11 if args.snapshots else 2
    grid, _ = lm.problem_geometry(problem)
    x0, keys = lm.trajectory_set(args.set, args.n, problem, grid * grid)
    if x0.shape[0] < args.n:
        raise SystemExit(f"set {args.set} has only {x0.shape[0]} trajectories for {problem}; --n {args.n} is too large")
    if args.sampler == "sde":
        fn = vmap(lambda y, k: solve_SDE(drift, diffusion, y, k, 0.0, lm.T_FORWARD, n_save, lm.DT0, rtol=lm.RTOL_SDE,
                                         atol=lm.ATOL_SDE, brownian_tolerance=lm.BROWNIAN_TOL, step_ts=knots))
    else:
        fn = vmap(lambda y, k: solve_ODE(drift, y, 0.0, lm.T_FORWARD, n_save, lm.DT0, rtol=lm.RTOL_ODE, atol=lm.ATOL_ODE,
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
