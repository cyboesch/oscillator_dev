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
Initial states and Brownian keys: latent_model.trajectory_set ('base' = the 500 trajectories of the ode_vs_sde
study; with --run paper, --set base reproduces job 4187940).

Writes out/latent_physical_SGM/<run>/samples/<sampler>_<set>_n<n>.npy (+ .json). Run under Slurm
(run_sample_latent_neuronic.sh).
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
    """Returns (params_history (100, n_theta), W (100, K, N), b (100, K), out_dir, provenance, problem)."""
    if run == "paper":
        sha = lm.sha256_file(lm.PAPER_CKPT)
        if sha != lm.PAPER_CKPT_SHA256:
            raise RuntimeError(f"paper checkpoint SHA mismatch: {sha}")
        th = np.array(np.load(lm.PAPER_CKPT, mmap_mode="r"), dtype=np.float64)
        return (th, np.zeros((th.shape[0], 0, lm.N)), np.zeros((th.shape[0], 0)), os.path.join(lm.OUT_ROOT, "paper_K0"),
                {"params_sha256": sha}, "mnist28")
    run_dir = os.path.join(lm.OUT_ROOT, run)
    if not os.path.exists(os.path.join(run_dir, "DONE")):
        raise FileNotFoundError(f"{run_dir} has no DONE marker (training incomplete)")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    problem = cfg.get("problem", "mnist28")
    hidden = cfg.get("hidden", "ising")
    th = np.load(os.path.join(run_dir, "params_history.npy"))
    hid = np.load(os.path.join(run_dir, "hidden_history.npz"))
    prov = {"params_sha256": lm.sha256_file(os.path.join(run_dir, "params_history.npy")), "hidden": hidden,
            "radius": cfg.get("radius"), "neighbourhood": cfg.get("neighbourhood", "chebyshev")}
    if "wl" in hid.files:
        prov["local"] = (hid["wl"], hid["bl"])
    if "Q" in hid.files:
        prov["quad"] = hid["Q"]
    return th, hid["W"], hid["b"], run_dir, prov, problem


def make_drift(theta_hist, W_hist, b_hist, sampler, problem="mnist28", hidden="ising", local=None, quad=None,
               radius=None, neighbourhood="chebyshev"):
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
    interp = interpolate_parameters(jnp.asarray(stacked), jnp.asarray(lm.forward_time_pts()))
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
    args = ap.parse_args()

    theta_hist, W_hist, b_hist, run_dir, prov, problem = load_params(args.run)
    out_dir = os.path.join(run_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{args.sampler}_{args.set}_n{args.n}")
    t_start = time.time()
    print(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
          f"jax {jax.__version__}; devices {jax.devices()}; run {args.run}; K {W_hist.shape[1]}; {args.sampler}, "
          f"set {args.set}, n {args.n}", flush=True)

    drift, diffusion = make_drift(theta_hist, W_hist, b_hist, args.sampler, problem, prov.get("hidden", "ising"),
                                  prov.pop("local", None), prov.pop("quad", None), prov.get("radius"),
                                  prov.get("neighbourhood", "chebyshev"))
    knots = jnp.asarray(np.sort(lm.T_FORWARD - lm.forward_time_pts())[1:-1])
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
        "interp": "linear", "step_ts": "98 interior knots", "chunk_seconds": chunk_times,
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


if __name__ == "__main__":
    main()
