#!/usr/bin/env python3
"""Save one real reverse-SDE trajectory and its conditional Potts probabilities.

CPU-only, low-memory visualization sampling: stream adjacent trained parameter
knots and integrate each interval with the production SRA1 solver/tolerances.
A single global VirtualBrownianTree is reconstructed from the same key on every
interval. Only the adaptive solver is restarted; the Brownian path is not.
No checkpoint or existing sample files are modified.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import diffrax
import lineax

from make_parameter_evolution import DEFAULT_RUN, HERE, connectivity, map_npz_member, signature
from make_force_comparison import decompose, digest_file


def make_drift(edges, temperature):
    i, j = jnp.asarray(edges[:, 0]), jnp.asarray(edges[:, 1])

    def drift(s, x, args):
        s0, s1, p0, p1 = args
        alpha = (s - s0) / (s1 - s0)
        p = jax.tree_util.tree_map(lambda a, b: a + alpha * (b - a), p0, p1)
        edge = -2.0 * p["c"] * (x[i] - x[j])
        pair = jnp.zeros_like(x).at[i].add(edge).at[j].add(-edge)
        onsite = -(p["k"] * x + p["u"] * x ** 3 + p["ell"])
        a = p["W"] @ x + p["Q"] @ (x * x) + p["b"]
        occupation = jax.nn.softmax(a / temperature)
        global_hidden = occupation @ p["W"] + 2.0 * x * (occupation @ p["Q"])
        local = p["wl"] * jnp.tanh((p["bl"] + p["wl"] * x) / temperature)
        return x + 2.0 * (pair + onsite + global_hidden + local)

    return drift


def make_interval_solver(drift, temperature, horizon, n, rtol, atol, brownian_tol, save_per_interval):
    def diffusion(s, x, args):
        return lineax.DiagonalLinearOperator(jnp.full_like(x, jnp.sqrt(2.0 * temperature)))

    @jax.jit
    def solve(y0, s0, s1, p0, p1, key):
        brownian = diffrax.VirtualBrownianTree(
            0.0, horizon, brownian_tol, (n,), key, diffrax.SpaceTimeLevyArea
        )
        terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian))
        solution = diffrax.diffeqsolve(
            terms, diffrax.SRA1(), t0=s0, t1=s1, dt0=1e-8, y0=y0,
            args=(s0, s1, p0, p1),
            saveat=diffrax.SaveAt(ts=jnp.linspace(s0, s1, save_per_interval)),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=1_000_000, throw=True,
        )
        return (solution.ys, solution.stats["num_steps"],
                solution.stats["num_accepted_steps"], solution.stats["num_rejected_steps"])

    return solve


def posterior(x, params, temperature):
    a = (params["W"] @ x + params["Q"] @ (x * x) + params["b"]) / temperature
    a -= np.max(a)
    p = np.exp(a)
    return p / p.sum()


def blend(p0, p1, alpha):
    return {key: p0[key] + alpha * (p1[key] - p0[key]) for key in p0}


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "figure_data/potts_video")
    parser.add_argument("--index", type=int, default=0, help="Index within the existing base-2000 trajectory set")
    parser.add_argument("--base-size", type=int, default=2000)
    parser.add_argument("--save-per-interval", type=int, default=5,
                        help="Includes both endpoints; 5 over 99 intervals gives 397 unique frames")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--brownian-tol", type=float, default=1e-12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.index < args.base_size or args.save_per_interval < 2:
        parser.error("Invalid trajectory index/base size or save count")
    if min(args.rtol, args.atol, args.brownian_tol) <= 0:
        parser.error("All tolerances must be positive")
    run, output = args.run.resolve(), args.output_dir.resolve()
    if output == run or run in output.parents:
        parser.error("Write visualization outputs outside the trained checkpoint directory")
    stem = f"trajectory_{args.index:03d}"
    final_path, info_path = output / f"{stem}.npz", output / f"{stem}.json"
    partial_path, partial_info = output / f"{stem}.partial.npz", output / f"{stem}.partial.json"
    if not args.overwrite and any(p.exists() for p in (final_path, info_path, partial_path, partial_info)):
        parser.error("Visualization files already exist; choose a new output directory or explicitly --overwrite")
    if not (run / "DONE").is_file():
        parser.error("The trained run must have a DONE marker")
    source_names = ["config.json", "params_history.npy", "hidden_history.npz", "state_class.npy", "train_log.txt"]
    sources = [run / name for name in source_names]
    source_stats = {p.name: signature(p) for p in sources}
    cfg = json.loads((run / "config.json").read_text())
    if (cfg.get("hidden"), cfg.get("quad"), cfg.get("local"), cfg.get("visible"), cfg.get("pair_terms")) != (
            "potts", True, True, "quartic", "lin"):
        parser.error("This visualization sampler expects global Potts + Q, local Ising, quartic on-site and linear springs")
    temperature = float(cfg["temp"])
    arrays = {key: map_npz_member(run / "hidden_history.npz", key) for key in ("W", "Q", "b", "wl", "bl", "t")}
    theta = np.load(run / "params_history.npy", mmap_mode="r", allow_pickle=False)
    tau = np.array(arrays["t"], copy=True)
    if not np.all(np.diff(tau) < 0) or not np.isclose(tau[-1], 0.0):
        parser.error("Expected descending forward times ending at zero")
    horizon = float(tau[0])
    generation_knots = horizon - tau
    nt, m, n = arrays["W"].shape
    grid = int(np.sqrt(n))
    edges = connectivity(grid, int(cfg["radius"]), cfg.get("neighbourhood", "chebyshev"))
    n_edges = len(edges)
    if grid * grid != n or theta.shape != (nt, 4 * n + 3 * n_edges):
        parser.error("Checkpoint geometry and visible parameter layout disagree")
    if np.any(theta[:, 2*n:3*n]) or np.any(theta[:, 3*n+n_edges:3*n+3*n_edges]):
        parser.error("Unexpected sextic or nonlinear pair coefficients")
    state_classes = np.load(run / "state_class.npy", allow_pickle=False)
    if state_classes.shape != (m,):
        parser.error("Initial state-class labels have the wrong shape")
    text = (run / "train_log.txt").read_text()
    normalization = re.search(r"normalisation mean ([+\-\d.eE]+) std ([+\-\d.eE]+)", text)
    if normalization is None:
        parser.error("No saved normalization statistics found for displaying image pixels")
    image_mean, image_std = map(float, normalization.groups())
    reference_path = run / "samples" / f"sde_base_n{args.base_size}.npy"
    reference_info_path = reference_path.with_suffix(".json")
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    reference_info = json.loads(reference_info_path.read_text())
    if reference.shape != (args.base_size, n) or reference_info.get("n") != args.base_size:
        parser.error("Reference sample set does not match the requested base size")
    theta_hash = digest_file(run / "params_history.npy")
    if reference_info.get("params_sha256") != theta_hash:
        parser.error("Reference samples were made with a different visible parameter history")
    source_stats[reference_path.name] = signature(reference_path)
    source_stats[reference_info_path.name] = signature(reference_info_path)

    def parameters(k):
        row = np.array(theta[k], copy=True)
        p = {key: np.array(arrays[key][k], copy=True) for key in ("W", "Q", "b", "wl", "bl")}
        p.update(k=row[:n], u=row[n:2*n], ell=row[-n:], c=row[3*n:3*n+n_edges])
        if not all(np.isfinite(v).all() for v in p.values()):
            raise ValueError(f"Nonfinite parameters at stored slice {k}")
        return p

    # Match latent_model.trajectory_set for this all-ten-digit problem exactly:
    # requesting one random row instead of the original shape would change it.
    if cfg.get("problem") == "mnist28":
        parser.error("Binary mnist28 uses a different legacy trajectory-set convention")
    initial_key, brownian_key = jr.split(jr.PRNGKey(2026))
    all_initial = np.asarray(jnp.sqrt(temperature) * jr.normal(initial_key, (args.base_size, n)), dtype=np.float64)
    keys = jr.split(brownian_key, args.base_size)
    x0, key = all_initial[args.index].copy(), keys[args.index]
    del all_initial
    drift = make_drift(edges, temperature)
    drift_check = jax.jit(drift)
    errors = []
    p_first = parameters(0)
    for k, alpha, x in [(0, .0, x0), (nt // 2, .37, np.asarray(reference[args.index])),
                         (nt - 2, 1.0, np.asarray(reference[args.index]))]:
        p0, p1 = (p_first, parameters(1)) if k == 0 else (parameters(k), parameters(k + 1))
        s0, s1 = generation_knots[k:k+2]
        actual = np.asarray(drift_check(s0 + alpha * (s1 - s0), jnp.asarray(x),
                                       (s0, s1, p0, p1)))
        expected = decompose(x[None], blend(p0, p1, alpha), edges, temperature, "potts")["sde_drift"][0]
        error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=2e-11, atol=2e-11):
            raise AssertionError(f"JAX drift disagrees with independently checked NumPy force decomposition: {error}")
        errors.append(error)
    del p0, p1, actual, expected
    solve_interval = make_interval_solver(drift, temperature, horizon, n, args.rtol, args.atol,
                                          args.brownian_tol, args.save_per_interval)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metadata = {
        "run": run.name, "config": cfg, "trajectory_index": args.index,
        "initial_set": "base", "initial_set_size": args.base_size, "initial_seed": 2026,
        "brownian_key": np.asarray(key).tolist(), "temperature": temperature,
        "solver": "Diffrax SRA1", "diffrax_version": diffrax.__version__, "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "rtol": args.rtol, "atol": args.atol, "brownian_tolerance": args.brownian_tol,
        "interval_initial_dt": 1e-8, "max_steps_per_interval": 1_000_000,
        "brownian_model": "One global VirtualBrownianTree on [0,5], shared key, SpaceTimeLevyArea; reconstructed identically each interval.",
        "solver_restart": "Adaptive solver restarted at every trained parameter knot. Same SDE, Brownian path, tolerances and linear parameter schedule; numerical endpoint need not be bitwise identical to the production solve.",
        "hidden_model": "Conditional mean force; no discrete hidden-state path or physical transition rates are simulated.",
        "state_labels": "Initial cluster majority labels, not independently reassigned final learned-state labels.",
        "normalization_mean": image_mean, "normalization_std": image_std,
        "normalization_precision": "Rounded training-log values, used only for displaying pixel intensities.",
        "source_file_stats": source_stats, "params_history_sha256": theta_hash,
        "sampler_script_sha256": digest_file(Path(__file__)),
        "jax_numpy_drift_max_abs_errors": errors,
        "reference_sample": str(reference_path), "reference_sample_manifest": reference_info,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "status": "running",
    }
    times, states = [0.0], [x0]
    probabilities = [posterior(x0, p_first, temperature)]
    interval_stats = []
    left = p_first
    y = jnp.asarray(x0)
    print(f"Starting CPU trajectory {args.index}; {m} Potts states; {nt-1} knot intervals; "
          f"{1+(nt-1)*(args.save_per_interval-1)} saved frames.", flush=True)
    print(f"Independent drift checks passed; max absolute error {max(errors):.3g}.", flush=True)

    def save(path):
        atomic_npz(path, generation_time=np.asarray(times), states=np.stack(states),
                   posterior=np.stack(probabilities), state_initial_classes=state_classes,
                   normalization_mean=np.asarray(image_mean), normalization_std=np.asarray(image_std))

    for k in range(nt - 1):
        right = parameters(k + 1)
        s0, s1 = generation_knots[k:k+2]
        interval_started = time.time()
        ys, steps, accepted, rejected = solve_interval(y, s0, s1, left, right, key)
        frame_states = np.asarray(ys)
        if not np.isfinite(frame_states).all():
            raise ValueError(f"Nonfinite trajectory at interval {k}")
        frame_times = np.linspace(s0, s1, args.save_per_interval)
        for row in range(1, args.save_per_interval):
            alpha = row / (args.save_per_interval - 1)
            params = right if row == args.save_per_interval - 1 else blend(left, right, alpha)
            probability = posterior(frame_states[row], params, temperature)
            if not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1.0):
                raise ValueError("Invalid saved posterior")
            times.append(float(frame_times[row]))
            states.append(frame_states[row].copy())
            probabilities.append(probability)
        y = ys[-1]
        left = right
        stats = {"interval": k, "s0": float(s0), "s1": float(s1), "steps": int(steps),
                 "accepted_steps": int(accepted), "rejected_steps": int(rejected),
                 "seconds": time.time() - interval_started}
        interval_stats.append(stats)
        if k == 0 or (k + 1) % 5 == 0 or k == nt - 2:
            print(f"interval {k+1}/{nt-1}, s={s1:.3f}, frames={len(times)}, "
                  f"steps={int(steps)}, accepted={int(accepted)}, "
                  f"elapsed={time.time()-started:.1f}s", flush=True)
        if (k + 1) % 10 == 0 or k == nt - 2:
            save(partial_path)
            partial_metadata = dict(metadata, status="partial", completed_intervals=k+1,
                                    elapsed_seconds=time.time()-started, interval_stats=interval_stats)
            partial_info.write_text(json.dumps(partial_metadata, indent=2) + "\n")
    for path in sources:
        if signature(path) != source_stats[path.name]:
            raise RuntimeError(f"Source changed during visualization sampling: {path}")
    for path in (reference_path, reference_info_path):
        if signature(path) != source_stats[path.name]:
            raise RuntimeError(f"Reference changed during visualization sampling: {path}")
    reference_final = np.asarray(reference[args.index])
    difference = states[-1] - reference_final
    metadata.update(status="complete", elapsed_seconds=time.time()-started, frames=len(times),
                    interval_stats=interval_stats,
                    final_reference_max_abs_difference=float(np.max(np.abs(difference))),
                    final_reference_l2_difference=float(np.linalg.norm(difference)),
                    final_argmax_state=int(np.argmax(probabilities[-1])),
                    final_max_state_probability=float(np.max(probabilities[-1])),
                    final_max_abs_amplitude=float(np.max(np.abs(states[-1]))))
    save(final_path)
    metadata["trajectory_sha256"] = digest_file(final_path)
    info_path.write_text(json.dumps(metadata, indent=2) + "\n")
    partial_path.unlink(missing_ok=True)
    partial_info.unlink(missing_ok=True)
    print(f"Saved {final_path}; elapsed {metadata['elapsed_seconds']:.1f}s; "
          f"final reference max|difference|={metadata['final_reference_max_abs_difference']:.6g}.", flush=True)


if __name__ == "__main__":
    main()
