#!/usr/bin/env python3
"""Compare force components on saved final samples, using one CPU thread.

No training, sampling, equilibrium relaxation or model/JAX import is performed.
The diagnostic uses f=-grad(F) at the final learned forward-time slice t=0.
The reverse-SDE drift is x+2f; it is recorded separately from energy forces.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np

from make_parameter_evolution import DEFAULT_OUTPUT, DEFAULT_RUN, connectivity, map_npz_member, signature


COMPONENTS = ["pair", "onsite", "global_hidden", "local_hidden", "hidden_total",
              "visible_total", "net_energy_force", "sde_drift"]
MAP_COMPONENTS = ["pair", "onsite", "global_hidden", "local_hidden", "net_energy_force"]


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decompose(x, params, edges, temperature, hidden):
    """Net force at each visible pixel, after summing all contributing edges."""
    pair = np.zeros_like(x)
    i, j = edges.T
    edge_force = -2.0 * params["c"] * (x[:, i] - x[:, j])
    np.add.at(pair, (slice(None), i), edge_force)
    np.add.at(pair, (slice(None), j), -edge_force)
    onsite = -(params["k"] * x + params["u"] * x ** 3 + params["ell"])
    a = (x @ params["W"].T + (x * x) @ params["Q"].T + params["b"]) / temperature
    if hidden == "potts":
        a -= a.max(axis=1, keepdims=True)
        occupation = np.exp(a)
        occupation /= occupation.sum(axis=1, keepdims=True)
    elif hidden == "ising":
        occupation = np.tanh(a)
    else:
        raise ValueError(f"Unsupported hidden type {hidden}")
    global_hidden = occupation @ params["W"] + 2.0 * x * (occupation @ params["Q"])
    local_hidden = params["wl"] * np.tanh((params["wl"] * x + params["bl"]) / temperature)
    hidden_total = global_hidden + local_hidden
    visible_total = pair + onsite
    total = visible_total + hidden_total
    return dict(zip(COMPONENTS, [pair, onsite, global_hidden, local_hidden, hidden_total,
                                  visible_total, total, x + 2.0 * total]))


def component_energies(x, params, edges, temperature, hidden):
    """Independent scalar energy expressions for directional-derivative checks."""
    pair = np.sum(params["c"] * (x[edges[:, 0]] - x[edges[:, 1]]) ** 2)
    onsite = np.sum(.5 * params["k"] * x ** 2 + .25 * params["u"] * x ** 4 + params["ell"] * x)
    a = (params["W"] @ x + params["Q"] @ (x * x) + params["b"]) / temperature
    if hidden == "potts":
        maximum = np.max(a)
        global_hidden = -temperature * (maximum + np.log(np.exp(a - maximum).sum()))
    else:
        global_hidden = -temperature * np.logaddexp(a, -a).sum()
    al = (params["wl"] * x + params["bl"]) / temperature
    local_hidden = -temperature * np.logaddexp(al, -al).sum()
    return np.array([pair, onsite, global_hidden, local_hidden])


def verify(x, params, edges, temperature, hidden):
    """Check force signs and scale against energies, plus internal-force balance."""
    rng = np.random.default_rng(407)
    worst = 0.0
    for row in x[:3]:
        direction = rng.normal(size=row.size)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        finite_difference = -(component_energies(row + eps * direction, params, edges, temperature, hidden)
                              - component_energies(row - eps * direction, params, edges, temperature, hidden)) / (2 * eps)
        forces = decompose(row[None], params, edges, temperature, hidden)
        analytic = np.array([np.dot(forces[key][0], direction) for key in COMPONENTS[:4]])
        scaled_error = np.max(np.abs(finite_difference - analytic) / (1.0 + np.abs(analytic)))
        worst = max(worst, float(scaled_error))
        if scaled_error > 2e-6:
            raise AssertionError(f"Energy-gradient check failed: {finite_difference}, {analytic}")
        if abs(forces["pair"].sum()) > 1e-10:
            raise AssertionError("Internal pair forces do not sum to zero")
    return {"directional_derivative_samples": min(3, len(x)), "step": 1e-6,
            "maximum_scaled_error": worst, "internal_pair_force_balance": "passed"}


def summarize(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"median": None, "q10": None, "q90": None, "mean": None}
    q10, median, q90 = np.quantile(finite, [.1, .5, .9])
    return {"median": float(median), "q10": float(q10), "q90": float(q90), "mean": float(np.mean(finite))}


def measure(run, samples, batch):
    if not (run / "DONE").is_file():
        raise ValueError("Only a completed run with a DONE marker can be read")
    cfg_path, theta_path, hidden_path = [run / name for name in
                                        ("config.json", "params_history.npy", "hidden_history.npz")]
    sample_info_path = samples.with_suffix(".json")
    inputs = [cfg_path, theta_path, hidden_path, samples, sample_info_path]
    before = {str(p): signature(p) for p in inputs}
    cfg, sample_info = json.loads(cfg_path.read_text()), json.loads(sample_info_path.read_text())
    if cfg.get("visible") != "quartic" or cfg.get("pair_terms") != "lin":
        raise ValueError("This diagnostic expects quartic on-site terms and linear springs only")
    if cfg.get("hidden") not in ("potts", "ising") or cfg.get("K", 0) <= 0:
        raise ValueError("The diagnostic requires global hidden units")
    if sample_info.get("run") != run.name:
        raise ValueError("Sample manifest belongs to a different run")
    theta_sha = digest_file(theta_path)
    if sample_info.get("params_sha256") != theta_sha:
        raise ValueError("Saved samples and visible parameter history do not match")
    theta = np.load(theta_path, mmap_mode="r", allow_pickle=False)
    times = np.asarray(map_npz_member(hidden_path, "t"))
    if not np.isclose(times[-1], 0.0) or not np.all(np.diff(times) < 0):
        raise ValueError("Expected descending forward-time history ending at zero")
    params = {name: np.array(map_npz_member(hidden_path, name)[-1], copy=True) for name in ("W", "b")}
    n = params["W"].shape[1]
    grid = int(np.sqrt(n))
    if grid * grid != n:
        raise ValueError("Expected a square visible lattice")
    if cfg.get("quad"):
        params["Q"] = np.array(map_npz_member(hidden_path, "Q")[-1], copy=True)
    else:
        params["Q"] = np.zeros_like(params["W"])
    if cfg.get("local"):
        params.update({name: np.array(map_npz_member(hidden_path, name)[-1], copy=True) for name in ("wl", "bl")})
    else:
        params.update(wl=np.zeros(n), bl=np.zeros(n))
    edges = connectivity(grid, int(cfg["radius"]), cfg.get("neighbourhood", "chebyshev"))
    p = len(edges)
    if theta.shape != (len(times), 4 * n + 3 * p):
        raise ValueError("Visible history shape does not match the geometry")
    final = np.array(theta[-1], copy=True)
    if np.any(final[2*n:3*n]) or np.any(final[3*n+p:3*n+3*p]):
        raise ValueError("Unexpected nonzero sextic or nonlinear pair coefficients")
    params.update(k=final[:n], u=final[n:2*n], c=final[3*n:3*n+p], ell=final[-n:])
    if not all(np.isfinite(v).all() for v in params.values()):
        raise ValueError("Nonfinite model coefficients")
    x = np.load(samples, mmap_mode="r", allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != n or x.shape[0] != sample_info.get("n"):
        raise ValueError("Sample shape and manifest do not match")
    finite_ids = np.flatnonzero(np.isfinite(x).all(axis=1))
    if not len(finite_ids):
        raise ValueError("No finite configurations are available")
    temperature = float(cfg["temp"])
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    checks = verify(np.array(x[finite_ids[:3]]), params, edges, temperature, cfg["hidden"])
    norms = np.empty((len(finite_ids), len(COMPONENTS)))
    cosine = np.full(len(finite_ids), np.nan)
    cancellation = np.full(len(finite_ids), np.nan)
    sum_squares = np.zeros((len(MAP_COMPONENTS), n))
    for start in range(0, len(finite_ids), batch):
        stop = min(start + batch, len(finite_ids))
        xb = np.array(x[finite_ids[start:stop]])
        forces = decompose(xb, params, edges, temperature, cfg["hidden"])
        if not all(np.isfinite(v).all() for v in forces.values()):
            raise ValueError("Nonfinite force evaluation; no configurations were silently removed")
        for col, key in enumerate(COMPONENTS):
            norms[start:stop, col] = np.linalg.norm(forces[key], axis=1)
        for row, key in enumerate(MAP_COMPONENTS):
            sum_squares[row] += np.sum(forces[key] ** 2, axis=0)
        nv, nh, net = [norms[start:stop, COMPONENTS.index(key)]
                       for key in ("visible_total", "hidden_total", "net_energy_force")]
        dot = np.sum(forces["visible_total"] * forces["hidden_total"], axis=1)
        np.divide(dot, nv * nh, out=cosine[start:stop], where=(nv * nh > 0))
        np.divide(net, nv + nh, out=cancellation[start:stop], where=(nv + nh > 0))
    ratio = np.full(len(finite_ids), np.nan)
    np.divide(norms[:, 0], norms[:, 4], out=ratio, where=(norms[:, 4] > 0))
    for path in inputs:
        if signature(path) != before[str(path)]:
            raise RuntimeError(f"Input changed during measurement: {path}")
    param_digest = hashlib.sha256()
    for key in sorted(params):
        param_digest.update(key.encode())
        param_digest.update(np.ascontiguousarray(params[key]).tobytes())
    data = {"sample_indices": finite_ids, "norms": norms,
            "pixel_rms": np.sqrt(sum_squares / len(finite_ids)),
            "cosine_visible_hidden": np.clip(cosine, -1, 1),
            "cancellation_ratio": cancellation, "pair_hidden_ratio": ratio}
    summary = {key: summarize(norms[:, i]) for i, key in enumerate(COMPONENTS)}
    metadata = {
        "run": run.name, "config": cfg, "K": int(cfg["K"]), "samples_file": samples.name,
        "sample_manifest": sample_info, "sample_sha256": digest_file(samples),
        "params_history_sha256": theta_sha, "final_parameter_arrays_sha256": param_digest.hexdigest(),
        "source_file_stats": {p.name: before[str(p)] for p in inputs},
        "n_samples": int(len(x)), "n_finite": int(len(finite_ids)),
        "n_nonfinite": int(len(x) - len(finite_ids)), "selection": "All finite saved final samples; no quality or amplitude filtering.",
        "grid": grid, "n_edges": p, "forward_time": 0.0, "generation_time": float(times[0]),
        "component_names": COMPONENTS, "pixel_rms_names": MAP_COMPONENTS,
        "hidden_label": "Potts" if cfg["hidden"] == "potts" else "Ising",
        "force_convention": "Energy force -grad F in standardized visible coordinates; vector components sum before taking L2 norm.",
        "sde_convention": "Reverse SDE drift is x + 2 * net_energy_force. Brownian noise is not included as a finite instantaneous force.",
        "norm_definition": "One Euclidean norm over all visible pixels per saved configuration, then statistics across configurations.",
        "pixel_map_definition": "At each pixel, sqrt(mean over saved configurations of squared net component force at that pixel).",
        "hidden_total_definition": "Vector sum of global hidden and local per-pixel hidden forces before computing the norm.",
        "cosine_definition": "Cosine between total visible force (pair+onsite) and total hidden force (global+local).",
        "cancellation_definition": "norm(net_energy_force)/(norm(visible_total)+norm(hidden_total)).",
        "gauge_caveat": "Common Potts W/Q shifts compensated by visible bias/stiffness preserve total F but redistribute global-hidden and on-site forces. Norms describe this saved parameterization, not causal importance.",
        "trajectory_caveat": "Final-time diagnostic only; no intermediate states inferred, no training or sampling run.",
        "summary": summary, "median_pair_hidden_ratio": summarize(ratio)["median"],
        "median_pair_global_hidden_ratio": summarize(np.divide(norms[:, 0], norms[:, 2],
            out=np.full(len(norms), np.nan), where=(norms[:, 2] > 0)))["median"],
        "median_cosine_visible_hidden": summarize(cosine)["median"],
        "median_cancellation_ratio": summarize(cancellation)["median"],
        "verification": checks,
    }
    return data, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--samples", type=Path, help="defaults to this run's samples/sde_base_n2000.npy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    run = args.run.resolve()
    samples = (args.samples or run / "samples/sde_base_n2000.npy").resolve()
    output = args.output_dir.resolve()
    if output == run or run in output.parents:
        parser.error("Write figures outside the checkpoint directory")
    data, metadata = measure(run, samples, args.batch)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "force_comparison_final_data.npz"
    np.savez_compressed(cache, **data)
    metadata["figure_data_sha256"] = digest_file(cache)
    from force_comparison_layout import render_force_comparison
    render_force_comparison(data, metadata, output)
    (output / "force_comparison_final_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Final-time force comparison: {metadata['n_finite']} of {metadata['n_samples']} saved samples.")
    for key, values in metadata["summary"].items():
        print(f"  {key}: median norm {values['median']:.6g}, 10–90% {values['q10']:.6g}–{values['q90']:.6g}")
    print(f"  Median pair/hidden norm ratio: {metadata['median_pair_hidden_ratio']:.6g}")
    print(f"  Median visible/hidden cosine: {metadata['median_cosine_visible_hidden']:.6g}")
    print(f"  Derivative check maximum scaled error: {metadata['verification']['maximum_scaled_error']:.3g}")
    print(f"Outputs: {output}; no training or sampling was run.")


if __name__ == "__main__":
    main()
