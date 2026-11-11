#!/usr/bin/env python3
"""Plot saved diffusion-time parameters; never run training or sampling.

Only a completed checkpoint is accepted. Uncompressed NPZ members are mapped
read-only so extracting one Potts state does not load the multi-GB history.
The exported small NPZ and JSON record exactly what is shown in the figure.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
DEFAULT_RUN = (
    PROJECT / "out/latent_physical_SGM/"
    "mnist28_d0123456789_K1536_ema_potts_quad_local_quartic_r1_lin_pooled"
)
DEFAULT_OUTPUT = HERE.parent / "figures"


def map_npz_member(path, key):
    """Map an np.savez array without decompressing or copying other states."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(key + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{key} is compressed; refusing a full-history load")
        with path.open("rb") as stream:
            stream.seek(info.header_offset)
            header = stream.read(30)
            if header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            name_length, extra_length = struct.unpack_from("<HH", header, 26)
            stream.seek(name_length + extra_length, 1)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY version {version}")
            offset = stream.tell()
    if dtype.hasobject:
        raise ValueError("Object arrays are not supported")
    return np.memmap(path, mode="r", dtype=dtype, shape=shape,
                     offset=offset, order="F" if fortran else "C")


def connectivity(grid, radius, neighbourhood):
    """Same sorted i<j, open-boundary order as network_fns.py, without JAX."""
    if neighbourhood not in ("chebyshev", "von_neumann"):
        raise ValueError(f"Unsupported neighbourhood: {neighbourhood}")
    pairs = []
    for row in range(grid):
        for col in range(grid):
            i = row * grid + col
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if neighbourhood == "von_neumann" and abs(dr) + abs(dc) > radius:
                        continue
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < grid and 0 <= cc < grid:
                        j = rr * grid + cc
                        if i < j:
                            pairs.append((i, j))
    return np.asarray(sorted(pairs), dtype=int)


def signature(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_trajectory(run):
    """Use the selected digit-2 replay; do not substitute a different digit."""
    path = HERE.parent / "figure_data/parameter_potts2/trajectory_021.npz"
    if not path.is_file():
        raise FileNotFoundError(
            "The selected digit-2 trajectory is missing; run "
            "run_parameter_trajectory_cpu.sh first or pass --trajectory PATH"
        )
    return path.resolve()


def attach_trajectory(run, data, metadata, path):
    """Attach actual recorded states at the exact displayed parameter times.

    Pixel normalization is shared across every frame. This function only reads
    a saved trajectory; it never samples, interpolates images or changes data.
    """
    path = path.resolve()
    info_path = path.with_suffix(".json")
    before = {p: signature(p) for p in (path, info_path)}
    info = json.loads(info_path.read_text())
    if info.get("status") != "complete" or info.get("run") != run.name:
        raise ValueError("Trajectory must be complete and come from the plotted run")
    if info.get("config") != metadata["config"]:
        raise ValueError("Trajectory configuration differs from the plotted checkpoint")
    trajectory_hash = file_digest(path)
    if info.get("trajectory_sha256") != trajectory_hash:
        raise ValueError("Trajectory checksum differs from its sampling manifest")
    params_hash = file_digest(run / "params_history.npy")
    if info.get("params_history_sha256") != params_hash:
        raise ValueError("Trajectory used a different visible parameter history")
    # The sampler records the hidden-history file identity without re-reading
    # the multi-GB checkpoint for a hash. Retain that check alongside the visible
    # parameter hash and exact saved run configuration.
    hidden_signature = info.get("source_file_stats", {}).get("hidden_history.npz")
    if hidden_signature != signature(run / "hidden_history.npz"):
        raise ValueError("Trajectory used a different hidden parameter history")
    with np.load(path, allow_pickle=False) as archive:
        times = np.asarray(archive["generation_time"], dtype=float)
        states = np.asarray(archive["states"], dtype=float)
        mean = float(archive["normalization_mean"])
        std = float(archive["normalization_std"])
    grid = int(data["grid"])
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or states.shape != (len(times), grid * grid)
            or not np.isfinite(states).all()
            or not np.isfinite(mean) or not np.isfinite(std) or std <= 0):
        raise ValueError("Invalid trajectory states, times or image normalization")
    if not (np.isclose(mean, info.get("normalization_mean", np.nan), rtol=0, atol=1e-12)
            and np.isclose(std, info.get("normalization_std", np.nan), rtol=0, atol=1e-12)):
        raise ValueError("Trajectory normalization differs from its sampling manifest")
    targets = np.asarray(data["generation_time"])[np.asarray(data["slices"], dtype=int)]
    indices = np.asarray([int(np.argmin(np.abs(times - t))) for t in targets])
    if not np.allclose(times[indices], targets, rtol=0, atol=1e-8):
        raise ValueError("Saved trajectory has no exact frames at the displayed parameter times")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Displayed parameter times must select distinct trajectory frames")
    data["trajectory_states"] = states[indices].copy()
    data["trajectory_images"] = np.clip(states[indices] * std + mean, 0, 1).reshape(-1, grid, grid)
    data["trajectory_times"] = times[indices].copy()
    data["trajectory_frame_indices"] = indices
    selection_path = path.parent / "selection.json"
    selection = (json.loads(selection_path.read_text()) if selection_path.is_file() else
                 {"description": "Existing saved trajectory; no new sample-quality selection by the renderer"})
    metadata["trajectory_panel"] = {
        "source": os.path.relpath(path, PROJECT),
        "manifest": os.path.relpath(info_path, PROJECT),
        "trajectory_sha256": trajectory_hash,
        "manifest_sha256": file_digest(info_path),
        "run": info["run"],
        "params_history_sha256": params_hash,
        "hidden_history_signature": hidden_signature,
        "trajectory_index": info["trajectory_index"],
        "initial_set": info.get("initial_set"),
        "initial_set_size": info.get("initial_set_size"),
        "initial_seed": info.get("initial_seed"),
        "selection": selection,
        "frame_indices": indices.tolist(),
        "generation_times": times[indices].tolist(),
        "forward_times": np.asarray(data["forward_time"])[data["slices"]].tolist(),
        "normalization_mean": mean,
        "normalization_std": std,
        "display": "clip(x * std + mean, 0, 1); shared grayscale limits, 0 black and 1 white",
        "image_interpolation": "none; recorded states at exact parameter times, original pixel grid",
        "solver": info.get("solver"),
        "solver_restart": info.get("solver_restart"),
        "final_reference_max_abs_difference": info.get("final_reference_max_abs_difference"),
    }
    for source, original in before.items():
        if signature(source) != original:
            raise RuntimeError(f"Trajectory source changed during figure extraction: {source}")


def extract(run, state, indices, patch_start, patch_size, extra_states=()):
    if not (run / "DONE").is_file():
        raise ValueError(f"No DONE marker: {run}")
    inputs = [run / name for name in ("config.json", "hidden_history.npz", "params_history.npy")]
    state_class_path = run / "state_class.npy"
    if state_class_path.is_file():
        inputs.append(state_class_path)
    before = {str(p): signature(p) for p in inputs}
    cfg = json.loads(inputs[0].read_text())
    if cfg.get("hidden") != "potts" or not cfg.get("quad"):
        raise ValueError("This draft expects a Potts model with linear and quadratic couplings")
    w = map_npz_member(inputs[1], "W")
    q = map_npz_member(inputs[1], "Q")
    t = np.array(map_npz_member(inputs[1], "t"), copy=True)
    theta = np.load(inputs[2], mmap_mode="r", allow_pickle=False)
    nt, n_states, n = w.shape
    grid = int(np.sqrt(n))
    if grid * grid != n or q.shape != w.shape or theta.shape[0] != nt:
        raise ValueError("Inconsistent checkpoint shapes")
    selected_states = [state] + list(extra_states)
    if len(selected_states) != len(set(selected_states)):
        raise ValueError("Selected Potts states must be distinct")
    if any(j < 0 or j >= n_states for j in selected_states) or any(i < 0 or i >= nt for i in indices):
        raise ValueError("State or time slice outside the checkpoint")
    initial_classes = np.full(len(selected_states), -1, dtype=int)
    if state_class_path.is_file():
        initial_classes = np.asarray(np.load(state_class_path, allow_pickle=False)[selected_states], dtype=int)
    if t.shape != (nt,) or not np.all(np.diff(t) < 0) or not np.isclose(t[-1], 0):
        raise ValueError("Expected descending forward times ending at zero")
    if indices != sorted(set(indices)) or len(indices) != 5:
        raise ValueError("This layout requires five increasing, distinct slice indices")
    radius = cfg.get("radius")
    if radius is None:
        raise ValueError("An explicit saved connectivity radius is required")
    pairs = connectivity(grid, int(radius), cfg.get("neighbourhood", "chebyshev"))
    if theta.shape[1] != 4 * n + 3 * len(pairs):
        raise ValueError("Visible parameter layout does not match saved geometry")
    row0, col0 = patch_start
    if not (0 <= row0 <= grid - patch_size and 0 <= col0 <= grid - patch_size):
        raise ValueError("Patch outside image")
    nodes = np.asarray([(row0 + r) * grid + col0 + c
                        for r in range(patch_size) for c in range(patch_size)])
    edge_ids = np.flatnonzero(np.isin(pairs[:, 0], nodes) & np.isin(pairs[:, 1], nodes))
    if not len(edge_ids):
        raise ValueError("No visible springs in this patch")
    all_c = np.array(theta[:, 3 * n:3 * n + len(pairs)], copy=True)
    degree = np.bincount(pairs.ravel(), minlength=n)
    if np.any(degree == 0):
        raise ValueError("Per-pixel spring RMS requires at least one edge per pixel")
    rms_maps = np.array([
        np.sqrt(np.bincount(pairs.ravel(), weights=np.repeat(row * row, 2), minlength=n) / degree)
        for row in all_c
    ])
    selected_w = np.stack([np.array(w[:, j, :], copy=True) for j in selected_states], axis=1)
    selected_q = np.stack([np.array(q[:, j, :], copy=True) for j in selected_states], axis=1)
    data = {
        "forward_time": t, "generation_time": t[0] - t,
        "slices": np.asarray(indices), "state": np.asarray(state),
        "states": np.asarray(selected_states), "state_initial_classes": initial_classes,
        "grid": np.asarray(grid), "patch_nodes": nodes,
        "patch_edges": pairs[edge_ids], "patch_edge_indices": edge_ids,
        "all_edges": pairs, "c_lin_all": all_c, "c_rms_map": rms_maps,
        "c_lin": np.array(theta[:, 3 * n + edge_ids], copy=True),
        "W": selected_w[:, 0, :], "Q": selected_q[:, 0, :],
        "W_states": selected_w, "Q_states": selected_q,
        "k_lin": np.array(theta[:, :n], copy=True),
        "k_duff": np.array(theta[:, n:2 * n], copy=True),
    }
    if not all(np.isfinite(data[key]).all() for key in ("W_states", "Q_states", "k_lin", "k_duff", "c_lin_all", "forward_time")):
        raise ValueError("Nonfinite plotted parameters")
    for p in inputs:
        if signature(p) != before[str(p)]:
            raise RuntimeError(f"Checkpoint changed during extraction: {p}")
    snapshots = sorted(str(p.name) for p in (run / "samples").glob("*_snapshots.npz"))
    metadata = {
        "run": run.name,
        "run_relative_to_project": os.path.relpath(run, PROJECT),
        "config": cfg, "source_file_stats": {p.name: before[str(p)] for p in inputs},
        "config_sha256": hashlib.sha256(inputs[0].read_bytes()).hexdigest(),
        "state_index_zero_based": state,
        "state_indices_zero_based": selected_states,
        "state_initial_majority_classes": initial_classes.tolist(),
        "state_selection": (
            "Default states 2,0,1,4: promote state 2 (initial cluster majority digit 2) "
            "to the first-page coupling rows as requested; retain fixed states 0,1,4 "
            "for comparison. The shown state is not asserted to control the selected trajectory."
            if selected_states == [2, 0, 1, 4] else
            "Explicit fixed state indices selected by the figure command; "
            "no state reassignment across time or assertion of control over the selected trajectory."
        ),
        "state_label_caveat": "Labels describe initial clusters; they do not establish the learned states' final class assignments.",
        "patch_selection": {"row0": row0, "col0": col0, "size": patch_size, "indexing": "zero based"},
        "slice_indices": indices,
        "forward_times": t[indices].tolist(),
        "generation_times": (t[0] - t[indices]).tolist(),
        "interpretation": "Learned parameters versus diffusion time, not optimizer iteration or a physical trajectory.",
        "time_order": "generation s=t_max-t; stored slice 0 is the noisy end",
        "maps": "Raw saved W and Q; no Potts common-mode subtraction or per-frame normalization.",
        "omitted_parameters": "Other Potts states, hidden biases, visible on-site linear biases and per-pixel Ising couplings/biases are not shown.",
        "onsite_energy": "k_lin*x_i^2/2 + k_duff*x_i^4/4 + bias*x_i (sextic term is zero in this quartic run).",
        "stiffness_caveat": "k_lin is a bare coefficient, not the full curvature of the visible energy or marginal free energy.",
        "graph": "c_lin is the coefficient of (x_i-x_j)^2, not twice that coefficient (spring stiffness).",
        "spring_map": "At pixel i: sqrt(mean over its existing neighbours j of c_ij squared). Magnitude only; not a single edge coefficient.",
        "global_spring_rms_at_displayed_times": np.sqrt(np.mean(all_c[indices] ** 2, axis=1)).tolist(),
        "patch_spring_rms_at_displayed_times": np.sqrt(np.mean(all_c[indices][:, edge_ids] ** 2, axis=1)).tolist(),
        "snapshot_files_found": snapshots,
        "trajectory_panel": "No trajectory attached by extract(); the figure command loads a verified saved replay.",
    }
    return data, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory", type=Path, default=None,
                        help="saved trajectory NPZ with adjacent sampling JSON; must match --run")
    parser.add_argument("--states", default="2,0,1,4", help="comma-separated fixed state indices")
    parser.add_argument("--state", type=int, default=None, help="plot a single state; overrides --states")
    parser.add_argument("--slices", default="0,25,50,75,99")
    parser.add_argument("--patch-start", default="12,12", help="zero-based row,column")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--spring-view", choices=("rms", "graph"), default="rms",
                        help="full-lattice per-pixel RMS (default), or the original selected subgraph")
    parser.add_argument("--colour-scale", choices=("linear", "log"), default="log",
                        help="log magnitudes and symmetric-log signed coefficients, or linear")
    parser.add_argument("--also-linear", action="store_true", help="also save a separate linear-scale comparison")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output_dir.resolve()
    data_dir = output.parent / "figure_data"
    for destination in (output, data_dir):
        if destination == run or run in destination.parents:
            raise ValueError("Figure outputs must be outside the checkpoint directory")
    indices = [int(i) for i in args.slices.split(",")]
    patch_start = tuple(int(i) for i in args.patch_start.split(","))
    if len(patch_start) != 2:
        raise ValueError("--patch-start needs row,column")
    states = [args.state] if args.state is not None else [int(j) for j in args.states.split(",")]
    if not states:
        raise ValueError("At least one state is required")
    data, metadata = extract(run, states[0], indices, patch_start, args.patch_size, states[1:])
    trajectory = args.trajectory.resolve() if args.trajectory is not None else default_trajectory(run)
    attach_trajectory(run, data, metadata, trajectory)
    output.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "parameter_evolution_data.npz"
    np.savez_compressed(cache, **data)
    metadata["figure_data_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    from parameter_evolution_layout import render_expanded
    render_expanded(data, metadata, output, args.spring_view, colour_scale=args.colour_scale)
    (data_dir / "parameter_evolution_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.also_linear and args.colour_scale != "linear":
        linear_metadata = json.loads(json.dumps(metadata))
        render_expanded(data, linear_metadata, output, args.spring_view,
                        colour_scale="linear", output_stem="parameter_evolution_linear")
        (data_dir / "parameter_evolution_linear_provenance.json").write_text(json.dumps(linear_metadata, indent=2) + "\n")
    print(f"Rendered saved parameters from {run.name}; states {states}; slices {indices}.")
    print(f"PDFs: {output}; data and SVGs: {data_dir}")
    print(f"Trajectory panel: recorded frames from {trajectory}; no training or sampling was run.")


if __name__ == "__main__":
    main()
