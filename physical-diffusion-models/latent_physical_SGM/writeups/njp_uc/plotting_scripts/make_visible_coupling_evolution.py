#!/usr/bin/env python3
"""Show every saved nearest-neighbour spring and compare global/patch magnitudes.

Read-only checkpoint access; no model imports, training or sampling. Requires
NumPy/Matplotlib and the adjacent make_parameter_evolution.py utilities.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from make_parameter_evolution import DEFAULT_RUN, DEFAULT_OUTPUT, connectivity, map_npz_member, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output_dir.resolve()
    if not (run / "DONE").is_file():
        raise ValueError("Only completed runs with a DONE marker are accepted")
    if output == run or run in output.parents:
        raise ValueError("Write figure outputs outside the checkpoint directory")
    paths = [run / k for k in ("config.json", "params_history.npy", "hidden_history.npz")]
    before = {p.name: signature(p) for p in paths}
    cfg = json.loads(paths[0].read_text())
    if cfg.get("radius") != 1 or cfg.get("neighbourhood") != "chebyshev":
        raise ValueError("This diagnostic expects radius-1 Chebyshev connectivity")
    theta = np.load(paths[1], mmap_mode="r", allow_pickle=False)
    w = map_npz_member(paths[2], "W")
    n = w.shape[-1]
    grid = int(np.sqrt(n))
    if grid * grid != n:
        raise ValueError("Expected a square visible lattice")
    edges = connectivity(grid, 1, "chebyshev")
    if theta.shape[1] != 4 * n + 3 * len(edges):
        raise ValueError("Parameter layout does not match saved geometry")
    c = np.array(theta[:, 3 * n:3 * n + len(edges)], copy=True)
    t = np.array(map_npz_member(paths[2], "t"), copy=True)
    s = t[0] - t
    indices = [0, 25, 50, 75, 99]
    if c.shape[0] != 100 or t.shape != (100,) or not np.all(np.diff(t) < 0):
        raise ValueError("Expected the saved 100-time descending schedule")
    if not np.isfinite(c).all():
        raise ValueError("Nonfinite spring coefficients")
    row, col = edges[:, 0] // grid, edges[:, 0] % grid
    dr = edges[:, 1] // grid - row
    dc = edges[:, 1] % grid - col
    directions = [(0, 1, "Horizontal", r"$(r,c)\to(r,c+1)$"),
                  (1, 0, "Vertical", r"$(r,c)\to(r+1,c)$"),
                  (1, 1, "Diagonal ↘", r"$(r,c)\to(r+1,c+1)$"),
                  (1, -1, "Diagonal ↙", r"$(r,c)\to(r+1,c-1)$")]
    masks = [(dr == rr) & (dc == cc) for rr, cc, _, _ in directions]
    if not np.all(np.sum(masks, axis=0) == 1):
        raise ValueError("Directions must partition every edge exactly once")
    maps = np.full((4, len(indices), grid, grid), np.nan)
    for direction, mask in enumerate(masks):
        for frame, k in enumerate(indices):
            maps[direction, frame, row[mask], col[mask]] = c[k, mask]
    patch_nodes = np.array([r * grid + cc for r in range(12, 16) for cc in range(12, 16)])
    patch_mask = np.isin(edges[:, 0], patch_nodes) & np.isin(edges[:, 1], patch_nodes)
    rms = np.sqrt(np.mean(c * c, axis=1))
    patch_rms = np.sqrt(np.mean(c[:, patch_mask] ** 2, axis=1))
    max_abs = float(np.abs(c[indices]).max())
    peaks = {}
    for label, values in [("all_edges", rms), ("central_patch", patch_rms)]:
        k = int(np.argmax(values))
        peaks[label] = {"slice": k, "forward_time": float(t[k]),
                        "generation_time": float(s[k]), "rms": float(values[k]),
                        "rms_at_displayed_times": values[indices].tolist()}
    if {p.name: signature(p) for p in paths} != before:
        raise RuntimeError("Source checkpoint changed during extraction")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(11, 10.8), facecolor="white")
    ink, muted = "#202c3b", "#566477"
    fig.text(.055, .965, "Visible springs across the full lattice", fontsize=18, weight="bold", color=ink)
    fig.text(.055, .936, f"All ten MNIST digits | Potts {cfg['K']:,} | {len(edges):,} distinct visible edges", color=muted)
    fig.text(.055, .908, "Each heat-map pixel is one edge, located at its first endpoint; grey indicates no edge.", fontsize=9, color=muted)
    xs = np.linspace(.235, .835, 5)
    for x, k in zip(xs, indices):
        fig.text(x, .873, f"$s={s[k]:.2f}$", ha="center", fontsize=11, color=ink)
        fig.text(x, .854, f"$t={t[k]:.2f}$", ha="center", fontsize=9, color=muted)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d5d9dd")
    norm = Normalize(-max_abs, max_abs)
    for d, (_, _, label, equation) in enumerate(directions):
        y = .702 - d * .148
        fig.text(.055, y + .079, label, fontsize=11, color=ink)
        fig.text(.055, y + .052, equation, fontsize=8, color=muted)
        for frame, x in enumerate(xs):
            ax = fig.add_axes([x - .064, y, .128, .131])
            ax.imshow(maps[d, frame], cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d5d9dd")
                spine.set_linewidth(.4)
    cax = fig.add_axes([.928, .405, .012, .28])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([-max_abs, 0, max_abs])
    cb.ax.set_yticklabels([f"{-max_abs:.3g}", "0", f"{max_abs:.3g}"])
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_linewidth(.4)
    fig.text(.94, .718, "$c_{ij}$", ha="center", fontsize=11, color=ink)
    fig.text(.17, .239, "One linear colour scale for every map; early-time coefficients are small on this scale.", fontsize=9, color=muted)

    ax = fig.add_axes([.12, .072, .58, .128])
    ax.semilogy(s, rms, color="#0072b2", linewidth=2, label=f"All {len(edges):,} edges")
    ax.semilogy(s, patch_rms, color="#d55e00", linewidth=2,
                label=f"Original 4 × 4 patch ({int(patch_mask.sum())} edges)")
    ax.scatter(s[indices], rms[indices], s=14, color="#0072b2", zorder=4)
    ax.scatter(s[indices], patch_rms[indices], s=14, color="#d55e00", zorder=4)
    ax.set_xlabel("Generation time $s=5-t$", fontsize=9)
    ax.set_ylabel("RMS of $c_{ij}$", fontsize=9)
    ax.set_xlim(0, 5)
    ax.grid(True, which="major", color="#e7ebef", linewidth=.6)
    ax.tick_params(labelsize=8)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    gp, pp = peaks["all_edges"], peaks["central_patch"]
    fig.text(.745, .175, "Peak spring magnitude", fontsize=10, weight="bold", color=ink)
    fig.text(.745, .145, f"Full lattice: $s={gp['generation_time']:.2f}$\nRMS = {gp['rms']:.3g}", fontsize=9, color="#0072b2", linespacing=1.5)
    fig.text(.745, .092, f"Central patch: $s={pp['generation_time']:.2f}$\nRMS = {pp['rms']:.3g}", fontsize=9, color="#d55e00", linespacing=1.5)
    fig.text(.055, .022, "$c_{ij}$ multiplies $(x_i-x_j)^2$. All maps and curves use saved coefficients, not sampled trajectories.", fontsize=8.5, color=muted)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output / f"visible_coupling_evolution.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    metadata = {"run": run.name, "source_stats": before, "n_edges": len(edges),
                "slices": indices, "generation_times": s[indices].tolist(),
                "forward_times": t[indices].tolist(), "shared_signed_colour_limit": max_abs,
                "directions_row_column": [[rr, cc] for rr, cc, _, _ in directions],
                "orientation_edge_counts": [int(mask.sum()) for mask in masks],
                "map_definition": "c_ij stored at row,column of first endpoint i; missing bonds masked grey",
                "patch_definition": "Both endpoints in rows/columns 12..15, zero based",
                "magnitude_definition": "sqrt(mean_edges(c_ij**2)) at each saved time",
                "peaks": peaks}
    (output / "visible_coupling_evolution_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(peaks, indent=2))
    print(f"Rendered directional spring heat maps and RMS curves to {output}")


if __name__ == "__main__":
    main()
