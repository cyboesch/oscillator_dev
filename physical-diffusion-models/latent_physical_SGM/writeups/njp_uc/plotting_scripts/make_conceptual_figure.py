#!/usr/bin/env python3
"""Render the 183 x 98 mm conceptual figure and a real four-digit Potts trajectory.

Usage (from the manuscript directory):
    python3 plotting_scripts/make_conceptual_figure.py

Requires NumPy and pdflatex. This renderer reads the small saved trajectory;
no model, training data, large checkpoint, JAX, or GPU is loaded. The trajectory
is reproduced separately by run_conceptual_trajectory_cpu.sh. Panel c shows
five recorded states from one trajectory, without image interpolation.
The network and learning icons in panels a/b are schematic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


EXPECTED_RUN = "mnist28_d0123_K784_ema_potts_quad_local_quartic_r1_lin_pooled"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_frames(path: Path, work: Path, requested_times: list[float]) -> dict:
    """Export measured pixels as vector tiles with one fixed display mapping."""
    source_info = json.loads(path.with_suffix(".json").read_text())
    if (source_info.get("run") != EXPECTED_RUN
            or source_info.get("status") != "complete"
            or source_info["config"].get("problem") != "mnist28_d0123"):
        raise ValueError("Expected a complete trajectory from the four-digit Potts784 run")
    trajectory_hash = digest(path)
    if source_info.get("trajectory_sha256") != trajectory_hash:
        raise ValueError("Trajectory checksum differs from its sampling manifest")
    with np.load(path, allow_pickle=False) as data:
        times = data["generation_time"]
        states = data["states"]
        mean = float(data["normalization_mean"])
        std = float(data["normalization_std"])
    if (states.shape != (len(times), 784) or len(times) < 5
            or not np.isfinite(states).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or not np.isfinite(mean) or not np.isfinite(std) or std <= 0):
        raise ValueError("Invalid saved states, times, or image normalization")
    if any(t < times[0] or t > times[-1] for t in requested_times):
        raise ValueError("Requested time lies outside the recorded trajectory")
    indices = [int(np.argmin(np.abs(times - t))) for t in requested_times]
    if len(set(indices)) != len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError("Select distinct frames in generation order")
    pixels = np.clip(states[indices] * std + mean, 0, 1).reshape(len(indices), 28, 28)
    for frame, values in enumerate(pixels):
        lines = ["% Actual recorded state; common [0,1] grayscale, white digits on black."]
        for row in range(28):
            for col in range(28):
                lines.append(
                    f"\\fill[white!{100 * values[row, col]:.5f}!black] "
                    f"({col},{-row}) rectangle ++(1,-1);"
                )
        (work / f"conceptual_frame_{frame}.tex").write_text("\n".join(lines) + "\n")
    return {
        "run": source_info["run"],
        "trajectory_file": str(path),
        "trajectory_sha256": trajectory_hash,
        "trajectory_index": source_info["trajectory_index"],
        "initial_set_size": source_info["initial_set_size"],
        "initial_seed": source_info["initial_seed"],
        "requested_generation_times": requested_times,
        "frame_indices": indices,
        "generation_times": times[indices].tolist(),
        "forward_times": (times[-1] - times[indices]).tolist(),
        "normalization_mean": mean,
        "normalization_std": std,
        "display": "clip(x * std + mean, 0, 1); 0 = black, 1 = white; identical scale in all frames",
        "image_interpolation": "none; original 28 x 28 pixels drawn as vector rectangles",
        "selection": "Trajectory index 0 fixed before viewing; five recorded states from noise to the final sample",
        "sampling": "Real reverse-SDE replay from the trained checkpoint; see adjacent trajectory_000.json",
        "artwork_mm": [183, 98],
        "label_font_pt": 7,
        "panel_letter_font_pt": 8,
        "timeline": "Tick-free qualitative timeline with compact inline schedule badge; five frames with schematic spacing",
        "image_tile_size_mm": 22,
        "timescale_cues": "Schematic rapid discrete switching for fast hidden equilibration and smooth slow visible motion; no measured rates or trajectories",
        "thermal_enclosures": "One unfilled red box labelled T around both complete network alternatives, including visible and hidden units at the same temperature; annotation, not additional coupling",
        "thermal_enclosure_colour": "#C43B36",
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=script_dir.parent / "figures")
    parser.add_argument("--trajectory", type=Path,
                        default=script_dir.parent / "figure_data/conceptual_potts4/trajectory_000.npz")
    parser.add_argument("--times", nargs=5, type=float, default=[0.0, 2.5, 4.0, 4.5, 5.0],
                        help="Generation times; choose nearest recorded states, never interpolate images")
    args = parser.parse_args()
    if shutil.which("pdflatex") is None:
        parser.error("Required rendering tool not found: pdflatex")
    if not args.trajectory.is_file():
        parser.error("Saved four-digit trajectory missing; run run_conceptual_trajectory_cpu.sh first")

    output_dir = args.output_dir.resolve()
    data_output = (output_dir.parent / "figure_data").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output.mkdir(parents=True, exist_ok=True)
    source = script_dir / "make_conceptual_figure.tex"
    with tempfile.TemporaryDirectory(prefix="conceptual_figure_") as temp:
        work = Path(temp)
        provenance = prepare_frames(args.trajectory.resolve(), work, args.times)
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                   "-output-directory", str(work), str(source)]
        # Two passes resolve the absolute current-page anchor consistently.
        for _ in range(2):
            result = subprocess.run(command, cwd=work, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"TeX rendering failed:\n{result.stdout}\n{result.stderr}")
        pdf = output_dir / "conceptual_draft.pdf"
        shutil.copy2(work / "make_conceptual_figure.pdf", pdf)

    provenance["renderer_sha256"] = digest(Path(__file__))
    provenance["tikz_source_sha256"] = digest(source)
    (data_output / "conceptual_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pdf)
    if shutil.which("pdftocairo"):
        svg = data_output / "conceptual_draft.svg"
        subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
        print(svg)


if __name__ == "__main__":
    main()
