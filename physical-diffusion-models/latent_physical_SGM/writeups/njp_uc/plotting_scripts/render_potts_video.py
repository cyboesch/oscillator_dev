#!/usr/bin/env python3
"""Animate saved visible configurations and conditional Potts probabilities.

This renderer never runs a model or sampler. It shows probability evolution,
not a discrete hidden-state path or physical transition rates. Digit groups
use the initial clusters' majority labels, not inferred final-state classes.

Required NPZ keys:
  generation_time (frames,), states (frames,pixels), posterior (frames,states),
  state_initial_classes (states,), normalization_mean (), normalization_std ().

Normal mode writes MP4 using an installed ffmpeg. In --raw-video mode, stdout
contains only RGBA frames for a host ffmpeg process; all messages go to stderr.
Example (from the workspace, with the existing container):
  apptainer exec ... python3 render_potts_video.py --data TRAJECTORY.npz \
      --raw-video --fps 20 | ffmpeg -f rawvideo -pixel_format rgba \
      -video_size 1280x800 -framerate 20 -i pipe:0 -an \
      -c:v libx264 -crf 18 -pix_fmt yuv420p VIDEO.mp4

Every saved configuration contributes exactly one frame. Frames/probabilities
are not interpolated, and no random hidden-state draws are fabricated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in (
            "generation_time", "states", "posterior", "state_initial_classes",
            "normalization_mean", "normalization_std")}
    times = np.asarray(data["generation_time"], dtype=float)
    x = np.asarray(data["states"], dtype=float)
    p = np.asarray(data["posterior"], dtype=float)
    initial_classes = np.asarray(data["state_initial_classes"])
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("Need at least two finite generation times")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Generation times must be strictly increasing")
    if not np.allclose(np.diff(times), np.mean(np.diff(times)), rtol=1e-5, atol=1e-8):
        raise ValueError("Uniformly spaced snapshots are required for one-frame-per-snapshot playback")
    if x.ndim != 2 or p.ndim != 2 or len(x) != len(times) or len(p) != len(times):
        raise ValueError("State, posterior and time dimensions do not match")
    grid = int(np.sqrt(x.shape[1]))
    if grid * grid != x.shape[1]:
        raise ValueError("Visible states must form a square image")
    if initial_classes.shape != (p.shape[1],) or not np.issubdtype(initial_classes.dtype, np.integer):
        raise ValueError("Initial class labels must be one integer per Potts state")
    if np.any((initial_classes < -1) | (initial_classes > 9)):
        raise ValueError("Initial class labels must be digits 0..9, or -1 for unknown")
    if not np.isfinite(x).all() or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Configurations and probabilities must be finite; probabilities nonnegative")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Each posterior must sum to one")
    mean, std = float(data["normalization_mean"]), float(data["normalization_std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid display normalization")
    data.update(generation_time=times, states=x, posterior=p,
                state_initial_classes=initial_classes.astype(int),
                normalization_mean=mean, normalization_std=std, grid=grid)
    return data


def prepare(data, top):
    p = data["posterior"]
    labels = data["state_initial_classes"]
    groups = list(range(10)) + ([-1] if np.any(labels == -1) else [])
    mass = np.column_stack([p[:, labels == digit].sum(axis=1) for digit in groups])
    top = min(top, p.shape[1])
    # A fixed set selected once from the complete recorded trajectory.
    selected = np.argsort(-np.max(p, axis=0), kind="stable")[:top]
    selected = np.asarray(sorted(selected, key=lambda j: (int(labels[j]), -float(p[:, j].max()), int(j))))
    displayed = np.column_stack([p[:, selected], np.maximum(0.0, 1.0 - p[:, selected].sum(axis=1))])
    def entropy(probability):
        return -np.sum(probability * np.log(np.maximum(probability, np.finfo(float).tiny)), axis=1)
    state_entropy = entropy(p)
    group_entropy = entropy(mass)
    within_entropy = np.maximum(0.0, state_entropy - group_entropy)
    return groups, mass, selected, displayed, group_entropy, within_entropy


def make_animation(data, top, width, height):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    ink, muted, grid_color = "#202c3b", "#566477", "#e1e6ec"
    times, x, p = data["generation_time"], data["states"], data["posterior"]
    labels, side = data["state_initial_classes"], data["grid"]
    groups, mass, selected, displayed, group_entropy, within_entropy = prepare(data, top)
    palette = plt.get_cmap("tab10")
    group_colors = [palette(g) if g >= 0 else "#838b95" for g in groups]
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    fig.text(.04, .956, "Which Potts states compete during generation?", fontsize=18, weight="bold", color=ink)
    status = fig.text(.04, .916, "", fontsize=11, color=muted)

    image_ax = fig.add_axes([.045, .483, .226, .360])
    image_ax.set_title("Visible configuration", loc="left", fontsize=11, weight="bold", pad=10)
    im = image_ax.imshow(np.clip(x[0].reshape(side, side) * data["normalization_std"] + data["normalization_mean"], 0, 1),
                         cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    image_ax.axis("off")
    fig.text(.045, .465, "Fixed intensity scale 0–1", fontsize=8.3, color=muted)

    digit_ax = fig.add_axes([.330, .515, .285, .325])
    digit_bars = digit_ax.bar(np.arange(len(groups)), mass[0], color=group_colors, width=.75)
    digit_ax.set_title("Probability by initial digit group", loc="left", fontsize=11, weight="bold", pad=10)
    digit_ax.set_ylim(0, 1.04)
    digit_ax.set_xlim(-.65, len(groups) - .35)
    digit_ax.set_xticks(np.arange(len(groups)), [str(g) if g >= 0 else "?" for g in groups])
    digit_ax.set_ylabel("Probability mass")
    digit_ax.set_xlabel("Initial cluster majority label")
    digit_ax.grid(axis="y", color=grid_color, linewidth=.6)
    digit_ax.set_axisbelow(True)

    state_ax = fig.add_axes([.748, .488, .220, .352])
    state_colors = [palette(labels[j]) if labels[j] >= 0 else "#838b95" for j in selected] + ["#b2bac4"]
    state_bars = state_ax.barh(np.arange(len(selected) + 1), displayed[0], color=state_colors, height=.7)
    state_ax.invert_yaxis()
    state_ax.set_xlim(0, 1.04)
    state_ax.set_yticks(np.arange(len(selected) + 1),
                       [f"j={int(j)}  [{int(labels[j]) if labels[j] >= 0 else '?'}]" for j in selected] + ["all other states"], fontsize=8)
    state_ax.set_xlabel("State probability")
    state_ax.set_title("Competing Potts states", loc="left", fontsize=11, weight="bold", pad=10)
    state_ax.grid(axis="x", color=grid_color, linewidth=.6)
    state_ax.set_axisbelow(True)
    fig.text(.748, .425, f"Fixed top {len(selected)} by peak p; [ ] = initial group", fontsize=7.8, color=muted)

    history_ax = fig.add_axes([.065, .160, .550, .205])
    history_ax.set_title("Digit-group probabilities along this trajectory", loc="left", fontsize=11, weight="bold", pad=12)
    history_ax.set_xlim(times[0], times[-1])
    history_ax.set_ylim(-.025, 1.025)
    history_ax.set_xlabel("Generation time s")
    history_ax.set_ylabel("Probability mass")
    history_lines = [history_ax.plot([], [], color=color, linewidth=1.7, label=str(group) if group >= 0 else "?")[0]
                     for group, color in zip(groups, group_colors)]
    history_cursor = history_ax.axvline(times[0], color=muted, linewidth=.9, linestyle="--")
    history_ax.legend(ncol=len(groups), loc="upper left", bbox_to_anchor=(0, 1.04),
                      frameon=False, fontsize=7.7, handlelength=1.0, columnspacing=.65, handletextpad=.3)
    history_ax.grid(color=grid_color, linewidth=.6)

    entropy_ax = fig.add_axes([.735, .160, .233, .205])
    entropy_ax.set_title("Where is the uncertainty?", loc="left", fontsize=11, weight="bold", pad=12)
    entropy_ax.set_xlim(times[0], times[-1])
    entropy_ax.set_ylim(0, max(float(group_entropy.max()), float(within_entropy.max()), .1) * 1.15)
    entropy_ax.set_xlabel("Generation time s")
    entropy_ax.set_ylabel("Entropy (nats)")
    entropy_group, = entropy_ax.plot([], [], color="#d27732", linewidth=1.8, label="between digit groups")
    entropy_within, = entropy_ax.plot([], [], color="#3e78a7", linewidth=1.8, label="states within groups")
    entropy_cursor = entropy_ax.axvline(times[0], color=muted, linewidth=.9, linestyle="--")
    entropy_ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    entropy_ax.grid(color=grid_color, linewidth=.6)
    for ax in (digit_ax, state_ax, history_ax, entropy_ax):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.text(.04, .072, "Conditional probabilities from the averaged model; no physical switching events or rates are shown.",
             fontsize=9, color=muted)
    fig.text(.04, .042, "Digit groups use initialization labels. They are approximate associations, not measured final-state classes.",
             fontsize=8.6, color=muted)

    def update(frame):
        im.set_data(np.clip(x[frame].reshape(side, side) * data["normalization_std"] + data["normalization_mean"], 0, 1))
        leading = int(np.argmax(p[frame]))
        label = int(labels[leading])
        status.set_text(f"s = {times[frame]:.3f}    t = {times[-1] - times[frame]:.3f}    |    "
                        f"Most probable state: j = {leading}    initial group: {label if label >= 0 else '?'}    "
                        f"p = {p[frame, leading]:.4f}")
        for bar, value in zip(digit_bars, mass[frame]):
            bar.set_height(value)
        for bar, value in zip(state_bars, displayed[frame]):
            bar.set_width(value)
        for line, probabilities in zip(history_lines, mass.T):
            line.set_data(times[:frame + 1], probabilities[:frame + 1])
        entropy_group.set_data(times[:frame + 1], group_entropy[:frame + 1])
        entropy_within.set_data(times[:frame + 1], within_entropy[:frame + 1])
        history_cursor.set_xdata([times[frame], times[frame]])
        entropy_cursor.set_xdata([times[frame], times[frame]])
        return fig

    update(0)
    return fig, update


def save_summary(data, path):
    import matplotlib.pyplot as plt
    times, x, p = data["generation_time"], data["states"], data["posterior"]
    groups, mass, _, _, _, _ = prepare(data, 12)
    palette = plt.get_cmap("tab10")
    ids = np.rint(np.linspace(0, len(times) - 1, 5)).astype(int)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2), gridspec_kw={"height_ratios": [1.0, .65]})
    fig.suptitle("Potts-state competition along one generated trajectory", fontsize=15, weight="bold", y=.985)
    for col, frame in enumerate(ids):
        image = np.clip(x[frame].reshape(data["grid"], data["grid"]) * data["normalization_std"] + data["normalization_mean"], 0, 1)
        axes[0, col].imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        leading = int(np.argmax(p[frame]))
        axes[0, col].set_title(f"s = {times[frame]:.2f}\nj = {leading}, p = {p[frame, leading]:.3f}", fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].bar(np.arange(len(groups)), mass[frame], color=[palette(g) if g >= 0 else "#838b95" for g in groups])
        axes[1, col].set_xticks(np.arange(len(groups)), [str(g) if g >= 0 else "?" for g in groups], fontsize=7)
        axes[1, col].set_ylim(0, 1.02)
        axes[1, col].set_xlabel("Initial digit group", fontsize=9)
        if col == 0:
            axes[1, col].set_ylabel("Probability mass", fontsize=9)
        for side in ("top", "right"):
            axes[1, col].spines[side].set_visible(False)
    fig.text(.5, .015, "Saved visible states and conditional probabilities; initialization labels are approximate. No physical switching rates are simulated.",
             ha="center", fontsize=8.3, color="#566477")
    fig.tight_layout(rect=(0, .045, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PAPER / "videos/potts_state_probabilities.mp4")
    parser.add_argument("--raw-video", action="store_true", help="write RGBA frames to stdout for a host ffmpeg pipe")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--summary", type=Path, help="optional PDF contact sheet of five saved frames")
    args = parser.parse_args()
    if args.fps <= 0 or args.top <= 0 or args.width < 1000 or args.height < 700:
        parser.error("Need positive fps/top and at least 1000x700 pixels")
    if args.width % 2 or args.height % 2:
        parser.error("Width and height must be even for H.264 playback")
    if args.summary is not None and args.summary.suffix.lower() != ".pdf":
        parser.error("--summary must be a PDF path")
    data = load_trajectory(args.data)
    frame_count = len(data["generation_time"])
    fig, update = make_animation(data, args.top, args.width, args.height)
    print(json.dumps({"frames": frame_count, "width": args.width, "height": args.height,
                      "fps": args.fps, "duration_seconds": frame_count / args.fps,
                      "mode": "raw_rgba" if args.raw_video else "mp4"}), file=sys.stderr, flush=True)
    if args.raw_video:
        for frame in range(frame_count):
            update(frame)
            fig.canvas.draw()
            sys.stdout.buffer.write(fig.canvas.buffer_rgba())
            if frame % 50 == 0 or frame == frame_count - 1:
                print(f"Rendered frame {frame + 1}/{frame_count}", file=sys.stderr, flush=True)
        sys.stdout.buffer.flush()
    else:
        from matplotlib.animation import FFMpegWriter
        if shutil.which("ffmpeg") is None:
            parser.error("ffmpeg is unavailable; use --raw-video piped from the container to host ffmpeg")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-crf", "18", "-pix_fmt", "yuv420p"],
                             metadata={"title": "Potts state probabilities along a generated trajectory"})
        with writer.saving(fig, str(args.output), dpi=100):
            for frame in range(frame_count):
                update(frame)
                writer.grab_frame(facecolor="white")
                if frame % 50 == 0 or frame == frame_count - 1:
                    print(f"Rendered frame {frame + 1}/{frame_count}", file=sys.stderr, flush=True)
        print(f"Saved {args.output}", file=sys.stderr, flush=True)
    import matplotlib.pyplot as plt
    plt.close(fig)
    if args.summary is not None:
        save_summary(data, args.summary)
        print(f"Saved {args.summary}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
