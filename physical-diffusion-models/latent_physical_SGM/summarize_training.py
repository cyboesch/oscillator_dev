#!/usr/bin/env python3
"""Summarise train_latent.py runs from their train_log.jsonl files.

Usage:  python summarize_training.py RUN [RUN ...] [--ref RUN]
For every run: validation loss per slice (fixed validation batch per slice, identical for runs on the same data),
the paper checkpoint's loss on the same batch, and for K = 1 runs the hidden unit's diagnostics
(|W| / |W*|, cos(W, W*), class accuracy of sign(a) on the noisy validation samples, <|m|>, softening v|W|^2/T).
--ref: the run whose validation loss is subtracted in the 'difference' panel (default: the paper checkpoint).
Writes out/latent_physical_SGM/summary/<job id>/: training_summary.png and summary.txt. CPU only, seconds.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "summary", os.environ.get("SLURM_JOB_ID", "manual"))
COLORS = ["#2a78d6", "#eb6834", "#2f9e44", "#8d4bd6", "#c2255c", "#868e96"]


def load(run):
    path = os.path.join(lm.OUT_ROOT, run, "train_log.jsonl")
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["slice"]] = r          # a resumed run may repeat a slice; keep the last record
    return [rows[k] for k in sorted(rows)]


def paired(run, ref, k):
    """Per-sample paired difference of the validation loss at slice k (same batch), or None if not stored."""
    try:
        a = np.load(os.path.join(lm.OUT_ROOT, run, "slices", f"slice_{k:03d}.npz"))
        b = np.load(os.path.join(lm.OUT_ROOT, ref, "slices", f"slice_{k:03d}.npz"))
        d = a["val_per_sample"].astype(np.float64) - b["val_per_sample"].astype(np.float64)
        if not np.array_equal(a["val_idx"], b["val_idx"]):
            return None
    except (FileNotFoundError, KeyError):
        return None
    return float(np.mean(d)), float(np.std(d) / np.sqrt(d.size)), float(np.median(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logs = {r: load(r) for r in args.runs}
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    if args.ref:
        out(f"paired validation-loss differences vs {args.ref} (same batch per slice; mean +- SE, median), "
            f"for slices with stored per-sample losses:")
        for run in args.runs:
            if run == args.ref:
                continue
            rows = []
            for r in logs[run]:
                p = paired(run, args.ref, r["slice"])
                if p is not None:
                    rows.append((r["slice"], r["t"]) + p)
            if not rows:
                out(f"  {run}: none stored")
                continue
            arr = np.array(rows)
            out(f"  {run}: {len(rows)} slices; mean difference {arr[:, 2].mean():.4g}; slices with a significantly "
                f"lower (higher) loss at 2 SE: {int(np.sum(arr[:, 2] < -2 * arr[:, 3]))} "
                f"({int(np.sum(arr[:, 2] > 2 * arr[:, 3]))})")
            for s, t, m, se, med in rows:
                if s % 10 == 0 or s >= len(logs[run]) - 2:
                    out(f"    slice {int(s):3d} t {t:.3f}: {m:+.4g} +- {se:.2g} (median {med:+.4g})")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax_l, ax_d, ax_w, ax_c = axes.ravel()
    for i, (run, rows) in enumerate(logs.items()):
        t = np.array([r["t"] for r in rows])
        val = np.array([r["val_loss"] for r in rows])
        col = COLORS[i % len(COLORS)]
        ax_l.plot(t, val, color=col, label=run)
        if args.ref and args.ref in logs:
            ref = {r["slice"]: r["val_loss"] for r in logs[args.ref]}
            d = np.array([r["val_loss"] - ref.get(r["slice"], np.nan) for r in rows])
            lab = f"{run} - {args.ref}"
        else:
            d = np.array([r["val_loss"] - r.get("val_loss_paper", np.nan) for r in rows])
            lab = f"{run} - paper"
        ax_d.plot(t, d, color=col, label=lab)
        if "W_norm" in rows[0]:
            ax_w.plot(t, [r["W_norm"] / r["Wref_norm"] for r in rows], color=col, label=f"{run}: |W| / |W*|")
            ax_w.plot(t, [r["cos_W_Wref"] for r in rows], color=col, linestyle="--", label=f"{run}: cos(W, W*)")
            ax_c.plot(t, [r["class_acc"] for r in rows], color=col, label=f"{run}: class accuracy of sign(a)")
            conf = [r.get("mean_abs_m", r.get("mean_max_p", np.nan)) for r in rows]
            ax_c.plot(t, conf, color=col, linestyle="--", label=f"{run}: <|m|> (Ising) or <max p> (Potts)")
        out(f"== {run}: {len(rows)} slices; total {sum(r['steps'] for r in rows)} steps, "
            f"{sum(r['seconds'] for r in rows) / 3600:.2f} h")
        hdr = f"{'slice':>5}{'t':>8}{'val':>14}{'val paper':>14}{'diff':>11}"
        if "W_norm" in rows[0]:
            hdr += f"{'|W|/|W*|':>10}{'cos':>7}{'acc':>7}{'<|m|>':>7}{'soft/T':>8}"
        out(hdr)
        for r in rows:
            if r["slice"] % 10 == 0 or r["slice"] >= len(rows) - 3:
                s = (f"{r['slice']:>5}{r['t']:>8.4f}{r['val_loss']:>14.6g}{r.get('val_loss_paper', np.nan):>14.6g}"
                     f"{r['val_loss'] - r.get('val_loss_paper', np.nan):>11.4g}")
                if "W_norm" in r:
                    s += (f"{r['W_norm'] / r['Wref_norm']:>10.3f}{r['cos_W_Wref']:>7.3f}{r['class_acc']:>7.3f}"
                          f"{r.get('mean_abs_m', r.get('mean_max_p', np.nan)):>7.3f}"
                          f"{r.get('softening_over_T', r.get('eff_states', np.nan)):>8.3g}")
                out(s)
    ax_l.set_yscale("symlog")
    ax_l.set_title("validation loss J_T per slice (same batch for runs on the joint data)", fontsize=9)
    ax_d.axhline(0, color="#999999", linewidth=0.8)
    ax_d.set_yscale("symlog", linthresh=1.0)
    ax_d.set_title("difference in validation loss (negative = better fit)", fontsize=9)
    ax_w.set_title("hidden unit: size and direction vs the Fisher reference", fontsize=9)
    ax_c.set_title("hidden unit as a classifier of the noisy samples", fontsize=9)
    for ax in axes.ravel():
        ax.set_xlim(5.05, -0.05)
        ax.set_xlabel("forward time t (generation runs left to right)")
        ax.grid(color="#e5e5e5", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_summary.png"), dpi=140)
    plt.close(fig)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
