#!/usr/bin/env python3
"""Figure: standard Inception FID across models, MNIST (ten digits) and Fashion-MNIST (ten classes) (2026-11-15).

Reads the report.txt of inception_fid.py runs (out/latent_physical_SGM/fid_inception/<job>/): MNIST 4253087, Fashion
4253088 and 4254741 (K1536 + patch8x8, seeds 1-3; shown as mean +- sd over seeds). report.txt, not metrics.json: the
metrics.json of these runs kept only the last 'faint zeroed' row (same key; fixed in inception_fid.py since).
Bars: FID against the 60k train statistics at n = 2000 samples, as sampled (error bar: bootstrap sd, or seed sd).
Dots: the same samples with faint pixels (< 0.1) set to 0 (a diagnostic for the sigma = 0.01 pixel noise, not the
model's number). Dashed line: real held-out test images at the same n (the floor). Dotted line: Wang & Achour, arXiv
2606.27294 (their best KuraSHIL model; their n and FID code are not stated).

Usage (CPU Slurm job): python plot_inception_fid.py
Writes out/latent_physical_SGM/figures/inception_fid_mnist_fashion.png and .pdf.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

FIDDIR = os.path.join(lm.OUT_ROOT, "fid_inception")
OUTDIR = os.path.join(lm.OUT_ROOT, "figures")

# reference palette (dataviz skill, light mode): series 1 blue, series 2 orange; ink, grid and surface tokens
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

PANELS = [
    ("MNIST, ten digits", "4253087", 27.6, [
        ("K784", "K784_start-equil_seed1"),
        ("K6144 (noise start)", "K6144-sweep2026-11-10_start-noise_seed1"),
        ("K784 + 7x7 patches", "K784+patch7x7-from-t2.2_start-equil_seed1"),
        ("K3072 + 7x7 patches", "K3072+patch7x7-from-t2.2_start-equil_seed1"),
        ("K6144 + 7x7 patches", "K6144+patch7x7-from-t2.2_start-equil_seed1"),
    ]),
    ("Fashion-MNIST, ten classes", "4253088", 80.8, [
        ("K784", "K784_start-equil_seed1"),
        ("K784 + 14x14 patches (64 states)", "K784+patch14x14-64states-from-t2.2_start-equil_seed1"),
        ("K784 + 7x7 patches", "K784+patch7x7-from-t2.2_start-equil_seed1"),
        ("K1536 + 7x7 patches", "K1536+patch7x7-from-t2.2_start-equil_seed1"),
        ("K3072 + 7x7 patches", "K3072+patch7x7-from-t2.2_start-equil_seed1"),
        ("K784 + 8x8 patches every 2 px", "K784+patch8x8-every2px-from-t2.2_start-equil_seed1"),
        ("K1536 + 8x8 patches every 2 px (3 seeds)", ("4254741", [
            f"K1536+patch8x8-every2px-from-t2.2_start-equil_seed{s}" for s in (1, 2, 3)])),
    ]),
]


def load(job):
    """Rows of report.txt in order -> ({label: [fid_train, boot_sd, fid_faint_zeroed]}, floor fid); the 'faint
    zeroed' diag row directly follows its set."""
    out, last, floor = {}, None, None
    num = r"(-?[\d.]+|nan)"
    pat = re.compile(rf"^(.*?)\s+(\d+)\s+(\d+)\s+{num}\s+{num}\s+{num}\s+{num}\s+{num}\s+{num}$")
    with open(os.path.join(FIDDIR, job, "report.txt")) as f:
        for line in f:
            line = line.rstrip("\n")
            mf = re.match(r"^real test images \(floor.*?\)\s+(\d+)\s+([\d.]+)", line)
            if mf:
                floor = float(mf.group(2))
                continue
            mm = pat.match(line)
            if not mm:
                continue
            label = mm.group(1).strip()
            if label.startswith("diag: same"):
                out[last][2] = float(mm.group(4))
            elif not label.startswith("diag:"):
                out[label], last = [float(mm.group(4)), float(mm.group(5)), np.nan], label
    return out, floor


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    table = []
    for ax, (title, job, wa, models) in zip(axes, PANELS, strict=True):
        rep, floor = load(job)
        rows = []
        for label, key in models:
            if isinstance(key, tuple):                       # seeds from another job: mean +- sd over seeds
                other, _ = load(key[0])
                v = np.array([other[k][0] for k in key[1]])
                z = np.array([other[k][2] for k in key[1]])
                rows.append((label, v.mean(), v.std(), z.mean()))
            else:
                rows.append((label, *rep[key]))
        rows.sort(key=lambda r: -r[1])                       # worst at the top, best at the bottom
        y = np.arange(len(rows))[::-1].astype(float)
        fid = np.array([r[1] for r in rows])
        sd = np.array([r[2] for r in rows])
        zero = np.array([r[3] for r in rows])
        ax.set_facecolor(SURFACE)
        ax.barh(y, fid, height=0.55, color=BLUE, zorder=2, label="as sampled")
        ax.errorbar(fid, y, xerr=sd, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
        ax.scatter(zero, y, s=44, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=4,
                   label="same samples, faint pixels (< 0.1) set to 0 (diagnostic)")
        for yi, f in zip(y, fid):
            ax.text(f + 2.5, yi, f"{f:.1f}", va="center", ha="left", fontsize=8, color=INK2, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
        top = len(rows) - 0.55
        ax.axvline(floor, color=INK2, ls="--", lw=1, zorder=1)
        ax.text(floor + 1.2, top, f"real test images {floor:.1f}", fontsize=7.5, color=INK2, va="bottom", zorder=5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))  # row 1
        ax.axvline(wa, color=INK, ls=":", lw=1.5, zorder=1)
        ax.text(wa + 1.2, top + 0.42, f"Wang & Achour {wa}", fontsize=7.5, color=INK, va="bottom", zorder=5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))  # row 2
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlim(0, max(fid.max() + sd.max(), wa) * 1.15)
        ax.set_ylim(-0.6, len(rows) + 0.35)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=14)
        ax.set_xlabel("Inception FID vs train set, n = 2000 samples (lower is better)")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        table += [(title, r[0], r[1], r[2], r[3]) for r in rows]
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "Equilibrated start and seed 1 unless noted; patches switch on at t = 2.2. Error bars: bootstrap "
             "sd over samples (sd over seeds for the 3-seed row).\nWang & Achour (arXiv 2606.27294): best KuraSHIL "
             "oscillator model; their sample count and FID code are not stated.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    for ext in ("png", "pdf"):
        p = os.path.join(OUTDIR, f"inception_fid_mnist_fashion.{ext}")
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        print("wrote", p)
    print(f"{'panel':<28}{'model':<44}{'FID':>8}{'sd':>6}{'zeroed':>8}")
    for t in table:
        print(f"{t[0]:<28}{t[1]:<44}{t[2]:>8.2f}{t[3]:>6.2f}{t[4]:>8.2f}")


if __name__ == "__main__":
    main()
