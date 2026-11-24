#!/usr/bin/env python3
"""Figure: standard Inception FID across models, a 2 x 2 grid (2026-11-19): MNIST (ten digits), Fashion-MNIST (ten
classes), CelebA 32x32 grey and CelebA 64x64 grey (first version 2026-11-15; CelebA 32x32 added 2026-11-16).

Reads the report.txt of inception_fid.py runs (out/latent_physical_SGM/fid_inception/<job>/; report.txt, not
metrics.json: some early metrics.json files lost rows to a key clash, fixed in inception_fid.py since).
Bars: FID against the train statistics at n = 2000 samples, as sampled by the reverse SDE (error bar: bootstrap sd,
or the sd over seeds). Dots (updated 2026-11-15 evening, the user: "replace the red circles with the quenched results"):
the same samples after a zero-temperature quench of 1 time unit at t = 0 (sample_latent --quench 1), the sampler of
our best results. Value labels: as sampled -> quenched. Dashed line: real held-out test images at the same n (the
floor). Dotted line: Wang & Achour, arXiv 2606.27294 (their best KuraSHIL model; their n and FID code are not stated).

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

SEEDS = (1, 2, 3)
# (panel title, job with the real-image floor, Wang & Achour, [(label, bar source, quench source)]);
# a source is (inception_fid job, report label) or (job, [labels]) for the mean +- sd over seeds
PANELS = [
    ("MNIST, ten digits", "4253087", 27.6, [
        ("K784", ("4253087", "K784_start-equil_seed1"), ("4256059", "K784_start-equil_quench1")),
        ("K6144 (noise start)", ("4253087", "K6144-sweep2026-11-10_start-noise_seed1"),
         ("4256048", "K6144-no-patches_start-noise_quench1")),
        ("K784 + 7x7 patches", ("4253087", "K784+patch7x7-from-t2.2_start-equil_seed1"),
         ("4255005", "K784+patch7x7-from-t2.2_start-equil_quench1")),
        ("K3072 + 7x7 patches", ("4253087", "K3072+patch7x7-from-t2.2_start-equil_seed1"),
         ("4256059", "K3072+patch7x7-from-t2.2_start-equil_quench1")),
        ("K6144 + 7x7 patches", ("4253087", "K6144+patch7x7-from-t2.2_start-equil_seed1"),
         ("4255378", "K6144+patch7x7-from-t2.2_start-equil_quench1")),
    ]),
    ("Fashion-MNIST, ten classes", "4253088", 80.8, [
        ("K784", ("4253088", "K784_start-equil_seed1"), ("4256060", "K784_start-equil_quench1")),
        ("K784 + 14x14 patches (64 states)", ("4253088", "K784+patch14x14-64states-from-t2.2_start-equil_seed1"),
         ("4256060", "K784+patch14x14-64states-from-t2.2_start-equil_quench1")),
        ("K784 + 7x7 patches", ("4253088", "K784+patch7x7-from-t2.2_start-equil_seed1"),
         ("4256060", "K784+patch7x7-from-t2.2_start-equil_quench1")),
        ("K1536 + 7x7 patches", ("4253088", "K1536+patch7x7-from-t2.2_start-equil_seed1"),
         ("4256060", "K1536+patch7x7-from-t2.2_start-equil_quench1")),
        ("K3072 + 7x7 patches", ("4253088", "K3072+patch7x7-from-t2.2_start-equil_seed1"),
         ("4256060", "K3072+patch7x7-from-t2.2_start-equil_quench1")),
        ("K784 + 8x8 patches every 2 px", ("4253088", "K784+patch8x8-every2px-from-t2.2_start-equil_seed1"),
         ("4255010", "K784+patch8x8-every2px-from-t2.2_start-equil_quench1")),
        ("K1536 + 8x8 patches every 2 px (3 seeds)",
         ("4255381", [f"K1536+patch8x8-every2px-from-t2.2_start-equil_seed{s}" for s in SEEDS]),
         ("4255381", [f"K1536+patch8x8-every2px-from-t2.2_start-equil_quench1_seed{s}" for s in SEEDS])),
    ]),
    # CelebA 32x32 grey (added 2026-11-16, updated 2026-11-19): noise start (the equilibrated start changes the quenched
    # FID by 0-1, 4256040); Wang & Achour report no CelebA result (None: no line)
    ("CelebA 32x32 grey (noise start)", "4256037", None, [
        ("K784", ("4256037", "K784_start-noise"), ("4256037", "K784_start-noise_quench1")),
        ("K3072", ("4256861", "K3072_start-noise"), ("4256861", "K3072_start-noise_quench1")),
        ("K784 + 8x8 patches every 2 px", ("4256037", "K784+patch8x8-every2px-from-t2.2_start-noise"),
         ("4256037", "K784+patch8x8-every2px-from-t2.2_start-noise_quench1")),
        ("K1536 + 8x8 patches every 2 px", ("4256037", "K1536+patch8x8-every2px-from-t2.2_start-noise"),
         ("4256037", "K1536+patch8x8-every2px-from-t2.2_start-noise_quench1")),
        ("K784 + 8x8 + 16x16 patches", ("4261206", "K784+patch8x8-every2px+patch16x16-every4px_start-noise"),
         ("4261206", "K784+patch8x8-every2px+patch16x16-every4px_start-noise_quench1")),
        ("K1536 + 8x8 + 16x16 patches", ("4261206", "K1536+patch8x8-every2px+patch16x16-every4px_start-noise"),
         ("4261206", "K1536+patch8x8-every2px+patch16x16-every4px_start-noise_quench1")),
        ("K784 + 8x8 + 4x4 + 16x16 patches", ("4262955",
                                              "K784+patch8x8-every2px+patch4x4-every2px+patch16x16-every4px_start-noise"),
         ("4262955", "K784+patch8x8-every2px+patch4x4-every2px+patch16x16-every4px_start-noise_quench1")),
        ("K1536 + 8x8 + 16x16, 2 corrector steps", ("4267210", "corrector-2-steps-snr0.16_start-noise"),
         ("4267210", "corrector-2-steps-snr0.16_start-noise_quench1")),
        ("K784 + 8x8 + 16x16, on-device training", (
            "4268089", "K784+8x8+16x16-S64-sampled-visit-count-global-and-patch_start-noise"),
         ("4268089", "K784+8x8+16x16-S64-sampled-visit-count-global-and-patch_start-noise_quench1")),
        ("K64 + 8x8 + 16x16 patches", ("4271878", "K64+8x8+16x16_start-noise"),
         ("4271878", "K64+8x8+16x16_start-noise_quench1")),
        ("K64 + 8x8 + 16x16 + 8-cluster label", ("4280007", "K64+8x8+16x16+label8-on-hidden-biases_start-noise"),
         ("4280007", "K64+8x8+16x16+label8-on-hidden-biases_start-noise_quench1")),
        # 2026-11-20: the coarse 16x16 set switched on early (t = 4.49; 8x8 at t = 2.22), with corrector / label
        ("K64 + early 16x16 + 8x8", ("4285111", "K64+early16x16"), ("4285111", "K64+early16x16_quench1")),
        ("K64 + early 16x16 + 8x8, 2 corrector steps", ("4285111", "K64+early16x16_corrector2"),
         ("4285111", "K64+early16x16_corrector2_quench1")),
        ("K784 + early 16x16 + 8x8 + label, 2 corrector steps", ("4285111", "K784+early16x16+label8_corrector2"),
         ("4285111", "K784+early16x16+label8_corrector2_quench1")),
        # 2026-11-20: the same K64 model with its history compressed to 5 / 9 / 9 time knots (10.7 M stored)
        ("K64 + early 16x16 + 8x8, knots 5/9/9", ("4286278", "knots559"), ("4286278", "knots559_quench1")),
        ("K64 + early 16x16 + 8x8, knots 5/9/9 + corrector", ("4286278", "knots559_corrector2"),
         ("4286278", "knots559_corrector2_quench1")),
    ]),
    # CelebA 64x64 grey (added 2026-11-19): noise start; 8x8 patches are too slow at 64x64, so 16x16 / 32x32
    ("CelebA 64x64 grey (noise start)", "4274103", None, [
        ("K1536 + 16x16 patches every 4 px", ("4266707", "K1536+patch16x16-every4px_start-noise"),
         ("4266707", "K1536+patch16x16-every4px_start-noise_quench1")),
        ("K3072 + 16x16 patches every 4 px", ("4266812", "K3072+patch16x16-every4px_start-noise"),
         ("4266812", "K3072+patch16x16-every4px_start-noise_quench1")),
        ("K64 + 16x16 + 32x32 patches", ("4274103", "K64+16x16+32x32_start-noise"),
         ("4274103", "K64+16x16+32x32_start-noise_quench1")),
        ("K64 + 16x16 + 32x32 + 8-cluster label", ("4274103", "K64+16x16+32x32+label8-on-hidden-biases_start-noise"),
         ("4274103", "K64+16x16+32x32+label8-on-hidden-biases_start-noise_quench1")),
        # 2026-11-20: the coarse 32x32 set switched on early (t = 4.49; 16x16 at t = 2.22)
        ("K64 + early 32x32 + 16x16", ("4285112", "K64+early32x32"), ("4285112", "K64+early32x32_quench1")),
        ("K64 + early 32x32 + 16x16 + 8-cluster label", ("4285112", "K64+early32x32+label8"),
         ("4285112", "K64+early32x32+label8_quench1")),
        ("K64 + early 32x32 + 16x16, 2 corrector steps", ("4285144", "K64+early32x32_corrector2"),
         ("4285144", "K64+early32x32_corrector2_quench1")),
        # 2026-11-20: history compressed to 5 / 9 / 9 time knots (42.7 M stored instead of 331.8 M)
        ("K64 + early 32x32 + 16x16, knots 5/9/9", ("4286279", "knots559"), ("4286279", "knots559_quench1")),
        ("K64 + early 32x32 + 16x16, knots 5/9/9 + corrector", ("4286279", "knots559_corrector2"),
         ("4286279", "knots559_corrector2_quench1")),
    ]),
]
_cache = {}


def load(job):
    """Rows of report.txt -> ({label: (fid_train, boot_sd)}, floor fid); diag rows are skipped."""
    if job in _cache:
        return _cache[job]
    out, floor = {}, None
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
            if mm and not mm.group(1).strip().startswith("diag"):
                out[mm.group(1).strip()] = (float(mm.group(4)), float(mm.group(5)))
    _cache[job] = (out, floor)
    return out, floor


def value(src):
    """(FID, sd): one set (bootstrap sd) or the mean over seeds (sd over seeds)."""
    job, key = src
    rep, _ = load(job)
    if isinstance(key, list):
        v = np.array([rep[k][0] for k in key])
        return float(v.mean()), float(v.std())
    return rep[key]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK, "text.color": INK})
    n_rows = [len(m) for _, _, _, m in PANELS]           # 2 x 2 grid; row heights follow the longer panel of the row
    hr = [max(n_rows[0:2]), max(n_rows[2:4])]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 0.42 * sum(hr) + 4.4), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": hr})
    axes = axes.ravel()
    table = []
    for i_ax, (ax, (title, floor_job, wa, models)) in enumerate(zip(axes, PANELS, strict=True)):
        n_slot = hr[i_ax // 2]                           # rows of the grid row: same bar spacing in both panels
        _, floor = load(floor_job)
        rows = [(label, *value(bar), *value(q)) for label, bar, q in models]
        rows.sort(key=lambda r: -r[3])                       # worst (quenched) at the top, best at the bottom
        y = np.arange(len(rows))[::-1].astype(float) + (n_slot - len(rows))   # stacked from the top
        fid = np.array([r[1] for r in rows])
        sd = np.array([r[2] for r in rows])
        qf = np.array([r[3] for r in rows])
        ax.set_facecolor(SURFACE)
        ax.barh(y, fid, height=0.55, color=BLUE, zorder=2, label="as sampled (reverse SDE)")
        ax.errorbar(fid, y, xerr=sd, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
        ax.scatter(qf, y, s=46, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=4,
                   label="after a zero-temperature quench of 1 at t = 0")
        for yi, f, q in zip(y, fid, qf):
            ax.text(f + 2.5, yi, f"{f:.1f} → {q:.1f}", va="center", ha="left", fontsize=8, color=INK2, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
        top = n_slot - 0.55
        y_lo = n_slot - len(rows) - 0.5                    # reference lines span the rows only
        ax.vlines(floor, y_lo, n_slot + 0.35, colors=INK2, linestyles="--", lw=1, zorder=1)
        ax.text(floor + 1.2, top, f"real test images {floor:.1f}", fontsize=7.5, color=INK2, va="bottom", zorder=5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))  # row 1
        if wa is not None:                                   # no published number for CelebA
            ax.vlines(wa, y_lo, n_slot + 0.35, colors=INK, linestyles=":", lw=1.5, zorder=1)
            ax.text(wa + 1.2, top + 0.42, f"Wang & Achour {wa}", fontsize=7.5, color=INK, va="bottom", zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))  # row 2
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlim(0, max(fid.max() + sd.max(), wa or 0.0) * 1.32)
        ax.set_ylim(-0.6, n_slot + 0.35)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=14)
        ax.set_xlabel("Inception FID vs train set, n = 2000 samples (lower is better)")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        table += [(title, r[0], r[1], r[2], r[3], r[4]) for r in rows]
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Rows sorted by the quenched FID (best at the bottom). Seed 1 unless noted; equilibrated start for "
             "MNIST and Fashion-MNIST, noise start for CelebA; patches switch on at t = 2.2 (\"early\": the coarsest set from t = 4.49).\n"
             "Error bars: bootstrap sd "
             "over samples (sd over seeds for the 3-seed row).\n"
             "Quench: the same samples relaxed for 1 time unit at zero temperature under the t = 0 energy. CelebA: "
             "aligned faces, centre crop, grey; 8x8 patches every 2 px, 4x4 every 2 px, 16x16 every 4 px, 32x32 every 8 px "
             "(64x64 only); 32 states per patch unit.\n"
             "Corrector: 2 Langevin corrector steps per knot ('+ corrector': the same). On-device training: hidden statistics from 64 sampled states, "
             "visit-count rule. 8-cluster label: a clamped k-means label on the hidden biases, drawn per sample.\n"
             "Knots 5/9/9: the trained 100-slice parameter history kept at 5 slices (global unit) and 9 (each patch "
             "set), linear in t between them, per-pixel units constant: 7.8x fewer stored numbers (post hoc, no retraining).\n"
             "The CelebA training images carry the paper's 1% pixel noise and score 11.1 themselves at 32x32 (real test "
             "images: 6.4). Wang & Achour (arXiv 2606.27294): best KuraSHIL oscillator model; their sample count and FID "
             "code are not stated; they report no CelebA result.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.965))
    for ext in ("png", "pdf"):
        p = os.path.join(OUTDIR, f"inception_fid_mnist_fashion.{ext}")
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        print("wrote", p)
    print(f"{'panel':<34}{'model':<44}{'FID':>8}{'sd':>6}{'quench1':>9}{'sd':>6}")
    for t in table:
        print(f"{t[0]:<34}{t[1]:<44}{t[2]:>8.2f}{t[3]:>6.2f}{t[4]:>9.2f}{t[5]:>6.2f}")


if __name__ == "__main__":
    main()
