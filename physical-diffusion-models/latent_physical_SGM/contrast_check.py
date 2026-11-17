#!/usr/bin/env python3
"""Contrast of generated faces against real ones (2026-11-17; the user: "lets 3", i.e. look into the low contrast of
the CelebA samples first).

For real held-out test images and every sample set LABEL=PATH (normalised units, mapped to pixels x * std + mean and
clipped to [0, 1], diverged rows dropped as in fid_score):
  - pixel-value histogram (all pixels of all images);
  - per image: mean brightness, standard deviation (contrast), share of dark pixels (< 0.1) and bright pixels (> 0.9);
  - per pixel position: mean over images and standard deviation over images (where the variety is missing:
    background, hair, face).
Report (report.txt): per set the percentiles 1/5/50/95/99 of the pixel values, the mean and the 10/50/90th
percentiles of the per-image std, the dark / bright shares, and the spread of the per-image mean (how different the
images are in overall brightness). Figures: contrast_hist.png (the three histograms), contrast_maps.png (the mean and
spread maps, real first).

Usage (CPU Slurm job): python contrast_check.py --problem P --out DIR LABEL=PATH ...
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import evaluate_samples as es  # noqa: E402
import fid_score as fs  # noqa: E402

N_REAL = 2000


def stats(P):
    """P (n, N) pixels in [0, 1] -> dict of summary statistics and the per-image arrays."""
    m, s = P.mean(1), P.std(1)
    dark, bright = (P < 0.1).mean(1), (P > 0.9).mean(1)
    return {"pct": np.percentile(P, [1, 5, 50, 95, 99]), "img_mean": m, "img_std": s, "dark": dark, "bright": bright,
            "map_mean": P.mean(0), "map_std": P.std(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lines = [f"contrast_check {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
             f"problem {args.problem}"]

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    prob = lm.make_problem(args.problem)
    g, mean, std = prob["grid"], prob["mean"], prob["std"]
    Xte = fs.load_mnist("test", g, lm.problem_dataset(args.problem))[0].astype(np.float32)
    rng = np.random.default_rng(0)
    sets = {"real test images": Xte[rng.choice(Xte.shape[0], min(N_REAL, Xte.shape[0]), replace=False)]}
    for spec in args.sets:
        label, path = spec.split("=", 1)
        full = path if os.path.isabs(path) else os.path.join(lm.OUT_ROOT, path)
        if not os.path.exists(full):
            out(f"{label}: skipped, no file {path}")
            continue
        x = np.asarray(es.load_set(path), np.float64)
        ok = np.all(np.isfinite(x), 1)
        ok[ok] = np.max(np.abs(x[ok]), 1) <= fs.DIV_ABS
        sets[label] = np.clip(x[ok] * std + mean, 0.0, 1.0).astype(np.float32)
    st = {k: stats(v) for k, v in sets.items()}
    out(f"{'set':<66}{'n':>5}  pixel pct 1/5/50/95/99            img std mean (p10/p50/p90)   dark  bright  "
        f"sd(img mean)")
    for k, v in st.items():
        p = v["pct"]
        q = np.percentile(v["img_std"], [10, 50, 90])
        out(f"{k:<66}{sets[k].shape[0]:>5}  {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f} {p[4]:.3f}   "
            f"{v['img_std'].mean():.3f} ({q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f})   {100 * v['dark'].mean():5.1f}% "
            f"{100 * v['bright'].mean():5.1f}%   {v['img_mean'].std():.3f}")
    out("(pixels in [0, 1]; img std = the standard deviation of the pixels within one image (its contrast); dark < 0.1, "
        "bright > 0.9; sd(img mean) = the spread of the images' mean brightness)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    bins = np.linspace(0, 1, 51)
    for k, v in st.items():
        kw = dict(color="#52514e", lw=2.2) if k.startswith("real") else dict(lw=1.3)
        axes[0].hist(sets[k].ravel(), bins=bins, density=True, histtype="step", label=k, **kw)
        axes[1].hist(v["img_std"], bins=40, density=True, histtype="step", label=k, **kw)
        axes[2].hist(v["img_mean"], bins=40, density=True, histtype="step", label=k, **kw)
    axes[0].set_yscale("log")
    for ax, t in zip(axes, ("pixel values (all pixels)", "per-image contrast (std of its pixels)",
                            "per-image mean brightness")):
        ax.set_title(t, fontsize=9)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{args.problem}: contrast of samples vs real test images", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "contrast_hist.png"), dpi=140)
    plt.close(fig)

    n = len(st)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.0))
    axes = np.asarray(axes).reshape(2, n)
    vmax_s = max(v["map_std"].max() for v in st.values())
    for c, (k, v) in enumerate(st.items()):
        axes[0, c].imshow(v["map_mean"].reshape(g, g), cmap="gray", vmin=0, vmax=1)
        axes[1, c].imshow(v["map_std"].reshape(g, g), cmap="magma", vmin=0, vmax=vmax_s)
        axes[0, c].set_title(k.replace("_", "\n", 2), fontsize=6)
        for r in (0, 1):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("mean image", fontsize=8)
    axes[1, 0].set_ylabel("spread over images", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "contrast_maps.png"), dpi=140)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "README.txt"), "w") as f:
        f.write(__doc__.split("Usage")[0] + "\nSets:\n" + "\n".join("  " + s for s in args.sets) + "\n")


if __name__ == "__main__":
    main()
