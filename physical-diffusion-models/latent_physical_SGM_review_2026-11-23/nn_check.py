#!/usr/bin/env python3
"""Memorisation check by nearest neighbours (2026-11-15; the user: "plot some nearest neighbours ... to check whether
it's not just memorizing", for the zero-temperature quench).

For each sample set LABEL=PATH (normalised units, as everywhere; mapped to pixels x * std + mean, clipped to [0, 1];
diverged rows dropped as in fid_score): the Euclidean pixel distance of every sample to its nearest TRAINING image (the
images the model was trained on, raw 8-bit / 255 without the paper's pixel noise) and to its nearest TEST image
(held out). Baseline: n random real test images and their nearest training image, i.e. how close a genuinely new real
image comes to the training set. Memorisation shows as samples much closer to the training set than that baseline
(near-copies), and as d_train well below d_test.

Report (report.txt): per set the median and 5th percentile of d_train and d_test, the share of samples closer to the
training set than the 1st / 5th percentile of the baseline, and the share with d_train < 1 (near-identical in pixel
space). Figures: per set nn_<k>_<label>.png, 18 triplets (sample | nearest train | nearest test, distances in the
titles): the first 12 are random samples, the last 6 the samples closest to the training set (the most suspicious);
and nn_hist.png, the d_train histograms against the baseline.

Usage (CPU Slurm job): python nn_check.py --problem P --out DIR LABEL=PATH ...
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

N_RANDOM, N_CLOSEST = 12, 6


def raw_train(problem, prob):
    """The training images as the model saw them, raw (no pixel noise), (n, grid^2) in [0, 1]."""
    if lm.problem_dataset(problem).startswith("celeba"):
        return lm.load_celeba_raw("train", lm.celeba_grid(problem))[0]
    path = lm.problem_data_path(problem)
    z = np.load(path)
    if "images" in z.files:
        return np.asarray(z["images"], np.float32)
    return np.clip(np.asarray(prob["data"]) * prob["std"] + prob["mean"], 0.0, 1.0).astype(np.float32)


def nearest(Q, R, chunk=2000):
    """For each row of Q its nearest row of R: (distance, index); Euclidean, float64 accumulation."""
    Q, R = np.asarray(Q, np.float32), np.asarray(R, np.float32)
    r2 = np.sum(R.astype(np.float64) ** 2, 1)
    d, ix = np.empty(Q.shape[0]), np.empty(Q.shape[0], np.int64)
    for i in range(0, Q.shape[0], chunk):
        q = Q[i:i + chunk]
        d2 = np.sum(q.astype(np.float64) ** 2, 1)[:, None] + r2[None, :] - 2.0 * (q @ R.T).astype(np.float64)
        j = np.argmin(d2, 1)
        ix[i:i + chunk] = j
        d[i:i + chunk] = np.sqrt(np.maximum(d2[np.arange(q.shape[0]), j], 0.0))
    return d, ix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lines = [f"nn_check {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
             f"problem {args.problem}"]

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    prob = lm.make_problem(args.problem)
    g, mean, std = prob["grid"], prob["mean"], prob["std"]
    Xtr = raw_train(args.problem, prob)
    Xte = fs.load_mnist("test", g, lm.problem_dataset(args.problem))[0].astype(np.float32)
    rng = np.random.default_rng(0)
    jb = rng.choice(Xte.shape[0], min(2000, Xte.shape[0]), replace=False)
    base_tr, _ = nearest(Xte[jb], Xtr)
    q1, q5 = np.percentile(base_tr, [1, 5])
    out(f"training images {Xtr.shape[0]}, test images {Xte.shape[0]}; pixel distances (Euclidean, [0, 1] pixels, "
        f"{g}x{g})")
    out(f"baseline: {jb.size} real test images -> nearest training image: median {np.median(base_tr):.2f}, "
        f"5th pct {q5:.2f}, 1st pct {q1:.2f}, min {base_tr.min():.2f}")
    out("")
    out(f"{'set':<58}{'n':>5}{'d_train med':>12}{'p5':>7}{'d_test med':>11}{'p5':>7}{'<base p1':>10}{'<base p5':>10}"
        f"{'d_tr<1':>8}")
    hist = {}
    for k, spec in enumerate(args.sets):
        label, path = spec.split("=", 1)
        full = path if os.path.isabs(path) else os.path.join(lm.OUT_ROOT, path)
        if not os.path.exists(full):
            out(f"{label}: skipped, no file {path}")
            continue
        x = np.asarray(es.load_set(path), np.float64)
        ok = np.all(np.isfinite(x), 1)
        ok[ok] = np.max(np.abs(x[ok]), 1) <= fs.DIV_ABS
        P = np.clip(x[ok] * std + mean, 0.0, 1.0).astype(np.float32)
        dtr, itr = nearest(P, Xtr)
        dte, ite = nearest(P, Xte)
        hist[label] = dtr
        out(f"{label:<58}{P.shape[0]:>5}{np.median(dtr):>12.2f}{np.percentile(dtr, 5):>7.2f}{np.median(dte):>11.2f}"
            f"{np.percentile(dte, 5):>7.2f}{100 * np.mean(dtr < q1):>9.1f}%{100 * np.mean(dtr < q5):>9.1f}%"
            f"{100 * np.mean(dtr < 1.0):>7.1f}%")
        pick = list(rng.choice(P.shape[0], N_RANDOM, replace=False)) + list(np.argsort(dtr)[:N_CLOSEST])
        fig, axes = plt.subplots(3, 18, figsize=(18 * 0.95, 3 * 1.15))
        for c, i in enumerate(pick):
            for r, (img, ttl) in enumerate([(P[i], "sample" if c < N_RANDOM else "closest"),
                                            (Xtr[itr[i]], f"train {dtr[i]:.1f}"), (Xte[ite[i]], f"test {dte[i]:.1f}")]):
                ax = axes[r, c]
                ax.imshow(img.reshape(g, g), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(ttl, fontsize=6.5, color="#b3261e" if (r == 1 and dtr[i] < q5) else "black")
        fig.suptitle(f"{label}: sample (top) | nearest training image (middle) | nearest test image (bottom); columns "
                     f"1-{N_RANDOM} random, {N_RANDOM + 1}-{N_RANDOM + N_CLOSEST} the samples closest to the training "
                     f"set; red = closer than 95% of real test images come", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(args.out, f"nn_{k:02d}_{label}.png"), dpi=130)
        plt.close(fig)
    out(f"(d = Euclidean distance in [0, 1] pixels; '<base p1/p5' = share of samples closer to the training set than "
        f"the 1st / 5th percentile of real test images; memorisation would show as large shares there and d_train << "
        f"d_test)")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, max(np.percentile(base_tr, 99.5), max(np.percentile(v, 99.5) for v in hist.values())), 60)
    ax.hist(base_tr, bins=bins, density=True, color="#9a9890", alpha=0.6, label="real test images (baseline)")
    for label, v in hist.items():
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.5, label=label)
    ax.set_xlabel("distance to the nearest training image (pixels in [0, 1])")
    ax.set_ylabel("density")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(args.title or args.problem, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "nn_hist.png"), dpi=150)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "README.txt"), "w") as f:
        f.write(__doc__.split("Usage")[0] + "\nSets:\n" + "\n".join("  " + s for s in args.sets) + "\n")


if __name__ == "__main__":
    main()
