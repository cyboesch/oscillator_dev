#!/usr/bin/env python3
"""Image gallery: real MNIST images next to generated samples, unsorted, for looking at rather than scoring.

For one problem and a list of sample sets NAME=PATH (PATH relative to out/latent_physical_SGM, '+'-joined parts
allowed, as in evaluate_samples.py), writes into --out:
  real.png               100 random real training images (the evaluation probes' pixel space: data * std + mean)
  <NAME>.png             the first 100 samples of the set (trajectory indices 0-99), in trajectory order, unsorted;
                         a diverged sample (|x| > 15 or not finite, as in the evaluation) is a red square
  compare.png            one row per set (first row: real images), 16 columns; for the sample sets column j is the
                         same trajectory index j, i.e. the same initial state and Brownian keys in every model
  nearest_<NAME>.png     12 samples (top) above their nearest training image (bottom), with the pixel-space distance
All images: white ink on black, pixel values clipped to [0, 1], nearest-neighbour upscaling, no smoothing.

Usage (CPU Slurm job): python make_gallery.py --problem mnist28 --out DIR --title TEXT NAME=PATH ... [--nearest NAME,...]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import evaluate_samples as es  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MAX_ABS = 15.0          # divergence threshold of the evaluation (blend_check.MAX_ABS)


def to_px(x, mean, std):
    x = np.asarray(x, dtype=np.float64)
    div = ~(np.max(np.abs(x), axis=1) <= MAX_ABS)
    P = np.where(np.isfinite(x), x, 0.0) * std + mean
    return np.clip(P, 0.0, 1.0), div


def canvas(P, div, g, rows, cols, pad=1):
    """rows x cols tiles of g x g images with pad-pixel dark-grey separators; RGB array."""
    H, W = rows * (g + pad) + pad, cols * (g + pad) + pad
    C = np.full((H, W, 3), 0.3)
    for i in range(min(P.shape[0], rows * cols)):
        r, c = divmod(i, cols)
        if div[i]:
            tile = np.zeros((g, g, 3))
            tile[..., 0] = 0.7
        else:
            tile = np.repeat(P[i].reshape(g, g)[..., None], 3, axis=2)
        y, x0 = pad + r * (g + pad), pad + c * (g + pad)
        C[y:y + g, x0:x0 + g] = tile
    return C


def save(C, path, title, scale, left_labels=None, row_h=None):
    C = np.repeat(np.repeat(C, scale, axis=0), scale, axis=1)
    H, W = C.shape[:2]
    top, left = 34, (190 if left_labels else 0)
    fig = plt.figure(figsize=((W + left) / 100, (H + top) / 100), dpi=100)
    ax = fig.add_axes([left / (W + left), 0, W / (W + left), H / (H + top)])
    ax.imshow(C, interpolation="nearest")
    ax.set_axis_off()
    fig.text(0.5, 1 - 0.5 * top / (H + top), title, ha="center", va="center", fontsize=10)
    if left_labels:
        for i, lab in enumerate(left_labels):
            yc = (i + 0.5) * row_h * scale
            fig.text((left - 8) / (W + left), 1 - (top + yc) / (H + top), lab, ha="right", va="center", fontsize=8)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="NAME=PATH")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--out", required=True, help="output folder (created)")
    ap.add_argument("--title", default="")
    ap.add_argument("--nearest", default="", help="comma list of set names for nearest-training-image pairs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    prob = lm.make_problem(args.problem)
    data, labels, mean, std, g = np.asarray(prob["data"]), prob["labels"], prob["mean"], prob["std"], prob["grid"]
    px = np.clip(data * std + mean, 0.0, 1.0)
    scale = max(1, 112 // g)                 # about 112 screen pixels per digit
    rng = np.random.default_rng(2026)
    real_idx = rng.choice(px.shape[0], size=100, replace=False)
    no_div = np.zeros(100, bool)
    save(canvas(px[real_idx], no_div, g, 10, 10), os.path.join(args.out, "real.png"),
         f"{args.title}: 100 random real training images", scale)

    sets = []
    for spec in args.sets:
        name, path = spec.split("=", 1)
        P, div = to_px(es.load_set(path), mean, std)
        sets.append((name, P, div))
        save(canvas(P[:100], div[:100], g, 10, 10), os.path.join(args.out, f"{name}.png"),
             f"{args.title}: {name}, samples 0-99 in trajectory order ({int(div[:100].sum())} diverged, red)", scale)
        print(f"{name}: {P.shape[0]} samples, {int(div.sum())} diverged; wrote {name}.png", flush=True)

    # side by side: same trajectory index in every column
    ncol = 16
    rows = [canvas(px[real_idx[:ncol]], np.zeros(ncol, bool), g, 1, ncol)]
    labs = ["real (random)"]
    for name, P, div in sets:
        rows.append(canvas(P[:ncol], div[:ncol], g, 1, ncol))
        labs.append(name)
    save(np.concatenate(rows, axis=0), os.path.join(args.out, "compare.png"),
         f"{args.title}: one row per model; column j = trajectory j in every model (same start and noise)", scale,
         left_labels=labs, row_h=g + 2)

    # nearest training images (pixel space, all training images)
    tr_n2 = np.sum(px * px, axis=1)
    for name in [s for s in args.nearest.split(",") if s and any(n == s for n, _, _ in sets)]:   # skip absent sets
        P, div = next((p, d) for n, p, d in sets if n == name)
        idx = np.where(~div)[0][:12]
        S = P[idx]
        d2 = np.sum(S * S, axis=1)[:, None] - 2.0 * S @ px.T + tr_n2[None, :]
        nn = np.argmin(d2, axis=1)
        dist = np.sqrt(np.maximum(d2[np.arange(len(idx)), nn], 0.0))
        pair = np.concatenate([S, px[nn]], axis=0)
        title = (f"{args.title}: {name}, samples (top) and their nearest training images (bottom); distances "
                 + " ".join(f"{v:.1f}" for v in dist))
        save(canvas(pair, np.zeros(pair.shape[0], bool), g, 2, len(idx)), os.path.join(args.out, f"nearest_{name}.png"),
             title, scale)
        print(f"{name}: nearest-image distances {np.round(dist, 2).tolist()}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
