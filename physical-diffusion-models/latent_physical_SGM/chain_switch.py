#!/usr/bin/env python3
"""Do fixed-slice Langevin chains mix between classes? (2026-11-15; the --fixed-langevin control of sample_latent.py.)

For the checkpoints S1 < S2 < ... of the same chains (<stem>_langevin-t<t>_S<S>.npy, row i = chain i in every file),
classifies every state with fid_score's classifier of the problem (MNIST or Fashion-MNIST) and reports per checkpoint:
the class shares, the share of confident states (max p >= 0.9), the mean distance to the nearest training image (pixel
units, clipped to [0, 1]; a proxy for "is it a clean image"), and between consecutive checkpoints the share of chains
whose class changed (among chains confident at both). If the dynamics mixed, the class shares would match the data's
(10% each for balanced classes) and chains would change class between checkpoints at that rate.

Usage (CPU Slurm job): python chain_switch.py --problem mnist28_d0123456789 PATH_S1 PATH_S2 ...
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import fid_score as fs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    prob = lm.make_problem(args.problem)
    mean, std, grid = prob["mean"], prob["std"], prob["grid"]
    feats, acc = fs.load_classifier(grid, lm.problem_dataset(args.problem))
    data_px = np.clip(np.asarray(prob["data"], np.float64) * std + mean, 0.0, 1.0)
    rng = np.random.default_rng(0)
    ref = data_px[rng.choice(data_px.shape[0], min(20000, data_px.shape[0]), replace=False)]
    ref_sq = np.sum(ref ** 2, 1)
    labels, conf = [], []
    print(f"problem {args.problem}; classifier test accuracy {acc:.4f}")
    print(f"{'checkpoint':<60}{'n':>5}{'div':>5}{'confident':>11}{'NN dist':>9}  class shares %")
    for p in args.paths:
        full = p if os.path.isabs(p) else os.path.join(lm.OUT_ROOT, p)
        x = np.load(full).astype(np.float64)
        ok = np.all(np.isfinite(x), 1)
        ok[ok] = np.max(np.abs(x[ok]), 1) <= fs.DIV_ABS
        P = np.where(ok[:, None], np.nan_to_num(x) * std + mean, 0.0)
        _, pr = feats(P)
        c = np.where(ok, pr.argmax(1), -1)
        cf = ok & (pr.max(1) >= 0.9)
        Pc = np.clip(P[ok], 0.0, 1.0)
        d2 = np.sum(Pc ** 2, 1)[:, None] + ref_sq[None, :] - 2.0 * Pc @ ref.T
        nn = np.sqrt(np.maximum(d2.min(1), 0.0))
        shares = " ".join(f"{d}:{100 * np.mean(c[ok] == d):.0f}" for d in range(10))
        print(f"{os.path.basename(p):<60}{x.shape[0]:>5}{int((~ok).sum()):>5}{100 * cf.mean():>10.1f}%"
              f"{nn.mean():>9.2f}  {shares}")
        labels.append(c)
        conf.append(cf)
    nn_real = []
    for i in range(500):
        j = rng.integers(0, data_px.shape[0])
        d = np.sqrt(np.maximum(np.sum((ref - data_px[j]) ** 2, 1), 0.0))
        d[d == 0] = np.inf
        nn_real.append(d.min())
    print(f"(NN dist of real training images to the reference, excluding themselves: {np.mean(nn_real):.2f})")
    print("")
    for a in range(len(args.paths) - 1):
        both = conf[a] & conf[a + 1]
        ch = np.mean(labels[a][both] != labels[a + 1][both]) if both.any() else float("nan")
        print(f"{os.path.basename(args.paths[a])} -> {os.path.basename(args.paths[a + 1])}: {100 * ch:.1f}% of "
              f"{int(both.sum())} chains (confident at both) changed class")


if __name__ == "__main__":
    main()
