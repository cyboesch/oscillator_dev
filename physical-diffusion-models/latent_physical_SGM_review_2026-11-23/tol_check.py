#!/usr/bin/env python3
"""Convergence of the reverse-SDE solver tolerance (2026-11-17; the user: "make sure that reducing does not change the
images"). Compares sample files of the SAME trajectories (same start states, same Brownian keys; the Brownian paths
do not depend on the solver's steps) produced at different --sde-tol-scale: per trajectory the max and the RMS of
|x_tight - x_ref| (normalised units), against the typical distance between two DIFFERENT samples of the reference
set (median nearest-neighbour RMS distance), plus the per-image contrast and dark share of each set.

Usage (CPU or GPU job): python tol_check.py --problem P REF_LABEL=PATH LABEL=PATH ...   (the first set is the reference)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--problem", required=True)
    args = ap.parse_args()
    prob = lm.make_problem(args.problem)
    m, s = prob["mean"], prob["std"]
    sets = []
    for spec in args.sets:
        label, path = spec.split("=", 1)
        full = path if os.path.isabs(path) else os.path.join(lm.OUT_ROOT, path)
        sets.append((label, np.load(full)))
    ref_label, ref = sets[0]
    d2 = np.sum((ref[:, None, :] - ref[None, :, :]) ** 2, axis=2) / ref.shape[1]
    np.fill_diagonal(d2, np.inf)
    nn_rms = np.median(np.sqrt(d2.min(1)))
    print(f"reference {ref_label}: n {ref.shape[0]}; median nearest-neighbour RMS distance between different samples "
          f"{nn_rms:.4f} (normalised units)")
    print(f"{'set':<34}{'max |diff|':>12}{'RMS diff med':>14}{'RMS diff max':>14}{'RMS / NN':>10}{'contrast':>10}"
          f"{'dark':>7}")
    for label, x in sets:
        P = np.clip(x * s + m, 0, 1)
        if x.shape != ref.shape:
            print(f"{label}: shape {x.shape} differs from the reference {ref.shape}")
            continue
        d = x - ref
        rms = np.sqrt(np.mean(d * d, axis=1))
        print(f"{label:<34}{np.abs(d).max():>12.2e}{np.median(rms):>14.2e}{rms.max():>14.2e}"
              f"{np.median(rms) / nn_rms:>10.1e}{P.std(1).mean():>10.4f}{100 * (P < 0.1).mean():>6.2f}%")


if __name__ == "__main__":
    main()
