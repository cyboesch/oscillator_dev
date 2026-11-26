#!/usr/bin/env python3
"""Has the training of each slice converged? (2026-11-25; works on finished and on running runs.)

From the validation curves the trainer stores (--val-every 1000: val_curves/slice_<k>.npz, the loss of the selected
(EMA) parameters on the same 4096 validation samples every 1000 steps): for every slice the change of the validation
loss over the last third of its steps and over the last check, each with the paired standard error over the
samples. A slice counts as "still falling" if the last-third change is negative beyond 3 standard errors.
For scale, between existing CelebA-32 runs 1 FID is a loss difference of about 50 at t ~ 1 and about 1000 at t = 0.

For a tree run also J per slice (from the slice files): rms J / T, and how much J changed from the previous slice
relative to its size (a table that still grows by a large fraction per slice is limited by its learning rate).

Usage (CPU job, from this folder):
  sbatch jobs/run_cpu.sh convergence_report.py LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--slices "20 40 60 80 90 95 99"]
RUN_DIR: absolute, or relative to the locked out root (out/latent_physical_SGM) or to this folder's out/.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tree_model as tm  # noqa: E402
lm = tm.lm
T = lm.TEMP


def find(run):
    for root in ("", lm.OUT_ROOT, os.path.join(HERE, "out")):
        p = os.path.join(root, run)
        if os.path.isdir(os.path.join(p, "slices")):
            return p
    raise SystemExit(f"run {run} not found")


def slice_row(run_dir, k):
    f = os.path.join(run_dir, "val_curves", f"slice_{k:03d}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    c, s = z["val_per_sample"].astype(np.float64), z["steps"]
    n = c.shape[0]
    third = max(1, (n - 1) // 3)
    d3, d1 = c[-1] - c[-1 - third], c[-1] - c[-2]
    se = lambda d: d.std() / np.sqrt(d.size)   # noqa: E731
    return {"t": float(z["t"]), "steps": int(s[-1]), "val": float(c[-1].mean()), "total": float(c[-1].mean() - c[0].mean()),
            "d3": float(d3.mean()), "se3": float(se(d3)), "steps3": int(s[-1] - s[-1 - third]),
            "d1": float(d1.mean()), "se1": float(se(d1)), "steps1": int(s[-1] - s[-2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--slices", default="")
    args = ap.parse_args()
    runs = [(r.split("=", 1)[0], find(r.split("=", 1)[1])) for r in args.runs]
    for label, d in runs:
        ks = sorted(int(f[6:9]) for f in os.listdir(os.path.join(d, "val_curves")) if f.startswith("slice_")) \
            if os.path.isdir(os.path.join(d, "val_curves")) else []
        print(f"\n=== {label}: {d}\n    {len(ks)} slices with validation curves" + (f" (0-{ks[-1]})" if ks else ""), flush=True)
        if not ks:
            continue
        show = [int(s) for s in args.slices.split()] if args.slices else ks
        rows = {k: slice_row(d, k) for k in ks}
        falling = [k for k in ks if rows[k]["d3"] < -3 * rows[k]["se3"]]
        print(f"    {'slice':>5s} {'t':>6s} {'steps':>6s} {'val loss':>11s} {'whole slice':>12s} {'last third':>20s} {'last check':>20s}  J: rms/T, change from the slice before")
        prevJ = {}
        for k in ks:
            r = rows[k]
            jtxt = ""
            sf = os.path.join(d, "slices", f"slice_{k:03d}.npz")
            if os.path.exists(sf):
                with np.load(sf) as z:
                    for kj in tm.J_KEYS:
                        if kj in z.files:
                            J = z[kj].astype(np.float64)
                            rms = np.sqrt(np.mean(J * J))
                            if rms > 0:
                                ch = np.sqrt(np.mean((J - prevJ[kj]) ** 2)) / rms if kj in prevJ else float("nan")
                                jtxt += f"  {kj} {rms / T:.3f}, {100 * ch:.0f}%"
                            prevJ[kj] = J
            if k in show:
                flag = " *" if k in falling else "  "
                print(f"    {k:5d} {r['t']:6.3f} {r['steps']:6d} {r['val']:11.1f} {r['total']:+12.1f} "
                      f"{r['d3']:+9.2f} +- {r['se3']:5.2f}{flag} {r['d1']:+9.2f} +- {r['se1']:5.2f}  {jtxt}")
        print(f"    still falling in the last third (beyond 3 standard errors, marked *): {len(falling)} of {len(ks)} slices"
              + (f": {falling[0]}-{falling[-1]} " + (f"(all of {falling[0]}-{falling[-1]})" if falling == list(range(falling[0], falling[-1] + 1)) else f"({falling})") if falling else ""))
        late = [k for k in ks if rows[k]["t"] < 0.5]
        if late:
            print(f"    slices with t < 0.5: last-third change summed {sum(rows[k]['d3'] for k in late):+.0f}, last-check change "
                  f"summed {sum(rows[k]['d1'] for k in late):+.0f} (1 FID is about 1000 at t = 0)")


if __name__ == "__main__":
    main()
