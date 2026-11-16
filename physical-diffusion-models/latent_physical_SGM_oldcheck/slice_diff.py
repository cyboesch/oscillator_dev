#!/usr/bin/env python3
"""Relative differences |a - b| / |b| (Frobenius) of the parameter arrays (theta, W, b, Q, wl, bl) and the logged
validation loss between the slice files of runs and one reference run, for the given slices (reads only; CPU).

Usage (from latent_physical_SGM/):
  sbatch jobs/run_cpu_neuronic.sh slice_diff.py <reference run> <slices, e.g. 0,1> <run> [<run> ...]
Writes out/latent_physical_SGM/slice_diff/<job>/report.txt.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(HERE), "out", "latent_physical_SGM")
JOB = os.environ.get("SLURM_JOB_ID", "manual")
OUT = os.path.join(OUT_ROOT, "slice_diff", JOB)


def log(msg=""):
    print(msg, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(msg + "\n")


def main():
    ref, slices, runs = sys.argv[1], [int(k) for k in sys.argv[2].split(",")], sys.argv[3:]
    os.makedirs(OUT, exist_ok=True)
    log(f"slice_diff job {JOB}, {time.strftime('%Y-%m-%d %H:%M:%S')}; reference {ref}; slices {slices}")
    for run in runs:
        for k in slices:
            with np.load(os.path.join(OUT_ROOT, run, "slices", f"slice_{k:03d}.npz")) as a, \
                    np.load(os.path.join(OUT_ROOT, ref, "slices", f"slice_{k:03d}.npz")) as b:
                rel = {m: float(np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-300))
                       for m in ("theta", "W", "b", "Q", "wl", "bl") if m in a.files and m in b.files}
                ra, rb = json.loads(str(a["record"])), json.loads(str(b["record"]))
                va, vb = ra["val_loss"], rb["val_loss"]
                ta, tb = ra.get("t"), rb.get("t")
            log(f"  {run} slice {k}: relative differences "
                + ", ".join(f"{m} {v:.2e}" for m, v in rel.items())
                + f"; val loss {va:.10g} vs {vb:.10g} (relative {abs(va - vb) / abs(vb):.1e})"
                + (f"; NOTE: different slice times t {ta:.5f} vs {tb:.5f} (another --time-grid)"
                   if ta is not None and tb is not None and abs(ta - tb) > 1e-9 else ""))


if __name__ == "__main__":
    main()
