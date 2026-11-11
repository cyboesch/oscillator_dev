#!/usr/bin/env python3
"""End-to-end checks of the training changes of 2026-11-11 against a run made with the code before them; a GPU job.

Reference: bench_mixst_K784 (job 4222240, 2 slices of four-digit Potts 784, stencil + mixed, --select ema), made
with the code before best_params was skipped under --select ema. Each case trains 2 slices with the reference's
flags (plus the case's own) under a new name, in a subprocess (train_latent.py as the pipelines run it):
  1. 'nobest': no extra flags (best_params is now skipped under --select ema): slice files bitwise equal to the
     reference (theta, W, b, Q, wl, bl, key, loss trace, validation per-sample losses and indices, and the record
     except timings and code hashes)
  2. 'lean': --step-impl lean: bitwise equal as well
  3. 'fast': --step-impl lean --ref-kmeans gpu (the reference changes at rounding level or more): start-up time and
     the per-slice records against the reference's
Wall times of every case from the process start to its end (2 slices, start-up included).

Usage (from latent_physical_SGM/):  sbatch --time=00:30:00 jobs/run_gpu_neuronic.sh train_checks.py [cases]
Writes out/latent_physical_SGM/train_checks/<job>/report.txt; the runs go to out/latent_physical_SGM/<case>_<job>/.
Exit code 1 if a bitwise check fails.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(HERE), "out", "latent_physical_SGM")
JOB = os.environ.get("SLURM_JOB_ID", "manual")
OUT = os.path.join(OUT_ROOT, "train_checks", JOB)
REF = "bench_mixst_K784"
BASE = ["--problem", "mnist28_d0123", "--K", "784", "--select", "ema", "--hidden", "potts", "--quad", "--local",
        "--visible", "quartic", "--radius", "1", "--pair-terms", "lin", "--proto-init", "pooled", "--max-slices", "2",
        "--visible-impl", "stencil", "--precision", "mixed"]
CASES = {"nobest": ([], True), "lean": (["--step-impl", "lean"], True),
         "fast": (["--step-impl", "lean", "--ref-kmeans", "gpu"], False)}
VOLATILE = {"seconds", "steps_per_s", "job", "code_sha256"}       # record fields that may differ
FAILED = []


def log(msg=""):
    print(msg, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(msg + "\n")


def compare_slices(run, bitwise):
    rows = []
    for k in (0, 1):
        a = np.load(os.path.join(OUT_ROOT, run, "slices", f"slice_{k:03d}.npz"))
        b = np.load(os.path.join(OUT_ROOT, REF, "slices", f"slice_{k:03d}.npz"))
        ra, rb = json.loads(str(a["record"])), json.loads(str(b["record"]))
        arrays = sorted((set(a.files) | set(b.files)) - {"record"})
        diff = [m for m in arrays if not (m in a.files and m in b.files and a[m].shape == b[m].shape
                                          and a[m].dtype == b[m].dtype and np.array_equal(a[m], b[m]))]
        rdiff = {f: (ra.get(f), rb.get(f)) for f in set(ra) | set(rb) if f not in VOLATILE and ra.get(f) != rb.get(f)}
        if bitwise:
            ok = not diff and not rdiff
            if not ok:
                FAILED.append(f"{run} slice {k}")
            log(f"  [{'ok' if ok else 'FAIL'}] {run} slice {k} vs {REF}: arrays differing {diff}; record fields "
                f"differing {rdiff}")
        else:
            rel = {m: float(np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-300))
                   for m in ("theta", "W", "b", "Q", "wl", "bl") if m in a.files and m in b.files}
            log(f"  {run} slice {k} vs {REF}: relative differences {json.dumps({m: f'{v:.2e}' for m, v in rel.items()})}")
            for f in ("val_loss", "best_loss", "best_epoch", "W_norm", "cos_W_Wref", "eff_states", "Q_norm"):
                log(f"      {f}: {ra.get(f)} (reference {rb.get(f)})")
        rows.append({"slice": k, "arrays_differing": diff, "record_differing": {f: list(v) for f, v in rdiff.items()},
                     "seconds": ra.get("seconds"), "ref_seconds": rb.get("seconds")})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    todo = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CASES)
    log(f"train_checks job {JOB}, {time.strftime('%Y-%m-%d %H:%M:%S')}; cases {todo}; reference {REF}")
    results = {}
    for case in todo:
        extra, bitwise = CASES[case]
        run = f"bench_{case}_K784_{JOB}"
        cmd = [sys.executable, "-u", os.path.join(HERE, "train_latent.py"), "--name", run] + BASE + extra
        log(f"\n=== {case}: {' '.join(cmd[2:])}")
        t = time.time()
        r = subprocess.run(cmd)
        wall = time.time() - t
        log(f"  exit code {r.returncode}; wall time {wall:.1f} s (2 slices, start-up included)")
        if r.returncode:
            FAILED.append(f"{case} exit {r.returncode}")
            continue
        with open(os.path.join(OUT_ROOT, run, "train_log.txt")) as f:
            for ln in f:
                if ln.startswith(("prototype reference", "slice ")):
                    log("  | " + ln.rstrip()[:260])
        results[case] = {"run": run, "wall_s": wall, "slices": compare_slices(run, bitwise)}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    log(f"\n{'ALL BITWISE CHECKS PASSED' if not FAILED else 'FAILED: ' + '; '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
