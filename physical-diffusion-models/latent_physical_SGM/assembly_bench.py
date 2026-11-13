#!/usr/bin/env python3
"""The final assembly of a finished run, timed both ways on a CPU node (reads the run; writes only to a new folder):

  files   train_latent.write_history with no slices held: every slice file is read, then params_history.npy and
          hidden_history.npz are written; the cost of the default assembly (--assemble files), whose inline code reads
          and writes the same arrays in the same order
  memory  the slices loaded into memory first (not timed: a --assemble memory run holds them already), then
          write_history writes the same two files from memory (--assemble memory)
'files' runs first, so that its reads do not come from this node's page cache. Both outputs are compared byte for
byte with the run's own params_history.npy and with every member of its hidden_history.npz (names, order, bytes;
the zip entries' time stamps aside), in chunks, and then deleted (each is as large as the run's own history).

Usage (from latent_physical_SGM/):  sbatch --mem=32G --time=00:30:00 jobs/run_cpu_neuronic.sh assembly_bench.py <run>
Writes out/latent_physical_SGM/assembly_bench/<job>/report.txt and results.json. Exit code 1 if a file differs.
"""

import json
import os
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import train_latent  # noqa: E402  (write_history)

JOB = os.environ.get("SLURM_JOB_ID", "manual")
OUT = os.path.join(lm.OUT_ROOT, "assembly_bench", JOB)
CHUNK = 1 << 26                                   # 64 MiB per read in the comparisons


def log(msg=""):
    print(msg, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(msg + "\n")


def same_stream(fa, fb):
    """True if two open binary files hold the same bytes (read in chunks)."""
    while True:
        a, b = fa.read(CHUNK), fb.read(CHUNK)
        if a != b:
            return False
        if not a:
            return True


def compare(new_dir, run_dir):
    """Names of the files / members of new_dir's history that differ from run_dir's (empty: byte-identical)."""
    bad = []
    with open(os.path.join(new_dir, "params_history.npy"), "rb") as fa, \
            open(os.path.join(run_dir, "params_history.npy"), "rb") as fb:
        if not same_stream(fa, fb):
            bad.append("params_history.npy")
    with zipfile.ZipFile(os.path.join(new_dir, "hidden_history.npz")) as za, \
            zipfile.ZipFile(os.path.join(run_dir, "hidden_history.npz")) as zb:
        if za.namelist() != zb.namelist():
            bad.append(f"members {za.namelist()} != {zb.namelist()}")
        for nm in zb.namelist():
            if nm in za.namelist():
                with za.open(nm) as fa, zb.open(nm) as fb:
                    if not same_stream(fa, fb):
                        bad.append(f"hidden_history.npz:{nm}")
    return bad


def main():
    run = sys.argv[1]
    run_dir = os.path.join(lm.OUT_ROOT, run)
    slice_dir = os.path.join(run_dir, "slices")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    local, quad = bool(cfg["local"]), bool(cfg["quad"]) and int(cfg["K"]) > 0
    harm, patch = bool(cfg.get("harmonic")), bool(cfg.get("patch"))     # --harmonic, --patch: their members too
    patch_quad = patch and bool(cfg.get("quad"))
    tail = ((["V", "c"] if harm else []) + ((["Wp", "bp"] + (["Qp"] if patch_quad else [])) if patch else []))
    taus = lm.time_grid_pts(lm.config_time_grid(cfg))      # the run's slice times (its --time-grid; older runs uniform)
    n = lm.N_TIME_STEPS
    log(f"assembly_bench job {JOB}, {time.strftime('%Y-%m-%d %H:%M:%S')}; run {run} (K {cfg['K']}, local {local}, "
        f"quad {quad}); history {os.path.getsize(os.path.join(run_dir, 'hidden_history.npz')) / 1e9:.2f} GB")
    res, failed = {"run": run}, []
    for mode in ("files", "memory"):
        d = os.path.join(OUT, mode)
        os.makedirs(d, exist_ok=True)
        held = {}
        if mode == "memory":                     # what a --assemble memory run holds at its end (not timed)
            t = time.time()
            for k in range(n):
                with np.load(os.path.join(slice_dir, f"slice_{k:03d}.npz")) as s:
                    held[k] = {g: s[g] for g in ["theta", "W", "b"] + (["wl", "bl"] if local else [])
                               + (["Q"] if quad else []) + tail}
            log(f"  (memory: the {n} slices loaded in {time.time() - t:.1f} s, not part of the assembly)")
        t = time.time()
        shape = train_latent.write_history(d, slice_dir, n, taus, local, quad, held, harmonic=harm, patch=patch,
                                           patch_quad=patch_quad)
        res[f"{mode}_s"] = time.time() - t
        del held
        bad = compare(d, run_dir)
        res[f"{mode}_differing"] = bad
        if bad:
            failed.append(mode)
        log(f"  [{'ok' if not bad else 'FAIL'}] {mode}: write_history ({shape}) in {res[f'{mode}_s']:.1f} s; equal to "
            f"the run's params_history.npy and hidden_history.npz byte for byte; differing: {bad}")
        for f in ("params_history.npy", "hidden_history.npz"):
            os.remove(os.path.join(d, f))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"\n{'ALL OK' if not failed else 'FAILED: ' + ', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
