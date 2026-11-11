#!/usr/bin/env python3
"""A finished run's final assembly and its time outside the training steps (a CPU job; reads only).

  1. params_history.npy and every member of hidden_history.npz (names, order, dtypes, shapes, bytes) against the
     slice files stacked as the default assembly (--assemble files) does: equal bytes in the .npy, equal member
     bytes in the .npz (the zip entries' time stamps aside)
  2. time outside the steps: per slice, the wall time between consecutive slice files minus the training time the
     slice logged (validation, diagnostics, checkpoint, and the next slice's set-up); start-up (config.json written
     -> slice 0's training start, from its file time minus its logged time and the median per-slice host time);
     assembly (last slice file -> DONE)

Usage (from latent_physical_SGM/):  sbatch jobs/run_cpu_neuronic.sh history_check.py <run> [<run> ...]
Writes out/latent_physical_SGM/history_check/<job>/report.txt.
"""

import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(HERE), "out", "latent_physical_SGM")
JOB = os.environ.get("SLURM_JOB_ID", "manual")
OUT = os.path.join(OUT_ROOT, "history_check", JOB)
FAILED = []


def log(msg=""):
    print(msg, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(msg + "\n")


def npy_bytes(a):
    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


def check_run(run):
    rdir = os.path.join(OUT_ROOT, run)
    with open(os.path.join(rdir, "config.json")) as f:
        cfg = json.load(f)
    sdir = os.path.join(rdir, "slices")
    n = 100
    local, quad = bool(cfg.get("local")), bool(cfg.get("quad")) and int(cfg["K"]) > 0
    groups = {g: [] for g in ["theta", "W", "b"] + (["wl", "bl"] if local else []) + (["Q"] if quad else [])}
    for k in range(n):
        with np.load(os.path.join(sdir, f"slice_{k:03d}.npz")) as s:
            for g in groups:
                groups[g].append(s[g])
    stacked = {g: np.stack(v) for g, v in groups.items()}
    bad = []
    with open(os.path.join(rdir, "params_history.npy"), "rb") as f:
        if f.read() != npy_bytes(stacked["theta"]):
            bad.append("params_history.npy")
    taus = np.load(os.path.join(rdir, "hidden_history.npz"))["t"]
    expect = ["W", "b", "t"] + (["wl", "bl"] if local else []) + (["Q"] if quad else [])
    with zipfile.ZipFile(os.path.join(rdir, "hidden_history.npz")) as z:
        names = [nm[:-4] for nm in z.namelist()]
        if names != expect:
            bad.append(f"members {names} != {expect}")
        for m in expect:
            ref = taus if m == "t" else stacked[m]
            if m in names and z.read(m + ".npy") != npy_bytes(ref):
                bad.append(f"hidden_history.npz:{m}")
    ok = not bad
    if not ok:
        FAILED.append(run)
    log(f"  [{'ok' if ok else 'FAIL'}] {run}: params_history.npy and hidden_history.npz members {expect} equal to the "
        f"stacked slice files byte for byte; differing: {bad}")

    recs = {}
    with open(os.path.join(rdir, "train_log.jsonl")) as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                recs[int(r["slice"])] = r
    mt = np.array([os.stat(os.path.join(sdir, f"slice_{k:03d}.npz")).st_mtime for k in range(n)])
    linked = [k for k in range(n) if os.path.islink(os.path.join(sdir, f"slice_{k:03d}.npz"))]
    sec = np.array([float(recs[k].get("seconds", np.nan)) for k in range(n)])
    own = [k for k in range(1, n) if k not in linked and (k - 1) not in linked
           and recs[k].get("job") == recs[k - 1].get("job")]
    host = np.array([mt[k] - mt[k - 1] - sec[k] for k in own])
    train = float(np.nansum(sec[[k for k in range(n) if k not in linked]]))
    res = {"run": run, "bitwise_ok": ok, "slices_linked": len(linked), "train_s": train}
    if host.size:
        res.update(host_median_s=float(np.median(host)), host_mean_s=float(np.mean(host)), host_max_s=float(host.max()),
                   host_total_s=float(np.sum(host)), host_share_of_train=float(np.sum(host) / train))
        log(f"  {run}: host time per slice (wall between slice files - logged training time, {host.size} slices): "
            f"median {res['host_median_s']:.3f} s, mean {res['host_mean_s']:.3f} s, max {res['host_max_s']:.3f} s, "
            f"total {res['host_total_s']:.1f} s = {100 * res['host_share_of_train']:.2f}% of the {train:.0f} s of "
            f"training")
    k0 = min(k for k in range(n) if k not in linked)
    started = datetime.strptime(cfg["started"], "%Y-%m-%d %H:%M:%S").timestamp()
    if k0 == 0:
        res["startup_s"] = float(mt[0] - sec[0] - res.get("host_median_s", 0.0) - started)
        log(f"  {run}: start-up (config.json written -> slice 0's training start) about {res['startup_s']:.1f} s; "
            f"slice 0 (with compilation) {sec[0]:.1f} s, slice 1 {sec[1]:.1f} s ({recs[1].get('steps_per_s', 0):.0f} "
            f"steps/s)")
    done = os.path.join(rdir, "DONE")
    if os.path.exists(done):
        res["assembly_s"] = float(os.stat(done).st_mtime - mt[-1])
        log(f"  {run}: assembly (last slice file -> DONE) {res['assembly_s']:.1f} s")
    with open(os.path.join(rdir, "train_log.txt")) as f:
        for ln in f:
            if ln.startswith(("prototype reference", "assembled")):
                log("  | " + ln.rstrip()[-200:])
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    log(f"history_check job {JOB}, {time.strftime('%Y-%m-%d %H:%M:%S')}; runs {sys.argv[1:]}")
    res = [check_run(r) for r in sys.argv[1:]]
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"\n{'ALL OK' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
