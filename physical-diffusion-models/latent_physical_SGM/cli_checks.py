#!/usr/bin/env python3
"""End-to-end checks of the train_latent.py options of 2026-11-11 through its command line, on the small problem
gauss2_8 (8x8 lattice, two classes, 10,000 images); a CPU job (it also runs on a GPU).

Every case trains with all the new options at once: Potts 8 --quad --local, --visible quartic --radius 1
--pair-terms lin, --select ema, --visible-impl stencil --precision mixed --step-impl lean --ref-kmeans gpu (the JAX
k-means, on whatever device JAX uses) and --steps-schedule '0:0.01,50:0.02' (101-201 steps per slice before slice
50, 401-601 after), each case in a subprocess as the pipelines run it:
  A  --assemble memory, killed (SIGKILL, as a time limit or a node failure ends a job) once slice 40 is written
  B  A's command again: resumes after A's last slice; its own slices come from memory, A's from their files
  C  the same options with --assemble files, in one go
  D  --init-from C --init-slices 90 --steps-schedule '90:0.03' --assemble memory (two keys of the --init-from free
     set; slices 0-89 are links to C's)
Checks: (1) config.json records the options; (2) every slice's step count and step factor follow the schedule;
(3) A+B's slice files equal C's bitwise (every array; the record except timings): an interrupted run resumes
exactly; (4) params_history.npy and every hidden_history.npz member (names, order, dtypes, values) equal the
stacked slice files byte for byte, for A+B, C and D, and A+B's equal C's (the zip entries' time stamps aside);
(5) the logs report the memory / file split of the assembly.

Usage (from latent_physical_SGM/):  sbatch --time=00:30:00 jobs/run_cpu_neuronic.sh cli_checks.py
Writes out/latent_physical_SGM/cli_checks/<job>/report.txt and results.json; the runs go to
out/latent_physical_SGM/cli_{mem,files,init}_<job>/. Exit code 1 if a check fails.
"""

import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import train_latent  # noqa: E402  (slice_steps, parse_steps_schedule, steps_factor, steps_schedule_str)

JOB = os.environ.get("SLURM_JOB_ID", "manual")
OUT = os.path.join(lm.OUT_ROOT, "cli_checks", JOB)
SCHEDULE, SCHEDULE_D = "0:0.01,50:0.02", "90:0.03"
FLAGS = ["--problem", "gauss2_8", "--K", "8", "--hidden", "potts", "--quad", "--local", "--select", "ema",
         "--visible", "quartic", "--radius", "1", "--pair-terms", "lin", "--proto-init", "pooled",
         "--visible-impl", "stencil", "--precision", "mixed", "--step-impl", "lean", "--ref-kmeans", "gpu"]
VOLATILE = {"seconds", "steps_per_s", "job", "code_sha256"}       # record fields that may differ between runs
KILL_AFTER = 40                                                   # A is killed once this slice's file exists
CHECKS = []


def log(msg=""):
    print(msg, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(msg + "\n")


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    log(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def command(name, *extra):
    return [sys.executable, "-u", os.path.join(HERE, "train_latent.py"), "--name", name] + FLAGS + list(extra)


def run_dir(name):
    return os.path.join(lm.OUT_ROOT, name)


def train(name, *extra, kill_after=None):
    """train_latent.py in a subprocess; with kill_after, SIGKILL once that slice's file exists. Returns (exit code,
    wall seconds)."""
    cmd = command(name, *extra)
    log(f"\n=== {' '.join(cmd[2:])}" + (f"   (killed once slice {kill_after} is written)" if kill_after else ""))
    t = time.time()
    p = subprocess.Popen(cmd)
    if kill_after is not None:
        target = os.path.join(run_dir(name), "slices", f"slice_{kill_after:03d}.npz")
        while p.poll() is None and not os.path.exists(target):
            time.sleep(0.1)
        if p.poll() is None:
            p.kill()
    rc = p.wait()
    wall = time.time() - t
    log(f"  exit code {rc}; wall time {wall:.1f} s")
    return rc, wall


def n_slices(name):
    k = 0
    while os.path.exists(os.path.join(run_dir(name), "slices", f"slice_{k:03d}.npz")):
        k += 1
    return k


def npy_bytes(a):
    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


def stacked_slices(name):
    """The slice files stacked as the default assembly (--assemble files) stacks them, and the history's members."""
    with open(os.path.join(run_dir(name), "config.json")) as f:
        cfg = json.load(f)
    local, quad = bool(cfg["local"]), bool(cfg["quad"]) and int(cfg["K"]) > 0
    groups = ["theta", "W", "b"] + (["wl", "bl"] if local else []) + (["Q"] if quad else [])
    parts = {g: [] for g in groups}
    for k in range(lm.N_TIME_STEPS):
        with np.load(os.path.join(run_dir(name), "slices", f"slice_{k:03d}.npz")) as s:
            for g in groups:
                parts[g].append(s[g])
    members = ["W", "b", "t"] + (["wl", "bl"] if local else []) + (["Q"] if quad else [])
    return {g: np.stack(v) for g, v in parts.items()}, members


def history_bytes(name):
    """params_history.npy's bytes and {member: bytes} of hidden_history.npz (in the archive's order)."""
    with open(os.path.join(run_dir(name), "params_history.npy"), "rb") as f:
        ph = f.read()
    with zipfile.ZipFile(os.path.join(run_dir(name), "hidden_history.npz")) as z:
        hh = {nm[:-4]: z.read(nm) for nm in z.namelist()}
    return ph, hh


def check_history(name):
    """(4): the run's history files against its stacked slice files, byte for byte."""
    stacked, members = stacked_slices(name)
    ph, hh = history_bytes(name)
    taus = np.asarray(lm.forward_time_pts(), dtype=np.float64)
    bad = [] if ph == npy_bytes(stacked["theta"]) else ["params_history.npy"]
    if list(hh) != members:
        bad.append(f"members {list(hh)} != {members}")
    bad += [m for m in members if m in hh and hh[m] != npy_bytes(taus if m == "t" else stacked[m])]
    check(f"{name}: params_history.npy and hidden_history.npz {members} equal the stacked slice files byte for byte",
          not bad, f"differing {bad}" if bad else "")


def records(name):
    out = {}
    for k in range(lm.N_TIME_STEPS):
        with np.load(os.path.join(run_dir(name), "slices", f"slice_{k:03d}.npz")) as s:
            out[k] = json.loads(str(s["record"]))
    return out


def check_steps(name, schedule, slices):
    """(2): step counts and factors of the given slices under the parsed schedule."""
    recs = records(name)
    bad = [k for k in slices if recs[k]["steps"] != train_latent.slice_steps(k, 1.0, schedule)
           or recs[k].get("steps_factor") != train_latent.steps_factor(schedule, k)]
    counts = sorted({recs[k]["steps"] for k in slices})
    check(f"{name}: step counts 100 m + 1 and factors of '{train_latent.steps_schedule_str(schedule)}' in slices "
          f"{min(slices)}-{max(slices)} (counts {counts[0]}..{counts[-1]})",
          not bad and all(c % 100 == 1 for c in counts), f"slices differing {bad}" if bad else "")


def check_config(name, expect):
    with open(os.path.join(run_dir(name), "config.json")) as f:
        cfg = json.load(f)
    bad = {k: (cfg.get(k), v) for k, v in expect.items() if cfg.get(k) != v}
    check(f"{name}: config.json records {sorted(expect)}", not bad, f"differing {bad}" if bad else "")


def assembly_line(name):
    with open(os.path.join(run_dir(name), "train_log.txt")) as f:
        lines = [ln.rstrip() for ln in f if ln.startswith("assembled")]
    return lines[-1] if lines else ""


def main():
    os.makedirs(OUT, exist_ok=True)
    log(f"cli_checks job {JOB}, {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sched, sched_d = train_latent.parse_steps_schedule(SCHEDULE), train_latent.parse_steps_schedule(SCHEDULE_D)
    mem, files, init = f"cli_mem_{JOB}", f"cli_files_{JOB}", f"cli_init_{JOB}"
    for name in (mem, files, init):
        if os.path.exists(run_dir(name)):
            raise SystemExit(f"{run_dir(name)} exists already; these checks need new run folders")
    res = {}

    rc_a, res["wall_A"] = train(mem, "--steps-schedule", SCHEDULE, "--assemble", "memory", kill_after=KILL_AFTER)
    n_a = n_slices(mem)
    check(f"A interrupted after slice {KILL_AFTER} (exit code {rc_a}, {n_a} slices written)",
          rc_a != 0 and KILL_AFTER < n_a < lm.N_TIME_STEPS)
    rc_b, res["wall_B"] = train(mem, "--steps-schedule", SCHEDULE, "--assemble", "memory")
    rc_c, res["wall_C"] = train(files, "--steps-schedule", SCHEDULE, "--assemble", "files")
    rc_d, res["wall_D"] = train(init, "--steps-schedule", SCHEDULE_D, "--assemble", "memory", "--init-from", files,
                                "--init-slices", "90")
    check("B, C and D complete (exit code 0, DONE written)",
          rc_b == rc_c == rc_d == 0 and all(os.path.exists(os.path.join(run_dir(n), "DONE")) for n in (mem, files, init)),
          f"exit codes {rc_b}, {rc_c}, {rc_d}")
    if rc_b or rc_c or rc_d:
        return finish(res)

    log("\n=== (1) config.json, (2) step counts")
    common = {"visible_impl": "stencil", "precision": "mixed", "step_impl": "lean", "ref_kmeans": "gpu"}
    check_config(mem, dict(common, steps_schedule=train_latent.steps_schedule_str(sched), assemble="memory"))
    check_config(files, dict(common, steps_schedule=train_latent.steps_schedule_str(sched), assemble="files"))
    check_config(init, dict(common, steps_schedule=train_latent.steps_schedule_str(sched_d), assemble="memory",
                            init_from=files, init_slices=90))
    for name in (mem, files):
        check_steps(name, sched, range(lm.N_TIME_STEPS))
    check_steps(init, sched_d, range(90, lm.N_TIME_STEPS))

    log("\n=== (3) A+B (killed and resumed) against C (one go), slice by slice")
    diff_a, diff_r = [], []
    for k in range(lm.N_TIME_STEPS):
        with np.load(os.path.join(run_dir(mem), "slices", f"slice_{k:03d}.npz")) as a, \
                np.load(os.path.join(run_dir(files), "slices", f"slice_{k:03d}.npz")) as c:
            names = sorted((set(a.files) | set(c.files)) - {"record"})
            diff_a += [(k, m) for m in names if not (m in a.files and m in c.files and a[m].dtype == c[m].dtype
                                                     and a[m].shape == c[m].shape and np.array_equal(a[m], c[m]))]
            ra, rc = json.loads(str(a["record"])), json.loads(str(c["record"]))
            diff_r += [(k, f) for f in sorted(set(ra) | set(rc)) if f not in VOLATILE and ra.get(f) != rc.get(f)]
    check("A+B slice files equal C's bitwise (every array; records except timings and job)", not diff_a and not diff_r,
          f"arrays differing {diff_a[:10]}, record fields differing {diff_r[:10]}" if diff_a or diff_r else "")
    linked = [k for k in range(90) if os.path.islink(os.path.join(run_dir(init), "slices", f"slice_{k:03d}.npz"))]
    check("D: slices 0-89 are links (--init-from)", len(linked) == 90, f"{len(linked)} links")

    log("\n=== (4) history files")
    for name in (mem, files, init):
        check_history(name)
    (ph_m, hh_m), (ph_f, hh_f) = history_bytes(mem), history_bytes(files)
    same = ph_m == ph_f and list(hh_m) == list(hh_f) and all(hh_m[m] == hh_f[m] for m in hh_f)
    check("A+B's history (memory and files) equals C's (files) byte for byte (params_history.npy, every member of "
          "hidden_history.npz)", same)

    log("\n=== (5) assembly log lines")
    pat = re.compile(r"\((\d+) slices from memory, (\d+) from their files\)")
    for name, n_mem in ((mem, lm.N_TIME_STEPS - n_a), (init, 10)):
        line = assembly_line(name)
        m = pat.search(line)
        check(f"{name}: '{line[:150]}'", m is not None and (int(m.group(1)), int(m.group(2))) ==
              (n_mem, lm.N_TIME_STEPS - n_mem), f"expected {n_mem} from memory")
        res[f"assembly_{name}"] = line
    line = assembly_line(files)
    check(f"{files}: '{line[:150]}' (the default assembly)", line.startswith("assembled params_history.npy (100,")
          and "from memory" not in line)
    res["slices_A"] = n_a
    finish(res)


def finish(res):
    n_fail = sum(not c["ok"] for c in CHECKS)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"results": res, "checks": CHECKS}, f, indent=1, default=str)
    log(f"\n=== {len(CHECKS)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
