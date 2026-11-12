#!/usr/bin/env python3
"""End-to-end checks of train_latent.py --time-grid and of both samplers on a time-grid run, through the command lines
as the pipelines run them, on the small problem gauss2_8 (8x8 lattice; Potts 8 --quad --local, quartic, radius 1, lin,
--select ema, stencil + mixed, --val-every 100); CPU jobs. On the CPU the default-path sampler needs about 7 min per
solve of 4 trajectories of these toy runs (job 4223634), so run the parts in two jobs of under 30 min:
U1,U2,P then S,X,R (with --runs-job = the first job's id).

Default path (no --time-grid) against the code from before the option: copies of latent_model.py, train_latent.py,
sample_latent.py, sample_fast.py and of the analysis scripts taken 2026-11-11 21:00
(out/latent_physical_SGM/time_grid_check/orig/), imported in-process with this folder's latent_model (only additions
there; the copy of latent_model.py is hashed, not run):
  U1  train_latent.py, 2 slices at --steps-scale 0.01, original vs current: slice files equal bit for bit (every
      array; the record except timings, job and code hashes); config.json equal but for the new 'time_grid' 'uniform'
  U2  a 100-slice uniform run (101 steps per slice): hidden_history.npz 't' = lm.forward_time_pts() bit for bit;
      sample_latent's default path as main() runs it (load_params, slice_times, make_drift, sample) against the
      original module's (load_params, make_drift, sample) on 4 trajectories: final states bit for bit (same process);
      the new manifest keys are exactly 'time_grid' ('uniform') and 'knots_t' (= forward_time_pts())
Power grid (--time-grid power:1.5), part P:
  A   5 slices at --steps-scale 0.01, killed (SIGKILL) once slice 2 is written;  B  A's command again: resumes at 3
  C   A's run with --time-grid uniform and with power:2: refused (the config differs); with 'power:1.50': accepted
  D   --init-from A+B --init-slices 2 --max-slices 5: on the uniform grid refused; on power:1.5 slices 0-1 linked,
      2-4 trained, equal to A+B's slices 2-4 bit for bit
  E   a 100-slice power:1.5 run (101 steps per slice): DONE; hidden_history.npz 't' = the grid bit for bit
  On A+B, D and E: config.json 'time_grid' = 'power:1.5'; every slice record's t = the grid's t_k, lr_v = the grid's
  learning rate and steps = slice_steps(k, scale, schedule, grid) (A+B: where the uniform grid's differ); the log
  has the grid line, every slice's t and the resume line with slice 3's t
  S   sample_latent.py (default path) and --fast on E through the command line, --s-n trajectories each: the start
      lines name the grid; the manifests record time_grid 'power:1.5' and knots_t = the grid; the fast sampler's
      _steps.npz s = sort(5 - grid); final states finite; fast vs default (about 1e-4 is the usual amplification of
      rounding by the step-size control). Control: the original sample_latent (which knows no grid) on E, i.e. a
      sampler interpolating a power-grid run on the uniform knots: its final states against the default path's
  T   'power:1.5@40' with U2's flags, 42 slices: slices 0-40 equal U2's bit for bit (uniform knots, lr and steps),
      slice 41 at the tail's t
  I   (added after review) --init-from across grids where the linked slices sit at identical times: a uniform run
      of 5 slices; power:1.5@3 --init-slices 4 from it accepted (slices 0-3 linked, 4-5 trained at the tail's t and
      steps), --init-slices 5 refused (slice 4 sits at another t); power:1.5 --init-slices 1 accepted (t = 5)
  X   every finished run under out/latent_physical_SGM (config.json and hidden_history.npz): its recorded 't' against
      its config's grid (latent_model.run_time_grid), i.e. sample_latent.load_params accepts every existing run
  R   the analysis scripts edited for --time-grid, original against current, in-process: summarize_training.py and
      plot_learning_curves.py give identical text output on two uniform runs (U2's and --r-uniform); on E against a
      uniform run they pair only slices 0 and 99 (t = 5 and 0) and name both runs' t; inspect_states.py draws E's
      slice data at the grid's t
Writes out/latent_physical_SGM/time_grid_check/<job>/ (report.txt, results.json, the subprocess logs and the
in-process outputs); the runs go to out/latent_physical_SGM/tgc_<case>_<runs job>/. Exit code 1 if a check fails.
Usage (from latent_physical_SGM/):
  J=$(sbatch --parsable --time=00:30:00 --cpus-per-task=4 --mem=16G jobs/run_cpu_neuronic.sh time_grid_check.py --parts U1,U2,P)
  sbatch --dependency=afterany:$J --time=00:30:00 --cpus-per-task=4 --mem=16G jobs/run_cpu_neuronic.sh time_grid_check.py --parts S,X,R --runs-job $J
"""

import argparse
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

JOB = os.environ.get("SLURM_JOB_ID", "manual")
ROOT = os.path.join(lm.OUT_ROOT, "time_grid_check")
OUT = os.path.join(ROOT, JOB)
ORIG = os.path.join(ROOT, "orig")                 # the modules before --time-grid (2026-11-11 21:00)
GRID = "power:1.5"
FLAGS = ["--problem", "gauss2_8", "--K", "8", "--hidden", "potts", "--quad", "--local", "--select", "ema",
         "--visible", "quartic", "--radius", "1", "--pair-terms", "lin", "--proto-init", "pooled",
         "--visible-impl", "stencil", "--precision", "mixed", "--val-every", "100"]
VOLATILE = {"seconds", "steps_per_s", "job", "code_sha256"}       # record fields that may differ between runs
CASES = {"U1": ("u1o", "u1n"), "U2": ("u2",), "P": ("pow", "du", "dp", "full"),
         "I": ("iu", "it", "ir", "ip")}                                     # the runs each part makes
CHECKS = []
RES = {}


def log(msg=""):
    print(msg, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(msg + "\n")


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    log(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def orig_module(name):
    """The original ORIG/<name>.py, imported in-process; its 'import latent_model' finds this folder's module.
    Executing it puts ORIG at the front of sys.path (each script inserts its own folder), so after this call a plain
    'import <name>' would find the original: load current modules with cur_module."""
    spec = importlib.util.spec_from_file_location(f"{name}_orig", os.path.join(ORIG, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cur_module(name):
    """A fresh instance of this folder's <name>.py, loaded by its path (not through sys.path, see orig_module)."""
    spec = importlib.util.spec_from_file_location(f"{name}_current", os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_dir(name):
    return os.path.join(lm.OUT_ROOT, name)


def subprocess_log(cmd, tag, kill_file=None):
    """Runs cmd with its output in OUT/<tag>.log; with kill_file, SIGKILL once that file exists. Returns (exit code,
    output text, wall seconds)."""
    path = os.path.join(OUT, f"{tag}.log")
    t = time.time()
    with open(path, "w") as fo:
        p = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT)
        if kill_file is not None:
            while p.poll() is None and not os.path.exists(kill_file):
                time.sleep(0.1)
            if p.poll() is None:
                p.kill()
        rc = p.wait()
    with open(path) as f:
        text = f.read()
    return rc, text, time.time() - t


def train(name, *extra, orig=False, kill_after=None, tag=None):
    """train_latent.py (orig: the original copy, through this script's --orig-train mode) in a subprocess."""
    script = [os.path.abspath(__file__), "--orig-train"] if orig else [os.path.join(HERE, "train_latent.py")]
    cmd = [sys.executable, "-u"] + script + ["--name", name] + FLAGS + list(extra)
    kill = os.path.join(run_dir(name), "slices", f"slice_{kill_after:03d}.npz") if kill_after is not None else None
    rc, text, wall = subprocess_log(cmd, tag or name, kill)
    log(f"\n=== {'original ' if orig else ''}train_latent.py --name {name} {' '.join(extra)}"
        + (f"  (killed once slice {kill_after} is written)" if kill_after is not None else "")
        + f": exit code {rc}, {wall:.0f} s")
    return rc, text


def n_slices(name):
    k = 0
    while os.path.exists(os.path.join(run_dir(name), "slices", f"slice_{k:03d}.npz")):
        k += 1
    return k


def slice_data(name, k):
    with np.load(os.path.join(run_dir(name), "slices", f"slice_{k:03d}.npz")) as s:
        return {m: s[m] for m in s.files if m != "record"}, json.loads(str(s["record"]))


def compare_slices(a, b, ks):
    """(arrays differing, record fields differing) between runs a and b over slices ks."""
    diff_a, diff_r = [], []
    for k in ks:
        (xa, ra), (xb, rb) = slice_data(a, k), slice_data(b, k)
        for m in sorted(set(xa) | set(xb)):
            if not (m in xa and m in xb and xa[m].dtype == xb[m].dtype and xa[m].shape == xb[m].shape
                    and np.array_equal(xa[m], xb[m])):
                diff_a.append((k, m))
        diff_r += [(k, f) for f in sorted(set(ra) | set(rb)) if f not in VOLATILE and ra.get(f) != rb.get(f)]
    return diff_a, diff_r


def config(name):
    with open(os.path.join(run_dir(name), "config.json")) as f:
        return json.load(f)


def check_grid_run(name, ks, steps_scale, schedule, grid=GRID):
    """config 'time_grid', and per slice: record t / lr_v / steps against the grid, the log line's t."""
    import train_latent
    taus, lrs = lm.time_grid_pts(grid), lm.time_grid_lr(grid)
    sched = train_latent.parse_steps_schedule(schedule)
    cfg = config(name)
    bad = [] if cfg.get("time_grid") == grid else [f"config time_grid {cfg.get('time_grid')!r}"]
    with open(os.path.join(run_dir(name), "train_log.txt")) as f:
        txt = f.read()
    logged = {int(m.group(1)): m.group(2) for m in re.finditer(r"^slice +(\d+) t ([0-9.]+):", txt, re.M)}
    for k in ks:
        _, r = slice_data(name, k)
        want = train_latent.slice_steps(k, steps_scale, sched, grid)
        if r["t"] != float(taus[k]):
            bad.append(f"slice {k} t {r['t']!r} != {float(taus[k])!r}")
        if r["lr_v"] != float(lrs[k]):
            bad.append(f"slice {k} lr_v {r['lr_v']!r} != {float(lrs[k])!r}")
        if r["steps"] != want:
            bad.append(f"slice {k} steps {r['steps']} != {want}")
        if logged.get(k) != f"{taus[k]:.4f}":
            bad.append(f"slice {k} logged t {logged.get(k)} != {taus[k]:.4f}")
    grid_line = [ln for ln in txt.splitlines() if ln.startswith(f"time grid {grid}:")]
    if not grid_line:
        bad.append("no 'time grid' line in train_log.txt")
    check(f"{name}: config time_grid {grid!r}; slices {min(ks)}-{max(ks)}: record t, lr_v, steps and the logged t "
          f"follow the grid; the log has the grid line", not bad, "; ".join(bad[:6]))
    return txt, grid_line[:1]


def sample_in_process(mod, run, stem, n, current):
    """sample_latent's default path of module mod on run's first n base trajectories, written to stem: as main() runs
    it (current: load_params, slice_times, make_drift and sample with taus) or as the original did (no taus)."""
    args = SimpleNamespace(run=run, set="base", n=n, chunk=n, sampler="sde", snapshots=False)
    th, W, b, _, prov, problem = mod.load_params(run)
    lp, qp = prov.pop("local", None), prov.pop("quad", None)
    pos = (th, W, b, "sde", problem, prov.get("hidden", "ising"), lp, qp, prov.get("radius"),
           prov.get("neighbourhood", "chebyshev"))
    if current:
        taus = mod.slice_times(prov)
        drift, diffusion = mod.make_drift(*pos, taus)
        return mod.sample(args, drift, diffusion, problem, stem, prov, W, time.time(), taus)
    drift, diffusion = mod.make_drift(*pos)
    return mod.sample(args, drift, diffusion, problem, stem, prov, W, time.time())


# ---------------------------------------------------------------------------------------------------- parts --
def part_u1(names):
    log("\n######## U1: train_latent.py without --time-grid, original vs current (2 slices)")
    two = ("--max-slices", "2", "--steps-scale", "0.01")
    rc_o, _ = train(names["u1o"], *two, orig=True)
    rc_n, _ = train(names["u1n"], *two)
    if rc_o or rc_n:
        return check("U1 both runs complete", False, f"exit codes {rc_o}, {rc_n}")
    da, dr = compare_slices(names["u1n"], names["u1o"], range(2))
    check("U1 current = original, bit for bit: every array of slices 0-1 and every record field but timings, job and "
          "code hashes", not da and not dr, f"arrays {da[:6]}, record fields {dr[:6]}" if da or dr else "")
    co, cn = config(names["u1o"]), config(names["u1n"])
    skip = {"name", "sha256", "started", "slurm_job"}
    extra_k = sorted(set(cn) - set(co))
    diff_k = sorted(k for k in set(co) & set(cn) if k not in skip and co[k] != cn[k])
    check("U1 config.json: the only new key is 'time_grid' = 'uniform'; all others equal (except name, code hashes, "
          "start time, job)", extra_k == ["time_grid"] and cn["time_grid"] == "uniform" and not diff_k
          and not (set(co) - set(cn)), f"new keys {extra_k}, differing {diff_k}")


def part_u2(names):
    log("\n######## U2: 100-slice uniform run; sample_latent default path, original vs current (in-process)")
    uni = lm.forward_time_pts()
    rc, _ = train(names["u2"], "--steps-schedule", "0:0.001")
    if rc or not os.path.exists(os.path.join(run_dir(names["u2"]), "DONE")):
        return check("U2 run complete", False, f"exit code {rc}")
    with np.load(os.path.join(run_dir(names["u2"]), "hidden_history.npz")) as z:
        t_rec = z["t"]
    check("U2 hidden_history.npz 't' = lm.forward_time_pts() bit for bit",
          t_rec.dtype == uni.dtype and t_rec.tobytes() == uni.tobytes())
    import jax
    import sample_latent as cur
    t0 = time.time()
    x_cur = sample_in_process(cur, names["u2"], os.path.join(OUT, "u2_current"), 4, True)
    t_cur = time.time() - t0
    x_orig = sample_in_process(orig_module("sample_latent"), names["u2"], os.path.join(OUT, "u2_original"), 4, False)
    d = float(np.max(np.abs(x_cur - x_orig)))
    RES["u2_sample_max_diff"] = d
    log(f"  4 trajectories: {t_cur:.0f} s (current, with compilation); finite {bool(np.all(np.isfinite(x_cur)))}; "
        f"max|x| {np.max(np.abs(x_cur)):.3f}; jax {jax.__version__}, {jax.devices()}")
    check("U2 sampling default path: current final states = original, bit for bit (max |diff|)",
          d == 0.0 and np.array_equal(x_cur, x_orig), f"{d:.3e}")
    with open(os.path.join(OUT, "u2_current.json")) as f:
        mc = json.load(f)
    with open(os.path.join(OUT, "u2_original.json")) as f:
        mo = json.load(f)
    new_keys = sorted(set(mc) - set(mo))
    same_vals = [k for k in set(mc) & set(mo) if k not in {"chunk_seconds", "sha256_sample_latent"} and mc[k] != mo[k]]
    check("U2 manifest: new keys exactly 'knots_t' (= forward_time_pts()) and 'time_grid' ('uniform'); the shared keys "
          "equal (but chunk_seconds and the module hash)",
          new_keys == ["knots_t", "time_grid"] and mc["time_grid"] == "uniform"
          and np.array_equal(np.array(mc["knots_t"]), uni) and not same_vals and not (set(mo) - set(mc)),
          f"new keys {new_keys}, differing {same_vals}")


def part_p(names):
    import train_latent
    g, uni = lm.time_grid_pts(GRID), lm.forward_time_pts()
    # A, B: interrupted and resumed
    log(f"\n######## A, B: --time-grid {GRID}, 5 slices at --steps-scale 0.01, killed after slice 2, resumed")
    five = ("--time-grid", GRID, "--max-slices", "5", "--steps-scale", "0.01")
    rc_a, _ = train(names["pow"], *five, kill_after=2, tag=names["pow"] + "_A")
    n_a = n_slices(names["pow"])
    check(f"A interrupted after slice 2 (exit code {rc_a}, {n_a} slices written)", rc_a != 0 and n_a == 3)
    rc_b, txt_b = train(names["pow"], *five, tag=names["pow"] + "_B")
    m = re.search(r"resuming at slice (\d+) \(t = ([0-9.]+)\)", txt_b)
    check("B resumed at slice 3 at the grid's t_3 and completed slices 3-4",
          rc_b == 0 and m is not None and m.group(1) == "3" and m.group(2) == f"{g[3]:.4f}" and n_slices(names["pow"]) == 5,
          f"exit code {rc_b}, resume line {m.group(0) if m else None}, grid t_3 {g[3]:.4f}")
    if n_slices(names["pow"]) == 5:
        check_grid_run(names["pow"], range(5), 0.01, "")
        st_u = [train_latent.slice_steps(k, 0.01) for k in range(5)]
        st_g = [slice_data(names["pow"], k)[1]["steps"] for k in range(5)]
        log(f"  steps of slices 0-4: power:1.5 {st_g}, the uniform grid would give {st_u}; t "
            f"{[round(slice_data(names['pow'], k)[1]['t'], 5) for k in range(5)]}, uniform {[round(float(v), 5) for v in uni[:5]]}")
        check("A+B steps differ from the uniform grid's where the paper's linear rule says so (slices 1-4)",
              all(st_g[k] != st_u[k] for k in range(1, 5)) and st_g[0] == st_u[0], f"{st_g} vs {st_u}")

    # C: resume with another grid
    log("\n######## C: A's run again with another --time-grid")
    for spec, ok_expected in (("uniform", False), ("power:2", False), ("power:1.50", True)):
        rc, text = train(names["pow"], "--time-grid", spec, "--max-slices", "5", "--steps-scale", "0.01",
                         tag=f"{names['pow']}_C_{spec.replace(':', '')}")
        refused = rc != 0 and "different config" in text and "time_grid" in text
        if ok_expected:
            check(f"C --time-grid {spec}: accepted (canonical {lm.parse_time_grid(spec)!r} = the run's); nothing to "
                  f"train", rc == 0 and n_slices(names["pow"]) == 5, f"exit code {rc}")
        else:
            line = next((ln for ln in text.splitlines() if "different config" in ln), "")
            check(f"C --time-grid {spec}: refused", refused, line[-160:])

    # D: --init-from across grids
    log("\n######## D: --init-from the power:1.5 run (slices 0-1)")
    init = ("--init-from", names["pow"], "--init-slices", "2", "--max-slices", "5", "--steps-scale", "0.01")
    rc, text = train(names["du"], *init)
    line = next((ln for ln in reversed(text.splitlines()) if ln.startswith("RuntimeError:")), "")   # the message, not the raise line
    check("D --init-from a power:1.5 run on the uniform grid: refused", rc != 0 and "time grid" in line
          and n_slices(names["du"]) == 0, line[-200:])
    rc, text = train(names["dp"], "--time-grid", GRID, *init)
    links = [k for k in range(2) if os.path.islink(os.path.join(run_dir(names["dp"]), "slices", f"slice_{k:03d}.npz"))]
    check("D --init-from on power:1.5: complete, slices 0-1 linked", rc == 0 and links == [0, 1]
          and n_slices(names["dp"]) == 5, f"exit code {rc}, links {links}")
    if rc == 0 and n_slices(names["dp"]) == 5:
        check_grid_run(names["dp"], range(2, 5), 0.01, "")
        da, dr = compare_slices(names["dp"], names["pow"], range(2, 5))
        check("D slices 2-4 = A+B's slices 2-4 bit for bit (arrays; records but timings)", not da and not dr,
              f"arrays {da[:6]}, record fields {dr[:6]}" if da or dr else "")

    # E: a full power:1.5 run
    log("\n######## E: a 100-slice power:1.5 run (101 steps per slice)")
    rc, _ = train(names["full"], "--time-grid", GRID, "--steps-schedule", "0:0.001")
    done = os.path.exists(os.path.join(run_dir(names["full"]), "DONE"))
    check("E complete (exit code 0, DONE)", rc == 0 and done, f"exit code {rc}")
    if done:
        _, gl = check_grid_run(names["full"], range(lm.N_TIME_STEPS), 1.0, "0:0.001")
        log(f"  {gl[0][:400] if gl else ''}")
        with np.load(os.path.join(run_dir(names["full"]), "hidden_history.npz")) as z:
            t_rec, members = z["t"], list(z.files)
        check(f"E hidden_history.npz 't' = the power:1.5 grid bit for bit (members {members})",
              t_rec.dtype == g.dtype and t_rec.tobytes() == g.tobytes())
        spec, taus = lm.run_time_grid(run_dir(names["full"]))
        check("E latent_model.run_time_grid: ('power:1.5', the grid)", spec == GRID and np.array_equal(taus, g))


def part_s(names, n):
    g = lm.time_grid_pts(GRID)
    run = names["full"]
    log(f"\n######## S: sample_latent.py on E = {run} (default path and --fast), {n} trajectories")
    if not os.path.exists(os.path.join(run_dir(run), "DONE")):
        return check("S needs the finished power:1.5 run E (part P)", False, run)
    base = [sys.executable, "-u", os.path.join(HERE, "sample_latent.py"), "--run", run, "--n", str(n), "--chunk", str(n)]
    stem = os.path.join(run_dir(run), "samples", f"sde_base_n{n}")
    rc_d, text_d, w_d = subprocess_log(base, "S_default")
    rc_f, text_f, w_f = subprocess_log(base + ["--fast", "--stem-suffix", "_fast"], "S_fast")
    log(f"  default: exit code {rc_d}, {w_d:.0f} s; fast: exit code {rc_f}, {w_f:.0f} s")
    if rc_d or rc_f:
        return check("S both samplers complete", False, f"exit codes {rc_d}, {rc_f}; see S_default.log / S_fast.log")
    start_d = next((ln for ln in text_d.splitlines() if ln.startswith("=== start")), "")
    start_f = next((ln for ln in text_f.splitlines() if ln.startswith("=== start")), "")
    log(f"  {start_d[-110:]}\n  {start_f[-150:]}")
    with open(stem + ".json") as f:
        md = json.load(f)
    with open(stem + "_fast.json") as f:
        mf = json.load(f)
    with np.load(stem + "_fast_steps.npz") as z:
        s_used = z["s"]
    x_d, x_f = np.load(stem + ".npy"), np.load(stem + "_fast.npy")
    ok = (md.get("time_grid") == GRID and mf.get("time_grid") == GRID
          and np.array_equal(np.array(md["knots_t"]), g) and np.array_equal(np.array(mf["knots_t"]), g)
          and s_used.tobytes() == np.sort(lm.T_FORWARD - g).tobytes()
          and f"time grid {GRID}" in start_d and f"time grid {GRID}" in start_f)
    check("S the start lines name the grid; both manifests record time_grid 'power:1.5' and knots_t = the grid; the "
          "fast sampler integrated over s = sort(5 - grid) (_steps.npz)", ok,
          f"manifests {md.get('time_grid')!r} / {mf.get('time_grid')!r}; knots_t[:3] {md['knots_t'][:3]}, "
          f"[-3:] {md['knots_t'][-3:]}; s[-3:] {s_used[-3:]}")
    dd = np.max(np.abs(x_d - x_f), axis=1)
    RES["fast_vs_default_per_traj"] = dd.tolist()
    check("S final states finite (default and fast)", bool(np.all(np.isfinite(x_d)) and np.all(np.isfinite(x_f))),
          f"max|x| {np.max(np.abs(x_d)):.3f} / {np.max(np.abs(x_f)):.3f}; fast vs default max|diff| per trajectory "
          f"{np.array2string(dd, precision=2)}; fast solver steps per trajectory {mf.get('steps_mean')} (mean), "
          f"rejected share {mf.get('rejected_share')}")
    # control: the original module (no grid) samples E on the uniform knots
    x_o = sample_in_process(orig_module("sample_latent"), run, os.path.join(OUT, "E_original_uniform_knots"), n, False)
    do = np.max(np.abs(x_o - x_d), axis=1)
    RES["control_uniform_knots_per_traj"] = do.tolist()
    log(f"  control: the original sampler (uniform knots) on the power:1.5 run vs the default path: max|diff| per "
        f"trajectory {np.array2string(do, precision=3)} (fast vs default: {np.array2string(dd, precision=2)}; "
        f"max|x| {np.max(np.abs(x_d)):.3f})")


def part_t(u2_run):
    """T: 'power:1.5@40' and the uniform grid with U2's flags for 42 slices each, in this job (XLA:CPU may round
    differently on another node): slices 0-40 of the tail run sit on the uniform knots with the uniform lr and steps,
    so they must equal the uniform run's bit for bit; slice 41 is at the tail's t. Against U2 (another job) only
    reported."""
    tail, k0 = "power:1.5@40", 40
    name, name_u = f"tgc_tail_{JOB}", f"tgc_tailu_{JOB}"
    log(f"\n######## T: --time-grid {tail} and uniform, 42 slices each")
    for nm in (name, name_u):
        if os.path.exists(run_dir(nm)):
            raise SystemExit(f"{run_dir(nm)} exists already")
    rc_u, _ = train(name_u, "--steps-schedule", "0:0.001", "--max-slices", "42")
    rc, _ = train(name, "--time-grid", tail, "--steps-schedule", "0:0.001", "--max-slices", "42")
    if rc or rc_u or n_slices(name) != 42 or n_slices(name_u) != 42:
        return check("T runs complete (42 slices each)", False, f"exit codes {rc}, {rc_u}")
    check_grid_run(name, range(42), 1.0, "0:0.001", grid=tail)
    da, dr = compare_slices(name, name_u, range(k0 + 1))
    check(f"T {tail}: slices 0-{k0} = the uniform run's bit for bit (every array; records but timings)",
          not da and not dr, f"arrays {da[:6]}, record fields {dr[:6]}" if da or dr else "")
    if os.path.exists(os.path.join(run_dir(u2_run), "DONE")):
        da2, dr2 = compare_slices(name, u2_run, range(k0 + 1))
        log(f"  (against U2 = {u2_run}, trained in another job: {len(da2)} arrays and {len(dr2)} record fields differ "
            f"in slices 0-{k0})")
    (xa, ra), (xb, rb) = slice_data(name, k0 + 1), slice_data(name_u, k0 + 1)
    gt = lm.time_grid_pts(tail)
    check(f"T slice {k0 + 1} at the tail's t {gt[k0 + 1]:.5f} (uniform {rb['t']:.5f}), steps {ra['steps']} and lr "
          f"{ra['lr_v']:.4g} of the grid, parameters no longer equal", ra["t"] == float(gt[k0 + 1]) and ra["t"] != rb["t"]
          and not np.array_equal(xa["theta"], xb["theta"]))


def part_i(names):
    """I: --init-from across grids, allowed exactly where the linked slices sit at identical times."""
    import train_latent
    tail = "power:1.5@3"
    gt, uni = lm.time_grid_pts(tail), lm.forward_time_pts()
    log(f"\n######## I: --init-from a uniform run onto {tail} (identical linked knots) and onto {GRID}")
    rc, _ = train(names["iu"], "--max-slices", "5", "--steps-scale", "0.01")
    check("I uniform source run complete (5 slices)", rc == 0 and n_slices(names["iu"]) == 5, f"exit code {rc}")
    if rc != 0:
        return
    common = ("--init-from", names["iu"], "--steps-scale", "0.01")
    rc, text = train(names["it"], "--time-grid", tail, "--init-slices", "4", "--max-slices", "6", *common)
    links = [k for k in range(6) if os.path.islink(os.path.join(run_dir(names["it"]), "slices", f"slice_{k:03d}.npz"))]
    line = next((ln for ln in text.splitlines() if "identical times" in ln), "")
    ok = rc == 0 and links == [0, 1, 2, 3] and n_slices(names["it"]) == 6 and bool(line)
    check(f"I {tail} --init-slices 4 from the uniform run: accepted, slices 0-3 linked, 4-5 trained", ok,
          f"exit code {rc}, links {links}, log '{line[-160:]}'")
    if ok:
        recs = [slice_data(names["it"], k)[1] for k in (4, 5)]
        good = all(r["t"] == float(gt[k]) and r["steps"] == train_latent.slice_steps(k, 0.01, (), tail)
                   for r, k in zip(recs, (4, 5)))
        check(f"I slices 4-5 at the tail's t ({gt[4]:.5f}, {gt[5]:.5f}; uniform {uni[4]:.5f}, {uni[5]:.5f}) with its "
              f"steps", good and float(gt[4]) != float(uni[4]),
              f"records t {[r['t'] for r in recs]}, steps {[r['steps'] for r in recs]}")
        check(f"I config.json time_grid = {tail!r}", config(names["it"]).get("time_grid") == tail)
    rc, text = train(names["ir"], "--time-grid", tail, "--init-slices", "5", "--max-slices", "6", *common)
    line = next((ln for ln in reversed(text.splitlines()) if ln.startswith("RuntimeError:")), "")   # the message, not the raise line
    check(f"I {tail} --init-slices 5 (slice 4 sits at another t): refused", rc != 0 and "other times" in line
          and n_slices(names["ir"]) == 0, line[-200:])
    rc, text = train(names["ip"], "--time-grid", GRID, "--init-slices", "1", "--max-slices", "2", *common)
    check(f"I {GRID} --init-slices 1 (slice 0 sits at t = 5 on every grid): accepted, slice 0 linked",
          rc == 0 and n_slices(names["ip"]) == 2
          and os.path.islink(os.path.join(run_dir(names["ip"]), "slices", "slice_000.npz")), f"exit code {rc}")


def part_x():
    log("\n######## X: recorded slice times of every finished run under out/latent_physical_SGM")
    n_ok, n_bitwise, worst, bad, grids = 0, 0, 0.0, [], {}
    for nm in sorted(os.listdir(lm.OUT_ROOT)):
        d = run_dir(nm)
        h, c = os.path.join(d, "hidden_history.npz"), os.path.join(d, "config.json")
        if not (os.path.isfile(h) and os.path.isfile(c)):
            continue
        try:
            with np.load(h) as z:
                t_rec = z["t"] if "t" in z.files else None
            spec, taus = lm.run_time_grid(d)
            n_ok += 1
            grids[spec] = grids.get(spec, 0) + 1
            if t_rec is not None:
                n_bitwise += int(t_rec.tobytes() == taus.tobytes())
                worst = max(worst, float(np.max(np.abs(t_rec - taus))))
        except Exception as e:                                      # noqa: BLE001  (report every failure)
            bad.append(f"{nm}: {type(e).__name__}: {str(e)[-160:]}")
    RES["existing_runs"] = {"accepted": n_ok, "bitwise": n_bitwise, "max_abs_diff": worst, "grids": grids,
                            "refused": bad}
    check(f"X every finished run is accepted (its recorded t matches its grid): {n_ok} accepted (grids {grids}), "
          f"{n_bitwise} of them bit for bit, max |diff| {worst:.2e}", not bad, "; ".join(bad[:5]))


def run_main(mod, argv, out):
    """mod.main() with sys.argv = argv and the module's output folder OUT set to out (in-process); a module with its
    own OUT_ROOT (plot_learning_curves.py, which derives it from its file's place) reads the runs from lm.OUT_ROOT,
    and a module-level report buffer LINES starts empty."""
    os.makedirs(out, exist_ok=True)
    mod.OUT = out
    if hasattr(mod, "OUT_ROOT"):
        mod.OUT_ROOT = lm.OUT_ROOT
    if hasattr(mod, "LINES"):
        mod.LINES = []
    saved, sys.argv = sys.argv, [mod.__file__] + list(argv)
    try:
        mod.main()
    finally:
        sys.argv = saved


def part_r(power_run, u1, u2):
    log(f"\n######## R: analysis scripts, original vs current (power run {power_run}; uniform runs {u1}, {u2})")
    g = lm.time_grid_pts(GRID)

    def text(path, drop=()):
        with open(path) as f:
            return [ln.replace(os.path.dirname(path), "<OUT>") for ln in f.read().splitlines()
                    if not any(ln.startswith(d) for d in drop)]

    for name, argv, fname, drop in (
            ("summarize_training", [u1, u2, "--ref", u1], "summary.txt", ()),
            ("plot_learning_curves", ["--group", "g", f"A={u1}", f"B={u2}", "--pair", "g:A,B", "--slices", "0,1,50,98,99"],
             "report.txt", ("=== start",))):
        cur, orig = cur_module(name), orig_module(name)
        run_main(cur, argv, os.path.join(OUT, f"R_{name}_current"))
        run_main(orig, argv, os.path.join(OUT, f"R_{name}_original"))
        a = text(os.path.join(OUT, f"R_{name}_current", fname), drop)
        b = text(os.path.join(OUT, f"R_{name}_original", fname), drop)
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        check(f"R {name}.py on two uniform runs: {fname} identical, original vs current ({len(a)} lines)",
              not diff and len(a) == len(b) and len(a) > 3, f"first differences {diff[:2]}, lines {len(a)} / {len(b)}")

    # the power run against a uniform run
    cur = cur_module("summarize_training")
    run_main(cur, [power_run, u1, "--ref", u1], os.path.join(OUT, "R_summarize_cross"))
    s = text(os.path.join(OUT, "R_summarize_cross", "summary.txt"))
    m = [ln for ln in s if ln.strip().startswith(f"{power_run}:") and "slices; mean difference" in ln]
    note = [ln for ln in s if "only the slices at equal t are paired" in ln]
    check("R summarize_training.py, power run vs uniform --ref: a note on the grids, and only 2 slices paired",
          bool(note) and bool(m) and m[0].strip().startswith(f"{power_run}: 2 slices;"), f"{(note + m)[:2]}")
    run_main(cur, [power_run, u1, "--states"], os.path.join(OUT, "R_summarize_states"))
    s = text(os.path.join(OUT, "R_summarize_states", "summary.txt"))
    heads = [ln.strip() for ln in s if "(time grid " in ln and "slices): most lost" in ln]
    near = [ln.strip() for ln in s if ln.strip().startswith("t ") and " eff " in ln and " lost " in ln]
    for ln in heads + near[:4] + near[10:14]:
        log(f"    {ln[:200]}")
    check("R summarize_training.py --states: a state table per run with its grid, at the slices nearest to t = 4.7 "
          "... 0 (10 rows per run, each run's own t), and states.png", len(heads) == 2 and f"time grid {GRID}" in heads[0]
          and "time grid uniform" in heads[1] and len(near) == 20
          and os.path.exists(os.path.join(OUT, "R_summarize_states", "states.png")), f"{heads}")
    cur = cur_module("plot_learning_curves")
    run_main(cur, ["--group", "h", f"U={u1}", f"P={power_run}", "--pair", "h:P,U", "--slices", "0,1,30,99"],
             os.path.join(OUT, "R_curves_cross"))
    r = text(os.path.join(OUT, "R_curves_cross", "report.txt"))
    skipped = [ln for ln in r if "slices not paired, their slice times differ" in ln]
    paired_line = [ln for ln in r if ln.startswith("pair P - U: final validation loss difference")]
    per_run_t = [ln for ln in r if "(t per run:" in ln]
    check("R plot_learning_curves.py, power run vs uniform: 98 slices not paired, 2 paired (t = 5, 0), the table names "
          "both runs' t", bool(skipped) and skipped[0].startswith("pair P - U: 98 slices") and bool(paired_line)
          and "of 2" in paired_line[0] and len(per_run_t) == 2, f"{(skipped + paired_line + per_run_t)[:4]}")
    cur, orig = cur_module("inspect_states"), orig_module("inspect_states")
    run_main(cur, [f"{power_run}:1,99"], os.path.join(OUT, "R_states_current"))
    run_main(orig, [f"{power_run}:1,99"], os.path.join(OUT, "R_states_original"))
    sc = text(os.path.join(OUT, "R_states_current", "summary.txt"))
    so = text(os.path.join(OUT, "R_states_original", "summary.txt"))
    tc = [ln.strip()[:24] for ln in sc if ln.strip().startswith("slice ")]
    to = [ln.strip()[:24] for ln in so if ln.strip().startswith("slice ")]
    check(f"R inspect_states.py on the power run: slice 1 at the grid's t = {g[1]:.3f} (original: uniform t), slice 99 "
          f"at t = 0 in both", len(tc) == 2 and f"(t = {g[1]:.3f})" in tc[0] and "(t = 0.000)" in tc[1]
          and len(to) == 2 and f"(t = {lm.forward_time_pts()[1]:.3f})" in to[0], f"current {tc}, original {to}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="U1,U2,P,S,X,R", help="parts to run, in this order")
    ap.add_argument("--runs-job", default=JOB, help="the job id in the names of the tgc_* runs (default this job)")
    ap.add_argument("--s-n", type=int, default=3, help="S: trajectories per sampler")
    ap.add_argument("--r-uniform", default="cli_files_4223239", help="R: a second uniform gauss2_8 run (read only)")
    args = ap.parse_args()
    parts = [p for p in args.parts.split(",") if p]
    unknown = set(parts) - {"U1", "U2", "P", "S", "T", "I", "X", "R"}
    if unknown:
        raise SystemExit(f"unknown parts {sorted(unknown)}")
    os.makedirs(OUT, exist_ok=True)
    log(f"time_grid_check job {JOB}, {time.strftime('%Y-%m-%d %H:%M:%S')}; parts {parts}; runs of job {args.runs_job}; "
        f"original modules in {ORIG}: " + ", ".join(f"{f} {lm.sha256_file(os.path.join(ORIG, f))[:16]}"
                                                    for f in sorted(os.listdir(ORIG)) if f.endswith(".py")))
    names = {c: f"tgc_{c}_{args.runs_job}" for cs in CASES.values() for c in cs}
    for p in parts:
        for c in CASES.get(p, ()):
            if os.path.exists(run_dir(names[c])):
                raise SystemExit(f"{run_dir(names[c])} exists already; part {p} needs new run folders")
    for p in parts:
        t0 = time.time()
        if p == "U1":
            part_u1(names)
        elif p == "U2":
            part_u2(names)
        elif p == "P":
            part_p(names)
        elif p == "S":
            part_s(names, args.s_n)
        elif p == "T":
            part_t(names["u2"])
        elif p == "I":
            part_i(names)
        elif p == "X":
            part_x()
        elif p == "R":
            part_r(names["full"], names["u2"], args.r_uniform)
        log(f"  (part {p}: {time.time() - t0:.0f} s)")
    finish()


def finish():
    n_fail = sum(not c["ok"] for c in CHECKS)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"results": RES, "checks": CHECKS}, f, indent=1, default=str)
    log(f"\n=== {len(CHECKS)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--orig-train"]:        # subprocess: the original train_latent.main() with these arguments
        sys.argv = [os.path.join(ORIG, "train_latent.py")] + sys.argv[2:]
        orig_module("train_latent").main()
    else:
        main()
