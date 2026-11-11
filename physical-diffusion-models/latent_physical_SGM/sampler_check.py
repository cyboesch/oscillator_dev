#!/usr/bin/env python3
"""Checks and timings of the fast sampler (sample_fast.py) against sample_latent.py; a GPU job.

  0. smoke test that fits a CPU job: pixels-only run, drift at two times, 2 trajectories over 3 knot intervals
  1. environment: jax / diffrax versions, device
  2. drift: FastSampler.drift (float64) against sample_latent.make_drift at random states and times, including t
     exactly at knots (from both neighbouring intervals) and at t = 0 and 5; the parameter interpolation (theta, b,
     local units) against interpolate_parameters; the --precision mixed drift's error for information
  3. trajectories: old sampler (sample_latent: one solve with step_ts = knots) against the fast sampler (float64,
     same Brownian tolerance) on the same initial states and Brownian keys: final states, solver steps per trajectory;
     and the old sampler in this job against the run's saved samples (same trajectories)
  4. speed: seconds per chunk for the fast sampler (configurations RUN:CHUNK:TOL:PRECISION), solver steps, loop
     iterations; cost of one drift evaluation and of one Brownian increment for the chunk
  5. samples for the distribution check: the fast sampler on RUN's base set (n, chunk, tolerance, precision), saved as
     <out>/<label>.npy for fid_score.py / evaluate_multiclass.py
  6. default path unchanged: sample_latent.sample with make_drift (no new flags) against a copy of the original
     sample_latent.py (sampler_check/orig/sample_latent.py) on the same trajectories, bit for bit; and job 4187940

Usage (from latent_physical_SGM/):
  sbatch --time=00:45:00 jobs/run_gpu_neuronic.sh sampler_check.py --sections 1,2,3 --runs p784,i784 ...
Runs: an alias (p784, p3072, i784, k0, paper) or a run name. Writes out/latent_physical_SGM/sampler_check/<job>/
(report.txt, results.json, sample files). Exit code 1 if a correctness check fails.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402  (enables float64)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import diffrax  # noqa: E402

import sample_latent  # noqa: E402
import sample_fast as sf  # noqa: E402
from physical_diffusion_fns.helper_fns import interpolate_parameters  # noqa: E402
from physical_diffusion_fns.network_fns import solve_SDE  # noqa: E402

ALIASES = {
    "p784": "mnist28_d0123_K784_ema_potts_quad_local_quartic_r1_lin_pooled",
    "p3072": "mnist28_d0123456789_K3072_ema_potts_quad_local_quartic_r1_lin_pooled",
    "i784": "mnist28_d0123_K784_ema_ising_quad_local_quartic_r1_lin",
    "k0": "mnist28_d0123456789_K0_ema_quartic_r1_lin",
    "paper": "paper",
}
JOB = os.environ.get("SLURM_JOB_ID", "manual")
OUT = os.path.join(lm.OUT_ROOT, "sampler_check", JOB)
ORIG = os.path.join(lm.OUT_ROOT, "sampler_check", "orig", "sample_latent.py")      # copy taken before this work
RESULTS = {"job": JOB, "checks": {}, "sections": {}}
FAILED = []
REPORT = None


def log(msg=""):
    print(msg, flush=True)
    with open(REPORT, "a") as f:
        f.write(msg + "\n")


def check(name, ok, detail):
    RESULTS["checks"][name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        FAILED.append(name)
    log(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")


def save_results():
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=float)


def run_name(alias):
    return ALIASES.get(alias, alias)


def load(alias):
    """sample_latent.load_params plus the arguments of make_drift / FastSampler."""
    th, W, b, run_dir, prov, problem = sample_latent.load_params(run_name(alias))
    prov = dict(prov)
    local, quad = prov.pop("local", None), prov.pop("quad", None)
    kw = dict(problem=problem, hidden=prov.get("hidden", "ising"), local=local, quad=quad, radius=prov.get("radius"),
              neighbourhood=prov.get("neighbourhood", "chebyshev"))
    return th, W, b, run_dir, kw


def saved_samples(alias, run_dir):
    """(n, array) of the run's saved base-set SDE samples (n = 500 preferred), or (None, None)."""
    if alias == "paper":
        return 500, np.load(lm.UNSPLIT_4187940)
    for n in (500, 2000):
        p = os.path.join(run_dir, "samples", f"sde_base_n{n}.npy")
        if os.path.exists(p):
            return n, np.load(p)
    return None, None


def trajectories(problem, n_set, m, set_name="base"):
    """The first m trajectories (initial states, keys) of the n_set-trajectory set, as sample_latent draws them."""
    grid, _ = lm.problem_geometry(problem)
    x0, keys = lm.trajectory_set(set_name, n_set, problem, grid * grid)
    return x0[:m], keys[:m]


# ------------------------------------------------------------------------------------------------- section 0 --
def smoke():
    """CPU-sized plumbing test: pixels-only run, drift at two times, 2 trajectories over the first 3 intervals."""
    th, W, b, run_dir, kw = load("k0")
    fs = sf.FastSampler(th, W, b, **kw)
    drift_old, _ = sample_latent.make_drift(th, W, b, "sde", kw["problem"], kw["hidden"], kw["local"], kw["quad"],
                                           kw["radius"], kw["neighbourhood"])
    x0, keys = trajectories(kw["problem"], 500, 2)
    for t in (0.0, 1.234):
        j = fs.interval_of(t)
        d_new = np.asarray(fs.drift_batch(np.full(2, t), x0, j))
        d_old = np.stack([np.asarray(drift_old(t, jnp.asarray(x), ())) for x in x0])
        log(f"  t {t}: interval {j}, relative drift difference {np.max(np.abs(d_new - d_old)) / np.max(np.abs(d_old)):.2e}")
    t0 = time.time()
    x, st, acc = fs.solve_chunk(x0, keys, intervals=3)
    log(f"  3 intervals, 2 trajectories: {time.time() - t0:.1f} s, steps {st.tolist()}, accepted {acc.tolist()}, "
        f"max|x| {np.max(np.abs(x)):.3f}, finite {bool(np.all(np.isfinite(x)))}")
    log(f"  settings {fs.settings()}")


# ------------------------------------------------------------------------------------------------- section 1 --
def env():
    log(f"jax {jax.__version__}, diffrax {diffrax.__version__}, devices {jax.devices()}, x64 {jax.config.jax_enable_x64}")
    log(f"default matmul precision {jax.config.jax_default_matmul_precision}")
    order, xp, s = sf.knot_times()
    log(f"knots: xp[0..2] {xp[:3]}, xp[-1] {xp[-1]}; s[0..2] {s[:3]}, s[-1] {s[-1]}; interval j has lo = 98 - j")
    RESULTS["sections"]["env"] = {"jax": jax.__version__, "diffrax": diffrax.__version__, "devices": str(jax.devices())}


# ------------------------------------------------------------------------------------------------- section 2 --
def drift_checks(aliases):
    out = {}
    rng = np.random.default_rng(0)
    order, xp, s = sf.knot_times()
    for alias in aliases:
        log(f"\n--- drift: {alias} ({run_name(alias)})")
        t_load = time.time()
        th, W, b, run_dir, kw = load(alias)
        K = W.shape[1]
        drift_old, _ = sample_latent.make_drift(th, W, b, "sde", kw["problem"], kw["hidden"], kw["local"], kw["quad"],
                                               kw["radius"], kw["neighbourhood"])
        # eager vmap: a jit of drift_old would embed its interpolation table (3.9 GB at K = 3072) as a constant
        old_batch = jax.vmap(drift_old, in_axes=(None, 0, None))
        fs = sf.FastSampler(th, W, b, **kw)
        fs_mixed = sf.FastSampler(th, W, b, precision="mixed", **kw) if K else None
        log(f"K {K}, hidden {kw['hidden']}, quad {kw['quad'] is not None}, local {kw['local'] is not None}, "
            f"visible {fs.visible}; loaded in {time.time() - t_load:.0f} s")
        n_saved, saved = saved_samples(alias, run_dir)
        grid, _ = lm.problem_geometry(kw["problem"])
        n = grid * grid
        # states: saved final samples carried to forward time tau by the forward process (exact marginal draw)
        rows = saved[rng.choice(saved.shape[0], 8, replace=False)] if saved is not None else rng.normal(size=(8, n))
        knots_j = [1, 37, 50, 98]
        ts = [(0.0, [0]), (lm.T_FORWARD, [sf.N_INTERVALS - 1])] + [(float(s[j]), [j - 1, j]) for j in knots_j]
        ts += [(float(t), [None]) for t in rng.uniform(0.0, lm.T_FORWARD, 14)]
        worst, worst_mixed, rows_out = 0.0, 0.0, []
        for t, js in ts:
            tau = lm.T_FORWARD - t
            eps = rng.normal(size=rows.shape)
            X = np.exp(-tau) * rows + np.sqrt(lm.TEMP * (1.0 - np.exp(-2.0 * tau))) * eps
            d_old = np.asarray(old_batch(t, jnp.asarray(X), ()))
            for j in js:
                j = fs.interval_of(t) if j is None else j
                d_new = np.asarray(fs.drift_batch(np.full(X.shape[0], t), X, j))
                err = np.max(np.abs(d_new - d_old)) / np.max(np.abs(d_old))
                l2 = np.max(np.linalg.norm(d_new - d_old, axis=1) / np.linalg.norm(d_old, axis=1))
                worst = max(worst, err)
                line = {"t": t, "interval": j, "rel_maxabs": float(err), "rel_l2": float(l2),
                        "max_abs_drift": float(np.max(np.abs(d_old)))}
                if fs_mixed is not None:
                    d_mix = np.asarray(fs_mixed.drift_batch(np.full(X.shape[0], t), X, j))
                    em = np.max(np.abs(d_mix - d_old)) / np.max(np.abs(d_old))
                    line["mixed_rel_maxabs"] = float(em)
                    line["mixed_rel_l2"] = float(np.max(np.linalg.norm(d_mix - d_old, axis=1)
                                                        / np.linalg.norm(d_old, axis=1)))
                    worst_mixed = max(worst_mixed, em)
                rows_out.append(line)
                log(f"  t {t:.6f} interval {j:2d}: max|d_new - d_old| / max|d_old| {err:.2e}, per-state l2 {l2:.2e}"
                    + (f"; mixed {line['mixed_rel_maxabs']:.2e} (l2 {line['mixed_rel_l2']:.2e})" if fs_mixed else ""))
        # the interpolation convention itself: theta, b and the local units against interpolate_parameters
        parts = [th] + ([b] if K else []) + ([kw["local"][0], kw["local"][1]] if kw["local"] is not None else [])
        interp = interpolate_parameters(jnp.asarray(np.concatenate(parts, axis=1)), jnp.asarray(lm.forward_time_pts()))
        ip_err = 0.0
        for t, js in ts:
            ref = np.asarray(interp(lm.T_FORWARD - t))
            for j in js:
                j = fs.interval_of(t) if j is None else j
                lo = sf.interval_lo(j)
                k_lo, k_hi = order[lo], order[lo + 1]
                l = ((lm.T_FORWARD - t) - xp[lo]) / (xp[lo + 1] - xp[lo])
                mine = np.concatenate([p[k_lo] + l * (p[k_hi] - p[k_lo]) for p in parts])
                ip_err = max(ip_err, float(np.max(np.abs(mine - ref) / (np.abs(ref) + 1e-300))))
        log(f"  interpolation of theta / b / local units vs interpolate_parameters: max relative difference {ip_err:.2e}")
        check(f"2 drift {alias} (float64) equals make_drift", worst < 1e-11, f"max relative error {worst:.2e}")
        out[alias] = {"worst_rel": worst, "worst_mixed_rel": worst_mixed, "interp_rel": ip_err, "points": rows_out,
                      "visible": fs.visible}
        del drift_old, old_batch, fs, fs_mixed
    RESULTS["sections"]["drift"] = out


# ------------------------------------------------------------------------------------------------- section 3 --
def trajectory_checks(aliases, m, tol):
    out = {}
    for alias in aliases:
        log(f"\n--- trajectories: {alias}, {m} trajectories, Brownian tolerance {tol:g}")
        th, W, b, run_dir, kw = load(alias)
        n_saved, saved = saved_samples(alias, run_dir)
        n_set = n_saved if n_saved is not None else 500
        x0, keys = trajectories(kw["problem"], n_set, m)
        drift, diffusion = sample_latent.make_drift(th, W, b, "sde", kw["problem"], kw["hidden"], kw["local"],
                                                    kw["quad"], kw["radius"], kw["neighbourhood"])
        knots = jnp.asarray(np.sort(lm.T_FORWARD - lm.forward_time_pts())[1:-1])
        fn = jax.vmap(lambda y, k: solve_SDE(drift, diffusion, y, k, 0.0, lm.T_FORWARD, 2, lm.DT0, rtol=lm.RTOL_SDE,
                                             atol=lm.ATOL_SDE, brownian_tolerance=tol, step_ts=knots))   # as sample_latent
        t0 = time.time()
        sol = fn(jnp.asarray(x0), jnp.asarray(keys))
        x_old = np.asarray(sol.ys[:, -1, :])
        st_old = np.asarray(sol.stats["num_steps"])
        acc_old = np.asarray(sol.stats["num_accepted_steps"])
        t_old = time.time() - t0
        fs = sf.FastSampler(th, W, b, brownian_tol=tol, **kw)
        t0 = time.time()
        x_new, st_new, acc_new = fs.solve_chunk(x0, keys)
        t_new = time.time() - t0
        d = np.max(np.abs(x_new - x_old), axis=1)
        same_steps = st_new.sum(1) == st_old
        log(f"  old sampler {t_old:.0f} s (incl. compilation), fast {t_new:.0f} s (incl. compilation)")
        log(f"  steps per trajectory old {np.array2string(st_old, max_line_width=150)}")
        log(f"  steps per trajectory new {np.array2string(st_new.sum(1), max_line_width=150)}")
        log(f"  accepted old {acc_old.sum()} new {acc_new.sum()}; same step count for {int(same_steps.sum())}/{m}")
        log(f"  max|x_new - x_old| per trajectory: {np.array2string(d, precision=1, max_line_width=150)}")
        log(f"  median {np.median(d):.2e}, max {np.max(d):.2e}; {int(np.sum(d > 1e-3))}/{m} above 1e-3; "
            f"max |x| {np.max(np.abs(x_old)):.2f}")
        res = {"m": m, "tol": tol, "max_diff": d.tolist(), "steps_old": st_old.tolist(),
               "steps_new": st_new.sum(1).tolist(), "accepted_old": acc_old.tolist(),
               "accepted_new": acc_new.sum(1).tolist(), "seconds_old": t_old, "seconds_new": t_new}
        if saved is not None and tol == lm.BROWNIAN_TOL:
            ds = np.max(np.abs(x_old - saved[:m]), axis=1)
            dn = np.max(np.abs(x_new - saved[:m]), axis=1)
            log(f"  old sampler in this job vs saved samples (n {n_set}): median {np.median(ds):.2e}, max {np.max(ds):.2e}; "
                f"fast vs saved: median {np.median(dn):.2e}, max {np.max(dn):.2e}")
            res["old_vs_saved"] = ds.tolist()
            res["new_vs_saved"] = dn.tolist()
        np.savez(os.path.join(OUT, f"traj_{alias}_tol{tol:g}.npz"), x_old=x_old, x_new=x_new, steps_old=st_old,
                 steps_new=st_new, acc_old=acc_old, acc_new=acc_new)
        check(f"3 trajectories {alias}: fast (float64) vs old, median max|diff|", np.median(d) < 1e-3,
              f"median {np.median(d):.2e}, {int(np.sum(d > 1e-3))}/{m} above 1e-3")
        out[alias] = res
        del drift, diffusion, fn, fs
    RESULTS["sections"].setdefault("trajectories", {}).update(out)


# ------------------------------------------------------------------------------------------------- section 4 --
def micro_costs(fs, x0, keys, reps=40):
    """ms per vmapped drift evaluation (the whole chunk) and per Brownian increment (two tree descents) for the
    chunk, from `reps` calls inside one jitted loop (interval 50)."""
    B = x0.shape[0]
    a = fs.interval_args(50)
    t_lo, t_hi = float(fs.s[50]), float(fs.s[51])
    tt = jnp.asarray(np.linspace(t_lo, t_hi, reps))
    f = jax.vmap(fs.drift, in_axes=(None, 0, None))

    @jax.jit
    def drift_loop(X, a):
        def body(i, acc):
            return acc + f(tt[i], X + 1e-9 * acc[:, None], a).sum(1)
        return jax.lax.fori_loop(0, reps, body, jnp.zeros(B))

    def one_bm(key):
        bm = diffrax.VirtualBrownianTree(0.0, lm.T_FORWARD, fs.brownian_tol, (fs.n,), key, diffrax.SpaceTimeLevyArea)

        def body(i, acc):
            inc = bm.evaluate(tt[i] + 1e-12 * acc, tt[i] + 1e-3, use_levy=True)
            return acc + inc.W.sum() * 1e-20 + inc.H.sum() * 1e-20
        return jax.lax.fori_loop(0, reps, body, jnp.zeros(()))

    bm_loop = jax.jit(jax.vmap(one_bm))
    X = jnp.asarray(x0)
    K = jnp.asarray(keys)
    jax.block_until_ready(drift_loop(X, a))
    jax.block_until_ready(bm_loop(K))
    t0 = time.time()
    jax.block_until_ready(drift_loop(X, a))
    t_drift = (time.time() - t0) / reps * 1e3
    t0 = time.time()
    jax.block_until_ready(bm_loop(K))
    t_bm = (time.time() - t0) / reps * 1e3
    return t_drift, t_bm


def speed(configs, micro=True):
    out = []
    loaded = {}
    for cfg in configs:
        alias, chunk, tol, prec = cfg.split(":")
        chunk, tol = int(chunk), float(tol)
        log(f"\n--- speed: {alias} chunk {chunk} tol {tol:g} precision {prec}")
        if alias not in loaded:
            loaded.clear()
            loaded[alias] = load(alias)
        th, W, b, run_dir, kw = loaded[alias]
        fs = sf.FastSampler(th, W, b, precision=prec, brownian_tol=tol, **kw)
        set_name = "base"
        if kw["problem"] == "mnist28" and chunk > 500:
            raise SystemExit("mnist28 base set has 500 trajectories")
        x0, keys = trajectories(kw["problem"], max(chunk, 500), chunk, set_name)
        t0 = time.time()
        fs.solve_chunk(x0, keys, intervals=1)                 # compilation (and interval 0)
        t_comp = time.time() - t0
        t0 = time.time()
        x, st, acc = fs.solve_chunk(x0, keys)
        t_chunk = time.time() - t0
        iters = int(st.max(0).sum())
        res = {"config": cfg, "seconds_chunk": t_chunk, "seconds_per_sample": t_chunk / chunk,
               "first_call_seconds": t_comp, "steps_mean": float(st.sum(1).mean()), "steps_max": int(st.sum(1).max()),
               "rejected_share": float(1 - acc.sum() / st.sum()), "loop_iterations": iters,
               "ms_per_iteration": t_chunk / iters * 1e3, "n_nonfinite": int(np.sum(~np.isfinite(x).all(1)))}
        if micro:
            res["ms_drift_eval"], res["ms_brownian_inc"] = micro_costs(fs, x0, keys)
        log(f"  {t_chunk:.1f} s per chunk ({t_chunk / chunk * 1e3:.1f} ms per sample); first call (compile + interval 0) "
            f"{t_comp:.1f} s; steps per trajectory mean {res['steps_mean']:.0f} max {res['steps_max']}, rejected "
            f"{res['rejected_share']:.1%}; loop iterations {iters} ({iters / res['steps_mean']:.2f} x mean steps), "
            f"{res['ms_per_iteration']:.2f} ms each"
            + (f"; one drift evaluation {res['ms_drift_eval']:.2f} ms, one Brownian increment {res['ms_brownian_inc']:.2f} ms"
               if micro else ""))
        out.append(res)
        RESULTS["sections"].setdefault("speed", []).append(res)
        save_results()
        del fs
    return out


# ------------------------------------------------------------------------------------------------- section 5 --
def dist_samples(specs):
    for spec in specs:
        alias, n, chunk, tol, prec = spec.split(":")
        n, chunk, tol = int(n), int(chunk), float(tol)
        label = f"{alias}_n{n}_c{chunk}_tol{tol:g}_{prec}"
        log(f"\n--- samples: {label}")
        th, W, b, run_dir, kw = load(alias)
        fs = sf.FastSampler(th, W, b, precision=prec, brownian_tol=tol, **kw)
        args = SimpleNamespace(run=run_name(alias), set="base", n=n, chunk=chunk, sampler="sde")
        prov = {"note": "sampler_check section 5"}
        t0 = time.time()
        x = sf.sample(args, fs, kw["problem"], os.path.join(OUT, label), prov, W, t0)
        log(f"  {n} samples in {time.time() - t0:.0f} s; non-finite {int(np.sum(~np.isfinite(x).all(1)))}; "
            f"saved {os.path.join(OUT, label)}.npy")
        RESULTS["sections"].setdefault("samples", {})[label] = {"seconds": time.time() - t0, "path": label + ".npy"}
        save_results()
        del fs


# ------------------------------------------------------------------------------------------------- section 6 --
def default_unchanged(aliases, m):
    spec = importlib.util.spec_from_file_location("sample_latent_orig", ORIG)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    log(f"original module {ORIG}: sha256 {lm.sha256_file(ORIG)}; current sample_latent.py "
        f"{lm.sha256_file(os.path.join(HERE, 'sample_latent.py'))}")
    for alias in aliases:
        log(f"\n--- default path: {alias}, {m} trajectories")
        res = {}
        for lab, mod in (("current", sample_latent), ("original", orig)):
            th, W, b, run_dir, prov, problem = mod.load_params(run_name(alias))
            local_p, quad_p = prov.pop("local", None), prov.pop("quad", None)
            drift, diffusion = mod.make_drift(th, W, b, "sde", problem, prov.get("hidden", "ising"), local_p, quad_p,
                                              prov.get("radius"), prov.get("neighbourhood", "chebyshev"))
            args = SimpleNamespace(run=run_name(alias), set="base", n=m, chunk=m, sampler="sde", snapshots=False)
            stem = os.path.join(OUT, f"default_{alias}_{lab}")
            res[lab] = mod.sample(args, drift, diffusion, problem, stem, prov, W, time.time())
        d = np.max(np.abs(res["current"] - res["original"]))
        check(f"6 default path {alias}: current sample_latent == original, bit for bit", d == 0.0, f"max|diff| {d:.3e}")
        if alias == "paper":
            ref = np.load(lm.UNSPLIT_4187940)[:m]
            dd = np.max(np.abs(res["current"] - ref), axis=1)
            check(f"6 default path reproduces job 4187940 (median max|diff|)", np.median(dd) < 1e-3,
                  f"per trajectory {np.array2string(dd, precision=2)}")


# ---------------------------------------------------------------------------------------------------- main --
def main():
    global REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2")
    ap.add_argument("--runs", default="p784,p3072,i784,k0,paper", help="section 2 (drift)")
    ap.add_argument("--traj-runs", default="p784", help="section 3")
    ap.add_argument("--ntraj", type=int, default=16)
    ap.add_argument("--traj-tol", type=float, default=lm.BROWNIAN_TOL)
    ap.add_argument("--speed", default="p784:250:1e-12:f64", help="section 4: RUN:CHUNK:TOL:PRECISION,...")
    ap.add_argument("--no-micro", action="store_true")
    ap.add_argument("--samples", default="", help="section 5: RUN:N:CHUNK:TOL:PRECISION,...")
    ap.add_argument("--default-runs", default="paper,p784", help="section 6")
    ap.add_argument("--default-n", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    REPORT = os.path.join(OUT, "report.txt")
    sections = set(args.sections.split(","))
    log(f"=== sampler_check job {JOB} {time.strftime('%Y-%m-%d %H:%M:%S')}; args {vars(args)}")
    RESULTS["args"] = vars(args)
    todo = [("0", smoke, ()), ("1", env, ()), ("2", drift_checks, (args.runs.split(","),)),
            ("3", trajectory_checks, (args.traj_runs.split(","), args.ntraj, args.traj_tol)),
            ("4", speed, ([c for c in args.speed.split(",") if c], not args.no_micro)),
            ("5", dist_samples, ([c for c in args.samples.split(",") if c],)),
            ("6", default_unchanged, (args.default_runs.split(","), args.default_n))]
    for key, fn, fargs in todo:
        if key not in sections:
            continue
        t0 = time.time()
        log(f"\n===== section {key}: {fn.__name__}")
        try:
            fn(*fargs)
        except Exception:
            FAILED.append(f"section {key} raised")
            log(traceback.format_exc())
        log(f"===== section {key} done in {time.time() - t0:.0f} s")
        save_results()
    log(f"\nfailed: {FAILED if FAILED else 'none'}")
    RESULTS["failed"] = FAILED
    save_results()
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
