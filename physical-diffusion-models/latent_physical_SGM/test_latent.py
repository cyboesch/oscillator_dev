#!/usr/bin/env python3
"""Validation of the latent-SGM code against the paper's code (GPU job; run before any long training).

  1. dense visible gradient and Laplacian  vs  the paper's analytic per-sample functions
  2. dense energy  vs  the paper's energy_fn; autodiff of the dense energy  vs  the dense gradient
  3. K = 0 loss and parameter gradient  vs  the paper's minimal-memory score-matching loss and gradient
     (paper initial parameters and three slices of the paper checkpoint)
  4. K = 3 hidden units: loss  vs  a full-autodiff reference (F from the paper's energy_fn and log 2cosh; Laplacian
     from the Hessian trace); gradient along random directions  vs  central differences of the reference loss
  5. 150 training steps of the new scan-based trainer  vs  the paper's training loop (same RNG chain, optax.adam)
  6. speed of the new trainer for K = 0 and K = 1
  7. sampler drift with a hidden unit  vs  x - 2 grad F by autodiff of the reference free energy
  8. sampler on the paper checkpoint (4 trajectories)  vs  the saved states of job 4187940; K = 1 sampler smoke
  ...
  13. the opt-in fast paths: stencil / dense_skip visible energy vs dense; the mixed-precision loss (float32 at
      HIGHEST) vs float64; --step-impl lean against the default runner (on the GPU bitwise for pixels only; equal up
      to rounding with hidden units, and on the CPU) and the default runner's reproducibility over two compilations;
      --steps-schedule; the GPU k-means (--ref-kmeans gpu) vs numpy; the history assembly from memory (--assemble
      memory) vs from the slice files
  14. the time grids (train_latent.py --time-grid): 'uniform' through the new code = the old slice times, learning
      rates and step counts bit for bit; 'power:1' = uniform to 1e-15; 'power:1.5' against hand-computed values (and
      the per-slice table and total steps); invalid specs; the knots the samplers take from a run (config 'time_grid'
      against hidden_history.npz 't'; a mismatch is refused); the fast sampler's interval weights, interpolated
      parameters and drift at the power:1.5 knots against jnp.interp and sample_latent.make_drift
  15. train_latent --split-dead: free-energy-preserving re-seeding of dead Potts states
  16. sample_latent --init-dist (start distributions of the reverse process at t = 5): data_stats and its cache, the
      Gaussian start (moments of 20,000 mapped draws, the rank-r starts), the Langevin equilibration on an analytic
      quadratic energy, the command-line guards, and through the command line on a small synthetic run (CPU
      subprocesses): the default path bit for bit against the module before the option, and each new option end to end
  17. train_latent --harmonic R (harmonic hidden modes): their gradient and Laplacian against autodiff, the loss with
      modes against a full-autodiff reference (K = 0, Potts and Ising, with and without --quad / --local), float64 vs
      --precision mixed; the closed-form reference (exact Gaussian score on synthetic Gaussian data and on mnist28_d0123's
      statistics, a stationary point of the expected loss, always confining) and the diagnostics; the sampler drift with
      modes against x - 2 grad F by autodiff; the default paths (loss, runner, drift, load_params, and through the
      command lines training, assembly and sampling) bit for bit against the modules before the option (HARM_ORIG);
      toy runs through the command line (--harmonic 4, --harmonic-fixed, K = 0, Ising, --time-grid), resume,
      --init-from and the guards; the default sampler on the trained toy runs, --fast refused; early_slices.py
  18. the review fixes of --harmonic: --harmonic-ref within (the within-cluster reference, the exact complement of
      the Potts prototype reference), --harmonic-lr-tmin, the diagnostic harm_n_beyond_vis; through the command line
      on the toy problem, the first --harmonic version (HARM_V1) against this one with the new options at their
      defaults; a toy run at about 30 x block 17's steps in slices 94-99
  19. train_latent --patch SIDE:STRIDE:STATES (per-patch Potts units, 2026-11-13): A the geometry (unfold / fold) and
      patch_terms against autodiff; B the loss and force with patches against a full-autodiff reference; C --precision
      mixed with patches; D the default paths bit for bit against the modules before the option (PATCH_ORIG); E the
      per-patch reference against prototype_reference, the learning-rate scales and the de-risk step (the judge's loss
      changes of job 4231616 to 1e-6, on the plain seed-1 study run); F the runner and the sampler with patches; G the
      command lines (guards, default paths old vs new, patch runs, resume, --init-from, --patch-on-slice, samplers,
      early_slices.py)
Usage (from latent_physical_SGM/):  sbatch jobs/run_gpu_neuronic.sh test_latent.py [13|14|15|16|17|18|19]   ('13' ...
'19': that block alone; all of them also run on the CPU, where block 13's bitwise comparisons test XLA:CPU's compilation
instead of the GPU's; block 16 there: sbatch --time=00:30:00 --cpus-per-task=4 --mem=16G jobs/run_cpu_neuronic.sh
test_latent.py 16; blocks 17, 18 and 19: sbatch --time=00:30:00 --cpus-per-task=8 --mem=32G jobs/run_cpu_neuronic.sh
test_latent.py 17). '19 ABCDEF' / '19 G' run those parts of block 19 (two jobs of about 15 min each).
Blocks 17-19 (about 15 min each on 8 CPUs, with parallel CPU subprocesses) run only when named: the full
sequence (no argument) runs blocks 1-16, as before them.
Writes out/latent_physical_SGM/tests/<job>/report.txt and checks.json; exit code 1 if any check fails.
"""

import contextlib
import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import optax  # noqa: E402

from physical_diffusion_fns.helper_fns import sample_forward_process  # noqa: E402
from physical_diffusion_fns.learning_fns_memory_efficient import setup_score_matching_kbT_loss_minimal_memory  # noqa: E402
from physical_diffusion_fns.network_fns import setup_energy_fn, solve_SDE  # noqa: E402
from physical_diffusion_fns.network_fns_memory_efficient import (  # noqa: E402
    setup_duffing_network_analytical_derivatives_memory_efficient,
)
import sample_latent  # noqa: E402
import train_latent  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "tests", os.environ.get("SLURM_JOB_ID", "manual"))
CHECKS = []


def report(line=""):
    print(line, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(line + "\n")


def check(name, value, tol, cmp="<"):
    ok = bool(value < tol) if cmp == "<" else bool(value == tol)
    CHECKS.append({"name": name, "value": float(value), "tol": float(tol), "ok": ok})
    report(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.3e} (tol {tol:.0e})")
    return ok


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def block_13(key, data, taus):
    """13: the opt-in fast paths (2026-11-11). data: the paper's 0/1 training images."""
    t13 = time.time()
    # stencil and dense_skip vs dense: all three pair groups (sextic), and the trained model's case (lin, quartic)
    for G, rad, kept, sextic in ((14, 2, ("duff", "lin", "om"), True), (28, 1, ("lin",), False)):
        conn_s = lm.connectivity(G, rad)
        n_s, P_s = G * G, conn_s.shape[0]
        k1, k2, key = jr.split(key, 3)
        th_s = jr.normal(k1, (4 * n_s + 3 * P_s,)) * 0.3
        for s in lm.dropped_pair_slices(n_s, P_s, kept) + (() if sextic else (lm.k6_slice(n_s),)):
            th_s = th_s.at[s].set(0.0)
        X_s = jr.normal(k2, (32, n_s)) * 0.8
        dense_s = lm.DenseVisible(conn_s, n_s)
        g_d, l_d = jax.jit(dense_s.grad_lap)(th_s, X_s)
        for impl in ("stencil", "dense_skip"):
            v = lm.make_visible(conn_s, n_s, impl, kept, sextic)
            g_v, l_v = jax.jit(v.grad_lap)(th_s, X_s)
            errs = [rel(g_v, g_d), rel(l_v, l_d)]
            if impl == "stencil":
                errs.append(rel(jax.jit(v.energy)(th_s, X_s), jax.jit(dense_s.energy)(th_s, X_s)))
            check(f"13 {impl} vs dense visible energy ({G}x{G} r{rad}, {','.join(kept)}, "
                  f"{'sextic' if sextic else 'quartic'}): grad, Laplacian" + (", energy" if impl == "stencil" else ""),
                  max(errs), 1e-11)

    # loss and gradient of a small random Potts model (8 states, --quad --local) on the trained model's lattice
    conn_s = lm.connectivity(28, 1)
    P_s = conn_s.shape[0]
    zs = lm.dropped_pair_slices(lm.N, P_s, ("lin",))
    kept_m = np.ones(4 * lm.N + 3 * P_s, bool)
    for s in zs + (lm.k6_slice(lm.N),):
        kept_m[s] = False
    k1, k2, k3, k4, k5, k6, k7, key = jr.split(key, 8)
    th_q = lm.quartic_initial_theta(lm.paper_theta_layout(P_s)[0], lm.N, P_s)
    th_q = jnp.where(kept_m, th_q + 0.2 * jr.normal(k1, th_q.shape), 0.0)
    Kp = 8
    pq = {"theta": th_q, "W": 0.02 * jr.normal(k2, (Kp, lm.N)), "b": 0.02 * jr.normal(k3, (Kp,)),
          "Q": 0.005 * jr.normal(k4, (Kp, lm.N)), "wl": 0.3 * jr.normal(k5, (lm.N,)), "bl": 0.02 * jr.normal(k6, (lm.N,))}
    t_q = float(taus[60])
    Xq = sample_forward_process(t_q, 64, lm.TEMP, lm.SIGMA_FORWARD, data, k7, beta=1)
    vis_d = lm.DenseVisible(conn_s)
    vis_st = lm.make_visible(conn_s, lm.N, "stencil", ("lin",), sextic=False)
    loss_d = lm.make_loss(vis_d, lm.TEMP, Kp, "potts", True)
    loss_m = lm.make_loss(vis_st, lm.TEMP, Kp, "potts", True, compute_dtype=jnp.float32)
    L_d, G_d = jax.jit(jax.value_and_grad(loss_d))(pq, Xq)
    L_s, G_s = jax.jit(jax.value_and_grad(lm.make_loss(vis_st, lm.TEMP, Kp, "potts", True)))(pq, Xq)
    L_m, G_m = jax.jit(jax.value_and_grad(loss_m))(pq, Xq)
    on = lambda g, a: np.asarray(a)[kept_m] if g == "theta" else np.asarray(a)   # noqa: E731  (theta: kept entries)
    check("13 loss + gradient, stencil vs dense (Potts 8 --quad --local, 28x28 r1 lin quartic)",
          max([abs(float(L_s) - float(L_d)) / abs(float(L_d))] + [rel(on(g, G_s[g]), on(g, G_d[g])) for g in G_d]),
          1e-10)
    nrel = {g: float(np.linalg.norm(on(g, G_m[g]) - on(g, G_d[g])) / np.linalg.norm(on(g, G_d[g]))) for g in G_d}
    report(f"   mixed vs float64 gradients (relative norm): {', '.join(f'{g} {e:.1e}' for g, e in nrel.items())}; "
           f"dtypes {sorted({str(np.asarray(v).dtype) for v in G_m.values()})}")
    check(f"13 mixed-precision loss vs float64 (relative; {float(L_d):.6g})", abs(float(L_m) - float(L_d)) / abs(float(L_d)),
          1e-5)
    check("13 mixed-precision gradient vs float64, worst group (relative norm)", max(nrel.values()), 1e-4)
    dots = [ln for ln in jax.jit(jax.value_and_grad(loss_m)).lower(pq, Xq).as_text().splitlines() if "dot_general" in ln]
    n_bad = sum(not ("HIGHEST" in ln and "xf32>" in ln and "xf64>" not in ln) for ln in dots)
    check(f"13 mixed precision: every product is float32 at HIGHEST ({len(dots)} dot_general)", n_bad + (not dots), 0.5)

    # --step-impl lean (best_params skipped under --select ema, theta masks, batches per chunk) against the default
    # runner: on the GPU bitwise for a pixels-only model; with hidden units (GEMMs), and on the CPU also without, XLA
    # compiles the changed step with other fusions, so the results differ at rounding level, amplified by Adam over
    # the steps (speed_check.py section 6, GPU: float64 about 1e-12, mixed precision about 1e-4 in the parameters
    # after 301 steps; a bug would be order 1)
    lean = {"track_best": False, "theta_masks": True, "chunk_batches": True}

    def run_variant(loss_, p0, visible="quartic", zero=zs, **kw):
        run_, adam_ = train_latent.build_runner(loss_, data, lm.N, 1000.0, visible, P_s, zero_slices=zero, **kw)
        c = train_latent.init_carry(p0, adam_, jr.PRNGKey(4))
        lr_ = {g: (1e-3 if g == "theta" else 3e-5) for g in p0}
        ls = []
        for nst in (100, 1):                                 # both compiled chunk lengths
            c, l = run_(c, t_q, lr_, nst)
            ls.append(np.asarray(l))
        return c, np.concatenate(ls)

    def leaves(c, groups=(0, 1, 2, 3, 5, 6, 7)):              # every carry group but best_params (4)
        return [np.asarray(x) for i in groups for x in jax.tree_util.tree_leaves(c[i])]

    def rel_groups(ca, cb, groups):                           # |a - b| / |b| over the leaves of the groups
        a, b = leaves(ca, groups), leaves(cb, groups)
        return float(np.sqrt(sum(np.sum((x - y) ** 2) for x, y in zip(a, b)) / sum(np.sum(y ** 2) for y in b)))

    p_k0 = {"theta": th_q, "W": jnp.zeros((0, lm.N)), "b": jnp.zeros((0,))}
    loss_k0 = lm.make_loss(vis_st, lm.TEMP, 0, compute_dtype=jnp.float32)
    (c0, l0), (c1, l1) = run_variant(loss_k0, p_k0), run_variant(loss_k0, p_k0, **lean)
    same = np.array_equal(l0, l1) and all(np.array_equal(x, y) for x, y in zip(leaves(c0), leaves(c1)))
    if jax.default_backend() == "gpu":        # bitwise on the L40 (speed_check.py section 6, jobs 4222690, 4222847)
        check("13 lean vs default runner, pixels only (stencil mixed, 101 steps): bitwise equal (losses, parameters, "
              "Adam state, key, best loss and step, EMA; differing)", 0.0 if same else 1.0, 0.5)
    else:                                     # XLA:CPU fuses the changed step otherwise (job 4223239): rounding level
        d0 = max([rel_groups(c1, c0, (g,)) for g in (0, 1, 7)] + [float(np.max(np.abs(l1 - l0) / np.abs(l0)))])
        check(f"13 lean vs default runner, pixels only (stencil mixed, 101 steps; on the {jax.default_backend()} equal "
              f"up to rounding, bitwise {same}): largest relative difference of parameters, Adam state, EMA, losses",
              d0, 1e-8)
    c0, l0 = run_variant(loss_m, pq)
    c2, l2 = run_variant(loss_m, pq)                          # a second build and compilation of the default runner
    same = np.array_equal(l0, l2) and all(np.array_equal(x, y) for x, y in zip(leaves(c0, range(8)), leaves(c2, range(8))))
    check("13 default runner reproducible over two compilations (Potts 8 --quad --local, stencil mixed; differing)",
          0.0 if same else 1.0, 0.5)
    # lean vs default, also on paths the sweep does not use by default (added 2026-11-11 after review): Ising units,
    # --select best (track_best kept: best_params compared too) and the paper's sextic visible energy (the k_6 floor
    # as a mask; the k_6 start around K6_FLOOR, so about half are floored at once; no quartic floors may act).
    # Tolerances: mixed 1e-5 (float32 rounding 6e-8; observed 0 on the GPU, 3.9e-12 on the CPU), float64 1e-8.
    k8, k9, key = jr.split(key, 3)                            # vis_d is the full sextic dense energy already
    th_6 = lm.paper_theta_layout(P_s)[0] + 0.3 * jr.normal(k8, th_q.shape)
    th_6 = th_6.at[lm.k6_slice(lm.N)].set(lm.K6_FLOOR + 0.01 * jr.normal(k9, (lm.N,)))
    cases = (("stencil mixed, Potts 8", loss_m, pq, 1e-5, {}, {}),
             ("dense f64, Potts 8", loss_d, pq, 1e-8, {}, {}),
             ("dense f64, Ising 8", lm.make_loss(vis_d, lm.TEMP, Kp, "ising", True), pq, 1e-8, {}, {}),
             ("dense f64, Potts 8, --select best", loss_d, pq, 1e-8, {}, {"track_best": True}),
             ("dense f64, Potts 8, sextic visible", loss_d, dict(pq, theta=th_6), 1e-8,
              {"visible": "sextic", "zero": ()}, {}))
    for lab, loss_, p0, tol, base, opt in cases:
        cd_, ld_ = (c0, l0) if lab.startswith("stencil mixed") else run_variant(loss_, p0, **base)
        cl_, ll_ = run_variant(loss_, p0, **base, **dict(lean, **opt))
        names = ("parameters", "Adam state", "EMA") + (("best_params",) if opt.get("track_best") else ())
        ds = [rel_groups(cl_, cd_, (g,)) for g in (0, 1, 7, 4)[:len(names)]]
        dl = float(np.max(np.abs(ll_ - ld_) / np.abs(ld_)))
        same_kb = all(np.array_equal(x, y) for x, y in zip(leaves(cl_, (2, 3)), leaves(cd_, (2, 3))))
        report(f"   lean vs default ({lab}, 101 steps): relative differences "
               f"{', '.join(f'{n} {d:.1e}' for n, d in zip(names, ds))}, losses max {dl:.1e}; key and step equal: "
               f"{same_kb}")
        check(f"13 lean vs default runner ({lab}; --quad --local, 101 steps): equal up to rounding (largest relative "
              f"difference of {', '.join(names)}, losses; key and step equal)", max(ds + [dl]) + (not same_kb), tol)

    # --steps-schedule: no schedule = the old step counts; a schedule gives counts 100 m + 1 near the target
    old = [max(1, int(round(lm.paper_steps(k) * sc))) for sc in (1.0, 0.1, 2.0) for k in range(lm.N_TIME_STEPS)]
    new = [train_latent.slice_steps(k, sc, ()) for sc in (1.0, 0.1, 2.0) for k in range(lm.N_TIME_STEPS)]
    check("13 steps schedule: none = the old step counts (scales 1, 0.1, 2; slices differing)",
          sum(a != b for a, b in zip(old, new)), 0.5)
    sch = train_latent.parse_steps_schedule("30:0.5,70:1,80:2")
    cnt = [train_latent.slice_steps(k, 1.0, sch) for k in range(lm.N_TIME_STEPS)]
    target = [lm.paper_steps(k) * (0.5 if 30 <= k < 70 else (2.0 if k >= 80 else 1.0)) for k in range(lm.N_TIME_STEPS)]
    bad = sum(c % 100 != 1 or abs(c - x) > 50 for c, x in zip(cnt, target))
    bad += sum(cnt[k] != lm.paper_steps(k) for k in list(range(30)) + list(range(70, 80)))
    bad += train_latent.parse_steps_schedule(train_latent.steps_schedule_str(sch)) != sch
    for spec in ("30:2,10:1", "5:0", "5:-1", "x", "100:1", "3:1:2"):
        try:
            train_latent.parse_steps_schedule(spec)
            bad += 1
        except ValueError:
            pass
    report(f"   steps schedule '30:0.5,70:1,80:2': slices 0/30/69/70/80/99 -> {[cnt[k] for k in (0, 30, 69, 70, 80, 99)]} "
           f"(paper {[lm.paper_steps(k) for k in (0, 30, 69, 70, 80, 99)]}); total {sum(cnt)} vs {sum(map(lm.paper_steps, range(100)))}")
    check("13 steps schedule '30:0.5,70:1,80:2': counts 100 m + 1 within 50 of the target, factor-1 slices unchanged, "
          "canonical form round trip, invalid specs rejected (failures)", bad, 0.5)

    # GPU k-means vs numpy on 3000 images, 24 centres
    Xk = np.asarray(data, dtype=np.float64)[:3000]
    Cc, ac = lm.kmeans(Xk, 24, 5)
    Cg, ag = lm.kmeans_jax(Xk, 24, 5)
    Cg2, ag2 = lm.kmeans_jax(Xk, 24, 5)
    rng = np.random.default_rng(5)
    idx = [int(rng.integers(Xk.shape[0]))]
    d2 = np.sum((Xk - Xk[idx[0]]) ** 2, axis=1)
    for _ in range(1, 24):
        idx.append(int(rng.choice(Xk.shape[0], p=d2 / d2.sum())))
        d2 = np.minimum(d2, np.sum((Xk - Xk[idx[-1]]) ** 2, axis=1))
    rng = np.random.default_rng(5)
    i0 = int(rng.integers(Xk.shape[0]))
    idx_g = np.asarray(lm._kmeanspp_device(jnp.asarray(Xk), jnp.asarray(i0), jnp.asarray(rng.random(23))))
    check("13 GPU k-means: seeding indices equal to numpy's (24 centres; differing)", int(np.sum(idx_g != np.array(idx))),
          0.5)
    obj_c, obj_g = float(np.sum((Xk - Cc[ac]) ** 2)), float(np.sum((Xk - Cg[ag]) ** 2))
    check(f"13 GPU k-means: objective vs numpy (relative; {obj_c:.8g}), assignments equal "
          f"{float(np.mean(ac == ag)):.4f}", abs(obj_g - obj_c) / obj_c, 1e-9)
    check("13 GPU k-means deterministic (two runs, differing entries)",
          int(np.sum(Cg != Cg2) + np.sum(ag != ag2)), 0.5)

    # history assembly: write_history from memory (all / half of the slices) vs the files, on synthetic slices
    hd = os.path.join(OUT, "history_check")
    sd = os.path.join(hd, "slices")
    os.makedirs(sd, exist_ok=True)
    rng = np.random.default_rng(13)
    held = {}
    for k in range(lm.N_TIME_STEPS):
        arrs = {"theta": rng.normal(size=5), "W": rng.normal(size=(3, 4)), "b": rng.normal(size=3),
                "Q": rng.normal(size=(3, 4)), "wl": rng.normal(size=4), "bl": rng.normal(size=4)}
        train_latent.save_atomic(os.path.join(sd, f"slice_{k:03d}.npz"), key=np.zeros(2, np.uint32), **arrs)
        held[k] = arrs
    outs = {}
    for lab, hk in (("files", {}), ("memory", held), ("half", {k: held[k] for k in range(50, 100)})):
        od = os.path.join(hd, lab)
        os.makedirs(od, exist_ok=True)
        train_latent.write_history(od, sd, lm.N_TIME_STEPS, taus, True, True, hk)
        with open(os.path.join(od, "params_history.npy"), "rb") as f:
            ph = f.read()
        z = np.load(os.path.join(od, "hidden_history.npz"))
        outs[lab] = (ph, list(z.files), {m: z[m] for m in z.files})
    ref_th = np.stack([held[k]["theta"] for k in range(lm.N_TIME_STEPS)])
    bad = int(not np.array_equal(np.load(os.path.join(hd, "files", "params_history.npy")), ref_th))
    for lab in ("memory", "half"):
        ph, files, arrs = outs[lab]
        bad += int(ph != outs["files"][0]) + int(files != outs["files"][1])
        bad += sum(not (arrs[m].dtype == outs["files"][2][m].dtype and np.array_equal(arrs[m], outs["files"][2][m]))
                   for m in files)
    report(f"   history check: hidden_history.npz members {outs['files'][1]}")
    check("13 history from memory (all / half of the slices) = from the slice files: params_history.npy bytes, "
          "hidden_history.npz members, order, dtypes and values (failures)", bad, 0.5)
    report(f"   block 13: {time.time() - t13:.0f} s")


def block_14():
    """14: the time grids of train_latent.py --time-grid (2026-11-11); CPU-sized. 'uniform' through the new code is the
    old grid and schedules bit for bit; 'power:1' is the uniform grid up to rounding; 'power:1.5' against
    hand-computed values; invalid specs; the knots the samplers take from a run (config.json 'time_grid' against
    hidden_history.npz 't', through sample_latent.load_params); the fast sampler's interval widths and weights,
    interpolated parameters and drift at the power:1.5 knots against jnp.interp (interpolate_parameters) and
    sample_latent.make_drift, on a small synthetic history (8x8 lattice, Potts 4 --quad --local)."""
    import math
    import sample_fast as sf
    from physical_diffusion_fns.helper_fns import interpolate_parameters
    t14 = time.time()
    n_sl = lm.N_TIME_STEPS
    uni = lm.forward_time_pts()

    def same(a, b):                                           # bit for bit: dtype, shape and bytes
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

    # 'uniform' through the new code = the old grid, learning rates, patience and step counts, bit for bit
    bad = int(not same(lm.time_grid_pts("uniform"), uni)) + int(not same(lm.time_grid_lr("uniform"), lm.lr_schedule()))
    bad += int(not same(lm.time_grid_patience("uniform"), lm.patience_schedule()))
    bad += sum(lm.time_grid_steps(k, "uniform") != lm.paper_steps(k) for k in range(n_sl))
    check("14 'uniform' = lm.forward_time_pts(), lr_schedule(), patience_schedule() and paper_steps(k), bit for bit "
          "(differing items)", bad, 0.5)

    def old_slice_steps(k, sc, sch):                          # train_latent.slice_steps before --time-grid
        if not sch:
            return max(1, int(round(lm.paper_steps(k) * sc)))
        x = lm.paper_steps(k) * sc * train_latent.steps_factor(sch, k)
        return 100 * max(1, int(round((x - 1.0) / 100.0))) + 1

    bad = 0
    for sch in ((), train_latent.parse_steps_schedule("30:0.5,70:1,80:2"),
                train_latent.parse_steps_schedule("0:0.01,50:0.02")):
        for sc in (1.0, 0.1, 2.0, 0.37):
            for k in range(n_sl):
                ref = old_slice_steps(k, sc, sch)
                bad += int(train_latent.slice_steps(k, sc, sch) != ref)
                bad += int(train_latent.slice_steps(k, sc, sch, "uniform") != ref)
    check("14 train_latent.slice_steps on 'uniform' (grid omitted and explicit) = the old rule (3 schedules x 4 "
          "scales x 100 slices; differing)", bad, 0.5)

    # 'power:1': the uniform grid up to rounding, with the same step counts
    p1 = lm.time_grid_pts("power:1")
    d_t = float(np.max(np.abs(p1 - uni)))
    d_lr = float(np.max(np.abs(lm.time_grid_lr("power:1") - lm.lr_schedule()) / lm.lr_schedule()))
    n_st = sum(lm.time_grid_steps(k, "power:1") != lm.paper_steps(k) for k in range(n_sl))
    report(f"   power:1 vs uniform: slice times max |diff| {d_t:.2e} ({int(np.sum(p1 != uni))} of {n_sl} not bitwise "
           f"equal); learning rates max relative diff {d_lr:.2e}; step counts differing {n_st}")
    check("14 'power:1' slice times = the uniform grid to 1e-15 (max |diff|)", d_t, 1e-15)
    check("14 'power:1' learning rates = the uniform grid's (max relative diff)", d_lr, 1e-14)
    check("14 'power:1' step counts = paper_steps(k) (slices differing)", n_st, 0.5)

    # 'power:1.5' against hand-computed values (t_k = 5 ((99 - k)/99)^1.5; lr and patience linear in t)
    g15 = lm.time_grid_pts("power:1.5")
    lr15, pat15 = lm.time_grid_lr("power:1.5"), lm.time_grid_patience("power:1.5")
    st15 = [lm.time_grid_steps(k, "power:1.5") for k in range(n_sl)]
    st_u = [lm.paper_steps(k) for k in range(n_sl)]
    lr_u = lm.lr_schedule()
    ks = (0, 1, 7, 15, 30, 50, 70, 90, 98, 99)
    steps_hand = {0: 10901, 1: 11301, 7: 13001, 15: 15301, 30: 19301, 50: 24001, 70: 27801, 90: 30401, 98: 30901,
                  99: 30901}                                  # worked out by hand from the formulas above
    bad = int(not (g15[0] == lm.T_FORWARD and g15[-1] == 0.0)) + int(not np.all(np.diff(g15) < 0.0))
    bad += int(not (lr15[0] == lm.LR_AT_T5 and lr15[-1] == lm.LR_AT_T0 and pat15[0] == lm.PATIENCE_AT_T5
                    and pat15[-1] == lm.PATIENCE_AT_T0))
    worst_t, worst_lr = 0.0, 0.0
    for k in ks:
        t_h = 5.0 * ((99 - k) / 99.0) ** 1.5
        lr_h = 2e-5 + (2e-3 - 2e-5) * t_h / 5.0
        st_h = 1001 + 100 * (math.ceil(300.0 - 40.0 * t_h) - 1)
        worst_t, worst_lr = max(worst_t, abs(g15[k] - t_h)), max(worst_lr, abs(lr15[k] - lr_h) / lr_h)
        bad += int(st15[k] != st_h) + int(st15[k] != steps_hand[k])
        bad += int(train_latent.slice_steps(k, 1.0, (), "power:1.5") != st15[k])
    last, first = g15[-2] - g15[-1], g15[0] - g15[1]
    rel_last = abs(last - 5.0 / 99.0 ** 1.5) / (5.0 / 99.0 ** 1.5)
    frac = pat15[1:-1] - np.floor(pat15[1:-1])
    near_int = float(np.min(np.minimum(frac, 1.0 - frac)))
    tot15, tot_u = sum(st15), sum(st_u)
    report(f"   power:1.5: first interval {first:.6f}, last {last:.6g} (5/99^1.5 = {5.0 / 99.0 ** 1.5:.6g}, relative "
           f"diff {rel_last:.1e}; uniform {uni[0] - uni[1]:.6f}); slice times vs hand max |diff| {worst_t:.1e}, "
           f"lr max relative diff {worst_lr:.1e}; interior patience closest to an integer: {near_int:.4f} away")
    report(f"   power:1.5 vs uniform per slice (t, lr, paper steps):")
    report(f"   {'slice':>5} {'t uniform':>10} {'t power1.5':>11} {'lr uniform':>11} {'lr power1.5':>12} "
           f"{'steps unif':>11} {'steps p1.5':>11}")
    for k in ks:
        report(f"   {k:5d} {uni[k]:10.5f} {g15[k]:11.5f} {lr_u[k]:11.4e} {lr15[k]:12.4e} {st_u[k]:11,d} {st15[k]:11,d}")
    report(f"   total paper steps: uniform {tot_u:,}, power:1.5 {tot15:,} ({100.0 * (tot15 / tot_u - 1.0):+.2f}%); slices "
           f"at t >= 3: uniform {int(np.sum(uni >= 3.0))}, power:1.5 {int(np.sum(g15 >= 3.0))}; slice 30 at t = "
           f"{uni[30]:.3f} (uniform) / {g15[30]:.3f} (power:1.5)")
    # where the slices and steps go: the window in which the four-digit Potts runs decide which states survive (the
    # uniform grid's slices 7-30, t = 4.646 ... 3.485), and the late region
    for lo_t, hi_t, lab in ((uni[30], uni[7], "state-survival window t 3.485-4.646 (uniform slices 7-30)"),
                            (3.0, 5.0, "t >= 3"), (0.0, 1.0, "t <= 1"), (0.0, 0.1, "t <= 0.1")):
        cells = []
        for nm, tt, st in (("uniform", uni, st_u), ("power:1.5", g15, st15)):
            ks_w = [k for k in range(n_sl) if lo_t - 1e-12 <= tt[k] <= hi_t + 1e-12]
            cells.append(f"{nm} {len(ks_w)} slices ({ks_w[0]}-{ks_w[-1]}), {sum(st[k] for k in ks_w):,} steps")
        report(f"   {lab}: " + "; ".join(cells))
    check("14 'power:1.5': t_0 = 5 and t_99 = 0 exactly, strictly decreasing, lr / patience exact at both ends, steps "
          "and slice_steps = hand-computed values at slices 0,1,7,15,30,50,70,90,98,99 (failures)", bad, 0.5)
    check("14 'power:1.5' slice times and learning rates = 5 ((99-k)/99)^1.5 and the linear lr (worst of |dt|, "
          "relative |dlr|)", max(worst_t, worst_lr), 1e-13)
    check("14 'power:1.5' last interval = 5 / 99^1.5 (relative diff)", rel_last, 1e-13)
    check(f"14 total paper steps: uniform = 2,095,000 (|diff|; power:1.5 {tot15:,})", abs(tot_u - 2095000), 0.5)
    bad = 0
    for p in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        sp = f"power:{p}"
        tp, lp = lm.time_grid_pts(sp), lm.time_grid_lr(sp)
        bad += int(not (tp[0] == 5.0 and tp[-1] == 0.0 and np.all(np.diff(tp) < 0.0)))
        bad += int(not (lp[0] == lm.LR_AT_T5 and lp[-1] == lm.LR_AT_T0))
        bad += int(lm.time_grid_steps(0, sp) != 10901) + int(lm.time_grid_steps(n_sl - 1, sp) != 30901)
    check("14 power:P for P = 0.5-4: exact endpoints, strictly decreasing, lr 2e-3 / 2e-5 and 10,901 / 30,901 steps "
          "at t = 5 / 0 (failures)", bad, 0.5)

    # 'power:P@K': the uniform grid's knots, learning rates and step counts for slices 0..K bit for bit, the power law
    # t_k = t_K ((99 - k) / (99 - K))^P below t_K
    tail, K0 = "power:1.5@40", 40
    gt, lr_t, pat_t = lm.time_grid_pts(tail), lm.time_grid_lr(tail), lm.time_grid_patience(tail)
    st_t = [lm.time_grid_steps(k, tail) for k in range(n_sl)]
    bad = int(not same(gt[:K0 + 1], uni[:K0 + 1])) + int(not same(lr_t[:K0 + 1], lr_u[:K0 + 1]))
    bad += int(not same(pat_t[:K0 + 1], lm.patience_schedule()[:K0 + 1]))
    bad += sum(st_t[k] != st_u[k] for k in range(K0 + 1))
    bad += int(not (gt[0] == lm.T_FORWARD and gt[-1] == 0.0 and np.all(np.diff(gt) < 0.0)))
    bad += int(not (lr_t[-1] == lm.LR_AT_T0 and pat_t[-1] == lm.PATIENCE_AT_T0))
    tail_hand = {41: 19301, 70: 26801, 98: 30901, 99: 30901}   # worked out by hand from t_40 = 5 * 59/99
    worst_tail = 0.0
    for k in (41, 55, 70, 90, 98, 99):
        t_h = float(uni[K0]) * ((99 - k) / (99.0 - K0)) ** 1.5
        lr_h = 2e-5 + (2e-3 - 2e-5) * t_h / 5.0
        worst_tail = max(worst_tail, abs(gt[k] - t_h), abs(lr_t[k] - lr_h) / lr_h)
        bad += int(st_t[k] != 1001 + 100 * (math.ceil(300.0 - 40.0 * t_h) - 1))
        bad += int(k in tail_hand and st_t[k] != tail_hand[k])
        bad += int(train_latent.slice_steps(k, 1.0, (), tail) != st_t[k])
    rel_last_t = abs((gt[-2] - gt[-1]) - float(uni[K0]) / 59.0 ** 1.5) / (float(uni[K0]) / 59.0 ** 1.5)
    tot_t = sum(st_t)
    report(f"   {tail}: slices 0-{K0} = the uniform grid (t >= {uni[K0]:.4f}); first interval below the junction "
           f"{gt[K0] - gt[K0 + 1]:.4f}, last {gt[-2] - gt[-1]:.6f} (= t_40 / 59^1.5, relative diff {rel_last_t:.1e}); "
           f"steps at slices 41/55/70/90/98: {[st_t[k] for k in (41, 55, 70, 90, 98)]} (uniform "
           f"{[st_u[k] for k in (41, 55, 70, 90, 98)]}); total {tot_t:,} ({100.0 * (tot_t / tot_u - 1.0):+.2f}%)")
    check(f"14 '{tail}': slices 0-{K0} bit for bit the uniform grid's (t, lr, patience, steps); exact ends, strictly "
          f"decreasing; steps below the junction = hand-computed values (failures)", bad, 0.5)
    check(f"14 '{tail}' slice times, learning rates and last interval below the junction = the formulas (worst of "
          f"|dt|, relative |dlr|, relative last-interval diff)", max(worst_tail, rel_last_t), 1e-13)

    # specs: canonical forms and rejections
    canon = {"uniform": "uniform", " UNIFORM ": "uniform", "power:1.5": "power:1.5", "Power:1.50": "power:1.5",
             " power: 1.5 ": "power:1.5", "power:1": "power:1.0", "power:2e0": "power:2.0",
             "Power:1.50@040": "power:1.5@40", "power:2@97": "power:2.0@97", "power:1.5@+1": "power:1.5@1",
             " power :1.5": "power:1.5", "POWER : 2 @ 97": "power:2.0@97", "power:4.8": "power:4.8"}
    bad = sum(lm.parse_time_grid(s) != c or lm.parse_time_grid(c) != c for s, c in canon.items())
    rejected = 0
    invalid = ("", "power", "power:", "power:0", "power:-1", "power:nan", "power:inf", "power:abc", "power:1.5:2",
               "uniform:1", "linear", "power;1.5", "power:1e-30", "power:5000", None, 1.5,
               "power:1.5@0", "power:1.5@98", "power:1.5@-1", "power:1.5@x", "power:1.5@40.0", "power:1.5@",
               "power:@40", "power:1.5@40@1", "power:0@40", "uniform@40",
               # added after review (2026-11-11): Python float() / int() forms that are typos here, and knots that
               # would lie closer than MIN_KNOT_GAP (t_98 = 5/99^4.9 = 8e-10; P = 9: s_98 = 5 - t_98 rounds to 5)
               "power:1_5", "power:１.5", "power:1.5@4_0", "power:4.9", "power:9")
    for s in invalid:
        try:
            lm.parse_time_grid(s)
        except ValueError:
            rejected += 1
    report(f"   specs: canonical forms {canon}; rejected {rejected} of {len(invalid)} invalid specs")
    check("14 time-grid specs: canonical forms and round trip, every invalid spec rejected with ValueError (failures)",
          bad + len(invalid) - rejected, 0.5)

    # the knots a sampler takes from a run: config.json 'time_grid' (missing = 'uniform') against hidden_history 't'
    rng = np.random.default_rng(14)
    G8, R8, K8 = 8, 1, 4
    n8, P8 = G8 * G8, lm.connectivity(G8, R8).shape[0]
    hist = {"theta": np.asarray(lm.paper_theta_layout(P8, n8)[0])[None] + 0.05 * rng.normal(size=(n_sl, 4 * n8 + 3 * P8)),
            "W": 0.05 * rng.normal(size=(n_sl, K8, n8)), "b": 0.05 * rng.normal(size=(n_sl, K8)),
            "Q": 0.01 * rng.normal(size=(n_sl, K8, n8)), "wl": 0.3 * rng.normal(size=(n_sl, n8)),
            "bl": 0.05 * rng.normal(size=(n_sl, n8))}
    root = os.path.join(OUT, "grid_runs")

    def make_run(name, cfg_grid, t_rec):
        """A finished run folder as train_latent.py leaves it (write_history with the given recorded t), relative to
        OUT_ROOT, for sample_latent.load_params."""
        d = os.path.join(root, name)
        os.makedirs(os.path.join(d, "slices"), exist_ok=True)
        held = {k: {g: hist[g][k] for g in ("theta", "W", "b")} for k in range(n_sl)}
        train_latent.write_history(d, os.path.join(d, "slices"), n_sl, t_rec, False, False, held)
        cfg = {"problem": "gauss2_8", "hidden": "potts", "K": K8, "radius": R8}
        if cfg_grid is not None:
            cfg["time_grid"] = cfg_grid
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        open(os.path.join(d, "DONE"), "w").close()
        return os.path.relpath(d, lm.OUT_ROOT)

    old_knots = np.asarray(jnp.asarray(np.sort(lm.T_FORWARD - lm.forward_time_pts())[1:-1]))   # sample() before
    old_kt = sf.knot_times()
    r_leg = make_run("legacy_uniform", None, uni)
    prov = sample_latent.load_params(r_leg)[4]
    tk = sample_latent.slice_times(prov)
    bad = int(prov["time_grid"] != "uniform") + int(not same(tk, uni))
    bad += int(not same(np.asarray(sample_latent.step_knots(tk)), old_knots))
    bad += int(not same(np.asarray(sample_latent.step_knots()), old_knots))
    bad += sum(int(not same(a, b)) for a, b in zip(sf.knot_times(tk), old_kt))
    spec_r, taus_r = lm.run_time_grid(os.path.join(lm.OUT_ROOT, r_leg))
    bad += int(spec_r != "uniform") + int(not same(taus_r, uni))
    check("14 knots from a run without 'time_grid' (recorded t = uniform): grid 'uniform'; slice times, step_ts and "
          "the fast sampler's (order, xp, s) are today's arrays bit for bit (failures)", bad, 0.5)
    r_pow = make_run("power15", "power:1.5", g15)
    prov = sample_latent.load_params(r_pow)[4]
    tk = sample_latent.slice_times(prov)
    order, xp, s = sf.knot_times(tk)
    bad = int(prov["time_grid"] != "power:1.5") + int(not same(tk, g15))
    bad += int(not same(np.asarray(sample_latent.step_knots(tk)), np.sort(lm.T_FORWARD - g15)[1:-1]))
    bad += int(not (same(xp, np.sort(g15)) and same(s, np.sort(lm.T_FORWARD - g15)) and same(g15[order], xp)))
    spec_r, taus_r = lm.run_time_grid(os.path.join(lm.OUT_ROOT, r_pow))
    bad += int(spec_r != "power:1.5") + int(not same(taus_r, g15))
    report(f"   power:1.5 run: step_ts {np.asarray(sample_latent.step_knots(tk))[:2]} ... "
           f"{np.asarray(sample_latent.step_knots(tk))[-2:]}; reverse-time knots s[-3:] {s[-3:]}")
    check("14 knots from a run with 'time_grid' power:1.5 (recorded t = the grid): its slice times, step_ts and the "
          "fast sampler's knots (failures)", bad, 0.5)
    outcomes = []
    for name, cg, tr, expect_ok in (("mismatch_power_cfg", "power:1.5", uni, False),
                                    ("mismatch_uniform_cfg", None, g15, False),
                                    ("mismatch_2e-12", "power:1.5", g15 + 2e-12, False),
                                    ("mismatch_shape", "power:1.5", g15[:-1], False),
                                    ("within_5e-13", "power:1.5", g15 + 5e-13, True)):
        rn = make_run(name, cg, tr)
        try:
            prov = sample_latent.load_params(rn)[4]
            ok = expect_ok and same(sample_latent.slice_times(prov), g15)
            msg = "accepted"
        except ValueError as e:
            ok, msg = not expect_ok, f"refused: {str(e)[-150:]}"
        outcomes.append(ok)
        report(f"   {name}: {msg}")
    check("14 a run whose recorded slice times differ from its config's grid by more than 1e-12 is refused, within "
          "1e-12 accepted (5 cases; failures)", outcomes.count(False), 0.5)

    # the fast sampler at the power:1.5 knots: interval widths and weights, interpolated parameters, drift
    kw8 = dict(problem="gauss2_8", hidden="potts", local=(hist["wl"], hist["bl"]), quad=hist["Q"], radius=R8,
               neighbourhood="chebyshev")
    fs = sf.FastSampler(hist["theta"], hist["W"], hist["b"], taus=g15, **kw8)
    drift_old, _ = sample_latent.make_drift(hist["theta"], hist["W"], hist["b"], "sde", kw8["problem"], kw8["hidden"],
                                           kw8["local"], kw8["quad"], R8, "chebyshev", taus=g15)
    old_batch = jax.jit(jax.vmap(drift_old, in_axes=(None, 0, None)))
    parts = [hist["theta"], hist["W"].reshape(n_sl, -1), hist["b"], hist["Q"].reshape(n_sl, -1), hist["wl"], hist["bl"]]
    stacked = np.concatenate(parts, axis=1)
    interp = interpolate_parameters(jnp.asarray(stacked), jnp.asarray(g15))
    widths = np.diff(np.sort(g15))                            # width of the interval above xp[lo], lo = 0..98
    X = 0.5 * rng.normal(size=(6, n8))
    pts = [(0.0, [0]), (lm.T_FORWARD, [sf.N_INTERVALS - 1])]
    pts += [(float(fs.s[j]), [j - 1, j]) for j in (1, 2, 50, 90, 97, 98)]
    pts += [(float(fs.s[j] + f * (fs.s[j + 1] - fs.s[j])), [j]) for j in (0, 1, 30, 70, 96, 97, 98)
            for f in (0.013, 0.5, 0.987)]
    pts += [(float(t), [None]) for t in rng.uniform(0.0, lm.T_FORWARD, 8)]
    w_bad, worst_ip, worst_dr, l_range = 0, 0.0, 0.0, [np.inf, -np.inf]
    for t, js in pts:
        tau = lm.T_FORWARD - t
        ref_p = np.asarray(interp(tau))
        d_old = np.asarray(old_batch(t, jnp.asarray(X), ()))
        for j in js:
            j = fs.interval_of(t) if j is None else j
            lo = sf.interval_lo(j)
            k_lo, k_hi = fs.order[lo], fs.order[lo + 1]
            a = fs.interval_args(j)
            w_bad += int(float(a["dxp"]) != widths[lo]) + int(float(a["xp_lo"]) != g15[k_lo])
            w_bad += int(g15[k_hi] - g15[k_lo] != widths[lo])
            l = (tau - fs.xp[lo]) / (fs.xp[lo + 1] - fs.xp[lo])            # this interval's own width
            l_range = [min(l_range[0], l), max(l_range[1], l)]
            mine = stacked[k_lo] + l * (stacked[k_hi] - stacked[k_lo])
            worst_ip = max(worst_ip, float(np.max(np.abs(mine - ref_p)) / np.max(np.abs(ref_p))))
            if js == [j] and 0.0 < l < 1.0:                     # strictly inside: jnp.interp's bracket is this one
                i = int(np.clip(np.searchsorted(fs.xp, tau, side="right"), 1, n_sl - 1))
                w_bad += int(i - 1 != lo) + int(fs.interval_of(t) != j)
            d_new = np.asarray(fs.drift_batch(np.full(X.shape[0], t), X, j))
            worst_dr = max(worst_dr, float(np.max(np.abs(d_new - d_old)) / np.max(np.abs(d_old))))
    fs_u = sf.FastSampler(hist["theta"], hist["W"], hist["b"], **kw8)          # uniform knots: must not match
    t_c = float(fs.s[97] + 0.5 * (fs.s[98] - fs.s[97]))
    d_c = np.asarray(old_batch(t_c, jnp.asarray(X), ()))
    ctrl = float(np.max(np.abs(np.asarray(fs_u.drift_batch(np.full(X.shape[0], t_c), X, fs_u.interval_of(t_c))) - d_c))
                 / np.max(np.abs(d_c)))
    report(f"   fast sampler at the power:1.5 knots: interval widths {widths.min():.4g}-{widths.max():.4g} (ratio "
           f"{widths.max() / widths.min():.1f}); weights l in [{l_range[0]:.3g}, {l_range[1]:.3g}]; interpolated "
           f"parameters vs interpolate_parameters max relative diff {worst_ip:.1e}; drift vs make_drift max relative "
           f"diff {worst_dr:.1e} over {len(pts)} times; control (fast sampler on the uniform knots) {ctrl:.2e}")
    check("14 fast sampler, power:1.5 knots: each interval's width, lower knot and jnp.interp bracket (failures)",
          w_bad, 0.5)
    # rounding: at a knot l is off by about ulp(5) / width = 2e-13 in the narrowest interval (job 4223624: 7e-14), and
    # these synthetic slices are independent draws, so P_hi - P_lo is as large as P itself
    check("14 fast sampler, power:1.5 knots: interpolated parameters (own-width weights) = interpolate_parameters "
          "(max relative diff)", worst_ip, 1e-11)
    check("14 fast sampler drift at power:1.5 knots = sample_latent.make_drift with the same knots (max relative "
          "diff)", worst_dr, 1e-11)
    check("14 control: the fast sampler on the uniform knots differs from the power:1.5 drift (> 1e-3 relative; "
          "0 = it does)", 0.0 if ctrl > 1e-3 else 1.0, 0.5)
    fs_t = sf.FastSampler(hist["theta"], hist["W"], hist["b"], taus=gt, **kw8)               # power:1.5@40
    drift_t, _ = sample_latent.make_drift(hist["theta"], hist["W"], hist["b"], "sde", kw8["problem"], kw8["hidden"],
                                         kw8["local"], kw8["quad"], R8, "chebyshev", taus=gt)
    old_t = jax.jit(jax.vmap(drift_t, in_axes=(None, 0, None)))
    s_j = float(lm.T_FORWARD - gt[K0])                        # reverse time of the junction knot t_40
    worst_t = 0.0
    for t in (s_j, s_j - 1e-3, s_j + 1e-3, float(fs_t.s[98] + 0.5 * (fs_t.s[99] - fs_t.s[98])), 0.3, 4.99):
        d_o = np.asarray(old_t(t, jnp.asarray(X), ()))
        for j in sorted({fs_t.interval_of(t), fs_t.interval_of(t) - (t == s_j)}):
            d_n = np.asarray(fs_t.drift_batch(np.full(X.shape[0], t), X, j))
            worst_t = max(worst_t, float(np.max(np.abs(d_n - d_o)) / np.max(np.abs(d_o))))
    check(f"14 fast sampler drift at {tail} knots (junction from both sides, last interval) = make_drift (max "
          f"relative diff)", worst_t, 1e-11)

    # latent_model.interp_history (train_latent's paper comparison on another grid) against interpolate_parameters
    worst = 0.0
    for taus_i in (uni, g15):
        ip = interpolate_parameters(jnp.asarray(stacked), jnp.asarray(taus_i))
        for t in list(rng.uniform(0.0, lm.T_FORWARD, 12)) + [float(g15[37]), float(uni[63])]:
            ref_p = np.asarray(ip(t))
            worst = max(worst, float(np.max(np.abs(lm.interp_history(stacked, taus_i, t) - ref_p)) / np.max(np.abs(ref_p))))
        worst = max(worst, float(np.max(np.abs(lm.interp_history(stacked, taus_i, float(taus_i[41])) - stacked[41]))))
    check("14 latent_model.interp_history = interpolate_parameters (uniform and power:1.5 knots, random t and knots; "
          "max relative diff)", worst, 1e-13)
    report(f"   block 14: {time.time() - t14:.0f} s")


def block_15(data):
    """15: train_latent --split-dead (split_dead_states, 2026-11-12): free-energy-preserving re-seeding of dead Potts
    states. A small Potts model (12 states, --quad --local, the trained model's lattice and visible terms) with 4 states
    made dead (bias -50): at spread 0 the score-matching loss and the free energy are unchanged to rounding; at spread
    0.5 every dead state is alive again, the donors are the 4 busiest states, each pair shares its donor's occupation,
    and the loss moves only slightly; the spec parser."""
    t15 = time.time()
    k1, k2, k3, k4, k5, k6, k7 = jr.split(jr.PRNGKey(15), 7)
    conn_s = lm.connectivity(28, 1)
    P_s = conn_s.shape[0]
    zs = lm.dropped_pair_slices(lm.N, P_s, ("lin",))
    kept_m = np.ones(4 * lm.N + 3 * P_s, bool)
    for s in zs + (lm.k6_slice(lm.N),):
        kept_m[s] = False
    th = lm.quartic_initial_theta(lm.paper_theta_layout(P_s)[0], lm.N, P_s)
    th = jnp.where(kept_m, th, 0.0)
    Kp, dead_idx = 12, np.array([2, 5, 7, 11])
    b0 = 0.02 * jr.normal(k2, (Kp,))
    b0 = b0.at[dead_idx].set(-50.0)
    p0 = {"theta": th, "W": 0.05 * jr.normal(k1, (Kp, lm.N)), "b": b0, "Q": 0.005 * jr.normal(k3, (Kp, lm.N)),
          "wl": 0.3 * jr.normal(k4, (lm.N,)), "bl": 0.02 * jr.normal(k5, (lm.N,))}
    X = sample_forward_process(1.0, 2048, lm.TEMP, lm.SIGMA_FORWARD, data, k6, beta=1)
    vis = lm.make_visible(conn_s, lm.N, "stencil", ("lin",), sextic=False)
    loss = jax.jit(lm.make_loss(vis, lm.TEMP, Kp, "potts", True))

    def occ(p):
        A = np.asarray(X) @ np.asarray(p["W"]).T + np.asarray(p["b"]) + (np.asarray(X) ** 2) @ np.asarray(p["Q"]).T
        Z = A / lm.TEMP
        P = np.exp(Z - Z.max(axis=1, keepdims=True))
        return (P / P.sum(axis=1, keepdims=True)).mean(axis=0), Z

    def free_energy(p):
        _, Z = occ(p)
        zm = Z.max(axis=1)
        return -lm.TEMP * (zm + np.log(np.exp(Z - zm[:, None]).sum(axis=1)))

    pbar0, _ = occ(p0)
    busiest = set(np.argsort(-pbar0)[:4].tolist())
    L0, F0 = float(loss(p0, X)), free_energy(p0)
    p_s0, info0 = train_latent.split_dead_states(p0, X, 0.0)
    L1, F1 = float(loss(p_s0, X)), free_energy(p_s0)
    report(f"   split, spread 0: {info0}; loss {L0:.10g} -> {L1:.10g}; max |dF| {np.max(np.abs(F1 - F0)):.2e}")
    check("15 split at spread 0: score-matching loss unchanged (relative)", abs(L1 - L0) / abs(L0), 1e-10)
    check("15 split at spread 0: free energy unchanged on every sample (max |dF| / T)",
          float(np.max(np.abs(F1 - F0))) / lm.TEMP, 1e-9)
    p_s, info = train_latent.split_dead_states(p0, X, 0.5)
    pbar1, Z1 = occ(p_s)
    L2, F2 = float(loss(p_s, X)), free_energy(p_s)
    donors = [int(j) for j in range(Kp) if j not in dead_idx and not np.allclose(np.asarray(p_s["W"][j]),
                                                                                 np.asarray(p0["W"][j]))]
    pair_share = max(abs(float(pbar1[d] + pbar1[j]) - float(pbar0[j])) / float(pbar0[j])
                     for d, j in zip(sorted(dead_idx.tolist()), sorted(donors, key=lambda j: -pbar0[j])))
    bad = int(info["dead"] != 4 or info["split"] != 4) + int(set(donors) != busiest)
    bad += int(np.any(pbar1[dead_idx] < 1e-6))
    report(f"   split, spread 0.5: {info}; donors {sorted(donors)} (busiest {sorted(busiest)}); occupations of the "
           f"re-seeded states {np.array2string(pbar1[dead_idx], precision=3)}; loss {L2:.10g} (relative change "
           f"{abs(L2 - L0) / abs(L0):.2e}); max dF / T {np.max(F2 - F0) / lm.TEMP:+.3f}, min {np.min(F2 - F0) / lm.TEMP:+.3f}; "
           f"largest relative change of a pair's total occupation {pair_share:.3f}")
    check("15 split at spread 0.5: 4 dead states re-seeded from the 4 busiest donors, all alive afterwards (failures)",
          bad, 0.5)
    check("15 split at spread 0.5: free energy only lowered, by at most T log cosh of a few SD (max dF / T <= 0; "
          "min dF / T)", max(float(np.max(F2 - F0)) / lm.TEMP, 0.0) + (float(np.min(F2 - F0)) / lm.TEMP < -5.0), 1e-9)
    check("15 split at spread 0.5: loss moves only slightly (relative)", abs(L2 - L0) / abs(L0), 1e-2)
    ok = train_latent.parse_split_slices("20, 25,20") == (20, 25) and train_latent.parse_split_slices("") == ()
    for spec in ("0", "100", "x", "20;25", ","):
        try:
            train_latent.parse_split_slices(spec)
            ok = False
        except ValueError:
            pass
    check("15 --split-dead spec parser (valid lists, invalid ones refused)", 0.0 if ok else 1.0, 0.5)
    report(f"   block 15: {time.time() - t15:.0f} s")


INIT_ORIG = os.path.join(lm.OUT_ROOT, "init_dist_dev", "orig")    # block 16: sample_latent.py before --init-dist


def synthetic_run(path, K=4, seed=16):
    """A small finished 'run' of train_latent.py for the command-line checks of block 16: gauss2_8 geometry (8x8,
    radius-1 Chebyshev pairs), Potts K with --quad --local, the quartic visible energy with lin couplings, 100 slices on
    the uniform grid. Weak parameters, smooth in t (from t = 5 to 0: k_lin 1 -> 1.5, k_duff 0.01 -> 0.21, c_lin 0.05 ->
    0.1, small fields; |W_j| about 0.2 -> 0.3, Q 0 -> 0.003, w_local about 0.05 -> 0.1), so that the reverse SDE is
    stable and cheap on the CPU while every force term of the sampler's drift is active."""
    n, conn = 64, lm.connectivity(8, 1)
    P = conn.shape[0]
    taus = lm.forward_time_pts()
    rng = np.random.default_rng(seed)
    s = (1.0 - taus / lm.T_FORWARD)[:, None]                                 # 0 at t = 5, 1 at t = 0
    th = np.zeros((lm.N_TIME_STEPS, 4 * n + 3 * P))
    th[:, :n] = 1.0 + 0.5 * s + 0.05 * s * rng.standard_normal(n)             # k_lin
    th[:, n:2 * n] = 0.01 + 0.2 * s                                           # k_duff (k_6 = 0: quartic)
    th[:, 3 * n:3 * n + P] = 0.05 * (1.0 + s) * (1.0 + 0.1 * rng.standard_normal(P))     # c_lin
    th[:, 3 * n + 3 * P:] = 0.02 * s * rng.standard_normal(n)                 # bias
    hid = {"W": (0.02 + 0.01 * s)[:, :, None] * rng.standard_normal((K, n))[None],
           "b": 0.01 * (1.0 + s) * rng.standard_normal(K), "Q": 0.003 * s[:, :, None] * rng.standard_normal((K, n))[None],
           "wl": 0.05 * (1.0 + s) * rng.standard_normal(n), "bl": 0.01 * np.repeat(rng.standard_normal((1, n)), 100, 0),
           "t": taus}
    os.makedirs(path)
    np.save(os.path.join(path, "params_history.npy"), th)
    np.savez(os.path.join(path, "hidden_history.npz"), **hid)
    cfg = {"name": os.path.basename(path), "problem": "gauss2_8", "K": K, "hidden": "potts", "quad": True, "local": True,
           "visible": "quartic", "pair_terms": "lin", "radius": 1, "neighbourhood": "chebyshev", "cls": "none",
           "data_sha256": lm.sha256_file(lm.problem_data_path("gauss2_8")), "synthetic": "test_latent.py block 16"}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    open(os.path.join(path, "DONE"), "w").close()


def _main16(argv):
    """sample_latent.main() in this process with sys.argv = ['sample_latent.py'] + argv: (exit code / message, or the
    name and text of any other exception, or None if it returned; stderr text)."""
    old, err = sys.argv, io.StringIO()
    sys.argv = ["sample_latent.py"] + list(argv)
    try:
        with contextlib.redirect_stderr(err):
            sample_latent.main()
        return None, err.getvalue()
    except SystemExit as e:
        return e.code, err.getvalue()
    except Exception as e:                                    # noqa: BLE001  (a guard that did not hold)
        return f"{type(e).__name__}: {e}", err.getvalue()
    finally:
        sys.argv = old


def block_16():
    """16: sample_latent.py --init-dist (2026-11-12), the start distributions of the reverse process at t = 5; CPU-sized
    (the command-line runs are CPU subprocesses).
    gauss: data_stats on the six study runs' problem (mnist28_d0123; cache in this job's folder) is written once, then
    reused, and recomputed when the data file's hash or the mean disagree; its mean and eigen-decomposition against the
    data's own covariance, and the numbers of CPU job 4229465 (|mu|^2, trace C, top eigenvalues, mean offset, PC1 excess,
    KL); Sigma^(1/2) Sigma^(1/2) = e^-10 C + T (1 - e^-10) I and gauss_start = m + Sigma^(1/2) z; 20,000 standard draws
    mapped by gauss_start have the target mean (along mu / |mu| and per pixel) and covariance (along the top 5 principal
    axes, in the top-10 block and in total); the rank-r starts (r = 0, 1, 2, 10, 784) equal the full one on the kept
    axes and the default draw plus the mean shift outside them (r = 0: m + x0 exactly, r = 784: the full start exactly).
    equil: on a quadratic slice-0 energy F_0 = sum_i A_i (x_i - a_i)^2 / 2 (K = 0, through make_drift's PF-ODE drift;
    the other slices differ, so a wrong slice shows) the Langevin drift is -grad F_0; equilibrate's states after S =
    0.5 follow the exact Ornstein-Uhlenbeck transition from their starts (pooled z-scores of 256 x 64), a stationary
    set stays stationary, and a direct vmap of solve_SDE with keys fold_in(key, 1) reproduces equilibrate. Command
    line: the guards (invalid specs, --init-rank without gauss or out of range, with --fast / --thin-knots /
    --oracle-start, no overwriting); on a small synthetic run (synthetic_run), as CPU subprocesses: the default path
    bit for bit against the module before the option (INIT_ORIG; manifests equal but for run, chunk_seconds and the
    module hash), and --init-dist gauss (--snapshots: the start states are gauss_start of the set's draws), equil:0.5
    (--snapshots: the start states are equilibrate's), and --sampler ode with gauss and with gauss --init-rank 2. On
    the CPU an SDE step costs about 0.1-0.3 s (the Brownian tree at tolerance 1e-12) and SRA1 takes about 1,000 steps
    per time unit, so the block takes about 15 min there (job 4229780: the six command-line runs 10 min in parallel)."""
    t16 = time.time()
    sl = sample_latent
    T = lm.TEMP
    v0 = lm.TEMP * lm.SIGMA_FORWARD ** 2
    e5 = float(np.exp(-lm.T_FORWARD / lm.SIGMA_FORWARD ** 2))

    # ---- the synthetic run, and the command-line runs on it (CPU subprocesses, while the checks below run) ------
    run_a, run_b = os.path.join(OUT, "b16_synth_a"), os.path.join(OUT, "b16_synth_b")
    synthetic_run(run_a)
    os.makedirs(run_b)
    for f in ("config.json", "params_history.npy", "hidden_history.npz", "DONE"):
        os.symlink(os.path.join("..", "b16_synth_a", f), os.path.join(run_b, f))
    name_a, name_b = os.path.relpath(run_a, lm.OUT_ROOT), os.path.relpath(run_b, lm.OUT_ROOT)
    st_toy = sl.data_stats("gauss2_8")             # the cache file the command-line gauss starts below will load
    report(f"   synthetic run {name_a} (and {name_b}, links to it); gauss2_8 statistics {st_toy['path']} "
           f"({'computed' if st_toy['recomputed'] else 'cached'}, sha256 {st_toy['file_sha256'][:16]})")
    py, cur, orig = sys.executable, os.path.join(HERE, "sample_latent.py"), os.path.join(INIT_ORIG, "sample_latent.py")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env["PYTHONPATH"] = os.pathsep.join([lm.HERE] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    four = ["--n", "4", "--chunk", "4"]
    cmds = {"default_current": [py, "-u", cur, "--run", name_a] + four,
            "gauss": [py, "-u", cur, "--run", name_a, "--init-dist", "gauss", "--snapshots"] + four,
            "equil": [py, "-u", cur, "--run", name_a, "--init-dist", "equil:0.5", "--snapshots"] + four,
            "ode_gauss": [py, "-u", cur, "--run", name_a, "--sampler", "ode", "--init-dist", "gauss"] + four,
            "ode_gauss_r2": [py, "-u", cur, "--run", name_a, "--sampler", "ode", "--init-dist", "gauss",
                             "--init-rank", "2"] + four}
    if os.path.exists(orig):
        cmds["default_original"] = [py, "-u", orig, "--run", name_b] + four
    else:
        report(f"   {orig} is missing: the default path is not compared with the module before --init-dist")
    procs = {}
    for tag, cmd in cmds.items():
        fo = open(os.path.join(OUT, f"b16_{tag}.log"), "w")
        procs[tag] = (subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, env=env), fo, time.time())

    # ---- gauss: the data statistics and their cache ------------------------------------------------------------
    stats_dir = os.path.join(OUT, "b16_data_stats")
    st = sl.data_stats("mnist28_d0123", cache_dir=stats_dir)
    st2 = sl.data_stats("mnist28_d0123", cache_dir=stats_dir)
    same = all(np.array_equal(st[k], st2[k]) for k in ("mu", "lam", "U")) and st2["file_sha256"] == st["file_sha256"]
    check("16 data_stats: computed and written once, then reused from the file (same arrays and file hash; failures)",
          int(not st["recomputed"]) + int(st2["recomputed"]) + int(not same), 0.5)
    with np.load(st["path"]) as z:
        rec = {k: z[k] for k in z.files}
    redo = []
    for tamper in ({"data_file_sha256": "0" * 64}, {"mu": rec["mu"] + 1e-6}, {"n_images": rec["n_images"] + 1}):
        np.savez(st["path"], **dict(rec, **tamper))
        st3 = sl.data_stats("mnist28_d0123", cache_dir=stats_dir)
        sgn = np.sign(np.sum(st3["U"] * st["U"], axis=0))
        redo.append(bool(st3["recomputed"]) and rel(st3["mu"], st["mu"]) < 1e-14 and rel(st3["lam"], st["lam"]) < 1e-12
                    and rel(st3["U"] * sgn, st["U"]) < 1e-9)
    check("16 data_stats: a cached file with another data-file hash, a shifted mean or another image count is "
          f"recomputed (and agrees with the first computation; {redo})", 0.0 if all(redo) else 1.0, 0.5)
    prob = lm.make_problem("mnist28_d0123", 1)
    X = np.asarray(prob["data"], dtype=np.float64)
    mu, lam, U = st["mu"], st["lam"], st["U"]
    C = np.cov(X, rowvar=False, bias=True)
    check("16 data_stats: mean = the data's mean (relative)", rel(mu, X.mean(axis=0)), 1e-13)
    check("16 data_stats: U diag(lam) U^T = the data's population covariance (relative), lam descending",
          rel((U * lam) @ U.T, C) + float(np.any(np.diff(lam) > 0)), 1e-12)
    ref = {"|mu|^2": (float(mu @ mu), 234.48, 0.006), "trace C": (float(lam.sum()), 549.5, 0.1)}
    for k, v in enumerate((92.39, 43.45, 40.38, 30.72, 24.49)):
        ref[f"lam_{k + 1}"] = (float(lam[k]), v, 0.01)
    kl_def = sl.gauss_kl(st, 0, shift=False)
    v0_, e_, delta = sl._gauss_axes(st, lm.T_FORWARD)
    ref["mean offset (sd)"] = (e5 * float(np.linalg.norm(mu)) / np.sqrt(v0), 0.730, 6e-4)
    ref["PC1 variance excess"] = (float(delta[0] / v0), 0.210, 6e-4)
    ref["KL mean part"] = (kl_def[0], 0.266, 6e-4)
    ref["KL covariance part"] = (kl_def[1], 0.0185, 1e-4)
    bad = [k for k, (val, want, tol) in ref.items() if not abs(val - want) <= tol]
    report("   mnist28_d0123 statistics (population covariance) vs job 4229465 (sample covariance, rounded): "
           + "; ".join(f"{k} {val:.5g} ({want:g})" for k, (val, want, _) in ref.items()))
    check(f"16 the statistics reproduce job 4229465's numbers within its rounding (outside: {bad})", len(bad), 0.5)
    sig = v0 + delta                                        # Sigma's eigenvalues
    m = e5 * mu
    Sigma = e5 ** 2 * C + v0 * (1.0 - e5 ** 2) * np.eye(mu.size)
    d = delta / (np.sqrt(sig) + np.sqrt(v0))
    L = np.sqrt(v0) * np.eye(mu.size) + (U * d) @ U.T
    check("16 gauss: Sigma^(1/2) Sigma^(1/2) = e^-10 C + T (1 - e^-10) I (relative), Sigma^(1/2) symmetric",
          rel(L @ L, Sigma) + rel(L, L.T), 1e-12)
    rng = np.random.default_rng(1616)
    Z = rng.standard_normal((20000, mu.size))
    X0 = np.sqrt(v0) * Z
    check("16 gauss: gauss_start(sqrt(T) z) = e^-5 mu + Sigma^(1/2) z (8 draws, relative)",
          rel(sl.gauss_start(X0[:8], st), m + Z[:8] @ L), 1e-12)
    Xg = sl.gauss_start(X0, st)
    nd = Xg.shape[0]
    mh = mu / np.linalg.norm(mu)
    z_off = float((Xg.mean(axis=0) - m) @ mh) / np.sqrt(float(mh @ Sigma @ mh) / nd)
    z_pix = float(np.max(np.abs(Xg.mean(axis=0) - m) / np.sqrt(np.diag(Sigma) / nd)))
    Y = (Xg - m) @ U[:, :10]
    var_rel = np.var(Y, axis=0) / sig[:10] - 1.0
    blk = float(np.linalg.norm(np.cov(Y, rowvar=False, bias=True) - np.diag(sig[:10])) / np.linalg.norm(sig[:10]))
    tot = float(np.var(Xg, axis=0).sum() / sig.sum() - 1.0)
    tot_none = float(sig.sum() - v0 * mu.size) / float(sig.sum())
    report(f"   20,000 draws: mean offset along mu {float(Xg.mean(axis=0) @ mh) / np.sqrt(v0):.4f} sd (target "
           f"{float(m @ mh) / np.sqrt(v0):.4f}; z-score {z_off:+.2f}), largest pixel z-score {z_pix:.2f}; variance along "
           f"the top 5 axes / target - 1: {np.array2string(var_rel[:5], precision=4)} (target / T: "
           f"{np.array2string(sig[:5] / v0, precision=3)}); top-10 block relative error {blk:.3f}; total variance "
           f"relative error {tot:+.2e} (the default start would be {-tot_none:+.2e})")
    check("16 gauss, 20,000 draws: mean along mu / |mu| (|z-score|)", abs(z_off), 5.0)
    check("16 gauss, 20,000 draws: mean per pixel (largest |z-score| of 784)", z_pix, 5.0)
    check("16 gauss, 20,000 draws: variance along the top 5 principal axes (largest relative error; SE 1%)",
          float(np.max(np.abs(var_rel[:5]))), 0.05)
    check("16 gauss, 20,000 draws: covariance in the top-10 principal block (relative Frobenius error; SE about 2%)",
          blk, 0.06)
    check("16 gauss, 20,000 draws: total variance (relative error; SE 1.1e-4, the excess is 1.5e-3)", abs(tot), 6e-4)
    def ratio(a, b):                                          # max |a| / max |b|
        return float(np.max(np.abs(a)) / np.max(np.abs(b)))

    bad, lines = 0, []
    for r in (0, 1, 2, 10, mu.size):
        Xr = sl.gauss_start(X0, st, r)
        if r == 0:
            bad += int(not np.array_equal(Xr, m + X0))
        if r == mu.size:
            bad += int(not np.array_equal(Xr, Xg))
        kept = ratio((Xr - Xg) @ U[:, :r], (Xg - m) @ U[:, :r]) if r else 0.0
        outside = ratio((Xr - m - X0) @ U[:, r:], X0 @ U[:, r:]) if r < mu.size else 0.0
        v1 = float(np.var((Xr - m) @ U[:, 0])) / (sig[0] if r else v0) - 1.0
        bad += int(kept > 1e-12) + int(outside > 1e-12) + int(abs(v1) > 0.05)
        lines.append(f"r {r}: kept axes vs full {kept:.1e}, outside vs default + shift {outside:.1e}, PC1 variance "
                     f"{v1:+.4f}, KL {sum(sl.gauss_kl(st, r)):.4f}")
        del Xr
    report("   rank-r starts: " + "; ".join(lines))
    check("16 gauss --init-rank r (0, 1, 2, 10, 784): = the full start on the kept axes, = default draw + mean shift "
          "outside them, PC1 variance as targeted; r = 0 is m + x0 and r = 784 the full start, bit for bit (failures)",
          bad, 0.5)
    kls = [sum(sl.gauss_kl(st, r)) for r in (0, 1, 2, 5, 10, 50, mu.size)]
    check("16 gauss_kl: decreasing with the rank, 0 at full rank (failures)",
          int(np.any(np.diff(kls) > 0)) + int(kls[-1] != 0.0), 0.5)
    del X, Xg, X0, Z, C, Sigma, L, Y

    # ---- equil: the Langevin routine on an analytic quadratic energy ------------------------------------------
    n8, conn8 = 64, lm.connectivity(8, 1)
    P8 = conn8.shape[0]
    rng = np.random.default_rng(1617)
    A = rng.uniform(0.5, 2.0, n8)
    a = 2.0 * np.sqrt(T) * rng.standard_normal(n8)
    th = np.zeros((lm.N_TIME_STEPS, 4 * n8 + 3 * P8))
    th[:, :n8] = 1.0                                         # every other slice: k_lin = 1, no field
    th[0, :n8] = A
    th[0, 3 * n8 + 3 * P8:] = -A * a                         # F_0 = sum_i A_i x_i^2 / 2 - A_i a_i x_i
    drift_ode, diffusion = sl.make_drift(th, np.zeros((lm.N_TIME_STEPS, 0, n8)), np.zeros((lm.N_TIME_STEPS, 0)), "ode",
                                         "gauss2_8", "potts", None, None, 1, "chebyshev")
    xr = 0.3 * rng.standard_normal((5, n8))
    ld = np.stack([np.asarray(drift_ode(0.0, jnp.asarray(x), ())) - x for x in xr])
    check("16 equil: Langevin drift drift_ode(0, x) - x = -grad F_0 = -A (x - a) (relative)", rel(ld, -A * (xr - a)),
          1e-12)
    # (on the CPU every SDE step costs about 0.1-0.3 s, mostly the Brownian tree at tolerance 1e-12, and SRA1 takes
    # about 1,000 steps per time unit even here: hence S = 0.5 and one chunk; the OU transition is exact for any S)
    ntr, S = 256, 0.5
    keys = np.asarray(jr.split(jr.PRNGKey(1618), ntr))
    x0 = np.sqrt(v0) * rng.standard_normal((ntr, n8))
    xs, info = sl.equilibrate(drift_ode, diffusion, x0, keys, S, ntr)
    dec = np.exp(-A * S)
    zz = (xs - (a + (x0 - a) * dec)) / np.sqrt(T / A * (1.0 - dec ** 2))
    z_none = (x0 - (a + (x0 - a) * dec)) / np.sqrt(T / A * (1.0 - dec ** 2))
    nz = zz.size
    report(f"   equilibrate, S = {S:g}, {ntr} trajectories x {n8}: accepted steps mean {info['steps_accepted_mean']:.0f} "
           f"(max {info['steps_accepted_max']}), rejected mean {info['steps_rejected_mean']:.0f}, "
           f"{sum(info['chunk_seconds']):.0f} s; z-scores against the exact OU transition from each start: mean "
           f"{zz.mean():+.4f}, mean square {np.mean(zz ** 2):.4f} (the starts themselves: mean square "
           f"{np.mean(z_none ** 2):.2f})")
    check(f"16 equil: states after S = {S:g} vs the exact OU transition from each start: mean of the z-scores "
          f"(SE {1 / np.sqrt(nz):.4f})", abs(float(zz.mean())), 5.0 / np.sqrt(nz))
    check(f"16 equil: ... mean square of the z-scores - 1 (SE {np.sqrt(2.0 / nz):.4f})",
          abs(float(np.mean(zz ** 2)) - 1.0), 5.0 * np.sqrt(2.0 / nz))
    keys2 = np.asarray(jr.split(jr.PRNGKey(1619), ntr))
    xst = a + np.sqrt(T / A) * rng.standard_normal((ntr, n8))
    xs2, _ = sl.equilibrate(drift_ode, diffusion, xst, keys2, S, ntr)
    z_b, z_a = (xst - a) / np.sqrt(T / A), (xs2 - a) / np.sqrt(T / A)
    report(f"   stationary start, S = {S:g}: z-scores against N(a, T/A) before: mean {z_b.mean():+.4f}, mean square "
           f"{np.mean(z_b ** 2):.4f}; after: {z_a.mean():+.4f}, {np.mean(z_a ** 2):.4f}")
    check(f"16 equil: a stationary set stays stationary (S = {S:g}): mean of the z-scores against N(a, T/A)",
          abs(float(z_a.mean())), 5.0 / np.sqrt(nz))
    check("16 equil: ... mean square of the z-scores - 1", abs(float(np.mean(z_a ** 2)) - 1.0), 5.0 * np.sqrt(2.0 / nz))
    x8, _ = sl.equilibrate(drift_ode, diffusion, x0[:8], keys[:8], S, 8)
    fn = jax.vmap(lambda y, k: solve_SDE(lambda s_, x_, a_: drift_ode(0.0, x_, a_) - x_, diffusion, y,
                                         jr.fold_in(k, 1), 0.0, S, 2, lm.DT0, rtol=lm.RTOL_SDE, atol=lm.ATOL_SDE,
                                         brownian_tolerance=lm.BROWNIAN_TOL))
    xd = np.asarray(fn(jnp.asarray(x0[:8]), jnp.asarray(keys[:8])).ys[:, -1, :])
    check("16 equil: a direct vmap of solve_SDE with keys fold_in(key, 1) reproduces equilibrate (8 trajectories, max "
          "|diff|)", float(np.max(np.abs(xd - x8))), 1e-9)

    # ---- command-line guards (in this process; none of them may start a solve) --------------------------------
    good = (sl.parse_init_dist("set") == ("set", None) and sl.parse_init_dist(" gauss") == ("gauss", None)
            and sl.parse_init_dist("equil:3") == ("equil", 3.0) and sl.parse_init_dist("equil:1e-1") == ("equil", 0.1)
            and sl.init_suffix("set") == "" and sl.init_suffix("gauss") == "_init-gauss"
            and sl.init_suffix("gauss", None, 0) == "_init-gauss_r0" and sl.init_suffix("gauss", None, 10) == "_init-gauss_r10"
            and sl.init_suffix("equil", 3.0) == "_init-equil3" and sl.init_suffix("equil", 0.5) == "_init-equil0.5")
    check("16 parse_init_dist / init_suffix on valid specs", 0.0 if good else 1.0, 0.5)
    nope = ["--run", "no_such_run_b16"]
    cases = [(nope + ["--init-dist", s], 2, "--init-dist") for s in
             ("foo", "equil", "equil:", "equil:0", "equil:-1", "equil:x", "equil:inf", "equil:nan", "gauss:2", "GAUSS")]
    cases += [(nope + ["--init-rank", "2"], 2, "--init-rank"),
              (nope + ["--init-dist", "equil:1", "--init-rank", "2"], 2, "--init-rank"),
              (nope + ["--init-dist", "gauss", "--init-rank", "-1"], 2, "--init-rank"),
              (nope + ["--init-dist", "gauss", "--fast"], 2, "default sampler only"),
              (nope + ["--init-dist", "equil:1", "--fast"], 2, "default sampler only"),
              (nope + ["--init-dist", "gauss", "--thin-knots", "0:1"], 2, "--thin-knots"),
              (nope + ["--init-dist", "equil:1", "--oracle-start", "5"], 2, "--oracle-start")]
    os.makedirs(os.path.join(run_a, "samples"), exist_ok=True)
    dummy = np.arange(6.0)
    for stem in ("sde_base_n3_init-gauss", "sde_base_n3_init-equil2"):
        np.save(os.path.join(run_a, "samples", stem + ".npy"), dummy)
    run3 = ["--run", name_a, "--n", "3", "--chunk", "3"]
    cases += [(run3 + ["--init-dist", "gauss", "--init-rank", "65"], "msg", "--init-rank 65"),
              (run3 + ["--init-dist", "gauss"], "msg", "does not overwrite"),
              (run3 + ["--init-dist", "equil:2"], "msg", "does not overwrite")]
    bad = []
    for argv, want, text in cases:
        code, err = _main16(argv)
        ok = (code == 2 and text in err) if want == 2 else (isinstance(code, str) and text in code)
        if not ok:
            bad.append((" ".join(argv[2:]), code, err.strip().splitlines()[-1:] if err.strip() else ""))
    kept = all(np.array_equal(np.load(os.path.join(run_a, "samples", s + ".npy")), dummy)
               for s in ("sde_base_n3_init-gauss", "sde_base_n3_init-equil2"))
    report(f"   {len(cases)} refused command lines; failures: {bad}")
    check(f"16 command-line guards: invalid --init-dist specs, --init-rank without gauss / negative / above the "
          f"dimension, --init-dist with --fast / --thin-knots / --oracle-start, existing files (failures of "
          f"{len(cases)}; existing files untouched: {kept})", len(bad) + int(not kept), 0.5)

    # ---- the command-line runs ---------------------------------------------------------------------------------
    rcs = {}
    for tag, (p, fo, t_p) in procs.items():
        try:
            rcs[tag] = p.wait(timeout=max(60.0, 1500.0 - (time.time() - t16)))
        except subprocess.TimeoutExpired:
            p.kill()
            rcs[tag] = "timeout"
        fo.close()
        with open(os.path.join(OUT, f"b16_{tag}.log")) as f:
            lines = [ln.rstrip() for ln in f if ln.startswith(("--init-dist", "  KL", "  start states", "  equilibrated",
                                                              "  diagnostic", "saved", "equilibration chunk"))]
        report(f"   {tag}: exit code {rcs[tag]}, {time.time() - t_p:.0f} s after its start; " + " | ".join(lines))
    samples = os.path.join(run_a, "samples")

    def load_out(run_dir, stem):
        p_ = os.path.join(run_dir, "samples", stem)
        if not (os.path.exists(p_ + ".npy") and os.path.exists(p_ + ".json")):
            return None, None
        with open(p_ + ".json") as f:
            return np.load(p_ + ".npy"), json.load(f)

    xa, ma = load_out(run_a, "sde_base_n4")
    if "default_original" in procs:
        xb, mb = load_out(run_b, "sde_base_n4")
        ok = rcs["default_current"] == 0 and rcs["default_original"] == 0 and xa is not None and xb is not None
        d = float(np.max(np.abs(xa - xb))) if ok else float("nan")
        check("16 default path through the command line (synthetic run, 4 trajectories): final states = the module "
              "before --init-dist, bit for bit (max |diff|)", d if ok and np.array_equal(xa, xb) else 1.0, 1e-300)
        skip = {"run", "chunk_seconds", "sha256_sample_latent"}
        diff_k = sorted(k for k in set(ma or {}) | set(mb or {}) if k not in skip and (ma or {}).get(k) != (mb or {}).get(k))
        check(f"16 default path: the manifests have the same keys (no 'init_dist') and values but for run, "
              f"chunk_seconds and the module hash (differing: {diff_k})",
              0.0 if ok and not diff_k and "init_dist" not in ma else 1.0, 0.5)
    x4, k4 = lm.trajectory_set("base", 4, "gauss2_8", 64)
    xg, mg = load_out(run_a, "sde_base_n4_init-gauss")
    bad = []
    if rcs["gauss"] != 0 or xg is None:
        bad.append(f"exit code {rcs['gauss']}")
    else:
        snap = np.load(os.path.join(samples, "sde_base_n4_init-gauss_snapshots.npz"))["states"]
        dg = rel(snap[:, 0, :], sl.gauss_start(x4, st_toy))
        ig = mg.get("init_dist", {})
        report(f"   gauss: start states (snapshot, float32) vs gauss_start of the set's draws {dg:.1e}; manifest "
               f"init_dist kind {ig.get('kind')}, rank {ig.get('rank')}, stats {ig.get('stats_file')} "
               f"{str(ig.get('stats_sha256'))[:16]}, KL this / default {ig.get('kl_this_start_nats')} / "
               f"{ig.get('kl_default_start_nats')}")
        bad += ["start states"] if dg > 2e-7 else []
        bad += ["manifest"] if (ig.get("kind") != "gauss" or ig.get("rank") != "full"
                                or ig.get("stats_sha256") != st_toy["file_sha256"]
                                or ig.get("stats_file") != st_toy["path"]) else []
        bad += ["not finite"] if not np.all(np.isfinite(xg)) else []
        bad += ["equal to the normal samples"] if xa is not None and np.array_equal(xg, xa) else []
    xr2, mr2 = load_out(run_a, "ode_base_n4_init-gauss_r2")
    if rcs["ode_gauss_r2"] != 0 or xr2 is None:
        bad.append(f"rank 2: exit code {rcs['ode_gauss_r2']}")
    else:
        ir = mr2.get("init_dist", {})
        bad += ["rank 2 manifest"] if ir.get("rank") != 2 or ir.get("numbers_used") != 64 + 2 * 64 - 1 else []
        bad += ["rank 2 not finite"] if not np.all(np.isfinite(xr2)) else []
    xo, mo = load_out(run_a, "ode_base_n4_init-gauss")
    if rcs["ode_gauss"] != 0 or xo is None:
        bad.append(f"ode: exit code {rcs['ode_gauss']}")
    else:
        bad += ["ode manifest"] if mo.get("sampler") != "ode" or mo.get("init_dist", {}).get("kind") != "gauss" else []
        bad += ["ode not finite"] if not np.all(np.isfinite(xo)) else []
        bad += ["rank 2 = full"] if xr2 is not None and np.array_equal(xr2, xo) else []
    check(f"16 --init-dist gauss through the command line (sde with --snapshots; --sampler ode, also with --init-rank "
          f"2): exit codes, start states = gauss_start of the set's draws, manifests, finite samples (failures: {bad})",
          len(bad), 0.5)
    bad = []
    xe, me = load_out(run_a, "sde_base_n4_init-equil0.5")
    if rcs["equil"] != 0 or xe is None:
        bad.append(f"exit code {rcs['equil']}")
    else:
        th_a, W_a, b_a, _, prov_a, _ = sl.load_params(name_a)
        lp, qp = prov_a.pop("local", None), prov_a.pop("quad", None)
        drift_a, diff_a = sl.make_drift(th_a, W_a, b_a, "ode", "gauss2_8", "potts", lp, qp, 1, "chebyshev")
        xeq, _ = sl.equilibrate(drift_a, diff_a, x4, k4, 0.5, 4)
        snap = np.load(os.path.join(samples, "sde_base_n4_init-equil0.5_snapshots.npz"))["states"]
        de = rel(snap[:, 0, :], xeq)
        ie = me.get("init_dist", {})
        report(f"   equil:0.5: start states (snapshot, float32) vs equilibrate in this process {de:.1e}; manifest "
               f"init_dist kind {ie.get('kind')}, S {ie.get('S')}, accepted steps mean {ie.get('steps_accepted_mean')}, "
               f"key {ie.get('key')}; diagnostic offset along mu before / after "
               f"{ie.get('diagnostic_only', {}).get('start_moments_before', {}).get('offset_along_mu_sd')} / "
               f"{ie.get('diagnostic_only', {}).get('start_moments_after', {}).get('offset_along_mu_sd')}")
        bad += ["start states"] if de > 2e-7 else []
        bad += ["manifest"] if (ie.get("kind") != "equil" or ie.get("S") != 0.5 or not ie.get("steps_accepted_mean")
                                or "fold_in" not in str(ie.get("key"))) else []
        bad += ["not finite"] if not np.all(np.isfinite(xe)) else []
    names = sorted(os.listdir(samples))
    want = sorted(["sde_base_n4.npy", "sde_base_n4.json", "sde_base_n3_init-gauss.npy", "sde_base_n3_init-equil2.npy",
                   "sde_base_n4_init-gauss.npy", "sde_base_n4_init-gauss.json", "sde_base_n4_init-gauss_snapshots.npz",
                   "ode_base_n4_init-gauss_r2.npy", "ode_base_n4_init-gauss_r2.json", "ode_base_n4_init-gauss.npy",
                   "ode_base_n4_init-gauss.json", "sde_base_n4_init-equil0.5.npy", "sde_base_n4_init-equil0.5.json",
                   "sde_base_n4_init-equil0.5_snapshots.npz"])
    bad += [f"files {sorted(set(names) ^ set(want))}"] if names != want else []
    check(f"16 --init-dist equil:0.5 through the command line: exit code, start states = equilibrate's, manifest; the "
          f"samples folder holds exactly the expected files (failures: {bad})", len(bad), 0.5)
    report(f"   block 16: {time.time() - t16:.0f} s")


# block 17: the modules from before --harmonic (latent_model, train_latent, sample_latent, sample_fast, test_latent and
# early_slices as installed on 2026-11-12 at 20:12), in a folder whose parent mirrors physical-diffusion-models (data,
# physical_diffusion_fns and out/latent_physical_SGM/data are links there), so that they run as they did
HARM_ORIG = os.environ.get("HARMONIC_ORIG", os.path.join(lm.OUT_ROOT, "harmonic_dev", "pkg", "latent_orig"))
# block 18: the first --harmonic version (as installed on 2026-11-12 at 21:17), laid out like HARM_ORIG
HARM_V1 = os.environ.get("HARMONIC_V1", os.path.join(lm.OUT_ROOT, "harmonic_dev", "pkg", "latent_v1"))
VOL17 = {"seconds", "steps_per_s", "job", "code_sha256"}          # record fields that may differ between two runs
# config.json keys added by train_latent.py --patch (2026-11-13), at their inactive values (blocks 17, 18 and 19)
PATCH_NEW_KEYS = {"patch": "", "patch_init_scale": 0.25, "patch_lr_scale": 1.0, "patch_on_slice": 0}


def _orig17(name):
    """The module `name` from before --harmonic (HARM_ORIG/<name>.py), imported in this process as <name>_orig17. Its
    own 'import latent_model' finds this process's latent_model (the copy of latent_model itself is the old code
    throughout); sys.path is restored afterwards."""
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(f"{name}_orig17", os.path.join(HARM_ORIG, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def _main17(module, argv):
    """module.main() in this process with sys.argv = ['<module>.py'] + argv, for command lines that must be refused
    before any work: (the code or message of a SystemExit, 'ExceptionName: message' of another exception, or None if it
    returned; stderr text)."""
    old, err = sys.argv, io.StringIO()
    sys.argv = [f"{module.__name__}.py"] + list(argv)
    try:
        with contextlib.redirect_stderr(err):
            module.main()
        return None, err.getvalue()
    except SystemExit as e:
        return e.code, err.getvalue()
    except Exception as e:                                    # noqa: BLE001  (a refusal is a ValueError / RuntimeError)
        return f"{type(e).__name__}: {e}", err.getvalue()
    finally:
        sys.argv = old


def _same17(a, b):
    """Bit for bit: dtype, shape and bytes."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _slices_differ17(dir_a, dir_b, ks, skip=()):
    """The differences between slices ks of two runs: missing files, other members, arrays that are not equal bit for
    bit, and record fields (but VOL17 and `skip`) that differ."""
    skip = set(VOL17) | set(skip)
    bad = []
    for k in ks:
        fa, fb = (os.path.join(d, "slices", f"slice_{k:03d}.npz") for d in (dir_a, dir_b))
        if not (os.path.exists(fa) and os.path.exists(fb)):
            bad.append(f"{k}: missing")
            continue
        with np.load(fa) as za, np.load(fb) as zb:
            if sorted(za.files) != sorted(zb.files):
                bad.append(f"{k}: members {sorted(set(za.files) ^ set(zb.files))}")
                continue
            for m in za.files:
                if m == "record":
                    ra, rb = json.loads(str(za[m])), json.loads(str(zb[m]))
                    d = sorted(f for f in set(ra) | set(rb)
                               if f not in skip and json.dumps(ra.get(f)) != json.dumps(rb.get(f)))
                    bad += [f"{k}: record {d}"] if d else []
                elif not _same17(za[m], zb[m]):
                    bad.append(f"{k}: {m}")
    return bad


def _synth_hist17(n, conn, K, R, seed, modes="random"):
    """A smooth synthetic 100-slice history (uniform grid) of a Potts K --quad --local model with the quartic visible
    energy and lin pair springs (as synthetic_run), in dimension n with pairs conn, plus R harmonic modes: modes
    'random' (rows of norm about 0.1 -> 0.3 from t = 5 to 0, fields about 0.05), 'zero' (V = c = 0) or None (no modes).
    Returns dict(theta, W, b, Q, wl, bl[, V, c], t) in hidden_history.npz's member order."""
    P = conn.shape[0]
    taus = lm.forward_time_pts()
    rng = np.random.default_rng(seed)
    s = (1.0 - taus / lm.T_FORWARD)[:, None]                                 # 0 at t = 5, 1 at t = 0
    th = np.zeros((lm.N_TIME_STEPS, 4 * n + 3 * P))
    th[:, :n] = 1.0 + 0.5 * s + 0.05 * s * rng.standard_normal(n)             # k_lin
    th[:, n:2 * n] = 0.01 + 0.2 * s                                           # k_duff (k_6 = 0: quartic)
    th[:, 3 * n:3 * n + P] = 0.05 * (1.0 + s) * (1.0 + 0.1 * rng.standard_normal(P))     # c_lin
    th[:, 3 * n + 3 * P:] = 0.02 * s * rng.standard_normal(n)                 # bias
    h = {"theta": th, "W": (0.02 + 0.01 * s)[:, :, None] * rng.standard_normal((K, n))[None],
         "b": 0.01 * (1.0 + s) * rng.standard_normal(K), "t": taus,
         "wl": 0.05 * (1.0 + s) * rng.standard_normal(n), "bl": 0.01 * np.repeat(rng.standard_normal((1, n)), 100, 0),
         "Q": 0.003 * s[:, :, None] * rng.standard_normal((K, n))[None]}
    if modes is not None:
        h["V"] = (0.1 + 0.2 * s)[:, :, None] * rng.standard_normal((R, n))[None] / np.sqrt(n)
        h["c"] = 0.05 * (1.0 + s) * rng.standard_normal(R)
        if modes == "zero":
            h["V"], h["c"] = np.zeros_like(h["V"]), np.zeros_like(h["c"])
    return h


def _write_run17(path, hist, cfg):
    """A finished 'run' folder from a synthetic history (_synth_hist17): params_history.npy, hidden_history.npz (the
    members in train_latent's order), config.json, DONE."""
    os.makedirs(path)
    np.save(os.path.join(path, "params_history.npy"), hist["theta"])
    np.savez(os.path.join(path, "hidden_history.npz"), **{k: v for k, v in hist.items() if k != "theta"})
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(dict(cfg, name=os.path.basename(path), synthetic="test_latent.py block 17"), f, indent=2)
    open(os.path.join(path, "DONE"), "w").close()


def block_17():
    """17: train_latent.py --harmonic R (2026-11-12), R harmonic hidden modes F_h = -1/2 sum_j (v_j . x + c_j)^2, and the
    sampler support; CPU-sized (the command-line runs are CPU subprocesses, partly in parallel).
    A. Formulas: harmonic_terms' gradient and Laplacian against autodiff of harmonic_free_energy; the loss
       (make_loss) and the force (make_force) with modes against a full-autodiff reference (paper-form visible energy
       with every pair group, Potts / Ising units with and without --quad, per-pixel units, the modes) for five model
       configurations, and the parameter gradient along a random direction against central differences; float64
       against --precision mixed with modes (Potts 8 --quad --local, 28x28 r1 lin quartic, 16 modes; every product
       float32 at HIGHEST).
    B. Default paths in this process against the modules before the option (_orig17): loss and parameter gradient bit
       for bit (and the same jaxpr) for six configurations (K = 0 dense f64; Potts --quad --local stencil f64 and mixed;
       Ising; sampled statistics with the jackknife), make_force, the runner (default and lean) over 2 x 50 steps,
       make_drift (SDE and PF-ODE) on a synthetic history, load_params.
    C. The closed-form reference: on synthetic Gaussian data (N = 64, spectrum 30 -> 0.002) at t = 5 ... 0 the
       reference modes plus the isotropic visible part give the exact Gaussian force T Sigma_t^-1 (x - m_t) and loss
       -tr Sigma^-1 + |Sigma^-1 (x - m)|^2 / 2 (R = N), exact on the top-R axes for R = 10; the precision T / iso - V^T V
       is positive (its smallest eigenvalue T / (e^2 lam_1 + sigma_t^2)); the strengths deviate from the plain formula
       (lam_0 = 0) by exactly -(lam_0 / lam_i + b) / (1 + b), b = e^2 lam_0 / sigma_t^2; the reference is a stationary
       point of the expected loss E_{N(m_t, Sigma_t)} L (exact through 2N sigma points, the loss being quadratic in x
       there; for R = N its global minimum, the Fisher divergence being 0) with no descent along 5 random directions
       (R = N and 10). On mnist28_d0123's statistics (data_stats, this job's own cache): the
       same exactness (R = 784, R = 64 on its axes) at t = 5 ... 0, confinement at all 100 slice times, finite positive
       learning-rate scales; a table of the reference; the quartic floor confines modes that exceed the springs.
       harmonic_unit_stats on the reference itself; visible_hessian_at_zero against autodiff.
    D. The sampler: make_drift with modes against x - s grad F by autodiff (s = 2 SDE, 1 ODE) with every parameter
       interpolated between the knots; load_params refuses a run with modes unless asked (with_harmonic=True), and
       --fast refuses it; a runner with modes trains them (lr > 0) and leaves them bit for bit (lr 0, the fixed case).
    E. Command lines (gauss2_8, Potts 8 --quad --local, quartic r1 lin, ema, stencil, mixed): the guards; 2-slice runs
       of the old and the new train_latent.py (study flags + lean; K = 0 defaults) equal bit for bit; 100-slice runs of
       both with the default assembly equal bit for bit (history files too); --harmonic 4 (lean, --assemble memory,
       --val-every 100) and --harmonic 4 --harmonic-fixed 100-slice runs: config, reference file, slice 0 = the
       reference at t = 5 bit for bit, later slices trained (lr = the reference scale), the fixed run = the reference
       at every t bit for bit, the history members and their bytes; K = 0 and Ising runs with modes; --time-grid
       power:1.5 (fixed: the grid's reference; trained: its learning rates); --harmonic-fixed-slices 2 (the reference
       in slices 0-1, trained from slice 2 on); resume after SIGKILL bit for bit, strict
       resume, --init-from (refused without the same modes, accepted with them, --harmonic-lr-scale free); the default
       sampler on 4 trajectories of both 100-slice runs (finite, manifest), and on a synthetic run old vs new bit for
       bit; early_slices.py on synthetic mnist28_d0123 runs with random, zero and no modes (zero = none exactly,
       random differs)."""
    t17 = time.time()
    sl, tl = sample_latent, train_latent
    T = lm.TEMP
    taus = lm.forward_time_pts()
    have_orig = all(os.path.exists(os.path.join(HARM_ORIG, f"{m}.py"))
                    for m in ("latent_model", "train_latent", "sample_latent"))
    report(f"   modules before --harmonic: {HARM_ORIG} ({'present' if have_orig else 'MISSING: comparisons skipped'})")
    check("17 the modules from before --harmonic are present (HARM_ORIG): the default-path comparisons below need them "
          "(without them those are skipped and this check fails)", 0.0 if have_orig else 1.0, 0.5)
    py, env = sys.executable, dict(os.environ, JAX_PLATFORMS="cpu")
    runs17 = os.path.join(OUT, "b17_runs")
    os.makedirs(runs17)
    st_toy = sl.data_stats("gauss2_8")            # the cache the command-line runs below load (written before they start)
    report(f"   gauss2_8 statistics {st_toy['path']} ({'computed' if st_toy['recomputed'] else 'cached'}, sha256 "
           f"{st_toy['file_sha256'][:16]})")
    FL = ["--problem", "gauss2_8", "--K", "8", "--hidden", "potts", "--quad", "--local", "--select", "ema",
          "--visible", "quartic", "--radius", "1", "--pair-terms", "lin", "--proto-init", "pooled",
          "--visible-impl", "stencil", "--precision", "mixed"]
    tl_cur, sl_cur = os.path.join(HERE, "train_latent.py"), os.path.join(HERE, "sample_latent.py")
    tl_old, sl_old = os.path.join(HARM_ORIG, "train_latent.py"), os.path.join(HARM_ORIG, "sample_latent.py")
    old_root = os.path.join(os.path.dirname(HARM_ORIG), "out", "latent_physical_SGM")   # the old modules' OUT_ROOT

    def tcmd(script, name, *extra, flags=FL):
        return [py, "-u", script, "--name", name, "--out-root", runs17] + list(flags) + list(extra)

    def scmd(script, run_rel):
        return [py, "-u", script, "--run", run_rel, "--n", "4", "--chunk", "4"]

    bg = {}

    def start(tag, cmds, extra_env=None):         # a background chain of commands, stopping at the first failure
        logp = os.path.join(OUT, f"b17_{tag}.log")
        chain = " && ".join(f"{shlex.join(c)} >> {shlex.quote(logp)} 2>&1" for c in cmds)
        bg[tag] = (subprocess.Popen(["bash", "-c", chain], env=dict(env, **(extra_env or {}))), time.time(), logp)

    # ---- background: the 100-slice runs (and their samples), the default sampler old vs new, early_slices -------
    run_a, run_b = os.path.join(OUT, "b17_synth_a"), os.path.join(OUT, "b17_synth_b")
    synthetic_run(run_a, K=4, seed=17)
    os.makedirs(run_b)
    for f in ("config.json", "params_history.npy", "hidden_history.npz", "DONE"):
        os.symlink(os.path.join("..", "b17_synth_a", f), os.path.join(run_b, f))
    harm_dir, fix_dir = os.path.join(runs17, "harm"), os.path.join(runs17, "fix")
    start("harm", [tcmd(tl_cur, "harm", "--step-impl", "lean", "--steps-schedule", "0:0.001", "--val-every", "100",
                        "--assemble", "memory", "--harmonic", "4"),
                   scmd(sl_cur, os.path.relpath(harm_dir, lm.OUT_ROOT))])
    start("fix", [tcmd(tl_cur, "fix", "--steps-schedule", "0:0.001", "--harmonic", "4", "--harmonic-fixed"),
                  scmd(sl_cur, os.path.relpath(fix_dir, lm.OUT_ROOT))])
    start("sdef_new", [scmd(sl_cur, os.path.relpath(run_a, lm.OUT_ROOT))])
    if have_orig:
        start("def100_old", [tcmd(tl_old, "def100_old", "--steps-schedule", "0:0.001")])
        start("def100_new", [tcmd(tl_cur, "def100_new", "--steps-schedule", "0:0.001")])
        start("sdef_old", [scmd(sl_old, os.path.relpath(run_b, old_root))])
    conn28 = lm.connectivity(28, 1)
    cfg_es = {"problem": "mnist28_d0123", "K": 4, "hidden": "potts", "quad": True, "local": True, "visible": "quartic",
              "pair_terms": "lin", "radius": 1, "neighbourhood": "chebyshev", "cls": "none"}
    es_runs = {}
    for lab, modes in (("h", "random"), ("n", None), ("z", "zero")):
        es_runs[lab] = os.path.join(OUT, f"b17_es_{lab}")
        _write_run17(es_runs[lab], _synth_hist17(lm.N, conn28, 4, 3, 1717, modes),
                     dict(cfg_es, harmonic=0 if modes is None else 3))
    es_job = f"b17es_{os.environ.get('SLURM_JOB_ID', 'manual')}"
    start("early_slices", [[py, "-u", os.path.join(HERE, "early_slices.py")]
                           + [f"{lab}={os.path.relpath(p, lm.OUT_ROOT)}" for lab, p in es_runs.items()]
                           + ["--slices", "0,50"]], {"SLURM_JOB_ID": es_job})

    # ---- A. formulas ---------------------------------------------------------------------------------------------
    n8, conn8 = 64, lm.connectivity(8, 1)
    P8 = conn8.shape[0]
    rng = np.random.default_rng(1701)
    V5, c5 = jnp.asarray(0.3 * rng.standard_normal((5, n8))), jnp.asarray(0.2 * rng.standard_normal(5))
    X6 = jnp.asarray(0.5 * rng.standard_normal((6, n8)))
    g_h, lap_h, s_h = lm.harmonic_terms(V5, c5, X6)
    f1 = lambda x: lm.harmonic_free_energy(V5, c5, x[None])[0]           # noqa: E731
    g_ad = jax.vmap(jax.grad(f1))(X6)
    lap_ad = jax.vmap(lambda x: jnp.trace(jax.hessian(f1)(x)))(X6)
    check("17 harmonic_terms: grad F_h vs autodiff of harmonic_free_energy", rel(g_h, g_ad), 1e-13)
    check("17 harmonic_terms: Laplacian of F_h (constant) vs the trace of the autodiff Hessian at 6 states",
          rel(jnp.full(6, lap_h), lap_ad), 1e-13)
    check("17 harmonic_terms: s = V x + c", rel(s_h, X6 @ V5.T + c5), 1e-15)

    vis8 = lm.DenseVisible(conn8, n8)
    th8 = jnp.asarray(0.3 * rng.standard_normal(vis8.n_theta))               # every visible group nonzero (sextic)
    X4 = jnp.asarray(0.5 * rng.standard_normal((4, n8)))

    def F_full(x, p, K, hidden, local):
        f = vis8.energy(p["theta"], x[None])[0]
        if K:
            a = p["W"] @ x + p["b"] + (p["Q"] @ (x * x) if "Q" in p else 0.0)
            f = f - T * (jax.scipy.special.logsumexp(a / T) if hidden == "potts"
                         else jnp.sum(jnp.logaddexp(a / T, -a / T)))
        if local:
            al = p["wl"] * x + p["bl"]
            f = f - T * jnp.sum(jnp.logaddexp(al / T, -al / T))
        return f + lm.harmonic_free_energy(p["V"], p["c"], x[None])[0]

    for K, hidden, quad, local in ((0, "potts", False, False), (3, "potts", True, True), (3, "potts", False, False),
                                   (3, "ising", True, True), (3, "ising", False, True)):
        p = {"theta": th8, "W": jnp.asarray(0.05 * rng.standard_normal((K, n8))),
             "b": jnp.asarray(0.05 * rng.standard_normal(K)), "V": V5, "c": c5}
        if quad:
            p["Q"] = jnp.asarray(0.01 * rng.standard_normal((K, n8)))
        if local:
            p.update(wl=jnp.asarray(0.3 * rng.standard_normal(n8)), bl=jnp.asarray(0.02 * rng.standard_normal(n8)))
        lab = f"K={K} {hidden}{' quad' if quad else ''}{' local' if local else ''}"

        def lref(p_, X_, K=K, hidden=hidden, local=local):
            def one(x):
                g = jax.grad(F_full)(x, p_, K, hidden, local)
                return -jnp.trace(jax.hessian(F_full)(x, p_, K, hidden, local)) / T + jnp.sum(g * g) / (2 * T * T)
            return jnp.mean(jax.lax.map(one, X_))

        lref = jax.jit(lref)
        lf = lm.make_loss(vis8, T, K, hidden, local)
        a_, b_ = float(jax.jit(lf)(p, X4)), float(lref(p, X4))
        check(f"17 loss with 5 modes vs full-autodiff reference, {lab} ({b_:.6g})", abs(a_ - b_) / max(abs(b_), 1.0), 1e-10)
        fo = np.asarray(jax.jit(lm.make_force(vis8, T, K, hidden, local))(p, X4))
        fr = np.asarray(jax.vmap(jax.grad(F_full), in_axes=(0, None, None, None, None))(X4, p, K, hidden, local))
        check(f"17 make_force with modes vs autodiff grad F, {lab}", rel(fo, fr), 1e-12)
        g = jax.jit(jax.grad(lf))(p, X4)
        kd = jr.fold_in(jr.PRNGKey(1702), K + 10 * quad + 100 * local)
        d = {kk: jr.normal(jr.fold_in(kd, i), p[kk].shape) * (jnp.abs(p[kk]) + 1e-3) for i, kk in enumerate(p)}
        fd = (float(lref({kk: p[kk] + 1e-5 * d[kk] for kk in p}, X4))
              - float(lref({kk: p[kk] - 1e-5 * d[kk] for kk in p}, X4))) / 2e-5
        an = float(sum(jnp.sum(g[kk] * d[kk]) for kk in p))
        check(f"17 loss gradient (every group, V and c too) along a random direction vs central difference, {lab} "
              f"({an:.6g})", abs(an - fd) / max(abs(fd), 1e-8), 1e-5)

    # float64 vs --precision mixed with modes, on the trained models' lattice and realistic magnitudes
    prob4 = lm.make_problem("mnist28_d0123", 1)
    data4 = prob4["data"]
    st4 = sl.data_stats("mnist28_d0123", cache_dir=os.path.join(OUT, "b17_data_stats"))
    mu4, lam4, U4 = st4["mu"], st4["lam"], st4["U"]
    P28 = conn28.shape[0]
    zs = lm.dropped_pair_slices(lm.N, P28, ("lin",))
    kept_m = np.ones(4 * lm.N + 3 * P28, bool)
    for s_ in zs + (lm.k6_slice(lm.N),):
        kept_m[s_] = False
    k1, k2, k3, k4, k5, k6, k7 = jr.split(jr.PRNGKey(1703), 7)
    th_q = lm.quartic_initial_theta(lm.paper_theta_layout(P28)[0], lm.N, P28)
    th_q = jnp.where(kept_m, th_q + 0.2 * jr.normal(k1, th_q.shape), 0.0)
    t_q = float(taus[60])
    Vq, cq, _ = lm.harmonic_reference(mu4, lam4, U4, 16, t_q)
    pq = {"theta": th_q, "W": 0.02 * jr.normal(k2, (8, lm.N)), "b": 0.02 * jr.normal(k3, (8,)),
          "Q": 0.005 * jr.normal(k4, (8, lm.N)), "wl": 0.3 * jr.normal(k5, (lm.N,)), "bl": 0.02 * jr.normal(k6, (lm.N,)),
          "V": jnp.asarray(Vq * (1.0 + 0.1 * rng.standard_normal(Vq.shape))), "c": jnp.asarray(cq + 0.01)}
    Xq = sample_forward_process(t_q, 64, T, lm.SIGMA_FORWARD, data4, k7, beta=1)
    vis_d = lm.DenseVisible(conn28)
    vis_st = lm.make_visible(conn28, lm.N, "stencil", ("lin",), sextic=False)
    L_d, G_d = jax.jit(jax.value_and_grad(lm.make_loss(vis_d, T, 8, "potts", True)))(pq, Xq)
    loss_m = lm.make_loss(vis_st, T, 8, "potts", True, compute_dtype=jnp.float32)
    L_m, G_m = jax.jit(jax.value_and_grad(loss_m))(pq, Xq)
    on = lambda g, a: np.asarray(a)[kept_m] if g == "theta" else np.asarray(a)   # noqa: E731
    nrel = {g: float(np.linalg.norm(on(g, G_m[g]) - on(g, G_d[g])) / np.linalg.norm(on(g, G_d[g]))) for g in G_d}
    report(f"   mixed vs float64 with 16 modes (t = {t_q:.3f}): loss {float(L_d):.8g}; gradients (relative norm): "
           f"{', '.join(f'{g} {e:.1e}' for g, e in nrel.items())}; dtypes {sorted({str(np.asarray(v).dtype) for v in G_m.values()})}")
    check(f"17 mixed-precision loss with modes vs float64 (relative; {float(L_d):.6g})",
          abs(float(L_m) - float(L_d)) / abs(float(L_d)), 1e-5)
    check("17 mixed-precision gradient with modes vs float64, worst group incl. V and c (relative norm)",
          max(nrel.values()), 1e-4)
    dots = [ln for ln in jax.jit(jax.value_and_grad(loss_m)).lower(pq, Xq).as_text().splitlines() if "dot_general" in ln]
    n_bad = sum(not ("HIGHEST" in ln and "xf32>" in ln and "xf64>" not in ln) for ln in dots)
    check(f"17 mixed precision with modes: every product float32 at HIGHEST ({len(dots)} dot_general)",
          n_bad + (not dots), 0.5)

    # ---- B. default paths in this process against the modules before the option -------------------------------
    if have_orig:
        lm_o, tl_o, sl_o = _orig17("latent_model"), _orig17("train_latent"), _orig17("sample_latent")
        p_q = {k: pq[k] for k in ("theta", "W", "b", "Q", "wl", "bl")}
        p_k0 = {"theta": th_q, "W": jnp.zeros((0, lm.N)), "b": jnp.zeros((0,))}
        p_is = {"theta": th_q, "W": pq["W"][:3], "b": pq["b"][:3], "wl": pq["wl"], "bl": pq["bl"]}
        vis_st_o = lm_o.make_visible(conn28, lm.N, "stencil", ("lin",), sextic=False)
        cases = [("K=0 dense f64", p_k0, lambda M, v: M.make_loss(v, T, 0), "dense", None),
                 ("Potts 8 quad local, stencil f64", p_q, lambda M, v: M.make_loss(v, T, 8, "potts", True), "stencil", None),
                 ("Potts 8 quad local, stencil mixed", p_q,
                  lambda M, v: M.make_loss(v, T, 8, "potts", True, compute_dtype=jnp.float32), "stencil", None),
                 ("Ising 3 local, dense f64", p_is, lambda M, v: M.make_loss(v, T, 3, "ising", True), "dense", None),
                 ("Potts 8 quad local, sampled S = 8, jackknife", p_q,
                  lambda M, v: M.make_loss(v, T, 8, "potts", True, samples=8, jackknife=True), "stencil", jr.PRNGKey(5)),
                 ("per-sample loss, Potts 8 quad local, dense f64", p_q,
                  lambda M, v: (lambda pp, XX, key=None: M.make_loss_per_sample(v, T, 8, "potts", True)(pp, XX)), "dense",
                  None)]
        bad, lines = [], []
        for lab, pp, mk, impl, key in cases:
            vn, vo = (vis_d, lm_o.DenseVisible(conn28)) if impl == "dense" else (vis_st, vis_st_o)
            fn_n, fn_o = mk(lm, vn), mk(lm_o, vo)
            if lab.startswith("per-sample"):
                vn_, vo_ = jax.jit(fn_n)(pp, Xq), jax.jit(fn_o)(pp, Xq)
                ok = _same17(vn_, vo_)
            else:
                (Ln, Gn), (Lo, Go) = (jax.jit(jax.value_and_grad(f_))(pp, Xq, key) for f_ in (fn_n, fn_o))
                ok = _same17(Ln, Lo) and all(_same17(Gn[g], Go[g]) for g in Gn) and sorted(Gn) == sorted(Go)
            jx_n = str(jax.make_jaxpr(fn_n)(pp, Xq, key) if key is not None else jax.make_jaxpr(fn_n)(pp, Xq))
            jx_o = str(jax.make_jaxpr(fn_o)(pp, Xq, key) if key is not None else jax.make_jaxpr(fn_o)(pp, Xq))
            lines.append(f"{lab}: values {'equal' if ok else 'DIFFER'}, jaxpr {'equal' if jx_n == jx_o else 'DIFFERS'}")
            bad += [lab] if not (ok and jx_n == jx_o) else []
        report("   default loss paths, new vs old module: " + "; ".join(lines))
        check(f"17 default loss paths without modes = the module before --harmonic, bit for bit, same jaxpr "
              f"({len(cases)} cases; failures {bad})", len(bad), 0.5)
        fn_ = [jax.jit(M.make_force(v, T, 8, "potts", True)) for M, v in ((lm, vis_d), (lm_o, lm_o.DenseVisible(conn28)))]
        check("17 default make_force (Potts 8 quad local) = the module before --harmonic, bit for bit",
              0.0 if _same17(fn_[0](p_q, Xq), fn_[1](p_q, Xq)) else 1.0, 0.5)
        # the runner (train_latent.build_runner is unchanged; this guards against an accidental change)
        prob8 = lm.make_problem("gauss2_8", 1)
        vis_8st = lm.make_visible(prob8["conn"], n8, "stencil", ("lin",), sextic=False)
        zs8 = lm.dropped_pair_slices(n8, P8, ("lin",))
        th0 = lm.quartic_initial_theta(lm.paper_theta_layout(P8, n8)[0], n8, P8)
        for s_ in zs8:
            th0 = th0.at[s_].set(0.0)
        p8 = {"theta": th0, "W": jnp.asarray(0.05 * rng.standard_normal((4, n8))),
              "b": jnp.asarray(0.02 * rng.standard_normal(4)), "Q": jnp.zeros((4, n8)),
              "wl": jnp.asarray(0.1 * rng.standard_normal(n8)), "bl": jnp.asarray(0.01 * rng.standard_normal(n8))}
        lrs8 = {"theta": 1e-3, "W": 1e-5, "b": 1e-5, "Q": 1e-5, "wl": 1e-5, "bl": 1e-5}
        loss8 = lm.make_loss(vis_8st, T, 4, "potts", True, compute_dtype=jnp.float32)
        bad = []
        for lean in ({}, {"theta_masks": True, "chunk_batches": True, "track_best": False}):
            outs = []
            for M in (tl, tl_o):
                run_, adam_ = M.build_runner(loss8, prob8["data"], n8, 1000.0, "quartic", P8, zero_slices=zs8, **lean)
                c_ = M.init_carry(p8, adam_, jr.PRNGKey(8))
                for _ in range(2):
                    c_, _l = run_(c_, 4.0, lrs8, 50)
                outs.append(jax.tree_util.tree_leaves(c_))
            bad += [f"lean {bool(lean)}"] if not all(_same17(a, b) for a, b in zip(*outs)) else []
        check(f"17 the runner (default and lean, 2 x 50 steps) = the module before --harmonic, bit for bit (failures "
              f"{bad})", len(bad), 0.5)
        # the sampler's drift and load_params on a synthetic run without modes
        name_a = os.path.relpath(run_a, lm.OUT_ROOT)
        out_n, out_o = sl.load_params(name_a, with_harmonic=True), sl_o.load_params(name_a)
        prov_n, prov_o = dict(out_n[4]), dict(out_o[4])
        ok = (all(_same17(a, b) for a, b in zip(out_n[:3], out_o[:3])) and out_n[3] == out_o[3] and out_n[5] == out_o[5]
              and sorted(prov_n) == sorted(prov_o) and "harmonic" not in prov_n
              and all(_same17(a, b) for a, b in zip(prov_n.pop("local"), prov_o.pop("local")))
              and _same17(prov_n.pop("quad"), prov_o.pop("quad")) and prov_n == prov_o)
        check("17 load_params on a run without modes (with_harmonic=True) = the module before --harmonic (arrays bit for "
              "bit, provenance equal, no 'harmonic' key)", 0.0 if ok else 1.0, 0.5)
        th_a, W_a, b_a, _, pv_a, _ = sl.load_params(name_a)
        lp_a, qp_a = pv_a.pop("local"), pv_a.pop("quad")
        xs = jnp.asarray(0.3 * rng.standard_normal((6, 64)))
        ts = [0.0, 0.01, 1.234567, 2.5, 4.97, 5.0]
        bad = []
        for smp in ("sde", "ode"):
            dn, _ = sl.make_drift(th_a, W_a, b_a, smp, "gauss2_8", "potts", lp_a, qp_a, 1, "chebyshev")
            do, _ = sl_o.make_drift(th_a, W_a, b_a, smp, "gauss2_8", "potts", lp_a, qp_a, 1, "chebyshev")
            dn_j, do_j = jax.jit(jax.vmap(dn, in_axes=(0, 0, None))), jax.jit(jax.vmap(do, in_axes=(0, 0, None)))
            tt = jnp.asarray(ts)
            bad += [smp] if not _same17(dn_j(tt, xs, None), do_j(tt, xs, None)) else []
        check(f"17 default make_drift (SDE and PF-ODE, Potts 4 quad local, 6 times) = the module before --harmonic, bit "
              f"for bit (failures {bad})", len(bad), 0.5)
        del lm_o, tl_o, sl_o

    # ---- C. the closed-form reference -------------------------------------------------------------------------
    Ns = 64
    rng = np.random.default_rng(1704)
    spec = np.concatenate([np.geomspace(30.0, 0.05, 20), np.zeros(Ns - 20)]) + 0.002
    Qm, _ = np.linalg.qr(rng.standard_normal((Ns, Ns)))
    Cs = (Qm * spec) @ Qm.T
    Cs = 0.5 * (Cs + Cs.T)
    mus = 2.0 * rng.standard_normal(Ns)
    lam_s, U_s = np.linalg.eigh(Cs)
    lam_s, U_s = lam_s[::-1].copy(), U_s[:, ::-1].copy()
    vis_iso = lm.DenseVisible(np.zeros((0, 2), dtype=np.int64), Ns)       # no pairs: theta = k_lin | k_duff | k_6 | bias

    def iso_params(n, iso, m, V, c):
        th = np.zeros(4 * n)
        th[:n], th[3 * n:] = T / iso, -T * np.asarray(m) / iso
        return {"theta": jnp.asarray(th), "W": jnp.zeros((0, n)), "b": jnp.zeros((0,)), "V": jnp.asarray(V),
                "c": jnp.asarray(c)}

    force_iso = jax.jit(lm.make_force(vis_iso, T, 0))
    lossps_iso = jax.jit(lm.make_loss_per_sample(vis_iso, T, 0))
    lines, worst = [], {"force": 0.0, "loss": 0.0, "top10": 0.0, "curv": 0.0, "plain": 0.0}
    for t in (5.0, 3.0, 1.0, 0.3, 0.05, 0.0):
        e = np.exp(-t)
        sig2 = T * (1.0 - e * e)
        Sig, m = e * e * Cs + sig2 * np.eye(Ns), e * mus
        V, c, iso = lm.harmonic_reference(mus, lam_s, U_s, Ns, t)
        Pinv = np.linalg.inv(Sig)
        Xg = m + rng.standard_normal((16, Ns)) @ np.linalg.cholesky(Sig).T
        p = iso_params(Ns, iso, m, V, c)
        ex_f = T * (Xg - m) @ Pinv
        ex_l = -np.trace(Pinv) + 0.5 * np.sum(((Xg - m) @ Pinv) ** 2, axis=1)
        worst["force"] = max(worst["force"], rel(force_iso(p, jnp.asarray(Xg)), ex_f))
        worst["loss"] = max(worst["loss"], rel(lossps_iso(p, jnp.asarray(Xg)), ex_l))
        f10 = np.asarray(force_iso(iso_params(Ns, iso, m, V[:10], c[:10]), jnp.asarray(Xg)))
        worst["top10"] = max(worst["top10"], rel(f10 @ U_s[:, :10], ex_f @ U_s[:, :10]))
        ev = np.linalg.eigvalsh(T / iso * np.eye(Ns) - V.T @ V)
        want = T / (e * e * lam_s[0] + sig2)
        worst["curv"] = max(worst["curv"], abs(ev[0] - want) / want + (1.0 if ev[0] <= 0 else 0.0))
        w, _ = lm.harmonic_reference_strengths(lam_s, t, 20)
        if t > 0:                              # w / w_plain - 1 = -(lam_0 / lam_i + b) / (1 + b), b = e^2 lam_0 / sigma_t^2
            w_plain = T * e * e * lam_s[:20] / (sig2 * (e * e * lam_s[:20] + sig2))
            b_ = e * e * lam_s[-1] / sig2
            dev_exact = -(lam_s[-1] / lam_s[:20] + b_) / (1.0 + b_)
            worst["plain"] = max(worst["plain"], float(np.max(np.abs((w / w_plain - 1.0) - dev_exact)
                                                              / np.abs(dev_exact))))
            lines.append(f"t {t:g}: T/iso {T / iso:.4g}, w_1 {w[0]:.4g} (plain {w_plain[0]:.4g}), w_20 {w[-1]:.3g}")
        else:
            lines.append(f"t 0: T/iso {T / iso:.4g} (= T / lam_0), w_1 {w[0]:.4g}, w_20 {w[-1]:.3g} (plain: infinite)")
    report("   synthetic Gaussian (N = 64, lam 30 -> 0.002): " + "; ".join(lines))
    check("17 reference + isotropic visible part = the exact Gaussian force T Sigma_t^-1 (x - m_t), R = N, t = 5 ... 0 "
          "(worst relative)", worst["force"], 1e-10)
    check("17 ... and the exact per-sample loss -tr Sigma^-1 + |Sigma^-1 (x - m)|^2 / 2 (worst relative)", worst["loss"],
          1e-10)
    check("17 ... R = 10: exact on the top-10 principal axes (worst relative)", worst["top10"], 1e-10)
    check("17 ... the reference's precision T/iso I - V^T V is positive, its smallest eigenvalue T / (e^2 lam_1 + "
          "sigma_t^2) (worst relative)", worst["curv"], 1e-9)
    check("17 ... the strengths deviate from the plain formula (lam_0 = 0) by exactly -(lam_0 / lam_i + b) / (1 + b), "
          "b = e^2 lam_0 / sigma_t^2 (worst relative error of the deviation, t > 0)", worst["plain"], 1e-8)

    # stationarity of the expected loss at the reference (exact through 2N sigma points: L is quadratic in x here)
    bad, lines = [], []
    for t, R in ((3.0, Ns), (3.0, 10), (0.3, Ns), (0.3, 10)):
        e = np.exp(-t)
        sig2 = T * (1.0 - e * e)
        ev_, U_ = np.linalg.eigh(e * e * Cs + sig2 * np.eye(Ns))
        Ls = (U_ * np.sqrt(ev_)) @ U_.T
        m = e * mus
        Xs = jnp.asarray(np.concatenate([m + np.sqrt(Ns) * Ls.T, m - np.sqrt(Ns) * Ls.T]))
        V, c, iso = lm.harmonic_reference(mus, lam_s, U_s, R, t)
        base = iso_params(Ns, iso, m, V, c)

        def EL(Vc, base=base, Xs=Xs):
            return jnp.mean(lossps_iso(dict(base, V=Vc[0], c=Vc[1]), Xs))

        ELj, gELj = jax.jit(EL), jax.jit(jax.grad(EL))
        Vc0 = (jnp.asarray(V), jnp.asarray(c))
        g0 = gELj(Vc0)
        r_ = np.random.default_rng(1705 + R)
        dirs = [(jnp.asarray(r_.standard_normal(V.shape) * np.abs(V).max()),
                 jnp.asarray(r_.standard_normal(c.shape) * max(np.abs(c).max(), 0.01))) for _ in range(5)]
        g1 = gELj((Vc0[0] + 1e-2 * dirs[0][0], Vc0[1] + 1e-2 * dirs[0][1]))
        norm = lambda g: float(np.sqrt(sum(float(jnp.sum(a * a)) for a in g)))      # noqa: E731
        ratio = norm(g0) / norm(g1)
        e0 = float(ELj(Vc0))
        rises = [float(ELj((Vc0[0] + 1e-2 * dv, Vc0[1] + 1e-2 * dc))) - e0 for dv, dc in dirs]
        lines.append(f"t {t:g}, R {R}: |grad E L| at the reference / at a 1% perturbation {ratio:.1e}; E L rises by "
                     f"{min(rises):.2e} .. {max(rises):.2e} along 5 random 1% perturbations (E L {e0:.6g})")
        bad += [f"t {t:g} R {R}"] if not (ratio < 1e-7 and min(rises) > 0.0) else []
    report("   expected loss at the reference: " + "; ".join(lines))
    check(f"17 the reference is a stationary point of the expected loss E_(N(m_t, Sigma_t)) L, and E L rises along 5 "
          f"random directions (R = N: the global minimum, where the Fisher divergence is 0; R = 10: no descent along "
          f"these 5, not a proof of a minimum: rotations of the modes among themselves are exactly flat) (t = 3 and "
          f"0.3; failures {bad})", len(bad), 0.5)

    # mnist28_d0123's statistics (this job's own cache, computed above)
    lam0 = max(float(lam4.min()), 0.0)
    report(f"   mnist28_d0123 statistics: |mu|^2 {float(mu4 @ mu4):.5g}, trace C {float(lam4.sum()):.5g}, lam_1..5 "
           f"{np.array2string(lam4[:5], precision=4)}, lam_16 {lam4[15]:.4g}, lam_64 {lam4[63]:.4g}, lam_0 (smallest) "
           f"{lam0:.4g}")
    rows = []
    for t in (5.0, 4.0, 3.0, 2.0, 1.0, 0.5, float(taus[98]), 0.0):
        w, iso = lm.harmonic_reference_strengths(lam4, t, 64)
        sV, sc = lm.harmonic_lr_scales(mu4, lam4, U4, 64, [t])
        e2 = np.exp(-2 * t)
        sig2 = T * (1.0 - e2)
        rp = (f"{w[0] / (T * e2 * lam4[0] / (sig2 * (e2 * lam4[0] + sig2))) - 1.0:+.1e}" if t > 0 else "inf")
        rows.append(f"t {t:.4g}: T/iso {T / iso:.4g}, |v_1|^2 {w[0]:.4g}, |v_16|^2 {w[15]:.4g}, |v_64|^2 {w[63]:.4g}, "
                    f"sum {w.sum():.4g}, rms V {sV[0]:.3g}, rms c {sc[0]:.3g}, v_1 vs plain formula {rp}")
    report("   mnist28_d0123 reference (R = 64):\n      " + "\n      ".join(rows))
    vis_iso4 = lm.DenseVisible(np.zeros((0, 2), dtype=np.int64), lm.N)
    force_iso4 = jax.jit(lm.make_force(vis_iso4, T, 0))
    w784 = {"full": 0.0, "top64": 0.0, "curv": 0.0}
    C4 = (U4 * lam4) @ U4.T
    for t in (5.0, 2.5, 1.0, float(taus[98]), 0.0):
        e = np.exp(-t)
        sig2 = T * (1.0 - e * e)
        m = e * mu4
        Sig = e * e * C4 + sig2 * np.eye(lm.N)
        Pinv = (U4 / (e * e * lam4 + sig2)) @ U4.T
        Xg = m + (rng.standard_normal((8, lm.N)) * np.sqrt(e * e * lam4 + sig2)) @ U4.T
        ex_f = T * (Xg - m) @ Pinv
        V, c, iso = lm.harmonic_reference(mu4, lam4, U4, lm.N, t)
        w784["full"] = max(w784["full"], rel(force_iso4(iso_params(lm.N, iso, m, V, c), jnp.asarray(Xg)), ex_f))
        f64_ = np.asarray(force_iso4(iso_params(lm.N, iso, m, V[:64], c[:64]), jnp.asarray(Xg)))
        w784["top64"] = max(w784["top64"], rel(f64_ @ U4[:, :64], ex_f @ U4[:, :64]))
        for R in (64, lm.N):
            ev = np.linalg.eigvalsh(T / iso * np.eye(lm.N) - V[:R].T @ V[:R])
            want = T / (e * e * lam4[0] + sig2)
            w784["curv"] = max(w784["curv"], abs(ev[0] - want) / want + (1.0 if ev[0] <= 0 else 0.0))
        del Sig
    check("17 mnist28_d0123: reference (R = 784) + isotropic part = the exact Gaussian force at t = 5, 2.5, 1, 0.05, 0 "
          "(worst relative)", w784["full"], 1e-9)
    check("17 mnist28_d0123: R = 64 exact on the top-64 principal axes (worst relative)", w784["top64"], 1e-9)
    check("17 mnist28_d0123: the reference precision is positive with its smallest eigenvalue T / (e^2 lam_1 + "
          "sigma_t^2) (R = 64 and 784; worst relative)", w784["curv"], 1e-8)
    bad = []
    for R in (16, 64, lm.N):
        sV, sc = lm.harmonic_lr_scales(mu4, lam4, U4, R, taus)
        top_ok = all(lm.harmonic_reference_strengths(lam4, t, R)[0].max()
                     < T / lm.harmonic_reference_strengths(lam4, t, R)[1] for t in taus)
        bad += [R] if not (np.all(np.isfinite(sV)) and np.all(sV > 0) and np.all(sc >= T) and top_ok) else []
    check(f"17 mnist28_d0123: at all 100 slice times every strength is below T / iso (a confining reference) and the "
          f"learning-rate scales are finite and positive (R = 16, 64, 784; failures {bad})", len(bad), 0.5)
    # the quartic floor confines modes that outgrow the springs (1.5 x the reference at t = 1, along PC 1)
    t = 1.0
    e = np.exp(-t)
    V, c, iso = lm.harmonic_reference(mu4, lam4, U4, 64, t)
    th_c = np.zeros(4 * lm.N)
    th_c[:lm.N], th_c[lm.N:2 * lm.N], th_c[3 * lm.N:] = T / iso, lm.K4_FLOOR, -T * e * mu4 / iso
    r_grid = np.concatenate([np.linspace(0.0, 10.0, 201), np.geomspace(10.0, 1e4, 400)[1:]])
    xr = e * mu4 + r_grid[:, None] * U4[:, 0]
    Fr = np.asarray(vis_iso4.energy(jnp.asarray(th_c), jnp.asarray(xr))
                    + lm.harmonic_free_energy(jnp.asarray(1.5 * V), jnp.asarray(1.5 * c), jnp.asarray(xr)))
    i_min = int(np.argmin(Fr))
    quad_coef = T / iso - 2.25 * float(np.sum(V[0] ** 2))
    report(f"   modes 1.5 x the reference at t = 1: along PC 1 the quadratic coefficient is {quad_coef:.4g} (< 0: the modes "
           f"exceed the springs); with k_duff = {lm.K4_FLOOR} the energy has its minimum at |x - m_t| = "
           f"{r_grid[i_min]:.4g} and rises to {Fr[-1] - Fr[i_min]:.3g} above it at |x - m_t| = {r_grid[-1]:.0f}")
    check("17 the quartic floor confines modes that exceed the springs (1.5 x the reference along PC 1: F has an "
          "interior minimum and grows without bound)", 0.0 if (quad_coef < 0 and 0 < i_min < r_grid.size - 1
                                                              and Fr[-1] > Fr[i_min] + 1e6) else 1.0, 0.5)
    # the diagnostics on the reference itself, and the visible Hessian at 0
    hs = lm.harmonic_unit_stats(V, c, V, c, U4[:, :64], iso, T, theta=th_c, conn=np.zeros((0, 2), dtype=np.int64))
    want_add = float(e * e * np.sum(lam4[:64] - lam0))
    hs2 = lm.harmonic_unit_stats(1.5 * V, 1.5 * c, V, c, U4[:, :64], iso, T)
    errs = [abs(hs["harm_cos_V_Vref"] - 1.0), abs(hs["harm_cos_M_Mref"] - 1.0), abs(hs["harm_in_ref_subspace"] - 1.0),
            abs(hs["harm_added_var"] - want_add) / want_add, abs(hs["harm_added_var_ref"] - want_add) / want_add,
            abs(hs["harm_V_norm"] - hs["harm_Vref_norm"]), float(hs["harm_n_beyond_iso"]),
            abs(hs["harm_vis_curv_min"] - T / iso) / (T / iso),
            abs(hs["harm_curv_min"] - T / (e * e * lam4[0] + T * (1 - e * e))) / (T / iso)]
    report(f"   harmonic_unit_stats(reference, reference) at t = 1: {json.dumps(hs)}; with 1.5 x the modes: beyond T/iso "
           f"{hs2['harm_n_beyond_iso']}, top added variance {hs2['harm_added_var_top']}")
    check("17 harmonic_unit_stats on the reference itself: cosines and subspace share 1, added variance = e^2 sum "
          "(lam_i - lam_0), none beyond T/iso, curvatures T/iso and T/(e^2 lam_1 + sigma^2) (worst error)", max(errs),
          1e-9)
    check("17 harmonic_unit_stats with 1.5 x the modes: modes beyond T/iso counted, the top one's added variance None",
          0.0 if hs2["harm_n_beyond_iso"] > 0 and hs2["harm_added_var_top"] is None else 1.0, 0.5)
    th_r = jnp.asarray(0.3 * rng.standard_normal(vis8.n_theta))
    H_ad = np.asarray(jax.hessian(lambda x: vis8.energy(th_r, x[None])[0])(jnp.zeros(n8)))
    check("17 visible_hessian_at_zero vs the autodiff Hessian of the visible energy at x = 0 (every pair group)",
          rel(lm.visible_hessian_at_zero(th_r, conn8, n8), H_ad), 1e-12)

    # ---- D. the sampler and the runner with modes ---------------------------------------------------------------
    h8 = _synth_hist17(n8, conn8, 4, 3, 1706)
    harm8 = (h8["V"], h8["c"])
    efn = setup_energy_fn(jnp.asarray(conn8), lm.paper_theta_layout(P8, n8)[1])

    def F_samp(x, q):
        a = q["W"] @ x + q["Q"] @ (x * x) + q["b"]
        al = q["wl"] * x + q["bl"]
        return (efn(x, q["theta"]) - T * jax.scipy.special.logsumexp(a / T) - T * jnp.sum(jnp.logaddexp(al / T, -al / T))
                - 0.5 * jnp.sum((q["V"] @ x + q["c"]) ** 2))

    worst = 0.0
    for smp, s_ in (("sde", 2.0), ("ode", 1.0)):
        dr, _ = sl.make_drift(h8["theta"], h8["W"], h8["b"], smp, "gauss2_8", "potts", (h8["wl"], h8["bl"]), h8["Q"], 1,
                              "chebyshev", harmonic=harm8)
        for t_rev in (0.0, 0.7777, 2.5, 4.9999):
            tau = lm.T_FORWARD - t_rev
            k = min(int(np.searchsorted(-taus, -tau, side="right")) - 1, lm.N_TIME_STEPS - 2)
            lw = (tau - taus[k + 1]) / (taus[k] - taus[k + 1])
            q = {g: jnp.asarray(h8[g][k + 1] + lw * (h8[g][k] - h8[g][k + 1]))
                 for g in ("theta", "W", "b", "Q", "wl", "bl", "V", "c")}
            x = jnp.asarray(0.4 * rng.standard_normal(n8))
            ref = x - s_ * jax.grad(F_samp)(x, q)
            worst = max(worst, rel(dr(t_rev, x, ()), ref))
    check("17 make_drift with modes (Potts 4 quad local + 3 modes; SDE and PF-ODE, 4 times between knots) vs x - s grad F "
          "by autodiff with interpolated parameters (worst relative)", worst, 1e-10)
    name_h = os.path.relpath(os.path.join(OUT, "b17_synth_harm"), lm.OUT_ROOT)
    _write_run17(os.path.join(OUT, "b17_synth_harm"), dict(h8, theta=h8["theta"]),
                 {"problem": "gauss2_8", "K": 4, "hidden": "potts", "quad": True, "local": True, "visible": "quartic",
                  "pair_terms": "lin", "radius": 1, "neighbourhood": "chebyshev", "cls": "none", "harmonic": 3})
    try:
        sl.load_params(name_h)
        refused = ""
    except RuntimeError as err:
        refused = str(err)
    out_h = sl.load_params(name_h, with_harmonic=True)
    ok = ("harmonic hidden modes" in refused and "harmonic" in out_h[4] and _same17(out_h[4]["harmonic"][0], h8["V"])
          and _same17(out_h[4]["harmonic"][1], h8["c"]))
    code, err_ = _main17(sl, ["--run", name_h, "--fast", "--n", "4", "--chunk", "4"])
    fast_ok = isinstance(code, str) and "--fast does not support runs with harmonic modes" in code
    report(f"   load_params refusal: {refused[:160]}...; --fast: {str(code)[:160]}")
    check("17 load_params refuses a run with modes unless with_harmonic=True (then prov['harmonic'] = V, c bit for "
          "bit); sample_latent --fast refuses it", 0.0 if ok and fast_ok else 1.0, 0.5)
    # the runner with modes: lr > 0 trains them; lr 0 (--harmonic-fixed, the freeze slices) leaves them bit for bit
    prob8 = lm.make_problem("gauss2_8", 1)
    vis_8st = lm.make_visible(prob8["conn"], n8, "stencil", ("lin",), sextic=False)
    zs8 = lm.dropped_pair_slices(n8, P8, ("lin",))
    th0 = lm.quartic_initial_theta(lm.paper_theta_layout(P8, n8)[0], n8, P8)
    for s_ in zs8:
        th0 = th0.at[s_].set(0.0)
    Vt, ct, _ = lm.harmonic_reference(st_toy["mu"], st_toy["lam"], st_toy["U"], 4, 3.0)
    p8h = {"theta": th0, "W": jnp.asarray(0.05 * rng.standard_normal((4, n8))), "b": jnp.zeros(4), "Q": jnp.zeros((4, n8)),
           "wl": jnp.zeros(n8), "bl": jnp.zeros(n8), "V": jnp.asarray(Vt), "c": jnp.asarray(ct)}
    loss8h = lm.make_loss(vis_8st, T, 4, "potts", True, compute_dtype=jnp.float32)
    bad = []
    for lean in ({}, {"theta_masks": True, "chunk_batches": True, "track_best": False}):
        run_, adam_ = tl.build_runner(loss8h, prob8["data"], n8, 1000.0, "quartic", P8, zero_slices=zs8, **lean)
        lr_on = {"theta": 1e-3, "W": 1e-5, "b": 1e-5, "Q": 1e-5, "wl": 1e-5, "bl": 1e-5, "V": 1e-4, "c": 1e-4}
        c_on, _ = run_(tl.init_carry(p8h, adam_, jr.PRNGKey(9)), 3.0, lr_on, 100)
        c_off, _ = run_(tl.init_carry(p8h, adam_, jr.PRNGKey(9)), 3.0, dict(lr_on, V=0.0, c=0.0), 100)
        moved = float(np.max(np.abs(np.asarray(c_on[0]["V"]) - Vt)))
        fin = all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(c_on))
        kept = all(_same17(c_off[i]["V"], Vt) and _same17(c_off[i]["c"], ct) for i in (0, 4, 7))
        bad += [f"lean {bool(lean)}: moved {moved:.1e}, finite {fin}, kept {kept}"] if not (moved > 0 and fin and kept) else []
    check(f"17 the runner with modes (default and lean): lr > 0 moves V, finite; lr 0 keeps V and c bit for bit in the "
          f"iterate, best and EMA (failures {bad})", len(bad), 0.5)

    # ---- E. command lines: guards (in this process, refused before any file is written) ------------------------
    guard_root = os.path.join(OUT, "b17_guards")
    base = ["--name", "g", "--out-root", guard_root, "--problem", "gauss2_8"]
    pk = ["--K", "4", "--hidden", "potts"]
    cases = [(["--harmonic", "-1"], "--harmonic -1"), (["--harmonic", "65"], "--harmonic 65"),
             (["--harmonic-fixed"], "--harmonic-fixed needs"), (["--harmonic", "4", "--harmonic-lr-scale", "0"], "must be > 0"),
             (["--harmonic", "4", "--harmonic-lr-scale", "nan"], "must be > 0"),
             (pk + ["--harmonic", "4", "--hidden-samples", "4"], "--harmonic needs exact"),
             (pk + ["--harmonic", "4", "--hidden-rule", "cd1"], "--harmonic needs exact"),
             (pk + ["--harmonic", "4", "--hidden-norm", "group"], "--harmonic needs exact"),
             (["--harmonic-fixed-slices", "2"], "--harmonic-fixed-slices 2"),
             (["--harmonic", "4", "--harmonic-fixed", "--harmonic-fixed-slices", "2"], "--harmonic-fixed-slices 2"),
             (["--harmonic", "4", "--harmonic-fixed-slices", "100"], "--harmonic-fixed-slices 100"),
             (["--harmonic", "4", "--harmonic-fixed-slices", "-1"], "--harmonic-fixed-slices -1")]
    bad = []
    for argv, text in cases:
        code, _e = _main17(tl, base + argv)
        if not (isinstance(code, str) and code.startswith("ValueError") and text in code):
            bad.append((" ".join(argv), code))
    check(f"17 train_latent guards: --harmonic out of 0..64, --harmonic-fixed alone, --harmonic-lr-scale 0 / nan, "
          f"--harmonic with --hidden-samples / cd1 / --hidden-norm group, --harmonic-fixed-slices without --harmonic, "
          f"with --harmonic-fixed, 100, -1: refused before writing (failures {bad}; no folder: "
          f"{not os.path.exists(guard_root)})", len(bad) + int(os.path.exists(guard_root)), 0.5)

    def train17(script, name, *extra, flags=FL, kill_after=None):
        """train_latent.py (script) in a subprocess; with kill_after, SIGKILL once that slice's file exists. Returns
        (exit code, the log text of this call only)."""
        cmd = tcmd(script, name, *extra, flags=flags)
        logp = os.path.join(OUT, f"b17_train_{name}.log")
        pos = os.path.getsize(logp) if os.path.exists(logp) else 0
        with open(logp, "a") as fo:
            fo.write("$ " + shlex.join(cmd[2:]) + "\n")
            fo.flush()
            p_ = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, env=env)
            if kill_after is not None:
                target = os.path.join(runs17, name, "slices", f"slice_{kill_after:03d}.npz")
                while p_.poll() is None and not os.path.exists(target):
                    time.sleep(0.05)
                if p_.poll() is None:
                    p_.kill()
            rc = p_.wait()
        with open(logp) as f:
            f.seek(pos)
            return rc, f.read()

    def records(name):
        with open(os.path.join(runs17, name, "train_log.jsonl")) as f:
            return {int(r["slice"]): r for r in (json.loads(ln) for ln in f if ln.strip())}

    two = ["--max-slices", "2", "--steps-scale", "0.01"]
    if have_orig:                              # the default path, old vs new train_latent.py, bit for bit
        bad = []
        for tag, flags, extra in (("study", FL, ["--step-impl", "lean"] + two), ("k0", ["--problem", "gauss2_8"], two)):
            rco, _ = train17(tl_old, f"d_{tag}_old", *extra, flags=flags)
            rcn, _ = train17(tl_cur, f"d_{tag}_new", *extra, flags=flags)
            diff = _slices_differ17(os.path.join(runs17, f"d_{tag}_old"), os.path.join(runs17, f"d_{tag}_new"), (0, 1))
            with open(os.path.join(runs17, f"d_{tag}_old", "config.json")) as f:
                co = json.load(f)
            with open(os.path.join(runs17, f"d_{tag}_new", "config.json")) as f:
                cn = json.load(f)
            new_keys = {k: cn[k] for k in set(cn) - set(co)}
            cdiff = sorted(k for k in set(co) & set(cn) if k not in ("name", "sha256", "started") and co[k] != cn[k])
            ok = (rco == 0 and rcn == 0 and not diff and not cdiff and set(co) <= set(cn)
                  and new_keys == {"harmonic": 0, "harmonic_fixed": False, "harmonic_lr_scale": 1.0,
                                   "harmonic_fixed_slices": 0, "harmonic_ref": "full", "harmonic_lr_tmin": 0.0,
                                   **PATCH_NEW_KEYS})
            report(f"   default path {tag}: exit codes {rco}/{rcn}; slice differences {diff}; config differences {cdiff}, "
                   f"new keys {new_keys}")
            bad += [tag] if not ok else []
        check(f"17 default training through the command line (study flags + lean; K = 0 defaults; 2 slices) = the "
              f"module before --harmonic: slice files bit for bit, config.json the same but for the six new keys (and "
              f"the four of --patch) at their defaults (failures {bad})", len(bad), 0.5)

    # modes with K = 0 and with Ising units (2 slices each)
    bad = []
    for tag, flags in (("k0h", ["--problem", "gauss2_8"]), ("isingh", ["--problem", "gauss2_8", "--K", "3",
                                                                      "--hidden", "ising"])):
        rc, txt = train17(tl_cur, tag, "--harmonic", "2", *two, flags=flags)
        ok = rc == 0
        if ok:
            with np.load(os.path.join(runs17, tag, "slices", "slice_001.npz")) as z1, \
                    np.load(os.path.join(runs17, tag, "slices", "slice_000.npz")) as z0:
                V0r, c0r, _ = lm.harmonic_reference(st_toy["mu"], st_toy["lam"], st_toy["U"], 2, float(taus[0]))
                ok = (_same17(z0["V"], V0r) and _same17(z0["c"], c0r) and z1["V"].shape == (2, n8)
                      and not _same17(z1["V"], z0["V"]) and np.all(np.isfinite(z1["V"])))
        bad += [f"{tag} (exit {rc})"] if not ok else []
    check(f"17 --harmonic 2 with K = 0 (sextic, dense, f64) and with 3 Ising units: slice 0 = the reference at t = 5 bit "
          f"for bit, slice 1 trained (failures {bad})", len(bad), 0.5)

    # resume after SIGKILL, strict resume, --init-from
    six = ["--harmonic", "4", "--max-slices", "6", "--steps-scale", "0.01"]
    rc_f, _ = train17(tl_cur, "res_full", *six)
    rc_k, _ = train17(tl_cur, "res_kill", *six, kill_after=2)
    n_k = sum(os.path.exists(os.path.join(runs17, "res_kill", "slices", f"slice_{k:03d}.npz")) for k in range(6))
    rc_r, txt_r = train17(tl_cur, "res_kill", *six)
    diff = _slices_differ17(os.path.join(runs17, "res_full"), os.path.join(runs17, "res_kill"), range(6))
    check(f"17 --harmonic 4 resumed after SIGKILL (exit {rc_k} with {n_k} slices; resumed: exit {rc_r}) = an "
          f"uninterrupted run, slices 0-5 bit for bit (exit {rc_f}; differences {diff})",
          0.0 if rc_f == 0 and rc_k != 0 and rc_r == 0 and n_k < 6 and not diff and "resuming at slice" in txt_r else 1.0,
          0.5)
    bad = []
    for extra in (["--harmonic", "8"], ["--harmonic", "4", "--harmonic-fixed"], ["--harmonic", "4",
                                                                                   "--harmonic-lr-scale", "2"], []):
        rc, txt = train17(tl_cur, "res_full", *(extra + ["--max-slices", "6", "--steps-scale", "0.01"]))
        bad += [" ".join(extra) or "no --harmonic"] if not (rc != 0 and "different config" in txt) else []
    check(f"17 resuming a --harmonic 4 run with other harmonic settings (8 modes, fixed, lr scale 2, none) is refused "
          f"(failures {bad})", len(bad), 0.5)
    bad = []
    src = os.path.relpath(os.path.join(runs17, "res_full"), runs17)
    if have_orig:
        rc, txt = train17(tl_cur, "if_nomodes", "--init-from", os.path.relpath(os.path.join(runs17, "d_study_new"), runs17),
                          "--init-slices", "2", "--harmonic", "4", "--step-impl", "lean", "--max-slices", "3",
                          "--steps-scale", "0.01")
        bad += ["source without modes accepted"] if not (rc != 0 and "different model config" in txt) else []
    rc, txt = train17(tl_cur, "if_same", "--init-from", src, "--init-slices", "3", *six)
    links = [os.path.islink(os.path.join(runs17, "if_same", "slices", f"slice_{k:03d}.npz")) for k in range(6)]
    diff = _slices_differ17(os.path.join(runs17, "res_full"), os.path.join(runs17, "if_same"), range(3, 6))
    bad += [f"same modes: exit {rc}, links {links}, differences {diff}"] if not (
        rc == 0 and links == [True] * 3 + [False] * 3 and not diff) else []
    rc, txt = train17(tl_cur, "if_none", "--init-from", src, "--init-slices", "3", "--max-slices", "6", "--steps-scale",
                      "0.01")
    bad += ["no modes from a source with modes accepted"] if not (rc != 0 and "different model config" in txt) else []
    rc, txt = train17(tl_cur, "if_fixed", "--init-from", src, "--init-slices", "3", "--harmonic-fixed", *six)
    bad += ["fixed from trained accepted"] if not (rc != 0 and "different model config" in txt) else []
    rc, txt = train17(tl_cur, "if_lr2", "--init-from", src, "--init-slices", "3", "--harmonic", "4",
                      "--harmonic-lr-scale", "2", "--max-slices", "4", "--steps-scale", "0.01")
    ok = rc == 0
    if ok:
        ra, rb = records("res_full")[3], records("if_lr2")[3]
        ok = abs(rb["lr_V"] / ra["lr_V"] - 2.0) < 1e-12 and abs(rb["lr_c"] / ra["lr_c"] - 2.0) < 1e-12
    bad += [f"--harmonic-lr-scale 2: exit {rc}"] if not ok else []
    check(f"17 --init-from: a source without modes is refused for --harmonic 4, a source with 4 modes continues bit for "
          f"bit (slices 0-2 linked, 3-5 = the source's), without --harmonic or with --harmonic-fixed refused, "
          f"--harmonic-lr-scale 2 accepted with doubled learning rates (failures {bad})", len(bad), 0.5)
    # --time-grid power:1.5: the fixed modes are the grid's reference, the trained ones get the grid's learning rates
    g_t = lm.time_grid_pts("power:1.5")
    bad = []
    rc, _ = train17(tl_cur, "grid_fix", "--time-grid", "power:1.5", "--harmonic", "3", "--harmonic-fixed",
                    "--max-slices", "3", "--steps-scale", "0.01")
    for k in range(3):
        if rc != 0:
            break
        Vr, cr, _ = lm.harmonic_reference(st_toy["mu"], st_toy["lam"], st_toy["U"], 3, float(g_t[k]))
        with np.load(os.path.join(runs17, "grid_fix", "slices", f"slice_{k:03d}.npz")) as z:
            bad += [f"fixed slice {k}"] if not (_same17(z["V"], Vr) and _same17(z["c"], cr)) else []
    bad += [f"fixed exit {rc}"] if rc != 0 else []
    rc, _ = train17(tl_cur, "grid_tr", "--time-grid", "power:1.5", "--harmonic", "3", "--max-slices", "3",
                    "--steps-scale", "0.01")
    if rc == 0:
        sVg, scg = lm.harmonic_lr_scales(st_toy["mu"], st_toy["lam"], st_toy["U"], 3, g_t)
        rr = records("grid_tr")
        for k in (1, 2):
            want_v, want_c = 5e-4 * sVg[k], 5e-4 * scg[k]
            bad += [f"lr slice {k}"] if not (abs(rr[k]["lr_V"] / want_v - 1) < 1e-12 and abs(rr[k]["lr_c"] / want_c - 1)
                                             < 1e-12 and abs(rr[k]["t"] - g_t[k]) < 1e-15) else []
        bad += ["slice 0 lr not 0"] if rr[0]["lr_V"] != 0.0 else []
    else:
        bad.append(f"trained exit {rc}")
    check(f"17 --harmonic with --time-grid power:1.5: fixed modes = the reference at the grid's t_k bit for bit; trained "
          f"modes' learning rates = 5e-4 x the reference scale at t_k, 0 in slice 0 (failures {bad})", len(bad), 0.5)
    # --harmonic-fixed-slices 2: the reference in slices 0-1, trained from slice 2 on (warm-started)
    rc, _ = train17(tl_cur, "fs2", "--harmonic", "3", "--harmonic-fixed-slices", "2", "--max-slices", "4",
                    "--steps-scale", "0.01")
    bad = [] if rc == 0 else [f"exit {rc}"]
    if rc == 0:
        rr = records("fs2")
        sV3, _sc3 = lm.harmonic_lr_scales(st_toy["mu"], st_toy["lam"], st_toy["U"], 3, taus)
        for k in range(4):
            Vr, cr, _ = lm.harmonic_reference(st_toy["mu"], st_toy["lam"], st_toy["U"], 3, float(taus[k]))
            with np.load(os.path.join(runs17, "fs2", "slices", f"slice_{k:03d}.npz")) as z:
                at_ref = _same17(z["V"], Vr) and _same17(z["c"], cr)
            if k < 2:
                bad += [f"slice {k} not the reference"] if not (at_ref and rr[k]["lr_V"] == 0.0) else []
            else:
                bad += [f"slice {k} not trained"] if (at_ref or not abs(rr[k]["lr_V"] / (5e-4 * sV3[k]) - 1) < 1e-12) \
                    else []
        with open(os.path.join(runs17, "fs2", "config.json")) as f:
            bad += ["config"] if json.load(f).get("harmonic_fixed_slices") != 2 else []
    check(f"17 --harmonic 3 --harmonic-fixed-slices 2: slices 0-1 = the reference bit for bit (lr 0), slices 2-3 "
          f"trained with lr = 5e-4 x the reference scale; config records 2 (failures {bad})", len(bad), 0.5)

    # ---- the background runs --------------------------------------------------------------------------------------
    rcs = {}
    for tag, (p_, t_p, logp) in bg.items():
        try:
            rcs[tag] = p_.wait(timeout=max(60.0, 1680.0 - (time.time() - t17)))
        except subprocess.TimeoutExpired:
            p_.kill()
            rcs[tag] = "timeout"
        report(f"   background {tag}: exit code {rcs[tag]}, {time.time() - t_p:.0f} s after its start ({logp})")
    if have_orig:                              # 100 slices with the default assembly: old vs new, history files too
        da, db = os.path.join(runs17, "def100_old"), os.path.join(runs17, "def100_new")
        diff = _slices_differ17(da, db, range(lm.N_TIME_STEPS)) if rcs["def100_old"] == 0 and rcs["def100_new"] == 0 \
            else ["exit codes"]
        hdiff = []
        for f_ in ("params_history.npy", "hidden_history.npz"):
            if not (os.path.exists(os.path.join(da, f_)) and os.path.exists(os.path.join(db, f_))):
                hdiff.append(f"{f_} missing")
        if not hdiff:
            hdiff += [] if _same17(np.load(os.path.join(da, "params_history.npy")),
                                   np.load(os.path.join(db, "params_history.npy"))) else ["params_history"]
            with np.load(os.path.join(da, "hidden_history.npz")) as za, np.load(os.path.join(db, "hidden_history.npz")) as zb:
                hdiff += [] if za.files == zb.files and all(_same17(za[m], zb[m]) for m in za.files) else \
                    [f"hidden_history {za.files} / {zb.files}"]
        check(f"17 100-slice default runs with the default assembly, old vs new train_latent.py: all slice files and "
              f"both history files (members, order, bytes) bit for bit (differences {diff[:5]} {hdiff})",
              len(diff) + len(hdiff), 0.5)
    # the --harmonic 4 runs (trained: lean, --assemble memory, --val-every 100; fixed: default assembly)
    for tag, d_, fixed in (("harm", harm_dir, False), ("fix", fix_dir, True)):
        bad = []
        if not os.path.exists(os.path.join(d_, "DONE")):
            check(f"17 --harmonic 4{' --harmonic-fixed' if fixed else ''}: 100-slice run complete", 1.0, 0.5)
            continue
        with open(os.path.join(d_, "config.json")) as f:
            cfg = json.load(f)
        bad += ["config"] if (cfg.get("harmonic"), cfg.get("harmonic_fixed"), cfg.get("harmonic_lr_scale")) != (
            4, fixed, 1.0) else []
        with open(os.path.join(d_, "harmonic_reference.json")) as f:
            hr = json.load(f)
        bad += ["reference file"] if (hr.get("R") != 4 or hr.get("stats_sha256") != st_toy["file_sha256"]
                                      or len(hr.get("lr_scale_V", [])) != 100) else []
        rr = records(tag)
        sVt, sct = lm.harmonic_lr_scales(st_toy["mu"], st_toy["lam"], st_toy["U"], 4, taus)
        Vs_, cs_ = [], []
        n_moved = 0
        for k in range(lm.N_TIME_STEPS):
            Vr, cr, _ = lm.harmonic_reference(st_toy["mu"], st_toy["lam"], st_toy["U"], 4, float(taus[k]))
            with np.load(os.path.join(d_, "slices", f"slice_{k:03d}.npz")) as z:
                Vk, ck = z["V"], z["c"]
            Vs_.append(Vk)
            cs_.append(ck)
            at_ref = _same17(Vk, Vr) and _same17(ck, cr)
            if fixed or k == 0:
                bad += [f"slice {k} not the reference"] if not at_ref else []
                bad += [f"slice {k} lr"] if rr[k]["lr_V"] != 0.0 or rr[k]["lr_c"] != 0.0 else []
            else:
                n_moved += int(not at_ref)
                bad += [f"slice {k} lr"] if not (abs(rr[k]["lr_V"] / (5e-4 * sVt[k]) - 1) < 1e-12
                                                  and abs(rr[k]["lr_c"] / (5e-4 * sct[k]) - 1) < 1e-12) else []
            bad += [f"slice {k} not finite"] if not (np.all(np.isfinite(Vk)) and np.all(np.isfinite(ck))) else []
            bad += [f"slice {k} diagnostics"] if not all(f"harm_{q}" in rr[k] for q in (
                "V_norm", "cos_V_Vref", "cos_M_Mref", "added_var", "added_var_ref", "curv_min", "vis_curv_min")) else []
        if not fixed:
            bad += [f"only {n_moved} of 99 trained slices moved"] if n_moved < 99 else []
        with np.load(os.path.join(d_, "hidden_history.npz")) as zh:
            members = list(zh.files)
            hist_ok = (members == ["W", "b", "t", "wl", "bl", "Q", "V", "c"] and _same17(zh["V"], np.stack(Vs_))
                       and _same17(zh["c"], np.stack(cs_)))
        bad += [f"history members {members}"] if not hist_ok else []
        with open(os.path.join(d_, "train_log.txt")) as f:
            bad += ["log line"] if "harmonic modes: 4" not in f.read() else []
        ks_ = (0, 1, 5, 10, 20, 30, 50, 80, 99)
        report(f"   {tag}: per-slice diagnostics (slice: |V| / ref, cos V, cos V^T V, added variance / ref, beyond T/iso, "
               f"curvature springs / with modes): " + "; ".join(
                   f"{k}: {rr[k]['harm_V_norm']:.3g}/{rr[k]['harm_Vref_norm']:.3g}, {rr[k]['harm_cos_V_Vref']:.3f}, "
                   f"{rr[k]['harm_cos_M_Mref']:.3f}, {rr[k]['harm_added_var']:.3g}/{rr[k]['harm_added_var_ref']:.3g}, "
                   f"{rr[k]['harm_n_beyond_iso']}, {rr[k]['harm_vis_curv_min']:.3g}/{rr[k]['harm_curv_min']:.3g}"
                   for k in ks_))
        check(f"17 --harmonic 4{' --harmonic-fixed' if fixed else ''} through the command line (100 slices): config, "
              f"reference file, " + ("every slice = the reference at its t bit for bit, lr 0" if fixed else
                                    "slice 0 = the reference at t = 5 bit for bit (lr 0), slices 1-99 trained with lr = "
                                    "5e-4 x the reference scale") + f", diagnostics, hidden_history members W b t wl bl Q "
              f"V c = the slice files (failures {bad[:6]})", len(bad), 0.5)
        xs_, ms_ = None, None
        p_s = os.path.join(d_, "samples", "sde_base_n4")
        if os.path.exists(p_s + ".npy") and os.path.exists(p_s + ".json"):
            xs_ = np.load(p_s + ".npy")
            with open(p_s + ".json") as f:
                ms_ = json.load(f)
        ok = (rcs[tag] == 0 and xs_ is not None and np.all(np.isfinite(xs_)) and ms_.get("harmonic_modes") == 4
              and ms_.get("sampler") == "sde")
        report(f"   {tag}: samples {None if xs_ is None else np.array2string(np.abs(xs_).max(axis=1), precision=3)} (max "
               f"|x| per trajectory), manifest harmonic_modes {None if ms_ is None else ms_.get('harmonic_modes')}")
        check(f"17 the default sampler on the {tag} run (4 trajectories): exit code 0, finite samples, manifest "
              f"harmonic_modes 4", 0.0 if ok else 1.0, 0.5)
    if have_orig:                              # the default sampler, old vs new module, on a synthetic run
        def load_out(run_dir):
            p_ = os.path.join(run_dir, "samples", "sde_base_n4")
            if not (os.path.exists(p_ + ".npy") and os.path.exists(p_ + ".json")):
                return None, None
            with open(p_ + ".json") as f:
                return np.load(p_ + ".npy"), json.load(f)

        (xa, ma), (xb, mb) = load_out(run_a), load_out(run_b)
        ok = rcs["sdef_new"] == 0 and rcs["sdef_old"] == 0 and xa is not None and xb is not None
        skip = {"run", "chunk_seconds", "sha256_sample_latent"}
        dk = sorted(k for k in set(ma or {}) | set(mb or {}) if k not in skip and (ma or {}).get(k) != (mb or {}).get(k))
        check(f"17 default sampler through the command line (synthetic run, 4 trajectories): final states = the module "
              f"before --harmonic bit for bit; manifests the same but for run, chunk_seconds and the module hash "
              f"(differing: {dk})", 0.0 if ok and _same17(xa, xb) and not dk else 1.0, 0.5)
    # early_slices.py on synthetic mnist28_d0123 runs: random modes h, no modes n, zero modes z
    res_p = os.path.join(lm.OUT_ROOT, "early_slices", es_job, "results.json")
    ok, detail = False, f"exit code {rcs['early_slices']}"
    if rcs["early_slices"] == 0 and os.path.exists(res_p):
        with open(res_p) as f:
            res = json.load(f)

        def close(a, b):                           # floats to 1e-12 relative, everything else equal
            if isinstance(a, float) and isinstance(b, float):
                return abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)
            return a == b

        exact_zn = all(json.dumps(res["z"][k]) == json.dumps(res["n"][k]) for k in ("0", "50"))
        same_zn = all(close(res["z"][k][q], res["n"][k][q]) for k in ("0", "50") for q in res["n"][k])
        diff_hn = all(abs(res["h"][k]["score_rel_err"] - res["n"][k]["score_rel_err"])
                      > 1e-9 * res["n"][k]["score_rel_err"] for k in ("0", "50"))
        ok = same_zn and diff_hn
        detail = (f"score errors at slices 0 / 50: modes {res['h']['0']['score_rel_err']:.6g} / "
                  f"{res['h']['50']['score_rel_err']:.6g}, none {res['n']['0']['score_rel_err']:.6g} / "
                  f"{res['n']['50']['score_rel_err']:.6g}; zero modes = none to 1e-12: {same_zn} (exactly: {exact_zn})")
    report(f"   early_slices.py: {detail} ({res_p})")
    check("17 early_slices.py with harmonic runs: zero modes give exactly the results of no modes, random modes change "
          "the score", 0.0 if ok else 1.0, 0.5)
    report(f"   block 17: {time.time() - t17:.0f} s")


def block_18():
    """18: the review fixes of train_latent.py --harmonic (2026-11-12, late): --harmonic-ref within, --harmonic-lr-tmin
    and the diagnostic harm_n_beyond_vis; CPU-sized (8 CPUs; the command-line runs are CPU subprocesses).
    A. prototype_reference(..., return_stats=True): W, b and the state classes bit for bit equal to the default call
       and to the module before --harmonic (HARM_ORIG), pooled and per class; S_w = the pooled within-cluster
       covariance of the k-means partition (kmeans recomputed here), the centres, counts and weights; the law of total
       covariance with the partition's member means (S_w differs from its within part only at second order in the
       distance of the k-means centres from those means).
    B. The within-cluster reference is the exact complement of the Potts prototype reference: on a synthetic
       three-cluster data set (N = 12) the prototype Potts unit plus the within modes (R = N) plus the isotropic
       visible part has the energy (up to a constant) and the force of the prototypes' Gaussian mixture
       sum_m pi_m N(e c_m, e^2 S_w + sigma_t^2 I) at t = 5 ... 0; with the full reference (C) instead the mean curvature
       along the top principal axis falls short of the mixture's (the between-cluster covariance counted twice).
    C. harmonic_unit_stats' harm_n_beyond_vis: the modes whose strength reaches the springs along their direction.
    D. Command lines (gauss2_8, Potts 8 --quad --local, quartic r1 lin, ema, stencil, mixed): the guards; --harmonic 4
       --harmonic-ref within (the spectrum in harmonic_within.npz = this process's; slice 0 = the within reference at
       t = 5 bit for bit; learning rates from it; config and reference file); with --harmonic-fixed (every slice the
       within reference); resume after SIGKILL bit for bit and strict resume; --init-from (another reference refused,
       --harmonic-lr-tmin free and applied); --harmonic-lr-tmin 4.9 with the full reference; a --harmonic 4 run of the
       first --harmonic version (HARM_V1) and of this one with the new options at their defaults: slice files bit for
       bit (records equal but for the new harm_n_beyond_vis), config.json the same but for the two new keys.
    E. Slices 94-99 at --steps-scale 0.1 (about 3,000 steps per slice, 30 x the minimum of block 17's runs; --init-from
       a 100-slice --harmonic 4 run at the minimum steps): finite, every mode within the springs along its direction
       (harm_n_beyond_vis 0), and how far the modes move against the short run."""
    t18 = time.time()
    sl, tl = sample_latent, train_latent
    T = lm.TEMP
    taus = lm.forward_time_pts()
    have_orig = os.path.exists(os.path.join(HARM_ORIG, "latent_model.py"))
    have_v1 = all(os.path.exists(os.path.join(HARM_V1, f"{m}.py")) for m in ("latent_model", "train_latent",
                                                                             "sample_latent"))
    report(f"   modules before --harmonic: {HARM_ORIG} ({'present' if have_orig else 'MISSING'}); the first --harmonic "
           f"version: {HARM_V1} ({'present' if have_v1 else 'MISSING'})")
    check("18 the comparison modules are present (HARM_ORIG, HARM_V1): without them the comparisons below are skipped "
          "and this check fails", 0.0 if have_orig and have_v1 else 1.0, 0.5)
    py, env = sys.executable, dict(os.environ, JAX_PLATFORMS="cpu")
    runs18 = os.path.join(OUT, "b18_runs")
    os.makedirs(runs18)
    st_toy = sl.data_stats("gauss2_8")
    mu_t, lam_t, U_t = (np.asarray(st_toy[k], dtype=np.float64) for k in ("mu", "lam", "U"))
    FL = ["--problem", "gauss2_8", "--K", "8", "--hidden", "potts", "--quad", "--local", "--select", "ema",
          "--visible", "quartic", "--radius", "1", "--pair-terms", "lin", "--proto-init", "pooled",
          "--visible-impl", "stencil", "--precision", "mixed"]
    tl_cur, tl_v1 = os.path.join(HERE, "train_latent.py"), os.path.join(HARM_V1, "train_latent.py")

    def tcmd(script, name, *extra, flags=FL):
        return [py, "-u", script, "--name", name, "--out-root", runs18] + list(flags) + list(extra)

    bg = {}

    def start(tag, cmds):                         # a background chain of commands, stopping at the first failure
        logp = os.path.join(OUT, f"b18_{tag}.log")
        chain = " && ".join(f"{shlex.join(c)} >> {shlex.quote(logp)} 2>&1" for c in cmds)
        bg[tag] = (subprocess.Popen(["bash", "-c", chain], env=env), time.time(), logp)

    # ---- E (background): 100 slices at the minimum steps, then slices 94-99 again at --steps-scale 0.1 ---------
    start("lowt", [tcmd(tl_cur, "lt_src", "--step-impl", "lean", "--steps-schedule", "0:0.001", "--harmonic", "4"),
                   tcmd(tl_cur, "lt_long", "--step-impl", "lean", "--steps-scale", "0.1", "--harmonic", "4",
                        "--init-from", "lt_src", "--init-slices", "94")])

    # ---- A. prototype_reference(..., return_stats=True) ---------------------------------------------------------
    prob8 = lm.make_problem("gauss2_8", 1)
    X8, y8 = np.asarray(prob8["data"], dtype=np.float64), np.asarray(prob8["labels"])
    n8, n_img = X8.shape[1], X8.shape[0]
    t3 = taus[[0, 50, 99]]
    lm_o = _orig17("latent_model") if have_orig else None
    bad, lines = [], []
    for pooled, M, seed in ((True, 8, 0), (False, 8, 100)):
        W0, b0, c0 = lm.prototype_reference(X8, y8, t3, T, M, seed=seed, pooled=pooled)
        W1, b1, c1, ps = lm.prototype_reference(X8, y8, t3, T, M, seed=seed, pooled=pooled, return_stats=True)
        same = _same17(W0, W1) and _same17(b0, b1) and _same17(c0, c1)
        if lm_o is not None:
            Wo, bo, co = lm_o.prototype_reference(X8, y8, t3, T, M, seed=seed, pooled=pooled)
            same = same and _same17(W0, Wo) and _same17(b0, bo) and _same17(c0, co)
        if pooled:
            Ck, asg = lm.kmeans(X8, M, seed)
            Rr, cnt, lab_all, Xo = X8 - Ck[asg], np.bincount(asg, minlength=M), asg, X8
        else:
            cl_ = np.unique(y8)
            parts = [lm.kmeans(X8[y8 == c], M // cl_.size, seed + int(c)) for c in cl_]
            Ck = np.concatenate([p_[0] for p_ in parts])
            Xo = np.concatenate([X8[y8 == c] for c in cl_])
            lab_all = np.concatenate([p_[1] + i * (M // cl_.size) for i, p_ in enumerate(parts)])
            Rr = Xo - Ck[lab_all]
            cnt = np.bincount(lab_all, minlength=M)
        e_sw = rel(ps["Sw"], Rr.T @ Rr / (n_img - M))
        ok_st = (_same17(ps["centres"], Ck) and _same17(ps["counts"], cnt)
                 and _same17(ps["pi"], np.maximum(cnt, 1) / n_img))
        mu8 = Xo.mean(axis=0)
        C8 = (Xo - mu8).T @ (Xo - mu8) / n_img
        xbar = np.stack([Xo[lab_all == j].mean(axis=0) for j in range(M)])
        Wn = (Xo - xbar[lab_all]).T @ (Xo - xbar[lab_all]) / n_img
        Bn = ((xbar - mu8).T * (cnt / n_img)) @ (xbar - mu8)
        e_tot, e_w2 = rel(Wn + Bn, C8), rel(ps["Sw"] * (n_img - M) / n_img, Wn)
        u1 = U_t[:, 0]
        lines.append(f"{'pooled' if pooled else 'per class'} M {M}: W/b/classes {'equal' if same else 'DIFFER'}, S_w vs "
                     f"R^T R / (n - M) {e_sw:.1e}, C = W_n + B_n {e_tot:.1e}, S_w (n - M) / n vs W_n {e_w2:.1e}; along "
                     f"PC 1 the prototypes carry {float(u1 @ Bn @ u1) / float(u1 @ C8 @ u1):.3f} of the variance")
        bad += [f"pooled {pooled}"] if not (same and ok_st and e_sw <= 1e-12 and e_tot <= 1e-12 and e_w2 <= 1e-6) else []
    report("   prototype_reference(..., return_stats=True): " + "; ".join(lines))
    check(f"18 prototype_reference(return_stats=True): W, b, classes bit for bit = the default call and the module before "
          f"--harmonic; S_w = R^T R / (n - M) of the k-means partition, centres / counts / weights equal; C = within + "
          f"between of the partition (pooled and per class; failures {bad})", len(bad), 0.5)

    # ---- B. the within reference is the exact complement of the prototype reference -----------------------------
    Nm, Mm = 12, 3
    rng = np.random.default_rng(1801)
    cent = 3.0 * rng.standard_normal((Mm, Nm))
    Qw, _ = np.linalg.qr(rng.standard_normal((Nm, Nm)))
    Lw = np.linalg.cholesky((Qw * np.geomspace(1.0, 0.05, Nm)) @ Qw.T)
    Xm = np.concatenate([cent[m] + rng.standard_normal((1000, Nm)) @ Lw.T for m in range(Mm)])
    ym = np.repeat(np.arange(Mm) % 2, 1000)                                  # labels: unused (pooled)
    Cm = np.cov(Xm.T, bias=True)
    lam_c, U_c = np.linalg.eigh(Cm)
    lam_c, U_c = lam_c[::-1].copy(), U_c[:, ::-1].copy()
    vis_m = lm.DenseVisible(np.zeros((0, 2), dtype=np.int64), Nm)
    force_m = jax.jit(lm.make_force(vis_m, T, Mm, "potts", False))

    def F_model(x, p):
        a = p["W"] @ x + p["b"]
        return (vis_m.energy(p["theta"], x[None])[0] - T * jax.scipy.special.logsumexp(a / T)
                + lm.harmonic_free_energy(p["V"], p["c"], x[None])[0])

    def F_mix(x, cents_t, S_inv, logpi):
        d = x[None, :] - cents_t
        return -T * jax.scipy.special.logsumexp(logpi - 0.5 * jnp.einsum("mi,ij,mj->m", d, S_inv, d))

    def model_params(W, b, m_c, lam_, U_, t):
        V, c, iso = lm.harmonic_reference(m_c, lam_, U_, Nm, t)
        th = np.zeros(4 * Nm)
        th[:Nm], th[3 * Nm:] = T / iso, -T * np.exp(-t) * m_c / iso
        return {"theta": jnp.asarray(th), "W": jnp.asarray(W), "b": jnp.asarray(b), "V": jnp.asarray(V),
                "c": jnp.asarray(c)}

    u1 = jnp.asarray(U_c[:, 0])
    worst, lines = {"force": 0.0, "energy": 0.0, "curv": 0.0}, []
    short = []
    for t in (5.0, 3.0, 1.0, 0.3, 0.05, 0.0):
        e = np.exp(-t)
        sig2 = T * (1.0 - e * e)
        Wm, bm, _, psm = lm.prototype_reference(Xm, ym, [t], T, Mm, seed=0, pooled=True, return_stats=True)
        lam_w, U_w = lm.harmonic_within_spectrum(psm["Sw"])
        m_c = psm["pi"] @ psm["centres"]                                   # the mixture's mean (sum_m pi_m c_m)
        p_w = model_params(Wm[0], bm[0], m_c, lam_w, U_w, t)
        S = e * e * psm["Sw"] + sig2 * np.eye(Nm)
        S_inv, cents_t = jnp.asarray(np.linalg.inv(S)), jnp.asarray(e * psm["centres"])
        logpi = jnp.asarray(np.log(psm["pi"]))
        r_ = np.random.default_rng(1802)
        comp = r_.integers(Mm, size=24)
        Xj = jnp.asarray(e * psm["centres"][comp] + r_.standard_normal((24, Nm)) @ np.linalg.cholesky(S).T)
        f_w = np.asarray(force_m(p_w, Xj))
        f_x = np.asarray(jax.vmap(jax.grad(F_mix), in_axes=(0, None, None, None))(Xj, cents_t, S_inv, logpi))
        worst["force"] = max(worst["force"], rel(f_w, f_x))
        dE = (np.asarray(jax.vmap(F_model, in_axes=(0, None))(Xj, p_w))
              - np.asarray(jax.vmap(F_mix, in_axes=(0, None, None, None))(Xj, cents_t, S_inv, logpi)))
        worst["energy"] = max(worst["energy"], float(np.std(dE)) / float(np.std(np.asarray(
            jax.vmap(F_mix, in_axes=(0, None, None, None))(Xj, cents_t, S_inv, logpi)))))
        # mean curvature along PC 1 at the mixture's samples: mixture, within reference, full reference (C)
        p_f = model_params(Wm[0], bm[0], m_c, lam_c, U_c, t)
        k_x = float(jnp.mean(jax.vmap(lambda x: u1 @ jax.hessian(F_mix)(x, cents_t, S_inv, logpi) @ u1)(Xj)))
        k_w = float(jnp.mean(jax.vmap(lambda x: u1 @ jax.hessian(F_model)(x, p_w) @ u1)(Xj)))
        k_f = float(jnp.mean(jax.vmap(lambda x: u1 @ jax.hessian(F_model)(x, p_f) @ u1)(Xj)))
        worst["curv"] = max(worst["curv"], abs(k_w - k_x) / abs(k_x))
        short.append(k_f / k_x)
        lines.append(f"t {t:g}: mean curvature along PC 1 mixture {k_x:.4g}, within {k_w:.4g}, full {k_f:.4g}")
    report(f"   synthetic three-cluster mixture (N = {Nm}; the prototypes carry "
           f"{1.0 - float(U_c[:, 0] @ psm['Sw'] @ U_c[:, 0]) / lam_c[0]:.3f} of the variance along PC 1): "
           + "; ".join(lines))
    check("18 within reference (R = N) + prototype Potts reference + isotropic springs = the prototypes' Gaussian "
          "mixture: force at t = 5 ... 0 (worst relative)", worst["force"], 1e-9)
    check("18 ... and its energy up to a constant (worst std of the difference / std of the energy)", worst["energy"],
          1e-9)
    check("18 ... and its mean curvature along PC 1 (worst relative)", worst["curv"], 1e-8)
    check(f"18 with the full reference (C) instead the mean curvature along PC 1 falls short of the mixture's at every t "
          f"(the between-cluster covariance counted twice; ratios full / mixture at t = 5 ... 0: "
          f"{', '.join(f'{v:.3g}' for v in short)})", max(short), 0.99)

    # ---- C. harm_n_beyond_vis ----------------------------------------------------------------------------------
    conn8 = lm.connectivity(8, 1)
    vis8 = lm.DenseVisible(conn8, n8)
    th_c = np.zeros(vis8.n_theta)
    th_c[:n8] = 0.8 + 0.4 * rng.random(n8)
    th_c[lm.clin_slice(n8, conn8.shape[0])] = 0.05 * rng.random(conn8.shape[0])
    ev, Qe = np.linalg.eigh(lm.visible_hessian_at_zero(th_c, conn8, n8))
    js, fs = [3, 17, 40, 60], [0.5, 0.9, 1.1, 1.5]
    Vb = np.stack([np.sqrt(f_ * ev[j]) * Qe[:, j] for j, f_ in zip(js, fs)])
    hs = lm.harmonic_unit_stats(Vb, np.zeros(4), Vb, np.zeros(4), Qe[:, js], T, T, theta=th_c, conn=conn8)
    hs0 = lm.harmonic_unit_stats(0.0 * Vb, np.zeros(4), Vb, np.zeros(4), Qe[:, js], T, T, theta=th_c, conn=conn8)
    report(f"   harm_n_beyond_vis: modes at 0.5 / 0.9 / 1.1 / 1.5 x the springs along their directions -> "
           f"{hs['harm_n_beyond_vis']}; zero modes -> {hs0['harm_n_beyond_vis']}; harm_n_beyond_iso (T/iso = 1) "
           f"{hs['harm_n_beyond_iso']}")
    check("18 harm_n_beyond_vis counts the modes at or beyond the springs along their own directions (2 of 4; zero "
          "modes 0)", abs(hs["harm_n_beyond_vis"] - 2) + abs(hs0["harm_n_beyond_vis"]), 0.5)

    # ---- D. command lines -----------------------------------------------------------------------------------------
    guard_root = os.path.join(OUT, "b18_guards")
    base = ["--name", "g", "--out-root", guard_root, "--problem", "gauss2_8"]
    wref = ["--harmonic-ref", "within"]
    cases = [(wref, "--harmonic-ref within"), (["--harmonic", "4"] + wref, "--harmonic-ref within"),
             (["--K", "3", "--hidden", "ising", "--harmonic", "4"] + wref, "--harmonic-ref within"),
             (["--K", "4", "--hidden", "potts", "--proto-init", "random", "--harmonic", "4"] + wref,
              "--harmonic-ref within"),
             (["--harmonic-lr-tmin", "1"], "--harmonic-lr-tmin 1"),
             (["--harmonic", "4", "--harmonic-lr-tmin", "-1"], "--harmonic-lr-tmin -1"),
             (["--harmonic", "4", "--harmonic-lr-tmin", "nan"], "--harmonic-lr-tmin nan"),
             (["--harmonic", "4", "--harmonic-lr-tmin", "6"], "--harmonic-lr-tmin 6")]
    bad = []
    for argv, text in cases:
        code, _e = _main17(tl, base + argv)
        if not (isinstance(code, str) and code.startswith("ValueError") and text in code):
            bad.append((" ".join(argv), code))
    check(f"18 train_latent guards: --harmonic-ref within without --harmonic, with K = 0, Ising units or --proto-init "
          f"random; --harmonic-lr-tmin without --harmonic, -1, nan, 6: refused before writing (failures {bad}; no "
          f"folder: {not os.path.exists(guard_root)})", len(bad) + int(os.path.exists(guard_root)), 0.5)

    def train18(script, name, *extra, flags=FL, kill_after=None):
        """train_latent.py (script) in a subprocess (as train17 in block 17). Returns (exit code, this call's log)."""
        cmd = tcmd(script, name, *extra, flags=flags)
        logp = os.path.join(OUT, f"b18_train_{name}.log")
        pos = os.path.getsize(logp) if os.path.exists(logp) else 0
        with open(logp, "a") as fo:
            fo.write("$ " + shlex.join(cmd[2:]) + "\n")
            fo.flush()
            p_ = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, env=env)
            if kill_after is not None:
                target = os.path.join(runs18, name, "slices", f"slice_{kill_after:03d}.npz")
                while p_.poll() is None and not os.path.exists(target):
                    time.sleep(0.05)
                if p_.poll() is None:
                    p_.kill()
            rc_ = p_.wait()
        with open(logp) as f:
            f.seek(pos)
            return rc_, f.read()

    def records18(name):
        with open(os.path.join(runs18, name, "train_log.jsonl")) as f:
            return {int(r["slice"]): r for r in (json.loads(ln) for ln in f if ln.strip())}

    def within_npz(name):
        with np.load(os.path.join(runs18, name, "harmonic_within.npz")) as zw:
            return np.asarray(zw["lam"]), np.asarray(zw["U"])

    def at_reference(name, k, lam_, U_):
        Vr, cr, _ = lm.harmonic_reference(mu_t, lam_, U_, 4, float(taus[k]))
        with np.load(os.path.join(runs18, name, "slices", f"slice_{k:03d}.npz")) as z:
            return _same17(z["V"], Vr) and _same17(z["c"], cr)

    three = ["--max-slices", "3", "--steps-scale", "0.01"]
    wfl = ["--harmonic", "4"] + wref
    # --harmonic 4 --harmonic-ref within (trained)
    rc, txt = train18(tl_cur, "w_tr", *wfl, *three)
    bad = [] if rc == 0 else [f"exit {rc}"]
    lam_s = U_s = None
    if rc == 0:
        d_ = os.path.join(runs18, "w_tr")
        lam_s, U_s = within_npz("w_tr")
        ps8 = lm.prototype_reference(X8, y8, taus, T, 8, seed=0, pooled=True, return_stats=True)[3]
        lam_p, U_p = lm.harmonic_within_spectrum(ps8["Sw"])
        e_sp = max(rel(lam_s, lam_p), float(np.max(np.abs(np.abs(np.sum(U_s[:, :4] * U_p[:, :4], axis=0)) - 1.0))))
        bad += [f"spectrum {e_sp:.1e}"] if not e_sp <= 1e-9 else []
        sVw, scw = lm.harmonic_lr_scales(mu_t, lam_s, U_s, 4, taus)
        rr = records18("w_tr")
        for k in range(3):
            at_ref = at_reference("w_tr", k, lam_s, U_s)
            if k == 0:
                bad += ["slice 0 not the within reference"] if not (at_ref and rr[0]["lr_V"] == 0.0) else []
            else:
                bad += [f"slice {k} lr / not trained"] if (at_ref or not (
                    abs(rr[k]["lr_V"] / (5e-4 * sVw[k]) - 1) < 1e-12 and abs(rr[k]["lr_c"] / (5e-4 * scw[k]) - 1) < 1e-12)) \
                    else []
            bad += [f"slice {k} harm_n_beyond_vis"] if not isinstance(rr[k].get("harm_n_beyond_vis"), int) else []
        with open(os.path.join(d_, "config.json")) as f:
            cfg = json.load(f)
        bad += ["config"] if (cfg.get("harmonic_ref"), cfg.get("harmonic_lr_tmin")) != ("within", 0.0) else []
        with open(os.path.join(d_, "harmonic_reference.json")) as f:
            hr = json.load(f)
        bad += ["reference file"] if not (
            hr.get("reference") == "within" and (hr.get("within") or {}).get("file_sha256")
            == lm.sha256_file(os.path.join(d_, "harmonic_within.npz")) and abs(hr["lr_scale_V"][1] / sVw[1] - 1) < 1e-12
            and hr.get("lam_top5") == [float(v) for v in lam_s[:5]]) else []
        with open(os.path.join(d_, "train_log.txt")) as f:
            bad += ["log line"] if "within-cluster" not in f.read() else []
        report(f"   w_tr: S_w lam top 3 {np.array2string(lam_s[:3], precision=4)} (C's "
               f"{np.array2string(lam_t[:3], precision=4)}); reference file 'within': {json.dumps(hr.get('within'))}; "
               f"slice records harm_n_beyond_vis {[rr[k].get('harm_n_beyond_vis') for k in range(3)]}")
    check(f"18 --harmonic 4 --harmonic-ref within: harmonic_within.npz = this process's S_w spectrum, slice 0 = the "
          f"within reference at t = 5 bit for bit (lr 0), slices 1-2 trained with lr = 5e-4 x its scale, config, "
          f"reference file, log line (failures {bad})", len(bad), 0.5)
    # with --harmonic-fixed: every slice the within reference
    rc, _ = train18(tl_cur, "w_fix", *wfl, "--harmonic-fixed", *three)
    bad = [] if rc == 0 else [f"exit {rc}"]
    if rc == 0:
        lam_f, U_f = within_npz("w_fix")
        bad += ["spectrum differs from w_tr's"] if lam_s is None or not (_same17(lam_f, lam_s) and _same17(U_f, U_s)) \
            else []
        rr = records18("w_fix")
        bad += [f"slice {k}" for k in range(3) if not (at_reference("w_fix", k, lam_f, U_f) and rr[k]["lr_V"] == 0.0)]
    check(f"18 --harmonic-ref within --harmonic-fixed: slices 0-2 = the within reference at t_k bit for bit, lr 0; the "
          f"spectrum = the trained run's bit for bit (failures {bad})", len(bad), 0.5)
    # resume after SIGKILL, strict resume
    four = wfl + ["--max-slices", "4", "--steps-scale", "0.01"]
    rc_f, _ = train18(tl_cur, "w_full", *four)
    rc_k, _ = train18(tl_cur, "w_kill", *four, kill_after=1)
    n_k = sum(os.path.exists(os.path.join(runs18, "w_kill", "slices", f"slice_{k:03d}.npz")) for k in range(4))
    rc_r, txt_r = train18(tl_cur, "w_kill", *four)
    diff = _slices_differ17(os.path.join(runs18, "w_full"), os.path.join(runs18, "w_kill"), range(4))
    check(f"18 --harmonic-ref within resumed after SIGKILL (exit {rc_k} with {n_k} slices; resumed: exit {rc_r}) = an "
          f"uninterrupted run, slices 0-3 bit for bit (exit {rc_f}; differences {diff})",
          0.0 if rc_f == 0 and rc_k != 0 and rc_r == 0 and n_k < 4 and not diff and "resuming at slice" in txt_r else 1.0,
          0.5)
    bad = []
    for extra in (["--harmonic", "4", "--max-slices", "4", "--steps-scale", "0.01"], four + ["--harmonic-lr-tmin", "1"]):
        rc, txt = train18(tl_cur, "w_full", *extra)
        bad += [" ".join(extra)] if not (rc != 0 and "different config" in txt) else []
    check(f"18 resuming a --harmonic-ref within run with the full reference or with --harmonic-lr-tmin 1 is refused "
          f"(failures {bad})", len(bad), 0.5)
    # --init-from: another reference refused; --harmonic-lr-tmin free and applied
    bad = []
    rc, txt = train18(tl_cur, "if_full", "--init-from", "w_full", "--init-slices", "2", "--harmonic", "4",
                      "--max-slices", "3", "--steps-scale", "0.01")
    bad += ["full from within accepted"] if not (rc != 0 and "different model config" in txt) else []
    rc, txt = train18(tl_cur, "if_tmin", "--init-from", "w_full", "--init-slices", "2", *wfl, "--harmonic-lr-tmin",
                      "4.9", "--max-slices", "4", "--steps-scale", "0.01")
    if rc == 0 and lam_s is not None:
        lam_i, U_i = within_npz("if_tmin")
        sV49, sc49 = lm.harmonic_lr_scales(mu_t, lam_i, U_i, 4, [4.9])
        rr = records18("if_tmin")
        bad += ["spectrum"] if not (_same17(lam_i, lam_s) and _same17(U_i, U_s)) else []
        bad += [f"lr slice {k}" for k in (2, 3) if not (abs(rr[k]["lr_V"] / (5e-4 * sV49[0]) - 1) < 1e-12
                                                        and abs(rr[k]["lr_c"] / (5e-4 * sc49[0]) - 1) < 1e-12)]
    else:
        bad.append(f"tmin under --init-from: exit {rc}")
    check(f"18 --init-from a --harmonic-ref within run: with the full reference refused; with --harmonic-lr-tmin 4.9 "
          f"accepted, its slices 2-3 (t < 4.9) with the scale at t = 4.9 (failures {bad})", len(bad), 0.5)
    # --harmonic-lr-tmin with the full reference
    rc, _ = train18(tl_cur, "tmin", "--harmonic", "4", "--harmonic-lr-tmin", "4.9", "--max-slices", "4",
                    "--steps-scale", "0.01")
    bad = [] if rc == 0 else [f"exit {rc}"]
    if rc == 0:
        sVc, scc = lm.harmonic_lr_scales(mu_t, lam_t, U_t, 4, np.maximum(taus, 4.9))
        sVu, _scu = lm.harmonic_lr_scales(mu_t, lam_t, U_t, 4, taus)
        rr = records18("tmin")
        bad += [f"lr slice {k}" for k in (1, 2, 3) if not (abs(rr[k]["lr_V"] / (5e-4 * sVc[k]) - 1) < 1e-12
                                                           and abs(rr[k]["lr_c"] / (5e-4 * scc[k]) - 1) < 1e-12)]
        bad += ["slice 1 (t > 4.9) not the plain scale"] if sVc[1] != sVu[1] else []
        bad += ["slices 2, 3 not held"] if not (sVc[2] == sVc[3] and sVc[2] != sVu[2]) else []
        with open(os.path.join(runs18, "tmin", "config.json")) as f:
            bad += ["config"] if json.load(f).get("harmonic_lr_tmin") != 4.9 else []
        with open(os.path.join(runs18, "tmin", "harmonic_reference.json")) as f:
            hr = json.load(f)
        bad += ["reference file"] if not (hr.get("lr_tmin") == 4.9 and hr.get("reference") == "full"
                                          and hr.get("within") is None
                                          and hr.get("lr_scale_V") == [float(v) for v in sVc]) else []
    check(f"18 --harmonic 4 --harmonic-lr-tmin 4.9: learning rates = 5e-4 x the scale at max(t_k, 4.9) (slice 1 at its "
          f"own t, slices 2-3 held), config and reference file record it (failures {bad})", len(bad), 0.5)
    # the first --harmonic version against this one, the new options at their defaults
    if have_v1:
        c_v1 = os.path.join(os.path.dirname(HARM_V1), "out", "latent_physical_SGM", "data_stats", "gauss2_8.npz")
        same_stats = False
        if os.path.exists(c_v1):
            with np.load(c_v1) as zc:
                same_stats = all(_same17(zc[k], st_toy[k]) for k in ("mu", "lam", "U"))
        report(f"   the first version's data_stats cache {c_v1}: " + ("the same mu, lam, U as this one's" if same_stats else
               "missing or different (the reference modes can then differ by rounding)"))
        rc1, _ = train18(tl_v1, "v1_h", "--harmonic", "4", *three)
        rc2, _ = train18(tl_cur, "cur_h", "--harmonic", "4", *three)
        bad = [] if rc1 == 0 and rc2 == 0 else [f"exit codes {rc1}/{rc2}"]
        if not bad:
            da, db = os.path.join(runs18, "v1_h"), os.path.join(runs18, "cur_h")
            bad += _slices_differ17(da, db, range(3), skip=("harm_n_beyond_vis",))
            with open(os.path.join(da, "config.json")) as f:
                co = json.load(f)
            with open(os.path.join(db, "config.json")) as f:
                cn = json.load(f)
            new_keys = {k: cn[k] for k in set(cn) - set(co)}
            cdiff = sorted(k for k in set(co) & set(cn) if k not in ("name", "sha256", "started", "slurm_job")
                           and co[k] != cn[k])
            bad += [f"config {cdiff} {new_keys}"] if cdiff or new_keys != {"harmonic_ref": "full",
                                                                          "harmonic_lr_tmin": 0.0,
                                                                          **PATCH_NEW_KEYS} else []
            with open(os.path.join(da, "harmonic_reference.json")) as f:
                ho = json.load(f)
            with open(os.path.join(db, "harmonic_reference.json")) as f:
                hn = json.load(f)
            # stats_file: each tree's own data_stats cache (the same bytes: stats_sha256 is compared)
            hdiff = sorted(k for k in set(ho) | set(hn) if k not in ("created", "slurm_job", "stats_file")
                           and ho.get(k) != hn.get(k))
            bad += [f"reference file {hdiff}"] if hdiff else []
            rn = records18("cur_h")
            bad += ["no harm_n_beyond_vis"] if not all("harm_n_beyond_vis" in rn[k] for k in range(3)) else []
        check(f"18 --harmonic 4, the first --harmonic version vs this one (new options at their defaults): slice files "
              f"bit for bit (records equal but for harm_n_beyond_vis), config.json the same but for harmonic_ref "
              f"'full' and harmonic_lr_tmin 0.0 (and the four --patch keys at their defaults), harmonic_reference.json "
              f"the same (failures {bad[:6]})", len(bad), 0.5)

    # ---- E. the long low-t slices ---------------------------------------------------------------------------------
    rcs = {}
    for tag, (p_, t_p, logp) in bg.items():
        try:
            rcs[tag] = p_.wait(timeout=max(60.0, 1560.0 - (time.time() - t18)))
        except subprocess.TimeoutExpired:
            p_.kill()
            rcs[tag] = "timeout"
        report(f"   background {tag}: exit code {rcs[tag]}, {time.time() - t_p:.0f} s after its start ({logp})")
    bad = [] if rcs.get("lowt") == 0 else [f"exit {rcs.get('lowt')}"]
    n_bey = []
    if not bad:
        rs, rl = records18("lt_src"), records18("lt_long")
        rows = []
        for k in range(94, 100):
            with np.load(os.path.join(runs18, "lt_src", "slices", f"slice_{k:03d}.npz")) as zs, \
                    np.load(os.path.join(runs18, "lt_long", "slices", f"slice_{k:03d}.npz")) as zl:
                Vs_, Vl_ = np.asarray(zs["V"]), np.asarray(zl["V"])
                fin = bool(np.all(np.isfinite(Vl_)) and np.all(np.isfinite(zl["c"])) and np.all(np.isfinite(zl["theta"])))
            dv = float(np.linalg.norm(Vl_ - Vs_) / np.linalg.norm(Vs_))
            bad += [f"slice {k} not finite"] if not fin else []
            bad += [f"slice {k} steps {rl[k]['steps']}"] if rl[k]["steps"] < 2500 else []
            n_bey.append(rl[k].get("harm_n_beyond_vis"))
            rows.append(f"{k}: steps {rs[k]['steps']}/{rl[k]['steps']}, |V| {rs[k]['harm_V_norm']:.3g}/"
                        f"{rl[k]['harm_V_norm']:.3g} (ref {rl[k]['harm_Vref_norm']:.3g}), |V_long - V_short| / |V_short| "
                        f"{dv:.3g}, cos V^T V {rs[k]['harm_cos_M_Mref']:.3f}/{rl[k]['harm_cos_M_Mref']:.3f}, curvature "
                        f"springs {rl[k]['harm_vis_curv_min']:.3g} -> with modes {rl[k]['harm_curv_min']:.3g}, beyond "
                        f"the springs {rl[k].get('harm_n_beyond_vis')}, lr_V {rl[k]['lr_V']:.3g}")
        report("   slices 94-99, minimum steps / --steps-scale 0.1:\n      " + "\n      ".join(rows))
    check(f"18 slices 94-99 at --steps-scale 0.1 (--init-from a 100-slice --harmonic 4 run): finite, >= 2500 steps per "
          f"slice (modes beyond the springs along their directions, a finding rather than a check: {n_bey}; failures "
          f"{bad})", len(bad), 0.5)
    report(f"   block 18: {time.time() - t18:.0f} s")


# block 19: the modules from before --patch (latent_model, train_latent, sample_latent, sample_fast, early_slices,
# test_latent and harmonic_analysis as installed on 2026-11-13 at 00:05), laid out like HARM_ORIG (PATCH_ORIG)
PATCH_ORIG = os.environ.get("PATCH_ORIG", os.path.join(lm.OUT_ROOT, "patch_dev", "pkg", "latent_orig"))
PLAIN19 = "mnist28_d0123_K784_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_seed1"     # job 4222701
# the judge's de-risk numbers (job 4231616, results_4231616.json, 'full' form): mean loss change on the plain seed-1
# slices when the patch reference at scale lam (7:3:32, k-means seeds 20000 + p) is added, draws of init_scale_check.py
E2E19 = {(0, 0.25): 0.18913879007784207, (50, 0.04): -1.8738779338858222, (50, 0.1): 22.221459564603787,
         (60, 0.1): 106.87821406111935}


def _orig19(name):
    """The module `name` from before --patch (PATCH_ORIG/<name>.py), imported in this process as <name>_orig19 (as
    _orig17: its own 'import latent_model' finds this process's latent_model; the copy of latent_model itself is the
    old code throughout)."""
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(f"{name}_orig19", os.path.join(PATCH_ORIG, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def _synth_hist19(n, conn, K, R, geom, S, seed, patch="random", quad=True):
    """_synth_hist17's smooth synthetic history (Potts K --quad --local, quartic lin; R harmonic modes if R > 0) plus
    per-patch Potts units of geometry geom with S states: patch 'random' (template entries about 0.03 -> 0.06 from t = 5
    to 0, fields about 0.01, Qp about 0.002 with quad), 'zero' (all zero) or None (no units). Members in
    hidden_history.npz's order: W, b, t, wl, bl, Q[, V, c][, Wp, bp[, Qp]]."""
    h = _synth_hist17(n, conn, K, R, seed, modes="random" if R else None)
    if patch is None:
        return h
    rng = np.random.default_rng(seed + 1900)
    s = (1.0 - lm.forward_time_pts() / lm.T_FORWARD)[:, None, None, None]
    h["Wp"] = (0.03 + 0.03 * s) * rng.standard_normal((geom.P, S, geom.d))[None]
    h["bp"] = 0.01 * (1.0 + s[..., 0]) * rng.standard_normal((geom.P, S))[None]
    if quad:
        h["Qp"] = 0.002 * s * rng.standard_normal((geom.P, S, geom.d))[None]
    if patch == "zero":
        for g in ("Wp", "bp", "Qp"):
            if g in h:
                h[g] = np.zeros_like(h[g])
    return h


def _np_fold19(geom, Gp):
    """fold by np.add.at: (P, B, d) -> (B, N)."""
    Gp = np.asarray(Gp, dtype=np.float64)
    out = np.zeros((geom.N, Gp.shape[1]))
    np.add.at(out, geom.idx.ravel(), np.transpose(Gp, (0, 2, 1)).reshape(-1, Gp.shape[1]))
    return out.T


def _sha19(*arrs):
    import hashlib
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def block_19(parts="ABCDEFG"):
    """19: per-patch Potts hidden units (train_latent.py --patch SIDE:STRIDE:STATES, 2026-11-13; spec
    out/latent_physical_SGM/patch_design/judge/final_spec.md section 7). CPU-sized (8 CPUs; G's command-line runs are
    CPU subprocesses, partly in parallel). 'test_latent.py 19' runs A-G; 'test_latent.py 19 G' (or any letters) a part.
    A. Formulas (float64): PatchGeometry 4:2 and 5:3 on 8x8, 7:3, 8:4 and 14:7 on 28x28: unfold = the gather X[:, idx]
       bit for bit, fold = np.add.at, the adjoint identity, patch_geometry from the shapes; patch_terms' gradient and
       Laplacian against autodiff of patch_free_energy (with and without Qp); one patch covering the grid =
       hidden_terms_quad(..., 'potts') and potts_terms.
    B. The loss and the force with patches against a full-autodiff reference (paper-form visible energy, global unit,
       per-pixel units, harmonic modes, patches) for five model configurations on 8x8, the parameter gradient along a
       random direction against central differences; zero patch arrays = absent keys bit for bit.
    C. --precision mixed with patches (Potts 8 quad local + 7:3:16 at the scale-1 reference of mnist28_d0123): gradient
       error against float64 relative to the gradient and to its batch noise at t = 5 ... 0; every product float32 at
       HIGHEST; the same scatter / gather counts in the jaxpr with and without patches.
    D. The default path in this process against the modules before --patch (PATCH_ORIG): loss and gradient bit for bit
       and the same jaxpr (block 17's cases + 16 harmonic modes in mixed), make_force, the runner (default and lean),
       make_drift (SDE / PF-ODE, with and without modes), load_params.
    E. The reference on mnist28_d0123: patch_reference(patch_clusters) = prototype_reference on the patch data bit for
       bit (patches 0 and 27, run seeds 1 and 2); the scale lam against a hand computation; the learning-rate scales;
       the de-risk step: the integrated loss reproduces the judge's loss changes (job 4231616) to 1e-6.
    F. The runner with patch keys and per-patch learning-rate arrays (moves / keeps, one trace); make_drift with patches
       (and modes) against x - s grad F by autodiff; load_params refuses patch runs unless asked; --fast refuses them.
    G. Command lines (gauss2_8, Potts 8 --quad --local, quartic r1 lin, ema, stencil, mixed; --patch 4:2:4): the guards;
       the default path old vs new (2 slices study + lean; 100 slices in both assembly modes; the default sampler);
       patch runs (lean, both assembly modes: identical slices and history bytes, members W b t wl bl Q Wp bp Qp; slice
       0 = 0.25 x the reference with lr 0; learning rates 5e-4 x the per-patch rms); SIGKILL and resume; strict resume;
       --init-from; --patch-on-slice 3 (slices 0-2 = a plain run's bit for bit, the start 0.06 x the reference at t_3);
       K = 0, Ising, --harmonic 2, --time-grid power:1.5 and --seed 2 with patches; the default sampler on a patch
       run; early_slices.py with zero, absent and random patch units."""
    import math
    t19 = time.time()
    sl, tl = sample_latent, train_latent
    T = lm.TEMP
    taus = lm.forward_time_pts()
    parts = "".join(parts).upper() or "ABCDEFG"
    have_orig = all(os.path.exists(os.path.join(PATCH_ORIG, f"{m}.py"))
                    for m in ("latent_model", "train_latent", "sample_latent"))
    report(f"   block 19 parts {parts}; modules before --patch: {PATCH_ORIG} ({'present' if have_orig else 'MISSING'})")
    if "D" in parts or "G" in parts:
        check("19 the modules from before --patch are present (PATCH_ORIG): the default-path comparisons need them",
              0.0 if have_orig else 1.0, 0.5)
    py, env = sys.executable, dict(os.environ, JAX_PLATFORMS="cpu")
    n8, conn8 = 64, lm.connectivity(8, 1)
    P8 = conn8.shape[0]
    conn28 = lm.connectivity(28, 1)
    P28 = conn28.shape[0]
    FL = ["--problem", "gauss2_8", "--K", "8", "--hidden", "potts", "--quad", "--local", "--select", "ema",
          "--visible", "quartic", "--radius", "1", "--pair-terms", "lin", "--proto-init", "pooled",
          "--visible-impl", "stencil", "--precision", "mixed"]
    PF = ["--patch", "4:2:4"]
    runs19 = os.path.join(OUT, "b19_runs")
    os.makedirs(runs19, exist_ok=True)
    bg = {}

    def start(tag, cmds, extra_env=None):         # a background chain of commands, stopping at the first failure
        logp = os.path.join(OUT, f"b19_{tag}.log")
        chain = " && ".join(f"{shlex.join(c)} >> {shlex.quote(logp)} 2>&1" for c in cmds)
        bg[tag] = (subprocess.Popen(["bash", "-c", chain], env=dict(env, **(extra_env or {}))), time.time(), logp)

    tl_cur, sl_cur = os.path.join(HERE, "train_latent.py"), os.path.join(HERE, "sample_latent.py")
    tl_old, sl_old = os.path.join(PATCH_ORIG, "train_latent.py"), os.path.join(PATCH_ORIG, "sample_latent.py")
    old_root = os.path.join(os.path.dirname(PATCH_ORIG), "out", "latent_physical_SGM")   # the old modules' OUT_ROOT

    def tcmd(script, name, *extra, flags=FL):
        return [py, "-u", script, "--name", name, "--out-root", runs19] + list(flags) + list(extra)

    def scmd(script, run_rel):
        return [py, "-u", script, "--run", run_rel, "--n", "4", "--chunk", "4"]

    geom8 = lm.PatchGeometry(8, 4, 2)

    # ---- G (background): 100-slice runs, the samplers, early_slices, independent short runs ----------------------
    if "G" in parts:
        st_toy = sl.data_stats("gauss2_8")       # the cache the --harmonic run below loads (written before it starts)
        mn = ["--steps-schedule", "0:0.001"]
        if have_orig:
            for tag, scr in (("d100_old_files", tl_old), ("d100_new_files", tl_cur)):
                start(tag, [tcmd(scr, tag, *mn)])
            for tag, scr in (("d100_old_mem", tl_old), ("d100_new_mem", tl_cur)):
                start(tag, [tcmd(scr, tag, *mn, "--assemble", "memory")])
        start("p_mem", [tcmd(tl_cur, "p_mem", *PF, "--step-impl", "lean", "--assemble", "memory", *mn)])
        start("p_files", [tcmd(tl_cur, "p_files", *PF, "--step-impl", "lean", *mn)])
        run_a, run_b = os.path.join(OUT, "b19_synth_a"), os.path.join(OUT, "b19_synth_b")
        synthetic_run(run_a, K=4, seed=1919)
        os.makedirs(run_b)
        for f in ("config.json", "params_history.npy", "hidden_history.npz", "DONE"):
            os.symlink(os.path.join("..", "b19_synth_a", f), os.path.join(run_b, f))
        start("sdef_new", [scmd(sl_cur, os.path.relpath(run_a, lm.OUT_ROOT))])
        if have_orig:
            start("sdef_old", [scmd(sl_old, os.path.relpath(run_b, old_root))])
        cfg8 = {"problem": "gauss2_8", "K": 4, "hidden": "potts", "quad": True, "local": True, "visible": "quartic",
                "pair_terms": "lin", "radius": 1, "neighbourhood": "chebyshev", "cls": "none"}
        run_p = os.path.join(OUT, "b19_synth_patch")
        _write_run17(run_p, _synth_hist19(n8, conn8, 4, 0, geom8, 4, 1931), dict(cfg8, patch="4:2:4"))
        start("samp_patch", [scmd(sl_cur, os.path.relpath(run_p, lm.OUT_ROOT))])
        g73_4 = lm.PatchGeometry(28, 7, 3)
        cfg_es = {"problem": "mnist28_d0123", "K": 4, "hidden": "potts", "quad": True, "local": True,
                  "visible": "quartic", "pair_terms": "lin", "radius": 1, "neighbourhood": "chebyshev", "cls": "none"}
        es_runs = {}
        for lab, pt in (("r", "random"), ("n", None), ("z", "zero")):
            es_runs[lab] = os.path.join(OUT, f"b19_es_{lab}")
            _write_run17(es_runs[lab], _synth_hist19(lm.N, conn28, 4, 0, g73_4, 4, 1919, patch=pt),
                         dict(cfg_es, patch="" if pt is None else "7:3:4"))
        es_job = f"b19es_{os.environ.get('SLURM_JOB_ID', 'manual')}"
        start("early_slices", [[py, "-u", os.path.join(HERE, "early_slices.py")]
                               + [f"{lab}={os.path.relpath(p, lm.OUT_ROOT)}" for lab, p in es_runs.items()]
                               + ["--slices", "0,50"]], {"SLURM_JOB_ID": es_job})
        two = ["--max-slices", "2", "--steps-scale", "0.01"]
        start("chainA", [tcmd(tl_cur, "on3", *PF, "--patch-on-slice", "3", "--patch-init-scale", "0.06",
                              "--max-slices", "5", "--steps-scale", "0.01"),
                         tcmd(tl_cur, "plain5", "--max-slices", "5", "--steps-scale", "0.01"),
                         tcmd(tl_cur, "k0p", *PF, *two, flags=["--problem", "gauss2_8"])])
        start("chainB", [tcmd(tl_cur, "isp", *PF, *two, flags=["--problem", "gauss2_8", "--K", "3", "--hidden", "ising"]),
                         tcmd(tl_cur, "hp", "--harmonic", "2", *PF, *two),
                         tcmd(tl_cur, "grid", *PF, "--time-grid", "power:1.5", "--max-slices", "3", "--steps-scale",
                              "0.01"),
                         tcmd(tl_cur, "seed2", *PF, "--seed", "2", *two)])

    # ---- A. formulas ----------------------------------------------------------------------------------------------
    rng = np.random.default_rng(1901)
    if "A" in parts:
        bad, lines = [], []
        for G, s, r in ((8, 4, 2), (8, 5, 3), (28, 7, 3), (28, 8, 4), (28, 14, 7)):
            geom = lm.PatchGeometry(G, s, r)
            X = rng.standard_normal((5, G * G))
            Gp = rng.standard_normal((geom.P, 5, geom.d))
            u = np.asarray(geom.unfold(jnp.asarray(X)))
            f = np.asarray(geom.fold(jnp.asarray(Gp)))
            e_un = _same17(u, np.transpose(X[:, geom.idx], (1, 0, 2)))
            e_fo = rel(f, _np_fold19(geom, Gp))
            a1, a2 = float(np.sum(f * X)), float(np.sum(Gp * u))
            e_adj = abs(a1 - a2) / max(abs(a1), abs(a2))
            g2 = lm.patch_geometry(G * G, geom.P, geom.d)
            same_g = (g2.s, g2.r, g2.n_p) == (s, r, geom.n_p) and np.array_equal(g2.idx, geom.idx)
            cov_ok = np.array_equal(geom.coverage, np.bincount(geom.idx.ravel(), minlength=G * G))
            lines.append(f"{s}:{r} on {G}x{G}: P {geom.P}, d {geom.d}, coverage {geom.coverage.min()}-{geom.coverage.max()} "
                         f"(mean {geom.coverage.mean():.2f}), central {int(geom.central.sum())}; unfold = gather {e_un}, "
                         f"fold vs np.add.at {e_fo:.1e}, adjoint {e_adj:.1e}")
            bad += [f"{s}:{r}/{G}"] if not (e_un and e_fo <= 1e-15 and e_adj <= 1e-13 and same_g and cov_ok) else []
        report("   PatchGeometry: " + "; ".join(lines))
        check(f"19A unfold = the gather X[:, idx] bit for bit, fold = np.add.at to 1e-15, <fold g, x> = <g, unfold x> to "
              f"1e-13, patch_geometry from the shapes = the geometry (4:2, 5:3 on 8x8; 7:3, 8:4, 14:7 on 28x28; failures "
              f"{bad})", len(bad), 0.5)
        g73 = lm.PatchGeometry(28, 7, 3)
        check(f"19A 7:3 on 28x28: P 64, d 49, coverage 1-9 (mean 4.00) as the spec",
              0.0 if (g73.P, g73.d, int(g73.coverage.min()), int(g73.coverage.max())) == (64, 49, 1, 9)
              and abs(g73.coverage.mean() - 4.0) < 1e-12 else 1.0, 0.5)
        worst = {"grad": 0.0, "lap": 0.0}
        for G, s, r, S in ((8, 4, 2, 3), (28, 7, 3, 4)):
            geom = lm.PatchGeometry(G, s, r)
            for quad in (True, False):
                Wp = jnp.asarray(0.05 * rng.standard_normal((geom.P, S, geom.d)))
                Qp = jnp.asarray(0.01 * rng.standard_normal((geom.P, S, geom.d))) if quad else None
                bp = jnp.asarray(0.02 * rng.standard_normal((geom.P, S)))
                X = jnp.asarray(0.6 * rng.standard_normal((3, G * G)))
                g, lap, A, p = lm.patch_terms(Wp, Qp, bp, X, T)
                f1 = lambda x, Wp=Wp, Qp=Qp, bp=bp: lm.patch_free_energy(Wp, Qp, bp, x[None], T)[0]   # noqa: E731
                worst["grad"] = max(worst["grad"], rel(g, jax.vmap(jax.grad(f1))(X)))
                worst["lap"] = max(worst["lap"], rel(lap, jax.vmap(lambda x, f1=f1: jnp.trace(jax.hessian(f1)(x)))(X)))
        check("19A patch_terms: grad F_patch vs autodiff of patch_free_energy (4:2:3 on 8x8, 7:3:4 on 28x28, with and "
              "without Qp; worst relative)", worst["grad"], 1e-12)
        check("19A patch_terms: Laplacian of F_patch vs the trace of the autodiff Hessian (worst relative)", worst["lap"],
              1e-12)
        W1 = jnp.asarray(0.05 * rng.standard_normal((5, n8)))
        Q1 = jnp.asarray(0.01 * rng.standard_normal((5, n8)))
        b1 = jnp.asarray(0.02 * rng.standard_normal(5))
        X = jnp.asarray(0.6 * rng.standard_normal((6, n8)))
        gq, lq, _, _ = lm.patch_terms(W1[None], Q1[None], b1[None], X, T)
        gh, lh, _, _ = lm.hidden_terms_quad(W1, Q1, b1, X, T, "potts")
        g0, l0, _, _ = lm.patch_terms(W1[None], None, b1[None], X, T)
        gp, lp_, _, _ = lm.potts_terms(W1, b1, X, T)
        check("19A one patch covering the 8x8 grid = hidden_terms_quad(..., 'potts') (with Q) and potts_terms (without): "
              "gradient and Laplacian (worst relative)", max(rel(gq, gh), rel(lq, lh), rel(g0, gp), rel(l0, lp_)), 1e-13)

    # ---- B. the loss and the force with patches ------------------------------------------------------------------
    if "B" in parts:
        vis8 = lm.DenseVisible(conn8, n8)
        th8 = jnp.asarray(0.3 * rng.standard_normal(vis8.n_theta))
        X4 = jnp.asarray(0.5 * rng.standard_normal((4, n8)))
        V3, c3 = jnp.asarray(0.3 * rng.standard_normal((3, n8))), jnp.asarray(0.2 * rng.standard_normal(3))
        Sp = 3

        def F_full(x, p, K, hidden, local):
            f = vis8.energy(p["theta"], x[None])[0]
            if K:
                a = p["W"] @ x + p["b"] + (p["Q"] @ (x * x) if "Q" in p else 0.0)
                f = f - T * (jax.scipy.special.logsumexp(a / T) if hidden == "potts"
                             else jnp.sum(jnp.logaddexp(a / T, -a / T)))
            if local:
                al = p["wl"] * x + p["bl"]
                f = f - T * jnp.sum(jnp.logaddexp(al / T, -al / T))
            if "V" in p:
                f = f + lm.harmonic_free_energy(p["V"], p["c"], x[None])[0]
            return f + lm.patch_free_energy(p["Wp"], p.get("Qp"), p["bp"], x[None], T)[0]

        for K, hidden, quad, local, modes in ((0, "potts", True, False, False), (3, "potts", True, True, False),
                                              (3, "potts", False, False, False), (3, "ising", True, True, False),
                                              (3, "potts", True, True, True)):
            p = {"theta": th8, "W": jnp.asarray(0.05 * rng.standard_normal((K, n8))),
                 "b": jnp.asarray(0.05 * rng.standard_normal(K)),
                 "Wp": jnp.asarray(0.05 * rng.standard_normal((geom8.P, Sp, geom8.d))),
                 "bp": jnp.asarray(0.03 * rng.standard_normal((geom8.P, Sp)))}
            if quad and K:
                p["Q"] = jnp.asarray(0.01 * rng.standard_normal((K, n8)))
            if quad:
                p["Qp"] = jnp.asarray(0.01 * rng.standard_normal((geom8.P, Sp, geom8.d)))
            if local:
                p.update(wl=jnp.asarray(0.3 * rng.standard_normal(n8)), bl=jnp.asarray(0.02 * rng.standard_normal(n8)))
            if modes:
                p.update(V=V3, c=c3)
            lab = (f"K={K} {hidden}{' quad' if quad else ''}{' local' if local else ''}{' + 3 modes' if modes else ''}"
                   f" + patches 4:2:3{'' if quad else ' (no Qp)'}")

            def lref(p_, X_, K=K, hidden=hidden, local=local):
                def one(x):
                    g = jax.grad(F_full)(x, p_, K, hidden, local)
                    return -jnp.trace(jax.hessian(F_full)(x, p_, K, hidden, local)) / T + jnp.sum(g * g) / (2 * T * T)
                return jnp.mean(jax.lax.map(one, X_))

            lref = jax.jit(lref)
            lf = lm.make_loss(vis8, T, K, hidden, local)
            a_, b_ = float(jax.jit(lf)(p, X4)), float(lref(p, X4))
            check(f"19B loss with patches vs full-autodiff reference, {lab} ({b_:.6g})", abs(a_ - b_) / max(abs(b_), 1.0),
                  1e-10)
            fo = np.asarray(jax.jit(lm.make_force(vis8, T, K, hidden, local))(p, X4))
            fr = np.asarray(jax.vmap(jax.grad(F_full), in_axes=(0, None, None, None, None))(X4, p, K, hidden, local))
            check(f"19B make_force with patches vs autodiff grad F, {lab}", rel(fo, fr), 1e-12)
            g = jax.jit(jax.grad(lf))(p, X4)
            kd = jr.fold_in(jr.PRNGKey(1902), K + 10 * quad + 100 * local + 1000 * modes)
            d = {kk: jr.normal(jr.fold_in(kd, i), p[kk].shape) * (jnp.abs(p[kk]) + 1e-3) for i, kk in enumerate(p)}
            fd = (float(lref({kk: p[kk] + 1e-5 * d[kk] for kk in p}, X4))
                  - float(lref({kk: p[kk] - 1e-5 * d[kk] for kk in p}, X4))) / 2e-5
            an = float(sum(jnp.sum(g[kk] * d[kk]) for kk in p))
            check(f"19B loss gradient (every group, Wp, bp{', Qp' if quad else ''} too) along a random direction vs "
                  f"central difference, {lab} ({an:.6g})", abs(an - fd) / max(abs(fd), 1e-8), 1e-5)
        # zero patch arrays = absent keys, bit for bit (Potts 3 quad local, float64 and mixed)
        p0 = {"theta": th8, "W": jnp.asarray(0.05 * rng.standard_normal((3, n8))),
              "b": jnp.asarray(0.05 * rng.standard_normal(3)), "Q": jnp.asarray(0.01 * rng.standard_normal((3, n8))),
              "wl": jnp.asarray(0.3 * rng.standard_normal(n8)), "bl": jnp.asarray(0.02 * rng.standard_normal(n8))}
        pz = dict(p0, Wp=jnp.zeros((geom8.P, Sp, geom8.d)), bp=jnp.zeros((geom8.P, Sp)), Qp=jnp.zeros((geom8.P, Sp, geom8.d)))
        bad = []
        for cdt in (None, jnp.float32):
            lps = jax.jit(lm.make_loss_per_sample(vis8, T, 3, "potts", True, compute_dtype=cdt))
            bad += [f"loss {cdt}"] if not _same17(lps(p0, X4), lps(pz, X4)) else []
        fz = jax.jit(lm.make_force(vis8, T, 3, "potts", True))
        bad += ["force"] if not _same17(fz(p0, X4), fz(pz, X4)) else []
        check(f"19B zero patch arrays = no patch keys, bit for bit: per-sample loss (float64 and mixed) and force "
              f"(failures {bad})", len(bad), 0.5)

    # ---- C. mixed precision with patches -------------------------------------------------------------------------
    if "C" in parts:
        import re
        prob4 = lm.make_problem("mnist28_d0123", 1)
        data4 = prob4["data"]
        g73 = lm.PatchGeometry(28, 7, 3)
        clus16 = lm.patch_clusters(np.asarray(data4, dtype=np.float64), g73, 16, lm.PATCH_SEED0)
        report(f"   19C: k-means of the 64 patch sets (S = 16) in {clus16['seconds']:.0f} s")
        zs = lm.dropped_pair_slices(lm.N, P28, ("lin",))
        kept_m = np.ones(4 * lm.N + 3 * P28, bool)
        for s_ in zs + (lm.k6_slice(lm.N),):
            kept_m[s_] = False
        k1, k2, k3, k4, k5, k6 = jr.split(jr.PRNGKey(1903), 6)
        th_q = lm.quartic_initial_theta(lm.paper_theta_layout(P28)[0], lm.N, P28)
        th_q = jnp.where(kept_m, th_q + 0.2 * jr.normal(k1, th_q.shape), 0.0)
        pq = {"theta": th_q, "W": 0.02 * jr.normal(k2, (8, lm.N)), "b": 0.02 * jr.normal(k3, (8,)),
              "Q": 0.005 * jr.normal(k4, (8, lm.N)), "wl": 0.3 * jr.normal(k5, (lm.N,)), "bl": 0.02 * jr.normal(k6, (lm.N,))}
        vis_d = lm.DenseVisible(conn28)
        vis_st = lm.make_visible(conn28, lm.N, "stencil", ("lin",), sextic=False)
        vg_d = jax.jit(jax.value_and_grad(lm.make_loss(vis_d, T, 8, "potts", True)))
        loss_m = lm.make_loss(vis_st, T, 8, "potts", True, compute_dtype=jnp.float32)
        vg_m = jax.jit(jax.value_and_grad(loss_m))
        on = lambda g, a: np.asarray(a)[kept_m] if g == "theta" else np.asarray(a).ravel()   # noqa: E731
        rows, bad = [], []
        for i, t in enumerate((5.0, 2.5, 1.0, 0.5, 0.0)):
            Wr, br = lm.patch_reference(clus16, [t], T)
            rmsw = float(np.sqrt(np.mean(Wr[0] ** 2)))
            p = dict(pq, Wp=jnp.asarray(Wr[0]), bp=jnp.asarray(br[0]),
                     Qp=jnp.asarray(0.05 * rmsw * rng.standard_normal(Wr[0].shape)))
            kx1, kx2 = jr.split(jr.fold_in(jr.PRNGKey(1904), i))
            X1 = sample_forward_process(t, 64, T, lm.SIGMA_FORWARD, data4, kx1, beta=1)
            X2 = sample_forward_process(t, 64, T, lm.SIGMA_FORWARD, data4, kx2, beta=1)
            (L_d, G_d), (_, G_d2), (L_m, G_m) = vg_d(p, X1), vg_d(p, X2), vg_m(p, X1)
            err = {g: float(np.linalg.norm(on(g, G_m[g]) - on(g, G_d[g]))) for g in G_d}
            nrm = {g: float(np.linalg.norm(on(g, G_d[g]))) for g in G_d}
            noise = {g: float(np.linalg.norm(on(g, G_d[g]) - on(g, G_d2[g]))) / np.sqrt(2.0) for g in G_d}
            relg = {g: err[g] / max(nrm[g], 1e-300) for g in G_d}
            eon = {g: err[g] / max(noise[g], 1e-300) for g in G_d}
            vd = np.concatenate([on(g, G_d[g]) for g in sorted(G_d)])
            vm = np.concatenate([on(g, G_m[g]) for g in sorted(G_d)])
            cos = float(vd @ vm / (np.linalg.norm(vd) * np.linalg.norm(vm)))
            med_rel, med_eon = float(np.median(list(relg.values()))), float(np.median(list(eon.values())))
            rows.append(f"t {t:g}: loss {float(L_d):.6g} (mixed rel {abs(float(L_m) - float(L_d)) / abs(float(L_d)):.1e}); "
                        f"relative gradient error median {med_rel:.1e} (Wp {relg['Wp']:.1e}, bp {relg['bp']:.1e}, Qp "
                        f"{relg['Qp']:.1e}, worst {max(relg.values()):.1e}); cosine {cos:.8f}; error / batch noise "
                        f"median {med_eon:.1e} (worst {max(eon.values()):.1e})")
            if t > 0:
                bad += [f"t {t:g}"] if not (med_rel <= 1e-3 and cos >= 0.9999 and med_eon <= 1e-2) else []
            else:
                bad += ["t 0"] if not med_eon <= 1e-2 else []
        report("   19C mixed vs float64 with patches 7:3:16 at the scale-1 reference (Potts 8 quad local):\n      "
               + "\n      ".join(rows))
        check(f"19C mixed precision with patches: median relative gradient error <= 1e-3, cosine >= 0.9999 and error / "
              f"batch noise median <= 1e-2 at t = 5, 2.5, 1, 0.5; at t = 0 error / noise <= 1e-2 (failures {bad})",
              len(bad), 0.5)
        dots = [ln for ln in vg_m.lower(p, X1).as_text().splitlines() if "dot_general" in ln]
        n_bad = sum(not ("HIGHEST" in ln and "xf32>" in ln and "xf64>" not in ln) for ln in dots)
        check(f"19C mixed precision with patches: every product float32 at HIGHEST, no float64 operand ({len(dots)} "
              f"dot_general)", n_bad + (not dots), 0.5)
        pat = re.compile(r"= (scatter[\w-]*|gather)\[")
        jx_p = str(jax.make_jaxpr(jax.value_and_grad(loss_m))(p, X1))
        jx_0 = str(jax.make_jaxpr(jax.value_and_grad(loss_m))(pq, X1))
        cnt = lambda s: sorted((m, len([x for x in pat.findall(s) if x == m])) for m in set(pat.findall(s)))  # noqa: E731
        report(f"   19C scatter / gather primitives in the jaxpr of value_and_grad (mixed): with patches {cnt(jx_p)}, "
               f"without {cnt(jx_0)}")
        check("19C the patch path adds no scatter or gather (the same counts in the jaxpr with and without patches)",
              0.0 if cnt(jx_p) == cnt(jx_0) else 1.0, 0.5)

    # ---- D. the default path against the modules before --patch -------------------------------------------------
    if "D" in parts and have_orig:
        lm_o, tl_o, sl_o = _orig19("latent_model"), _orig19("train_latent"), _orig19("sample_latent")
        prob4 = lm.make_problem("mnist28_d0123", 1)
        zs = lm.dropped_pair_slices(lm.N, P28, ("lin",))
        kept_m = np.ones(4 * lm.N + 3 * P28, bool)
        for s_ in zs + (lm.k6_slice(lm.N),):
            kept_m[s_] = False
        k1, k2, k3, k4, k5, k6, k7, k8 = jr.split(jr.PRNGKey(1905), 8)
        th_q = lm.quartic_initial_theta(lm.paper_theta_layout(P28)[0], lm.N, P28)
        th_q = jnp.where(kept_m, th_q + 0.2 * jr.normal(k1, th_q.shape), 0.0)
        p_q = {"theta": th_q, "W": 0.02 * jr.normal(k2, (8, lm.N)), "b": 0.02 * jr.normal(k3, (8,)),
               "Q": 0.005 * jr.normal(k4, (8, lm.N)), "wl": 0.3 * jr.normal(k5, (lm.N,)),
               "bl": 0.02 * jr.normal(k6, (lm.N,))}
        p_h = dict(p_q, V=jnp.asarray(0.1 * rng.standard_normal((16, lm.N)) / np.sqrt(lm.N)),
                   c=jnp.asarray(0.05 * rng.standard_normal(16)))
        p_k0 = {"theta": th_q, "W": jnp.zeros((0, lm.N)), "b": jnp.zeros((0,))}
        p_is = {"theta": th_q, "W": p_q["W"][:3], "b": p_q["b"][:3], "wl": p_q["wl"], "bl": p_q["bl"]}
        Xq = sample_forward_process(float(taus[60]), 64, T, lm.SIGMA_FORWARD, prob4["data"], k7, beta=1)
        vis_d = lm.DenseVisible(conn28)
        vis_st = lm.make_visible(conn28, lm.N, "stencil", ("lin",), sextic=False)
        vis_st_o = lm_o.make_visible(conn28, lm.N, "stencil", ("lin",), sextic=False)
        cases = [("K=0 dense f64", p_k0, lambda M, v: M.make_loss(v, T, 0), "dense", None),
                 ("Potts 8 quad local, stencil f64", p_q, lambda M, v: M.make_loss(v, T, 8, "potts", True), "stencil", None),
                 ("Potts 8 quad local, stencil mixed", p_q,
                  lambda M, v: M.make_loss(v, T, 8, "potts", True, compute_dtype=jnp.float32), "stencil", None),
                 ("Ising 3 local, dense f64", p_is, lambda M, v: M.make_loss(v, T, 3, "ising", True), "dense", None),
                 ("Potts 8 quad local, sampled S = 8, jackknife", p_q,
                  lambda M, v: M.make_loss(v, T, 8, "potts", True, samples=8, jackknife=True), "stencil", jr.PRNGKey(5)),
                 ("per-sample loss, Potts 8 quad local, dense f64", p_q,
                  lambda M, v: (lambda pp, XX, key=None: M.make_loss_per_sample(v, T, 8, "potts", True)(pp, XX)), "dense",
                  None),
                 ("Potts 8 quad local + 16 harmonic modes, stencil mixed", p_h,
                  lambda M, v: M.make_loss(v, T, 8, "potts", True, compute_dtype=jnp.float32), "stencil", None)]
        bad, lines = [], []
        for lab, pp, mk, impl, key in cases:
            vn, vo = (vis_d, lm_o.DenseVisible(conn28)) if impl == "dense" else (vis_st, vis_st_o)
            fn_n, fn_o = mk(lm, vn), mk(lm_o, vo)
            if lab.startswith("per-sample"):
                ok = _same17(jax.jit(fn_n)(pp, Xq), jax.jit(fn_o)(pp, Xq))
            else:
                (Ln, Gn), (Lo, Go) = (jax.jit(jax.value_and_grad(f_))(pp, Xq, key) for f_ in (fn_n, fn_o))
                ok = _same17(Ln, Lo) and all(_same17(Gn[g], Go[g]) for g in Gn) and sorted(Gn) == sorted(Go)
            jx_n = str(jax.make_jaxpr(fn_n)(pp, Xq, key) if key is not None else jax.make_jaxpr(fn_n)(pp, Xq))
            jx_o = str(jax.make_jaxpr(fn_o)(pp, Xq, key) if key is not None else jax.make_jaxpr(fn_o)(pp, Xq))
            lines.append(f"{lab}: values {'equal' if ok else 'DIFFER'}, jaxpr {'equal' if jx_n == jx_o else 'DIFFERS'}")
            bad += [lab] if not (ok and jx_n == jx_o) else []
        report("   19D default loss paths, new vs before --patch: " + "; ".join(lines))
        check(f"19D default loss paths without patches = the module before --patch: loss and every gradient bit for bit, "
              f"the same jaxpr ({len(cases)} cases incl. 16 harmonic modes in mixed; failures {bad})", len(bad), 0.5)
        bad = []
        for lab, pp in (("Potts 8 quad local", p_q), ("+ 16 modes", p_h)):
            fn_ = [jax.jit(M.make_force(v, T, 8, "potts", True)) for M, v in ((lm, vis_d), (lm_o, lm_o.DenseVisible(conn28)))]
            bad += [lab] if not _same17(fn_[0](pp, Xq), fn_[1](pp, Xq)) else []
        check(f"19D default make_force (Potts 8 quad local, with and without 16 modes) = the module before --patch, bit "
              f"for bit (failures {bad})", len(bad), 0.5)
        prob8 = lm.make_problem("gauss2_8", 1)
        vis_8st = lm.make_visible(prob8["conn"], n8, "stencil", ("lin",), sextic=False)
        zs8 = lm.dropped_pair_slices(n8, P8, ("lin",))
        th0 = lm.quartic_initial_theta(lm.paper_theta_layout(P8, n8)[0], n8, P8)
        for s_ in zs8:
            th0 = th0.at[s_].set(0.0)
        p8 = {"theta": th0, "W": jnp.asarray(0.05 * rng.standard_normal((4, n8))),
              "b": jnp.asarray(0.02 * rng.standard_normal(4)), "Q": jnp.zeros((4, n8)),
              "wl": jnp.asarray(0.1 * rng.standard_normal(n8)), "bl": jnp.asarray(0.01 * rng.standard_normal(n8))}
        lrs8 = {"theta": 1e-3, "W": 1e-5, "b": 1e-5, "Q": 1e-5, "wl": 1e-5, "bl": 1e-5}
        loss8 = lm.make_loss(vis_8st, T, 4, "potts", True, compute_dtype=jnp.float32)
        bad = []
        for lean in ({}, {"theta_masks": True, "chunk_batches": True, "track_best": False}):
            outs = []
            for M in (tl, tl_o):
                run_, adam_ = M.build_runner(loss8, prob8["data"], n8, 1000.0, "quartic", P8, zero_slices=zs8, **lean)
                c_ = M.init_carry(p8, adam_, jr.PRNGKey(8))
                for _ in range(2):
                    c_, _l = run_(c_, 4.0, lrs8, 50)
                outs.append(jax.tree_util.tree_leaves(c_))
            bad += [f"lean {bool(lean)}"] if not all(_same17(a, b) for a, b in zip(*outs)) else []
        check(f"19D the runner (default and lean, 2 x 50 steps) = the module before --patch, bit for bit (failures {bad})",
              len(bad), 0.5)
        # make_drift and load_params on synthetic runs without patches: no modes (synthetic_run), 3 modes
        run_d = os.path.join(OUT, "b19_synth_d")
        synthetic_run(run_d, K=4, seed=1906)
        h8 = _synth_hist17(n8, conn8, 4, 3, 1907)
        run_h = os.path.join(OUT, "b19_synth_dh")
        _write_run17(run_h, h8, {"problem": "gauss2_8", "K": 4, "hidden": "potts", "quad": True, "local": True,
                                 "visible": "quartic", "pair_terms": "lin", "radius": 1, "neighbourhood": "chebyshev",
                                 "cls": "none", "harmonic": 3})
        bad = []
        xs = jnp.asarray(0.3 * rng.standard_normal((6, n8)))
        tt = jnp.asarray([0.0, 0.01, 1.234567, 2.5, 4.97, 5.0])
        for rd in (run_d, run_h):
            name_ = os.path.relpath(rd, lm.OUT_ROOT)
            out_n, out_o = sl.load_params(name_, with_harmonic=True, with_patch=True), sl_o.load_params(name_,
                                                                                                       with_harmonic=True)
            prov_n, prov_o = dict(out_n[4]), dict(out_o[4])
            ok = (all(_same17(a, b) for a, b in zip(out_n[:3], out_o[:3])) and out_n[3] == out_o[3]
                  and out_n[5] == out_o[5] and sorted(prov_n) == sorted(prov_o) and "patch" not in prov_n
                  and "patch_units" not in prov_n)
            arrs = {}
            for g in ("local", "quad", "harmonic"):
                if g in prov_n:
                    a_n, a_o = prov_n.pop(g), prov_o.pop(g)
                    a_n, a_o = (a_n if isinstance(a_n, tuple) else (a_n,)), (a_o if isinstance(a_o, tuple) else (a_o,))
                    ok = ok and all(_same17(a, b) for a, b in zip(a_n, a_o))
                    arrs[g] = a_n if len(a_n) > 1 else a_n[0]
            ok = ok and prov_n == prov_o
            bad += [f"load_params {os.path.basename(rd)}"] if not ok else []
            th_, W_, b_ = out_n[:3]
            for smp in ("sde", "ode"):
                dn, _ = sl.make_drift(th_, W_, b_, smp, "gauss2_8", "potts", arrs.get("local"), arrs.get("quad"), 1,
                                      "chebyshev", harmonic=arrs.get("harmonic"))
                do, _ = sl_o.make_drift(th_, W_, b_, smp, "gauss2_8", "potts", arrs.get("local"), arrs.get("quad"), 1,
                                        "chebyshev", harmonic=arrs.get("harmonic"))
                dn_j, do_j = jax.jit(jax.vmap(dn, in_axes=(0, 0, None))), jax.jit(jax.vmap(do, in_axes=(0, 0, None)))
                bad += [f"make_drift {smp} {os.path.basename(rd)}"] if not _same17(dn_j(tt, xs, None),
                                                                                   do_j(tt, xs, None)) else []
        check(f"19D load_params(run, with_harmonic=True, with_patch=True) on runs without patches (without and with 3 "
              f"modes) = the module before --patch (arrays bit for bit, provenance equal, no 'patch' key); make_drift "
              f"(SDE and PF-ODE, 6 times) bit for bit (failures {bad})", len(bad), 0.5)
        del lm_o, tl_o, sl_o

    # ---- E. the reference on mnist28_d0123 and the de-risk step --------------------------------------------------
    if "E" in parts:
        prob4 = lm.make_problem("mnist28_d0123", 1)
        data = np.asarray(prob4["data"], dtype=np.float64)
        labels = prob4["labels"]
        g73 = lm.PatchGeometry(28, 7, 3)
        clus1 = lm.patch_clusters(data, g73, 32, lm.PATCH_SEED0)

        class _Sub:                                # patches 0..27 of 7:3 only (seed 2: the bit-for-bit check)
            P, idx = 28, g73.idx[:28]

        clus2 = lm.patch_clusters(data, _Sub, 32, lm.PATCH_SEED0 + 1000)
        cnt = clus1["counts"]
        report(f"   19E k-means of the 64 patch sets (7:3:32, seeds 20000 + p): {clus1['seconds']:.0f} s; patches 0-27 "
               f"with seeds 21000 + p: {clus2['seconds']:.0f} s; explained variance median "
               f"{np.median(clus1['explained_variance']):.3f} (min {np.min(clus1['explained_variance']):.3f}); empty "
               f"clusters {int(np.sum(cnt == 0))}, < 20 images {int(np.sum(cnt < 20))} (judge 4231616: 0 and 19)")
        t4 = taus[[0, 31, 60, 99]]
        bad = []
        for cl, seed0 in ((clus1, lm.PATCH_SEED0), (clus2, lm.PATCH_SEED0 + 1000)):
            Wr, br = lm.patch_reference(cl, t4, T)
            for p_ in (0, 27):
                W_lm, b_lm, _ = lm.prototype_reference(data[:, g73.idx[p_]], labels, t4, T, 32, seed=seed0 + p_,
                                                       pooled=True)
                if not (_same17(Wr[:, p_], W_lm) and _same17(br[:, p_], b_lm)):
                    bad.append(f"seed0 {seed0} patch {p_}: W {rel(Wr[:, p_], W_lm):.1e}, b {rel(br[:, p_], b_lm):.1e}")
        check(f"19E patch_reference(patch_clusters) = lm.prototype_reference(data[:, idx_p], pooled=True, seed = seed0 + p) "
              f"bit for bit (patches 0 and 27, run seeds 1 and 2, slices 0, 31, 60, 99; failures {bad})", len(bad), 0.5)
        # the scale lam against a hand computation (patch 27, slice 50, lam 0.06), and lam W(1) = W(lam)
        t50, lam = float(taus[50]), 0.06
        C, Sw, pi = clus1["C"][27], clus1["Sw"][27], clus1["pi"][27]
        e = np.exp(-t50)
        mu = e * C
        Ws = T * mu @ np.linalg.inv(e * e * Sw + T * (1.0 - e * e) * np.eye(49))
        W_h = lam * (Ws - pi @ Ws)
        b0_ = T * np.log(pi) - (T * np.log(pi)) @ pi
        b1_ = -0.5 * (np.sum(Ws * mu, axis=1) - np.sum(Ws * mu, axis=1) @ pi)
        W_l, b_l = (a[0, 27] for a in lm.patch_reference(clus1, [t50], T, scale=lam))
        W_1, b_1 = (a[0, 27] for a in lm.patch_reference(clus1, [t50], T))
        errs = [rel(W_l, W_h), rel(b_l, b0_ + lam * b1_), rel(W_l, lam * W_1), rel(b_1, b0_ + b1_)]
        check(f"19E patch_reference(scale = lam) = lam (W* - pi W*), b = centred T log pi - lam W*.mu / 2 by a hand "
              f"computation (inverse instead of solve; patch 27, slice 50, lam 0.06), and = lam x the scale-1 templates "
              f"(worst relative; {', '.join(f'{v:.1e}' for v in errs)})", max(errs), 1e-10)
        # the learning-rate scales
        sWp, sbp = lm.patch_lr_scales(clus1, taus, T)
        bad = [] if (np.all(np.isfinite(sWp)) and np.all(sWp > 0) and np.all(np.isfinite(sbp)) and np.all(sbp >= T)
                     and sWp.shape == (100, 64) and sbp.shape == (100, 64)) else ["not finite / positive / shapes"]
        for k in (0, 31, 45, 60, 99):
            W_k, b_k = lm.patch_reference(clus1, [taus[k]], T)
            bad += [f"slice {k}"] if not (
                _same17(sWp[k], np.sqrt(np.mean(W_k[0] ** 2, axis=(1, 2))))
                and _same17(sbp[k], np.maximum(np.sqrt(np.mean(b_k[0] ** 2, axis=1)), T))) else []
        meds = {k: 5e-4 * float(np.median(sWp[k])) for k in (31, 45, 60)}
        spec_meds = {31: 1.6e-5, 45: 2.7e-5, 60: 4.0e-5}
        report(f"   19E lr_Wp medians 5e-4 x rms: " + ", ".join(f"slice {k} {v:.3g} (spec {spec_meds[k]:.2g})"
                                                         for k, v in meds.items())
               + f"; per-patch spread (max/min) at slices 0 / 31 / 60 / 80 / 99: "
               + ", ".join(f"{sWp[k].max() / sWp[k].min():.2f}" for k in (0, 31, 60, 80, 99)))
        check(f"19E patch_lr_scales: finite, positive, = the rms of each patch's scale-1 W* (b*: floored at T) bit for bit "
              f"at slices 0, 31, 45, 60, 99 (failures {bad})", len(bad), 0.5)
        check("19E the lr_Wp medians at slices 31 / 45 / 60 = the spec's 1.6e-5 / 2.7e-5 / 4.0e-5 (two digits; worst "
              "relative)", max(abs(meds[k] / spec_meds[k] - 1.0) for k in meds), 0.04)
        # the de-risk step: the integrated functions reproduce the judge's loss changes (job 4231616) to 1e-6
        roots = [os.environ.get("PATCH_PLAIN_ROOT", ""), lm.OUT_ROOT, "/physical-diffusion-models/out/latent_physical_SGM"]
        plain = next((os.path.join(r_, PLAIN19) for r_ in roots if r_ and os.path.exists(
            os.path.join(r_, PLAIN19, "slices", "slice_060.npz"))), None)
        if plain is None:
            report(f"   WARNING: the plain seed-1 run {PLAIN19} is absent: the end-to-end check of 19E is skipped")
        else:
            vis = lm.make_visible(prob4["conn"], lm.N, "stencil", ("lin",), sextic=False)
            f_ps = jax.jit(lm.make_loss_per_sample(vis, T, 784, "potts", True))
            rng_e = np.random.default_rng(1807)
            res, lines = {}, []
            for k in (0, 50, 60):
                t = float(taus[k])
                e = math.exp(-t)                  # as init_scale_check.py draws them
                pick = rng_e.choice(data.shape[0], size=4096, replace=True)
                X = jnp.asarray(e * data[pick] + math.sqrt(T * (1 - e * e)) * rng_e.standard_normal((4096, lm.N)))
                with np.load(os.path.join(plain, "slices", f"slice_{k:03d}.npz")) as z:
                    params = {g: jnp.asarray(z[g]) for g in ("theta", "W", "b", "Q", "wl", "bl")}
                L0 = np.asarray(f_ps(params, X))
                for (kk, lam), want in E2E19.items():
                    if kk != k:
                        continue
                    Wl, bl_ = lm.patch_reference(clus1, [t], T, scale=lam)
                    L = np.asarray(f_ps(dict(params, Wp=jnp.asarray(Wl[0]), bp=jnp.asarray(bl_[0])), X))
                    d_ = L - L0
                    res[(k, lam)] = float(np.mean(d_))
                    lines.append(f"slice {k} lam {lam:g}: {res[(k, lam)]:+.10g} +- {np.std(d_) / np.sqrt(d_.size):.2g} "
                                 f"(judge {want:+.10g}; relative difference "
                                 f"{abs(res[(k, lam)] - want) / abs(want):.1e})")
            report(f"   19E de-risk, the plain seed-1 slices ({plain}) + lam x the patch reference, 4096 draws of "
                   f"init_scale_check.py (rng 1807):\n      " + "\n      ".join(lines))
            check("19E de-risk: lm.patch_clusters / patch_reference / make_loss_per_sample reproduce the judge's mean loss "
                  "changes (job 4231616: +0.1891 at slice 0 lam 0.25, -1.874 and +22.22 at slice 50 lam 0.04 / 0.1, "
                  "+106.9 at slice 60 lam 0.1) to 1e-6 relative (worst)",
                  max(abs(res[key] - want) / abs(want) for key, want in E2E19.items()) if len(res) == len(E2E19)
                  else 1.0, 1e-6)

    # ---- F. the runner and the sampler with patches -------------------------------------------------------------
    if "F" in parts:
        prob8 = lm.make_problem("gauss2_8", 1)
        vis_8st = lm.make_visible(prob8["conn"], n8, "stencil", ("lin",), sextic=False)
        zs8 = lm.dropped_pair_slices(n8, P8, ("lin",))
        th0 = lm.quartic_initial_theta(lm.paper_theta_layout(P8, n8)[0], n8, P8)
        for s_ in zs8:
            th0 = th0.at[s_].set(0.0)
        Sp = 4
        p8p = {"theta": th0, "W": jnp.asarray(0.05 * rng.standard_normal((4, n8))), "b": jnp.zeros(4),
               "Q": jnp.zeros((4, n8)), "wl": jnp.zeros(n8), "bl": jnp.zeros(n8),
               "Wp": jnp.asarray(0.03 * rng.standard_normal((geom8.P, Sp, geom8.d))),
               "bp": jnp.asarray(0.01 * rng.standard_normal((geom8.P, Sp))), "Qp": jnp.zeros((geom8.P, Sp, geom8.d))}
        loss8p = lm.make_loss(vis_8st, T, 4, "potts", True, compute_dtype=jnp.float32)
        bad = []
        for lean in ({}, {"theta_masks": True, "chunk_batches": True, "track_best": False}):
            ntrace = [0]

            def loss_c(params, X, key=None, ntrace=ntrace):
                ntrace[0] += 1
                return loss8p(params, X)

            run_, adam_ = tl.build_runner(loss_c, prob8["data"], n8, 1000.0, "quartic", P8, zero_slices=zs8, **lean)
            base_lr = {"theta": 1e-3, "W": 1e-5, "b": 1e-5, "Q": 1e-5, "wl": 1e-5, "bl": 1e-5}
            # built as train_latent.py builds them (jnp.asarray of numpy float64: strongly typed; jnp.full with a
            # Python scalar would be weakly typed and trace the runner again)
            lr_on = dict(base_lr, Wp=jnp.asarray(1e-4 * np.ones((geom8.P, 1, 1))), bp=jnp.asarray(1e-4 * np.ones((geom8.P, 1))),
                         Qp=jnp.asarray(1e-4 * np.ones((geom8.P, 1, 1))))
            lr_off = dict(base_lr, Wp=jnp.asarray(np.zeros((geom8.P, 1, 1))), bp=jnp.asarray(np.zeros((geom8.P, 1))),
                          Qp=jnp.asarray(np.zeros((geom8.P, 1, 1))))
            c_off, _ = run_(tl.init_carry(p8p, adam_, jr.PRNGKey(9)), 3.0, lr_off, 100)
            n1 = ntrace[0]
            c_on, _ = run_(tl.init_carry(p8p, adam_, jr.PRNGKey(9)), 2.5, lr_on, 100)
            n2 = ntrace[0]
            moved = [g for g in ("Wp", "bp", "Qp") if not _same17(c_on[0][g], p8p[g])]
            fin = all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(c_on))
            kept = all(_same17(c_off[i][g], p8p[g]) for i in (0, 4, 7) for g in ("Wp", "bp", "Qp"))
            others = not _same17(c_off[0]["theta"], p8p["theta"])
            report(f"   19F runner {'lean' if lean else 'default'}: traces {n1} then {n2}; moved {moved}; finite {fin}; "
                   f"kept at lr 0 {kept}")
            bad += [f"lean {bool(lean)}"] if not (moved == ["Wp", "bp", "Qp"] and fin and kept and others and n2 == n1
                                                  and n1 >= 1) else []
        check(f"19F the runner with patch keys and per-patch learning-rate arrays (default and lean): lr > 0 moves Wp, bp, "
              f"Qp (finite); lr 0 keeps them bit for bit in the iterate, best and EMA; one trace across learning rates and "
              f"t (failures {bad})", len(bad), 0.5)
        # make_drift with patches (and modes) against x - s grad F by autodiff
        efn = setup_energy_fn(jnp.asarray(conn8), lm.paper_theta_layout(P8, n8)[1])

        def F_samp(x, q):
            a = q["W"] @ x + q["Q"] @ (x * x) + q["b"]
            al = q["wl"] * x + q["bl"]
            f = efn(x, q["theta"]) - T * jax.scipy.special.logsumexp(a / T) - T * jnp.sum(jnp.logaddexp(al / T, -al / T))
            if "V" in q:
                f = f - 0.5 * jnp.sum((q["V"] @ x + q["c"]) ** 2)
            return f + lm.patch_free_energy(q["Wp"], q.get("Qp"), q["bp"], x[None], T)[0]

        worst = 0.0
        for R, quad in ((3, True), (0, False)):
            h = _synth_hist19(n8, conn8, 4, R, geom8, Sp, 1908 + R, quad=quad)
            harm = (h["V"], h["c"]) if R else None
            patch_h = (h["Wp"], h["bp"], h.get("Qp"))
            for smp, s_ in (("sde", 2.0), ("ode", 1.0)):
                dr, _ = sl.make_drift(h["theta"], h["W"], h["b"], smp, "gauss2_8", "potts", (h["wl"], h["bl"]), h["Q"], 1,
                                      "chebyshev", harmonic=harm, patch=patch_h)
                for t_rev in (0.0, 0.7777, 2.5, 4.9999):
                    tau = lm.T_FORWARD - t_rev
                    k = min(int(np.searchsorted(-taus, -tau, side="right")) - 1, lm.N_TIME_STEPS - 2)
                    lw = (tau - taus[k + 1]) / (taus[k] - taus[k + 1])
                    q = {g: jnp.asarray(h[g][k + 1] + lw * (h[g][k] - h[g][k + 1]))
                         for g in ("theta", "W", "b", "Q", "wl", "bl", "V", "c", "Wp", "bp", "Qp") if g in h}
                    x = jnp.asarray(0.4 * rng.standard_normal(n8))
                    worst = max(worst, rel(dr(t_rev, x, ()), x - s_ * jax.grad(F_samp)(x, q)))
        check("19F make_drift with patches 4:2:4 (Potts 4 quad local + 3 modes + Qp; and without modes and Qp; SDE and "
              "PF-ODE, 4 times between knots) vs x - s grad F by autodiff with interpolated parameters (worst relative)",
              worst, 1e-10)
        h = _synth_hist19(n8, conn8, 4, 0, geom8, Sp, 1909)
        run_fp = os.path.join(OUT, "b19_synth_fp")
        _write_run17(run_fp, h, {"problem": "gauss2_8", "K": 4, "hidden": "potts", "quad": True, "local": True,
                                 "visible": "quartic", "pair_terms": "lin", "radius": 1, "neighbourhood": "chebyshev",
                                 "cls": "none", "patch": "4:2:4"})
        name_fp = os.path.relpath(run_fp, lm.OUT_ROOT)
        refused = []
        for kw in ({}, {"with_harmonic": True}):
            try:
                sl.load_params(name_fp, **kw)
                refused.append("")
            except RuntimeError as err:
                refused.append(str(err))
        out_p = sl.load_params(name_fp, with_harmonic=True, with_patch=True)
        pv = out_p[4]
        ok = (all("per-patch Potts units" in r_ for r_ in refused) and "patch" in pv and pv.get("patch_units") == "4:2:4"
              and _same17(pv["patch"][0], h["Wp"]) and _same17(pv["patch"][1], h["bp"]) and _same17(pv["patch"][2], h["Qp"]))
        code, _e = _main17(sl, ["--run", name_fp, "--fast", "--n", "4", "--chunk", "4"])
        fast_ok = isinstance(code, str) and "--fast does not support runs with patch units" in code
        report(f"   19F load_params refusal: {refused[0][:150]}...; --fast: {str(code)[:150]}")
        check("19F load_params refuses a patch run unless with_patch=True (then prov['patch'] = Wp, bp, Qp bit for bit, "
              "patch_units '4:2:4'); sample_latent --fast refuses it", 0.0 if ok and fast_ok else 1.0, 0.5)

    # ---- G. command lines --------------------------------------------------------------------------------------
    if "G" in parts:
        guard_root = os.path.join(OUT, "b19_guards")
        base = ["--name", "g", "--out-root", guard_root, "--problem", "gauss2_8"]
        pk = ["--K", "4", "--hidden", "potts"]
        cases = [(["--patch", s_], f"--patch '{s_}'") for s_ in ("7:3:32", "4:3:4", "4:5:4", "4:2:1", "4:2", "0:1:4",
                                                                 "9:1:4")]
        cases += [(pk + PF + ["--hidden-samples", "4"], "--patch needs exact"),
                  (pk + PF + ["--hidden-rule", "cd1"], "--patch needs exact"),
                  (pk + PF + ["--hidden-norm", "group"], "--patch needs exact"),
                  (pk + PF + ["--hidden-norm", "state"], "--patch needs exact"),
                  (pk + PF + ["--hidden-norm", "occupancy"], "--patch needs exact"),
                  (pk + PF + ["--hidden-fixed"], "--patch needs exact"),
                  (PF + ["--cls", "0"], "--patch needs exact"),
                  (PF + ["--patch-lr-scale", "0"], "--patch-lr-scale 0.0: must be finite and > 0"),
                  (PF + ["--patch-lr-scale", "nan"], "--patch-lr-scale nan: must be finite and > 0"),
                  (PF + ["--patch-init-scale", "0"], "--patch-init-scale 0.0: must be finite and > 0"),
                  (PF + ["--patch-init-scale", "-1"], "--patch-init-scale -1.0: must be finite and > 0"),
                  (PF + ["--patch-init-scale", "nan"], "--patch-init-scale nan: must be finite and > 0"),
                  (["--patch-init-scale", "0.25"], "need --patch"), (["--patch-lr-scale", "2"], "need --patch"),
                  (["--patch-on-slice", "3"], "need --patch"),
                  (PF + ["--patch-on-slice", "99", "--patch-init-scale", "0.06"], "--patch-on-slice 99: must lie in 0..98"),
                  (PF + ["--patch-on-slice", "-1", "--patch-init-scale", "0.06"], "--patch-on-slice -1: must lie in 0..98"),
                  (PF + ["--patch-on-slice", "3"], "needs an explicit --patch-init-scale")]
        bad = []
        for argv, text in cases:
            code, _e = _main17(tl, base + argv)
            if not (isinstance(code, str) and code.startswith("ValueError") and text in code):
                bad.append((" ".join(argv), code))
        check(f"19G train_latent guards: invalid --patch specs (7:3:32 on 8x8, 4:3:4, 4:5:4, 4:2:1, 4:2, 0:1:4, 9:1:4); "
              f"--patch with --hidden-samples, cd1, --hidden-norm group / state / occupancy, --hidden-fixed, --cls 0; "
              f"--patch-lr-scale 0 / nan; --patch-init-scale 0 / -1 / nan; the other patch options without --patch; "
              f"--patch-on-slice 99 / -1, and > 0 without --patch-init-scale: refused before writing ({len(cases)} cases; "
              f"failures {bad}; no folder: {not os.path.exists(guard_root)})", len(bad) + int(os.path.exists(guard_root)),
              0.5)

        def train19(script, name, *extra, flags=FL, kill_after=None):
            """train_latent.py (script) in a subprocess (as train17). Returns (exit code, this call's log text)."""
            cmd = tcmd(script, name, *extra, flags=flags)
            logp = os.path.join(OUT, f"b19_train_{name}.log")
            pos = os.path.getsize(logp) if os.path.exists(logp) else 0
            with open(logp, "a") as fo:
                fo.write("$ " + shlex.join(cmd[2:]) + "\n")
                fo.flush()
                p_ = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, env=env)
                if kill_after is not None:
                    target = os.path.join(runs19, name, "slices", f"slice_{kill_after:03d}.npz")
                    while p_.poll() is None and not os.path.exists(target):
                        time.sleep(0.05)
                    if p_.poll() is None:
                        p_.kill()
                rc_ = p_.wait()
            with open(logp) as f:
                f.seek(pos)
                return rc_, f.read()

        def records19(name):
            with open(os.path.join(runs19, name, "train_log.jsonl")) as f:
                return {int(r["slice"]): r for r in (json.loads(ln) for ln in f if ln.strip())}

        def cfg19(name):
            with open(os.path.join(runs19, name, "config.json")) as f:
                return json.load(f)

        def zslice(name, k):
            with np.load(os.path.join(runs19, name, "slices", f"slice_{k:03d}.npz")) as z:
                return {m: z[m] for m in z.files}

        prob8 = lm.make_problem("gauss2_8", 1)
        clus8 = lm.patch_clusters(np.asarray(prob8["data"], dtype=np.float64), geom8, 4, lm.PATCH_SEED0)
        sW8, sb8 = lm.patch_lr_scales(clus8, taus, T)

        def lr_ok(rec, k, sW=sW8, sb=sb8, f=1.0):
            want_w = [5e-4 * f * v for v in (np.min(sW[k]), np.median(sW[k]), np.max(sW[k]))]
            want_b = [5e-4 * f * v for v in (np.min(sb[k]), np.median(sb[k]), np.max(sb[k]))]
            return (all(abs(a / b - 1.0) < 1e-12 for a, b in zip(rec["lr_Wp"], want_w))
                    and all(abs(a / b - 1.0) < 1e-12 for a, b in zip(rec["lr_bp"], want_b)))

        two = ["--max-slices", "2", "--steps-scale", "0.01"]
        if have_orig:                              # the default path, old vs new train_latent.py (2 slices, study + lean)
            rco, _ = train19(tl_old, "d_study_old", "--step-impl", "lean", *two)
            rcn, _ = train19(tl_cur, "d_study_new", "--step-impl", "lean", *two)
            diff = _slices_differ17(os.path.join(runs19, "d_study_old"), os.path.join(runs19, "d_study_new"), (0, 1))
            co, cn = cfg19("d_study_old"), cfg19("d_study_new")
            new_keys = {k: cn[k] for k in set(cn) - set(co)}
            cdiff = sorted(k for k in set(co) & set(cn) if k not in ("name", "sha256", "started") and co[k] != cn[k])
            report(f"   default path (study + lean, 2 slices): exit codes {rco}/{rcn}; slice differences {diff}; config "
                   f"differences {cdiff}, new keys {new_keys}")
            check(f"19G default training through the command line (study flags + lean, 2 slices) = the module before "
                  f"--patch: slice files bit for bit (records but for timing), config.json the same but for the four new "
                  f"keys at their inactive values", 0.0 if (rco == 0 and rcn == 0 and not diff and not cdiff
                                                            and set(co) <= set(cn) and new_keys == PATCH_NEW_KEYS) else 1.0,
                  0.5)
        # resume after SIGKILL, strict resume, --init-from
        six = PF + ["--max-slices", "6", "--steps-scale", "0.01"]
        rc_f, _ = train19(tl_cur, "p_full", *six)
        rc_k, _ = train19(tl_cur, "p_kill", *six, kill_after=2)
        n_k = sum(os.path.exists(os.path.join(runs19, "p_kill", "slices", f"slice_{k:03d}.npz")) for k in range(6))
        rc_r, txt_r = train19(tl_cur, "p_kill", *six)
        diff = _slices_differ17(os.path.join(runs19, "p_full"), os.path.join(runs19, "p_kill"), range(6))
        check(f"19G --patch 4:2:4 resumed after SIGKILL (exit {rc_k} with {n_k} slices; resumed: exit {rc_r}) = an "
              f"uninterrupted run, slices 0-5 bit for bit (exit {rc_f}; differences {diff})",
              0.0 if rc_f == 0 and rc_k != 0 and rc_r == 0 and n_k < 6 and not diff and "resuming at slice" in txt_r
              else 1.0, 0.5)
        bad = []
        for extra in (["--patch", "4:2:8"], PF + ["--patch-init-scale", "0.5"],
                      PF + ["--patch-on-slice", "2", "--patch-init-scale", "0.25"], []):
            rc, txt = train19(tl_cur, "p_full", *(extra + ["--max-slices", "6", "--steps-scale", "0.01"]))
            bad += [" ".join(extra) or "no --patch"] if not (rc != 0 and "different config" in txt) else []
        check(f"19G resuming a --patch 4:2:4 run with another spec, init scale, on-slice or no --patch is refused "
              f"(failures {bad})", len(bad), 0.5)
        bad = []
        if have_orig:
            rc, txt = train19(tl_cur, "if_nop", "--init-from", "d_study_new", "--init-slices", "2", *PF, "--step-impl",
                              "lean", "--max-slices", "3", "--steps-scale", "0.01")
            bad += ["source without patches accepted"] if not (rc != 0 and "different model config" in txt) else []
        rc, txt = train19(tl_cur, "if_same", "--init-from", "p_full", "--init-slices", "3", *six)
        links = [os.path.islink(os.path.join(runs19, "if_same", "slices", f"slice_{k:03d}.npz")) for k in range(6)]
        diff = _slices_differ17(os.path.join(runs19, "p_full"), os.path.join(runs19, "if_same"), range(3, 6))
        bad += [f"same patches: exit {rc}, links {links}, differences {diff}"] if not (
            rc == 0 and links == [True] * 3 + [False] * 3 and not diff) else []
        rc, txt = train19(tl_cur, "if_none", "--init-from", "p_full", "--init-slices", "3", "--max-slices", "6",
                          "--steps-scale", "0.01")
        bad += ["no patches from a source with patches accepted"] if not (rc != 0 and "different model config" in txt) \
            else []
        rc, txt = train19(tl_cur, "if_lr2", "--init-from", "p_full", "--init-slices", "3", *PF, "--patch-lr-scale", "2",
                          "--max-slices", "4", "--steps-scale", "0.01")
        ok = rc == 0
        if ok:
            ra, rb = records19("p_full")[3], records19("if_lr2")[3]
            ok = all(abs(b / a - 2.0) < 1e-12 for a, b in zip(ra["lr_Wp"] + ra["lr_bp"], rb["lr_Wp"] + rb["lr_bp"]))
        bad += [f"--patch-lr-scale 2: exit {rc}"] if not ok else []
        check(f"19G --init-from: a source without patches is refused for --patch 4:2:4, a source with the same patches "
              f"continues bit for bit (slices 0-2 linked, 3-5 = the source's), without --patch refused, --patch-lr-scale 2 "
              f"accepted with doubled learning rates (failures {bad})", len(bad), 0.5)

        # ---- the background runs ------------------------------------------------------------------------------
        rcs = {}
        for tag, (p_, t_p, logp) in bg.items():
            try:
                rcs[tag] = p_.wait(timeout=max(60.0, 1600.0 - (time.time() - t19)))
            except subprocess.TimeoutExpired:
                p_.kill()
                rcs[tag] = "timeout"
            report(f"   background {tag}: exit code {rcs[tag]}, {time.time() - t_p:.0f} s after its start ({logp})")

        def hist_bytes(d_):
            import zipfile
            with open(os.path.join(d_, "params_history.npy"), "rb") as f:
                ph = f.read()
            with zipfile.ZipFile(os.path.join(d_, "hidden_history.npz")) as z:
                return ph, [nm[:-4] for nm in z.namelist()], {nm[:-4]: z.read(nm) for nm in z.namelist()}

        def npy_b(a):
            buf = io.BytesIO()
            np.save(buf, a)
            return buf.getvalue()

        if have_orig:                              # 100 slices, old vs new, both assembly modes
            for mode in ("files", "mem"):
                da, db = os.path.join(runs19, f"d100_old_{mode}"), os.path.join(runs19, f"d100_new_{mode}")
                ok = rcs.get(f"d100_old_{mode}") == 0 and rcs.get(f"d100_new_{mode}") == 0
                diff = _slices_differ17(da, db, range(lm.N_TIME_STEPS)) if ok else ["exit codes"]
                hdiff = []
                if ok:
                    pa, ma, za = hist_bytes(da)
                    pb, mb, zb = hist_bytes(db)
                    hdiff += [] if pa == pb else ["params_history.npy"]
                    hdiff += [] if ma == mb and all(za[m] == zb[m] for m in ma) else [f"hidden_history {ma} / {mb}"]
                    co, cn = cfg19(f"d100_old_{mode}"), cfg19(f"d100_new_{mode}")
                    new_keys = {k: cn[k] for k in set(cn) - set(co)}
                    cdiff = sorted(k for k in set(co) & set(cn) if k not in ("name", "sha256", "started") and co[k] != cn[k])
                    hdiff += [f"config {cdiff} {new_keys}"] if cdiff or new_keys != PATCH_NEW_KEYS else []
                check(f"19G 100-slice default runs (--steps-schedule 0:0.001, --assemble {'files' if mode == 'files' else 'memory'}), "
                      f"old vs new train_latent.py: every slice file and both history files bit for bit, config the same "
                      f"but for the four new keys (differences {diff[:5]} {hdiff})", len(diff) + len(hdiff), 0.5)

            def load_out(run_dir):
                p_ = os.path.join(run_dir, "samples", "sde_base_n4")
                if not (os.path.exists(p_ + ".npy") and os.path.exists(p_ + ".json")):
                    return None, None
                with open(p_ + ".json") as f:
                    return np.load(p_ + ".npy"), json.load(f)

            (xa, ma), (xb, mb) = load_out(run_a), load_out(run_b)
            ok = rcs.get("sdef_new") == 0 and rcs.get("sdef_old") == 0 and xa is not None and xb is not None
            skip = {"run", "chunk_seconds", "sha256_sample_latent"}
            dk = sorted(k for k in set(ma or {}) | set(mb or {}) if k not in skip and (ma or {}).get(k) != (mb or {}).get(k))
            check(f"19G default sampler through the command line (synthetic run, 4 trajectories): final states = the "
                  f"module before --patch bit for bit; manifests the same keys and values but for run, chunk_seconds and "
                  f"the module hash (differing: {dk})", 0.0 if ok and _same17(xa, xb) and not dk else 1.0, 0.5)
        # the 100-slice patch runs: lean + memory and lean + files
        dm, dfi = os.path.join(runs19, "p_mem"), os.path.join(runs19, "p_files")
        bad = [] if rcs.get("p_mem") == 0 and rcs.get("p_files") == 0 and os.path.exists(os.path.join(dm, "DONE")) \
            and os.path.exists(os.path.join(dfi, "DONE")) else [f"exit codes {rcs.get('p_mem')}/{rcs.get('p_files')}"]
        if not bad:
            bad += _slices_differ17(dm, dfi, range(lm.N_TIME_STEPS))[:5]
            pa, ma, za = hist_bytes(dm)
            pb, mb, zb = hist_bytes(dfi)
            want_m = ["W", "b", "t", "wl", "bl", "Q", "Wp", "bp", "Qp"]
            bad += [f"members {ma} / {mb}"] if not (ma == want_m and mb == want_m) else []
            bad += ["history bytes memory vs files"] if not (pa == pb and all(za[m] == zb.get(m) for m in ma)) else []
            stk = {g: np.stack([zslice("p_mem", k)[g] for k in range(lm.N_TIME_STEPS)]) for g in ("Wp", "bp", "Qp")}
            bad += [f"history {g} != stacked slices" for g in ("Wp", "bp", "Qp") if za.get(g) != npy_b(stk[g])]
            W0r, b0r = (a[0] for a in lm.patch_reference(clus8, taus[:1], T, scale=0.25))
            z0 = zslice("p_mem", 0)
            bad += ["slice 0 not 0.25 x the reference"] if not (_same17(z0["Wp"], W0r) and _same17(z0["bp"], b0r)
                                                                 and not np.any(z0["Qp"])) else []
            rr = records19("p_mem")
            bad += ["slice 0 lr"] if rr[0]["lr_Wp"] != [0.0, 0.0, 0.0] or rr[0]["lr_bp"] != [0.0, 0.0, 0.0] else []
            bad += [f"slice {k} lr" for k in range(1, lm.N_TIME_STEPS) if not lr_ok(rr[k], k)][:5]
            keys_ = ("patch_W_norm", "patch_cos_W_Wref", "patch_eff_states_median", "patch_states_below_1e-6",
                     "force_rms_patch", "force_rms_glob", "force_rms_loc", "force_rms_vis")
            bad += [f"slice {k} diagnostics" for k in range(lm.N_TIME_STEPS) if not all(q in rr[k] for q in keys_)][:3]
            n_moved = sum(not _same17(zslice("p_mem", k)["Wp"], zslice("p_mem", k - 1)["Wp"]) for k in range(1, 100))
            bad += [f"only {n_moved} of 99 slices moved Wp"] if n_moved < 99 else []
            with open(os.path.join(dm, "patch_reference.json")) as f:
                pr_ = json.load(f)
            bad += ["patch_reference.json"] if not (pr_.get("spec") == "4:2:4" and pr_.get("P") == 9 and pr_.get("d") == 16
                                                    and pr_.get("start_sha256") == _sha19(W0r, b0r)
                                                    and pr_.get("kmeans_seeds") == [20000, 20008]) else []
            with open(os.path.join(dm, "train_log.txt")) as f:
                bad += ["log line"] if "patch units 4:2:4: P 9, d 16, S 4" not in f.read() else []
            ks_ = (1, 30, 60, 99)
            report("   p_mem (lean, memory): per-slice patch diagnostics (|Wp| / ref, cos, eff median / min, dead, <max p>, "
                   "force patch / glob / loc / vis): " + "; ".join(
                       f"{k}: {rr[k]['patch_W_norm']:.3g}/{rr[k]['patch_Wref_norm']:.3g}, {rr[k]['patch_cos_W_Wref']:.3f}, "
                       f"{rr[k]['patch_eff_states_median']:.2f}/{rr[k]['patch_eff_states_min']:.2f}, "
                       f"{rr[k]['patch_states_below_1e-6']}, {rr[k]['patch_mean_max_p']:.3f}, {rr[k]['force_rms_patch']:.3g}"
                       f"/{rr[k]['force_rms_glob']:.3g}/{rr[k]['force_rms_loc']:.3g}/{rr[k]['force_rms_vis']:.3g}"
                       for k in ks_))
        check(f"19G --patch 4:2:4 100-slice runs, lean with --assemble memory and with files: slice files bit for bit, "
              f"history members W b t wl bl Q Wp bp Qp with the same bytes (= the stacked slices), slice 0 = 0.25 x the "
              f"reference bit for bit with lr 0, slices 1-99 trained with lr = 5e-4 x the per-patch rms, diagnostics, "
              f"patch_reference.json, log line (failures {bad[:6]})", len(bad), 0.5)
        # --patch-on-slice 3 against a plain run
        bad = [] if rcs.get("chainA") == 0 else [f"chain A exit {rcs.get('chainA')}"]
        if not bad:
            for k in range(3):
                za, zb = zslice("on3", k), zslice("plain5", k)
                for m in ("theta", "W", "b", "Q", "wl", "bl", "val_per_sample", "key", "loss_trace"):
                    bad += [f"slice {k} {m}"] if not _same17(za[m], zb[m]) else []
                bad += [f"slice {k} patch arrays"] if not (za["Wp"].shape == (9, 4, 16) and not np.any(za["Wp"])
                                                           and za["bp"].shape == (9, 4) and not np.any(za["bp"])
                                                           and za["Qp"].shape == (9, 4, 16) and not np.any(za["Qp"])) else []
                ra, rb = json.loads(str(za["record"])), json.loads(str(zb["record"]))
                d_ = sorted(f for f in set(ra) | set(rb) if f not in VOL17 and json.dumps(ra.get(f)) != json.dumps(rb.get(f)))
                bad += [f"slice {k} record {d_}"] if d_ else []
            W3, b3 = (a[0] for a in lm.patch_reference(clus8, taus[3:4], T, scale=0.06))
            with open(os.path.join(runs19, "on3", "patch_reference.json")) as f:
                pr3 = json.load(f)
            z3 = zslice("on3", 3)
            cos3 = float(np.sum(z3["Wp"] * W3) / (np.linalg.norm(z3["Wp"]) * np.linalg.norm(W3)))
            r3 = records19("on3")
            bad += ["start != 0.06 x the reference at t_3"] if pr3.get("start_sha256") != _sha19(W3, b3) else []
            bad += [f"slice 3 cos to the start {cos3:.4f}"] if not cos3 > 0.99 else []
            bad += ["slice 3 lr"] if not lr_ok(r3[3], 3) else []
            bad += ["slice 3/4 not trained"] if _same17(z3["Wp"], zslice("on3", 4)["Wp"]) else []
            c3 = cfg19("on3")
            bad += ["config"] if (c3.get("patch"), c3.get("patch_on_slice"), c3.get("patch_init_scale")) != (
                "4:2:4", 3, 0.06) else []
            with open(os.path.join(runs19, "on3", "train_log.txt")) as f:
                bad += ["switch-on log line"] if "switched on at 0.06 x the reference" not in f.read() else []
        check(f"19G --patch-on-slice 3 --patch-init-scale 0.06: slices 0-2 = a plain run's bit for bit (theta, W, b, Q, wl, "
              f"bl, val_per_sample, key, loss trace; records but for timing) with zero patch arrays; slice 3 starts at "
              f"0.06 x the reference at t_3 (patch_reference.json fingerprint) and trains with 5e-4 x the per-patch rms "
              f"(failures {bad})", len(bad), 0.5)
        # K = 0, Ising, --harmonic 2, --time-grid power:1.5, --seed 2 with patches
        bad = []
        for tag, chain, want, qp in (("k0p", "chainA", ["Wp", "bp"], False), ("isp", "chainB", ["Wp", "bp"], False),
                                     ("hp", "chainB", ["V", "c", "Wp", "bp", "Qp"], True)):
            if not os.path.exists(os.path.join(runs19, tag, "slices", "slice_001.npz")):
                bad.append(f"{tag}: missing (chain {chain} exit {rcs.get(chain)})")
                continue
            z0, z1 = zslice(tag, 0), zslice(tag, 1)
            W0r, b0r = (a[0] for a in lm.patch_reference(clus8, taus[:1], T, scale=0.25))
            ok = (all(m in z1 for m in want) and ("Qp" in z1) == qp and _same17(z0["Wp"], W0r) and _same17(z0["bp"], b0r)
                  and not _same17(z1["Wp"], z0["Wp"]) and all(np.all(np.isfinite(z1[m])) for m in want)
                  and lr_ok(records19(tag)[1], 1))
            bad += [tag] if not ok else []
        if os.path.exists(os.path.join(runs19, "grid", "slices", "slice_002.npz")):
            g_t = lm.time_grid_pts("power:1.5")
            sWg, sbg = lm.patch_lr_scales(clus8, g_t, T)
            rg = records19("grid")
            Wg0, bg0 = (a[0] for a in lm.patch_reference(clus8, g_t[:1], T, scale=0.25))
            bad += ["grid"] if not (all(lr_ok(rg[k], k, sWg, sbg) and abs(rg[k]["t"] - g_t[k]) < 1e-15 for k in (1, 2))
                                    and _same17(zslice("grid", 0)["Wp"], Wg0)) else []
        else:
            bad.append(f"grid missing (chain B exit {rcs.get('chainB')})")
        if os.path.exists(os.path.join(runs19, "seed2", "slices", "slice_001.npz")):
            r2, r1 = records19("seed2"), records19("p_full")
            with open(os.path.join(runs19, "seed2", "patch_reference.json")) as f:
                pr2 = json.load(f)
            bad += ["seed 2"] if not (r2[1]["lr_Wp"] == r1[1]["lr_Wp"] and r2[1]["lr_bp"] == r1[1]["lr_bp"]
                                      and not _same17(zslice("seed2", 0)["Wp"], zslice("p_full", 0)["Wp"])
                                      and pr2.get("kmeans_seeds") == [21000, 21008]
                                      and pr2.get("lr_kmeans_seeds") == [20000, 20008]) else []
        else:
            bad.append(f"seed2 missing (chain B exit {rcs.get('chainB')})")
        check(f"19G patches with K = 0 (sextic dense f64, no Qp), 3 Ising units, --harmonic 2 (slice members V c Wp bp Qp), "
              f"--time-grid power:1.5 (the grid's reference and learning rates) and --seed 2 (its own k-means seeds "
              f"21000 + p, the seed-1 learning rates): slice 0 = 0.25 x the reference, slice 1 trained, finite (failures "
              f"{bad})", len(bad), 0.5)
        # the default sampler on a (synthetic) patch run
        xs_, ms_ = None, None
        p_s = os.path.join(run_p, "samples", "sde_base_n4")
        if os.path.exists(p_s + ".npy") and os.path.exists(p_s + ".json"):
            xs_ = np.load(p_s + ".npy")
            with open(p_s + ".json") as f:
                ms_ = json.load(f)
        ok = (rcs.get("samp_patch") == 0 and xs_ is not None and np.all(np.isfinite(xs_))
              and ms_.get("patch_units") == "4:2:4" and "patch" not in ms_)
        report(f"   samp_patch: samples {None if xs_ is None else np.array2string(np.abs(xs_).max(axis=1), precision=3)} "
               f"(max |x| per trajectory), manifest patch_units {None if ms_ is None else ms_.get('patch_units')}")
        check("19G the default sampler on a patch run (synthetic, 4:2:4 with Qp; 4 trajectories): exit code 0, finite "
              "samples, manifest patch_units '4:2:4'", 0.0 if ok else 1.0, 0.5)
        # early_slices.py: random r, absent n, zero z patch units
        res_p = os.path.join(lm.OUT_ROOT, "early_slices", es_job, "results.json")
        ok, detail = False, f"exit code {rcs.get('early_slices')}"
        if rcs.get("early_slices") == 0 and os.path.exists(res_p):
            with open(res_p) as f:
                res = json.load(f)

            def close(a, b):
                if isinstance(a, float) and isinstance(b, float):
                    return abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)
                return a == b

            exact_zn = all(json.dumps({q: res["z"][k][q] for q in res["n"][k]}) == json.dumps(res["n"][k])
                           for k in ("0", "50"))
            same_zn = all(close(res["z"][k][q], res["n"][k][q]) for k in ("0", "50") for q in res["n"][k])
            diff_rn = all(abs(res["r"][k]["score_rel_err"] - res["n"][k]["score_rel_err"])
                          > 1e-9 * res["n"][k]["score_rel_err"] for k in ("0", "50"))
            has_rows = all("patch_eff_median" in res["r"][k] and "patch_eff_median" not in res["n"][k] for k in ("0", "50"))
            ok = same_zn and diff_rn and has_rows
            detail = (f"score errors at slices 0 / 50: random {res['r']['0']['score_rel_err']:.6g} / "
                      f"{res['r']['50']['score_rel_err']:.6g}, none {res['n']['0']['score_rel_err']:.6g} / "
                      f"{res['n']['50']['score_rel_err']:.6g}; zero = none to 1e-12: {same_zn} (exactly: {exact_zn}); "
                      f"patch rows {has_rows}")
        report(f"   early_slices.py: {detail} ({res_p})")
        check("19G early_slices.py with patch runs: zero units give the results of no units, random units change the score, "
              "patch rows for patch runs", 0.0 if ok else 1.0, 0.5)
    report(f"   block 19: {time.time() - t19:.0f} s")


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; jax {jax.__version__}; devices {jax.devices()}")
    if sys.argv[1:] == ["13"]:          # block 13 alone (e.g. on a CPU node while GPUs are scarce); default: all blocks
        data, _, _, _ = lm.load_data()
        block_13(jr.PRNGKey(13), data, lm.forward_time_pts())
        finish(t_all)
    if sys.argv[1:] == ["14"]:          # block 14 alone (CPU-sized)
        block_14()
        finish(t_all)
    if sys.argv[1:] == ["15"]:          # block 15 alone (CPU-sized)
        data, _, _, _ = lm.load_data()
        block_15(data)
        finish(t_all)
    if sys.argv[1:] == ["16"]:          # block 16 alone (CPU-sized; its command-line runs are CPU subprocesses)
        block_16()
        finish(t_all)
    if sys.argv[1:] == ["17"]:          # block 17 alone (CPU-sized: 8 CPUs; its command-line runs are CPU subprocesses)
        block_17()
        finish(t_all)
    if sys.argv[1:] == ["18"]:          # block 18 alone (CPU-sized: 8 CPUs; its command-line runs are CPU subprocesses)
        block_18()
        finish(t_all)
    if sys.argv[1:2] == ["19"]:         # block 19 alone (CPU-sized: 8 CPUs); '19 G' (or other letters): those parts
        block_19(sys.argv[2:] or "ABCDEFG")
        finish(t_all)
    key = jr.PRNGKey(123)
    conn = lm.connectivity()
    connj = jnp.asarray(conn)
    vis = lm.DenseVisible(conn)
    theta0, unflatten = lm.paper_theta_layout(vis.P)
    report(f"pairs {vis.P}, parameters {vis.n_theta} (paper: 180810, 545566)")
    check("parameter count", abs(vis.n_theta - 545566), 1)
    energy_fn = setup_energy_fn(connj, unflatten)
    gradient_fn, trace_hessian_fn, _, _ = setup_duffing_network_analytical_derivatives_memory_efficient(connj, unflatten)
    paper_loss, paper_grad = setup_score_matching_kbT_loss_minimal_memory(gradient_fn, trace_hessian_fn, k_b=1.0, T=lm.TEMP)
    paper_loss, paper_grad = jax.jit(paper_loss), jax.jit(paper_grad)
    loss0 = jax.jit(lm.make_loss(vis, lm.TEMP, 0))
    grad0 = jax.jit(jax.grad(lm.make_loss(vis, lm.TEMP, 0)))
    ckpt = np.load(lm.PAPER_CKPT, mmap_mode="r")
    data, labels, _, _ = lm.load_data()
    taus = lm.forward_time_pts()

    # 1-2: random parameters (all seven groups non-zero) and random states
    k1, k2, key = jr.split(key, 3)
    th_r = jr.normal(k1, theta0.shape) * 0.3
    X_r = jr.normal(k2, (16, lm.N))
    g_d, l_d = jax.jit(vis.grad_lap)(th_r, X_r)
    g_p = jax.vmap(gradient_fn, in_axes=(0, None))(X_r, th_r)
    l_p = jax.vmap(trace_hessian_fn, in_axes=(0, None))(X_r, th_r)
    check("1 dense grad E vs paper analytic (random theta)", rel(g_d, g_p), 1e-11)
    check("1 dense Laplacian vs paper analytic (random theta)", rel(l_d, l_p), 1e-11)
    e_d = jax.jit(vis.energy)(th_r, X_r)
    e_p = jax.vmap(energy_fn, in_axes=(0, None))(X_r, th_r)
    check("2 dense energy vs paper energy_fn", rel(e_d, e_p), 1e-11)
    g_ad = jax.vmap(jax.grad(lambda x, th: vis.energy(th, x[None])[0]), in_axes=(0, None))(X_r, th_r)
    check("2 autodiff of dense energy vs dense grad", rel(g_ad, g_d), 1e-11)

    # 3: K = 0 loss and gradient vs the paper's training functions
    for lab, th, t in [("paper init, t=5", theta0, taus[0])] + [
            (f"paper ckpt slice {k} (t={taus[k]:.2f})", jnp.asarray(np.array(ckpt[k], dtype=np.float64)), taus[k])
            for k in (0, 50, 99)]:
        k3, key = jr.split(key)
        X = sample_forward_process(float(t), 64, lm.TEMP, lm.SIGMA_FORWARD, data, k3, beta=1)
        a, b = float(loss0({"theta": th, "W": jnp.zeros((0, lm.N)), "b": jnp.zeros((0,))}, X)), float(paper_loss(th, X))
        check(f"3 loss vs paper, {lab} ({b:.6g})", abs(a - b) / max(abs(b), 1.0), 1e-10)
        ga = grad0({"theta": th, "W": jnp.zeros((0, lm.N)), "b": jnp.zeros((0,))}, X)["theta"]
        gb = paper_grad(th, X, None)
        check(f"3 parameter gradient vs paper, {lab}", rel(ga, gb), 1e-9)

    # 4: hidden units vs a full-autodiff reference
    th = jnp.asarray(np.array(ckpt[90], dtype=np.float64))
    t = float(taus[90])
    kx, kw, kb, kd, key = jr.split(key, 5)
    X = sample_forward_process(t, 4, lm.TEMP, lm.SIGMA_FORWARD, data, kx, beta=1)
    W = jr.normal(kw, (3, lm.N)) * 0.03
    bb = jr.normal(kb, (3,)) * 0.05

    def F_ref(x, th, W, b):
        a = W @ x + b
        return energy_fn(x, th) - lm.TEMP * jnp.sum(jnp.logaddexp(a / lm.TEMP, -a / lm.TEMP))

    def loss_ref(p, X):
        def one(x):
            g = jax.grad(F_ref)(x, p["theta"], p["W"], p["b"])
            lap = jnp.trace(jax.hessian(F_ref)(x, p["theta"], p["W"], p["b"]))
            return -lap / lm.TEMP + jnp.sum(g * g) / (2 * lm.TEMP**2)
        return jnp.mean(jax.lax.map(one, X))

    loss_ref = jax.jit(loss_ref)
    loss3 = jax.jit(lm.make_loss(vis, lm.TEMP, 3))
    grad3 = jax.jit(jax.grad(lm.make_loss(vis, lm.TEMP, 3)))
    p = {"theta": th, "W": W, "b": bb}
    lr_, lm_ = float(loss_ref(p, X)), float(loss3(p, X))
    check(f"4 K=3 loss vs full-autodiff reference ({lr_:.6g})", abs(lm_ - lr_) / max(abs(lr_), 1.0), 1e-10)
    g = grad3(p, X)
    for gi, part in enumerate(["theta", "W", "b", "all"]):
        kd, sub = jr.split(kd)
        d = {kk: (jr.normal(jr.fold_in(sub, i), p[kk].shape) * (jnp.abs(p[kk]) + 1e-3)
                  if part in (kk, "all") else jnp.zeros_like(p[kk])) for i, kk in enumerate(p)}
        eps = 1e-5
        fd = (float(loss_ref({kk: p[kk] + eps * d[kk] for kk in p}, X))
              - float(loss_ref({kk: p[kk] - eps * d[kk] for kk in p}, X))) / (2 * eps)
        an = float(sum(jnp.sum(g[kk] * d[kk]) for kk in p))
        check(f"4 gradient along a random {part} direction vs central difference of reference ({an:.6g})",
              abs(an - fd) / max(abs(fd), 1e-8), 1e-5)

    # 5: trainer vs the paper's training loop, 150 steps of slice 0 from the paper's initial parameters
    n_cmp = 150
    optimization_key, _, _ = jr.split(jr.PRNGKey(lm.KEY_SEED), 3)
    _, sub0 = jr.split(optimization_key)
    lr0 = float(lm.lr_schedule()[0])
    t0 = float(taus[0])
    sched = optax.exponential_decay(init_value=lr0, transition_steps=lm.LR_DECAY_STEPS, decay_rate=lm.LR_DECAY_RATE,
                                    staircase=False)
    opt = optax.adam(learning_rate=sched)

    @jax.jit
    def paper_step(params, opt_state, Xb, subkey):
        lv = paper_loss(params, Xb)
        dp = paper_grad(params, Xb, subkey)
        upd, opt_state = opt.update(dp, opt_state)
        return optax.apply_updates(params, upd), opt_state, lv

    params_p, st, kk_ = theta0, opt.init(theta0), sub0
    best_l, best_p, best_e, losses_p = np.inf, theta0, 0, []
    tp = time.time()
    for ep in range(n_cmp):
        kk_, k_e, k_g = jr.split(kk_, 3)
        Xb = sample_forward_process(t0, lm.BATCH_SIZE, lm.TEMP, lm.SIGMA_FORWARD, data, k_e, beta=1)
        params_p, st, lv = paper_step(params_p, st, Xb, k_g)
        params_p = params_p.at[lm.k6_slice()].set(jnp.maximum(params_p[lm.k6_slice()], lm.K6_FLOOR))
        losses_p.append(float(lv))
        if lv < best_l:
            best_l, best_p, best_e = lv, params_p, ep
    tp = time.time() - tp
    run, adam = train_latent.build_runner(lm.make_loss(vis, lm.TEMP, 0), data)
    pr = {"theta": theta0, "W": jnp.zeros((0, lm.N)), "b": jnp.zeros((0,))}
    carry = train_latent.init_carry(pr, adam, sub0)
    carry, l1 = run(carry, t0, {"theta": lr0, "W": 0.0, "b": 0.0}, 100)
    carry, l2 = run(carry, t0, {"theta": lr0, "W": 0.0, "b": 0.0}, n_cmp - 100)
    losses_n = np.concatenate([np.asarray(l1), np.asarray(l2)])
    report(f"5 paper loop {n_cmp} steps in {tp:.1f} s; losses {losses_p[0]:.6g} -> {losses_p[-1]:.6g}")
    check("5 per-step losses, new trainer vs paper loop", rel(losses_n, np.array(losses_p)), 1e-7)
    check("5 final parameters, new trainer vs paper loop", rel(carry[0]["theta"], params_p), 1e-7)
    check("5 best parameters, new trainer vs paper loop", rel(carry[4]["theta"], best_p), 1e-7)
    check("5 best epoch equal", abs(int(carry[6]) - best_e), 1)

    # 6: speed
    W_ref, b_ref = lm.fisher_reference(data, labels, taus, lm.TEMP)
    for K in (0, 1):
        run, adam = train_latent.build_runner(lm.make_loss(vis, lm.TEMP, K), data)
        pr = {"theta": jnp.asarray(np.array(ckpt[50], dtype=np.float64)),
              "W": jnp.asarray(W_ref[50:51] if K else np.zeros((0, lm.N))), "b": jnp.asarray(b_ref[50:51] if K else np.zeros((0,)))}
        carry = train_latent.init_carry(pr, adam, jr.PRNGKey(5))
        lrs = {"theta": 1e-4, "W": 1e-6 if K else 0.0, "b": 1e-6 if K else 0.0}
        carry, _ = run(carry, float(taus[50]), lrs, 100)
        jax.block_until_ready(carry)
        ts = time.time()
        for _ in range(5):
            carry, l = run(carry, float(taus[50]), lrs, 100)
        jax.block_until_ready(carry)
        sps = 500 / (time.time() - ts)
        report(f"6 trainer speed K={K}: {sps:.1f} steps/s -> full paper schedule (2.09M steps) in "
               f"{sum(lm.paper_steps(k) for k in range(100)) / sps / 3600:.1f} h")
    report(f"   Fisher reference: |W*(t=5)| {np.linalg.norm(W_ref[0]):.4g}, |W*(t=0)| {np.linalg.norm(W_ref[-1]):.4g}; "
           f"b* {b_ref[0]:.4g} -> {b_ref[-1]:.4g}")
    Xf = np.asarray(data)
    acc = [float(np.mean(((Xf * np.exp(-taus[k])) @ W_ref[k] + b_ref[k] > 0) == (labels == 1))) for k in (0, 50, 99)]
    report(f"   Fisher unit on the noiseless (scaled) training images: accuracy {acc} at slices 0, 50, 99")

    # 7: sampler drift with a hidden unit vs autodiff of the reference free energy
    th_hist = np.array(ckpt[:], dtype=np.float64)
    Wh = W_ref[:, None, :]
    bh = b_ref[:, None]
    for sampler in ("sde", "ode"):
        drift, _ = sample_latent.make_drift(th_hist, Wh, bh, sampler)
        kx, key = jr.split(key)
        x = jr.normal(kx, (lm.N,)) * 0.5
        for t in (0.3, 2.71, 4.99):
            tau = lm.T_FORWARD - t
            ks = np.searchsorted(np.sort(taus), tau)
            ts_ = np.sort(taus)
            w = (tau - ts_[ks - 1]) / (ts_[ks] - ts_[ks - 1])
            order = np.argsort(taus)
            interp = lambda arr: (1 - w) * arr[order][ks - 1] + w * arr[order][ks]
            th_i, W_i, b_i = interp(th_hist), interp(Wh), interp(bh)
            sc = 2.0 if sampler == "sde" else 1.0
            ref = x - sc * jax.grad(F_ref)(x, jnp.asarray(th_i), jnp.asarray(W_i), jnp.asarray(b_i))
            check(f"7 {sampler} drift at t={t} vs x - {sc:g} grad F (autodiff)", rel(drift(t, x, ()), ref), 1e-10)

    # 8: sampler reproduction of job 4187940 (paper checkpoint, first 4 trajectories), and a K = 1 smoke run
    knots = jnp.asarray(np.sort(lm.T_FORWARD - taus)[1:-1])
    x0, keys = lm.trajectory_set("base", 4)
    for lab, Wx, bx in [("paper K=0", np.zeros((100, 0, lm.N)), np.zeros((100, 0))), ("paper theta + Fisher unit K=1", Wh, bh)]:
        drift, diffusion = sample_latent.make_drift(th_hist, Wx, bx, "sde")
        fn = jax.vmap(lambda y, k: solve_SDE(drift, diffusion, y, k, 0.0, lm.T_FORWARD, 2, lm.DT0, rtol=lm.RTOL_SDE,
                                             atol=lm.ATOL_SDE, brownian_tolerance=lm.BROWNIAN_TOL, step_ts=knots))
        ts = time.time()
        xs = np.asarray(fn(jnp.asarray(x0), jnp.asarray(keys)).ys[:, -1, :])
        report(f"8 {lab}: 4 trajectories in {time.time() - ts:.0f} s; max|x| {np.max(np.abs(xs)):.3g}")
        if Wx.shape[1] == 0:
            ref = np.load(lm.UNSPLIT_4187940)[:4]
            d = np.max(np.abs(xs - ref), axis=1)
            report(f"   vs job 4187940: max|diff| per trajectory {np.array2string(d, precision=2)}")
            check("8 sampler reproduces job 4187940 (median max|diff|)", float(np.median(d)), 1e-3)
        else:
            check("8 K=1 sampler gives finite states", float(np.sum(~np.isfinite(xs))), 0.5)

    # 9: small problems: dense functions vs the paper's analytic ones with the problem's lattice; data sanity
    for pname in ("mnist14", "gauss2_8"):
        prob = lm.make_problem(pname)
        n = prob["n"]
        connj_s = jnp.asarray(prob["conn"])
        vis_s = lm.DenseVisible(prob["conn"], n)
        th0_s, unflat_s = lm.paper_theta_layout(vis_s.P, n)
        gfn_s, tfn_s, _, _ = setup_duffing_network_analytical_derivatives_memory_efficient(connj_s, unflat_s)
        efn_s = setup_energy_fn(connj_s, unflat_s)
        k1, k2, key = jr.split(key, 3)
        th_s = jr.normal(k1, th0_s.shape) * 0.3
        X_s = jr.normal(k2, (16, n))
        g_d, l_d = jax.jit(vis_s.grad_lap)(th_s, X_s)
        check(f"9 {pname}: dense grad vs paper analytic", rel(g_d, jax.vmap(gfn_s, in_axes=(0, None))(X_s, th_s)), 1e-11)
        check(f"9 {pname}: dense Laplacian vs paper analytic", rel(l_d, jax.vmap(tfn_s, in_axes=(0, None))(X_s, th_s)), 1e-11)
        check(f"9 {pname}: dense energy vs paper energy_fn",
              rel(jax.jit(vis_s.energy)(th_s, X_s), jax.vmap(efn_s, in_axes=(0, None))(X_s, th_s)), 1e-11)
        d = np.asarray(prob["data"])
        report(f"9 {pname}: {d.shape[0]} samples x {n} pixels, {vis_s.P} pairs, {vis_s.n_theta} parameters; "
               f"normalised range [{d.min():.3f}, {d.max():.3f}], mean {d.mean():.2e}, std {d.std():.4f}; "
               f"class 1 share {prob['labels'].mean():.3f}")
        x0s, ks = lm.trajectory_set("base", 8, pname, n)
        check(f"9 {pname}: trajectory set shape", abs(x0s.shape[1] - n) + abs(ks.shape[0] - 8), 1)

    # 10: Potts hidden unit: loss vs full autodiff, gradient vs central differences, M = 2 equals one Ising unit,
    #     sampler drift, and the prototype initialisation on the training images
    th = jnp.asarray(np.array(ckpt[85], dtype=np.float64))
    t = float(taus[85])
    kx, kw, kb, kd, key = jr.split(key, 5)
    X = sample_forward_process(t, 4, lm.TEMP, lm.SIGMA_FORWARD, data, kx, beta=1)
    Wp = jr.normal(kw, (4, lm.N)) * 0.03
    bp = jr.normal(kb, (4,)) * 0.05

    def F_potts(x, th, W, b):
        a = W @ x + b
        return energy_fn(x, th) - lm.TEMP * jax.scipy.special.logsumexp(a / lm.TEMP)

    def loss_ref_potts(p, X):
        def one(x):
            g = jax.grad(F_potts)(x, p["theta"], p["W"], p["b"])
            lap = jnp.trace(jax.hessian(F_potts)(x, p["theta"], p["W"], p["b"]))
            return -lap / lm.TEMP + jnp.sum(g * g) / (2 * lm.TEMP**2)
        return jnp.mean(jax.lax.map(one, X))

    loss_ref_potts = jax.jit(loss_ref_potts)
    lossP = jax.jit(lm.make_loss(vis, lm.TEMP, 4, "potts"))
    gradP = jax.jit(jax.grad(lm.make_loss(vis, lm.TEMP, 4, "potts")))
    p = {"theta": th, "W": Wp, "b": bp}
    a_, b_ = float(lossP(p, X)), float(loss_ref_potts(p, X))
    check(f"10 Potts M=4 loss vs full-autodiff reference ({b_:.6g})", abs(a_ - b_) / max(abs(b_), 1.0), 1e-10)
    g = gradP(p, X)
    kd, sub = jr.split(kd)
    d = {kk: jr.normal(jr.fold_in(sub, i), p[kk].shape) * (jnp.abs(p[kk]) + 1e-3) for i, kk in enumerate(p)}
    eps = 1e-5
    fd = (float(loss_ref_potts({kk: p[kk] + eps * d[kk] for kk in p}, X))
          - float(loss_ref_potts({kk: p[kk] - eps * d[kk] for kk in p}, X))) / (2 * eps)
    an = float(sum(jnp.sum(g[kk] * d[kk]) for kk in p))
    check(f"10 Potts gradient along a random direction vs central difference ({an:.6g})", abs(an - fd) / max(abs(fd), 1e-8), 1e-5)
    W1, b1 = Wp[:1], bp[:1]
    li = float(jax.jit(lm.make_loss(vis, lm.TEMP, 1, "ising"))({"theta": th, "W": W1, "b": b1}, X))
    lp = float(jax.jit(lm.make_loss(vis, lm.TEMP, 2, "potts"))(
        {"theta": th, "W": jnp.concatenate([W1, -W1]), "b": jnp.concatenate([b1, -b1])}, X))
    check("10 Potts M=2 with (W, -W), (b, -b) equals one Ising unit", abs(lp - li) / max(abs(li), 1.0), 1e-12)
    Wph = np.repeat(np.asarray(Wp)[None], 100, axis=0)
    bph = np.repeat(np.asarray(bp)[None], 100, axis=0)
    drift, _ = sample_latent.make_drift(th_hist, Wph, bph, "sde", "mnist28", "potts")
    kx, key = jr.split(key)
    x = jr.normal(kx, (lm.N,)) * 0.5
    tt = 1.37
    tau = lm.T_FORWARD - tt
    ts_ = np.sort(taus)
    ks = np.searchsorted(ts_, tau)
    w = (tau - ts_[ks - 1]) / (ts_[ks] - ts_[ks - 1])
    order = np.argsort(taus)
    th_i = (1 - w) * th_hist[order][ks - 1] + w * th_hist[order][ks]
    ref = x - 2.0 * jax.grad(F_potts)(x, jnp.asarray(th_i), Wp, bp)
    check("10 Potts sde drift vs x - 2 grad F (autodiff)", rel(drift(tt, x, ()), ref), 1e-10)
    Wr, br, scls = lm.prototype_reference(data, labels, taus, lm.TEMP, 8)
    Xf = np.asarray(data)
    for kk in (99, 50):
        A = (Xf * np.exp(-taus[kk])) @ Wr[kk].T + br[kk]
        acc = float(np.mean(scls[np.argmax(A, axis=1)] == labels))
        report(f"   prototype unit (8 states) on the scaled training images, slice {kk}: class accuracy of the top state "
               f"{acc:.4f}; states used {np.bincount(np.argmax(A, axis=1), minlength=8).tolist()}")
        if kk == 99:
            check("10 prototype initialisation classifies the training images (slice 99)", 1.0 - acc, 0.05)

    # 11: one Ising unit per pixel (--local), together with one global unit
    kx, kw, kb, kl, kc, key = jr.split(key, 6)
    X = sample_forward_process(t, 4, lm.TEMP, lm.SIGMA_FORWARD, data, kx, beta=1)
    Wg, bg = jr.normal(kw, (1, lm.N)) * 0.03, jr.normal(kb, (1,)) * 0.05
    wl, bl = jr.normal(kl, (lm.N,)) * 0.3, jr.normal(kc, (lm.N,)) * 0.05

    def F_loc(x, th, W, b, wl, bl):
        a = W @ x + b
        al = wl * x + bl
        return (energy_fn(x, th) - lm.TEMP * jnp.sum(jnp.logaddexp(a / lm.TEMP, -a / lm.TEMP))
                - lm.TEMP * jnp.sum(jnp.logaddexp(al / lm.TEMP, -al / lm.TEMP)))

    def loss_ref_loc(p, X):
        def one(x):
            args_ = (p["theta"], p["W"], p["b"], p["wl"], p["bl"])
            g = jax.grad(F_loc)(x, *args_)
            lap = jnp.trace(jax.hessian(F_loc)(x, *args_))
            return -lap / lm.TEMP + jnp.sum(g * g) / (2 * lm.TEMP**2)
        return jnp.mean(jax.lax.map(one, X))

    loss_ref_loc = jax.jit(loss_ref_loc)
    lossL = jax.jit(lm.make_loss(vis, lm.TEMP, 1, "ising", True))
    gradL = jax.jit(jax.grad(lm.make_loss(vis, lm.TEMP, 1, "ising", True)))
    p = {"theta": th, "W": Wg, "b": bg, "wl": wl, "bl": bl}
    a_, b_ = float(lossL(p, X)), float(loss_ref_loc(p, X))
    check(f"11 local + global units: loss vs full-autodiff reference ({b_:.6g})", abs(a_ - b_) / max(abs(b_), 1.0), 1e-10)
    g = gradL(p, X)
    kd, sub = jr.split(kd)
    d = {kk: jr.normal(jr.fold_in(sub, i), p[kk].shape) * (jnp.abs(p[kk]) + 1e-3) for i, kk in enumerate(p)}
    fd = (float(loss_ref_loc({kk: p[kk] + 1e-5 * d[kk] for kk in p}, X))
          - float(loss_ref_loc({kk: p[kk] - 1e-5 * d[kk] for kk in p}, X))) / 2e-5
    an = float(sum(jnp.sum(g[kk] * d[kk]) for kk in p))
    check(f"11 local units: gradient along a random direction vs central difference ({an:.6g})",
          abs(an - fd) / max(abs(fd), 1e-8), 1e-5)
    Wgh, bgh = np.repeat(np.asarray(Wg)[None], 100, 0), np.repeat(np.asarray(bg)[None], 100, 0)
    wlh, blh = np.repeat(np.asarray(wl)[None], 100, 0), np.repeat(np.asarray(bl)[None], 100, 0)
    drift, _ = sample_latent.make_drift(th_hist, Wgh, bgh, "sde", "mnist28", "ising", (wlh, blh))
    kx, key = jr.split(key)
    x = jr.normal(kx, (lm.N,)) * 0.5
    ref = x - 2.0 * jax.grad(F_loc)(x, jnp.asarray(th_i), Wg, bg, wl, bl)
    check("11 local units: sde drift vs x - 2 grad F (autodiff)", rel(drift(tt, x, ()), ref), 1e-10)
    thr = (0.5 - 0.12478) / 0.30467
    wr, cr = lm.local_reference(data, taus, lm.TEMP, thr)
    Xf = np.asarray(data)
    used = np.abs(wr[-1]) > 0
    pred = (Xf * wr[-1] + cr[-1]) > 0
    acc = float(np.mean(pred[:, used] == (Xf[:, used] > thr)))
    report(f"   local reference: {int(used.sum())} pixels with both modes; at t = 0 the units' sign matches ink / "
           f"background on {acc:.4f} of (image, pixel) pairs; rms w* {np.sqrt(np.mean(wr[-1, used] ** 2)):.3g}")
    check("11 local reference separates ink from background at t = 0", 1.0 - acc, 0.05)

    # 12: hidden units that also couple to the squared pixels (--quad), Ising (K = 1) and Potts (M = 4)
    for hid, Kq in (("ising", 1), ("potts", 4)):
        kx, kw, kq, kb, key = jr.split(key, 5)
        X = sample_forward_process(t, 4, lm.TEMP, lm.SIGMA_FORWARD, data, kx, beta=1)
        Wq, Qq, bq = jr.normal(kw, (Kq, lm.N)) * 0.03, jr.normal(kq, (Kq, lm.N)) * 0.01, jr.normal(kb, (Kq,)) * 0.05

        def F_quad(x, th, W, Q, b, hid=hid):
            a = W @ x + Q @ (x * x) + b
            if hid == "ising":
                return energy_fn(x, th) - lm.TEMP * jnp.sum(jnp.logaddexp(a / lm.TEMP, -a / lm.TEMP))
            return energy_fn(x, th) - lm.TEMP * jax.scipy.special.logsumexp(a / lm.TEMP)

        def loss_ref_quad(p, X, F=F_quad):
            def one(x):
                args_ = (p["theta"], p["W"], p["Q"], p["b"])
                g = jax.grad(F)(x, *args_)
                lap = jnp.trace(jax.hessian(F)(x, *args_))
                return -lap / lm.TEMP + jnp.sum(g * g) / (2 * lm.TEMP**2)
            return jnp.mean(jax.lax.map(one, X))

        lref = jax.jit(loss_ref_quad)
        p = {"theta": th, "W": Wq, "Q": Qq, "b": bq}
        lq = jax.jit(lm.make_loss(vis, lm.TEMP, Kq, hid))
        a_, b_ = float(lq(p, X)), float(lref(p, X))
        check(f"12 quad {hid} K={Kq}: loss vs full-autodiff reference ({b_:.6g})", abs(a_ - b_) / max(abs(b_), 1.0), 1e-10)
        g = jax.jit(jax.grad(lm.make_loss(vis, lm.TEMP, Kq, hid)))(p, X)
        kd, sub = jr.split(kd)
        d = {kk: jr.normal(jr.fold_in(sub, i), p[kk].shape) * (jnp.abs(p[kk]) + 1e-3) for i, kk in enumerate(p)}
        fd = (float(lref({kk: p[kk] + 1e-5 * d[kk] for kk in p}, X)) - float(lref({kk: p[kk] - 1e-5 * d[kk] for kk in p}, X))) / 2e-5
        an = float(sum(jnp.sum(g[kk] * d[kk]) for kk in p))
        check(f"12 quad {hid}: gradient along a random direction vs central difference ({an:.6g})",
              abs(an - fd) / max(abs(fd), 1e-8), 1e-5)
        rep_ = lambda arr: np.repeat(np.asarray(arr)[None], 100, 0)
        drift, _ = sample_latent.make_drift(th_hist, rep_(Wq), rep_(bq), "sde", "mnist28", hid, None, rep_(Qq))
        kx, key = jr.split(key)
        x = jr.normal(kx, (lm.N,)) * 0.5
        ref = x - 2.0 * jax.grad(F_quad)(x, jnp.asarray(th_i), Wq, Qq, bq)
        check(f"12 quad {hid}: sde drift vs x - 2 grad F (autodiff)", rel(drift(tt, x, ()), ref), 1e-10)

    block_13(key, data, taus)
    block_14()
    block_15(data)
    block_16()
    # blocks 17 and 18 (--harmonic; about 15 min each with parallel CPU subprocesses) run only when named
    finish(t_all)


def finish(t_all):
    """Summary line, checks.json, and the exit code (1 if any check failed)."""
    n_fail = sum(not c["ok"] for c in CHECKS)
    report(f"=== {len(CHECKS)} checks, {n_fail} failed; total {time.time() - t_all:.0f} s")
    with open(os.path.join(OUT, "checks.json"), "w") as f:
        json.dump(CHECKS, f, indent=2)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
