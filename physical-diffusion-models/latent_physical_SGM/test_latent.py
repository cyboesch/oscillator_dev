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
Usage (from latent_physical_SGM/):  sbatch jobs/run_gpu_neuronic.sh test_latent.py [13|14]   ('13' / '14': that block
alone; both also run on the CPU, where block 13's bitwise comparisons test XLA:CPU's compilation instead of the GPU's).
Writes out/latent_physical_SGM/tests/<job>/report.txt and checks.json; exit code 1 if any check fails.
"""

import json
import os
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
