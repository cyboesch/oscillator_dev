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
      HIGHEST) vs float64; the per-step trims (best_params skipped under --select ema, --step-impl lean) bitwise
      against the default runner; --steps-schedule; the GPU k-means (--ref-kmeans gpu) vs numpy; the history
      assembly from memory (--assemble memory) vs from the slice files
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

    # per-step trims: best_params skipped (--select ema) and --step-impl lean, bitwise against the default runner
    lrs = {g: (1e-3 if g == "theta" else 3e-5) for g in pq}

    def run_variant(loss_, **kw):
        run_, adam_ = train_latent.build_runner(loss_, data, lm.N, 1000.0, "quartic", P_s, zero_slices=zs, **kw)
        c = train_latent.init_carry(pq, adam_, jr.PRNGKey(4))
        ls = []
        for nst in (100, 100, 1):                            # both compiled chunk lengths
            c, l = run_(c, t_q, lrs, nst)
            ls.append(np.asarray(l))
        return c, np.concatenate(ls)

    lean = {"track_best": False, "theta_masks": True, "chunk_batches": True}
    for lab, loss_, variants in (("stencil mixed", loss_m, (("best skipped", {"track_best": False}), ("lean", lean))),
                                 ("dense f64", loss_d, (("best skipped", {"track_best": False}),))):
        c0, l0 = run_variant(loss_)
        for vname, kw in variants:
            c1, l1 = run_variant(loss_, **kw)
            same = np.array_equal(l0, l1) and all(
                np.array_equal(np.asarray(a), np.asarray(b))
                for i in (0, 1, 2, 3, 5, 6, 7) for a, b in zip(jax.tree_util.tree_leaves(c0[i]),
                                                               jax.tree_util.tree_leaves(c1[i])))
            check(f"13 {vname} ({lab}, Potts 8 --quad --local, 201 steps): losses, parameters, Adam state, key, best "
                  f"loss and step, EMA (the --select ema result) bitwise equal to the default runner", 0.0 if same else 1.0,
                  0.5)

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


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; jax {jax.__version__}; devices {jax.devices()}")
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

    n_fail = sum(not c["ok"] for c in CHECKS)
    report(f"=== {len(CHECKS)} checks, {n_fail} failed; total {time.time() - t_all:.0f} s")
    with open(os.path.join(OUT, "checks.json"), "w") as f:
        json.dump(CHECKS, f, indent=2)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
