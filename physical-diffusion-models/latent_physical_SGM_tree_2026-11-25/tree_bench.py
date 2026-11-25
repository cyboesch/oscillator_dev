#!/usr/bin/env python3
"""Cost of the tree on a trained locked model, and a first probe of what J can do (2026-11-25, GPU).

For a run of the locked recipe (K16 global Potts unit + two patch sets) and a few of its slices:

1. Consistency: with J = 0 the tree loss equals the locked code's loss on the run's own slice parameters (float64).
2. Step cost at the training batch (512, --precision mixed = float32 loss): value and gradient with respect to all
   parameters, for (a) the locked model, (b) the tree with J = 0, (c) the tree with a random J of rms 1 T; the
   gradient with respect to J only; peak GPU memory; float32 against float64 for (c). From (a) and (c): the estimated
   time of a full 100-slice training.
3. Sampler cost: the force at the sampler's chunk (250, float64), locked against tree.
4. J-only probe: all trained parameters frozen, J = 0 at the start, Adam on J alone for --probe-steps steps with fresh
   forward samples of the training images (as the training does). Before and after: the implicit score-matching loss
   on held-out images (4096 test images, the same noise draws: a paired difference with its standard error), rms J / T,
   the mutual information I(g; s_p) of the tree's joint posterior and the effective number of global states.
   A lower held-out loss means the coupling carries information the pixels did not; this is only a hint (the other
   parameters are not re-trained, and no samples are drawn).

Usage (GPU job, from this folder): sbatch jobs/run_gpu.sh tree_bench.py --run RUN [--slices "30 60 80 99"]
Writes out/tree_bench/<job>/report.txt, metrics.json and J_probe_slice<k>.npz.
"""
import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tree_model as tm  # noqa: E402
lm = tm.lm

T = lm.TEMP
OUT = os.path.join(HERE, "out", "tree_bench", os.environ.get("SLURM_JOB_ID", "manual"))
LINES = []
MODEL_KEYS = ("theta", "W", "b", "Q", "wl", "bl")


def report(s=""):
    print(s, flush=True)
    LINES.append(s)


def timeit(fn, *a, n=20):
    jax.block_until_ready(fn(*a))                       # compile
    t0 = time.time()
    for _ in range(n):
        out = fn(*a)
    jax.block_until_ready(out)
    return (time.time() - t0) / n


def peak_gb():
    try:
        return jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2 ** 30
    except Exception:
        return float("nan")


def noised(x0, t, key):
    e = jnp.exp(-t)
    return e * x0 + jnp.sqrt(T * (1.0 - e * e)) * jr.normal(key, x0.shape, x0.dtype)


def tree_info(params, X):
    """rms J / T per set, mean I(g; s_p) per patch (nats; the joint posterior of the tree) and effective global states."""
    _, _, pi, qs, pcs = tm.tree_terms(params["W"], params.get("Q"), params["b"], tm.tree_sets(params), X, T, with_lap=False)
    eff = float(jnp.mean(jnp.exp(-jnp.sum(pi * jnp.log(pi + 1e-300), axis=1))))
    mi = []
    for q, pc in zip(qs, pcs):
        joint = jnp.einsum("bk,pbks->pbks", pi, pc)
        ratio = jnp.log(pc + 1e-300) - jnp.log(q[:, :, None, :] + 1e-300)
        mi.append(float(jnp.mean(jnp.sum(joint * ratio, axis=(2, 3)))))
    return eff, mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--slices", default="30 60 80 99")
    ap.add_argument("--batch", type=int, default=lm.BATCH_SIZE)
    ap.add_argument("--probe-steps", type=int, default=1500)
    ap.add_argument("--probe-lr-scale", type=float, default=1.0)
    ap.add_argument("--val-n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    run_dir = os.path.join(lm.OUT_ROOT, args.run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    problem, K = cfg["problem"], int(cfg["K"])
    report(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}; tree_bench; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
           f"devices {jax.devices()}; jax {jax.__version__}")
    report(f"run {args.run} ({problem}); K {K}; patches {cfg.get('patch')} from slice {cfg.get('patch_on_slice')}, "
           f"{cfg.get('patch2')} from slice {cfg.get('patch2_on_slice')}; batch {args.batch}")
    if cfg.get("hidden") != "potts" or not cfg.get("quad") or not cfg.get("local") or cfg.get("visible") != "quartic":
        raise SystemExit("tree_bench is written for the locked recipe (--hidden potts --quad --local --visible quartic)")
    prob = lm.make_problem(problem, radius=cfg.get("radius"), neighbourhood=cfg.get("neighbourhood", "chebyshev"))
    N = prob["n"]
    pair_terms = lm.parse_pair_terms(cfg.get("pair_terms", "all"))
    vis = lm.make_visible(prob["conn"], N, "stencil", pair_terms, sextic=False)
    train = jnp.asarray(np.asarray(prob["data"]), jnp.float32)
    ds = lm.problem_dataset(problem)
    raw_te = lm.load_cifar_raw("test")[0] if ds == "cifar" else lm.load_celeba_raw("test", lm.celeba_grid(problem))[0]
    rng = np.random.default_rng(args.seed)
    te = raw_te[rng.choice(raw_te.shape[0], args.val_n, replace=False)]
    test = jnp.asarray((te + lm.STD_ADDED_NOISE * rng.standard_normal(te.shape) - prob["mean"]) / prob["std"])
    taus, lr_v = lm.forward_time_pts(), lm.lr_schedule()
    on = [cfg.get("patch_on_slice") or 0, cfg.get("patch2_on_slice") or 0, cfg.get("patch3_on_slice") or 0]
    base64 = lm.make_loss_per_sample(vis, T, K, "potts", True)
    tree64 = tm.make_tree_loss_per_sample(vis, T, True)
    base32 = lm.make_loss(vis, T, K, "potts", True, compute_dtype=jnp.float32)
    tree32 = tm.make_tree_loss(vis, T, True, compute_dtype=jnp.float32)
    vg_base = jax.jit(jax.value_and_grad(base32))
    vg_tree = jax.jit(jax.value_and_grad(tree32))
    vg_tree64 = jax.jit(jax.value_and_grad(tm.make_tree_loss(vis, T, True)))
    force_base = jax.jit(lm.make_force(vis, T, K, "potts", True))
    force_tree = jax.jit(tm.make_tree_force(vis, T, True))
    val_tree = jax.jit(tree64)
    key = jr.PRNGKey(args.seed)
    metrics, t_base, t_tree = {}, {}, {}
    for k in [int(s) for s in args.slices.split()]:
        t = float(taus[k])
        z = np.load(os.path.join(run_dir, "slices", f"slice_{k:03d}.npz"))
        params = {n: jnp.asarray(z[n]) for n in MODEL_KEYS}
        sets = []
        for i, (kw, kb, kq) in enumerate(lm.PATCH_SETS):
            if kw in z.files and k >= on[i]:
                params.update({kw: jnp.asarray(z[kw]), kb: jnp.asarray(z[kb])})
                if kq in z.files:
                    params[kq] = jnp.asarray(z[kq])
                sets.append((kw, tm.J_KEYS[i]))
        kk = jr.fold_in(key, k)
        report()
        report(f"--- slice {k}, t = {t:.3f}; patch sets active: {[f'{kw} {tuple(params[kw].shape)}' for kw, _ in sets]}")
        if not sets:
            report("  no patch units at this slice: nothing to couple")
            continue
        J0 = {kj: jnp.zeros((params[kw].shape[0], K, params[kw].shape[1])) for kw, kj in sets}
        Jr = {kj: T * jr.normal(jr.fold_in(kk, 7 + i), v.shape) for i, (kj, v) in enumerate(J0.items())}
        Xb = noised(train[jr.choice(jr.fold_in(kk, 1), train.shape[0], (args.batch,), replace=False)].astype(jnp.float64),
                    t, jr.fold_in(kk, 2))
        Xv = noised(test, t, jr.fold_in(kk, 3))
        m = metrics.setdefault(str(k), {"t": t})

        # 1. consistency with the locked code (float64, 64 samples)
        ref, new = base64(params, Xb[:64]), tree64(dict(params, **J0), Xb[:64])
        e = float(jnp.max(jnp.abs(new - ref) / jnp.abs(ref)))
        report(f"  1. J = 0: tree loss per sample = locked loss, max rel. diff {e:.1e} (float64, 64 samples) "
               f"{'OK' if e < 1e-9 else 'MISMATCH'}")
        m["j0_rel_diff"] = e

        # 2. step cost (float32 loss, all parameters)
        tb = timeit(vg_base, params, Xb)
        mem_b = peak_gb()
        t0 = timeit(vg_tree, dict(params, **J0), Xb)
        tr = timeit(vg_tree, dict(params, **Jr), Xb)
        mem_t = peak_gb()
        vgJ = jax.jit(jax.value_and_grad(lambda Jd, X: tree32(dict(params, **Jd), X)))
        tj = timeit(vgJ, Jr, Xb)
        v32, g32 = vg_tree(dict(params, **Jr), Xb[:128])
        v64, g64 = vg_tree64(dict(params, **Jr), Xb[:128])
        rel = {n: float(jnp.linalg.norm(g32[n] - g64[n]) / (jnp.linalg.norm(g64[n]) + 1e-300)) for n in g64}
        report(f"  2. value + gradient, batch {args.batch}, float32: locked {tb * 1e3:.1f} ms; tree J = 0 {t0 * 1e3:.1f} ms; "
               f"tree random J (rms 1 T) {tr * 1e3:.1f} ms = {tr / tb:.1f} x locked; gradient w.r.t. J only {tj * 1e3:.1f} ms; "
               f"peak GPU memory {mem_b:.1f} -> {mem_t:.1f} GB")
        report(f"     float32 against float64 (random J, 128 samples): value rel. {abs(float(v32 - v64) / float(v64)):.1e}; "
               f"gradient rel. error max {max(rel.values()):.1e} ({max(rel, key=rel.get)}), J {max(rel[kj] for _, kj in sets):.1e}")
        t_base[k], t_tree[k] = tb, tr
        m.update(ms_locked=tb * 1e3, ms_tree_j0=t0 * 1e3, ms_tree=tr * 1e3, ms_j_only=tj * 1e3, grad_rel_err=rel)

        # 3. sampler force (float64, chunk 250)
        tf_b = timeit(force_base, params, Xb[:250])
        tf_t = timeit(force_tree, dict(params, **Jr), Xb[:250])
        report(f"  3. force at chunk 250 (float64): locked {tf_b * 1e3:.2f} ms; tree {tf_t * 1e3:.2f} ms = {tf_t / tf_b:.1f} x")
        m.update(ms_force_locked=tf_b * 1e3, ms_force_tree=tf_t * 1e3)

        # 4. J-only probe
        lr = float(lr_v[k]) * args.probe_lr_scale
        opt = optax.adam(lr)

        @jax.jit
        def step(Jd, st, kstep):
            k1, k2 = jr.split(kstep)
            X = noised(train[jr.choice(k1, train.shape[0], (args.batch,))].astype(jnp.float64), t, k2)
            v, g = jax.value_and_grad(lambda Jx: tree32(dict(params, **Jx), X))(Jd)
            up, st = opt.update(g, st)
            return optax.apply_updates(Jd, up), st, v

        def val(Jd):
            return np.concatenate([np.asarray(val_tree(dict(params, **Jd), Xv[i:i + 512])) for i in range(0, Xv.shape[0], 512)])

        v_before = val(J0)
        Jd, st = J0, opt.init(J0)
        t_start, trace = time.time(), []
        for i in range(args.probe_steps):
            Jd, st, v = step(Jd, st, jr.fold_in(jr.fold_in(kk, 99), i))
            if i % 100 == 0 or i == args.probe_steps - 1:
                trace.append(float(v))
        dt = time.time() - t_start
        v_after = val(Jd)
        d = v_after - v_before
        eff0, mi0 = tree_info(dict(params, **J0), Xv[:1024])
        eff1, mi1 = tree_info(dict(params, **Jd), Xv[:1024])
        rms = {kj: float(jnp.sqrt(jnp.mean(v * v))) / T for kj, v in Jd.items()}
        z_score = float(d.mean() / (d.std() / np.sqrt(d.size)))
        report(f"  4. J-only probe: {args.probe_steps} Adam steps, lr {lr:.2e} ({dt:.0f} s, {dt / args.probe_steps * 1e3:.0f} ms per step); "
               f"training loss {trace[0]:.3f} -> {trace[-1]:.3f}")
        report(f"     held-out loss ({d.size} test images, same noise): {v_before.mean():.3f} -> {v_after.mean():.3f}; paired "
               f"difference {d.mean():+.4f} +- {d.std() / np.sqrt(d.size):.4f} ({z_score:+.1f} standard errors); the loss's "
               f"spread over images is {v_before.std():.1f}")
        report(f"     rms J / T {', '.join(f'{kj} {v:.3f}' for kj, v in rms.items())}; I(g; s_p) per patch "
               f"{', '.join(f'{a:.4f} -> {b:.4f}' for a, b in zip(mi0, mi1))} nats; effective global states {eff0:.2f} -> {eff1:.2f}")
        m.update(probe={"steps": args.probe_steps, "lr": lr, "val_before": float(v_before.mean()), "val_after": float(v_after.mean()),
                        "diff": float(d.mean()), "diff_se": float(d.std() / np.sqrt(d.size)), "rms_J_over_T": rms,
                        "mi_before": mi0, "mi_after": mi1, "eff_before": eff0, "eff_after": eff1, "train_trace": trace})
        np.savez(os.path.join(OUT, f"J_probe_slice{k:03d}.npz"), **{kj: np.asarray(v) for kj, v in Jd.items()})

    if t_tree:
        ks = sorted(t_tree)
        steps = np.array([lm.paper_steps(j) for j in range(lm.N_TIME_STEPS)], float)
        def per_slice(tt):                       # the measured step times, interpolated over the slices with patches
            return np.interp(np.arange(lm.N_TIME_STEPS), ks, [tt[j] for j in ks])
        first = min(o for o in on[:2] if o) if any(on[:2]) else 0
        tb_all, tt_all = per_slice(t_base), per_slice(t_tree)
        tt_all[:first] = tb_all[:first]          # before the first patch set there is nothing to couple
        report()
        report(f"Estimated full training (100 slices, {int(steps.sum()):,} steps, step cost only): locked "
               f"{np.sum(steps * tb_all) / 3600:.1f} h; tree {np.sum(steps * tt_all) / 3600:.1f} h "
               f"({np.sum(steps * tt_all) / np.sum(steps * tb_all):.1f} x)")
        metrics["estimate_hours"] = {"locked": float(np.sum(steps * tb_all) / 3600), "tree": float(np.sum(steps * tt_all) / 3600)}
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump({"run": args.run, "slices": metrics}, fh, indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
