#!/usr/bin/env python3
"""J on top of a trained locked model (2026-11-25): every trained parameter frozen, only the tables J_p[g, s] between
the global unit and the patch units are trained, slice by slice, with the training's own loss (implicit score
matching, exact statistics, float32 loss as --precision mixed) and fresh forward samples of the training images.

 - Parameters: the run's selected (EMA) history, i.e. the model the sampler uses (params_history.npy,
   hidden_history.npz), from the first slice with patch units to the last.
 - J of a set starts at zero at the set's switch-on slice; every later slice starts from the previous slice's
   result (as the training warm-starts its slices). Adam, constant step --lr, --steps steps per slice; the slice's
   result is the exponential average of the iterates over --ema-steps steps (as --select ema).
 - Per slice: the loss on held-out test images (same images and noise for every variant): J = 0, the warm start,
   the result. A negative difference to J = 0 means the coupling carries information the pixels did not.
 - Output: a mirror of the run below this folder's out/, named <run>__J-<tag>: links to the run's files, its own
   samples/ folder, J_history.npz (J, J2: (100, P, K, S) float32, zero before the switch-on) and train_J.json.
   sample_tree.py samples it.

The other parameters are not re-trained: this is the cheapest test of the coupling, not the tree's full capacity.

Usage (GPU job, from this folder): sbatch jobs/run_gpu.sh train_J.py --run RUN --tag lr5e-5 [--lr 5e-5 --steps 2000]
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
OUT_ROOT = os.path.join(HERE, "out")


def log(s=""):
    print(s, flush=True)


def noised(x0, t, key):
    e = jnp.exp(-t)
    return e * x0 + jnp.sqrt(T * (1.0 - e * e)) * jr.normal(key, x0.shape, x0.dtype)


def make_mirror(run_dir, mirror):
    """A folder that sample_tree.py reads like the run: relative links to every file of the run except samples/."""
    os.makedirs(os.path.join(mirror, "samples"), exist_ok=True)
    for name in os.listdir(run_dir):
        if name in ("samples", "LOCK"):
            continue
        dst = os.path.join(mirror, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.relpath(os.path.join(run_dir, name), mirror), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ema-steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=lm.BATCH_SIZE)
    ap.add_argument("--val-n", type=int, default=2048)
    ap.add_argument("--val-chunk", type=int, default=512, help="samples per validation call (64x64: 128, GPU memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-slices", type=int, default=lm.N_TIME_STEPS, help="stop after this slice index (tests)")
    args = ap.parse_args()
    run_dir = os.path.join(lm.OUT_ROOT, args.run)        # an absolute path (e.g. a run of the review folder) stays as it is
    mirror = os.path.join(OUT_ROOT, f"{os.path.basename(os.path.normpath(args.run))}__J-{args.tag}")   # always below this folder's out/
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    problem, K = cfg["problem"], int(cfg["K"])
    if cfg.get("hidden") != "potts" or not cfg.get("quad") or not cfg.get("local") or cfg.get("visible") != "quartic":
        raise SystemExit("train_J is written for the locked recipe (--hidden potts --quad --local --visible quartic)")
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}; train_J; job {os.environ.get('SLURM_JOB_ID', 'manual')}; {jax.devices()}")
    log(f"run {args.run}; K {K}; patches {cfg.get('patch')} from slice {cfg.get('patch_on_slice')}, {cfg.get('patch2')} "
        f"from slice {cfg.get('patch2_on_slice')}; lr {args.lr}, {args.steps} steps per slice, EMA {args.ema_steps}; -> {mirror}")
    prob = lm.make_problem(problem, radius=cfg.get("radius"), neighbourhood=cfg.get("neighbourhood", "chebyshev"))
    N = prob["n"]
    vis = lm.make_visible(prob["conn"], N, "stencil", lm.parse_pair_terms(cfg.get("pair_terms", "all")), sextic=False)
    train = jnp.asarray(np.asarray(prob["data"]), jnp.float32)
    ds = lm.problem_dataset(problem)
    raw_te = lm.load_cifar_raw("test")[0] if ds == "cifar" else lm.load_celeba_raw("test", lm.celeba_grid(problem))[0]
    rng = np.random.default_rng(args.seed)
    te = raw_te[rng.choice(raw_te.shape[0], args.val_n, replace=False)]
    test = jnp.asarray((te + lm.STD_ADDED_NOISE * rng.standard_normal(te.shape) - prob["mean"]) / prob["std"])
    theta_hist = np.load(os.path.join(run_dir, "params_history.npy"))
    with np.load(os.path.join(run_dir, "hidden_history.npz")) as z:
        hid = {name: np.asarray(z[name]) for name in z.files}
    taus = np.asarray(hid["t"])
    on = [cfg.get("patch_on_slice") or 0, cfg.get("patch2_on_slice") or 0, cfg.get("patch3_on_slice") or 0]
    set_ids = [i for i, (kw, _, _) in enumerate(lm.PATCH_SETS) if kw in hid]
    if not set_ids:
        raise SystemExit("the run has no patch units")
    J_hist = {tm.J_KEYS[i]: np.zeros((lm.N_TIME_STEPS, hid[lm.PATCH_SETS[i][0]].shape[1], K,
                                      hid[lm.PATCH_SETS[i][0]].shape[2]), np.float32) for i in set_ids}
    tree32 = tm.make_tree_loss(vis, T, True, compute_dtype=jnp.float32)
    val_ps = jax.jit(tm.make_tree_loss_per_sample(vis, T, True))
    opt = optax.adam(args.lr)
    w_ema = 1.0 / args.ema_steps

    @jax.jit
    def run_slice(params, Jd, t, key):
        """--steps Adam steps on J (params frozen); returns the EMA of the iterates and the mean training loss."""
        def body(carry, i):
            Jc, st, ema, acc = carry
            k1, k2 = jr.split(jr.fold_in(key, i))
            X = noised(train[jr.choice(k1, train.shape[0], (args.batch,))].astype(jnp.float64), t, k2)
            v, g = jax.value_and_grad(lambda Jx: tree32(dict(params, **Jx), X))(Jc)
            up, st = opt.update(g, st)
            Jc = optax.apply_updates(Jc, up)
            ema = jax.tree_util.tree_map(lambda e, j: e + w_ema * (j - e), ema, Jc)
            return (Jc, st, ema, acc + v), None
        (Jc, _, ema, acc), _ = jax.lax.scan(body, (Jd, opt.init(Jd), Jd, jnp.zeros((), jnp.float64)), jnp.arange(args.steps))
        return ema, acc / args.steps

    def val(params, Jd, Xv):
        c = args.val_chunk
        return np.concatenate([np.asarray(val_ps(dict(params, **Jd), Xv[i:i + c])) for i in range(0, Xv.shape[0], c)])

    key = jr.PRNGKey(args.seed)
    Jd, records, t0 = {}, [], time.time()
    first = min(on[i] for i in set_ids)
    for k in range(first, min(lm.N_TIME_STEPS, args.max_slices + 1)):
        t = float(taus[k])
        params = {"theta": jnp.asarray(theta_hist[k]), "W": jnp.asarray(hid["W"][k]), "b": jnp.asarray(hid["b"][k]),
                  "Q": jnp.asarray(hid["Q"][k]), "wl": jnp.asarray(hid["wl"][k]), "bl": jnp.asarray(hid["bl"][k])}
        for i in set_ids:
            kw, kb, kq = lm.PATCH_SETS[i]
            if k >= on[i]:
                params[kw], params[kb] = jnp.asarray(hid[kw][k], jnp.float64), jnp.asarray(hid[kb][k], jnp.float64)
                if kq in hid:
                    params[kq] = jnp.asarray(hid[kq][k], jnp.float64)
                if tm.J_KEYS[i] not in Jd:                     # the set switches on: its J starts at zero
                    Jd[tm.J_KEYS[i]] = jnp.zeros(J_hist[tm.J_KEYS[i]].shape[1:], jnp.float64)
        kk = jr.fold_in(key, k)
        Xv = noised(test, t, jr.fold_in(kk, 1))
        J0 = {kj: jnp.zeros_like(v) for kj, v in Jd.items()}
        v0, vw = val(params, J0, Xv), val(params, Jd, Xv)
        Jd, tr = run_slice(params, Jd, t, jr.fold_in(kk, 2))
        v1 = val(params, Jd, Xv)
        d, dw = v1 - v0, vw - v0
        for kj, v in Jd.items():
            J_hist[kj][k] = np.asarray(v, np.float32)
        rec = {"slice": k, "t": t, "val_J0": float(v0.mean()), "gain": float(d.mean()), "gain_se": float(d.std() / np.sqrt(d.size)),
               "gain_warm_start": float(dw.mean()), "train_loss": float(tr),
               "rms_J_over_T": {kj: float(jnp.sqrt(jnp.mean(v * v))) / T for kj, v in Jd.items()}}
        records.append(rec)
        log(f"slice {k:2d} t {t:.3f}: held-out loss {v0.mean():11.2f} (J = 0); warm start {dw.mean():+8.2f}; result "
            f"{d.mean():+8.2f} +- {rec['gain_se']:.2f}; rms J / T "
            + ", ".join(f"{kj} {v:.3f}" for kj, v in rec["rms_J_over_T"].items()) + f"; {time.time() - t0:.0f} s")
    make_mirror(run_dir, mirror)
    np.savez(os.path.join(mirror, "J_history.npz"), **J_hist)
    with open(os.path.join(mirror, "train_J.json"), "w") as fh:
        json.dump({"run": args.run, "tag": args.tag, "lr": args.lr, "steps": args.steps, "ema_steps": args.ema_steps,
                   "batch": args.batch, "seed": args.seed, "slurm_job": os.environ.get("SLURM_JOB_ID", "manual"),
                   "slices": records}, fh, indent=1)
    g = np.array([r["gain"] for r in records])
    log(f"gain summed over the {len(records)} slices {g.sum():+.1f}; slices with a gain beyond 3 standard errors: "
        f"{int(sum(r['gain'] < -3 * r['gain_se'] for r in records))}; worse beyond 3: "
        f"{int(sum(r['gain'] > 3 * r['gain_se'] for r in records))}")
    log(f"wrote {mirror} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
