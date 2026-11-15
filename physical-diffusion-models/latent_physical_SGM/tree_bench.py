#!/usr/bin/env python3
"""GPU cost and precision of the deep tree (train_latent.py --tree) at the real size, before the overnight runs
(2026-11-14). Parameters from a finished plain patch run's slice files (default: four digits, K784, patches 7:3:32 from
slice 31, seed 1), J = 0 and random J of rms 1 T and 3 T.

  1. one training step's loss and parameter gradient (make_loss, stencil visible, --precision mixed, batch 512):
     seconds per call (median of 10 after compilation) and the compiled executable's temporary memory, for the plain
     patch model (no J) and the tree with chunks of 32, 64 and 128 samples; the ratio tree / plain.
  2. mixed against float64 (dense visible) for the tree at slices 31 and 55: relative gradient error per group, cosine.
  3. the sampler drift (sample_latent.make_drift, the deep tree's 5-tuple; float64, vmapped over 250 trajectories as
     sample_latent --chunk 250): seconds per call and temporary memory, against the plain patch drift.
  4. the validation loss (make_loss_per_sample, dense float64, 4096 samples) with the tree: seconds and memory.
  5. the estimated training time: the plain run's steps per slice x the measured step time ratio.
Usage (from latent_physical_SGM/): sbatch --time=00:40:00 jobs/run_gpu_neuronic.sh ../latent_physical_SGM_tree_candidate/tree_bench.py
Writes out/latent_physical_SGM/tree_bench/<job>/report.txt."""

import argparse
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

from physical_diffusion_fns.helper_fns import sample_forward_process  # noqa: E402
import sample_latent as sl  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "tree_bench", os.environ.get("SLURM_JOB_ID", "manual"))
RUN = "mnist28_d0123_K784_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch7s3S32on31_seed1"


def out(s=""):
    print(s, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(s + "\n")


def timed(fn, *args, n=10):
    lowered = jax.jit(fn).lower(*args)
    comp = lowered.compile()
    try:
        mem = comp.memory_analysis()
        temp = getattr(mem, "temp_size_in_bytes", float("nan")) / 2**30
    except Exception:                                   # noqa: BLE001
        temp = float("nan")
    r = comp(*args)
    jax.block_until_ready(r)
    ts = []
    for _ in range(n):
        t0 = time.time()
        jax.block_until_ready(comp(*args))
        ts.append(time.time() - t0)
    return float(np.median(ts)), temp, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out(f"=== tree_bench {time.strftime('%Y-%m-%d %H:%M:%S')}; jax {jax.__version__}; devices {jax.devices()}; run {args.run}")
    T = lm.TEMP
    run_dir = os.path.join(lm.OUT_ROOT, args.run)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    prob = lm.make_problem(cfg["problem"], cfg.get("radius"))
    data, conn = prob["data"], prob["conn"]
    vis_st = lm.make_visible(conn, lm.N, "stencil", ("lin",), sextic=False)
    vis_d = lm.DenseVisible(conn)
    taus = lm.forward_time_pts()
    rng = np.random.default_rng(14)

    def params_at(k, j_rms):
        s = np.load(os.path.join(run_dir, "slices", f"slice_{k:03d}.npz"))
        p = {g: jnp.asarray(s[g]) for g in ("theta", "W", "b", "Q", "wl", "bl", "Wp", "bp", "Qp")}
        P, S = p["Wp"].shape[0], p["Wp"].shape[1]
        if j_rms is not None:
            p["J"] = jnp.asarray(j_rms * T * rng.standard_normal((P, p["W"].shape[0], S)))
        return p

    K = int(cfg["K"])
    # ---- 1. training step -----------------------------------------------------------------------------------------
    k = 55
    X = sample_forward_process(float(taus[k]), lm.BATCH_SIZE, T, lm.SIGMA_FORWARD, data, jr.PRNGKey(1), beta=1)
    loss_m = lm.make_loss(vis_st, T, K, "potts", True, compute_dtype=jnp.float32)
    vg = jax.value_and_grad(loss_m)
    p_plain = params_at(k, None)
    t_plain, m_plain, _ = timed(vg, p_plain, X)
    out(f"1. training step at slice {k} (t = {taus[k]:.3f}), batch {lm.BATCH_SIZE}, mixed: plain patches {t_plain * 1e3:.1f} ms, "
        f"temp memory {m_plain:.2f} GiB")
    best = None
    for chunk in (32, 64, 128):
        lm.TREE_CHUNK = chunk
        p_tree = params_at(k, 1.0)
        try:
            t_tree, m_tree, (L, G) = timed(vg, p_tree, X)
            out(f"   tree, chunk {chunk}: {t_tree * 1e3:.1f} ms ({t_tree / t_plain:.2f} x plain), temp memory {m_tree:.2f} "
                f"GiB; loss {float(L):.6g}, |grad J| {float(jnp.linalg.norm(G['J'])):.3g}, |grad bp| "
                f"{float(jnp.linalg.norm(G['bp'])):.3g}")
            if best is None or t_tree < best[1]:
                best = (chunk, t_tree)
        except Exception as e:                          # noqa: BLE001
            out(f"   tree, chunk {chunk}: FAILED {type(e).__name__}: {str(e)[:300]}")
    lm.TREE_CHUNK = 64
    out(f"   fastest chunk: {best}")

    # ---- 2. precision -------------------------------------------------------------------------------------------------
    vg_d = jax.jit(jax.value_and_grad(lm.make_loss(vis_d, T, K, "potts", True)))
    vg_m = jax.jit(vg)
    for k in (31, 55, 80):
        for j_rms in (0.0, 1.0, 3.0):
            p = params_at(k, j_rms)
            Xk = sample_forward_process(float(taus[k]), 256, T, lm.SIGMA_FORWARD, data, jr.PRNGKey(10 + k), beta=1)
            (Ld, Gd), (Lm, Gm) = vg_d(p, Xk), vg_m(p, Xk)
            relg = {g: float(jnp.linalg.norm(Gm[g] - Gd[g]) / jnp.maximum(jnp.linalg.norm(Gd[g]), 1e-300)) for g in Gd}
            vd = jnp.concatenate([Gd[g].ravel() for g in sorted(Gd)])
            vm = jnp.concatenate([Gm[g].ravel().astype(vd.dtype) for g in sorted(Gd)])
            cos = float(vd @ vm / (jnp.linalg.norm(vd) * jnp.linalg.norm(vm)))
            st = lm.tree_stats(p["W"], p["Q"], p["b"], p["Wp"], p["Qp"], p["bp"], p["J"], Xk, T)
            out(f"2. slice {k} (t = {taus[k]:.3f}), J rms {j_rms:g} T: loss {float(Ld):.6g} (mixed rel "
                f"{abs(float(Lm) - float(Ld)) / abs(float(Ld)):.1e}); gradient rel error J {relg['J']:.1e}, Wp "
                f"{relg['Wp']:.1e}, W {relg['W']:.1e}, theta {relg['theta']:.1e}, worst {max(relg.values()):.1e}; "
                f"cosine {cos:.8f}; I(g; s_p) {st['tree_mi_mean']:.3g}, eff global {st['tree_eff_global_median']:.3g}")

    # ---- 3. sampler drift ---------------------------------------------------------------------------------------------
    th, W, b, _, prov, problem = sl.load_params(args.run, with_harmonic=True, with_patch=True)
    Wp, bp, Qp = prov["patch"][:3]
    Jh = (1.0 * T * rng.standard_normal((Wp.shape[0], Wp.shape[1], W.shape[1], Wp.shape[2]))).astype(np.float32)
    kw = dict(problem=problem, hidden="potts", local=prov["local"], quad=prov["quad"], radius=prov["radius"],
              neighbourhood=prov["neighbourhood"], taus=sl.slice_times(prov))
    x = jnp.asarray(np.asarray(sample_forward_process(2.2, 250, T, lm.SIGMA_FORWARD, data, jr.PRNGKey(3), beta=1)))
    for lab, ptup in (("plain patches", (Wp, bp, Qp, "potts")), ("tree", (Wp, bp, Qp, "potts", Jh))):
        dr, _ = sl.make_drift(th, W, b, "sde", patch=ptup, **kw)
        f = jax.vmap(dr, in_axes=(None, 0, None))
        t_d, m_d, _ = timed(lambda x_: f(2.8, x_, ()), x)
        out(f"3. drift, {lab}, 250 trajectories, float64: {t_d * 1e3:.1f} ms per call, temp memory {m_d:.2f} GiB")
        del dr, f

    # ---- 4. validation loss ---------------------------------------------------------------------------------------------
    k = 55
    p = params_at(k, 1.0)
    Xv = sample_forward_process(float(taus[k]), 4096, T, lm.SIGMA_FORWARD, data, jr.PRNGKey(7), beta=1)
    t_v, m_v, _ = timed(lm.make_loss_per_sample(vis_d, T, K, "potts", True), p, Xv, n=2)
    out(f"4. validation loss, 4096 samples, dense float64, tree: {t_v:.2f} s, temp memory {m_v:.2f} GiB")
    t_s = time.time()
    lm.tree_stats(p["W"], p["Q"], p["b"], p["Wp"], p["Qp"], p["bp"], p["J"], Xv, T)
    out(f"   tree_stats on 4096 samples: {time.time() - t_s:.2f} s (with compilation)")

    # ---- 5. training time estimate --------------------------------------------------------------------------------------
    with open(os.path.join(run_dir, "train_log.jsonl")) as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    on = int(cfg.get("patch_on_slice", 0))
    sec = np.array([r.get("seconds", np.nan) for r in recs], dtype=float)
    ratio = best[1] / t_plain if best else float("nan")
    est = np.nansum(sec[:on]) + ratio * np.nansum(sec[on:])
    out(f"5. the plain run: {np.nansum(sec) / 3600:.2f} h ({np.nansum(sec[:on]) / 3600:.2f} h before slice {on}); with the "
        f"tree at {ratio:.2f} x per step from slice {on}: about {est / 3600:.2f} h (from slice 55: "
        f"{(np.nansum(sec[:55]) + ratio * np.nansum(sec[55:])) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
