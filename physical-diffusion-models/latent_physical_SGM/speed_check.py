#!/usr/bin/env python3
"""Checks and timings for the faster training paths (--visible-impl, --precision mixed); a GPU job.

  1. environment: jax version, device, default matmul precision
  2. visible energy: StencilVisible and DenseVisible(terms=...) against DenseVisible on several lattices (gradient,
     Laplacian, energy; autodiff of the stencil energy and the trace of its Hessian); loss and parameter gradient
     (K = 0, and Potts K = 3 --quad --local); determinism of the stencil gradient
  3. HLO of the mixed loss: every product float32 at HIGHEST precision, no float64 product left; GEMM kernel counts
  4. float32 vs float64 on trained slices: loss and gradients on real batches; relative error, cosine, error /
     batch-to-batch noise, bias, rarely visited Potts states, field errors; with HIGHEST and with DEFAULT precision
     (a large DEFAULT error means XLA uses TF32 for float32 products by default)
  5. speed of the training step (train_latent.build_runner) for dense / dense_skip / stencil x f64 / mixed at
     K = 0 (pixels only), 784, 3072, 6144 (Potts --quad --local), and of loss + gradient alone (the difference is
     sampling and optimizer bookkeeping)

Usage (from latent_physical_SGM/):  sbatch --time=02:00:00 jobs/run_gpu_neuronic.sh speed_check.py
Writes out/latent_physical_SGM/speed_check/<job>/report.txt and results.json. Exit code 1 if a correctness check
or the float32 acceptance criteria fail (so dependent jobs do not start).
"""

import json
import os
import sys
import time
import traceback
from functools import partial

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402  (enables float64)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from physical_diffusion_fns.helper_fns import sample_forward_process  # noqa: E402
import train_latent  # noqa: E402

T = lm.TEMP
JOB = os.environ.get("SLURM_JOB_ID", "manual")
OUT = os.path.join(lm.OUT_ROOT, "speed_check", JOB)
os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(OUT, "report.txt")
RESULTS = {"job": JOB, "checks": {}, "sections": {}}
FAILED = []

# float32 acceptance criteria (per slice, per parameter group, on the 'mixed' path vs float64)
ACC = {"median_rel": 1e-3, "min_cos": 0.9999, "median_ratio": 1e-2, "max_ratio": 5e-2}


def log(msg=""):
    print(msg, flush=True)
    with open(REPORT, "a") as f:
        f.write(msg + "\n")


def check(name, ok, detail):
    RESULTS["checks"][name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        FAILED.append(name)
    log(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")


def relmax(a, b):
    """max |a - b| / max |b|"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300)) if b.size else 0.0


def section(name, fn):
    log(f"\n=== {name} ===")
    t0 = time.time()
    try:
        out = fn()
        RESULTS["sections"][name] = out if out is not None else "done"
    except Exception:
        tb = traceback.format_exc()
        log(tb)
        FAILED.append(f"section {name} crashed")
        RESULTS["sections"][name] = {"crashed": tb}
    log(f"--- {name}: {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------------------------------------------
# 1. environment
# ---------------------------------------------------------------------------------------------------------------
def env():
    try:
        prec = jax.config.jax_default_matmul_precision
    except Exception as e:  # noqa: BLE001
        prec = f"? ({e})"
    info = {"jax": jax.__version__, "devices": str(jax.devices()), "default_matmul_precision": str(prec),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"), "XLA_FLAGS": os.environ.get("XLA_FLAGS")}
    for k, v in info.items():
        log(f"  {k}: {v}")
    return info


# ---------------------------------------------------------------------------------------------------------------
# 2. visible energy: stencil and dense_skip vs dense
# ---------------------------------------------------------------------------------------------------------------
def rand_theta(key, n, P, sextic=True):
    ks = jr.split(key, 7)
    return jnp.concatenate([1.0 + 0.3 * jr.uniform(ks[0], (n,)),
                            0.5 + 0.3 * jr.uniform(ks[1], (n,)),
                            (0.1 + 0.1 * jr.uniform(ks[2], (n,))) if sextic else jnp.zeros(n),
                            0.3 * jr.normal(ks[3], (P,)),
                            0.3 * jr.normal(ks[4], (P,)),
                            0.2 * jr.uniform(ks[5], (P,)),
                            0.2 * jr.normal(ks[6], (n,))])


def lin_only(theta, n, P):
    """theta of the --pair-terms lin quartic model: c_om, c_duff and k_6 exactly zero."""
    return (theta.at[lm.com_slice(n, P)].set(0.0).at[lm.cduff_slice(n, P)].set(0.0)
            .at[lm.k6_slice(n)].set(0.0))


def visible_checks():
    tol = 1e-11
    out = {}
    lattices = [(28, 1, "chebyshev"), (28, 2, "chebyshev"), (28, 1, "von_neumann"), (28, 3, "von_neumann"),
                (14, 1, "chebyshev"), (8, 7, "chebyshev")]
    for i, (G, rad, nb) in enumerate(lattices):
        conn = lm.connectivity(G, rad, nb)
        n, P = G * G, conn.shape[0]
        dense, st = lm.DenseVisible(conn, n), lm.StencilVisible(conn, n)
        theta = rand_theta(jr.PRNGKey(10 + i), n, P)
        X = 0.8 * jr.normal(jr.PRNGKey(20 + i), (64, n))
        gd, ld = jax.jit(dense.grad_lap)(theta, X)
        gs, ls = jax.jit(st.grad_lap)(theta, X)
        ed, es = jax.jit(dense.energy)(theta, X), jax.jit(st.energy)(theta, X)
        g_ad = jax.jit(jax.vmap(jax.grad(lambda x: st.energy(theta, x[None])[0])))(X)
        hess_tr = jnp.stack([jnp.trace(jax.hessian(lambda x: st.energy(theta, x[None])[0])(X[s])) for s in range(2)])
        r = {"pairs": int(P), "offsets": len(st.offsets), "grad": relmax(gs, gd), "lap": relmax(ls, ld),
             "energy": relmax(es, ed), "grad_vs_autodiff": relmax(g_ad, gs), "lap_vs_hessian": relmax(hess_tr, ls[:2])}
        out[f"{G}x{G} {nb} r{rad}"] = r
        check(f"stencil {G}x{G} {nb} r{rad} ({P} pairs, {len(st.offsets)} offsets)",
              max(r["grad"], r["lap"], r["energy"], r["grad_vs_autodiff"], r["lap_vs_hessian"]) < tol,
              {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in r.items()})

    # the trained model's case: 28x28 radius 1, --pair-terms lin, quartic
    conn = lm.connectivity(28, 1)
    n, P = 784, conn.shape[0]
    theta = lin_only(rand_theta(jr.PRNGKey(3), n, P), n, P)
    X = 0.8 * jr.normal(jr.PRNGKey(4), (512, n))
    dense = lm.DenseVisible(conn, n)
    skip = lm.make_visible(conn, n, "dense_skip", ("lin",), sextic=False)
    st = lm.make_visible(conn, n, "stencil", ("lin",), sextic=False)
    gd, ld = jax.jit(dense.grad_lap)(theta, X)
    gk, lk = jax.jit(skip.grad_lap)(theta, X)
    gs, ls = jax.jit(st.grad_lap)(theta, X)
    bitwise = bool(np.array_equal(np.asarray(gk), np.asarray(gd)) and np.array_equal(np.asarray(lk), np.asarray(ld)))
    check("dense_skip vs dense (lin, quartic)", max(relmax(gk, gd), relmax(lk, ld)) < tol,
          f"grad {relmax(gk, gd):.2e}, lap {relmax(lk, ld):.2e}, bitwise equal: {bitwise}")
    check("stencil vs dense (lin, quartic)", max(relmax(gs, gd), relmax(ls, ld)) < tol,
          f"grad {relmax(gs, gd):.2e}, lap {relmax(ls, ld):.2e}")
    out["dense_skip_bitwise"] = bitwise

    # loss and parameter gradient: K = 0, and Potts K = 3 --quad --local
    k1, k2, k3, k4, k5 = jr.split(jr.PRNGKey(6), 5)
    cases = {"K0": (0, "ising", False, {"theta": theta, "W": jnp.zeros((0, n)), "b": jnp.zeros((0,))}),
             "potts3_quad_local": (3, "potts", True, {"theta": theta, "W": 0.1 * jr.normal(k1, (3, n)),
                                                      "b": 0.05 * jr.normal(k2, (3,)),
                                                      "Q": 0.05 * jr.normal(k3, (3, n)),
                                                      "wl": 0.5 * jr.normal(k4, (n,)),
                                                      "bl": 0.1 * jr.normal(k5, (n,))})}
    for name, (K, hidden, local, params) in cases.items():
        vg_d = jax.jit(jax.value_and_grad(lm.make_loss(dense, T, K, hidden, local)))
        vg_s = jax.jit(jax.value_and_grad(lm.make_loss(st, T, K, hidden, local)))
        vg_k = jax.jit(jax.value_and_grad(lm.make_loss(skip, T, K, hidden, local)))
        vg_m = jax.jit(jax.value_and_grad(lm.make_loss(st, T, K, hidden, local, compute_dtype=jnp.float32)))
        (Ld, gd), (Ls, gs), (Lk, gk), (Lm, gm) = vg_d(params, X), vg_s(params, X), vg_k(params, X), vg_m(params, X)
        kept = np.ones(params["theta"].shape[0], bool)
        for s in (lm.com_slice(n, P), lm.cduff_slice(n, P), lm.k6_slice(n)):
            kept[s] = False
        errs = {g: relmax(np.asarray(gs[g])[kept] if g == "theta" else gs[g],
                          np.asarray(gd[g])[kept] if g == "theta" else gd[g]) for g in gd if np.asarray(gd[g]).size}
        errs_k = {g: relmax(np.asarray(gk[g])[kept] if g == "theta" else gk[g],
                            np.asarray(gd[g])[kept] if g == "theta" else gd[g]) for g in gd if np.asarray(gd[g]).size}
        dropped_zero = bool(np.all(np.asarray(gs["theta"])[~kept] == 0.0))
        check(f"loss + gradient, stencil vs dense, {name}",
              relmax(Ls, Ld) < 1e-10 and max(errs.values()) < 1e-10 and dropped_zero,
              f"loss {relmax(Ls, Ld):.2e}; grads {', '.join(f'{g} {e:.1e}' for g, e in errs.items())}; "
              f"dropped theta entries zero: {dropped_zero}")
        check(f"loss + gradient, dense_skip vs dense, {name}",
              relmax(Lk, Ld) < 1e-10 and max(errs_k.values()) < 1e-10,
              f"loss {relmax(Lk, Ld):.2e}; grads {', '.join(f'{g} {e:.1e}' for g, e in errs_k.items())}")
        em = {g: relmax(np.asarray(gm[g])[kept] if g == "theta" else gm[g],
                        np.asarray(gd[g])[kept] if g == "theta" else gd[g]) for g in gd if np.asarray(gd[g]).size}
        log(f"  (info) mixed vs f64, {name}: loss {relmax(Lm, Ld):.2e}; grads "
            f"{', '.join(f'{g} {e:.1e}' for g, e in em.items())}; grad dtypes "
            f"{sorted({str(np.asarray(v).dtype) for v in gm.values()})}")
        # determinism: the same compiled function twice, and a fresh compilation
        L2, g2 = vg_s(params, X)
        vg_s2 = jax.jit(jax.value_and_grad(lm.make_loss(st, T, K, hidden, local)))
        L3, g3 = vg_s2(params, X)
        same = all(np.array_equal(np.asarray(gs[g]), np.asarray(g2[g])) and np.array_equal(np.asarray(gs[g]),
                                                                                         np.asarray(g3[g]))
                   for g in gs) and float(Ls) == float(L2) == float(L3)
        check(f"stencil gradient deterministic, {name}", same, "bitwise equal over 3 evaluations (2 compilations)")
    return out


# ---------------------------------------------------------------------------------------------------------------
# 3. HLO of the mixed loss
# ---------------------------------------------------------------------------------------------------------------
def kernel_counts(text):
    return {"cublas": text.count("cublas"), "triton": text.count("triton"),
            "loop_fusions": text.count("kind=kLoop"), "input_fusions": text.count("kind=kInput"),
            "output_fusions": text.count("kind=kOutput"), "custom_fusions": text.count("kind=kCustom")}


def hlo_checks():
    conn = lm.connectivity(28, 1)
    n, P, K = 784, conn.shape[0], 784
    theta = lin_only(rand_theta(jr.PRNGKey(3), n, P), n, P)
    k1, k2, k3, k4 = jr.split(jr.PRNGKey(7), 4)
    params = {"theta": theta, "W": 0.1 * jr.normal(k1, (K, n)), "b": jnp.zeros(K), "Q": 0.02 * jr.normal(k2, (K, n)),
              "wl": 0.5 * jr.normal(k3, (n,)), "bl": 0.1 * jr.normal(k4, (n,))}
    X = 0.8 * jr.normal(jr.PRNGKey(8), (512, n))
    out = {}
    for impl in ("stencil", "dense"):
        vis = lm.make_visible(conn, n, impl, ("lin",), sextic=False)
        lowered = jax.jit(jax.value_and_grad(lm.make_loss(vis, T, K, "potts", True,
                                                          compute_dtype=jnp.float32))).lower(params, X)
        dots = [ln for ln in lowered.as_text().splitlines() if "dot_general" in ln]
        n_f32 = sum("xf32>" in ln for ln in dots)
        n_f64 = sum("xf64>" in ln for ln in dots)
        n_hi = sum("HIGHEST" in ln for ln in dots)
        out[impl] = {"dots": len(dots), "f32": n_f32, "f64": n_f64, "highest": n_hi}
        check(f"HLO of the mixed loss ({impl} visible, Potts 784 quad local): all products float32 at HIGHEST",
              len(dots) > 0 and n_f64 == 0 and n_hi == len(dots),
              f"{len(dots)} dot_general, {n_f32} float32, {n_f64} float64, {n_hi} HIGHEST")
        if impl == "stencil":
            log(f"  (info) example product: {dots[0].strip()[:300]}")
        out[impl]["compiled"] = kernel_counts(lowered.compile().as_text())
        log(f"  (info) compiled kernels, mixed {impl}: {out[impl]['compiled']}")
    dense = lm.DenseVisible(conn, n)
    comp = jax.jit(jax.value_and_grad(lm.make_loss(dense, T, K, "potts", True))).lower(params, X).compile()
    out["f64_dense_compiled"] = kernel_counts(comp.as_text())
    log(f"  (info) compiled kernels, float64 dense (today): {out['f64_dense_compiled']}")
    return out


# ---------------------------------------------------------------------------------------------------------------
# 4. float32 vs float64 on trained slices
# ---------------------------------------------------------------------------------------------------------------
RUNS = [("mnist28_d0123_K784_ema_potts_quad_local_quartic_r1_lin_pooled", (0, 50, 80, 90, 95, 99)),
        ("mnist28_d0123456789_K3072_ema_potts_quad_local_quartic_r1_lin_pooled", (0, 50, 80, 95, 99)),
        ("mnist28_d0123_K784_ema_ising_quad_local_quartic_r1_lin", (0, 50, 80, 95, 99))]
NB = 16                       # batches of 512 per slice


def norm(a):
    return float(jnp.sqrt(jnp.sum(jnp.square(a))))


def precision_checks():
    taus = lm.forward_time_pts()
    out = {}
    for run, slices in RUNS:
        rdir = os.path.join(lm.OUT_ROOT, run)
        if not os.path.exists(os.path.join(rdir, "config.json")):
            log(f"  {run}: missing, skipped")
            continue
        with open(os.path.join(rdir, "config.json")) as f:
            cfg = json.load(f)
        prob = lm.make_problem(cfg["problem"], cfg.get("radius"), cfg.get("neighbourhood", "chebyshev"))
        n, conn, data = prob["n"], prob["conn"], prob["data"]
        terms = lm.parse_pair_terms(cfg.get("pair_terms", "duff,lin,om"))
        sextic = cfg.get("visible", "sextic") == "sextic"
        K, hidden, local = int(cfg["K"]), cfg.get("hidden", "ising"), bool(cfg.get("local", False))
        vis_d = lm.DenseVisible(conn, n)
        vis_s = lm.make_visible(conn, n, "stencil", terms, sextic)
        ps_plain = lm.make_loss_per_sample(vis_s, T, K, hidden, local)
        # theta entries that training uses: not the dropped pair groups (re-zeroed after every step) and not k_6 of
        # the quartic model (its gradient is zeroed); the dense path computes discarded gradients there, the stencil 0
        kept = np.ones(vis_d.n_theta, bool)
        for s in lm.dropped_pair_slices(n, vis_d.P, terms) + (() if sextic else (lm.k6_slice(n),)):
            kept[s] = False
        kept = jnp.asarray(kept)

        def mixed_default(params, X, key=None):        # float32 at XLA's DEFAULT precision (TF32 detection)
            p = {k: v.astype(jnp.float32) for k, v in params.items()}
            return jnp.mean(ps_plain(p, X.astype(jnp.float32)).astype(jnp.float64))

        fns = {"f64": lm.make_loss(vis_d, T, K, hidden, local),
               "f64_stencil": lm.make_loss(vis_s, T, K, hidden, local),
               "mixed": lm.make_loss(vis_s, T, K, hidden, local, compute_dtype=jnp.float32),
               "mixed_default": mixed_default}
        vg = {k: jax.jit(jax.value_and_grad(f)) for k, f in fns.items()}
        ps64 = jax.jit(lm.make_loss_per_sample(vis_d, T, K, hidden, local))
        ps32 = jax.jit(lm.make_loss_per_sample(vis_s, T, K, hidden, local, compute_dtype=jnp.float32))

        @jax.jit
        def fields64(W, Q, b, X):
            return X @ W.T + (X * X) @ Q.T + b

        @jax.jit
        def fields32_hi(W, Q, b, X):
            with jax.default_matmul_precision("float32"):
                W, Q, b, X = (a.astype(jnp.float32) for a in (W, Q, b, X))
                return X @ W.T + (X * X) @ Q.T + b

        @jax.jit
        def fields32_def(W, Q, b, X):
            W, Q, b, X = (a.astype(jnp.float32) for a in (W, Q, b, X))
            return X @ W.T + (X * X) @ Q.T + b

        log(f"\n  run {run} (K {K} {hidden}, local {local}, pair terms {terms}, {'sextic' if sextic else 'quartic'})")
        out[run] = {}
        for k in slices:
            path = os.path.join(rdir, "slices", f"slice_{k:03d}.npz")
            if not os.path.exists(path):
                log(f"    slice {k}: missing, skipped")
                continue
            z = np.load(path)
            params = {g: jnp.asarray(z[g]) for g in ("theta", "W", "b", "Q", "wl", "bl") if g in z.files}
            t = float(taus[k])
            groups = [g for g in params if params[g].size]
            variants = ("f64_stencil", "mixed", "mixed_default")
            st = {v: {g: {"rel": [], "cos": [], "ratio": [], "sum_e": None, "sum_e2": 0.0} for g in groups}
                  for v in variants}
            rare = {v: {"rel": [], "ratio": []} for v in variants}
            noise = {g: [] for g in groups}
            loss_rel_med, loss_rel_max, n_rare = [], [], []
            fld = {}
            prev = None
            for j in range(NB):
                X = sample_forward_process(t, lm.BATCH_SIZE, T, lm.SIGMA_FORWARD, data,
                                           jr.fold_in(jr.PRNGKey(4242 + k), j), beta=1)
                res = {v: vg[v](params, X) for v in vg}
                res = {v: (L, dict(g, theta=g["theta"][kept])) for v, (L, g) in res.items()}
                g64 = res["f64"][1]
                d = np.abs(np.asarray(ps32(params, X), np.float64) - np.asarray(ps64(params, X)))
                rel_ps = d / np.maximum(np.abs(np.asarray(ps64(params, X))), 1e-300)
                loss_rel_med.append(float(np.median(rel_ps)))
                loss_rel_max.append(float(np.max(rel_ps)))
                rare_mask = None
                if K > 0 and "Q" in params:
                    A64 = fields64(params["W"], params["Q"], params["b"], X)
                    if j == 0:
                        dz_hi = np.abs(np.asarray(fields32_hi(params["W"], params["Q"], params["b"], X), np.float64)
                                       - np.asarray(A64)) / T
                        dz_def = np.abs(np.asarray(fields32_def(params["W"], params["Q"], params["b"], X), np.float64)
                                        - np.asarray(A64)) / T
                        fld = {"max_abs_A_over_T": float(np.max(np.abs(np.asarray(A64)))) / T,
                               "dz_highest_p50_p99_max": [float(np.percentile(dz_hi, 50)),
                                                          float(np.percentile(dz_hi, 99)), float(np.max(dz_hi))],
                               "dz_default_p50_p99_max": [float(np.percentile(dz_def, 50)),
                                                          float(np.percentile(dz_def, 99)), float(np.max(dz_def))]}
                    if hidden == "potts":
                        occ = jnp.mean(jax.nn.softmax(A64 / T, axis=1), axis=0)
                        rare_mask = occ < 1e-4
                        n_rare.append(int(jnp.sum(rare_mask)))
                for g in groups:
                    if prev is not None:
                        noise[g].append(norm(g64[g] - prev[g]) / np.sqrt(2.0))
                for v in variants:
                    gv = res[v][1]
                    for g in groups:
                        e = gv[g] - g64[g]
                        en, gn = norm(e), norm(g64[g])
                        s = st[v][g]
                        s["rel"].append(en / max(gn, 1e-300))
                        s["cos"].append(float(jnp.sum(gv[g] * g64[g])) / max(norm(gv[g]) * gn, 1e-300))
                        if prev is not None:
                            s["ratio"].append(en / max(noise[g][-1], 1e-300))
                        s["sum_e"] = e if s["sum_e"] is None else s["sum_e"] + e
                        s["sum_e2"] += en * en
                    if rare_mask is not None and bool(jnp.any(rare_mask)) and prev is not None:
                        er = jnp.concatenate([(gv[g] - g64[g])[rare_mask].ravel() for g in ("W", "Q") if g in gv])
                        gr = jnp.concatenate([g64[g][rare_mask].ravel() for g in ("W", "Q") if g in gv])
                        nr = jnp.concatenate([(g64[g] - prev[g])[rare_mask].ravel() for g in ("W", "Q") if g in gv])
                        rare[v]["rel"].append(norm(er) / max(norm(gr), 1e-300))
                        rare[v]["ratio"].append(norm(er) / max(norm(nr) / np.sqrt(2.0), 1e-300))
                prev = g64
            rec = {"t": t, "loss_rel_median": float(np.median(loss_rel_med)), "loss_rel_max": float(max(loss_rel_max)),
                   "fields": fld, "n_rare_states_median": float(np.median(n_rare)) if n_rare else None}
            verdict = True
            lines = []
            for v in variants:
                rec[v] = {}
                for g in groups:
                    s = st[v][g]
                    bias = norm(s["sum_e"] / NB)
                    rms = np.sqrt(s["sum_e2"] / NB)
                    r = {"rel_median": float(np.median(s["rel"])), "rel_max": float(np.max(s["rel"])),
                         "cos_min": float(np.min(s["cos"])), "ratio_median": float(np.median(s["ratio"])),
                         "ratio_max": float(np.max(s["ratio"])),
                         "bias_z": float(bias / max(rms / np.sqrt(NB), 1e-300)),
                         "bias_over_noise": float(bias / max(np.median(noise[g]), 1e-300))}
                    rec[v][g] = r
                    if v == "mixed":
                        ok = (r["rel_median"] <= ACC["median_rel"] and r["cos_min"] >= ACC["min_cos"]
                              and r["ratio_median"] <= ACC["median_ratio"] and r["ratio_max"] <= ACC["max_ratio"])
                        verdict = verdict and ok
                    lines.append(f"      {v:13s} {g:5s} rel {r['rel_median']:.1e} (max {r['rel_max']:.1e}), "
                                 f"cos {r['cos_min']:.7f}, err/noise {r['ratio_median']:.1e} "
                                 f"(max {r['ratio_max']:.1e}), bias z {r['bias_z']:.1f}, "
                                 f"bias/noise {r['bias_over_noise']:.1e}")
                if rare[v]["rel"]:
                    rec[v]["rare_rows"] = {"rel_median": float(np.median(rare[v]["rel"])),
                                           "ratio_median": float(np.median(rare[v]["ratio"])),
                                           "ratio_max": float(np.max(rare[v]["ratio"]))}
                    lines.append(f"      {v:13s} rare  states (occupation < 1e-4, median {rec['n_rare_states_median']:.0f}): "
                                 f"rel {rec[v]['rare_rows']['rel_median']:.1e}, err/noise "
                                 f"{rec[v]['rare_rows']['ratio_median']:.1e} (max {rec[v]['rare_rows']['ratio_max']:.1e})")
            rec["mixed_pass"] = bool(verdict)
            out[run][k] = rec
            check(f"float32 (mixed) gradients, {run.split('_ema')[0]} {hidden}, slice {k} (t {t:.3f})", verdict,
                  f"per-sample loss rel error median {rec['loss_rel_median']:.1e} max {rec['loss_rel_max']:.1e}; "
                  f"fields {json.dumps(fld)}")
            for ln in lines:
                log(ln)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 5. speed of the training step
# ---------------------------------------------------------------------------------------------------------------
def speed():
    prob = lm.make_problem("mnist28_d0123", 1)
    conn, n, data = prob["conn"], prob["n"], prob["data"]
    P = conn.shape[0]
    terms = ("lin",)
    theta0 = lm.quartic_initial_theta(lm.paper_theta_layout(P, n)[0], n, P)
    zero_slices = lm.dropped_pair_slices(n, P, terms)
    for s in zero_slices:
        theta0 = theta0.at[s].set(0.0)
    t_bench = float(lm.forward_time_pts()[90])
    X = sample_forward_process(t_bench, lm.BATCH_SIZE, T, lm.SIGMA_FORWARD, data, jr.PRNGKey(9), beta=1)

    def make_params(K):
        if K == 0:
            return {"theta": theta0, "W": jnp.zeros((0, n)), "b": jnp.zeros((0,))}
        k1, k2, k3, k4 = jr.split(jr.PRNGKey(5), 4)
        return {"theta": theta0, "W": 0.1 * jr.normal(k1, (K, n)), "b": jnp.zeros(K),
                "Q": 0.02 * jr.normal(k2, (K, n)), "wl": 0.5 * jr.normal(k3, (n,)), "bl": 0.1 * jr.normal(k4, (n,))}

    def timed(call):
        """seconds per call of `call` (100 steps), after one warm-up (compilation) call"""
        tc = time.perf_counter()
        jax.block_until_ready(call())
        t_first = time.perf_counter() - tc
        tc = time.perf_counter()
        jax.block_until_ready(call())
        t1 = time.perf_counter() - tc
        reps = int(min(50, max(2, 5.0 / max(t1, 1e-3))))
        tc = time.perf_counter()
        for _ in range(reps):
            r = call()
        jax.block_until_ready(r)
        return (time.perf_counter() - tc) / reps, t_first

    variants = [("dense", "f64"), ("dense_skip", "f64"), ("stencil", "f64"), ("dense", "mixed"), ("stencil", "mixed")]
    out = {}
    log("  ms per step (full training step: sampling, loss + gradient, Adam, update, EMA, best) | loss + gradient "
        "+ a plain update only | first call (compile + 100 steps)")
    for K in (0, 784, 3072, 6144):
        for impl, prec in variants:
            name = f"K{K} {impl} {prec}"
            try:
                vis = lm.make_visible(conn, n, impl, terms, sextic=False)
                loss = lm.make_loss(vis, T, K, "potts", K > 0,
                                    compute_dtype=jnp.float32 if prec == "mixed" else None)
                run, adam = train_latent.build_runner(loss, data, n, 1000.0, "quartic", P, zero_slices=zero_slices)
                params = make_params(K)
                state = {"carry": train_latent.init_carry(params, adam, jr.PRNGKey(1))}
                lrs = {g: 1e-6 for g in params}

                def call_full():
                    state["carry"], l = run(state["carry"], t_bench, lrs, 100)
                    return l

                vg = jax.value_and_grad(loss)

                @partial(jax.jit, static_argnums=2)
                def loop(p, X, nsteps):
                    def body(p, _):
                        l, g = vg(p, X)
                        return jax.tree_util.tree_map(lambda a, b: a - 1e-12 * b, p, g), l
                    return jax.lax.scan(body, p, None, length=nsteps)

                pstate = {"p": params}

                def call_lg():
                    pstate["p"], l = loop(pstate["p"], X, 100)
                    return l

                s_full, first = timed(call_full)
                s_lg, _ = timed(call_lg)
                finite = bool(np.all(np.isfinite(np.asarray(call_full()))))
                r = {"ms_step": 10.0 * s_full, "ms_lossgrad": 10.0 * s_lg, "first_call_s": first, "finite": finite}
                if K == 784 and impl == "stencil" and prec == "mixed":
                    r["compiled_step_kernels"] = kernel_counts(
                        run.lower(state["carry"], t_bench, lrs, 100).compile().as_text())
                out[name] = r
                log(f"  {name:28s} {r['ms_step']:8.3f} ms | {r['ms_lossgrad']:8.3f} ms | {first:6.1f} s"
                    + ("" if finite else "   (non-finite losses)")
                    + (f"   kernels {r['compiled_step_kernels']}" if "compiled_step_kernels" in r else ""))
                del run, adam, state, pstate, loop
            except Exception:
                tb = traceback.format_exc()
                log(f"  {name}: crashed\n{tb}")
                out[name] = {"crashed": tb}
                FAILED.append(f"speed {name} crashed")
    base = {K: out.get(f"K{K} dense f64", {}).get("ms_step") for K in (0, 784, 3072, 6144)}
    log("\n  speed-up of the full step over dense f64:")
    for K in (0, 784, 3072, 6144):
        if base[K]:
            log(f"    K {K:5d}: " + ", ".join(f"{impl} {prec} {base[K] / out[f'K{K} {impl} {prec}']['ms_step']:.1f}x"
                                            for impl, prec in variants
                                            if out.get(f"K{K} {impl} {prec}", {}).get("ms_step")))
    return out


def main():
    # optional argument: the sections to run, e.g. "1,4" (default: all)
    todo = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"1", "2", "3", "4", "5"}
    log(f"speed_check job {JOB}, {time.strftime('%Y-%m-%d %H:%M:%S')}; sections {sorted(todo)}")
    for key, name, fn in (("1", "1 environment", env),
                          ("2", "2 visible energy: stencil / dense_skip vs dense", visible_checks),
                          ("3", "3 HLO of the mixed-precision loss", hlo_checks),
                          ("4", "4 float32 vs float64 on trained slices", precision_checks),
                          ("5", "5 speed of the training step", speed)):
        if key in todo:
            section(name, fn)
    RESULTS["failed"] = FAILED
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)
    log(f"\n{'ALL CHECKS PASSED' if not FAILED else 'FAILED: ' + '; '.join(FAILED)}")
    log(f"report: {REPORT}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
