#!/usr/bin/env python3
"""How good is the local CD-1 (short-release) rule for the hidden parameters on a trained model?

DESIGN section 2: with x' = x - (dt/2) grad F(x) + sqrt(T dt) xi (a short release of the pixels while the fast hidden
units stay in equilibrium),
    g_cd1 = 2 / (T^2 dt) * [ <dF/dtheta_h>_data - <dF/dtheta_h>_after release ]  =  dJ_T/dtheta_h + O(dt)
in expectation over the release noise. Here dF/dtheta_h are the local Hebbian terms (-p_m x_i, -p_m, -p_m x_i^2 for a
Potts unit; -m_i x_i, -m_i for the per-pixel units).

For each chosen slice of a trained run, on one batch X ~ p_tau of the training distribution:
  * exact: dJ_T/dtheta_h by autodiff of the training loss (what the runs were trained with);
  * minibatch noise: the exact gradient on a second, independent batch (the noise the training already tolerates);
  * CD-1 for each release length dt and number of releases R per sample: the error of one estimate, its cosine with
    the exact gradient, and the error of the mean of `draws` estimates (bias plus noise / sqrt(draws));
    with --cd1-samples S > 0 the hidden statistics of the CD-1 terms are measured from S sampled hidden states per
    released configuration (R * S at the data) instead of being exact averages;
  * S = <sum_m p_m |G_m|^2> (or max wl^2): the fields' gradient scale; dt = kappa^2 T / S moves a field by kappa T.
Errors are relative, |g_hat - g| / |g|, per parameter group (W, b, Q, wl, bl).

Usage (GPU job): python cd1_check.py --run mnist14_K128_ema_potts_quad_local [--slices 1,50,99] [--dts ...]
Writes out/latent_physical_SGM/cd1_check/<job>/report.txt and metrics.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402  (enables float64)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from physical_diffusion_fns.helper_fns import sample_forward_process  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(np.ravel(a - b)) / max(np.linalg.norm(np.ravel(b)), 1e-300))


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under out/latent_physical_SGM")
    ap.add_argument("--slices", default="1,25,50,75,90,99")
    ap.add_argument("--dts", default="1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5")
    ap.add_argument("--releases", default="1,16")
    ap.add_argument("--cd1-samples", default="0",
                    help="CD-1 with the hidden statistics measured from S sampled hidden states per released "
                         "configuration (releases * S at the data); comma list, 0 = exact averages")
    ap.add_argument("--draws", type=int, default=32, help="independent CD-1 estimates per (slice, dt, R)")
    ap.add_argument("--batch", type=int, default=lm.BATCH_SIZE)
    ap.add_argument("--samples", default="",
                    help="also: force matching with S sampled hidden states per training sample (comma list, e.g. "
                         "4,16,64,256): error of the plug-in gradient of all parameters against the exact one; SxR "
                         "(e.g. 64x4) averages R independent S-draw estimates (make_loss replicates); a suffix o2 "
                         "(e.g. 16o2, 16x16o2) uses the second-order jackknife (needs --samples-jk, S divisible by 4)")
    ap.add_argument("--skip-cd1", action="store_true")
    ap.add_argument("--samples-jk", action="store_true", help="--samples with the jackknife bias correction")
    ap.add_argument("--local-rule", action="store_true",
                    help="also: the explicit local force-matching rule (latent_model.force_matching_local_grad for Potts "
                         "units, latent_model.ising_force_matching_local_grad for Ising units) against autodiff; "
                         "agreement to rounding error verifies the derivation")
    args = ap.parse_args()

    job = os.environ.get("SLURM_JOB_ID", "manual")
    out_dir = os.path.join(lm.OUT_ROOT, "cd1_check", job)
    os.makedirs(out_dir, exist_ok=True)
    rep = open(os.path.join(out_dir, "report.txt"), "w")

    def say(line=""):
        print(line, flush=True)
        rep.write(line + "\n")

    run_dir = os.path.join(lm.OUT_ROOT, args.run)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    K, hidden, local = cfg["K"], cfg.get("hidden", "ising"), cfg.get("local", False)
    prob = lm.make_problem(cfg.get("problem", "mnist28"), cfg.get("radius"), cfg.get("neighbourhood", "chebyshev"))
    data, n = prob["data"], prob["n"]
    T = lm.TEMP
    vis = lm.DenseVisible(prob["conn"], n)
    taus = lm.forward_time_pts()
    slices = [int(s) for s in args.slices.split(",")]
    dts = [float(s) for s in args.dts.split(",")]
    Rs = [int(s) for s in args.releases.split(",")]

    loss = lm.make_loss(vis, T, K, hidden, local)
    g_exact = jax.jit(jax.grad(loss))
    Ss = args.samples.split(",") if args.samples else []          # tokens "S" or "SxR"

    def s_r(tok):                                 # "S", "SxR", with an optional suffix "o2" (second-order jackknife)
        order = 2 if tok.endswith("o2") else 1
        S, _, R = tok[:-2].partition("x") if order == 2 else tok.partition("x")
        return int(S), int(R or 1), order
    if any(s_r(s)[2] == 2 for s in Ss) and not args.samples_jk:
        raise SystemExit("--samples tokens with o2 need --samples-jk")

    def jk_of(tok):
        return (2 if s_r(tok)[2] == 2 else True) if args.samples_jk else False
    g_sampled = {s: jax.jit(jax.grad(lm.make_loss(vis, T, K, hidden, local, samples=s_r(s)[0],
                                                  jackknife=jk_of(s), replicates=s_r(s)[1])))
                 for s in Ss}
    cSs = [int(s) for s in args.cd1_samples.split(",")]
    cd1 = {s: jax.jit(lm.make_cd1_hidden_grad(vis, T, K, hidden, local, samples=s), static_argnums=4) for s in cSs}
    combos = [(dt, R, s) for dt in dts for R in Rs for s in cSs]

    def label(dt, R, s):
        return f"dt={dt:.0e} R={R}" + (f" S={s}" if s else "")
    scale = jax.jit(lambda p, X: lm.hidden_field_scale(p, X, T, K, hidden, local))
    rule_fn = lm.ising_force_matching_local_grad if hidden == "ising" else lm.force_matching_local_grad
    local_rule = jax.jit(lambda p, X: rule_fn(p, X, T, vis, local))

    say(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; cd1_check; job {job}; run {args.run}; "
        f"problem {cfg.get('problem', 'mnist28')}, hidden {hidden}, K {K}, quad {cfg.get('quad', False)}, local {local}, "
        f"visible {cfg.get('visible', 'sextic')}; batch {args.batch}; draws {args.draws}; T {T}")
    say("rel = |g_hat - g| / |g| on the same batch; 'minibatch' = exact gradient on an independent batch; "
        "'one' = one CD-1 estimate; 'mean' = mean of the draws (bias + noise/sqrt(draws)); cos of one estimate")
    metrics = {"run": args.run, "job": job, "slices": {}}
    for k in slices:
        z = np.load(os.path.join(run_dir, "slices", f"slice_{k:03d}.npz"))
        params = {"theta": jnp.asarray(z["theta"]), "W": jnp.asarray(z["W"]), "b": jnp.asarray(z["b"])}
        for extra in ("Q", "wl", "bl"):
            if extra in z.files:
                params[extra] = jnp.asarray(z[extra])
        t = float(taus[k])
        X = sample_forward_process(t, args.batch, T, lm.SIGMA_FORWARD, data, jr.PRNGKey(10_000 + k), beta=1)
        X2 = sample_forward_process(t, args.batch, T, lm.SIGMA_FORWARD, data, jr.PRNGKey(20_000 + k), beta=1)
        g = g_exact(params, X)
        g2 = g_exact(params, X2)
        keys = [h for h in lm.HIDDEN_KEYS if h in params]
        S = float(scale(params, X))
        say("")
        say(f"slice {k} (t = {t:.4f}): field scale S = {S:.4g}; dt for kappa = 0.3 / 0.1: "
            f"{0.09 * T / S:.3g} / {0.01 * T / S:.3g}")
        sm = {"t": t, "S": S, "groups": {}, "samples": {}}
        if args.local_rule and K > 0:              # Potts: force_matching_local_grad; Ising: its Ising counterpart
            gl = local_rule(params, X)
            errs = {h: rel(np.asarray(gl[h]), np.asarray(g[h])) for h in gl}
            say("  local force-matching rule vs autodiff, relative difference: "
                + ", ".join(f"{h} {e:.2e}" for h, e in errs.items()))
            sm["local_rule_rel"] = errs
        if Ss:                                    # force matching with S sampled hidden states, all parameter groups
            say(f"  force matching with S sampled hidden states" + (" (jackknife)" if args.samples_jk else " (plug-in)")
                + f" (all groups): rel one / rel mean of {args.draws} / cos; "
                f"minibatch = exact gradient on an independent batch")
            say(f"  {'group':5s} {'|g|':>10s} {'minibatch':>10s}  " + "  ".join(f"{'S=' + str(s):>20s}" for s in Ss))
            est = {s: [g_sampled[s](params, X, jr.fold_in(jr.PRNGKey(40_000 + k), d)) for d in range(args.draws)]
                   for s in Ss}
            for h in ["theta"] + keys:
                gh, g2h = np.asarray(g[h]), np.asarray(g2[h])
                cells, rec = [], {"norm": float(np.linalg.norm(gh)), "minibatch_rel": rel(g2h, gh)}
                for s in Ss:
                    E = np.stack([np.asarray(e[h]) for e in est[s]])
                    one = float(np.mean([rel(e, gh) for e in E]))
                    mean = rel(E.mean(axis=0), gh)
                    c = float(np.mean([cos(e, gh) for e in E]))
                    cells.append(f"{one:6.3g}/{mean:6.3g}/{c:5.2f}")
                    rec[f"S={s}"] = {"one_rel": one, "mean_rel": mean, "cos": c}
                say(f"  {h:5s} {rec['norm']:10.4g} {rec['minibatch_rel']:10.3g}  " + "  ".join(f"{c:>20s}" for c in cells))
                sm["samples"][h] = rec
        if args.skip_cd1:
            metrics["slices"][str(k)] = sm
            continue
        say(f"  {'group':5s} {'|g|':>10s} {'minibatch':>10s}  " + "  ".join(
            f"{label(dt, R, s):>22s}" for dt, R, s in combos))
        say(f"  {'':5s} {'':>10s} {'rel':>10s}  " + "  ".join(f"{'one/mean/cos':>22s}" for _ in combos))
        ests = {}
        for dt, R, s in combos:
            draws = [cd1[s](params, X, jr.fold_in(jr.PRNGKey(30_000 + k), d), dt, R) for d in range(args.draws)]
            ests[(dt, R, s)] = {h: np.stack([np.asarray(e[h]) for e in draws]) for h in keys}
        for h in keys:
            gh, g2h = np.asarray(g[h]), np.asarray(g2[h])
            cells, rec = [], {"norm": float(np.linalg.norm(gh)), "minibatch_rel": rel(g2h, gh), "cd1": {}}
            for dt, R, s in combos:
                E = ests[(dt, R, s)][h]
                one = float(np.mean([rel(e, gh) for e in E]))
                mean = rel(E.mean(axis=0), gh)
                c = float(np.mean([cos(e, gh) for e in E]))
                cells.append(f"{one:7.3g}/{mean:6.3g}/{c:5.2f}")
                rec["cd1"][f"dt={dt:g},R={R}" + (f",S={s}" if s else "")] = {"one_rel": one, "mean_rel": mean, "cos": c}
            say(f"  {h:5s} {rec['norm']:10.4g} {rec['minibatch_rel']:10.3g}  " + "  ".join(f"{s:>22s}" for s in cells))
            sm["groups"][h] = rec
        metrics["slices"][str(k)] = sm
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    say("")
    say("reading: CD-1 is usable where 'one' (R releases) is not much larger than the minibatch noise, and unbiased "
        "where 'mean' falls like 'one'/sqrt(draws) as dt shrinks")
    rep.close()


if __name__ == "__main__":
    main()
