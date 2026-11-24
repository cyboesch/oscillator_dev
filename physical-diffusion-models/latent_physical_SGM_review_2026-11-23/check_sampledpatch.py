#!/usr/bin/env python3
"""Check of measured (sampled) hidden statistics for patch units (2026-11-17): on a trained run with patch units, at
given slices and on one batch of forward-process samples, the force-matching loss and its parameter gradient with
S sampled hidden states per unit and sample (latent_model.make_loss, samples = S, jackknife) against the exact ones
(samples = 0), in float64 with the dense visible model. Per S and parameter group: the cosine between the sampled and
the exact gradient (mean over keys, one key at a time), the relative error of one key's gradient, and the relative
error of the mean over the keys (the bias, if the noise averages out). The patch units' gradients (Wp, bp, Qp) are
the new part; the global unit's (W, b, Q) are the known reference.

Usage (GPU job): python check_sampledpatch.py --run RUN --slices 60,85 --S 16,64,256 --keys 8 --batch 128
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
from physical_diffusion_fns.helper_fns import sample_forward_process  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--slices", default="60,85")
    ap.add_argument("--S", default="16,64,256")
    ap.add_argument("--keys", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    run_dir = os.path.join(lm.OUT_ROOT, args.run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    prob = lm.make_problem(cfg["problem"], radius=cfg.get("radius"))
    n = prob["grid"] ** 2
    vis = lm.DenseVisible(prob["conn"], n)
    K = int(cfg["K"])
    taus = lm.forward_time_pts()
    data = jnp.asarray(prob["data"])
    for k in [int(v) for v in args.slices.split(",")]:
        s = np.load(os.path.join(run_dir, "slices", f"slice_{k:03d}.npz"))
        params = {g: jnp.asarray(s[g]) for g in ("theta", "W", "b", "Q", "wl", "bl", "Wp", "bp", "Qp") if g in s.files}
        if "Wp" not in params or not np.any(s["Wp"]):
            print(f"slice {k}: no active patch units, skipped")
            continue
        t = float(taus[k])
        X = sample_forward_process(t, args.batch, lm.TEMP, lm.SIGMA_FORWARD, data, jr.PRNGKey(4242 + k), beta=1)
        exact = jax.jit(jax.value_and_grad(lm.make_loss(vis, lm.TEMP, K, "potts", True, samples=0)))
        l0, g0 = exact(params, X)
        print(f"== slice {k} (t = {t:.4f}), batch {args.batch}: exact loss {float(l0):.6g}; |grad| "
              + ", ".join(f"{g} {float(jnp.linalg.norm(v)):.3g}" for g, v in g0.items()), flush=True)
        for S in [int(v) for v in args.S.split(",")]:
            fn = jax.jit(jax.value_and_grad(lm.make_loss(vis, lm.TEMP, K, "potts", True, samples=S, jackknife=True)))
            ls, gs = [], {g: [] for g in g0}
            for r in range(args.keys):
                l, g = fn(params, X, jr.PRNGKey(1000 * S + r))
                ls.append(float(l))
                for q in g0:
                    gs[q].append(np.asarray(g[q]))
            ls = np.array(ls)
            row = [f"S {S:4d}: loss {ls.mean():.6g} +- {ls.std() / np.sqrt(len(ls)):.2g} (exact {float(l0):.6g})"]
            for q in ("W", "b", "Q", "Wp", "bp", "Qp", "theta"):
                if q not in g0:
                    continue
                e = np.asarray(g0[q]).ravel()
                G = np.stack([v.ravel() for v in gs[q]])
                ne = np.linalg.norm(e) + 1e-300
                cos = np.mean([float(v @ e / (np.linalg.norm(v) * ne + 1e-300)) for v in G])
                err1 = np.median([np.linalg.norm(v - e) / ne for v in G])
                errm = np.linalg.norm(G.mean(0) - e) / ne
                row.append(f"{q}: cos {cos:.4f}, err(1 key) {err1:.3f}, err(mean of {len(G)}) {errm:.3f}")
            print("   " + "\n     ".join(row), flush=True)
    print("(err = |g_sampled - g_exact| / |g_exact|; with unbiased noise the mean of n keys has about err(1)/sqrt(n))")


if __name__ == "__main__":
    main()
