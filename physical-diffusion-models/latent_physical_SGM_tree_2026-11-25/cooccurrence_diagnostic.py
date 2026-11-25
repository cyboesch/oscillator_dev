#!/usr/bin/env python3
"""Do our generated images coordinate the hidden states as real images do? (2026-11-25; no training.)

The question behind Krotov's hierarchy: in today's model the global unit and the patch units are coupled to the pixels
only. If the generated images already reproduce the real images' co-occurrence of (global state, patch state) and of
the states of distant patches, a direct coupling J has little to add; if they fall short, that is the measurable
target, and it shows whether the missing link is global <-> patch or patch <-> patch.

Method. For a trained run and some of its slices k (time t_k): real train images, real held-out (test) images and the
run's generated images (final samples) all get the forward noise of t_k (x_t = e^-t x + sqrt(T (1 - e^-2t)) eps; real
test and generated images first get the training data's 0.01 pixel noise), and the slice's own hidden units assign
states: pi(g | x) of the global unit and q(s | x) of every patch unit (exact posteriors; given x they are independent,
so the joint posterior is their product).
 - Reference tables from N_FIT real train images: R_p[g, s] (global x patch) and R_pq[s, s'] for pairs of
   NON-overlapping patches: 'adjacent' (touching: 4 patch steps apart, horizontal and vertical), 'far' (8 steps) and
   'mirror' (left-right mirror images of each other, >= 4 steps apart).
 - Cross gain of an image set: mean over its images of log [R(a, b) / (R(a) R(b))] under the set's own posteriors,
   in nats per patch (or pair). For held-out real images this is the mutual information of the real data (how much
   the global state tells about a patch state); for generated images it measures how far they follow the real
   images' assembling rules. Its bias is small (the tables come from N_FIT images), so real and generated compare
   directly. The real value is given as mean +- sd over disjoint blocks of the generated sets' size.
 - Plug-in mutual information of each set by itself at the same n (biased upwards, equally for real blocks and
   generated sets): catches coordination that is strong but different from the real one.
 - Usage: KL(set || real train) of the global-state and patch-state frequencies.

Usage (GPU job, from this folder):
  sbatch jobs/run_gpu.sh cooccurrence_diagnostic.py --run RUN --gen LABEL=samples/FILE.npy [...] [--slices "99 94 79"]
Writes out/cooccurrence/<job>/report.txt and metrics.json.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tree_model as tm  # noqa: E402
lm = tm.lm

T = lm.TEMP
BATCH = 2000
OUT = os.path.join(HERE, "out", "cooccurrence", os.environ.get("SLURM_JOB_ID", "manual"))
LINES = []


def report(s=""):
    print(s, flush=True)
    LINES.append(s)


def pair_lists(n_p, min_off=4):
    """{class: (a (n,), b (n,))} patch indices of non-overlapping pairs on an n_p x n_p patch grid."""
    idx = np.arange(n_p * n_p).reshape(n_p, n_p)
    out = {}
    for name, off in (("adjacent", min_off), ("far", 2 * min_off)):
        if off < n_p:
            a = np.concatenate([idx[:, :-off].ravel(), idx[:-off, :].ravel()])
            b = np.concatenate([idx[:, off:].ravel(), idx[off:, :].ravel()])
            out[name] = (a, b)
    cols = [j for j in range(n_p // 2) if n_p - 1 - 2 * j >= min_off]
    if cols:
        out["mirror"] = (np.concatenate([idx[:, j] for j in cols]), np.concatenate([idx[:, n_p - 1 - j] for j in cols]))
    return out


def make_tables(params, pairs):
    """jit: X (B, N) -> soft count tables {g: (K,), gs<i>: (P, K, S), s<i>: (P, S), pair<i>_<class>: (n, S, S)}."""
    @jax.jit
    def f(X):
        pi, qs = tm.hidden_posteriors(params, X, T)
        out = {"g": jnp.sum(pi, axis=0)}
        for i, q in enumerate(qs):
            out[f"gs{i}"] = jnp.einsum("bk,pbs->pks", pi, q)
            out[f"s{i}"] = jnp.sum(q, axis=1)
            for name, (a, b) in pairs[i].items():
                out[f"pair{i}_{name}"] = jnp.einsum("nbs,nbt->nst", q[a], q[b])
        return out
    return f


def accumulate(f, X, key, t):
    """Tables of the images X (normalised units) noised to time t, summed over batches; the counts' n."""
    e = np.exp(-t)
    sd = np.sqrt(T * (1.0 - e * e))
    tot = None
    for i in range(0, X.shape[0], BATCH):
        xb = jnp.asarray(X[i:i + BATCH], jnp.float64)
        xb = e * xb + sd * jr.normal(jr.fold_in(key, i), xb.shape)
        out = {k: np.asarray(v, np.float64) for k, v in f(xb).items()}
        tot = out if tot is None else {k: tot[k] + out[k] for k in out}
    return tot, X.shape[0]


def mi_cross(E, n, R, nR, alpha=1.0):
    """Per unit (first axis): sum_ab (E / n) log [R(a, b) / (R(a) R(b))], R smoothed by alpha pseudo-counts in all."""
    Pj = (R + alpha / (R.shape[-1] * R.shape[-2])) / (nR + alpha)
    L = np.log(Pj) - np.log(Pj.sum(-1, keepdims=True)) - np.log(Pj.sum(-2, keepdims=True))
    return np.sum(E / n * L, axis=(-1, -2))


def mi_plugin(E, n):
    Pj = E / n
    Pa, Pb = Pj.sum(-1, keepdims=True), Pj.sum(-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(Pj > 0, Pj * (np.log(Pj) - np.log(Pa) - np.log(Pb)), 0.0)
    return np.sum(term, axis=(-1, -2))


def kl(p, q, eps=1e-9):
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    return np.sum(np.where(p > 0, p * (np.log(p + eps) - np.log(q + eps)), 0.0), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gen", action="append", required=True, help="LABEL=PATH below the run folder (repeat)")
    ap.add_argument("--slices", default="99 97 94 89 79 59 39")
    ap.add_argument("--n-fit", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    run_dir = os.path.join(lm.OUT_ROOT, args.run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    problem = cfg["problem"]
    report(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}; cooccurrence_diagnostic; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
           f"devices {jax.devices()}")
    report(f"run {args.run} ({problem}); K {cfg['K']}; patch sets {cfg.get('patch')} (from slice {cfg.get('patch_on_slice')}), "
           f"{cfg.get('patch2')} (from slice {cfg.get('patch2_on_slice')})")
    prob = lm.make_problem(problem)
    mean, std, grid = prob["mean"], prob["std"], prob["grid"]
    rng = np.random.default_rng(args.seed)
    train = np.asarray(prob["data"])
    fit = train[rng.choice(train.shape[0], min(args.n_fit, train.shape[0]), replace=False)]
    ds = lm.problem_dataset(problem)
    raw_te = lm.load_cifar_raw("test")[0] if ds == "cifar" else lm.load_celeba_raw("test", lm.celeba_grid(problem))[0]
    test = (raw_te + lm.STD_ADDED_NOISE * rng.standard_normal(raw_te.shape) - mean) / std
    gens = {}
    for spec in args.gen:
        label, path = spec.split("=", 1)
        x = np.load(os.path.join(run_dir, path)).astype(np.float64)
        ok = np.isfinite(x).all(1) & (np.abs(x).max(1) <= 15.0)
        gens[label] = x[ok] + lm.STD_ADDED_NOISE / std * rng.standard_normal(x[ok].shape)
        report(f"generated set {label}: {path}, {int(ok.sum())} of {x.shape[0]} samples")
    n_gen = min(v.shape[0] for v in gens.values())
    test = test[rng.permutation(test.shape[0])]
    blocks = [test[i:i + n_gen] for i in range(0, test.shape[0] - n_gen + 1, n_gen)]
    report(f"real: {fit.shape[0]} train images for the reference tables; {test.shape[0]} test images in {len(blocks)} "
           f"blocks of {n_gen}; pixel noise {lm.STD_ADDED_NOISE} added to test and generated images")
    slices = [int(s) for s in args.slices.split()]
    with np.load(os.path.join(run_dir, "hidden_history.npz")) as z:      # each member is read once (the file is large)
        taus = np.asarray(z["t"])
        hid = {name: np.asarray(z[name])[slices] for name in z.files if name != "t"}
    hid_files = set(hid)
    on = [cfg.get("patch_on_slice") or 0, cfg.get("patch2_on_slice") or 0, cfg.get("patch3_on_slice") or 0]
    key = jr.PRNGKey(args.seed)
    metrics = {}
    for ik, k in enumerate(slices):
        t = float(taus[k])
        params = {"W": jnp.asarray(hid["W"][ik]), "b": jnp.asarray(hid["b"][ik]), "Q": jnp.asarray(hid["Q"][ik])}
        pairs, names = [], []
        for i, (kw, kb, kq) in enumerate(lm.PATCH_SETS):
            if kw in hid_files and k >= on[i]:
                params[kw], params[kb] = jnp.asarray(hid[kw][ik], jnp.float64), jnp.asarray(hid[kb][ik], jnp.float64)
                if kq in hid_files:
                    params[kq] = jnp.asarray(hid[kq][ik], jnp.float64)
                geom = lm.patch_geometry(prob["n"], params[kw].shape[0], params[kw].shape[2])
                pairs.append(pair_lists(geom.n_p, geom.s // geom.r))
                names.append(f"{geom.s}x{geom.s} patches ({geom.P} units, {params[kw].shape[1]} states)")
        f = make_tables(params, pairs)
        kk = jr.fold_in(key, k)
        R, nR = accumulate(f, fit, jr.fold_in(kk, 0), t)
        B = [accumulate(f, b, jr.fold_in(kk, 10 + j), t)[0] for j, b in enumerate(blocks)]
        G = {lab: accumulate(f, x, jr.fold_in(kk, 1000 + j), t)[0] for j, (lab, x) in enumerate(gens.items())}
        nG = {lab: x.shape[0] for lab, x in gens.items()}
        pg = R["g"] / nR
        report()
        report(f"--- slice {k}, t = {t:.3f}: noise sd {np.sqrt(T * (1 - np.exp(-2 * t))):.3f}, signal x {np.exp(-t):.3f}; "
               f"effective global states (real train) {np.exp(-np.sum(pg * np.log(pg + 1e-300))):.2f} of {pg.size}")
        m = metrics.setdefault(str(k), {"t": t})
        rows = [("global-state usage, KL to real train", "g", None)]
        for i, nm in enumerate(names):
            rows.append((f"set {i + 1} {nm}: patch-state usage, KL to real train (mean per unit)", f"s{i}", None))
        for label, tk, _ in rows:
            vb = [float(np.mean(kl(b[tk], R[tk]))) for b in B]
            line = f"  {label}: real blocks {np.mean(vb):.4f} +- {np.std(vb):.4f}"
            for lab in G:
                v = float(np.mean(kl(G[lab][tk], R[tk])))
                line += f"; {lab} {v:.4f}"
                m.setdefault("usage_kl", {}).setdefault(tk, {})[lab] = v
            m["usage_kl"][tk]["real_blocks"] = [float(np.mean(vb)), float(np.std(vb))]
            report(line)
        report(f"  {'statistic (nats per unit or pair)':58s} {'pairs':>5s} {'real: cross gain':>18s} {'real: plug-in':>16s}  "
               + "  ".join(f"{lab + ' cross / plug-in / % of real':>44s}" for lab in G))
        stats = []
        for i, nm in enumerate(names):
            stats.append((f"set {i + 1} global <-> patch ({nm.split(' (')[0]})", f"gs{i}"))
            for cls in pairs[i]:
                stats.append((f"set {i + 1} patch <-> patch, {cls}", f"pair{i}_{cls}"))
        for label, tk in stats:
            cb = np.array([np.mean(mi_cross(b[tk], n_gen, R[tk], nR)) for b in B])
            pb = np.array([np.mean(mi_plugin(b[tk], n_gen)) for b in B])
            line = f"  {label:58s} {R[tk].shape[0]:5d} {cb.mean():9.4f} +- {cb.std():6.4f} {pb.mean():8.4f} +- {pb.std():5.4f}  "
            mm = m.setdefault(tk, {"real_cross": [float(cb.mean()), float(cb.std())],
                                   "real_plugin": [float(pb.mean()), float(pb.std())], "units": int(R[tk].shape[0])})
            for lab in G:
                c = float(np.mean(mi_cross(G[lab][tk], nG[lab], R[tk], nR)))
                p = float(np.mean(mi_plugin(G[lab][tk], nG[lab])))
                pct = 100.0 * c / cb.mean() if cb.mean() > 1e-6 else float("nan")
                line += f"  {c:14.4f} / {p:8.4f} / {pct:6.1f}%" + " " * 9
                mm[lab] = {"cross": c, "plugin": p, "pct_of_real_cross": pct}
            report(line)
    report()
    report("Reading: 'cross gain' of the real blocks = the real data's mutual information between the two states (how much "
           "one tells about the other). A generated set at 100% follows the real images' rule as closely as held-out real "
           "images do; clearly below 100% = the model's samples coordinate these states less than real images (or "
           "differently: then its plug-in value stays high while the cross gain drops).")
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump({"run": args.run, "gen": args.gen, "n_fit": int(fit.shape[0]), "n_gen": int(n_gen), "slices": metrics}, fh, indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
