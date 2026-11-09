#!/usr/bin/env python3
"""What do the K Ising hidden units of a trained run represent, slice by slice? (inspect_states.py does this for Potts
units and skips everything else.)

The unit tree is rebuilt from the run's config (latent_model.split_reference with the run's K, --seed and
--proto-init; an old K = 1 run is the Fisher unit, i.e. the perclass tree with K = 1) and compared with the run's
split_tree.npz, if there is one. For each run and slice: data x ~ p_tau (fixed key), the fields
a_j = b_j + W_j.x + Q_j.x^2 and m_j = tanh(a_j / T), and
  * the summary numbers of latent_model.ising_unit_stats (as in the training log);
  * per unit: its node (depth, child sizes and purity), the class mix of the samples on either side (sign a_j), its
    minority share and its straddle (the share of the samples outside the unit's subtree with |a_j| < 2T);
  * images, for the first --max-units units: the tree's two child means (clean pixels), W_j and, with --quad, Q_j;
  * the |cos| matrix of the rows of W.
With --samples STEM (the file <run>/samples/STEM.npy of sample_latent.py, e.g. sde_base_n500), under the t = 0
parameters (slice 99):
  * codes sign(a) of the generated samples, of --probes real probes (training images held out of the code book) and
    of the remaining training images (the code book);
  * for every sample and probe the Hamming distance to the nearest code-book code, counted only over the units with
    |a| > 2T in both; the novel-code rate (distance >= 1) of the samples against that of the real probes;
  * the per-class activity of each unit (the share of a class's training images with a_j > 0) and the samples' one;
  * a gallery of the most novel samples.

Usage (CPU Slurm job): python inspect_units.py RUN[:SLICE,SLICE,...] ... [--samples sde_base_n500] [--max-units 32]
Writes out/latent_physical_SGM/units/<job>/: <run>_s<slice>.png (units), <run>_s<slice>_cos.png, <run>_novel.png,
<run>_activity.txt, <run>_metrics.json and summary.txt.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

import jax.random as jr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "units", os.environ.get("SLURM_JOB_ID", "manual"))
N_DATA = 4096


def nearest_code_distance(Aq, Ab, T, chunk=256):
    """Hamming distance of each query code sign(Aq) to the nearest code sign(Ab), counted only over the units with
    |a| > 2T in both. Returns (distances (q,) int, decided units per query (q,) int)."""
    sq, dq = np.sign(Aq), (np.abs(Aq) > 2.0 * T).astype(np.float64)
    sb, db = np.sign(Ab), (np.abs(Ab) > 2.0 * T).astype(np.float64)
    dsb = db * sb
    out = np.empty(Aq.shape[0])
    for c in range(0, Aq.shape[0], chunk):
        q = slice(c, c + chunk)
        both = dq[q] @ db.T                                  # units decided in both codes
        agree = (dq[q] * sq[q]) @ dsb.T                      # among those: +1 same sign, -1 different
        out[q] = np.min(0.5 * (both - agree), axis=1)
    return np.rint(out).astype(int), dq.sum(axis=1).astype(int)


def unit_figure(path, title, tree, W, Q, mean, std, grid, units):
    rows = 4 if Q is not None else 3
    fig, axes = plt.subplots(rows, len(units), figsize=(1.3 * len(units), 1.45 * rows), squeeze=False)
    for c, j in enumerate(units):
        for r, img in enumerate((tree["mu_left"][j], tree["mu_right"][j])):
            axes[r, c].imshow((img * std + mean).reshape(grid, grid), cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"{j}: d{int(tree['depth'][j])} {int(tree['n_left'][j])}/{int(tree['n_right'][j])}",
                             fontsize=6)
        v = np.abs(W[j]).max() + 1e-12
        axes[2, c].imshow(W[j].reshape(grid, grid), cmap="RdBu_r", vmin=-v, vmax=v)
        if Q is not None:
            vq = np.abs(Q[j]).max() + 1e-12
            axes[3, c].imshow(Q[j].reshape(grid, grid), cmap="RdBu_r", vmin=-vq, vmax=vq)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for r, name in enumerate(["left child\n(mean)", "right child\n(mean)", "W_j", "Q_j"][:rows]):
        axes[r, 0].set_ylabel(name, fontsize=7)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cos_figure(path, title, W):
    Wn = W / np.maximum(np.linalg.norm(W, axis=1), 1e-300)[:, None]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(np.abs(Wn @ Wn.T), cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("unit")
    ax.set_ylabel("unit")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="RUN or RUN:SLICE,SLICE (default slices 0,50,75,99)")
    ap.add_argument("--max-units", type=int, default=32, help="units shown in the images and the per-unit table")
    ap.add_argument("--samples", default="", help="STEM of <run>/samples/STEM.npy (sample_latent.py) for the codes")
    ap.add_argument("--probes", type=int, default=1000, help="real probes held out of the code book")
    ap.add_argument("--top-novel", type=int, default=32, help="samples in the gallery of the most novel codes")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    T = lm.TEMP
    taus = lm.forward_time_pts()
    lines = [f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; inspect_units; job {os.environ.get('SLURM_JOB_ID', 'manual')}"]
    probs = {}
    for spec in args.runs:
        run, _, sl = spec.partition(":")
        slices = [int(s) for s in sl.split(",")] if sl else [0, 50, 75, 99]
        run_dir = os.path.join(lm.OUT_ROOT, run)
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        K = int(cfg.get("K", 0))
        if cfg.get("hidden", "ising") != "ising" or K == 0:
            lines.append(f"{run}: not an Ising run with K >= 1; skipped")
            continue
        pname = cfg.get("problem", "mnist28")
        if pname not in probs:
            probs[pname] = lm.make_problem(pname, cfg.get("radius"), cfg.get("neighbourhood", "chebyshev"))
        prob = probs[pname]
        data, labels, grid = np.asarray(prob["data"]), prob["labels"], prob["grid"]
        n_img, n = data.shape
        n_cls = int(labels.max()) + 1
        seed = int(cfg.get("seed", lm.KEY_SEED))
        proto = cfg.get("proto_init", "perclass")
        pooled = proto in ("pooled", "random")
        W_ref, _, tree = lm.split_reference(data, labels, taus, T, K, seed=100 * (seed - lm.KEY_SEED), pooled=pooled)
        lines.append("")
        lines.append(f"{run} ({pname}, {K} Ising units, init {cfg.get('init')}, proto_init {proto}, seed {seed}, quad "
                     f"{cfg.get('quad', False)}, local {cfg.get('local', False)}, fixed {cfg.get('hidden_fixed', False)})")
        tpath = os.path.join(run_dir, "split_tree.npz")
        if os.path.exists(tpath):
            zt = np.load(tpath)
            same = set(zt.files) == set(tree) and all(np.array_equal(zt[k_], tree[k_], equal_nan=k_ == "unit_purity")
                                                      for k_ in tree)
            lines.append(f"  rebuilt tree {'equals' if same else 'DIFFERS FROM'} the run's split_tree.npz")
        sizes = np.bincount(tree["leaf_of"], minlength=K + 1)
        lines.append(f"  tree: depth <= {int(tree['depth'].max())}, leaf sizes {int(sizes.min())}-{int(sizes.max())}, "
                     f"root purity {tree['unit_purity'][0]:.3f}, root classes left/right "
                     f"{tree['unit_class'][0, 0]}/{tree['unit_class'][0, 1]}; split kinds class/2-means/median "
                     f"{'/'.join(str(int(np.sum(tree['kind'] == q))) for q in range(3))}")
        metrics = {"run": run, "slices": {}}
        units = list(range(min(K, args.max_units)))
        for k in slices:
            path = os.path.join(run_dir, "slices", f"slice_{k:03d}.npz")
            if not os.path.exists(path):
                lines.append(f"  slice {k}: missing; skipped")
                continue
            z = np.load(path)
            W, b = z["W"], z["b"]
            Q = z["Q"] if "Q" in z.files else None
            prev = os.path.join(run_dir, "slices", f"slice_{k - 1:03d}.npz")
            W_prev = np.load(prev)["W"] if k > 0 and os.path.exists(prev) else None
            t = float(taus[k])
            kidx, knoise = jr.split(jr.fold_in(jr.PRNGKey(123), k))
            idx = np.asarray(jr.choice(kidx, n_img, shape=(N_DATA,), replace=False))
            e = np.exp(-t)
            X = e * data[idx] + np.sqrt(T * (1 - e * e)) * np.asarray(jr.normal(knoise, (N_DATA, n)))
            lab, leaf = labels[idx], tree["leaf_of"][idx]
            rec, arr = lm.ising_unit_stats(W, b, Q, X, T, W_ref=W_ref[k], W_prev=W_prev, tree=tree, leaf=leaf,
                                           labels=lab)
            metrics["slices"][str(k)] = rec
            st = rec.get("straddle_off")
            lines.append(f"  slice {k} (t = {t:.3f}): |W| {rec['W_norm']:.4g} (ref {rec['Wref_norm']:.4g}), overlap "
                         f"{rec['overlap_ref']:.3f}, norm ratio min/median {rec['norm_ratio_min']:.3g}/"
                         f"{rec['norm_ratio_median']:.3g}; dead/weak/decided {rec['n_dead']}/{rec['n_weak']}/"
                         f"{rec['n_decided']}, gated {rec['n_gated']}; max |cos| {rec['max_pair_cos']:.3f}"
                         + (f", min cos to slice {k - 1} {rec['min_cos_prev']:.3f}" if 'min_cos_prev' in rec else "")
                         + f"; unit-0 acc {rec['class_acc']:.3f}, leaf acc {rec['leaf_acc']:.3f}; <|m|> "
                         f"{rec['mean_abs_m']:.3f}, <v> {rec['mean_v']:.3f}, softening/T {rec['softening_over_T']:.3g}, "
                         f"n_eff(m) {rec['n_eff_m']:.2f}, straddle " + ("n/a" if st is None else f"{st:.3f}")
                         + (f", own balance error {rec['own_balance_err']:.3f}" if rec.get('own_balance_err') is not None
                            else ""))
            A = lm.ising_fields(W, b, Q, X)
            for j in units:
                neg, pos = A[:, j] <= 0, A[:, j] > 0
                mix_n = "/".join(str(v) for v in np.bincount(lab[neg], minlength=n_cls))
                mix_p = "/".join(str(v) for v in np.bincount(lab[pos], minlength=n_cls))
                sj = arr["unit_straddle"][j]
                lines.append(f"    unit {j:3d}: depth {int(tree['depth'][j])}, children {int(tree['n_left'][j])}/"
                             f"{int(tree['n_right'][j])} (purity {tree['unit_purity'][j]:.2f}); classes a<0 {mix_n}, "
                             f"a>0 {mix_p}; minority {arr['unit_minority'][j]:.3f}, <|m|> {arr['unit_abs_m'][j]:.3f}, "
                             f"straddle " + ("n/a" if not np.isfinite(sj) else f"{sj:.3f}")
                             + f", |W_j| {arr['unit_norm'][j]:.3g}, cos ref {arr['unit_cos_ref'][j]:.3f}")
            unit_figure(os.path.join(OUT, f"{run}_s{k:03d}.png"),
                        f"{run}, slice {k} (t = {t:.2f}): child means, W_j" + (", Q_j" if Q is not None else "")
                        + "; titles: unit, depth, n_left/n_right", tree, W, Q, prob["mean"], prob["std"], grid, units)
            cos_figure(os.path.join(OUT, f"{run}_s{k:03d}_cos.png"), f"{run}, slice {k}: |cos(W_i, W_j)|", W)

        if args.samples:                            # codes under the t = 0 parameters
            k99 = lm.N_TIME_STEPS - 1
            path = os.path.join(run_dir, "slices", f"slice_{k99:03d}.npz")
            spath = os.path.join(run_dir, "samples", f"{args.samples}.npy")
            if not (os.path.exists(path) and os.path.exists(spath)):
                lines.append(f"  codes: needs slice {k99} and {spath}; skipped")
            else:
                z = np.load(path)
                W, b = z["W"], z["b"]
                Q = z["Q"] if "Q" in z.files else None
                Xs = np.load(spath)
                fin = np.all(np.isfinite(Xs), axis=1)
                Xs = Xs[fin]
                rng = np.random.default_rng(0)
                probe = np.sort(rng.choice(n_img, size=min(args.probes, n_img // 2), replace=False))
                book = np.setdiff1d(np.arange(n_img), probe)
                A_all = lm.ising_fields(W, b, Q, data)
                A_s = lm.ising_fields(W, b, Q, Xs)
                d_s, dec_s = nearest_code_distance(A_s, A_all[book], T)
                d_p, dec_p = nearest_code_distance(A_all[probe], A_all[book], T)

                def hist(d):
                    return "/".join(str(int(v)) for v in (np.sum(d == 0), np.sum(d == 1), np.sum(d == 2), np.sum(d >= 3)))
                rate_s, rate_p = float(np.mean(d_s >= 1)) if d_s.size else float("nan"), float(np.mean(d_p >= 1))
                lines.append(f"  codes at t = 0 ({args.samples}: {int(fin.sum())} finite samples of {fin.size}; "
                             f"{probe.size} real probes; code book of {book.size} training images): novel-code rate "
                             f"samples {rate_s:.3f} vs real probes {rate_p:.3f}; distance 0/1/2/>=3: samples {hist(d_s)}, "
                             f"probes {hist(d_p)}; decided units per code: samples {dec_s.mean() if dec_s.size else 0:.1f}, "
                             f"probes {dec_p.mean():.1f} of {K}")
                metrics["codes"] = {"samples": args.samples, "n_samples": int(fin.sum()), "n_probes": int(probe.size),
                                    "novel_rate_samples": rate_s, "novel_rate_probes": rate_p,
                                    "mean_distance_samples": float(d_s.mean()) if d_s.size else None,
                                    "mean_distance_probes": float(d_p.mean()),
                                    "decided_samples": float(dec_s.mean()) if dec_s.size else None,
                                    "decided_probes": float(dec_p.mean())}
                act = np.stack([np.mean(A_all[labels == c] > 0, axis=0) for c in range(n_cls)])     # (C, K)
                act_s = np.mean(A_s > 0, axis=0) if A_s.shape[0] else np.full(K, np.nan)
                with open(os.path.join(OUT, f"{run}_activity.txt"), "w") as f:
                    f.write(f"# {run}: share of each class's training images (and of the samples) with a_j > 0 at t = 0\n")
                    f.write("unit " + " ".join(f"class{c:<3d}" for c in range(n_cls)) + " samples\n")
                    for j in range(K):
                        f.write(f"{j:4d} " + " ".join(f"{act[c, j]:8.3f}" for c in range(n_cls)) + f" {act_s[j]:8.3f}\n")
                if d_s.size:
                    order = np.lexsort((-dec_s, -d_s))[:args.top_novel]          # most novel first
                    cols = 8
                    rows = int(np.ceil(order.size / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(1.3 * cols, 1.45 * rows), squeeze=False)
                    for ax in axes.ravel():
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.axis("off")
                    for i, s in enumerate(order):
                        ax = axes[i // cols, i % cols]
                        ax.imshow(np.clip(Xs[s] * prob["std"] + prob["mean"], 0, 1).reshape(grid, grid), cmap="gray",
                                  vmin=0, vmax=1)
                        ax.set_title(f"d={d_s[s]} ({dec_s[s]} dec)", fontsize=6)
                    fig.suptitle(f"{run}: most novel samples ({args.samples}); d = Hamming distance to the nearest "
                                 f"training code over units decided in both", fontsize=7)
                    fig.tight_layout()
                    fig.savefig(os.path.join(OUT, f"{run}_novel.png"), dpi=110)
                    plt.close(fig)
        with open(os.path.join(OUT, f"{run}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
