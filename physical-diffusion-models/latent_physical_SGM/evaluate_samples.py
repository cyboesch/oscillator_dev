#!/usr/bin/env python3
"""Blend check (same reference and thresholds as blend_check.py) plus per-class pixel statistics, for any sample sets.

Usage:  python evaluate_samples.py LABEL=PATH [LABEL=PATH ...] [--pair LABEL_A,LABEL_B ...]
  PATH: .npy of final states (n, 784), normalised units; relative paths are under out/latent_physical_SGM.
        PATH_A+PATH_B concatenates files (e.g. samples of the 0-only and 1-only models). Paired tests treat rows as
        independent trajectories, so pair '+'-unions only if their parts are disjoint trajectory sets (base+ext),
        never unions that reuse trajectories (cls0+cls1); pairs need the same part lengths.
  --pair: paired comparison of two sets that used the same initial states and Brownian keys: McNemar for blend, far
          and diverged; a paired bootstrap over trajectories for the class-standardised clean yield and far rate.
Class-resolved rates: P(clean | class), P(far | class), P(blend | class) among non-diverged samples with that Fisher
class, and their 50/50 averages ('std'), which do not depend on the model's class mix.
Categories (blend_check.py): diverged, blend (G above the 5%-of-real-images threshold), clean / malformed 0 and 1
(nearest training image within / beyond 4.84; for --problem mnist14 / gauss2_8 the probes' p90 distance), and the
Fisher class score.
Per-class pixel statistics (pixel space, non-diverged samples, class from the Fisher score):
  RMSE of the class mean and std images against the data's, ink fraction (pixels > 0.5), and the 'ghost'
  intensity: mean pixel value of class-0 samples in the 1-stroke region (data mu_1 - mu_0 > 0.25), and of class-1
  samples in the ring region (mu_0 - mu_1 > 0.25).
Writes out/latent_physical_SGM/eval/<job id>/: report.txt, metrics.json, examples.png, class_means.png. CPU only.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import blend_check as bc  # noqa: E402  (scores, frac and constants; its main() is not run)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "eval", os.environ.get("SLURM_JOB_ID", "manual"))
GHOST_MARGIN = 0.25


def report(line=""):
    print(line, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(line + "\n")


def reference(problem="mnist28"):
    prob = lm.make_problem(problem)
    data, labels, mean_px, std_px = np.asarray(prob["data"]), prob["labels"], prob["mean"], prob["std"]
    if np.unique(labels).size != 2:
        raise SystemExit(f"{problem}: evaluate_samples.py handles two classes only")
    px = data * std_px + mean_px
    tr_n2 = np.sum(px**2, axis=1)
    rng = np.random.default_rng(bc.PROBE_SEED)
    probe = np.sort(rng.choice(px.shape[0], size=bc.N_PROBE, replace=False))
    is_probe = np.zeros(px.shape[0], bool)
    is_probe[probe] = True
    X0, X1 = px[~is_probe & (labels == 0)], px[~is_probe & (labels == 1)]
    mu0, mu1 = X0.mean(0), X1.mean(0)
    Sw = ((X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)) / (X0.shape[0] + X1.shape[0] - 2)
    lam = 1e-2 * np.trace(Sw) / Sw.shape[0]
    w = np.linalg.solve(Sw + lam * np.eye(Sw.shape[0]), mu1 - mu0)
    lda = lambda P: (P @ w - mu0 @ w) / ((mu1 - mu0) @ w)
    nn_p, G_p = bc.scores(px[probe], px, tr_n2, labels, self_idx=probe)
    s_p = lda(px[probe])
    lab_p = labels[probe]
    # 'far' threshold: 4.84 for the paper's problem (as in the ode_vs_sde study), else the probes' p90
    far_thr = bc.FAR_THRESHOLD if problem == "mnist28" else float(np.percentile(nn_p, 90))
    copy_thr = float(np.percentile(nn_p, 5))      # closer to a training image than 95% of held-out images are
    ref = dict(px=px, tr_n2=tr_n2, labels=labels, probe=probe, lda=lda, mean_px=mean_px, std_px=std_px, far_thr=far_thr,
               copy_thr=copy_thr, digits=prob.get("digits"),
               grid=prob["grid"],
               G_thr=float(np.quantile(G_p, 1.0 - bc.FP_RATE)),
               q0=float(np.quantile(s_p[lab_p == 0], 0.99)), q1=float(np.quantile(s_p[lab_p == 1], 0.01)),
               nn_p=nn_p, G_p=G_p, s_p=s_p)
    if problem == "gauss2_8":
        # exact data distribution: two Gaussians (pixel noise 0.15, plus the paper's 0.01) around the patterns
        ring, bar = lm.toy_patterns(prob["grid"])
        sig2 = 0.15**2 + lm.STD_ADDED_NOISE**2
        ref["toy"] = dict(ring=ring, bar=bar, sig2=sig2)
        ll_p = toy_loglik(ref["toy"], px[probe])[0]
        ref["toy"]["ll_thr"] = float(np.quantile(ll_p, 0.005))
    m0, m1 = px[labels == 0].mean(0), px[labels == 1].mean(0)        # unclipped: defines the ring / stroke masks
    pc = np.clip(px, 0.0, 1.0)                                        # samples are clipped too before comparing
    ref.update(mu=(pc[labels == 0].mean(0), pc[labels == 1].mean(0)), sd=(pc[labels == 0].std(0), pc[labels == 1].std(0)),
               ring=(m0 - m1) > GHOST_MARGIN, stroke=(m1 - m0) > GHOST_MARGIN)
    return ref


def texture(P, grid):
    """Per-image texture statistics in pixel space (clipped to [0, 1]): roughness = mean squared 5-point Laplacian
    (interior pixels), grey = share of pixels strictly between 0.2 and 0.8."""
    Q = np.clip(P, 0.0, 1.0).reshape(-1, grid, grid)
    lap = 4 * Q[:, 1:-1, 1:-1] - Q[:, :-2, 1:-1] - Q[:, 2:, 1:-1] - Q[:, 1:-1, :-2] - Q[:, 1:-1, 2:]
    return np.mean(lap**2, axis=(1, 2)), np.mean((Q > 0.2) & (Q < 0.8), axis=(1, 2))


def toy_loglik(toy, P):
    """Exact log-likelihood (pixel space) under the gauss2_8 data distribution, and the posterior of class 1."""
    d0 = np.sum((P - toy["ring"]) ** 2, axis=1) / (2 * toy["sig2"])
    d1 = np.sum((P - toy["bar"]) ** 2, axis=1) / (2 * toy["sig2"])
    n = P.shape[1]
    ll = np.logaddexp(-d0, -d1) - np.log(2.0) - 0.5 * n * np.log(2 * np.pi * toy["sig2"])
    return ll, 1.0 / (1.0 + np.exp(np.clip(d1 - d0, -700, 700)))


def classify(ref, x=None):
    if x is None:
        P, nn, G, s = ref["px"][ref["probe"]], ref["nn_p"], ref["G_p"], ref["s_p"]
        div = np.zeros(P.shape[0], bool)
    else:
        x = np.asarray(x, dtype=np.float64)
        div = ~(np.max(np.abs(x), axis=1) <= bc.MAX_ABS)
        P = np.where(np.isfinite(x), x, 0.0) * ref["std_px"] + ref["mean_px"]
        nn, G = bc.scores(P, ref["px"], ref["tr_n2"], ref["labels"])
        s = ref["lda"](P)
    ok = ~div
    blend = ok & (G > ref["G_thr"])
    far = div | (nn > ref["far_thr"])
    cls1 = s >= 0.5
    rest = ok & ~blend
    r = {"n": int(P.shape[0]), "diverged": bc.frac(div), "blend": bc.frac(blend),
         "gap": bc.frac(ok & (s > ref["q0"]) & (s < ref["q1"])),
         "clean0": bc.frac(rest & ~far & ~cls1), "clean1": bc.frac(rest & ~far & cls1),
         "malformed0": bc.frac(rest & far & ~cls1), "malformed1": bc.frac(rest & far & cls1),
         "far": bc.frac(far), "blend_share_of_far": bc.frac(blend & far, denom=far & ok),
         "class1_share": bc.frac(rest & cls1, denom=rest), "class1_share_all": bc.frac(ok & cls1, denom=ok),
         "copy": bc.frac(ok & (nn < ref["copy_thr"]), denom=ok), "nn_median": float(np.median(nn[ok])) if ok.any() else float("nan")}
    if "toy" in ref:
        ll, post1 = toy_loglik(ref["toy"], P)
        r["toy_off"] = bc.frac(ok & (ll < ref["toy"]["ll_thr"]), denom=ok)
        r["toy_w1"] = bc.frac(ok & (post1 > 0.5), denom=ok)
        r["toy_ll_median"] = float(np.median(ll[ok])) if ok.any() else float("nan")
    # class-resolved rates among non-diverged samples (class from the Fisher score) and their 50/50 standardisation:
    # real 0s are 'far' much more often than real 1s, so the raw rates reward a model for drawing more 1s
    for c, cm in ((0, ok & ~cls1), (1, ok & cls1)):
        r[f"clean|{c}"] = bc.frac(cm & ~blend & ~far, denom=cm)
        r[f"far|{c}"] = bc.frac(cm & far, denom=cm)
        r[f"blend|{c}"] = bc.frac(cm & blend, denom=cm)
    for k in ("clean", "far", "blend"):
        r[f"{k}_std"] = 0.5 * (r[f"{k}|0"][0] + r[f"{k}|1"][0])
    rough, grey = texture(P, ref["grid"])
    r["roughness"] = float(np.mean(rough[ok])) if ok.any() else float("nan")
    r["grey_frac"] = float(np.mean(grey[ok])) if ok.any() else float("nan")
    # per-class pixel statistics over all non-diverged samples, class from the Fisher score
    for c, mask in ((0, ok & ~cls1), (1, ok & cls1)):
        Q = np.clip(P[mask], 0.0, 1.0)
        if Q.shape[0] < 2:
            continue
        r[f"pix{c}"] = {
            "n": int(Q.shape[0]),
            "rmse_mean": float(np.sqrt(np.mean((Q.mean(0) - ref["mu"][c]) ** 2))),
            "rmse_mean_floor": float(np.sqrt(np.mean(ref["sd"][c] ** 2) / Q.shape[0])),   # finite-n noise of the mean
            "rmse_std": float(np.sqrt(np.mean((Q.std(0) - ref["sd"][c]) ** 2))),
            "ink_frac": float(np.mean(Q > 0.5)),
            "ghost": float(np.mean(Q[:, ref["stroke"] if c == 0 else ref["ring"]])),
        }
    return r, dict(P=P, div=div, blend=blend, far=far, cls1=cls1, s=s, G=G)


def load_set(spec, with_parts=False):
    """Concatenate the files of 'PATH[+PATH...]'; with_parts=True also returns the length of each part."""
    parts = []
    for p in spec.split("+"):
        path = p if os.path.isabs(p) else os.path.join(lm.OUT_ROOT, p)
        parts.append(np.load(path))
    x = np.concatenate(parts)
    return (x, [q.shape[0] for q in parts]) if with_parts else x


def std_rates(d, idx):
    """50/50 class-standardised clean yield and far rate among non-diverged samples (as clean_std / far_std of
    classify), for every row of trajectory indices idx (B, n). Returns two arrays (B,); NaN where a class is empty."""
    ok, blend, far, cls1 = (~d["div"])[idx], d["blend"][idx], d["far"][idx], d["cls1"][idx]
    clean_std, far_std = np.zeros(idx.shape[0]), np.zeros(idx.shape[0])
    for cm in (ok & ~cls1, ok & cls1):
        n_c = cm.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            clean_std += 0.5 * (cm & ~blend & ~far).sum(axis=1) / n_c
            far_std += 0.5 * (cm & far).sum(axis=1) / n_c
    return clean_std, far_std


N_BOOT = 10000
MIN_CLASS_N = 20      # bootstrap the standardised rates only if every class has this many non-diverged samples


def paired_bootstrap(da, db, seed=0):
    """Paired bootstrap over trajectories (the same resampled indices for both sets) of the differences a - b in the
    standardised clean yield and far rate. McNemar does not apply here, because the class is itself an outcome.
    Returns (m, out): m = smallest class count (non-diverged, Fisher class) in either set; out is None if
    m < MIN_CLASS_N, else {'clean_std' / 'far_std': (difference on all trajectories, 2.5% and 97.5% quantiles,
    two-sided p, number of resamples with both classes present)}."""
    m = min(int(np.sum(~d["div"] & cm)) for d in (da, db) for cm in (~d["cls1"], d["cls1"]))
    if m < MIN_CLASS_N:
        return m, None
    n = da["div"].size
    idx = np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n), dtype=np.int32)
    full = np.arange(n, dtype=np.int32)[None, :]
    out = {}
    for k, (va, vb, fa, fb) in zip(("clean_std", "far_std"),
                                   zip(std_rates(da, idx), std_rates(db, idx), std_rates(da, full), std_rates(db, full))):
        diff = va - vb
        fin = np.isfinite(diff)
        diff = diff[fin]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p = min(1.0, 2.0 * min(np.mean(diff <= 0.0), np.mean(diff >= 0.0)))
        out[k] = (float(fa[0] - fb[0]), float(lo), float(hi), float(p), int(fin.sum()))
    return m, out


def json_safe(v):
    """Non-finite floats (e.g. the rates of an empty class) -> None, tuples -> lists, so metrics.json is strict JSON."""
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired booleans; returns (n_a_only, n_b_only, p)."""
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n        # exact integer arithmetic (no overflow)
    return n10, n01, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--problem", default="mnist28", help="mnist28, mnist14, gauss2_8 or a two-class digit subset mnist28_d<ab> / mnist14_d<ab>")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; evaluate_samples; job {os.environ.get('SLURM_JOB_ID', 'manual')}")
    ref = reference(args.problem)
    report(f"problem {args.problem}; far threshold {ref['far_thr']:.3f}"
           + (f"; class 0 = digit {ref['digits'][0]}, class 1 = digit {ref['digits'][1]} (columns 0 / 1 below)"
              if ref.get("digits") else ""))
    report(f"reference: blend threshold G > {ref['G_thr']:.3f}; class-score 99% bounds {ref['q0']:.3f} / {ref['q1']:.3f}; "
           f"ring pixels {int(ref['ring'].sum())}, stroke pixels {int(ref['stroke'].sum())}")
    sets = [("data (probes)", None)] + [tuple(s.split("=", 1)) for s in args.sets]
    rows, keep = {}, {}
    report("")
    report(f"{'sample set':<30}{'n':>5}{'%div':>6}{'%blend':>8}{'+-':>5}{'%gap':>6}{'%clean0':>8}{'%clean1':>8}"
           f"{'%malf0':>7}{'%malf1':>7}{'%far':>6}{'+-':>5}{'%cls1':>7}")
    for name, spec in sets:
        x, lens = (None, None) if spec is None else load_set(spec, with_parts=True)
        r, d = classify(ref, x)
        d["parts"] = lens or [r["n"]]
        rows[name], keep[name] = r, d
        pc = lambda k: 100 * r[k][0]
        report(f"{name:<30}{r['n']:>5}{pc('diverged'):>6.1f}{pc('blend'):>8.1f}{100 * r['blend'][1]:>5.1f}{pc('gap'):>6.1f}"
               f"{pc('clean0'):>8.1f}{pc('clean1'):>8.1f}{pc('malformed0'):>7.1f}{pc('malformed1'):>7.1f}{pc('far'):>6.1f}"
               f"{100 * r['far'][1]:>5.1f}{100 * r['class1_share_all'][0]:>7.1f}")
    report("(%cls1 = share of 1s by Fisher score among all non-diverged samples)")
    report("")
    report(f"memorisation check: copy = nearest training image closer than {ref['copy_thr']:.3f} (5% of held-out images by "
           f"construction); median nearest-image distance")
    for name in rows:
        report(f"{name:<30} copy {100 * rows[name]['copy'][0]:5.1f}% +- {100 * rows[name]['copy'][1]:.1f}   "
               f"median NN distance {rows[name]['nn_median']:.3f}")
    report("")
    report("texture (pixel space, non-diverged): roughness = mean squared 5-point Laplacian; grey = share of pixels in (0.2, 0.8)")
    for name in rows:
        report(f"{name:<30} roughness {rows[name]['roughness']:.4f}   grey {rows[name]['grey_frac']:.4f}")
    report("")
    report("class-resolved rates among non-diverged samples (class by Fisher score), and their 50/50 standardisation "
           "(std = mean over the two classes):")
    report(f"{'sample set':<30}{'n0':>5}{'n1':>5}{'clean|0':>8}{'clean|1':>8}{'clean std':>10}{'far|0':>7}{'far|1':>7}{'far std':>8}"
           f"{'blend|0':>8}{'blend|1':>8}{'blend std':>10}")
    for name in rows:
        r = rows[name]
        report(f"{name:<30}{r['clean|0'][3]:>5}{r['clean|1'][3]:>5}"
               f"{100 * r['clean|0'][0]:>8.1f}{100 * r['clean|1'][0]:>8.1f}{100 * r['clean_std']:>10.1f}"
               f"{100 * r['far|0'][0]:>7.1f}{100 * r['far|1'][0]:>7.1f}{100 * r['far_std']:>8.1f}"
               f"{100 * r['blend|0'][0]:>8.1f}{100 * r['blend|1'][0]:>8.1f}{100 * r['blend_std']:>10.1f}")
    if "toy" in ref:
        report("")
        report("exact data distribution (gauss2_8): off = log-likelihood below the 0.5% quantile of real samples; "
               "w1 = share of class 1 by exact posterior (data: 50%)")
        for name in rows:
            r = rows[name]
            report(f"{name:<30} off {100 * r['toy_off'][0]:5.1f}% +- {100 * r['toy_off'][1]:.1f}; "
                   f"w1 {100 * r['toy_w1'][0]:5.1f}% +- {100 * r['toy_w1'][1]:.1f}; median log-lik {r['toy_ll_median']:.1f}")
    report("")
    report("per-class pixel statistics (pixel space; class by Fisher score; non-diverged):")
    report(f"{'sample set':<30}{'cls':>4}{'n':>5}{'RMSE mean':>10}{'(floor)':>9}{'RMSE std':>9}{'ink frac':>9}{'ghost':>7}")
    for c in (0, 1):
        dmask = ref["labels"] == c
        Q = ref["px"][dmask]
        report(f"{'all training images':<30}{c:>4}{int(dmask.sum()):>5}{0.0:>10.4f}{'':>9}{0.0:>9.4f}{np.mean(Q > 0.5):>9.4f}"
               f"{np.mean(np.clip(Q, 0, 1)[:, ref['stroke'] if c == 0 else ref['ring']]):>7.4f}")
        for name in rows:
            p = rows[name].get(f"pix{c}")
            if p:
                report(f"{name:<30}{c:>4}{p['n']:>5}{p['rmse_mean']:>10.4f}{p['rmse_mean_floor']:>9.4f}{p['rmse_std']:>9.4f}{p['ink_frac']:>9.4f}"
                       f"{p['ghost']:>7.4f}")
    pairs = []
    for spec in args.pair:
        a, b = spec.split(",")
        if a not in keep or b not in keep or keep[a]["P"].shape != keep[b]["P"].shape \
                or keep[a]["parts"] != keep[b]["parts"]:
            report(f"pair {spec}: not comparable (missing or different sizes)")
            continue
        report("")
        report(f"paired comparison {a}  vs  {b} (same initial states and Brownian keys):")
        for what in ("blend", "far", "div"):
            x, y = keep[a][what], keep[b][what]
            n10, n01, pv = mcnemar(x, y)
            report(f"  {what:<6} {100 * x.mean():5.1f}% vs {100 * y.mean():5.1f}%: only {a}: {n10}, only {b}: {n01}; "
                   f"McNemar p = {pv:.2g}")
            pairs.append({"a": a, "b": b, "what": what, "a_only": n10, "b_only": n01, "p": pv})
        m, boot = paired_bootstrap(keep[a], keep[b])
        if boot is None:
            report(f"  clean_std / far_std: not estimable (smallest class count {m} < {MIN_CLASS_N})")
            pairs.append({"a": a, "b": b, "what": "std", "estimable": False, "min_class_n": m})
        for what, (d0, lo, hi, pv, nfin) in (boot or {}).items():
            ptxt = f"< {2.0 / nfin:.0e}" if pv == 0.0 else f"= {pv:.2g}"
            report(f"  {what:<9} {100 * rows[a][what]:5.1f}% vs {100 * rows[b][what]:5.1f}%: difference {100 * d0:+.1f} "
                   f"points, 95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]; paired bootstrap p {ptxt}"
                   + ("" if nfin == N_BOOT else f" ({nfin}/{N_BOOT} resamples with both classes)"))
            pairs.append({"a": a, "b": b, "what": what, "diff": d0, "ci95": [lo, hi], "p_boot": pv,
                          "n_resamples": nfin, "min_class_n": m})
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(json_safe({"G_threshold": ref["G_thr"], "sets": {n: s for n, s in sets[1:]},
                             "rows": rows, "pairs": pairs}), f, indent=2, allow_nan=False)

    # example grid: 8 random members of each category per set
    cats = [("clean 0", lambda d: ~d["div"] & ~d["blend"] & ~d["far"] & ~d["cls1"]),
            ("clean 1", lambda d: ~d["div"] & ~d["blend"] & ~d["far"] & d["cls1"]),
            ("blend", lambda d: d["blend"]),
            ("malformed 0", lambda d: ~d["div"] & ~d["blend"] & d["far"] & ~d["cls1"]),
            ("malformed 1", lambda d: ~d["div"] & ~d["blend"] & d["far"] & d["cls1"])]
    names = list(keep)
    cols = 8
    fig, axes = plt.subplots(len(cats) * len(names), cols, figsize=(1.1 * cols, 1.1 * len(cats) * len(names)),
                             squeeze=False)
    pick = np.random.default_rng(0)
    for si, name in enumerate(names):
        d = keep[name]
        for ci, (cname, sel) in enumerate(cats):
            idx = np.where(sel(d))[0]
            idx = pick.choice(idx, size=min(cols, idx.size), replace=False) if idx.size else idx
            row = si * len(cats) + ci
            for c in range(cols):
                ax = axes[row, c]
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if c < idx.size:
                    ax.imshow(d["P"][idx[c]].reshape(ref["grid"], ref["grid"]), cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_ylabel(f"{name}\n{cname}", rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("Random members of each category; pixel space, fixed limits [0, 1]", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "examples.png"), dpi=110)
    plt.close(fig)

    # class mean images per set
    fig, axes = plt.subplots(2, len(names) + 1, figsize=(1.6 * (len(names) + 1), 3.4), squeeze=False)
    for c in (0, 1):
        axes[c, 0].imshow(ref["mu"][c].reshape(ref["grid"], ref["grid"]), cmap="gray", vmin=0, vmax=1)
        axes[c, 0].set_title(f"data {c}", fontsize=7)
        for si, name in enumerate(names):
            d = keep[name]
            m = ~d["div"] & (d["cls1"] if c else ~d["cls1"])
            if m.sum():
                axes[c, si + 1].imshow(np.clip(d["P"][m], 0, 1).mean(0).reshape(ref["grid"], ref["grid"]), cmap="gray",
                                       vmin=0, vmax=1)
            axes[c, si + 1].set_title(f"{name[:18]}\n{c} (n={int(m.sum())})", fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "class_means.png"), dpi=120)
    plt.close(fig)
    report(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
