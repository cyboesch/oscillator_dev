#!/usr/bin/env python3
"""Sample evaluation for C >= 2 classes: evaluate_samples.py generalised from two classes to C.

For C = 2 it reproduces evaluate_samples.py: the same probes, nearest-image distance, blend score, far and copy
thresholds, class assignment and 'between classes' test. Run it on a two-class problem to check this.
  class        argmax of the ridge-regularised LDA discriminants (pooled within-class covariance + 1e-2 tr/N, equal
               priors); for C = 2 this is 'Fisher score >= 0.5'
  between      ('gap') best class a, runner-up b: the a-b Fisher score s_ab (mu_a -> 0, mu_b -> 1) lies beyond the 99%
               quantile of s_ab over real a's and beyond the 1% quantile over real b's (for C = 2: q0 < s < q1)
  blend        G = 1 - r2 / r1 above its 95% quantile over the probes; r1 = best single template (non-negative scale)
               among the K_NN nearest training images of every class; r2 = best non-negative pair of templates from
               two different classes, minimised over the class pairs (for C = 2: blend_check.scores)
  far / copy   nearest training image farther than the probes' p90 (4.84 for the paper's mnist28) / closer than p5
  clean c      assigned class c, not a blend, not far; malformed c: assigned c, not a blend, far
Class-resolved rates among non-diverged samples, and their standardisation (mean over the C classes; for C = 2 the
'50/50' rates of evaluate_samples.py). Paired comparisons: exact McNemar for blend / far / diverged, paired bootstrap
for the standardised clean yield and far rate.

Usage (CPU Slurm job): python evaluate_multiclass.py --problem mnist14_d147 LABEL=PATH[+PATH] ... [--pair A,B ...]
Writes out/latent_physical_SGM/eval_mc/<job>/report.txt and metrics.json.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import blend_check as bc  # noqa: E402
import evaluate_samples as es  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "eval_mc", os.environ.get("SLURM_JOB_ID", "manual"))
_lines = []


def report(line=""):
    print(line, flush=True)
    _lines.append(line)


def scores_mc(Q, px_train, tr_n2, labels, n_cls, self_idx=None):
    """Nearest-image distance and the C-class blend score G for query images Q (pixel space); bc.scores for C = 2."""
    n = Q.shape[0]
    nn, G = np.empty(n), np.empty(n)
    idx = [np.where(labels == c)[0] for c in range(n_cls)]
    for c0 in range(0, n, bc.CHUNK):
        X = Q[c0:c0 + bc.CHUNK]
        xx = np.sum(X**2, axis=1)
        d2 = xx[:, None] + tr_n2[None, :] - 2.0 * X @ px_train.T
        if self_idx is not None:
            d2[np.arange(X.shape[0]), self_idx[c0:c0 + bc.CHUNK]] = np.inf
        nn[c0:c0 + bc.CHUNK] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        nb = [ic[np.argpartition(d2[:, ic], bc.K_NN, axis=1)[:, :bc.K_NN]] for ic in idx]
        Z = [px_train[b] for b in nb]
        xz = [np.einsum("cd,ckd->ck", X, z) for z in Z]
        zz = [tr_n2[b] for b in nb]
        r1 = xx - np.max(np.concatenate([np.maximum(a, 0)**2 / b for a, b in zip(xz, zz)], axis=1), axis=1)
        r2 = r1.copy()
        for a, b in itertools.combinations(range(n_cls), 2):
            C = np.einsum("cad,cbd->cab", Z[a], Z[b])
            A, B = zz[a][:, :, None], zz[b][:, None, :]
            ca, cb = xz[a][:, :, None], xz[b][:, None, :]
            det = A * B - C**2
            with np.errstate(divide="ignore", invalid="ignore"):
                al = (B * ca - C * cb) / det
                be = (A * cb - C * ca) / det
                r2p = xx[:, None, None] - (al * ca + be * cb)
            valid = (al >= 0) & (be >= 0) & (det > 0)
            r2p = np.where(valid, r2p, np.inf).reshape(X.shape[0], -1)
            r2 = np.minimum(r2, r2p.min(axis=1))
        G[c0:c0 + bc.CHUNK] = 1.0 - np.maximum(r2, 0.0) / np.maximum(r1, 1e-300)
    return nn, G


def reference(problem):
    prob = lm.make_problem(problem)
    data, labels, mean_px, std_px = np.asarray(prob["data"]), prob["labels"], prob["mean"], prob["std"]
    n_cls = int(labels.max()) + 1
    px = data * std_px + mean_px
    tr_n2 = np.sum(px**2, axis=1)
    rng = np.random.default_rng(bc.PROBE_SEED)                          # the probes of evaluate_samples.py
    probe = np.sort(rng.choice(px.shape[0], size=bc.N_PROBE, replace=False))
    is_probe = np.zeros(px.shape[0], bool)
    is_probe[probe] = True
    Xc = [px[~is_probe & (labels == c)] for c in range(n_cls)]
    mu = np.stack([x.mean(0) for x in Xc])
    Sw = sum((x - m).T @ (x - m) for x, m in zip(Xc, mu)) / (sum(x.shape[0] for x in Xc) - n_cls)
    lam = 1e-2 * np.trace(Sw) / Sw.shape[0]
    Sinv_mu = np.linalg.solve(Sw + lam * np.eye(Sw.shape[0]), mu.T).T  # (C, N): w_c = (Sw + lam I)^-1 mu_c
    disc_b = -0.5 * np.sum(Sinv_mu * mu, axis=1)
    ref = dict(px=px, tr_n2=tr_n2, labels=labels, probe=probe, mean_px=mean_px, std_px=std_px, grid=prob["grid"],
               n_cls=n_cls, digits=prob.get("digits"), mu_lda=mu, w=Sinv_mu, disc_b=disc_b)
    nn_p, G_p = scores_mc(px[probe], px, tr_n2, labels, n_cls, self_idx=probe)
    ref.update(nn_p=nn_p, G_p=G_p, G_thr=float(np.quantile(G_p, 1.0 - bc.FP_RATE)),
               far_thr=bc.FAR_THRESHOLD if problem == "mnist28" else float(np.percentile(nn_p, 90)),
               copy_thr=float(np.percentile(nn_p, 5)))
    # pairwise Fisher scores s_ab (mu_a -> 0, mu_b -> 1) and the 99% quantile of s_ab over the probes of class a
    lab_p = labels[probe]
    q = np.full((n_cls, n_cls), np.nan)
    for a in range(n_cls):
        for b in range(n_cls):
            if a != b:
                q[a, b] = float(np.quantile(pair_score(ref, px[probe][lab_p == a], a, b), 0.99))
    ref["q"] = q
    pc = np.clip(px, 0.0, 1.0)
    ref.update(mu=[pc[labels == c].mean(0) for c in range(n_cls)], sd=[pc[labels == c].std(0) for c in range(n_cls)])
    return ref


def pair_score(ref, P, a, b):
    """Fisher score of P on the a-b axis of the LDA, scaled so that mu_a -> 0 and mu_b -> 1."""
    w = ref["w"][b] - ref["w"][a]
    return (P @ w - ref["mu_lda"][a] @ w) / ((ref["mu_lda"][b] - ref["mu_lda"][a]) @ w)


def classify(ref, x=None):
    n_cls = ref["n_cls"]
    if x is None:
        P, nn, G = ref["px"][ref["probe"]], ref["nn_p"], ref["G_p"]
        div = np.zeros(P.shape[0], bool)
    else:
        x = np.asarray(x, dtype=np.float64)
        div = ~(np.max(np.abs(x), axis=1) <= bc.MAX_ABS)
        P = np.where(np.isfinite(x), x, 0.0) * ref["std_px"] + ref["mean_px"]
        nn, G = scores_mc(P, ref["px"], ref["tr_n2"], ref["labels"], n_cls)
    disc = P @ ref["w"].T + ref["disc_b"]                               # (n, C)
    order = np.argsort(-disc, axis=1)
    cls, second = order[:, 0], order[:, 1]
    s = np.array([pair_score(ref, P[i:i + 1], a, b)[0] for i, (a, b) in enumerate(zip(cls, second))]) \
        if P.shape[0] else np.zeros(0)
    gap = (s > ref["q"][cls, second]) & (s < 1.0 - ref["q"][second, cls])
    ok = ~div
    blend = ok & (G > ref["G_thr"])
    far = div | (nn > ref["far_thr"])
    rest = ok & ~blend
    r = {"n": int(P.shape[0]), "diverged": bc.frac(div), "blend": bc.frac(blend), "gap": bc.frac(ok & gap),
         "far": bc.frac(far), "copy": bc.frac(ok & (nn < ref["copy_thr"]), denom=ok),
         "nn_median": float(np.median(nn[ok])) if ok.any() else float("nan")}
    for c in range(n_cls):
        cm = ok & (cls == c)
        r[f"clean{c}"] = bc.frac(rest & ~far & (cls == c))
        r[f"malformed{c}"] = bc.frac(rest & far & (cls == c))
        r[f"share{c}"] = bc.frac(cm, denom=ok)
        r[f"clean|{c}"] = bc.frac(cm & ~blend & ~far, denom=cm)
        r[f"far|{c}"] = bc.frac(cm & far, denom=cm)
        r[f"blend|{c}"] = bc.frac(cm & blend, denom=cm)
    for k in ("clean", "far", "blend"):
        r[f"{k}_std"] = float(np.mean([r[f"{k}|{c}"][0] for c in range(n_cls)]))
    rough, grey = es.texture(P, ref["grid"])
    r["roughness"] = float(np.mean(rough[ok])) if ok.any() else float("nan")
    r["grey_frac"] = float(np.mean(grey[ok])) if ok.any() else float("nan")
    for c in range(n_cls):
        Qc = np.clip(P[ok & (cls == c)], 0.0, 1.0)
        if Qc.shape[0] >= 2:
            r[f"pix{c}"] = {"n": int(Qc.shape[0]),
                            "rmse_mean": float(np.sqrt(np.mean((Qc.mean(0) - ref["mu"][c]) ** 2))),
                            "rmse_mean_floor": float(np.sqrt(np.mean(ref["sd"][c] ** 2) / Qc.shape[0])),
                            "rmse_std": float(np.sqrt(np.mean((Qc.std(0) - ref["sd"][c]) ** 2))),
                            "ink_frac": float(np.mean(Qc > 0.5))}
    return r, dict(P=P, div=div, blend=blend, far=far, cls=cls, G=G, nn=nn)


def std_rates(d, idx, n_cls):
    """Class-standardised clean yield and far rate (mean over the C classes) for every row of indices idx (B, n)."""
    ok, blend, far, cls = (~d["div"])[idx], d["blend"][idx], d["far"][idx], d["cls"][idx]
    clean_std, far_std = np.zeros(idx.shape[0]), np.zeros(idx.shape[0])
    for c in range(n_cls):
        cm = ok & (cls == c)
        n_c = cm.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            clean_std += (cm & ~blend & ~far).sum(axis=1) / n_c / n_cls
            far_std += (cm & far).sum(axis=1) / n_c / n_cls
    return clean_std, far_std


def paired_bootstrap(da, db, n_cls, seed=0):
    """As evaluate_samples.paired_bootstrap, for C classes."""
    m = min(int(np.sum(~d["div"] & (d["cls"] == c))) for d in (da, db) for c in range(n_cls))
    if m < es.MIN_CLASS_N:
        return m, None
    n = da["div"].size
    idx = np.random.default_rng(seed).integers(0, n, size=(es.N_BOOT, n), dtype=np.int32)
    full = np.arange(n, dtype=np.int32)[None, :]
    out = {}
    for k, (va, vb, fa, fb) in zip(("clean_std", "far_std"),
                                   zip(std_rates(da, idx, n_cls), std_rates(db, idx, n_cls),
                                       std_rates(da, full, n_cls), std_rates(db, full, n_cls))):
        diff = va - vb
        fin = np.isfinite(diff)
        diff = diff[fin]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p = min(1.0, 2.0 * min(np.mean(diff <= 0.0), np.mean(diff >= 0.0)))
        out[k] = (float(fa[0] - fb[0]), float(lo), float(hi), float(p), int(fin.sum()))
    return m, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--problem", required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; evaluate_multiclass; job {os.environ.get('SLURM_JOB_ID', 'manual')}")
    ref = reference(args.problem)
    C = ref["n_cls"]
    names = [f"digit {d}" for d in ref["digits"]] if ref.get("digits") else [f"class {c}" for c in range(C)]
    report(f"problem {args.problem}; {C} classes: " + ", ".join(f"{c} = {nm}" for c, nm in enumerate(names))
           + f"; far threshold {ref['far_thr']:.3f}; blend threshold G > {ref['G_thr']:.3f}; copy < {ref['copy_thr']:.3f}")
    sets = [("data (probes)", classify(ref))]
    raw = {}
    for spec in args.sets:
        label, path = spec.split("=", 1)
        x = es.load_set(path)
        r, d = classify(ref, x)
        sets.append((label, (r, d)))
        raw[label] = d
    report("")
    head = f"{'sample set':<30}{'n':>5}{'%div':>6}{'%blend':>8}{'%gap':>6}{'%far':>6}" + "".join(
        f"{'%clean' + str(c):>8}" for c in range(C)) + "".join(f"{'%share' + str(c):>8}" for c in range(C))
    report(head)
    for label, (r, _) in sets:
        report(f"{label:<30}{r['n']:>5}{100 * r['diverged'][0]:>6.1f}{100 * r['blend'][0]:>8.1f}{100 * r['gap'][0]:>6.1f}"
               f"{100 * r['far'][0]:>6.1f}" + "".join(f"{100 * r[f'clean{c}'][0]:>8.1f}" for c in range(C))
               + "".join(f"{100 * r[f'share{c}'][0]:>8.1f}" for c in range(C)))
    report("(%share c = share of class c among non-diverged samples, by LDA class)")
    report("")
    report("class-resolved rates among non-diverged samples, and their standardisation (mean over the classes):")
    report(f"{'sample set':<30}" + "".join(f"{'n' + str(c):>6}" for c in range(C))
           + "".join(f"{'clean|' + str(c):>9}" for c in range(C)) + f"{'clean std':>10}"
           + "".join(f"{'far|' + str(c):>8}" for c in range(C)) + f"{'far std':>9}{'blend std':>10}")
    for label, (r, d) in sets:
        ns = [int(np.sum(~d["div"] & (d["cls"] == c))) for c in range(C)] if "cls" in d else []
        report(f"{label:<30}" + "".join(f"{v:>6}" for v in ns)
               + "".join(f"{100 * r[f'clean|{c}'][0]:>9.1f}" for c in range(C)) + f"{100 * r['clean_std']:>10.1f}"
               + "".join(f"{100 * r[f'far|{c}'][0]:>8.1f}" for c in range(C)) + f"{100 * r['far_std']:>9.1f}"
               + f"{100 * r['blend_std']:>10.1f}")
    report("")
    report("memorisation and texture: copy rate, median nearest-image distance, roughness (mean squared 5-point Laplacian)")
    for label, (r, _) in sets:
        report(f"{label:<30} copy {100 * r['copy'][0]:5.1f}%   median NN {r['nn_median']:.3f}   roughness {r['roughness']:.4f}")
    report("")
    report("per-class mean images: RMSE of the class mean against the data's (finite-sample floor), ink fraction")
    for label, (r, _) in sets[1:]:
        report(f"{label:<30}" + "   ".join(
            f"{c}: {r[f'pix{c}']['rmse_mean']:.4f} ({r[f'pix{c}']['rmse_mean_floor']:.4f}) ink {r[f'pix{c}']['ink_frac']:.3f}"
            if f"pix{c}" in r else f"{c}: -" for c in range(C)))
    pairs_out = {}
    for pr in args.pair:
        a, b = pr.split(",")
        da, db = raw[a], raw[b]
        if da["div"].shape != db["div"].shape:
            report(f"pair {a},{b}: not comparable (different sizes)")
            continue
        report("")
        report(f"paired comparison {a}  vs  {b} (same initial states and Brownian keys):")
        res = {}
        for k in ("blend", "far", "div"):
            xa = da["div"] if k == "div" else da[k]
            xb = db["div"] if k == "div" else db[k]
            n10, n01, p = es.mcnemar(xa, xb)
            res[k] = dict(a=float(xa.mean()), b=float(xb.mean()), only_a=n10, only_b=n01, p=p)
            report(f"  {k:<6} {100 * xa.mean():5.1f}% vs {100 * xb.mean():5.1f}%: only {a}: {n10}, only {b}: {n01}; "
                   f"McNemar p = {p:.2g}")
        m, bo = paired_bootstrap(da, db, C)
        if bo is None:
            report(f"  standardised rates: not estimable (smallest class count {m} < {es.MIN_CLASS_N})")
        else:
            ra = dict(sets)[a][0]
            rb = dict(sets)[b][0]
            for k, (diff, lo, hi, p, nf) in bo.items():
                ptxt = f"p < {2 / es.N_BOOT:.0e}" if p == 0 else f"p = {p:.2g}"
                report(f"  {k:<9} {100 * ra[k]:5.1f}% vs {100 * rb[k]:5.1f}%: difference {100 * diff:+.1f} points, "
                       f"95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]; paired bootstrap {ptxt}")
                res[k] = dict(diff=diff, lo=lo, hi=hi, p=p, n_finite=nf)
        pairs_out[f"{a},{b}"] = res
    report(f"total time {time.time() - t0:.0f} s")
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(_lines) + "\n")
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(es.json_safe({"problem": args.problem, "n_classes": C, "names": names,
                                "thresholds": {"G": ref["G_thr"], "far": ref["far_thr"], "copy": ref["copy_thr"]},
                                "sets": {lab: r for lab, (r, _) in sets}, "pairs": pairs_out}), f, indent=1,
                  allow_nan=False)


if __name__ == "__main__":
    main()
