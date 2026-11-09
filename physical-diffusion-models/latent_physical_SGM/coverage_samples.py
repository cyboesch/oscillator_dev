#!/usr/bin/env python3
"""Coverage, diversity and templating of sample sets, per class, against held-out real images (the probes).

Usage:  python coverage_samples.py [--problem mnist28|mnist14] LABEL=PATH [LABEL=PATH ...]
  PATH as in evaluate_samples.py: .npy of final states (n, N) in normalised units, relative to out/latent_physical_SGM;
  PATH_A+PATH_B concatenates files.
Classes are those of evaluate_samples.py (Fisher score >= 0.5 is a 1), for samples and probes alike; diverged samples
(max |x| > 15) are left out. Images are compared in pixel space clipped to [0, 1]. Per class, every row uses the same
matched size n (n_match, or less if a class has fewer than 2 n_match probes; a set with fewer samples of the class
than n is not scored). S random subsamples of n images; every set is scored against the same S probe subsets
(common random numbers), and its own subsamples come from a generator seeded by its label (no order dependence):
  - k-NN precision and recall (Kynkaanniemi et al. 2019, k = 3) against n probes of the class. Precision: share of
    samples inside the real images' k-NN balls (fidelity). Recall: share of real images inside the samples' k-NN
    balls (coverage; it falls if the samples concentrate on fewer shapes than the data has);
  - spread: mean pairwise distance among the n images;
  - distinct nearest training images among the n images (training images = all non-probe images), a mode-collapse
    indicator;
  - and, computed once on all non-diverged images of the class (no subsampling, no uncertainty): the median distance
    to the nearest of 16 k-means centroids of the class (non-probe images). Samples closer to the centroids than real
    images are averaged or templated.
'mean +- std' is the spread over the S subsamples drawn from one fixed pool. It is NOT an error bar: for pools barely
larger than n it understates the noise of one n-draw (factor ~ sqrt((pool - n) / (pool - 1)); 0 when pool = n), so do
not read non-overlap between rows as significance.
Reference row 'real vs real': two disjoint random halves of the probes of the class, one as the reference and the
other as 'samples'; its mean is the target for real images.
Writes out/latent_physical_SGM/coverage/<job id>/report.txt and metrics.json. CPU only.
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import blend_check as bc  # noqa: E402
import evaluate_samples as es  # noqa: E402  (reference, load_set, json_safe; its main() is not run)

OUT = os.path.join(lm.OUT_ROOT, "coverage", os.environ.get("SLURM_JOB_ID", "manual"))
N_CENTROIDS = 16


def report(line=""):
    print(line, flush=True)
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(line + "\n")


def sqdist(A, B):
    return np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0)


def knn_radius2(A, k):
    """Squared distance of every row of A to its k-th nearest other row."""
    D = sqdist(A, A)
    np.fill_diagonal(D, np.inf)
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def precision_recall(R, G, k):
    """k-NN precision and recall of samples G against real images R (both (n, d))."""
    D = sqdist(G, R)
    precision = float(np.mean(np.any(D <= knn_radius2(R, k)[None, :], axis=1)))
    recall = float(np.mean(np.any(D <= knn_radius2(G, k)[:, None], axis=0)))
    return precision, recall


def spread(A):
    D = np.sqrt(sqdist(A, A))
    return float(D[np.triu_indices(A.shape[0], 1)].mean())


def nearest_index(Q, T, chunk=500):
    """Index of the nearest row of T for every row of Q."""
    t2 = np.sum(T * T, 1)
    out = np.empty(Q.shape[0], dtype=np.int64)
    for c0 in range(0, Q.shape[0], chunk):
        X = Q[c0:c0 + chunk]
        out[c0:c0 + chunk] = np.argmin(t2[None, :] - 2.0 * X @ T.T, axis=1)   # |x|^2 is constant per row
    return out


def summarise(prec, rec, spr, dist, n, pool, cdist):
    ms = lambda v: (float(np.mean(v)), float(np.std(v)))
    return {"pool": int(pool), "n_used": int(n), "precision": ms(prec), "recall": ms(rec), "spread": ms(spr),
            "distinct_nn": ms(dist), "centroid_dist_median": float(np.median(cdist))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--problem", default="mnist28", help="mnist28, mnist14 or a two-class digit subset mnist28_d<ab> / mnist14_d<ab>")
    ap.add_argument("--n-match", type=int, default=150)
    ap.add_argument("--subsamples", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; coverage_samples; job {os.environ.get('SLURM_JOB_ID', 'manual')}")
    n_cls = lm.problem_classes(args.problem)
    if n_cls == 2:                       # evaluate_samples.py: Fisher score >= 0.5 is class 1
        ref = es.reference(args.problem)
        assign = lambda P: (ref["lda"](P) >= 0.5).astype(np.int64)
        real_cls = (ref["s_p"] >= 0.5).astype(np.int64)
    else:                                # evaluate_multiclass.py: argmax of the LDA discriminants
        import evaluate_multiclass as emc
        ref = emc.reference(args.problem)
        assign = lambda P: np.argmax(P @ ref["w"].T + ref["disc_b"], axis=1)
        real_cls = assign(ref["px"][ref["probe"]])
    CLS = range(n_cls)
    px, probe = ref["px"], ref["probe"]
    is_probe = np.zeros(px.shape[0], bool)
    is_probe[probe] = True
    train = np.clip(px[~is_probe], 0.0, 1.0)                    # the non-probe images
    train_lab = ref["labels"][~is_probe]
    real = np.clip(px[probe], 0.0, 1.0)

    real_by_cls = {c: real[real_cls == c] for c in CLS}
    cents = {c: lm.kmeans(train[train_lab == c], N_CENTROIDS, seed=c)[0] for c in CLS}
    S, k = args.subsamples, args.k
    n_ref = {c: min(args.n_match, real_by_cls[c].shape[0] // 2) for c in CLS}
    report(f"problem {args.problem}; {train.shape[0]} non-probe training images; {real.shape[0]} probes "
           f"(" + " / ".join(f"{real_by_cls[c].shape[0]} class {c}" for c in CLS) + " by LDA class); matched n "
           f"{' / '.join(str(n_ref[c]) for c in CLS)} (n_match {args.n_match}), {S} subsamples, k = {k}; {N_CENTROIDS} k-means centroids "
           f"per class")

    # common random numbers: every set is scored against the same S probe subsets of each class
    prng = np.random.default_rng(1)
    probe_sub = {c: [prng.choice(real_by_cls[c].shape[0], size=n_ref[c], replace=False) for _ in range(S)]
                 for c in CLS}
    real_nn = nearest_index(real, train)
    results = {}

    # real vs real: disjoint halves of the probes of the class (a new random split for every subsample)
    rng = np.random.default_rng(0)
    rr = {}
    for c in CLS:
        Rc, Rnn, n = real_by_cls[c], real_nn[real_cls == c], n_ref[c]
        prec, rec, spr, dist = [], [], [], []
        for _ in range(S):
            perm = rng.permutation(Rc.shape[0])
            A, B = perm[:Rc.shape[0] // 2], perm[Rc.shape[0] // 2:]
            ai, bi = rng.choice(A, size=n, replace=False), rng.choice(B, size=n, replace=False)
            p, r = precision_recall(Rc[ai], Rc[bi], k)
            prec.append(p)
            rec.append(r)
            spr.append(spread(Rc[bi]))
            dist.append(np.unique(Rnn[bi]).size)
        rr[c] = summarise(prec, rec, spr, dist, n, Rc.shape[0], np.sqrt(np.min(sqdist(Rc, cents[c]), axis=1)))
    results["real vs real"] = rr

    for spec in args.sets:
        name, path = spec.split("=", 1)
        x = np.asarray(es.load_set(path), dtype=np.float64)
        div = ~(np.max(np.abs(x), axis=1) <= bc.MAX_ABS)
        P = np.where(np.isfinite(x), x, 0.0) * ref["std_px"] + ref["mean_px"]
        cls = assign(P)
        G_all = np.clip(P, 0.0, 1.0)
        nn_all = nearest_index(G_all, train)
        res = {"n": int(x.shape[0]), "diverged": int(div.sum())}
        for c in CLS:
            m = ~div & (cls == c)
            G, Gnn, n = G_all[m], nn_all[m], n_ref[c]
            if G.shape[0] < n:
                res[c] = {"pool": int(G.shape[0]), "n_used": 0}
                continue
            srng = np.random.default_rng([c, zlib.crc32(name.encode())])
            prec, rec, spr, dist = [], [], [], []
            for s in range(S):
                gi = srng.choice(G.shape[0], size=n, replace=False)
                p, r = precision_recall(real_by_cls[c][probe_sub[c][s]], G[gi], k)
                prec.append(p)
                rec.append(r)
                spr.append(spread(G[gi]))
                dist.append(np.unique(Gnn[gi]).size)
            res[c] = summarise(prec, rec, spr, dist, n, G.shape[0], np.sqrt(np.min(sqdist(G, cents[c]), axis=1)))
        results[name] = res

    for c in CLS:
        report("")
        report(f"class {c}: matched n = {n_ref[c]}; mean +- std over {S} subsamples from one pool (a subsampling spread, "
               f"not an error bar); centroid d on the whole pool")
        report(f"{'sample set':<30}{'pool':>5}{'precision':>15}{'recall':>15}{'spread':>15}{'distinct NN':>15}"
               f"{'centroid d':>11}")
        for name, res in results.items():
            r = res[c]
            if not r.get("n_used"):
                report(f"{name:<30}{r['pool']:>5}  fewer samples of this class than the matched n {n_ref[c]}; not scored")
                continue
            f2 = lambda v, d=3: f"{v[0]:.{d}f} +- {v[1]:.{d}f}"
            report(f"{name:<30}{r['pool']:>5}{f2(r['precision']):>15}{f2(r['recall']):>15}{f2(r['spread'], 2):>15}"
                   f"{f2(r['distinct_nn'], 1):>15}{r['centroid_dist_median']:>11.3f}")
    report("")
    report("pool = non-diverged samples of the class; precision ~ fidelity; recall ~ coverage of the real images' "
           "variety; spread = mean pairwise distance; distinct NN = number of different nearest training images "
           "among the n images; centroid d = median distance to the nearest of the class's k-means centroids "
           "(smaller than real = averaged / templated)")
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(es.json_safe({"problem": args.problem, "n_match": n_ref, "subsamples": S, "k": k,
                                "sets": {s.split("=", 1)[0]: s.split("=", 1)[1] for s in args.sets},
                                "results": {n: {str(c): v for c, v in r.items()} for n, r in results.items()}}),
                  f, indent=2, allow_nan=False)
    report(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
