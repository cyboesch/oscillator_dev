#!/usr/bin/env python3
"""FID and KID of generated digits, in the features of a small MNIST classifier (in addition to the % measures of
evaluate_multiclass.py).

Classifier: a small CNN (conv 3x3 x32 - avgpool 2 - conv 3x3 x64 - avgpool 2 - dense 128 - dense 10), trained once per
resolution on all ten digits of the MNIST train split (tensorflow_datasets 'mnist', /255; 14x14 = 2x2 average pooling,
as make_problem), with the paper's N(0, 0.01^2) pixel noise as augmentation. Cached in
out/latent_physical_SGM/fid/classifier_<grid>.npz with its test accuracy. Features = the 128 hidden units (after ReLU).

Reference: the MNIST TEST images of the problem's digits (never seen by any generative model), classes balanced by
subsampling (seed 0), plus the paper's N(0, 0.01^2) noise. Samples are mapped to pixel units (x * std + mean of the
problem); rows that diverged (max |x| > 15 in normalised units, as blend_check) or are not finite are left out and
counted. Every set is scored twice: on the raw pixel values (as the % measures) and clipped to [0, 1] (as the
galleries).

Reported per set: n used; FID; KID x 1000 (unbiased MMD^2, cubic polynomial kernel on features scaled to unit rms over the reference, reference subsampled to <= 2000);
bootstrap sd of FID (200 resamples of the set); the classifier's view: share of samples it assigns to the problem's
digits with max p >= 0.9 ("confident"), and the class shares.
Floors: FID / KID of n random real images (the problem's training images, the generative models' data) against the
reference, mean +- sd over 10 draws, at the sample sets' n (FID is biased upwards at small n: compare against this
floor, not against 0). And the training images at full size.
--pair A,B: paired bootstrap (same trajectory indices for both sets) of FID_A - FID_B and KID_A - KID_B: difference,
95% interval, two-sided bootstrap p.

Usage (CPU Slurm job): python fid_score.py --problem mnist28_d0123 LABEL=PATH[+PATH] ... [--pair A,B ...]
                       python fid_score.py --train-only --grid 28     (train and cache the classifier)
Writes out/latent_physical_SGM/fid/<job>/report.txt and metrics.json.
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
import evaluate_samples as es  # noqa: E402

FID_DIR = os.path.join(lm.OUT_ROOT, "fid")
OUT = os.path.join(FID_DIR, os.environ.get("SLURM_JOB_ID", "manual"))
DIV_ABS = 15.0
NOISE = 0.01
N_BOOT = 200
N_FLOOR = 10
KID_REF_MAX = 2000
PR_K, N_PR = 3, 5          # k-NN precision / recall: k, number of reference draws
_lines = []


def report(s=""):
    print(s, flush=True)
    _lines.append(s)


# ---------------------------------------------------------------- MNIST -------------------------------------------
def load_mnist(split, grid):
    import tensorflow_datasets as tfds
    b = tfds.builder("mnist")
    b.download_and_prepare()
    X, y = [], []
    for img, lab in tfds.as_numpy(b.as_dataset(split=split, as_supervised=True, batch_size=10000)):
        X.append(img.astype(np.float32).reshape(img.shape[0], 28, 28) / 255.0)
        y.append(lab.astype(np.int64))
    X, y = np.concatenate(X), np.concatenate(y)
    if grid == 14:
        X = X.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
    return X.reshape(X.shape[0], -1), y


# ---------------------------------------------------------------- classifier --------------------------------------
def init_params(grid, rng):
    flat = (((grid // 2) // 2) ** 2) * 64

    def he(shape, fan_in):
        return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
    return {"c1": he((3, 3, 1, 32), 9), "b1": np.zeros(32, np.float32),
            "c2": he((3, 3, 32, 64), 9 * 32), "b2": np.zeros(64, np.float32),
            "d1": he((flat, 128), flat), "e1": np.zeros(128, np.float32),
            "d2": he((128, 10), 128), "e2": np.zeros(10, np.float32)}


def make_net(grid):
    import jax
    import jax.numpy as jnp
    from jax import lax

    def conv(h, w):
        return lax.conv_general_dilated(h, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))

    def pool(h):
        return lax.reduce_window(h, 0.0, lax.add, (1, 2, 2, 1), (1, 2, 2, 1), "VALID") / 4.0

    def forward(p, x):
        h = x.reshape(-1, grid, grid, 1)
        h = pool(jax.nn.relu(conv(h, p["c1"]) + p["b1"]))
        h = pool(jax.nn.relu(conv(h, p["c2"]) + p["b2"]))
        f = jax.nn.relu(h.reshape(h.shape[0], -1) @ p["d1"] + p["e1"])
        return f, f @ p["d2"] + p["e2"]
    return jax.jit(forward)


def train_classifier(grid, epochs=5, batch=128, lr=1e-3, seed=0):
    import jax
    import jax.numpy as jnp
    t0 = time.time()
    Xtr, ytr = load_mnist("train", grid)
    Xte, yte = load_mnist("test", grid)
    rng = np.random.default_rng(seed)
    p = {k: jnp.asarray(v) for k, v in init_params(grid, rng).items()}
    forward = make_net(grid)

    def loss(p, x, y):
        _, logits = forward(p, x)
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], 1))

    @jax.jit
    def step(p, m, v, t, x, y):
        g = jax.grad(loss)(p, x, y)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        p = jax.tree_util.tree_map(lambda q, a, b: q - lr * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8),
                                   p, m, v)
        return p, m, v

    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    t = 0
    for ep in range(epochs):
        perm = rng.permutation(Xtr.shape[0])
        for i in range(0, Xtr.shape[0] - batch + 1, batch):
            idx = perm[i:i + batch]
            x = Xtr[idx] + NOISE * rng.standard_normal((batch, Xtr.shape[1])).astype(np.float32)
            t += 1
            p, m, v = step(p, m, v, t, jnp.asarray(x), jnp.asarray(ytr[idx]))
        acc = test_accuracy(forward, p, Xte, yte)
        print(f"classifier {grid}x{grid}: epoch {ep + 1}, test accuracy {acc:.4f} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(FID_DIR, exist_ok=True)
    path = os.path.join(FID_DIR, f"classifier_{grid}.npz")
    np.savez(path, test_accuracy=acc, epochs=epochs, **{k: np.asarray(val) for k, val in p.items()})
    print(f"wrote {path}", flush=True)
    return path


def test_accuracy(forward, p, X, y):
    pred = np.concatenate([np.asarray(forward(p, X[i:i + 2000])[1]).argmax(1) for i in range(0, X.shape[0], 2000)])
    return float(np.mean(pred == y))


def load_classifier(grid):
    import jax.numpy as jnp
    path = os.path.join(FID_DIR, f"classifier_{grid}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run fid_score.py --train-only --grid {grid} first")
    z = np.load(path)
    p = {k: jnp.asarray(z[k]) for k in ("c1", "b1", "c2", "b2", "d1", "e1", "d2", "e2")}
    forward = make_net(grid)

    def feats(X):
        F, P = [], []
        for i in range(0, X.shape[0], 2000):
            f, logits = forward(p, np.asarray(X[i:i + 2000], np.float32))
            F.append(np.asarray(f, np.float64))
            P.append(np.asarray(logits, np.float64))
        L = np.concatenate(P)
        L = np.exp(L - L.max(1, keepdims=True))
        return np.concatenate(F), L / L.sum(1, keepdims=True)
    return feats, float(z["test_accuracy"])


# ---------------------------------------------------------------- distances ---------------------------------------
def stats(F):
    return F.mean(0), np.cov(F, rowvar=False)


def fid_from_stats(m1, C1, m2, C2):
    w, U = np.linalg.eigh(C1)
    C1h = (U * np.sqrt(np.maximum(w, 0.0))) @ U.T
    ev = np.linalg.eigvalsh(C1h @ C2 @ C1h)
    return float(np.sum((m1 - m2) ** 2) + np.trace(C1) + np.trace(C2) - 2.0 * np.sum(np.sqrt(np.maximum(ev, 0.0))))


def kernel(A, B):
    return (A @ B.T / A.shape[1] + 1.0) ** 3


def kid_from_kernels(Kxx, Kxy, kyy_mean):
    """Unbiased MMD^2 with the off-diagonal mean of Kxx (positions, so repeated bootstrap rows count as distinct)."""
    n = Kxx.shape[0]
    kxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    return float(kxx + kyy_mean - 2.0 * Kxy.mean())


def sqdist(A, B):
    return np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0)


def prec_rec(S, R, k=PR_K):
    """k-NN precision and recall (Kynkaanniemi et al. 2019) of samples S against real images R (rows)."""
    def radii(Y):
        D = sqdist(Y, Y)
        np.fill_diagonal(D, np.inf)
        return np.sqrt(np.partition(D, k - 1, axis=1)[:, k - 1])
    D = np.sqrt(sqdist(S, R))
    return float(np.mean((D <= radii(R)[None, :]).any(1))), float(np.mean((D <= radii(S)[:, None]).any(0)))


class Scorer:
    def __init__(self, Fref):
        self.mr, self.Cr = stats(Fref)
        self.s = float(np.sqrt(np.mean(Fref ** 2)))          # KID on features scaled to unit rms (reference)
        rng = np.random.default_rng(1)
        self.Fk = (Fref if Fref.shape[0] <= KID_REF_MAX else Fref[rng.choice(Fref.shape[0], KID_REF_MAX, replace=False)]) / self.s
        Kyy = kernel(self.Fk, self.Fk)
        m = self.Fk.shape[0]
        self.kyy = float((Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)))

    def fid(self, F):
        return fid_from_stats(*stats(F), self.mr, self.Cr)

    def kid(self, F):
        G = F / self.s
        return kid_from_kernels(kernel(G, G), kernel(G, self.Fk), self.kyy)


# ---------------------------------------------------------------- main --------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--problem")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--grid", type=int, choices=(14, 28))
    args = ap.parse_args()
    if args.train_only:
        train_classifier(args.grid)
        return
    if not args.problem or not args.sets:
        raise SystemExit("--problem and at least one LABEL=PATH are needed")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; fid_score; job {os.environ.get('SLURM_JOB_ID', 'manual')}")
    prob = lm.make_problem(args.problem)
    grid, digits = prob["grid"], prob["digits"]
    mean, std = prob["mean"], prob["std"]
    feats, acc = load_classifier(grid)
    # reference: balanced MNIST test images of the problem's digits, with the paper's pixel noise
    Xte, yte = load_mnist("test", grid)
    rng = np.random.default_rng(0)
    nmin = min(int(np.sum(yte == d)) for d in digits)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yte == d), nmin, replace=False) for d in digits]))
    Xref = Xte[idx] + NOISE * rng.standard_normal((idx.size, Xte.shape[1]))
    Fref, _ = feats(Xref)
    sc = Scorer(Fref)
    report(f"problem {args.problem} (digits {digits}, {grid}x{grid}); classifier test accuracy (all ten digits) "
           f"{acc:.4f}; reference: {idx.size} MNIST test images ({nmin} per digit) + N(0, {NOISE}^2) noise; "
           f"KID reference {sc.Fk.shape[0]} of them")
    # real training images (the generative models' data), pixel units
    Ptr = np.asarray(prob["data"], np.float64) * std + mean
    Ftr, Ptr_cls = feats(Ptr)
    sets, metrics = {}, {}
    for spec in args.sets:
        label, path = spec.split("=", 1)
        missing = [p for p in path.split("+") if not os.path.exists(p if os.path.isabs(p) else os.path.join(lm.OUT_ROOT, p))]
        if missing:
            report(f"{label}: skipped, no samples ({', '.join(missing)})")
            continue
        x = np.asarray(es.load_set(path), np.float64)
        ok = np.all(np.isfinite(x), 1)
        ok[ok] = np.max(np.abs(x[ok]), 1) <= DIV_ABS
        P = np.where(ok[:, None], np.nan_to_num(x) * std + mean, 0.0)
        F_raw, Pr_raw = feats(P)
        F_clip, _ = feats(np.clip(P, 0.0, 1.0))
        sets[label] = dict(ok=ok, raw=F_raw, clip=F_clip, prob=Pr_raw, px=np.clip(P, 0.0, 1.0))
    report("")
    report(f"{'set':<34}{'n':>5}{'div':>5}{'FID':>9}{'(sd)':>7}{'KIDx1e3':>9}{'FID clip':>10}{'KID clip':>10}"
           f"{'confident':>11}  class shares (classifier, among non-diverged)")
    n_typ = int(np.median([s["ok"].sum() for s in sets.values()]))
    fl = []
    for r in range(N_FLOOR):
        j = np.random.default_rng(100 + r).choice(Ftr.shape[0], n_typ, replace=False)
        fl.append((sc.fid(Ftr[j]), 1e3 * sc.kid(Ftr[j])))
    fl = np.asarray(fl)
    conf_tr = np.mean(np.isin(Ptr_cls.argmax(1), digits) & (Ptr_cls.max(1) >= 0.9))
    report(f"{'real training images (floor)':<34}{n_typ:>5}{'':>5}{fl[:, 0].mean():>9.2f}{fl[:, 0].std():>7.2f}"
           f"{fl[:, 1].mean():>9.2f}{'':>10}{'':>10}{100 * conf_tr:>10.1f}%  ({N_FLOOR} draws of n; sd over draws)")
    report(f"{'real training images (all)':<34}{Ftr.shape[0]:>5}{'':>5}{sc.fid(Ftr):>9.2f}{'':>7}"
           f"{1e3 * sc.kid(Ftr[np.random.default_rng(7).choice(Ftr.shape[0], min(Ftr.shape[0], 5000), replace=False)]):>9.2f}")
    metrics["floor"] = {"n": n_typ, "fid_mean": float(fl[:, 0].mean()), "fid_sd": float(fl[:, 0].std()),
                        "kid_mean": float(fl[:, 1].mean()), "kid_sd": float(fl[:, 1].std()), "confident": float(conf_tr)}
    brng = np.random.default_rng(11)
    for label, s in sets.items():
        ok = s["ok"]
        Fr, Fc, Pp = s["raw"][ok], s["clip"][ok], s["prob"][ok]
        fid_r, kid_r = sc.fid(Fr), 1e3 * sc.kid(Fr)
        fid_c, kid_c = sc.fid(Fc), 1e3 * sc.kid(Fc)
        boot = [sc.fid(Fr[brng.integers(0, Fr.shape[0], Fr.shape[0])]) for _ in range(N_BOOT)]
        cls = Pp.argmax(1)
        conf = np.mean(np.isin(cls, digits) & (Pp.max(1) >= 0.9))
        shares = " ".join(f"{d}:{100 * np.mean(cls == d):.0f}" for d in range(10) if np.mean(cls == d) >= 0.005)
        report(f"{label:<34}{int(ok.sum()):>5}{int((~ok).sum()):>5}{fid_r:>9.2f}{np.std(boot):>7.2f}{kid_r:>9.2f}"
               f"{fid_c:>10.2f}{kid_c:>10.2f}{100 * conf:>10.1f}%  {shares}")
        metrics[label] = {"n": int(ok.sum()), "diverged": int((~ok).sum()), "fid": fid_r, "fid_boot_sd": float(np.std(boot)),
                          "kid_x1e3": kid_r, "fid_clip": fid_c, "kid_clip_x1e3": kid_c, "confident": float(conf),
                          "class_shares": {int(d): float(np.mean(cls == d)) for d in range(10)}}
    report("(FID in the classifier's 128 features, raw pixel values unless 'clip'; KID x 1000, unbiased, can be < 0 by "
           "chance; confident = classified as one of the problem's digits with max p >= 0.9)")
    # k-NN precision / recall against held-out test images (the FID reference), at matched n
    Rpx = np.clip(Xref, 0.0, 1.0)
    Ptr_px = np.clip(Ptr, 0.0, 1.0)

    def pr_row(F, X, n):
        out = []
        for r in range(N_PR):                                    # same reference draws for every set
            g = np.random.default_rng(500 + r)
            jr = g.choice(Fref.shape[0], n, replace=False)
            js = g.choice(F.shape[0], n, replace=False) if F.shape[0] > n else np.arange(F.shape[0])
            out.append(prec_rec(F[js], Fref[jr]) + prec_rec(X[js], Rpx[jr]))
        return np.mean(out, 0)
    n_cap = Fref.shape[0] // 2
    report("")
    report(f"k-NN precision / recall (k = {PR_K}, Kynkaanniemi et al. 2019) against n held-out MNIST test images (the FID "
           f"reference), n = the set's size (at most {n_cap}), mean over {N_PR} draws; in the classifier's features (F) "
           f"and in pixel space clipped to [0, 1] (px). Recall = share of held-out real images inside the samples' k-NN "
           f"balls: it drops if the samples cover fewer shapes than real digits have (e.g. K prototypes plus noise).")
    report(f"{'set':<34}{'n':>5}{'prec F':>9}{'rec F':>8}{'prec px':>9}{'rec px':>8}")
    n_fl = min(n_typ, n_cap)
    v = pr_row(Ftr, Ptr_px, n_fl) if Ftr.shape[0] > n_fl else None
    if v is not None:
        report(f"{'real training images (target)':<34}{n_fl:>5}{v[0]:>9.3f}{v[1]:>8.3f}{v[2]:>9.3f}{v[3]:>8.3f}")
        metrics["floor"].update(prec_feat=v[0], rec_feat=v[1], prec_px=v[2], rec_px=v[3], pr_n=n_fl)
    for label, s in sets.items():
        ok = s["ok"]
        n = min(int(ok.sum()), n_cap)
        v = pr_row(s["raw"][ok], s["px"][ok], n)
        report(f"{label:<34}{n:>5}{v[0]:>9.3f}{v[1]:>8.3f}{v[2]:>9.3f}{v[3]:>8.3f}")
        metrics[label].update(prec_feat=float(v[0]), rec_feat=float(v[1]), prec_px=float(v[2]), rec_px=float(v[3]), pr_n=n)
    report("(compare rows at the same n: recall rises with n)")
    for pr in args.pair:
        a, b = pr.split(",")
        if a not in sets or b not in sets:
            report(f"pair {a},{b}: a set is missing, skipped")
            continue
        A, B = sets[a], sets[b]
        n = A["ok"].size
        if B["ok"].size != n:
            report(f"pair {a},{b}: different numbers of trajectories, skipped")
            continue
        GA, GB = A["raw"] / sc.s, B["raw"] / sc.s
        KA = (kernel(GA, GA), kernel(GA, sc.Fk))
        KB = (kernel(GB, GB), kernel(GB, sc.Fk))

        def both(j):
            ja, jb = j[A["ok"][j]], j[B["ok"][j]]
            fa, fb = sc.fid(A["raw"][ja]), sc.fid(B["raw"][jb])
            ka = kid_from_kernels(KA[0][np.ix_(ja, ja)], KA[1][ja], sc.kyy)
            kb = kid_from_kernels(KB[0][np.ix_(jb, jb)], KB[1][jb], sc.kyy)
            return fa - fb, 1e3 * (ka - kb)
        d0 = both(np.arange(n))
        bs = np.asarray([both(brng.integers(0, n, n)) for _ in range(N_BOOT)])
        out = {}
        for k, name in enumerate(("FID", "KIDx1e3")):
            lo, hi = np.percentile(bs[:, k], [2.5, 97.5])
            p = min(1.0, 2.0 * min(np.mean(bs[:, k] <= 0), np.mean(bs[:, k] >= 0)))
            report(f"paired {a} - {b}: {name} {d0[k]:+.2f} [{lo:+.2f}, {hi:+.2f}], bootstrap p = {max(p, 1.0 / N_BOOT):.3f}"
                   + (" (<)" if p == 0 else ""))
            out[name] = [float(d0[k]), float(lo), float(hi), float(p)]
        metrics[f"pair {a},{b}"] = out
    report(f"total time {time.time() - t0:.0f} s")
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(_lines) + "\n")
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
