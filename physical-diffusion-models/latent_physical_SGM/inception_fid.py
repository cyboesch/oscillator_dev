#!/usr/bin/env python3
"""Standard FID (Inception-v3 pool_3 features, Heusel et al. 2017) of generated MNIST / Fashion-MNIST images, for
comparison with published numbers (2026-11-15). fid_score.py stays the project's main metric (a small classifier's
features, cheap, sensitive to stroke shape); this script gives the number papers report.

Network: the original TTUR graph (inception-2015-12-05.tgz, classify_image_graph_def.pb; sha256 below), downloaded once
to out/latent_physical_SGM/fid/inception/. pytorch-fid and clean-fid reproduce these weights. The graph is imported
with its input 'ExpandDims:0' remapped to a placeholder of free batch size, and features are read from 'pool_3:0'
(2048-d). Images go in as TTUR feeds them: grey 28x28 (CelebA, CIFAR-10 grey: 32x32) replicated to 3 channels, values 0..255 (float), and the graph's own
ResizeBilinear to 299x299 and (x - 128) / 128. Generated samples are mapped to pixel units (x * std + mean of the
problem), clipped to [0, 1] and quantised to 8 bit (as a saved PNG would be); rows that diverged (max |x| > 15 in
normalised units or not finite, as fid_score.py) are left out and counted.

Reference statistics: the dataset's full TRAIN split (60,000; the usual convention in papers) and TEST split (10,000),
raw 8-bit images, no noise; cached as out/latent_physical_SGM/fid/inception/stats_<dataset>_<split>.npz.
Reported per set: FID against train and against test at the set's n; FID_inf (Chong & Forsyth 2020: FID at n draws
for several n, least-squares fit FID(n) = FID_inf + b / n; the n -> infinity value, comparable with 50k-sample numbers
in papers); KID x 1000 against test (Binkowski et al. 2018: cubic kernel (x.y / 2048 + 1)^3, unbiased, mean over
subsets of 1000). Floor: the same for random real TEST images at the same n (held-out real data: the best a model can
get at that n).

Usage (GPU Slurm job): python inception_fid.py --problem mnist28_d0123456789 LABEL=PATH ...
                       python inception_fid.py --smoke        (GPU check, batch independence, speed)
Writes out/latent_physical_SGM/fid_inception/<job>/report.txt and metrics.json.
"""
import argparse
import json
import math
import os
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"                     # this process needs no JAX on the GPU; TensorFlow gets it
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import evaluate_samples as es  # noqa: E402
import fid_score as fs  # noqa: E402

INC_DIR = os.path.join(lm.OUT_ROOT, "fid", "inception")
GRAPH = os.path.join(INC_DIR, "classify_image_graph_def.pb")
GRAPH_SHA256 = "009d6814d1bc560d4e7b236e170e9b2d5ca6f4b57bd8037f6db05776204415c6"
OUT = os.path.join(lm.OUT_ROOT, "fid_inception", os.environ.get("SLURM_JOB_ID", "manual"))
BATCH = 250
N_FLOOR = 10
N_GRID = (400, 600, 800, 1000, 1400, 2000)     # FID_inf fit; sizes above the set's n are dropped
N_GRID_DRAWS = 3
KID_SUB, KID_DRAWS = 1000, 20
N_BOOT = 10
FAINT = 0.1             # diagnostics: pixels in (0, FAINT) count as faint; "faint zeroed" sets them to 0
NOISE_FLOORS = (0.01, 0.02)   # diagnostics: real test images + N(0, sigma^2) pixel noise
_lines = []


def report(s=""):
    print(s, flush=True)
    _lines.append(s)


def make_inception():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)     # float32 convolutions, as the original
    if lm.sha256_file(GRAPH) != GRAPH_SHA256:
        raise SystemExit(f"{GRAPH}: sha256 mismatch")
    gd = tf.compat.v1.GraphDef()
    with open(GRAPH, "rb") as f:
        gd.ParseFromString(f.read())
    g = tf.Graph()
    with g.as_default():
        x = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name="x")
        (pool,) = tf.import_graph_def(gd, input_map={"ExpandDims:0": x}, return_elements=["pool_3:0"], name="inc")
    sess = tf.compat.v1.Session(graph=g)

    def feats(P, batch=BATCH):
        """P (n, 784) pixel values; clipped to [0, 1], 8-bit, 3 channels, 0..255 -> (n, 2048) float64."""
        P = np.asarray(P, np.float64)
        G = math.isqrt(P.shape[1])
        img = np.round(np.clip(P, 0.0, 1.0) * 255.0).astype(np.float32).reshape(-1, G, G, 1)
        out = []
        for i in range(0, img.shape[0], batch):
            b = np.repeat(img[i:i + batch], 3, axis=3)
            out.append(sess.run(pool, {x: b}).reshape(b.shape[0], -1))
        return np.concatenate(out).astype(np.float64)
    return feats, [d.name for d in gpus]


def ref_stats(feats, dataset, split, grid=28):
    path = os.path.join(INC_DIR, f"stats_{dataset}_{split}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["mu"], z["sigma"], int(z["n"])
    X, _ = fs.load_mnist(split, grid, dataset)
    F = feats(X)
    mu, sigma = fs.stats(F)
    np.savez(path, mu=mu, sigma=sigma, n=F.shape[0])
    if split == "test":
        np.save(os.path.join(INC_DIR, f"feats_{dataset}_test.npy"), F.astype(np.float32))
    return mu, sigma, F.shape[0]


def kid(F, R, rng):
    d = F.shape[1]
    vals = []
    for _ in range(KID_DRAWS):
        m = min(KID_SUB, F.shape[0], R.shape[0])
        A, B = F[rng.choice(F.shape[0], m, replace=False)], R[rng.choice(R.shape[0], m, replace=False)]
        kxx, kyy, kxy = (A @ A.T / d + 1) ** 3, (B @ B.T / d + 1) ** 3, (A @ B.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return 1e3 * float(np.mean(vals))


def fid_vs(F, mu, sigma):
    m, C = fs.stats(F)
    return fs.fid_from_stats(m, C, mu, sigma)


def fid_inf(F, mu, sigma, rng):
    ns = [n for n in N_GRID if n <= F.shape[0]]
    pts = []
    for n in ns:
        draws = 1 if n == F.shape[0] else N_GRID_DRAWS
        pts.append(np.mean([fid_vs(F[rng.choice(F.shape[0], n, replace=False)], mu, sigma) for _ in range(draws)]))
    A = np.stack([np.ones(len(ns)), 1.0 / np.asarray(ns, float)], 1)
    coef = np.linalg.lstsq(A, np.asarray(pts), rcond=None)[0]
    return float(coef[0]), dict(zip(ns, map(float, pts)))


def smoke():
    t0 = time.time()
    feats, gpus = make_inception()
    print(f"GPUs seen by TensorFlow: {gpus}; graph loaded in {time.time() - t0:.1f} s", flush=True)
    X, _ = fs.load_mnist("test", 28, "mnist")
    F1 = np.concatenate([feats(X[i:i + 1], batch=1) for i in range(8)])
    F8 = feats(X[:8], batch=8)
    print(f"batch independence: max |F(batch 1) - F(batch 8)| = {np.max(np.abs(F1 - F8)):.2e} "
          f"(feature rms {np.sqrt(np.mean(F8 ** 2)):.3f}); features >= 0: {bool(np.all(F8 >= 0))}", flush=True)
    t1 = time.time()
    F = feats(X[:2000])
    dt = time.time() - t1
    print(f"2000 images in {dt:.1f} s ({2000 / dt:.0f} img/s)", flush=True)
    m1, C1 = fs.stats(F[:1000])
    m2, C2 = fs.stats(F[1000:2000])
    print(f"FID(test[:1000], test[1000:2000]) = {fs.fid_from_stats(m1, C1, m2, C2):.2f} (real vs real, n = 1000)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--problem")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    if not args.problem or not args.sets:
        raise SystemExit("--problem and at least one LABEL=PATH are needed")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; inception_fid; job {os.environ.get('SLURM_JOB_ID', 'manual')}")
    prob = lm.make_problem(args.problem)
    grid = prob["grid"]
    if grid not in (28, 32, 64):
        raise SystemExit("Inception FID is set up for 28x28 (MNIST, Fashion) and 32x32 / 64x64 (CelebA, CIFAR grey) problems")
    mean, std = prob["mean"], prob["std"]
    dataset = lm.problem_dataset(args.problem)
    feats, gpus = make_inception()
    mu_tr, S_tr, n_tr = ref_stats(feats, dataset, "train", grid)
    mu_te, S_te, n_te = ref_stats(feats, dataset, "test", grid)
    Fte = np.load(os.path.join(INC_DIR, f"feats_{dataset}_test.npy")).astype(np.float64)
    report(f"problem {args.problem} ({dataset}); Inception-v3 pool_3 (TTUR graph, sha256 {GRAPH_SHA256[:12]}...); "
           f"TensorFlow GPUs {gpus}; reference statistics: {dataset} train ({n_tr}) and test ({n_te}), raw 8-bit images")
    Xte, _ = fs.load_mnist("test", grid, dataset)
    metrics = {}

    def faint_pct(P):
        q = np.round(np.clip(P, 0.0, 1.0) * 255.0)
        return 100.0 * float(np.mean((q > 0) & (q < FAINT * 255.0)))

    def flush():
        with open(os.path.join(OUT, "report.txt"), "w") as f:
            f.write("\n".join(_lines) + "\n")
        with open(os.path.join(OUT, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)

    rng = np.random.default_rng(0)

    def row(label, F, P, n_div=0, full=True):
        f_tr, f_te = fid_vs(F, mu_tr, S_tr), fid_vs(F, mu_te, S_te)
        f_inf, curve = fid_inf(F, mu_tr, S_tr, rng) if full else (float("nan"), {})
        k = kid(F, Fte, rng)
        boot = [fid_vs(F[rng.integers(0, F.shape[0], F.shape[0])], mu_tr, S_tr) for _ in range(N_BOOT)] if full else [np.nan]
        fp = faint_pct(P)
        report(f"{label:<56}{F.shape[0]:>6}{n_div:>5}{f_tr:>11.2f}{np.std(boot):>7.2f}{f_te:>10.2f}{f_inf:>9.2f}"
               f"{k:>9.2f}{fp:>8.1f}")
        metrics[label] = {"n": F.shape[0], "diverged": n_div, "fid_train": f_tr, "fid_train_boot_sd": float(np.std(boot)),
                          "fid_test": f_te, "fid_inf": f_inf, "fid_curve": curve, "kid_test_x1e3": k, "faint_pct": fp}
        flush()

    report("")
    report(f"{'set':<56}{'n':>6}{'div':>5}{'FID train':>11}{'(sd)':>7}{'FID test':>10}{'FID_inf':>9}{'KIDx1e3':>9}"
           f"{'faint%':>8}")
    n_typ = 2000
    fl = [fid_vs(Fte[np.random.default_rng(100 + r).choice(Fte.shape[0], n_typ, replace=False)], mu_tr, S_tr)
          for r in range(N_FLOOR)]
    fl_inf, _ = fid_inf(Fte, mu_tr, S_tr, rng)
    metrics["floor"] = {"n": n_typ, "fid_train_mean": float(np.mean(fl)), "fid_train_sd": float(np.std(fl)),
                        "fid_train_all_test": fid_vs(Fte, mu_tr, S_tr), "fid_inf": fl_inf, "faint_pct": faint_pct(Xte)}
    report(f"{'real test images (floor, ' + str(N_FLOOR) + ' draws of n)':<56}{n_typ:>6}{'':>5}{np.mean(fl):>11.2f}"
           f"{np.std(fl):>7.2f}{'':>10}{fl_inf:>9.2f}{'':>9}{faint_pct(Xte):>8.1f}")
    report(f"{'real test images (all)':<56}{Fte.shape[0]:>6}{'':>5}{metrics['floor']['fid_train_all_test']:>11.2f}")
    j = np.random.default_rng(1).choice(Xte.shape[0], n_typ, replace=False)
    for sig in NOISE_FLOORS:          # diagnostic: what faint pixel noise alone costs
        P = Xte[j] + sig * np.random.default_rng(2).standard_normal((n_typ, Xte.shape[1]))
        row(f"diag: real test + N(0, {sig}^2) noise", feats(P), P, full=False)
    for spec in args.sets:
        label, path = spec.split("=", 1)
        missing = [p for p in lm.split_set_spec(path) if not os.path.exists(p if os.path.isabs(p) else os.path.join(lm.OUT_ROOT, p))]
        if missing:
            report(f"{label}: skipped, no samples ({', '.join(missing)})")
            continue
        x = np.asarray(es.load_set(path), np.float64)
        ok = np.all(np.isfinite(x), 1)
        ok[ok] = np.max(np.abs(x[ok]), 1) <= fs.DIV_ABS
        P = x[ok] * std + mean
        row(label, feats(P), P, int((~ok).sum()))
        Pz = np.where(P < FAINT, 0.0, P)  # diagnostic: faint pixels set to 0 (not the model's number)
        row(f"  diag: same, faint (<{FAINT}) zeroed", feats(Pz), Pz, full=False)
        metrics[f"{label} [faint zeroed]"] = metrics.pop(f"  diag: same, faint (<{FAINT}) zeroed")  # unique key
        flush()
    report("(Inception-v3 pool_3 FID, 8-bit clipped images; 'FID train' = against the 60k train statistics at the set's n, "
           f"sd = bootstrap over the set ({N_BOOT}); FID_inf = n -> infinity extrapolation, the number to compare with "
           "papers' 50k-sample FIDs; KID x 1000 against the test images, subsets of 1000; faint% = share of pixels with "
           f"0 < value < {FAINT}; 'diag' rows are diagnostics, not model numbers)")
    report(f"total time {time.time() - t0:.0f} s")
    flush()


if __name__ == "__main__":
    main()
