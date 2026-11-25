#!/usr/bin/env python3
"""Check of the cifar32g problem (2026-11-25), run once after make_cifar_data.py (CPU Slurm job):

1. the files: cifar_gray32_train / test (50,000 / 10,000 x 1024 uint8), labels 5,000 / 1,000 per class, grey equals
   PIL convert('L') of the RGB files on 200 random images;
2. make_problem('cifar32g'): 32x32, 1024 units, radius 16, one class (labels all 0), data finite with mean 0 and
   std 1 (the paper's normalisation), the problem's data path and dataset name;
3. fid_score.load_mnist('test', 32, 'cifar') (the Inception reference images): 10,000 x 1024 in [0, 1];
4. the existing problems are unchanged: dataset name, geometry, classes and data path of celeba32g, celeba64g,
   fashion28_c0123456789, mnist28_d0123456789, and the sha256 of celeba32g's data file equals the one recorded in the
   locked CelebA-32 run's config.json; 'cifar32g' matches no other problem pattern and the others do not match it.

Usage (from latent_physical_SGM/): sbatch --mem 16G jobs/run_cpu_neuronic.sh check_cifar_problem.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import fid_score as fs  # noqa: E402

FAIL = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def main():
    # 1. files
    for split, n in (("train", 50000), ("test", 10000)):
        g = np.load(os.path.join(lm.CIFAR_RAW_DIR, f"cifar_gray32_{split}.npz"))
        c = np.load(os.path.join(lm.CIFAR_RAW_DIR, f"cifar_rgb32_{split}.npz"))
        X, y, R = g["images"], g["labels"], c["images"]
        check(X.shape == (n, 1024) and X.dtype == np.uint8, f"{split} grey shape {X.shape} {X.dtype}")
        check(R.shape == (n, 32, 32, 3) and R.dtype == np.uint8, f"{split} rgb shape {R.shape} {R.dtype}")
        check(np.all(np.bincount(y, minlength=10) == n // 10), f"{split} labels {np.bincount(y, minlength=10).tolist()}")
        check(np.array_equal(y, c["labels"]), f"{split} grey and rgb labels equal")
        from PIL import Image
        idx = np.random.default_rng(0).choice(n, 200, replace=False)
        ref = np.stack([np.asarray(Image.fromarray(R[i], "RGB").convert("L"), np.uint8).reshape(-1) for i in idx])
        check(np.array_equal(ref, X[idx]), f"{split} grey = PIL convert('L') of rgb (200 images)")
        f601 = 0.299 * R[idx, ..., 0] + 0.587 * R[idx, ..., 1] + 0.114 * R[idx, ..., 2]
        d = np.abs(f601.reshape(200, -1) - X[idx].astype(float)).max()
        check(d <= 1.0, f"{split} grey within 1 level of 0.299 R + 0.587 G + 0.114 B (max diff {d:.3f})")
        print(f"  {split}: grey mean {X.mean() / 255:.4f}, std {X.std() / 255:.4f}", flush=True)

    # 2. the problem
    p = lm.make_problem("cifar32g")
    D = np.asarray(p["data"])
    check(p["grid"] == 32 and p["n"] == 1024 and p["radius"] == 16, f"geometry grid {p['grid']} n {p['n']} radius {p['radius']}")
    check(D.shape == (50000, 1024) and np.isfinite(D).all(), f"data {D.shape}, finite")
    check(abs(D.mean()) < 1e-3 and abs(D.std() - 1.0) < 1e-3, f"normalised: mean {D.mean():.2e}, std {D.std():.6f}")
    check(np.all(p["labels"] == 0) and lm.problem_classes("cifar32g") == 1, "one class, labels all 0")
    check(lm.problem_dataset("cifar32g") == "cifar", f"dataset {lm.problem_dataset('cifar32g')}")
    check(lm.problem_data_path("cifar32g").endswith("data/CIFAR10/cifar_gray32_train.npz"),
          f"data path {lm.problem_data_path('cifar32g')}")
    print(f"  pixel mean {p['mean']:.5f}, std {p['std']:.5f} (pixel units incl. the 0.01 noise)", flush=True)

    # 3. Inception reference images
    Xte, yte = fs.load_mnist("test", 32, "cifar")
    check(Xte.shape == (10000, 1024) and Xte.min() >= 0 and Xte.max() <= 1, f"test images {Xte.shape}, [{Xte.min()}, {Xte.max()}]")

    # 4. existing problems unchanged
    want = {"celeba32g": ("celeba", (32, 16), 1, "data/CelebA/celeba_gray32_train.npz"),
            "celeba64g": ("celeba64", (64, 32), 1, "data/CelebA/celeba_gray64_train.npz"),
            "fashion28_c0123456789": ("fashion", (28, 14), 10, "fashion_0123456789_28.npz"),
            "mnist28_d0123456789": ("mnist", (28, 14), 10, "digits_0123456789_28.npz")}
    for name, (ds, geo, ncls, path) in want.items():
        got = (lm.problem_dataset(name), tuple(lm.problem_geometry(name)), lm.problem_classes(name), lm.problem_data_path(name))
        check(got[:3] == (ds, geo, ncls) and got[3].endswith(path), f"{name} unchanged: {got[:3]}, {os.path.basename(got[3])}")
    check(lm._CIFAR_PROBLEM.fullmatch("cifar32g") is not None and lm._CELEBA_PROBLEM.fullmatch("cifar32g") is None
          and lm._FASHION_PROBLEM.fullmatch("cifar32g") is None and lm._DIGIT_PROBLEM.fullmatch("cifar32g") is None
          and all(lm._CIFAR_PROBLEM.fullmatch(n) is None for n in want), "problem patterns do not overlap")
    run = os.path.join(lm.OUT_ROOT, "celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1")
    rec = json.load(open(os.path.join(run, "config.json")))["data_sha256"]
    check(lm.sha256_file(lm.problem_data_path("celeba32g")) == rec, "celeba32g data file sha256 = the locked run's record")

    print(f"check_cifar_problem: {'ALL PASSED' if not FAIL else str(len(FAIL)) + ' FAILED'}", flush=True)
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
