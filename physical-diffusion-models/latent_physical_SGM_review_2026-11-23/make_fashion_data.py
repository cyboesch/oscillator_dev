#!/usr/bin/env python3
"""Fashion-MNIST class subsets for the latent physical SGM (2026-11-14), in the format of make_digit_data.py.

Reads the raw Zalando idx files in physical-diffusion-models/data/FashionMNIST (latent_model.load_fashion_raw; the
official files, md5 train-images 8d4fb7e6..., train-labels 25c81989..., t10k-images bef4ecab..., t10k-labels
bb300cfd...), takes the train split, /255 (28x28 as given), balances the requested classes by subsampling with numpy
default_rng(42) (Fashion-MNIST is balanced already: 6,000 per class) and shuffles with the same generator, as the digit
subsets. Writes out/latent_physical_SGM/data/fashion_<c1><c2>..._28.npz: images (n, 784) float32 in [0, 1] and labels
(n,) int64 (the Fashion-MNIST class 0-9: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag,
Ankle boot), plus a JSON summary out/latent_physical_SGM/data/fashion_<job>.json. The problem is then
fashion28_c<c1><c2>... (latent_model.make_problem).

Usage (CPU Slurm job): python make_fashion_data.py --classes 0123456789 [--classes ...]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", action="append", required=True, help="class ids as digits, e.g. 0123456789")
    args = ap.parse_args()
    X, y = lm.load_fashion_raw("train")
    Xt, yt = lm.load_fashion_raw("test")
    summary = {"raw_dir": lm.FASHION_RAW_DIR, "train": int(X.shape[0]), "test": int(Xt.shape[0]),
               "train_counts": np.bincount(y, minlength=10).tolist(), "test_counts": np.bincount(yt, minlength=10).tolist(),
               "pixel_mean_train": float(X.mean()), "files": {}, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "slurm_job": os.environ.get("SLURM_JOB_ID", "manual")}
    os.makedirs(lm.DIGITS_DIR, exist_ok=True)
    for spec in args.classes:
        cls = sorted(int(c) for c in spec)
        if len(set(cls)) != len(cls) or len(cls) < 2:
            raise SystemExit(f"--classes {spec}: needs at least two distinct class ids")
        rng = np.random.default_rng(42)
        nmin = min(int(np.sum(y == c)) for c in cls)
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), nmin, replace=False) for c in cls])
        rng.shuffle(idx)
        path = os.path.join(lm.DIGITS_DIR, f"fashion_{''.join(map(str, cls))}_28.npz")
        np.savez(path, images=X[idx].astype(np.float32), labels=y[idx].astype(np.int64))
        summary["files"][path] = {"n": int(idx.size), "per_class": nmin, "classes": cls,
                                  "names": [lm.FASHION_CLASSES[c] for c in cls], "sha256": lm.sha256_file(path)}
        print(f"wrote {path}: {idx.size} images ({nmin} per class)", flush=True)
    out = os.path.join(lm.DIGITS_DIR, f"fashion_{os.environ.get('SLURM_JOB_ID', 'manual')}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
