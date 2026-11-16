#!/usr/bin/env python3
"""MNIST digit subsets for the hidden-unit study, preprocessed exactly like the paper's scripts/preprocess-mnist.py
(train split of tensorflow_datasets 'mnist', /255, linear resize to 28x28, classes balanced by subsampling with
numpy default_rng(42), then shuffled with the same generator), and saved WITH their labels.

Usage (CPU Slurm job; tfds reads its cache in ~/tensorflow_datasets):
    python make_digit_data.py --digits 1,4 --digits 4,9 ... [--verify]
Writes out/latent_physical_SGM/data/digits_<d1><d2>..._28.npz: images (n, 784) float32 in [0, 1] (the paper's file
format) and labels (n,) int64 (the MNIST digit), plus a JSON summary in out/latent_physical_SGM/data/<job>.json.
--verify: for the digit sets the paper's data folder already holds (0,1 / 0,1,2 / 0,1,2,3, balanced, 28x28), the
images must equal that file exactly, row by row; it also checks the 0/1 labels against the ones-only file.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT = os.path.join(PKG, "out", "latent_physical_SGM", "data")
PAPER_FILES = {(0, 1): "mnist_labels_[0, 1]_resolution_(28, 28)_balanced_True.npy",
               (0, 1, 2): "mnist_labels_[0, 1, 2]_resolution_(28, 28)_balanced_True.npy",
               (0, 1, 2, 3): "mnist_labels_[0, 1, 2, 3]_resolution_(28, 28)_balanced_True.npy"}
ONES_PATH = os.path.join(PKG, "data", "MNIST", "mnist_labels_[1]_resolution_(28, 28).npy")


def load_and_preprocess(digits, resolution=(28, 28), step=1):
    """scripts/preprocess-mnist.py: load_and_preprocess_mnist, verbatim in effect."""
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    import tensorflow_datasets as tfds

    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    ds = ds_builder.as_dataset(split="train", as_supervised=True)
    ds = ds.filter(lambda _, label: tf.reduce_any([tf.equal(label, l) for l in digits]))
    ds = ds.enumerate().filter(lambda idx, _: idx % step == 0).map(lambda _, x: x)
    images, labels = [], []
    for image, label in tfds.as_numpy(ds):
        image = image.astype(np.float32) / 255.0
        resized = jax.image.resize(jnp.array(image), shape=(*resolution, image.shape[-1]), method="linear")
        images.append(resized)
        labels.append(label)
    return jnp.stack(images), jnp.array(labels)


def balance_classes(images, labels_array, seed=42):
    """scripts/preprocess-mnist.py: balance_classes, verbatim in effect."""
    import jax.numpy as jnp

    unique_labels = np.unique(labels_array)
    counts = {int(label): int(np.sum(labels_array == label)) for label in unique_labels}
    min_count = min(counts.values())
    rng = np.random.default_rng(seed)
    bi, bl = [], []
    for label in unique_labels:
        indices = np.where(labels_array == label)[0]
        sel = rng.choice(indices, size=min_count, replace=False)
        bi.append(images[sel])
        bl.append(labels_array[sel])
    bi = jnp.concatenate(bi, axis=0)
    bl = jnp.concatenate(bl, axis=0)
    perm = rng.permutation(len(bl))
    return bi[perm], bl[perm], counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", action="append", required=True, help="comma-separated digits, e.g. 1,4 (repeatable)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "manual")
    summary = {"job": job, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "sets": {}}
    sets = [tuple(sorted(int(d) for d in s.split(","))) for s in args.digits]
    if args.verify:
        sets += [s for s in PAPER_FILES if s not in sets]
    for digits in sets:
        t0 = time.time()
        images, labels = load_and_preprocess(list(digits))
        images, labels, counts = balance_classes(images, labels)
        X = np.asarray(images.reshape(images.shape[0], -1), dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        tag = "".join(str(d) for d in digits)
        path = os.path.join(OUT, f"digits_{tag}_28.npz")
        np.savez(path, images=X, labels=y)
        rec = {"path": path, "n": int(X.shape[0]), "counts_before": counts,
               "per_class": {int(d): int(np.sum(y == d)) for d in digits}, "seconds": round(time.time() - t0, 1)}
        if digits in PAPER_FILES:
            ref = np.load(os.path.join(PKG, "data", "MNIST", PAPER_FILES[digits]))
            rec["paper_file"] = PAPER_FILES[digits]
            rec["identical_to_paper_file"] = bool(ref.shape == X.shape and ref.dtype == X.dtype and np.array_equal(ref, X))
            if digits == (0, 1):
                ones = np.load(ONES_PATH)
                ones_set = {r.tobytes() for r in np.round(ones * 255).astype(np.uint8)}
                lab1 = np.array([r.tobytes() in ones_set for r in np.round(X * 255).astype(np.uint8)])
                rec["labels_match_ones_file"] = bool(np.array_equal(lab1, y == 1))
        print(json.dumps({"digits": digits, **rec}), flush=True)
        summary["sets"][tag] = rec
    with open(os.path.join(OUT, f"{job}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    bad = [t for t, r in summary["sets"].items()
           if r.get("identical_to_paper_file") is False or r.get("labels_match_ones_file") is False]
    if bad:
        print(f"VERIFY FAILED for {bad}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
