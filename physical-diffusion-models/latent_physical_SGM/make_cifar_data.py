#!/usr/bin/env python3
"""CIFAR-10, 32x32 grey, for the latent physical SGM (2026-11-25; after the meeting with Dmitry Krotov: grey CIFAR-10
with the locked recipe first, colour later).

Source: the official binary version, https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz (Krizhevsky 2009),
downloaded once to physical-diffusion-models/data/CIFAR10/download/. The archive is read in memory with tarfile, never
extracted to disk, and no pickle is used (the binary records are parsed with numpy): nothing from the download is
executed. Each record is 1 label byte + 3072 pixel bytes (1024 R, 1024 G, 1024 B, each row-major 32x32).
Checks: the archive's md5 against the published value (a mismatch is reported), five train batches + one test batch
of 10,000 records each, 5,000 / 1,000 images per class.

Grey conversion: PIL convert('L') of the RGB image (ITU-R 601 luma, L = 0.299 R + 0.587 G + 0.114 B, as for CelebA),
8 bit. No resizing, no crop (the images are 32x32 already). The problem uses no labels (one class, as CelebA); the
true labels are kept in the files for later use.

Writes data/CIFAR10/cifar_gray32_<split>.npz (images (n, 1024) uint8 row-major, labels (n,) int64 0-9) and
cifar_rgb32_<split>.npz (images (n, 32, 32, 3) uint8, labels; for the colour step later), for split train (50,000)
and test (10,000); a preview data/CIFAR10/preview_gray32.png (first 100 train images, grey) and preview_rgb32.png;
data/CIFAR10/cifar_gray32_<job>.json (counts, md5 / sha256 of the archive and outputs).

Usage (CPU Slurm job, from latent_physical_SGM/): sbatch jobs/run_cpu_neuronic.sh make_cifar_data.py
"""
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"        # published md5 of cifar-10-binary.tar.gz
MEMBERS = {"train": [f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)],
           "test": ["cifar-10-batches-bin/test_batch.bin"]}
RECORD = 1 + 3 * 1024
PER_BATCH = 10000
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def download(dl_dir):
    os.makedirs(dl_dir, exist_ok=True)
    p = os.path.join(dl_dir, os.path.basename(URL))
    if not os.path.exists(p):
        req = urllib.request.Request(URL, headers={"User-Agent": "python-urllib"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r, open(p + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(p + ".part", p)
        print(f"downloaded {p} ({os.path.getsize(p) / 1e6:.1f} MB, {time.time() - t0:.0f} s)", flush=True)
    return p


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_split(tar_path, split):
    """(rgb (n, 32, 32, 3) uint8, labels (n,) int64) from the archive's batch files, read in memory."""
    rgb, lab = [], []
    with tarfile.open(tar_path, "r:gz") as tf:
        for name in MEMBERS[split]:
            m = tf.getmember(name)
            if not m.isfile() or m.size != RECORD * PER_BATCH:
                raise RuntimeError(f"{name}: not a regular file of {RECORD * PER_BATCH} bytes ({m.size})")
            buf = np.frombuffer(tf.extractfile(m).read(), np.uint8).reshape(PER_BATCH, RECORD)
            lab.append(buf[:, 0].astype(np.int64))
            rgb.append(buf[:, 1:].reshape(PER_BATCH, 3, 32, 32).transpose(0, 2, 3, 1))
    rgb, lab = np.concatenate(rgb), np.concatenate(lab)
    counts = np.bincount(lab, minlength=10)
    want = len(MEMBERS[split]) * PER_BATCH // 10
    if lab.max() > 9 or not np.all(counts == want):
        raise RuntimeError(f"{split}: class counts {counts.tolist()}, expected {want} each")
    return np.ascontiguousarray(rgb), lab


def to_grey(rgb):
    """PIL convert('L') of every image (as make_celeba_data.py), (n, 1024) uint8."""
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(im, "RGB").convert("L"), np.uint8).reshape(-1) for im in rgb])


def preview(grey, rgb, raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tiles = grey[:100].reshape(10, 10, 32, 32).transpose(0, 2, 1, 3).reshape(320, 320)
    plt.imsave(os.path.join(raw, "preview_gray32.png"), tiles, cmap="gray", vmin=0, vmax=255)
    tiles = rgb[:100].reshape(10, 10, 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(320, 320, 3)
    plt.imsave(os.path.join(raw, "preview_rgb32.png"), tiles)


def main():
    raw = lm.CIFAR_RAW_DIR
    tar_path = download(os.path.join(raw, "download"))
    md5 = md5_file(tar_path)
    if md5 != MD5:
        print(f"WARNING: md5 {md5} differs from the published {MD5}; the record and class-count checks below still "
              f"apply", flush=True)
    summary = {"source": URL, "md5": md5, "md5_published": MD5, "sha256": lm.sha256_file(tar_path),
               "grey": "PIL convert('L')", "classes": CLASSES, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "slurm_job": os.environ.get("SLURM_JOB_ID", "manual"), "files": {}}
    for split in ("train", "test"):
        rgb, lab = read_split(tar_path, split)
        grey = to_grey(rgb)
        for kind, X in (("gray", grey), ("rgb", rgb)):
            path = os.path.join(raw, f"cifar_{kind}32_{split}.npz")
            np.savez(path, images=X, labels=lab)
            summary["files"][path] = {"n": int(X.shape[0]), "shape": list(X.shape), "mean": float(X.mean() / 255.0),
                                      "sha256": lm.sha256_file(path)}
            print(f"wrote {path}: {X.shape}, mean {X.mean() / 255:.4f}", flush=True)
        if split == "train":
            preview(grey, rgb, raw)
    with open(os.path.join(raw, f"cifar_gray32_{summary['slurm_job']}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["files"], indent=1), flush=True)


if __name__ == "__main__":
    main()
