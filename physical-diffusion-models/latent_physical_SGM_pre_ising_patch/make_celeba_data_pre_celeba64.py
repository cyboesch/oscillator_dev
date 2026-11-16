#!/usr/bin/env python3
"""CelebA, 32x32 grey, for the latent physical SGM (2026-11-15; the user: "lets try CelebA 32x32, gray").

Source: the Hugging Face parquet export of CelebA 'img_align+identity+attr' (dataset flwrlabs/celeba; the official
aligned 178x218 JPEGs, with the official train / valid / test partition of list_eval_partition.txt: 162,770 / 19,867 /
19,962 images). Downloaded once to physical-diffusion-models/data/CelebA/parquet/ (if a shard is missing).

Preprocessing (the usual one for small CelebA, e.g. DDPM's 64x64 CelebA): centre crop 140x140 (box (19, 39, 159, 179)
of the 178x218 aligned image), PIL convert('L') (ITU-R 601 luma, L = 0.299 R + 0.587 G + 0.114 B), resize to 32x32
with PIL's antialiased bicubic filter, 8 bit. No labels are used (the problem has one class).

Writes data/CelebA/celeba_gray32_<split>.npz: images (n, 1024) uint8 (row-major 32x32), index (n,) int64 (row order
of the parquet shards), and a preview data/CelebA/preview_gray32.png (the first 100 train images) plus
data/CelebA/celeba_gray32_<job>.json (counts, sha256 of every shard and output).

Usage (CPU Slurm job): python make_celeba_data.py
"""
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

URL = "https://huggingface.co/api/datasets/flwrlabs/celeba/parquet/img_align+identity+attr/{split}/{i}.parquet"
SHARDS = {"train": 19, "valid": 3, "test": 3}
EXPECTED = {"train": 162770, "valid": 19867, "test": 19962}
CROP = (19, 39, 159, 179)          # (left, upper, right, lower): the centre 140x140 of 178x218
SIZE = 32


def download(pq_dir):
    os.makedirs(pq_dir, exist_ok=True)
    paths = {}
    for split, n in SHARDS.items():
        paths[split] = []
        for i in range(n):
            p = os.path.join(pq_dir, f"{split}_{i:02d}.parquet")
            if not os.path.exists(p):
                req = urllib.request.Request(URL.format(split=split, i=i), headers={"User-Agent": "python-urllib"})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=600) as r, open(p + ".part", "wb") as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                os.replace(p + ".part", p)
                print(f"downloaded {p} ({os.path.getsize(p) / 1e6:.1f} MB, {time.time() - t0:.0f} s)", flush=True)
            paths[split].append(p)
    return paths


def image_column(table):
    """The name of the column holding the encoded images (a struct with a 'bytes' field)."""
    import pyarrow as pa
    for field in table.schema:
        if pa.types.is_struct(field.type) and any(c.name == "bytes" for c in field.type):
            return field.name
    raise RuntimeError(f"no image column in {table.schema}")


def convert_shard(p):
    """One parquet shard -> (n, 1024) uint8 (in row order)."""
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(p)
    col = image_column(t)
    out = []
    for k, rec in enumerate(t.column(col).to_pylist()):
        im = Image.open(io.BytesIO(rec["bytes"]))
        if im.size != (178, 218):
            raise RuntimeError(f"{p}: image {k} has size {im.size}, expected the aligned 178x218")
        g = im.convert("L").crop(CROP).resize((SIZE, SIZE), Image.BICUBIC)
        out.append(np.asarray(g, np.uint8).reshape(-1))
    print(f"  {p}: {t.num_rows} images", flush=True)
    return np.stack(out)


def convert(paths):
    """All shards of a split, in shard order; shards in parallel over the job's CPUs."""
    from multiprocessing import Pool
    n_proc = max(1, min(len(paths), int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    with Pool(n_proc) as pool:
        parts = pool.map(convert_shard, paths)
    return np.concatenate(parts)


def preview(X, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tiles = X[:100].reshape(10, 10, SIZE, SIZE).transpose(0, 2, 1, 3).reshape(10 * SIZE, 10 * SIZE)
    plt.imsave(path, tiles, cmap="gray", vmin=0, vmax=255)


def main():
    raw = lm.CELEBA_RAW_DIR
    paths = download(os.path.join(raw, "parquet"))
    summary = {"source": URL, "crop": CROP, "size": SIZE, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "slurm_job": os.environ.get("SLURM_JOB_ID", "manual"), "shards": {}, "files": {}}
    for split, ps in paths.items():
        for p in ps:
            summary["shards"][os.path.basename(p)] = lm.sha256_file(p)
        X = convert(ps)
        if X.shape[0] != EXPECTED[split]:
            print(f"WARNING: {split} has {X.shape[0]} images, expected {EXPECTED[split]}", flush=True)
        path = os.path.join(raw, f"celeba_gray32_{split}.npz")
        np.savez(path, images=X, index=np.arange(X.shape[0], dtype=np.int64))
        summary["files"][path] = {"n": int(X.shape[0]), "mean": float(X.mean() / 255.0), "sha256": lm.sha256_file(path)}
        print(f"wrote {path}: {X.shape}, mean {X.mean() / 255:.4f}", flush=True)
        if split == "train":
            preview(X, os.path.join(raw, "preview_gray32.png"))
    with open(os.path.join(raw, f"celeba_gray32_{summary['slurm_job']}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["files"], indent=1), flush=True)


if __name__ == "__main__":
    main()
