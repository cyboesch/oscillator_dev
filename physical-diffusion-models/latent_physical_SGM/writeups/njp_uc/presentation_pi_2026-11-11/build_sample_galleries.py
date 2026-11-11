#!/usr/bin/env python3
"""Render first-100 galleries from two completed saved sample sets; no sampling.

Uses only NumPy/Matplotlib. All checkpoint and sample access is read-only.
Outputs PDF galleries and their provenance beside this script in assets/.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RUNS = ROOT / "physical-diffusion-models/out/latent_physical_SGM"
MODELS = {
    "potts": ("mnist28_d0123456789_K1536_ema_potts_quad_local_quartic_r1_lin_pooled", 1536),
    "ising": ("mnist28_d0123456789_K784_ema_ising_quad_local_quartic_r1_lin", 784),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main():
    output = HERE / "assets"
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    shared_normalization = None
    for family, (name, count) in MODELS.items():
        run = RUNS / name
        if not (run / "DONE").is_file():
            raise ValueError(f"Run is incomplete: {run}")
        paths = [run / "config.json", run / "train_log.txt", run / "params_history.npy",
                 run / "samples/sde_base_n2000.npy", run / "samples/sde_base_n2000.json"]
        before = {str(p.relative_to(run)): signature(p) for p in paths}
        cfg = json.loads(paths[0].read_text())
        expected = {"hidden": family, "K": count, "problem": "mnist28_d0123456789",
                    "local": True, "quad": True, "radius": 1, "neighbourhood": "chebyshev",
                    "pair_terms": "lin", "visible": "quartic", "hidden_rule": "exact"}
        if any(cfg.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Unexpected model configuration: {name}")
        match = re.search(r"normalisation mean ([+\-\d.eE]+) std ([+\-\d.eE]+)", paths[1].read_text())
        if match is None:
            raise ValueError(f"Missing normalization: {name}")
        normalization = tuple(map(float, match.groups()))
        if shared_normalization is None:
            shared_normalization = normalization
        if normalization != shared_normalization:
            raise ValueError("Models do not share the same saved display normalization")
        mean, std = normalization
        manifest = json.loads(paths[4].read_text())
        params_hash = digest(paths[2])
        if manifest.get("params_sha256") != params_hash:
            raise ValueError(f"Sample/checkpoint mismatch: {name}")
        if manifest.get("n") != 2000 or manifest.get("set") != "base" or manifest.get("sampler") != "sde":
            raise ValueError(f"Unexpected sample manifest: {name}")
        samples = np.load(paths[3], mmap_mode="r", allow_pickle=False)
        if samples.shape != (2000, 784):
            raise ValueError(f"Unexpected sample shape: {samples.shape}")
        selected = np.asarray(samples[:100])
        if not np.isfinite(selected).all():
            raise ValueError("Selected samples contain nonfinite values; none have been excluded")
        pixels = np.clip(selected * std + mean, 0.0, 1.0)
        mosaic = np.full((291, 291), .20)
        for index, sample in enumerate(pixels):
            row, col = divmod(index, 10)
            y, x = 1 + row * 29, 1 + col * 29
            mosaic[y:y + 28, x:x + 28] = sample.reshape(28, 28)
        fig = plt.figure(figsize=(6, 6), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(mosaic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        filename = output / f"sample_{family}.pdf"
        fig.savefig(filename, metadata={"Title": f"First 100 base-2000 samples: {family} {count}"})
        plt.close(fig)
        after = {str(p.relative_to(run)): signature(p) for p in paths}
        if after != before:
            raise RuntimeError("Source files changed during the read-only gallery rendering")
        records[family] = {
            "run": name, "configuration": expected, "sample_manifest": manifest,
            "sample_path_relative_to_workspace": str(paths[3].relative_to(ROOT)),
            "sample_sha256": digest(paths[3]), "params_history_sha256": params_hash,
            "source_file_stats": before, "sample_indices": list(range(100)),
            "selection": "First 100, in saved trajectory order; no sorting or quality filtering.",
            "n_nonfinite_selected": 0, "normalization_mean": mean, "normalization_std": std,
            "display": "clip(x * std + mean, 0, 1); white ink on black; nearest-neighbour pixels.",
            "normalization_precision": "Rounded training-log values, identical for both models and the video; display only.",
            "output": filename.name,
        }
        print(f"Saved {filename}: first 100 unfiltered {family} {count} base-2000 samples")
    (output / "sample_gallery_provenance.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
