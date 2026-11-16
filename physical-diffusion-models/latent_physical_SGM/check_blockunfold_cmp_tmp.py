#!/usr/bin/env python3
"""Temporary (2026-11-16): compare the slice files of tmpcheck_blk_slices and tmpcheck_blk_blocks (deleted after)."""
import glob
import json
import os

import numpy as np

OUT = "/physical-diffusion-models/out/latent_physical_SGM"
a = sorted(glob.glob(f"{OUT}/tmpcheck_blk_slices/slices/slice_*.npz"))
b = sorted(glob.glob(f"{OUT}/tmpcheck_blk_blocks/slices/slice_*.npz"))
print("slices", len(a), len(b))
bad = 0
for pa, pb in zip(a, b):
    za, zb = np.load(pa), np.load(pb)
    out = []
    for k in za.files:
        if k == "record":
            continue
        x, y = za[k], zb[k]
        if x.dtype.kind in "fc":
            eq = np.array_equal(x, y, equal_nan=True)
            if not eq:
                rel = float(np.max(np.abs(x - y)) / (np.max(np.abs(x)) + 1e-300))
                out.append(f"{k} rel {rel:.1e}")
                bad += rel > 1e-4
        elif not np.array_equal(x, y):
            out.append(f"{k} DIFFERS")
            bad += 1
    ra, rb = json.loads(str(za["record"])), json.loads(str(zb["record"]))
    print(os.path.basename(pa), "all arrays bitwise" if not out else "; ".join(out),
          f"| val {ra.get('val')} vs {rb.get('val')}")
print("TRAIN RESULT:", "PASS (bitwise or < 1e-4)" if bad == 0 else f"FAIL ({bad})")
raise SystemExit(1 if bad else 0)
