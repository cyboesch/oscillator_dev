#!/usr/bin/env python3
"""Temporary check (2026-11-16, --patch3 refactor; deleted after use): (1) a --patch + --patch2 run, old vs new code,
slice files bit-identical; (2) the three-scale run's history holds all three sets, nonzero after the switch-on, the
diagnostics patch2_* and patch3_* are recorded, and the samples are finite with patch_units of three specs."""
import glob
import json
import os
import sys

import numpy as np

OUT = "/physical-diffusion-models/out/latent_physical_SGM"
fail = 0
old, new = (sorted(glob.glob(f"{OUT}/{n}/slices/slice_*.npz")) for n in ("tmpcheck_p3_old", "tmpcheck_p3_new"))
print("regression slices", len(old), len(new))
if not old or len(old) != len(new):
    print("FAIL: slice lists differ")
    fail += 1
for a, b in zip(old, new):
    za, zb = np.load(a), np.load(b)
    if sorted(za.files) != sorted(zb.files):
        print("FAIL keys", os.path.basename(a), sorted(set(za.files) ^ set(zb.files)))
        fail += 1
    bad = [k for k in za.files if k != "record" and k in zb.files
           and not (za[k].shape == zb[k].shape and (np.array_equal(za[k], zb[k], equal_nan=True)
                                                     if za[k].dtype.kind in "fc" else np.array_equal(za[k], zb[k])))]
    ra, rb = json.loads(str(za["record"])), json.loads(str(zb["record"]))
    rbad = sorted(set(ra) ^ set(rb))
    print(f"{os.path.basename(a)}: arrays {'all bitwise' if not bad else 'DIFFER ' + str(bad)}; record keys "
          f"{'same' if not rbad else 'DIFFER ' + str(rbad)}")
    fail += bool(bad) + bool(rbad)

run = f"{OUT}/tmpcheck_p3_three"
cfg = json.load(open(f"{run}/config.json"))
print("three-scale config:", cfg.get("patch"), cfg.get("patch2"), cfg.get("patch3"))
h = np.load(f"{run}/hidden_history.npz")
print("hidden_history members:", h.files)
for g in ("Wp", "bp", "Qp", "Wp2", "bp2", "Qp2", "Wp3", "bp3", "Qp3"):
    if g not in h.files:
        print("FAIL missing", g)
        fail += 1
        continue
    nz = [k for k in range(h[g].shape[0]) if np.any(h[g][k])]
    print(f"  {g} {h[g].shape}; first nonzero slice {nz[0] if nz else None}")
    if not nz or nz[0] != 55:
        fail += 1
rec = json.loads(str(np.load(f"{run}/slices/slice_099.npz")["record"]))
p3 = sorted(k for k in rec if k.startswith(("patch2", "patch3", "force_rms_patch")))
print("slice 99 record keys:", p3)
fail += not any(k.startswith("patch3_") for k in p3) or "force_rms_patch3" not in p3
for m in sorted(glob.glob(f"{run}/samples/*.json")):
    print(os.path.basename(m), "patch_units:", json.load(open(m)).get("patch_units"))
for m in sorted(glob.glob(f"{run}/samples/*.npy")):
    x = np.load(m)
    ok = bool(np.all(np.isfinite(x)))
    print(os.path.basename(m), x.shape, "finite", ok)
    fail += not ok
print("RESULT:", "PASS" if fail == 0 else f"FAIL ({fail})")
sys.exit(1 if fail else 0)
