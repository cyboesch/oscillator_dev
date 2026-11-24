#!/usr/bin/env python3
"""REVIEW COPY: validation loss per slice of every run of this folder minus the locked baseline's (same validation
batch per slice; negative = better). Reads train_log.jsonl only. Usage: python3 val_compare.py [slice ...]"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "out", "latent_physical_SGM",
                    "celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1")
def load(d):
    return {r["slice"]: r for r in map(json.loads, open(os.path.join(d, "train_log.jsonl")))}
b = load(BASE)
ks = [int(a) for a in sys.argv[1:]] or [12, 30, 45, 54, 56, 60, 66, 72, 78, 84, 90, 94, 96, 98, 99]
print("slice".ljust(74) + "".join(f"{k:>8d}" for k in ks))
print("baseline val".ljust(74) + "".join(f"{b[k]['val_loss']:8.0f}" for k in ks))
for d in sorted(glob.glob(os.path.join(HERE, "out", "celeba32g_*"))):
    try:
        r = load(d)
    except FileNotFoundError:
        continue
    name = os.path.basename(d)[len("celeba32g_"):].replace("_seed1", "").replace("locked-patches-64states_", "")[:72]
    print(name.ljust(74) + "".join((f"{r[k]['val_loss'] - b[k]['val_loss']:8.1f}" if k in r else "       .") for k in ks)
          + f"   (done {max(r)})")
