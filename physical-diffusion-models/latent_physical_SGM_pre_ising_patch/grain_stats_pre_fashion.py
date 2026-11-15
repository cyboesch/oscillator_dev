#!/usr/bin/env python3
"""Grain and stroke statistics of sample sets against real held-out digits (CPU job; reads only).

Section A of out/latent_physical_SGM/patch_design/eval_physics/baseline_measure.py (job 4231229), generalised to any
LABEL=PATH sets. Per image, in pixel units (x * std + mean of the problem's normalisation, clipped to [0, 1]):
  area      ink pixels (> 0.5)
  width     stroke width 2 area / (number of 4-neighbour ink/background boundary edges): a stroke of width w and length
            L has area w L and a boundary of about 2 L
  ncomp     8-connected ink components;  specks: components of at most 2 pixels;  holes: background pixels whose 4
            neighbours are all ink
  speckle   mean |x - median3x3(x)| (zero padding)
  mid_frac  share of pixels in (0.1, 0.9)
Per set: mean, SE (SD / sqrt(n)) and SD; samples that are not finite or exceed 15 (normalised units) in absolute value
count as diverged and are left out. References: 2000 real test digits of the problem's classes (the MNIST test split,
plus the paper's N(0, 0.01^2) pixel noise) and 2000 training images, drawn as baseline_measure.py draws them (rng 3,
training first), so the numbers equal 4231229's (real_test specks 0.01, components 1.03; plain seed 1 0.52, 1.84).
Also each set's deviation from the reference in units of the reference's per-image SD and, with --pair A,B (sets of
the same trajectories, e.g. two runs sampled from the same trajectory set), the paired difference A - B by trajectory
(images finite in both) with its SE.

Usage (from latent_physical_SGM/):
  sbatch --time=00:20:00 --cpus-per-task=4 --mem=16G jobs/run_cpu_neuronic.sh grain_stats.py --problem mnist28_d0123 \
      LABEL=RUN/samples/sde_base_n500.npy ... [--pair A,B ...]
PATH: a .npy of samples (n, N) in normalised units, relative to out/latent_physical_SGM/ or absolute.
Writes out/latent_physical_SGM/grain_stats/<job>/report.txt and results.json.
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

from scipy import ndimage  # noqa: E402

STAT_KEYS = ["area", "width", "ncomp", "specks", "holes", "speckle", "mid_frac"]
OUT = os.path.join(lm.OUT_ROOT, "grain_stats", os.environ.get("SLURM_JOB_ID", "manual"))
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def image_stats(P):
    """Per-image statistics of P (n, G, G), pixel units in [0, 1] (baseline_measure.image_stats, its subset)."""
    n = P.shape[0]
    ink = P > 0.5
    area = ink.sum(axis=(1, 2)).astype(np.float64)
    I = np.pad(ink, ((0, 0), (1, 1), (1, 1)))
    edges = ((I[:, 1:, :] != I[:, :-1, :]).sum(axis=(1, 2))
             + (I[:, :, 1:] != I[:, :, :-1]).sum(axis=(1, 2))).astype(np.float64)
    width = np.where(edges > 0, 2.0 * area / np.maximum(edges, 1.0), np.nan)
    st = np.zeros((3, 3, 3), bool)
    st[1] = True                                          # 8-connectivity within each image, none across images
    lab_, nl = ndimage.label(ink, structure=st)
    flat = lab_.ravel()
    ids, first = np.unique(flat, return_index=True)
    keep = ids > 0
    ids, first = ids[keep], first[keep]
    sizes = np.bincount(flat, minlength=nl + 1)[ids]
    img = first // (P.shape[1] * P.shape[2])
    ncomp = np.bincount(img, minlength=n).astype(np.float64)
    specks = np.bincount(img[sizes <= 2], minlength=n).astype(np.float64)
    holes = ((~ink) & I[:, :-2, 1:-1] & I[:, 2:, 1:-1] & I[:, 1:-1, :-2] & I[:, 1:-1, 2:]).sum(axis=(1, 2))
    med = ndimage.median_filter(P, size=(1, 3, 3), mode="constant", cval=0.0)
    speckle = np.mean(np.abs(P - med), axis=(1, 2))
    mid = np.mean((P > 0.1) & (P < 0.9), axis=(1, 2))
    return {"area": area, "width": width, "ncomp": ncomp, "specks": specks, "holes": holes.astype(np.float64),
            "speckle": speckle, "mid_frac": mid}


def summary(st):
    mu = {k: float(np.nanmean(st[k])) for k in STAT_KEYS}
    sd = {k: float(np.nanstd(st[k])) for k in STAT_KEYS}
    se = {k: sd[k] / float(np.sqrt(max(np.sum(np.isfinite(st[k])), 1))) for k in STAT_KEYS}
    return mu, se, sd


def fmt(k, v):
    return f"{v:9.4f}" if k in ("width", "speckle", "mid_frac") else f"{v:9.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="LABEL=PATH (.npy of samples in normalised units)")
    ap.add_argument("--problem", default="mnist28_d0123")
    ap.add_argument("--pair", action="append", default=[], help="A,B: paired difference A - B by trajectory")
    ap.add_argument("--max-abs", type=float, default=15.0, help="samples beyond this |x| count as diverged")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    out(f"=== grain_stats {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
        f"problem {args.problem}")
    prob = lm.make_problem(args.problem)
    G = int(prob["grid"])
    mean, std = float(prob["mean"]), float(prob["std"])
    digits = list(prob["digits"]) if prob.get("digits") is not None else None
    Xtr = np.asarray(prob["data"], dtype=np.float64)

    def px(x):
        return np.clip(np.asarray(x, np.float64) * std + mean, 0.0, 1.0)

    rows = {}
    rng = np.random.default_rng(3)
    j = rng.choice(Xtr.shape[0], 2000, replace=False)
    rows["real_train"] = (image_stats(px(Xtr[j]).reshape(-1, G, G)), 2000, 0, None)
    ref = "real_train"
    try:
        import fid_score
        Xte, yte = fid_score.load_mnist("test", G)
        sel = np.isin(yte, digits) if digits is not None else np.ones(yte.shape[0], bool)
        Xte_px = np.asarray(Xte[sel], np.float64)
        j = rng.choice(Xte_px.shape[0], 2000, replace=False)
        P = np.clip(Xte_px[j] + 0.01 * rng.standard_normal((j.size, G * G)), 0.0, 1.0)
        rows["real_test"] = (image_stats(P.reshape(-1, G, G)), j.size, 0, None)
        ref = "real_test"
    except Exception as e:                                # noqa: BLE001
        out(f"could not load the MNIST test split ({type(e).__name__}: {e}); the reference is real_train")
    for spec in args.sets:
        label, path = spec.split("=", 1)
        fpath = path if os.path.isabs(path) else os.path.join(lm.OUT_ROOT, path)
        if not os.path.exists(fpath):
            out(f"  {label}: missing {fpath}")
            continue
        x = np.load(fpath)
        ok = np.all(np.isfinite(x), axis=1)
        ok[ok] = np.max(np.abs(x[ok]), axis=1) <= args.max_abs
        st = image_stats(px(np.where(ok[:, None], x, 0.0)).reshape(-1, G, G))
        rows[label] = (st, int(ok.sum()), int((~ok).sum()), {"path": path, "ok": ok})
    out(f"\n  {'set':<16}{'n':>5}{'div':>5}" + "".join(f"{k:>9}" for k in STAT_KEYS) + "   (means; SE below)")
    res = {"problem": args.problem, "reference": ref, "sets": {}}
    for label, (st, n, nd, info) in rows.items():
        ok = info["ok"] if info else np.ones(n, bool)
        stk = {k: st[k][ok] for k in STAT_KEYS}
        mu, se, sd = summary(stk)
        res["sets"][label] = {"n": n, "diverged": nd, "path": info["path"] if info else None, "mean": mu, "se": se,
                              "sd": sd}
        out(f"  {label:<16}{n:>5}{nd:>5}" + "".join(fmt(k, mu[k]) for k in STAT_KEYS))
        out(f"  {'':<16}{'':>5}{'':>5}" + "".join(fmt(k, se[k]) for k in STAT_KEYS))
    out(f"\n  deviation from {ref} in units of its per-image SD, (set mean - {ref} mean) / {ref} SD:")
    r_ = res["sets"][ref]
    for label in res["sets"]:
        d = {k: (res["sets"][label]["mean"][k] - r_["mean"][k]) / max(r_["sd"][k], 1e-12) for k in STAT_KEYS}
        res["sets"][label]["dev_sd"] = d
        out(f"  {label:<16}" + "".join(f"{d[k]:+9.2f}" for k in STAT_KEYS))
    res["pairs"] = {}
    for pr in args.pair:
        a, b = pr.split(",")
        if a not in rows or b not in rows or rows[a][3] is None or rows[b][3] is None:
            out(f"  pair {pr}: both must be sample sets")
            continue
        (sa, na, _, ia), (sb, nb, _, ib) = rows[a], rows[b]
        if sa["area"].shape != sb["area"].shape:
            out(f"  pair {pr}: different numbers of samples ({sa['area'].shape[0]} / {sb['area'].shape[0]})")
            continue
        both = ia["ok"] & ib["ok"]
        cells, rp = [], {}
        for k in STAT_KEYS:
            dlt = sa[k][both] - sb[k][both]
            dlt = dlt[np.isfinite(dlt)]
            m, s = float(np.mean(dlt)), float(np.std(dlt) / np.sqrt(max(dlt.size, 1)))
            rp[k] = [m, s]
            cells.append(f"{k} {m:+.4g} +- {s:.2g}")
        res["pairs"][pr] = {"n": int(both.sum()), **rp}
        out(f"  paired {a} - {b} ({int(both.sum())} trajectories): " + "; ".join(cells))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    out(f"\nwrote {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
