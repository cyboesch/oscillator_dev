"""Post-hoc compression of a trained run's time dependence (2026-11-19): how many degrees of freedom in time do the
parameters need? (the user: "can we implement the time basis functions to reduce parameters?")

A run stores one parameter set per time slice (100 slices: params_history.npy, hidden_history.npz). This tool replaces
every parameter group's history X (n_active x P, the slices where the group exists; patch sets are zero before their
switch-on and stay so) by a compressed one and writes it as a new run directory that sample_latent.py reads like any
run (config.json copied with a 'time_compression' record; cond_clusters.npz copied for a --cond run):
  knots:B  B of the active slices (evenly spaced in slice index, first and last always kept), linear interpolation in t
           between them: stored B x P per group (hardware: B coupling sets);
  svd:R    X ~ mean + sum_{r<=R} a_r(t) V_r (truncated SVD of the time-centred history): stored (R + 1) x P + R x n_active
           per group (hardware: R coupling patterns driven by R scalar control signals a_r(t), plus the mean);
  svd:full / knots:all   no compression (a check: reproduces the history to rounding / exactly).
Writes report.txt (per group: P, active slices, stored numbers, relative reconstruction error) in the new run.

  python compress_history.py --run RUN --mode svd:4 [--out-name NAME]
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402


def compress(X, t, mode):
    """X (n_slices, P) -> (reconstruction, stored count, active slices, relative error)."""
    n = X.shape[0]
    nz = np.flatnonzero(np.any(X != 0.0, axis=1))
    if nz.size == 0:
        return X.copy(), 0, 0, 0.0
    k0 = int(nz[0])                                   # active from k0 on (patch sets: their switch-on slice)
    A = X[k0:]
    na, P = A.shape
    kind, _, arg = mode.partition(":")
    if kind == "knots":
        B = na if arg == "all" else min(int(arg), na)
        idx = np.unique(np.round(np.linspace(0, na - 1, B)).astype(int))
        tt = t[k0:]
        order = np.argsort(tt[idx])                   # np.interp needs increasing abscissae (t decreases with slice)
        xs = tt[idx][order]
        # linear interpolation in t between the kept slices, per parameter (vectorised)
        pos = np.interp(tt, xs, np.arange(xs.size, dtype=np.float64))
        lo = np.clip(np.floor(pos).astype(int), 0, xs.size - 1)
        hi = np.clip(lo + 1, 0, xs.size - 1)
        w = (pos - lo)[:, None]
        Ak = A[idx][order]
        R = (1.0 - w) * Ak[lo] + w * Ak[hi]
        stored = idx.size * P
    elif kind == "svd":
        mu = A.mean(axis=0)
        U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
        r = S.size if arg == "full" else min(int(arg), S.size)
        R = mu + (U[:, :r] * S[:r]) @ Vt[:r]
        stored = (r + 1) * P + r * na
    else:
        raise ValueError(f"--mode {mode!r}: knots:B, knots:all, svd:R or svd:full")
    out = X.copy()
    out[k0:] = R
    err = float(np.linalg.norm(R - A) / max(np.linalg.norm(A), 1e-300))
    return out, stored, na, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", required=True, help="knots:B | knots:all | svd:R | svd:full")
    ap.add_argument("--out-name", default="")
    args = ap.parse_args()
    t0 = time.time()
    src = os.path.join(lm.OUT_ROOT, args.run)
    if not os.path.exists(os.path.join(src, "DONE")):
        raise SystemExit(f"{src}: no DONE marker")
    tag = args.mode.replace(":", "")
    name = args.out_name or f"{args.run}_time-{tag}"
    dst = os.path.join(lm.OUT_ROOT, name)
    if os.path.exists(os.path.join(dst, "DONE")):
        raise SystemExit(f"{dst} exists")
    os.makedirs(dst, exist_ok=True)
    th = np.load(os.path.join(src, "params_history.npy"))
    hid = np.load(os.path.join(src, "hidden_history.npz"))
    t = np.asarray(hid["t"], dtype=np.float64)
    lines = [f"compress_history {args.mode} of {args.run} -> {name}; {th.shape[0]} slices",
             f"{'group':<8}{'per slice':>12}{'active':>8}{'stored':>14}{'full':>14}{'rel. error':>12}"]
    tot_slice = tot_stored = tot_full = 0
    arrays = {}
    groups = [("theta", th)] + [(g, hid[g]) for g in hid.files if g != "t"]
    for g, a in groups:
        X = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
        Rx, stored, na, err = compress(X, t, args.mode)
        arrays[g] = Rx.reshape(a.shape).astype(a.dtype)
        P = X.shape[1]
        tot_slice += P
        tot_stored += stored
        tot_full += na * P
        lines.append(f"{g:<8}{P:>12,}{na:>8}{stored:>14,}{na * P:>14,}{err:>12.2e}")
    lines.append(f"{'total':<8}{tot_slice:>12,}{'':>8}{tot_stored:>14,}{tot_full:>14,}   (stored / full "
                 f"{tot_stored / max(tot_full, 1):.3f})")
    np.save(os.path.join(dst, "params_history.npy"), arrays.pop("theta"))
    np.savez(os.path.join(dst, "hidden_history.npz"), t=hid["t"], **arrays)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg["time_compression"] = {"mode": args.mode, "source_run": args.run, "stored": int(tot_stored),
                               "full": int(tot_full), "per_slice": int(tot_slice),
                               "job": os.environ.get("SLURM_JOB_ID", "manual")}
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for extra in ("cond_clusters.npz",):
        if os.path.exists(os.path.join(src, extra)):
            shutil.copy2(os.path.join(src, extra), os.path.join(dst, extra))
    lines.append(f"total {time.time() - t0:.0f} s")
    with open(os.path.join(dst, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(dst, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
