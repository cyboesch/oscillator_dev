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
  knotsarc:B  (2026-11-20) like knots:B, but the B kept slices sit at equal steps of the path length of the group's
           history (dense where the parameters move fast); knotsopt:B greedy, hat:B / cheb:B least squares (see compress).
  --knot-share set / all (2026-11-20): one knot schedule per parameter set (global / each patch set; knotsarc on the
           set's joint path length) or for all groups, so the co-adapted groups of a set are interpolated together.
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


def _interp_rows(A, tt, idx):
    """Linear interpolation in t of the rows A[idx] onto every row time tt (per column)."""
    order = np.argsort(tt[idx])                       # np.interp needs increasing abscissae (t decreases with slice)
    xs = tt[idx][order]
    pos = np.interp(tt, xs, np.arange(xs.size, dtype=np.float64))
    lo = np.clip(np.floor(pos).astype(int), 0, xs.size - 1)
    hi = np.clip(lo + 1, 0, xs.size - 1)
    w = (pos - lo)[:, None]
    Ak = A[idx][order]
    return (1.0 - w) * Ak[lo] + w * Ak[hi]


def _basis(tt, kind, B):
    """Least-squares basis (na x B) on the active times: 'hat' = B piecewise-linear hats on uniform knots in t,
    'cheb' = Chebyshev polynomials of degree 0..B-1 in t mapped to [-1, 1]."""
    lo, hi = float(tt.min()), float(tt.max())
    if kind == "hat":
        tau = np.linspace(lo, hi, B)
        d = (hi - lo) / max(B - 1, 1)
        return np.maximum(0.0, 1.0 - np.abs(tt[:, None] - tau[None, :]) / d)
    s_ = 2.0 * (tt - lo) / max(hi - lo, 1e-300) - 1.0
    return np.polynomial.chebyshev.chebvander(s_, B - 1)


def _choose_knots(na, B, kind, s=None):
    """Kept slices (local indices 0..na-1, first and last included): 'knots' evenly spaced in slice index;
    'knotsarc' at equal steps of the cumulative path length s (length na); duplicates filled by splitting the longest
    gap (in slices) until B are kept (B + 1 if the last steps have zero length)."""
    if kind == "knots":
        return np.unique(np.round(np.linspace(0, na - 1, B)).astype(int))
    keep = set(np.searchsorted(s, np.linspace(0.0, s[-1], B), side="left").clip(0, na - 1).tolist())
    keep |= {0, na - 1}
    while len(keep) < B:
        ks = sorted(keep)
        g = int(np.argmax(np.diff(ks)))
        keep.add((ks[g] + ks[g + 1]) // 2)
    return np.array(sorted(keep))


def set_of(g):
    """2026-11-20: the parameter set a group belongs to: the patch sets (Wp / bp / Qp / Bpc + suffix '', '2', '3')
    or the global set (theta, W, b, wl, bl, Q, Bc, hc, ...). Groups of one set are trained together per slice."""
    for p in ("Wp", "bp", "Qp", "Bpc"):
        if g.startswith(p) and g[len(p):] in ("", "2", "3"):
            return "patch" + (g[len(p):] or "1")
    return "global"


def compress(X, t, mode, shared=None):
    """X (n_slices, P) -> (reconstruction, stored count, active slices, relative error, knot times or None).
    Modes (2026-11-19): knots:B (B evenly spaced kept slices, interpolation), knotsopt:B (B kept slices chosen
    greedily where the interpolation error is largest), hat:B / cheb:B (least-squares fit with B piecewise-linear hats
    or B Chebyshev polynomials in t), svd:R (mean + rank R), knots:all / svd:full (no compression).
    shared (2026-11-20; knots / knotsarc only): kept slices given as absolute slice indices (chosen for a whole set,
    --knot-share); the group keeps those inside its active range plus its first and last active slice."""
    nz = np.flatnonzero(np.any(X != 0.0, axis=1))
    if nz.size == 0:
        return X.copy(), 0, 0, 0.0, None
    k0 = int(nz[0])                                   # active from k0 on (patch sets: their switch-on slice)
    A = X[k0:]
    tt = t[k0:]
    na, P = A.shape
    kind, _, arg = mode.partition(":")
    if shared is not None and kind in ("knots", "knotsarc"):
        idx = np.array(sorted({0, na - 1} | {int(i) - k0 for i in shared if k0 < int(i) < k0 + na}))
        R = _interp_rows(A, tt, idx)
        stored = idx.size * P + idx.size
    elif kind == "knots":
        B = na if arg == "all" else min(int(arg), na)
        idx = _choose_knots(na, B, "knots")
        R = _interp_rows(A, tt, idx)
        stored = idx.size * P
    elif kind == "knotsopt":
        B = min(int(arg), na)
        keep = [0, na - 1]
        while len(keep) < B:
            R = _interp_rows(A, tt, np.array(sorted(keep)))
            e = np.sum((R - A) ** 2, axis=1)
            e[keep] = -1.0
            keep.append(int(np.argmax(e)))
        idx = np.array(sorted(keep))
        R = _interp_rows(A, tt, idx)
        stored = idx.size * P + idx.size
    elif kind == "knotsarc":
        # 2026-11-20: B kept slices at equal steps of the path length s_k = sum_{j<=k} |A_j - A_{j-1}| (dense where
        # the parameters move fast); duplicates are filled by splitting the longest gap (in slices) until B are kept.
        B = min(int(arg), na)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))])
        idx = _choose_knots(na, B, "knotsarc", s)
        R = _interp_rows(A, tt, idx)
        stored = idx.size * P + idx.size
    elif kind in ("hat", "cheb"):
        B = min(int(arg), na)
        Phi = _basis(tt, kind, B)
        C, *_ = np.linalg.lstsq(Phi, A, rcond=None)
        R = Phi @ C
        stored = B * P
    elif kind == "svd":
        mu = A.mean(axis=0)
        U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
        r = S.size if arg == "full" else min(int(arg), S.size)
        R = mu + (U[:, :r] * S[:r]) @ Vt[:r]
        stored = (r + 1) * P + r * na
    else:
        raise ValueError(f"mode {mode!r}: knots:B, knotsopt:B, knotsarc:B, hat:B, cheb:B, svd:R, knots:all or "
                         f"svd:full")
    out = X.copy()
    out[k0:] = R
    err = float(np.linalg.norm(R - A) / max(np.linalg.norm(A), 1e-300))
    knot_t = tt[idx] if kind.startswith("knots") else None
    return out, stored, na, err, knot_t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", required=True, help="knots:B | knotsopt:B | knotsarc:B | hat:B | cheb:B | svd:R | knots:all | svd:full")
    ap.add_argument("--group-modes", default="", help="per-group overrides, e.g. 'wl=svd:4,bl=svd:4'")
    ap.add_argument("--out-name", default="")
    ap.add_argument("--dry-run", action="store_true", help="only the error report; no run directory is written")
    ap.add_argument("--knot-share", default="group", choices=["group", "set", "all"],
                    help="(2026-11-20; knots:B / knotsarc:B) choose the kept slices per group (default, as before), "
                         "once per parameter set (global / patch sets; knotsarc on the set's joint path length), or "
                         "once for all groups; groups with a --group-modes override keep their own mode")
    args = ap.parse_args()
    t0 = time.time()
    src = os.path.join(lm.OUT_ROOT, args.run)
    if not os.path.exists(os.path.join(src, "DONE")):
        raise SystemExit(f"{src}: no DONE marker")
    gmodes = dict(kv.split("=", 1) for kv in args.group_modes.split(",") if kv.strip())
    mkind, _, marg = args.mode.partition(":")
    if args.knot_share != "group" and (mkind not in ("knots", "knotsarc") or not marg.isdigit()):
        raise SystemExit("--knot-share set / all needs --mode knots:B or knotsarc:B")
    tag = args.mode.replace(":", "") + "".join(f"_{g}-{m.replace(':', '')}" for g, m in sorted(gmodes.items()))
    if args.knot_share != "group":
        tag += f"_share-{args.knot_share}"
    name = args.out_name or f"{args.run}_time-{tag}"
    dst = os.path.join(lm.OUT_ROOT, name)
    if not args.dry_run and os.path.exists(os.path.join(dst, "DONE")):
        raise SystemExit(f"{dst} exists")
    if not args.dry_run:
        os.makedirs(dst, exist_ok=True)
    th = np.load(os.path.join(src, "params_history.npy"))
    hid = np.load(os.path.join(src, "hidden_history.npz"))
    t = np.asarray(hid["t"], dtype=np.float64)
    lines = [f"compress_history {args.mode} (group overrides {gmodes or 'none'}) of {args.run} -> {name}; "
             f"{th.shape[0]} slices",
             f"{'group':<8}{'per slice':>12}{'active':>8}{'stored':>14}{'full':>14}{'rel. error':>12}"]
    tot_slice = tot_stored = tot_full = 0
    arrays = {}
    groups = [("theta", th)] + [(g, hid[g]) for g in hid.files if g != "t"]
    shared = {}                                       # set -> kept absolute slice indices (--knot-share set / all)
    if args.knot_share != "group":
        n = th.shape[0]
        d2, k0s = {}, {}                              # per set: squared step lengths (switch-on steps excluded)
        for g, a in groups:
            if g in gmodes:
                continue
            S = set_of(g) if args.knot_share == "set" else "all"
            X = np.asarray(a, dtype=np.float64).reshape(n, -1)
            nz = np.flatnonzero(np.any(X != 0.0, axis=1))
            if nz.size == 0:
                continue
            k0 = int(nz[0])
            step = np.zeros(n)
            step[k0 + 1:] = np.sum(np.diff(X[k0:], axis=0) ** 2, axis=1)
            d2[S] = d2.get(S, 0.0) + step
            k0s[S] = min(k0s.get(S, n), k0)
        for S in sorted(d2):
            k0 = k0s[S]
            s = np.concatenate([[0.0], np.cumsum(np.sqrt(d2[S][k0 + 1:]))])
            idx = _choose_knots(n - k0, min(int(marg), n - k0), mkind, s)
            shared[S] = k0 + idx
            lines.insert(1, f"knot share {args.knot_share}: set {S} keeps slices "
                            + " ".join(str(int(i)) for i in shared[S])
                            + "  (t = " + " ".join(f"{t[int(i)]:.2f}" for i in shared[S]) + ")")
    for g, a in groups:
        X = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
        sh = None if (g in gmodes or not shared) else shared.get(set_of(g) if args.knot_share == "set" else "all")
        Rx, stored, na, err, knot_t = compress(X, t, gmodes.get(g, args.mode), shared=sh)
        arrays[g] = Rx.reshape(a.shape).astype(a.dtype)
        P = X.shape[1]
        tot_slice += P
        tot_stored += stored
        tot_full += na * P
        kt = ("   knots at t = " + " ".join(f"{v:.2f}" for v in knot_t)) if knot_t is not None and knot_t.size <= 20 \
            else ""
        lines.append(f"{g:<8}{P:>12,}{na:>8}{stored:>14,}{na * P:>14,}{err:>12.2e}{kt}")
    lines.append(f"{'total':<8}{tot_slice:>12,}{'':>8}{tot_stored:>14,}{tot_full:>14,}   (stored / full "
                 f"{tot_stored / max(tot_full, 1):.3f})")
    if args.dry_run:
        print("\n".join(lines), flush=True)
        return
    np.save(os.path.join(dst, "params_history.npy"), arrays.pop("theta"))
    np.savez(os.path.join(dst, "hidden_history.npz"), t=hid["t"], **arrays)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg["time_compression"] = {"mode": args.mode, "group_modes": gmodes, "source_run": args.run,
                               "knot_share": args.knot_share,
                               "stored": int(tot_stored),
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
