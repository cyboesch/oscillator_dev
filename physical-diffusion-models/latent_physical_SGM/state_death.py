#!/usr/bin/env python3
"""How do Potts states die in the early slices (exact statistics)? Merging / shadowing against starving.

For each run (Potts units) and each slice 0..40 and 99: the fields a_m(x) / T on a FIXED set of N_DATA clean training
images, noised at the slice's t with FIXED noise (common random numbers across slices: x_t = e^-t x0 + sqrt(T (1 -
e^-2t)) xi), the posteriors p(m | x) and each state's log10 mean occupation. A state is dead where that is below -6
(as states_below_1e-6 in the training log, which uses the validation batch instead).

A state "dies in the window" if it is alive at slice 5 and dead at slice 30. For each such state m, at its first dead
slice d (and its last alive slice d - 1):
  - its absorber m*: the state that wins most of the data m owned at slice 5 (argmax p), and that share (the
    concentration: 1 = one state took all of m's data; small = its data went to many states);
  - the field difference D(x) = (a_m*(x) - a_m(x)) / T over all data: mean and SD. A near-copy of m* (merging /
    shadowing) has D almost constant (small SD against its mean); a starving state loses to different states on
    different data;
  - in parameter space: cos(W_m, W_m*) and |W_m - W_m*| / |W_m*|, against the nearest-neighbour distances of the
    states alive at d;
  - its occupation trajectory (slices from first below 1e-2 to below 1e-6) and how far W_m moves per slice after
    death compared with living states (frozen or drifting);
  - the class (majority label) of the data it owned at slice 5.
Also: dead states that come back by slice 40 / 99, and which states that die in one run survive in another run with
the same seed and initialisation (e.g. 1x vs early 2x steps).

Usage (CPU Slurm job): python state_death.py LABEL=RUN ... [--pairs A,B ...]
Writes out/latent_physical_SGM/state_death/<job>/: report.txt, results.json, dying_<label>.png.
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

import jax.random as jr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "state_death", os.environ.get("SLURM_JOB_ID", "manual"))
N_DATA = 4096
LOG_DEAD = -6.0
SLICES = list(range(41)) + [99]
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def q(v, ps=(10, 50, 90)):
    """'p10 / p50 / p90' of an array (or '-' if empty)."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return "-" if v.size == 0 else " / ".join(f"{np.percentile(v, p):.3g}" for p in ps)


def load_slice(run, k):
    z = np.load(os.path.join(lm.OUT_ROOT, run, "slices", f"slice_{k:03d}.npz"))
    return z["W"], z["b"], (z["Q"] if "Q" in z.files else None)


def fields(W, b, Q, X):
    A = X @ W.T + b
    if Q is not None:
        A = A + (X * X) @ Q.T
    return A / lm.TEMP


def log_post(A):
    am = A.max(axis=1, keepdims=True)
    return A - (am + np.log(np.exp(A - am).sum(axis=1, keepdims=True)))


def log10_occ(logp):
    lmax = logp.max(axis=0)
    return (lmax + np.log(np.mean(np.exp(logp - lmax), axis=0))) / np.log(10.0)


def analyse(label, run, x0, xi, labels, n_cls, taus):
    """Per-slice occupations and the death analysis of one run; returns a dict of results."""
    occ, owner, Ws, A_cache = {}, {}, {}, {}
    for k in SLICES:
        W, b, Q = load_slice(run, k)
        t = float(taus[k])
        e = np.exp(-t)
        X = e * x0 + np.sqrt(lm.TEMP * (1.0 - e * e)) * xi
        A = fields(W, b, Q, X)
        lp = log_post(A)
        occ[k] = log10_occ(lp)
        owner[k] = A.argmax(axis=1)
        Ws[k] = W
        if 6 <= k <= 30:                                   # the possible death slices (kept for the analysis)
            A_cache[k] = A
    K = Ws[0].shape[0]
    dead = {k: occ[k] < LOG_DEAD for k in SLICES}
    out(f"\n=== {label} = {run} (K {K})")
    out("  dead states (log10 mean occupation < -6) at slices 0/5/10/15/19/25/30/40/99: "
        + "/".join(str(int(dead[k].sum())) for k in (0, 5, 10, 15, 19, 25, 30, 40, 99)))
    dying = np.flatnonzero(~dead[5] & dead[30])
    revive40 = np.flatnonzero(dead[30] & ~dead[40])
    revive99 = np.flatnonzero(dead[30] & ~dead[99])
    out(f"  alive at slice 5 and dead at slice 30: {dying.size}; of the states dead at 30, alive again at 40: "
        f"{revive40.size}, at 99: {revive99.size}")

    # class of the data each state owned at slice 5
    own5 = owner[5]
    state_cls = -np.ones(K, int)
    for m in range(K):
        sel = own5 == m
        if sel.any():
            state_cls[m] = int(np.bincount(labels[sel], minlength=n_cls).argmax())
    cls_all = np.bincount(state_cls[(state_cls >= 0) & ~dead[5]], minlength=n_cls)
    cls_dying = np.bincount(state_cls[dying][state_cls[dying] >= 0], minlength=n_cls)
    out(f"  majority class of the data owned at slice 5: states alive at 5 {'/'.join(map(str, cls_all))}; dying "
        f"{'/'.join(map(str, cls_dying))} (classes 0..{n_cls - 1}; {int(np.sum(state_cls[dying] < 0))} dying states "
        f"owned no data at slice 5)")

    rows = []
    for m in dying:
        d = next(k for k in range(6, 31) if dead[k][m])
        kb = d - 1
        W, A = Ws[d], A_cache[d]
        mine = own5 == m                                   # the data m owned at slice 5
        if mine.sum() >= 1:
            winners = A[mine].argmax(axis=1)
        else:                                              # owned nothing: the points where m came closest
            gap = A.max(axis=1) - A[:, m]
            near = np.argsort(gap)[: max(10, N_DATA // 100)]
            winners = A[near].argmax(axis=1)
        cnt = np.bincount(winners, minlength=K)
        ms = int(cnt.argmax())
        conc = float(cnt[ms] / winners.size)
        D = A[:, ms] - A[:, m]
        mu, sd = float(D.mean()), float(D.std())
        wm, ws = W[m], W[ms]
        cos = float(wm @ ws / (np.linalg.norm(wm) * np.linalg.norm(ws) + 1e-300))
        rel = float(np.linalg.norm(wm - ws) / (np.linalg.norm(ws) + 1e-300))
        # occupation trajectory: first slice below 1e-2 (log10 -2) to the first dead slice
        k2 = next((k for k in range(0, 31) if occ[k][m] < -2.0), d)
        # movement of W_m per slice after death vs living states (slices d..min(d+5, 40))
        mv_dead, mv_live = [], []
        for k in range(d, min(d + 5, 40)):
            dW = np.linalg.norm(Ws[k + 1] - Ws[k], axis=1) / (np.linalg.norm(Ws[k], axis=1) + 1e-300)
            mv_dead.append(dW[m])
            mv_live.append(np.median(dW[~dead[k] & ~dead[k + 1]]))
        rows.append({"m": int(m), "d": d, "absorber": ms, "conc": conc, "mu": mu, "sd": sd, "cos": cos, "rel": rel,
                     "owned5": int(mine.sum()), "slices_to_death": d - k2, "move_dead": float(np.mean(mv_dead)),
                     "move_live": float(np.mean(mv_live)), "cls": int(state_cls[m]), "kb": kb})

    # reference: nearest-neighbour W distances among living states at slice 19
    W19 = Ws[19]
    live19 = np.flatnonzero(~dead[19])
    Wl = W19[live19]
    nrm = np.linalg.norm(Wl, axis=1)
    G = Wl @ Wl.T
    d2 = nrm[:, None] ** 2 + nrm[None, :] ** 2 - 2 * G
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.maximum(d2.min(axis=1), 0.0)) / nrm
    cosm = G / (nrm[:, None] * nrm[None, :])
    np.fill_diagonal(cosm, -np.inf)
    nn_cos = cosm.max(axis=1)

    if rows:
        R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        out(f"  death slice (first dead), p10 / p50 / p90: {q(R['d'])}; slices from occupation < 1e-2 to dead: "
            f"{q(R['slices_to_death'])}")
        out(f"  data owned at slice 5 by a dying state: {q(R['owned5'])} of {N_DATA} images")
        out(f"  absorber concentration (share of the state's slice-5 data won by one state at its death slice): "
            f"{q(R['conc'])}; >= 0.8 in {int(np.sum(R['conc'] >= 0.8))} of {len(rows)}")
        out(f"  field gap to the absorber D = (a_m* - a_m)/T over all data: mean {q(R['mu'])}; SD {q(R['sd'])}; "
            f"SD / mean {q(R['sd'] / np.maximum(R['mu'], 1e-12))}")
        out(f"  W: cos(W_m, W_m*) {q(R['cos'])}; |W_m - W_m*| / |W_m*| {q(R['rel'])}  (living states at slice 19: "
            f"nearest-neighbour cos {q(nn_cos)}, relative distance {q(nn)})")
        out(f"  W movement per slice after death, relative: dying {q(R['move_dead'])}; living states (median) "
            f"{q(R['move_live'])}")
        absorbers = np.bincount(R["absorber"], minlength=K)
        out(f"  absorbers: {int(np.sum(absorbers > 0))} distinct states absorb the {len(rows)} dying ones; the busiest "
            f"absorbs {int(absorbers.max())}; absorbers that die themselves by slice 30: "
            f"{int(np.sum(dead[30][np.flatnonzero(absorbers > 0)]))}")
        merge = (R["conc"] >= 0.8) & (R["cos"] >= 0.9)
        out(f"  classification (thresholds: concentration >= 0.8 and cos >= 0.9 = merging / shadowing by one state): "
            f"merging {int(merge.sum())}, other (data spread over several states or a dissimilar winner) "
            f"{int((~merge).sum())}")
    del A_cache
    return {"label": label, "run": run, "K": K, "dead_counts": {str(k): int(dead[k].sum()) for k in SLICES},
            "dying": [int(m) for m in dying], "rows": rows, "occ": {str(k): occ[k].tolist() for k in (5, 10, 15, 19, 25, 30, 40, 99)},
            "nn_cos19": nn_cos.tolist(), "nn_rel19": nn.tolist(), "Ws": Ws, "own5": own5}


def figure(res, x0_clean, grid, n_show=12):
    rows = res["rows"]
    if not rows:
        return
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(rows), size=min(n_show, len(rows)), replace=False))
    fig, axes = plt.subplots(4, len(pick), figsize=(1.35 * len(pick), 6.0), squeeze=False)
    own5 = res["own5"]
    for j, i in enumerate(pick):
        r = rows[i]
        m, ms = r["m"], r["absorber"]
        sel = own5 == m
        img = x0_clean[sel].mean(axis=0) if sel.any() else np.zeros(grid * grid)
        axes[0, j].imshow(img.reshape(grid, grid), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"m {m}, n {int(sel.sum())}\ndies at {r['d']}", fontsize=6)
        for row, (W, ttl) in enumerate(((res["Ws"][r["kb"]][m], f"W_m s{r['kb']}"),
                                        (res["Ws"][r["d"]][ms], f"W_m* s{r['d']}")), start=1):
            v = np.abs(W).max() + 1e-12
            axes[row, j].imshow(W.reshape(grid, grid), cmap="RdBu_r", vmin=-v, vmax=v)
            axes[row, j].set_title(ttl if j == 0 else "", fontsize=6)
        sel_s = own5 == ms
        img_s = x0_clean[sel_s].mean(axis=0) if sel_s.any() else np.zeros(grid * grid)
        axes[3, j].imshow(img_s.reshape(grid, grid), cmap="gray", vmin=0, vmax=1)
        axes[3, j].set_title(f"m* {ms}: conc {r['conc']:.2f}\ncos {r['cos']:.2f}", fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("data of m\n(slice 5)", fontsize=7)
    axes[1, 0].set_ylabel("W_m before\ndeath", fontsize=7)
    axes[2, 0].set_ylabel("W of absorber\nat death", fontsize=7)
    axes[3, 0].set_ylabel("data of m*\n(slice 5)", fontsize=7)
    fig.suptitle(f"{res['label']}: dying states (alive at slice 5, dead at 30) and their absorbers", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"dying_{res['label']}.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=RUN (Potts runs)")
    ap.add_argument("--pairs", nargs="*", default=[], help="A,B: of the states dying in A, how many survive in B "
                    "(same seed and initialisation, so the state index is the same unit early on)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; state_death; job {os.environ.get('SLURM_JOB_ID', 'manual')}")
    specs = [s.split("=", 1) for s in args.runs]
    with open(os.path.join(lm.OUT_ROOT, specs[0][1], "config.json")) as f:
        cfg = json.load(f)
    prob = lm.make_problem(cfg.get("problem", "mnist28"), cfg.get("radius"), cfg.get("neighbourhood", "chebyshev"))
    data, labels, grid = np.asarray(prob["data"], dtype=np.float64), np.asarray(prob["labels"]), prob["grid"]
    n_cls = int(labels.max()) + 1
    kidx, knoise = jr.split(jr.PRNGKey(123))
    idx = np.asarray(jr.choice(kidx, data.shape[0], shape=(N_DATA,), replace=False))
    x0 = data[idx]
    xi = np.asarray(jr.normal(knoise, x0.shape), dtype=np.float64)
    lab = labels[idx]
    x0_clean = x0 * prob["std"] + prob["mean"]
    out(f"problem {cfg.get('problem')}: {N_DATA} fixed training images (classes {np.bincount(lab).tolist()}), fixed "
        f"noise; slices {SLICES[0]}-{SLICES[-2]} and 99")
    res = {}
    for label, run in specs:
        with open(os.path.join(lm.OUT_ROOT, run, "config.json")) as f:
            c = json.load(f)
        if c.get("hidden") != "potts":
            out(f"{label}: not a Potts run; skipped")
            continue
        _, taus = lm.run_time_grid(os.path.join(lm.OUT_ROOT, run), c)
        t0 = time.time()
        res[label] = analyse(label, run, x0, xi, lab, n_cls, taus)
        figure(res[label], x0_clean, grid)
        out(f"  ({time.time() - t0:.0f} s)")
    for p in args.pairs:
        a, b = p.split(",")
        if a in res and b in res:
            da = np.array(res[a]["dying"], int)
            db = res[b]["dead_counts"]
            surv = np.array([res[b]["occ"]["30"][m] >= LOG_DEAD for m in da])
            out(f"\npair {a} -> {b}: of the {da.size} states dying in {a} (slices 5-30), {int(surv.sum())} are alive at "
                f"slice 30 in {b}; {b} has {db['30']} dead at 30 against {res[a]['dead_counts']['30']} in {a}")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({lb: {k: v for k, v in r.items() if k not in ("Ws", "own5")} for lb, r in res.items()}, f, indent=1)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    out(f"wrote {OUT}")


if __name__ == "__main__":
    main()
