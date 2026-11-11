"""Learning curves per diffusion slice, to judge by eye whether training has converged.

Usage (CPU job; submit from latent_physical_SGM/):
  sbatch jobs/run_cpu_neuronic.sh plot_learning_curves.py \
      --group NAME LABEL=RUN [LABEL=RUN ...] [--group NAME2 ...] \
      [--pair NAME:A,B] [--mark NAME:LABEL:FRACTION] [--slices 0,10,...] [--window 1000]

RUN is a run folder under out/latent_physical_SGM/. For every group:
  NAME_train.png      the stored mini-batch training losses (every 10th step: slices/slice_kkk.npz, loss_trace) as a
                      running mean over --window steps; one panel per selected slice, the group's runs overlaid
                      (sequential colours in the order given when there are more than three runs).
  NAME_summary.png    per slice against t, from the raw traces: (left) the share of the slice's improvement still made
                      in its second half, f = (L_mid - L_end) / (L_start - L_end) (window means: first 5%, 45-55%,
                      last 10%; about 0 = flat by mid-slice, about 0.5 = still falling linearly); (right) the late
                      slope, mean(87.5-100%) - mean(75-87.5%) in units of its standard error (mini-batches are
                      independent draws; below -2 = still falling at the end of the slice).
  NAME_val.png        (runs trained with --val-every) the EMA validation loss on the fixed validation batch, paired
                      per sample against its final value (mean, band = +-2 SE), one panel per selected slice.
  NAME_val_summary.png  (same runs) per slice: validation improvement over the second half and over the quarter
                      before it (N/4 -> N/2 and N/2 -> N), in loss units and in SE.
  NAME_pair.png       (--pair NAME:A,B) per slice: final validation loss of A minus B, paired per sample on the
                      shared validation images (mean +- 2 SE), against t. Negative = A better.
--mark NAME:LABEL:F draws a vertical line at the fraction F of every slice of that run (e.g. the paper schedule's
budget, F = 0.5, in a --steps-scale 2 run).
Writes out/latent_physical_SGM/learning_curves/<job>/ (PNG + report.txt).
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(HERE), "out", "latent_physical_SGM")
OUT = os.path.join(OUT_ROOT, "learning_curves", os.environ.get("SLURM_JOB_ID", "manual"))
TRACE_EVERY = 10          # train_latent.py stores losses[::10]
CATEGORICAL = ["#1f77b4", "#ff7f0e", "#2ca02c"]

LINES = []


def report(msg=""):
    print(msg, flush=True)
    LINES.append(msg)


def load_run(run):
    d = run if os.path.isabs(run) else os.path.join(OUT_ROOT, run)
    recs = {}
    with open(os.path.join(d, "train_log.jsonl")) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "slice" in r:
                    recs[int(r["slice"])] = r          # a resumed run may log a slice twice: the last one counts
    slices = {}
    for k in sorted(recs):
        f = os.path.join(d, "slices", f"slice_{k:03d}.npz")
        if not os.path.exists(f):
            continue
        s = {"t": float(recs[k]["t"]), "steps": int(recs[k]["steps"]), "rec": recs[k]}
        with np.load(f) as z:
            s["trace"] = z["loss_trace"].astype(np.float64)
            if "val_per_sample" in z.files:
                s["val"] = z["val_per_sample"].astype(np.float64)
                s["val_idx"] = z["val_idx"] if "val_idx" in z.files else np.arange(len(s["val"]))
        vf = os.path.join(d, "val_curves", f"slice_{k:03d}.npz")
        if os.path.exists(vf):
            with np.load(vf) as z:
                s["vc_steps"] = z["steps"].astype(np.int64)
                s["vc"] = z["val_per_sample"].astype(np.float64)
        slices[k] = s
    return slices


def smooth(x, y, w):
    w = max(1, min(w, len(y)))
    c = np.cumsum(np.insert(y, 0, 0.0))
    return (x[: len(x) - w + 1] + x[w - 1:]) / 2, (c[w:] - c[:-w]) / w


def seg(y, a, b):
    n = len(y)
    i, j = int(a * n), max(int(b * n), int(a * n) + 2)
    s = y[i:j]
    return s.mean(), s.var(ddof=1) / len(s)


def trace_stats(y):
    L0, _ = seg(y, 0.0, 0.05)
    Lm, _ = seg(y, 0.45, 0.55)
    Le, _ = seg(y, 0.9, 1.0)
    f = (Lm - Le) / (L0 - Le) if L0 - Le > 0 else np.nan
    a, va = seg(y, 0.75, 0.875)
    b, vb = seg(y, 0.875, 1.0)
    return f, b - a, (b - a) / np.sqrt(va + vb)


def paired(a, b):
    d = a - b
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def colours(n):
    if n <= len(CATEGORICAL):
        return CATEGORICAL[:n]
    return [plt.cm.Blues(v) for v in np.linspace(0.4, 0.95, n)]


def grid(n):
    cols = 4 if n > 6 else min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    report(f"wrote {p}")


def do_group(name, runs, sel, window, marks, title):
    labels = [l for l, _ in runs]
    data = {l: load_run(r) for l, r in runs}
    col = dict(zip(labels, colours(len(labels))))
    report(f"\n=== group {name}: " + "; ".join(f"{l} = {r} ({len(data[l])} slices)" for l, r in runs))

    # training curves
    ks = [k for k in sel if any(k in data[l] for l in labels)]
    fig, axes = grid(len(ks))
    for ax, k in zip(axes, ks):
        lo, hi = np.inf, -np.inf
        for l in labels:
            s = data[l].get(k)
            if s is None:
                continue
            y = s["trace"]
            x = np.arange(len(y)) * TRACE_EVERY
            xs, ys = smooth(x, y, window // TRACE_EVERY)
            ax.plot(xs, ys, color=col[l], lw=1.2, label=l)
            late = ys[xs >= 0.1 * x[-1]]
            if len(late):
                lo, hi = min(lo, late.min()), max(hi, late.max())
            if (name, l) in marks:
                ax.axvline(marks[(name, l)] * s["steps"], color="0.4", lw=0.8, ls="--")
        if np.isfinite(lo):
            pad = 0.05 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
        t = next(data[l][k]["t"] for l in labels if k in data[l])
        ax.set_title(f"slice {k}, t = {t:.3f}", fontsize=8)
        ax.set_xlabel("training step in the slice", fontsize=7)
        ax.set_ylabel(f"training loss (running mean, {window} steps)", fontsize=7)
        style(ax)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{title}: training loss within each slice", fontsize=9)
    save(fig, f"{name}_train.png")

    # per-slice summary from the traces
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    report(f"per slice (selected): f = share of the improvement in the 2nd half; z = late slope / SE")
    report("slice      t  " + "  ".join(f"{l[:14]:>14} f / z" for l in labels))
    stats = {}
    for l in labels:
        rows = [(k, s["t"]) + trace_stats(s["trace"]) for k, s in sorted(data[l].items())]
        stats[l] = {r[0]: r for r in rows}
        tt = np.array([r[1] for r in rows])
        a1.plot(tt, [r[2] for r in rows], color=col[l], lw=1.2, marker="o", ms=2.5, label=l)
        a2.plot(tt, [r[4] for r in rows], color=col[l], lw=1.2, marker="o", ms=2.5, label=l)
    for k in sel:
        if any(k in stats[l] for l in labels):
            t = next(stats[l][k][1] for l in labels if k in stats[l])
            report(f"{k:5d} {t:6.3f}  " + "  ".join(
                f"{stats[l][k][2]:9.2f} / {stats[l][k][4]:5.1f}" if k in stats[l] else f"{'-':>17}" for l in labels))
    for lo_k, hi_k in ((0, 29), (30, 79), (80, 99)):
        report(f"slices {lo_k}-{hi_k}, medians: " + "; ".join(
            f"{l}: f {np.nanmedian([r[2] for kk, r in stats[l].items() if lo_k <= kk <= hi_k]):.2f}, "
            f"z {np.nanmedian([r[4] for kk, r in stats[l].items() if lo_k <= kk <= hi_k]):.1f}"
            for l in labels if any(lo_k <= kk <= hi_k for kk in stats[l])))
    a1.axhline(0.0, color="0.5", lw=0.6)
    a1.axhline(0.5, color="0.5", lw=0.6, ls=":")
    a1.set_ylabel("share of the slice's improvement made in its 2nd half", fontsize=7)
    a2.axhline(0.0, color="0.5", lw=0.6)
    a2.axhline(-2.0, color="0.5", lw=0.6, ls=":")
    a2.set_ylabel("late slope / SE (last eighth vs the one before)", fontsize=7)
    for ax in (a1, a2):
        ax.set_xlabel("diffusion time t of the slice (training runs from t = 5 to 0)", fontsize=7)
        ax.invert_xaxis()
        style(ax)
    a1.legend(fontsize=6, frameon=False)
    fig.suptitle(f"{title}: is each slice still improving at the end of its budget? (training mini-batches)", fontsize=9)
    save(fig, f"{name}_summary.png")

    # validation curves (runs with --val-every)
    vlabels = [l for l in labels if any("vc" in s for s in data[l].values())]
    if not vlabels:
        return data
    ks = [k for k in sel if any(k in data[l] and "vc" in data[l][k] for l in vlabels)]
    fig, axes = grid(len(ks))
    for ax, k in zip(axes, ks):
        for l in vlabels:
            s = data[l].get(k)
            if s is None or "vc" not in s:
                continue
            D = s["vc"] - s["vc"][-1]
            m, se = D.mean(1), D.std(1, ddof=1) / np.sqrt(D.shape[1])
            ax.plot(s["vc_steps"], m, color=col[l], lw=1.2, label=l)
            ax.fill_between(s["vc_steps"], m - 2 * se, m + 2 * se, color=col[l], alpha=0.25, lw=0)
            if (name, l) in marks:
                ax.axvline(marks[(name, l)] * s["steps"], color="0.4", lw=0.8, ls="--")
            tail = m[len(m) // 4:]
            ax.set_ylim(min(tail.min(), 0) * 1.1 - 1e-6, max(tail.max(), 0) * 1.1 + 1e-6)
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_title(f"slice {k}, t = {data[vlabels[0]][k]['t']:.3f}", fontsize=8)
        ax.set_xlabel("training step in the slice", fontsize=7)
        ax.set_ylabel("validation loss minus its final value", fontsize=7)
        style(ax)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{title}: EMA validation loss (fixed batch, paired per sample; band +-2 SE; dashed = mark)",
                 fontsize=9)
    save(fig, f"{name}_val.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    for l in vlabels:
        rows = []
        for k, s in sorted(data[l].items()):
            if "vc" not in s:
                continue
            st, V = s["vc_steps"], s["vc"]
            iq, im = np.argmin(np.abs(st - s["steps"] / 4)), np.argmin(np.abs(st - s["steps"] / 2))
            g1, e1 = paired(V[im], V[iq])
            g2, e2 = paired(V[-1], V[im])
            rows.append((s["t"], g1, e1, g2, e2))
        R = np.array(rows)
        a1.plot(R[:, 0], R[:, 1], color=col[l], lw=1.0, ls=":", marker="o", ms=2, label=f"{l}: N/4 -> N/2")
        a1.plot(R[:, 0], R[:, 3], color=col[l], lw=1.2, marker="o", ms=2.5, label=f"{l}: N/2 -> N")
        a2.plot(R[:, 0], R[:, 1] / R[:, 2], color=col[l], lw=1.0, ls=":", marker="o", ms=2, label=f"{l}: N/4 -> N/2")
        a2.plot(R[:, 0], R[:, 3] / R[:, 4], color=col[l], lw=1.2, marker="o", ms=2.5, label=f"{l}: N/2 -> N")
        late = R[R[:, 0] <= R[:, 0].min() + 0.2 * (R[:, 0].max() - R[:, 0].min())]
        report(f"{l}: validation change N/4 -> N/2 and N/2 -> N (EMA, paired; mean over slices with t in the last "
               f"fifth): {late[:, 1].mean():+.1f} and {late[:, 3].mean():+.1f}; all slices: "
               f"{R[:, 1].mean():+.1f} and {R[:, 3].mean():+.1f}; slices with N/2 -> N below -2 SE: "
               f"{int(np.sum(R[:, 3] / R[:, 4] < -2))} of {len(R)}")
    for ax, yl in ((a1, "validation change (loss units; negative = better)"), (a2, "validation change / SE")):
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_xlabel("diffusion time t of the slice", fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.invert_xaxis()
        style(ax)
    a2.axhline(-2.0, color="0.5", lw=0.6, ls=":")
    a1.legend(fontsize=6, frameon=False)
    fig.suptitle(f"{title}: validation gain in the second half (N/2 -> N) vs the quarter before (N/4 -> N/2)",
                 fontsize=9)
    save(fig, f"{name}_val_summary.png")
    return data


def do_pair(name, a, b, data, title):
    rows = []
    for k in sorted(set(data[a]) & set(data[b])):
        sa, sb = data[a][k], data[b][k]
        if "val" not in sa or "val" not in sb:
            continue
        _, ia, ib = np.intersect1d(sa["val_idx"], sb["val_idx"], return_indices=True)
        if len(ia) < 10:
            continue
        m, se = paired(sa["val"][ia], sb["val"][ib])
        rows.append((k, sa["t"], m, se, len(ia)))
    if not rows:
        report(f"pair {a} - {b}: no shared validation samples")
        return
    R = np.array(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.errorbar(R[:, 1], R[:, 2], yerr=2 * R[:, 3], fmt="o", ms=2.5, lw=0.8, color=CATEGORICAL[0])
    a2.plot(R[:, 1], R[:, 2] / R[:, 3], "o-", ms=2.5, lw=1.0, color=CATEGORICAL[0])
    for ax, yl in ((a1, f"final validation loss, {a} minus {b}"), (a2, "difference / SE")):
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_xlabel("diffusion time t of the slice", fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.invert_xaxis()
        style(ax)
    a2.axhline(-2.0, color="0.5", lw=0.6, ls=":")
    a2.axhline(2.0, color="0.5", lw=0.6, ls=":")
    fig.suptitle(f"{title}: {a} minus {b}, per slice (paired on {int(R[0, 4])} validation images; "
                 f"negative = {a} better)", fontsize=9)
    save(fig, f"{name}_pair.png")
    report(f"pair {a} - {b}: final validation loss difference, mean over slices {R[:, 2].mean():+.1f}; "
           f"slices with difference below -2 SE: {int(np.sum(R[:, 2] / R[:, 3] < -2))}, above +2 SE: "
           f"{int(np.sum(R[:, 2] / R[:, 3] > 2))}, of {len(R)}")
    for lo_k, hi_k in ((0, 29), (30, 79), (80, 99)):
        m = (R[:, 0] >= lo_k) & (R[:, 0] <= hi_k)
        if m.any():
            report(f"  slices {lo_k}-{hi_k}: mean difference {R[m, 2].mean():+.1f} (mean z {np.mean(R[m, 2] / R[m, 3]):+.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar="NAME LABEL=RUN",
                    help="a group name followed by LABEL=RUN entries (repeat for several groups)")
    ap.add_argument("--title", action="append", default=[], help="NAME:title text for a group's figures")
    ap.add_argument("--pair", action="append", default=[], help="NAME:A,B (labels in group NAME)")
    ap.add_argument("--mark", action="append", default=[], help="NAME:LABEL:FRACTION")
    ap.add_argument("--slices", default="0,10,20,30,40,50,60,70,80,90,95,99")
    ap.add_argument("--window", type=int, default=1000, help="running-mean window in training steps")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; plot_learning_curves; job "
           f"{os.environ.get('SLURM_JOB_ID', 'manual')}")
    sel = [int(v) for v in args.slices.split(",")]
    marks = {}
    for m in args.mark:
        g, l, f = m.rsplit(":", 2)
        marks[(g, l)] = float(f)
    titles = dict(t.split(":", 1) for t in args.title)
    for g in args.group:
        name, entries = g[0], g[1:]
        runs = []
        for e in entries:
            l, r = e.split("=", 1)
            if os.path.exists(os.path.join(OUT_ROOT, r, "train_log.jsonl")):
                runs.append((l, r))
            else:
                report(f"skip {l}: {r} has no train_log.jsonl")
        if not runs:
            continue
        data = do_group(name, runs, sel, args.window, marks, titles.get(name, name))
        for p in args.pair:
            pg, ab = p.split(":", 1)
            if pg == name:
                a, b = ab.split(",")
                if a in data and b in data:
                    do_pair(name, a, b, data, titles.get(name, name))
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
