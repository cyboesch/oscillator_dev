#!/usr/bin/env python3
"""What do the Potts states of a trained run represent, slice by slice?

For each run and slice: data x ~ p_tau (the training distribution of that slice at the run's slice time tau, i.e. on
its --time-grid; fixed key per slice index), the state posteriors
p(m | x), and for the states that carry the most data (by mean occupation):
  * the mean training image (clean, t = 0 pixels) of the samples whose most likely state is m, and its class mix;
  * the coupling W_m and, with the spring, Q_m (as images).
Summary numbers per slice: effective number of states exp(H(pbar)), share of the data in the top states, class
purity (share of argmax-assigned samples belonging to the state's majority class), mean max p.

Usage (CPU Slurm job): python inspect_states.py RUN[:SLICE,SLICE,...] ... [--top 16]
Writes out/latent_physical_SGM/states/<job>/<run>_s<slice>.png and summary.txt.
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from physical_diffusion_fns.helper_fns import sample_forward_process  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "states", os.environ.get("SLURM_JOB_ID", "manual"))
N_DATA = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="RUN or RUN:SLICE,SLICE (default slices 50,75,99)")
    ap.add_argument("--top", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lines = [f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; inspect_states; job {os.environ.get('SLURM_JOB_ID', 'manual')}"]
    probs = {}
    for spec in args.runs:
        run, _, sl = spec.partition(":")
        slices = [int(s) for s in sl.split(",")] if sl else [50, 75, 99]
        with open(os.path.join(lm.OUT_ROOT, run, "config.json")) as f:
            cfg = json.load(f)
        if cfg.get("hidden") != "potts" or cfg.get("K", 0) == 0:
            lines.append(f"{run}: not a Potts run; skipped")
            continue
        # the run's slice times (train_latent.py --time-grid; 'uniform' = lm.forward_time_pts() for older runs),
        # checked against its hidden_history.npz 't' when the run is finished
        tgrid, taus = lm.run_time_grid(os.path.join(lm.OUT_ROOT, run), cfg)
        pname = cfg.get("problem", "mnist28")
        if pname not in probs:
            probs[pname] = lm.make_problem(pname, cfg.get("radius"), cfg.get("neighbourhood", "chebyshev"))
        prob = probs[pname]
        data, labels, grid = prob["data"], prob["labels"], prob["grid"]
        clean_px = np.asarray(data) * prob["std"] + prob["mean"]
        n_cls = int(labels.max()) + 1
        lines.append("")
        lines.append(f"{run} ({pname}, {cfg['K']} states, quad {cfg.get('quad', False)}, local {cfg.get('local', False)}, "
                     f"fixed {cfg.get('hidden_fixed', False)}" + (f", time grid {tgrid}" if tgrid != "uniform" else "") + ")")
        for k in slices:
            z = np.load(os.path.join(lm.OUT_ROOT, run, "slices", f"slice_{k:03d}.npz"))
            W, b = z["W"], z["b"]
            Q = z["Q"] if "Q" in z.files else None
            t = float(taus[k])
            kidx, knoise = jr.split(jr.fold_in(jr.PRNGKey(123), k))
            idx = np.asarray(jr.choice(kidx, data.shape[0], shape=(N_DATA,), replace=False))
            e = np.exp(-t)
            X = e * np.asarray(data)[idx] + np.sqrt(lm.TEMP * (1 - e * e)) * np.asarray(jr.normal(knoise, (N_DATA, data.shape[1])))
            A = X @ W.T + b + ((X * X) @ Q.T if Q is not None else 0.0)
            A = A / lm.TEMP
            A = A - A.max(axis=1, keepdims=True)
            p = np.exp(A)
            p /= p.sum(axis=1, keepdims=True)
            pbar = p.mean(axis=0)
            eff = float(np.exp(-np.sum(pbar * np.log(pbar + 1e-300))))
            top = np.argsort(-pbar)[:args.top]
            arg = p.argmax(axis=1)
            lab = labels[idx]
            purity = []
            state_cls = -np.ones(W.shape[0], int)          # majority class of the samples a state wins; -1 if none
            for m in range(W.shape[0]):
                sel = arg == m
                if sel.sum():
                    cnt = np.bincount(lab[sel], minlength=n_cls)
                    purity.append((sel.sum(), cnt.max()))
                    state_cls[m] = int(cnt.argmax())
            pur = sum(c for _, c in purity) / max(sum(n for n, _ in purity), 1)
            per_cls = np.bincount(state_cls[state_cls >= 0], minlength=n_cls)
            acc = [float(np.mean(state_cls[arg[lab == c]] == c)) for c in range(n_cls)]
            lines.append(f"  slice {k} (t = {t:.3f}): effective states {eff:.1f}; data in the top {args.top} states "
                         f"{pbar[top].sum():.2f}; states used as argmax {len(purity)}; class purity of the argmax "
                         f"assignment {pur:.3f}; mean max p {p.max(axis=1).mean():.3f}")
            lines.append(f"    states per majority class {'/'.join(str(v) for v in per_cls)}; share of each class's "
                         f"samples won by a state of its own class {'/'.join(f'{a:.3f}' for a in acc)}")
            # occupation in the log domain (exp underflows): log10 of each state's mean occupation over the data and
            # of its largest posterior on any data point; a sampled rule sees a state only through its draws
            logp = A - np.log(np.exp(A).sum(axis=1, keepdims=True))
            lmax = logp.max(axis=0)
            lbar = (lmax + np.log(np.mean(np.exp(logp - lmax), axis=0))) / np.log(10.0)
            lmax = lmax / np.log(10.0)
            edges = [0.0, -2.0, -4.0, -8.0, -16.0, -32.0, -np.inf]
            occ = [int(np.sum((lbar < hi) & (lbar >= lo))) for hi, lo in zip([np.inf] + edges[1:-1], edges[1:])]
            lines.append(f"    states by log10 mean occupation [>= -2, -4, -8, -16, -32, below]: "
                         f"{'/'.join(str(v) for v in occ)}; largest posterior on any data point below 1e-6: "
                         f"{int(np.sum(lmax < -6))} states (min log10 {lmax.min():.0f})")
            rows = 3 if Q is not None else 2
            fig, axes = plt.subplots(rows, args.top, figsize=(1.3 * args.top, 1.45 * rows), squeeze=False)
            for j, m in enumerate(top):
                sel = arg == m
                img = clean_px[idx][sel].mean(axis=0) if sel.any() else np.zeros(grid * grid)
                mix = np.bincount(lab[sel], minlength=n_cls) if sel.any() else np.zeros(n_cls, int)
                axes[0, j].imshow(img.reshape(grid, grid), cmap="gray", vmin=0, vmax=1)
                axes[0, j].set_title(f"{100 * pbar[m]:.1f}% " + "/".join(str(v) for v in mix), fontsize=6)
                v = np.abs(W[m]).max() + 1e-12
                axes[1, j].imshow(W[m].reshape(grid, grid), cmap="RdBu_r", vmin=-v, vmax=v)
                if Q is not None:
                    vq = np.abs(Q[m]).max() + 1e-12
                    axes[2, j].imshow(Q[m].reshape(grid, grid), cmap="RdBu_r", vmin=-vq, vmax=vq)
            for ax in axes.ravel():
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0, 0].set_ylabel("data\n(mean)", fontsize=7)
            axes[1, 0].set_ylabel("W_m", fontsize=7)
            if Q is not None:
                axes[2, 0].set_ylabel("Q_m", fontsize=7)
            fig.suptitle(f"{run}, slice {k} (t = {t:.2f}): effective states {eff:.1f}; titles: occupation, class counts",
                         fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"{run}_s{k:03d}.png"), dpi=110)
            plt.close(fig)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
