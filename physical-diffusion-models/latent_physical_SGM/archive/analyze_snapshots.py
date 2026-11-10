#!/usr/bin/env python3
"""When does generation commit to a class? Follows one Ising hidden unit along sampled trajectories.

Usage:  python analyze_snapshots.py RUN [SNAPSHOT_FILE]
  RUN: a K = 1 Ising run under out/latent_physical_SGM; SNAPSHOT_FILE defaults to samples/sde_base_n*_snapshots.npz
       (written by sample_latent.py --snapshots).
For every trajectory and snapshot time t (forward time tau = 5 - t): a = b(tau) + W(tau) . x_t and m = tanh(a / T),
with W, b interpolated linearly between the 100 knots as in the sampler. Reports the first t at which |m| > 0.9 and
stays there, how often the sign of m changes after that, and whether the final sign agrees with the final image's
class (Fisher class score of the final state, as in evaluate_samples.py).
Writes out/latent_physical_SGM/snapshots/<job id>/: report.txt, commitment.png. CPU only, seconds.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import evaluate_samples as ev  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "snapshots", os.environ.get("SLURM_JOB_ID", "manual"))


def main():
    run = sys.argv[1]
    run_dir = os.path.join(lm.OUT_ROOT, run)
    snap_path = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(run_dir, "samples", "*_snapshots.npz")))[0]
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    snap = np.load(snap_path)
    X, ts = snap["states"].astype(np.float64), snap["t"]          # (n, n_t, N), (n_t,)
    hid = np.load(os.path.join(run_dir, "hidden_history.npz"))
    W_h, b_h, taus = hid["W"][:, 0, :], hid["b"][:, 0], hid["t"]
    order = np.argsort(taus)
    tau_s, W_s, b_s = taus[order], W_h[order], b_h[order]
    m = np.empty(X.shape[:2])
    for i, t in enumerate(ts):
        tau = lm.T_FORWARD - t
        W = np.array([np.interp(tau, tau_s, W_s[:, j]) for j in range(W_s.shape[1])])
        b = np.interp(tau, tau_s, b_s)
        m[:, i] = np.tanh((X[:, i, :] @ W + b) / lm.TEMP)
    ref = ev.reference("mnist28")
    r, d = ev.classify(ref, X[:, -1, :])
    final_cls = d["cls1"].astype(int)
    ok = ~d["div"]
    committed = np.abs(m) > 0.9
    t_commit = np.full(X.shape[0], np.nan)
    for k in range(X.shape[0]):
        for i in range(len(ts)):
            if committed[k, i:].all():
                t_commit[k] = ts[i]
                break
    flips = np.array([np.sum(np.diff(np.sign(m[k][committed[k]])) != 0) for k in range(X.shape[0])])
    agree = (np.sign(m[:, -1]) > 0) == (final_cls == 1)
    out(f"run {run}; {X.shape[0]} trajectories, snapshots at t = {np.round(ts, 2).tolist()}")
    out(f"share with |m| > 0.9 at each snapshot: {np.round(np.mean(committed, axis=0), 3).tolist()}")
    out(f"share of trajectories committed (|m| > 0.9 for good) by t: "
        f"{[(float(t), round(float(np.mean(t_commit <= t + 1e-9)), 3)) for t in ts]}")
    out(f"median commitment time t {np.nanmedian(t_commit):.2f} (forward time tau {lm.T_FORWARD - np.nanmedian(t_commit):.2f}); "
        f"never committed {int(np.sum(np.isnan(t_commit)))}")
    out(f"sign flips while committed: {np.bincount(flips).tolist()} (count of trajectories with 0, 1, ... flips)")
    out(f"final sign of m agrees with the final image's class for {np.mean(agree[ok]):.3f} of non-diverged samples")
    for lab, mask in (("blend", d["blend"]), ("not blend", ok & ~d["blend"])):
        if mask.any():
            out(f"  {lab}: n {int(mask.sum())}, median commitment t {np.nanmedian(t_commit[mask]):.2f}, "
                f"never committed {int(np.sum(np.isnan(t_commit[mask])))}, |m| at the end {np.mean(np.abs(m[mask, -1])):.3f}")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k in range(min(X.shape[0], 200)):
        ax.plot(lm.T_FORWARD - ts, m[k], color="#2a78d6" if final_cls[k] else "#eb6834", alpha=0.25, linewidth=0.8)
    ax.set_xlim(5.05, -0.05)
    ax.set_xlabel("forward time tau (generation runs left to right)")
    ax.set_ylabel("hidden unit m = tanh(a / T)")
    ax.set_title(f"{run}: hidden unit along generation (blue: final image a 1, orange: a 0)", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "commitment.png"), dpi=140)
    plt.close(fig)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
