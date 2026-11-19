"""Diagnostic (2026-11-19): what the forward marginal p_t looks like, and where the model's reverse trajectory departs
from it (the user: "at 2.2 how does the distribution look? is it already a gaussian more or less or still very
different?").

Part A, the data only (forward process x_t = e^-t x0 + sigma_t xi, sigma_t^2 = T (1 - e^-2t)):
  - per principal component k of the training images (eigenvalue lam_k, from sample_latent.data_stats): the
    signal-to-noise ratio SNR_k(t) = e^-2t lam_k / sigma_t^2; the number of components with SNR > 1 and the share of
    the signal variance they carry;
  - non-Gaussianity of the 1-D marginals of x_t along the leading components: excess kurtosis and the
    Kullback-Leibler divergence to the Gaussian of the same mean and variance (histogram estimate, nats);
  - how much a noised image still determines the face: for held-out test images noised to t, the posterior over a
    reference set of training images, p(i | x_t) ~ exp(-|x_t - e^-t x0_i|^2 / 2 sigma_t^2); its effective number
    exp(entropy) (1: the face is determined; the reference-set size: nothing is left).
  - examples: test faces noised to each t.
Part B (--snapshots FILE, the reverse-process states of sample_latent --snapshots): along the same components, the
  model's signal variance Var(U_k . x_t) - sigma_t^2 against the data's e^-2t lam_k, at the snapshot times, per group
  of components; and the excess kurtosis of the leading components, model vs data.

  python forward_marginal_check.py --problem celeba32g --out DIR [--snapshots RUN/samples/sde_base_nN_snapshots.npz]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import latent_model as lm  # noqa: E402

T_LIST = (0.0, 0.2, 0.71, 1.46, 2.22, 2.98, 3.48, 3.99, 5.0)
N_REF, N_QUERY, N_KURT, N_LEAD = 20000, 500, 20000, 10
GROUPS = ((0, 1), (1, 5), (5, 20), (20, 100), (100, 1024))   # component index ranges [lo, hi)


def sig2(t):
    return lm.TEMP * (1.0 - np.exp(-2.0 * t))


def kl_to_gauss(v, bins=80):
    """KL(p || N(mean, var)) of the 1-D sample v from a histogram (nats)."""
    m, s = v.mean(), v.std()
    h, edges = np.histogram(v, bins=bins, range=(m - 6 * s, m + 6 * s), density=True)
    c, w = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    g = np.exp(-0.5 * ((c - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    ok = h > 0
    return float(np.sum(h[ok] * np.log(h[ok] / g[ok]) * w[ok]))


def ex_kurt(v):
    z = (v - v.mean()) / v.std()
    return float(np.mean(z ** 4) - 3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problem", default="celeba32g")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshots", default="", help="Part B: a sample_latent --snapshots file (relative to OUT_ROOT)")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    import sample_latent as sl
    prob = lm.make_problem(args.problem, 1)
    data = np.asarray(prob["data"], dtype=np.float64)
    g, mean_px, std_px = prob["grid"], prob["mean"], prob["std"]
    st = sl.data_stats(args.problem)
    mu, lam, U = np.asarray(st["mu"]), np.asarray(st["lam"]), np.asarray(st["U"])
    out(f"problem {args.problem}: {data.shape[0]} training images of {data.shape[1]} pixels; T = {lm.TEMP}; total "
        f"variance {lam.sum():.1f} (normalised units); leading eigenvalues {', '.join(f'{v:.1f}' for v in lam[:5])}")
    rng = np.random.default_rng(0)

    # ---- Part A: SNR spectrum ----------------------------------------------------------------------------------
    out("\nA1. Signal-to-noise per principal component: SNR_k = e^-2t lam_k / sigma_t^2")
    out(f"{'t':>6} {'sigma_t':>8} {'SNR_1':>9} {'SNR_10':>8} {'SNR_100':>8} {'n(SNR>1)':>9} {'n(SNR>10)':>10} "
        f"{'signal var in SNR>1 comps':>26}")
    snr = {}
    for t in T_LIST[1:]:
        s2 = sig2(t)
        r = np.exp(-2 * t) * lam / s2
        snr[t] = r
        share = float(lam[r > 1].sum() / lam.sum())
        out(f"{t:6.2f} {np.sqrt(s2):8.4f} {r[0]:9.1f} {r[9]:8.2f} {r[99]:8.3f} {int((r > 1).sum()):9d} "
            f"{int((r > 10).sum()):10d} {100 * share:25.1f}%")

    # ---- Part A: non-Gaussianity of the leading components -----------------------------------------------------
    idx = rng.choice(data.shape[0], N_KURT, replace=False)
    A = (data[idx] - mu) @ U[:, :N_LEAD]                       # (n, N_LEAD) projections of x0
    xi = rng.standard_normal(A.shape)
    out(f"\nA2. Non-Gaussianity of x_t along the leading components ({N_KURT} training images): excess kurtosis and KL "
        f"to the moment-matched Gaussian (nats; histogram, floor about 0.003)")
    out(f"{'t':>6}  " + "  ".join(f"{'PC' + str(k + 1) + ' kurt/KL':>14}" for k in range(5)))
    kurt_data = {}
    for t in T_LIST:
        P = np.exp(-t) * A + np.sqrt(sig2(t)) * xi
        kurt_data[t] = [ex_kurt(P[:, k]) for k in range(N_LEAD)]
        out(f"{t:6.2f}  " + "  ".join(f"{ex_kurt(P[:, k]):+6.2f}/{kl_to_gauss(P[:, k]):6.3f}" for k in range(5)))

    # ---- Part A: how much x_t determines the face ------------------------------------------------------------------
    raw_te, _ = lm.load_celeba_raw("test", lm.celeba_grid(args.problem)) if args.problem.startswith("celeba") \
        else (None, None)
    if raw_te is not None:
        Q = (np.asarray(raw_te, np.float64)[rng.choice(raw_te.shape[0], N_QUERY, replace=False)] - mean_px) / std_px
    else:
        Q = data[rng.choice(data.shape[0], N_QUERY, replace=False)]
    R = data[rng.choice(data.shape[0], N_REF, replace=False)]
    out(f"\nA3. How much x_t determines the face: {N_QUERY} held-out test images noised to t, posterior over {N_REF} "
        f"training images; effective number exp(entropy) (1: determined; {N_REF}: nothing left)")
    out(f"{'t':>6} {'eff. number: median':>20} {'p10':>8} {'p90':>8} {'top-1 posterior median':>24}")
    eff_med = {}
    nr = np.sum(R * R, axis=1)
    for t in T_LIST[1:]:
        s2, e = sig2(t), np.exp(-t)
        Xt = e * Q + np.sqrt(s2) * rng.standard_normal(Q.shape)
        L = (2 * e * Xt @ R.T - e * e * nr[None, :]) / (2 * s2)     # log-likelihood up to a per-query constant
        L -= L.max(axis=1, keepdims=True)
        Pp = np.exp(L)
        Pp /= Pp.sum(axis=1, keepdims=True)
        H = -np.sum(np.where(Pp > 0, Pp * np.log(np.where(Pp > 0, Pp, 1.0)), 0.0), axis=1)
        eff = np.exp(H)
        eff_med[t] = float(np.median(eff))
        out(f"{t:6.2f} {np.median(eff):20.1f} {np.percentile(eff, 10):8.1f} {np.percentile(eff, 90):8.1f} "
            f"{np.median(Pp.max(axis=1)):24.3f}")

    # ---- Part B: the model's trajectory ------------------------------------------------------------------------------
    ratio = None
    if args.snapshots:
        z = np.load(os.path.join(lm.OUT_ROOT, args.snapshots))
        S, t_rev = np.asarray(z["states"], np.float64), np.asarray(z["t"])
        ok = np.all(np.isfinite(S.reshape(S.shape[0], -1)), axis=1)
        S = S[ok]
        t_fwd = lm.T_FORWARD - t_rev
        out(f"\nB. The model's reverse trajectory ({args.snapshots}, {S.shape[0]} samples): signal variance along the "
            f"principal components, Var(U_k . x_t) - sigma_t^2, over the data's e^-2t lam_k (1: as the forward "
            f"process; < 1: too little structure)")
        hdr = "  ".join(f"PC{lo + 1}-{hi}".rjust(10) for lo, hi in GROUPS)
        out(f"{'t':>6}  {hdr}   {'PC1-5 kurtosis model / data':>30}")
        ratio = []
        for j in range(S.shape[1]):
            t = float(t_fwd[j])
            Pm = (S[:, j, :] - np.exp(-t) * mu) @ U
            vm = Pm.var(axis=0) - sig2(t)
            vd = np.exp(-2 * t) * lam
            row = [float(vm[lo:hi].sum() / vd[lo:hi].sum()) for lo, hi in GROUPS]
            ratio.append((t, row))
            Pd = np.exp(-t) * A[:, :5] + np.sqrt(sig2(t)) * rng.standard_normal((A.shape[0], 5))
            km = [ex_kurt(Pm[:, k]) for k in range(5)]
            kd = [ex_kurt(Pd[:, k]) for k in range(5)]
            out(f"{t:6.2f}  " + "  ".join(f"{v:10.3f}" for v in row) + "   "
                + " ".join(f"{a:+.2f}/{b:+.2f}" for a, b in zip(km, kd)))
    out(f"\ntotal {time.time() - t0:.0f} s")
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figures -------------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    BLUE = ["#bcd4f2", "#8fb5e8", "#5f95dd", "#2a78d6", "#1f5aa5", "#163f75", "#0d2647", "#08172b"]   # light -> dark
    ts = T_LIST[1:]
    faces = Q[:8] if raw_te is not None else data[:8]
    fig, axes = plt.subplots(len(T_LIST), 8, figsize=(8 * 1.1, len(T_LIST) * 1.15), facecolor=SURF)
    for r, t in enumerate(T_LIST):
        Xt = np.exp(-t) * faces + np.sqrt(sig2(t)) * rng.standard_normal(faces.shape)
        Xs = Xt * np.exp(t) if t > 0 else Xt                     # rescaled by e^t, so the signal keeps its size
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(np.clip(Xs[c] * std_px + mean_px, 0, 1).reshape(g, g), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[r, 0].set_ylabel(f"t = {t:g}", fontsize=8, color=INK)
    fig.suptitle("Held-out faces noised to forward time t, shown as e^t x_t (signal at its original size)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "noised_examples.png"), dpi=110, facecolor=SURF)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3 if ratio else 2, figsize=(15 if ratio else 10.5, 4.2), facecolor=SURF)
    ax = axes[0]
    k = np.arange(1, lam.size + 1)
    for i, t in enumerate(ts):
        ax.loglog(k, snr[t], color=BLUE[min(i, len(BLUE) - 1)], lw=2, label=f"t = {t:g}")
    ax.axhline(1.0, color=INK2, ls="--", lw=1)
    ax.text(1.2, 1.15, "signal = noise", fontsize=8, color=INK2)
    ax.set_xlabel("principal component k of the training faces")
    ax.set_ylabel("signal-to-noise ratio of x_t along k")
    ax.set_title("Which components still carry the face", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax = axes[1]
    nsig = [int((snr[t] > 1).sum()) for t in ts]
    ax.plot(ts, nsig, color=BLUE[3], lw=2, marker="o", ms=6, label="components with SNR > 1")
    if eff_med:
        ax.plot(ts, [eff_med[t] for t in ts], color="#eb6834", lw=2, marker="o", ms=6,
                label=f"training faces consistent with x_t (of {N_REF})")
    ax.set_yscale("log")
    ax.axvline(2.22, color=INK2, ls=":", lw=1)
    ax.text(2.27, 1.3, "patches on", fontsize=8, color=INK2)
    ax.set_xlabel("forward time t")
    ax.set_title("How much of the face is left at time t", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False)
    if ratio:
        ax = axes[2]
        tt = [r[0] for r in ratio]
        for i, (lo, hi) in enumerate(GROUPS):
            ax.plot(tt, [r[1][i] for r in ratio], lw=2, marker="o", ms=5, color=BLUE[min(2 * i, len(BLUE) - 1)],
                    label=f"PC {lo + 1}-{hi}")
        ax.axhline(1.0, color=INK2, ls="--", lw=1)
        ax.axvline(2.22, color=INK2, ls=":", lw=1)
        ax.set_xlabel("forward time t (reverse process runs right to left)")
        ax.set_ylabel("model signal variance / data")
        ax.set_title("Structure in the model's x_t vs the forward process", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "forward_marginal.png"), dpi=130, facecolor=SURF)
    plt.close(fig)
    with open(os.path.join(args.out, "README.txt"), "w") as f:
        f.write(__doc__ + f"\nJob: {os.environ.get('SLURM_JOB_ID', 'manual')}\n")


if __name__ == "__main__":
    main()
