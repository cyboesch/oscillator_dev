#!/usr/bin/env python3
"""Are the samplers' failures 0/1 blends, or badly formed single digits?

Classifies existing samples of the paper checkpoint (no new sampling):
  diverged     max|x| > 15 in normalised units
  blend        adding a template of the other class explains the sample much better than any one template
  clean 0 / 1  otherwise, nearest training image within the 'far' threshold (4.84, as in the ode_vs_sde study)
  malformed    otherwise, beyond that threshold
Labels: the 11,846 training images are labelled exactly, by matching their pixels against the ones-only file.
Blend score G = 1 - r2^2 / r1^2:
  r1 = best single template among the K_NN nearest 0s and K_NN nearest 1s, with a non-negative scale;
  r2 = best non-negative combination of one of those 0s and one of those 1s.
The G threshold is the (1 - FP_RATE) quantile over held-out training images (probes), so FP_RATE of real images
count as blends by construction; what matters is the excess over that rate.
Class side: ridge-regularised Fisher discriminant between 0s and 1s, scaled so the class means sit at 0 and 1,
fitted on the training images that are not probes.
Read-only; CPU only. Writes out/diagnostics/blend_check/<job id>/.
Run under Slurm (cd physical-diffusion-models/latent_physical_SGM && sbatch run_latent_checks_neuronic.sh).
"""

import json
import os
import re
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.append(PKG)
from physical_diffusion_fns.helper_fns import normalize_samples  # noqa: E402

# Must match scripts/physical_diffusion_model_MNIST.py and the ode_vs_sde study
KEY_SEED = 1
STD_ADDED_NOISE = 0.01
FAR_THRESHOLD = 4.84        # p90 of the held-out nearest-image distance in the ode_vs_sde study (seed 11)
MAX_ABS = 15.0
# This check
K_NN = 10
N_PROBE = 1000
PROBE_SEED = 3
FP_RATE = 0.05
CHUNK = 500

DATA_PATH = os.path.join(PKG, "data", "MNIST", "mnist_labels_[0, 1]_resolution_(28, 28)_balanced_True.npy")
ONES_PATH = os.path.join(PKG, "data", "MNIST", "mnist_labels_[1]_resolution_(28, 28).npy")
RUN_DIR = os.path.join(
    PKG, "out", "2026_07_08", "problems", "MNIST_generation",
    "labels_[0, 1]_resolution_28_x_28",
    "n_couplings_14_Temp_0.02_energy_fn_type_6th_order_duffing_coupling",
    "added_gaussian_noise_std_0.01_key_seed_1_additional_rescaling_1",
    "training_method_SM_at_kbT_minimal_memory_solve_opt_in_reverse_True_exp_time_pts_False_"
    "t_forward_5.0_n_timesteps_100_sigma_forward_1.0_lr_2e-05_to_0.002_epochs_100000_batch_512_"
    "window_1000_tol_0.001_to_0.001_patience_300_to_100_",
    "opt_per_time_plots",
)
ABLATION_DIR = os.path.join(RUN_DIR, "interpolation_ablation_v1")
SAVED_ODE = "final_states_ablation_params_raw_interp_linear_step_knots_True_n40_matched_rng_probability_flow_ode_atol_1e-09_rtol_1e-07.npy"
SAVED_SDE = "final_states_ablation_params_raw_interp_linear_step_knots_True_n40_matched_rng_sde_atol_1e-06_rtol_0.0001_btol_1e-12.npy"
FINE_NPZ = os.path.join(PKG, "out", "diagnostics", "ode_noise_window_fine", "4183779", "final_states.npz")
UNSPLIT = os.path.join(PKG, "out", "diagnostics", "sde_split_control", "4187940", "final_states_sde_unsplit.npy")
OUT_DIR = os.path.join(PKG, "out", "diagnostics", "blend_check", os.environ.get("SLURM_JOB_ID", "manual"))


def report(line=""):
    print(line, flush=True)
    with open(os.path.join(OUT_DIR, "report.txt"), "a") as f:
        f.write(line + "\n")


def pretty(key):
    m = re.match(r"^SDE_([\d.]+)_([\d.]+)$", key)
    if m:
        return f"SDE[{m.group(1)},{m.group(2)}]"
    return key.replace("_then_", ">").replace("_at_", "@")


def scores(Q, px_train, tr_n2, labels, self_idx=None):
    """Nearest-image distance and blend score G for query images Q (pixel space)."""
    n = Q.shape[0]
    nn = np.empty(n)
    G = np.empty(n)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    for c0 in range(0, n, CHUNK):
        X = Q[c0:c0 + CHUNK]
        xx = np.sum(X**2, axis=1)
        d2 = xx[:, None] + tr_n2[None, :] - 2.0 * X @ px_train.T
        if self_idx is not None:                       # leave the probe itself out
            d2[np.arange(X.shape[0]), self_idx[c0:c0 + CHUNK]] = np.inf
        nn[c0:c0 + CHUNK] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        nb0 = idx0[np.argpartition(d2[:, idx0], K_NN, axis=1)[:, :K_NN]]
        nb1 = idx1[np.argpartition(d2[:, idx1], K_NN, axis=1)[:, :K_NN]]
        Z0, Z1 = px_train[nb0], px_train[nb1]                     # (c, k, 784)
        xz0, xz1 = np.einsum("cd,ckd->ck", X, Z0), np.einsum("cd,ckd->ck", X, Z1)
        zz0, zz1 = tr_n2[nb0], tr_n2[nb1]
        # best single template with a non-negative scale: r^2 = |x|^2 - max(x.z, 0)^2 / |z|^2
        r1 = xx - np.max(np.concatenate([np.maximum(xz0, 0)**2 / zz0, np.maximum(xz1, 0)**2 / zz1], axis=1), axis=1)
        # best pair (one 0, one 1), both weights >= 0
        C = np.einsum("cad,cbd->cab", Z0, Z1)
        A, B = zz0[:, :, None], zz1[:, None, :]
        c0v, c1v = xz0[:, :, None], xz1[:, None, :]
        det = A * B - C**2
        with np.errstate(divide="ignore", invalid="ignore"):
            al = (B * c0v - C * c1v) / det
            be = (A * c1v - C * c0v) / det
            r2p = xx[:, None, None] - (al * c0v + be * c1v)
        valid = (al >= 0) & (be >= 0) & (det > 0)
        r2p = np.where(valid, r2p, np.inf).reshape(X.shape[0], -1)
        best = np.argmin(r2p, axis=1)
        r2 = np.minimum(r1, r2p[np.arange(X.shape[0]), best])
        G[c0:c0 + CHUNK] = 1.0 - np.maximum(r2, 0.0) / np.maximum(r1, 1e-300)
    return nn, G


def frac(mask, denom=None):
    k = int(np.sum(mask))
    n = int(mask.size if denom is None else np.sum(denom))
    p = k / n if n else float("nan")
    return p, (np.sqrt(p * (1 - p) / n) if n else float("nan")), k, n


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; blend check; jax {jax.__version__}")

    # Training images, labels and the normalisation used everywhere else
    raw = np.load(DATA_PATH)
    ones = np.load(ONES_PATH)
    as_bytes = lambda a: [r.tobytes() for r in np.round(np.asarray(a) * 255).astype(np.uint8)]
    ones_set = set(as_bytes(ones))
    labels = np.array([r in ones_set for r in as_bytes(raw)], dtype=int)
    n_unique = len(set(as_bytes(raw)))
    report(f"training images {raw.shape[0]} ({n_unique} distinct); labelled 1 by exact match: {labels.sum()}, "
           f"0: {(labels == 0).sum()} (expected 5923 / 5923)")
    rawj = jnp.array(raw)
    _, _, image_noise_added_key = jr.split(jr.PRNGKey(KEY_SEED), 3)
    _, subkey = jr.split(image_noise_added_key)
    data, mean_px, std_px = normalize_samples(rawj + jr.normal(subkey, rawj.shape) * STD_ADDED_NOISE)
    data, mean_px, std_px = np.asarray(data), float(mean_px), float(std_px)
    px_train = data * std_px + mean_px
    tr_n2 = np.sum(px_train**2, axis=1)
    report(f"normalisation: mean {mean_px:.5f}, std {std_px:.5f} (CODE_NOTES §4: 0.12478, 0.30467)")

    rng = np.random.default_rng(PROBE_SEED)
    probe = np.sort(rng.choice(raw.shape[0], size=N_PROBE, replace=False))
    is_probe = np.zeros(raw.shape[0], bool)
    is_probe[probe] = True

    # Fisher discriminant on the non-probe images
    X0, X1 = px_train[~is_probe & (labels == 0)], px_train[~is_probe & (labels == 1)]
    mu0, mu1 = X0.mean(0), X1.mean(0)
    Sw = ((X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)) / (X0.shape[0] + X1.shape[0] - 2)
    lam = 1e-2 * np.trace(Sw) / Sw.shape[0]
    w = np.linalg.solve(Sw + lam * np.eye(Sw.shape[0]), mu1 - mu0)
    lda = lambda P: (P @ w - mu0 @ w) / ((mu1 - mu0) @ w)

    # Reference: held-out training images
    nn_p, G_p = scores(px_train[probe], px_train, tr_n2, labels, self_idx=probe)
    s_p = lda(px_train[probe])
    lab_p = labels[probe]
    G_thr = float(np.quantile(G_p, 1.0 - FP_RATE))
    q0_hi, q1_lo = float(np.quantile(s_p[lab_p == 0], 0.99)), float(np.quantile(s_p[lab_p == 1], 0.01))
    acc = float(np.mean((s_p >= 0.5) == (lab_p == 1)))
    report(f"probes: {N_PROBE} held-out training images ({lab_p.sum()} ones); Fisher accuracy {acc:.4f}; "
           f"class-score 99% of 0s below {q0_hi:.3f}, 99% of 1s above {q1_lo:.3f}")
    report(f"probes: nearest-image distance p90 {np.percentile(nn_p, 90):.2f} (ode_vs_sde threshold {FAR_THRESHOLD}); "
           f"blend-score threshold G > {G_thr:.3f} ({100 * FP_RATE:.0f}% of real images by construction)")

    # Sample sets
    sets = [("data (probes)", None)]
    sets.append(("SDE, one tree (4187940)", np.load(UNSPLIT)))
    fine = np.load(FINE_NPZ)
    fine_keys = list(fine.files)
    for k in ("ODE", "SDE"):
        sets.append((f"{k} (4183779)", fine[k]))
    sets.append(("SDE paper (40)", np.load(os.path.join(ABLATION_DIR, SAVED_SDE))))
    sets.append(("ODE paper (40)", np.load(os.path.join(ABLATION_DIR, SAVED_ODE))))
    for k in fine_keys:
        if k not in ("ODE", "SDE"):
            sets.append((pretty(k), fine[k]))

    rows, keep = {}, {}
    hdr = (f"{'sample set':<26}{'n':>5}{'%div':>6}{'%blend':>8}{'+-':>5}{'%gap':>6}{'%clean0':>8}{'%clean1':>8}"
           f"{'%malf0':>7}{'%malf1':>7}{'%far':>6}{'blend/far':>10}{'%class1':>8}")
    report("")
    report("blend = G above the probe threshold (5% of real images by construction); gap = Fisher class score between "
           "the 99% ranges of 0s and 1s; far = nearest image > 4.84 or diverged; blend/far = share of the far, "
           "non-diverged samples that are blends; class1 = share of 1s among non-diverged non-blends")
    report(hdr)
    for name, x in sets:
        if x is None:
            P, nn, G, s = px_train[probe], nn_p, G_p, s_p
            div = np.zeros(N_PROBE, bool)
        else:
            x = np.asarray(x, dtype=np.float64)
            div = ~(np.max(np.abs(x), axis=1) <= MAX_ABS)       # NaN rows count as diverged
            P = x * std_px + mean_px
            nn, G = scores(P, px_train, tr_n2, labels)
            s = lda(P)
        ok = ~div
        blend = ok & (G > G_thr)
        far = div | (nn > FAR_THRESHOLD)
        cls1 = s >= 0.5
        gap = ok & (s > q0_hi) & (s < q1_lo)
        rest = ok & ~blend
        r = {
            "n": int(len(nn)),
            "diverged": frac(div), "blend": frac(blend), "gap": frac(gap),
            "clean0": frac(rest & ~far & ~cls1), "clean1": frac(rest & ~far & cls1),
            "malformed0": frac(rest & far & ~cls1), "malformed1": frac(rest & far & cls1),
            "far": frac(far), "blend_share_of_far": frac(blend & far, denom=far & ok),
            "class1_share": frac(rest & cls1, denom=rest),
        }
        rows[name] = r
        keep[name] = dict(P=P, nn=nn, G=G, s=s, div=div, blend=blend, far=far, cls1=cls1)
        pc = lambda key: 100 * r[key][0]
        report(f"{name:<26}{r['n']:>5}{pc('diverged'):>6.1f}{pc('blend'):>8.1f}{100 * r['blend'][1]:>5.1f}{pc('gap'):>6.1f}"
               f"{pc('clean0'):>8.1f}{pc('clean1'):>8.1f}{pc('malformed0'):>7.1f}{pc('malformed1'):>7.1f}{pc('far'):>6.1f}"
               f"{pc('blend_share_of_far'):>10.1f}{pc('class1_share'):>8.1f}")
    report("")
    report("sanity: % far should reproduce the ode_vs_sde study: SDE one tree 35.6, ODE 47.6, SDE 34.6, SDE paper 22.5")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"G_threshold": G_thr, "class_score_q0_99": q0_hi, "class_score_q1_01": q1_lo, "fisher_accuracy": acc,
                   "probe_nn_p90": float(np.percentile(nn_p, 90)), "n_labelled_ones": int(labels.sum()),
                   "rows": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                            for k, v in rows.items()}}, f, indent=2)

    # Examples: random members of each category, for the two main samplers and the data
    cats = [("clean 0", lambda d: ~d["div"] & ~d["blend"] & ~d["far"] & ~d["cls1"]),
            ("clean 1", lambda d: ~d["div"] & ~d["blend"] & ~d["far"] & d["cls1"]),
            ("blend", lambda d: d["blend"]),
            ("malformed 0", lambda d: ~d["div"] & ~d["blend"] & d["far"] & ~d["cls1"]),
            ("malformed 1", lambda d: ~d["div"] & ~d["blend"] & d["far"] & d["cls1"])]
    show = ["data (probes)", "SDE, one tree (4187940)", "ODE (4183779)"]
    cols = 8
    fig, axes = plt.subplots(len(cats) * len(show), cols, figsize=(1.1 * cols, 1.1 * len(cats) * len(show)))
    pick = np.random.default_rng(0)
    for si, name in enumerate(show):
        d = keep[name]
        for ci, (cname, sel) in enumerate(cats):
            idx = np.where(sel(d))[0]
            idx = pick.choice(idx, size=min(cols, idx.size), replace=False) if idx.size else idx
            row = si * len(cats) + ci
            for c in range(cols):
                ax = axes[row, c]
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if c < idx.size:
                    ax.imshow(d["P"][idx[c]].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_ylabel(f"{name}\n{cname}", rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("Random members of each category; pixel space, fixed limits [0, 1]", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "blend_examples.png"), dpi=120)
    plt.close(fig)

    # Where samples sit: class score vs blend score
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, color, label in [("data (probes)", "#8c8c8c", "held-out data"),
                               ("SDE, one tree (4187940)", "#2a78d6", "SDE (500)"),
                               ("ODE (4183779)", "#eb6834", "ODE (500)")]:
        d = keep[name]
        m = ~d["div"]
        ax.scatter(np.clip(d["s"][m], -1, 2), d["G"][m], s=8, color=color, alpha=0.6, linewidths=0, label=label)
    ax.axhline(G_thr, color="#555555", linewidth=1, linestyle="--")
    ax.text(0.01, G_thr + 0.01, "blend threshold", fontsize=8, color="#555555", transform=ax.get_yaxis_transform())
    for q in (q0_hi, q1_lo):
        ax.axvline(q, color="#bbbbbb", linewidth=1, linestyle=":")
    ax.set_xlabel("Fisher class score (0 = mean zero, 1 = mean one; clipped to [-1, 2])")
    ax.set_ylabel("blend score G (gain from a second, other-class template)")
    ax.set_title("Diverged samples omitted", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "blend_scatter.png"), dpi=150)
    plt.close(fig)
    report(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
