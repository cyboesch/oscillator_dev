#!/usr/bin/env python3
"""Do the Potts states just recall prototypes (a dense associative memory), or do the samples vary like real images?

For each trained Potts run, under its t = 0 couplings (slice 99): fields a_m(x) = b_m + W_m.x + Q_m.x^2 (normalised
units), posterior p(m | x) = softmax(a / T), and the most likely state argmax_m a_m(x) of
  * the training images (minus the evaluation's 1,000 probes; blend_check.PROBE_SEED), which define each state's
    cluster: its members and their mean image (the "prototype"),
  * the probes (real images the clusters are not built from: the baseline),
  * the generated samples (<run>/samples/STEM.npy, non-finite rows dropped).
Distances are Euclidean in pixel units (x * std + mean), as in evaluate_samples.py. Reported:
  * distance to the own state's prototype: samples against probes (median, and the ratio). Samples much closer
    than probes = "prototype plus blur"; about equal = they vary around the prototype like real images;
  * within-state spread: mean pairwise distance of the samples assigned to a state, against the same number of
    that state's training members (drawn at random), pooled over states with >= 3 samples;
  * whether the nearest training image of a sample lies in its own state (against the probes' share);
  * posterior sharpness: mean max_m p(m | x) and mean entropy, samples against probes (low max p = between states);
  * state usage: effective number of states exp(H) of the samples' assignments against the training images'.
Figure per run: for the 8 most used states, the prototype, 6 training members and up to 6 samples of that state.

Usage (CPU Slurm job): python prototype_test.py RUN ... [--samples sde_base_n500]
Writes out/latent_physical_SGM/prototype_test/<job>/<run>.png and summary.txt.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import blend_check as bc   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "prototype_test", os.environ.get("SLURM_JOB_ID", "manual"))


def fields(p, X):
    A = X @ p["W"].T + p["b"]
    if p.get("Q") is not None:
        A = A + (X * X) @ p["Q"].T
    return A


def posterior_stats(A, T):
    Z = (A - A.max(1, keepdims=True)) / T
    P = np.exp(Z)
    P /= P.sum(1, keepdims=True)
    H = -np.sum(np.where(P > 0, P * np.log(np.maximum(P, 1e-300)), 0.0), 1)
    return P.max(1), H


def pdist_mean(Y):
    n = Y.shape[0]
    if n < 2:
        return np.nan
    n2 = np.sum(Y * Y, 1)
    D2 = np.maximum(n2[:, None] + n2[None, :] - 2.0 * Y @ Y.T, 0.0)
    return float(np.sqrt(D2)[np.triu_indices(n, 1)].mean())


def eff(assign, K):
    c = np.bincount(assign, minlength=K).astype(float)
    p = c[c > 0] / c.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def one_run(name, stem, rng, lines):
    rdir = os.path.join(lm.OUT_ROOT, name)
    cfg = json.load(open(os.path.join(rdir, "config.json")))
    if cfg.get("hidden") != "potts":
        lines.append(f"{name}: not a Potts run, skipped")
        return
    prob = lm.make_problem(cfg["problem"])
    X = np.asarray(prob["data"], dtype=np.float64)
    mean_px, std_px = prob["mean"], prob["std"]
    n, N = X.shape
    g = int(round(np.sqrt(N)))
    s = np.load(os.path.join(rdir, "slices", "slice_099.npz"))
    p = {"W": np.asarray(s["W"], np.float64), "b": np.asarray(s["b"], np.float64),
         "Q": np.asarray(s["Q"], np.float64) if "Q" in s.files else None}
    K = p["W"].shape[0]
    S = np.load(os.path.join(rdir, "samples", f"{stem}.npy")).astype(np.float64)
    S = S[np.all(np.isfinite(S), 1)]
    prng = np.random.default_rng(bc.PROBE_SEED)                 # the evaluation's probes
    probe = np.sort(prng.choice(n, size=bc.N_PROBE, replace=False))
    is_probe = np.zeros(n, bool)
    is_probe[probe] = True
    Xtr, Xpr = X[~is_probe], X[is_probe]
    T = lm.TEMP
    a_tr, a_pr, a_s = fields(p, Xtr), fields(p, Xpr), fields(p, S)
    m_tr, m_pr, m_s = a_tr.argmax(1), a_pr.argmax(1), a_s.argmax(1)
    px = lambda Z: Z * std_px + mean_px                         # noqa: E731
    Ptr, Ppr, Ps = px(Xtr), px(Xpr), px(S)
    counts = np.bincount(m_tr, minlength=K)
    C = np.zeros((K, N))
    for m in np.flatnonzero(counts):
        C[m] = Ptr[m_tr == m].mean(0)
    ok = counts >= 5                                            # states with a usable cluster
    d_s = np.linalg.norm(Ps - C[m_s], axis=1)[ok[m_s]]
    d_p = np.linalg.norm(Ppr - C[m_pr], axis=1)[ok[m_pr]]
    d_t = np.linalg.norm(Ptr - C[m_tr], axis=1)[ok[m_tr]]
    # within-state spread: samples against the same number of training members of that state
    sp_s, sp_t, w = [], [], []
    for m in np.flatnonzero(ok):
        idx = np.flatnonzero(m_s == m)
        if idx.size < 3:
            continue
        mem = np.flatnonzero(m_tr == m)
        k = min(idx.size, mem.size)
        sp_s.append(pdist_mean(Ps[idx[:k]]))
        sp_t.append(pdist_mean(Ptr[rng.choice(mem, size=k, replace=False)]))
        w.append(k)
    w = np.asarray(w, float)
    # nearest training image in the own state?
    tr_n2 = np.sum(Ptr * Ptr, 1)

    def nn_same(Pq, mq):
        out = []
        for i0 in range(0, Pq.shape[0], 256):
            Q_ = Pq[i0:i0 + 256]
            D2 = tr_n2[None, :] - 2.0 * Q_ @ Ptr.T
            out.append(m_tr[D2.argmin(1)] == mq[i0:i0 + 256])
        return float(np.concatenate(out).mean())
    mp_s, H_s = posterior_stats(a_s, T)
    mp_p, H_p = posterior_stats(a_pr, T)
    L = [f"{name} ({cfg['problem']}, Potts {K}, visible {cfg.get('visible')}, pair terms {cfg.get('pair_terms', 'all')}, "
         f"neighbourhood {cfg.get('neighbourhood', 'chebyshev')}, radius {cfg.get('radius')})",
         f"  states with >= 5 training members: {int(ok.sum())} of {K}; samples {S.shape[0]}, probes {Xpr.shape[0]}, "
         f"training images {Xtr.shape[0]}",
         f"  distance to own state's prototype, median: samples {np.median(d_s):.3f}, probes {np.median(d_p):.3f}, "
         f"training members {np.median(d_t):.3f}; samples / probes {np.median(d_s) / np.median(d_p):.3f}",
         f"    quartiles samples {np.percentile(d_s, 25):.3f} / {np.percentile(d_s, 75):.3f}, probes "
         f"{np.percentile(d_p, 25):.3f} / {np.percentile(d_p, 75):.3f}",
         f"  within-state spread (mean pairwise distance, {len(w)} states with >= 3 samples, weighted by size): "
         f"samples {np.sum(w * np.asarray(sp_s)) / w.sum():.3f}, training members (same counts) "
         f"{np.sum(w * np.asarray(sp_t)) / w.sum():.3f}; ratio {np.sum(w * np.asarray(sp_s)) / np.sum(w * np.asarray(sp_t)):.3f}",
         f"  nearest training image in the own state: samples {nn_same(Ps, m_s):.3f}, probes {nn_same(Ppr, m_pr):.3f}",
         f"  posterior at t = 0: mean max p samples {mp_s.mean():.3f}, probes {mp_p.mean():.3f}; mean entropy "
         f"samples {H_s.mean():.3f}, probes {H_p.mean():.3f} nats; share with max p < 0.5: samples "
         f"{np.mean(mp_s < 0.5):.3f}, probes {np.mean(mp_p < 0.5):.3f}",
         f"  state usage exp(H): samples {eff(m_s, K):.1f}, probes {eff(m_pr, K):.1f}, training images "
         f"{eff(m_tr, K):.1f}; states used by samples {np.unique(m_s).size}, by training images "
         f"{np.unique(m_tr).size}"]
    lines += L
    print("\n".join(L), flush=True)
    # figure: 8 most used states (by samples): prototype | 6 members | up to 6 samples
    top = [m for m in np.argsort(-np.bincount(m_s, minlength=K)) if ok[m]][:8]
    fig, ax = plt.subplots(len(top), 13, figsize=(13 * 0.9, len(top) * 0.95))
    for r, m in enumerate(top):
        mem = np.flatnonzero(m_tr == m)
        mem = rng.choice(mem, size=min(6, mem.size), replace=False)
        smp = np.flatnonzero(m_s == m)[:6]
        imgs = [C[m]] + [Ptr[i] for i in mem] + [None] * (6 - mem.size) + [Ps[i] for i in smp] + [None] * (6 - smp.size)
        for c, im in enumerate(imgs):
            a = ax[r, c]
            a.set_xticks([]); a.set_yticks([])
            if im is not None:
                a.imshow(np.clip(im, 0, 1).reshape(g, g), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                a.set_facecolor("0.3")
            if r == 0:
                a.set_title("prototype" if c == 0 else ("real" if c <= 6 else "sample"), fontsize=7)
        ax[r, 0].set_ylabel(f"state {m}\n{counts[m]} real\n{np.sum(m_s == m)} smp", fontsize=6)
    fig.suptitle(f"{name}: per state, prototype (mean of its training members), 6 members, up to 6 samples", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--samples", default="sde_base_n500")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []
    for r in args.runs:
        one_run(r, args.samples, rng, lines)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
