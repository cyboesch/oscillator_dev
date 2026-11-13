#!/usr/bin/env python3
"""What do more training steps change in the early slices (high t), where the digit of a sample is decided?

For each run and slice k (forward time t_k), on N_X held-out MNIST test images of the problem's digits, noised at
t_k with fixed noise (x = e^-t x0 + sqrt(T (1 - e^-2t)) xi, the same x for every run):
  - the exact score of the noised TRAINING distribution (an equal-weight Gaussian mixture around e^-t x0_i over all
    training images): s*(x) = (e^-t m*(x) - x) / sigma^2, m*(x) = sum_i w_i(x) x0_i, and its class posterior
    p*(c | x) = sum_{i in c} w_i(x);
  - the model's score from its sampler drift at that knot, s(x) = (drift(5 - t_k, x) - x) / (2 T)
    (sample_latent.make_drift: drift = x - 2 grad F = x + 2 T score; with the harmonic modes of train_latent.py
    --harmonic runs);
  - relative score error E|s - s*|^2 / E|s*|^2 (the validation loss measures the same error but offset by the large
    constant E|s*|^2 / 2, which hides relative differences);
  - the denoised means (Tweedie) m = (x + sigma^2 s) / e^-t of model and truth, their relative error, and the digit the
    FID classifier reads from each: agreement of the model's digit with the exact one and with the image's label;
  - the Potts states: effective number, dead (< 1e-6), and how many states each class has (majority class of the test
    images a state wins), and the class purity of the argmax state;
  - for runs with per-patch Potts units (train_latent.py --patch; their force is part of the drift): the patch units'
    effective states (median / min over the patches), dead states (summed) and <max p> (latent_model.patch_unit_stats).
Usage (CPU Slurm job): python early_slices.py LABEL=RUN ... [--slices 0,3,...]
Writes out/latent_physical_SGM/early_slices/<job>/report.txt and results.json.
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
import sample_latent as sl  # noqa: E402
import fid_score  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "early_slices", os.environ.get("SLURM_JOB_ID", "manual"))
N_X = 2048
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def exact_posterior(X, x0, onehot, e, s2, chunk=256):
    """m*(x) = sum_i w_i x0_i and p*(c|x) for the mixture of N(e x0_i, s2 I) with equal weights."""
    sq = 0.5 * e * e * np.sum(x0 * x0, axis=1)
    M, PC = [], []
    for i in range(0, X.shape[0], chunk):
        L = (e * (X[i:i + chunk] @ x0.T) - sq) / s2
        L -= L.max(axis=1, keepdims=True)
        w = np.exp(L)
        w /= w.sum(axis=1, keepdims=True)
        M.append(w @ x0)
        PC.append(w @ onehot)
    return np.concatenate(M), np.concatenate(PC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=RUN")
    ap.add_argument("--slices", default="0,3,6,9,12,15,19,22,25,30,40,60,80,99")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; early_slices; job {os.environ.get('SLURM_JOB_ID', 'manual')}")
    specs = [s.split("=", 1) for s in args.runs]
    slices = [int(s) for s in args.slices.split(",")]
    with open(os.path.join(lm.OUT_ROOT, specs[0][1], "config.json")) as f:
        cfg = json.load(f)
    pname = cfg["problem"]
    digits = [int(c) for c in pname.split("_d")[1]]
    prob = lm.make_problem(pname, cfg.get("radius"), cfg.get("neighbourhood", "chebyshev"))
    x0_train = np.asarray(prob["data"], dtype=np.float64)
    y_train = np.asarray(prob["labels"])
    n_cls = len(digits)
    onehot = np.eye(n_cls)[y_train]
    mean, std = float(prob["mean"]), float(prob["std"])
    Xte, yte = fid_score.load_mnist("test", 28)
    sel = np.isin(yte, digits)
    Xte, yte = Xte[sel], np.searchsorted(digits, yte[sel])
    rng = np.random.default_rng(0)
    pick = rng.choice(Xte.shape[0], size=N_X, replace=False)
    x0 = (Xte[pick].astype(np.float64) - mean) / std
    lab = yte[pick]
    xi = rng.standard_normal(x0.shape)
    feats, acc = fid_score.load_classifier(28)
    out(f"problem {pname}: {x0_train.shape[0]} training images; {N_X} held-out test images (classes "
        f"{np.bincount(lab, minlength=n_cls).tolist()}); classifier test accuracy {acc:.4f}; slices {slices}")

    def digit_of(m):                                   # the classifier's digit (restricted to the problem's digits)
        _, P = feats(np.clip(m * std + mean, 0.0, 1.0))
        return np.argmax(P[:, digits], axis=1)

    truth = {}
    taus = lm.forward_time_pts()
    for k in slices:
        t = float(taus[k])
        e = np.exp(-t)
        s2 = lm.TEMP * (1.0 - e * e)
        X = e * x0 + np.sqrt(s2) * xi
        m_true, pc = exact_posterior(X, x0_train, onehot, e, s2)
        s_true = (e * m_true - X) / s2
        truth[k] = {"t": t, "e": e, "s2": s2, "X": X, "m": m_true, "s": s_true, "pc": pc,
                    "dig": digit_of(m_true)}
        H = -np.sum(pc * np.log(pc + 1e-300), axis=1)
        out(f"  slice {k:2d} (t = {t:.3f}): exact class posterior: max p median {np.median(pc.max(axis=1)):.3f}, "
            f"entropy median {np.median(H):.3f} (log {n_cls} = {np.log(n_cls):.3f}); argmax = the image's label "
            f"{np.mean(pc.argmax(axis=1) == lab):.3f}; classifier digit of m* = label {np.mean(truth[k]['dig'] == lab):.3f}")

    res = {}
    for label, run in specs:
        t0 = time.time()
        th, W, b, run_dir, prov, problem = sl.load_params(run, with_harmonic=True, with_patch=True)
        local_p, quad_p = prov.pop("local", None), prov.pop("quad", None)
        harm_p = prov.pop("harmonic", None)          # harmonic modes (train_latent --harmonic): part of the score
        patch_p = prov.pop("patch", None)            # per-patch units (train_latent --patch): part of the score
        drift, _ = sl.make_drift(th, W, b, "sde", problem, prov.get("hidden", "ising"), local_p, quad_p,
                                 prov.get("radius"), prov.get("neighbourhood", "chebyshev"), sl.slice_times(prov),
                                 harmonic=harm_p, patch=patch_p)
        geom_p = None if patch_p is None else lm.patch_geometry(lm.problem_geometry(problem)[0] ** 2,
                                                                patch_p[0].shape[1], patch_p[0].shape[3])
        f = jax.jit(jax.vmap(drift, in_axes=(None, 0, None)))
        rows = {}
        for k in slices:
            tr = truth[k]
            D = np.asarray(f(jnp.asarray(lm.T_FORWARD - tr["t"]), jnp.asarray(tr["X"]), None), dtype=np.float64)
            s = (D - tr["X"]) / (2.0 * lm.TEMP)
            err = float(np.mean(np.sum((s - tr["s"]) ** 2, axis=1)) / np.mean(np.sum(tr["s"] ** 2, axis=1)))
            m = (tr["X"] + tr["s2"] * s) / tr["e"]
            mvar = np.mean(np.sum((tr["m"] - tr["m"].mean(axis=0)) ** 2, axis=1))
            derr = float(np.mean(np.sum((m - tr["m"]) ** 2, axis=1)) / mvar)
            dig = digit_of(m)
            # Potts states on these x
            Wk, bk = W[k], b[k]
            A = tr["X"] @ Wk.T + bk
            if quad_p is not None:
                A = A + (tr["X"] ** 2) @ quad_p[k].T
            Z = A / lm.TEMP
            P = np.exp(Z - Z.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
            pbar = P.mean(axis=0)
            arg = P.argmax(axis=1)
            st_cls = -np.ones(Wk.shape[0], int)
            for mm in np.unique(arg):
                st_cls[mm] = int(np.bincount(lab[arg == mm], minlength=n_cls).argmax())
            per_cls = np.bincount(st_cls[st_cls >= 0], minlength=n_cls)
            purity = float(np.mean(st_cls[arg] == lab))
            rows[k] = {"score_rel_err": err, "denoise_rel_err": derr, "digit_agree_exact": float(np.mean(dig == tr["dig"])),
                       "digit_acc": float(np.mean(dig == lab)), "eff": float(np.exp(-np.sum(pbar * np.log(pbar + 1e-300)))),
                       "dead": int(np.sum(pbar < 1e-6)), "argmax_states": int(np.unique(arg).size),
                       "states_per_class": per_cls.tolist(), "purity": purity}
            if patch_p is not None and not (np.any(np.asarray(patch_p[0][k])) or np.any(np.asarray(patch_p[1][k]))):
                rows[k]["patch_absent"] = True       # --patch-on-slice runs: all-zero units before the switch-on
            elif patch_p is not None:                # the patch units' occupations on these x (their slice-k values)
                st_p, _ = lm.patch_unit_stats(patch_p[0][k], patch_p[1][k], None if patch_p[2] is None else
                                              patch_p[2][k], tr["X"], lm.TEMP, None, geom_p)
                rows[k].update(patch_eff_median=st_p["patch_eff_states_median"],
                               patch_eff_min=st_p["patch_eff_states_min"],
                               patch_dead=st_p["patch_states_below_1e-6"], patch_mean_max_p=st_p["patch_mean_max_p"])
        res[label] = rows
        del th, W, b, drift, f
        out(f"\n=== {label} = {run} ({time.time() - t0:.0f} s)")
        out("  slice      t   score err   denoise err   digit = exact   digit = label   eff  dead  argmax states  "
            "states per class   purity")
        for k in slices:
            r = rows[k]
            out(f"  {k:5d}  {truth[k]['t']:.3f}   {r['score_rel_err']:9.4f}   {r['denoise_rel_err']:11.4f}   "
                f"{r['digit_agree_exact']:13.3f}   {r['digit_acc']:13.3f}   {r['eff']:5.0f}  {r['dead']:4d}  "
                f"{r['argmax_states']:13d}  {'/'.join(map(str, r['states_per_class'])):>16}   {r['purity']:.3f}")
            if "patch_eff_median" in r:              # patch runs: one extra row per slice
                out(f"         patch units: effective states {r['patch_eff_median']:.2f} / {r['patch_eff_min']:.2f} "
                    f"(median / min over the patches), dead {r['patch_dead']}, <max p> {r['patch_mean_max_p']:.3f}")
            elif r.get("patch_absent"):
                out("         patch units: absent at this slice (all zero; switched on later)")
    # group means (label prefix before '_s' / digits) for a compact comparison
    out("\nmeans over the runs of each group (group = label without a trailing seed number):")
    groups = {}
    for label in res:
        g = label.rstrip("0123456789").rstrip("_s").rstrip("_")
        groups.setdefault(g, []).append(label)
    for k in slices:
        parts = []
        for g, labs in groups.items():
            v = np.mean([[res[lb][k][q] for q in ("score_rel_err", "denoise_rel_err", "digit_agree_exact")] for lb in labs], axis=0)
            parts.append(f"{g} ({len(labs)}): {v[0]:.4f} / {v[1]:.4f} / {v[2]:.3f}")
        out(f"  slice {k:2d}: score err / denoise err / digit = exact: " + ";  ".join(parts))
    with open(os.path.join(OUT, "results.json"), "w") as fo:
        json.dump(res, fo, indent=1)
    with open(os.path.join(OUT, "report.txt"), "w") as fo:
        fo.write("\n".join(LINES) + "\n")
    out(f"wrote {OUT}")


if __name__ == "__main__":
    main()
