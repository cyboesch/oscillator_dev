#!/usr/bin/env python3
"""Track 2 analysis (2026-11-12; revised the same night after review): how do the closed-form harmonic modes compare
with what a trained study run already has? Read-only use of a finished run (default: the plain seed-1 study run) and of
the production statistics cache.

For each listed slice k (forward time t_k: the run's own slice times, hidden_history.npz 't' checked against its
config's time grid), along the principal axes e_1, e_16, e_64 of the run's training data (the statistics cache
<root>/data_stats/<problem>[_cls<c>].npz, loaded directly: never recomputed or rewritten here):
  - the full reference (latent_model.harmonic_reference, train_latent.py --harmonic-ref full): isotropic spring
    T / iso_t, mode strengths w_1, w_16, w_64 (anti-spring constants along e_i);
  - the within-cluster reference (--harmonic-ref within; Potts runs with prototype inits): the modes of the shared
    component covariance S_w of the run's own Potts prototype init (latent_model.prototype_reference with the run's
    k-means seed, return_stats=True), their anti-spring along e_i, sum_j w_j^w (u_j^w . e_i)^2, and the share of the
    data variance along e_i that S_w keeps (e_i^T S_w e_i / lam_i; the rest is the prototypes' spread);
  - the run's visible springs at x = 0 (latent_model.visible_hessian_at_zero): smallest eigenvalue and the margins
    min eig(H_vis(0) - V^T V) with the full reference's modes (R 16, 64) and with the within reference's (R 64);
    negative: the modes would exceed the springs there and only the quartic floor would confine (a global measure:
    at the data the hidden units' curvature counts too, below);
  - the run's mean curvature at the data along u = e_i, on n_x noised training images x = e x0 + sigma_t xi ~ p_t:
      visible   V_i = E_x u^T Hess E_vis(x) u (autodiff of the dense visible energy: the springs and the quartic
                terms, which curve the energy away from x = 0),
      Potts     A_P = E_x [Var_p(G_m . u) / T + 2 sum_l u_l^2 E_p Q_{m,l}], G_m = W_m + 2 Q_m x (the mean Hessian of
                the Potts free energy is -Cov_p(G) / T - 2 diag(E_p Q), i.e. -A_P along u),
      local     A_L = E_x sum_l u_l^2 (wl_l^2 / T) (1 - tanh^2((wl_l x_l + bl_l) / T)) (the per-pixel units),
      modes     A_M = u^T V^T V u for a run with harmonic modes (train_latent.py --harmonic);
    the model's mean curvature V_i - A_P - A_L - A_M, and the same with the full reference's 64 modes and with the
    within reference's 64 modes added (hypothetical: the run's other parameters were trained without them);
  - the targets: the Gaussian precision T / (e^2 lam_i + sigma_t^2), and for t >= 3.4 (slices 0-31 of the uniform
    grid) the exact mean curvature of the training distribution p_t (the equal mixture of N(e x0_j, sigma_t^2 I) over
    the training images), T u^T J u = T / sigma_t^2 - T e^2 E_x Var_w(x0 . u) / sigma_t^4 (w: the posterior over the
    training images; J the Fisher information, at least the inverse covariance). Caveat: as t falls the posterior
    over the 23,692 training images concentrates on a few of them (by t = 3 the exact curvature is already T / sigma_t^2
    along every axis: the empirical distribution's memorisation regime), so below t of about 4.5 this target is the
    finite training set's, an upper reference: a generalising model sits between the two targets.
Writes out/latent_physical_SGM/harmonic_analysis/<job>/report.txt and results.json.
Usage (CPU, from latent_physical_SGM/): sbatch --time=00:30:00 --cpus-per-task=8 --mem=32G jobs/run_cpu_neuronic.sh
  harmonic_analysis.py --run NAME [--slices 0,5,...] [--n-x 2048] [--root OUT_ROOT]
(job 4230722: the first version on the plain seed-1 study run, without the Q, local, data-curvature, exact and within
terms)
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "harmonic_analysis", os.environ.get("SLURM_JOB_ID", "manual"))
LINES = []
STATS_MU_TOL = 1e-9                      # as sample_latent.STATS_MU_TOL


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def fisher_curvature(X, x0, E, e, s2, T, chunk=256):
    """T u^T J u along the columns u of E for the equal mixture of N(e x0_j, s2 I) over the rows of x0, averaged over
    the rows of X: T / s2 - T e^2 E_X Var_w(x0 . u) / s2^2, w the posterior over j. Returns (len(E.T),)."""
    proj = x0 @ E                                         # (n_train, 3)
    n0 = np.sum(x0 * x0, axis=1)
    acc = np.zeros(E.shape[1])
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i + chunk]
        logw = (e * (Xc @ x0.T) - 0.5 * e * e * n0[None, :]) / s2     # -|x - e x0|^2 / (2 s2) + const(x)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        m1 = w @ proj
        acc += np.sum(w @ (proj * proj) - m1 * m1, axis=0)
    return T / s2 - T * e * e * (acc / X.shape[0]) / (s2 * s2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mnist28_d0123_K784_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_seed1")
    ap.add_argument("--slices", default="0,1,5,10,15,19,25,30,40,50,60,70,80,90,98,99")
    ap.add_argument("--n-x", type=int, default=2048)
    ap.add_argument("--root", default=lm.OUT_ROOT, help="folder of the runs and of data_stats/ (read only)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    run_dir = os.path.join(args.root, args.run)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    problem, cls = cfg["problem"], str(cfg.get("cls", "none"))
    st = np.load(os.path.join(args.root, "data_stats", problem + ("" if cls == "none" else f"_cls{cls}") + ".npz"))
    mu, lam, U = st["mu"], st["lam"], st["U"]
    th = np.load(os.path.join(run_dir, "params_history.npy"), mmap_mode="r")
    hid = np.load(os.path.join(run_dir, "hidden_history.npz"))
    if "Wp" in hid.files:                                      # per-patch Potts units (train_latent.py --patch)
        raise SystemExit(f"{args.run} has per-patch Potts units (train_latent.py --patch {cfg.get('patch')}): "
                         f"harmonic_analysis.py would silently omit their anti-spring; not supported")
    W_all, b_all = hid["W"], hid["b"]
    Q_all = hid["Q"] if "Q" in hid.files else None
    wl_all = hid["wl"] if "wl" in hid.files else None
    bl_all = hid["bl"] if "bl" in hid.files else None
    V_all = hid["V"] if "V" in hid.files else None             # harmonic modes (train_latent.py --harmonic)
    K = int(W_all.shape[1])
    potts = cfg.get("hidden", "ising") == "potts" and K > 0     # else no Potts anti-spring (reported 0)
    taus = lm.checked_time_pts(lm.config_time_grid(cfg), hid["t"] if "t" in hid.files else None, where=args.run)
    prob = lm.make_problem(problem, cfg.get("radius"), cfg.get("neighbourhood", "chebyshev"))
    conn, n = prob["conn"], prob["n"]
    data, labels = np.asarray(prob["data"], dtype=np.float64), np.asarray(prob["labels"])
    data_train = data if cls == "none" else data[labels == int(cls)]
    mu_err = float(np.max(np.abs(np.asarray(mu) - data_train.mean(axis=0))))
    if not mu_err <= STATS_MU_TOL:
        raise SystemExit(f"the statistics do not describe {args.run}'s training data (mean difference {mu_err:.3g})")
    T = lm.TEMP
    lam0 = max(float(lam.min()), 0.0)
    E = U[:, [0, 15, 63]]                                      # e_1, e_16, e_64
    out(f"=== harmonic_analysis {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
        f"run {args.run} ({problem}, cls {cls}, K {K}, radius {cfg.get('radius')}, pair terms {cfg.get('pair_terms')}, "
        f"visible {cfg.get('visible')}, time grid {lm.config_time_grid(cfg)}); statistics: lam_1..3 {lam[:3]}, lam_16 "
        f"{lam[15]:.4g}, lam_64 {lam[63]:.4g}, lam_0 {lam0:.4g} (smallest), |mu|^2 {float(mu @ mu):.5g}")
    # the within-cluster reference: the run's own prototype init (its k-means seed), and S_w
    within = None
    if potts and cfg.get("proto_init") in ("pooled", "perclass"):          # prototype inits (not random ones)
        km_seed = 100 * (int(cfg.get("seed", lm.KEY_SEED)) - lm.KEY_SEED)
        pooled = cfg.get("proto_init") == "pooled"
        t_km = time.time()
        W_r, _b_r, _cls_r, ps = lm.prototype_reference(
            data, labels, taus[:1], T, K, seed=km_seed, pooled=pooled,
            kmeans_fn=lm.kmeans_jax if cfg.get("ref_kmeans") == "gpu" else None, return_stats=True)
        lam_w, U_w = lm.harmonic_within_spectrum(ps["Sw"])
        keep = np.einsum("ij,ik,kj->j", E, ps["Sw"], E) / lam[[0, 15, 63]]
        init_dev = float(np.max(np.abs(W_r[0] - np.asarray(W_all[0]))))
        within = {"lam_w": lam_w, "U_w": U_w, "keep": keep, "init_dev": init_dev}
        out(f"within-cluster reference: k-means seed {km_seed} ({'pooled' if pooled else 'per class'}, {K} states; "
            f"{time.time() - t_km:.0f} s); trace S_w {lam_w.sum():.5g} vs trace C {lam.sum():.5g} "
            f"({lam_w.sum() / lam.sum():.3f}); S_w lam_1..5 {np.array2string(lam_w[:5], precision=4)}, lam_16 "
            f"{lam_w[15]:.4g}, lam_64 {lam_w[63]:.4g}, smallest {lam_w[-1]:.4g}; share of the data variance S_w keeps "
            f"along e_1/e_16/e_64 {keep[0]:.3f}/{keep[1]:.3f}/{keep[2]:.3f}; slice 0's W vs this init: largest "
            f"difference {init_dev:.3g} (0: the same partition as the run's)")
    rng = np.random.default_rng(0)
    pick = rng.choice(data_train.shape[0], size=args.n_x, replace=False)
    xi = rng.standard_normal((args.n_x, n))
    vis = lm.DenseVisible(conn, n)

    @jax.jit
    def vis_curv(theta, X, Ej):                                # E_x u^T Hess E_vis(x) u for the columns u of Ej
        def one(x, u):
            f = lambda y: vis.energy(theta, y[None])[0]                                      # noqa: E731
            return jax.jvp(lambda y: jax.jvp(f, (y,), (u,))[1], (x,), (u,))[1]
        return jnp.mean(jax.vmap(lambda u: jax.vmap(lambda x: one(x, u))(X))(Ej.T), axis=1)

    rows = {}
    for k in [int(s) for s in args.slices.split(",")]:
        tk = time.time()
        t = float(taus[k])
        e = math.exp(-t)
        sig2 = T * (1.0 - e * e)
        w64, iso = lm.harmonic_reference_strengths(lam, t, 64)
        w3 = w64[[0, 15, 63]]
        gauss = T / (e * e * lam[[0, 15, 63]] + sig2)          # the Gaussian precision along e_1, e_16, e_64
        X = e * data_train[pick] + math.sqrt(sig2) * xi
        fisher = fisher_curvature(X, data_train, E, e, sig2, T) if t >= 3.4 else np.full(3, np.nan)
        thk = np.asarray(th[k], dtype=np.float64)
        H = lm.visible_hessian_at_zero(thk, conn, n)
        ev_H = np.linalg.eigvalsh(H)
        margins = {}
        for R in (16, 64):
            V, _, _ = lm.harmonic_reference(mu, lam, U, R, t)
            margins[R] = float(np.linalg.eigvalsh(H - V.T @ V)[0])
        A_wref = np.full(3, np.nan)
        if within is not None:
            Vw, _, _ = lm.harmonic_reference(mu, within["lam_w"], within["U_w"], 64, t)
            margins["w64"] = float(np.linalg.eigvalsh(H - Vw.T @ Vw)[0])
            A_wref = np.sum((Vw @ E) ** 2, axis=0)
        curv_vis = np.asarray(vis_curv(jnp.asarray(thk), jnp.asarray(X), jnp.asarray(E)))
        A_P, A_L, A_M, P = np.zeros(3), np.zeros(3), np.zeros(3), np.full((1, 1), np.nan)
        if potts:                                              # Hess F_P = -Cov_p(G) / T - 2 diag(E_p Q)
            W, b = np.asarray(W_all[k], dtype=np.float64), np.asarray(b_all[k], dtype=np.float64)
            A = X @ W.T + b
            Qk = np.asarray(Q_all[k], dtype=np.float64) if Q_all is not None else None
            if Qk is not None:
                A = A + (X * X) @ Qk.T
            Z = A / T
            P = np.exp(Z - Z.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
            GU = np.stack([np.broadcast_to(W @ E[:, j], P.shape)
                           + (2.0 * (X * E[:, j]) @ Qk.T if Qk is not None else 0.0)
                           for j in range(E.shape[1])], axis=-1)   # (n_x, M, 3): G_m . u
            m1 = np.einsum("xm,xmj->xj", P, GU)
            m2 = np.einsum("xm,xmj->xj", P, GU * GU)
            A_P = np.mean(m2 - m1 * m1, axis=0) / T
            if Qk is not None:                                 # + 2 E_x sum_l u_l^2 E_p Q_l
                A_P = A_P + 2.0 * np.mean((P @ Qk) @ (E * E), axis=0)
        if wl_all is not None and cfg.get("local"):           # the per-pixel units: -(wl^2 / T)(1 - tanh^2) per pixel
            wl, bl = np.asarray(wl_all[k], dtype=np.float64), np.asarray(bl_all[k], dtype=np.float64)
            th_l = np.tanh((X * wl + bl) / T)
            A_L = np.mean(((wl * wl / T) * (1.0 - th_l * th_l)) @ (E * E), axis=0)
        if V_all is not None:                                  # the run's own modes
            Vk = np.asarray(V_all[k], dtype=np.float64)
            A_M = np.sum((Vk @ E) ** 2, axis=0)
            margins["run"] = float(np.linalg.eigvalsh(H - Vk.T @ Vk)[0])
        total = curv_vis - A_P - A_L - A_M
        rows[k] = {"t": t, "iso_spring": T / iso, "w": w3.tolist(), "gauss_precision": gauss.tolist(),
                   "fisher_curvature": fisher.tolist(), "H_min": float(ev_H[0]),
                   "H0_curv": np.einsum("ij,ik,kj->j", E, H, E).tolist(), "vis_curv_data": curv_vis.tolist(),
                   "potts_antispring": A_P.tolist(), "local_antispring": A_L.tolist(), "total_curv": total.tolist(),
                   "total_plus_full64": (total - w3).tolist(), "within_w": A_wref.tolist(),
                   "total_plus_within64": (total - A_wref).tolist(), "margin16": margins[16], "margin64": margins[64],
                   "margin_within64": margins.get("w64"), "mean_max_p": float(np.mean(P.max(axis=1)))}
        if V_all is not None:
            rows[k].update(modes_antispring=A_M.tolist(), margin_run=margins["run"])
        f3 = lambda a: "/".join(f"{v:.4g}" for v in a)                                     # noqa: E731
        out(f"slice {k:2d} t {t:.3f}: T/iso {T / iso:.4g}; full reference w {f3(w3)}; within reference along e_i "
            f"{f3(A_wref)}; targets: Gaussian precision {f3(gauss)}, exact (Fisher) {f3(fisher)} | run: H_vis(0) min "
            f"{ev_H[0]:.4g}; at the data: visible {f3(curv_vis)}, minus the anti-springs of the Potts layer "
            f"{f3(A_P)}, of the local units {f3(A_L)}"
            + (f", of the own modes {f3(A_M)}" if V_all is not None else "")
            + f"; total {f3(total)}; + full-reference modes {f3(total - w3)}; + within-reference modes "
            f"{f3(total - A_wref)} | margins min eig(H_vis(0) - V^T V): full R 16 {margins[16]:.4g}, R 64 "
            f"{margins[64]:.4g}, within R 64 {margins.get('w64', float('nan')):.4g}"
            + (f", own modes {margins['run']:.4g}" if V_all is not None else "")
            + f"; <max p> {rows[k]['mean_max_p']:.3f} ({time.time() - tk:.0f} s)")
    # the paper's initial visible energy (slice 0 starts from it): k_lin = 1, c_lin = 1 on every kept pair (k_duff = 1
    # in the quartic model, k_6 = 1 in the sextic one; neither curves the energy at x = 0)
    P_ = conn.shape[0]
    th0 = lm.paper_theta_layout(P_, n)[0]
    if cfg.get("visible", "sextic") == "quartic":
        th0 = lm.quartic_initial_theta(th0, n, P_)
    th0 = np.array(th0)                                        # a writable copy (the dropped groups are zeroed below)
    for s_ in lm.dropped_pair_slices(n, P_, lm.parse_pair_terms(cfg.get("pair_terms", "all"))):
        th0[s_] = 0.0
    H0 = lm.visible_hessian_at_zero(th0, conn, n)
    V5, _, _ = lm.harmonic_reference(mu, lam, U, 64, float(taus[0]))
    line = (f"initial visible energy (paper init, {cfg.get('visible', 'sextic')}, pairs {cfg.get('pair_terms')}): H_vis "
            f"min {np.linalg.eigvalsh(H0)[0]:.4g}, along e_1/16/64 {np.einsum('ij,ik,kj->j', E, H0, E)}, min eig of "
            f"H_vis - V_ref(t_0)^T V_ref(t_0): full {np.linalg.eigvalsh(H0 - V5.T @ V5)[0]:.4g}")
    if within is not None:
        V5w, _, _ = lm.harmonic_reference(mu, within["lam_w"], within["U_w"], 64, float(taus[0]))
        line += f", within {np.linalg.eigvalsh(H0 - V5w.T @ V5w)[0]:.4g}"
    out(line)
    summary = {"run": args.run, "slices": rows}
    if within is not None:
        summary["within"] = {"lam_w_top5": within["lam_w"][:5].tolist(), "trace_Sw": float(within["lam_w"].sum()),
                             "trace_C": float(lam.sum()), "keep_e1_e16_e64": within["keep"].tolist(),
                             "slice0_W_vs_init": within["init_dev"]}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    out(f"wrote {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
