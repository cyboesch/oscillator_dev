#!/usr/bin/env python3
"""Where along the reverse path does the SDE's noise matter?  (fine grid, 500 trajectories)

Follow-up to ode_noise_window.py.  The reverse time t in [0, 5] (forward time tau = 5 - t) is cut
into 10 intervals of length 0.5.  Every sampler is a mode string with one letter per interval:
O = probability-flow ODE, S = reverse SDE.  Variants:

  ODE / SDE         : OOOOOOOOOO / SSSSSSSSSS
  SDE>ODE@tau       : SDE for forward times above tau, ODE below   (tau = 4.5, 4, ..., 0.5)
  ODE>SDE@tau       : ODE above tau, SDE below
  SDE[hi,lo]        : SDE only while tau is in [lo, hi], ODE elsewhere

Noise: the library's solve_SDE builds its VirtualBrownianTree on the segment's own [t0, t1], so the
same key on two segments would replay a rescaled copy of the same noise.  Each (trajectory,
interval) therefore gets its own key, fold_in(trajectory key, interval).  All variants that run
the SDE on a given interval see the same noise there (paired comparison).

Variants sharing a prefix share its computation (trie over the mode strings).  States that have
diverged (max|x| > MAX_ABS_FREEZE) are frozen; they are counted as diverged and as far.
Trajectories 0-39 are the paper's shared initial states/keys; 40.. are new draws from the same
distribution (sqrt(Temp) * N(0, 1)).  Progress is checkpointed after every interval; rerunning
with the same NOISE_WINDOW_OUT resumes.

Read-only on the checkpoint.  Writes out/diagnostics/ode_noise_window_fine/<job id>/.
Run under Slurm (cd physical-diffusion-models/ode_vs_sde && sbatch run_ode_noise_window_fine_neuronic.sh).
"""

import gc
import json
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import flatten_util, vmap  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.append(PKG)

from physical_diffusion_fns.helper_fns import interpolate_parameters, normalize_samples  # noqa: E402
from physical_diffusion_fns.network_fns import (  # noqa: E402
    create_2d_square_lattice_connectivity,
    setup_energy_fn,
    setup_overdamped_SDE,
    solve_ODE,
    solve_SDE,
)

# Must match scripts/physical_diffusion_model_MNIST.py
KEY_SEED = 1
STD_ADDED_NOISE = 0.01
TEMP = 0.02
SIGMA_FORWARD = 1.0
GRID = 28
N_NEIGHBOUR_COUPLINGS = 14
T_FORWARD = 5.0
N_TIME_STEPS = 100
RTOL_SDE, ATOL_SDE, BROWNIAN_TOL = 1e-4, 1e-6, 1e-12
RTOL_ODE, ATOL_ODE = 1e-7, 1e-9
DT0 = 1e-8

# Experiment settings
N_TRAJ = int(os.environ.get("N_TRAJ", 500))
CHUNK = int(os.environ.get("CHUNK", 250))
NEW_SEED = 2026
N_INT = 10
DT_INT = T_FORWARD / N_INT
SWITCH_TAUS = [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
WINDOWS = [(4.0, 1.0), (4.0, 2.0), (3.0, 1.0), (3.5, 1.5)]  # (hi, lo) in forward time
MAX_ABS_FREEZE = 15.0
OUT_OF_RANGE_MARGIN = 0.2
REF_SEED = 11

DATA_PATH = os.path.join(PKG, "data", "MNIST", "mnist_labels_[0, 1]_resolution_(28, 28)_balanced_True.npy")
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
CHECKPOINT_PATH = os.path.join(RUN_DIR, "params_history.npy")
ABLATION_DIR = os.path.join(RUN_DIR, "interpolation_ablation_v1")
SAVED_ODE = "final_states_ablation_params_raw_interp_linear_step_knots_True_n40_matched_rng_probability_flow_ode_atol_1e-09_rtol_1e-07.npy"
SAVED_SDE = "final_states_ablation_params_raw_interp_linear_step_knots_True_n40_matched_rng_sde_atol_1e-06_rtol_0.0001_btol_1e-12.npy"
OUT_DIR = os.environ.get("NOISE_WINDOW_OUT") or os.path.join(
    PKG, "out", "diagnostics", "ode_noise_window_fine", os.environ.get("SLURM_JOB_ID", "manual"))
PROGRESS = os.path.join(OUT_DIR, "progress.npz")

LINES = []


def report(line=""):
    print(line, flush=True)
    LINES.append(line)
    with open(os.path.join(OUT_DIR, "report.txt"), "a") as f:
        f.write(line + "\n")


def k_of(tau):
    """Number of intervals with forward time above tau."""
    k = (T_FORWARD - tau) / DT_INT
    assert abs(k - round(k)) < 1e-9, tau
    return int(round(k))


def build_variants():
    v = {"ODE": "O" * N_INT, "SDE": "S" * N_INT}
    for tau in SWITCH_TAUS:
        k = k_of(tau)
        v[f"SDE>ODE@{tau:g}"] = "S" * k + "O" * (N_INT - k)
    for tau in SWITCH_TAUS:
        k = k_of(tau)
        v[f"ODE>SDE@{tau:g}"] = "O" * k + "S" * (N_INT - k)
    for hi, lo in WINDOWS:
        a, b = k_of(hi), k_of(lo)
        v[f"SDE[{hi:g},{lo:g}]"] = "O" * a + "S" * (b - a) + "O" * (N_INT - b)
    return v


def nn_distance(px_samples, px_train, chunk=2000):
    best = np.full(px_samples.shape[0], np.inf)
    s2 = np.sum(px_samples**2, axis=1)[:, None]
    for i in range(0, px_train.shape[0], chunk):
        tr = px_train[i:i + chunk]
        d2 = s2 + np.sum(tr**2, axis=1)[None, :] - 2.0 * px_samples @ tr.T
        best = np.minimum(best, np.min(d2, axis=1))
    return np.sqrt(np.maximum(best, 0.0))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; jax {jax.__version__}; devices {jax.devices()}; "
           f"default float {jnp.zeros(1).dtype}; N_TRAJ {N_TRAJ}; CHUNK {CHUNK}")

    # Data (for metrics only)
    raw = jnp.array(np.load(DATA_PATH))
    _, _, image_noise_added_key = jr.split(jr.PRNGKey(KEY_SEED), 3)
    _, subkey = jr.split(image_noise_added_key)
    data, mean_px, std_px = normalize_samples(raw + jr.normal(subkey, raw.shape) * STD_ADDED_NOISE)
    data, mean_px, std_px = np.asarray(data), float(mean_px), float(std_px)
    raw = np.asarray(raw)
    blank = np.all(raw == 0.0, axis=0)
    pix_min, pix_max = (0.0 - mean_px) / std_px, (1.0 - mean_px) / std_px
    px_train = data * std_px + mean_px
    rng = np.random.default_rng(REF_SEED)
    held = rng.choice(data.shape[0], size=200, replace=False)
    nn_ref = nn_distance(px_train[held], px_train[np.setdiff1d(np.arange(data.shape[0]), held)])
    blend_threshold = float(np.percentile(nn_ref, 90))
    report(f"reference: held-out data nearest-image distance median {np.median(nn_ref):.2f}, p90 {blend_threshold:.2f}"
           f" (samples above the p90 are counted as 'far')")

    # Model, parameters and the paper's interpolation
    ckpt = np.load(CHECKPOINT_PATH, mmap_mode="r")
    params_history = jnp.asarray(np.array(ckpt, dtype=np.float64))
    forward_time_pts = jnp.linspace(0.0, T_FORWARD, N_TIME_STEPS)[::-1]
    interp = interpolate_parameters(params_history, forward_time_pts)
    n = GRID * GRID
    connectivity = create_2d_square_lattice_connectivity(grid_size=GRID, n_neighbour_couplings=N_NEIGHBOUR_COUPLINGS)
    p = int(connectivity.shape[0])
    template = (jnp.ones(n), jnp.zeros(n), jnp.ones(n), jnp.ones(p), jnp.zeros(p), jnp.zeros(p), jnp.zeros(n))
    theta_init, unflatten = flatten_util.ravel_pytree(template)
    energy_fn = setup_energy_fn(connectivity, unflatten)
    theta_lin = jnp.zeros_like(theta_init).at[0:n].set(1.0)
    tau_of = lambda t: T_FORWARD - t
    theta_pf = lambda t: interp(tau_of(t)) - theta_lin
    theta_rev = lambda t: 2.0 * interp(tau_of(t)) - theta_lin
    drift_ode, _ = setup_overdamped_SDE(energy_fn, theta_pf, n, time_dependent_parms=True, Temp=TEMP)
    drift_sde, diffusion_sde = setup_overdamped_SDE(energy_fn, theta_rev, n, time_dependent_parms=True, Temp=TEMP)
    knots = np.sort(np.asarray(T_FORWARD - forward_time_pts))[1:-1]

    def knots_in(t0, t1):
        k = knots[(knots > t0 + 1e-12) & (knots < t1 - 1e-12)]
        return jnp.asarray(k) if k.size else None

    # Initial states and keys: the paper's 40, then new draws from the same distribution
    x_paper = np.load(os.path.join(ABLATION_DIR, "shared_initial_states.npy"))
    keys_paper = np.load(os.path.join(ABLATION_DIR, "shared_brownian_keys.npy"))
    n_new = N_TRAJ - x_paper.shape[0]
    k_init, k_brown = jr.split(jr.PRNGKey(NEW_SEED))
    x_new = np.asarray(jnp.sqrt(TEMP) * SIGMA_FORWARD * jr.normal(k_init, (n_new, n)))
    keys_new = np.asarray(jr.split(k_brown, n_new))
    x_init = np.concatenate([x_paper, x_new]).astype(np.float64)
    keys_traj = jnp.asarray(np.concatenate([keys_paper, keys_new.astype(keys_paper.dtype)]))
    report(f"{N_TRAJ} trajectories ({x_paper.shape[0]} paper + {n_new} new, seed {NEW_SEED}); "
           f"init std {x_init.std():.4f} (sqrt T = {np.sqrt(TEMP):.4f}); keys {keys_traj.shape} {keys_traj.dtype}")

    variants = build_variants()
    report(f"{len(variants)} variants over {N_INT} intervals of {DT_INT:g}:")
    for lab, m in variants.items():
        report(f"  {lab:<14} {m}")

    def solve_group(mode, x_rows, traj_rows, t0, t1, interval):
        """Advance rows of states over [t0, t1] with one solver; frozen (diverged) rows are skipped."""
        out = x_rows.copy()
        live = np.where(np.max(np.abs(x_rows), axis=1) <= MAX_ABS_FREEZE)[0]
        if live.size == 0:
            return out, 0
        st = knots_in(t0, t1)
        if mode == "O":
            fn = vmap(lambda y: solve_ODE(drift_ode, y, t0, t1, 2, DT0, rtol=RTOL_ODE, atol=ATOL_ODE, step_ts=st))
        else:
            keys_int = vmap(lambda k: jr.fold_in(k, interval))(keys_traj)
            fn = vmap(lambda y, k: solve_SDE(drift_sde, diffusion_sde, y, k, t0, t1, 2, DT0, rtol=RTOL_SDE,
                                             atol=ATOL_SDE, brownian_tolerance=BROWNIAN_TOL, step_ts=st))
        for c0 in range(0, live.size, CHUNK):
            idx = live[c0:c0 + CHUNK]
            pad = np.concatenate([idx, np.repeat(idx[-1:], CHUNK - idx.size)])  # fixed shape: one compile
            y = jnp.asarray(x_rows[pad])
            sol = fn(y) if mode == "O" else fn(y, keys_int[traj_rows[pad]])
            out[idx] = np.asarray(sol.ys[:, -1, :])[: idx.size]
        return out, int(x_rows.shape[0] - live.size)

    # Trie over mode strings; resume from the last finished interval if present
    states = {"": x_init}
    done = 0
    if os.path.exists(PROGRESS):
        with np.load(PROGRESS) as f:
            done = int(f["__interval__"])
            states = {k: f[k] for k in f.files if k != "__interval__"}
        report(f"resuming after interval {done} ({len(states)} stored prefixes)")

    for i in range(done, N_INT):
        t0, t1 = i * DT_INT, (i + 1) * DT_INT
        new = sorted({m[: i + 1] for m in variants.values()})
        t_i = time.time()
        msg = []
        for mode in ("O", "S"):
            group = [pr for pr in new if pr[-1] == mode]
            if not group:
                continue
            x_rows = np.concatenate([states[pr[:-1]] for pr in group])
            traj_rows = np.tile(np.arange(N_TRAJ), len(group))
            t_m = time.time()
            x_out, frozen = solve_group(mode, x_rows, traj_rows, t0, t1, i)
            for g, pr in enumerate(group):
                states[pr] = x_out[g * N_TRAJ:(g + 1) * N_TRAJ]
            msg.append(f"{mode}: {len(group)} prefixes, {frozen} frozen rows, {time.time() - t_m:.0f} s")
            gc.collect()
            jax.clear_caches()  # each compiled solve embeds the 436 MB parameter table
        states = {k: v for k, v in states.items() if len(k) == i + 1}
        np.savez(PROGRESS + ".tmp.npz", __interval__=np.array(i + 1), **states)
        os.replace(PROGRESS + ".tmp.npz", PROGRESS)
        report(f"interval {i} (t {t0:.1f}-{t1:.1f}, tau {T_FORWARD - t0:.1f}-{T_FORWARD - t1:.1f}) "
               f"done in {time.time() - t_i:.0f} s; " + "; ".join(msg))

    finals = {lab: states[m] for lab, m in variants.items()}
    np.savez(os.path.join(OUT_DIR, "final_states.npz"),
             **{k.replace("@", "_at_").replace(">", "_then_").replace("[", "_").replace("]", "").replace(",", "_"): v
                for k, v in finals.items()})

    # Sanity checks on the paper's 40 trajectories
    saved_ode = np.load(os.path.join(ABLATION_DIR, SAVED_ODE))
    saved_sde = np.load(os.path.join(ABLATION_DIR, SAVED_SDE))
    d_ode = np.max(np.abs(finals["ODE"][:40] - saved_ode), axis=1)
    report("")
    report(f"sanity: ODE (split at the interval boundaries) vs saved paper ODE, trajectories 0-39: "
           f"median max|diff| {np.median(d_ode):.2e}, max {np.max(d_ode):.2e}")
    report("(the SDE here uses different noise from the paper's single [0, 5] Brownian tree, so it is not "
           "expected to match the saved SDE samples; the saved SDE row below is an independent reference)")

    # Metrics
    nn = {lab: nn_distance(x * std_px + mean_px, px_train) for lab, x in finals.items()}
    nn["SDE (paper, 40)"] = nn_distance(saved_sde * std_px + mean_px, px_train)
    finals_m = dict(finals, **{"SDE (paper, 40)": saved_sde})
    far_sde = (nn["SDE"] > blend_threshold) | (np.max(np.abs(finals["SDE"]), axis=1) > MAX_ABS_FREEZE)
    report("")
    report("far = nearest-image distance above the data's p90, or diverged (max|x| > 15); +-: binomial SE; "
           "dfar: paired difference to the SDE row with its SE; %out = pixels beyond the data range +- 0.2")
    report(f"{'variant':<16}{'mode':>12}{'NN med':>8}{'% far':>7}{'+-':>5}{'dfar':>7}{'+-':>5}"
           f"{'bg within':>10}{'%out':>7}{'%div':>6}")
    rows = {}
    for lab, x in finals_m.items():
        div = np.max(np.abs(x), axis=1) > MAX_ABS_FREEZE
        far = (nn[lab] > blend_threshold) | div
        out = (x < pix_min - OUT_OF_RANGE_MARGIN) | (x > pix_max + OUT_OF_RANGE_MARGIN)
        m = far.size
        r = {
            "mode": variants.get(lab, "paper"), "n": int(m),
            "nn_median": float(np.median(nn[lab])), "frac_far": float(far.mean()),
            "se_far": float(np.sqrt(far.mean() * (1 - far.mean()) / m)),
            "bg_within": float(np.median(x[~div][:, blank].std(axis=1))) if (~div).any() else float("nan"),
            "frac_out": float(out.mean()), "frac_diverged": float(div.mean()),
        }
        if m == N_TRAJ:
            dd = far.astype(float) - far_sde.astype(float)
            r["dfar_vs_sde"], r["se_dfar"] = float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(m))
        rows[lab] = r
        dtxt = (f"{100 * r['dfar_vs_sde']:>7.1f}{100 * r['se_dfar']:>5.1f}" if "dfar_vs_sde" in r else f"{'':>12}")
        report(f"{lab:<16}{r['mode']:>12}{r['nn_median']:>8.2f}{100 * r['frac_far']:>7.1f}{100 * r['se_far']:>5.1f}"
               f"{dtxt}{r['bg_within']:>10.4f}{100 * r['frac_out']:>7.2f}{100 * r['frac_diverged']:>6.1f}")
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"blend_threshold": blend_threshold, "variants": rows}, f, indent=2)

    # Curve: % far against the switch point, both directions (endpoints are the pure samplers)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    taus = [5.0] + SWITCH_TAUS + [0.0]
    curves = [
        ("SDE above τ, ODE below", "#2a78d6", ["ODE"] + [f"SDE>ODE@{t:g}" for t in SWITCH_TAUS] + ["SDE"]),
        ("ODE above τ, SDE below", "#eb6834", ["SDE"] + [f"ODE>SDE@{t:g}" for t in SWITCH_TAUS] + ["ODE"]),
    ]
    for name, color, labs in curves:
        y = np.array([100 * rows[l]["frac_far"] for l in labs])
        e = np.array([100 * rows[l]["se_far"] for l in labs])
        ax.fill_between(taus, y - e, y + e, color=color, alpha=0.15, linewidth=0)
        ax.plot(taus, y, color=color, linewidth=2, marker="o", markersize=5, label=name)
        above = y[0] >= (100 * rows["SDE"]["frac_far"] + 100 * rows["ODE"]["frac_far"]) / 2
        ax.annotate(name, (taus[0], y[0]), textcoords="offset points", xytext=(4, 10 if above else -16),
                    fontsize=8, color="#333333")  # label at tau = 5, where the curves are furthest apart
    ax.axhline(10, color="#999999", linewidth=1, linestyle="--")
    ax.text(4.95, 11, "held-out data (10% by construction)", fontsize=8, color="#666666")
    ax.set_xlim(5.05, -0.05)
    ax.set_xlabel("switch point τ (forward time; generation runs left to right)")
    ax.set_ylabel("% samples far from every training image")
    ax.set_title(f"Where the noise matters ({N_TRAJ} trajectories, band = ±1 SE)", fontsize=10)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "noise_window_curve.png"), dpi=150)
    plt.close(fig)

    cols = 10
    labels = list(finals)
    fig, axes = plt.subplots(len(labels), cols, figsize=(1.2 * cols, 1.2 * len(labels)))
    for r_, lab in enumerate(labels):
        for c in range(cols):
            a = axes[r_, c]
            a.imshow((finals[lab][c] * std_px + mean_px).reshape(GRID, GRID), cmap="gray", vmin=0, vmax=1)
            a.set_xticks([])
            a.set_yticks([])
        axes[r_, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Same 10 trajectories under each sampler; pixel space, fixed limits [0, 1]", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "noise_window_grid.png"), dpi=120)
    plt.close(fig)
    report(f"total time this run {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
