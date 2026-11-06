#!/usr/bin/env python3
"""Where along the reverse path does the SDE's noise matter?

Hybrid samplers on the paper checkpoint (raw parameters, linear interpolation, solver steps
forced at the knots, the paper's shared initial states and Brownian keys):

  ODE        : probability-flow ODE on the whole path (re-run; sanity check vs the saved file)
  burst@tau  : ODE down to tau, Langevin at the frozen theta(tau) for S_BURST time units, ODE to 0
  SDE>ODE@tau: reverse SDE down to tau, then ODE to 0   (is early noise enough?)
  ODE>SDE@tau: ODE down to tau, then reverse SDE to 0   (is late noise enough?)
  SDE        : the saved reverse-SDE samples (reference)

Reverse time t in [0, 5], forward time tau = 5 - t.
Metric for blended digits: pixel-space distance to the nearest training image.
Read-only on the checkpoint. Writes out/diagnostics/ode_noise_window/<job id>/.
Run under Slurm (cd physical-diffusion-models/ode_vs_sde && sbatch run_ode_noise_window_neuronic.sh).
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
GRID = 28
N_NEIGHBOUR_COUPLINGS = 14
T_FORWARD = 5.0
N_TIME_STEPS = 100
RTOL_SDE, ATOL_SDE, BROWNIAN_TOL = 1e-4, 1e-6, 1e-12
RTOL_ODE, ATOL_ODE = 1e-7, 1e-9
DT0 = 1e-8

# Experiment settings
BURST_TAUS = [4.5, 3.5, 2.5, 1.5, 0.5]
S_BURST = 0.5
SWITCH_TAUS = [4.0, 3.0, 2.0, 1.0]
LATE_SWITCH_TAUS = [4.0, 2.0, 1.0]
MAX_ABS_FOR_BURST = 15.0     # diverged states skip the burst (stiff; would stall the batch)
OUT_OF_RANGE_MARGIN = 0.2
BURST_SEED = 11

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
OUT_DIR = os.path.join(PKG, "out", "diagnostics", "ode_noise_window", os.environ.get("SLURM_JOB_ID", "manual"))

LINES = []


def report(line=""):
    print(line, flush=True)
    LINES.append(line)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


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
    report(f"noise-window test; jax {jax.__version__}; devices {jax.devices()}; default float {jnp.zeros(1).dtype}")

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
    rng = np.random.default_rng(BURST_SEED)
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
    tau = lambda t: T_FORWARD - t
    theta_pf = lambda t: interp(tau(t)) - theta_lin
    theta_rev = lambda t: 2.0 * interp(tau(t)) - theta_lin
    drift_ode, _ = setup_overdamped_SDE(energy_fn, theta_pf, n, time_dependent_parms=True, Temp=TEMP)
    drift_sde, diffusion_sde = setup_overdamped_SDE(energy_fn, theta_rev, n, time_dependent_parms=True, Temp=TEMP)
    knots = np.sort(np.asarray(T_FORWARD - forward_time_pts))[1:-1]

    def knots_in(t0, t1):
        k = knots[(knots > t0 + 1e-12) & (knots < t1 - 1e-12)]
        return jnp.asarray(k) if k.size else None

    x_init = jnp.asarray(np.load(os.path.join(ABLATION_DIR, "shared_initial_states.npy")))
    keys_b = jnp.asarray(np.load(os.path.join(ABLATION_DIR, "shared_brownian_keys.npy")))
    n_traj = x_init.shape[0]

    def run_ode(x, t0, t1):
        if t1 <= t0:
            return x
        st = knots_in(t0, t1)
        sol = vmap(lambda y: solve_ODE(drift_ode, y, t0, t1, 2, DT0, rtol=RTOL_ODE, atol=ATOL_ODE, step_ts=st))(x)
        return sol.ys[:, -1, :]

    def run_sde(x, t0, t1):
        # NB: solve_SDE builds its VirtualBrownianTree on [t0, t1], so this segment gets a fresh Brownian path
        # (valid noise, but not the paper's [0, 5] path).
        if t1 <= t0:
            return x
        st = knots_in(t0, t1)
        sol = vmap(lambda y, k: solve_SDE(drift_sde, diffusion_sde, y, k, t0, t1, 2, DT0, rtol=RTOL_SDE,
                                          atol=ATOL_SDE, brownian_tolerance=BROWNIAN_TOL, step_ts=st))(x, keys_b)
        return sol.ys[:, -1, :]

    def run_burst(x, tau_b):
        x = np.array(x)  # writable copy
        ok = np.max(np.abs(x), axis=1) <= MAX_ABS_FOR_BURST
        theta_b = interp(tau_b)
        drift_b, diff_b = setup_overdamped_SDE(energy_fn, theta_b, n, Temp=TEMP)
        keys = jr.split(jr.PRNGKey(BURST_SEED + int(round(10 * tau_b))), int(ok.sum()))
        sol = vmap(lambda y, k: solve_SDE(drift_b, diff_b, y, k, 0.0, S_BURST, 2, DT0, rtol=RTOL_SDE,
                                          atol=ATOL_SDE, brownian_tolerance=BROWNIAN_TOL))(jnp.asarray(x[ok]), keys)
        x[ok] = np.asarray(sol.ys[:, -1, :])
        return jnp.asarray(x), int((~ok).sum())

    variants = {}

    def record(label, x, t_start, note=""):
        variants[label] = np.asarray(x)
        gc.collect()
        jax.clear_caches()  # each solve embeds the 436 MB parameter table; free compiled programs
        report(f"  {label:<14} done in {time.time() - t_start:6.0f} s {note}")

    report(f"{n_traj} trajectories; knots at the parameter grid; ODE tol {RTOL_ODE}/{ATOL_ODE}, SDE tol {RTOL_SDE}/{ATOL_SDE}")
    t_s = time.time()
    x_ode = run_ode(x_init, 0.0, T_FORWARD)
    saved = np.load(os.path.join(ABLATION_DIR, SAVED_ODE))
    record("ODE", x_ode, t_s, f"(max |diff| to the saved ODE samples: {np.max(np.abs(np.asarray(x_ode) - saved)):.2e})")
    variants["SDE (saved)"] = np.load(os.path.join(ABLATION_DIR, SAVED_SDE))

    for tau_b in BURST_TAUS:
        t_s = time.time()
        x = run_ode(x_init, 0.0, T_FORWARD - tau_b)
        x, skipped = run_burst(x, tau_b)
        x = run_ode(x, T_FORWARD - tau_b, T_FORWARD)
        record(f"burst@{tau_b:g}", x, t_s, f"(burst skipped for {skipped} diverged states)" if skipped else "")

    for tau_s in SWITCH_TAUS:
        t_s = time.time()
        x = run_sde(x_init, 0.0, T_FORWARD - tau_s)
        x = run_ode(x, T_FORWARD - tau_s, T_FORWARD)
        record(f"SDE>ODE@{tau_s:g}", x, t_s)

    for tau_s in LATE_SWITCH_TAUS:
        t_s = time.time()
        x = run_ode(x_init, 0.0, T_FORWARD - tau_s)
        x = run_sde(x, T_FORWARD - tau_s, T_FORWARD)
        record(f"ODE>SDE@{tau_s:g}", x, t_s)

    np.savez(os.path.join(OUT_DIR, "final_states.npz"), **{k.replace("@", "_at_").replace(">", "_then_").replace(" ", "_"): v
                                                           for k, v in variants.items()})

    report("")
    report("far = nearest-image distance above the data's p90; bg within = median per-sample std of blank pixels "
           "(data 0.033); %out = pixels beyond the data range +- 0.2")
    report(f"{'variant':<14}{'NN med':>8}{'NN p90':>8}{'% far':>7}{'bg within':>10}{'%out':>7}{'max|x|':>8}")
    rows = {}
    for label, x in variants.items():
        nn = nn_distance(x * std_px + mean_px, px_train)
        out = (x < pix_min - OUT_OF_RANGE_MARGIN) | (x > pix_max + OUT_OF_RANGE_MARGIN)
        rows[label] = {
            "nn_median": float(np.median(nn)), "nn_p90": float(np.percentile(nn, 90)),
            "frac_far": float(np.mean(nn > blend_threshold)),
            "bg_within": float(np.median(x[:, blank].std(axis=1))),
            "frac_out": float(out.mean()), "max_abs": float(np.max(np.abs(x))),
        }
        r = rows[label]
        report(f"{label:<14}{r['nn_median']:>8.2f}{r['nn_p90']:>8.2f}{100 * r['frac_far']:>7.1f}"
               f"{r['bg_within']:>10.4f}{100 * r['frac_out']:>7.2f}{r['max_abs']:>8.2f}")
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)

    cols = 10
    labels = list(variants)
    fig, axes = plt.subplots(len(labels), cols, figsize=(1.2 * cols, 1.2 * len(labels)))
    for r, lab in enumerate(labels):
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow((variants[lab][c] * std_px + mean_px).reshape(GRID, GRID), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Same 10 trajectories under each hybrid sampler; pixel space, fixed limits [0, 1]", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "noise_window_grid.png"), dpi=150)
    plt.close(fig)
    report(f"total time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
