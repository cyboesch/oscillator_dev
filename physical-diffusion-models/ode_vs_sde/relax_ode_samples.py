#!/usr/bin/env python3
"""Corrector test: relax the saved probability-flow-ODE samples with physical Langevin dynamics
at the final energy theta(t=0) and the bath temperature, and see whether they un-blend.

    dx = -grad_x E(x; theta(0)) ds + sqrt(2 k_B T) dW,   s in [0, S_MAX]

This is the dynamics the reverse SDE adds on top of the ODE at its last slice, applied after the
fact. The SDE final states get the same treatment as a control. Snapshots are compared by:
background statistics, out-of-range pixels, distance to the nearest training image (pixel space;
blended digits are far from every real digit), and E/T at theta(0).

Read-only on the checkpoint and the ablation outputs. Writes to
out/diagnostics/ode_relaxation/<job id>/. Run under Slurm (cd physical-diffusion-models/ode_vs_sde && sbatch run_relax_ode_neuronic.sh).
"""

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

from physical_diffusion_fns.helper_fns import normalize_samples  # noqa: E402
from physical_diffusion_fns.network_fns import (  # noqa: E402
    create_2d_square_lattice_connectivity,
    setup_energy_fn,
    setup_overdamped_SDE,
    solve_SDE,
)

# Must match scripts/physical_diffusion_model_MNIST.py
KEY_SEED = 1
STD_ADDED_NOISE = 0.01
TEMP = 0.02
GRID = 28
N_NEIGHBOUR_COUPLINGS = 14
RTOL, ATOL, BROWNIAN_TOL = 1e-4, 1e-6, 1e-12  # the paper's SDE tolerances
EXPECTED_SHA256 = "37cfe46ef805a39129c265a0c17aaf6cf2b4e3b2ec7d8f55725e1cd66c227249"

# Relaxation settings
S_MAX = 0.5
N_SAVE = 26                 # snapshots every 0.02 time units
REPORT_S = [0.0, 0.02, 0.1, 0.5]
MAX_ABS_INPUT = 15.0        # skip diverged trajectories (stiff, would stall the vmapped batch)
OUT_OF_RANGE_MARGIN = 0.2
RELAX_SEED = 7

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
VARIANTS = {
    "ODE": "final_states_ablation_params_raw_interp_linear_step_knots_True_n40_matched_rng_probability_flow_ode_atol_1e-09_rtol_1e-07.npy",
    "SDE": "final_states_ablation_params_raw_interp_linear_step_knots_True_n40_matched_rng_sde_atol_1e-06_rtol_0.0001_btol_1e-12.npy",
}
OUT_DIR = os.path.join(PKG, "out", "diagnostics", "ode_relaxation", os.environ.get("SLURM_JOB_ID", "manual"))

LINES = []


def report(line=""):
    print(line, flush=True)
    LINES.append(line)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


def load_data():
    raw = jnp.array(np.load(DATA_PATH))
    _, _, image_noise_added_key = jr.split(jr.PRNGKey(KEY_SEED), 3)
    _, subkey = jr.split(image_noise_added_key)
    noisy = raw + jr.normal(subkey, raw.shape) * STD_ADDED_NOISE
    data, mean_px, std_px = normalize_samples(noisy)
    return np.asarray(raw), np.asarray(data), float(mean_px), float(std_px)


def nn_distance(px_samples, px_train, chunk=2000):
    """Euclidean distance (pixel space) from each sample to its nearest training image."""
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
    report(f"ODE relaxation test; jax {jax.__version__}; devices {jax.devices()}; default float {jnp.zeros(1).dtype}")

    raw, data, mean_px, std_px = load_data()
    pix_min, pix_max = (0.0 - mean_px) / std_px, (1.0 - mean_px) / std_px
    blank = np.all(raw == 0.0, axis=0)
    px_train = data * std_px + mean_px
    report(f"data {data.shape}; pixel range [{pix_min:.4f}, {pix_max:.4f}]; {int(blank.sum())} always-blank pixels")

    ckpt = np.load(CHECKPOINT_PATH, mmap_mode="r")
    theta0 = jnp.asarray(np.array(ckpt[-1], dtype=np.float64))  # row 99 is t = 0
    n = GRID * GRID
    connectivity = create_2d_square_lattice_connectivity(grid_size=GRID, n_neighbour_couplings=N_NEIGHBOUR_COUPLINGS)
    p = int(connectivity.shape[0])
    template = (jnp.ones(n), jnp.zeros(n), jnp.ones(n), jnp.ones(p), jnp.zeros(p), jnp.zeros(p), jnp.zeros(n))
    _, unflatten = flatten_util.ravel_pytree(template)
    energy_fn = setup_energy_fn(connectivity, unflatten)
    drift_fn, diffusion_fn = setup_overdamped_SDE(energy_fn, theta0, n, Temp=TEMP)
    energy_batch = jax.jit(vmap(lambda x: energy_fn(x, theta0)))

    # Reference: nearest-neighbour distance of held-out data images to the rest of the data.
    rng = np.random.default_rng(RELAX_SEED)
    held = rng.choice(data.shape[0], size=200, replace=False)
    rest = np.setdiff1d(np.arange(data.shape[0]), held)
    nn_data = nn_distance(px_train[held], px_train[rest])
    e_data = np.asarray(energy_batch(jnp.asarray(data[held]))) / TEMP
    report(f"reference (200 held-out data images): NN distance median {np.median(nn_data):.2f} "
           f"[p90 {np.percentile(nn_data, 90):.2f}]; E/T median {np.median(e_data):.1f}")
    report("")

    ts = np.linspace(0.0, S_MAX, N_SAVE)
    results = {}
    for label, fname in VARIANTS.items():
        x0 = np.asarray(np.load(os.path.join(ABLATION_DIR, fname)), dtype=np.float64)
        keep = np.max(np.abs(x0), axis=1) <= MAX_ABS_INPUT
        skipped = np.where(~keep)[0].tolist()
        x0 = x0[keep]
        report(f"{label}: relaxing {x0.shape[0]} samples for S = {S_MAX} at theta(0), T = {TEMP}"
               f" (skipped diverged samples {skipped})")
        keys = jr.split(jr.PRNGKey(RELAX_SEED + (0 if label == "ODE" else 1)), x0.shape[0])
        t_run = time.time()
        solve_one = lambda x, k: solve_SDE(
            drift_fn, diffusion_fn, x, k, 0.0, S_MAX, N_SAVE, 1e-8,
            rtol=RTOL, atol=ATOL, brownian_tolerance=BROWNIAN_TOL,
        )
        sol = vmap(solve_one)(jnp.asarray(x0), keys)
        ys = np.asarray(sol.ys)  # (n_samples, N_SAVE, n)
        steps = np.asarray(sol.stats["num_steps"]).reshape(-1)
        report(f"  solved in {time.time() - t_run:.0f} s; solver steps median {np.median(steps):.0f}, max {np.max(steps):.0f}")
        np.save(os.path.join(OUT_DIR, f"relaxed_{label}.npy"), ys)
        results[label] = ys

        report(f"  {'S':>5} {'bg within':>10} {'bg offset':>10} {'%out':>6} {'max|x|':>7} "
               f"{'NN med':>7} {'NN p90':>7} {'E/T med':>9}")
        bg_level = data[:, blank].mean()
        for s_val in REPORT_S:
            j = int(np.argmin(np.abs(ts - s_val)))
            x = ys[:, j, :]
            xb = x[:, blank]
            out = (x < pix_min - OUT_OF_RANGE_MARGIN) | (x > pix_max + OUT_OF_RANGE_MARGIN)
            nn = nn_distance(x * std_px + mean_px, px_train)
            e = np.asarray(energy_batch(jnp.asarray(x))) / TEMP
            report(f"  {ts[j]:>5.2f} {np.median(xb.std(axis=1)):>10.4f} "
                   f"{np.median(np.abs(xb.mean(axis=1) - bg_level)):>10.4f} {100 * out.mean():>6.2f} "
                   f"{np.max(np.abs(x)):>7.2f} {np.median(nn):>7.2f} {np.percentile(nn, 90):>7.2f} {np.median(e):>9.1f}")
        report(f"  (data: bg within {np.median(data[held][:, blank].std(axis=1)):.4f}, offset ~0, %out 0)")
        report("")

    # Figures: same 10 samples at each snapshot, pixel space, fixed limits.
    cols = 10
    rows = [(f"{lab} S={s:g}", lab, s) for lab in results for s in REPORT_S]
    fig, axes = plt.subplots(len(rows), cols, figsize=(1.2 * cols, 1.25 * len(rows)))
    for r, (name, lab, s_val) in enumerate(rows):
        j = int(np.argmin(np.abs(ts - s_val)))
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow((results[lab][c, j] * std_px + mean_px).reshape(GRID, GRID), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(f"Langevin relaxation at theta(0), T={TEMP}; pixel space, fixed limits [0, 1]", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "relaxation_grid.png"), dpi=150)
    plt.close(fig)

    for lab in results:
        ys = results[lab]
        fig, axes = plt.subplots(2 * 4, cols, figsize=(1.2 * cols, 1.25 * 8))
        for half, j in enumerate([0, N_SAVE - 1]):
            for idx in range(min(40, ys.shape[0])):
                ax = axes[half * 4 + idx // cols, idx % cols]
                ax.imshow((ys[idx, j] * std_px + mean_px).reshape(GRID, GRID), cmap="gray", vmin=0, vmax=1)
            for ax in axes[half * 4:(half + 1) * 4].ravel():
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"{lab}: all samples, top 4 rows S=0, bottom 4 rows S={S_MAX}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"relaxation_all_{lab}.png"), dpi=150)
        plt.close(fig)

    report(f"total time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
