"""Replay the paper's MNIST reverse-SDE run (Fig. 4b) and keep the whole trajectories.

The paper sampler (scripts/physical_diffusion_model_MNIST.py, non-ablation branch, run of
2026-07-21) integrated all 40 trajectories but saved only the final states. This script
repeats exactly that solve -- same checkpoint, same RNG splits, same solver settings, no
step_ts -- and saves the states at n_save evenly spaced reverse times s in [0, tau].
It then checks the replayed final states against the saved ones and draws frame strips.

Read-only with respect to the paper checkpoint; writes only to <this folder>/out/<job id>/.
"""
import os
import sys
import json
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
PDM = os.path.dirname(os.path.dirname(HERE))  # physical-diffusion-models/
sys.path.append(PDM)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import flatten_util, vmap

from physical_diffusion_fns.helper_fns import interpolate_parameters
from physical_diffusion_fns.network_fns import (
    setup_energy_fn,
    setup_overdamped_SDE,
    solve_SDE,
    create_2d_square_lattice_connectivity,
)

jax.config.update("jax_enable_x64", True)

# ---------------- paper configuration (copied from the MNIST driver) ----------------
key_seed = 1
Temp = 0.02
sigma_forward = 1.0
t_forward = 5.0
n_time_steps = 100
grid = 28
n_neighbour_couplings = 14
n_trajectories = 40
rtol_sde, atol_sde, brownian_tolerance, dt0 = 1e-4, 1e-6, 1e-12, 1e-8
clip_min, clip_max = -0.4096, 2.8727      # normalised pixel range, as in Fig. 4b
expected_sha = "37cfe46ef805a39129c265a0c17aaf6cf2b4e3b2ec7d8f55725e1cd66c227249"

# ---------------- what this replay adds ----------------
n_save = 501                               # saved reverse times, s = 0, 0.01, ..., 5
traj_of_interest = [0, 4]                  # first "0" in Fig. 4b, and the cleaner "0" at position 5
fig4b_trajectories = list(range(20))       # the 20 samples shown in Fig. 4b

RUN = os.path.join(
    PDM, "out", "2026_07_08", "problems", "MNIST_generation",
    "labels_[0, 1]_resolution_28_x_28",
    "n_couplings_14_Temp_0.02_energy_fn_type_6th_order_duffing_coupling",
    "added_gaussian_noise_std_0.01_key_seed_1_additional_rescaling_1",
    "training_method_SM_at_kbT_minimal_memory_solve_opt_in_reverse_True_exp_time_pts_False_"
    "t_forward_5.0_n_timesteps_100_sigma_forward_1.0_lr_2e-05_to_0.002_epochs_100000_batch_512_"
    "window_1000_tol_0.001_to_0.001_patience_300_to_100_",
    "opt_per_time_plots",
)
CKPT = os.path.join(RUN, "params_history.npy")
SDE_SUFFIX = ("non_smoothed_sde_memory_efficient_atol_1e-06_rtol_0.0001_brownian_tolerance_1e-12"
              "_init_method_asymptotic_gaussian_use_same_initial_condition_for_all_trajectories_False")
SAVED_FINAL = {
    "paper_2026_07_21": os.path.join(RUN, f"final_states_{SDE_SUFFIX}.npy"),
    "reproduction_check_2026_08_22": os.path.join(RUN, f"final_states_{SDE_SUFFIX}_reproduction_check.npy"),
}
SAVED_INIT = os.path.join(RUN, "interpolation_ablation_v1", "shared_initial_states.npy")

job = os.environ.get("SLURM_JOB_ID", "local")
OUT = os.path.join(HERE, "out", job)
os.makedirs(OUT, exist_ok=True)
print("Output:", OUT)
print("Devices:", jax.devices())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


sha = sha256(CKPT)
assert sha == expected_sha, f"checkpoint SHA mismatch: {sha}"
params_history = jnp.asarray(np.load(CKPT))
assert params_history.shape == (100, 545566), params_history.shape

# ---------------- network ----------------
N = grid * grid
connectivity = create_2d_square_lattice_connectivity(grid_size=grid, n_neighbour_couplings=n_neighbour_couplings)
P = connectivity.shape[0]
template = (jnp.ones(N), jnp.zeros(N), jnp.ones(N), jnp.ones(P), jnp.zeros(P), jnp.zeros(P), jnp.zeros(N))
params_flat0, unflatten = flatten_util.ravel_pytree(template)
energy_fn = setup_energy_fn(connectivity, unflatten)

forward_time_pts = jnp.linspace(0.0, t_forward, n_time_steps)[::-1]
interp = interpolate_parameters(params_history, forward_time_pts)
forward_params = jnp.zeros_like(params_flat0).at[0:N].set(1.0)
params_reverse = lambda s: 2 * interp(t_forward - s) - forward_params / sigma_forward**2
drift_fn, diffusion_fn = setup_overdamped_SDE(energy_fn, params_reverse, N, time_dependent_parms=True, Temp=Temp)

# ---------------- RNG, exactly as the driver ----------------
master = jr.PRNGKey(key_seed)
_, reverse_key, _ = jr.split(master, 3)
k = reverse_key
for _ in range(2):
    k, _ = jr.split(k)
k, init_sub = jr.split(k)
initial_states = jnp.sqrt(Temp) * sigma_forward * jr.normal(init_sub, shape=(n_trajectories, N))
k, brown_sub = jr.split(k)
brownian_keys = jr.split(brown_sub, n_trajectories)

init_check = None
if os.path.exists(SAVED_INIT):
    init_check = float(np.max(np.abs(np.load(SAVED_INIT) - np.asarray(initial_states))))
    print(f"max |initial states - saved ablation initial states| = {init_check:.3e}")

# ---------------- solve (paper sampler, no step_ts) ----------------
solve_one = lambda x0, kb: solve_SDE(
    drift_fn, diffusion_fn, x0, kb, 0.0, t_forward, n_save, dt0,
    brownian_tolerance=brownian_tolerance, rtol=rtol_sde, atol=atol_sde,
)
sol = vmap(solve_one, in_axes=(0, 0))(initial_states, brownian_keys)
ys = np.asarray(sol.ys)                     # (40, n_save, 784)
s_grid = np.linspace(0.0, t_forward, n_save)
num_steps = np.asarray(sol.stats["num_steps"]) if "num_steps" in sol.stats else None
assert np.all(np.isfinite(ys)), "non-finite states"

np.save(os.path.join(OUT, "trajectories_all40_float32.npy"), ys.astype(np.float32))
np.save(os.path.join(OUT, "reverse_times_s.npy"), s_grid)
for i in traj_of_interest:
    np.save(os.path.join(OUT, f"trajectory_{i:02d}_float64.npy"), ys[i])

# ---------------- does the replay match the saved paper samples? ----------------
final = ys[:, -1, :]
checks = {}
for name, path in SAVED_FINAL.items():
    if os.path.exists(path):
        saved = np.load(path)
        d = np.max(np.abs(final - saved), axis=1)
        checks[name] = {"max_abs_diff_per_traj": d.tolist(), "max_abs_diff": float(d.max())}
        print(f"{name}: max |replay - saved| over all pixels/trajectories = {d.max():.3e}; "
              f"traj {traj_of_interest}: {[float(d[i]) for i in traj_of_interest]}")


# ---------------- figures ----------------
def show(ax, x, mode):
    img = np.clip(x.reshape(grid, grid), clip_min, clip_max)
    if mode == "fixed":   # one grey scale for every frame: [clip_min, clip_max]
        ax.imshow(img, cmap="gray", vmin=clip_min, vmax=clip_max)
    else:                 # per-frame autoscale, as Fig. 4b was drawn
        ax.imshow(img, cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])


def strip(traj, s_values, mode, path, title):
    idx = [int(round(s / t_forward * (n_save - 1))) for s in s_values]
    fig, axes = plt.subplots(1, len(idx), figsize=(1.6 * len(idx), 2.0))
    for ax, j in zip(np.atleast_1d(axes), idx):
        show(ax, ys[traj, j], mode)
        ax.set_title(f"s={s_grid[j]:.2f}", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


dense_s = np.round(np.arange(0.0, t_forward + 1e-9, 0.25), 2)
five_s = [0.0, 2.5, 3.5, 4.25, 5.0]
for i in traj_of_interest:
    for mode in ("fixed", "autoscale"):
        strip(i, dense_s, mode, os.path.join(OUT, f"traj{i:02d}_dense_every0.25_{mode}.png"),
              f"Paper MNIST model, reverse SDE, trajectory {i} (Fig. 4b position {i + 1}); grey scale: {mode}")
        strip(i, five_s, mode, os.path.join(OUT, f"traj{i:02d}_five_frames_{mode}.png"),
              f"trajectory {i}, s = {five_s}; grey scale: {mode}")
        # individual frames for the figure
        for s in dense_s:
            j = int(round(s / t_forward * (n_save - 1)))
            fig, ax = plt.subplots(figsize=(2, 2))
            show(ax, ys[i, j], mode)
            ax.axis("off")
            fig.savefig(os.path.join(OUT, f"frame_traj{i:02d}_s{s:.2f}_{mode}.png"),
                        dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

# overview: the 20 Fig. 4b trajectories at 11 reverse times, to pick a different one if wanted
ov_s = np.round(np.arange(0.0, t_forward + 1e-9, 0.5), 2)
for mode in ("fixed", "autoscale"):
    fig, axes = plt.subplots(len(fig4b_trajectories), len(ov_s), figsize=(1.1 * len(ov_s), 1.1 * len(fig4b_trajectories)))
    for r, i in enumerate(fig4b_trajectories):
        for c, s in enumerate(ov_s):
            show(axes[r, c], ys[i, int(round(s / t_forward * (n_save - 1)))], mode)
            if r == 0:
                axes[r, c].set_title(f"s={s:.1f}", fontsize=7)
        axes[r, 0].set_ylabel(f"traj {i}", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"overview_fig4b_trajectories_0-19_{mode}.png"), dpi=150)
    plt.close(fig)

summary = {
    "job": job,
    "checkpoint": CKPT,
    "checkpoint_sha256": sha,
    "sampler": "reverse SDE, SRA1, PIDController(rtol=1e-4, atol=1e-6), dt0=1e-8, "
               "VirtualBrownianTree tol 1e-12, no step_ts (as the paper sampler)",
    "n_save": n_save,
    "saved_states_note": "states between accepted solver steps are diffrax's local linear interpolation",
    "init_states_vs_saved_ablation_max_abs_diff": init_check,
    "final_state_checks": checks,
    "num_steps": num_steps.tolist() if num_steps is not None else None,
    "devices": [str(d) for d in jax.devices()],
    "versions": {"jax": jax.__version__, "numpy": np.__version__},
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Done.")
