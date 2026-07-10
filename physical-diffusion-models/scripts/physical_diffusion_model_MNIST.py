print("Script started.")
import os
import sys
import gc
import shutil
from datetime import datetime

# Make the physical_diffusion_fns package importable (parent dir of scripts/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import flatten_util, vmap
import jax.numpy as jnp
import jax.random as jr

from physical_diffusion_fns.helper_fns import (
    normalize_samples,
    sample_forward_process,
    smooth_parameters,
    interpolate_parameters,
)
from physical_diffusion_fns.learning_fns import solve_score_matching_cg
from physical_diffusion_fns.plotting_fns import (
    plot_forward_marginals,
    plot_parameter_as_fn_of_time,
    visualize_connectivity_with_non_local_couplings,
)
from physical_diffusion_fns.network_fns import (
    setup_energy_fn,
    setup_overdamped_SDE,
    solve_SDE,
    create_2d_square_lattice_connectivity,
)
from physical_diffusion_fns.network_fns_memory_efficient import (
    setup_duffing_network_analytical_derivatives_memory_efficient,
)
from physical_diffusion_fns.learning_fns_memory_efficient import (
    setup_score_matching_kbT_loss_minimal_memory,
)

jax.config.update("jax_enable_x64", True)


def print_memory_usage(label=""):
    """Print current resident memory usage."""
    rss = psutil.Process(os.getpid()).memory_info().rss
    print(f"Memory usage {label}: {rss / 1024 / 1024:.2f} MB")


def optimize_memory():
    """Force garbage collection and clear JAX caches."""
    gc.collect()
    jax.clear_caches()


print_memory_usage("at start")
print(f"Number of devices: {jax.local_device_count()}")

#####################################
# Configuration
#####################################
# --- RNG ---
key_seed = 1
key_seed_reverse = None  # None -> derived from master key; set an int for an independent reverse-sampling seed
master_key = jax.random.PRNGKey(key_seed)
optimization_key, _default_reverse_sde_key, image_noise_added_key = jr.split(master_key, 3)
reverse_sde_key = jax.random.PRNGKey(key_seed_reverse) if key_seed_reverse is not None else _default_reverse_sde_key

# --- Data ---
labels = [0, 1]
resolution = (28, 28)
balanced = True
std_of_added_noise = 0.01     # Gaussian noise added to the data for regularization
additional_rescaling = 1      # extra scalar applied to the (normalized) target samples

# --- Forward (noising) Ornstein-Uhlenbeck process ---
Temp = 0.02                   # bath temperature = diffusion constant D
sigma_forward = 1.0
t_forward = 5.0
n_time_steps = 100
# The energy is trained separately at each forward time. We optimize from the noisy
# (large-t) end down to the data (t=0), warm-starting each slice from the previous one,
# so the time points and every per-slice schedule below are stored in reverse order.
forward_time_pts = jnp.linspace(0.0, t_forward, n_time_steps)[::-1]
print("forward_time_pts", forward_time_pts)

# --- Convex solver: matrix-free ridge-regularized CG (one exact solve per forward-time slice) ---
# The per-slice implicit-score-matching loss is an exact convex quadratic in theta (the energy is
# linear in theta and all units are visible), so instead of SGD we solve the ridge-regularized
# normal equations (H + lambda I) theta = c directly and matrix-free (H is never formed; P ~ 5e5).
# See physical_diffusion_fns/learning_fns.py:solve_score_matching_cg.
# cg_ridge_rel and cg_sample_size are the two regularization dials the exact convex solve needs
# (SGD used to set them implicitly). They are env-overridable so a sweep is just a shell loop --
# each (ridge, sample) combo lands in its own output directory:
#   CG_RIDGE_REL=1e-2 CG_SAMPLE_SIZE=16384 python3 physical_diffusion_model_MNIST.py
cg_sample_size = int(os.environ.get("CG_SAMPLE_SIZE", 8192))          # fixed sample of p_t per slice used to FIT (>~ P/N_osc for full rank)
cg_heldout_size = int(os.environ.get("CG_HELDOUT_SIZE", cg_sample_size))  # independent sample for the held-out-loss diagnostic
cg_ridge_rel = float(os.environ.get("CG_RIDGE_REL", 1e-2))           # Tikhonov ridge: lambda = cg_ridge_rel * (mean eigenvalue of H)
cg_tol = 1e-5            # conjugate-gradient relative-residual tolerance
cg_maxiter = 500         # conjugate-gradient iteration cap
cg_active_set_iters = 8  # max active-set passes for the k_6 >= epsilon box constraint

# --- Network: 2D lattice of coupled 6th-order Duffing oscillators, one oscillator per pixel ---
n_neighbour_couplings = 14

# --- Reverse (generation) SDE ---
n_trajectories = 40
rtol_sde = 1e-4
atol_sde = 1e-6
brownian_tolerance = 1e-12
# Use Savitzky-Golay-smoothed theta(t) for generation. The per-slice convex solves are independent,
# so theta(t) can be jaggier than the old warm-started-SGD trajectory; smoothing steadies the
# reverse dynamics. Env-overridable (USE_SMOOTHED_REVERSE=0 to disable).
use_smoothed_params_for_reverse = bool(int(os.environ.get("USE_SMOOTHED_REVERSE", 1)))

# --- Checkpointing ---
start_from_scratch = False

# Fixed labels used only for output-directory naming.
training_method = "SM_at_kbT_convex_cg"   # note: still starts with "SM_at_kbT" -> reverse-SDE scaling unchanged
energy_fn_type = "6th_order_duffing_coupling"

#####################################
# Output directories
#####################################
problem_type_folder = "MNIST_generation"
system_specifics = f"n_couplings_{n_neighbour_couplings}_Temp_{Temp}_energy_fn_type_{energy_fn_type}"
MNIST_specifics = f"labels_{labels}_resolution_{resolution[0]}_x_{resolution[1]}"
data_folder = f"added_gaussian_noise_std_{std_of_added_noise}_key_seed_{key_seed}_additional_rescaling_{additional_rescaling}"
optimization_folder = (
    f"training_method_{training_method}_solve_opt_in_reverse_True_"
    f"exp_time_pts_False_"
    f"t_forward_{t_forward}_"
    f"n_timesteps_{n_time_steps}_"
    f"sigma_forward_{sigma_forward}_"
    f"cg_sample_{cg_sample_size}_"
    f"ridge_rel_{cg_ridge_rel}_"
    f"cg_tol_{cg_tol}_"
    f"cg_maxiter_{cg_maxiter}_"
)

here = os.path.dirname(os.path.abspath(__file__))
current_date = datetime.now().strftime("%Y_%m_%d")
base_dir = os.path.join(here, "..", "out", current_date, "problems")
output_dir_root = os.path.join(base_dir, problem_type_folder, MNIST_specifics, system_specifics, data_folder, optimization_folder)
output_dir = os.path.join(output_dir_root, "opt_per_time_plots")
plot_folder = os.path.join(output_dir_root, "final_plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)
print(f"Output directory: {output_dir}")
print(f"Plot directory: {plot_folder}")

#####################################
# Load MNIST data
#####################################
path_to_data = os.path.join(here, "..", "data", "MNIST", f"mnist_labels_{labels}_resolution_{resolution}_balanced_{balanced}.npy")
images_flat_raw = jnp.array(np.load(path_to_data))
n_samples = images_flat_raw.shape[0]
print("Data shape:", images_flat_raw.shape)

# Add small Gaussian noise for regularization, then standardize to zero-mean / unit-variance.
_, subkey_image_noise_added = jr.split(image_noise_added_key)
images_flat_true = images_flat_raw + jr.normal(subkey_image_noise_added, images_flat_raw.shape) * std_of_added_noise
samples_target_unscaled, mean_MNIST, std_MNIST = normalize_samples(images_flat_true)
# Original pixel range [0, 1] mapped through the normalization (x - mean) / std
pixel_min_normalized = (0.0 - mean_MNIST) / std_MNIST
pixel_max_normalized = (1.0 - mean_MNIST) / std_MNIST
print(f"Normalized pixel range: [{pixel_min_normalized:.4f}, {pixel_max_normalized:.4f}]")
samples_target = samples_target_unscaled * additional_rescaling

#####################################
# Setup network (energy function + analytical derivatives)
#####################################
N_osc = resolution[0] ** 2
connectivity = create_2d_square_lattice_connectivity(grid_size=resolution[0], n_neighbour_couplings=n_neighbour_couplings)
num_connections = connectivity.shape[0]
visualize_connectivity_with_non_local_couplings(
    connectivity, grid_size_x=resolution[0], grid_size_y=resolution[1],
    n_neighbour_couplings=n_neighbour_couplings, save_fig=True, path=output_dir,
)
print(f"Network size: {N_osc} oscillators, {num_connections} connections")

# Initial energy parameters. Layout (flattened by ravel_pytree in this order):
#   per-oscillator: k_lin, k_duff, k_6, bias ;  per-edge: c_lin, c_optomech, c_duff
k_lin_0 = jnp.ones(N_osc)
k_duff_0 = jnp.zeros(N_osc)
k_6_0 = jnp.ones(N_osc)
c_lin_0 = jnp.ones(num_connections)
c_optomech_0 = jnp.zeros(num_connections)
c_duff_0 = jnp.zeros(num_connections)
biases_0 = jnp.zeros(N_osc)
params_initial = (k_lin_0, k_duff_0, k_6_0, c_lin_0, c_optomech_0, c_duff_0, biases_0)
params_names = ["k_lin", "k_duff", "k_6", "c_lin", "c_optomech", "c_duff", "biases"]
params_flattened_initial, unflatten = flatten_util.ravel_pytree(params_initial)
# Keep the k_6 (sextic) self-coefficients >= epsilon during training so the potential stays confining.
constraint_indices = jnp.arange(2 * N_osc, 3 * N_osc)

energy_fn = setup_energy_fn(connectivity, unflatten)
# The minimal-memory score-matching loss only needs the energy gradient and Hessian trace;
# it obtains the parameter gradient by autodiff, so the analytical param-Jacobians are unused.
gradient_fn, trace_hessian_fn, _, _ = setup_duffing_network_analytical_derivatives_memory_efficient(connectivity, unflatten)
print(f"Total parameters: {len(params_flattened_initial)}")
print_memory_usage("after network setup")

#####################################
# Learning objective: implicit score matching at temperature Temp (minimal-memory autodiff)
#####################################
loss_fn_per_batch, gradient_fn_per_batch = setup_score_matching_kbT_loss_minimal_memory(
    gradient_fn, trace_hessian_fn, k_b=1.0, T=Temp,
)
print(f"Score matching (minimal-memory autodiff) at k_bT = {Temp}")

#####################################
# Optimization loop (with resume-from-checkpoint support)
#####################################
params_history_path = f"{output_dir}/params_history.npy"
time_index_path = f"{output_dir}/current_time_index.txt"

if start_from_scratch:
    print("Starting from scratch")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plot_folder, exist_ok=True)
    start_t_idx = 0
    params_history_all_t = []
    current_params = params_flattened_initial
elif os.path.exists(params_history_path) and os.path.exists(time_index_path):
    print("Loading existing parameters from", params_history_path)
    full_params_history = jnp.load(params_history_path)
    current_params = full_params_history[-1]
    params_history_all_t = full_params_history.tolist()
    with open(time_index_path, "r") as f:
        start_t_idx = int(f.read())
    print(f"Resuming from time index {start_t_idx} ({len(params_history_all_t)} parameter sets loaded)")
else:
    print("No existing parameters found, starting from beginning")
    start_t_idx = 0
    params_history_all_t = []
    current_params = params_flattened_initial

# Per-slice diagnostics accumulated this session (fit-vs-held-out loss detects overfitting).
cg_diag_t, cg_diag_fit, cg_diag_heldout, cg_diag_resid = [], [], [], []

if start_t_idx < len(forward_time_pts):
    for t_idx in range(start_t_idx, len(forward_time_pts)):
        t_curr = forward_time_pts[t_idx]
        optimization_key, subkey_sample, subkey_solve, subkey_heldout = jr.split(optimization_key, 4)

        # On the first slice, sanity-check that the forward process reaches the target Gaussian.
        if t_idx == 0:
            samples_0 = sample_forward_process(t_forward, n_samples, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=optimization_key)
            plot_forward_marginals(samples_0, t_forward, sigma_forward, Temp=Temp, beta=1.0, path=plot_folder, save_fig=True, fontsize=16, plot_show=True)

        # One large fixed sample of the noised marginal p_{t_curr}, held constant across the CG solve.
        batch = sample_forward_process(
            t_curr, n_samples=cg_sample_size, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=subkey_sample, beta=1,
        )

        # Exact convex solve for this slice, warm-started from the previous slice's parameters.
        current_params, info = solve_score_matching_cg(
            gradient_fn_per_batch,
            current_params,
            batch,
            ridge_rel=cg_ridge_rel,
            constraint_indices=constraint_indices,
            epsilon=0.01,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
            active_set_iters=cg_active_set_iters,
            loss_fn_per_batch=loss_fn_per_batch,
            key=subkey_solve,
        )

        # Held-out diagnostic: SM loss on an INDEPENDENT fresh sample of p_{t_curr}. info['loss_after']
        # is in-sample (biased low); if loss_heldout >> loss_fit the solve is overfitting -> raise
        # cg_ridge_rel and/or cg_sample_size.
        heldout_batch = sample_forward_process(
            t_curr, n_samples=cg_heldout_size, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=subkey_heldout, beta=1,
        )
        heldout_loss = float(loss_fn_per_batch(current_params, heldout_batch))
        print(
            f"t_idx {t_idx:3d}  t={float(t_curr):.3f}  loss_fit {info['loss_after']:.2f}  loss_heldout {heldout_loss:.2f}  "
            f"ridge={info['ridge']:.2e}  k6_at_floor={info['n_active']}/{N_osc}  free_residual={info['residual']:.2e}"
        )
        cg_diag_t.append(float(t_curr)); cg_diag_fit.append(info["loss_after"])
        cg_diag_heldout.append(heldout_loss); cg_diag_resid.append(info["residual"])

        params_history_all_t.append(current_params)
        optimize_memory()

        # Checkpoint after each slice so the run is resumable.
        jnp.save(params_history_path, jnp.array(params_history_all_t))
        with open(time_index_path, "w") as f:
            f.write(str(t_idx + 1))  # next time index to resume from

    print("Optimization complete; saved parameters to", output_dir)

# Save + plot the fit-vs-held-out loss diagnostic for slices run this session.
if cg_diag_t:
    np.savez(
        f"{output_dir}/cg_loss_diagnostics.npz",
        t=np.array(cg_diag_t), loss_fit=np.array(cg_diag_fit),
        loss_heldout=np.array(cg_diag_heldout), free_residual=np.array(cg_diag_resid),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(cg_diag_t)
    ax.plot(np.array(cg_diag_t)[order], np.array(cg_diag_fit)[order], "-o", ms=3, label="fit (in-sample)")
    ax.plot(np.array(cg_diag_t)[order], np.array(cg_diag_heldout)[order], "-o", ms=3, label="held-out")
    ax.set_xlabel("forward time t"); ax.set_ylabel("score-matching loss")
    ax.set_title(f"Fit vs held-out SM loss (ridge_rel={cg_ridge_rel}, sample={cg_sample_size})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_folder}/cg_fit_vs_heldout_loss.png", dpi=150)
    plt.close(fig)
    print(f"Saved CG loss diagnostics to {output_dir}/cg_loss_diagnostics.npz and plot to {plot_folder}/cg_fit_vs_heldout_loss.png")

params_history_all_t = jnp.array(params_history_all_t)

#####################################
# Interpolate the learned parameters as a function of forward time
#####################################
print(f"params_history_all_t shape: {params_history_all_t.shape}, forward_time_pts shape: {forward_time_pts.shape}")

smoothed_params = smooth_parameters(
    params_history_all_t,
    window_lengths=[10] * params_history_all_t.shape[1],
    poly_orders=[3] * params_history_all_t.shape[1],
)
params_interpolator_smoothed = interpolate_parameters(smoothed_params, forward_time_pts)
params_interpolator_non_smoothed = interpolate_parameters(params_history_all_t, forward_time_pts)

plot_parameter_as_fn_of_time(
    params_names, forward_time_pts, forward_time_pts, params_history_all_t,
    params_interpolator_smoothed, unflatten, N_osc, save_fig=True, path=plot_folder, log_scale=False,
)

#####################################
# Reverse (generation) SDE
#####################################
def run_reverse_process_SDE(params_interpolator, suffix, key):
    """Integrate the reverse-time Langevin SDE to generate samples.

    For the OU forward process the reverse drift is (1/sigma^2) x - 2 grad_x E_theta(tau(t)).
    Exploiting the energy's linearity in the parameters, this is assembled purely by parameter
    arithmetic: theta_reverse(t) = 2 * theta(tau(t)) - theta_linear, where theta_linear is the
    pure harmonic reference (k_lin = 1) and tau(t) = t_forward - t. The initial condition is the
    asymptotic forward Gaussian N(0, Temp * sigma_forward^2 I). Only final states are kept.
    """
    tau = lambda t: t_forward - t
    forward_params = jnp.zeros_like(params_flattened_initial).at[0:N_osc].set(1.0)
    params_reverse = lambda t: 2 * params_interpolator(tau(t)) - forward_params / sigma_forward**2

    drift_fn, diffusion_fn = setup_overdamped_SDE(energy_fn, params_reverse, N_osc, time_dependent_parms=True, Temp=Temp)

    t0, t1 = 0.0, t_forward
    n_time_steps_sde = 100          # number of saved output points (does not affect solver accuracy)
    ts = jnp.linspace(t0, t1, n_time_steps_sde)
    dt0 = 1e-8

    # Initial states ~ N(0, Temp * sigma_forward^2 I), and one Brownian key per trajectory.
    key, subkey_init = jr.split(key)
    initial_states = jnp.sqrt(Temp) * sigma_forward * jr.normal(subkey_init, shape=(n_trajectories, N_osc))
    key, subkey_brownian = jr.split(key)
    keys_brownian = jr.split(subkey_brownian, n_trajectories)

    print(f"Running reverse SDE for {n_trajectories} trajectories...")
    solve_one = lambda init_state, key_b: solve_SDE(
        drift_fn, diffusion_fn, init_state, key_b, t0, t1, ts.shape[0], dt0,
        brownian_tolerance=brownian_tolerance, rtol=rtol_sde, atol=atol_sde,
    )
    solutions = vmap(solve_one, in_axes=(0, 0))(initial_states, keys_brownian)
    final_samples_scaled = solutions.ys[:, -1, :]  # (n_trajectories, N_osc), final states only

    final_states_path = f"{output_dir}/final_states_{suffix}.npy"
    jnp.save(final_states_path, final_samples_scaled)
    print(f"Final states saved to {final_states_path}")

    del solutions
    optimize_memory()
    final_samples = final_samples_scaled / additional_rescaling
    return final_samples.reshape(-1, resolution[0], resolution[1])


print(f"Running reverse SDE with final time: {t_forward}")
reverse_interpolator = params_interpolator_smoothed if use_smoothed_params_for_reverse else params_interpolator_non_smoothed
params_tag = "smoothed" if use_smoothed_params_for_reverse else "non_smoothed"
reverse_sde_suffix = (
    f"{params_tag}_sde_memory_efficient_atol_{atol_sde}_rtol_{rtol_sde}_brownian_tolerance_{brownian_tolerance}"
    f"_init_method_asymptotic_gaussian"
    f"_use_same_initial_condition_for_all_trajectories_False"
    + (f"_reverse_rng_seed_{key_seed_reverse}" if key_seed_reverse is not None else "")
)
images_generated_sde = run_reverse_process_SDE(reverse_interpolator, reverse_sde_suffix, reverse_sde_key)
print_memory_usage("after reverse SDE")

#####################################
# Plotting: true vs. generated digit grids
#####################################
def plot_true_vs_generated_grid(true_imgs, generated_imgs, generated_title, save_path):
    """Top block: true images; bottom block: generated images (<=10 columns per row).
    Robust to any n_trajectories (not only multiples of 10)."""
    n_show = n_trajectories
    ncol = min(10, n_show)
    n_rows = max(1, -(-n_show // ncol))  # ceil(n_show / ncol)
    fig, axes = plt.subplots(2 * n_rows, ncol, figsize=(1.5 * ncol, 3 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")  # blank any unused cells (partial last row)
    for idx in range(n_show):
        for row_block, imgs in ((0, true_imgs), (n_rows, generated_imgs)):
            ax = axes[idx // ncol + row_block, idx % ncol]
            img = imgs[idx]
            if img.shape[-1] == 1:  # drop singleton channel if present
                img = img.squeeze(-1)
            ax.imshow(np.array(img), cmap="gray")
    axes[0, 0].set_title("True images", loc="left", fontsize=16)
    axes[n_rows, 0].set_title(generated_title, loc="left", fontsize=16)
    plt.subplots_adjust(wspace=0.01, hspace=0.5)
    plt.savefig(save_path)
    plt.close()


images_true = images_flat_true.reshape(-1, resolution[0], resolution[1])
plot_true_vs_generated_grid(
    images_true,
    images_generated_sde,
    f"SDE sampled images (Memory Efficient), rtol={rtol_sde}, atol={atol_sde}, brownian_tolerance={brownian_tolerance}",
    f"{plot_folder}/samples_{reverse_sde_suffix}.png",
)

# Same comparison, but with true and generated images clipped to the normalized pixel range.
clip_min, clip_max = pixel_min_normalized, pixel_max_normalized
images_true_clipped = np.clip(np.array(samples_target_unscaled.reshape(-1, resolution[0], resolution[1])), clip_min, clip_max)
images_generated_clipped = np.clip(np.array(images_generated_sde), clip_min, clip_max)
plot_true_vs_generated_grid(
    images_true_clipped,
    images_generated_clipped,
    f"SDE sampled images (range:[{clip_min:.4f},{clip_max:.4f}]), rtol={rtol_sde}, atol={atol_sde}, brownian_tolerance={brownian_tolerance}",
    f"{plot_folder}/samples_{reverse_sde_suffix}_clipped_{clip_min:.4f}_to_{clip_max:.4f}.png",
)

print("Script completed successfully!")
