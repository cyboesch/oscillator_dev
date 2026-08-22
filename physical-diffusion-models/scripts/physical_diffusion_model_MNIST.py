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
from physical_diffusion_fns.learning_fns import run_optimization
from physical_diffusion_fns.plotting_fns import (
    plot_forward_marginals,
    plot_parameter_evolution,
    plot_parameter_as_fn_of_time,
    visualize_connectivity_with_non_local_couplings,
)
from physical_diffusion_fns.network_fns import (
    setup_energy_fn,
    setup_overdamped_SDE,
    solve_ODE,
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

# --- Optimization (one fit per forward-time slice) ---
learning_rate_start = 2e-5
learning_rate_end = 0.002
learning_rate_schedule = jnp.linspace(learning_rate_start, learning_rate_end, n_time_steps)[::-1]
lr_decay_rate = 0.95
lr_decay_steps = 6000
n_epochs = 100000
batch_size = 512
window_size = 1000
tolerance_start = 1e-3
tolerance_end = 1e-3
tolerance_schedule = jnp.linspace(tolerance_start, tolerance_end, n_time_steps)[::-1]
patience_start = 300
patience_end = 100
patience_schedule = jnp.linspace(patience_start, patience_end, n_time_steps)[::-1]

# --- Network: 2D lattice of coupled 6th-order Duffing oscillators, one oscillator per pixel ---
n_neighbour_couplings = 14

# --- Reverse generation ---
n_trajectories = 40
use_probability_flow_ODE = True
run_reverse_sde_sanity_check = True
rtol_sde = 1e-4
atol_sde = 1e-6
brownian_tolerance = 1e-12
rtol_ode = 1e-4
atol_ode = 1e-6

# --- Checkpointing / memory management ---
start_from_scratch = False
sampling_only = True
checkpoint_date = "2026_07_08"  # paper checkpoint used for the ODE/SDE comparison
clear_cache_every_n_steps = 5
force_gc_every_n_steps = 10

# Fixed labels used only for output-directory naming.
training_method = "SM_at_kbT_minimal_memory"
energy_fn_type = "6th_order_duffing_coupling"
maximize = False

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
    f"lr_{learning_rate_start}_to_{learning_rate_end}_"
    f"epochs_{n_epochs}_"
    f"batch_{batch_size}_"
    f"window_{window_size}_"
    f"tol_{tolerance_start}_to_{tolerance_end}_"
    f"patience_{patience_start}_to_{patience_end}_"
)

here = os.path.dirname(os.path.abspath(__file__))
current_date = checkpoint_date if checkpoint_date is not None else datetime.now().strftime("%Y_%m_%d")
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
if not sampling_only:
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

if sampling_only and start_from_scratch:
    raise ValueError("sampling_only=True is incompatible with start_from_scratch=True")

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
    if sampling_only and (
        start_t_idx != len(forward_time_pts)
        or full_params_history.shape
        != (len(forward_time_pts), len(params_flattened_initial))
    ):
        raise RuntimeError(
            "Sampling-only mode requires a complete checkpoint with exactly "
            f"{len(forward_time_pts)} time slices; found index {start_t_idx} and "
            f"shape {full_params_history.shape}."
        )
elif sampling_only:
    raise FileNotFoundError(
        "Sampling-only mode cannot proceed because the paper checkpoint is missing: "
        f"{params_history_path} and/or {time_index_path}"
    )
else:
    print("No existing parameters found, starting from beginning")
    start_t_idx = 0
    params_history_all_t = []
    current_params = params_flattened_initial

plot_per_step = True  # save per-time-slice optimization diagnostics

if not sampling_only and start_t_idx < len(forward_time_pts):
    for t_idx in range(start_t_idx, len(forward_time_pts)):
        t_curr = forward_time_pts[t_idx]
        print(f"t_idx: {t_idx}, t_curr: {t_curr}")

        optimization_key, optimization_subkey = jr.split(optimization_key)

        # On the first slice, sanity-check that the forward process reaches the target Gaussian.
        if t_idx == 0:
            samples_0 = sample_forward_process(t_forward, n_samples, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=optimization_key)
            plot_forward_marginals(samples_0, t_forward, sigma_forward, Temp=Temp, beta=1.0, path=plot_folder, save_fig=True, fontsize=16, plot_show=True)

        # Draw fresh noised mini-batches from the forward marginal p_{t_curr}.
        sampler = lambda key: sample_forward_process(
            t_curr, n_samples=batch_size, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=key, beta=1,
        )

        params_history, loss_history, current_params, best_loss, best_epoch = run_optimization(
            loss_fn_per_batch=loss_fn_per_batch,
            params_initial=current_params,
            sampler=sampler,
            gradient_fn_per_batch=gradient_fn_per_batch,
            key=optimization_subkey,
            learning_rate=learning_rate_schedule[t_idx],
            n_epochs=n_epochs,
            maximize=maximize,
            window_size=window_size,
            tolerance=tolerance_schedule[t_idx],
            patience=patience_schedule[t_idx],
            constraint_indices=constraint_indices,
            lr_decay_rate=lr_decay_rate,
            lr_decay_steps=lr_decay_steps,
        )

        # Downsample the (epoch, loss/params) history to <=100 points for the diagnostic plot.
        if plot_per_step:
            length = len(loss_history)
            idxs = jnp.unique(jnp.concatenate([
                jnp.array([0], dtype=int),
                jnp.linspace(0, length - 1, min(100, length)).astype(int),
                jnp.array([length - 1], dtype=int),
            ]))
            params_ds = [params_history[i] for i in idxs.tolist()]
            loss_ds = jnp.array(loss_history)[idxs]

        params_history_all_t.append(current_params)

        # Free the large per-slice history and reclaim memory.
        del params_history, loss_history
        optimize_memory()
        if t_idx % clear_cache_every_n_steps == 0:
            jax.clear_caches()
        if t_idx % force_gc_every_n_steps == 0:
            gc.collect()

        # Checkpoint after each slice so the run is resumable.
        jnp.save(params_history_path, jnp.array(params_history_all_t))
        with open(time_index_path, "w") as f:
            f.write(str(t_idx + 1))  # next time index to resume from

        if plot_per_step:
            plot_parameter_evolution(
                params_history=params_ds,
                loss_history=loss_ds,
                best_loss=best_loss,
                best_idx=best_epoch,
                best_params=current_params,
                time=t_curr,
                time_index=t_idx,
                unflatten=unflatten,
                N_osc=N_osc,
                title=f"{training_method} (t = {t_curr:.3f})",
                maximize=maximize,
                labels_on=True,
                save_fig=True,
                path=output_dir,
                param_names=params_names,
            )
        print(f"--- done t_idx {t_idx} ({len(params_history_all_t)} slices total)")

    print("Optimization complete; saved parameters to", output_dir)

params_history_all_t = jnp.array(params_history_all_t)

#####################################
# Interpolate the learned parameters as a function of forward time
#####################################
print(f"params_history_all_t shape: {params_history_all_t.shape}, forward_time_pts shape: {forward_time_pts.shape}")

params_interpolator_non_smoothed = interpolate_parameters(params_history_all_t, forward_time_pts)

if not sampling_only:
    smoothed_params = smooth_parameters(
        params_history_all_t,
        window_lengths=[10] * params_history_all_t.shape[1],
        poly_orders=[3] * params_history_all_t.shape[1],
    )
    params_interpolator_smoothed = interpolate_parameters(smoothed_params, forward_time_pts)
    plot_parameter_as_fn_of_time(
        params_names, forward_time_pts, forward_time_pts, params_history_all_t,
        params_interpolator_smoothed, unflatten, N_osc, save_fig=True, path=plot_folder, log_scale=False,
    )

#####################################
# Reverse generation: probability-flow ODE and reference SDE
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
    # The original sampler (used in the paper) performed two extra jr.split(key) calls before drawing; advance
    # past them so we land on the same subkeys and reproduce the historical samples exactly.
    for _ in range(2):
        key, _ = jr.split(key)
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


def run_reverse_process_ODE(params_interpolator, suffix, key):
    """Integrate the deterministic probability-flow ODE.

    The learned-score contribution is half of the reverse-SDE contribution:
    theta_probability_flow(t) = theta(tau(t)) - theta_linear. The harmonic
    reference is not halved, so the drift is x/sigma^2 - grad_x E_theta.
    The initial Gaussian draw deliberately consumes the same historical JAX
    subkeys as ``run_reverse_process_SDE``; there is no Brownian key or noise.
    """
    tau = lambda t: t_forward - t
    forward_params = jnp.zeros_like(params_flattened_initial).at[0:N_osc].set(1.0)
    params_probability_flow = lambda t: params_interpolator(tau(t)) - forward_params / sigma_forward**2

    drift_fn, _ = setup_overdamped_SDE(
        energy_fn, params_probability_flow, N_osc, time_dependent_parms=True, Temp=Temp,
    )

    t0, t1 = 0.0, t_forward
    n_time_steps_ode = 100          # number of saved output points (does not affect solver accuracy)
    ts = jnp.linspace(t0, t1, n_time_steps_ode)
    dt0 = 1e-8

    # Reuse the exact historical initial-condition subkey used by the SDE.
    for _ in range(2):
        key, _ = jr.split(key)
    key, subkey_init = jr.split(key)
    initial_states = jnp.sqrt(Temp) * sigma_forward * jr.normal(
        subkey_init, shape=(n_trajectories, N_osc)
    )

    print(f"Running probability-flow ODE for {n_trajectories} trajectories...")
    solve_one = lambda init_state: solve_ODE(
        drift_fn, init_state, t0, t1, ts.shape[0], dt0, rtol=rtol_ode, atol=atol_ode,
    )
    solutions = vmap(solve_one)(initial_states)
    final_samples_scaled = solutions.ys[:, -1, :]  # (n_trajectories, N_osc), final states only

    final_states_path = f"{output_dir}/final_states_{suffix}.npy"
    jnp.save(final_states_path, final_samples_scaled)
    print(f"Final probability-flow ODE states saved to {final_states_path}")

    del solutions
    optimize_memory()
    final_samples = final_samples_scaled / additional_rescaling
    return final_samples.reshape(-1, resolution[0], resolution[1])


reverse_sde_suffix = (
    f"non_smoothed_sde_memory_efficient_atol_{atol_sde}_rtol_{rtol_sde}_brownian_tolerance_{brownian_tolerance}"
    f"_init_method_asymptotic_gaussian"
    f"_use_same_initial_condition_for_all_trajectories_False"
    + (f"_reverse_rng_seed_{key_seed_reverse}" if key_seed_reverse is not None else "")
)
images_generated_non_smoothed_sde = None
if run_reverse_sde_sanity_check or not use_probability_flow_ODE:
    sde_output_suffix = reverse_sde_suffix + ("_reproduction_check" if use_probability_flow_ODE else "")
    print(f"Running reverse SDE with final time: {t_forward}")
    images_generated_non_smoothed_sde = run_reverse_process_SDE(
        params_interpolator_non_smoothed, sde_output_suffix, reverse_sde_key,
    )
    print_memory_usage("after reverse SDE")

reverse_ode_suffix = (
    f"non_smoothed_probability_flow_ode_memory_efficient_atol_{atol_ode}_rtol_{rtol_ode}"
    f"_init_method_asymptotic_gaussian"
    f"_use_same_initial_condition_for_all_trajectories_False"
    + (f"_reverse_rng_seed_{key_seed_reverse}" if key_seed_reverse is not None else "")
)
images_generated_non_smoothed_ode = None
if use_probability_flow_ODE:
    print(f"Running probability-flow ODE with final time: {t_forward}")
    images_generated_non_smoothed_ode = run_reverse_process_ODE(
        params_interpolator_non_smoothed, reverse_ode_suffix, reverse_sde_key,
    )
    print_memory_usage("after probability-flow ODE")

#####################################
# Plotting: true vs. generated digit grids
#####################################
def plot_true_vs_generated_grid(true_imgs, generated_imgs, generated_title, save_path):
    """Top rows: true images; bottom rows: generated images (10 columns per row)."""
    n_show = n_trajectories
    n_rows = n_show // 10
    fig, axes = plt.subplots(2 * n_rows, 10, figsize=(15, 3 * n_rows))
    for idx in range(n_show):
        for row_block, imgs in ((0, true_imgs), (n_rows, generated_imgs)):
            ax = axes[idx // 10 + row_block, idx % 10]
            img = imgs[idx]
            if img.shape[-1] == 1:  # drop singleton channel if present
                img = img.squeeze(-1)
            ax.imshow(np.array(img), cmap="gray")
            ax.axis("off")
    axes[0, 0].set_title("True images", loc="left", fontsize=16)
    axes[n_rows, 0].set_title(generated_title, loc="left", fontsize=16)
    plt.subplots_adjust(wspace=0.01, hspace=0.5)
    plt.savefig(save_path)
    plt.close()


images_true = images_flat_true.reshape(-1, resolution[0], resolution[1])
clip_min, clip_max = pixel_min_normalized, pixel_max_normalized
images_true_clipped = np.clip(np.array(samples_target_unscaled.reshape(-1, resolution[0], resolution[1])), clip_min, clip_max)


def save_sample_grids(generated_images, suffix, title, clipped_title):
    """Save unclipped and normalized-range-clipped comparisons to the true images."""
    plot_true_vs_generated_grid(
        images_true,
        generated_images,
        title,
        f"{plot_folder}/samples_{suffix}.png",
    )
    generated_images_clipped = np.clip(np.array(generated_images), clip_min, clip_max)
    plot_true_vs_generated_grid(
        images_true_clipped,
        generated_images_clipped,
        clipped_title,
        f"{plot_folder}/samples_{suffix}_clipped_{clip_min:.4f}_to_{clip_max:.4f}.png",
    )


if images_generated_non_smoothed_sde is not None:
    save_sample_grids(
        images_generated_non_smoothed_sde,
        sde_output_suffix,
        f"SDE sampled images (Memory Efficient), rtol={rtol_sde}, atol={atol_sde}, brownian_tolerance={brownian_tolerance}",
        f"SDE sampled images (range:[{clip_min:.4f},{clip_max:.4f}]), rtol={rtol_sde}, atol={atol_sde}, brownian_tolerance={brownian_tolerance}",
    )

if images_generated_non_smoothed_ode is not None:
    save_sample_grids(
        images_generated_non_smoothed_ode,
        reverse_ode_suffix,
        f"Probability-flow ODE sampled images, rtol={rtol_ode}, atol={atol_ode}",
        f"Probability-flow ODE sampled images (range:[{clip_min:.4f},{clip_max:.4f}]), rtol={rtol_ode}, atol={atol_ode}",
    )

print("Script completed successfully!")
