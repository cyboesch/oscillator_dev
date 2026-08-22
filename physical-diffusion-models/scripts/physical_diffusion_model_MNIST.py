print("Script started.")
import os
import sys
import gc
import json
import hashlib
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
    interpolate_parameters_pchip,
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


def sha256_file(path):
    """Return a streaming SHA-256 digest without loading the file twice into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
run_interpolation_ablation = True
ablation_run_reverse_sde = True
ablation_run_probability_flow_ode = True
rtol_sde = 1e-4
atol_sde = 1e-6
brownian_tolerance = 1e-12
rtol_ode = 1e-7
atol_ode = 1e-9
ablation_savgol_window = 9
ablation_savgol_polyorder = 3
ablation_k6_floor = 0.01
ablation_step_at_parameter_knots = True
ablation_name = "interpolation_ablation_v1"
expected_paper_checkpoint_sha256 = "37cfe46ef805a39129c265a0c17aaf6cf2b4e3b2ec7d8f55725e1cd66c227249"

# --- Checkpointing / memory management ---
start_from_scratch = False
sampling_only = True
checkpoint_date = "2026_07_08"  # paper checkpoint used for the ODE/SDE comparison
clear_cache_every_n_steps = 5
force_gc_every_n_steps = 10

if run_interpolation_ablation and (not sampling_only or start_from_scratch):
    raise ValueError(
        "The interpolation ablation must be sampling-only and load the verified paper checkpoint"
    )

smoothed_parameter_label = (
    f"savgol_w{ablation_savgol_window}_p{ablation_savgol_polyorder}"
    f"_k6_log_floor_{ablation_k6_floor:g}"
)

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
sampling_output_dir = os.path.join(output_dir, ablation_name) if run_interpolation_ablation else output_dir
sampling_plot_folder = os.path.join(plot_folder, ablation_name) if run_interpolation_ablation else plot_folder
os.makedirs(sampling_output_dir, exist_ok=True)
os.makedirs(sampling_plot_folder, exist_ok=True)
print(f"Output directory: {output_dir}")
print(f"Plot directory: {plot_folder}")
if run_interpolation_ablation:
    print(f"Ablation state directory: {sampling_output_dir}")
    print(f"Ablation plot directory: {sampling_plot_folder}")

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
checkpoint_sha256 = None

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
    # Sampling does not need a mutable Python list. Keeping the checkpoint as one
    # array avoids materializing 54.5 million Python float objects.
    params_history_all_t = full_params_history if sampling_only else full_params_history.tolist()
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
    if run_interpolation_ablation:
        checkpoint_sha256 = sha256_file(params_history_path)
        if checkpoint_sha256 != expected_paper_checkpoint_sha256:
            raise RuntimeError(
                "The interpolation ablation requires the exact paper checkpoint; "
                f"expected SHA-256 {expected_paper_checkpoint_sha256}, found {checkpoint_sha256}."
            )
        print(f"Verified paper checkpoint SHA-256: {checkpoint_sha256}")
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

params_interpolator_non_smoothed = None
if not run_interpolation_ablation:
    params_interpolator_non_smoothed = interpolate_parameters(params_history_all_t, forward_time_pts)

smoothed_params_ablation = None
if run_interpolation_ablation:
    print(
        "Building constrained Savitzky-Golay parameter history "
        f"(window={ablation_savgol_window}, polyorder={ablation_savgol_polyorder})..."
    )
    smoothed_params_ablation = smooth_parameters(
        params_history_all_t,
        window_lengths=ablation_savgol_window,
        poly_orders=ablation_savgol_polyorder,
    )

    # Direct polynomial smoothing can make constrained sextic coefficients
    # negative. Smooth this positive group in log-space, transform back, and
    # restore the same floor used by training.
    raw_k6 = params_history_all_t[:, constraint_indices]
    smoothed_log_k6 = smooth_parameters(
        jnp.log(raw_k6),
        window_lengths=ablation_savgol_window,
        poly_orders=ablation_savgol_polyorder,
    )
    smoothed_k6_before_floor = jnp.exp(smoothed_log_k6)
    n_k6_projected = int(jnp.sum(smoothed_k6_before_floor < ablation_k6_floor))
    smoothed_k6 = jnp.maximum(smoothed_k6_before_floor, ablation_k6_floor)
    smoothed_params_ablation = smoothed_params_ablation.at[:, constraint_indices].set(smoothed_k6)
    if not bool(jnp.all(jnp.isfinite(smoothed_params_ablation))):
        raise RuntimeError("Non-finite values were produced while smoothing the parameters")
    if float(jnp.min(smoothed_params_ablation[:, constraint_indices])) < ablation_k6_floor:
        raise RuntimeError("The smoothed parameter history violates the k6 confinement floor")
    print(
        f"Constrained smoothed k6 minimum: {float(jnp.min(smoothed_k6)):.6g}; "
        f"projected {n_k6_projected} knot values to {ablation_k6_floor}."
    )

    # Validate shape preservation on a dense time grid without materializing all
    # 545,566 interpolated parameters at once.
    dense_validation_times = jnp.linspace(0.0, t_forward, 10 * (n_time_steps - 1) + 1)
    ablation_k6_minima = {}
    for parameter_label, parameter_values in (
        ("raw", params_history_all_t),
        (smoothed_parameter_label, smoothed_params_ablation),
    ):
        k6_values = parameter_values[:, constraint_indices]
        for interpolation_label, interpolation_factory in (
            ("linear", interpolate_parameters),
            ("pchip", interpolate_parameters_pchip),
        ):
            k6_interpolator = interpolation_factory(k6_values, forward_time_pts)
            minimum = float(jnp.min(k6_interpolator(dense_validation_times)))
            ablation_k6_minima[f"{parameter_label}_{interpolation_label}"] = minimum
            if minimum < ablation_k6_floor - 1e-12:
                raise RuntimeError(
                    f"{parameter_label}/{interpolation_label} interpolation violates "
                    f"the k6 floor: {minimum}"
                )
            print(f"Dense-grid k6 minimum ({parameter_label}, {interpolation_label}): {minimum:.6g}")

if not sampling_only:
    smoothed_params_for_plot = smooth_parameters(
        params_history_all_t,
        window_lengths=[10] * params_history_all_t.shape[1],
        poly_orders=[3] * params_history_all_t.shape[1],
    )
    params_interpolator_smoothed = interpolate_parameters(smoothed_params_for_plot, forward_time_pts)
    plot_parameter_as_fn_of_time(
        params_names, forward_time_pts, forward_time_pts, params_history_all_t,
        params_interpolator_smoothed, unflatten, N_osc, save_fig=True, path=plot_folder, log_scale=False,
    )

#####################################
# Reverse generation: probability-flow ODE, reference SDE, and interpolation ablation
#####################################
def save_solver_stats(solutions, suffix):
    """Save per-trajectory Diffrax step statistics for reproducibility."""
    stats = {name: np.asarray(value) for name, value in solutions.stats.items()}
    stats_path = os.path.join(sampling_output_dir, f"solver_stats_{suffix}.npz")
    np.savez(stats_path, **stats)
    if "num_steps" in stats:
        num_steps = stats["num_steps"]
        print(
            f"Solver steps for {suffix}: min={np.min(num_steps)}, "
            f"median={np.median(num_steps):.1f}, max={np.max(num_steps)}"
        )
    return stats_path


def run_reverse_process_SDE(
    params_interpolator,
    suffix,
    initial_states,
    keys_brownian,
    parameter_knot_times=None,
):
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

    print(f"Running reverse SDE for {n_trajectories} trajectories...")
    solve_one = lambda init_state, key_b: solve_SDE(
        drift_fn, diffusion_fn, init_state, key_b, t0, t1, ts.shape[0], dt0,
        brownian_tolerance=brownian_tolerance, rtol=rtol_sde, atol=atol_sde,
        step_ts=parameter_knot_times,
    )
    solutions = vmap(solve_one, in_axes=(0, 0))(initial_states, keys_brownian)
    final_samples_scaled = solutions.ys[:, -1, :]  # (n_trajectories, N_osc), final states only

    final_states_path = os.path.join(sampling_output_dir, f"final_states_{suffix}.npy")
    jnp.save(final_states_path, final_samples_scaled)
    print(f"Final states saved to {final_states_path}")
    stats_path = save_solver_stats(solutions, suffix)

    final_samples = np.asarray(final_samples_scaled / additional_rescaling)
    del solutions
    optimize_memory()
    return final_samples.reshape(-1, resolution[0], resolution[1]), final_states_path, stats_path


def run_reverse_process_ODE(
    params_interpolator,
    suffix,
    initial_states,
    parameter_knot_times=None,
):
    """Integrate the deterministic probability-flow ODE.

    The learned-score contribution is half of the reverse-SDE contribution:
    theta_probability_flow(t) = theta(tau(t)) - theta_linear. The harmonic
    reference is not halved, so the drift is x/sigma^2 - grad_x E_theta.
    The caller supplies the Gaussian initial states shared by every ablation
    variant; there is no Brownian key or noise.
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

    print(f"Running probability-flow ODE for {n_trajectories} trajectories...")
    solve_one = lambda init_state: solve_ODE(
        drift_fn, init_state, t0, t1, ts.shape[0], dt0, rtol=rtol_ode, atol=atol_ode,
        step_ts=parameter_knot_times,
    )
    solutions = vmap(solve_one)(initial_states)
    final_samples_scaled = solutions.ys[:, -1, :]  # (n_trajectories, N_osc), final states only

    final_states_path = os.path.join(sampling_output_dir, f"final_states_{suffix}.npy")
    jnp.save(final_states_path, final_samples_scaled)
    print(f"Final probability-flow ODE states saved to {final_states_path}")
    stats_path = save_solver_stats(solutions, suffix)

    final_samples = np.asarray(final_samples_scaled / additional_rescaling)
    del solutions
    optimize_memory()
    return final_samples.reshape(-1, resolution[0], resolution[1]), final_states_path, stats_path


# Draw the historical Gaussian initial states and Brownian keys once. Passing the
# resulting arrays into every variant makes the factorial comparison explicitly
# paired, rather than merely relying on repeated pure-key computations.
shared_key = reverse_sde_key
for _ in range(2):
    shared_key, _ = jr.split(shared_key)
shared_key, shared_init_subkey = jr.split(shared_key)
shared_initial_states = jnp.sqrt(Temp) * sigma_forward * jr.normal(
    shared_init_subkey, shape=(n_trajectories, N_osc)
)
shared_key, shared_brownian_subkey = jr.split(shared_key)
shared_brownian_keys = jr.split(shared_brownian_subkey, n_trajectories)

sampling_results = []
ablation_manifest = None
if run_interpolation_ablation:
    initial_states_path = os.path.join(sampling_output_dir, "shared_initial_states.npy")
    brownian_keys_path = os.path.join(sampling_output_dir, "shared_brownian_keys.npy")
    np.save(initial_states_path, np.asarray(shared_initial_states))
    np.save(brownian_keys_path, np.asarray(shared_brownian_keys))

    reverse_parameter_knot_times = None
    if ablation_step_at_parameter_knots:
        reverse_parameter_knot_times = jnp.sort(t_forward - forward_time_pts)[1:-1]

    ablation_manifest = {
        "ablation_name": ablation_name,
        "checkpoint_path": os.path.abspath(params_history_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_shape": list(params_history_all_t.shape),
        "source_files": {
            "physical_diffusion_model_MNIST.py": {
                "path": os.path.abspath(__file__),
                "sha256": sha256_file(os.path.abspath(__file__)),
            },
            "helper_fns.py": {
                "path": os.path.abspath(os.path.join(here, "..", "physical_diffusion_fns", "helper_fns.py")),
                "sha256": sha256_file(os.path.join(here, "..", "physical_diffusion_fns", "helper_fns.py")),
            },
            "network_fns.py": {
                "path": os.path.abspath(os.path.join(here, "..", "physical_diffusion_fns", "network_fns.py")),
                "sha256": sha256_file(os.path.join(here, "..", "physical_diffusion_fns", "network_fns.py")),
            },
        },
        "key_seed": key_seed,
        "key_seed_reverse": key_seed_reverse,
        "n_trajectories": n_trajectories,
        "initial_states_file": os.path.basename(initial_states_path),
        "initial_states_sha256": sha256_file(initial_states_path),
        "brownian_keys_file": os.path.basename(brownian_keys_path),
        "brownian_keys_sha256": sha256_file(brownian_keys_path),
        "savgol_window": ablation_savgol_window,
        "savgol_polyorder": ablation_savgol_polyorder,
        "k6_smoothing": "log_savgol_then_floor",
        "k6_floor": ablation_k6_floor,
        "k6_values_projected": n_k6_projected,
        "dense_grid_k6_minima": ablation_k6_minima,
        "step_at_parameter_knots": ablation_step_at_parameter_knots,
        "ode_solver": "Tsit5",
        "ode_rtol": rtol_ode,
        "ode_atol": atol_ode,
        "sde_solver": "SRA1",
        "sde_rtol": rtol_sde,
        "sde_atol": atol_sde,
        "brownian_tolerance": brownian_tolerance,
        "variants": [],
    }

    parameter_variants = (
        ("raw", params_history_all_t),
        (smoothed_parameter_label, smoothed_params_ablation),
    )
    interpolation_variants = (
        ("linear", interpolate_parameters),
        ("pchip", interpolate_parameters_pchip),
    )

    for parameter_label, parameter_values in parameter_variants:
        for interpolation_label, interpolation_factory in interpolation_variants:
            print(f"Preparing ablation variant: params={parameter_label}, interpolation={interpolation_label}")
            params_interpolator = interpolation_factory(parameter_values, forward_time_pts)
            variant_tag = (
                f"ablation_params_{parameter_label}_interp_{interpolation_label}"
                f"_step_knots_{ablation_step_at_parameter_knots}_n{n_trajectories}_matched_rng"
            )

            if ablation_run_reverse_sde:
                suffix = (
                    f"{variant_tag}_sde_atol_{atol_sde}_rtol_{rtol_sde}"
                    f"_btol_{brownian_tolerance}"
                )
                images, states_path, stats_path = run_reverse_process_SDE(
                    params_interpolator,
                    suffix,
                    shared_initial_states,
                    shared_brownian_keys,
                    parameter_knot_times=reverse_parameter_knot_times,
                )
                sampling_results.append({
                    "sampler": "sde",
                    "parameter_variant": parameter_label,
                    "interpolation": interpolation_label,
                    "suffix": suffix,
                    "images": images,
                    "states_path": states_path,
                    "stats_path": stats_path,
                    "title": (
                        f"SDE: {parameter_label}, {interpolation_label}; "
                        f"rtol={rtol_sde}, atol={atol_sde}"
                    ),
                })
                print_memory_usage(f"after SDE ({parameter_label}, {interpolation_label})")

            if ablation_run_probability_flow_ode:
                suffix = (
                    f"{variant_tag}_probability_flow_ode_atol_{atol_ode}_rtol_{rtol_ode}"
                )
                images, states_path, stats_path = run_reverse_process_ODE(
                    params_interpolator,
                    suffix,
                    shared_initial_states,
                    parameter_knot_times=reverse_parameter_knot_times,
                )
                sampling_results.append({
                    "sampler": "probability_flow_ode",
                    "parameter_variant": parameter_label,
                    "interpolation": interpolation_label,
                    "suffix": suffix,
                    "images": images,
                    "states_path": states_path,
                    "stats_path": stats_path,
                    "title": (
                        f"Probability-flow ODE: {parameter_label}, {interpolation_label}; "
                        f"rtol={rtol_ode}, atol={atol_ode}"
                    ),
                })
                print_memory_usage(f"after ODE ({parameter_label}, {interpolation_label})")

            del params_interpolator
            optimize_memory()
else:
    reverse_sde_suffix = (
        f"non_smoothed_sde_memory_efficient_atol_{atol_sde}_rtol_{rtol_sde}_brownian_tolerance_{brownian_tolerance}"
        f"_init_method_asymptotic_gaussian"
        f"_use_same_initial_condition_for_all_trajectories_False"
        + (f"_reverse_rng_seed_{key_seed_reverse}" if key_seed_reverse is not None else "")
    )
    if run_reverse_sde_sanity_check or not use_probability_flow_ODE:
        sde_output_suffix = reverse_sde_suffix + ("_reproduction_check" if use_probability_flow_ODE else "")
        images, states_path, stats_path = run_reverse_process_SDE(
            params_interpolator_non_smoothed,
            sde_output_suffix,
            shared_initial_states,
            shared_brownian_keys,
        )
        sampling_results.append({
            "sampler": "sde",
            "suffix": sde_output_suffix,
            "images": images,
            "states_path": states_path,
            "stats_path": stats_path,
            "title": (
                f"SDE sampled images; rtol={rtol_sde}, atol={atol_sde}, "
                f"brownian_tolerance={brownian_tolerance}"
            ),
        })

    if use_probability_flow_ODE:
        reverse_ode_suffix = (
            f"non_smoothed_probability_flow_ode_memory_efficient_atol_{atol_ode}_rtol_{rtol_ode}"
            f"_init_method_asymptotic_gaussian"
            f"_use_same_initial_condition_for_all_trajectories_False"
            + (f"_reverse_rng_seed_{key_seed_reverse}" if key_seed_reverse is not None else "")
        )
        images, states_path, stats_path = run_reverse_process_ODE(
            params_interpolator_non_smoothed,
            reverse_ode_suffix,
            shared_initial_states,
        )
        sampling_results.append({
            "sampler": "probability_flow_ode",
            "suffix": reverse_ode_suffix,
            "images": images,
            "states_path": states_path,
            "stats_path": stats_path,
            "title": f"Probability-flow ODE; rtol={rtol_ode}, atol={atol_ode}",
        })

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
    unclipped_path = os.path.join(sampling_plot_folder, f"samples_{suffix}.png")
    clipped_path = os.path.join(
        sampling_plot_folder,
        f"samples_{suffix}_clipped_{clip_min:.4f}_to_{clip_max:.4f}.png",
    )
    plot_true_vs_generated_grid(
        images_true,
        generated_images,
        title,
        unclipped_path,
    )
    generated_images_clipped = np.clip(np.array(generated_images), clip_min, clip_max)
    plot_true_vs_generated_grid(
        images_true_clipped,
        generated_images_clipped,
        clipped_title,
        clipped_path,
    )
    return unclipped_path, clipped_path


for result in sampling_results:
    generated_images = np.asarray(result["images"])
    expected_shape = (n_trajectories, resolution[0], resolution[1])
    if generated_images.shape != expected_shape or not np.all(np.isfinite(generated_images)):
        raise RuntimeError(
            f"Invalid {result['sampler']} output for {result['suffix']}: "
            f"shape={generated_images.shape}, finite={np.all(np.isfinite(generated_images))}"
        )
    unclipped_path, clipped_path = save_sample_grids(
        generated_images,
        result["suffix"],
        result["title"],
        f"{result['title']} (range:[{clip_min:.4f},{clip_max:.4f}])",
    )
    result["unclipped_plot_path"] = unclipped_path
    result["clipped_plot_path"] = clipped_path

    if ablation_manifest is not None:
        ablation_manifest["variants"].append({
            "sampler": result["sampler"],
            "parameter_variant": result["parameter_variant"],
            "interpolation": result["interpolation"],
            "suffix": result["suffix"],
            "final_states_file": os.path.basename(result["states_path"]),
            "final_states_sha256": sha256_file(result["states_path"]),
            "solver_stats_file": os.path.basename(result["stats_path"]),
            "unclipped_plot_file": os.path.relpath(unclipped_path, sampling_output_dir),
            "clipped_plot_file": os.path.relpath(clipped_path, sampling_output_dir),
        })

if ablation_manifest is not None:
    manifest_path = os.path.join(sampling_output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(ablation_manifest, f, indent=2, sort_keys=True)
    print(f"Ablation manifest saved to {manifest_path}")

print("Script completed successfully!")
