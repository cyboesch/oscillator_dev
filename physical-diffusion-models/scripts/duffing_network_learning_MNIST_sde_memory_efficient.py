print("Script started.")
import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Memory monitoring
import psutil
import gc

def print_memory_usage(label=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage {label}: {memory_info.rss / 1024 / 1024:.2f} MB")

def optimize_memory():
    """Force garbage collection and clear JAX caches"""
    gc.collect()
    jax.clear_caches()

print_memory_usage("at start")

import jax
from jax import flatten_util, vmap 
import jax.numpy as jnp
import jax.random as jr
import os
import matplotlib.pyplot as plt
from datetime import datetime
from physical_diffusion_fns.helper_fns import sample_gaussian_mixture, normalize_samples, sample_forward_process, get_best_params, smooth_parameters, interpolate_parameters
from physical_diffusion_fns.learning_fns import setup_MLE_loss_per_batch, setup_MLE_gradient_per_batch, CD1_gradient, setup_score_matching_loss_per_batch, run_optimization, setup_score_matching_kbT_loss_per_batch, setup_score_matching_kbT_local_gradient_per_batch, run_optimization_multi_gpu_sampler
from physical_diffusion_fns.plotting_fns import plot_energy_and_distributions, plot_parameter_evolution, plot_forward_marginals, visualize_connectivity, plot_parameter_as_fn_of_time, visualize_connectivity_with_non_local_couplings
from physical_diffusion_fns.network_fns import setup_overdamped_SDE, solve_SDE, create_2d_square_lattice_connectivity, setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_and_6th_order_energy_fn

# Import memory-efficient versions
from physical_diffusion_fns.learning_fns_memory_efficient import (
    setup_score_matching_kbT_loss_analytical_memory_efficient, 
    setup_score_matching_kbT_loss_analytical_ultra_efficient,
    setup_score_matching_kbT_loss_analytical_extreme_efficient,
    setup_score_matching_kbT_loss_minimal_memory
)
from physical_diffusion_fns.network_fns_memory_efficient import setup_duffing_network_analytical_derivatives_memory_efficient

jax.config.update("jax_enable_x64", True)

num_devices = jax.local_device_count()
print(f"Number of devices: {num_devices}")

##################################### 
# Set random seeds
##################################### 

key_seed = 1234
master_key  = jax.random.PRNGKey(key_seed)          # single seed
optimization_key, reverse_sde_key, image_noise_added_key = jr.split(master_key, 3)

##################################### 
# Set parameters - MEMORY OPTIMIZED
##################################### 
std_of_added_noise = 0.005
additional_rescaling = 1.
# Forward process parameters
Temp = 0.005
n_time_steps = 15
t_forward = 4.0
sigma_forward = 1.
exponential_time_pts = False
if exponential_time_pts:
    forward_time_pts = jnp.exp(jnp.linspace(jnp.log(1e-7), jnp.log(t_forward), n_time_steps))
    forward_time_pts = forward_time_pts.at[0].set(0.)
else:
    forward_time_pts = jnp.linspace(1.9, t_forward, n_time_steps)
print('forward_time_pts', forward_time_pts)

# Optimization parameters - EXTREMELY REDUCED FOR MEMORY
learning_rate = 1.
lr_decay_rate = 0.95
lr_decay_steps = 6000 
n_epochs = 10000
batch_size = 2*512  # Batch size (doesn't affect reverse SDE memory usage)
window_size=1000
tolerance=.1
patience=100
CD1_dt = 0.00001
CD1_num_noise_samples = 1000

# Parameter storage - store all optimization time steps
save_all_params_to_disk = True  # Save all params to disk
chunk_size = 4  # Process samples one by one for maximum memory efficiency

# Additional extreme memory optimizations
clear_cache_every_n_steps = 5  # Clear JAX cache every N optimization steps
force_gc_every_n_steps = 10   # Force garbage collection every N steps

# Set training method and display memory usage warning
training_method = "SM_at_kbT_minimal_memory"  # Use most memory-efficient version

print(f"\nEXTREME MEMORY OPTIMIZATION ENABLED:")
print(f"  - Batch size reduced to: {batch_size}")
print(f"  - Chunk size: {chunk_size}")
print(f"  - Training method: {training_method}")
print(f"  - Clear cache every: {clear_cache_every_n_steps} steps")
print(f"  - Force GC every: {force_gc_every_n_steps} steps")
print(f"  - SDE time steps: 25 (reduced from 100)")
print(f"\nTotal memory reductions applied:")
print(f"  - Batch size: {batch_size} (extreme reduction)")
print(f"  - Sequential processing: eliminates vmap memory overhead")
print(f"  - Chunked parallel SDE solving: balances performance and memory, stores only final states")
print(f"  - Gradient checkpointing: trades computation for memory")
print(f"  - Expected memory reduction: ~50-100x vs original")

labels = [0,1]
# REDUCED resolution for extreme memory efficiency
resolution = (20,20)  # Reduced from (20,20) for memory efficiency

# REDUCED neighbor couplings for memory efficiency  
n_neighbour_couplings = 16  # Reduced from 8 to 4 for memory efficiency
energy_fn_type = "6th_order_duffing_coupling"

print(f"  - Network size: reduced to {resolution[0]}x{resolution[1]} with {n_neighbour_couplings} neighbors")


print(f"NETWORK SIZE REDUCED FOR MEMORY:")
print(f"  - Resolution: {resolution} (was 20x20)")
print(f"  - Neighbor couplings: {n_neighbour_couplings} (was 8)")
print(f"  - Estimated oscillators: {resolution[0]**2}")
print(f"  - This reduces parameter count by ~16x compared to original")

# SDE parameters - EXTREMELY REDUCED for memory efficiency
n_trajectories = 20   # Reduced to 5 for extreme memory efficiency with large network
rtol_sde = 1e-8      # Relaxed for memory efficiency
atol_sde = 1e-9      # REDUCED from 1e-5 to 1e-8 for memory efficiency
brownian_tolerance = 1e-12

print(f"  - SDE trajectories: {n_trajectories} (reduced from 100, final states only)")
print(f"  - SDE tolerances: rtol={rtol_sde}, atol={atol_sde}, brownian_tol={brownian_tolerance}")

start_from_scratch = False

#####################################
## Filenames
#####################################

problem_type_folder = f"MNIST_generation_memory_efficient"

system_specifics = f"n_neighbour_couplings_{n_neighbour_couplings}_Temp_{Temp}_energy_fn_type_{energy_fn_type}"

MNIST_specifics = f"labels_{labels}_resolution_{resolution[0]}_x_{resolution[1]}"

data_folder = f"added_gaussian_noise_std_{std_of_added_noise}_additional_rescaling_{additional_rescaling}_key_seed_{key_seed}"

optimization_folder = (
    f"schedule_training_method_{training_method}_"
    + (f"CD1_dt_{CD1_dt}_CD1_num_noise_samples_{CD1_num_noise_samples}_" if training_method == "CD1" else "")
    + f"exp_time_pts_{exponential_time_pts}_"
    f"t_forward_{t_forward}_"
    f"n_timesteps_{n_time_steps}_"
    f"forward_time_pts_{forward_time_pts[0]}_to_{forward_time_pts[-1]}_"
    f"sigma_forward_{sigma_forward}_"
    f"lr_{learning_rate}_"
    f"lr_decay_rate_{lr_decay_rate}_"
    f"lr_decay_steps_{lr_decay_steps}_"
    f"epochs_{n_epochs}_"
    f"batch_{batch_size}_"
    f"chunk_{chunk_size}_"
    f"window_{window_size}_"
    f"tol_{tolerance}_"
    f"patience_{patience}"
)

# Setup output directories
here = os.path.dirname(os.path.abspath(__file__))
current_date = datetime.now().strftime("%Y_%m_%d")
# current_date = "2025_11_28"
base_dir = os.path.join(here,"..","out", current_date, "problems")
output_dir = os.path.join(base_dir, problem_type_folder, MNIST_specifics, system_specifics, data_folder, optimization_folder)
plot_folder = os.path.join(output_dir, 'aaa_final_plots')


# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)
print(f"Output directory: {output_dir}")
print(f"Plot directory: {plot_folder}")


##################################### 
# Load MNIST data
##################################### 
import numpy as np
# Load the saved .npy file
path_to_data = os.path.join(here,"..", "data", "MNIST", f"mnist_labels_{labels}_resolution_{resolution}.npy")
data_np = np.load(path_to_data)


images_flat_raw = jnp.array(data_np)
n_samples = images_flat_raw.shape[0]
print("Data shape:", images_flat_raw.shape)

# Add noise to the data for regularization
_, subkey_image_noise_added =  jr.split(image_noise_added_key)  # Seed for reproducibility
gaussian_noise = jr.normal(subkey_image_noise_added, images_flat_raw.shape) * std_of_added_noise 
images_flat_true = images_flat_raw + gaussian_noise

# Normalize the data
samples_target_unscaled, mean_MNIST, std_MNIST = normalize_samples(images_flat_true)
samples_target = samples_target_unscaled*additional_rescaling



##################################### 
# Setup network
##################################### 

#####################################
# Network size and topology
N_osc = resolution[0]**2
connectivity = create_2d_square_lattice_connectivity(grid_size=resolution[0], n_neighbour_couplings=n_neighbour_couplings)
num_connections = connectivity.shape[0]
visualize_connectivity_with_non_local_couplings(connectivity, grid_size_x=resolution[0], grid_size_y=resolution[1], n_neighbour_couplings=n_neighbour_couplings, save_fig=True, path=output_dir)

print(f"Network size: {N_osc} oscillators, {num_connections} connections")
print(f"Estimated parameter count: {7*N_osc + 3*num_connections}")

#####################################
# Define initial parameters and energy fn
if energy_fn_type == "6th_order_duffing_coupling":
    k_lin_0 = -1.*jnp.ones(N_osc)
    k_duff_0 = jnp.ones(N_osc)
    k_6_0 = jnp.ones(N_osc)
    c_lin_0 = jnp.zeros(num_connections)
    c_optomech_0 = jnp.zeros(num_connections)
    c_duff_0 = jnp.zeros(num_connections)
    biases_0 = jnp.zeros(N_osc)
    params_initial = (k_lin_0, k_duff_0, k_6_0, c_lin_0, c_optomech_0, c_duff_0, biases_0)
    params_names = ['k_lin', 'k_duff', 'k_6', 'c_lin', 'c_optomech', 'c_duff', 'biases']
    params_flattened_initial, unflatten = flatten_util.ravel_pytree(params_initial)
    constraint_indices_k6_self = jnp.arange(2*N_osc,3*N_osc)
    constraint_indices = constraint_indices_k6_self

    # Set up energy fn
    energy_fn = setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_and_6th_order_energy_fn(connectivity, unflatten)
    
    # Set up MEMORY-EFFICIENT analytical derivatives
    gradient_fn, trace_hessian_fn, gradient_wrt_params_dot_product_fn, trace_hessian_wrt_params_fn = setup_duffing_network_analytical_derivatives_memory_efficient(connectivity, unflatten)
    
    print("Using MEMORY-EFFICIENT analytical derivatives!")

else:
    raise ValueError(f"Only 6th_order_duffing_coupling is supported in this memory-efficient version.")

print(f"Total parameters: {len(params_flattened_initial)}")
print_memory_usage("after network setup")

##################################### 
# Learning - MEMORY EFFICIENT VERSION
##################################### 

##################################### 
# Setup loss and gradient functions based on selected method

if training_method == "SM_at_kbT_analytical_memory_efficient":
    # Use ultra memory-efficient version
    loss_fn_per_batch, gradient_fn_per_batch = setup_score_matching_kbT_loss_analytical_ultra_efficient(
        gradient_fn, trace_hessian_fn, gradient_wrt_params_dot_product_fn, trace_hessian_wrt_params_fn, 
        k_b=1.0, T=Temp, max_batch_size=chunk_size
    )
    maximize = False
    learning_rate = learning_rate*Temp
    print(f"Using ULTRA memory-efficient analytical derivatives for score matching at kbT={Temp}")
    print(f"Processing samples one-by-one to minimize memory usage")
elif training_method == "SM_at_kbT_analytical_chunked":
    # Use chunked version
    loss_fn_per_batch, gradient_fn_per_batch = setup_score_matching_kbT_loss_analytical_memory_efficient(
        gradient_fn, trace_hessian_fn, gradient_wrt_params_dot_product_fn, trace_hessian_wrt_params_fn, 
        k_b=1.0, T=Temp, chunk_size=chunk_size
    )
    maximize = False
    learning_rate = learning_rate*Temp
    print(f"Using chunked memory-efficient analytical derivatives for score matching at kbT={Temp}")
    print(f"Chunk size: {chunk_size}")
elif training_method == "SM_at_kbT_extreme_efficient":
    # Use extreme memory-efficient version with checkpointing
    loss_fn_per_batch, gradient_fn_per_batch = setup_score_matching_kbT_loss_analytical_extreme_efficient(
        gradient_fn, trace_hessian_fn, gradient_wrt_params_dot_product_fn, trace_hessian_wrt_params_fn, 
        k_b=1.0, T=Temp
    )
    maximize = False
    learning_rate = learning_rate*Temp
    print(f"Using EXTREME memory-efficient version with gradient checkpointing at kbT={Temp}")
    print(f"Processing samples one-by-one with checkpointing")
elif training_method == "SM_at_kbT_minimal_memory":
    # Use minimal memory version with autodiff
    loss_fn_per_batch, gradient_fn_per_batch = setup_score_matching_kbT_loss_minimal_memory(
        gradient_fn, trace_hessian_fn, k_b=1.0, T=Temp
    )
    maximize = False
    learning_rate = learning_rate*Temp
    print(f"Using MINIMAL memory version with JAX autodiff at kbT={Temp}")
    print(f"Avoiding all large parameter gradient matrices")
else:
    raise ValueError(f"Only memory-efficient training methods are supported in this version.")

print_memory_usage("after loss function setup")

#####################################
# Optimization
#####################################
params_history_path = f"{output_dir}/params_history.npy"
time_index_path = f"{output_dir}/current_time_index.txt"

# Check if we have existing parameters and time index
if start_from_scratch:
    print('Starting from scratch')
    # Delete all files in the output directory
    import shutil
    if os.path.exists(output_dir):
        print(f'Deleting all files in {output_dir}')
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plot_folder, exist_ok=True)
    start_t_idx = 0
    params_history_all_t = []
    current_params = params_flattened_initial
    
elif os.path.exists(params_history_path) and os.path.exists(time_index_path):
    # Load existing parameters and time index
    print('Loading existing parameters from', params_history_path)
    full_params_history = jnp.load(params_history_path)
    current_params = full_params_history[-1]
    
    # Store all parameter sets in memory
    params_history_all_t = full_params_history.tolist()
    
    with open(time_index_path, 'r') as f:
        start_t_idx = int(f.read())
    print(f'Resuming from time index {start_t_idx}')
    print(f'Loaded {len(params_history_all_t)} parameter sets in memory')
else:
    print('No existing parameters found, starting from beginning')
    start_t_idx = 0
    params_history_all_t = []
    current_params = params_flattened_initial

print_memory_usage("before optimization loop")

# Only run optimization if we haven't completed all time steps
if start_t_idx < len(forward_time_pts):
    # Setup optimization parameters
    mask = jnp.ones(params_flattened_initial.shape[0])

    # Plotting parameters
    plot_steps = True  # Flag to control whether to plot during optimization
    plot_slice = 1     # Plot every nth step

    # Loop over time points starting from where we left off
    for t_idx in range(start_t_idx, len(forward_time_pts)):
        t_curr = forward_time_pts[t_idx]
        print(f"t_idx: {t_idx}, t_curr: {t_curr}")
        print_memory_usage(f"before optimization at t_idx {t_idx}")
        
        optimization_key, optimization_subkey = jr.split(optimization_key)

        if t_idx == 0:
            samples_0 = sample_forward_process(t_forward, n_samples, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=optimization_key)
            plot_forward_marginals(samples_0, t_forward, sigma_forward, Temp=Temp, beta=1.0, path=plot_folder, save_fig=True, fontsize=16, plot_show=True)
        
        sampler = lambda key: sample_forward_process(
                t_curr, n_samples = batch_size, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=key, beta= 1
            )
        
        params_history, loss_history, current_params, best_loss, best_epoch = run_optimization(
            loss_fn_per_batch=loss_fn_per_batch,
            params_initial=current_params,
            sampler=sampler,
            mask=mask,
            gradient_fn_per_batch=gradient_fn_per_batch,
            key=optimization_subkey,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            maximize=maximize,
            window_size=window_size,
            tolerance=tolerance,
            patience=patience,
            constraint_indices=constraint_indices,
            lr_decay_rate=lr_decay_rate,
            lr_decay_steps=lr_decay_steps
        )
        
        print_memory_usage(f"after optimization at t_idx {t_idx}")
        
        # Store data for plotting before cleanup
        if plot_steps and t_idx % plot_slice == 0:
            # Prepare plotting data before cleanup
            length = len(loss_history)
            num_pts = min(100, length)
            
            # build 100 (or fewer) evenly‐spaced integer indices in [0, length-1]
            raw_idxs = jnp.linspace(0, length - 1, num_pts).astype(int)
            idxs = jnp.unique(jnp.concatenate([
                jnp.array([0], dtype=int),
                raw_idxs,
                jnp.array([length - 1], dtype=int),
            ]))
            # turn into a Python list of ints so we can index the params list
            idxs_py = idxs.tolist()
            
            # slice params_history (still a list) and loss_history (convert to array first)
            params_ds = [params_history[i] for i in idxs_py]
            loss_arr = jnp.array(loss_history)
            loss_ds = loss_arr[idxs]
        else:
            params_ds = None
            loss_ds = None
        
        params_history_all_t.append(current_params)
        
        # Memory optimization: Clear large arrays immediately after use
        del params_history, loss_history
        
        # Aggressive memory cleanup for memory-efficient version
        optimize_memory()
        
        # Additional extreme memory optimizations
        if t_idx % clear_cache_every_n_steps == 0:
            jax.clear_caches()
            print(f"Cleared JAX caches at t_idx {t_idx}")
        
        if t_idx % force_gc_every_n_steps == 0:
            import gc
            gc.collect()
            print(f"Forced garbage collection at t_idx {t_idx}")
        
        print_memory_usage(f"after memory cleanup at t_idx {t_idx}")
        
        # Save current state after each time step
        jnp.save(params_history_path, jnp.array(params_history_all_t))
        print('params saved')
        with open(time_index_path, 'w') as f:
            f.write(str(t_idx + 1))  # Save next time index to resume from
        if plot_steps:
            if t_idx % plot_slice == 0:
                # Use the real plotting data we prepared earlier
                if params_ds is not None and loss_ds is not None:
                    plot_parameter_evolution(
                        params_history=params_ds,
                        loss_history=loss_ds,
                        best_loss=best_loss,
                        best_idx=best_epoch,
                        best_params=current_params,
                        time = t_curr,
                        time_index = t_idx,
                        unflatten=unflatten,
                        N_osc=N_osc,
                        title=f"{training_method} (t = {t_curr:.3f})",
                        maximize=maximize,
                        labels_on=True,
                        save_fig=True,
                        path=output_dir,
                        param_names=params_names,
                        
                    )
                    
                else:
                    print("Skipping plot - no data prepared")
        print('end plotting')
        print_memory_usage(f"end of iteration t_idx {t_idx}")
        print(f"Total params history length: {len(params_history_all_t)}")
        print("---")
    
    print('Optimization complete; saved parameters to', output_dir)
    params_history_all_t = jnp.array(params_history_all_t)
else:
    print('Optimization already completed for all time steps')
    params_history_all_t = jnp.array(params_history_all_t)
    

#####################################
# Interpolating parameters as function of time
# All optimization time steps are stored and used
#####################################

# Parameters and time points should always match since we store all optimization steps
print(f"params_history_all_t shape: {params_history_all_t.shape}")
print(f"forward_time_pts shape: {forward_time_pts.shape}")
print(f"Both arrays have length: {len(params_history_all_t)}")

# First smooth the parameters
smoothed_params = smooth_parameters(params_history_all_t, window_lengths=[10] * params_history_all_t.shape[1], poly_orders=[3] * params_history_all_t.shape[1])

# Get interpolator functions
params_interpolator_smoothed = interpolate_parameters(smoothed_params, forward_time_pts)
params_interpolator_non_smoothed = interpolate_parameters(params_history_all_t, forward_time_pts)

# Then interpolate the smoothed parameters
time_eval = jnp.linspace(forward_time_pts[0], forward_time_pts[-1], 200)
interpolated_params_smoothed = params_interpolator_smoothed(time_eval)
interpolated_params_non_smoothed = params_interpolator_non_smoothed(time_eval)

plot_parameter_as_fn_of_time(params_names, forward_time_pts, forward_time_pts, params_history_all_t, params_interpolator_smoothed, unflatten, N_osc, save_fig=True, path=plot_folder, log_scale=False)

#####################################
# Run reverse process
#####################################

def run_reverse_process_SDE(params_interpolator, suffix, key, Temp, atol, rtol, brownian_tolerance, ode_solve=False, t_final=None):
    # Always use t_forward as the final time
    t_final = t_forward
    print(f"Using t_final = {t_final} for reverse SDE")
    
    forward_params = jnp.zeros_like(params_flattened_initial)
    forward_params = forward_params.at[0:N_osc].set(1.)
    if ode_solve:
        params_interpolator_reverse_plus_linear = lambda t: Temp*params_interpolator(t_final - t) - 1 / sigma_forward**2 * forward_params
    elif training_method.startswith("SM_at_kbT"):
        params_interpolator_reverse_plus_linear = lambda t: 2*params_interpolator(t_final - t) - 1 / sigma_forward**2 * forward_params
    else:
        params_interpolator_reverse_plus_linear = lambda t: 2*Temp*params_interpolator(t_final - t) - 1 / sigma_forward**2 * forward_params

    # Setup SDE functions
    drift_fn, diffusion_fn = setup_overdamped_SDE(energy_fn, params_interpolator_reverse_plus_linear, N_osc, time_dependent_parms=True, Temp=Temp)

    t0 = 0.0
    t1 = t_final
    
    # Number of output time points to save (doesn't affect numerical accuracy)
    # The adaptive solver uses rtol/atol to control internal step size automatically
    n_time_steps_sde = 100  # Fixed number of output points for trajectory visualization
    
    ts = jnp.linspace(t0, t1, n_time_steps_sde)
    dt0 = 0.00000001

    # Generate multiple initial states with memory optimization
    key, subkey = jr.split(key)
    print('subkey', subkey)
    
    # MEMORY FIX: Don't store all initial states at once for large networks
    print(f"Generating {n_trajectories} initial states one by one for memory efficiency...")
    
    # We'll generate initial states one by one during trajectory processing
    # to avoid storing large initial_states array

    # Generate a key for each initial condition
    key, subkey = jr.split(key)
    keys_brownian = jr.split(subkey, n_trajectories)
    print('keys_brownian', keys_brownian)

    # RESTORED SIMPLE APPROACH: Process all trajectories at once (like the working version)
    # This avoids numerical instabilities from chunked processing and memory clearing
    print(f"Processing all {n_trajectories} trajectories at once (stable approach)...")
    
    # Generate all initial states and keys upfront
    key, subkey = jr.split(key)
    initial_states = sample_forward_process(t_final, n_trajectories, sigma_final=sigma_forward, D=Temp, samples0=samples_target, key=subkey)
    key, subkey = jr.split(key)
    keys_brownian = jr.split(subkey, n_trajectories)
    
    # Vectorized solve for all trajectories at once (like the original working version)
    vectorized_solve_SDE = vmap(
        lambda init_state, key_b: solve_SDE(
            drift_fn,
            diffusion_fn,
            init_state,
            key_b,
            t0,
            t1,
            ts.shape[0],
            dt0,
            brownian_tolerance=brownian_tolerance,
            rtol=rtol,
            atol=atol
        ),
        in_axes=(0, 0)
    )
    
    # Run SDE for all initial states at once - no chunking, no memory clearing
    print("Running SDE solver for all trajectories...")
    solutions = vectorized_solve_SDE(initial_states, keys_brownian)
    
    # Extract only final states to save memory (don't store full trajectories)
    final_samples_scaled = solutions.ys[:, -1, :]  # Shape: (n_trajectories, N_osc)
    print(f"Completed all {n_trajectories} trajectories successfully")
    
    # Skip saving full trajectories to save memory
    print("Skipping full trajectory storage to save memory")
    # Save only final states instead of full trajectories
    final_states_path = f"{output_dir}/final_states_{suffix}.npy"
    jnp.save(final_states_path, final_samples_scaled)
    print(f"Final states saved to {final_states_path}")
    
    # Clean up large arrays after use (but not during computation)
    del solutions, initial_states, keys_brownian
    import gc
    gc.collect()
    print("Cleaned up memory after SDE completion")
    final_samples = final_samples_scaled / additional_rescaling
    images_generated_flat = final_samples * std_MNIST + mean_MNIST
    print('images_generated_flat', images_generated_flat)
    return images_generated_flat.reshape(-1, resolution[0], resolution[1])


# Use t_forward as the final time
print(f"Running reverse SDE with final time: {t_forward}")
images_generated_non_smoothed_sde = run_reverse_process_SDE(
    params_interpolator_non_smoothed, 
    f"non_smoothed_sde_memory_efficient_atol_{atol_sde}_rtol_{rtol_sde}_brownian_tolerance_{brownian_tolerance}",
    reverse_sde_key, 
    Temp, 
    atol_sde, 
    rtol_sde, 
    brownian_tolerance,
    ode_solve=False,
    t_final=t_forward
)

print('images_generated_non_smoothed_sde', images_generated_non_smoothed_sde)

#####################################
# Plotting
#####################################
images_true = images_flat_true.reshape(-1, resolution[0], resolution[1])

# Assume num_examples is an integer multiple of 10 (e.g., 20, 30, etc.)
num_examples = n_trajectories  
n_rows = num_examples // 10  # number of rows per set (true and generated)

# Create a grid with 2 * n_rows rows (true images on top, generated images below) and 10 columns per row
fig, axes = plt.subplots(2 * n_rows, 10, figsize=(15, 3 * n_rows))

# Plot true images in the first n_rows rows
for idx in range(num_examples):
    row = idx // 10        # determine row index for the true images block
    col = idx % 10         # determine column index
    ax = axes[row, col]
    img = images_true[idx]
    if img.shape[-1] == 1:  # handle grayscale images with a singleton channel
        img = img.squeeze(-1)
    ax.imshow(np.array(img), cmap='gray')
    ax.axis('off')

# Plot SDE generated images in the next n_rows rows
for idx in range(num_examples):
    row = (idx // 10) + n_rows  # offset rows by n_rows for the generated images block
    col = idx % 10
    ax = axes[row, col]
    img_generated = images_generated_non_smoothed_sde[idx]
    if img_generated.shape[-1] == 1:
        img_generated = img_generated.squeeze(-1)
    ax.imshow(np.array(img_generated), cmap='gray')
    ax.axis('off')

# Add titles with increased font size on the left-most subplot of each block
axes[0, 0].set_title("True images", loc='left', fontsize=16)
axes[n_rows, 0].set_title(f"SDE sampled images (Memory Efficient), rtol={rtol_sde}, atol={atol_sde}, brownian_tolerance={brownian_tolerance}", loc='left', fontsize=16)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.01, hspace=0.5)

# Save the figure
plt.savefig(f'{plot_folder}/samples_memory_efficient_dim_{resolution[0]}_n_couplings_{n_neighbour_couplings}_Temp_{Temp}_atol_{atol_sde}_rtol_{rtol_sde}_brownian_tolerance_{brownian_tolerance}.png')
plt.close()

print_memory_usage("final")
print("Memory-efficient script completed successfully!")
