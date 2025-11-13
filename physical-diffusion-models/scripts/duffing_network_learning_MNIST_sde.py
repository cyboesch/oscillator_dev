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
from physical_diffusion_fns.learning_fns import setup_MLE_loss_per_batch, setup_MLE_gradient_per_batch, CD1_gradient, setup_score_matching_loss_per_batch, run_optimization, setup_score_matching_kbT_loss_per_batch, setup_score_matching_kbT_local_gradient_per_batch, run_optimization_multi_gpu_sampler,setup_score_matching_kbT_loss_analytical
from physical_diffusion_fns.plotting_fns import plot_energy_and_distributions, plot_parameter_evolution, plot_forward_marginals, visualize_connectivity, plot_parameter_as_fn_of_time, visualize_connectivity_with_non_local_couplings
from physical_diffusion_fns.network_fns import setup_overdamped_SDE, solve_SDE, create_2d_square_lattice_connectivity, setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_and_6th_order_energy_fn, setup_duffing_network_analytical_derivatives

jax.config.update("jax_enable_x64", True)

num_devices = jax.local_device_count()
print(f"Number of devices: {num_devices}")

##################################### 
# Set random seeds
##################################### 

key_seed = 123
master_key  = jax.random.PRNGKey(key_seed)          # single seed
optimization_key, reverse_sde_key, image_noise_added_key = jr.split(master_key, 3)

##################################### 
# Set parameters
##################################### 
std_of_added_noise = 0.005
additional_rescaling = 1.
# Forward process parameters
Temp = 0.005
n_time_steps = 15
t_forward = 5.0
sigma_forward = 1.
exponential_time_pts = False
if exponential_time_pts:
    forward_time_pts = jnp.exp(jnp.linspace(jnp.log(1e-7), jnp.log(t_forward), n_time_steps))
    forward_time_pts = forward_time_pts.at[0].set(0.)
else:
    forward_time_pts = jnp.linspace(0., t_forward, n_time_steps)
print('forward_time_pts', forward_time_pts)

# Optimization parameters
learning_rate = 1.
lr_decay_rate = 0.95
lr_decay_steps = 6000
n_epochs = 100000
batch_size = 4*128
window_size=1000
tolerance=.1
patience=100
CD1_dt = 0.00001
CD1_num_noise_samples = 1000

# Memory optimization parameters
max_params_history = 50  # Keep only last 50 parameter sets in memory
save_all_params_to_disk = True  # Save all params to disk but keep limited in memory

# Set training method and display memory usage warning
training_method = "SM_at_kbT_analytical"


labels = [0,1]
resolution = (12,12)

n_neighbour_couplings = 1
energy_fn_type = "6th_order_duffing_coupling"

# SDE parameters
n_trajectories = 100
rtol_sde = 1e-9
atol_sde = 1e-10

start_from_scratch = False

#####################################
## Filenames
#####################################

problem_type_folder = f"MNIST_generation"

system_specifics = f"n_neighbour_couplings_{n_neighbour_couplings}_Temp_{Temp}_energy_fn_type_{energy_fn_type}"

MNIST_specifics = f"labels_{labels}_resolution_{resolution[0]}_x_{resolution[1]}"

data_folder = f"added_gaussian_noise_std_{std_of_added_noise}_additional_rescaling_{additional_rescaling}_key_seed_{key_seed}"

optimization_folder = (
    f"schedule_training_method_{training_method}_"
    + (f"CD1_dt_{CD1_dt}_CD1_num_noise_samples_{CD1_num_noise_samples}_" if training_method == "CD1" else "")
    + f"exp_time_pts_{exponential_time_pts}_"
    f"t_forward_{t_forward}_"
    f"n_timesteps_{n_time_steps}_"
    f"sigma_forward_{sigma_forward}_"
    f"lr_{learning_rate}_"
    f"lr_decay_rate_{lr_decay_rate}_"
    f"lr_decay_steps_{lr_decay_steps}_"
    f"epochs_{n_epochs}_"
    f"batch_{batch_size}_"
    f"window_{window_size}_"
    f"tol_{tolerance}_"
    f"patience_{patience}"
)

# Setup output directories
here = os.path.dirname(os.path.abspath(__file__))
current_date = datetime.now().strftime("%Y_%m_%d")
# current_date = "2025_07_11"
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
    # Set up analytical derivatives
    gradient_fn, trace_hessian_fn, gradient_wrt_params_fn, trace_hessian_wrt_params_fn = setup_duffing_network_analytical_derivatives(connectivity, unflatten)
elif energy_fn_type == "duffing_coupling":
    k_lin_0 = -1.*jnp.ones(N_osc)
    k_duff_0 = jnp.ones(N_osc)
    c_lin_0 = jnp.zeros(num_connections)
    c_optomech_0 = jnp.zeros(num_connections)
    c_duff_0 = jnp.zeros(num_connections)
    biases_0 = jnp.zeros(N_osc)
    params_initial = (k_lin_0, k_duff_0, c_lin_0, c_optomech_0, c_duff_0, biases_0)
    params_names = ['k_lin', 'k_duff', 'c_lin', 'c_optomech', 'c_duff', 'biases']
    params_flattened_initial, unflatten = flatten_util.ravel_pytree(params_initial)
    constraint_indices_duff_self = jnp.arange(N_osc,2*N_osc)
    constraint_indices_duff_coupling = jnp.arange(2*N_osc+2*num_connections,2*N_osc+3*num_connections)
    constraint_indices = jnp.concatenate([constraint_indices_duff_self, constraint_indices_duff_coupling])
    # Set up energy fn
    energy_fn = setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_energy_fn(connectivity, unflatten)
    # Set up analytical derivatives
    gradient_fn, trace_hessian_fn, gradient_wrt_params_fn, trace_hessian_wrt_params_fn = setup_duffing_network_analytical_derivatives(connectivity, unflatten)
elif energy_fn_type == "duffing_optomech_coupling":
    k_lin_0 = -1.*jnp.ones(N_osc)
    k_duff_0 = jnp.ones(N_osc)
    c_lin_0 = jnp.zeros(num_connections)
    c_optomech_0 = jnp.zeros(num_connections)
    biases_0 = jnp.zeros(N_osc)
    params_initial = (k_lin_0, k_duff_0, c_lin_0, c_optomech_0, biases_0)
    params_names = ['k_lin', 'k_duff', 'c_lin', 'c_optomech', 'biases']
    params_flattened_initial, unflatten = flatten_util.ravel_pytree(params_initial)
    constraint_indices = jnp.arange(N_osc,2*N_osc)
    # Set up energy fn
    energy_fn = setup_duffing_network_with_external_force_energy_fn(connectivity, unflatten)
    # Set up analytical derivatives (Note: this function is designed for 6th order duffing coupling, may need adjustment for other energy types)
    gradient_fn, trace_hessian_fn, gradient_wrt_params_fn, trace_hessian_wrt_params_fn = setup_duffing_network_analytical_derivatives(connectivity, unflatten)
else:
    raise ValueError(f"Unknown energy function type: {energy_fn_type}. Choose from '6th_order_duffing_coupling' or 'duffing_coupling'.")

##################################### 
# Learning
##################################### 

##################################### 
# Setup loss and gradient functions based on selected method

if training_method == "SM":
    loss_fn_per_batch = setup_score_matching_loss_per_batch(energy_fn)
    gradient_fn_per_batch = None
    maximize = False
elif training_method == "SM_at_kbT":
    loss_fn_per_batch = setup_score_matching_kbT_loss_per_batch_hessian(energy_fn, k_b=1.0, T=Temp)
    gradient_fn_per_batch = None
    maximize = False
    learning_rate = learning_rate*Temp
    print(f"Using exact Hessian computation for score matching at kbT={Temp}")
elif training_method == "SM_at_kbT_analytical":
    # Use analytical version for much better performance
    loss_fn_per_batch, gradient_fn_per_batch = setup_score_matching_kbT_loss_analytical(
        gradient_fn, trace_hessian_fn, gradient_wrt_params_fn, trace_hessian_wrt_params_fn, 
        k_b=1.0, T=Temp
    )
    maximize = False
    learning_rate = learning_rate*Temp
    print(f"Using analytical derivatives for score matching at kbT={Temp} - MUCH more efficient!")
elif training_method == "SM_local_at_kbT":
    # For local gradient, we use the Hutchinson version for the loss (more memory efficient)
    # but the local gradient computation is separate
    loss_fn_per_batch = setup_score_matching_kbT_loss_per_batch(energy_fn, k_b=1.0, T=Temp)
    gradient_fn_per_batch = setup_score_matching_kbT_local_gradient_per_batch(energy_fn, k_b=1.0, T=Temp)
    maximize = False
    learning_rate = learning_rate*Temp
elif training_method == "CD1":
    loss_fn_per_batch = setup_score_matching_loss_per_batch(energy_fn)
    gradient_fn_per_batch = lambda flattened_args, batch, key_CD1: -CD1_gradient(energy_fn, batch, flattened_args, dt=CD1_dt, D=Temp, key=key_CD1, num_noise_samples=CD1_num_noise_samples)
    learning_rate = learning_rate*Temp
    maximize = False
elif training_method == "MLE":
    loss_fn_per_batch = setup_MLE_loss_per_batch(energy_fn)
    gradient_fn_per_batch = setup_MLE_gradient_per_batch(energy_fn)
    maximize = True
else:
    raise ValueError(f"Unknown training method: {training_method}. Choose from 'SM', 'SM_at_kbT', 'SM_at_kbT_analytical', 'SM_local_at_kbT', 'CD1', or 'MLE'.")


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
    
    # Memory optimization: Keep only last N parameter sets in memory
    if len(full_params_history) > max_params_history:
        params_history_all_t = full_params_history[-max_params_history:].tolist()
    else:
        params_history_all_t = full_params_history.tolist()
    
    with open(time_index_path, 'r') as f:
        start_t_idx = int(f.read())
    print(f'Resuming from time index {start_t_idx}')
    print(f'Keeping last {len(params_history_all_t)} parameter sets in memory')
else:
    print('No existing parameters found, starting from beginning')
    start_t_idx = 0
    params_history_all_t = []
    current_params = params_flattened_initial



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
        
        params_history, loss_history = run_optimization(
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
        
        # Get best parameters BEFORE deleting the history
        best_loss, current_params, best_idx = get_best_params(params_history, loss_history, maximize=maximize)
        print(f"Best loss: {best_loss:.4e}")
        print(f"Found at epoch: {best_idx}")
        
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
        
        # Memory optimization: Keep only last N parameter sets in memory
        if len(params_history_all_t) > max_params_history:
            params_history_all_t = params_history_all_t[-max_params_history:]
        
        # Memory optimization: Clear large arrays immediately after use
        del params_history, loss_history
        
        # More frequent garbage collection for hessian computation
        if training_method == "SM_at_kbT":
            # Force memory cleanup after each iteration for hessian computation
            optimize_memory()
            print_memory_usage(f"after hessian computation cleanup at t_idx {t_idx}")
        elif training_method == "SM_at_kbT_analytical":
            # Analytical version is much more memory efficient, but still do some cleanup
            if t_idx % 2 == 0:  # Less frequent cleanup needed
                optimize_memory()
                print_memory_usage(f"after analytical computation cleanup at t_idx {t_idx}")
        
        # Force garbage collection every few iterations
        if t_idx % 3 == 0:
            optimize_memory()
            print_memory_usage(f"after memory optimization at t_idx {t_idx}")
        
        # Save current state after each time step
        if save_all_params_to_disk:
            # For disk storage, we need to load, append, and save to keep all history
            if os.path.exists(params_history_path):
                full_params_history = jnp.load(params_history_path).tolist()
                full_params_history.append(current_params)
                jnp.save(params_history_path, jnp.array(full_params_history))
            else:
                jnp.save(params_history_path, jnp.array([current_params]))
        else:
            # Save only current limited history
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
        print(f"Params history length in memory: {len(params_history_all_t)}")
        print("---")
    
    print('Optimization complete; saved parameters to', output_dir)
    params_history_all_t = jnp.array(params_history_all_t)
else:
    print('Optimization already completed for all time steps')
    params_history_all_t = jnp.array(params_history_all_t)
    

#####################################
# Interpolating parameters as function of time
#####################################

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

def run_reverse_process_SDE(params_interpolator, suffix,key, Temp, atol, rtol, ode_solve=False):
    forward_params = jnp.zeros_like(params_flattened_initial)
    forward_params = forward_params.at[0:N_osc].set(1.)
    if ode_solve:
        params_interpolator_reverse_plus_linear = lambda t: Temp*params_interpolator(t_forward - t) - 1 / sigma_forward**2 * forward_params
    elif training_method == "CD1" or training_method == "SM_at_kbT" or training_method == "SM_at_kbT_analytical" or training_method == "SM_local_at_kbT":
        params_interpolator_reverse_plus_linear = lambda t: 2*params_interpolator(t_forward - t) - 1 / sigma_forward**2 * forward_params
    else:
        params_interpolator_reverse_plus_linear = lambda t: 2*Temp*params_interpolator(t_forward - t) - 1 / sigma_forward**2 * forward_params

    # Setup SDE functions
    drift_fn, diffusion_fn = setup_overdamped_SDE(energy_fn, params_interpolator_reverse_plus_linear, N_osc, time_dependent_parms=True, Temp=Temp)

    t0 = 0.0
    t1 = t_forward
    ts = jnp.linspace(t0, t1, 100)
    dt0 = 0.00000001

    # Generate multiple initial states
    key, subkey = jr.split(key)
    print('subkey', subkey)
    initial_states = sample_forward_process(t_forward, n_trajectories, sigma_final=sigma_forward, D=Temp, samples0=samples_target, key=subkey)
    print('initial_states', initial_states)

    # Generate a key for each initial condition
    key, subkey = jr.split(key)
    keys_brownian = jr.split(subkey, n_trajectories)
    print('keys_brownian', keys_brownian)

    # Vectorize solve_SDE across both initial states and keys
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
            rtol=rtol,
            atol=atol
        ),
        in_axes=(0, 0)
    )

    # Run SDE for all initial states at once
    solutions = vectorized_solve_SDE(initial_states, keys_brownian)
    all_trajectories = solutions.ys  # Shape: (n_trajectories, n_timesteps, N_osc)
    reverse_trajectories_path = f"{output_dir}/reverse_trajectories_{suffix}.npy"
    jnp.save(reverse_trajectories_path, all_trajectories)
    print(f"Reverse trajectories saved to {reverse_trajectories_path}")

    final_samples_scaled = all_trajectories[:, -1, :]  # Shape: (n_trajectories, N_osc)
    final_samples = final_samples_scaled / additional_rescaling
    images_generated_flat = final_samples * std_MNIST + mean_MNIST
    print('images_generated_flat', images_generated_flat)
    return images_generated_flat.reshape(-1, resolution[0], resolution[1])


# Run reverse process for both smoothed and non-smoothed parameters
images_generated_non_smoothed_sde = run_reverse_process_SDE(params_interpolator_non_smoothed, "non_smoothed_sde",reverse_sde_key, Temp, atol_sde, rtol_sde, ode_solve=False)

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
axes[n_rows, 0].set_title(f"SDE sampled images, rtol={rtol_sde}, atol={atol_sde}", loc='left', fontsize=16)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.01, hspace=0.5)

# Save the figure
plt.savefig(f'{plot_folder}/samples_dim_{resolution[0]}_n_couplings_{n_neighbour_couplings}_Temp_{Temp}_rtol_sde_{rtol_sde}_atol_sde_{atol_sde}.png')
plt.close()

