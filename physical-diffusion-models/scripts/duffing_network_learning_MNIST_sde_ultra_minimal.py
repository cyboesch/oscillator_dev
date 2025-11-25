print("Ultra-minimal memory script started.")
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
from physical_diffusion_fns.learning_fns import run_optimization
from physical_diffusion_fns.plotting_fns import plot_parameter_evolution, plot_forward_marginals, visualize_connectivity_with_non_local_couplings
from physical_diffusion_fns.network_fns import setup_overdamped_SDE, solve_SDE, create_2d_square_lattice_connectivity, setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_and_6th_order_energy_fn

jax.config.update("jax_enable_x64", True)

# Force JAX to use minimal memory allocation
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use only 70% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate all memory

num_devices = jax.local_device_count()
print(f"Number of devices: {num_devices}")

##################################### 
# Set random seeds
##################################### 

key_seed = 123
master_key  = jax.random.PRNGKey(key_seed)          # single seed
optimization_key, reverse_sde_key, image_noise_added_key = jr.split(master_key, 3)

##################################### 
# Set parameters - ULTRA MINIMAL FOR MEMORY
##################################### 
std_of_added_noise = 0.005
additional_rescaling = 1.
# Forward process parameters
Temp = 0.005
n_time_steps = 5  # REDUCED from 15 to 5
t_forward = 5.0
sigma_forward = 1.
exponential_time_pts = False
if exponential_time_pts:
    forward_time_pts = jnp.exp(jnp.linspace(jnp.log(1e-7), jnp.log(t_forward), n_time_steps))
    forward_time_pts = forward_time_pts.at[0].set(0.)
else:
    forward_time_pts = jnp.linspace(0., t_forward, n_time_steps)
print('forward_time_pts', forward_time_pts)

# Optimization parameters - ULTRA MINIMAL
learning_rate = 1.
lr_decay_rate = 0.95
lr_decay_steps = 1000  # REDUCED
n_epochs = 100  # HEAVILY REDUCED from 10000 to 100
batch_size = 2  # ULTRA SMALL - just 2 samples at a time
window_size = 50  # REDUCED
tolerance = 0.5  # RELAXED
patience = 20  # REDUCED
CD1_dt = 0.00001
CD1_num_noise_samples = 100  # REDUCED

# Memory optimization parameters
max_params_history = 10  # Keep only last 10 parameter sets
save_all_params_to_disk = True
chunk_size = 1  # Process ONE sample at a time

# Use simplified training method
training_method = "SM_at_kbT_analytical_ultra_minimal"

print(f"ULTRA MINIMAL MEMORY OPTIMIZATION:")
print(f"  - Batch size: {batch_size}")
print(f"  - Chunk size: {chunk_size}")
print(f"  - Epochs: {n_epochs}")
print(f"  - Time steps: {n_time_steps}")

labels = [0,1]
resolution = (10, 10)  # REDUCED from (20,20) to (10,10) - 4x fewer oscillators!

n_neighbour_couplings = 2  # REDUCED from 6 to 2 - much fewer connections
energy_fn_type = "6th_order_duffing_coupling"

# SDE parameters
n_trajectories = 20  # REDUCED from 100 to 20
rtol_sde = 1e-6  # RELAXED
atol_sde = 1e-7  # RELAXED

start_from_scratch = True  # Always start fresh for testing

print(f"NETWORK SIZE REDUCTION:")
print(f"  - Resolution: {resolution} (was 20x20)")
print(f"  - Neighbor couplings: {n_neighbour_couplings} (was 6)")
print(f"  - Trajectories: {n_trajectories} (was 100)")

#####################################
## Filenames
#####################################

problem_type_folder = f"MNIST_generation_ultra_minimal"

system_specifics = f"n_neighbour_couplings_{n_neighbour_couplings}_Temp_{Temp}_energy_fn_type_{energy_fn_type}"

MNIST_specifics = f"labels_{labels}_resolution_{resolution[0]}_x_{resolution[1]}"

data_folder = f"added_gaussian_noise_std_{std_of_added_noise}_additional_rescaling_{additional_rescaling}_key_seed_{key_seed}"

optimization_folder = (
    f"ultra_minimal_training_method_{training_method}_"
    f"t_forward_{t_forward}_"
    f"n_timesteps_{n_time_steps}_"
    f"sigma_forward_{sigma_forward}_"
    f"lr_{learning_rate}_"
    f"epochs_{n_epochs}_"
    f"batch_{batch_size}_"
    f"chunk_{chunk_size}"
)

# Setup output directories
here = os.path.dirname(os.path.abspath(__file__))
current_date = datetime.now().strftime("%Y_%m_%d")
base_dir = os.path.join(here,"..","out", current_date, "problems")
output_dir = os.path.join(base_dir, problem_type_folder, MNIST_specifics, system_specifics, data_folder, optimization_folder)
plot_folder = os.path.join(output_dir, 'aaa_final_plots')

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)
print(f"Output directory: {output_dir}")

##################################### 
# Load MNIST data - REDUCED RESOLUTION
##################################### 
import numpy as np

# Check if reduced resolution data exists
path_to_data = os.path.join(here,"..", "data", "MNIST", f"mnist_labels_{labels}_resolution_{resolution}.npy")
if not os.path.exists(path_to_data):
    print(f"Creating reduced resolution MNIST data at {path_to_data}")
    # Load original 20x20 data and downsample
    original_path = os.path.join(here,"..", "data", "MNIST", f"mnist_labels_{labels}_resolution_(20, 20).npy")
    if os.path.exists(original_path):
        original_data = np.load(original_path)
        # Reshape to images, downsample, and flatten
        original_images = original_data.reshape(-1, 20, 20)
        # Simple downsampling by taking every other pixel
        downsampled_images = original_images[:, ::2, ::2]  # 20x20 -> 10x10
        downsampled_data = downsampled_images.reshape(-1, resolution[0] * resolution[1])
        np.save(path_to_data, downsampled_data)
        print(f"Created {path_to_data} with shape {downsampled_data.shape}")
    else:
        print(f"ERROR: Could not find original MNIST data at {original_path}")
        print("Please create 10x10 MNIST data first")
        sys.exit(1)

data_np = np.load(path_to_data)
images_flat_raw = jnp.array(data_np)
n_samples = images_flat_raw.shape[0]
print("Data shape:", images_flat_raw.shape)

# Add noise to the data for regularization
_, subkey_image_noise_added =  jr.split(image_noise_added_key)
gaussian_noise = jr.normal(subkey_image_noise_added, images_flat_raw.shape) * std_of_added_noise 
images_flat_true = images_flat_raw + gaussian_noise

# Normalize the data
samples_target_unscaled, mean_MNIST, std_MNIST = normalize_samples(images_flat_true)
samples_target = samples_target_unscaled*additional_rescaling

print_memory_usage("after data loading")

##################################### 
# Setup network - MINIMAL SIZE
##################################### 

N_osc = resolution[0]**2  # 100 oscillators instead of 400
connectivity = create_2d_square_lattice_connectivity(grid_size=resolution[0], n_neighbour_couplings=n_neighbour_couplings)
num_connections = connectivity.shape[0]

print(f"Network size: {N_osc} oscillators, {num_connections} connections")
estimated_params = 7*N_osc + 3*num_connections
print(f"Estimated parameter count: {estimated_params}")

if estimated_params > 5000:
    print("WARNING: Still too many parameters! Consider reducing network size further.")

# Visualize connectivity (but don't save to reduce memory)
# visualize_connectivity_with_non_local_couplings(connectivity, grid_size_x=resolution[0], grid_size_y=resolution[1], n_neighbour_couplings=n_neighbour_couplings, save_fig=False, path=output_dir)

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
    
    print("Using STANDARD JAX autodiff (most memory efficient for small networks)")

else:
    raise ValueError(f"Only 6th_order_duffing_coupling is supported in this minimal version.")

print(f"Total parameters: {len(params_flattened_initial)}")
print_memory_usage("after network setup")

##################################### 
# Learning - ULTRA SIMPLE VERSION
##################################### 

def setup_ultra_simple_score_matching_loss(energy_fn, k_b=1.0, T=1.0):
    """
    Ultra simple version using JAX autodiff - most memory efficient for small networks.
    """
    def loss_fn_per_batch(flattened_args, batch):
        n_samples = batch.shape[0]
        
        def score_loss_per_sample(x):
            # Use JAX autodiff for everything - simpler and more memory efficient for small networks
            def log_prob_unnorm(x):
                return -energy_fn(x, flattened_args) / (k_b * T)
            
            # Score function
            score = jax.grad(log_prob_unnorm)(x)
            
            # Hessian trace using JAX
            hessian_fn = jax.hessian(log_prob_unnorm)
            hess_matrix = hessian_fn(x)
            trace_hess = jnp.trace(hess_matrix)
            
            # Score matching loss
            return (trace_hess + 0.5 * jnp.sum(score**2)) / n_samples
        
        # Process samples one by one to minimize memory
        total_loss = 0.0
        for i in range(n_samples):
            total_loss += score_loss_per_sample(batch[i])
        
        return total_loss
    
    return loss_fn_per_batch

# Setup loss function
loss_fn_per_batch = setup_ultra_simple_score_matching_loss(energy_fn, k_b=1.0, T=Temp)
gradient_fn_per_batch = None  # Use JAX autodiff
maximize = False
learning_rate = learning_rate * Temp

print(f"Using ultra-simple JAX autodiff for score matching at kbT={Temp}")
print_memory_usage("after loss function setup")

#####################################
# Optimization - MINIMAL VERSION
#####################################
params_history_path = f"{output_dir}/params_history.npy"
time_index_path = f"{output_dir}/current_time_index.txt"

# Always start from scratch for testing
start_t_idx = 0
params_history_all_t = []
current_params = params_flattened_initial

print_memory_usage("before optimization loop")

# Only run optimization if we haven't completed all time steps
if start_t_idx < len(forward_time_pts):
    # Setup optimization parameters
    mask = jnp.ones(params_flattened_initial.shape[0])

    # Loop over time points starting from where we left off
    for t_idx in range(start_t_idx, len(forward_time_pts)):
        t_curr = forward_time_pts[t_idx]
        print(f"t_idx: {t_idx}, t_curr: {t_curr}")
        print_memory_usage(f"before optimization at t_idx {t_idx}")
        
        optimization_key, optimization_subkey = jr.split(optimization_key)

        if t_idx == 0:
            samples_0 = sample_forward_process(t_forward, n_samples, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=optimization_key)
            # Skip plotting to save memory
            # plot_forward_marginals(samples_0, t_forward, sigma_forward, Temp=Temp, beta=1.0, path=plot_folder, save_fig=True, fontsize=16, plot_show=True)
        
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
        
        params_history_all_t.append(current_params)
        
        # Memory optimization: Keep only last N parameter sets in memory
        if len(params_history_all_t) > max_params_history:
            params_history_all_t = params_history_all_t[-max_params_history:]
        
        # Memory optimization: Clear large arrays immediately after use
        del params_history, loss_history
        
        # Aggressive memory cleanup
        optimize_memory()
        print_memory_usage(f"after memory cleanup at t_idx {t_idx}")
        
        # Save current state after each time step
        if save_all_params_to_disk:
            if os.path.exists(params_history_path):
                full_params_history = jnp.load(params_history_path).tolist()
                full_params_history.append(current_params)
                jnp.save(params_history_path, jnp.array(full_params_history))
            else:
                jnp.save(params_history_path, jnp.array([current_params]))
        
        with open(time_index_path, 'w') as f:
            f.write(str(t_idx + 1))
        
        print(f"Completed t_idx {t_idx}, params history length: {len(params_history_all_t)}")
        print("---")
    
    print('Optimization complete; saved parameters to', output_dir)
    params_history_all_t = jnp.array(params_history_all_t)
else:
    print('Optimization already completed for all time steps')
    params_history_all_t = jnp.array(params_history_all_t)

print_memory_usage("after optimization")

#####################################
# Simplified reverse process (skip for now to test optimization)
#####################################

print("Skipping reverse process and plotting to minimize memory usage")
print("If optimization completes successfully, we can add these back")

print_memory_usage("final")
print("Ultra-minimal script completed successfully!")
print(f"Final parameter shape: {params_history_all_t.shape}")
