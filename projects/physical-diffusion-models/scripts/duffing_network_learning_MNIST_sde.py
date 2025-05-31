print("Script started.")
import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import jax
from jax import flatten_util, vmap 
import jax.numpy as jnp
import jax.random as jr
import os
import matplotlib.pyplot as plt
from physical_diffusion_fns.helper_fns import sample_gaussian_mixture, normalize_samples, sample_forward_process, get_best_params, smooth_parameters, interpolate_parameters
from physical_diffusion_fns.learning_fns import setup_MLE_loss_per_batch, setup_MLE_gradient_per_batch, CD1_gradient, setup_score_matching_loss_per_batch, run_optimization
from physical_diffusion_fns.plotting_fns import plot_energy_and_distributions, plot_parameter_evolution, plot_forward_marginals, visualize_connectivity, plot_parameter_as_fn_of_time, visualize_connectivity_with_non_local_couplings
from physical_diffusion_fns.network_fns import setup_duffing_network_with_external_force_energy_fn, setup_overdamped_SDE, solve_SDE, create_2d_square_lattice_connectivity, setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_energy_fn,setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_and_6th_order_energy_fn

jax.config.update("jax_enable_x64", True)

##################################### 
# Set random seeds
##################################### 

key_seed = 0
master_key  = jax.random.PRNGKey(key_seed)          # single seed
optimization_key, reverse_sde_key = jr.split(master_key, 2)

##################################### 
# Set parameters
##################################### 
std_of_added_noise = 0.01
additional_rescaling = 1.
# Forward process parameters
n_time_steps = 15
t_forward = 4.
sigma_forward = 1.
exponential_time_pts = False
if exponential_time_pts:
    forward_time_pts = jnp.exp(jnp.linspace(jnp.log(1e-7), jnp.log(t_forward), n_time_steps))
    forward_time_pts = forward_time_pts.at[0].set(0.)
else:
    forward_time_pts = jnp.linspace(0., t_forward, n_time_steps)
print('forward_time_pts', forward_time_pts)

# Optimization parameters
training_method = "CD1"
learning_rate = 0.01
n_epochs = 10000
batch_size = 6*128
window_size=100
tolerance=1e-3
patience=10

Temp = 0.01

key_seed = 0

labels = [0,1]
resolution = (10,10)

n_neighbour_couplings = 3

energy_fn_type = "6th_order_duffing_coupling"

# SDE parameters
n_trajectories = 100
rtol_sde = 1e-7
atol_sde = 1e-9

#####################################
## Filenames
#####################################

problem_type_folder = f"MNIST_generation/Energy_fn_type_{energy_fn_type}_n_neighbour_couplings_{n_neighbour_couplings}_Temp_{Temp}"
data_folder = f"added_gaussian_noise_std_{std_of_added_noise}_additional_rescaling_{additional_rescaling}_key_seed_{key_seed}_training_method_{training_method}"

optimization_folder = (
           f"exp_time_pts_{exponential_time_pts}_"
           f"t_forward_{t_forward}_"
           f"n_timesteps_{n_time_steps}_"
           f"sigma_forward_{sigma_forward}_"
           f"lr_{learning_rate}_"
           f"epochs_{n_epochs}_"
           f"batch_{batch_size}_"
           f"window_{window_size}_"
           f"tol_{tolerance}_"
           f"patience_{patience}")

# Setup output directories
here = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(here,"..", "out", "problems")
output_dir = os.path.join(base_dir, problem_type_folder, f"MNIST_labels_{labels}_resolution_{resolution}", data_folder, optimization_folder)
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

# Optionally convert to a JAX array
images_flat_raw = jnp.array(data_np)
n_samples = images_flat_raw.shape[0]
print("Data shape:", images_flat_raw.shape)

# Add noise to the data for regularization
key =  jr.PRNGKey(key_seed)  
key, subkey =  jr.split(key)  # Seed for reproducibility
gaussian_noise = jr.normal(subkey, images_flat_raw.shape) * std_of_added_noise 
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
elif training_method == "CD1":
    loss_fn_per_batch = setup_score_matching_loss_per_batch(energy_fn)
    gradient_fn_per_batch = lambda flattened_args, batch, key_CD1: -CD1_gradient(energy_fn, batch, flattened_args, dt=0.001, D=Temp, key=key_CD1, num_noise_samples=1000)
    maximize = False
elif training_method == "MLE":
    loss_fn_per_batch = setup_MLE_loss_per_batch(energy_fn)
    gradient_fn_per_batch = setup_MLE_gradient_per_batch(energy_fn)
    maximize = True
else:
    raise ValueError(f"Unknown training method: {training_method}. Choose from 'SM', 'CD1', or 'MLE'.")


#####################################
# Optimization
#####################################
params_history_path = f"{output_dir}/params_history.npy"
time_index_path = f"{output_dir}/current_time_index.txt"

# Check if we have existing parameters and time index
if os.path.exists(params_history_path) and os.path.exists(time_index_path):
    # Load existing parameters and time index
    print('Loading existing parameters from', params_history_path)
    params_history_all_t = jnp.load(params_history_path)
    with open(time_index_path, 'r') as f:
        start_t_idx = int(f.read())
    print(f'Resuming from time index {start_t_idx}')
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
        
        optimization_key, optimization_subkey = jr.split(optimization_key)

        if t_idx == 0:
            samples_0 = sample_forward_process(t_forward, n_samples, D=1, sigma_final=sigma_forward, samples0=samples_target, key=optimization_key)
            plot_forward_marginals(samples_0, t_forward, sigma_forward, beta=1.0, path=output_dir, save_fig=True, fontsize=16, plot_show=True)
        
        sampler = lambda key: sample_forward_process(
                t_curr, n_samples = batch_size, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=key, beta= 1
            )
        
        # Perform optimization starting from previous best parameters
        key, subkey = jr.split(key)
        
        params_history, loss_history = run_optimization(
            loss_fn_per_batch=loss_fn_per_batch,
            params_initial=current_params,
            sampler = sampler,
            mask=mask,
            gradient_fn_per_batch=gradient_fn_per_batch,
            key=optimization_subkey,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            maximize=maximize,
            window_size=window_size,
            tolerance=tolerance,
            patience=patience,
            constraint_indices=constraint_indices
        )
        
        _, current_params, _ = get_best_params(params_history, loss_history, maximize=maximize)
        params_history_all_t.append(current_params)
        
        # Save current state after each time step
        jnp.save(params_history_path, jnp.array(params_history_all_t))
        with open(time_index_path, 'w') as f:
            f.write(str(t_idx + 1))  # Save next time index to resume from
        
        if plot_steps:
            if t_idx % plot_slice == 0:
                # Plot optimization progress
                plot_parameter_evolution(
                    params_history=params_history,
                    loss_history=loss_history,
                    time = t_curr,
                    time_index = t_idx,
                    unflatten=unflatten,
                    N_osc=N_osc,
                    slicing=10,
                    title=f"{training_method} (t = {t_curr:.3f})",
                    maximize=maximize,
                    labels_on=True,
                    save_fig=True,
                    path=output_dir,
                    param_names=params_names,
                )
    
    print('Optimization complete; saved parameters to', output_dir)
    params_history_all_t = jnp.array(params_history_all_t)
else:
    print('Optimization already completed for all time steps')
    

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
    elif training_method == "CD1":
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
    initial_states = sample_forward_process(t_forward, n_trajectories, sigma_final=sigma_forward, D=Temp, samples0=samples_target, key=subkey)

    # Generate a key for each initial condition
    key, subkey = jr.split(key)
    keys_brownian = jr.split(subkey, n_trajectories)

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
    return images_generated_flat.reshape(-1, resolution[0], resolution[1])


# Run reverse process for both smoothed and non-smoothed parameters
key, subkey = jr.split(key)
images_generated_non_smoothed_sde = run_reverse_process_SDE(params_interpolator_non_smoothed, "non_smoothed_sde",subkey, Temp, atol_sde, rtol_sde, ode_solve=False)

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
plt.savefig(f'{plot_folder}/true_vs_generated_images_comparison_rtol_sde_{rtol_sde}_atol_sde_{atol_sde}_n_samples_{n_trajectories}.png')
plt.close()

