
import jax
from jax import flatten_util, vmap 
import jax.numpy as jnp
import jax.random as jr
import os
import matplotlib.pyplot as plt
from physical_diffusion_fns.helper_fns import sample_gaussian_mixture, normalize_samples, sample_forward_process, get_best_params, smooth_parameters, interpolate_parameters
from physical_diffusion_fns.learning_fns import setup_MLE_loss_per_batch, setup_MLE_gradient_per_batch, CD1_gradient, setup_score_matching_loss_per_batch, run_optimization
from physical_diffusion_fns.plotting_fns import plot_energy_and_distributions, plot_parameter_evolution, plot_forward_marginals, visualize_connectivity, plot_parameter_as_fn_of_time
from physical_diffusion_fns.network_fns import setup_duffing_network_with_external_force_energy_fn, setup_overdamped_SDE, solve_SDE, create_2d_square_grid_connectivity


jax.config.update("jax_enable_x64", True)


##################################### 
# Load MNIST data
##################################### 
import numpy as np
# Load the saved .npy file
data_np = np.load("data/MNIST/MNIST_0_1_8x8pix/mnist_0_1_8x8pix.npy")

# Optionally convert to a JAX array
images_flat_raw = jnp.array(data_np)
n_samples = images_flat_raw.shape[0]
print("Data shape:", images_flat_raw.shape)

mean = jnp.mean(images_flat_raw)  # This calculates mean over all samples and pixels
std = jnp.std(images_flat_raw)    # This calculates std over all samples and pixels
samples_target, mean_MNIST, std_MNIST = normalize_samples(images_flat_raw)

##################################### 
# Setup network
##################################### 

#####################################
# Network size and topology
N_osc = 64
connectivity = create_2d_square_grid_connectivity(grid_size=8)
num_connections = connectivity.shape[0]

#####################################
# Define initial parameters
k_lin_0 = -1.*jnp.ones(N_osc)
k_duff_0 = jnp.ones(N_osc)
c_lin_0 = jnp.zeros(num_connections)
c_optomech_0 = jnp.zeros(num_connections)
biases_0 = jnp.zeros(N_osc)
params_initial = (k_lin_0, k_duff_0, c_lin_0, c_optomech_0, biases_0)
params_names = ['k_lin', 'k_duff', 'c_lin', 'c_optomech', 'biases']
params_flattened_initial, unflatten = flatten_util.ravel_pytree(params_initial)

#####################################
# Bringe energy into correct form
energy_fn = setup_duffing_network_with_external_force_energy_fn(connectivity, unflatten)

##################################### 
# Learning
##################################### 

##################################### 
# Choose training method
training_method = "SM"  # Options: "SM" (Score Matching), "CD1" (Contrastive Divergence), "MLE" (Maximum Likelihood)

# Setup loss and gradient functions based on selected method
if training_method == "SM":
    loss_fn_per_batch = setup_score_matching_loss_per_batch(energy_fn)
    gradient_fn_per_batch = None
    maximize = False
elif training_method == "CD1":
    loss_fn_per_batch = setup_score_matching_loss_per_batch(energy_fn)
    gradient_fn_per_batch = lambda flattened_args, batch: -CD1_gradient(energy_fn, batch, flattened_args, dt=0.001, D=1, key=jnp.random.PRNGKey(1535), num_noise_samples=1000)
    maximize = False
elif training_method == "MLE":
    loss_fn_per_batch = setup_MLE_loss_per_batch(energy_fn)
    gradient_fn_per_batch = setup_MLE_gradient_per_batch(energy_fn)
    maximize = True
else:
    raise ValueError(f"Unknown training method: {training_method}. Choose from 'SM', 'CD1', or 'MLE'.")


#####################################
## Optimization parameters and filenames

# Forward process parameters
n_time_steps = 5
t_forward = 1.0
sigma_forward = .5
forward_time_pts = jnp.exp(jnp.linspace(jnp.log(1e-9), jnp.log(t_forward), n_time_steps))
forward_time_pts = forward_time_pts.at[0].set(0.)
print('forward_time_pts', forward_time_pts)

# Optimization parameters
learning_rate = 10.
n_epochs = 100000//2
batch_size = 128
window_size=1000
tolerance=1e-8
patience=50


comment = "with_external_force"
optimization_folder = (f"{training_method}_"
           f"t_forward_{t_forward}_"
           f"n_timesteps_{n_time_steps}_"
           f"lr_{learning_rate}_"
           f"epochs_{n_epochs}_"
           f"batch_{batch_size}_"
           f"window_{window_size}_"
           f"tol_{tolerance}_"
           f"patience_{patience}_"
           f"{comment}")

# Setup output directories
base_dir = "out/problems"
problem_type_folder = "MNIST"
problem_folder = f"only_0_and_1_resol_8x8"  # Replace with your actual parameters

output_dir = os.path.join(base_dir, problem_type_folder, problem_folder,optimization_folder)

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)


#####################################
# Print final distribution of the forward

key = jr.PRNGKey(0)
key, subkey = jr.split(key)
samples_t = sample_forward_process(t_forward, n_samples, D=1, sigma_final=sigma_forward, samples0=samples_target, key=subkey)
plot_forward_marginals(samples_t, t_forward, sigma_forward, beta=1.0, path=output_dir, save_fig=True, fontsize=16)


#####################################
# Optimization
#####################################

# Setup optimization parameters
mask = jnp.ones(params_flattened_initial.shape[0])

# Plotting parameters
plot_steps = True  # Flag to control whether to plot during optimization
plot_slice = 1     # Plot every nth step


# Initialize storage for parameters at each time step
params_history_all_t = []
current_params = params_flattened_initial

# Loop over time points
for t_idx, t_curr in enumerate(forward_time_pts):
    print(f"Optimization for time {t_curr}")
    
    # Generate samples at current time
    key, subkey = jr.split(key)
    samples_t = sample_forward_process(
        t_curr, n_samples, D=1, sigma_final=sigma_forward, samples0=samples_target, key=subkey
    )
    
    # Perform optimization starting from previous best parameters
    key, subkey = jr.split(key)
    
    params_history, loss_history = run_optimization(
        loss_fn_per_batch=loss_fn_per_batch,
        params_initial=current_params,
        samples=samples_t,
        mask=mask,
        gradient_fn_per_batch=gradient_fn_per_batch,
        key=subkey,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        maximize=maximize,
        window_size=window_size,
        tolerance=tolerance,
        patience=patience
    )
    
    _, current_params, _ = get_best_params(params_history, loss_history, maximize=maximize)
    
    params_history_all_t.append(current_params)
    
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
                param_names=params_names
            )
        
# Convert params_history_all_t to array for easier analysis
params_history_all_t = jnp.array(params_history_all_t)
jnp.save(f"{output_dir}/params_history.npy", params_history_all_t)

print('Optimization complete; saved parameters to', output_dir)
