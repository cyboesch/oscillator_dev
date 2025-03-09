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


##############################
# Generate samples from a Gaussian mixture


import numpy as np
# Load the saved .npy file
data_np = np.load("data/MNIST/MNIST_0_1_8x8pix/mnist_0_1_8x8pix.npy")

# Optionally convert to a JAX array
images_flat_raw = jnp.array(data_np)
n_samples = images_flat_raw.shape[0]
print("Data shape:", images_flat_raw.shape)

samples_target, mean_MNIST, std_MNIST = normalize_samples(images_flat_raw)


##############################
# Network size and topology


N_osc = 64
connectivity = create_2d_square_grid_connectivity(grid_size=8)
num_connections = connectivity.shape[0]
visualize_connectivity(connectivity, grid_size_x=8, grid_size_y=8)


##############################
# Define initial parameters


k_lin_0 = -1.*jnp.ones(N_osc)
k_duff_0 = jnp.ones(N_osc)
c_lin_0 = jnp.zeros(num_connections)
c_optomech_0 = jnp.zeros(num_connections)
biases_0 = jnp.zeros(N_osc)
params_initial = (k_lin_0, k_duff_0, c_lin_0, c_optomech_0, biases_0)
params_names = ['k_lin', 'k_duff', 'c_lin', 'c_optomech', 'biases']
params_flattened_initial, unflatten = flatten_util.ravel_pytree(params_initial)



##############################
# Bringe energy into correct form


energy_fn = setup_duffing_network_with_external_force_energy_fn(connectivity, unflatten)


##############################
# Plot initial energy landscape


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


##############################
## Optimization parameters and filenames

# Forward process parameters
n_time_steps = 50
t_forward = 1.0
sigma_forward = .5
forward_time_pts = jnp.exp(jnp.linspace(jnp.log(1e-9), jnp.log(t_forward), n_time_steps))
forward_time_pts = forward_time_pts.at[0].set(0.)
print('forward_time_pts', forward_time_pts)

# Optimization parameters
learning_rate = 100.
n_epochs = 100000//2
batch_size = 128
window_size=1000
tolerance=1e-8
patience=50


comment = "with_external_force_and_duff_constraint_positive_only"
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

##############################
# Load parameters
key = jr.PRNGKey(0)

params_history_all_t = jnp.load(f"{output_dir}/params_history.npy")

##############################
# Smoothen and interpolate parameter evolution


use_smoothed_params = False
if use_smoothed_params:
    # First smooth the parameters
    smoothed_params = smooth_parameters(params_history_all_t, window_lengths = [10] * params_history_all_t.shape[1], poly_orders = [3] * params_history_all_t.shape[1])
    # Get interpolator function
    params_interpolator = interpolate_parameters(smoothed_params, forward_time_pts)
else:
    # Get interpolator function
    params_interpolator = interpolate_parameters(params_history_all_t, forward_time_pts)

# Then interpolate the smoothed parameters
time_eval = jnp.linspace(forward_time_pts[0],forward_time_pts[-1],200)
# interpolated_params = interpolate_parameters(smoothed_params, forward_time_pts, time_dense)
interpolated_params = params_interpolator(time_eval)


plot_parameter_as_fn_of_time(params_names, forward_time_pts,forward_time_pts, params_history_all_t, params_interpolator, unflatten, N_osc) 


##############################
# Run reverse SDE:


forward_params = jnp.zeros_like(params_flattened_initial)
forward_params = forward_params.at[0:N_osc].set(1.)
params_interpolator_reverse_plus_linear = lambda t: 2*params_interpolator(t_forward - t) - 1/sigma_forward**2 * forward_params
# Setup SDE functions
drift_fn, diffusion_fn = setup_overdamped_SDE(energy_fn, params_interpolator_reverse_plus_linear, N_osc, time_dependent_parms=True)


t0 = 0.0
t1 = t_forward
ts = jnp.linspace(t0, t1, 100)
dt0 = 0.00000001

# Generate multiple initial states
n_trajectories = 10
key, subkey = jr.split(key)
initial_states = sample_forward_process(t_forward, n_trajectories, sigma_final=sigma_forward, D=1, samples0=samples_target, key=subkey)

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
        rtol=1e-3,
        atol=1e-6
    ),
    in_axes=(0, 0)
)


# Run SDE for all initial states at once
solutions = vectorized_solve_SDE(initial_states, keys_brownian)
all_trajectories = solutions.ys  # Shape: (n_trajectories, n_timesteps, N_osc)
np.save(f"{output_dir}/diffusion_trajectories.npy", all_trajectories)

# Save the final states as diffusion samples

final_states = all_trajectories[:, -1, :]  # Shape: (n_trajectories, N_osc)
final_states_rescaled = (final_states + mean_MNIST) * std_MNIST


images_generated = final_states_rescaled.reshape(-1, 8, 8)
# Plot a few examples
num_examples = n_trajectories  # number of examples to display
fig, axes = plt.subplots(1, num_examples, figsize=(15, 2))
for i in range(num_examples):
    # Remove channel dimension if it exists (i.e., converting 8x8x1 to 8x8)
    img = images_generated[i]
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    axes[i].imshow(np.array(img), cmap='gray')
    axes[i].axis('off')
plt.tight_layout()
# Save the figure
plt.savefig(f"{output_dir}/diffusion_samples_figure.png")
plt.show()




##############################
# Equillibirum Sampling

params_for_equillibrium_sampling = params_history_all_t[0,:]
# Setup SDE functions
drift_equilibrium_fn, diffusion_equilibrium_fn = setup_overdamped_SDE(energy_fn, params_for_equillibrium_sampling, N_osc, time_dependent_parms=False)


t0 = 0.0
t1 = 10
ts = jnp.linspace(t0, t1, 100)
dt0 = 0.00000001

key, subkey = jr.split(key)

# Select random initial states from samples_target
initial_states_indices = jr.randint(subkey, (n_trajectories,), 0, samples_target.shape[0])
initial_states = samples_target[initial_states_indices]

# Generate a key for each initial condition
key, subkey = jr.split(key)
keys_brownian = jr.split(subkey, n_trajectories)

# Vectorize solve_SDE across both initial states and keys
vectorized_solve_SDE_equilibrium = vmap(
    lambda init_state, key_b: solve_SDE(
        drift_equilibrium_fn, 
        diffusion_equilibrium_fn, 
        init_state,
        key_b, 
        t0, 
        t1, 
        ts.shape[0], 
        dt0,
        rtol=1e-6,
        atol=1e-9
    ),
    in_axes=(0, 0)
)

# Run SDE for all initial states at once
solutions = vectorized_solve_SDE_equilibrium(initial_states, keys_brownian)
all_trajectories_equilibrium = solutions.ys  # Shape: (n_trajectories, n_timesteps, N_osc)
np.save(f"{output_dir}/diffusion_trajectories_equilibrium.npy", all_trajectories_equilibrium)




