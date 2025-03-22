
#%%
import jax
from jax import flatten_util, vmap 
import jax.numpy as jnp
import jax.random as jr
import os
import matplotlib.pyplot as plt
from physical_diffusion_fns.helper_fns import sample_gaussian_mixture, normalize_samples, sample_forward_process, get_best_params, smooth_parameters, interpolate_parameters
from physical_diffusion_fns.learning_fns import setup_MLE_loss_per_batch, setup_MLE_gradient_per_batch, CD1_gradient, setup_score_matching_loss_per_batch, run_optimization
from physical_diffusion_fns.plotting_fns import plot_energy_and_distributions, plot_parameter_evolution, plot_forward_marginals, visualize_connectivity, plot_parameter_as_fn_of_time, visualize_connectivity_with_non_local_couplings
from physical_diffusion_fns.network_fns import setup_duffing_network_with_external_force_energy_fn, setup_overdamped_SDE, solve_SDE, create_2d_square_grid_connectivity, create_2d_square_grid_connectivity_with_diagonals, setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_energy_fn,setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_and_6th_order_energy_fn, setup_overdamped_ODE, solve_ODE, create_2d_square_grid_connectivity_with_long_range

jax.config.update("jax_enable_x64", True)


#%%
##################################### 
# Set parameters
##################################### 
std_of_added_noise = 0.01
additional_rescaling = 1.
# Forward process parameters
n_time_steps = 20 
t_forward = 2.5
sigma_forward = 1.
exponential_time_pts = True
if exponential_time_pts:
    forward_time_pts = jnp.exp(jnp.linspace(jnp.log(1e-4), jnp.log(t_forward), n_time_steps))
    forward_time_pts = forward_time_pts.at[0].set(0.)
else:
    forward_time_pts = jnp.linspace(0., t_forward, n_time_steps)
print('forward_time_pts', forward_time_pts)

# Optimization parameters
training_method = "SM"
learning_rate = 0.01
n_epochs = 1000000
batch_size = 128
window_size=100
tolerance=1e-6
patience=50

key_seed = 0

labels = [1,7]
resolution = (8,8)

with_diagonal_connections = True



#%%

#####################################
# Network size and topology
# N_osc = resolution[0]**2
# if with_diagonal_connections:
#     connectivity = create_2d_square_grid_connectivity_with_diagonals(grid_size=resolution[0])
# else:
#     connectivity = create_2d_square_grid_connectivity(grid_size=resolution[0])
# num_connections = connectivity.shape[0]

connectivity_w_non_local_couplings = create_2d_square_grid_connectivity_with_long_range(grid_size=4, n_neighbour_couplings=2)


#%%
visualize_connectivity_with_non_local_couplings(connectivity_w_non_local_couplings, grid_size_x=4, grid_size_y=4,)

# %%
connectivity_w_non_local_couplings
# %%
