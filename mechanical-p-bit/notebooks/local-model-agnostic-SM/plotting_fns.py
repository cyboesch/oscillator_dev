import jax
from jax import flatten_util, vmap, grad 
import jax.numpy as jnp
import jax.random as jr
import diffrax
from diffrax import ControlTerm, MultiTerm, ODETerm
import matplotlib.pyplot as plt
from functools import partial
import optax
from helper_fns_general import sample_gaussian_mixture, normalize_samples, CD1_gradient, setup_score_matching_loss_per_batch, run_optimization
jax.config.update("jax_enable_x64", True)


def plot_energy_and_distributions(energy_fn, param_list, samples, 
                                x1_range=(-2, 2), x2_range=(-2, 2), n_points=100,
                                titles=None, 
                                figsize=None,
                                fontsize=12,
                                sample_stride=10,
                                suptitle="Energy Landscapes and Probability Distributions"):
    """
    Plot energy landscapes and probability distributions for multiple parameter sets.
    
    Args:
        energy_fn: Function that takes (x, params) and returns energy
        param_list: List of parameter sets to evaluate
        samples: Training samples to plot (shape: [n_samples, 2])
        x1_range: Tuple of (min, max) for x₁ axis (default: (-2, 2))
        x2_range: Tuple of (min, max) for x₂ axis (default: (-2, 2))
        n_points: Number of points in each dimension for grid (default: 100)
        titles: List of titles for each row (default: None)
        figsize: Figure size as (width, height) tuple (default: None)
        fontsize: Base font size for plots (default: 12)
        suptitle: Super title for the entire figure (default: "Energy Landscapes...")
    """
    n_rows = len(param_list)
    if figsize is None:
        figsize = (16, 8 * n_rows)
    
    if titles is None:
        titles = [f"Parameter Set {i+1}" for i in range(n_rows)]
    
    # Create meshgrid
    x1 = jnp.linspace(x1_range[0], x1_range[1], n_points)
    x2 = jnp.linspace(x2_range[0], x2_range[1], n_points)
    X1, X2 = jnp.meshgrid(x1, x2)
    
    # Create figure with n_rows x 2 subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Get grid spacing for normalization
    dx = x1[1] - x1[0]
    dy = x2[1] - x2[0]
    
    for row, (params, title) in enumerate(zip(param_list, titles)):
        # Compute energy landscape
        energy = jnp.zeros_like(X1)
        for i in range(n_points):
            for j in range(n_points):
                x = jnp.array([X1[i,j], X2[i,j]])
                energy = energy.at[i,j].set(energy_fn(x, params))
                
        # Compute probability distribution
        prob = jnp.exp(-energy)
        prob = prob / (jnp.sum(prob) * dx * dy)  # Normalize
        
        # Plot energy landscape
        contour1 = axes[row,0].contourf(X1, X2, energy, levels=20, cmap='viridis', alpha=0.7)
        plt.colorbar(contour1, ax=axes[row,0], label='Energy')
        axes[row,0].scatter(samples[::sample_stride, 0], samples[::sample_stride, 1], c='black', s=1, alpha=0.9, 
                          label='Training Samples')
        axes[row,0].set_xlabel('x₁', fontsize=fontsize)
        axes[row,0].set_ylabel('x₂', fontsize=fontsize)
        axes[row,0].set_title(f'{title}\nEnergy Landscape', fontsize=fontsize+2)
        axes[row,0].legend(fontsize=fontsize-2)
        axes[row,0].tick_params(labelsize=fontsize-2)
        
        # Plot probability distribution
        contour2 = axes[row,1].contourf(X1, X2, prob, levels=20, cmap='viridis', alpha=0.7)
        plt.colorbar(contour2, ax=axes[row,1], label='Probability Density')
        axes[row,1].scatter(samples[::sample_stride, 0], samples[::sample_stride, 1], c='black', s=1, alpha=0.9, 
                          label='Training Samples')
        axes[row,1].set_xlabel('x₁', fontsize=fontsize)
        axes[row,1].set_ylabel('x₂', fontsize=fontsize)
        axes[row,1].set_title(f'{title}\nProbability Distribution', fontsize=fontsize+2)
        axes[row,1].legend(fontsize=fontsize-2)
        axes[row,1].tick_params(labelsize=fontsize-2)
        
        # Print normalization check for each distribution
        total_prob = jnp.sum(prob) * dx * dy
        print(f"{title} probability distribution total (should be ≈ 1): {total_prob:.6f}")
    
    # plt.tight_layout()
    if suptitle:
        plt.suptitle(suptitle, y=1.02, fontsize=fontsize+4)
    plt.show()
    
    
def plot_parameter_evolution(params_history, loss_history, unflatten, N_osc, slicing=10, figsize=(6, 6), title="Parameter Evolution"):
    """
    Plot the evolution of parameters during optimization.
    
    Args:
        params_history: History of flattened parameters
        loss_history: History of loss values
        unflatten: Function to unflatten parameters
        N_osc: Number of oscillators
        slicing: Plot every nth point (default: 10)
        figsize: Figure size as (width, height) tuple (default: (6, 6))
        title: Super title for the plot (default: "Parameter Evolution")
    
    Returns:
        tuple: Best parameters (k_lin_best, k_duff_best, c_lin_best, c_optomech_best)
    """
    # Convert histories to arrays for plotting
    params_history = jnp.array(params_history[::slicing])
    k_lin_history = jnp.zeros((len(params_history), N_osc))
    k_duff_history = jnp.zeros((len(params_history), N_osc))
    c_lin_history = jnp.zeros((len(params_history), 1))
    c_optomech_history = jnp.zeros((len(params_history), 1))

    for i in range(len(params_history)):
        unflattened_params = unflatten(params_history[i])
        k_lin, k_duff, c_lin, c_optomech = unflattened_params
        k_lin_history = k_lin_history.at[i].set(k_lin)
        k_duff_history = k_duff_history.at[i].set(k_duff)
        c_lin_history = c_lin_history.at[i].set(c_lin)
        c_optomech_history = c_optomech_history.at[i].set(c_optomech)

    loss_history = jnp.array(loss_history[::slicing])

    # Find index of lowest loss
    best_idx = jnp.argmin(loss_history)

    # Get parameters corresponding to lowest loss
    k_lin_best = k_lin_history[best_idx]
    k_duff_best = k_duff_history[best_idx]
    c_lin_best = c_lin_history[best_idx]
    c_optomech_best = c_optomech_history[best_idx]

    print(f"Best loss: {loss_history[best_idx]:.4e}")
    print(f"Found at epoch: {best_idx * slicing}")

    # Create figure with 4 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot k_lin evolution
    for i in range(N_osc):
        axes[0, 0].plot(k_lin_history[:, i], label=f'k_lin[{i}]')
    axes[0, 0].set_title(f'Evolution of k_lin\nBest values:\n' + 
                         '\n'.join([f'k_lin[{i}] = {k_lin_best[i]:.3f}' 
                                   for i in range(N_osc)]))
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Linear Strength')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot k_duff evolution
    for i in range(N_osc):
        axes[0, 1].plot(k_duff_history[:, i], label=f'k_duff[{i}]')
    axes[0, 1].set_title(f'Evolution of k_duff\nBest values:\n' + 
                         '\n'.join([f'k_duff[{i}] = {k_duff_best[i]:.3f}' 
                                   for i in range(N_osc)]))
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Duffing Strength')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot c_lin and c_optomech evolution
    axes[1, 0].plot(c_lin_history, label='c_lin')
    axes[1, 0].plot(c_optomech_history, label='c_optomech')
    axes[1, 0].set_title(f'Evolution of Coupling Parameters\nBest values:\n' +
                         f'c_lin = {c_lin_best[0]:.3f}\n' +
                         f'c_optomech = {c_optomech_best[0]:.3f}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Coupling Strength')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot loss evolution
    axes[1, 1].plot(loss_history, label='Loss')
    axes[1, 1].set_title(f'Evolution of Loss\nBest value: {loss_history[best_idx]:.3e}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

    return k_lin_best, k_duff_best, c_lin_best, c_optomech_best