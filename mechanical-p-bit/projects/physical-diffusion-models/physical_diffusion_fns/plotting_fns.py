import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from physical_diffusion_fns.helper_fns import get_best_params, reformat_optimization_results

jax.config.update("jax_enable_x64", True)

########################################################################################
# Plotting energy, distributions, and samples
########################################################################################

def plot_energy_and_distributions(energy_fn, param_list, samples, 
                                x1_range=(-2, 2), 
                                x2_range=(-2, 2), 
                                n_points=100,
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

########################################################################################
# Plotting parameter evolution
########################################################################################

def plot_parameter_evolution(params_history, loss_history, time, time_index, unflatten, N_osc, param_names, slicing=10, figsize=(4, 14), title="Parameter Evolution", maximize=False, labels_on=True, save_fig=False, path=None):
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
        param_names: List of parameter names for plotting
    """
    
    best_loss, best_params, best_idx = get_best_params(params_history, loss_history, maximize)
    
    # Unpack parameter histories
    param_histories = reformat_optimization_results(params_history, loss_history, unflatten, slicing=slicing, maximize=maximize)

    print(f"Best loss: {loss_history[best_idx]:.4e}")
    print(f"Found at epoch: {best_idx * slicing}")

    # Create figure with (num_params + 1) x 1 subplots
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params + 1, 1, figsize=figsize)

    # Plot each parameter evolution
    for idx, param_name in enumerate(param_names):
        param_history = param_histories[idx]
        if labels_on and N_osc <= 2:
            for i in range(param_history.shape[1]):
                axes[idx].plot(param_history[:, i], label=f'{param_name}[{i}]')
        else:
            axes[idx].plot(param_history)
        axes[idx].set_title(f'Evolution of {param_name}')
        axes[idx].set_xlabel(f'Epoch (x {slicing})')
        axes[idx].set_ylabel(f'{param_name} Value')
        if labels_on and N_osc <= 2:
            axes[idx].legend()
        axes[idx].grid(True)

    # Plot loss evolution
    axes[-1].plot(loss_history, label='Loss')
    axes[-1].set_title(f'Evolution of Loss\nBest value: {best_loss:.3e}')
    axes[-1].set_xlabel(f'Epoch (x {slicing})')
    axes[-1].set_ylabel('Loss')
    axes[-1].legend()
    axes[-1].grid(True)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    if save_fig and path is not None:
        print(f"Saving figure to {path}")
        plt.savefig(path + f"/parameter_evolution_timeidx{time_index}_time{time}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return best_loss, best_params, best_idx

########################################################################################
# Plotting forward diffusion process
########################################################################################
def plot_forward_marginals(samples_t, t_forward, sigma_final, beta=1.0, path=None, save_fig=False, fontsize=16):
    """
    Plot marginal distributions of samples at forward time t.
    
    Args:
        samples_t: Array of samples shape (n_samples, n_dim)
        t_forward: Forward time value
        sigma_final: Final sigma value for Gaussian comparison
        path: Path to save figure (default: None)
        save_fig: Boolean flag to save figure (default: False)
        fontsize: Base font size for the plot (default: 16)
    """
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Get number of dimensions
    n_dim = samples_t.shape[1]
    show_labels = n_dim <= 2
    
    # Plot marginal distributions for all dimensions
    colors = plt.cm.tab10(jnp.linspace(0, 1, n_dim))  # Get distinct colors
    for i in range(n_dim):
        label = f'Dimension {i} distribution' if show_labels else None
        plt.hist(samples_t[:, i], bins=50, density=True, alpha=0.3, color=colors[i], label=label)

    # Add Gaussian N(0,sigma) for comparison
    x = jnp.linspace(-3, 3, 1000)  # Adjust range as needed
    gaussian_pdf = (1 / jnp.sqrt(2 * jnp.pi*sigma_final**2)) * jnp.exp(-0.5 * x**2/sigma_final**2)
    plt.plot(x, gaussian_pdf, 'r--', linewidth=2, label=r'Gaussian N(0,$\sigma$)')

    plt.title(f'Marginal Distributions at t = {t_forward}, $\sigma$ = {sigma_final:.2f}', fontsize=fontsize+2)
    plt.xlabel('Value', fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    if show_labels:
        plt.legend(fontsize=fontsize-2)
    
    # Set tick label sizes
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)

    plt.tight_layout()
    
    if save_fig and path is not None:
        print(f"Saving figure to {path}")
        plt.savefig(path + f"/final_forward_distribution_sigma_{sigma_final:.2f}_beta_{beta:.2f}.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
########################################################################################
# Plotting parameter as function of time
########################################################################################  
def plot_parameter_as_fn_of_time(params_names, forward_time_pts, time_eval, 
                                 params_history_all_t, params_interpolator, 
                                 unflatten, N_osc, log_scale=True):
    # --- Reconstruct Historical Data ---
    first_params = unflatten(params_history_all_t[0])
    n_groups = len(first_params)
    n_time = len(forward_time_pts)
    
    # Initialize a list of JAX arrays for each parameter group.
    hist_arrays = [jnp.zeros((n_time,) + jnp.array(param).shape) for param in first_params]
    
    # Fill in the history using JAX's immutable update semantics.
    for i in range(n_time):
        unflattened = unflatten(params_history_all_t[i])
        for j, param in enumerate(unflattened):
            hist_arrays[j] = hist_arrays[j].at[i].set(param)
    
    # --- Reconstruct Interpolated Data ---
    n_time_interp = len(time_eval)
    interp_arrays = [jnp.zeros((n_time_interp,) + jnp.array(param).shape) for param in first_params]
    
    for i in range(n_time_interp):
        unflattened = unflatten(params_interpolator(time_eval[i]))
        for j, param in enumerate(unflattened):
            interp_arrays[j] = interp_arrays[j].at[i].set(param)

    # --- Plotting ---
    fig, axes = plt.subplots(n_groups, 1, figsize=(6, 3 * n_groups))
    if n_groups == 1:
        axes = [axes]
    
    fig.suptitle('Parameter Evolution Over Time: Raw and Interpolated', fontsize=16)
    
    # Extended list of colors to cycle through
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    
    # Loop over each parameter group
    for j in range(n_groups):
        ax = axes[j]
        param_shape = jnp.array(first_params[j]).shape
        
        # If parameter is defined per oscillator (its first dimension equals N_osc), plot each oscillator.
        if len(param_shape) > 0 and param_shape[0] == N_osc:
            for osc in range(N_osc):
                color = colors[osc % len(colors)]
                ax.plot(forward_time_pts, hist_arrays[j][:, osc],
                        color + '-', alpha=0.3)
                ax.plot(time_eval, interp_arrays[j][:, osc],
                        color + '-', linewidth=2)
        else:
            # Otherwise, plot a single curve.
            ax.plot(forward_time_pts, hist_arrays[j],
                    colors[0] + '-', alpha=0.3)
            ax.plot(time_eval, interp_arrays[j],
                    colors[0] + '-', linewidth=2)
        
        ax.set_title(f'{params_names[j]} Evolution')
        ax.set_xlabel('Time')
        if log_scale:
            ax.set_xscale('log')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to reduce gap
    plt.show()

########################################################################################
# Plotting connectivity
########################################################################################   
def visualize_connectivity(connectivity, grid_size_x=8, grid_size_y=8):
    """
    Visualize the connectivity pattern of the grid.
    
    Args:
        connectivity (jnp.ndarray): Connectivity matrix
        grid_size (int): Size of the square grid
    """    
    plt.figure(figsize=(3, 3))
    
    # Plot oscillators
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            plt.plot(j, i, 'ko')
    
    # Plot connections
    for conn in connectivity:
        i1, j1 = divmod(conn[0], grid_size_y)
        i2, j2 = divmod(conn[1], grid_size_x)
        plt.plot([j1, j2], [i1, i2], 'b-', alpha=0.3)
    
    plt.grid(True)
    plt.axis('equal')
    plt.title(f'2D Grid Connectivity ({grid_size_x}x{grid_size_y})')
    plt.show()