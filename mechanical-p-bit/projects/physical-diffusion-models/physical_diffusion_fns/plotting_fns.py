import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


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

def reformat_optimization_results(params_history, loss_history, unflatten, N_osc, N_connections, slicing=1, maximize=False):
    # Convert histories to arrays for plotting
    params_history = jnp.array(params_history[::slicing])
    k_lin_history = jnp.zeros((len(params_history), N_osc))
    k_duff_history = jnp.zeros((len(params_history), N_osc))
    c_lin_history = jnp.zeros((len(params_history), N_connections))
    c_optomech_history = jnp.zeros((len(params_history), N_connections))

    for i in range(len(params_history)):
        unflattened_params = unflatten(params_history[i])
        k_lin, k_duff, c_lin, c_optomech = unflattened_params
        k_lin_history = k_lin_history.at[i].set(k_lin)
        k_duff_history = k_duff_history.at[i].set(k_duff)
        c_lin_history = c_lin_history.at[i].set(c_lin)
        c_optomech_history = c_optomech_history.at[i].set(c_optomech)

    loss_history = jnp.array(loss_history[::slicing])

    return k_lin_history, k_duff_history, c_lin_history, c_optomech_history, loss_history

def get_best_params(params_history, loss_history, maximize=False):
    loss_history = jnp.array(loss_history)
    # Find index of lowest loss
    if maximize:
        best_idx = jnp.argmax(loss_history)
    else:
        best_idx = jnp.argmin(loss_history)
    return loss_history[best_idx], params_history[best_idx], best_idx
    
def plot_parameter_evolution(params_history, loss_history, time, time_index, unflatten, N_osc, N_connections=1, slicing=10, figsize=(6, 6), title="Parameter Evolution", maximize=False, labels_on=True, save_fig=False, path=None):
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
    """
    
    best_loss, best_params, best_idx = get_best_params(params_history, loss_history, maximize)
    
    k_lin_history, k_duff_history, c_lin_history, c_optomech_history, loss_history = reformat_optimization_results(params_history, loss_history, unflatten, N_osc, N_connections, slicing=slicing, maximize=maximize)

    print(f"Best loss: {loss_history[best_idx]:.4e}")
    print(f"Found at epoch: {best_idx * slicing}")

    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Plot k_lin evolution
    if labels_on:
        for i in range(N_osc):
            axes[0, 0].plot(k_lin_history[:, i], label=f'k_lin[{i}]')
    else:
        axes[0, 0].plot(k_lin_history)
    axes[0, 0].set_title(f'Evolution of k_lin')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Linear Strength')
    if labels_on:
        axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot k_duff evolution
    if labels_on:
        for i in range(N_osc):
            axes[0, 1].plot(k_duff_history[:, i], label=f'k_duff[{i}]')
    else:
        axes[0, 1].plot(k_duff_history)
    axes[0, 1].set_title(f'Evolution of k_duff')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Duffing Strength')
    if labels_on:
        axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot c_lin evolution
    if labels_on:
        for i in range(N_connections):
            axes[1, 0].plot(c_lin_history[:, i], label=f'c_lin[{i}]')
    else:
        axes[1, 0].plot(c_lin_history)
    axes[1, 0].set_title(f'Evolution of Linear Coupling')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Linear Coupling Strength')
    if labels_on:
        axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot c_optomech evolution
    if labels_on:
        for i in range(N_connections):
            axes[1, 1].plot(c_optomech_history[:, i], label=f'c_optomech[{i}]')
    else:
        axes[1, 1].plot(c_optomech_history)
    axes[1, 1].set_title(f'Evolution of Optomechanical Coupling')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Optomechanical Coupling Strength')
    if labels_on:
        axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plot loss evolution
    axes[2, 0].plot(loss_history, label='Loss')
    axes[2, 0].set_title(f'Evolution of Loss\nBest value: {best_loss:.3e}')
    axes[2, 0].set_xlabel(f'Epoch (x {slicing})')
    axes[2, 0].set_ylabel('Loss')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Hide the empty subplot
    axes[2, 1].set_visible(False)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    if save_fig and path is not None:
        print(f"Saving figure to {path}")
        plt.savefig(path + f"/parameter_evolution_timeidx{time_index}_time{time}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return best_loss, best_params, best_idx


def plot_forward_marginals(samples_t, t_forward, sigma_final, beta=1.0, path=None, save_fig=False, fontsize=16):
    """
    Plot marginal distributions of samples at forward time t.
    
    Args:
        samples_t: Array of samples shape (n_samples, 2)
        t_forward: Forward time value
        sigma_final: Final sigma value for Gaussian comparison
        path: Path to save figure (default: None)
        save_fig: Boolean flag to save figure (default: False)
        fontsize: Base font size for the plot (default: 16)
    """
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot marginal distributions
    plt.hist(samples_t[:, 0], bins=50, density=True, alpha=0.7, label='X distribution')
    plt.hist(samples_t[:, 1], bins=50, density=True, alpha=0.7, label='Y distribution')

    # Add Gaussian N(0,sigma) for comparison
    x = jnp.linspace(-1, 1, 1000)  # Adjust range as needed
    gaussian_pdf = (1 / jnp.sqrt(2 * jnp.pi*sigma_final**2)) * jnp.exp(-0.5 * x**2/sigma_final**2)
    plt.plot(x, gaussian_pdf, 'r--', linewidth=2, label=r'Gaussian N(0,$\sigma$)')

    plt.title(f'Marginal Distributions at t = {t_forward}, $\sigma$ = {sigma_final:.2f}', fontsize=fontsize+2)
    plt.xlabel('Value', fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    plt.legend(fontsize=fontsize-2)
    
    # Set tick label sizes
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)

    plt.tight_layout()
    
    if save_fig and path is not None:
        print(f"Saving figure to {path}")
        plt.savefig(path + f"/final_forward_distribution_sigma_{sigma_final:.2f}_beta_{beta:.2f}.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    
    
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