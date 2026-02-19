import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from physical_diffusion_fns.helper_fns import get_best_params, reformat_optimization_results


import math
import matplotlib
import numpy as np
# Use non-interactive backend for headless environments
# matplotlib.use('Agg')
import shutil

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


def plot_parameter_evolution(params_history, loss_history, best_params, best_loss, best_idx, time, time_index, unflatten,
                             N_osc, param_names, figsize=(6, 14), title="Parameter Evolution",
                             maximize=False, labels_on=True, save_fig=False, path=None,
                             plot_show=False, small_fig=False):
    """
    Plot the evolution of parameters during optimization at 100 fixed time steps.

    Args:
        params_history: History of flattened parameters
        loss_history: History of loss values
        time: Array of time points
        time_index: Index of current time for filename
        unflatten: Function to unflatten parameters
        N_osc: Number of oscillators (for labeling)
        param_names: List of parameter names for plotting
        figsize: Base figure size (width, height)
        title: Super title for the plot
        maximize: If True, best loss is maximum instead of minimum
        labels_on: Whether to show labels for N_osc <= 2
        save_fig: Boolean to save figure
        path: Path to save figure
        plot_show: Whether to call plt.show()
        small_fig: If True, scale figsize down by 0.2 for quick testing
    """
    # Scale down figure if requested
    if small_fig:
        fig_w, fig_h = figsize
        figsize = (fig_w * 0.2, fig_h * 0.2)

    # Determine best

    # Unpack full parameter histories (no slicing)
    param_histories = reformat_optimization_results(params_history, loss_history,
                                                    unflatten, slicing=1, maximize=maximize)

    # Subsample indices for up to 100 equally‐spaced steps (inclusive endpoints)
    n_points = param_histories[0].shape[0]
    num_samples = min(100, n_points)
    # Generate equally spaced floats, round to nearest int, then unique and sort
    raw_idxs = np.linspace(0, n_points - 1, num_samples)
    idxs = np.unique(np.round(raw_idxs).astype(int))

    print(f"Best loss: {best_loss:.4e}")
    print(f"Found at epoch: {best_idx}")

    # Create subplots: one per parameter plus loss plus zoomed loss
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params + 2, 1, figsize=figsize)

    # Plot each parameter evolution at fixed indices
    for idx, param_name in enumerate(param_names):
        full_hist = param_histories[idx]
        hist = full_hist[idxs]
        ax = axes[idx]
        if labels_on and N_osc <= 2:
            for j in range(hist.shape[1]):
                ax.plot(idxs, hist[:, j], label=f'{param_name}[{j}]')
            ax.legend()
        else:
            ax.plot(idxs, hist)
        ax.set_title(f'Evolution of {param_name}')
        ax.set_xlabel('Step index')
        ax.set_ylabel(param_name)
        ax.grid(True)

    # Plot subsampled loss
    loss_plot = np.array(loss_history)[idxs]
    ax_loss = axes[-2]
    ax_loss.plot(idxs, loss_plot, label='Loss')
    ax_loss.set_title(f'Evolution of Loss\nBest value: {best_loss:.3e}')
    ax_loss.set_xlabel('Step index')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)

    # Plot last 30 steps of loss evolution (zoom-in)
    ax_loss_zoom = axes[-1]
    loss_array = np.array(loss_history)
    n_total = len(loss_array)
    start_idx = max(0, n_total - 30)
    last_30_indices = np.arange(start_idx, n_total)
    last_30_loss = loss_array[start_idx:]
    
    ax_loss_zoom.plot(last_30_indices, last_30_loss, label='Loss (Last 30 steps)', color='red')
    ax_loss_zoom.set_title(f'Loss Evolution - Last 30 Steps\nFinal value: {loss_array[-1]:.3e}')
    ax_loss_zoom.set_xlabel('Step index')
    ax_loss_zoom.set_ylabel('Loss')
    ax_loss_zoom.legend()
    ax_loss_zoom.grid(True)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)

    if save_fig and path:
        filename = f"{path}/parameter_opt_evolution_timeidx_{time_index}_time_{time:.5f}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if plot_show:
        plt.show()
    plt.close(fig)
    return best_loss, best_params, best_idx



########################################################################################
# Plotting forward diffusion process
########################################################################################
def plot_forward_marginals(samples_t, t_forward, sigma_final, Temp=1.0, beta=1.0,
                            path=None, save_fig=False, fontsize=16, plot_show=False,
                            max_plot_points=None, random_seed=None):
    """
    Plot marginal distributions of samples at forward time t using individual subplots.

    Args:
        samples_t: Array of samples shape (n_samples, n_dim)
        t_forward: Forward time value
        sigma_final: Final sigma value for Gaussian comparison
        path: Path to save figure (default: None)
        save_fig: Boolean flag to save figure (default: False)
        fontsize: Base font size for the plot (default: 16)
        plot_show: Whether to call plt.show() (default: False)
        max_plot_points: Maximum number of points to plot per marginal (subsamples if larger)
        random_seed: Seed for subsampling reproducibility (default: None)
        small_fig: If True, use a very small figure size for testing
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_dim = samples_t.shape[1]
    ncols = int(math.ceil(math.sqrt(n_dim)))
    nrows = int(math.ceil(n_dim / ncols))


    figsize = (2 * ncols, 2 * nrows)

    fig = plt.figure(figsize=figsize)

    # Prepare Gaussian overlay data using numpy to avoid JAX sync issues
    x = np.linspace(-3 * math.sqrt(Temp) * sigma_final,
                     3 * math.sqrt(Temp) * sigma_final, 1000)
    gaussian_pdf = (1 / math.sqrt(2 * math.pi * Temp * sigma_final**2)) * \
                   np.exp(-0.5 * x**2 / (Temp * sigma_final**2))

    for i in range(n_dim):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        data = np.array(samples_t[:, i])
        if max_plot_points is not None and data.size > max_plot_points:
            idx = np.random.choice(data.size, size=max_plot_points, replace=False)
            data_to_plot = data[idx]
        else:
            data_to_plot = data

        ax.hist(data_to_plot, bins=50, density=True, alpha=0.3)
        ax.plot(x, gaussian_pdf, 'r--', linewidth=2)
        ax.set_title(f'Dimension {i}', fontsize=fontsize)
        ax.set_xlabel('Value', fontsize=fontsize - 2)
        ax.set_ylabel('Density', fontsize=fontsize - 2)
        ax.tick_params(labelsize=fontsize - 2)

    fig.suptitle(f'Marginal Distributions at t = {t_forward}, σ = {sigma_final:.2f}',
                 fontsize=fontsize + 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_fig and path:
        plt.savefig(f"{path}/forward_marginals_grid_sigma_{sigma_final:.2f}_beta_{beta:.2f}.png", 
                    dpi=300, bbox_inches='tight')
    if plot_show:
        plt.show()
    plt.close(fig)

########################################################################################
# Plotting parameter as function of time
########################################################################################  
def plot_parameter_as_fn_of_time(params_names, forward_time_pts, time_eval, 
                                 params_history_all_t, params_interpolator, 
                                 unflatten, N_osc, log_scale=True, save_fig=False, path=None, plot_show=False):
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
    if save_fig and path is not None:
        plt.savefig(f"{path}/parameter_evolution_over_time.png", dpi=300, bbox_inches='tight')
    if plot_show:
        plt.show()
    else:
        plt.close()
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
    plt.close()


def visualize_connectivity_with_non_local_couplings(connectivity, grid_size_x=8, grid_size_y=8, n_neighbour_couplings=1, save_fig=False, path=None, plot_show=False):
    """
    Visualize the connectivity of a square grid.
    
    Nodes are arranged in a grid (row-major ordering). Each connection is drawn using a color
    that depends on its "shell" number, defined as:
    
         shell = max(|row2 - row1|, |col2 - col1|)
    
    - Shell 1 (immediate neighbors: horizontal, vertical, diagonal) are drawn as straight lines.
    - For shells 2 and higher, a curved line (quadratic Bézier curve) is drawn.
    
    Different colors are assigned to each shell. For example:
        shell 1: blue, shell 2: red, shell 3: green, etc.
    
    Args:
        connectivity (jnp.ndarray): Array of shape (num_connections, 2) with each row [node1, node2].
        grid_size_x (int): Number of columns.
        grid_size_y (int): Number of rows.
        n_neighbour_couplings (int): Maximum shell (range) considered.
    """
    plt.figure(figsize=(6, 6))
    
    # Plot the nodes.
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            plt.plot(j, i, 'ko', markersize=4)
    
    # Define a list of colors for each shell.
    # Extend this list if you need more shells.
    shell_colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta", "brown"]
    
    # Iterate over each connection.
    for conn in connectivity:
        node1, node2 = int(conn[0]), int(conn[1])
        # Recover grid coordinates (assumes row-major ordering).
        row1, col1 = divmod(node1, grid_size_x)
        row2, col2 = divmod(node2, grid_size_x)
        # Determine the shell number based on the maximum coordinate difference.
        shell = max(abs(row2 - row1), abs(col2 - col1))
        
        # Get the color based on the shell (using the list, with shell 1 -> index 0, etc.)
        if shell - 1 < len(shell_colors):
            color = shell_colors[shell - 1]
        else:
            color = "black"  # fallback if shell number exceeds our defined colors
        
        if shell == 1:
            # Immediate neighbors: draw a straight line.
            plt.plot([col1, col2], [row1, row2], color=color, alpha=0.7, linewidth=1.5)
        else:
            # For longer-range couplings, draw a curved line.
            # Compute the midpoint.
            mid_x = (col1 + col2) / 2.0
            mid_y = (row1 + row2) / 2.0
            # Compute the perpendicular direction.
            dx = col2 - col1
            dy = row2 - row1
            norm = jnp.sqrt(dx**2 + dy**2)
            if norm == 0:
                perp_x, perp_y = 0, 0
            else:
                perp_x, perp_y = -dy / norm, dx / norm
            # Bend factor increases with the shell (adjust multiplier as desired).
            bend = 0.2 * shell
            cp_x = mid_x + bend * norm * perp_x
            cp_y = mid_y + bend * norm * perp_y
            
            # Generate points along a quadratic Bézier curve.
            t = jnp.linspace(0, 1, 50)
            curve_x = (1 - t)**2 * col1 + 2 * (1 - t) * t * cp_x + t**2 * col2
            curve_y = (1 - t)**2 * row1 + 2 * (1 - t) * t * cp_y + t**2 * row2
            plt.plot(curve_x, curve_y, color=color, alpha=0.7, linewidth=1.5)
    
    plt.grid(True)
    plt.axis('equal')
    plt.title(f'2D Grid Connectivity ({grid_size_x}x{grid_size_y})\nShells 1 to {n_neighbour_couplings}')
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    if plot_show:
        plt.show()
    if save_fig and path is not None:
        plt.savefig(path + f"/connectivity_n_neighbour_couplings_{n_neighbour_couplings}.png", dpi=300, bbox_inches='tight')
    plt.close()
    


from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Print-friendly color schemes with high contrast
# Option 1: Grayscale with high contrast (best for B&W printing)
grayscale_cmap = LinearSegmentedColormap.from_list("grayscale_contrast", 
                                                   ["white", "lightgray", "gray", "darkgray", "black"])

# Option 2: Blue-white-red diverging (good contrast, colorblind friendly)
bwr_cmap = LinearSegmentedColormap.from_list("blue_white_red", 
                                             ["darkblue", "blue", "lightblue", "white", 
                                              "lightcoral", "red", "darkred"])

# Option 3: Viridis-like but with higher contrast
viridis_contrast = LinearSegmentedColormap.from_list("viridis_contrast",
                                                     ["#440154", "#31688e", "#35b779", "#fde725"])

# Option 4: Custom high-contrast scheme (purple to yellow)
purple_yellow = LinearSegmentedColormap.from_list("purple_yellow",
                                                  ["#2d1b69", "#5d2a8a", "#8e44ad", "#c39bd3", 
                                                   "#f7dc6f", "#f4d03f", "#f1c40f"])

def plot_sampels_energy_marginals_for_SGM_vs_ES(
    energy_fn,
    params_reverse,
    params_equil,
    reverse_samples,
    sde_samples_direct,
    weights, means, covs,       # GMM params
    *,                           # everything below here must be named
    x1_range=(-1.5, 1.5),
    x2_range=(-1.5, 1.5),
    n_points=200,
    num_bins=50,
    sample_stride=10,
    figsize=(7, 4),
    fontsize=10,
    label_fontsize=14,
    iso_levels=5,
    colormap='grayscale',        # NEW: colormap option
    contour_linewidth=2.0,       # NEW: thicker contour lines
    scatter_size=3,              # NEW: larger scatter points
    scatter_alpha=0.8,           # NEW: higher alpha for visibility
    plot_folder=None,
    n_trajectories=None,
    atol=None,
    rtol=None,
    Temp=None,
):
    """
    Print-friendly version with improved color schemes and visibility.
    
    colormap options:
    - 'grayscale': High-contrast grayscale (best for B&W printing)
    - 'bwr': Blue-white-red diverging
    - 'viridis_contrast': High-contrast viridis-like
    - 'purple_yellow': Purple to yellow high contrast
    - 'plasma': Original plasma (for comparison)
    """
    
    # Select colormap
    cmap_dict = {
        'grayscale': grayscale_cmap,
        'bwr': bwr_cmap,
        'viridis_contrast': viridis_contrast,
        'purple_yellow': purple_yellow,
        'plasma': 'plasma'
    }
    selected_cmap = cmap_dict.get(colormap, grayscale_cmap)
    
    # 1) Grid
    x1 = jnp.linspace(x1_range[0], x1_range[1], n_points)
    x2 = jnp.linspace(x2_range[0], x2_range[1], n_points)
    X1, X2 = jnp.meshgrid(x1, x2)
    dx, dy = x1[1] - x1[0], x2[1] - x2[0]

    # 2) Exact marginals for histograms
    x_lin = jnp.linspace(x1_range[0], x1_range[1], 1000)
    marginal_x1 = sum(
        w * jnp.exp(-0.5*((x_lin - m[0])**2)/C[0,0]) / jnp.sqrt(2*jnp.pi*C[0,0])
        for w,m,C in zip(weights, means, covs)
    )
    marginal_x2 = sum(
        w * jnp.exp(-0.5*((x_lin - m[1])**2)/C[1,1]) / jnp.sqrt(2*jnp.pi*C[1,1])
        for w,m,C in zip(weights, means, covs)
    )

    # 3) True 2D GMM density on grid
    true_prob = jnp.zeros_like(X1)
    for w, m, C in zip(weights, means, covs):
        diff = jnp.stack([X1 - m[0], X2 - m[1]], axis=-1)
        invC = jnp.linalg.inv(C)
        exponent = jnp.einsum('...i,ij,...j->...', diff, invC, diff)
        norm = jnp.sqrt((2*jnp.pi)**2 * jnp.linalg.det(C))
        true_prob += w * jnp.exp(-0.5 * exponent) / norm

    # 4) Compute E & model P for both methods
    E_list, P_list = [], []
    for params in (params_reverse, params_equil):
        E = jnp.zeros_like(X1)
        for i in range(n_points):
            for j in range(n_points):
                x = jnp.array([X1[i,j], X2[i,j]])
                E = E.at[i,j].set(energy_fn(x, params))
        P = jnp.exp(-E)
        P /= (jnp.sum(P) * dx * dy)
        E_list.append(E)
        P_list.append(P)

    # 5) Figure setup with improved styling
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    labels = ['(a)','(b)','(c)','(d)','(e)','(f)']
    for ax, lab in zip(axes.flatten(), labels):
        ax.text(0.02, 0.95, lab,
                transform=ax.transAxes,
                fontsize=label_fontsize,
                fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    row_titles = ['Reverse Trajectory', 'Equilibrium Sampling']

    # 6) Plot rows with improved visibility
    for row, (title, samples) in enumerate(zip(row_titles,
                                              (reverse_samples, sde_samples_direct))):
        # — combined marginals with thicker lines
        ax = axes[row,0]
        ax.hist(samples[:,0], bins=num_bins, density=True,
                alpha=0.6, color='blue', label='x$_1$', edgecolor='darkblue', linewidth=0.5)
        ax.hist(samples[:,1], bins=num_bins, density=True,
                alpha=0.6, color='red',  label='x$_2$', edgecolor='darkred', linewidth=0.5)
        ax.plot(x_lin, marginal_x1, '--', color='darkblue', lw=3, label='Exact $p(x_1)$')
        ax.plot(x_lin, marginal_x2, '--', color='darkred', lw=3, label='Exact $p(x_2)$')
        ax.set_ylim(0, 13)
        ax.set_xlabel('x$_1$, x$_2$', fontsize=fontsize)
        ax.set_ylabel('Density', fontsize=fontsize)
        ax.legend(fontsize=fontsize-2, loc='upper right')
        ax.tick_params(labelsize=fontsize-2)
        ax.grid(True, alpha=0.3)

        # — energy map with improved colormap
        ax = axes[row,1]
        cf = ax.contourf(X1, X2, E_list[row], levels=30,
                         cmap=selected_cmap, alpha=0.95)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(r'$\hat{E}_{\theta(0)}$', fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=label_fontsize-2)
        
        # Improved scatter points
        scatter_color = 'white' if colormap == 'grayscale' else 'black'
        ax.scatter(samples[::sample_stride,0],
                   samples[::sample_stride,1],
                   c=scatter_color, s=scatter_size, alpha=scatter_alpha,
                   edgecolors='black' if scatter_color == 'white' else 'white',
                   linewidths=0.5)
        ax.set_xlabel('x$_1$', fontsize=fontsize)
        ax.set_ylabel('x$_2$', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize-2)

        # — probability map with improved contours
        ax = axes[row,2]
        cf = ax.contourf(X1, X2, P_list[row], levels=30,
                         cmap=selected_cmap, alpha=0.95)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(
            r'$p_{\theta(0)} = \exp[-\hat{E}_{\theta(0)}/k_\mathrm{B}T]/Z_{\theta(0)}$',
            fontsize=label_fontsize
        )
        cbar.ax.tick_params(labelsize=label_fontsize-2)
        
        # Improved scatter points
        ax.scatter(samples[::sample_stride,0],
                   samples[::sample_stride,1],
                   c=scatter_color, s=scatter_size, alpha=scatter_alpha,
                   edgecolors='black' if scatter_color == 'white' else 'white',
                   linewidths=0.5)
        
        # Thicker, more visible contour lines
        iso = ax.contour(X1, X2, true_prob,
                         levels=iso_levels,
                         colors='yellow',
                         linestyles='--',
                         linewidths=contour_linewidth)
        ax.clabel(iso, fmt='%.2f', fontsize=fontsize-2, inline=True)
        
        iso_proxy = Line2D(
            [0],[0],
            color='yellow',
            linestyle='--',
            linewidth=contour_linewidth,
            label=r'$p(x_1,x_2)$'
        )
        ax.legend(handles=[iso_proxy], fontsize=fontsize-2, loc='upper right')

        ax.set_xlabel('x$_1$', fontsize=fontsize)
        ax.set_ylabel('x$_2$', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize-2)

        # sanity check
        tot = jnp.sum(P_list[row]) * dx * dy
        print(f"{title:>22} total prob ≈ {tot:.6f}")

    # fig.suptitle(f'Print-Friendly Version (colormap: {colormap})', fontsize=label_fontsize+2)
    fig.tight_layout(rect=[0,0,1,0.95])
    
    # Save with colormap info in filename
    if plot_folder is not None:
        plt.savefig(f"{plot_folder}/fig_2D_mixture_comparison_print_friendly_{colormap}_Temp_{Temp}_num_traj_{n_trajectories}_atol_{atol}_rtol_{rtol}.png", 
                    dpi=300, bbox_inches='tight')
    plt.show()
