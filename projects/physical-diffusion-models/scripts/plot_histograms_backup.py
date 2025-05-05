plt.savefig(f'{plot_folder}/true_vs_generated_images_comparison_rtol_sde_{rtol_sde}_atol_sde_{atol_sde}_n_samples_{num_examples}.png')
plt.show()

#####################################
# Plot pixel-wise histograms
#####################################
# Reshape the data for pixel-wise comparison
pixels_true = images_flat_true.reshape(-1, N_osc)  # Shape: (n_samples, N_osc)
pixels_generated = images_generated_non_smoothed_sde.reshape(-1, N_osc)  # Shape: (n_trajectories, N_osc)

# Create a grid of subplots for histograms
n_cols = 10  # Number of columns in the grid
n_rows = (N_osc + n_cols - 1) // n_cols  # Ceiling division to get number of rows
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2*n_rows))
fig.suptitle('Pixel-wise Distribution Comparison', fontsize=16, y=1.02)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Plot histograms for each pixel
for i in range(N_osc):
    ax = axes[i]
    # Plot histograms
    ax.hist(pixels_true[:, i], bins=30, alpha=0.5, label='True', density=True)
    ax.hist(pixels_generated[:, i], bins=30, alpha=0.5, label='Generated', density=True)
    ax.set_title(f'Pixel {i}', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Add legend to the first subplot only
    if i == 0:
        ax.legend(fontsize=6)

# Hide unused subplots
for i in range(N_osc, len(axes)):
    axes[i].axis('off')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{plot_folder}/pixel_wise_distributions_comparison.png', bbox_inches='tight')
plt.show()

