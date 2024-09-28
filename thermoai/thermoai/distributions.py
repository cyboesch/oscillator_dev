import jax
#%%
import jax.numpy as jnp
import jax.random as random
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML
import matplotlib.animation as animation

# @jax.jit
def sample_mog(key, means, covariances, weights):
    n_components = weights.shape[0]
    
    key_select, key_sample = random.split(key)
    
    component_index = random.choice(
        key_select, 
        n_components, 
        p=weights
    )
    

    selected_mean = means[component_index]
    selected_cov = covariances[component_index]
    
    if selected_mean.ndim == 0:
        sample = random.normal(key_sample) * jnp.sqrt(selected_cov) + selected_mean
    else:
        sample = random.multivariate_normal(
            key_sample, 
            selected_mean, 
            selected_cov
    )
    
    return sample

def mog_get_mean(means, weights):
    return jnp.sum(weights * means)

@jax.jit
def mog_pdf(x, means, covariances, weights):
    def component_pdf(mean, cov):
        return multivariate_normal.pdf(x, mean=mean, cov=cov)
    
    pdfs = jax.vmap(component_pdf)(means, covariances)
    return jnp.sum(weights*pdfs)

# @jax.jit
# def mog_logpdf(x, means, covariances, weights):    
#     pdf = mog_pdf(x, means, covariances, weights)
#     return jnp.log(pdf)

## Geoefrrey more elaborate version
@jax.jit
def mog_logpdf(x, means, covariances, weights):
    def component_logpdf(mean, cov):
        return multivariate_normal.logpdf(x, mean=mean, cov=cov)
    
    logpdfs = jax.vmap(component_logpdf)(means, covariances)
    return logsumexp(jnp.log(weights) + logpdfs)

# @jax.jit
# def mog_logpdf(x, means, covariances, weights):
#     def component_pdf(mean, cov):
#         return multivariate_normal.pdf(x, mean=mean, cov=cov)
    
#     pdfs = jax.vmap(component_pdf)(means, covariances)
#     pdfs_sum = jnp.sum(weights* pdfs)
#     return jnp.log(pdfs_sum)

#%%
if __name__ == "__main__":
    means = jnp.array([
        [1.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0]
    ])
    covariances = jnp.array([
        [[0.01, 0.01], [0.01, 0.4]],
        [[0.1, -0.03], [-0.03, 0.1]],
        [[0.2, 0.0], [0.0, 0.2]]
    ])
    weights = jnp.array([0.3, 0.3, 0.4])

    # Number of samples to generate
    n_samples = 1000

    # JAX random key
    key = random.PRNGKey(42)

    # Generate samples
    keys = random.split(key, n_samples)
    samples = jax.vmap(lambda key: sample_mog(key, means, covariances, weights))(keys)

    # Create a grid for the contour plot
    x_lim, y_lim = 2, 2
    res = 0.01
    x, y = jnp.mgrid[-x_lim:x_lim:res, -y_lim:y_lim:res]
    pos = jnp.dstack((x, y))

    # Compute the PDF, logPDF, and energy
    z_pdf = jax.vmap(lambda p: mog_pdf(p, means, covariances, weights))(pos.reshape(-1, 2)).reshape(x.shape)
    z_logpdf = jax.vmap(lambda p: mog_logpdf(p, means, covariances, weights))(pos.reshape(-1, 2)).reshape(x.shape)
    z_energy = jax.vmap(lambda p: mog_energy(p, means, covariances, weights))(pos.reshape(-1, 2)).reshape(x.shape)

    # Create a new figure with a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mixture of Gaussians Visualization', fontsize=16)

    # 2D Contour plots
    contour_plots = [
        (axs[0, 0], z_pdf, 'PDF'),
        (axs[0, 1], z_logpdf, 'Log PDF'),
        (axs[0, 2], z_energy, 'Energy')
    ]

    for ax, z, title in contour_plots:
        cf = ax.contourf(x, y, z, levels=20, cmap='viridis')
        ax.set_title(f'{title} (2D Contour)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(cf, ax=ax)
        ax.scatter(samples[:, 0], samples[:, 1], color='red', alpha=0.5, s=1)

    # 3D Surface plots
    surface_plots = [
        (axs[1, 0], z_pdf, 'PDF'),
        (axs[1, 1], z_logpdf, 'Log PDF'),
        (axs[1, 2], z_energy, 'Energy')
    ]

    for ax, z, title in surface_plots:
        ax.remove()
        ax = fig.add_subplot(2, 3, 4 + surface_plots.index((ax, z, title)), projection='3d')
        surf = ax.plot_surface(x, y, z, cmap='viridis')
        ax.set_title(f'{title} (3D Surface)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel(title)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()
