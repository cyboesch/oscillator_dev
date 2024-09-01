from jax.scipy.stats import multivariate_normal
import jax.random as random
import jax.numpy as jnp
import matplotlib.pyplot as plt

def sample_1d_mog(mu1, stddev1, mu2, stddev2, p1, key):
    """
    Samples from a superposition of two Gaussian distributions in 2D using JAX.
    """
    u_rng, z_rng = random.split(key)
    u = random.uniform(u_rng)
    mu = jnp.where(u < p1, mu1, mu2)
    stddev = jnp.where(u < p1, stddev1, stddev2)
    return mu + stddev*random.normal(z_rng)

def sample_from_superposition(mu1, sigma1, mu2, sigma2, p1, n_samples, key):
    """
    Samples from a superposition of two Gaussian distributions in 2D using JAX.
    """
    key1, key2, key3 = random.split(key, 3)
    
    # Sample from both Gaussians
    samples1 = random.multivariate_normal(key1, mu1, sigma1, shape=(n_samples,))
    samples2 = random.multivariate_normal(key2, mu2, sigma2, shape=(n_samples,))
    
    # Generate random numbers to decide which sample to keep
    choices = random.uniform(key3, shape=(n_samples,)) < p1
    
    # Select samples based on the choices
    samples = jnp.where(choices[:, None], samples1, samples2)
    
    return samples

def gaussian_pdf(x1, x2, mu, sigma):
    """Compute the PDF of a 2D Gaussian distribution."""
    pos = jnp.dstack((x1, x2))
    return multivariate_normal.pdf(pos, mean=mu, cov=sigma)


if __name__ == "__main__":
    # Parameters for the first Gaussian
    mu1 = jnp.array([1, 1])
    sigma1 = jnp.array([[0.01, 0.01], [0.01, 0.4]])

    # Parameters for the second Gaussian
    mu2 = jnp.array([-1, 0])
    sigma2 = jnp.array([[0.1, -0.03], [-.03, 0.1]])

    # Probability of sampling from the first Gaussian
    p1 = 0.5

    # Number of samples to generate
    n_samples = 1000

    # JAX random key
    key = random.PRNGKey(0)

    # Generate samples
    samples = sample_from_superposition(mu1, sigma1, mu2, sigma2, p1, n_samples, key)

    # Create a grid for the contour plot
    x_lim = 2
    y_lim = 2
    x, y = jnp.mgrid[-x_lim:x_lim:.01, -y_lim:y_lim:.01]

    # Compute the PDF for both Gaussians
    z1 = gaussian_pdf(x, y, mu1, sigma1)
    z2 = gaussian_pdf(x, y, mu2, sigma2)

    # Combine the PDFs according to the mixture weights
    z = p1 * z1 + (1 - p1) * z2

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot the contour of the combined distribution
    plt.contourf(x, y, z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Probability Density')

    # Plot the samples
    plt.scatter(samples[:, 0], samples[:, 1], color='red', alpha=0.5, s=10, label='Samples')

    plt.title('Superposition of Two Gaussians in 2D with Samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()