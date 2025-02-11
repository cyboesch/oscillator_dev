import jax
from jax import grad, vmap
import jax.numpy as jnp
import jax.random as jr
import diffrax
import matplotlib.pyplot as plt
from functools import partial
from jax.scipy.stats import multivariate_normal

def sample_gaussian_mixture(key, n_samples, weights, means, covs):
    """
    Sample from a mixture of multivariate Gaussians
    
    Args:
        key: JAX random key
        n_samples: number of samples to draw
        weights: mixture weights (shape: N_mixgauss)
        means: means of Gaussians (shape: N_mixgauss x N)
        covs: covariance matrices (shape: N_mixgauss x N x N)
    
    Returns:
        samples: array of shape (n_samples x N)
    """
    # Split key for component selection and sampling
    key_components, key_sample = jr.split(key)
    
    # Select components based on weights
    components = jr.choice(
        key_components,
        jnp.arange(len(weights)),
        shape=(n_samples,),
        p=weights
    )
    
    # Generate standard normal samples
    keys_sample = jr.split(key_sample, n_samples)
    
    def sample_one(key, component_idx):
        mean = means[component_idx]
        cov = covs[component_idx]
        # Sample from standard normal and transform
        z = jr.multivariate_normal(key, jnp.zeros_like(mean), jnp.eye(mean.shape[0]))
        # Use Cholesky decomposition for numerical stability
        L = jnp.linalg.cholesky(cov)
        return mean + jnp.dot(L, z)
    
    samples = jax.vmap(sample_one)(keys_sample, components)
    return samples

def rescale_to_pi_interval(data):
    """
    Rescale and shift data to lie in [-π, π] interval
    
    Args:
        data: array of shape (n_samples, N_dimensions)
    
    Returns:
        scaled_data: array of same shape, rescaled to [-π, π]
    """
    # Compute min and max for each dimension
    data_min = jnp.min(data, axis=0)
    data_max = jnp.max(data, axis=0)
    data_range = data_max - data_min
    
    # First shift to [0, data_range]
    shifted = data - data_min
    
    # Then rescale to [0, 2π]
    scaled = (shifted / data_range) * (2 * jnp.pi)
    
    # Finally shift to [-π, π]
    return scaled - jnp.pi


def XY_energy(x, W, h):
    """
    Compute the energy of the XY model
    Args:
        x: array of phases (shape: N_osc)
        W: coupling matrix (shape: N_osc x N_osc)
        h: local biases (shape: N_osc)
    Returns:
        energy: scalar
    """
    # Compute coupling energy
    coupling = -jnp.sum(W * jnp.cos(x[:, None] - x[None, :]))
    # Compute bias energy
    bias = -jnp.sum(h * jnp.cos(x))
    return coupling + bias

def drift_x_fn(t, state, args):
    """
    Drift function for the XY model with effective temperature
    Args:
        t: time
        state: array of phases (shape: N_osc)
        args: tuple containing (W, h)
    Returns:
        Array of phase velocities
    """
    W, h = args
    # Compute force from energy gradient
    dE_dtheta = jax.grad(XY_energy, argnums=0)(state, W, h)
    
    # Damping term
    gamma = 1.0  # Damping coefficient, adjust as needed
    
    dtheta_dt = -gamma * dE_dtheta
    
    return dtheta_dt

def diffusion_x_fn(t, state, args):
    """
    Diffusion function for the XY model with effective temperature
    Args:
        t: time
        state: array of phases (shape: N_osc)
        args: additional arguments
    Returns:
        Noise matrix (shape: N_osc x 1)
    """
    T_eff = 1.0  # Effective temperature, adjust as needed
    gamma = 1.0  # Same damping coefficient as in drift
    
    noise_strength = jnp.sqrt(2 * gamma * T_eff)
    noise_matrix = jnp.eye(N_osc)
    return noise_strength * noise_matrix


def compute_gradient_difference(samples, param_idx, dt, key, num_noise_samples=1000):
    """
    Computes the difference in energy gradients between current and evolved samples for XY model.
    When num_noise_samples=0, computes deterministic gradient without noise averaging.
    
    Args:
        samples: Input samples (angles)
        param_idx: Parameter index for gradient computation
        dt: Time step
        key: Random key
        num_noise_samples: Number of noise realizations to average over. If 0, computes deterministically.
    """
    # Compute current gradients
    grad_energy_curr = grad(XY_energy, argnums=(param_idx))
    grad_energy_x_curr = lambda x: grad_energy_curr(x, J)
    grad_batch_curr = vmap(grad_energy_x_curr)
    current_grads = grad_batch_curr(samples)
    current_grads_avg = jnp.mean(current_grads, axis=0)
    
    # Get drift for all samples
    drift_batch = vmap(lambda x: drift_x_fn(0., x, None))
    drifts = drift_batch(samples)
    
    if num_noise_samples == 0:
        # Deterministic evolution with only drift
        evolved_samples = samples + drifts * dt/2
        
        # Compute evolved gradients
        grad_energy = grad(XY_energy, argnums=(param_idx))
        grad_energy_x = lambda x: grad_energy(x, J)
        grad_batch = vmap(grad_energy_x)
        evolved_grads = grad_batch(evolved_samples)
        avg_evolved_grads = jnp.mean(evolved_grads, axis=0)
    else:
        # Split key for multiple noise samples
        keys = random.split(key, num_noise_samples)
        
        # Function to compute evolved gradients for one noise realization
        def compute_evolved_grads(key):
            noise_std = jnp.sqrt(2*T*dt)  # XY model noise scale
            noise = random.normal(key, shape=samples.shape) * noise_std
            evolved_samples = samples + drifts * dt/2 + noise
            
            grad_energy = grad(XY_energy, argnums=(param_idx))
            grad_energy_x = lambda x: grad_energy(x, J)
            grad_batch = vmap(grad_energy_x)
            return grad_batch(evolved_samples)
        
        # Compute and average evolved gradients over noise realizations
        evolved_grads_all = vmap(compute_evolved_grads)(keys)
        avg_evolved_grads = jnp.mean(jnp.mean(evolved_grads_all, axis=0), axis=0)
    
    # Compute final gradient difference
    avg_grad_diff = current_grads_avg - avg_evolved_grads
    return -avg_grad_diff