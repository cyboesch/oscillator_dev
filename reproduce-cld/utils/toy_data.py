import jax
import jax.numpy as jnp
from typing import Callable
from jax.scipy.stats import multivariate_normal

def get_diamond_sample() -> Callable:
    WIDTH: int = 3
    BOUND: float = 0.5
    NOISE: float = 0.04
    ROTATION_MATRIX: jnp.ndarray = jnp.array([[1., -1.], [1., 1.]]) / jnp.sqrt(2.)

    # Generate grid of means
    xs: jnp.ndarray = jnp.linspace(-BOUND, BOUND, WIDTH)
    ys: jnp.ndarray = jnp.linspace(-BOUND, BOUND, WIDTH)
    means: jnp.ndarray = jnp.array([(x, y) for x in xs for y in ys])
    
    # Apply rotation
    means = means @ ROTATION_MATRIX

    covariance_factor: jnp.ndarray = NOISE * jnp.eye(2)
    covariance: jnp.ndarray = covariance_factor @ covariance_factor.T  # Full covariance matrix

    def sample(key: jnp.ndarray) -> jnp.ndarray:
        component_key, noise_key = jax.random.split(key)
        
        # Select a random component
        index = jax.random.randint(component_key, (), 0, WIDTH**2)
        
        # Generate noise
        noise = jax.random.normal(noise_key, (2,))
        
        # Combine selected mean with noise
        return means[index] + noise @ covariance_factor

    def logpdf(x: jnp.ndarray) -> float:
        # Compute the log PDF of x under each Gaussian component
        def component_logpdf(mean):
            return multivariate_normal.logpdf(x, mean, covariance)

        # Vectorize over all components
        logpdfs = jax.vmap(component_logpdf)(means)
        
        # Calculate the log-sum-exp across components (equal mixing weights)
        return jax.scipy.special.logsumexp(logpdfs) - jnp.log(WIDTH ** 2)

    return sample, logpdf

def get_multimodal_swissroll_sample() -> Callable:
    NOISE: float = 0.2
    MULTIPLIER: float = 0.01
    OFFSETS: jnp.ndarray = jnp.array([
        [0.8, 0.8], [0.8, -0.8], [-0.8, -0.8], [-0.8, 0.8], [0.0, 0.0]
    ])

    def jax_swiss_roll(key):
        t = jax.random.uniform(key, (1,), minval=1.5, maxval=4.5) * 3
        height = jax.random.uniform(key, (1,), minval=0, maxval=1)
        x = t * jnp.cos(t)
        y = height
        z = t * jnp.sin(t)
        return jnp.column_stack([x, z])

    def sample(key: jnp.ndarray) -> jnp.ndarray:
        component_key, swissroll_key, noise_key = jax.random.split(key, 3)
        
        # Select a random component
        component = jax.random.randint(component_key, (), 0, 5)
        
        # Generate Swiss roll data
        x = jax_swiss_roll(swissroll_key)
        x = x * MULTIPLIER
        
        # Add noise
        noise = jax.random.normal(noise_key, shape=(1, 2)) * NOISE * MULTIPLIER
        x = x + noise
        
        # Add offset for the selected component
        x = x + OFFSETS[component]
        
        return x[0]  # Return as a 1D array

    return sample
