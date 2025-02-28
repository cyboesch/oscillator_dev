import jax
from jax import grad, vmap, hessian
import jax.numpy as jnp
import jax.random as jr
import diffrax
from functools import partial
import optax


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

def normalize_samples(samples):
    """
    Normalize samples to have mean 0 and variance 1 along each dimension.
    
    Args:
        samples: Array of shape (n_samples, n_dimensions)
    
    Returns:
        Normalized samples with same shape as input
    """
    # Calculate mean and std along each dimension
    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0)
    
    # Normalize
    normalized_samples = (samples - mean) / std
    
    return normalized_samples


def setup_score_matching_loss_per_batch(energy_fn):
    def loss_fn_per_batch(flattened_args,batch):
        n_samples = batch.shape[0]
        def log_propability_unnormalized(x):
            return -energy_fn(x, flattened_args)
        
        current_score = grad(log_propability_unnormalized)
        current_score2 = hessian(log_propability_unnormalized)
        
        current_score_loss_per_sample = lambda x: (jnp.trace(current_score2(x)) + 1/2 * jnp.sum(current_score(x)**2))/n_samples   
        return jnp.sum(vmap(current_score_loss_per_sample)(batch))
    return loss_fn_per_batch


def CD1_gradient(energy_fn, samples, flattened_args, dt, D, key, num_noise_samples=1000):
    """
    Computes the difference in energy gradients between current and evolved samples
    for a generic energy function.
    
    Args:
        energy_fn: Energy function that takes (x, *args) as input
        samples: Input samples
        args: Tuple of parameters for the energy function
        dt: Time step
        D: Noise strength (diffusion constant)
        key: Random key
        num_noise_samples: Number of noise realizations to average over. If 0, computes deterministically.
    
    Returns:
        Gradient differences with respect to all parameters in args
    """
    # Compute current gradients for all parameters
    grad_energy_curr = grad(energy_fn, argnums=1)
    grad_energy_x_curr = lambda x: grad_energy_curr(x, flattened_args)
    grad_batch_curr = vmap(grad_energy_x_curr)
    current_grads = grad_batch_curr(samples)
    current_grads_avg = jnp.mean(current_grads, axis=0)
    
    # Get drift for all samples
    drift_fn = lambda x: -grad(energy_fn, argnums=0)(x, flattened_args)
    drift_batch = vmap(drift_fn)
    drifts = drift_batch(samples)
    
    if num_noise_samples == 0:
        # Deterministic evolution with only drift
        evolved_samples = samples + drifts * dt/2
        
        # Compute evolved gradients
        grad_energy = grad(energy_fn, argnums=1)
        grad_energy_x = lambda x: grad_energy(x, flattened_args)
        grad_batch = vmap(grad_energy_x)
        evolved_grads = grad_batch(evolved_samples)
        avg_evolved_grads = jnp.mean(evolved_grads, axis=0)
    else:
        # Split key for multiple noise samples
        keys = jr.split(key, num_noise_samples)
        
        # Function to compute evolved gradients for one noise realization
        def compute_evolved_grads(key):
            noise_std = jnp.sqrt(D*dt)
            noise = jr.normal(key, shape=samples.shape) * noise_std
            evolved_samples = samples + drifts * dt/2 + noise
            
            grad_energy = grad(energy_fn, argnums=1)
            grad_energy_x = lambda x: grad_energy(x, flattened_args)
            grad_batch = vmap(grad_energy_x)
            return grad_batch(evolved_samples)
        
        # Compute and average evolved gradients over noise realizations
        evolved_grads_all = vmap(compute_evolved_grads)(keys)
        avg_evolved_grads = jnp.mean(jnp.mean(evolved_grads_all, axis=0), axis=0)
    
    # Compute final gradient difference
    avg_grad_diff = current_grads_avg - avg_evolved_grads
    return -avg_grad_diff/(dt/2) # the negative sign ensures that this is in fact the same gradient as in eq. (1) in the paper "Connections Between Score Matching, Contrastive Divergence, and Pseudolikelihood for Continuous-Valued Variables" by Hyvärinen


def run_optimization(loss_fn_per_batch, params_initial, samples, gradient_fn_per_batch = None, key = jr.PRNGKey(0),batch_size=128, learning_rate=0.001, n_epochs=20000):
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params_initial)
    
    @partial(jax.jit, static_argnums=(3,))
    def training_step(params, opt_state, samples, batch_size, key):
        """Single training step using batched samples"""
        # Get random batch of samples
        key, subkey = jr.split(key)
        n_samples = len(samples)
        idx = jr.randint(subkey, (batch_size,), 0, n_samples)
        batch = samples[idx]
        
        # Setup loss for this batch
        loss_fn = lambda params_current: loss_fn_per_batch(params_current, batch)
        loss_val = loss_fn(params)
        
        # Get gradient of parameters for this batch
        if gradient_fn_per_batch is None:
            dparams = grad(loss_fn)(params) 
        else:
            dparams = gradient_fn_per_batch(params, batch)

        
        # Apply updates
        updates, opt_state = optimizer.update(dparams, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, key, loss_val
    
    # Training loop
    params_history = []
    loss_history = []

    params = params_initial

    for epoch in range(n_epochs):
        key, subkey = jr.split(key)
        params, opt_state, key, loss = training_step(
            params, opt_state, samples, batch_size, subkey)
        
        # Store parameters and loss
        params_history.append(params)
        loss_history.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}")
            print(f"params:\n{params}")
            print(f"Loss: {loss:.4f}")
            print("---")
    return params_history, loss_history