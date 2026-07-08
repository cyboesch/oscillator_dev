import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jr
from scipy.signal import savgol_filter


########################################################################################
# Sampling
########################################################################################

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

########################################################################################
# Normalizing samples
########################################################################################

def normalize_samples(samples):
    """
    Normalize samples to have mean 0 and variance 1 along each dimension.
    
    Args:
        samples: Array of shape (n_samples, n_dimensions)
    
    Returns:
        Normalized samples with same shape as input
    """
    # Calculate mean and std along each dimension
    mean = jnp.mean(samples)
    std = jnp.std(samples)
    
    # Normalize
    normalized_samples = (samples - mean) / std
    
    return normalized_samples, mean, std

########################################################################################
# Forward diffusion process - sampling from marginal distribution
########################################################################################

def sample_forward_process(t, n_samples, D, sigma_final, samples0, key, beta=1.0):
    """
    Forward diffusion process: dx = -beta/sigma_final^2 * x dt + sqrt(2*D*beta) dw,
    Samples from the marginal distribution 
      p_t(x_t) = (1/M) sum_{i=1}^M p(x_t|x0^{(i)})
    where for the SDE
         dx = -beta/sigma_final^2 * x dt + sqrt(2*D*beta) dw,
    the solution is:
         x_t = exp(-beta*t/sigma_final^2) * x0 + sqrt(D*sigma_final^2*(1 - exp(-2*beta*t/sigma_final^2))) * epsilon,
         with epsilon ~ N(0, I).
    
    Parameters:
      t          : time (scalar)
      n_samples  : number of samples to generate from p_t
      D          : noise strength parameter (affects the variance)
      sigma_final: parameter such that the final variance is (ideally) sigma_final^2 
                   (if parameters are chosen so that at final time t_final: 
                   D*(1 - exp(-2*beta*t_final/sigma_final^2)) = 1).
      beta       : diffusion rate
      samples0   : array of shape (M, N) containing the initial data points {x0}
      key        : JAX random key
      
    Returns:
      x_t_samples: array of shape (n_samples, N) of samples from p_t(x_t)
    """
    # Split the random key for index selection and noise generation.
    key_idx, key_noise = jax.random.split(key)
    
    # Number of initial samples in the dataset.
    M = samples0.shape[0]
    
    # Randomly choose indices (with replacement) from the dataset.
    indices = jax.random.choice(key_idx, M, shape=(n_samples,), replace=True)
    x0_samples = samples0[indices]
    
    # Compute the deterministic decay factor.
    mean_factor = jnp.exp(-beta * t / (sigma_final**2))
    
    # Compute the variance factor for the added noise.
    var_factor = D * (sigma_final**2) * (1 - jnp.exp(-2 * beta * t / (sigma_final**2)))
    
    # Sample standard Gaussian noise with the same shape as the selected initial samples.
    noise = jax.random.normal(key_noise, shape=x0_samples.shape)
    
    # Combine the deterministic and stochastic parts.
    x_t_samples = mean_factor * x0_samples + jnp.sqrt(var_factor) * noise
    return x_t_samples


########################################################################################
# Reformatting optimization results
########################################################################################

def reformat_optimization_results(params_history, loss_history, unflatten, slicing=1, maximize=False):
    # Unflatten the first set of parameters to determine the structure
    first_params = unflatten(params_history[0])
    param_histories = [jnp.zeros((len(params_history[::slicing]), *param.shape)) for param in first_params]

    # Convert histories to arrays for plotting
    params_history = jnp.array(params_history[::slicing])

    for i in range(len(params_history)):
        unflattened_params = unflatten(params_history[i])
        for j, param in enumerate(unflattened_params):
            param_histories[j] = param_histories[j].at[i].set(param)

    loss_history = jnp.array(loss_history[::slicing])

    return (*param_histories, loss_history)
########################################################################################
# Smoothing parameters
########################################################################################
def smooth_parameters(params, window_lengths, poly_orders):
    """
    Smooths multiple parameter trajectories using Savitzky-Golay filter
    
    Args:
        params: Array of shape (n_timesteps, n_params) containing parameter trajectories
        window_lengths: List/array of window lengths for each parameter dimension
        poly_orders: List/array of polynomial orders for each parameter dimension
    
    Returns:
        Array of shape (n_timesteps, n_params) containing smoothed parameter trajectories
    """
    n_params = params.shape[1]
    smoothed = []
    
    for i in range(n_params):
        # Ensure window length is odd and valid
        window_length = min(window_lengths[i], len(params) - (1 - len(params) % 2))
        if window_length % 2 == 0:
            window_length -= 1
            
        # Smooth using Savitzky-Golay filter
        smoothed.append(savgol_filter(params[:, i], window_length, poly_orders[i]))
    
    return jnp.column_stack(smoothed)

########################################################################################
# Interpolating parameters
########################################################################################

def interpolate_parameters(params, time_points):
    """
    Interpolates parameter trajectories using jnp.interp in a vectorized fashion.

    Args:
        params: Array of shape (n_timesteps, n_params) containing parameter trajectories.
        time_points: 1D array of shape (n_timesteps,) corresponding to the time points.

    Returns:
        A function that takes time points `t` (scalar or 1D array) and returns
        the interpolated parameters. For scalar `t`, the result is shape (n_params,);
        for vector `t`, the result is shape (len(t), n_params).
    """

    # IMPORTANT: jnp.interp requires `xp` (here: time_points) to be increasing.
    # We may optimize/store parameters in reverse time order (e.g. T -> 0),
    # so we sort time_points and permute params accordingly to make interpolation well-defined.
    params = jnp.asarray(params)
    time_points = jnp.asarray(time_points)
    sort_idx = jnp.argsort(time_points)
    time_points_sorted = time_points[sort_idx]
    params_sorted = params[sort_idx]

    def _interpolator(t):
        # Transpose params to shape (n_params, n_timesteps) so that each row is a trajectory.
        # Use vmap to apply jnp.interp over each parameter trajectory.
        interpolated = jax.vmap(lambda param: jnp.interp(t, time_points_sorted, param))(params_sorted.T)
        # If t is an array, jnp.interp returns an array for each parameter, resulting in a
        # (n_params, len(t)) array. Transpose it so that each row corresponds to a time point.
        if jnp.ndim(t) > 0:
            return interpolated.T
        return interpolated

    return _interpolator
