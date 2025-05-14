import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
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


# --------------------------------------------------------------------------
# helper:   log p_t(points)  for a Gaussian mixture, computed in chunks
# --------------------------------------------------------------------------
def _log_p_mixture(points, means, var_t, chunk=1024):
    """
    Parameters
    ----------
    points : (P, dim) array
        Proposal points x whose density we need.
    means  : (M, dim) array
        Time‑evolved component means μ_i(t).
    var_t  : scalar
        Common (isotropic) variance of each component.
    chunk  : int
        Number of mixture components processed at once.

    Returns
    -------
    (P,) array of log‑densities  log p_t(points).
    """
    P, dim = points.shape
    M      = means.shape[0]
    log_norm = -0.5 * dim * jnp.log(2.0 * jnp.pi * var_t)

    # initial values for numerically stable running log‑sum‑exp
    init_max = -jnp.inf * jnp.ones(P)
    init_sum = jnp.zeros(P)

    def body(carry, start):
        cur_max, cur_sum = carry
        # slice  'chunk'  means  starting at  'start'
        m_chunk = lax.dynamic_slice_in_dim(means, start, chunk, axis=0)  # (chunk, dim)

        # (P, chunk, dim)  →  (P, chunk)
        diff   = points[:, None, :] - m_chunk[None, :, :]
        log_c  = log_norm - 0.5 * jnp.sum(diff ** 2, axis=-1) / var_t

        # merge this chunk with previous partial log‑sum‑exp
        new_max = jnp.maximum(cur_max, jnp.max(log_c, axis=1))
        new_sum = (
            cur_sum * jnp.exp(cur_max - new_max) +
            jnp.sum(jnp.exp(log_c - new_max[:, None]), axis=1)
        )
        return (new_max, new_sum), None

    (log_max, exp_sum), _ = lax.scan(
        body, (init_max, init_sum), jnp.arange(0, M, chunk)
    )
    return log_max + jnp.log(exp_sum) - jnp.log(M)


# --------------------------------------------------------------------------
# main sampler with chunked log‑density evaluation
# --------------------------------------------------------------------------
def sample_total_distribution(
    t, n_samples, D, sigma_final, samples0, key, k, beta=1.0,
    *, exact=True, oversample_factor=4, chunk=1024
):
    """
    Sample x_t ~ q_t whose score is 2∇_x log p_t(x) + kx
    using either an exact quadratic‑mixture sampler (for small M)
    or an importance‑resampling scheme with *chunked* log p_t.
    """
    key1, key2, key3 = jr.split(key, 3)
    M, dim = samples0.shape

    # Ornstein–Uhlenbeck marginal parameters
    a_t   = jnp.exp(-beta * t / (sigma_final ** 2))
    var_t = D * sigma_final ** 2 * (1.0 - jnp.exp(-2 * beta * t / sigma_final ** 2))

    if k * var_t >= 2.0:
        raise ValueError(
            f"Density not normalisable: k*var_t = {k*var_t:.3f} ≥ 2."
        )

    # transformed covariance / mean factors
    new_var    = var_t / (2.0 - k * var_t)
    mean_scale = 1.0 / (1.0 - 0.5 * k * var_t)

    # ----------------------------------------------------------------------
    # 1)  exact mixture (small M only)
    # ----------------------------------------------------------------------
    if exact and M <= 250:
        means = a_t * samples0                      # (M, dim)

        mu_i  = means[:, None, :]
        mu_j  = means[None, :, :]
        mu_ij = 0.5 * (mu_i + mu_j)                # (M, M, dim)

        sq_norm_diff = jnp.sum((mu_i - mu_j) ** 2, axis=-1)
        sq_norm_mu   = jnp.sum(mu_ij ** 2, axis=-1)

        log_c  = -0.25 / var_t * sq_norm_diff
        log_tc = -0.5   * sq_norm_mu / (var_t / 2.0 + 1.0 / k)
        log_w  = (log_c + log_tc).reshape(-1)

        log_w  = log_w - jsp.special.logsumexp(log_w)
        pair_idx = jr.categorical(key1, log_w, shape=(n_samples,))
        mu_sel   = mu_ij.reshape(-1, dim)[pair_idx] * mean_scale

        eps = jr.normal(key2, shape=(n_samples, dim))
        return mu_sel + jnp.sqrt(new_var) * eps

    # ----------------------------------------------------------------------
    # 2)  importance‑resampling with chunked log p_t
    # ----------------------------------------------------------------------
    prop_N = oversample_factor * n_samples

    # 2.1  proposal samples from p_t
    prop = sample_forward_process(
        t, prop_N, D, sigma_final, samples0, key1, beta=beta
    )                                               # (prop_N, dim)

    # 2.2  log p_t(prop) via chunked mixture evaluation
    means = a_t * samples0                          # (M, dim)
    log_p = _log_p_mixture(prop, means, var_t, chunk=chunk)  # (prop_N,)

    # 2.3  importance weights  w(x) ∝ p_t(x) · exp(+½ k‖x‖²)
    log_w = log_p + 0.5 * k * jnp.sum(prop ** 2, axis=-1)
    log_w = log_w - jsp.special.logsumexp(log_w)
    w     = jnp.exp(log_w)

    # 2.4  systematic resampling
    idx = jr.choice(key3, prop_N, shape=(n_samples,), p=w)
    return prop[idx]


# def sample_total_distribution(
#     t, n_samples, D, sigma_final, samples0, key, k, beta=1.0,
#     *, exact=True, oversample_factor=4
# ):
#     """
#     Sample x_t ~ q_t whose score is 2∇_x log p_t(x) + kx.

#     Parameters
#     ----------
#     t : float
#         Diffusion time.
#     n_samples : int
#         Number of required samples.
#     D, sigma_final, beta, samples0
#         Same meaning as in `sample_forward_process`.
#     key : PRNGKey
#     k : float
#         Coefficient in the +k x term (scalar, isotropic case).
#     exact : bool (default True)
#         If True and dataset is small (M≤250) use an exact Gaussian mixture sampler.
#     oversample_factor : int
#         Only for the fallback path: proposal batch size = oversample_factor*n_samples.

#     Returns
#     -------
#     (n_samples, dim) array with samples from q_t.
#     """
#     key1, key2, key3 = jax.random.split(key, 3)
#     M, dim = samples0.shape

#     # --- OU marginal parameters ------------------------------------------------
#     a_t   = jnp.exp(-beta * t / (sigma_final ** 2))                      # mean decay
#     var_t = D * sigma_final ** 2 * (1.0 - jnp.exp(-2 * beta * t / sigma_final ** 2))

#     # --- check normalisability -------------------------------------------------
#     if k * var_t >= 2.0:
#         raise ValueError(
#             f"Density not normalisable at this (t, k): k*var_t = {k*var_t:.3f} ≥ 2."
#         )

#     # new component covariance and mean‑scaling factors (isotropic)
#     new_var   = var_t / (2.0 - k * var_t)                 # Σ''  in the derivation
#     mean_scale = 1.0 / (1.0 - 0.5 * k * var_t)            # μ  → μ / (1 - k var/2)

#     # --------------------------------------------------------------------------
#     # 1) exact mixture sampling  (works up to a few hundred modes without fuss)
#     # --------------------------------------------------------------------------
#     if exact and M <= 250:
#         # time‑evolved means of the *original* mixture components
#         means = a_t * samples0                  # shape (M, dim)

#         # pairwise (i,j) component means μ_ij and log‑weights  log w_ij
#         mu_i   = means[:, None, :]              # (M,1,dim)
#         mu_j   = means[None, :, :]              # (1,M,dim)
#         mu_ij  = 0.5 * (mu_i + mu_j)            # (M,M,dim)

#         # log‑weight pieces  c_ij  and  tilde{c}_ij  (constants drop out)
#         sq_norm_diff = jnp.sum((mu_i - mu_j) ** 2, axis=-1)        # ‖μ_i-μ_j‖²
#         sq_norm_mu   = jnp.sum(mu_ij ** 2, axis=-1)                # ‖μ_ij‖²

#         log_c    = -0.25 / var_t * sq_norm_diff
#         log_tc   = -0.5   * sq_norm_mu / (var_t / 2.0 + 1.0 / k)
#         log_w    = (log_c + log_tc).reshape(-1)

#         # normalise logits for categorical sampling
#         log_w   = log_w - jsp.special.logsumexp(log_w)

#         # sample component indices and draw from the corresponding Gaussian
#         pair_idx = jax.random.categorical(key1, log_w, shape=(n_samples,))
#         mu_pairs = mu_ij.reshape(-1, dim)[pair_idx] * mean_scale

#         eps      = jax.random.normal(key2, shape=(n_samples, dim))
#         return mu_pairs + jnp.sqrt(new_var) * eps

#     # --------------------------------------------------------------------------
#     # 2) importance‑resampling fallback  (cheaper memory, scales to large M)
#     # --------------------------------------------------------------------------
#     prop_N  = oversample_factor * n_samples

#     # 2.1 draw proposal points from the known p_t
#     prop = sample_forward_process(
#         t, prop_N, D, sigma_final, samples0, key1, beta=beta
#     )

#     # 2.2 compute log p_t(prop)   ---- fully vectorised
#     means = a_t * samples0                                            # (M, dim)

#     def log_gauss(x, m):
#         return -0.5 * (
#             dim * jnp.log(2.0 * jnp.pi * var_t) +
#             jnp.sum((x - m) ** 2, axis=-1) / var_t
#         )                                                             # scalar

#     def log_p_single(x):
#         # mixture log‑density  log p_t(x)
#         log_comp = log_gauss(x, means)                                # (M,)
#         return jsp.special.logsumexp(log_comp) - jnp.log(M)           # scalar

#     log_p = jax.vmap(log_p_single)(prop)                              # (prop_N,)

#     # 2.3 importance weights  w(x) ∝ p_t(x) exp(+½ k‖x‖²)
#     log_w = log_p + 0.5 * k * jnp.sum(prop ** 2, axis=-1)
#     log_w = log_w - jsp.special.logsumexp(log_w)
#     w     = jnp.exp(log_w)

#     # 2.4 systematic resampling
#     idx   = jax.random.choice(key3, prop_N, shape=(n_samples,), p=w)
#     return prop[idx]

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
# Getting best parameters
########################################################################################
def get_best_params(params_history, loss_history, maximize=False):
    loss_history = jnp.array(loss_history)
    # Find index of lowest loss
    if maximize:
        best_idx = jnp.argmax(loss_history)
    else:
        best_idx = jnp.argmin(loss_history)
    return loss_history[best_idx], params_history[best_idx], best_idx
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
    
    def _interpolator(t):
        # Transpose params to shape (n_params, n_timesteps) so that each row is a trajectory.
        # Use vmap to apply jnp.interp over each parameter trajectory.
        interpolated = jax.vmap(lambda param: jnp.interp(t, time_points, param))(params.T)
        # If t is an array, jnp.interp returns an array for each parameter, resulting in a
        # (n_params, len(t)) array. Transpose it so that each row corresponds to a time point.
        if jnp.ndim(t) > 0:
            return interpolated.T
        return interpolated

    return _interpolator
