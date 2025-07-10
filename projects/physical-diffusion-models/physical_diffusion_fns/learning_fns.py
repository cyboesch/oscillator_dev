import jax
from jax import grad, vmap,pmap, hessian, jacfwd
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import pjit
import jax.numpy as jnp
import jax.random as jr
import diffrax
from diffrax import ControlTerm, MultiTerm, ODETerm
from functools import partial
from jax.experimental.pjit import pjit
import optax
from physical_diffusion_fns.network_fns import setup_overdamped_SDE, solve_SDE


########################################################################################
# Maximum log likelihood
#####################################################################################

def setup_MLE_loss_per_batch(energy_fn):
    def loss_fn_per_batch(flattened_args,batch):
        _loss_fn_per_batch = lambda x: -energy_fn(x,flattened_args)
        return jnp.sum(vmap(_loss_fn_per_batch)(batch))
    return loss_fn_per_batch

def setup_MLE_gradient_per_batch(energy_fn, N_osc, gamma=1.0, k_b=1.0, T=1.0, t0=0.0, t1=10.0, N_samples=1000, dt0=0.01, initial_state=jnp.array([0.,0.])):
    
    def gradient_fn_per_batch(flattened_args,batch):
        drift_fn, diffusion_fn = setup_overdamped_SDE(energy_fn, flattened_args, N_osc, gamma, k_b, T)
        solution = solve_SDE(drift_fn, diffusion_fn, initial_state, t0, t1, N_samples, dt0)
        
        d_energy_d_params = lambda x: grad(energy_fn, argnums=1)(x, flattened_args)
        expected_val_d_energy_d_params_clamped_batch = jnp.sum(vmap(d_energy_d_params)(batch), axis=0)/batch.shape[0]
        
        expected_val_d_energy_d_params_free = jnp.sum(vmap(d_energy_d_params)(solution.ys), axis=0)/solution.ys.shape[0]

        return expected_val_d_energy_d_params_free-expected_val_d_energy_d_params_clamped_batch
         
    return gradient_fn_per_batch

########################################################################################
# Score matching 
########################################################################################

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

def setup_denoising_score_matching_loss_per_batch(energy_fn, k_b=1.0, T=1.0):
    """
    Returns a loss function of the form

      loss = loss_fn_per_batch(flattened_args, batch)

    where `batch` is a tuple
      (x_t_batch, eps_over_sigma_batch)
    both of shape (n_samples, dim).

    It computes
      1/(2n) * sum_i || score(x_t_i) + eps_over_sigma_i ||^2
    where score = ∇_x log p_theta = -∇_x energy_fn.
    """
    def loss_fn_per_batch(flattened_args, batch):
        x_t_batch, eps_over_sigma = batch   # each (n, dim)
        n = x_t_batch.shape[0]

        # model‐score: ∇_x log p_theta(x)
        def log_unnorm(x):
            return -energy_fn(x, flattened_args)/(k_b*T)
        score_fn = grad(log_unnorm)

        # per‐sample DSM loss: ½‖s(x_t) + ε/σ_t‖², averaged over n
        def loss_per_sample(x_t, eps_s):
            s = score_fn(x_t)                  # (dim,)
            return 0.5 * jnp.sum((s + eps_s)**2) / n

        losses = vmap(loss_per_sample)(x_t_batch, eps_over_sigma)
        return jnp.sum(losses)

    return loss_fn_per_batch

def setup_score_matching_kbT_loss_per_batch(energy_fn, k_b=1.0, T=1.0):
    def loss_fn_per_batch(flattened_args,batch):
        n_samples = batch.shape[0]
        def log_propability_unnormalized(x):
            return -energy_fn(x, flattened_args)
        
        current_score = grad(log_propability_unnormalized)
        current_score2 = hessian(log_propability_unnormalized)
        
        current_score_loss_per_sample = lambda x: (jnp.trace(current_score2(x))/(k_b*T) + 1/2 * jnp.sum(current_score(x)**2)/(k_b*T)**2)/n_samples   
        return jnp.sum(vmap(current_score_loss_per_sample)(batch))
    return loss_fn_per_batch

def setup_score_matching_kbT_local_gradient_per_batch(energy_fn, k_b=1.0, T=1.0):
    def gradient_fn_per_batch(params, batch, subkey_gradient=None):
        """
        params:     array of shape (P,)
        batch:      array of shape (N, D)
        returns:    array of shape (P,)
        """
        kBT = k_b * T  # just to shorten expressions

        def per_sample_grad(x):
            # 1) ∇ₓE(x; params)   shape (D,)
            dE_dx = grad(energy_fn, argnums=0)(x, params)

            # 3) ∂²E/(∂p ∂x):   shape (P, D)
            d2E_dpdx = jacfwd(lambda xx: grad(energy_fn, argnums=1)(xx, params))(x)

            # 4) ∂³E/(∂p ∂x ∂x): shape (P, D, D)
            d3E = jacfwd(lambda xx: jacfwd(lambda yy: grad(energy_fn, argnums=1)(yy, params))(xx))(x)

            # 5) trace over the last two dims → shape (P,)
            trace_term = jnp.trace(d3E, axis1=1, axis2=2)

            # 6) dot the mixed second derivative with ∇ₓE → shape (P,)
            dot_term = jnp.dot(d2E_dpdx, dE_dx)

            # combine
            return -trace_term / kBT + dot_term / kBT**2

        # 7) vectorize over the batch and take the mean
        grads = vmap(per_sample_grad)(batch)    # shape (N, P)
        return jnp.mean(grads, axis=0)          # shape (P,)
    return gradient_fn_per_batch

########################################################################################
# CD-1 gradient
########################################################################################

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
    return -avg_grad_diff/(D**2*dt/2) # the negative sign ensures that this is in fact the same gradient as in eq. (1) in the paper "Connections Between Score Matching, Contrastive Divergence, and Pseudolikelihood for Continuous-Valued Variables" by Hyvärinen




# #######################################################################################
# Optimization
# #######################################################################################


def run_optimization(loss_fn_per_batch, 
                     params_initial, 
                     sampler, 
                     gradient_fn_per_batch=None,
                     mask=None, 
                     key=jr.PRNGKey(0), 
                     learning_rate=0.001,
                     n_epochs=20000, 
                     maximize=False, 
                     window_size=1000, 
                     tolerance=1e-16, 
                     patience=50,
                     constraint_indices=None,
                     lr_decay_rate=1.,
                     lr_decay_steps=100):
    """
    Runs gradient-based optimization using mini-batches with an early stopping criterion based
    on the moving average of the loss.

    Args:
        loss_fn_per_batch: Callable that computes the loss for a batch of samples.
        params_initial: Initial parameters (any pytree).
        samples: Array of training samples.
        gradient_fn_per_batch: Optional callable to compute gradients.
        mask: Optional mask to apply to the gradients.
        key: PRNG key for random batch sampling.
        learning_rate: Step size for the Adam optimizer.
        n_epochs: Maximum number of optimization steps.
        maximize: Whether to maximize (rather than minimize) the loss.
        window_size: Number of recent epochs over which to compute the moving average.
        tolerance: Minimum improvement required in the moving average to reset the patience counter.
        patience: Number of consecutive windows without sufficient improvement before stopping.

    Returns:
        params_history: List of parameter values at each optimization step.
        loss_history: List of loss values at each optimization step.
    """
    #########################################################
    # Define an exponential decay learning rate schedule
    lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=lr_decay_steps,  # number of steps after which to decay
            decay_rate=lr_decay_rate,
            staircase=False
        )
        
    # Initialize the optimizer with the learning rate schedule
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params_initial)
    if mask is None:
        mask = jnp.ones(params_initial.shape[0])
    #########################################################
    

    @partial(jax.jit)
    def training_step(params, opt_state, samples_batched, subkey_gradient):
        # Define loss for current batch
        loss_fn = lambda params_current: loss_fn_per_batch(params_current, samples_batched)
        loss_val = loss_fn(params)
        # Compute gradients
        if gradient_fn_per_batch is None:
            dparams = jax.grad(loss_fn)(params)
        else:
            dparams = gradient_fn_per_batch(params, samples_batched, subkey_gradient)
        if maximize:
            dparams = -dparams
        dparams = dparams * mask
        updates, opt_state = optimizer.update(dparams, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    params_history = []
    loss_history = []
    best_moving_avg = jnp.inf
    patience_counter = 0

    params = params_initial

    for epoch in range(n_epochs):
        key, subkey_epoch, subkey_gradient = jr.split(key,3)
        samples_batched = sampler(subkey_epoch)
        params, opt_state, loss = training_step(params, opt_state, samples_batched, subkey_gradient)
        
        if constraint_indices is not None:
            # Define a small positive constant epsilon to ensure strict positivity.
            epsilon = .01
            # Project the subset of parameters (e.g., those at index 0:10) to be at least epsilon:
            params = params.at[constraint_indices].set(jnp.maximum(params[constraint_indices], epsilon))

        params_history.append(params)
        loss_history.append(loss)

        # Check convergence if we have enough history
        if epoch % 100 == 0 and len(loss_history) >= window_size:
            current_moving_avg = jnp.mean(jnp.array(loss_history[-window_size:]))
            # If the moving average hasn't improved by the tolerance, increase the counter
            if current_moving_avg < best_moving_avg - tolerance:
                best_moving_avg = current_moving_avg
                patience_counter = 0
            else:
                patience_counter += 1

        # Optionally print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss:.4f}")
        # If we haven't seen sufficient improvement for 'patience' consecutive windows, stop training
        if patience_counter >= patience:
            print(f"Convergence reached at epoch {epoch}. Stopping optimization.")
            break

    return params_history, loss_history


def run_optimization_multi_gpu_sampler(
    loss_fn_per_batch, 
    params_initial, 
    sampler, 
    gradient_fn_per_batch=None,
    mask=None, 
    key=jr.PRNGKey(0), 
    learning_rate=0.001,
    n_epochs=20000, 
    maximize=False, 
    window_size=1000, 
    tolerance=1e-16, 
    patience=50,
    constraint_indices=None,
    lr_decay_rate=1.,
    lr_decay_steps=100,
    batch_size=128
):
    """
    Multi-GPU optimization using pmap and a sampler function for batches.

    Args:
        loss_fn_per_batch: Callable that computes the loss for a batch of samples.
        params_initial: Initial parameters (any pytree).
        sampler: Callable(key, batch_size) -> batch of samples.
        gradient_fn_per_batch: Optional callable to compute gradients.
        mask: Optional mask to apply to the gradients.
        key: PRNG key for random batch sampling.
        learning_rate: Step size for the Adam optimizer.
        n_epochs: Maximum number of optimization steps.
        maximize: Whether to maximize (rather than minimize) the loss.
        window_size: Number of recent epochs over which to compute the moving average.
        tolerance: Minimum improvement required in the moving average to reset the patience counter.
        patience: Number of consecutive windows without sufficient improvement before stopping.
        constraint_indices: Indices of parameters to constrain to be positive.
        lr_decay_rate: Learning rate decay rate.
        lr_decay_steps: Steps between learning rate decays.
        batch_size: Total batch size (will be split across devices).

    Returns:
        params_history: List of parameter values at each optimization step.
        loss_history: List of loss values at each optimization step.
    """
    num_devices = jax.local_device_count()
    local_batch_size = batch_size // num_devices

    lr_schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=lr_decay_steps,
        decay_rate=lr_decay_rate,
        staircase=False
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params_initial)
    if mask is None:
        mask = jnp.ones(params_initial.shape[0])

    # Replicate params and opt_state across devices
    params = jax.device_put_replicated(params_initial, jax.devices())
    opt_state = jax.device_put_replicated(opt_state, jax.devices())
    mask = jax.device_put_replicated(mask, jax.devices())

    def training_step(params, opt_state, batch, mask, subkey_gradient):
        loss_fn = lambda p: loss_fn_per_batch(p, batch)
        loss_val = loss_fn(params)
        if gradient_fn_per_batch is None:
            dparams = jax.grad(loss_fn)(params)
        else:
            dparams = gradient_fn_per_batch(params, batch, subkey_gradient)
        if maximize:
            dparams = -dparams
        dparams = dparams * mask
        updates, opt_state = optimizer.update(dparams, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    p_training_step = jax.pmap(training_step, in_axes=(0, 0, 0, 0, 0))

    params_history = []
    loss_history = []
    best_moving_avg = jnp.inf
    patience_counter = 0

    for epoch in range(n_epochs):
        key, subkey_epoch, subkey_gradient = jr.split(key, 3)
        # Split keys for each device
        device_keys = jr.split(subkey_epoch, num_devices)
        device_grad_keys = jr.split(subkey_gradient, num_devices)
        # Sample a batch for each device
        batches = [sampler(device_keys[i], local_batch_size) for i in range(num_devices)]
        # Stack batches for pmap
        batches = jax.device_put_sharded(batches, jax.devices())
        # Run training step
        params, opt_state, loss = p_training_step(params, opt_state, batches, mask, device_grad_keys)

        # Optionally enforce constraints
        if constraint_indices is not None:
            epsilon = 0.01
            params = params.at[:, constraint_indices].set(jnp.maximum(params[:, constraint_indices], epsilon))

        # Aggregate loss and params
        loss_val = jax.device_get(loss).mean()
        loss_history.append(loss_val)
        params_history.append(jax.device_get(params))

        # Early stopping
        if epoch % 100 == 0 and len(loss_history) >= window_size:
            current_moving_avg = jnp.mean(jnp.array(loss_history[-window_size:]))
            if current_moving_avg < best_moving_avg - tolerance:
                best_moving_avg = current_moving_avg
                patience_counter = 0
            else:
                patience_counter += 1

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss_val:.4f}")
        if patience_counter >= patience:
            print(f"Convergence reached at epoch {epoch}. Stopping optimization.")
            break

    return params_history, loss_history