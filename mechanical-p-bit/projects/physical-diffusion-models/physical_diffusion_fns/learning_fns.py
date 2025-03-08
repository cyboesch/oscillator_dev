import jax
from jax import grad, vmap, hessian
import jax.numpy as jnp
import jax.random as jr
import diffrax
from diffrax import ControlTerm, MultiTerm, ODETerm
from functools import partial
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
# Score matching gradient
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
    return -avg_grad_diff/(dt/2) # the negative sign ensures that this is in fact the same gradient as in eq. (1) in the paper "Connections Between Score Matching, Contrastive Divergence, and Pseudolikelihood for Continuous-Valued Variables" by Hyvärinen




########################################################################################
# Optimization
########################################################################################

def run_optimization(loss_fn_per_batch, 
                     params_initial, 
                     samples, 
                     gradient_fn_per_batch=None,
                     mask=None, 
                     key=jr.PRNGKey(0), 
                     batch_size=128,
                     learning_rate=0.001,
                     n_epochs=20000, 
                     maximize=False, 
                     window_size=1000, 
                     tolerance=1e-16, 
                     patience=50):
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
        batch_size: Number of samples per optimization step.
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
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params_initial)
    if mask is None:
        mask = jnp.ones(params_initial.shape[0])

    @partial(jax.jit, static_argnums=(3,))
    def training_step(params, opt_state, samples, batch_size, key):
        key, subkey = jr.split(key)
        n_samples = len(samples)
        idx = jr.randint(subkey, (batch_size,), 0, n_samples)
        batch = samples[idx]
        # Define loss for current batch
        loss_fn = lambda params_current: loss_fn_per_batch(params_current, batch)
        loss_val = loss_fn(params)
        # Compute gradients
        if gradient_fn_per_batch is None:
            dparams = jax.grad(loss_fn)(params)
        else:
            dparams = gradient_fn_per_batch(params, batch)
        if maximize:
            dparams = -dparams
        dparams = dparams * mask
        updates, opt_state = optimizer.update(dparams, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, key, loss_val

    params_history = []
    loss_history = []
    best_moving_avg = jnp.inf
    patience_counter = 0

    params = params_initial

    for epoch in range(n_epochs):
        key, subkey = jr.split(key)
        params, opt_state, key, loss = training_step(params, opt_state, samples, batch_size, subkey)
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