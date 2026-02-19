import jax
from jax import grad, vmap, hessian, jacfwd
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import optax
from collections import deque


########################################################################################
# Score matching 
########################################################################################
def setup_score_matching_kbT_loss_per_batch(energy_fn, k_b=1.0, T=1.0):
    """
    Construct a score matching loss function using automatic differentiation.

    Implements the score matching objective from Hyvarinen (2005), adapted to
    a Boltzmann distribution at temperature T. The loss for each sample x is:

        L(x) = (1 / k_b T) * tr(H(x)) + (1 / 2 (k_b T)^2) * ||s(x)||^2

    where s(x) = grad log p(x) is the score (gradient of the unnormalized
    log-density) and H(x) is the Hessian of the log-density, both computed
    via JAX autodiff from the provided energy function.

    Args:
        energy_fn: Callable (x, params) -> scalar energy. The unnormalized
            log-probability is defined as -energy_fn(x, params).
        k_b: Boltzmann constant (default 1.0).
        T: Temperature (default 1.0).

    Returns:
        loss_fn_per_batch: Callable (flattened_args, batch) -> scalar loss
            averaged over the batch, suitable for use with gradient-based
            optimizers.
    """
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


def setup_score_matching_kbT_loss_analytical(gradient_fn, trace_hessian_fn, gradient_wrt_params_fn, trace_hessian_wrt_params_fn, k_b=1.0, T=1.0):
    """
    Analytical version of score matching loss using precomputed analytical derivatives.
    Much more efficient than automatic differentiation.
    
    Args:
        gradient_fn: Analytical gradient function ∇E(x, params)
        trace_hessian_fn: Analytical trace of Hessian function Tr(∇²E(x, params))  
        gradient_wrt_params_fn: Analytical ∂(∇E)/∂params function
        trace_hessian_wrt_params_fn: Analytical ∂(Tr(∇²E))/∂params function
        k_b: Boltzmann constant
        T: Temperature
    """
    def loss_fn_per_batch(flattened_args, batch):
        n_samples = batch.shape[0]
        
        def score_loss_per_sample(x):
            # Score function: ∇ log p = -∇E / (k_b * T)
            score = -gradient_fn(x, flattened_args) / (k_b * T)
            
            # Trace of Hessian of log p: -Tr(∇²E) / (k_b * T)  
            trace_hess_log_p = -trace_hessian_fn(x, flattened_args) / (k_b * T)
            
            # Score matching loss: Tr(∇²log p) + (1/2)||∇log p||²
            return (trace_hess_log_p + 0.5 * jnp.sum(score**2)) / n_samples
        
        return jnp.sum(vmap(score_loss_per_sample)(batch))
    
    def loss_gradient_fn_per_batch(flattened_args, batch, subkey_gradient=None):
        """Analytical gradient of the loss w.r.t. parameters"""
        n_samples = batch.shape[0]
        
        def score_loss_gradient_per_sample(x):
            # Current values
            score = -gradient_fn(x, flattened_args) / (k_b * T)  # Shape: [n_oscillators]
            trace_hess_log_p = -trace_hessian_fn(x, flattened_args) / (k_b * T)  # Scalar 
            
            # Parameter derivatives
            grad_wrt_params = -gradient_wrt_params_fn(x, flattened_args) / (k_b * T)  # Shape: [n_oscillators, n_params]
            trace_hess_wrt_params = -trace_hessian_wrt_params_fn(x, flattened_args) / (k_b * T)  # Shape: [n_params]
            
            # Gradient of score matching loss w.r.t. parameters
            # ∂/∂θ [Tr(∇²log p) + (1/2)||∇log p||²]
            # = ∂(Tr(∇²log p))/∂θ + ∇log p · ∂(∇log p)/∂θ
            
            # First term: gradient of trace of Hessian
            grad_trace_term = trace_hess_wrt_params
            
            # Second term: gradient of ||score||² = 2 * score · ∂score/∂θ
            # score is [n_oscillators], grad_wrt_params is [n_oscillators, n_params]
            # Result should be [n_params]
            grad_score_norm_term = jnp.sum(score[:, None] * grad_wrt_params, axis=0)
            
            return (grad_trace_term + grad_score_norm_term) / n_samples
        
        return jnp.sum(vmap(score_loss_gradient_per_sample)(batch), axis=0)
    
    return loss_fn_per_batch, loss_gradient_fn_per_batch



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
                     lr_decay_rate=0.99,
                     lr_decay_steps=10):
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
        tolerance: Minimum RELATIVE improvement required in the moving average to reset the patience counter.
                   E.g., tolerance=0.001 means 0.1% improvement required.
        patience: Number of consecutive windows without sufficient improvement before stopping.

    Returns:
        params_history: List of parameter values at each optimization step.
        loss_history: List of loss values at each optimization step.
        best_params: Parameters that achieved the best (lowest) loss.
        best_loss: The best (lowest) loss value achieved.
        best_epoch: The epoch at which the best loss was achieved.
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

    # Sparse history for plotting (every 100 epochs) - bounded memory
    params_history = []
    loss_history = []
    
    # Fixed-size window for early stopping - constant memory regardless of epochs
    loss_window = deque(maxlen=window_size)
    best_moving_avg = float('inf')
    patience_counter = 0
    
    # Track best parameters throughout optimization
    best_loss = jnp.inf if not maximize else -jnp.inf
    best_params = params_initial
    best_epoch = 0

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

        # Store history sparsely (every 100 epochs) to save memory
        if epoch % 100 == 0:
            params_history.append(params)
            loss_history.append(loss)
        
        # Update best parameters if current loss is better
        if (not maximize and loss < best_loss) or (maximize and loss > best_loss):
            best_loss = loss
            best_params = params.copy()  # Make a copy to avoid reference issues
            best_epoch = epoch

        # Add to fixed-size window for early stopping (deque auto-drops old values)
        loss_window.append(float(loss))

        # Check convergence if we have enough history
        if epoch % 100 == 0 and len(loss_window) >= window_size:
            current_moving_avg = sum(loss_window) / len(loss_window)
            # If the moving average hasn't improved by the relative tolerance, increase the counter
            # Use relative improvement: (best - current) / |best| > tolerance
            if best_moving_avg == 0:
                # Avoid division by zero
                relative_improvement = abs(current_moving_avg)
            else:
                relative_improvement = (best_moving_avg - current_moving_avg) / abs(best_moving_avg)
            
            if relative_improvement > tolerance:
                best_moving_avg = current_moving_avg
                patience_counter = 0
            else:
                patience_counter += 1

        # Optionally print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss:.4f} - Best Loss: {best_loss:.4f} (Epoch {best_epoch})")
        # If we haven't seen sufficient improvement for 'patience' consecutive windows, stop training
        if patience_counter >= patience:
            print(f"Convergence reached at epoch {epoch}. Stopping optimization.")
            break

    return params_history, loss_history, best_params, best_loss, best_epoch

