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
def setup_score_matching_LOSS_per_batch(energy_fn, k_b=1.0, T=1.0):
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

########################################################################################
# Force matching - gradient using automatic differentiation
# The analytic version below is computationally more efficient. 
########################################################################################
def setup_force_matching_GRADIENT_per_batch_using_AD(energy_fn, k_b=1.0, T=1.0):
    """
    Constructs the parameter gradient of the score matching loss using
    automatic differentiation (``jacfwd`` / ``grad``).

    For a Boltzmann distribution p(x) ∝ exp(-E(x;θ) / kT), the implicit
    score matching loss (Hyvärinen 2005) is

        L(θ) = (1/kT) Tr(∇²_x E) − (1/2kT²) ||∇_x E||²

    This function returns a callable that computes ∂L/∂θ averaged over a
    mini-batch, by evaluating three derivative objects per sample via AD:

        ∇_x E,   ∂²E/∂θ∂x,   ∂³E/∂θ∂x²

    and combining them as

        ∂L/∂θ = −(1/kT) Tr(∂³E/∂θ∂x²) + (1/kT²) (∂²E/∂θ∂x) · ∇_x E
        
    Force matching refers to the fact that the gradient is computed using the force, which is the negative gradient of the energy.

    Args:
        energy_fn: Callable (x, params) -> scalar energy.
        k_b: Boltzmann constant (default 1.0).
        T: Temperature (default 1.0).

    Returns:
        gradient_fn_per_batch: Callable (params, batch, subkey_gradient=None)
            -> parameter gradient of shape (P,), averaged over the batch.
    """
    def gradient_fn_per_batch(params, batch, subkey_gradient=None):
        """
        Compute the batch-averaged parameter gradient of the score matching loss.

        Args:
            params: Flattened parameter vector, shape (P,).
            batch: Sample array, shape (N, D).
            subkey_gradient: Unused; accepted for interface compatibility with
                stochastic gradient estimators.

        Returns:
            Array of shape (P,) — mean ∂L/∂θ over the batch.
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
# Force matching - loss and gradient using analytical derivatives
########################################################################################
def setup_force_matching_LOSS_and_GRADIENT_per_batch_analytical(gradient_fn, trace_hessian_fn, gradient_wrt_params_fn, trace_hessian_wrt_params_fn, k_b=1.0, T=1.0):
    """
    Constructs the score matching loss **and** its parameter gradient using
    hand-derived (analytical) derivative functions, bypassing automatic
    differentiation entirely.  This is the performant counterpart of
    ``setup_force_matching_GRADIENT_per_batch_using_AD``; the two produce
    identical results, but this version avoids the cost of ``jacfwd`` and
    third-order AD by delegating to precomputed derivative closures.

    For p(x) ∝ exp(−E(x;θ) / kT), the implicit score matching loss
    (Hyvärinen 2005) is

        L(θ) = (1/kT) Tr(∇²_x E) − (1/2kT²) ||∇_x E||²

    and its parameter gradient decomposes as

        ∂L/∂θ = −(1/kT) ∂Tr(∇²_x E)/∂θ
                 + (1/kT²) [∂(∇_x E)/∂θ]ᵀ · ∇_x E

    Both quantities are averaged over a mini-batch of samples.

    The four derivative callables are typically produced by
    ``setup_duffing_network_analytical_derivatives`` in ``network_fns.py``.

    Args:
        gradient_fn: Callable (x, params) -> array (D,).
            Analytical energy gradient ∇_x E.
        trace_hessian_fn: Callable (x, params) -> scalar.
            Analytical trace of the energy Hessian, Tr(∇²_x E).
        gradient_wrt_params_fn: Callable (x, params) -> array (D, P).
            Jacobian of the energy gradient w.r.t. parameters, ∂(∇_x E)/∂θ.
        trace_hessian_wrt_params_fn: Callable (x, params) -> array (P,).
            Gradient of the Hessian trace w.r.t. parameters,
            ∂Tr(∇²_x E)/∂θ.
        k_b: Boltzmann constant (default 1.0).
        T: Temperature (default 1.0).

    Returns:
        loss_fn_per_batch:
            Callable (flattened_args, batch) -> scalar loss averaged over
            the batch.
        loss_gradient_fn_per_batch:
            Callable (flattened_args, batch, subkey_gradient=None) -> array
            (P,), the batch-averaged parameter gradient ∂L/∂θ.
    """
    def loss_fn_per_batch(flattened_args, batch):
        n_samples = batch.shape[0]
        
        def score_loss_per_sample(x):
            score = -gradient_fn(x, flattened_args) / (k_b * T)
            trace_hess_log_p = -trace_hessian_fn(x, flattened_args) / (k_b * T)
            return (trace_hess_log_p + 0.5 * jnp.sum(score**2)) / n_samples
        
        return jnp.sum(vmap(score_loss_per_sample)(batch))
    
    def loss_gradient_fn_per_batch(flattened_args, batch, subkey_gradient=None):
        """
        Batch-averaged parameter gradient of the score matching loss.

        Args:
            flattened_args: Flat parameter vector, shape (P,).
            batch: Sample array, shape (N, D).
            subkey_gradient: Unused; accepted for interface compatibility
                with stochastic gradient estimators.

        Returns:
            Array of shape (P,) — mean ∂L/∂θ over the batch.
        """
        n_samples = batch.shape[0]
        
        def score_loss_gradient_per_sample(x):
            score = -gradient_fn(x, flattened_args) / (k_b * T)
            grad_wrt_params = -gradient_wrt_params_fn(x, flattened_args) / (k_b * T)
            
            grad_trace_term = -trace_hessian_wrt_params_fn(x, flattened_args) / (k_b * T)
            grad_score_norm_term = jnp.sum(score[:, None] * grad_wrt_params, axis=0)
            
            return (grad_trace_term + grad_score_norm_term) / n_samples
        
        return jnp.sum(vmap(score_loss_gradient_per_sample)(batch), axis=0)
    
    return loss_fn_per_batch, loss_gradient_fn_per_batch


def solve_quadratic_score_matching(
    params_initial,
    sampler,
    hessian_fn,
    gradient_fn_per_batch,
    loss_fn_per_batch=None,
    mask=None,
    key=jr.PRNGKey(0),
    maximize=False,
    constraint_indices=None,
    epsilon=0.01,
    hessian_reg=1e-6,
):
    """
    One-step Newton solve for the score matching loss (quadratic in params).

    Solves θ* = θ - (H + λI)⁻¹ ∇L in a single step. No learning rate or iteration needed.

    Args:
        params_initial: Initial parameter vector, shape (P,).
        sampler: Callable (key) -> batch of samples, shape (N, D).
        hessian_fn: Callable (x, flattened_args) -> Hessian matrix (P, P).
        gradient_fn_per_batch: Callable (params, batch, subkey=None) -> gradient (P,).
        loss_fn_per_batch: Optional, for computing loss at solution.
        mask: Optional mask of shape (P,). Zeros out updates for masked params.
        key: PRNG key for sampling.
        maximize: If True, maximize instead of minimize (negate gradient).
        constraint_indices: Optional indices to project to >= epsilon.
        epsilon: Minimum value for constrained params (default 0.01).
        hessian_reg: Ridge regularization added to Hessian (H + λI) for numerical
            stability when H is singular or ill-conditioned (default 1e-6).

    Returns:
        params_history: List with single element [params] (for compatibility).
        loss_history: List with single element [loss] if loss_fn_per_batch provided.
        params: Optimal parameters from one Newton step.
        best_loss: Loss at solution (or None).
        best_epoch: 0 (for compatibility).
    """
    subkey_sampler, = jr.split(key, 1)
    batch = sampler(subkey_sampler)

    H = hessian_fn(batch, params_initial)
    g = gradient_fn_per_batch(params_initial, batch, subkey_gradient=None)

    if maximize:
        g = -g

    if mask is not None:
        g = g * mask
        # Zero out Hessian rows/cols for masked params; set H[i,i]=1 so Δθ[i]=0
        H = H * mask[:, None] * mask[None, :]
        H = H + (1.0 - mask) * jnp.eye(H.shape[0])

    # Add ridge regularization for numerical stability (singular/ill-conditioned H)
    H = H + hessian_reg * jnp.eye(H.shape[0])

    # Solve H @ Δθ = -g  =>  Δθ = -H⁻¹ @ g
    delta_params = -jnp.linalg.solve(H, g)

    if mask is not None:
        delta_params = delta_params * mask

    params = params_initial + delta_params

    if constraint_indices is not None:
        params = params.at[constraint_indices].set(
            jnp.maximum(params[constraint_indices], epsilon)
        )

    loss = loss_fn_per_batch(params, batch) if loss_fn_per_batch is not None else None
    params_history = [params]
    loss_history = [loss] if loss is not None else []

    return params_history, loss_history, params, loss, 0


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

