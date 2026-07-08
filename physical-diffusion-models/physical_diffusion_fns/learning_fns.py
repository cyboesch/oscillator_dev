import jax
from jax import grad, vmap, hessian, jacfwd
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.sparse.linalg import cg
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


# #######################################################################################
# Convex solve: matrix-free ridge-regularized conjugate gradient
# #######################################################################################

def solve_score_matching_cg(
    gradient_fn_per_batch,
    params_initial,
    batch,
    ridge_rel=1e-4,
    ridge_abs=0.0,
    constraint_indices=None,
    epsilon=0.01,
    cg_tol=1e-6,
    cg_maxiter=500,
    active_set_iters=5,
    n_probes=3,
    loss_fn_per_batch=None,
    key=jr.PRNGKey(0),
):
    """
    Solve one forward-time slice of score matching by directly solving the (convex,
    quadratic) normal equations, matrix-free — replacing the per-slice SGD loop.

    For an energy that is LINEAR in the parameters theta and a Boltzmann model with
    only visible units, the implicit score-matching loss is an exact convex quadratic

        L(theta) = 1/2 theta^T H theta - c^T theta,          H = grad^2_theta L >= 0,

    so its gradient is AFFINE:  g(theta) = grad_theta L = H theta - c.  Therefore

        c    = -g(0)                              (one gradient evaluation at theta=0)
        H v  = jvp(g, v)                          (one gradient-sized pass; H never formed)

    We solve the ridge-regularized system  (H + lambda I) theta = c  by conjugate
    gradient using only these two matrix-free products.  H is only PSD (and, on a finite
    batch, can be rank-deficient / ill-conditioned), so a Tikhonov ridge

        lambda = ridge_abs + ridge_rel * scale,   scale ~ mean eigenvalue of H

    (estimated with a few Hutchinson probes v^T H v / v^T v) makes the problem
    well-posed (H + lambda I > 0) and caps the condition number.  lambda*||theta||^2/2
    is equivalently a Gaussian prior (MAP score matching).

    If ``constraint_indices`` is given, the box constraint theta[idx] >= epsilon (a
    convex set) is enforced by active-set refinement: coordinates whose unconstrained
    solution violates the floor are pinned at epsilon and the reduced system is
    re-solved, repeated until the active set stabilises, with a final clamp as a
    feasibility safety net.

    Args:
        gradient_fn_per_batch: callable (theta, batch[, subkey]) -> grad_theta L, i.e.
            the affine gradient g(theta) = H theta - c (e.g. the one returned by
            ``setup_score_matching_kbT_loss_minimal_memory``).
        params_initial: flat parameter vector (P,); also used to warm-start CG.
        batch: one fixed sample of the noised marginal p_t, shape (S, D). Use S large
            enough (>~ P / n_oscillators) for H to be well-conditioned.
        ridge_rel, ridge_abs: Tikhonov ridge lambda = ridge_abs + ridge_rel*scale.
        constraint_indices, epsilon: box constraint theta[constraint_indices] >= epsilon.
        cg_tol, cg_maxiter: conjugate-gradient tolerance and iteration cap.
        active_set_iters: max active-set refinement passes.
        n_probes: Hutchinson probes used to estimate the eigenvalue scale for ridge_rel.
        loss_fn_per_batch: optional (theta, batch) -> scalar, for before/after diagnostics.
        key: PRNG key for the Hutchinson probes.

    Returns:
        (theta_star, info) where info is a dict of diagnostics (ridge, scale, n_active,
        active_set_iters_used, residual, and loss_before/loss_after if a loss is given).
    """
    g = lambda theta: gradient_fn_per_batch(theta, batch)
    zeros = jnp.zeros_like(params_initial)

    # Affine gradient => c = -g(0) and H v = jvp(g, v) (base point irrelevant).
    c = -g(zeros)

    def Hv(v):
        return jax.jvp(g, (zeros,), (v,))[1]

    # Estimate the eigenvalue scale of H for the relative ridge.
    scale = 1.0
    if ridge_rel > 0 and n_probes > 0:
        probes = []
        for _ in range(n_probes):
            key, sub = jr.split(key)
            v = jr.normal(sub, params_initial.shape)
            probes.append(jnp.vdot(v, Hv(v)) / jnp.vdot(v, v))
        scale = max(float(jnp.mean(jnp.stack(probes))), 0.0)
    lam = ridge_abs + ridge_rel * scale

    A = lambda v: Hv(v) + lam * v

    # Unconstrained ridge solve, warm-started from params_initial.
    theta, _ = cg(A, c, x0=params_initial, tol=cg_tol, maxiter=cg_maxiter)

    info = {"ridge": lam, "scale": scale, "n_active": 0, "active_set_iters_used": 0}

    # Active-set refinement for the box constraint theta[constraint_indices] >= epsilon.
    if constraint_indices is not None:
        active_prev = None
        for it in range(active_set_iters):
            active = theta[constraint_indices] < epsilon
            info["active_set_iters_used"] = it + 1
            if not bool(jnp.any(active)):
                break
            fixed_mask = jnp.zeros_like(theta, dtype=bool).at[constraint_indices].set(active)
            free_mask = ~fixed_mask
            fixed_vals = jnp.where(fixed_mask, epsilon, 0.0)

            # Reduced operator: A on the free block, identity on the pinned block (keeps it SPD).
            def A_free(v):
                out = A(jnp.where(free_mask, v, 0.0))
                return jnp.where(free_mask, out, v)

            rhs = jnp.where(free_mask, c - A(fixed_vals), 0.0)
            theta_free, _ = cg(A_free, rhs, x0=jnp.where(free_mask, theta, 0.0),
                               tol=cg_tol, maxiter=cg_maxiter)
            theta = jnp.where(free_mask, theta_free, fixed_vals)

            cur = tuple(int(i) for i in jnp.nonzero(active)[0])
            if cur == active_prev:
                break
            active_prev = cur

        # Feasibility safety net.
        theta = theta.at[constraint_indices].set(jnp.maximum(theta[constraint_indices], epsilon))

    # Diagnostics. For a constrained solution the stationarity residual is measured on the
    # FREE coordinates only (the pinned coords carry a nonzero KKT multiplier by design), and
    # n_active counts coordinates sitting at the floor in the final solution.
    resid_vec = A(theta) - c
    if constraint_indices is not None:
        at_floor = theta[constraint_indices] <= epsilon + 1e-9
        info["n_active"] = int(jnp.sum(at_floor))
        pinned_mask = jnp.zeros_like(theta, dtype=bool).at[constraint_indices].set(at_floor)
        resid_vec = jnp.where(pinned_mask, 0.0, resid_vec)
    info["residual"] = float(jnp.linalg.norm(resid_vec))
    if loss_fn_per_batch is not None:
        info["loss_before"] = float(loss_fn_per_batch(params_initial, batch))
        info["loss_after"] = float(loss_fn_per_batch(theta, batch))
    return theta, info
