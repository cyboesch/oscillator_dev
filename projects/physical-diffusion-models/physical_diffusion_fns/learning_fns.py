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
                     constraint_indices=None):
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
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params_initial)
    if mask is None:
        mask = jnp.ones(params_initial.shape[0])

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


def run_optimization_multi_gpu(loss_fn_per_batch, 
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
                     patience=50,
                     constraint_indices=None):
    """
    Runs gradient-based optimization using mini-batches with an early stopping criterion.
    """
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params_initial)
    if mask is None:
        mask = jax.numpy.ones(params_initial.shape[0])

    # Define training step without jax.jit, as pmap will handle compilation
    def training_step(params, opt_state, batch, batch_size, key):
        key, subkey = jr.split(key)
        # Define loss for current batch; note: batch here is per-device (e.g. shape (local_batch_size, ...))
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

    # Get number of devices and determine per-device batch size
    num_devices = jax.local_device_count()  # e.g., 2
    local_batch_size = batch_size // num_devices

    # Replicate parameters and optimizer state across devices
    params = jax.device_put_replicated(params_initial, jax.devices())
    opt_state = jax.device_put_replicated(opt_state, jax.devices())

    # Wrap the training_step with pmap.
    p_training_step = jax.pmap(training_step, static_broadcasted_argnums=(3,))

    params_history = []
    loss_history = []
    best_moving_avg = jax.numpy.inf
    patience_counter = 0

    for epoch in range(n_epochs):
        key, subkey = jr.split(key)
        # Sample a global batch of size `batch_size`
        n_samples = len(samples)
        idx = jr.randint(subkey, (batch_size,), 0, n_samples)
        global_batch = samples[idx]
        # Reshape the global batch to have shape (num_devices, local_batch_size, ...)
        sharded_batch = global_batch.reshape((num_devices, local_batch_size, *global_batch.shape[1:]))
        
        # Split the key for each device
        device_keys = jr.split(key, num_devices)
        params, opt_state, device_keys, loss = p_training_step(params, opt_state, sharded_batch, local_batch_size, device_keys)
        
        # Optionally, if you need to enforce constraints after each update
        if constraint_indices is not None:
            epsilon = 0.01
            params = params.at[:, constraint_indices].set(jax.numpy.maximum(params[:, constraint_indices], epsilon))

        # Here, loss is per-device; you might average or sum them as needed.
        loss_val = jax.numpy.mean(loss)
        loss_history.append(loss_val)
        params_history.append(params)

        # Convergence check (using a moving average over epochs)
        if epoch % 100 == 0 and len(loss_history) >= window_size:
            current_moving_avg = jax.numpy.mean(jax.numpy.array(loss_history[-window_size:]))
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




# --------------------------------------------------------------------------
# main function
# --------------------------------------------------------------------------
def run_optimization_pjit(
        loss_fn_per_batch,                    # params, batch -> scalar loss
        params_initial,                       # PyTree of arrays  (leading axis = shardable)
        samples,                              # (N, ...) full dataset (on host)
        gradient_fn_per_batch=None,  
        mask=None,
        key=jr.PRNGKey(0),
        batch_size=128,
        learning_rate=1e-3,
        n_epochs=20_000,
        maximize=False,
        window_size=1_000,
        tolerance=1e-16,
        patience=50,
        constraint_indices=None               # optional positivity constraint
    ):
    """Adam optimisation with parameter *and* batch sharding via `pjit`."""

    # ----------------------------------------------------------------------
    # 0.  build 1‑D mesh over all local GPUs
    # ----------------------------------------------------------------------
    devices   = jax.devices()
    n_devices = len(devices)
    mesh      = Mesh(devices, ('dp',))        # single axis called 'dp'
    param_ps  = P('dp',)                      # leading axis sharded
    batch_ps  = P('dp',)                      # ditto for batch
    
    # helper to replicate if axis too small
    def maybe_shard(x):
        if x.ndim and x.shape[0] >= n_devices:
            # 1. split leading axis evenly across devices
            per_dev = [x[i::n_devices] for i in range(n_devices)]  # list length = n_devices

            # 2. place each slice onto its target GPU
            per_dev = [
                jax.device_put(arr, device=devices[i])   # NOW each slice lives on cuda:0, cuda:1, ...
                for i, arr in enumerate(per_dev)
            ]

            # 3. build a sharded array whose sharding matches param_ps
            return jax.make_array_from_single_device_arrays(
                x.shape,
                jax.sharding.NamedSharding(mesh, param_ps),
                per_dev,
            )
        else:
            # small leading axis → just replicate
            return jax.device_put_replicated(x, devices)

    params = jax.tree_map(maybe_shard, params_initial)

    # ----------------------------------------------------------------------
    # 1.  optimiser state
    # ----------------------------------------------------------------------
    opt      = optax.adam(learning_rate)
    opt_state = jax.device_put_replicated(opt.init(params_initial), devices)

    # ----------------------------------------------------------------------
    # 2.  pjit‑compiled step -------------------------------------------------
    # ----------------------------------------------------------------------
    if gradient_fn_per_batch is None:
        val_and_grad = jax.value_and_grad(loss_fn_per_batch)
    else:
        def val_and_grad(p, b):
            return loss_fn_per_batch(p, b), gradient_fn_per_batch(p, b)

    @pjit(
    in_shardings=(param_ps, batch_ps, None, None, None),
    out_shardings=(param_ps, P(), param_ps, P()),
    )
    def train_step(params, batch, opt_state, mask, key):
        loss, grads = val_and_grad(params, batch)
        if maximize:
            grads = jax.tree_map(lambda g: -g, grads)
        grads = jax.tree_map(lambda g, m: g * m, grads, mask)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, loss, opt_state, key

    # ----------------------------------------------------------------------
    # 3.  main loop ----------------------------------------------------------
    # ----------------------------------------------------------------------
    mask   = jnp.ones(params_initial.shape[0])
    best   = jnp.inf
    wait   = 0
    loss_history = []

    for epoch in range(n_epochs):
        key, sub = jr.split(key)
        idx      = jr.randint(sub, (batch_size,), 0, len(samples))
        global_b = samples[idx]

        # batch must be (n_devices, local_batch, ...)
        local_bs = batch_size // n_devices
        batch    = global_b.reshape(n_devices, local_bs, *global_b.shape[1:])
        batch    = jax.device_put_sharded(list(batch), devices)

        params, loss, opt_state, key = train_step(params, batch, opt_state,
                                                  mask, key)
        loss   = jax.device_get(loss).mean()
        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"epoch {epoch:>6d}  loss={loss:.4e}")

        # simple moving‑average early‑stop
        if epoch % 100 == 0 and len(loss_history) >= window_size:
            mov = jnp.mean(jnp.array(loss_history[-window_size:]))
            if mov < best - tolerance:
                best, wait = mov, 0
            else:
                wait += 1
            if wait >= patience:
                print(f"stop @ epoch {epoch}")
                break

        # optional positivity constraint
        if constraint_indices is not None:
            eps = 1e-2
            def clip_fn(p):
                p = p.at[constraint_indices].set(jnp.maximum(
                        p[constraint_indices], eps))
                return p
            params = jax.tree_map(clip_fn, params)

    return params, loss_history