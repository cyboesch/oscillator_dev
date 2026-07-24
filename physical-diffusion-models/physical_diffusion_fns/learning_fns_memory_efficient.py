"""
Memory-efficient versions of learning functions for large-scale diffusion models.
These implementations avoid creating large intermediate tensors and use gradient accumulation.
"""

import jax
import jax.numpy as jnp


def setup_score_matching_kbT_loss_minimal_memory(
    gradient_fn, trace_hessian_fn, k_b=1.0, T=1.0
):
    """
    Minimal memory version that uses JAX autodiff instead of analytical parameter gradients.
    This avoids the need for large gradient_wrt_params matrices entirely.
    """

    @jax.checkpoint
    def loss_fn_per_batch(flattened_args, batch):
        n_samples = batch.shape[0]

        def score_loss_per_sample(x):
            score = -gradient_fn(x, flattened_args) / (k_b * T)
            trace_hess_log_p = -trace_hessian_fn(x, flattened_args) / (k_b * T)
            return (trace_hess_log_p + 0.5 * jnp.sum(score**2)) / n_samples

        # Process samples one by one
        def accumulate_loss(carry, x):
            total_loss = carry
            sample_loss = score_loss_per_sample(x)
            return total_loss + sample_loss, None

        final_loss, _ = jax.lax.scan(accumulate_loss, 0.0, batch)
        return final_loss

    def loss_gradient_fn_per_batch_minimal_memory(flattened_args, batch, subkey_gradient=None):
        """
        Use JAX autodiff on the loss function itself to avoid explicit parameter gradients.
        This is the most memory-efficient approach as it avoids all large intermediate matrices.
        """
        n_samples = batch.shape[0]

        # Define single sample loss for autodiff
        @jax.checkpoint
        def single_sample_loss(params, x):
            score = -gradient_fn(x, params) / (k_b * T)
            trace_hess_log_p = -trace_hessian_fn(x, params) / (k_b * T)
            return trace_hess_log_p + 0.5 * jnp.sum(score**2)

        # Get gradient function
        grad_fn = jax.grad(single_sample_loss, argnums=0)

        # Process samples one by one and accumulate gradients
        def accumulate_gradient(carry, x):
            accumulated_grad = carry
            sample_grad = grad_fn(flattened_args, x)
            return accumulated_grad + sample_grad, None

        # Use scan for memory efficiency
        initial_grad = jnp.zeros_like(flattened_args)
        final_grad, _ = jax.lax.scan(accumulate_gradient, initial_grad, batch)

        return final_grad / n_samples

    return loss_fn_per_batch, loss_gradient_fn_per_batch_minimal_memory


def setup_score_matching_kbT_loss_chunked(gradient_fn, trace_hessian_fn, k_b=1.0, T=1.0, chunk=1024):
    """
    Same loss/gradient as ``setup_score_matching_kbT_loss_minimal_memory`` but the
    gradient vmaps per-sample grads over chunks of the batch (scan over chunks) instead
    of scanning sample-by-sample. On GPU this is ~2x faster per gradient and per
    jvp-through-gradient (the CG matvec in solve_score_matching_cg), at the price of
    materializing a (chunk, P) intermediate: chunk=1024 with P~5.5e5 peaks at ~8.4 GB
    in float64 through the jvp; use chunk=512 if anything else is resident on the GPU.
    The loss itself stays sample-by-sample (it is called only a few times per slice).
    """
    loss_fn_per_batch, _ = setup_score_matching_kbT_loss_minimal_memory(
        gradient_fn, trace_hessian_fn, k_b=k_b, T=T
    )

    @jax.checkpoint
    def single_sample_loss(params, x):
        score = -gradient_fn(x, params) / (k_b * T)
        trace_hess_log_p = -trace_hessian_fn(x, params) / (k_b * T)
        return trace_hess_log_p + 0.5 * jnp.sum(score**2)

    grad_fn = jax.grad(single_sample_loss, argnums=0)
    vgrad = jax.vmap(grad_fn, in_axes=(None, 0))

    def gradient_fn_per_batch_chunked(flattened_args, batch, subkey_gradient=None):
        n = batch.shape[0]
        # batch shapes are static under jit, so this runs at trace time
        c = min(chunk, n)
        while n % c:
            c //= 2
        batch_r = batch.reshape(n // c, c, batch.shape[-1])

        def body(acc, xs):
            return acc + jnp.sum(vgrad(flattened_args, xs), axis=0), None

        total, _ = jax.lax.scan(body, jnp.zeros_like(flattened_args), batch_r)
        return total / n

    return loss_fn_per_batch, gradient_fn_per_batch_chunked


def setup_denoising_score_matching_loss_chunked(gradient_fn, k_b=1.0, T=1.0, chunk=512):
    """Denoising score matching (Vincent 2011) per-slice objective, chunked-vmap gradient.

    The model score is s_theta(x) = grad_x log p_theta = -gradient_fn(x, theta) / (k_b T), which is
    LINEAR in theta because the energy is linear in theta. DSM regresses that score onto the
    analytic score of the forward transition kernel:

        target = grad_{x_t} log q(x_t | x_0) = -(x_t - mean_factor x_0) / var_factor

    (supplied by ``sample_forward_pairs``). The per-slice loss

        L(theta) = (1 / 2N) sum_i || s_theta(x_t^i) - target_i ||^2

    is an EXACT convex quadratic in theta. Its Hessian, H = (1/(N (k_b T)^2)) sum_i J_i^T J_i with
    J_i = d(grad_x E)/d theta, is IDENTICAL to the implicit-score-matching Hessian (both come from
    the ||s_theta||^2 term); only the linear coefficient differs. So the same convex solver,
    exact-diagonal preconditioner (``exact_score_matching_diag``), ridge/prox regularization and
    prox chain all carry over unchanged — the solver picks up the DSM linear term automatically via
    c = -grad(0). Unlike implicit SM, DSM's target pins down the relative mass of well-separated
    modes, which is what fixes the 0-vs-1 imbalance.

    The batch is a tuple ``(x_t, target)`` (both shape (N_samples, N_dim)); pass ``x_t`` alone to
    ``exact_score_matching_diag`` for the preconditioner.
    """

    # x_t and target are packed into a single (n, 2*D) array so the scan/vmap has the SAME
    # single-array structure as the ISM chunked gradient. Scanning over a python tuple
    # (x_t, target) instead blocks XLA's fusion of the per-chunk (chunk, P) gradient tensor and
    # blows up memory under the jvp used for the CG matvec (empirically OOMs where ISM fits).
    @jax.checkpoint
    def single_sample_loss(params, xt_tg):
        d = xt_tg.shape[-1] // 2
        x_t = xt_tg[:d]
        target = xt_tg[d:]
        score = -gradient_fn(x_t, params) / (k_b * T)
        return 0.5 * jnp.sum((score - target) ** 2)

    def loss_fn_per_batch(flattened_args, batch):
        x_t, target = batch
        packed = jnp.concatenate([x_t, target], axis=-1)
        n = packed.shape[0]

        def accumulate(carry, row):
            return carry + single_sample_loss(flattened_args, row), None

        total, _ = jax.lax.scan(accumulate, 0.0, packed)
        return total / n

    grad_fn = jax.grad(single_sample_loss, argnums=0)
    vgrad = jax.vmap(grad_fn, in_axes=(None, 0))

    def gradient_fn_per_batch(flattened_args, batch, subkey_gradient=None):
        x_t, target = batch
        packed = jnp.concatenate([x_t, target], axis=-1)
        n = packed.shape[0]
        c = min(chunk, n)
        while n % c:
            c //= 2
        packed_r = packed.reshape(n // c, c, packed.shape[-1])

        def body(acc, rows):
            return acc + jnp.sum(vgrad(flattened_args, rows), axis=0), None

        total, _ = jax.lax.scan(body, jnp.zeros_like(flattened_args), packed_r)
        return total / n

    return loss_fn_per_batch, gradient_fn_per_batch
