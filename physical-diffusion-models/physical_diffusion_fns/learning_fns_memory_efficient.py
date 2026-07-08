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
