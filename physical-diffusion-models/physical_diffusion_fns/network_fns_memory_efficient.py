"""
Memory-efficient versions of network functions that avoid creating large intermediate matrices.
"""

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial


def setup_duffing_network_analytical_derivatives_memory_efficient(connectivity, unflatten):
    """
    Memory-efficient version of analytical derivatives that avoids creating large matrices.
    
    Key optimizations:
    1. Compute parameter gradients on-demand rather than storing full matrices
    2. Use sparse operations where possible
    3. Avoid materializing [n_oscillators, n_params] matrices
    """
    
    # Keep the original gradient and trace hessian functions (they're already efficient)
    def gradient_self_oscillator(x, k_lin, k_duff, k_6, bias):
        return x + k_lin * x + k_duff * x**3 + k_6 * x**5 + bias
    
    def hessian_diag_self_oscillator(x, k_lin, k_duff, k_6, bias):
        return 1 +k_lin + 3 * k_duff * x**2 + 5 * k_6 * x**4
    
    def gradient_coupling_pair_wrt_x(x, y, c_lin, c_optomech, c_duff):
        return (2 * c_lin * x - 2 * c_lin * y + 
                2 * c_optomech * x * y + 
                c_duff * (x - y)**3)
    
    def gradient_coupling_pair_wrt_y(x, y, c_lin, c_optomech, c_duff):
        return (-2 * c_lin * x + 2 * c_lin * y + 
                c_optomech * x**2 - 
                c_duff * (x - y)**3)
    
    def hessian_diag_coupling_pair_wrt_x(x, y, c_lin, c_optomech, c_duff):
        return (2 * c_lin + 
                2 * c_optomech * y + 
                3 * c_duff * (x - y)**2)
    
    def hessian_diag_coupling_pair_wrt_y(x, y, c_lin, c_optomech, c_duff):
        return (2 * c_lin + 
                3 * c_duff * (x - y)**2)
    
    def gradient_duffing_network(x, flattened_args, connectivity):
        """Analytical gradient of the full network energy w.r.t. x"""
        k_lin, k_duffing, k_6, c_lin, c_optomech, c_duff, bias = unflatten(flattened_args)
        
        # Initialize gradient vector
        grad = jnp.zeros_like(x)
        
        # Add self oscillator contributions
        grad_self = vmap(gradient_self_oscillator)(x, k_lin, k_duffing, k_6, bias)
        grad = grad + grad_self
        
        # Add coupling contributions using vectorized operations
        def _compute_coupling_gradients(c_lin_pair, c_optomech_pair, c_duff_pair, pair):
            i, j = pair
            grad_i = gradient_coupling_pair_wrt_x(x[i], x[j], c_lin_pair, c_optomech_pair, c_duff_pair)
            grad_j = gradient_coupling_pair_wrt_y(x[i], x[j], c_lin_pair, c_optomech_pair, c_duff_pair)
            return jnp.array([i, j]), jnp.array([grad_i, grad_j])
        
        # Vectorize over all coupling pairs
        indices, grad_values = vmap(_compute_coupling_gradients)(c_lin, c_optomech, c_duff, connectivity)
        
        # Add coupling contributions to gradient using scatter_add
        grad = grad.at[indices[:, 0]].add(grad_values[:, 0])
        grad = grad.at[indices[:, 1]].add(grad_values[:, 1])
        
        return grad
    
    def trace_hessian_duffing_network(x, flattened_args, connectivity):
        """Analytical trace of Hessian of the full network energy"""
        k_lin, k_duffing, k_6, c_lin, c_optomech, c_duff, bias = unflatten(flattened_args)
        
        # Initialize trace vector (diagonal elements of Hessian)
        trace_hess = jnp.zeros_like(x)
        
        # Add self oscillator contributions to diagonal
        hess_diag_self = vmap(hessian_diag_self_oscillator)(x, k_lin, k_duffing, k_6, bias)
        trace_hess = trace_hess + hess_diag_self
        
        # Add coupling contributions to diagonal using vectorized operations
        def _compute_coupling_hessian_diag(c_lin_pair, c_optomech_pair, c_duff_pair, pair):
            i, j = pair
            hess_ii = hessian_diag_coupling_pair_wrt_x(x[i], x[j], c_lin_pair, c_optomech_pair, c_duff_pair)
            hess_jj = hessian_diag_coupling_pair_wrt_y(x[i], x[j], c_lin_pair, c_optomech_pair, c_duff_pair)
            return jnp.array([i, j]), jnp.array([hess_ii, hess_jj])
        
        # Vectorize over all coupling pairs
        indices, hess_values = vmap(_compute_coupling_hessian_diag)(c_lin, c_optomech, c_duff, connectivity)
        
        # Add coupling contributions to Hessian diagonal using scatter_add
        trace_hess = trace_hess.at[indices[:, 0]].add(hess_values[:, 0])
        trace_hess = trace_hess.at[indices[:, 1]].add(hess_values[:, 1])
        
        return jnp.sum(trace_hess)  # Return scalar trace
    
    def compute_parameter_gradient_dot_product_efficiently(x, flattened_args, score, connectivity):
        """
        Compute score · ∂(∇E)/∂params efficiently without creating the full matrix.
        
        This is the key optimization: instead of creating a [n_oscillators, n_params] matrix
        and then computing the dot product, we compute the result directly.
        """
        unflattened_params = unflatten(flattened_args)
        n_params = len(flattened_args)
        
        # Initialize result
        result = jnp.zeros(n_params)
        
        # Handle different parameter structures
        if len(unflattened_params) == 7:  # 6th order duffing coupling
            k_lin, k_duffing, k_6, c_lin, c_optomech, c_duff, bias = unflattened_params
            has_k6 = True
        elif len(unflattened_params) == 6:  # duffing coupling without k_6
            k_lin, k_duffing, c_lin, c_optomech, c_duff, bias = unflattened_params
            k_6 = jnp.zeros_like(k_lin)
            has_k6 = False
        elif len(unflattened_params) == 5:  # duffing optomech coupling
            k_lin, k_duffing, c_lin, c_optomech, bias = unflattened_params
            k_6 = jnp.zeros_like(k_lin)
            c_duff = jnp.zeros_like(c_lin)
            has_k6 = False
        else:
            raise ValueError(f"Unsupported parameter structure with {len(unflattened_params)} parameter groups")
        
        # Self oscillator contributions - compute dot products directly
        n_osc = len(x)
        
        # k_lin contributions: score[i] * x[i] for each oscillator i
        k_lin_contributions = score * x  # Element-wise multiplication
        result = result.at[:n_osc].set(k_lin_contributions)
        
        # k_duff contributions: score[i] * x[i]^3 for each oscillator i
        k_duff_contributions = score * (x**3)
        result = result.at[n_osc:2*n_osc].set(k_duff_contributions)
        
        # k_6 contributions (if present): score[i] * x[i]^5 for each oscillator i
        if has_k6:
            k_6_contributions = score * (x**5)
            result = result.at[2*n_osc:3*n_osc].set(k_6_contributions)
            bias_start = 3*n_osc
            c_lin_start = 4*n_osc
        else:
            bias_start = 2*n_osc
            c_lin_start = 3*n_osc
        
        # bias contributions: score[i] * 1 for each oscillator i
        result = result.at[n_params-n_osc:].set(score)  # bias is always last
        
        # Coupling contributions - this is more complex but we can still avoid the full matrix
        c_optomech_start = c_lin_start + len(c_lin)
        c_duff_start = c_optomech_start + len(c_optomech)
        
        def compute_coupling_contributions(c_lin_pair, c_optomech_pair, c_duff_pair, pair, pair_idx):
            i, j = pair
            x_i, x_j = x[i], x[j]
            score_i, score_j = score[i], score[j]
            
            # c_lin contribution
            c_lin_contrib = score_i * (2*x_i - 2*x_j) + score_j * (-2*x_i + 2*x_j)
            
            # c_optomech contribution  
            c_optomech_contrib = score_i * (2*x_i*x_j) + score_j * (x_i**2)
            
            # c_duff contribution (if exists)
            if len(c_duff) > 0:
                c_duff_contrib = score_i * (x_i - x_j)**3 + score_j * (-(x_i - x_j)**3)
            else:
                c_duff_contrib = 0.0
            
            return jnp.array([
                c_lin_start + pair_idx,
                c_optomech_start + pair_idx,
                c_duff_start + pair_idx if len(c_duff) > 0 else c_duff_start
            ]), jnp.array([
                c_lin_contrib,
                c_optomech_contrib, 
                c_duff_contrib
            ])
        
        # Vectorize over coupling pairs
        pair_indices = jnp.arange(len(connectivity))
        coupling_param_indices, coupling_contributions = vmap(compute_coupling_contributions)(
            c_lin, c_optomech, c_duff, connectivity, pair_indices
        )
        
        # Add coupling contributions
        for i in range(len(connectivity)):
            if has_k6 or len(c_duff) > 0:  # Only add c_duff if it exists
                result = result.at[coupling_param_indices[i, 0]].add(coupling_contributions[i, 0])  # c_lin
                result = result.at[coupling_param_indices[i, 1]].add(coupling_contributions[i, 1])  # c_optomech
                if len(c_duff) > 0:
                    result = result.at[coupling_param_indices[i, 2]].add(coupling_contributions[i, 2])  # c_duff
            else:
                result = result.at[coupling_param_indices[i, 0]].add(coupling_contributions[i, 0])  # c_lin
                result = result.at[coupling_param_indices[i, 1]].add(coupling_contributions[i, 1])  # c_optomech
        
        return result
    
    def gradient_wrt_params_of_trace_hessian_efficient(x, flattened_args, connectivity):
        """
        Compute ∂(Tr(∇²E))/∂params efficiently without creating large matrices.
        """
        unflattened_params = unflatten(flattened_args)
        n_params = len(flattened_args)
        
        # Initialize result
        result = jnp.zeros(n_params)
        
        # Handle different parameter structures
        if len(unflattened_params) == 7:  # 6th order duffing coupling
            k_lin, k_duffing, k_6, c_lin, c_optomech, c_duff, bias = unflattened_params
            has_k6 = True
        elif len(unflattened_params) == 6:  # duffing coupling without k_6
            k_lin, k_duffing, c_lin, c_optomech, c_duff, bias = unflattened_params
            k_6 = jnp.zeros_like(k_lin)
            has_k6 = False
        elif len(unflattened_params) == 5:  # duffing optomech coupling
            k_lin, k_duffing, c_lin, c_optomech, bias = unflattened_params
            k_6 = jnp.zeros_like(k_lin)
            c_duff = jnp.zeros_like(c_lin)
            has_k6 = False
        else:
            raise ValueError(f"Unsupported parameter structure")
        
        n_osc = len(x)
        
        # Self oscillator contributions to trace Hessian derivatives
        # ∂(Tr(∇²E_self))/∂k_lin = 1 for each oscillator
        result = result.at[:n_osc].set(1.0)
        
        # ∂(Tr(∇²E_self))/∂k_duff = 3*x²  for each oscillator
        result = result.at[n_osc:2*n_osc].set(3 * x**2)
        
        # ∂(Tr(∇²E_self))/∂k_6 = 5*x⁴ for each oscillator (if present)
        if has_k6:
            result = result.at[2*n_osc:3*n_osc].set(5 * x**4)
            c_lin_start = 3*n_osc + n_osc  # After k_lin, k_duff, k_6, bias
        else:
            c_lin_start = 2*n_osc + n_osc  # After k_lin, k_duff, bias
        
        # bias contributions to trace Hessian: 0 (bias doesn't affect second derivatives)
        # (already initialized to zero)
        
        # Coupling contributions to trace Hessian derivatives
        c_optomech_start = c_lin_start + len(c_lin)
        c_duff_start = c_optomech_start + len(c_optomech)
        
        def compute_coupling_trace_contributions(c_lin_pair, c_optomech_pair, c_duff_pair, pair, pair_idx):
            i, j = pair
            x_i, x_j = x[i], x[j]
            
            # ∂(Tr(∇²E_coupling))/∂c_lin = 2 + 2 = 4 (from both oscillators)
            c_lin_trace_contrib = 4.0
            
            # ∂(Tr(∇²E_coupling))/∂c_optomech = 2*x_j (only from first oscillator)
            c_optomech_trace_contrib = 2 * x_j
            
            # ∂(Tr(∇²E_coupling))/∂c_duff = 3*(x_i-x_j)² + 3*(x_i-x_j)² = 6*(x_i-x_j)²
            if len(c_duff) > 0:
                c_duff_trace_contrib = 6 * (x_i - x_j)**2
            else:
                c_duff_trace_contrib = 0.0
            
            return jnp.array([
                c_lin_start + pair_idx,
                c_optomech_start + pair_idx,
                c_duff_start + pair_idx if len(c_duff) > 0 else c_duff_start
            ]), jnp.array([
                c_lin_trace_contrib,
                c_optomech_trace_contrib,
                c_duff_trace_contrib
            ])
        
        # Vectorize over coupling pairs
        pair_indices = jnp.arange(len(connectivity))
        coupling_param_indices, coupling_trace_contributions = vmap(compute_coupling_trace_contributions)(
            c_lin, c_optomech, c_duff, connectivity, pair_indices
        )
        
        # Add coupling contributions
        for i in range(len(connectivity)):
            result = result.at[coupling_param_indices[i, 0]].add(coupling_trace_contributions[i, 0])  # c_lin
            result = result.at[coupling_param_indices[i, 1]].add(coupling_trace_contributions[i, 1])  # c_optomech
            if len(c_duff) > 0:
                result = result.at[coupling_param_indices[i, 2]].add(coupling_trace_contributions[i, 2])  # c_duff
        
        return result
    
    # Return the functions
    gradient_fn = lambda x, flattened_args: gradient_duffing_network(x, flattened_args, connectivity)
    trace_hessian_fn = lambda x, flattened_args: trace_hessian_duffing_network(x, flattened_args, connectivity)
    
    # Memory-efficient versions that compute dot products directly
    def gradient_wrt_params_dot_product_fn(x, flattened_args, score):
        return compute_parameter_gradient_dot_product_efficiently(x, flattened_args, score, connectivity)
    
    def trace_hessian_wrt_params_fn(x, flattened_args):
        return gradient_wrt_params_of_trace_hessian_efficient(x, flattened_args, connectivity)
    
    return gradient_fn, trace_hessian_fn, gradient_wrt_params_dot_product_fn, trace_hessian_wrt_params_fn

