import jax
from jax import grad, vmap, hessian
import jax.numpy as jnp
import jax.random as jr
import diffrax
from diffrax import ControlTerm, MultiTerm, ODETerm

########################################################################################
# Connectivity functions
########################################################################################

def create_2d_square_lattice_connectivity(grid_size, n_neighbour_couplings):
    """
    Create connectivity for a square lattice (grid_size x grid_size) by coupling each node
    to every other node whose Chebyshev distance (shell) is between 1 and n_neighbour_couplings.
    
    For example:
      - n_neighbour_couplings = 1: Connect to immediate neighbors (horizontal, vertical, and diagonal).
      - n_neighbour_couplings = 2: Also include next-nearest neighbors (shell 2).
      - n_neighbour_couplings = grid_size: All-to-all connectivity.
    
    Args:
        grid_size (int): Side length of the square grid.
        n_neighbour_couplings (int): Maximum coupling shell to include.
        
    Returns:
        jnp.ndarray: Array of shape (num_connections, 2) with each row [node1, node2].
    """
    connections = set()
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            # Loop over all possible offsets from -n_neighbour_couplings to +n_neighbour_couplings.
            for di in range(-n_neighbour_couplings, n_neighbour_couplings + 1):
                for dj in range(-n_neighbour_couplings, n_neighbour_couplings + 1):
                    # Skip self connection.
                    if di == 0 and dj == 0:
                        continue
                    # Only include if the offset is within the desired shell.
                    # (This is actually always true since di,dj are within [-n_c, n_c].)
                    ni = i + di
                    nj = j + dj
                    # Make sure neighbor is within the grid.
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        neighbor = ni * grid_size + nj
                        # To avoid duplicates, only add if current node index is less than neighbor.
                        if node < neighbor:
                            connections.add((node, neighbor))
    # Convert the set of tuples into a sorted JAX array.
    connections = sorted(list(connections))
    return jnp.array(connections)



########################################################################################
# Energy functions
########################################################################################

# with external force and duffing nonlinear coupling
def setup_energy_fn(connectivity, unflatten):
    def energy_self_oscillator(x, k_lin, k_duff, k_6, bias):
        return 1 / 2 * k_lin * x**2 + 1 / 4 * k_duff * x**4 + 1 / 6 * k_6 * x**6 + bias * x

    def energy_self_network(x, k_lin, k_duff, k_6, bias):
        return jnp.sum(vmap(energy_self_oscillator)(x, k_lin, k_duff, k_6, bias))


    def energy_coupling_pair(x, y, c_lin, c_optomech, c_duff):
        return c_lin * x * (x - y) + c_lin * y * (y - x) + c_optomech * (x**2) * y + c_duff/4 * (x-y)**4

    def energy_coupling_network(x, c_lin, c_optomech, c_duff, connectivity):
        # get contribution to total energy from each pair of oscillators
        
        def _coupling_energy_pair(x, c_lin, c_optomech, c_duff, pair):
            # get energy of one pair of oscillators
            i, j = pair
            return energy_coupling_pair(x[i], x[j], c_lin, c_optomech, c_duff)

        # get energy of all pairs of oscillators
        return jnp.sum(
            vmap(_coupling_energy_pair, in_axes=(None, 0, 0, 0, 0))(
                x, c_lin, c_optomech, c_duff, connectivity
            )
        )

    def _energy_duffing_network(x, k_lin, k_duff, k_6, c_lin, c_optomech, c_duff, bias, connectivity):
        # returns energy of network of oscillators given input parameters
        return (
            energy_self_network(x, k_lin, k_duff, k_6, bias) +
            energy_coupling_network(x, c_lin, c_optomech,c_duff, connectivity)
        ) 

    def energy_duffing_network(x, flattened_args, connectivity):
        k_lin, k_duffing, k_6, c_lin, c_optomech, c_duff, bias = unflatten(flattened_args)
        return _energy_duffing_network(x, k_lin, k_duffing, k_6, c_lin, c_optomech, c_duff, bias, connectivity)

    energy_fn = lambda x, flattened_args: energy_duffing_network(x, flattened_args, connectivity)
    return energy_fn


# Analytical gradient and Hessian trace for the duffing network with external force and nonlinear coupling
def setup_duffing_network_analytical_derivatives(connectivity, unflatten):
    """
    Constructs closed-form derivative functions for the Duffing oscillator network
    energy, avoiding the overhead of automatic differentiation.

    The total network energy is E = E_self + E_coupling, where

        E_self(x_i) = (1/2) k_lin x_i² + (1/4) k_duff x_i⁴
                     + (1/6) k_6 x_i⁶ + bias x_i

        E_coupling(x_i, x_j) = c_lin x_i(x_i - x_j) + c_lin x_j(x_j - x_i)
                              + c_optomech x_i² x_j + (c_duff/4)(x_i - x_j)⁴

    Supports three parameter layouts via ``unflatten``:
        - 7 groups: (k_lin, k_duff, k_6, c_lin, c_optomech, c_duff, bias)
        - 6 groups: (k_lin, k_duff, c_lin, c_optomech, c_duff, bias)  — no 6th-order term
        - 5 groups: (k_lin, k_duff, c_lin, c_optomech, bias)           — no Duffing coupling

    Args:
        connectivity: int array of shape (n_pairs, 2) listing coupled oscillator
            index pairs, as produced by ``create_2d_square_lattice_connectivity``.
        unflatten: callable that maps a flat parameter vector to the tuple of
            per-oscillator and per-pair parameter arrays described above.

    Returns:
        Tuple of four functions, each with signature ``(x, flattened_args)``:
            gradient_fn:
                ∇_x E — gradient of the energy w.r.t. oscillator coordinates.
            trace_hessian_fn:
                Tr(∇²_x E) — scalar trace of the Hessian (Laplacian of E).
            gradient_wrt_params_fn:
                ∂(∇_x E)/∂θ — Jacobian of the gradient w.r.t. parameters,
                shape (n_oscillators, n_params).
            trace_hessian_wrt_params_fn:
                ∂(Tr(∇²_x E))/∂θ — gradient of the Hessian trace w.r.t.
                parameters, shape (n_params,).
    """
    
    def gradient_self_oscillator(x, k_lin, k_duff, k_6, bias):
        """Analytical gradient of self oscillator energy w.r.t. x"""
        return k_lin * x + k_duff * x**3 + k_6 * x**5 + bias
    
    def hessian_diag_self_oscillator(x, k_lin, k_duff, k_6, bias):
        """Analytical diagonal of Hessian (second derivative) of self oscillator energy w.r.t. x"""
        return k_lin + 3 * k_duff * x**2 + 5 * k_6 * x**4
    
    def gradient_coupling_pair_wrt_x(x, y, c_lin, c_optomech, c_duff):
        """Analytical gradient of coupling energy w.r.t. x for pair (x,y)"""
        return (2 * c_lin * x - 2 * c_lin * y + 
                2 * c_optomech * x * y + 
                c_duff * (x - y)**3)
    
    def gradient_coupling_pair_wrt_y(x, y, c_lin, c_optomech, c_duff):
        """Analytical gradient of coupling energy w.r.t. y for pair (x,y)"""
        return (-2 * c_lin * x + 2 * c_lin * y + 
                c_optomech * x**2 - 
                c_duff * (x - y)**3)
    
    def hessian_diag_coupling_pair_wrt_x(x, y, c_lin, c_optomech, c_duff):
        """Analytical diagonal Hessian of coupling energy w.r.t. x for pair (x,y)"""
        return (2 * c_lin + 
                2 * c_optomech * y + 
                3 * c_duff * (x - y)**2)
    
    def hessian_diag_coupling_pair_wrt_y(x, y, c_lin, c_optomech, c_duff):
        """Analytical diagonal Hessian of coupling energy w.r.t. y for pair (x,y)"""
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
    
    # Analytical parameter derivatives
    def gradient_wrt_params_of_gradient(x, flattened_args, connectivity):
        """Analytical parameter derivatives of the gradient: ∂(∇E)/∂params"""
        # Unflatten parameters to get the actual structure
        unflattened_params = unflatten(flattened_args)
        n_oscillators = len(x)
        n_params = len(flattened_args)
        
        # Initialize parameter gradient matrix [n_oscillators, n_params]
        param_grad_matrix = jnp.zeros((n_oscillators, n_params))
        
        # Handle different parameter structures based on what's available
        if len(unflattened_params) == 7:  # 6th order duffing coupling
            k_lin, k_duffing, k_6, c_lin, c_optomech, c_duff, bias = unflattened_params
            has_k6 = True
        elif len(unflattened_params) == 6:  # duffing coupling without k_6
            k_lin, k_duffing, c_lin, c_optomech, c_duff, bias = unflattened_params
            k_6 = jnp.zeros_like(k_lin)  # Dummy k_6 for compatibility
            has_k6 = False
        elif len(unflattened_params) == 5:  # duffing optomech coupling
            k_lin, k_duffing, c_lin, c_optomech, bias = unflattened_params
            k_6 = jnp.zeros_like(k_lin)  # Dummy values for compatibility
            c_duff = jnp.zeros_like(c_lin)
            has_k6 = False
        else:
            raise ValueError(f"Unsupported parameter structure with {len(unflattened_params)} parameter groups")
        
        # Self oscillator parameter derivatives
        # ∂(∂E_self/∂x)/∂k_lin = x, ∂(∂E_self/∂x)/∂k_duff = x³, etc.
        def _self_param_derivatives(osc_idx, x_i, k_lin_i, k_duff_i, k_6_i, bias_i):
            # Create derivatives array with correct size
            derivs = jnp.zeros(n_params)
            
            # k_lin derivatives (first N_osc parameters)
            derivs = derivs.at[osc_idx].set(x_i)
            
            # k_duff derivatives (second N_osc parameters)  
            derivs = derivs.at[len(k_lin) + osc_idx].set(x_i**3)
            
            # k_6 derivatives (third N_osc parameters if present)
            if has_k6:
                derivs = derivs.at[2*len(k_lin) + osc_idx].set(x_i**5)
            
            # bias derivatives (last N_osc parameters)
            derivs = derivs.at[n_params - len(k_lin) + osc_idx].set(1.0)
            
            return derivs
        
        # Add self oscillator contributions
        oscillator_indices = jnp.arange(len(x))
        self_derivs = vmap(_self_param_derivatives)(oscillator_indices, x, k_lin, k_duffing, k_6, bias)
        param_grad_matrix = param_grad_matrix + self_derivs
        
        # Coupling parameter derivatives (more complex due to pairwise interactions)
        def _coupling_param_derivatives_x(x_i, x_j, c_lin_pair, c_optomech_pair, c_duff_pair, pair_idx):
            # Create derivatives array with correct size
            derivs = jnp.zeros(n_params)
            
            # Find the parameter indices for coupling parameters
            n_osc = len(k_lin)
            if has_k6:
                c_lin_start = 3 * n_osc  # After k_lin, k_duff, k_6
            else:
                c_lin_start = 2 * n_osc  # After k_lin, k_duff
            
            c_optomech_start = c_lin_start + len(c_lin)
            c_duff_start = c_optomech_start + len(c_optomech)
            
            # ∂(∂E_coupling/∂x_i)/∂c_lin = 2*x_i - 2*x_j
            derivs = derivs.at[c_lin_start + pair_idx].set(2*x_i - 2*x_j)
            
            # ∂(∂E_coupling/∂x_i)/∂c_optomech = 2*x_i*x_j  
            derivs = derivs.at[c_optomech_start + pair_idx].set(2*x_i*x_j)
            
            # ∂(∂E_coupling/∂x_i)/∂c_duff = (x_i - x_j)³ (if c_duff exists)
            if len(c_duff) > 0:
                derivs = derivs.at[c_duff_start + pair_idx].set((x_i - x_j)**3)
            
            return derivs
        
        def _coupling_param_derivatives_y(x_i, x_j, c_lin_pair, c_optomech_pair, c_duff_pair, pair_idx):
            # Create derivatives array with correct size
            derivs = jnp.zeros(n_params)
            
            # Find the parameter indices for coupling parameters
            n_osc = len(k_lin)
            if has_k6:
                c_lin_start = 3 * n_osc  # After k_lin, k_duff, k_6
            else:
                c_lin_start = 2 * n_osc  # After k_lin, k_duff
            
            c_optomech_start = c_lin_start + len(c_lin)
            c_duff_start = c_optomech_start + len(c_optomech)
            
            # ∂(∂E_coupling/∂x_j)/∂c_lin = -2*x_i + 2*x_j
            derivs = derivs.at[c_lin_start + pair_idx].set(-2*x_i + 2*x_j)
            
            # ∂(∂E_coupling/∂x_j)/∂c_optomech = x_i²
            derivs = derivs.at[c_optomech_start + pair_idx].set(x_i**2)
            
            # ∂(∂E_coupling/∂x_j)/∂c_duff = -(x_i - x_j)³ (if c_duff exists)
            if len(c_duff) > 0:
                derivs = derivs.at[c_duff_start + pair_idx].set(-(x_i - x_j)**3)
            
            return derivs
        
        # Add coupling contributions
        def _compute_coupling_param_derivs(c_lin_pair, c_optomech_pair, c_duff_pair, pair, pair_idx):
            i, j = pair
            derivs_i = _coupling_param_derivatives_x(x[i], x[j], c_lin_pair, c_optomech_pair, c_duff_pair, pair_idx)
            derivs_j = _coupling_param_derivatives_y(x[i], x[j], c_lin_pair, c_optomech_pair, c_duff_pair, pair_idx)
            return jnp.array([i, j]), jnp.stack([derivs_i, derivs_j])
        
        # Vectorize over all coupling pairs with pair indices
        pair_indices = jnp.arange(len(connectivity))
        indices, coupling_derivs = vmap(_compute_coupling_param_derivs)(c_lin, c_optomech, c_duff, connectivity, pair_indices)
        
        # Add coupling contributions using scatter_add
        param_grad_matrix = param_grad_matrix.at[indices[:, 0]].add(coupling_derivs[:, 0])
        param_grad_matrix = param_grad_matrix.at[indices[:, 1]].add(coupling_derivs[:, 1])
        
        return param_grad_matrix
    
    def gradient_wrt_params_of_trace_hessian(x, flattened_args, connectivity):
        """Analytical parameter derivatives of trace of Hessian: ∂(Tr(∇²E))/∂params"""
        # Unflatten parameters to get the actual structure
        unflattened_params = unflatten(flattened_args)
        n_params = len(flattened_args)
        
        # Initialize parameter gradient vector [n_params]
        param_grad_vector = jnp.zeros(n_params)
        
        # Handle different parameter structures based on what's available
        if len(unflattened_params) == 7:  # 6th order duffing coupling
            k_lin, k_duffing, k_6, c_lin, c_optomech, c_duff, bias = unflattened_params
            has_k6 = True
        elif len(unflattened_params) == 6:  # duffing coupling without k_6
            k_lin, k_duffing, c_lin, c_optomech, c_duff, bias = unflattened_params
            k_6 = jnp.zeros_like(k_lin)  # Dummy k_6 for compatibility
            has_k6 = False
        elif len(unflattened_params) == 5:  # duffing optomech coupling
            k_lin, k_duffing, c_lin, c_optomech, bias = unflattened_params
            k_6 = jnp.zeros_like(k_lin)  # Dummy values for compatibility
            c_duff = jnp.zeros_like(c_lin)
            has_k6 = False
        else:
            raise ValueError(f"Unsupported parameter structure with {len(unflattened_params)} parameter groups")
        
        # Self oscillator parameter derivatives of Hessian diagonal
        # ∂(∂²E_self/∂x²)/∂k_lin = 1, ∂(∂²E_self/∂x²)/∂k_duff = 3*x², etc.
        def _self_hess_param_derivatives(osc_idx, x_i, k_lin_i, k_duff_i, k_6_i, bias_i):
            # Create derivatives array with correct size
            derivs = jnp.zeros(n_params)
            
            # k_lin derivatives (first N_osc parameters)
            derivs = derivs.at[osc_idx].set(1.0)
            
            # k_duff derivatives (second N_osc parameters)  
            derivs = derivs.at[len(k_lin) + osc_idx].set(3*x_i**2)
            
            # k_6 derivatives (third N_osc parameters if present)
            if has_k6:
                derivs = derivs.at[2*len(k_lin) + osc_idx].set(5*x_i**4)
            
            # bias derivatives are 0 for Hessian
            
            return derivs
        
        # Sum over all oscillators for self contributions
        oscillator_indices = jnp.arange(len(x))
        self_hess_derivs = vmap(_self_hess_param_derivatives)(oscillator_indices, x, k_lin, k_duffing, k_6, bias)
        param_grad_vector = param_grad_vector + jnp.sum(self_hess_derivs, axis=0)
        
        # Coupling parameter derivatives of Hessian diagonal
        def _coupling_hess_param_derivatives(x_i, x_j, c_lin_pair, c_optomech_pair, c_duff_pair, pair_idx):
            # Create derivatives array with correct size
            derivs = jnp.zeros(n_params)
            
            # Find the parameter indices for coupling parameters
            n_osc = len(k_lin)
            if has_k6:
                c_lin_start = 3 * n_osc  # After k_lin, k_duff, k_6
            else:
                c_lin_start = 2 * n_osc  # After k_lin, k_duff
            
            c_optomech_start = c_lin_start + len(c_lin)
            c_duff_start = c_optomech_start + len(c_optomech)
            
            # For both oscillators i and j
            # ∂(∂²E_coupling/∂x_i²)/∂c_lin = 2, ∂(∂²E_coupling/∂x_j²)/∂c_lin = 2
            derivs = derivs.at[c_lin_start + pair_idx].set(4.0)  # 2 + 2 from both oscillators
            
            # ∂(∂²E_coupling/∂x_i²)/∂c_optomech = 2*x_j, ∂(∂²E_coupling/∂x_j²)/∂c_optomech = 0
            derivs = derivs.at[c_optomech_start + pair_idx].set(2*x_j)  # Only from oscillator i
            
            # ∂(∂²E_coupling/∂x_i²)/∂c_duff = 3*(x_i-x_j)², ∂(∂²E_coupling/∂x_j²)/∂c_duff = 3*(x_i-x_j)²
            if len(c_duff) > 0:
                derivs = derivs.at[c_duff_start + pair_idx].set(6*(x_i-x_j)**2)  # 3 + 3 from both oscillators
            
            return derivs
        
        # Sum over all coupling pairs
        pair_indices = jnp.arange(len(connectivity))
        coupling_hess_derivs = vmap(_coupling_hess_param_derivatives)(
            x[connectivity[:, 0]], x[connectivity[:, 1]], c_lin, c_optomech, c_duff, pair_indices
        )
        param_grad_vector = param_grad_vector + jnp.sum(coupling_hess_derivs, axis=0)
        
        return param_grad_vector

    gradient_fn = lambda x, flattened_args: gradient_duffing_network(x, flattened_args, connectivity)
    trace_hessian_fn = lambda x, flattened_args: trace_hessian_duffing_network(x, flattened_args, connectivity)
    gradient_wrt_params_fn = lambda x, flattened_args: gradient_wrt_params_of_gradient(x, flattened_args, connectivity)
    trace_hessian_wrt_params_fn = lambda x, flattened_args: gradient_wrt_params_of_trace_hessian(x, flattened_args, connectivity)
    
    return gradient_fn, trace_hessian_fn, gradient_wrt_params_fn, trace_hessian_wrt_params_fn


def setup_score_matching_loss_hessian(gradient_wrt_params_fn, k_b=1.0, T=1.0):
    """
    Constructs the analytical Hessian of the score matching loss w.r.t. parameters.

    The score matching loss is quadratic in the parameters, so its Hessian is constant
    (independent of params) and depends only on the batch of states x.

    For p(x) ∝ exp(−E(x;θ) / kT), the loss per sample is
        L(x, θ) = −Tr(∇²_x E)/(k_b T) + (1/2) ||∇_x E||²/(k_b T)²

    The first term is linear in θ (Hessian = 0). The second term is quadratic:
    since ∇_x E is linear in θ, we have ||∇_x E||² = θᵀ G(x)ᵀ G(x) θ where
    G(x) = ∂(∇_x E)/∂θ. Thus ∂²L/∂θ² = G(x)ᵀ G(x) / (k_b T)² per sample.

    Args:
        gradient_wrt_params_fn: Callable (x, flattened_args) -> array (D, P).
            Jacobian ∂(∇_x E)/∂θ, from setup_duffing_network_analytical_derivatives.
        k_b: Boltzmann constant (default 1.0).
        T: Temperature (default 1.0).

    Returns:
        hessian_fn: Callable (x, flattened_args) -> array (P, P).
            If x has shape (D,), returns Hessian for a single sample.
            If x has shape (N, D), returns batch-averaged Hessian.
            flattened_args is used only for structure (values ignored); n_params is
            extracted from gradient_wrt_params_fn output shape.
    """
    kBT_sq = (k_b * T) ** 2

    def hessian_fn(x, flattened_args):
        if x.ndim == 1:
            G = gradient_wrt_params_fn(x, flattened_args)
            return (G.T @ G) / kBT_sq
        else:
            G_batch = vmap(lambda xi: gradient_wrt_params_fn(xi, flattened_args))(x)
            H = jnp.mean(vmap(lambda Gi: Gi.T @ Gi)(G_batch), axis=0)
            return H / kBT_sq

    return hessian_fn


########################################################################################
# Overdamped SDE
########################################################################################

def setup_overdamped_SDE(energy_fn, flattened_args, N_osc, gamma=1.0, k_b=1.0, Temp=1.0, time_dependent_parms=False, debug=False):
    """
    Sets up drift and diffusion functions for overdamped dynamics
    
    Args:
        energy_fn: function taking (state, args) as input
        flattened_args: parameters for the energy function
        gamma: damping coefficient
        k_b: Boltzmann constant
        T: temperature
        
    Returns:
        drift_fn: function taking (t, state, args) as input
        diffusion_fn: function taking (t, state, args) as input
    """
    # Reduce energy function to only depend on state
    if time_dependent_parms:
        energy_of_state_time = lambda t, state: energy_fn(state, flattened_args(t))
    else:
        energy_of_state_time = lambda t, state: energy_fn(state, flattened_args) 
    
    if debug:   
        def drift_fn(t, state, args):
            jax.debug.print("Current time: {}", t)
            """Drift function for overdamped dynamics"""
            dE_dx = grad(energy_of_state_time, argnums=1)(t, state)
            return -1/gamma * dE_dx
    else:
        def drift_fn(t, state, args):
            """Drift function for overdamped dynamics"""
            dE_dx = grad(energy_of_state_time, argnums=1)(t, state)
            return -1/gamma * dE_dx
    
    def diffusion_fn(t, state, args):
        """Diffusion function for overdamped dynamics"""
        noise_strength = jnp.sqrt(2 * 1/gamma * k_b * Temp)
        return noise_strength * jnp.eye(N_osc)
    
    return drift_fn, diffusion_fn

def solve_SDE(drift_fn, diffusion_fn, initial_state, key_brownian, t0, t1, N_save, dt0, rtol=1e-3, atol=1e-6, brownian_tolerance=1e-12):
    N_osc = initial_state.shape[0]
    ts = jnp.linspace(t0, t1, N_save)
    w_shape = (N_osc,)  # state is just phases for each oscillator
    brownian_motion = diffrax.VirtualBrownianTree(
        t0, t1, brownian_tolerance, w_shape, key_brownian, diffrax.SpaceTimeLevyArea
    )
    terms = MultiTerm(ODETerm(drift_fn), ControlTerm(diffusion_fn, brownian_motion))
    saveat = diffrax.SaveAt(ts=ts)


    solution = diffrax.diffeqsolve(
        terms,
        solver=diffrax.SRA1(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=initial_state,
        args=(),
        saveat=saveat,
        progress_meter=diffrax.TqdmProgressMeter(),
        max_steps=1000000000,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),  # Enable adaptive stepping
    )
    return solution
