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


# setup 6th order local linear coupling energy function
def setup_6th_order_local_linear_coupling_energy_fn(connectivity, unflatten):
    def energy_self_oscillator(x, k_lin, k_duff, k_6, bias):
        return 1 / 2 * k_lin * x**2 + 1 / 4 * k_duff * x**4 + 1 / 6 * k_6 * x**6 + bias * x

    def energy_self_network(x, k_lin, k_duff, k_6, bias):
        return jnp.sum(vmap(energy_self_oscillator)(x, k_lin, k_duff, k_6, bias))


    def energy_coupling_pair(x, y, c_lin):
        return c_lin * x * (x - y) + c_lin * y * (y - x)

    def energy_coupling_network(x, c_lin, connectivity):
        # get contribution to total energy from each pair of oscillators
        
        def _coupling_energy_pair(x, c_lin, pair):
            # get energy of one pair of oscillators
            i, j = pair
            return energy_coupling_pair(x[i], x[j], c_lin)

        # get energy of all pairs of oscillators
        return jnp.sum(
            vmap(_coupling_energy_pair, in_axes=(None, 0, 0))(
                x, c_lin, connectivity
            )
        )

    def _energy_duffing_network(x, k_lin, k_duff, k_6, c_lin, bias, connectivity):
        # returns energy of network of oscillators given input parameters
        return (
            energy_self_network(x, k_lin, k_duff, k_6, bias) +
            energy_coupling_network(x, c_lin, connectivity)
        ) 

    def energy_duffing_network(x, flattened_args, connectivity):
        k_lin, k_duffing, k_6, c_lin, bias = unflatten(flattened_args)
        return _energy_duffing_network(x, k_lin, k_duffing, k_6, c_lin, bias, connectivity)

    energy_fn = lambda x, flattened_args: energy_duffing_network(x, flattened_args, connectivity)
    return energy_fn


# with external force and duffing nonlinear coupling
def setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_energy_fn(connectivity, unflatten):
    def energy_self_oscillator(x, k_lin, k_duff, bias):
        return 1 / 2 * k_lin * x**2 + 1 / 4 * k_duff * x**4 + bias * x

    def energy_self_network(x, k_lin, k_duff, bias):
        return jnp.sum(vmap(energy_self_oscillator)(x, k_lin, k_duff, bias))


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

    def _energy_duffing_network(x, k_lin, k_duff, c_lin, c_optomech, c_duff, bias, connectivity):
        # returns energy of network of oscillators given input parameters
        return (
            energy_self_network(x, k_lin, k_duff, bias) +
            energy_coupling_network(x, c_lin, c_optomech,c_duff, connectivity)
        ) 

    def energy_duffing_network(x, flattened_args, connectivity):
        k_lin, k_duffing, c_lin, c_optomech, c_duff, bias = unflatten(flattened_args)
        return _energy_duffing_network(x, k_lin, k_duffing, c_lin, c_optomech, c_duff, bias, connectivity)

    energy_fn = lambda x, flattened_args: energy_duffing_network(x, flattened_args, connectivity)
    return energy_fn


# with external force and duffing nonlinear coupling
def setup_duffing_network_with_external_force_and_nonlinear_duffing_coupling_and_6th_order_energy_fn(connectivity, unflatten):
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

# # general polynomial network
# def setup_general_polynomial_network_with_external_force_energy_fn(connectivity, unflatten):

#     # Self oscillator energy: here only a linear bias is used to mimic an external force.
#     def energy_self_oscillator(x, bias):
#         return bias * x

#     def energy_self_network(x, bias):
#         # Apply the self oscillator energy to each oscillator.
#         # Here we create a bias array of the same shape as x.
#         bias_array = jnp.full_like(x, bias)
#         return jnp.sum(vmap(energy_self_oscillator)(x, bias_array))

#     # Coupling energy for a pair (x, y)
#     def energy_coupling_pair(x, y, a1, a2, a3):
#         z = a1 * x + a2 * y + a3 * x**2
#         return z**4 - z**2

#     def energy_coupling_network(x, a1, a2, a3, connectivity):
#         # For every pair given in connectivity, compute the pair energy.
#         def _coupling_energy_pair(x, a1, a2, a3, pair):
#             i, j = pair
#             return energy_coupling_pair(x[i], x[j], a1, a2, a3)
#         return jnp.sum(vmap(_coupling_energy_pair, in_axes=(None, 0, 0, 0, 0))(
#     x, a1, a2, a3, connectivity))

#     # Total energy is the sum of the self energies and the coupling energies.
#     def _energy_general_polynomial_network(x, a1, a2, a3, bias, connectivity):
#         return energy_self_network(x, bias) + energy_coupling_network(x, a1, a2, a3, connectivity)

#     def energy_general_polynomial_network(x, flattened_args, connectivity):
#         # Unflatten to get the coupling parameters a1, a2, a3.
#         a1, a2, a3, bias = unflatten(flattened_args)
#         return _energy_general_polynomial_network(x, a1, a2, a3, bias, connectivity)

#     energy_fn = lambda x, flattened_args: energy_general_polynomial_network(x, flattened_args, connectivity)
#     return energy_fn


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

def solve_SDE(drift_fn, diffusion_fn, initial_state, key_brownian, t0, t1, N_save, dt0, rtol=1e-3, atol=1e-6):
    N_osc = initial_state.shape[0]
    ts = jnp.linspace(t0, t1, N_save)
    w_shape = (N_osc,)  # state is just phases for each oscillator
    brownian_motion = diffrax.VirtualBrownianTree(
        t0, t1, 1.e-11, w_shape, key_brownian, diffrax.SpaceTimeLevyArea
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
