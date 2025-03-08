import jax
from jax import grad, vmap, hessian
import jax.numpy as jnp
import jax.random as jr
import diffrax
from diffrax import ControlTerm, MultiTerm, ODETerm

########################################################################################
# Energy functions
########################################################################################

# without external force
def setup_duffing_network_energy_fn(connectivity, unflatten):
    def energy_self_oscillator(x, k_lin, k_duff):
        return 1 / 2 * k_lin * x**2 + 1 / 4 * k_duff * x**4

    def energy_self_network(x, k_lin, k_duff):
        return jnp.sum(vmap(energy_self_oscillator)(x, k_lin, k_duff))


    def energy_coupling_pair(x, y, c_lin, c_optomech):
        return c_lin * x * (x - y) + c_lin * y * (y - x) + c_optomech * (x**2) * y

    def energy_coupling_network(x, c_lin, c_optomech, connectivity):
        # get contribution to total energy from each pair of oscillators
        
        def _coupling_energy_pair(x, c_lin, c_optomech, pair):
            # get energy of one pair of oscillators
            i, j = pair
            return energy_coupling_pair(x[i], x[j], c_lin, c_optomech)

        # get energy of all pairs of oscillators
        return jnp.sum(
            vmap(_coupling_energy_pair, in_axes=(None, 0, 0, 0))(
                x, c_lin, c_optomech, connectivity
            )
        )


    def _energy_duffing_network(x, k_lin, k_duff, c_lin, c_optomech, connectivity):
        # returns energy of network of oscillators given input parameters
        return (
            energy_self_network(x, k_lin, k_duff) +
            energy_coupling_network(x, c_lin, c_optomech, connectivity)
        ) 

    def energy_duffing_network(x, flattened_args, connectivity):
        k_lin, k_duffing, c_lin, c_optomech = unflatten(flattened_args)
        return _energy_duffing_network(x, k_lin, k_duffing, c_lin, c_optomech, connectivity)

    energy_fn = lambda x, flattened_args: energy_duffing_network(x, flattened_args, connectivity)
    return energy_fn

# with external force
def setup_duffing_network_with_external_force_energy_fn(connectivity, unflatten):
    def energy_self_oscillator(x, k_lin, k_duff, bias):
        return 1 / 2 * k_lin * x**2 + 1 / 4 * k_duff * x**4 + bias * x

    def energy_self_network(x, k_lin, k_duff, bias):
        return jnp.sum(vmap(energy_self_oscillator)(x, k_lin, k_duff, bias))


    def energy_coupling_pair(x, y, c_lin, c_optomech):
        return c_lin * x * (x - y) + c_lin * y * (y - x) + c_optomech * (x**2) * y

    def energy_coupling_network(x, c_lin, c_optomech, connectivity):
        # get contribution to total energy from each pair of oscillators
        
        def _coupling_energy_pair(x, c_lin, c_optomech, pair):
            # get energy of one pair of oscillators
            i, j = pair
            return energy_coupling_pair(x[i], x[j], c_lin, c_optomech)

        # get energy of all pairs of oscillators
        return jnp.sum(
            vmap(_coupling_energy_pair, in_axes=(None, 0, 0, 0))(
                x, c_lin, c_optomech, connectivity
            )
        )

    def _energy_duffing_network(x, k_lin, k_duff, c_lin, c_optomech, bias, connectivity):
        # returns energy of network of oscillators given input parameters
        return (
            energy_self_network(x, k_lin, k_duff, bias) +
            energy_coupling_network(x, c_lin, c_optomech, connectivity)
        ) 

    def energy_duffing_network(x, flattened_args, connectivity):
        k_lin, k_duffing, c_lin, c_optomech, bias = unflatten(flattened_args)
        return _energy_duffing_network(x, k_lin, k_duffing, c_lin, c_optomech, bias, connectivity)

    energy_fn = lambda x, flattened_args: energy_duffing_network(x, flattened_args, connectivity)
    return energy_fn

# general polynomial network
def setup_general_polynomial_network_with_external_force_energy_fn(connectivity, unflatten):
    import jax.numpy as jnp
    from jax import vmap

    # Self oscillator energy: here only a linear bias is used to mimic an external force.
    def energy_self_oscillator(x, bias):
        return bias * x

    def energy_self_network(x, bias):
        # Apply the self oscillator energy to each oscillator.
        # Here we create a bias array of the same shape as x.
        bias_array = jnp.full_like(x, bias)
        return jnp.sum(vmap(energy_self_oscillator)(x, bias_array))

    # Coupling energy for a pair (x, y)
    def energy_coupling_pair(x, y, a1, a2, a3):
        z = a1 * x + a2 * y + a3 * x**2
        return z**4 - z**2

    def energy_coupling_network(x, a1, a2, a3, connectivity):
        # For every pair given in connectivity, compute the pair energy.
        def _coupling_energy_pair(x, a1, a2, a3, pair):
            i, j = pair
            return energy_coupling_pair(x[i], x[j], a1, a2, a3)
        return jnp.sum(vmap(_coupling_energy_pair, in_axes=(None, 0, 0, 0, 0))(
    x, a1, a2, a3, connectivity))

    # Total energy is the sum of the self energies and the coupling energies.
    def _energy_general_polynomial_network(x, a1, a2, a3, bias, connectivity):
        return energy_self_network(x, bias) + energy_coupling_network(x, a1, a2, a3, connectivity)

    def energy_general_polynomial_network(x, flattened_args, connectivity):
        # Unflatten to get the coupling parameters a1, a2, a3.
        a1, a2, a3, bias = unflatten(flattened_args)
        return _energy_general_polynomial_network(x, a1, a2, a3, bias, connectivity)

    energy_fn = lambda x, flattened_args: energy_general_polynomial_network(x, flattened_args, connectivity)
    return energy_fn


########################################################################################
# Overdamped SDE
########################################################################################

def setup_overdamped_SDE(energy_fn, flattened_args, N_osc, gamma=1.0, k_b=1.0, T=1.0, time_dependent_parms=False):
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
    
    def drift_fn(t, state, args):
        """Drift function for overdamped dynamics"""
        dE_dx = grad(energy_of_state_time, argnums=1)(t, state)
        return -gamma * dE_dx
    
    def diffusion_fn(t, state, args):
        """Diffusion function for overdamped dynamics"""
        noise_strength = jnp.sqrt(2 * gamma * k_b * T)
        return noise_strength * jnp.eye(N_osc)
    
    return drift_fn, diffusion_fn

def solve_SDE(drift_fn, diffusion_fn, initial_state, key_brownian, t0, t1, N_samples, dt0, rtol=1e-3, atol=1e-6):
    N_osc = initial_state.shape[0]
    ts = jnp.linspace(t0, t1, N_samples)
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