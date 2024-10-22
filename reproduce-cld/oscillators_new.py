import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import equinox as eqx
import jax.numpy as jnp
from typing import List, Tuple, Callable, Optional
from typing import Callable
import jax

from jax import Array
from jax import config, jit, random, grad, vmap, flatten_util, hessian
from functools import partial, reduce
from operator import mul
from scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp

# Commented out old functions
# def energy_self_oscillator(x, k_lin, k_duff):
#     return 1 / 2 * k_lin * x**2 + 1 / 4 * k_duff * x**4

# unitary_potential = energy_self_oscillator

# def energy_coupling_pair(x, y, c_lin, c_optomech):
#     return c_lin * x * (x - y) + c_lin * y * (y - x) + c_optomech * (x**2) * y

# pairwise_potential = energy_coupling_pair

# def energy_self_network(x, k_lin, k_duff):
#     return jnp.sum(vmap(energy_self_oscillator)(x, k_lin, k_duff))

# def energy_coupling_network(x, c_lin, c_optomech, connectivity):
#     # get contribution to total energy from each pair of oscillators

#     def _coupling_energy_pair(x, c_lin, c_optomech, pair):
#         # get energy of one pair of oscillators
#         i, j = pair
#         return energy_coupling_pair(x[i], x[j], c_lin, c_optomech)

#     # get energy of all pairs of oscillators
#     return jnp.sum(
#         vmap(_coupling_energy_pair, in_axes=(None, 0, 0, 0))(
#             x, c_lin, c_optomech, connectivity
#         )
#     )

# @jit
# def energy_network(x, k_lin, k_duff, c_lin, c_optomech, connectivity, k_b=1.0, T=1.0):
#     # returns energy of network of oscillators given input parameters
#     return (
#         energy_self_network(x, k_lin, k_duff) +
#         energy_coupling_network(x, c_lin, c_optomech, connectivity)
#     ) / (2 * k_b * T)

# New functions start here

def energy_potential_single(x, k_harmonic, k_duffing):
    """Potential energy of a single oscillator
    
    Args:
        x: Displacement from equilibrium
        k_harmonic: Harmonic spring constant
        k_duffing: Duffing nonlinearity coefficient
        
    Returns:
        V(x) = ½ * k_harmonic * x² + ¼ * k_duffing * x⁴
    """
    return 0.5 * k_harmonic * x**2 + 0.25 * k_duffing * x**4

def energy_linear_coupling(x_i, x_j, c_lin):
    """Linear coupling energy between two oscillators
    
    Args:
        x_i, x_j: Displacements of two oscillators
        c_lin: Linear coupling strength
        
    Returns:
        V_c(x_i, x_j) = c_lin * (x_i - x_j)²
    """
    return c_lin * (x_i - x_j)**2

def energy_optomechanical(x, y, g_opt):
    """Optomechanical coupling energy between two oscillators
    
    Args:
        x_i, x_j: Displacements of two oscillators
        g_opt: Optomechanical coupling strength
        
    Returns:
        V_opt(x, y) = g_opt * x² * y
    """
    return g_opt * x**2 * y

def energy_self_network(x, k_harmonic, k_duffing):
    """Total potential energy of all uncoupled oscillators
    
    Args:
        x: Array of oscillator displacements
        k_harmonic: Harmonic spring constants (can be scalar or array)
        k_duffing: Duffing coefficients (can be scalar or array)
        
    Returns:
        Total potential energy of the network
    """
    return jnp.sum(vmap(lambda x_i: energy_potential_single(x_i, k_harmonic, k_duffing))(x))

def energy_coupling_network(x, c_lin, g_opt, connectivity):
    """Total coupling energy in the network
    
    Args:
        x: Array of oscillator displacements
        c_lin: Linear coupling strength
        g_opt: Optomechanical coupling strength
        connectivity: Array of (i, j) pairs indicating coupled oscillators
        
    Returns:
        Total coupling energy of the network
    """
    def _coupling_energy_pair(pair):
        i, j = pair
        return (
            energy_linear_coupling(x[i], x[j], c_lin) +
            energy_optomechanical(x[i], x[j], g_opt)
        )
        
    # Sum over all pairs specified in connectivity
    return jnp.sum(vmap(_coupling_energy_pair)(connectivity))

@jit
def energy_network(x, k_harmonic, k_duffing, c_lin, g_opt, connectivity, k_b=1.0, T=1.0):
    """Total energy of the oscillator network
    
    H = Σᵢ [½ * k_harmonic * xᵢ² + ¼ * k_duffing * xᵢ⁴] 
        + Σ₍ᵢ,ⱼ₎ [c_lin * (xᵢ - xⱼ)² + g_opt * xᵢ * xⱼ]
    
    Args:
        x: Array of oscillator displacements
        k_harmonic: Harmonic spring constant(s)
        k_duffing: Duffing nonlinearity coefficient(s)
        c_lin: Linear coupling strength
        g_opt: Optomechanical coupling strength
        connectivity: Array of (i, j) pairs indicating coupled oscillators
        k_b: Boltzmann constant
        T: Temperature
        
    Returns:
        Total normalized energy of the network
    """
    # Potential energy of individual oscillators
    single_oscillator_energy = energy_self_network(x, k_harmonic, k_duffing)
    # Coupling energy between oscillators
    coupling_energy = energy_coupling_network(x, c_lin, g_opt, connectivity)
    # Total energy normalized by temperature
    total_energy = (single_oscillator_energy + coupling_energy) / (2 * k_b * T)
    return total_energy

def setup_integration_grid(
    num_dims: int,
    integration_limits: Tuple[float, float] = (-2.0, 2.0),
    num_points: int = 10,
) -> Tuple[Array, float]:
    """Creates integration grid and volume element for numerical integration.
    
    Args:
        num_dims: Number of dimensions to integrate over
        integration_limits: (lower, upper) bounds for integration
        num_points: Number of points per dimension
        
    Returns:
        grid_points: Array of shape (num_points^num_dims, num_dims) containing all grid points
        vol_element: Volume element for integration
    """
    # Create meshgrid for integration points
    points_1d = jnp.linspace(integration_limits[0], integration_limits[1], num_points)
    grid = jnp.meshgrid(*[points_1d for _ in range(num_dims)])
    
    # Reshape to (num_points^num_dims, num_dims)
    grid_points = jnp.stack([g.ravel() for g in grid], axis=-1)
    
    # Compute volume element
    vol_element = ((integration_limits[1] - integration_limits[0]) / (num_points - 1)) ** num_dims
    
    return grid_points, vol_element

@partial(jit, static_argnums=(0,))
def marginalize_dofs(
    energy_fn: Callable,
    x_observed: Array,
    marginalized_idxs: Array,
    observed_idxs: Array,
    grid_points: Array,
    vol_element: float,
    *args
) -> Array:
    """Marginalizes out specified degrees of freedom from the energy function.
    
    Args:
        energy_fn: Original energy function to marginalize
        x_observed: Values of observed (non-marginalized) DOFs
        marginalized_idxs: Indices of DOFs to marginalize out
        observed_idxs: Indices of DOFs to keep
        grid_points: Integration grid points from setup_integration_grid
        vol_element: Volume element for integration
        *args: Additional arguments to pass to energy_fn
        
    Returns:
        Marginalized free energy
    """
    def energy_at_point(marginalized_values: Array) -> Array:
        # Construct full state vector
        x_full = jnp.zeros(len(marginalized_idxs) + len(observed_idxs))
        x_full = x_full.at[observed_idxs].set(x_observed)
        x_full = x_full.at[marginalized_idxs].set(marginalized_values)
        return -energy_fn(x_full, *args)
    
    # Vectorize over all grid points
    energies = vmap(energy_at_point)(grid_points)
    
    # Compute log of partition function using logsumexp for numerical stability
    log_Z = logsumexp(energies) + jnp.log(vol_element)
    
    return -log_Z  # Return marginalized free energy

if __name__ == "__main__":
    # Set random seed for reproducibility
    key = random.PRNGKey(0)
    
    # Test basic network setup
    x = jnp.array([0.1, -0.2, 0.3])  # 3 oscillators
    k_harmonic = 1.0
    k_duffing = 0.1
    c_lin = 0.5
    g_opt = 0.2
    connectivity = jnp.array([[0, 1], [1, 2]])  # Chain configuration
    
    # Test energy calculations
    energy = energy_network(x, k_harmonic, k_duffing, c_lin, g_opt, connectivity)
    assert jnp.isfinite(energy), "Energy calculation failed"
    
    # Test single oscillator energy
    single_energy = energy_potential_single(x[0], k_harmonic, k_duffing)
    assert jnp.isfinite(single_energy), "Single oscillator energy calculation failed"
    
    # Test coupling energies
    linear_coupling = energy_linear_coupling(x[0], x[1], c_lin)
    assert jnp.isfinite(linear_coupling), "Linear coupling energy calculation failed"
    
    optomech_coupling = energy_optomechanical(x[0], x[1], g_opt)
    assert jnp.isfinite(optomech_coupling), "Optomechanical coupling energy calculation failed"
    
    # Test marginalization
    num_marginalized = 1
    grid_points, vol_element = setup_integration_grid(
        num_dims=num_marginalized,
        integration_limits=(-2.0, 2.0),
        num_points=10
    )
    
    marginalized_idxs = jnp.array([0])
    observed_idxs = jnp.array([1, 2])
    x_observed = jnp.array([0.1, 0.2])
    
    marginalized_energy = marginalize_dofs(
        energy_network,
        x_observed,
        marginalized_idxs,
        observed_idxs,
        grid_points,
        vol_element,
        k_harmonic,
        k_duffing,
        c_lin,
        g_opt,
        connectivity
    )
    
    assert jnp.isfinite(marginalized_energy), "Marginalization failed"
    
    print("All tests passed!")
