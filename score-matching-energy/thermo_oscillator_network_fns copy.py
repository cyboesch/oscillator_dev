import jax
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax import config, jit, random, grad, vmap, flatten_util, hessian
# from jax_md.quantity import force
from jax import random

from functools import reduce
from operator import mul

from scipy.stats import multivariate_normal


def energy_self_oscillator(x, k_lin, k_duff):
    return 1/2*k_lin*x**2 + 1/4*k_duff*x**4
        
def energy_coupling_pair(x,y,c_lin, c_optomech):
    return c_lin*x*(x-y) + c_lin*y*(y-x) + c_optomech*(x**2)*y + c_optomech*(y**2)*x

def energy_self_network(x, k_lin, k_duff):
    return jnp.sum(vmap(energy_self_oscillator)(x, k_lin, k_duff))

def energy_coupling_network(x, c_lin, c_optomech, connectivity):
    def _coupling_energy_pair(x,c_lin, c_optomech,pair):
        i, j = pair
        return energy_coupling_pair(x[i], x[j], c_lin, c_optomech)
    return jnp.sum(vmap(_coupling_energy_pair, in_axes=(None,0,0,0))(x, c_lin, c_optomech, connectivity))
@jit
def energy_network(x, k_lin, k_duff, c_lin, c_optomech, connectivity):
    return energy_self_network(x, k_lin, k_duff) + energy_coupling_network(x, c_lin, c_optomech, connectivity)



def setup_integration(marginalized_dofs, integration_limits=(-2, 2), num_integration_points=10):
    integration_grids = [jnp.linspace(integration_limits[0], integration_limits[1], num_integration_points) 
                                for _ in marginalized_dofs]
    marginalized_combinations = jnp.meshgrid(*integration_grids)
    marginalized_points = jnp.stack([grid.flatten() for grid in marginalized_combinations], axis=-1)

    def inegrator(fn):
        # Combine all possible combinations of marginalized DOFs
        integrated_vals = jnp.sum(vmap(fn)(marginalized_points)) * ((integration_limits[1] - integration_limits[0]) / num_integration_points) ** len(marginalized_dofs)
        return integrated_vals
    return inegrator


def integrate_dofs(inegrator,energy_fn, marginalized_dofs, non_marginalized_dofs):
    """
    Integrate out marginalized DOFs from the energy function.
    
    Args:
    energy_fn: The original energy function.
    marginalized_dofs: List of indices of DOFs to be integrated out.
    non_marginalized_dofs: List of indices of DOFs to keep.
    integration_limits: Tuple of (lower, upper) limits for integration.
    num_integration_points: Number of points for numerical integration.
    
    Returns:
    A new energy function that depends only on non-marginalized DOFs.
    """
            # Create integration grid for marginalized DOFs

    
    @jit
    def log_exp_energy_fn_marg(x_non_marginalized, *args):

        
        # Function to compute energy for each marginalized point
        def energy_for_marginalized_point(marginalized_point):
            # Combine non-marginalized and marginalized points
            full_x = jnp.zeros(len(marginalized_dofs) + len(non_marginalized_dofs))
            full_x = full_x.at[non_marginalized_dofs].set(x_non_marginalized)
            full_x = full_x.at[marginalized_dofs].set(marginalized_point)
            
            return jnp.exp(-energy_fn(full_x, *args))
        
        return jnp.log(inegrator(energy_for_marginalized_point))
    
    return log_exp_energy_fn_marg



