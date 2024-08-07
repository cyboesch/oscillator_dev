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


# # MARGINALIZATION
# def integrate_dofs(energy_fn, x_mins, x_maxs, N):
#     # Create a list of ranges for each dimension
#     x_ranges = [jnp.linspace(x_min, x_max, N) for x_min, x_max in zip(x_mins, x_maxs)]
    
#     # Create a meshgrid of all combinations
#     meshgrid = jnp.meshgrid(*x_ranges, indexing='ij')
    
#     # Flatten the meshgrid to create a list of points
#     points = jnp.stack([grid.flatten() for grid in meshgrid], axis=-1)
    
#     # Compute the integrand for all points
#     integrand = vmap(energy_fn)(points)
    
#     # Reshape the integrand back to the original shape
#     integrand = integrand.reshape([N] * len(x_mins))
    
#     # Compute the volume element
#     dV = reduce(mul, [(x_max - x_min) / (N - 1) for x_min, x_max in zip(x_mins, x_maxs)])
    
#     # Perform the integration
#     return jnp.sum(jnp.exp(-integrand)) * dV


