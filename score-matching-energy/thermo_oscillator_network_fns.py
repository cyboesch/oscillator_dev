from typing import Callable
import jax

# jax.config.update("jax_platform_name", "cpu")
from jax import Array

import jax.numpy as jnp
from jax import config, jit, random, grad, vmap, flatten_util, hessian

# from jax_md.quantity import force
from jax import random

from functools import reduce
from operator import mul

from scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp


def energy_self_oscillator(x, k_lin, k_duff):
    return 1 / 2 * k_lin * x**2 + 1 / 4 * k_duff * x**4


def energy_coupling_pair(x, y, c_lin, c_optomech):
    return (
        c_lin * x * (x - y)
        + c_lin * y * (y - x)
        + c_optomech * (x**2) * y
    )


def energy_self_network(x, k_lin, k_duff):
    return jnp.sum(vmap(energy_self_oscillator)(x, k_lin, k_duff))


def energy_coupling_network(x, c_lin, c_optomech, connectivity):
    def _coupling_energy_pair(x, c_lin, c_optomech, pair):
        i, j = pair
        return energy_coupling_pair(x[i], x[j], c_lin, c_optomech)

    return jnp.sum(
        vmap(_coupling_energy_pair, in_axes=(None, 0, 0, 0))(
            x, c_lin, c_optomech, connectivity
        )
    )


@jit
def energy_network(x, k_lin, k_duff, c_lin, c_optomech, connectivity,k_b=1.,T=1.):
    return (energy_self_network(x, k_lin, k_duff) + energy_coupling_network(
        x, c_lin, c_optomech, connectivity
    ))/(2*k_b*T)


def setup_integration(
    marginalized_dofs: Array,
    integration_limits: tuple[int, int] = (-2, 2),
    num_integration_points: int = 10,
) -> Callable[[Callable], Array]:

    # Create a len(dofs_to_marginalize)-dimensional grid of indices
    grid_indices = jnp.mgrid[(slice(0, num_integration_points),) * len(marginalized_dofs)]
    
    # Transform indices to points .
    marginalized_points = integration_limits[0] + (integration_limits[1] - integration_limits[0]) * grid_indices / (num_integration_points - 1)
    
    # Reshape to (num_points, num_dimensions)
    marginalized_points = marginalized_points.reshape(-1, len(marginalized_dofs))
    
    # --- compute volume element
    vol_elt = ((integration_limits[1] - integration_limits[0]) / num_integration_points) ** len(
        marginalized_dofs
    )

    def compute_integrands(energy_fn: Callable[[Array], Array]) -> Array:
        ### returns contributions to the integral across marginalized DOFs
        return vmap(energy_fn)(marginalized_points) * vol_elt

    return compute_integrands


def integrate_dofs(
    compute_integrands, energy_fn, marginalized_dofs, non_marginalized_dofs
):
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
    def logsumexp_energy_fn_marg(x_non_marginalized, *args):

        def energy_for_marginalized_point(marginalized_point: Array) -> Array:
            # Combine non-marginalized and marginalized points
            full_x = jnp.zeros(len(marginalized_dofs) + len(non_marginalized_dofs))
            full_x = full_x.at[non_marginalized_dofs].set(x_non_marginalized)
            full_x = full_x.at[marginalized_dofs].set(marginalized_point)

            return -energy_fn(full_x, *args)

        return logsumexp(compute_integrands(energy_for_marginalized_point))

    return logsumexp_energy_fn_marg