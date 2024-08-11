from typing import Callable
import jax

jax.config.update("jax_platform_name", "cpu")
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
        + c_optomech * (y**2) * x
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
def energy_network(x, k_lin, k_duff, c_lin, c_optomech, connectivity):
    return energy_self_network(x, k_lin, k_duff) + energy_coupling_network(
        x, c_lin, c_optomech, connectivity
    )


def setup_integration(
    dofs_to_marginalize: Array,
    integration_limits: tuple[int, int] = (-2, 2),
    num_integration_points: int = 10,
) -> Callable[[Callable], Array]:
    """Create a function that numerically integrates out a subest of DOFs from an energy function,
    using a uniform grid."""

    # Create integration grid between limits
    integration_grids: list[Array] = [
        jnp.linspace(
            integration_limits[0], integration_limits[1], num_integration_points
        )
        for _ in dofs_to_marginalize
    ]
    marginalized_combinations: list[Array] = jnp.meshgrid(*integration_grids)
    marginalized_points: Array = jnp.stack(
        [grid.flatten() for grid in marginalized_combinations], axis=-1
    )

    def compute_integrands(fn: Callable[[Array], Array]) -> Array:
        """Numerically integrates fn over a uniform grid."""
        # --- compute the volume element of the integration grid.
        const_volume_element = jnp.prod(
            jnp.array(
                [
                    (integration_limits[1] - integration_limits[0])
                    / num_integration_points
                    for _ in dofs_to_marginalize
                ]
            )
        )

        # --- sum fn over all marginalized dofs, weighted by volume
        integrand_vals = vmap(fn)(marginalized_points) * const_volume_element
        return integrand_vals

    # return integrator
    return compute_integrands


# def integrate_dofs(integrator, energy_fn, marginalized_dofs, non_marginalized_dofs):
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
    def log_exp_energy_fn_marg(x_non_marginalized, *args):

        def log_energy_for_marginalized_point(marginalized_point: Array) -> Array:
            # Combine non-marginalized and marginalized points
            full_x = jnp.zeros(len(marginalized_dofs) + len(non_marginalized_dofs))
            full_x = full_x.at[non_marginalized_dofs].set(x_non_marginalized)
            full_x = full_x.at[marginalized_dofs].set(marginalized_point)

            return -energy_fn(full_x, *args)

        return logsumexp(compute_integrands(log_energy_for_marginalized_point))

    return log_exp_energy_fn_marg
