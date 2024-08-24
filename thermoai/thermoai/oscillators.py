import equinox as eqx
import jax.numpy as jnp
from typing import List, Tuple, Callable, Optional
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

unitary_potential = energy_self_oscillator

def energy_coupling_pair(x, y, c_lin, c_optomech):
    return c_lin * x * (x - y) + c_lin * y * (y - x) + c_optomech * (x**2) * y

pairwise_potential = energy_coupling_pair

def energy_self_network(x, k_lin, k_duff):
    return jnp.sum(vmap(energy_self_oscillator)(x, k_lin, k_duff))

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


@jit
def energy_network(x, k_lin, k_duff, c_lin, c_optomech, connectivity, k_b=1.0, T=1.0):
    # get energy of network of oscillators given input parameters
    return (
        energy_self_network(x, k_lin, k_duff) +
        energy_coupling_network(x, c_lin, c_optomech, connectivity)
    ) / (2 * k_b * T)


def setup_integration(
    marginalized_dofs: Array,
    integration_limits: tuple[int, int] = (-2, 2),
    num_integration_points: int = 10,
) -> Callable[[Callable], Array]:

    # Create a len(dofs_to_marginalize)-D grid of indices
    grid_indices = jnp.mgrid[
        (slice(0, num_integration_points),) * len(marginalized_dofs)
    ]

    # Transform indices to points that span the integration limits.
    marginalized_points = integration_limits[0] + (
        integration_limits[1] - integration_limits[0]
    ) * grid_indices / (num_integration_points - 1)

    # Reshape to (num_points, num_dimensions)
    marginalized_points = marginalized_points.reshape(-1, len(marginalized_dofs))

    # --- compute volume element
    vol_elt = (
        (integration_limits[1] - integration_limits[0]) / num_integration_points
    ) ** len(marginalized_dofs)

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


#%%
# %% [markdown]
# aim: re
#%%


#%%
class Oscillator(eqx.Module):  # node
    m: float = 1
    position: jnp.ndarray = jnp.zeros(2)
    velocity: jnp.ndarray = jnp.zeros(2)
    
class Spring(eqx.Module):  # (self) edge
    dof1_index: int
    dof2_index: Optional[int] = None
    stiffness: float = 1
    rest_length: float = 1
    force_function: Callable[[float], float] = lambda x: x


class OscillatorNetwork(eqx.Module):
    dofs: List[Oscillator]
    springs: List[Spring]

    def __init__(self):
        self.dofs = []
        self.springs = []

    def add_dof(self, position: jnp.ndarray, mass: float) -> int:
        self.dofs.append(Oscillator(position, mass))
        return len(self.dofs) - 1

    def add_spring(self, 
                   dof1_index: int, 
                   dof2_index: Optional[int], 
                   stiffness: float, 
                   rest_length: float, 
                   force_function: Callable[[float], float]) -> int:
        self.springs.append(Spring(dof1_index, dof2_index, stiffness, rest_length, force_function))
        return len(self.springs) - 1

    def get_state(self) -> jnp.ndarray:
        return jnp.array([dof.position for dof in self.dofs])

    def set_state(self, state: jnp.ndarray) -> 'OscillatorNetwork':
        new_dofs = [Oscillator(pos, dof.mass) for pos, dof in zip(state, self.dofs)]
        return eqx.tree_at(lambda t: t.dofs, self, new_dofs)

def dynamics(network: OscillatorNetwork, t: float, params: eqx.Module) -> Tuple[jnp.ndarray, jnp.ndarray]:
    forces = network.compute_forces()
    accelerations = forces / jnp.array([dof.mass for dof in network.dofs])
    velocities = network.get_state()  # Assuming the state contains velocities
    return velocities, accelerations

# Create a JIT-compiled version of the dynamics function
jit_dynamics = eqx.filter_jit(dynamics)
import jax

# Initialize the network
if __name__ == "__main__":
    network = OscillatorNetwork()
    dof1 = network.add_dof(jnp.array([0.0, 0.0]), 1.0)
    dof2 = network.add_dof(jnp.array([1.0, 0.0]), 1.0)
    network.add_spring(dof1, dof2, 1.0, 1.0, lambda x: x)  # Linear spring

    # Set up initial conditions
    initial_state = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # positions and velocities
    network = network.set_state(initial_state)

    # Simulate
    t = 0.0
    dt = 0.01
    for _ in range(100):
        velocities, accelerations = jit_dynamics(network, t, None)
        new_state = network.get_state() + velocities * dt + 0.5 * accelerations * dt**2
        network = network.set_state(new_state)
        t += dt