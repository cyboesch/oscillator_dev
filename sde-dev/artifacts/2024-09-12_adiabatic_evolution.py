# %%
# %%
# %%
from dataclasses import dataclass
from typing import Callable, Union
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from diffrax import Euler, ItoMilstein, UnsafeBrownianPath
from jax import config, devices, jit
from jaxtyping import Scalar, Array, Float, Int
import jax.random as jr

from plot_helpers import plot_time_dependent_energy

from thermoai.distributions import mog_energy as mog_energy_fn, sample_mog
from cld import CriticallyDampedLangevinDynamics

config.update("jax_enable_x64", True)

cpu_devices = devices("cpu")


@dataclass
class CLDConfig:
    state_dim: int
    beta: float
    M: float
    gamma: float
    Gamma: Union[float, None] = None
    U: Union[Callable, None] = None  # potential energy function


# %%

# Create time-dependent energy function 
means = jnp.array([-4.0, 0.0])
covariances = jnp.array([1.0, 1.0])
weights = jnp.array([0.5, 0.5])

# --- Make time-dependent weights
weights_t = lambda t: jnp.array(
    [weights[0] * (1 - t), weights[1] + t * (1 - weights[1])]
)

params_fn = lambda t: {
    "means": means,
    "covariances": covariances,
    "weights": weights_t(t),
}

def U_x_t(x: Float, t: Float):
    # At time t=0, this function returns the energy of a 1D bimodal Gaussian
    # At time t=1, this function returns the energy of a 1D Normal
    return mog_energy_fn(x, **params_fn(t))


# %%

# Plot weights as a sanity check.
plt.plot(
    jnp.linspace(0, 1, 100),
    jax.vmap(weights_t)(jnp.linspace(0, 1, 100)),
    label=["$\pi_0(t)$", "$\pi_1(t)$"],
)
plt.legend()
plt.show()
# Plot the energy landscape at different times.

plot_time_dependent_energy(U_x_t)

# %%

# Create SDE terms
time_dependent_potential = CLDConfig(
    state_dim=1,
    beta=0.01,
    M=1.0,
    gamma=1.0,
    U=U_x_t,
)

cld = CriticallyDampedLangevinDynamics(**time_dependent_potential.__dict__)

# %%
# Set up solve parameters
solver = Euler()


def _split_state(y):
    state_dim = time_dependent_potential.state_dim
    return y[:state_dim], y[state_dim:]


def solve_fwd(key, x0, v0, dt, t0=0.0, t1=1.0):
    brownian = UnsafeBrownianPath(
        shape=(2 * time_dependent_potential.state_dim,), key=key
    )
    terms = cld.get_terms(brownian)

    @jax.jit
    def solve_step(carry, _):
        y, t = carry
        next_y = solver.step(terms, t, t + dt, y, None, None, made_jump=False)[0]
        return (next_y, t + dt), next_y
    
    (y, t), ys = jax.lax.scan(solve_step, (jnp.array([x0, v0]), t0), length=int((t1 - t0) / dt))
    x, v = _split_state(y)
    xs, vs = jax.vmap(_split_state)(ys)
    return (x, v, t), (xs, vs)


# %%
# Example usage for single trajectory:

key = jax.random.PRNGKey(0)
x0 = 0.0
v0 = 0.0
dt = 1e-5
t0, t1 = 0.0, 1.0

(x, v, t), (xs, vs) = solve_fwd(key, x0, v0, dt, t0, t1)

# %%
# Example usage for sampling N trajectories in parallel:
N = 1000
dt = 1e-2
t0, t1 = 0.0, 1.0


# --- Get randomness
x0s_key, v0s_key, solve_key = jax.random.split(key, 3)
x0s_keys = jax.random.split(x0s_key, N)
v0s_keys = jax.random.split(v0s_key, N)
solve_keys = jax.random.split(solve_key, N)

# --- Sample initial conditions at t=0.
x0s = jax.vmap(lambda key: sample_mog(key, **params_fn(0)))(x0s_keys)
v0s = jax.vmap(lambda key: jr.normal(key))(v0s_keys)


# --- Solve N trajectories in parallel, with different key, x0, v0.
(xs, vs, ts), _ = jax.vmap(
    lambda key, x0, v0: solve_fwd(key, x0, v0, dt, t0, t1)
)(solve_keys, x0s, v0s)


# %%
# Visualize initial and final distributions.
# intiial
plt.hist(x0s, bins=50, alpha=0.5, label="$x_0$")
plt.legend()
plt.hist(xs, bins=50, alpha=0.5, label="$x_T$")
plt.legend()
plt.show()

# v0
plt.hist(v0s, bins=50, alpha=0.5, label="$v_0$")
plt.legend()
plt.hist(vs, bins=50, alpha=0.5, label="$v_T$")
plt.legend()
plt.show()


# %%
