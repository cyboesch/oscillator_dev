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

from thermoai.distributions import mog_energy as mog_energy_fn, sample_mog, mog_logpdf
from cld import CriticallyDampedLangevinDynamics

config.update("jax_enable_x64", True)

cpu_devices = devices("cpu")


@dataclass
class CLDConfig(dict):
    state_dim: int
    beta: float
    M: float
    gamma: float
    Gamma: Union[float, None] = None
    U: Union[Callable, None] = None  # potential energy function
    V: Union[Callable, None] = None  # kinetic energy function
    score_fn: Union[Callable, None] = None  # score function
    
    def __post_init__(self):
        # Initialize the dict with the instance's attributes
        for key, value in self.__dict__.items():
            self[key] = value

    def __iter__(self):
        return iter(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]


# %%
# Create a time-dependent energy function 
state_dim = 1
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

score_fn_x = lambda x, t: jax.grad(mog_logpdf)(x, **params_fn(t))
score_fn_x(jnp.array([0.0]), 1.0)
#%%

def score_fn(u, t):
    x, v = u[:state_dim], u[state_dim:]
    return jnp.array([jnp.zeros_like(v), score_fn_x(x, t)]).squeeze()

score_fn(jnp.array([0.0, 0.0]), 1.0)
# %%

# Create SDE terms
# NOTE: implicitly, final T is 1.0 in Dockhorn paper.
time_dependent_config = CLDConfig(
    state_dim=state_dim,
    beta=100.0,  # This controls speed at which system equilibrates
    M=1.0,
    gamma=1.0,
    U=U_x_t,
    score_fn=score_fn,
)

cld = CriticallyDampedLangevinDynamics(**time_dependent_config) 
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
# Set up solve parameters
solver = Euler()  # Trust in dt -> 0...


def _split_state(y):
    state_dim = cld.state_dim
    return y[:state_dim], y[state_dim:]


def solve_fwd(key, x0, v0, dt, t0=0.0, t1=1.0):
    brownian = UnsafeBrownianPath(
        shape=(2 * cld.state_dim,), key=key
    )
    terms = cld.get_fwd_terms(brownian)

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

(x, v, t), (xTs, vTs) = solve_fwd(key, x0, v0, dt, t0, t1)

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
(xTs, vTs, ts), _ = jax.vmap(
    lambda key, x0, v0: solve_fwd(key, x0, v0, dt, t0, t1)
)(solve_keys, x0s, v0s)


# --- Plot histograms of initial and final distributions.
plt.hist(x0s, bins=50, alpha=0.5, label="$x_0$")
plt.legend()
plt.hist(xTs, bins=50, alpha=0.5, label="$x_T$")
plt.legend()
plt.show()

# v0
plt.hist(v0s, bins=50, alpha=0.5, label="$v_0$")
plt.legend()
plt.hist(vTs, bins=50, alpha=0.5, label="$v_T$")
plt.legend()
plt.show()


# %%
# Now reverse the process, running it backwards in time.
def solve_bwd(key, xT, vT, dt, t0=0.0, t1=1.0):
    brownian = UnsafeBrownianPath(
        shape=(2 * cld.state_dim,), key=key
    )
    terms = cld.get_bwd_terms(brownian)

    @jax.jit
    def solve_step(carry, _):
        y, t = carry
        jax.debug.print("{t}", t=t)
        jax.debug.print("{y}", y=y)
        next_y = solver.step(terms, t, t - dt, y, None, None, made_jump=False)[0]
        jax.debug.print("{next_y}", next_y=next_y)
        return (next_y, t - dt), next_y
    
    (y, t), ys = jax.lax.scan(solve_step, (jnp.array([xT, vT]), t1), length=int((t1 - t0) / dt))
    x, v = _split_state(y)
    xs, vs = jax.vmap(_split_state)(ys)
    return (x, v, t), (xs[::-1], vs[::-1])  # Reverse the trajectories

# %%
# Example usage for N trajectories in parallel.

# --- Get randomness
key, bwd_solve_key = jax.random.split(key, 2)
bwd_solve_keys = jax.random.split(bwd_solve_key, N)


# --- Solve N trajectories in parallel, with different key, x0, v0.
(x0s_bwd, v0s_bwd, ts), _ = jax.vmap(
    lambda key, xT, vT: solve_bwd(key, xT, vT, dt, t0, t1)
)(bwd_solve_keys, xTs[:,0], vTs[:,0])


#%%
x0s_bwd
# %%
# xTs histogram
plt.hist(xTs[:,0], bins=50, alpha=0.5, label="$x_T$")
plt.legend()
plt.show()
#%%
# x0s_bwd histogram
plt.hist(x0s_bwd, bins=50, alpha=0.5, label="$x_0$")
plt.legend()
plt.show()
# %%

# vTs histogram
plt.hist(vTs, bins=50, alpha=0.5, label="$v_T$")
plt.legend()
plt.show()

# v0s_bwd histogram
plt.hist(v0s_bwd, bins=50, alpha=0.5, label="$v_0$")
plt.legend()
plt.show()

#%%