# %%
# make sure we can import from artifacts/cld.py and artifacts/plot_helpers.py
import jax
import sys
sys.path.append('../artifacts')
sys.path.append('../../thermoai/thermoai')
# cpu only
jax.config.update("jax_platform_name", "cpu")

from dataclasses import dataclass
from typing import Callable, Union
import matplotlib.pyplot as plt
import jax.numpy as jnp
from diffrax import Euler, ItoMilstein, UnsafeBrownianPath, diffeqsolve, DirectAdjoint, SaveAt, ShARK, ODETerm, ControlTerm, MultiTerm
from jax import config, devices, jit
from jaxtyping import Scalar, Array, Float, Int
import jax.random as jr



from plot_helpers import plot_time_dependent_energy
from thermoai.distributions import mog_energy as mog_energy_fn, sample_mog, mog_logpdf, mog_get_mean
from cld import CriticallyDampedLangevinDynamics
from diffrax import VirtualBrownianTree, SpaceTimeLevyArea

cpu_devices = devices("cpu")

script_key = jax.random.PRNGKey(0)
# %%
# Create a time-dependent energy function
state_dim = 1
means = jnp.array([-9.0, 3.0])
covariances = 1.0*jnp.ones(2)
weights = jnp.array([0.8, 0.2])

# --- Make time-dependent covariances
T = 10.0
# T = 1.0
temp_final = 10.0
temp_fn = lambda t: temp_final #1+temp_final * t/T

params_fn = lambda t: {
    "means": means,
    "covariances": covariances*temp_fn(t),
    "weights": weights,
}
params = {
    "means": means,
    "covariances": temp_final*covariances,
    "weights": weights,
}

#%%
print(params_fn(0))
print(params)
#%%
@jax.jit
def sample_fwd_ics(key, N, t=0):
    x_key, v_key = jax.random.split(key, 2)
    x0s_keys = jax.random.split(x_key, N)
    v0s_keys = jax.random.split(v_key, N)
    x0s = jax.vmap(lambda key: sample_mog(key, **params_fn(t)))(x0s_keys)
    v0s = jax.vmap(lambda key: jr.normal(key))(v0s_keys)
    return x0s, v0s


def sample_bwd_ics(key, N, t=T):
    x_key, v_key = jax.random.split(key, 2)
    x0s_keys = jax.random.split(x_key, N)
    v0s_keys = jax.random.split(v_key, N)
    x0s = jax.vmap(lambda key: sample_mog(key, **params_fn(t)))(x0s_keys)
    v0s = jax.vmap(lambda key: jr.normal(key))(v0s_keys)
    return x0s, v0s


def U_x_t(x: Float, t_0_to_1: Float, t_fn: Callable = lambda t: t):
    """t_0_to_1 is the time input to the potential energy function.
    # t_fn is a function that maps the input time from 0 to 1, in some arbitrary way (e.g. nonlinear)
    # At time t=0, this function returns the energy of a 1D bimodal Gaussian
    # At time t=1, this function returns the energy of a 1D Normal"""
    return mog_energy_fn(x, **params_fn(t_fn(t_0_to_1)))

@jax.jit
def U_x_t_simplified(x: Float, t_0_to_1: Float, t_fn: Callable = lambda t: t):
    """t_0_to_1 is the time input to the potential energy function.
    # t_fn is a function that maps the input time from 0 to 1, in some arbitrary way (e.g. nonlinear)
    # At time t=0, this function returns the energy of a 1D bimodal Gaussian
    # At time t=1, this function returns the energy of a 1D Normal"""
    return mog_energy_fn(x,**params)


fwd_cld = CriticallyDampedLangevinDynamics(
    state_dim=state_dim,
    beta=100.0,  # This controls speed at which system equilibrates
    M=1.0,
    gamma=2.0,
    U=U_x_t,
)

V = lambda v: 0.5*jnp.dot(v, v) / fwd_cld.M


@jax.jit
def fwd_drift(t, u, args):
    x, v = u[:fwd_cld.state_dim], u[fwd_cld.state_dim:]
    
    # Compute Hamiltonian termsplot
    x_dot = jax.grad(V)(v)         # e.g., M^-1 v
    v_dot = -1.0 * jax.grad(U_x_t_simplified)(x, t)  # e.g., -x
    friction_term = -fwd_cld.Gamma * x_dot  # e.g., -Γ M^-1 v
    hamiltonian_terms = jnp.concatenate([x_dot, 
                                            v_dot])
    ou_process_terms = jnp.concatenate([jnp.zeros_like(x), 
                                        friction_term])

    return fwd_cld.beta * (hamiltonian_terms + ou_process_terms)

@jax.jit

def fwd_diffusion(t, u, args):
    zero_block = jnp.zeros((fwd_cld.state_dim, fwd_cld.state_dim))
    sigma = jnp.sqrt(2 * fwd_cld.Gamma * fwd_cld.beta)
    diffusion_block = sigma * jnp.eye(fwd_cld.state_dim)

    G = jnp.block([[zero_block, zero_block], [zero_block, diffusion_block]])

    return G
    
def get_fwd_terms(bm):
    drift_term = ODETerm(fwd_drift)
    diffusion_term = ControlTerm(fwd_diffusion, bm)
    return MultiTerm(drift_term, diffusion_term)

plot_time_dependent_energy(U_x_t, T=T)
solver = ItoMilstein()

@jax.jit
def _split_state(y):
    state_dim = fwd_cld.state_dim
    return y[:state_dim], y[state_dim:]

def solve_fwd(key, x0, v0, dt, t0=0.0, t1=1.0):
    """Solve (simulate) the forward diffusion process for a single trajectory.

    Example usage:

        # x0 = 0.0
        # v0 = 0.0
        # dt = 1e-5
        # t0, t1 = 0.0, 1.0

        # (x, v, t), (xTs, vTs) = solve_fwd(key, x0, v0, dt, t0, t1)
    """
    brownian = UnsafeBrownianPath(shape=(2 * fwd_cld.state_dim,), key=key)
    terms = get_fwd_terms(brownian)
    
    @jax.jit
    def solve_step(carry, _):
        y, t = carry
        next_y = solver.step(terms, t, t + dt, y, None, None, made_jump=False)[0]
        return (next_y, t + dt), next_y

    (y, t), ys = jax.lax.scan(
        solve_step, (jnp.array([x0, v0]), t0), length=int((t1 - t0) / dt)
    )
    x, v = _split_state(y)
    xs, vs = jax.vmap(_split_state)(ys)
    return (x, v, t), (xs, vs)  # xs.shape = (N, num_steps)

#%%
# Example usage for sampling N trajectories in parallel:
N = 100000
N_to_save = 1000
dt = 1e-3
dt0 = dt
t0, t1 = 0.0, T  # T = 1.0
script_key, sample_key, solve_key = jax.random.split(script_key, 3)
x0s, v0s = sample_fwd_ics(sample_key, N, t=0)

solve_keys = jax.random.split(solve_key, N)
(xTs, vTs, ts), (xTs_all, vTs_all) = jax.vmap(lambda key, x0, v0: solve_fwd(key, x0, v0, dt, t0, t1))(
    solve_keys, x0s, v0s
)

# --- Plot histograms of initial and final distributions.
plt.figure(figsize=(12, 6))
plt.hist(x0s, bins=50, alpha=0.5, label="$x_0$s")
plt.hist(xTs, bins=50, alpha=0.5, label="$x_T$s")
plt.legend()
plt.title(f"Position Distributions\nMeans: {means}, Covariances: {covariances}, Weights: {weights}")
plt.xlabel("Position")
plt.ylabel("Count")
plt.show()

#%%
# --- Solve N trajectories in parallel, with unique key, x0, v0.
x0s, v0s = sample_fwd_ics(sample_key, N, t=0)
solve_keys = jax.random.split(solve_key, N)

(xTs, vTs, ts), (xTs_all, vTs_all) = jax.vmap(lambda key, x0, v0: solve_fwd(key, x0, v0, dt, t0, t1))(
    solve_keys, x0s, v0s
)
xTs = np.array(xTs)
mean_xTs = np.mean(xTs)

print(f"Mean of xTs: {mean_xTs}")
# rolling mean


# %%



# Select different time steps to plot
time_steps = [0, 25, 50, 75, -1]  # -1 represents the final time step

plt.figure(figsize=(6, 4))

for step in time_steps:
    if step == -1:
        rolling_mean = np.cumsum(xTs, axis=0) / np.arange(1, N+1)[:, np.newaxis]
        label = f"t = {T:.2f} (final)"
    else:
        rolling_mean = np.cumsum(xTs_all[:, step], axis=0) / np.arange(1, N+1)[:, np.newaxis]
        label = f"t = {step * dt:.2f}"
    
    plt.plot(rolling_mean, alpha=0.5, label=label)

plt.axhline(mog_get_mean(means, weights), linestyle='--', linewidth=2, label="True mean", color="red")
plt.xlabel("Number of samples")
plt.ylabel("Rolling mean")
plt.title("Rolling mean of x at different time steps")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# --- Plot histograms of initial and final distributions.
plt.figure(figsize=(12, 6))
plt.hist(x0s, bins=50, alpha=0.5, label="$x_0$")
plt.hist(xTs, bins=50, alpha=0.5, label="$x_T$")
plt.hist(x0s_bwd, bins=50, alpha=0.5, label="$x_0$ for backward process")
plt.legend()
plt.title(f"Position Distributions\nMeans: {means}, Covariances: {covariances}, Weights: {weights}")
plt.xlabel("Position")
plt.ylabel("Count")
plt.show()

# v0
plt.figure(figsize=(12, 6))
plt.hist(v0s, bins=50, alpha=0.5, label="$v_0$")
plt.hist(vTs, bins=50, alpha=0.5, label="$v_T$")
plt.legend()
plt.title(f"Velocity Distributions\nMeans: {means}, Covariances: {covariances}, Weights: {weights}")
plt.xlabel("Velocity")
plt.ylabel("Count")
plt.show()

# %%
# Set up solve parameters
solver = Euler()  # Trust in dt -> 0...


def _split_state(y):
    state_dim = fwd_cld.state_dim
    return y[:state_dim], y[state_dim:]


def solve_bwd(key, x0, v0, dt, t0=0.0, t1=1):
    """Solve (simulate) the forward diffusion process for a single trajectory.

    Example usage:

        # x0 = 0.0
        # v0 = 0.0
        # dt = 1e-5
        # t0, t1 = 0.0, 1.0

        # (x, v, t), (xTs, vTs) = solve_fwd(key, x0, v0, dt, t0, t1)
    """
    brownian = UnsafeBrownianPath(shape=(2 * bwd_cld.state_dim,), key=key)
    terms = bwd_cld.get_fwd_terms(brownian)

    @jax.jit
    def solve_step(carry, _):
        y, t = carry
        next_y = solver.step(terms, t, t + dt, y, None, None, made_jump=False)[0]
        return (next_y, t + dt), next_y

    (y, t), ys = jax.lax.scan(
        solve_step, (jnp.array([x0, v0]), t0), length=int((t1 - t0) / dt)
    )
    x, v = _split_state(y)
    xs, vs = jax.vmap(_split_state)(ys)
    return (x, v, t), (xs, vs)



# Example usage for sampling N trajectories in parallel:
N = 1000
dt = 1e-2
t0, t1 = 0.0, T



# --- Solve N trajectories in parallel, with unique key, x0, v0.
solve_keys_bwd = jax.random.split(bwd_solve_key, N)
(xTs_bwd, vTs_bwd, ts), _ = jax.vmap(lambda key, x0, v0: solve_bwd(key, x0, v0, dt, t0, t1))(
    solve_keys_bwd, xTs.reshape(-1), vTs.reshape(-1)
)

# (
#     solve_keys_bwd, x0s_bwd, v0s_bwd
# )


# %%
# --- Plot histograms of initial and final distributions.
plt.hist(x0s_bwd, bins=50, alpha=0.5, label="$x_0$")
plt.legend()
plt.hist(xTs, bins=50, alpha=0.5, label="$x_T$")
plt.legend()
plt.show()

# %%
# --- Plot histograms of initial and final distributions.
plt.hist(x0s_bwd, bins=50, alpha=0.5, label="$x_T$")
plt.legend()
plt.hist(xTs_bwd, bins=50, alpha=0.5, label="$xbwd_0$")
plt.legend()
plt.hist(x0s, bins=50, alpha=0.5, label="$x_0$")
plt.legend()
plt.show()

# v0
plt.hist(v0s_bwd, bins=50, alpha=0.5, label="$v_0$")
plt.legend()
plt.hist(vTs_bwd, bins=50, alpha=0.5, label="$v_T$")
plt.legend()
plt.show()

# %%



