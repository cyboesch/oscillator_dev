# %%
# %%
# %%
import jax.numpy as jnp
import jax.random as jrnd
from diffrax import MultiTerm, ODETerm, ControlTerm, ItoMilstein, UnsafeBrownianPath
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from jax import jit

import diffrax
from matplotlib import gridspec
import matplotlib.pyplot as plt
from diffrax import ControlTerm, MultiTerm, ODETerm
from jax import grad, vmap
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from scipy import stats

from jax import jit, devices, config

config.update("jax_enable_x64", True)

from thermoai.distributions import sample_1d_mog

cpu_devices = devices("cpu")

from dataclasses import dataclass
import jax.random as jr

@dataclass
class SDEConfig:
    beta: float
    M: float
    Gamma: float
    gamma: float
    state_dim: int

@dataclass
class InitConfig:
    N: int
    rng: jr.PRNGKey

@dataclass
class SimulationConfig:
    init_seed: int
    init_config: InitConfig
    sde_config: SDEConfig

    @classmethod
    def create(cls, init_seed: int = 0):
        return cls(
            init_seed=init_seed,
            init_config=InitConfig(
                N=10000,
                rng=jr.PRNGKey(init_seed)
            ),
            sde_config=SDEConfig(
                beta=0.1,
                M=1.0,
                Gamma=1.0,
                gamma=1.0,
                state_dim=1
            )
        )

config = SimulationConfig.create()

MoG_1D_params = {
    "p1": 0.5,  # p2 = 1 - p1
    "mu1": -5.0,
    "mu2": 5.0,
    "stddev1": 0.5,
    "stddev2": 0.5,
}


normal_params = {
    "p1": 1.0,  # p2 = 1 - p1
    "mu1": 0.0,
    "mu2": 0.0,  # Not used
    "stddev1": jnp.sqrt(config.sde_config.gamma * config.sde_config.M),
    "stddev2": 0.0,  # Not used
}


# Sample from the MoG and the normal distribution.
def sample_x0_p0(key):
    x_key, p_key = jr.split(key, 2)
    x = sample_1d_mog(**MoG_1D_params, key=x_key)
    p = sample_1d_mog(**normal_params, key=p_key)
    return jnp.array([x, p])  # (2,)


# Sample initial conditions and put them on the cpu
init_x0s_p0s = jax.device_put(
    vmap(sample_x0_p0)(jr.split(config.init_config.rng, config.init_config.N)), cpu_devices[0]
)  # (n_samples, 2)


# %%
# Create the plot
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(3, 3)

# Main scatter plot
ax_main = fig.add_subplot(gs[1:, :2])
ax_main.scatter(init_x0s_p0s[:, 0], init_x0s_p0s[:, 1], alpha=0.5, s=1)
ax_main.set_xlabel("x")
ax_main.set_ylabel("p")

# Top marginal plot (for x)
ax_top = fig.add_subplot(gs[0, :2], sharex=ax_main)
ax_top.hist(init_x0s_p0s[:, 0], bins=50, density=True, alpha=0.6)
ax_top.set_ylabel("Density")
ax_top.set_title("MoG Distribution (x)")
ax_top.tick_params(labelbottom=False)

# Right marginal plot (for p)
ax_right = fig.add_subplot(gs[1:, 2], sharey=ax_main)
ax_right.hist(
    init_x0s_p0s[:, 1], bins=50, density=True, alpha=0.6, orientation="horizontal"
)

ax_right.set_xlabel("Density")
ax_right.set_title("Normal Distribution (p)", rotation=270, x=1.1, y=0.5)
ax_right.tick_params(labelleft=False)

# 2D contour plot
ax_contour = fig.add_subplot(gs[1:, :2], sharex=ax_main, sharey=ax_main)
x = init_x0s_p0s[:, 0]
y = init_x0s_p0s[:, 1]
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
ax_contour.contourf(xx, yy, f, cmap="jet", alpha=0.5)
ax_contour.set_xlabel("x")
ax_contour.set_ylabel("p")

# Adjust layout and display
plt.tight_layout()
plt.show()

# %%


def H(z):
    x, p = z
    return 0.5 * (x**2 + p**2 * (1 / config.sde_config.M))


grad_z_H = jit(grad(H))

# Plot H and its gradient
plt.figure(figsize=(12, 4))
x = jnp.linspace(xmin, xmax, 1000)
y = jnp.linspace(ymin, ymax, 1000)
xx, yy = jnp.meshgrid(x, y)
xxyy = jnp.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
zz = vmap(H)(xxyy)
print(zz.shape)
zz = zz.reshape(xx.shape)
plt.subplot(1, 3, 1)
plt.contourf(xx, yy, zz, alpha=1.0)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("p")
plt.title("$H(z)$")

zz = vmap(grad_z_H)(xxyy)
zz_x = zz[:, 0].reshape(xx.shape)
zz_p = zz[:, 1].reshape(xx.shape)
plt.subplot(1, 3, 2)
plt.contourf(xx, yy, zz_x, alpha=1.0)
plt.colorbar()
plt.title("$\\nabla_x H(z)$")
plt.xlabel("x")
plt.ylabel("p")

plt.subplot(1, 3, 3)
plt.contourf(xx, yy, zz_p, alpha=1.0)
plt.colorbar()
plt.title("$\\nabla_p H(z)$")
plt.xlabel("x")
plt.ylabel("p")

plt.tight_layout()
plt.show()

del xxyy, xx, yy, x, y, zz, zz_x, zz_p


# %%
def diffusion(t, state, args):
    return lx.DiagonalLinearOperator(
        jnp.array([0, jnp.sqrt(2 * config.sde_config.Gamma * config.sde_config.beta)])
    )


def drift(t, state, args):
    Q = jnp.array(
        [
            [0.0, config.sde_config.beta],
            [
                -config.sde_config.beta,
                -config.sde_config.Gamma * config.sde_config.beta * (1 / config.sde_config.M),
            ],
        ]
    )
    return 1.0 * Q @ grad_z_H(state)  # (2,2) @ (2,) -> (2,)@


t0 = 0.0
t1 = 1.0
dt0 = 0.001

w_shape = (2 * config.sde_config.state_dim,)
"""     t0: RealScalarLike,
        t1: RealScalarLike,
        tol: RealScalarLike,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
        ] = BrownianIncrement,
        _spline: _Spline = "sqrt","""


rng = jrnd.PRNGKey(4)
bm = UnsafeBrownianPath(shape=(2 * config.sde_config.state_dim,), key=rng)


# Set up simulation parameters
num_steps = 100000
step_size = 0.1
t_final = num_steps * step_size

# Define SDE term
sde_term = MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))
y0 = jnp.zeros(2 * config.sde_config.state_dim)

# Set up solver
solver = ItoMilstein()
solver_state = solver.init(sde_term, t0=0.0, t1=t_final, y0=y0, args=None)

@jit
def ItoMilstein_step(t, y):
    next_t = t + step_size
    next_y = solver.step(sde_term, t, next_t, y, None, solver_state, made_jump=False)[0]
    return next_t, next_y

@jit
def simulation_step(carry):
    t, y, trajectory, i = carry
    next_t, next_y = ItoMilstein_step(t, y)
    trajectory = trajectory.at[i].set(next_y)
    return next_t, next_y, trajectory, i + 1

@jit
def simulation_loop(init_carry):
    def cond_fun(carry):
        t, _, _, i = carry
        return (t < t_final) & (i < num_steps)

    return jax.lax.while_loop(cond_fun, simulation_step, init_carry)

# Initialize trajectory array
trajectory = jnp.zeros((num_steps, 2 * config.sde_config.state_dim))

# Run the simulation
init_carry = (0.0, y0, trajectory, 0)
final_t, final_y, full_trajectory, _ = simulation_loop(init_carry)

# Generate time array
ts = jnp.linspace(0, final_t, num_steps)

# random subsample of 10000 for plotting
rng, subkey = jax.random.split(rng)
idxs = jax.random.choice(subkey, jnp.arange(num_steps), (10000,), replace=False)
ys = full_trajectory[idxs]
ts_plot = ts[idxs]

# %%
ys.shape
ys
#%%

# Plot the results
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    "Stochastic Oscillator System: Initial vs Final Distributions", fontsize=16
)

# Initial distribution of x
sns.histplot(init_x0s_p0s[:, 0], kde=True, color="blue", alpha=0.6, ax=axes[0, 0])
axes[0, 0].set_title("Initial Distribution of X", fontsize=14)
axes[0, 0].set_xlabel("X", fontsize=12)
axes[0, 0].set_ylabel("Density", fontsize=12)

# Initial distribution of p
sns.histplot(init_x0s_p0s[:, 1], kde=True, color="green", alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title("Initial Distribution of P", fontsize=14)
axes[0, 1].set_xlabel("P", fontsize=12)
axes[0, 1].set_ylabel("Density", fontsize=12)

# Initial joint distribution of x and p
sns.kdeplot(
    x=init_x0s_p0s[:, 0],
    y=init_x0s_p0s[:, 1],
    cmap="YlGnBu",
    shade=True,
    cbar=True,
    ax=axes[0, 2],
)
# sns.scatterplot(x=init_x0s_p0s[:, 0], y=init_x0s_p0s[:, 1], color='red', alpha=0.6, s=10, ax=axes[0, 2])
axes[0, 2].set_title("Initial Joint Distribution of X and P", fontsize=14)
axes[0, 2].set_xlabel("X", fontsize=12)
axes[0, 2].set_ylabel("P", fontsize=12)

# Final distribution of x
sns.histplot(ys[-1000:, 0], kde=True, color="red", alpha=0.6, ax=axes[1, 0])
axes[1, 0].set_title("Final Distribution of X", fontsize=14)
axes[1, 0].set_xlabel("X", fontsize=12)
axes[1, 0].set_ylabel("Density", fontsize=12)

# Final distribution of p
sns.histplot(ys[-1000:, 1], kde=True, color="purple", alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title("Final Distribution of P", fontsize=14)
axes[1, 1].set_xlabel("P", fontsize=12)
axes[1, 1].set_ylabel("Density", fontsize=12)

# Final joint distribution of x and p
sns.kdeplot(
    x=ys[-1000:, 0],
    y=ys[-1000:, 1],
    cmap="YlGnBu",
    shade=True,
    cbar=True,
    ax=axes[1, 2],
)
# sns.scatterplot(x=y_array[-1000:, 0], y=y_array[-1000:, 1], color='red', alpha=0.6, s=10, ax=axes[1, 2])
axes[1, 2].set_title("Final Joint Distribution of X and P", fontsize=14)
axes[1, 2].set_xlabel("X", fontsize=12)
axes[1, 2].set_ylabel("P", fontsize=12)

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Final time: {ts[-1]:.2f}")
print(f"Initial mean state: {jnp.mean(init_x0s_p0s, axis=0)}")
print(f"Initial state standard deviation: {jnp.std(init_x0s_p0s, axis=0)}")
print(f"Final mean state: {jnp.mean(ys[-1000:], axis=0)}")
print(f"Final state standard deviation: {jnp.std(ys[-1000:], axis=0)}")

# %%
