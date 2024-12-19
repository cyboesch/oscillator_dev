# %%
from typing import Callable, Dict, NamedTuple, Union
import diffrax
import jax
from jax.typing import ArrayLike
from tabulate import tabulate

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
from plot_helpers import plot_time_dependent_energy
from ctmc import ContinuousTimeMarkovChain
import matplotlib.pyplot as plt
from thermoai.distributions import mog_logpdf, sample_mog
import seaborn as sns

# --- Types

Time = float  # Real number representing time
StateDim = int  # Dimension of a position in state space
Count = int  # Natural number representing a quantity
Temperature = float  # Positive real number representing temperature
Scalar = float  # Positive real number representing a scalar quantity
Matrix = ArrayLike  # shape: (StateDim, StateDim)
Vector = ArrayLike  # shape: (StateDim,)

Position = ArrayLike  # shape: (StateDim,)
Velocity = ArrayLike  # shape: (StateDim,)
State = Union[Position, Velocity]  # shape: (2*StateDim,)


class MixtureParams(Dict):
    means: Vector
    covariances: Matrix
    weights: Vector


class SystemParams(NamedTuple):
    mass: Scalar | Matrix
    damping: Scalar | Matrix


class Args(NamedTuple):
    params_fn: Callable[[Time], MixtureParams]
    system: SystemParams


# --- Functions to unpack and pack states correctly

def make_state_fns(state_dim: int) -> tuple[Callable, ...]:
    def get_position(state: State) -> Position:
        return state[..., :state_dim]

    def get_velocity(state: State) -> Velocity:
        return state[..., state_dim:]

    def unpack_state(state: State) -> tuple[Position, Velocity]:
        return get_position(state), get_velocity(state)

    def pack_state(position: Position, velocity: Velocity) -> State:
        return jnp.hstack([position, velocity])

    return get_position, get_velocity, unpack_state, pack_state


################
################
################
################
# %%

# CONFIGURE SYSTEM PARAMETERS

state_dim: StateDim = 1
Zero: Matrix = jnp.zeros((state_dim, state_dim))
Identity: Matrix = jnp.eye(state_dim)
mass: Matrix = 1.0 * Identity
damping: Scalar = jnp.sqrt(4.0 * mass)

# Make state accessor functions
get_position, get_velocity, unpack_state, pack_state = make_state_fns(state_dim)

# Make single object to hold system parameters
sys_params: SystemParams = SystemParams(mass=mass, damping=damping)

# CONFIGURE SIMULATION PARAMETERS AND FNs

seed = 0
dt0: Time = 0.22
t0: Time = 0.0
T: Time = 1000.0
num_timesteps: Count = int((T - t0) / dt0) + 1
time_points: ArrayLike = jnp.linspace(t0, T, num_timesteps)
temp_final: Temperature = 1000.0
temp_0: Temperature = 1.0


def sigmoid(x: Position) -> Position:
    return 1 / (1 + jnp.exp(-x))


def reverse_sigmoid(x: Position, k: Scalar = 10.0) -> Position:
    return 1 - sigmoid(k * (x - 0.5))


def temp_fn(t: Time) -> Scalar:
    return temp_0 + (temp_final - temp_0) * reverse_sigmoid(t / T)


# CONFIGURE ENERGY FUNCTION PARAMETERS AND FNs

num_mixture_components: Count = 2
means = jnp.array([-4.0, 7.0])
covariances = jnp.array([0.1, 0.1])
weights = jnp.array([0.8, 0.2])


def get_params(t: Time) -> MixtureParams:
    return MixtureParams(
        means=means,
        covariances=covariances * temp_fn(t),
        weights=weights,
    )


def target_logdensity_fn(t: Time, x: Position, args: Args) -> Scalar:
    return mog_logpdf(x, **args.params_fn(t))


# %%
# Plot temperatures
plt.figure(figsize=(12, 6))
temperatures = jax.vmap(temp_fn)(time_points)
plt.plot(time_points, temperatures, color="red", linewidth=2)

# Customize the plot
plt.title("Temperature Profile", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Temperature", fontsize=12)
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# %%
# Create functions for CTMC

def U(t: Time, y: State, args: Args) -> Scalar:
    return -1.0 * target_logdensity_fn(t, get_position(y), args)


def V(t: Time, y: State, args: Args) -> Scalar:
    M_inv = jnp.linalg.inv(args.system.mass)
    v = get_velocity(y)
    return 0.5 * jnp.dot(v, (M_inv @ v))


def H(t: Time, z: State, args: Args) -> Scalar:
    return U(t, z, args) + V(t, z, args)


def D(t: Time, z: State, args: Args) -> Matrix:
    return jnp.block(
        [
            [Zero, Zero],
            [Zero, damping * Identity],
        ]
    )


def Q(t: Time, z: State, args: Args) -> Matrix:
    return jnp.block(
        [
            [Zero, Identity],
            [-Identity, Zero],
        ]
    )


# Set to zero for CLD so we don't waste time computing it.
def tau_fn(t: Time, z: State, args: Args) -> Vector:
    return jnp.zeros((2 * state_dim,))


################
################
################
################

# %%

def sample_initial_condition(key: jrnd.PRNGKey, args: Args, t: Time) -> State:
    position = sample_mog(key, **args.params_fn(t))
    velocity = jnp.zeros(state_dim)
    return pack_state(position, velocity)


target_ctmc = ContinuousTimeMarkovChain(H_fn=H, D_fn=D, Q_fn=Q, Gamma_fn=tau_fn)
solver = diffrax.ItoMilstein()


def run_time_evolving_cdl(
    y0: State, args: Args, bm_key: jrnd.PRNGKey
) -> tuple[Position, Velocity]:
    brownian_motion = diffrax.UnsafeBrownianPath(shape=(2 * state_dim,), key=bm_key)
    target_ctmc_terms = target_ctmc.get_terms(brownian_motion)

    ctmc_solution = diffrax.diffeqsolve(
        terms=target_ctmc_terms,
        solver=solver,
        t0=t0,
        t1=T,
        dt0=dt0,
        y0=y0,
        args=args,
        adjoint=diffrax.DirectAdjoint(),
        max_steps=num_timesteps,
        solver_state=None,  # ItoMilstein does not need solver_state
        saveat=diffrax.SaveAt(ts=jnp.arange(t0, T, dt0)),
        progress_meter=diffrax.TqdmProgressMeter(),
    )

    return jax.vmap(unpack_state)(ctmc_solution.ys)  # xs, vs


# %%
seed_ini = 1
key_ini = jrnd.PRNGKey(seed_ini)
N_initialconds = 1000
keys = jrnd.split(key_ini, N_initialconds)
samples_t = 0
args = Args(params_fn=get_params, system=sys_params)

# SAMPLE INTIIAL POSITION AND VELOCITY
x0s_and_v0s: State = jax.vmap(lambda key: sample_initial_condition(key, args, samples_t))(keys)
seed_bm = 2
key_bm = jrnd.PRNGKey(seed_bm)
keys_bm = jrnd.split(key_bm, N_initialconds)

# %%

# RUN SIMULATION
xs, vs = jax.vmap(run_time_evolving_cdl, in_axes=(0, None, 0))(x0s_and_v0s, args, keys_bm)

# %%

U_xt: Callable[[ArrayLike, Time], Scalar] = lambda x, t: -target_logdensity_fn(
    t, x, args
)

plot_time_dependent_energy(
    U_xt, T=T, n_lines=50, x_min=-20.0, x_max=20.0, y_min=0.5, y_max=10.0, reverse=False
)
# %%
# Import seaborn and update matplotlib style
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

N_samples = 10000

sns.set_theme(style="darkgrid")
key_sample = jrnd.PRNGKey(seed_ini)
keys = jrnd.split(key_sample, N_samples)
samples_t = T
mog_samples = jax.vmap(lambda key: sample_mog(key, **args.params_fn(samples_t)))(keys)


font_size = 20
# Create a figure with four subplots
fig, ((ax1, ax3, ax4)) = plt.subplots(3, 1, figsize=(20, 16))

ax1_right = ax1.twinx()
# Plot histograms for x
sns.histplot(
    xs[:, -1].flatten(),
    stat="density",
    label="CTMC Samples",
    color="skyblue",
    alpha=0.6,
    ax=ax1,
)
sns.histplot(
    mog_samples.flatten(),
    stat="density",
    label=f"MoG Samples at t={samples_t}",
    color="salmon",
    alpha=0.6,
    ax=ax1,
)


x_values = jnp.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], num_timesteps)
U_values = jax.vmap(lambda x: U(samples_t, jnp.array([x, 0.0]), args))(x_values)
exp_neg_U = jnp.exp(-U_values)
# ax1_right = ax1.twinx()
ax1_right.plot(x_values, exp_neg_U, color="green", label="p(x)")

# Customize the density plot for x
ax1.set_title(
    f"Comparison adiabatic sampling vs true probability distribution; final temperature = {temp_final}; integration time = {T}",
    fontsize=font_size,
)
ax1.set_xlabel("Value", fontsize=font_size)
ax1.set_ylabel("Density", fontsize=font_size)


# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_right.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=font_size, loc="upper left")

# Set the same y-axis limits for both left and right axes
y_min = min(ax1.get_ylim()[0], ax1_right.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax1_right.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax1_right.set_ylim(y_min, y_max)

# After the existing plot code, add the following:

# Create meshgrids for the colormaps
x_range = jnp.linspace(-20, 20, 500)
p_range = jnp.linspace(-20, 20, 500)
t_range = jnp.linspace(t0, T, 500)
X, T_x = jnp.meshgrid(x_range, t_range)
P, T_p = jnp.meshgrid(p_range, t_range)

# Calculate the energy values for position
energy_values_x = jax.vmap(jax.vmap(lambda x, t: U(t, jnp.array([x, 0.0]), args)))(
    X, T_x
)

# Calculate the energy values for momentum (using potential function V)
energy_values_p = jax.vmap(jax.vmap(lambda p, t: V(t, jnp.array([0.0, p]), args)))(
    P, T_p
)

# Plot colormaps
im_x = ax3.pcolormesh(
    T_x, X, energy_values_x, cmap="viridis", norm=LogNorm(), shading="auto"
)  # norm=LogNorm(),
im_p = ax4.pcolormesh(
    T_p, P, energy_values_p, cmap="viridis", shading="auto"
)  # norm=LogNorm(),

# Add colorbars
fig.colorbar(im_x, ax=ax3, label="Potential Energy")
fig.colorbar(im_p, ax=ax4, label="Kinetic Energy")

# Plot individual trajectories
num_trajectories = min(50, N_initialconds)  # Limit to 100 trajectories for clarity

# %%
# %%
for i in range(num_trajectories):
    ax3.plot(
        time_points,
        xs[i],
        color="white",
        alpha=0.1,
        linewidth=3.0,
    )
    ax4.plot(
        time_points,
        vs[i],
        color="white",
        alpha=0.1,
        linewidth=3.0,
    )

# Customize the plots
ax3.set_title("Position Trajectories on Potential Energy Landscape", fontsize=font_size)
ax3.set_xlabel("Time", fontsize=font_size)
ax3.set_ylabel("Position (x)", fontsize=font_size)
ax3.set_ylim(-20, 20)

ax4.set_title("Momentum Trajectories on Kinetic Energy Landscape", fontsize=font_size)
ax4.set_xlabel("Time", fontsize=font_size)
ax4.set_ylabel("Momentum (p)", fontsize=font_size)
ax4.set_ylim(-20, 20)

plt.tight_layout()
plt.show()

# %%
