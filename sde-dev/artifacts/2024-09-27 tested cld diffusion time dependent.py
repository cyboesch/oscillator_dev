# %%
from typing import Callable, Dict, NamedTuple, Union
import diffrax
import jax
# jax.config.update("jax_disable_jit", True)
from jax.typing import ArrayLike
from tabulate import tabulate

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
from plot_helpers import plot_time_dependent_energy
from ctmc_physical import ContinuousTimeMarkovChain
import matplotlib.pyplot as plt
from thermoai.distributions import mog_logpdf, sample_mog
import seaborn as sns

# %%
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


# from jax import config
################
################
################
################
# %%


seed = 0

state_dim: StateDim = 1

dt0: Time = 0.022
t0: Time = 0.0
T: Time = 500.0
num_timesteps: Count = int((T - t0) / dt0) + 1

time_points: ArrayLike = jnp.linspace(t0, T, num_timesteps)

temp_final: Temperature = 1000.0
temp_0: Temperature = 4.0

num_mixture_components: Count = 3

means = jnp.array([-4.0, 7.0, 4.0])
covariances = jnp.array([0.6, 0.03, 0.01])
weights = jnp.array([0.7, 0.2, 0.1])

# %%
def create_block_diagonal_cov(variances: ArrayLike, state_dim: StateDim) -> Matrix:
    """
    Create a block diagonal covariance matrix.
    
    Args:
    covariances (ArrayLike): 1D array of covariance values for each component.
    state_dim (int): Dimension of the state space.
    
    Returns:
    ArrayLike: Block diagonal covariance matrix.
    """
    num_components = len(variances)
    covariances_nd = jnp.zeros((num_components, state_dim, state_dim))
    for i, c in enumerate(variances):
        covariances_nd = covariances_nd.at[i].set(jnp.eye(state_dim) * c)
    return covariances_nd

# --- Mixture of 2 100-D Gaussians

def create_means_nd(means: Vector, num_mixture_components: Count, state_dim: StateDim) -> Matrix:
    return jnp.ones((num_mixture_components, state_dim)) * means[:, None]

means_nd = create_means_nd(means, num_mixture_components, state_dim)
covariances_nd = create_block_diagonal_cov(covariances, state_dim)
print(f"means_nd.shape: {means_nd.shape}")
print(f"covariances_nd.shape: {covariances_nd.shape}")
#%%

# --- Helper functions to unpack and pack states
def make_state_fns(state_dim: int) -> tuple[Callable, ...]:
    def get_position(state: State) -> Position:
        if state.shape[0] != 2 * state_dim:
            raise ValueError(f"State must have {2 * state_dim} elements, got {state.shape[0]}")
        return state[..., :state_dim]

    def get_velocity(state: State) -> Velocity:
        if state.shape[0] != 2 * state_dim:
            raise ValueError(f"State must have {2 * state_dim} elements, got {state.shape[0]}")
        return state[..., state_dim:]

    def unpack_state(state: State) -> tuple[Position, Velocity]:
        if state.shape[0] != 2 * state_dim:
            raise ValueError(f"State must have {2 * state_dim} elements, got {state.shape[0]}")
        return get_position(state), get_velocity(state)

    def pack_state(position: Position, velocity: Velocity) -> State:
        if position.shape[0] != state_dim:
            raise ValueError(f"Position must have {state_dim} elements, got {position.shape[0]}")
        if velocity.shape[0] != state_dim:
            raise ValueError(f"Velocity must have {state_dim} elements, got {velocity.shape[0]}")
        return jnp.hstack([position, velocity])

    return get_position, get_velocity, unpack_state, pack_state


get_position, get_velocity, unpack_state, pack_state = make_state_fns(state_dim)
#%%
# --- Make time-dependent energy
# weights_t_fn = lambda t: jnp.array(
#     [weights[0] * (1 - (t-t0)/(T-t0)), weights[1] * (t-t0)/(T-t0)]
# )
# temp_fn = lambda t: jnp.where(t<T, jnp.array(
#     temp_final - (temp_final - temp_0) * (t/(T))
# ), jnp.array(
#     temp_0
# ))

temp_fn = lambda t: 1/((1/temp_0-1/temp_final)*(t/T) + 1/temp_final)

# def sigmoid(x):
#     return 1 / (1 + jnp.exp(-x))

# def reverse_sigmoid(x, k=13):
#     return 1 - sigmoid(k * (x - 0.5))

# temp_fn = lambda t: temp_0 + (temp_final - temp_0) * reverse_sigmoid(t / T)

# def temp_fn(t: Time) -> Scalar:
#     # Constants (you may want to define these in your Args or as global constants)
#     a = 1  # Acceleration parameter (adjust as needed)

#     numerator = 1.0
#     denominator = (
#         ((1 / temp_0 - 1 / temp_final) / (T + a * T**2 / 2)) * (t + a * t**2 / 2)
#         + 1 / temp_final
#     )
    
#     return numerator / denominator


# Calculate temperatures
temperatures = jax.vmap(temp_fn)(time_points)
temperatures[-1]
# %%

# Plot the temperature profile
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


def generate_params_1d(t: Time) -> MixtureParams:
    return MixtureParams(
        means=means,
        covariances=covariances*temp_0 ,
        weights=weights,
    )
    
def generate_params_nd(t: Time) -> MixtureParams:
    return MixtureParams(
        means=means_nd,
        covariances=covariances_nd*temp_0,
        weights=weights,
    )


def target_logdensity_fn(t: Time, x: Position, args: Args) -> Scalar:
    return mog_logpdf(x, **args.params_fn(t))

#%%
# --- System parameters
Zero: Matrix = jnp.zeros((state_dim, state_dim))
Identity: Matrix = jnp.eye(state_dim)
mass: Matrix = 1.0 * Identity
damping: Scalar = jnp.sqrt(4.0 * mass)

sys_params: SystemParams = SystemParams(mass=mass, damping=damping)
args: Args = Args(params_fn=generate_params_nd, system=sys_params)

#%%
generate_params_1d(0.0).values()
for x in generate_params_nd(0.0).values():
    print(x.shape)
val = target_logdensity_fn(0.0, get_position(jnp.zeros(2 * state_dim)), args)
print(val)

#%%
def U(t: Time, y: State, args: Args) -> Scalar:
    return -1.0 * target_logdensity_fn(t, get_position(y), args)


def V(t: Time, y: State, args: Args) -> Scalar:
    M_inv = jnp.linalg.inv(args.system.mass)
    v = get_velocity(y)
    return 0.5 * jnp.sum(v * (M_inv @ v), axis=-1)


H: Callable[[Time, State, Args], Scalar] = lambda t, z, args: U(t, z, args) + V(
    t, z, args
)

# %%
D: Callable[[Time, State, Args], Matrix] = lambda t, z, args: jnp.block(
    [
        [Zero, Zero],
        [Zero, damping * Identity*temp_fn(t)],
    ]
)

Q: Callable[[Time, State, Args], Matrix] = lambda t, z, args: jnp.block(
    [
        [Zero, Identity],
        [-Identity, Zero],
    ]
)

damping_fn: Callable[[Time, State, Args], Matrix] = lambda t, z, args: jnp.block(
    [
        [Zero, Zero],
        [Zero, damping],
    ]
)

# Set to zero for CLD so we don't waste time computing it.
tau_fn: Callable[[Time, State, Args], Vector] = lambda t, z, args: jnp.zeros(
    (2 * state_dim,)
)
################
################
################
################

# %%


def sample_position(key: jrnd.PRNGKey, args: Args, t: Time) -> Position:
    return sample_mog(key, **args.params_fn(t))


def sample_velocity(key: jrnd.PRNGKey, args: Args, t: Time) -> Velocity:
    return jnp.zeros(state_dim)


def sample_initial_condition(key: jrnd.PRNGKey, args: Args, t: Time) -> State:
    pos_key, vel_key = jrnd.split(key)
    position = sample_position(pos_key, args, t)
    velocity = sample_velocity(vel_key, args, t)
    return pack_state(position, velocity)


target_ctmc = ContinuousTimeMarkovChain(H_fn=H, D_fn=D, Q_fn=Q, damping_fn=damping_fn, Gamma_fn=tau_fn)
solver = diffrax.ItoMilstein()
saveat_ts = jnp.arange(t0, T, dt0)


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
        saveat=diffrax.SaveAt(ts=saveat_ts),
        progress_meter=diffrax.TqdmProgressMeter(),
    )

    return jax.vmap(unpack_state)(ctmc_solution.ys)  # xs, vs


# %%
seed_ini = 1
key_ini = jrnd.PRNGKey(seed_ini)
N_initialconds = 10000//2
keys = jrnd.split(key_ini, N_initialconds)
samples_t = 0
y0: ArrayLike = jax.vmap(lambda key: sample_initial_condition(key, args, samples_t))(
    keys
)

# %%
y0

# %%

seed_bm = 2
key_bm = jrnd.PRNGKey(seed_bm)
keys_bm = jrnd.split(key_bm, N_initialconds)

# %%
# -- Print out simulation parameters and then run simulation
parameters = [
    ["state_dim", state_dim],
    ["dt0", dt0],
    ["t0", t0],
    ["T", T],
    ["N_timesteps", num_timesteps],
    ["temp_final", temp_final],
    ["temp_0", temp_0],
    ["means", means],
    ["covariances", covariances],
    ["weights", weights],
]

table = tabulate(parameters, headers=["Parameter", "Value"], tablefmt="grid")
print("Simulation Parameters:")
print(table)


xs, vs = jax.vmap(run_time_evolving_cdl, in_axes=(0, None, 0))(y0, args, keys_bm)
print(f"xs.shape: {xs.shape}\nvs.shape: {vs.shape}")

# %%
vs
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

# N_samples = 10000

# sns.set_theme(style="darkgrid")
# key_sample = jrnd.PRNGKey(seed_ini)
# keys = jrnd.split(key_sample, N_samples)
# samples_t = T
# mog_samples = jax.vmap(lambda key: sample_mog(key, **generate_params_1d(samples_t)))(keys)


font_size = 20
# Create a figure with four subplots
fig, ((ax1, ax3, ax4)) = plt.subplots(3, 1, figsize=(20, 16))

ax1_right = ax1.twinx()
# Plot histograms for x
sns.histplot(
    xs[:, -1].flatten(),
    bins=100,
    stat="density",
    label="CTMC Samples",
    color="skyblue",
    alpha=0.6,
    ax=ax1,
)
# sns.histplot(
#     mog_samples.flatten(),
#     stat="density",
#     label=f"MoG Samples at t={samples_t}",
#     color="salmon",
#     alpha=0.6,
#     ax=ax1,
# )


x_values = jnp.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], num_timesteps)
U_values = jax.vmap(lambda x: U(time_points[-1], jnp.array([x, 0.0]), args))(x_values)
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
num_trajectories = min(30, N_initialconds)  # Limit to 100 trajectories for clarity


for i in range(num_trajectories):
    ax3.plot(
        time_points,
        xs[i+100],
        color="white",
        alpha=0.1,
        linewidth=5.0,
    )
    ax4.plot(
        time_points,
        vs[i],
        color="white",
        alpha=0.1,
        linewidth=5.0,
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
#%%
