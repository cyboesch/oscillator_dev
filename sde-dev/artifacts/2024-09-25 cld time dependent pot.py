# %%
import diffrax
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
from plot_helpers import plot_time_dependent_energy
from ctmc import ContinuousTimeMarkovChain
import matplotlib.pyplot as plt
from thermoai.distributions import mog_logpdf, sample_mog
import seaborn as sns

# from jax import config
################
################
################
################
#%%
seed = 0

state_dim = 1

dt0 = 0.22
t0 = 0.0
T = 1000.0
N_timesteps = int((T - t0) / dt0) + 1
time_points = jnp.linspace(t0, T, N_timesteps)
temp_final = 1000.0
temp_0 = 1.0

means = jnp.array([-4.0, 7.0])
covariances = jnp.array([.1, .1])
weights = jnp.array([0.8, 0.2])
# --- Make time-dependent energy
# weights_t_fn = lambda t: jnp.array(
#     [weights[0] * (1 - (t-t0)/(T-t0)), weights[1] * (t-t0)/(T-t0)]
# )
# temp_fn = lambda t: jnp.where(t<T/10, jnp.array(
#     temp_final - (temp_final - temp_0) * (t/(T/10))
# ), jnp.array(
#     temp_0
# ))

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def reverse_sigmoid(x, k=10):
    return 1 - sigmoid(k * (x - 0.5))

temp_fn = lambda t: temp_0 + (temp_final - temp_0) * reverse_sigmoid(t / T)

plt.figure(figsize=(12, 6))


# Calculate temperatures
temperatures = jax.vmap(temp_fn)(time_points)
temperatures[-1]
#%%

# Plot the temperature profile
plt.plot(time_points, temperatures, color='red', linewidth=2)

# Customize the plot
plt.title("Temperature Profile", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Temperature", fontsize=12)
plt.grid(True)


# Show the plot
plt.tight_layout()
plt.show()


#%%
params_fn = lambda t: {
    "means": means,
    "covariances": covariances * temp_fn(t),
    "weights": weights,
}
target_logdensity_fn = jax.jit(lambda t, x, args: mog_logpdf(x, **params_fn(t)))
#%%
Zero = jnp.zeros((state_dim, state_dim))
Identity = jnp.eye(state_dim)
mass = 1.0 * Identity
damping = jnp.sqrt(4.0 * mass)
def U(t, z, args):
    x = z[0]
    return -1. * target_logdensity_fn(t, x, args)

def V(t, z, args):
    p = z[state_dim:]
    return 0.5 * jnp.dot(p, jnp.linalg.inv(mass) @ p)

H = lambda t, z, args: U(t, z, args) + V(t, z, args)

D = lambda t, z, args: jnp.block(
    [
        [Zero, Zero],
        [Zero, damping * Identity],
    ]
)

Q = lambda t, z, args: jnp.block(
    [
        [Zero, Identity],
        [-Identity, Zero],
    ]
)

# Set to zero for CLD so we don't waste time computing it.
tau = lambda t, z, args: jnp.zeros((2 * state_dim,))
################
################
################
################

#%%
target_ctmc = ContinuousTimeMarkovChain(H_fn=H, D_fn=D, Q_fn=Q, Gamma_fn=tau)
solver = diffrax.ItoMilstein()

def run_time_evolving_cdl(y0,key):
    key, subkey = jrnd.split(key)

    key, bm_key = jrnd.split(key)
    brownian_motion = diffrax.UnsafeBrownianPath(shape=(2 * state_dim,), key=bm_key)
    target_ctmc_terms = target_ctmc.get_terms(brownian_motion)

    ctmc_solution = diffrax.diffeqsolve(
        terms=target_ctmc_terms,
        solver=solver,
        t0=t0,
        t1=T,
        dt0=dt0,
        y0=y0,
        args=None,
        adjoint=diffrax.DirectAdjoint(),
        max_steps=N_timesteps,
        solver_state=solver.init(target_ctmc_terms, t0, T, y0, None),
        saveat=diffrax.SaveAt(ts=jnp.arange(t0, T, dt0)),
        progress_meter=diffrax.TqdmProgressMeter(),
    )
    ctmc_samples_x = ctmc_solution.ys[:, :state_dim]
    ctmc_samples_p = ctmc_solution.ys[:, state_dim:]
    
    return ctmc_samples_x, ctmc_samples_p
# %%
seed_ini = 1
key_ini = jrnd.PRNGKey(seed_ini)
N_initialconds = 1000
keys = jrnd.split(key_ini, N_initialconds)
samples_t = 0
initial_position = jax.vmap(lambda key: sample_mog(key, **params_fn(samples_t)))(keys)
initial_velocity = jnp.zeros_like(initial_position)
z0 = jnp.vstack([initial_position, initial_velocity])

seed_bm = 2
key_bm = jrnd.PRNGKey(seed_bm)
keys_bm = jrnd.split(key_bm, N_initialconds)

#%%
ctmc_samples_x, ctmc_samples_p = jax.vmap(run_time_evolving_cdl)(z0.T, keys_bm)

#%%

U_xt = lambda x, t: -target_logdensity_fn(t, x,None)

plot_time_dependent_energy(U_xt, T=T, n_lines=50, x_min=-20.0, x_max=20.0, y_min=0.5, y_max=10.0, reverse=False)
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
mog_samples = jax.vmap(lambda key: sample_mog(key, **params_fn(samples_t)))(keys)


font_size = 20
# Create a figure with four subplots
fig, ((ax1,ax3, ax4)) = plt.subplots(3, 1, figsize=(20, 16))

ax1_right = ax1.twinx()

# Plot histograms for x
sns.histplot(
    ctmc_samples_x[:,-1].flatten(),
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



x_values = jnp.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], N_timesteps)
U_values = jax.vmap(lambda x: U(samples_t, jnp.array([x, 0.0]), None))(x_values)
exp_neg_U = jnp.exp(-U_values)
# ax1_right = ax1.twinx()
ax1_right.plot(x_values, exp_neg_U, color='green', label='p(x)')

# Customize the density plot for x
ax1.set_title(f"Comparison adiabatic sampling vs true probability distribution; final temperature = {temp_final}; integration time = {T}", fontsize=font_size)
ax1.set_xlabel("Value", fontsize=font_size)
ax1.set_ylabel("Density", fontsize=font_size)


# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_right.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=font_size, loc='upper left')

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
energy_values_x = jax.vmap(jax.vmap(lambda x, t: U(t, jnp.array([x, 0.0]), None)))(X, T_x)

# Calculate the energy values for momentum (using potential function V)
energy_values_p = jax.vmap(jax.vmap(lambda p, t: V(t, jnp.array([0.0, p]), None)))(P, T_p)

# Plot colormaps
im_x = ax3.pcolormesh(T_x, X, energy_values_x, cmap='viridis', norm=LogNorm(), shading='auto')#norm=LogNorm(),
im_p = ax4.pcolormesh(T_p, P, energy_values_p, cmap='viridis',  shading='auto')#norm=LogNorm(),

# Add colorbars
fig.colorbar(im_x, ax=ax3, label='Potential Energy')
fig.colorbar(im_p, ax=ax4, label='Kinetic Energy')

# Plot individual trajectories
num_trajectories = min(50, N_initialconds)  # Limit to 100 trajectories for clarity
for i in range(num_trajectories):
    ax3.plot(time_points, ctmc_samples_x[i], color='white', alpha=0.1, linewidth=3.)
    ax4.plot(time_points, ctmc_samples_p[i], color='white', alpha=0.1, linewidth=3.)

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
