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
################
################
################
################
#%%
seed = 0

state_dim = 1

dt0 = 0.22
t0 = 0.0
T = 10000.0
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
temp_fn = lambda t: jnp.array(
    temp_final - (temp_final - temp_0) * (t/T)
)
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

U_xt = lambda x, t: -target_logdensity_fn(t, x,None)

plot_time_dependent_energy(U_xt, T=T, n_lines=50, x_min=-20.0, x_max=20.0, y_min=0.5, y_max=10.0, reverse=False)

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
    
    return ctmc_samples_x.reshape(-1)[-1], ctmc_samples_p.reshape(-1)[-1]
# %%
seed_ini = 1
key_ini = jrnd.PRNGKey(seed_ini)
N_initialconds = 10
keys = jrnd.split(key_ini, N_initialconds)
samples_t = 0
initial_position = jax.vmap(lambda key: sample_mog(key, **params_fn(samples_t)))(keys)
initial_velocity = jnp.zeros_like(initial_position)
z0 = jnp.vstack([initial_position, initial_velocity])

seed_brownian = 2
key_brownian = jrnd.PRNGKey(seed_brownian)
print(key_brownian.shape)
keys_brownian = jrnd.split(key_ini, N_initialconds)
keys_brownian[:,0]
#%%

key = jrnd.PRNGKey(seed)
initial_position = jnp.array([-2.0]) #jnp.zeros(state_dim)
initial_velocity = jnp.zeros(state_dim)
y0 = jnp.concatenate([initial_position, initial_velocity])

ctmc_samples_x, ctmc_samples_p = run_time_evolving_cdl(y0,key)

# %%
# Import seaborn and update matplotlib style
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
keys = jrnd.split(key, N_timesteps)
samples_t = T
mog_samples = jax.vmap(lambda key: sample_mog(key, **params_fn(samples_t)))(keys)



# Create a figure with four subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

ax1_right = ax1.twinx()

# Plot histograms for x
sns.histplot(
    ctmc_samples_x.flatten(),
    kde=True,
    stat="density",
    label="CTMC Samples",
    color="skyblue",
    alpha=0.6,
    ax=ax1,
)
sns.histplot(
    mog_samples.flatten(),
    kde=True,
    stat="density",
    label=f"MoG Samples at t={samples_t}",
    color="salmon",
    alpha=0.6,
    ax=ax1,
)



x_values = jnp.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], N_timesteps)
U_values = jax.vmap(lambda x: U(samples_t, jnp.array([x, 0.0]), None))(x_values)
exp_neg_U = jnp.exp(-U_values)
ax1_right = ax1.twinx()
ax1_right.plot(x_values, exp_neg_U, color='green', label='exp(-U(x))')

# Customize the density plot for x
ax1.set_title("Comparison of CTMC Samples (x) vs Prior", fontsize=16)
ax1.set_xlabel("Value", fontsize=16)
ax1.set_ylabel("Density", fontsize=16)

# Remove the y-axis label and tick labels for the right axis, keep green ticks
ax1_right.set_ylabel("")
ax1_right.tick_params(axis='y', colors='green', labelcolor='green')
ax1_right.set_yticklabels([])  # Remove tick labels from right axis

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_right.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')

# Set the same y-axis limits for both left and right axes
y_min = min(ax1.get_ylim()[0], ax1_right.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax1_right.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax1_right.set_ylim(y_min, y_max)

# ... rest of the existing code ...

# Plot histograms for p
sns.histplot(
    ctmc_samples_p.flatten(),
    kde=True,
    stat="density",
    label="CTMC Samples (p)",
    color="lightgreen",
    alpha=0.6,
    ax=ax2,
)

# Customize the density plot for p
ax2.set_title("Distribution of CTMC Samples (p)", fontsize=16)
ax2.set_xlabel("Value", fontsize=16)
ax2.set_ylabel("Density", fontsize=16)
ax2.legend(fontsize=10)

# Plot trace plot for x on the third subplot
ax3.plot(jnp.arange(len(ctmc_samples_x)), ctmc_samples_x.flatten(), alpha=0.6)
ax3.set_title("Trace Plot of CTMC Samples (x)", fontsize=16)
ax3.set_xlabel("Sample Index", fontsize=16)
ax3.set_ylabel("Value", fontsize=16)

# Plot trace plot for p on the fourth subplot
ax4.plot(jnp.arange(len(ctmc_samples_p)), ctmc_samples_p.flatten(), alpha=0.6)
ax4.set_title("Trace Plot of CTMC Samples (p)", fontsize=16)
ax4.set_xlabel("Sample Index", fontsize=16)
ax4.set_ylabel("Value", fontsize=16)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
#%%

