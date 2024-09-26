# %%
# No GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# %%
import diffrax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
from ctmc import ContinuousTimeMarkovChain
import matplotlib.pyplot as plt
from thermoai.distributions import mog_logpdf, sample_mog
import seaborn as sns

seed = 0
dt0 = 0.22

# - Create a time-dependent energy function
state_dim = 1
means = jnp.array([-4.0, 4.0])
covariances = jnp.array([0.1, 0.1])
weights = jnp.array([0.1, 0.9])

# - Make time-dependent covariances
T = 10000.0
t0 = 0.0
temp_final = 10.0
Zero = jnp.zeros((state_dim, state_dim))
All_Ones = jnp.ones((state_dim, state_dim))
Identity = jnp.eye(state_dim)
mass = 1.0 * Identity
damping = jnp.sqrt(4.0 * mass)

# --- Make time-dependent weights between t0 and T
# weights_t_fn = lambda t: jnp.array(
#     [weights[0] * (1 - (t-t0)/(T-t0)), weights[1] * (t-t0)/(T-t0)]
# )

params_fn = lambda t: {
    "means": means,
    "covariances": covariances * temp_final,
    "weights": weights,
}

initial_position = jnp.zeros(state_dim)
initial_velocity = jnp.zeros(state_dim)
y0 = jnp.concatenate([initial_position, initial_velocity])

target_logdensity_fn = jax.jit(lambda t, x, args: mog_logpdf(x, **params_fn(t)))

def U(t, z, args):
    x = z[:state_dim]
    return -1.0 * target_logdensity_fn(t, x, args)

def V(t, z, args):
    p = z[state_dim:]
    return 0.5 * jnp.dot(p, jnp.linalg.inv(mass) @ p)

H = lambda t, z, args: U(t, z, args) + V(t, z, args)

key = jrnd.PRNGKey(seed)
key, subkey = jrnd.split(key)

key, bm_key = jrnd.split(key)
brownian_motion = diffrax.UnsafeBrownianPath(shape=(2*state_dim,), key=bm_key)

D = lambda z, args: jnp.block(
    [
        [Zero, Zero],
        [Zero, damping * Identity],
    ]
)

print(f"D shape: {D(y0, None).shape}")

Q = lambda z, args: jnp.block(
    [
        [Zero, Identity],
        [-Identity, Zero],
    ]
)
print(f"Q shape: {Q(y0, None).shape}")
#%%
target_ctmc = ContinuousTimeMarkovChain(H_fn=H, D_fn=D, Q_fn=Q)
target_ctmc_terms = target_ctmc.get_terms(brownian_motion)
solver = diffrax.ItoMilstein()

# %%
# Generate samples using VBT MALA
ctmc_solution = diffrax.diffeqsolve(
    terms=target_ctmc_terms,
    solver=solver,
    t0=t0,
    t1=T,
    dt0=dt0,
    y0=y0,
    args=None,
    adjoint=diffrax.DirectAdjoint(),
    max_steps=int((T - t0) / dt0) + 1,
    solver_state=solver.init(target_ctmc_terms, t0, T, y0, None),
    saveat=diffrax.SaveAt(ts=jnp.arange(t0, T, dt0)),
    progress_meter=diffrax.TqdmProgressMeter(),
)
ctmc_samples = ctmc_solution.ys
# %%
# Import seaborn and update matplotlib style
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
keys = jrnd.split(key, 100000)
mog_samples = jax.vmap(lambda key: sample_mog(key, **params_fn(0)))(keys)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

# Plot histograms using seaborn on the first subplot
sns.histplot(
    ctmc_samples.flatten(),
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
    label="True MoG Samples",
    color="salmon",
    alpha=0.6,
    ax=ax1,
)

# Customize the density plot
ax1.set_title("Comparison of CTMC Samples vs Directly Sampled Mixture of Gaussians", fontsize=16)
ax1.set_xlabel("Value", fontsize=12)
ax1.set_ylabel("Density", fontsize=12)
ax1.legend(fontsize=10)

# Plot trace plot on the second subplot
ax2.plot(jnp.arange(len(ctmc_samples)), ctmc_samples.flatten(), alpha=0.6)
ax2.set_title("Trace Plot of MALA Samples", fontsize=16)
ax2.set_xlabel("Sample Index", fontsize=12)
ax2.set_ylabel("Value", fontsize=12)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
# %%
