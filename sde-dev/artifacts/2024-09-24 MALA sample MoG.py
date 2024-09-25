#%%
from typing import Callable
from pathlib import Path
from diffrax import Euler, VirtualBrownianTree
import diffrax
import pytest
import jax
import jax.numpy as jnp
import jax.random as jrnd
from ctmc import ContinuousTimeMarkovChain
from mh_solver import MetropolisAdjustedSolver
from mala import sample as mala_sample
from jax.scipy.stats import multivariate_normal as MVN
import matplotlib.pyplot as plt
from thermoai.distributions import mog_logpdf, sample_mog
from jaxtyping import Float
import tqdm
import seaborn as sns

from mala import compute_log_rho

jax.config.update("jax_enable_x64", True)
seed = 0
step_size = 0.22
# VBT MALA setup

# # Create a time-dependent energy function
# state_dim = 1
# means = jnp.array([-4.0, 0.0])
# covariances = jnp.array([1.0, 1.0])
# weights = jnp.array([0.5, 0.5])
#- Create a time-dependent energy function
state_dim = 1
means = jnp.array([-4.0, 4.0])
covariances = jnp.array([0.1, 0.1])
weights = jnp.array([0.1, 0.9])
# - Make time-dependent covariances
T=10000.0
t0 = 0.0
temp_final = 10.0

# --- Make time-dependent weights between t0 and T
# weights_t_fn = lambda t: jnp.array(
#     [weights[0] * (1 - (t-t0)/(T-t0)), weights[1] * (t-t0)/(T-t0)]
# )

params_fn = lambda t: {
    "means": means,
    "covariances": covariances*temp_final,
    "weights": weights,
}

initial_position = jnp.zeros(state_dim)
target_logdensity_fn = jax.jit(lambda t, x, args: mog_logpdf(x, **params_fn(t)))
H = lambda t, z, args:  -1.0*target_logdensity_fn(t, z, args)

key = jrnd.PRNGKey(seed)
key, subkey = jrnd.split(key)

bm_key, mh_key = jrnd.split(key)
bm_vbt = VirtualBrownianTree(t0=t0, t1=T, tol=step_size, shape=(state_dim,), key=key)
bm_ubt = diffrax.UnsafeBrownianPath(shape=(state_dim,), key=bm_key)
target_ctmc = ContinuousTimeMarkovChain(H_fn=H)
sde_terms_vbt = target_ctmc.get_terms(bm_ubt)
mala_solver = MetropolisAdjustedSolver(logdensity_val_and_grad_fn=jax.jit(jax.value_and_grad(target_logdensity_fn, argnums=1)), log_accept_ratio_fn=jax.jit(compute_log_rho), solver=Euler())


# Generate samples using VBT MALA
vbt_solution = diffrax.diffeqsolve(
    terms=sde_terms_vbt,
    solver=mala_solver,
    t0=t0,
    t1=T,
    dt0=step_size,
    y0=initial_position,
    args=None,
    adjoint=diffrax.DirectAdjoint(),
    max_steps=int((T - t0) / step_size) + 1,
    solver_state=mala_solver.init(bm_ubt, t0, T, initial_position, None, mh_key),
    saveat=diffrax.SaveAt(ts=jnp.arange(t0, T, step_size)),
    progress_meter=diffrax.TqdmProgressMeter()
)
vbt_samples = vbt_solution.ys

# Import seaborn and update matplotlib style
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
keys = jrnd.split(key, 100000)
mog_samples = jax.vmap(lambda key: sample_mog(key, **params_fn(0)))(keys)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

# Plot histograms using seaborn on the first subplot
sns.histplot(vbt_samples.flatten(), kde=True, stat="density", label="VBT MALA Samples", color="skyblue", alpha=0.6, ax=ax1)
sns.histplot(mog_samples.flatten(), kde=True, stat="density", label="True MoG Samples", color="salmon", alpha=0.6, ax=ax1)

# Customize the density plot
ax1.set_title("Comparison of MALA Samples vs True Mixture of Gaussians", fontsize=16)
ax1.set_xlabel("Value", fontsize=12)
ax1.set_ylabel("Density", fontsize=12)
ax1.legend(fontsize=10)

# Plot trace plot on the second subplot
ax2.plot(jnp.arange(len(vbt_samples)), vbt_samples.flatten(), alpha=0.6)
ax2.set_title("Trace Plot of MALA Samples", fontsize=16)
ax2.set_xlabel("Sample Index", fontsize=12)
ax2.set_ylabel("Value", fontsize=12)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
# %%
