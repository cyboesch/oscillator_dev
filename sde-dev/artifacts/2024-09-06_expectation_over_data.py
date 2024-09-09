# %%
# %%
# %%
from typing import Union
import jax.numpy as jnp
import jax.random as jrnd
from diffrax import MultiTerm, ODETerm, ControlTerm, ItoMilstein, UnsafeBrownianPath
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from jax import jit
from jax.lax import map

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
# Plot the results
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

config.update("jax_enable_x64", True)

from thermoai.distributions import sample_1d_mog

cpu_devices = devices("cpu")

from dataclasses import dataclass
import jax.random as jr
from cld import CriticallyDampedLangevinDynamics



@dataclass
class CLDConfig:
    state_dim: int
    beta: float
    M: float
    gamma: float
    Gamma: Union[float, None] = None

@dataclass
class InitConfig:
    N: int
    rng: jr.PRNGKey
    step_size: float

@dataclass
class SimulationConfig:
    init_seed: int
    init_config: InitConfig
    cld_config: CLDConfig

    @classmethod
    def create(cls, init_seed: int = 0):
        return cls(
            init_seed=init_seed,
            init_config=InitConfig(
                N=1_000_000,
                rng=jr.PRNGKey(init_seed),
                step_size=0.01
            ),
            cld_config=CLDConfig(
                state_dim=1,
                beta=1.0,
                M=1.0,
                gamma=1.0,
            )
        )
config = SimulationConfig.create()
cld = CriticallyDampedLangevinDynamics(**config.cld_config.__dict__)

#%%
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
    "stddev1": jnp.sqrt(config.cld_config.gamma * config.cld_config.M),
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


rng = jrnd.PRNGKey(4)
bm = UnsafeBrownianPath(shape=(2 * cld.state_dim,), key=rng)

sde_terms = cld.get_terms(bm)

# Set up simulation parameters
num_steps = config.init_config.N 
step_size = config.init_config.step_size
t_final = num_steps * step_size


# Set up solver
solver = ItoMilstein()
solver_state = solver.init(sde_terms, t0=0.0, t1=t_final, y0=jnp.empty(2 * cld.state_dim), args=None)

@jit
def ItoMilstein_step(t, y):
    next_t = t + step_size
    next_y = solver.step(sde_terms, t, next_t, y, None, solver_state, made_jump=False)[0]
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

# Modify the simulation to accept initial conditions
@jit
def run_simulation(y0):
    trajectory = jnp.zeros((num_steps, 2 * config.cld_config.state_dim))
    init_carry = (0.0, y0, trajectory, 0)
    _, _, full_trajectory, _ = simulation_loop(init_carry)
    return full_trajectory

# Set up multiple initial conditions
n_data_samples = 100
rand_init_idxs = np.random.choice(np.arange(config.init_config.N), n_data_samples, replace=False)
inits = init_x0s_p0s[rand_init_idxs]  # This should be your (n_inits, 2) array of initial conditions

# Run the simulation for all initial conditions using vmap
all_trajectories = jax.vmap(run_simulation)(inits)

# %%

#%%
# A_vv*(mu_v(t)-v(t))
print(inits.shape)
xs = all_trajectories[:, :, 0]
vs = all_trajectories[:, :, 1]
init_xs = inits[:, 0, None]
init_vs = inits[:, 1, None]
ts = jnp.linspace(0, t_final, num_steps)
#%%
HSM_cov = {
    "Sigma_0_xx": 0.0,
    "Sigma_0_vv": config.cld_config.gamma * config.cld_config.M,
}
Sigma_inv_xxs = vmap(lambda t: cld.Sigma_inv_xx_t(t, **HSM_cov))(ts)
Sigma_inv_vvs = vmap(lambda t: cld.Sigma_inv_vv_t(t, **HSM_cov))(ts)
Sigma_inv_xvs = vmap(lambda t: cld.Sigma_inv_xv_t(t, **HSM_cov))(ts)
mus = vmap(lambda x, v: vmap(lambda t: cld.mean(t, x, v))(ts))(init_xs, init_vs)
mus_xs = mus[:,:, 0]
mus_vs = mus[:,:, 1]
#%%
print(xs.shape)
print(mus_xs.shape)
#%%
# Vectorize over time and initial conditions
norms_vv = vmap(lambda v, mu_v: vmap(lambda t: jnp.abs(Sigma_inv_vvs[t] * (mu_v[t] - v[t])))(jnp.arange(len(ts))))(vs, mus_vs)

norms_xx = vmap(lambda x, mu_x: vmap(lambda t: jnp.abs(Sigma_inv_xxs[t] * (mu_x[t] - x[t])))(jnp.arange(len(ts))))(xs, mus_xs)

norms_xv = vmap(lambda x, mu_x: vmap(lambda t: jnp.abs(Sigma_inv_xvs[t] * (mu_x[t] - x[t])))(jnp.arange(len(ts))))(xs, mus_xs)
#%%
#%%
mean_norms_vv = jnp.nanmean(norms_vv, axis=0)
mean_norms_xx = jnp.nanmean(norms_xx, axis=0)
mean_norms_xv = jnp.nanmean(norms_xv, axis=0)

min_step = 10000
max_step = 100000
min_t = ts[min_step]
max_t = ts[max_step]

# Plot the differences for both vv and xx
ts_to_plot = ts[min_step:max_step]
norms_vv_to_plot = mean_norms_vv[min_step:max_step]
norms_xx_to_plot = mean_norms_xx[min_step:max_step]
norms_xv_to_plot = mean_norms_xv[min_step:max_step]
rollmean_norms_vv_to_plot = jnp.cumsum(norms_vv_to_plot) / jnp.arange(1, len(norms_vv_to_plot) + 1)
rollmean_norms_xx_to_plot = jnp.cumsum(norms_xx_to_plot) / jnp.arange(1, len(norms_xx_to_plot) + 1)
rollmean_norms_xv_to_plot = jnp.cumsum(norms_xv_to_plot) / jnp.arange(1, len(norms_xv_to_plot) + 1)

# ratio of avv to axv
ratio_vv2xx = norms_vv_to_plot / norms_xx_to_plot
ratio_vv2xv = norms_vv_to_plot / norms_xv_to_plot
rollmean_ratio_vv2xx = jnp.cumsum(ratio_vv2xx) / jnp.arange(1, len(ratio_vv2xx) + 1)
rollmean_ratio_vv2xv = jnp.cumsum(ratio_vv2xv) / jnp.arange(1, len(ratio_vv2xv) + 1)

#%%
# Create the mosaic layout
mosaic = """
AABBCC
DDEEFF
GGGGGG
"""
n_to_plot = 1000
burn_in =int(0.50*config.init_config.N)
plot_idxs = np.random.choice(np.arange(burn_in, config.init_config.N), n_to_plot, replace=False)
fig = plt.figure(figsize=(10, 9), dpi=600)
ax_dict = fig.subplot_mosaic(mosaic)
times_info_str = r"$t_{min} = $" + f"{min_t:.2f}, "+r"$t_{max} = $" + f"{max_t:.1f}"
fig.suptitle(
    "Critically Damped Langevin Dynamics at "+
              r"$\beta = $" + f"{config.cld_config.beta}, "+r"$\gamma = $" + f"{config.cld_config.gamma}", fontsize=16,
    y=1.02
)

# Color scheme
x_color = "#1f77b4"  # blue
p_color = "#ff7f0e"  # orange
joint_cmap = "viridis"  # better heatmap coloring for joint distributions

# Initial distribution of x
sns.histplot(init_x0s_p0s[rand_init_idxs, 0], kde=True, color=x_color, alpha=0.6, ax=ax_dict['A'])
ax_dict['A'].set_title(r"$p_0(x)$", fontsize=14)
ax_dict['A'].set_xlabel("X", fontsize=12)
ax_dict['A'].set_ylabel("Density", fontsize=12)

# Initial distribution of p
sns.histplot(init_x0s_p0s[rand_init_idxs, 1], kde=True, color=p_color, alpha=0.6, ax=ax_dict['B'])
ax_dict['B'].set_title(r"$p_0(v)$", fontsize=14)
ax_dict['B'].set_xlabel("P", fontsize=12)
ax_dict['B'].set_ylabel("Density", fontsize=12)

# Initial joint distribution of x and p
sns.kdeplot(
    x=init_x0s_p0s[rand_init_idxs, 0],
    y=init_x0s_p0s[rand_init_idxs, 1],
    cmap=joint_cmap,
    fill=True,
    cbar=True,
    ax=ax_dict['C'],
)
ax_dict['C'].set_title(r"$p_0(x,v)$", fontsize=14)
ax_dict['C'].set_xlabel("x_0", fontsize=12)
ax_dict['C'].set_ylabel("v_0", fontsize=12)

# Final distribution of x
sns.histplot(xs[0,plot_idxs], kde=True, color=x_color, alpha=0.6, ax=ax_dict['D'])
ax_dict['D'].set_title(r"$p_{T}(x)$", fontsize=14)
ax_dict['D'].set_xlabel("x_T", fontsize=12)
ax_dict['D'].set_ylabel("Density", fontsize=12)

# Final distribution of v
sns.histplot(vs[0,plot_idxs], kde=True, color=p_color, alpha=0.6, ax=ax_dict['E'])
ax_dict['E'].set_title(r"$p_{T}(v)$", fontsize=14)
ax_dict['E'].set_xlabel("v_T", fontsize=12)
ax_dict['E'].set_ylabel("Density", fontsize=12)


# Final joint distribution of x and p at the final time step
sns.kdeplot(
    x=xs[0,plot_idxs],
    y=vs[0,plot_idxs],
    cmap=joint_cmap,
    fill=True,
    cbar=True,
    ax=ax_dict['F'],
)
ax_dict['F'].set_title("Final Joint Distribution of X and P", fontsize=14)
ax_dict['F'].set_xlabel("X", fontsize=12)
ax_dict['F'].set_ylabel("P", fontsize=12)

# Convergence means plot
ax_dict['G'].plot(ts_to_plot, rollmean_norms_xv_to_plot, label=r'$\frac{1}{N}\sum_{n=0}^N\|A_{xv}(t)(\mu_{x_n}(t) - x_n(t))\|$', color=x_color, alpha=1.0)
ax_dict['G'].plot(ts_to_plot, rollmean_norms_vv_to_plot, label=r'$\frac{1}{N}\sum_{n=0}^N\|A_{vv}(t)(\mu_{v_n}(t) - v_n(t))\|$', color=p_color, alpha=1.0)
ax_dict['G'].set_title(r"Rolling Means", fontsize=14, y=1.05)

# add red line for minimum value of ratio vv/vx
ax_dict['G'].set_xlabel("Time", fontsize=12)
ax_dict['G'].set_ylabel("Value", fontsize=12)
ts_freq = 10
ax_dict['G'].set_xticks(ts_to_plot[::max_step//ts_freq])
ax_dict['G'].set_xticklabels([f"{t:.1f}" for t in ts_to_plot[::max_step//ts_freq]])
ax_dict['G'].legend(fontsize=10)

# log
ax_dict['G'].set_yscale('log')
# ax_dict['G'].set_ylim(1e-2, 1e2)
# put yticklabels on the right side
ax_dict['G'].yaxis.tick_right()
ax_dict['G'].yaxis.tick_left()

plt.tight_layout()
plt.show()
# %%

