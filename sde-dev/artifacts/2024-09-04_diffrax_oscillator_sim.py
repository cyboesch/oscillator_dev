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
                N=100000,
                rng=jr.PRNGKey(init_seed),
                step_size=0.01
            ),
            cld_config=CLDConfig(
                state_dim=1,
                beta=0.0005,
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


# # %%
# # Create the plot
# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(3, 3)

# # Main scatter plot
# ax_main = fig.add_subplot(gs[1:, :2])
# ax_main.scatter(init_x0s_p0s[:, 0], init_x0s_p0s[:, 1], alpha=0.5, s=1)
# ax_main.set_xlabel("x")
# ax_main.set_ylabel("p")

# # Top marginal plot (for x)
# ax_top = fig.add_subplot(gs[0, :2], sharex=ax_main)
# ax_top.hist(init_x0s_p0s[:, 0], bins=50, density=True, alpha=0.6)
# ax_top.set_ylabel("Density")
# ax_top.set_title("MoG Distribution (x)")
# ax_top.tick_params(labelbottom=False)

# # Right marginal plot (for p)
# ax_right = fig.add_subplot(gs[1:, 2], sharey=ax_main)
# ax_right.hist(
#     init_x0s_p0s[:, 1], bins=50, density=True, alpha=0.6, orientation="horizontal"
# )

# ax_right.set_xlabel("Density")
# ax_right.set_title("Normal Distribution (p)", rotation=270, x=1.1, y=0.5)
# ax_right.tick_params(labelleft=False)

# # 2D contour plot
# ax_contour = fig.add_subplot(gs[1:, :2], sharex=ax_main, sharey=ax_main)
# x = init_x0s_p0s[:, 0]
# y = init_x0s_p0s[:, 1]
# xmin, xmax = x.min(), x.max()
# ymin, ymax = y.min(), y.max()
# xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([xx.ravel(), yy.ravel()])
# values = np.vstack([x, y])
# kernel = stats.gaussian_kde(values)
# f = np.reshape(kernel(positions).T, xx.shape)
# ax_contour.contourf(xx, yy, f, cmap="jet", alpha=0.5)
# ax_contour.set_xlabel("x")
# ax_contour.set_ylabel("p")

# # Adjust layout and display
# plt.tight_layout()
# plt.show()

# # %%


# def H(z):
#     x, p = z
#     return 0.5 * (x**2 + p**2 * (1 / config.cld_config.M))


# grad_z_H = jit(grad(H))

# # Plot H and its gradient
# plt.figure(figsize=(12, 4))
# x = jnp.linspace(xmin, xmax, 1000)
# y = jnp.linspace(ymin, ymax, 1000)
# xx, yy = jnp.meshgrid(x, y)
# xxyy = jnp.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
# zz = vmap(H)(xxyy)
# zz = zz.reshape(xx.shape)
# plt.subplot(1, 3, 1)
# plt.contourf(xx, yy, zz, alpha=1.0)
# plt.colorbar()
# plt.xlabel("x")
# plt.ylabel("p")
# plt.title("$H(z)$")

# zz = vmap(grad_z_H)(xxyy)
# zz_x = zz[:, 0].reshape(xx.shape)
# zz_p = zz[:, 1].reshape(xx.shape)
# plt.subplot(1, 3, 2)
# plt.contourf(xx, yy, zz_x, alpha=1.0)
# plt.colorbar()
# plt.title("$\\nabla_x H(z)$")
# plt.xlabel("x")
# plt.ylabel("p")

# plt.subplot(1, 3, 3)
# plt.contourf(xx, yy, zz_p, alpha=1.0)
# plt.colorbar()
# plt.title("$\\nabla_p H(z)$")
# plt.xlabel("x")
# plt.ylabel("p")

# plt.tight_layout()
# plt.show()

# del xxyy, xx, yy, x, y, zz, zz_x, zz_p


# %%
# def diffusion(t, state, args):
#     return lx.DiagonalLinearOperator(
#         jnp.array([0, jnp.sqrt(2 * config.cld_config.Gamma * config.cld_config.beta)])
#     )


# def drift(t, state, args):
#     Q = jnp.array(
#         [
#             [0.0, config.cld_config.beta],
#             [
#                 -config.cld_config.beta,
#                 -config.cld_config.Gamma * config.cld_config.beta * (1 / config.cld_config.M),
#             ],
#         ]
#     )
#     return 1.0 * Q @ grad_z_H(state)  # (2,2) @ (2,) -> (2,)@


t0 = 0.0
t1 = 1.0
dt0 = 0.001

rng = jrnd.PRNGKey(4)
bm = UnsafeBrownianPath(shape=(2 * cld.state_dim,), key=rng)

sde_terms = cld.get_terms(bm)

# Set up simulation parameters
num_steps = config.init_config.N 
step_size = config.init_config.step_size
t_final = num_steps * step_size

# # Define SDE term
y0 = jnp.zeros(2 * cld.state_dim)

# Set up solver
solver = ItoMilstein()
solver_state = solver.init(sde_terms, t0=0.0, t1=t_final, y0=y0, args=None)

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

# Initialize trajectory array
trajectory = jnp.zeros((num_steps, 2 * config.cld_config.state_dim))

# Run the simulation
init_carry = (0.0, y0, trajectory, 0)
final_t, final_y, full_trajectory, _ = simulation_loop(init_carry)

# Generate time array
ts = jnp.linspace(0, final_t, num_steps)

# random subsample of 10000 for plotting
rng, subkey = jax.random.split(rng)
idxs = jax.random.choice(subkey, jnp.arange(num_steps), (10000,), replace=False)
ys_to_plot = full_trajectory[idxs]
ts_to_plot = [idxs]

#%%
# A_vv*(mu_v(t)-v(t))
# Assuming HSM
HSM_cov = {
    "Sigma_0_xx": 0.0,
    "Sigma_0_vv": config.cld_config.gamma * config.cld_config.M,
}
HSM_mean = {
    "x_0": jnp.array([y0[0]]),
    "v_0": jnp.array([0.0]),
}

A_vvs = vmap(lambda t: cld.Lambda_vv_t(t, **HSM_cov))(ts)
mu_vs = vmap(lambda t: cld.mean(t, **HSM_mean))(ts)[:, 1]
vs = ys_to_plot[:, 1]
norms_vv = vmap(lambda t: jnp.abs(A_vvs.at[t].get() * (mu_vs.at[t].get() - vs.at[t].get())))(jnp.arange(len(ts)))

A_xxs = vmap(lambda t: cld.Lambda_xx_t(t, **HSM_cov))(ts)
mu_xs = vmap(lambda t: cld.mean(t, **HSM_mean))(ts)[:, 0]
xs = ys_to_plot[:, 0]
norms_xx = vmap(lambda t: jnp.abs(A_xxs.at[t].get() * (mu_xs.at[t].get() - xs.at[t].get())))(jnp.arange(len(ts)))

A_xvs = vmap(lambda t: cld.Lambda_xv_t(t, **HSM_cov))(ts)
mu_xv = vmap(lambda t: cld.mean(t, **HSM_mean))(ts)[:, 1]
xv = ys_to_plot[:, 1]
norms_xv = vmap(lambda t: jnp.abs(A_xvs.at[t].get() * (mu_xv.at[t].get() - xv.at[t].get())))(jnp.arange(len(ts)))

#%%
# compute rolling mean

min_step = 0
max_step = num_steps
# Plot the differences for both vv and xx
ts_to_plot = ts[min_step:max_step]
norms_vv_to_plot = norms_vv[min_step:max_step]
norms_xx_to_plot = norms_xx[min_step:max_step]
norms_xv_to_plot = norms_xv[min_step:max_step]
rollmean_vv = jnp.cumsum(norms_vv_to_plot) / jnp.arange(1, len(norms_vv_to_plot) + 1)
rollmean_xx = jnp.cumsum(norms_xx_to_plot) / jnp.arange(1, len(norms_xx_to_plot) + 1)
#%%
#The ratio plot (option 1) or the log-scale ratio plot (option 2) are often good starting points, as they directly show the relative magnitude of the two quantities.

# ratio of avv to axv
ratio_vv2xx = norms_vv_to_plot / norms_xx_to_plot
ratio_vv2xv = norms_vv_to_plot / norms_xv_to_plot
rollmean_ratio_vv2xx = jnp.cumsum(ratio_vv2xx) / jnp.arange(1, len(ratio_vv2xx) + 1)
rollmean_ratio_vv2xv = jnp.cumsum(ratio_vv2xv) / jnp.arange(1, len(ratio_vv2xv) + 1)

# Create the mosaic layout
mosaic = """
AABBCC
DDEEFF
GGGGGG
"""

fig = plt.figure(figsize=(10, 9), dpi=600)
ax_dict = fig.subplot_mosaic(mosaic)

fig.suptitle(
    "Stochastic Oscillator System: Initial vs Final Distributions and Convergence", fontsize=16,
    y=1.02
)

# Color scheme
x_color = "#1f77b4"  # blue
p_color = "#ff7f0e"  # orange
joint_cmap = "viridis"  # better heatmap coloring for joint distributions

# Initial distribution of x
sns.histplot(init_x0s_p0s[idxs, 0], kde=True, color=x_color, alpha=0.6, ax=ax_dict['A'])
ax_dict['A'].set_title("Initial Distribution of X", fontsize=14)
ax_dict['A'].set_xlabel("X", fontsize=12)
ax_dict['A'].set_ylabel("Density", fontsize=12)

# Initial distribution of p
sns.histplot(init_x0s_p0s[idxs, 1], kde=True, color=p_color, alpha=0.6, ax=ax_dict['B'])
ax_dict['B'].set_title("Initial Distribution of P", fontsize=14)
ax_dict['B'].set_xlabel("P", fontsize=12)
ax_dict['B'].set_ylabel("Density", fontsize=12)

# Initial joint distribution of x and p
sns.kdeplot(
    x=init_x0s_p0s[idxs, 0],
    y=init_x0s_p0s[idxs, 1],
    cmap=joint_cmap,
    fill=True,
    cbar=True,
    ax=ax_dict['C'],
)
ax_dict['C'].set_title("Initial Joint Distribution of X and P", fontsize=14)
ax_dict['C'].set_xlabel("X", fontsize=12)
ax_dict['C'].set_ylabel("P", fontsize=12)

# Final distribution of x
sns.histplot(ys_to_plot[-1000:, 0], kde=True, color=x_color, alpha=0.6, ax=ax_dict['D'])
ax_dict['D'].set_title("Final Distribution of X", fontsize=14)
ax_dict['D'].set_xlabel("X", fontsize=12)
ax_dict['D'].set_ylabel("Density", fontsize=12)

# Final distribution of p
sns.histplot(ys_to_plot[-1000:, 1], kde=True, color=p_color, alpha=0.6, ax=ax_dict['E'])
ax_dict['E'].set_title("Final Distribution of P", fontsize=14)
ax_dict['E'].set_xlabel("P", fontsize=12)
ax_dict['E'].set_ylabel("Density", fontsize=12)

# Final joint distribution of x and p
sns.kdeplot(
    x=ys_to_plot[-1000:, 0],
    y=ys_to_plot[-1000:, 1],
    cmap=joint_cmap,
    fill=True,
    cbar=True,
    ax=ax_dict['F'],
)
ax_dict['F'].set_title("Final Joint Distribution of X and P", fontsize=14)
ax_dict['F'].set_xlabel("X", fontsize=12)
ax_dict['F'].set_ylabel("P", fontsize=12)

# Convergence means plot
ax_dict['G'].plot(ts_to_plot, rollmean_ratio_vv2xx, label=r"$r_{vv/xx}$", color=p_color, alpha=1.0)
ax_dict['G'].plot(ts_to_plot, rollmean_ratio_vv2xv, label=r"$r_{vv/xv}$", color=x_color, alpha=1.0)
# ax_dict['G'].plot(ts_to_plot, norms_vv_to_plot, label=r"$\|A_{vv}(t)(\mu_v(t) - v(t))\|$", color=p_color, alpha=0.3
# ax_dict['G'].plot(ts_to_plot, norms_xx_to_plot, label=r"$\|A_{xx}(t)(\mu_x(t) - x(t))\|$", color=x_color, alpha=0.3)
# ax_dict['G'].plot(ts_to_plot, rollmean_vv, '-.', label="Rolling Mean vv", color=p_color, linewidth=2, zorder=10)
# ax_dict['G'].plot(ts_to_plot, rollmean_xx, '-.', label="Rolling Mean xx", color=x_color, linewidth=2, zorder=10)
ax_dict['G'].set_title(r"$t_0 = $" + f"{t0}, $t_1 = $" + f"{final_t:.2f}, " + 
              r"$\beta = $" + f"{config.cld_config.beta}, "+r"$\gamma = $" + f"{config.cld_config.gamma}" + 
              "\n" + r"$r_{\frac{vv}{xx}} = $" + r"$\frac{\|A_{vv}(t)(\mu_{v}(t) - v(t))\|}{\|A_{xx}(t)(\mu_{x}(t) - x(t))\|}$" + " and " + r"$r_{\frac{vv}{xv}} = $" + r"$\frac{\|A_{vv}(t)(\mu_{v}(t) - v(t))\|}{\|A_{xv}(t)(\mu_{x}(t) - x(t))\|}$", fontsize=14, y=1.05)
ax_dict['G'].set_xlabel("Time", fontsize=12)
ax_dict['G'].set_ylabel("Value", fontsize=12)
ax_dict['G'].set_xticklabels([f"{t:.2f}" for t in ts_to_plot[::max_step//10]])
ax_dict['G'].set_xticks(ts_to_plot[::max_step//10])
ax_dict['G'].legend(fontsize=10)

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Final time: {ts[-1]:.2f}")
print(f"Initial mean state: {jnp.mean(init_x0s_p0s, axis=0)}")
print(f"Initial state standard deviation: {jnp.std(init_x0s_p0s, axis=0)}")
print(f"Final mean state: {jnp.mean(ys_to_plot[-1000:], axis=0)}")
print(f"Final state standard deviation: {jnp.std(ys_to_plot[-1000:], axis=0)}")

# %%
