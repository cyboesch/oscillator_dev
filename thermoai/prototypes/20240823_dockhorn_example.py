# %%
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


init_seed = 0

init_config = {
    "N": 10000,
    "rng": jr.PRNGKey(init_seed),
}

sde_config = {
    "beta": 1.0,
    "M": 1.0,
    "Gamma": 1.0, #jnp.sqrt(4 * 1),  # -- from M
    "gamma": 1.0,
    "state_dim": 1,
}


MoG_1D_params = {
    "p1": 0.5,  # p2 = 1 - p1
    "mu1": -1.0,
    "mu2": 1.0,
    "stddev1": 0.5,
    "stddev2": 0.5,
}


normal_params = {
    "p1": 1.0,  # p2 = 1 - p1
    "mu1": 0.0,
    "mu2": 0.0,  # Not used
    "stddev1": jnp.sqrt(sde_config["gamma"] * sde_config["M"]),
    "stddev2": 0.0,  # Not used
}


# Sample from the MoG and the normal distribution.
def sample_x0_p0(key):
    x_key, p_key = jr.split(key, 2)
    x = sample_1d_mog(**MoG_1D_params, key=x_key)
    p = sample_1d_mog(**normal_params, key=p_key)
    return jnp.array([x, p])  # (2,)


init_x0s_p0s = jax.device_put(
    vmap(sample_x0_p0)(jr.split(init_config["rng"], init_config["N"])), cpu_devices[0]
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
    return 0.5 * (x**2 + p**2 * (1 / sde_config["M"]))


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
def diffusion_fn(t, state, args) -> lx.DiagonalLinearOperator:
    diff = lx.DiagonalLinearOperator(
        jnp.array(
            [0, jnp.sqrt(2 * sde_config["Gamma"] * sde_config["beta"])]
        )  # (2*state_dim, )
    )
    return jnp.array([0, 0])


def drift_fn(t, state, args):
    Q = jnp.array(
        [
            [0.0, sde_config["beta"]],
            [
                -sde_config["beta"],
                -sde_config["Gamma"] * sde_config["beta"] * (1 / sde_config["M"]),
            ],
        ]
    )
    # return 1.0 * Q @ grad_z_H(state)  # (2,2) @ (2,) -> (2,)@
    return jnp.array([0, 0])


t0 = 0.0
t1 = 1.0
dt0 = 0.001

w_shape = (2 * sde_config["state_dim"],)
"""     t0: RealScalarLike,
        t1: RealScalarLike,
        tol: RealScalarLike,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
        ] = BrownianIncrement,
        _spline: _Spline = "sqrt","""
        
brownian_motion = diffrax.VirtualBrownianTree(
    t0=t0,
    t1=t1,
    tol=2**-14,
    shape=w_shape,
    key=jr.PRNGKey(4),
    levy_area=diffrax.SpaceTimeLevyArea,
)


ShARK = diffrax.SRA1()

# %%
ShARK.term_structure

terms = MultiTerm(ODETerm(drift_fn), ControlTerm(diffusion_fn, brownian_motion))

saveat = diffrax.SaveAt(steps=True)

# Update the diffeqsolve function to use the new y0
y0 = init_x0s_p0s[0]

solutions = diffrax.diffeqsolve(
    terms,
    solver=ShARK,
    t0=t0,
    t1=t1,
    y0=np.zeros(2),
    saveat=saveat,
    progress_meter=diffrax.TqdmProgressMeter(),
    max_steps=1_000,
)

#%%
xs = solutions.ys[:, 0]  # (n_steps,)
ps = solutions.ys[:, 1]  # (n_steps,)
print(ps)
up_to_idx = xs.shape[0]
print(jnp.nanmean(xs[:up_to_idx]))
print(jnp.nanmean(ps[:up_to_idx]))
# %%

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(3, 3)

# Main scatter plot
ax_main = fig.add_subplot(gs[1:, :2])
ax_main.scatter(xs[:up_to_idx], ps[:up_to_idx], alpha=0.5, s=1)
ax_main.set_xlabel("x")
ax_main.set_ylabel("p")

# Top marginal plot (for x)
ax_top = fig.add_subplot(gs[0, :2], sharex=ax_main)
ax_top.hist(xs[:up_to_idx], bins=50, density=True, alpha=0.6)
ax_top.set_ylabel("Density")
ax_top.set_title("MoG Distribution (x)")
ax_top.tick_params(labelbottom=False)

# Right marginal plot (for p)
ax_right = fig.add_subplot(gs[1:, 2], sharey=ax_main)
ax_right.hist(
    ps[:up_to_idx], bins=50, density=True, alpha=0.6, orientation="horizontal"
)
ax_right.set_xlabel("Density")
ax_right.set_title("Normal Distribution (p)", rotation=270, x=1.1, y=0.5)
ax_right.tick_params(labelleft=False)
# %%
plt.subplot(1, 2, 2)
plt.hist(ps[:1000], bins=50, density=True, alpha=0.6, label="Histogram")
plt.title("Initial p0 Density")
plt.xlabel("p0")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
# 2D contour plot
ax_contour = fig.add_subplot(gs[1:, :2], sharex=ax_main, sharey=ax_main)
x = xs
y = ps
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
plt.figure(figsize=(8, 8))
plt.plot(solutions.ys[0, :, 0], solutions.ys[0, :, 1])
plt.title("Solutions")
plt.xlabel("t")
plt.ylabel("x")
plt.show()
# %%
plt.figure(figsize=(10, 6))
xs_sol = solutions.ys[:, 0, 0]
ps_sol = solutions.ys[:, 0, 1]

plt.scatter(xs_sol, ps_sol)
plt.title("Solutions")
plt.xlabel("x")
plt.ylabel("p")
plt.show()
# Plot the density of initial conditions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
kde_x0 = stats.gaussian_kde(x0_samples)
x_range_x0 = np.linspace(x0_samples.min(), x0_samples.max(), 1000)
plt.plot(x_range_x0, kde_x0(x_range_x0), label="KDE")
plt.hist(x0_samples, bins=50, density=True, alpha=0.6, label="Histogram")
plt.title("Initial x0 Density")
plt.xlabel("x0")
plt.ylabel("Density")
plt.legend()


# Plot the density of solutions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x_solutions = solutions.ys[:, 0, 0]
kde_x_sol = stats.gaussian_kde(x_solutions)
x_range_sol = np.linspace(x_solutions.min(), x_solutions.max(), 1000)
plt.plot(x_range_sol, kde_x_sol(x_range_sol), label="KDE")
plt.hist(x_solutions, bins=100, density=True, alpha=0.6, label="Histogram")
plt.title("Final x Density")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()

plt.subplot(1, 2, 2)
p_solutions = solutions.ys[:, 0, 1]
kde_p_sol = stats.gaussian_kde(p_solutions)
p_range_sol = np.linspace(p_solutions.min(), p_solutions.max(), 1000)
plt.plot(p_range_sol, kde_p_sol(p_range_sol), label="KDE")
plt.hist(p_solutions, bins=100, density=True, alpha=0.6, label="Histogram")
plt.title("Final p Density")
plt.xlabel("p")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()

# Calculate statistics of the solutions array
x_solutions = solutions.ys[:, 0, 0]
p_solutions = solutions.ys[:, 0, 1]


# Function to calculate mode (most frequent value)
def calculate_mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]


# Statistics for x
x_mean = np.mean(x_solutions)
x_mode = calculate_mode(x_solutions)
x_max = np.max(x_solutions)
x_min = np.min(x_solutions)

# Statistics for p
p_mean = np.mean(p_solutions)
p_mode = calculate_mode(p_solutions)
p_max = np.max(p_solutions)
p_min = np.min(p_solutions)

print("Statistics for x:")
print(f"Mean: {x_mean:.4f}")
print(f"Mode: {x_mode:.4f}")
print(f"Max: {x_max:.4f}")
print(f"Min: {x_min:.4f}")

print("\nStatistics for p:")
print(f"Mean: {p_mean:.4f}")
print(f"Mode: {p_mode:.4f}")
print(f"Max: {p_max:.4f}")
print(f"Min: {p_min:.4f}")

solutions.ys
# %%
