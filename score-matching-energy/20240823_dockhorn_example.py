# %%
import diffrax
import matplotlib.pyplot as plt
from diffrax import ControlTerm, MultiTerm, ODETerm
from jax import grad, vmap
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from scipy import stats

# from jax import config
# config.update("jax_disable_jit", True)
dim = 1
beta = 1
M = 1
B = jnp.sqrt(4 * M)  # from 1
gamma = 0.5
# %%
key = jr.PRNGKey(0)
weights = 0.5 * jnp.array([1.0, 1.0])
mean = jnp.array([-1.0, 1.0])
normals = 1.0 * jr.normal(key, shape=(1000, 2))
samples = jnp.array([normals * mean[0] * weights[0], normals * mean[1] * weights[1]])
print(samples.shape)

# estimate the pdf of the samples in 1d
# %%
print(samples)

# Create single Gaussian 1D samples
key = jr.PRNGKey(0)
n_samples = 10000
mean = 0.0
std = jnp.sqrt(gamma * M)

# Draw initial conditions
n_initial_conditions = 1000

# Generate samples from a single normal distribution
samples = jr.normal(key, (n_samples,)) * std + mean

# Plot the density
plt.figure(figsize=(10, 6))
kde = stats.gaussian_kde(samples)
x_range = np.linspace(samples.min(), samples.max(), 1000)
plt.plot(x_range, kde(x_range), label="KDE")
plt.hist(samples, bins=50, density=True, alpha=0.6, label="Histogram")
plt.title("Single Gaussian 1D Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# %%
# do same for bimodal

# Create bimodal 1D samples
key = jr.PRNGKey(1)  # Using a different key to avoid overwriting previous samples
n_samples_bimodal = 10000
weights_bimodal = jnp.array([0.5, 0.5])
means_bimodal = jnp.array([-1.0, 1.0])
stds_bimodal = jnp.array([0.5, 0.5])

# Generate samples from two normal distributions
samples1_bimodal = (
    jr.normal(key, (n_samples_bimodal // 2,)) * stds_bimodal[0] + means_bimodal[0]
)
samples2_bimodal = (
    jr.normal(jr.split(key)[1], (n_samples_bimodal // 2,)) * stds_bimodal[1]
    + means_bimodal[1]
)

# Combine samples
samples_bimodal = jnp.concatenate([samples1_bimodal, samples2_bimodal])

# Plot the density for bimodal distribution
plt.figure(figsize=(10, 6))
kde_bimodal = stats.gaussian_kde(samples_bimodal)
x_range_bimodal = np.linspace(samples_bimodal.min(), samples_bimodal.max(), 1000)
plt.plot(x_range_bimodal, kde_bimodal(x_range_bimodal), label="KDE")
plt.hist(samples_bimodal, bins=50, density=True, alpha=0.6, label="Histogram")
plt.title("Bimodal 1D Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# %%

# %%

# x0 from bimodal distribution
key_x0 = jr.PRNGKey(2)
samples1_x0 = (
    jr.normal(key_x0, (n_initial_conditions // 2,)) * stds_bimodal[0] + means_bimodal[0]
)
samples2_x0 = (
    jr.normal(jr.split(key_x0)[1], (n_initial_conditions // 2,)) * stds_bimodal[1]
    + means_bimodal[1]
)
x0_samples = jnp.concatenate([samples1_x0, samples2_x0])

# p0 from regular (single Gaussian) distribution
key_p0 = jr.PRNGKey(3)
p0_samples = jr.normal(key_p0, (n_initial_conditions,)) * std + mean

# Combine x0 and p0 to form initial conditions
initial_conditions = jnp.column_stack((x0_samples, p0_samples))

# Plot the initial conditions
plt.figure(figsize=(10, 6))
plt.scatter(x0_samples, p0_samples, alpha=0.5)
plt.title("Initial Conditions (x0 vs p0)")
plt.xlabel("x0 (Bimodal)")
plt.ylabel("p0 (Gaussian)")
plt.show()
# %%
# Update y0 to use one of these initial conditions (e.g., the first one)


# %%

dim = 1
beta = 1.0
M = 1.0
B = jnp.sqrt(4 * M)  # from 1
gamma = 0.5

w_shape = (2 * dim,)
mean_scale = 1.0


def H(z):
    x, p = z[:dim], z[dim:]
    return (0.5 * (x**2 + p**2 * (1 / M))).squeeze()


grad_z_H = grad(H)


def diffusion_fn(t, state, args) -> lx.DiagonalLinearOperator:
    return jnp.array([0, jnp.sqrt(2 * B * beta)])


def drift_fn(t, state, args):
    Q = jnp.array([[0, beta], [-beta, -B * beta]])
    return Q @ grad_z_H(state)  # (2,2) @ (2,) -> (2,)


t0 = 0.0
t1 = 1.0
dt0 = 0.001

brownian_motion = diffrax.VirtualBrownianTree(
    t0, t1, dt0, w_shape, jr.PRNGKey(0), diffrax.BrownianIncrement
)

terms = MultiTerm(ODETerm(drift_fn), ControlTerm(diffusion_fn, brownian_motion))

saveat = diffrax.SaveAt(steps=True)

# Update the diffeqsolve function to use the new y0
solutions = vmap(
    lambda y0: diffrax.diffeqsolve(
        terms,
        solver=diffrax.SRA1(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        progress_meter=diffrax.TqdmProgressMeter(),
        max_steps=100000,
    ),
    in_axes=0,
)(initial_conditions[:2])


# %%
solutions.ys[0,-1,:]
# plot the solutions
# %%
solutions.ys.shape
#%%
plt.figure(figsize=(10, 6))
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

plt.subplot(1, 2, 2)
kde_p0 = stats.gaussian_kde(p0_samples)
p_range_p0 = np.linspace(p0_samples.min(), p0_samples.max(), 1000)
plt.plot(p_range_p0, kde_p0(p_range_p0), label="KDE")
plt.hist(p0_samples, bins=50, density=True, alpha=0.6, label="Histogram")
plt.title("Initial p0 Density")
plt.xlabel("p0")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

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
