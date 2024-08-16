# %%
from typing import Callable

# jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax import Array, random, grad, vmap
from jax import random as jrnd

from jax import random
import diffrax
from diffrax import ControlTerm, MultiTerm, ODETerm
import lineax as lx


from thermo_oscillator_network_fns import (
    energy_network,
    setup_integration,
    integrate_dofs,
)

from scipy.stats import multivariate_normal

import os

import matplotlib.pyplot as plt


# %%
def sample_from_superposition(mu1, sigma1, mu2, sigma2, p1, n_samples, key):
    """
    Samples from a superposition of two Gaussian distributions in 2D using JAX.
    """
    key1, key2, key3 = random.split(key, 3)

    # Sample from both Gaussians
    samples1 = random.multivariate_normal(key1, mu1, sigma1, shape=(n_samples,))
    samples2 = random.multivariate_normal(key2, mu2, sigma2, shape=(n_samples,))

    # Generate random numbers to decide which sample to keep
    choices = random.uniform(key3, shape=(n_samples,)) < p1

    # Select samples based on the choices
    samples = jnp.where(choices[:, None], samples1, samples2)

    return samples


def gaussian_pdf(x, y, mu, sigma):
    """Compute the PDF of a 2D Gaussian distribution."""
    pos = jnp.dstack((x, y))
    return multivariate_normal.pdf(pos, mean=mu, cov=sigma)


# Parameters for the first Gaussian
mu1 = jnp.array([1, 1])
sigma1 = jnp.array([[0.01, 0.01], [0.01, 0.4]])

# Parameters for the second Gaussian
mu2 = jnp.array([-1, 0])
sigma2 = jnp.array([[0.1, -0.03], [-0.03, 0.1]])

# Probability of sampling from the first Gaussian
p1 = 0.5

# Number of samples to generate
n_samples = 1000

# JAX random key
key = random.PRNGKey(0)

# Generate samples
samples = sample_from_superposition(mu1, sigma1, mu2, sigma2, p1, n_samples, key)

# Create a grid for the contour plot
x_lim = 2
y_lim = 2
x, y = jnp.mgrid[-x_lim:x_lim:0.01, -y_lim:y_lim:0.01]

# Compute the PDF for both Gaussians
z1 = gaussian_pdf(x, y, mu1, sigma1)
z2 = gaussian_pdf(x, y, mu2, sigma2)

# Combine the PDFs according to the mixture weights
z = p1 * z1 + (1 - p1) * z2

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the contour of the combined distribution
plt.contourf(x, y, z, levels=20, cmap="viridis", alpha=0.7)
plt.colorbar(label="Probability Density")

# Plot the samples
plt.scatter(samples[:, 0], samples[:, 1], color="red", alpha=0.5, s=10, label="Samples")

plt.title("Superposition of Two Gaussians in 2D with Samples")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

# %%
# # # Duffing params
# k_lin= jnp.array([1., 1.,1.,1.])
# k_duff = jnp.array([0., 0.,0.,0.])
# c_lin = jnp.array([.0,.0,.0])
# c_optomech = jnp.array([0., 0.,0.])
# connectivity = jnp.array([[0, 1],[1, 2],[2, 3]])

# marginalized_dofs = jnp.array([1,2])
# non_marginalized_dofs = jnp.array([0,3])

# %%
# # # Duffing params
k_b = 1.0
T = 100.0

m = jnp.array([1.0, 1.0, 1.0])
eig_freq = jnp.array([1.0, 1.0, 1.0])
k_lin = eig_freq**2 * m
k_duff = jnp.array([1.0, 1.0, 1.0])
c_lin = jnp.zeros(2)
c_optomech = jnp.array([1.0, 1.0])
connectivity = jnp.array([[0, 1], [1, 2]])

marginalized_dofs = jnp.array([1])
non_marginalized_dofs = jnp.array([0, 2])

k_lin = jnp.array([1335.6998, -5119.987, 4098.15])
k_duff = jnp.array([2343.9277, 907.47296, -418.0096])
c_lin = jnp.array([3501.9744, 533.9826])
c_optomech = jnp.array([492.3586, -859.08386])


######     #    ######     #    #     #  #####
#     #   # #   #     #   # #   ##   ## #     #
#     #  #   #  #     #  #   #  # # # # #
######  #     # ######  #     # #  #  #  #####
#       ####### #   #   ####### #     #       #
#       #     # #    #  #     # #     # #     #
#       #     # #     # #     # #     #  #####


# %%
integration_limits = (-2, 2)
num_integration_points = 10
if marginalized_dofs is not None:
    get_integrands = setup_integration(
        marginalized_dofs,
        integration_limits=integration_limits,
        num_integration_points=10,
    )
    sign_hess = 1
else:
    get_integrands = setup_integration(
        non_marginalized_dofs, integration_limits=(-2, 2), num_integration_points=10
    )
    sign_hess = -1


# %%
if marginalized_dofs is not None:
    energy_network_marg = integrate_dofs(
        get_integrands, energy_network, marginalized_dofs, non_marginalized_dofs
    )
else:
    energy_network_marg = energy_network

# %%
# Define the grid for plotting using integration limits
x1_values = jnp.linspace(integration_limits[0], integration_limits[1], 100)
x2_values = jnp.linspace(integration_limits[0], integration_limits[1], 100)
X1, X2 = jnp.meshgrid(x1_values, x2_values)

# Calculate energy values for original parameters
energy_values_original = vmap(
    lambda x1, x2: -1
    * sign_hess
    * energy_network_marg(
        jnp.array([x1, x2]), k_lin, k_duff, c_lin, c_optomech, connectivity, k_b, T
    )
)(X1.ravel(), X2.ravel())
energy_values_original = energy_values_original.reshape(X1.shape)

# Calculate exp(-energy) values
exp_neg_energy = jnp.exp(-energy_values_original)


# %%
# Create the plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot original energy
contour1 = ax1.contourf(X1, X2, energy_values_original, levels=50, cmap="viridis")
ax1.set_title("Network Energy (Original Parameters)")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
cbar1 = fig.colorbar(contour1, ax=ax1, label="Energy")

# Add contour lines for better visibility of energy levels
ax1.contour(
    X1, X2, energy_values_original, levels=20, colors="k", alpha=0.3, linewidths=0.5
)

# Plot exp(-energy)
contour2 = ax2.contourf(X1, X2, exp_neg_energy, levels=50, cmap="viridis")
ax2.set_title("exp(-Energy) (Original Parameters)")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
cbar2 = fig.colorbar(contour2, ax=ax2, label="exp(-Energy)")

# Add contour lines for better visibility of exp(-energy) levels
ax2.contour(X1, X2, exp_neg_energy, levels=20, colors="k", alpha=0.3, linewidths=0.5)

plt.tight_layout()
plt.show()

# Print original parameter values for reference
print(
    f"Original parameters: k_lin={k_lin}, k_duff={k_duff}, c_lin={c_lin}, c_optomech={c_optomech}"
)
print(f"Integration limits: {integration_limits}")


# %%
# Calculate energy values for original k
energy_values_original = vmap(
    lambda x1, x2: -sign_hess
    * energy_network_marg(
        jnp.array([x1, x2]).T, k_lin, k_duff, c_lin, c_optomech, connectivity, k_b, T
    )
)(X1.ravel(), X2.ravel())
energy_values_original = energy_values_original.reshape(X1.shape)

# Calculate energy values for optimized k
energy_values_optimized = vmap(
    lambda x1, x2: -sign_hess
    * energy_network_marg(
        jnp.array([x1, x2]).T, k_lin, k_duff, c_lin, c_optomech, connectivity, k_b, T
    )
)(X1.ravel(), X2.ravel())
energy_values_optimized = energy_values_optimized.reshape(X1.shape)

prob_from_energy_vals_fn = vmap(
    lambda x: jnp.exp(
        sign_hess
        * energy_network_marg(x, k_lin, k_duff, c_lin, c_optomech, connectivity, k_b, T)
    )
)  # (X1.

X = jnp.stack([X1.ravel(), X2.ravel()], axis=-1)

# Apply the function to the combined array
prob_from_energy_vals = prob_from_energy_vals_fn(X)

# Reshape the result back to the original shape
prob_from_energy_vals = prob_from_energy_vals.reshape(X1.shape)


inegrator_norm = setup_integration(
    non_marginalized_dofs,
    integration_limits=integration_limits,
    num_integration_points=num_integration_points,
)

vol_elt = (
    (integration_limits[1] - integration_limits[0]) / num_integration_points
) ** 2
norm_const_opt = jnp.sum(jnp.exp(-prob_from_energy_vals)) * vol_elt

prob_from_energy_normalized = prob_from_energy_vals / norm_const_opt


# %%
# ... (previous code remains the same)

# Create 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))

# Increase font size for all text elements
plt.rcParams.update({"font.size": 14})

# Plot original k energy
contour1 = ax1.contourf(X1, X2, energy_values_original, levels=50, cmap="viridis")
cbar1 = fig.colorbar(contour1, ax=ax1)
cbar1.set_label("Energy", fontsize=14)
ax1.set_title("Network Deformation Energy (Initial)", fontsize=16)
ax1.set_xlabel("x1", fontsize=14)
ax1.set_ylabel("x2", fontsize=14)
ax1.grid(True)

# Plot optimized k energy
contour2 = ax2.contourf(X1, X2, energy_values_optimized, levels=50, cmap="viridis")
cbar2 = fig.colorbar(contour2, ax=ax2)
cbar2.set_label("Energy", fontsize=14)
ax2.set_title("Network Deformation Energy (Optimized)", fontsize=16)
ax2.set_xlabel("x1", fontsize=14)
ax2.set_ylabel("x2", fontsize=14)
ax2.grid(True)

# Plot target distribution
x, y = jnp.mgrid[-x_lim:x_lim:0.01, -y_lim:y_lim:0.01]
z1 = gaussian_pdf(x, y, mu1, sigma1)
z2 = gaussian_pdf(x, y, mu2, sigma2)
z = p1 * z1 + (1 - p1) * z2

contour3 = ax3.contourf(x, y, z, levels=20, cmap="viridis", alpha=0.7)
cbar3 = fig.colorbar(contour3, ax=ax3)
cbar3.set_label("Probability Density", fontsize=14)
ax3.scatter(samples[:, 0], samples[:, 1], color="red", alpha=0.5, s=10, label="Samples")
ax3.set_title("Target Distribution: Superposition of Two Gaussians", fontsize=16)
ax3.set_xlabel("X", fontsize=14)
ax3.set_ylabel("Y", fontsize=14)
ax3.legend(fontsize=12)
ax3.axis("equal")

# Plot probability from optimized energy
contour4 = ax4.contourf(
    X1, X2, prob_from_energy_normalized, levels=20, cmap="viridis", alpha=0.7
)
cbar4 = fig.colorbar(contour4, ax=ax4)
cbar4.set_label("Probability Density", fontsize=14)
ax4.set_title("Probability from Optimized Energy", fontsize=16)
ax4.set_xlabel("X", fontsize=14)
ax4.set_ylabel("Y", fontsize=14)
ax4.axis("equal")
# ax4.set_xlim(x_lim, -x_lim)
# ax4.set_ylim(y_lim, -y_lim)

# Increase font size for colorbar labels
for cbar in [cbar1, cbar2, cbar3, cbar4]:
    cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

# filename = convergence_filename.replace("_convergence.png", ".png")

# # Save the figure to the results folder
# plt.savefig(f'results/{folder_filename}/{filename}', dpi=300, bbox_inches='tight')

plt.show()


# Print k values for reference
print(
    f"Optimized k_lin: {k_lin}, k_duff: {k_duff}, c_lin: {c_lin}, c_optomech: {c_optomech}"
)

# %%
N = k_lin.shape[0]
y0 = jnp.zeros(2 * N)
w_shape = (2 * N,)  # state is (x1, x2, p1, p2)
N = k_lin.shape[0]
diag_B: Array = k_b * T * jnp.ones(N)
sys_params: dict = {
  "diag_B": diag_B, 
  "diag_M_inv": jnp.ones((N,))
  }
state_dim = k_lin.shape[0] * 2

params = (k_lin, k_duff)
consts = (c_lin, c_optomech, connectivity)

# --- Potential function (positive energy)
U_fn: Callable[[jnp.ndarray, tuple[float, float]], jnp.ndarray] = (
    lambda x, params: energy_network(x, *params, *consts)
)
grad_x_U_fn = grad(U_fn)

# --- drift and difusion
def diffusion_fn(t, state, args) -> lx.DiagonalLinearOperator:
    diag = jnp.concatenate([jnp.zeros(N), args[1]])
    return lx.DiagonalLinearOperator(diag)

def drift_fn(_, state, args):
    x, p = state.at[:N].get(), state.at[N:].get()  #(2*N, )
    diag_M_inv, diag_B, params = args  #(N,) (2*N,), (2*N,)
    xdot = diag_M_inv * p
    pdot = -grad_x_U_fn(x, params) - diag_B * xdot
    return jnp.concatenate([xdot, pdot])


diag_B = sys_params.get("diag_B", 0.5 * jnp.ones((N,)))
diag_M_inv = sys_params.get("diag_M_inv", jnp.ones((N,)))

sample_params = {
    "t0": 0.0,
    "t1": 100.0,
    "num_samples": 1000,
}
t0 = sample_params.get("t0", 0.0)
t1 = sample_params.get("t1", 100.0)
num_samples = sample_params.get("num_samples", 1000)

brownian_motion = diffrax.VirtualBrownianTree(
    t0, t1, 0.05, w_shape, jrnd.PRNGKey(0), diffrax.SpaceTimeLevyArea
)

terms = MultiTerm(ODETerm(drift_fn), ControlTerm(diffusion_fn, brownian_motion))
ts = jnp.linspace(t0, t1, num_samples)
saveat = diffrax.SaveAt(ts=ts)

solution = diffrax.diffeqsolve(
    terms,
    solver=diffrax.SRA1(),
    t0=t0,
    t1=t1,
    dt0=0.1,
    y0=y0,
    args=(diag_M_inv, diag_B, params),
    saveat=saveat,
    progress_meter=diffrax.TqdmProgressMeter(),
)

solution.ys[:, :N]  # Return only position coordinates

# %%
