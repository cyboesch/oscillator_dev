# %%
from tqdm.auto import tqdm
from collections import namedtuple
from typing import NamedTuple
from jax import jvp, vmap, jit
import jax.random as jrnd
import jax.numpy as jnp
from diffrax import ODETerm, Euler
import matplotlib.pyplot as plt
import numpy as np


from diffrax._custom_types import RealScalarLike, Y, Args

# config.update("jax_disable_jit", True)
# config.update("jax_log_compiles", True)

# %%

# Define system dimensions and parameters
state_dim = 3
unstable_subspace_dim = 1  # Number of basis vectors needed to span unstable subspace

# Define Lorenz system parameters for chaotic behavior
sigma = 10.0  # Prandtl number
rho = 28.0    # Rayleigh number
beta = 8.0 / 3.0  # Physical proportion

# Pack parameters into a named tuple for better readability
LorenzParams = namedtuple('LorenzParams', ['sigma', 'rho', 'beta'])
lorenz_params = LorenzParams(sigma, rho, beta)

# Set args to the Lorenz parameters
args = lorenz_params

# Define system dynamics.
def lorenz63(t, y, args):
    x, y, z = y
    sigma, rho, beta = args

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return jnp.array([dx, dy, dz])


@jit
def dynamics(t, y, args):
    return lorenz63(t, y, args)



# %%

seed = 42
rng = jrnd.PRNGKey(seed)


class Solution(NamedTuple):
    t: RealScalarLike
    y: jnp.ndarray
    Q: jnp.ndarray


class StepInfo(NamedTuple):
    t: RealScalarLike
    center_span: jnp.ndarray
    unstable_basis_directions: jnp.ndarray
    unstable_basis_magnitudes: jnp.ndarray


# %%
# --- Diffrax API helpers


def convert_to_ts(num_steps, step_size):
    t0 = 0
    t1 = num_steps * step_size
    return t0, t1


# --- Linear algebra helpers


def orthogonal_projection_v_onto_u(v, u):
    return jnp.dot(v, u) / jnp.dot(u, u) * u


@jit
def orth_proj_V_onto_u(V, u):
    # Project each column of V onto u.
    return vmap(
        lambda v: orthogonal_projection_v_onto_u(v, u),
        in_axes=(1),  # vectorize over columns of V_in
        out_axes=(1),  # output each result as a column of V_out
    )(V)


def test_orthogonal_projection():
    # Generate random vectors
    key = jrnd.PRNGKey(0)
    key, subkey = jrnd.split(key)
    V = jrnd.normal(subkey, (5, 3))  # 5x3 matrix
    key, subkey = jrnd.split(key)
    u = jrnd.normal(subkey, (5,))  # 3D vector

    # Perform the projection
    V_proj = orth_proj_V_onto_u(V, u)

    # Check that V_proj is parallel to u
    parallel_check = jnp.allclose(
        V_proj, jnp.outer(u, jnp.sum(V_proj.T * u, axis=1) / jnp.dot(u, u))
    )

    # Check that (V - V_proj) is orthogonal to u
    orthogonal_check = jnp.allclose(jnp.dot((V - V_proj).T, u), jnp.zeros(3), atol=1e-6)

    assert parallel_check, "Projection is parallel to u"
    assert orthogonal_check, "(V - V_proj) is orthogonal to u"


test_orthogonal_projection()

# %%

# Define dynamics of state y and vector w in tangent space of `dynamics` at y(t).
augmented_dynamics_y_w = lambda t, y, w, args: jvp(
    lambda x: dynamics(t, x, args), (y,), (w,)
)

# Define `augmented_dynamics` vectorized over w in tangent space of `dynamics` at y(t).
augmented_dynamics_y_W = vmap(  # vmap over columns of W: (y, W) -> (y_dot, W_dot),
    lambda t, y_aug, args: augmented_dynamics_y_w(t, *y_aug, args),
    in_axes=(None, (None, 1), None),
    out_axes=((None, 1)),
)


# %%

#####  ####### #     # ####### ###  #####   #####
#     # #     # ##    # #        #  #     # #     #
#       #     # # #   # #        #  #       #
#       #     # #  #  # #####    #  #  ####  #####
#       #     # #   # # #        #  #     #       #
#     # #     # #    ## #        #  #     # #     #
#####  ####### #     # #       ###  #####   #####

# %%
# Set up initial value problem whose solution is basis for unstable subspace.

# Initialize state and basis estimates
y0 = jnp.array([1.0, 1.0, 1.0])
rng, Q_rng = jrnd.split(rng)
A_rand = jrnd.normal(Q_rng, (state_dim, unstable_subspace_dim))
Q0, R0 = jnp.linalg.qr(A_rand)

# Configure the ODE solver.
num_steps = 10_000
step_size = 0.01
prev_t, t_final = convert_to_ts(num_steps, step_size)
solver = Euler()
augmented_term = ODETerm(augmented_dynamics_y_W)
solution = {"t": [prev_t], "y": [y0], "Q": [Q0]}
step_info = {
    "t": [prev_t],
    "center_span": [dynamics(0, y0, args)],
    "unstable_basis_directions": [Q0],
    "unstable_basis_magnitudes": [R0],
}


# %%
@jit
def Euler_step(t, y_aug):
    next_t = t + step_size
    next_y_aug = solver.step(
        augmented_term, t, next_t, y_aug, args, None, made_jump=False
    )[0]
    return next_t, next_y_aug


def remove_subspace(V, subspace_basis):
    V_along_subspace = orth_proj_V_onto_u(V, subspace_basis)
    return V - V_along_subspace


def estimate_center_subspace_span(t, y, args):
    return


# %%
# Solve the IVP for the unstable subspace basis.

prev_y, prev_Q = y0, Q0

# Calculate total number of steps
total_steps = int((t_final - prev_t) / step_size) + 1

# Create tqdm progress bar
progress_bar = tqdm(total=total_steps, desc="Solving IVP", unit="step")

# Wrap the while loop with tqdm

with progress_bar:
    while prev_t < t_final:
        # Take one step forward in the augmented dynamics.
        cur_t, (cur_y, cur_W) = Euler_step(prev_t, (prev_y, prev_Q))

        # Estimate and remove center subspace component from cur_W.
        center_subspace_basis =  dynamics(cur_t, cur_y, args)
        cur_W_no_center_component = cur_W - orth_proj_V_onto_u(
            cur_W, center_subspace_basis
        )

        # Separate dir and magnitude of the unstable subspace basis
        cur_Q, cur_R = jnp.linalg.qr(jnp.stack(cur_W_no_center_component))

        # Update step info.
        step_info["t"].append(cur_t)
        step_info["center_span"].append(center_subspace_basis)
        step_info["unstable_basis_directions"].append(cur_Q)
        step_info["unstable_basis_magnitudes"].append(jnp.diag(cur_R))

        # Update solution.
        solution["t"].append(cur_t)
        solution["y"].append(cur_y)
        solution["Q"].append(cur_Q)

        prev_t = min(cur_t, t_final)
        prev_y = cur_y
        prev_Q = cur_Q  # Next step's initial condition in tangent space has unit norm.
        progress_bar.update(1)


# %%

# # Print statistics of the solution
# print("Solution statistics:")
# print(f"Number of time steps: {len(solution['y'])}")
# print(f"Final time: {solution['t'][-1]:.4f}")
# print(f"shape of solution['t]: {jnp.array(solution['t']).shape}")
# print(f"Final state: {solution['y'][-1]}")
# print(f"Shape of final state: {jnp.array(solution['y']).shape}")
# print(f"Shape of final Q matrix: {solution['Q'][-1].shape}")

# # Compute and print some additional statistics
# y_array = jnp.array(solution["y"])
# print(f"Mean state: {jnp.mean(y_array, axis=0)}")
# print(f"State standard deviation: {jnp.std(y_array, axis=0)}")

# # Print statistics about the unstable subspace
Q_array = jnp.array(solution["Q"])
# print(
#     f"Mean Frobenius norm of Q matrices: {jnp.mean(jnp.linalg.norm(Q_array, axis=(1,2))):.4f}"
# )

# # Print some information about the center subspace
# center_spans = jnp.array(step_info["center_span"])
# print(
#     f"Mean magnitude of center subspace vector: {jnp.mean(jnp.linalg.norm(center_spans, axis=1)):.4f}"
# )
# %%

# %%
# # Calculate alpha values that increase over time
# num_steps = len(y_array)
# alpha_values = np.linspace(0.1, 1, num_steps)

y_array = jnp.array(solution["y"])
t_array = jnp.array(solution["t"])

# fig = plt.figure(figsize=(20, 5))
# dim_triples = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]

# for i, (dim1, dim2, dim3) in enumerate(dim_triples):
#     ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    
#     # Use alpha_values for the color array, mapping to a colormap
#     colors = plt.cm.viridis(alpha_values)
    
#     # Plot the trajectory with increasing alpha
#     for j in range(1, num_steps):
#         ax.plot(y_array[j-1:j+1, dim1], y_array[j-1:j+1, dim2], t_array[j-1:j+1], 
#                 color=colors[j], alpha=alpha_values[j], linewidth=1)
    
#     ax.set_xlabel(f"y_{dim1}")
#     ax.set_ylabel(f"y_{dim2}")
#     ax.set_zlabel("Time")
#     ax.set_title(f"y_{dim1} vs y_{dim2} vs Time")
    
#     # Create a ScalarMappable with the correct colormap for the colorbar
#     sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, label='Time progression')
#     cbar.set_ticks([0, 0.5, 1])
#     cbar.set_ticklabels(['Start', 'Middle', 'End'])

# plt.tight_layout()
# plt.show()

# # %%
# # Plot all 3 dimensions together
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(y_array[:, 0], y_array[:, 1], y_array[:, 2])

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('3D Plot of Lorenz Attractor')

# plt.tight_layout()
# plt.show()

# %%
center_spans = jnp.array(step_info["center_span"])
# Plot the unstable subspace basis vector and center tangent space direction in Q along the trajectory in X-Y, Y-Z, and Z-X planes
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# Function to plot trajectory, unstable subspace basis vectors, and center tangent space directions
def plot_trajectory_and_vectors(ax, x_idx, y_idx, x_label, y_label):
    ax.plot(y_array[:, x_idx], y_array[:, y_idx], color='blue', alpha=0.5, linewidth=0.5, label='Trajectory')
    
    plot_interval = len(y_array) // 200  # Plot vectors at regular intervals
    for i in range(0, len(y_array), plot_interval):
        y = y_array[i, [x_idx, y_idx]]
        q = Q_array[i, [x_idx, y_idx], 0]
        
        # Calculate center tangent space direction
        center_dir = center_spans[i]
        
        scale = 2.0
        q_scaled = q * scale
        center_dir_scaled = center_dir * scale / jnp.linalg.norm(center_dir)
        
        # Plot unstable subspace basis vector increased in size:
        
        
        # Plot center tangent space direction
        ax.arrow(y[0], y[1], center_dir_scaled[0], center_dir_scaled[1], 
                 color='magenta', width=0.1, head_width=0.3, alpha=1.0,
                 label='Center tangent space' if i == 0 else "")
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(f'Trajectory, Unstable Subspace, and Center Tangent Space in {x_label}-{y_label} Plane')
    ax3.legend()
plt.title("Time and unstable directions in tangent dynamics.")

# Plot for X-Y plane
plot_trajectory_and_vectors(ax1, 0, 1, 'X', 'Y')

# Plot for Y-Z plane
plot_trajectory_and_vectors(ax2, 1, 2, 'Y', 'Z')

# Plot for Z-X plane
plot_trajectory_and_vectors(ax3, 2, 0, 'Z', 'X')

plt.tight_layout()
plt.show()
# #%%
# %%
import plotly.graph_objects as go
import numpy as np

# Create a 3D scatter plot of the trajectory
trace_trajectory = go.Scatter3d(
    x=y_array[:, 0],
    y=y_array[:, 1],
    z=y_array[:, 2],
    mode='lines',
    name='Trajectory',
    line=dict(color='black', width=0.5),
    marker=dict(opacity=0.1)
)

# Create lists to store the arrow traces
traces_unstable = []
traces_center = []

# Plot vectors at regular intervals (20 points)
plot_indices = np.linspace(0, len(y_array) - 1, 400, dtype=int)

for i in plot_indices:
    y = y_array[i]
    q = Q_array[i, :, 0]
    center_dir = center_spans[i]
    
    scale = 2.0
    q_scaled = q * scale
    center_dir_scaled = center_dir * scale / np.linalg.norm(center_dir)
    
    # Unstable subspace vector
    traces_unstable.append(go.Scatter3d(
        x=[y[0], y[0] + q_scaled[0]],
        y=[y[1], y[1] + q_scaled[1]],
        z=[y[2], y[2] + q_scaled[2]],
        mode='lines',
        line=dict(color='cyan', width=6),
        showlegend=bool(i==0),
        name='Unstable subspace'
    ))
    
    # Center tangent space vector
    traces_center.append(go.Scatter3d(
        x=[y[0], y[0] + center_dir_scaled[0]],
        y=[y[1], y[1] + center_dir_scaled[1]],
        z=[y[2], y[2] + center_dir_scaled[2]],
        mode='lines',
        line=dict(color='magenta', width=6),
        showlegend=bool(i==0),
        name='Center tangent space'
    ))

# Combine all traces
data = [trace_trajectory] + traces_unstable + traces_center

# Create the layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(showticklabels=False, showgrid=False, showbackground=False, title=''),
        yaxis=dict(showticklabels=False, showgrid=False, showbackground=False, title=''),
        zaxis=dict(showticklabels=False, showgrid=False, showbackground=False, title=''),
        aspectmode='cube',
        aspectratio=dict(x=1, y=1, z=1),
    ),
    title='3D Trajectory with Unstable Subspace and Center Tangent Space Vectors',
    showlegend=True
)

# Create the figure and show it
fig = go.Figure(data=data, layout=layout)
fig.show()