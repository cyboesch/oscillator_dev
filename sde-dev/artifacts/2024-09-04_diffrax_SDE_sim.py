#%%
import jax.numpy as jnp
import jax.random as jrnd
from diffrax import MultiTerm, ODETerm, ControlTerm, ItoMilstein, UnsafeBrownianPath
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from jax import jit

# Define Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
args = (sigma, rho, beta)

# Define system dynamics
def lorenz63(t, y, args):
    x, y, z = y
    sigma, rho, beta = args
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return jnp.array([dx, dy, dz])

# Define SDE drift and diffusion functions
def drift(t, y, args):
    return lorenz63(t, y, args)

def diffusion(t, y, args):
    # Constant diffusion for simplicity
    return 5.0 * jnp.eye(3)

# Set up simulation parameters
y0 = jnp.array([1.0, 1.0, 1.0])  # Initial state
num_steps = 10000
step_size = 0.01
t_final = num_steps * step_size

# Create a Brownian motion path
rng = jrnd.PRNGKey(42)
bm = UnsafeBrownianPath(shape=(3,), key=rng)

# Define SDE term
sde_term = MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))

# Set up solver
solver = ItoMilstein()
solver_state = solver.init(sde_term, t0=0.0, t1=t_final, y0=y0, args=args)

# Initialize storage for results
solution = {"t": [0.0], "y": [y0]}

# Run the simulation
prev_t, prev_y = 0.0, y0
@jit
def ItoMilstein_step(t, y, args):
    next_t = t + step_size
    next_y = solver.step(
        sde_term, t, next_t, y, args, solver_state, made_jump=False
    )[0]
    return next_t, next_y


with tqdm(total=num_steps, desc="Simulating SDE", unit="step") as progress_bar:
    # Run SRA1
    while prev_t < t_final:
        # Step the SDE
        cur_t, cur_y = ItoMilstein_step(prev_t, prev_y, args)
        
        # Update storage
        solution["t"].append(cur_t)
        solution["y"].append(cur_y)
        
        prev_t = min(cur_t, t_final)
        prev_y = cur_y
        
        progress_bar.update(1)
        
# Convert lists to arrays for easier plotting
t_array = jnp.array(solution["t"])
y_array = jnp.array(solution["y"])
#%%
# Plot the results
fig = plt.figure(figsize=(15, 5))

fig.suptitle("Stochastic Lorenz 63 System with Ito Milstein Method")

# 3D trajectory plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(y_array[:, 0], y_array[:, 1], y_array[:, 2], linewidth=0.5, alpha=0.9)
ax1.set_title("3D Trajectory")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Time series plot
ax2 = fig.add_subplot(132)
ax2.plot(t_array, y_array[:, 0], label="X", linewidth=0.9, alpha=0.7)
ax2.plot(t_array, y_array[:, 1], label="Y", linewidth=0.9, alpha=0.7)
ax2.plot(t_array, y_array[:, 2], label="Z", linewidth=0.9, alpha=0.7)
ax2.set_title("Time Series")
ax2.set_xlabel("Time")
ax2.set_ylabel("Value")
ax2.legend()

# Phase space plot (X vs Y)
ax3 = fig.add_subplot(133)
ax3.plot(y_array[:, 0], y_array[:, 1], linewidth=0.9, alpha=0.7)
ax3.set_title("Phase Space (X vs Y)")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Final time: {t_array[-1]:.2f}")
print(f"Final state: {y_array[-1]}")
print(f"Mean state: {jnp.mean(y_array, axis=0)}")
print(f"State standard deviation: {jnp.std(y_array, axis=0)}")
# %%
