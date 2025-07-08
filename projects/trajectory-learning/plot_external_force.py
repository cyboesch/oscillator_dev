import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Problem setup & hyperparameters
N = 3               # number of DOFs (you can keep this generic)
T = 1.0
num_steps = 500
ts = jnp.linspace(0.0, T, num_steps)

def external_force(t):
    # a sine plus a brief impulse
    return jnp.sin(2*jnp.pi*t) + jnp.where((t>0.3)&(t<0.35), 5.0, 0.0)

def target_x(t):
    return jnp.sin(2*jnp.pi*t)

def target_v(t):
    return 2*jnp.pi * jnp.cos(2*jnp.pi*t)

# Plot external force over time T
plt.figure(figsize=(12, 8))

# Plot 1: External Force
plt.subplot(2, 1, 1)
plt.plot(ts, external_force(ts), 'b-', linewidth=2, label='External Force')
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('External Force vs Time')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Target trajectory for reference
plt.subplot(2, 1, 2)
plt.plot(ts, target_x(ts), 'r-', linewidth=2, label='Target Position')
plt.plot(ts, target_v(ts), 'g-', linewidth=2, label='Target Velocity')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Target Trajectory vs Time')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Print some key information
print(f"Time range: 0 to {T}")
print(f"Number of time steps: {num_steps}")
print(f"Time step size: {T/num_steps:.4f}")
print(f"External force at t=0: {external_force(0.0):.4f}")
print(f"External force at t=0.5: {external_force(0.5):.4f}")
print(f"External force at t=0.32 (during impulse): {external_force(0.32):.4f}") 