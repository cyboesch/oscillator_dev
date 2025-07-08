import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt

# Problem setup & hyperparameters
N = 3               # number of DOFs (you can keep this generic)
T = 1.0
num_steps = 500
ts = jnp.linspace(0.0, T, num_steps)

# "True" parameters for generating the target trajectory
theta_true = jnp.array([1.0, 0.1])   # [k, alpha]

def spring_chain_forces(x, v, theta):
    k, alpha = theta
    # linear nearest‐neighbor springs
    f_lin = jnp.zeros_like(x)
    f_lin = f_lin.at[1:-1].set(-k * (2*x[1:-1] - x[0:-2] - x[2:]))
    f_lin = f_lin.at[0].set(-k * (x[0] - x[1]))
    f_lin = f_lin.at[-1].set(-k * (x[-1] - x[-2]))
    # duffing (cubic) nonlinearity
    f_duf = -alpha * x**3
    return f_lin + f_duf

def external_force(t):
    # a sine plus a brief impulse
    return jnp.sin(2*jnp.pi*t) + jnp.where((t>0.3)&(t<0.35), 5.0, 0.0)

def target_x(t):
    return jnp.sin(2*jnp.pi*t)

def target_v(t):
    return 2*jnp.pi * jnp.cos(2*jnp.pi*t)

# Cell 4: Clamped dynamics → generate (x,h) & f_target(t)
def clamp_dynamics(state, t, theta):
    # state for DOFs 0..N-2: [x0..x_{N-2}, v0..v_{N-2}]
    n1 = state.shape[0]//2
    x = state[:n1]
    v = state[n1:]
    # extend with clamped last DOF
    x_ext = jnp.concatenate([x, jnp.array([target_x(t)])])
    v_ext = jnp.concatenate([v, jnp.array([target_v(t)])])
    f_int = spring_chain_forces(x_ext, v_ext, theta)
    f_int = f_int.at[0].add(external_force(t))
    a_ext = f_int   # assume M = I
    # only hidden DOFs evolve
    a = a_ext[:n1]
    return jnp.concatenate([v, a])

# integrate the clamped system
init_clamp = jnp.zeros(2*(N-1))
sol_clamp = odeint(clamp_dynamics, init_clamp, ts, theta_true)

# reconstruct full-state & f_target
x_clamp = jnp.concatenate([sol_clamp[:,:N-1], target_x(ts)[:,None]], axis=1)
v_clamp = jnp.concatenate([sol_clamp[:,N-1:], target_v(ts)[:,None]], axis=1)
f_target = spring_chain_forces(x_clamp, v_clamp, theta_true)

# Plot the clamped solution
plt.figure(figsize=(15, 12))

# Plot 1: Positions of all DOFs
plt.subplot(3, 2, 1)
for i in range(N):
    if i < N-1:
        plt.plot(ts, x_clamp[:, i], linewidth=2, label=f'DOF {i} (hidden)')
    else:
        plt.plot(ts, x_clamp[:, i], 'r--', linewidth=2, label=f'DOF {i} (clamped)')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Clamped Solution: Positions')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Velocities of all DOFs
plt.subplot(3, 2, 2)
for i in range(N):
    if i < N-1:
        plt.plot(ts, v_clamp[:, i], linewidth=2, label=f'DOF {i} (hidden)')
    else:
        plt.plot(ts, v_clamp[:, i], 'r--', linewidth=2, label=f'DOF {i} (clamped)')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Clamped Solution: Velocities')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 3: Forces on all DOFs
plt.subplot(3, 2, 3)
for i in range(N):
    plt.plot(ts, f_target[:, i], linewidth=2, label=f'DOF {i}')
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('Clamped Solution: Forces')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 4: External force
plt.subplot(3, 2, 4)
plt.plot(ts, external_force(ts), 'b-', linewidth=2, label='External Force')
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('External Force')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 5: Target trajectory vs actual clamped trajectory
plt.subplot(3, 2, 5)
plt.plot(ts, target_x(ts), 'r--', linewidth=2, label='Target Position')
plt.plot(ts, x_clamp[:, -1], 'r-', linewidth=2, label='Clamped Position')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Target vs Clamped (Last DOF)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 6: Target velocity vs actual clamped velocity
plt.subplot(3, 2, 6)
plt.plot(ts, target_v(ts), 'g--', linewidth=2, label='Target Velocity')
plt.plot(ts, v_clamp[:, -1], 'g-', linewidth=2, label='Clamped Velocity')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Target vs Clamped (Last DOF)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Print some statistics
print("Clamped Solution Statistics:")
print(f"Number of DOFs: {N}")
print(f"Number of hidden DOFs: {N-1}")
print(f"Number of clamped DOFs: 1")
print(f"Time range: 0 to {T}")
print(f"Number of time steps: {num_steps}")
print(f"Initial positions: {x_clamp[0, :]}")
print(f"Final positions: {x_clamp[-1, :]}")
print(f"Max force magnitude: {jnp.max(jnp.abs(f_target)):.4f}")
print(f"Mean force magnitude: {jnp.mean(jnp.abs(f_target)):.4f}") 