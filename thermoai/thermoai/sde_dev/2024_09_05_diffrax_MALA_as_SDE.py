#%%
import jax.numpy as jnp
import jax.random as jrnd
from diffrax import MultiTerm, ODETerm, ControlTerm, Euler, ItoMilstein, UnsafeBrownianPath
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from jax import jit

# Define Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
args = (sigma, rho, beta)
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import diffrax

class TargetSDE(eqx.Module):
    H: callable
    D: callable
    Q: callable

    def __init__(self, H, D, Q):
        self.H = H  # Target distribution H(z)
        self.D = D  # Positive semidefinite diffusion matrix
        self.Q = Q  # Skew-symmetric curl matrix

    def drift(self, t, z, args):
        grad_H = jax.grad(self.H)(z)
        DQ = jnp.add(self.D(z), self.Q(z))
        f = -jnp.dot(DQ, grad_H) + self.gamma(z)
        return f

    def gamma(self, z):
        def partial_sum(i):
            return jnp.sum(jax.jacfwd(lambda z: self.D(z)[i, :] + self.Q(z)[i, :])(z))
        return jax.vmap(partial_sum)(jnp.arange(z.shape[0]))

    def diffusion(self, t, z, args):
        return jnp.sqrt(2 * self.D(z))

    def sample_trajectory(self, key, z0, t0, t1, dt):
        solver = diffrax.Euler()
        brownian_motion = diffrax.UnsafeBrownianPath(shape=(z0.shape[0],), key=key)
        solution = diffrax.diffeqsolve(
            MultiTerm(ODETerm(self.drift), ControlTerm(self.diffusion, brownian_motion)),
            solver,
            t0,
            t1,
            dt,
            y0=z0,
            adjoint=diffrax.DirectAdjoint(),
            args=None,
            max_steps=100000,
            saveat=diffrax.SaveAt(steps=True),
        )
        return solution.ys, solution.ts

# Example usage:
def H(z):
    return jnp.sum(z**2)  # Example: standard normal distribution

def D(z):
    return jnp.eye(z.shape[0])  # Constant diffusion

def Q(z):
    return jnp.zeros((z.shape[0], z.shape[0]))  # No curl

sde = TargetSDE(H, D, Q)

# Define initial conditions in a square around (0,0)
num_trajectories = 4
initial_conditions = jnp.linspace(-8, 8, int(jnp.sqrt(num_trajectories)))
initial_conditions = jnp.array(jnp.meshgrid(initial_conditions, initial_conditions)).T.reshape(-1, 2)

t0, t1 = 0.0, 10.0
dt = 0.1

# Simulate multiple trajectories
all_trajectories = []
all_ts = []

for i, z0 in tqdm(enumerate(initial_conditions)):
    key = random.PRNGKey(i)  # Use different keys for each trajectory
    trajectory, ts = sde.sample_trajectory(key, z0, t0, t1, dt)
    all_trajectories.append(trajectory)
    all_ts.append(ts)
#%%
# Plot the results
plt.figure(figsize=(12, 10))

# Plot isocontours of H
x = jnp.linspace(-8, 8, 100)
y = jnp.linspace(-8, 8, 100)
X, Y = jnp.meshgrid(x, y)

# Correctly use vmap with JAX
Z = jax.vmap(lambda x, y: H(jnp.array([x, y])))(X.ravel(), Y.ravel())
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
plt.colorbar(label='H(z)')

# Plot all trajectories
for trajectory, ts in tqdm(zip(all_trajectories, all_ts)):
    # Plot the trajectory with time-based coloring
    points = plt.scatter(trajectory[:, 0], trajectory[:, 1], c=ts, cmap='cool', alpha=1.0, s=5)
    
    # Plot the trajectory line with time-based coloring
    segments = jnp.array([trajectory[:-1], trajectory[1:]]).transpose(1, 0, 2)
    lc = mpl.collections.LineCollection(segments, cmap='cool', alpha=1.0, linewidth=1)
    lc.set_array(ts[:-1])
    plt.gca().add_collection(lc)

plt.colorbar(points, label='Time')

plt.xlabel('z[0]')
plt.ylabel('z[1]')
plt.title('Multiple SDE Trajectories and H(z) Contours')
plt.show()
# %%
