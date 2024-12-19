# %%# %%#%%
import numpy as np
import matplotlib.pyplot as plt

def conditional_brownian_bridge(start1, start2, end, T=1.0, n_steps=1000):
    """
    Simulate two Brownian motions conditioned to meet at the same end point.
    
    :param start1: Starting point of the first Brownian motion
    :param start2: Starting point of the second Brownian motion
    :param end: Common end point for both Brownian motions
    :param T: Total time
    :param n_steps: Number of time steps
    :return: Time array and two arrays for the Brownian bridges
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    
    # Generate two standard Brownian motions
    dB1 = np.random.normal(0, np.sqrt(dt), n_steps)
    dB2 = np.random.normal(0, np.sqrt(dt), n_steps)
    
    B1 = np.cumsum(dB1)
    B2 = np.cumsum(dB2)
    
    B1 = np.insert(B1, 0, 0)
    B2 = np.insert(B2, 0, 0)
    
    # Convert to Brownian bridges
    X1 = start1 + B1 - t * (B1[-1] + start1 - end) / T
    X2 = start2 + B2 - t * (B2[-1] + start2 - end) / T
    
    return t, X1, X2

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
start1 = 5.0
start2 = -5.0
end = 0.5
T = 1.0
n_steps = 1000

# Simulate multiple pairs of conditional Brownian bridges
n_pairs = 3

plt.figure(figsize=(10, 6))

for i in range(n_pairs):
    t, X1, X2 = conditional_brownian_bridge(start1, start2, end, T, n_steps)
    plt.plot(t, X1, label=f'Path 1 (Pair {i+1})')
    plt.plot(t, X2, label=f'Path 2 (Pair {i+1})')

plt.title('Conditional Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
#%%

from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

def estimate_diamond(key, num_points):
    x = random.uniform(key, shape=(num_points,))
    y = random.uniform(key, shape=(num_points,))
    inside_diamond = (jnp.abs(x) + jnp.abs(y)) <= 1
    diamond_estimate = jnp.mean(inside_diamond)
    return diamond_estimate

def run_simulations(key, num_simulations, points_per_simulation):
    keys = random.split(key, num_simulations)
    estimates = jax.vmap(estimate_diamond, in_axes=(0, None))(keys, points_per_simulation)
    return estimates

def plot_diamond_trajectories(diamond_estimates, num_trajectories=5):
    plt.figure(figsize=(12, 8))
    
    # Plot individual trajectories
    for i in range(num_trajectories):
        cumulative_mean = jnp.cumsum(diamond_estimates[i::num_trajectories]) / jnp.arange(1, len(diamond_estimates[i::num_trajectories]) + 1)
        plt.plot(cumulative_mean, alpha=0.5, label=f'Trajectory {i+1}')
    
    # Plot overall mean
    overall_mean = jnp.cumsum(diamond_estimates) / jnp.arange(1, len(diamond_estimates) + 1)
    plt.plot(overall_mean, 'k-', linewidth=2, label='Overall Mean')
    
    plt.xlabel('Number of Simulations')
    plt.ylabel('Estimated Diamond Area')
    plt.title('Convergence of Diamond Area Estimates')
    plt.legend()
    plt.grid(True)
    plt.show()

# Set up the simulation
num_simulations = 1000
points_per_simulation = 10000

# Initialize the random key
key = random.PRNGKey(0)

# JIT-compile the simulation function
run_simulations_jit = run_simulations

# Run the simulations
diamond_estimates = run_simulations_jit(key, num_simulations, points_per_simulation)

# Calculate mean and standard deviation of the estimates
mean_estimate = jnp.mean(diamond_estimates)
std_estimate = jnp.std(diamond_estimates)

print(f"Estimated Diamond Area: {mean_estimate:.6f} ± {std_estimate:.6f}")

# Plot the trajectories
plot_diamond_trajectories(diamond_estimates)

#%%

# Simulate multiple pairs of conditional Brownian bridges
n_pairs = 3

plt.figure(figsize=(10, 6))

for i in range(n_pairs):
    t, X1, X2 = conditional_brownian_bridge(start1, start2, end, T, n_steps)
    plt.plot(t, X1, label=f'Path 1 (Pair {i+1})')
    plt.plot(t, X2, label=f'Path 2 (Pair {i+1})')
    
    # Plot starting and end points
    plt.scatter(0, start1, color='blue', marker='o')  # Start point of X1
    plt.scatter(0, start2, color='orange', marker='o')  # Start point of X2
    plt.scatter(T, end, color='red', marker='x')  # End point for both X1 and X2

plt.title('Conditional Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

def conditional_brownian_bridge(start1, start2, end, T=1.0, n_steps=1000):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    
    dB1 = np.random.normal(0, np.sqrt(dt), n_steps)
    dB2 = np.random.normal(0, np.sqrt(dt), n_steps)
    
    B1 = np.cumsum(dB1)
    B2 = np.cumsum(dB2)
    
    B1 = np.insert(B1, 0, 0)
    B2 = np.insert(B2, 0, 0)
    
    X1 = start1 + B1 - t * (B1[-1] + start1 - end) / T
    X2 = start2 + B2 - t * (B2[-1] + start2 - end) / T
    
    return t, X1, X2

def merge_brownian_bridges(starts, end, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps+1)
    paths = [np.zeros(n_steps+1) for _ in range(n)]
    
    for i in range(n):
        paths[i][0] = starts[i]
    
    partition_points = np.log2(n).astype(int)
    for _ in range(partition_points):
        new_starts = []
        for i in range(0, len(starts), 2):
            t, X1, X2 = conditional_brownian_bridge(starts[i], starts[i+1], end, T, n_steps)
            new_starts.append(end)
            for j in range(n_steps+1):
                paths[i][j] = X1[j]
                paths[i+1][j] = X2[j]
        starts = new_starts
    
    return t, paths

# Parameters
N = 8
starts = np.linspace(0, 1, N)
end = 0.5
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths = merge_brownian_bridges(starts, end, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting and end points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')  # Start points
plt.scatter(T, end, color='red', marker='x')  # End point

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

def conditional_brownian_bridge(start1, start2, end, T=1.0, n_steps=1000):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    
    dB1 = np.random.normal(0, np.sqrt(dt), n_steps)
    dB2 = np.random.normal(0, np.sqrt(dt), n_steps)
    
    B1 = np.cumsum(dB1)
    B2 = np.cumsum(dB2)
    
    B1 = np.insert(B1, 0, 0)
    B2 = np.insert(B2, 0, 0)
    
    X1 = start1 + B1 - t * (B1[-1] + start1 - end) / T
    X2 = start2 + B2 - t * (B2[-1] + start2 - end) / T
    
    return t, X1, X2

def merge_brownian_bridges(starts, end, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps+1)
    paths = [np.zeros(n_steps+1) for _ in range(n)]
    
    for i in range(n):
        paths[i][0] = starts[i]
    
    partition_points = np.sort(np.random.uniform(0, T, int(np.log2(n))))
    
    for partition in partition_points:
        new_starts = []
        indices = np.random.permutation(len(starts))
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i+1]
            t, X1, X2 = conditional_brownian_bridge(starts[idx1], starts[idx2], end, partition, n_steps)
            new_starts.append(end)
            for j in range(n_steps+1):
                paths[idx1][j] = X1[j]
                paths[idx2][j] = X2[j]
        starts = new_starts
    
    return t, paths

# Parameters
N = 8
starts = np.linspace(0, 1, N)
end = 0.5
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths = merge_brownian_bridges(starts, end, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting and end points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')  # Start points
plt.scatter(T, end, color='red', marker='x')  # End point

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

def conditional_brownian_bridge(start1, start2, end, T=1.0, n_steps=1000):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    
    dB1 = np.random.normal(0, np.sqrt(dt), n_steps)
    dB2 = np.random.normal(0, np.sqrt(dt), n_steps)
    
    B1 = np.cumsum(dB1)
    B2 = np.cumsum(dB2)
    
    B1 = np.insert(B1, 0, 0)
    B2 = np.insert(B2, 0, 0)
    
    X1 = start1 + B1 - t * (B1[-1] + start1 - end) / T
    X2 = start2 + B2 - t * (B2[-1] + start2 - end) / T
    
    return t, X1, X2

def merge_brownian_bridges(starts, end, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps+1)
    paths = [np.zeros(n_steps+1) for _ in range(n)]
    
    for i in range(n):
        paths[i][0] = starts[i]
    
    partition_points = np.sort(np.random.uniform(0, T, int(np.log2(n))))
    
    for partition in partition_points:
        new_starts = []
        indices = np.random.permutation(len(starts))
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i+1]
            t, X1, X2 = conditional_brownian_bridge(starts[idx1], starts[idx2], end, partition, n_steps)
            new_starts.append(end)
            for j in range(n_steps+1):
                paths[idx1][j] = X1[j]
                paths[idx2][j] = X2[j]
        starts = new_starts
    
    return t, paths, partition_points

# Parameters
N = 8
starts = np.linspace(0, 1, N)
end = 0.5
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths, partition_points = merge_brownian_bridges(starts, end, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting and end points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')  # Start points
plt.scatter(T, end, color='red', marker='x')  # End point

# Plot partition points as red dashed vertical lines
for partition in partition_points:
    plt.axvline(x=partition, color='red', linestyle='--')

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

def conditional_brownian_bridge(start1, start2, end, T=1.0, n_steps=1000):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    
    dB1 = np.random.normal(0, np.sqrt(dt), n_steps)
    dB2 = np.random.normal(0, np.sqrt(dt), n_steps)
    
    B1 = np.cumsum(dB1)
    B2 = np.cumsum(dB2)
    
    B1 = np.insert(B1, 0, 0)
    B2 = np.insert(B2, 0, 0)
    
    X1 = start1 + B1 - t * (B1[-1] + start1 - end) / T
    X2 = start2 + B2 - t * (B2[-1] + start2 - end) / T
    
    return t, X1, X2

def merge_brownian_bridges(starts, end, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps+1)
    paths = [np.zeros(n_steps+1) for _ in range(n)]
    
    for i in range(n):
        paths[i][0] = starts[i]
    
    partition_points = np.sort(np.random.uniform(0, T, int(np.log2(n))))
    
    for partition in partition_points:
        new_starts = []
        indices = np.random.permutation(len(starts))
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i+1]
            _, X1, X2 = conditional_brownian_bridge(starts[idx1], starts[idx2], end, partition, n_steps)
            new_starts.append(end)
            for j in range(n_steps+1):
                paths[idx1][j] = X1[j]
                paths[idx2][j] = X2[j]
        starts = new_starts
        if len(starts) == 1:
            break
    
    return t, paths, partition_points

# Parameters
N = 8
starts = np.linspace(0, 1, N)
end = 0.5
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths, partition_points = merge_brownian_bridges(starts, end, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting and end points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')  # Start points
plt.scatter(T, end, color='red', marker='x')  # End point

# Plot partition points as red dashed vertical lines
for partition in partition_points:
    plt.axvline(x=partition, color='red', linestyle='--')

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt

def conditional_brownian_bridge(start1, start2, end, T=1.0, n_steps=1000):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    
    dB1 = np.random.normal(0, np.sqrt(dt), n_steps)
    dB2 = np.random.normal(0, np.sqrt(dt), n_steps)
    
    B1 = np.cumsum(dB1)
    B2 = np.cumsum(dB2)
    
    B1 = np.insert(B1, 0, 0)
    B2 = np.insert(B2, 0, 0)
    
    X1 = start1 + B1 - t * (B1[-1] + start1 - end) / T
    X2 = start2 + B2 - t * (B2[-1] + start2 - end) / T
    
    return t, X1, X2

def merge_brownian_bridges(starts, end, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps+1)
    paths = [np.zeros(n_steps+1) for _ in range(n)]
    
    for i in range(n):
        paths[i][0] = starts[i]
    
    partition_points = np.sort(np.random.uniform(0, T, int(np.log2(n))))
    
    for partition in partition_points:
        new_starts = []
        indices = np.random.permutation(len(starts))
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i+1]
            _, X1, X2 = conditional_brownian_bridge(starts[idx1], starts[idx2], end, partition, n_steps)
            new_starts.append(end)
            for j in range(n_steps+1):
                paths[idx1][j] = X1[j]
                paths[idx2][j] = X2[j]
        starts = new_starts
        if len(starts) == 1:
            break
    
    return t, paths, partition_points

# Parameters
N = 4
starts = np.linspace(0, 1, N)
end = 0.5
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths, partition_points = merge_brownian_bridges(starts, end, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting and end points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')  # Start points
plt.scatter(T, end, color='red', marker='x')  # End point

# Plot partition points as red dashed vertical lines
for partition in partition_points:
    plt.axvline(x=partition, color='red', linestyle='--')

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

def conditional_brownian_bridge(start1, start2, end, T=1.0, n_steps=1000):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    
    dB1 = np.random.normal(0, np.sqrt(dt), n_steps)
    dB2 = np.random.normal(0, np.sqrt(dt), n_steps)
    
    B1 = np.cumsum(dB1)
    B2 = np.cumsum(dB2)
    
    B1 = np.insert(B1, 0, 0)
    B2 = np.insert(B2, 0, 0)
    
    X1 = start1 + B1 - t * (B1[-1] + start1 - end) / T
    X2 = start2 + B2 - t * (B2[-1] + start2 - end) / T
    
    return t, X1, X2

def merge_brownian_bridges(starts, end, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps+1)
    paths = [np.zeros(n_steps+1) for _ in range(n)]
    
    for i in range(n):
        paths[i][0] = starts[i]
    
    partition_points = np.sort(np.random.uniform(0, T, int(np.log2(n))))
    
    for partition in partition_points:
        new_starts = []
        indices = np.random.permutation(len(starts))
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i+1]
            t, X1, X2 = conditional_brownian_bridge(starts[idx1], starts[idx2], end, partition, n_steps)
            new_start = (X1[int(partition * n_steps / T)] + X2[int(partition * n_steps / T)]) / 2
            new_starts.append(new_start)
            for j in range(int(partition * n_steps / T) + 1):
                paths[idx1][j] = X1[j]
                paths[idx2][j] = X2[j]
        starts = new_starts
        if len(starts) == 1:
            break
    
    return t, paths, partition_points

# Parameters
N = 4
starts = np.linspace(0, 1, N)
end = 0.5
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths, partition_points = merge_brownian_bridges(starts, end, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting and end points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')  # Start points
plt.scatter(T, end, color='red', marker='x')  # End point

# Plot partition points as red dashed vertical lines
for partition in partition_points:
    plt.axvline(x=partition, color='red', linestyle='--')

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

def conditional_brownian_bridge(start1, start2, end, T=1.0, n_steps=1000):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    
    dB1 = np.random.normal(0, np.sqrt(dt), n_steps)
    dB2 = np.random.normal(0, np.sqrt(dt), n_steps)
    
    B1 = np.cumsum(dB1)
    B2 = np.cumsum(dB2)
    
    B1 = np.insert(B1, 0, 0)
    B2 = np.insert(B2, 0, 0)
    
    X1 = start1 + B1 - t * (B1[-1] + start1 - end) / T
    X2 = start2 + B2 - t * (B2[-1] + start2 - end) / T
    
    return t, X1, X2

def merge_brownian_bridges(starts, end, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps+1)
    paths = [np.zeros(n_steps+1) for _ in range(n)]
    
    for i in range(n):
        paths[i][0] = starts[i]
    
    partition_points = np.sort(np.random.uniform(0, T, int(np.log2(n))))
    
    for partition in partition_points:
        new_starts = []
        indices = np.random.permutation(len(starts))
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i+1]
            _, X1, X2 = conditional_brownian_bridge(starts[idx1], starts[idx2], end, partition, n_steps)
            coalesce_point = (X1[int(partition * n_steps / T)] + X2[int(partition * n_steps / T)]) / 2
            new_starts.append(coalesce_point)
            for j in range(int(partition * n_steps / T) + 1):
                paths[idx1][j] = X1[j]
                paths[idx2][j] = X2[j]
        starts = new_starts
        if len(starts) == 1:
            break
    
    return t, paths, partition_points

# Parameters
N = 4
starts = np.linspace(0, 1, N)
end = 0.5
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths, partition_points = merge_brownian_bridges(starts, end, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting and end points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')  # Start points
plt.scatter(T, end, color='red', marker='x')  # End point

# Plot partition points as red dashed vertical lines
for partition in partition_points:
    plt.axvline(x=partition, color='red', linestyle='--')

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

def conditional_brownian_bridge(start1, start2, end, T=1.0, n_steps=1000):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    
    dB1 = np.random.normal(0, np.sqrt(dt), n_steps)
    dB2 = np.random.normal(0, np.sqrt(dt), n_steps)
    
    B1 = np.cumsum(dB1)
    B2 = np.cumsum(dB2)
    
    B1 = np.insert(B1, 0, 0)
    B2 = np.insert(B2, 0, 0)
    
    X1 = start1 + B1 - t * (B1[-1] + start1 - end) / T
    X2 = start2 + B2 - t * (B2[-1] + start2 - end) / T
    
    return t, X1, X2

def merge_brownian_bridges(starts, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps+1)
    paths = [np.zeros(n_steps+1) for _ in range(n)]
    
    for i in range(n):
        paths[i][0] = starts[i]
    
    partition_points = np.sort(np.random.uniform(0, T, int(np.log2(n))))
    
    for partition in partition_points:
        new_starts = []
        indices = np.random.permutation(len(starts))
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i+1]
            _, X1, X2 = conditional_brownian_bridge(starts[idx1], starts[idx2], starts[idx1], partition, n_steps)
            coalesce_point = X1[int(partition * n_steps / T)]
            new_starts.append(coalesce_point)
            for j in range(int(partition * n_steps / T) + 1):
                paths[idx1][j] = X1[j]
                paths[idx2][j] = X2[j]
        starts = new_starts
        if len(starts) == 1:
            break
    
    return t, paths, partition_points

# Parameters
N = 4
starts = np.linspace(0, 1, N)
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths, partition_points = merge_brownian_bridges(starts, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting and end points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')  # Start points
plt.scatter(T, starts[0], color='red', marker='x')  # End point

# Plot partition points as red dashed vertical lines
for partition in partition_points:
    plt.axvline(x=partition, color='red', linestyle='--')

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

def brownian_bridge(start, end, T, n_steps):
    t = np.linspace(0, T, n_steps)
    dt = T / (n_steps - 1)
    dB = np.random.normal(0, np.sqrt(dt), n_steps - 1)
    B = np.cumsum(dB)
    B = np.insert(B, 0, 0)
    return start + B - t * (B[-1] + start - end) / T

def merge_brownian_bridges(starts, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps)
    paths = [np.zeros(n_steps) for _ in range(n)]
    partition_points = [0.25, 0.5, 0.75]
    
    # First bridge
    paths[0][:int(0.25*n_steps)] = brownian_bridge(starts[0], starts[1], 0.25, int(0.25*n_steps))
    
    # Second bridge
    paths[1][:int(0.5*n_steps)] = brownian_bridge(starts[1], starts[2], 0.5, int(0.5*n_steps))
    
    # Remaining bridges
    for i in range(2, n):
        paths[i][:int(0.75*n_steps)] = brownian_bridge(starts[i], starts[3], 0.75, int(0.75*n_steps))
    
    # Coalesce at 0.25
    coalesce_point_1 = paths[0][int(0.25*n_steps)-1]
    paths[1][int(0.25*n_steps):int(0.5*n_steps)] = brownian_bridge(coalesce_point_1, paths[1][int(0.5*n_steps)-1], 0.25, int(0.25*n_steps))
    
    # Coalesce at 0.5
    coalesce_point_2 = paths[2][int(0.5*n_steps)-1]
    paths[2][int(0.5*n_steps):int(0.75*n_steps)] = brownian_bridge(coalesce_point_2, paths[3][int(0.75*n_steps)-1], 0.25, int(0.25*n_steps))

    
    # Coalesce at 0.75 and continue to end
    coalesce_point_3 = paths[2][int(0.75*n_steps)-1]
    final_path = brownian_bridge(coalesce_point_3, starts[0], 0.25, int(0.25*n_steps))
    for i in range(n):
        paths[i][int(0.75*n_steps):] = final_path
    
    return t, paths, partition_points

# Parameters
N = 4
starts = np.linspace(0, 5, N)
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths, partition_points = merge_brownian_bridges(starts, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')

# Plot partition points as red dashed vertical lines
for partition in partition_points:
    plt.axvline(x=partition, color='red', linestyle='--')

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

def brownian_bridge(start, end, T, n_steps):
    t = np.linspace(0, T, n_steps)
    dt = T / (n_steps - 1)
    dB = np.random.normal(0, np.sqrt(dt), n_steps - 1)
    B = np.cumsum(dB)
    B = np.insert(B, 0, 0)
    return start + B - t * (B[-1] + start - end) / T

def merge_brownian_bridges(starts, T=1.0, n_steps=1000):
    n = len(starts)
    t = np.linspace(0, T, n_steps)
    paths = [np.zeros(n_steps) for _ in range(n)]
    partition_points = [0.25, 0.5, 0.75]
    
    # First bridge
    paths[0][:int(0.25*n_steps)] = brownian_bridge(starts[0], starts[1], 0.25, int(0.25*n_steps))
    paths[1][:int(0.25*n_steps)] = brownian_bridge(starts[1], starts[1], 0.5, int(0.25*n_steps))
   
    # second bridge
    paths[2][:int(0.5*n_steps)] = brownian_bridge(starts[2], starts[2], 0.75, int(0.5*n_steps))
    paths[3][:int(0.5*n_steps)] = brownian_bridge(starts[3], starts[2], 0.75, int(0.5*n_steps))
    
    # fourth bridge starts at .25 and ends at .75
    paths[4][int(0.25*n_steps):int(0.75*n_steps)] = brownian_bridge(paths[0][int(0.25*n_steps)-1], paths[1][int(0.75*n_steps)-1], 0.5, int(0.5*n_steps))

    
    # continue the second bridge with a new brownian bridge to 0.75
    
    
    
    # # Remaining bridges
    # for i in range(2, n):
    #     paths[i][:int(0.75*n_steps)] = brownian_bridge(starts[i], starts[3], 0.75, int(0.75*n_steps))
    
    # # Continue first bridge from 0.25 to 0.75
    # paths[0][int(0.25*n_steps):int(0.75*n_steps)] = brownian_bridge(paths[0][int(0.25*n_steps)-1], paths[2][int(0.75*n_steps)-1], 0.5, int(0.5*n_steps))
    
    # # Continue second bridge from 0.5 to 0.75
    # paths[1][int(0.5*n_steps):int(0.75*n_steps)] = brownian_bridge(paths[1][int(0.5*n_steps)-1], paths[2][int(0.75*n_steps)-1], 0.25, int(0.25*n_steps))
    
    # # Coalesce at 0.75 and continue to end
    # coalesce_point = paths[2][int(0.75*n_steps)-1]
    # final_path = brownian_bridge(coalesce_point, starts[0], 0.25, int(0.25*n_steps))
    # for i in range(n):
    #     paths[i][int(0.75*n_steps):] = final_path
    
    return t, paths, partition_points

# Parameters
N = 7
starts = np.linspace(0, 6, N)
T = 1.0
n_steps = 1000

# Simulate the merging process
t, paths, partition_points = merge_brownian_bridges(starts, T, n_steps)

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, paths[i], label=f'Path {i+1}')

# Plot starting points
for start in starts:
    plt.scatter(0, start, color='blue', marker='o')

# Plot partition points as red dashed vertical lines
for partition in partition_points:
    plt.axvline(x=partition, color='red', linestyle='--')

plt.title('Merging Brownian Bridges')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
# %%
