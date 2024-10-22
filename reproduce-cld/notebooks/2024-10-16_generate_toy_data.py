# This notebook demonstrates how to generate toy data

#%%
# get parent directory and add to sys path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils.toy_data import get_multimodal_swissroll_sample, get_diamond_sample

#%%
# Create the distribution
multimodal_swissroll_sample, multimodal_swissroll_logpdf = get_multimodal_swissroll_sample()

# Example usage with vmap for multiple samples
key = jax.random.PRNGKey(0)
n_samples: int = 10_000

# Generate multiple samples using vmap
sample_vmap = jax.vmap(multimodal_swissroll_sample)
keys = jax.random.split(key, n_samples)
samples: jnp.ndarray = sample_vmap(keys)

# Plot samples
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
# set square axes
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.show()

# compute logpdf
logpdfs = jax.vmap(multimodal_swissroll_logpdf)(samples)
print(f"logpdfs: {logpdfs}")
#%%
# Create the diamond distribution
diamond_sample, diamond_logpdf = get_diamond_sample()

# Example usage with vmap for multiple inputs
key = jax.random.PRNGKey(0)
n_samples: int = 10_000

# Generate multiple samples using vmap
diamond_sample_vmap = jax.vmap(diamond_sample)
diamond_keys = jax.random.split(key, n_samples)
diamond_samples: jnp.ndarray = diamond_sample_vmap(diamond_keys)

# Plot samples
plt.scatter(diamond_samples[:, 0], diamond_samples[:, 1], alpha=0.5, s=1)
# set square axes
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.show()

# compute logpdf
logpdfs = jax.vmap(diamond_logpdf)(diamond_samples)
print(f"logpdfs: {logpdfs}")
#%%
