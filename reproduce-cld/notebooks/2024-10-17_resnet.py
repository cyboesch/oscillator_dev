#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.random as random
import jax.numpy as jnp
#%%
from resnet import ResNet


input_dim = 2
index_dim = 1
hidden_dim = 64
n_hidden_layers = 20

key = random.PRNGKey(0)
model = ResNet(input_dim, index_dim, hidden_dim, n_hidden_layers)

u = random.uniform(key, (1, input_dim))
t = jnp.array([1.0])

output = model(u, t)
print("Output:", output)


#%%