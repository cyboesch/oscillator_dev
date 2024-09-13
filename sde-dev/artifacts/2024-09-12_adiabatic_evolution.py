# %%
# %%
# %%
from typing import Callable, Union
import jax.numpy as jnp
import jax.random as jrnd
from diffrax import MultiTerm, ODETerm, ControlTerm, ItoMilstein, UnsafeBrownianPath
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from jax import jit
from jax.lax import map

import diffrax
from matplotlib import gridspec
import matplotlib.pyplot as plt
from diffrax import ControlTerm, MultiTerm, ODETerm
from jax import grad, vmap
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from scipy import stats

from jax import jit, devices, config
# Plot the results
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

config.update("jax_enable_x64", True)

from thermoai.distributions import mog_energy 

cpu_devices = devices("cpu")

from dataclasses import dataclass
import jax.random as jr
from cld import CriticallyDampedLangevinDynamics
@dataclass
class CLDConfig:
    dim: int
    beta: float
    M: float
    gamma: float
    Gamma: Union[float, None] = None
    H: Union[Callable, None] = None
@dataclass
class InitConfig:
    N: int
    rng: jr.PRNGKey
    step_size: float
#%%

#%%



#%%
# Create energy function with time-varying parameters
# TODO: Implement a function that defines the energy landscape
# with parameters that change over time
energy_fn = lambda u, t: None 

#%%
# Create SDE terms


cld_config = CLDConfig(dim=1, beta=1.0, M=1.0, gamma=1.0, H=energy_fn)

cld = CriticallyDampedLangevinDynamics(**cld_config)
#%%
# Create a simple solver for the SDE

# Set up solver
solver = ItoMilstein()
solver_state = solver.init(sde_terms, t0=0.0, t1=t_final, y0=jnp.empty(2 * cld.state_dim), args=None)

@jit
def ItoMilstein_step(t, y):
    next_t = t + step_size
    next_y = solver.step(sde_terms, t, next_t, y, None, solver_state, made_jump=False)[0]
    return next_t, next_y

@jit
def simulation_step(carry):
    t, y, trajectory, i = carry
    next_t, next_y = ItoMilstein_step(t, y)
    trajectory = trajectory.at[i].set(next_y)
    return next_t, next_y, trajectory, i + 1

@jit
def simulation_loop(init_carry):
    def cond_fun(carry):
        t, _, _, i = carry
        return (t < t_final) & (i < num_steps)

    return jax.lax.while_loop(cond_fun, simulation_step, init_carry)

# Modify the simulation to accept initial conditions
@jit
def run_simulation(y0):
    trajectory = jnp.zeros((num_steps, 2 * config.cld_config.state_dim))
    init_carry = (0.0, y0, trajectory, 0)
    _, _, full_trajectory, _ = simulation_loop(init_carry)
    return full_trajectory

# Set up multiple initial conditions
n_data_samples = 10
rand_init_idxs = np.random.choice(np.arange(config.init_config.N), n_data_samples, replace=False)
inits = init_x0s_p0s[rand_init_idxs]  # This should be your (n_inits, 2) array of initial conditions

# Run the simulation for all initial conditions using vmap
all_trajectories = jax.vmap(run_simulation)(inits)