#%%
import jax
import jax.numpy as jnp
import jax.random as jrnd
from diffrax import Euler
from ctmc import ContinuousTimeMarkovChain
from mh import MetropolisHastingsController
from brownian import AuditableVirtualBrownianTree
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffrax._custom_types import BM
from diffrax import SaveAt
jax.config.update("jax_enable_x64", True)

# Setup parameters
key = jrnd.PRNGKey(42)
loc, scale = jnp.array([0.0]), jnp.array([1.0])
args = (loc, scale)
y0 = jnp.array([0.0])
tol = 1e-8
t0, t1 = 0.0, 1.0
dt =.1 
dim = 1
saveat = SaveAt(ts=jnp.arange(t0, t1, dt))

# Define stationary distribution
def target_logdensity_fn(x, args):
    loc, scale = args
    return jax.scipy.stats.norm.logpdf(x, loc=loc, scale=scale)[0]


jax.value_and_grad(target_logdensity_fn)(jnp.array([-0.02640058]), args)
# Create Brownian motion and SDE terms
target_ctmc = ContinuousTimeMarkovChain(target_logdensity_fn)

# Create MH controller
rng, bm_rng, mh_rng = jax.random.split(key, 3)
bm: BM = AuditableVirtualBrownianTree(t0=t0, t1=t1, tol=tol, shape=(dim,), key=bm_rng)
solver = Euler()
mh_adjust = MetropolisHastingsController(target_logdensity_fn, mh_rng)

# Initialize the solution
cur_sde_terms = target_ctmc.get_terms(bm)
cur_t = t0
final_t = t1
cur_y = y0

# Initialize the controller
next_t, cur_mh_state = mh_adjust.init(cur_sde_terms, t0, t1, y0, dt, args, None, 1)

trajectory = [y0]
accepts = []
n_steps = int((t1 - t0) / dt)
with tqdm(total=n_steps, desc="Forward sampling") as pbar:
    # Solve the SDE in a while loop
    for i in range(n_steps):

        # --- take the step
        y_candidate = solver.step(
            terms=cur_sde_terms,
            t0=cur_t,
            t1=next_t,
            y0=cur_y,
            args=args,
            solver_state=None,
            made_jump=False,
        )[0]

        # --- accept or reject the step
        do_accept, next_t0, next_t1, _, next_mh_state, _ = mh_adjust.adapt_step_size(
            t0=cur_t,
            t1=next_t,
            y0=cur_y,
            y1_candidate=y_candidate,
            args=args,
            y_error=None,
            error_order=1,
            controller_state=cur_mh_state,
        )
        jax.debug.print("{x}", x=do_accept)
        jax.debug.print("{x}", x=next_t0)
        jax.debug.print("{x}", x=next_t1)
        jax.debug.print("{x}", x=next_mh_state)
        jax.debug.print("{x}", x=cur_mh_state)
        jax.debug.print("{x}", x=cur_t)
        jax.debug.print("{x}", x=next_t)
        jax.debug.print("{x}", x=cur_y)
        jax.debug.print("{x}", x=y_candidate)
        
        # --- update the solution
        cur_y = jnp.where(do_accept, y_candidate, cur_y)
        cur_t = next_t0
        next_t = next_t1
        cur_mh_state = next_mh_state


        # --- update the RNG key
        bm_rng = jax.lax.cond(
            do_accept,
            lambda _: bm_rng,
            lambda _: jrnd.split(bm_rng, 2)[0],
            operand=None
        )
       
       
        
        cur_sde_terms = target_ctmc.get_terms(AuditableVirtualBrownianTree(t0=t0, t1=t1, tol=tol, shape=(dim,), key=bm_rng))
        
        trajectory.append(cur_y)
        accepts.append(do_accept)
        pbar.update(1)
# plot rolling mean oftrajectory

trajectory = jnp.array(trajectory)
accepts = jnp.array(accepts)
rollmean_trj = jnp.cumsum(trajectory) / jnp.arange(1, len(trajectory) + 1)
plt.plot(rollmean_trj)
plt.text(0.5, 0.5, f"Acceptance rate: {accepts.mean()}", transform=plt.gca().transAxes)
plt.show()
# %%
# Backward adjoint computation
adjoint = jnp.zeros_like(trajectory[-1])
t = t1

with tqdm(total=len(trajectory) - 1, desc="Backward adjoint") as pbar:
    for i in range(len(trajectory) - 1, 0, -1):
        t_prev = max(t - dt0, t0)

        # Compute adjoint update
        drift_jacobian = jax.jacrev(drift_term, argnums=1)(t, trajectory[i], None)
        diffusion_jacobian = jax.jacrev(diffusion_term, argnums=1)(
            t, trajectory[i], None
        )

        adjoint = adjoint + jnp.dot(adjoint, drift_jacobian) * (t - t_prev)
        adjoint = adjoint + jnp.dot(adjoint, diffusion_jacobian) @ bm.evaluate(
            t_prev, t
        )

        t = t_prev
        pbar.update(1)

print("Forward trajectory:", trajectory)
print("Final adjoint:", adjoint)

# %%
