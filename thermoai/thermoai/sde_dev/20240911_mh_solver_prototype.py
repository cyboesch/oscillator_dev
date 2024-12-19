#%%
import jax
import jax.numpy as jnp
import diffrax
from ctmc import ContinuousTimeMarkovChain
#%%

class MetropolisHastingsSolver(diffrax.Euler):
    proposal_std: float
    
    def __init__(self, proposal_std=0.1):
        super().__init__()
        self.proposal_std = proposal_std

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        # Original Euler step
        y1, error, dense_info, solver_state, results = super().step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )

        # Metropolis-Hastings adjustment
        noise = jax.random.normal(jax.random.PRNGKey(0), shape=y0.shape) * self.proposal_std
        y_proposal = y1 + noise

        # Compute acceptance probability (simplified version)
        acceptance_prob = jnp.minimum(1.0, jnp.exp(-jnp.sum((y_proposal - y1)**2)))

        # Use a differentiable approximation of accept/reject
        accept = jax.nn.sigmoid((acceptance_prob - jax.random.uniform(jax.random.PRNGKey(1))) / 0.1)

        # Blend between original and proposed state
        y_adjusted = accept * y_proposal + (1 - accept) * y1

        return y_adjusted, error, dense_info, solver_state, results
    

# Set up the SDE terms

# Set a global seed for JAX
key = jax.random.PRNGKey(0)

# Use this key when creating the VirtualBrownianTree
brownian = diffrax.VirtualBrownianTree(t0=0, t1=1, tol=1e-3,shape=(), key=key)
drift_term = diffrax.ODETerm(drift)
diffusion_term = diffrax.ControlTerm(diffusion, brownian)
terms = diffrax.MultiTerm(drift_term, diffusion_term)

# Set up the solver
solver = MetropolisHastingsSolver(proposal_std=0.1)

# Solve the SDE
solution = diffrax.diffeqsolve(
    terms=terms,
    solver=solver,
    t0=0,
    t1=1,
    dt0=0.01,
    y0=jnp.array([1.0]),
    saveat=diffrax.SaveAt(ts=jnp.linspace(0, 1, 101)),
)
#%%
import matplotlib.pyplot as plt
plt.plot(solution.ts, solution.ys)
plt.show()
#%%
