#%%
import jax
import jax.numpy as jnp
from diffrax import AbstractAdaptiveStepSizeController, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree, diffeqsolve
from diffrax._misc import split_by_tree
from ctmc import ContinuousTimeMarkovChain

jax.config.update("jax_enable_x64", True)

# Setup parameters
key = jrnd.PRNGKey(42)
loc, scale = jnp.array([0.0]), jnp.array([1.0])
args = (loc, scale)
y0 = jnp.array([0.0])
tol = 1e-8
t0, t1 = 0.0, 100.0
dt =.5 
dim = 1


class MetropolisSolver(AbstractAdaptiveSolver[_SolverState], AbstractWrappedSolver[_SolverState]):
    """Wraps another solver and applies a Metropolis adjustment step."""

    solver: AbstractSolver[_SolverState]
    log_prob_func: Callable[[Y], RealScalarLike]

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        # Step with the wrapped solver
        y1, y_error, dense_info, new_solver_state, result = self.solver.step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )

        # Compute acceptance probability
        log_prob_old = self.log_prob_func(y0)
        log_prob_new = self.log_prob_func(y1)
        log_acceptance_ratio = log_prob_new - log_prob_old

        # Metropolis acceptance step
        key = jr.PRNGKey(0)  # You might want to pass this as an argument
        u = jr.uniform(key)
        accept = jnp.log(u) < log_acceptance_ratio

        # Update y1 and error estimate based on acceptance
        y1 = jnp.where(accept, y1, y0)
        y_error = jnp.where(accept, y_error, jnp.zeros_like(y_error))

        return y1, y_error, dense_info, new_solver_state, result

    # Implement other necessary methods (init, func, etc.) as in HalfSolver
    # ...

# Define stationary distribution
def target_logdensity_fn(x, args):
    loc, scale = args
    return jax.scipy.stats.norm.logpdf(x, loc=loc, scale=scale)[0]

# Create Brownian motion and SDE terms
target_ctmc = ContinuousTimeMarkovChain(target_logdensity_fn)

class MetropolisHastingsController(AbstractAdaptiveStepSizeController):
    def __init__(self, rtol, atol, proposal_std=0.1):
        super().__init__(rtol, atol)
        self.proposal_std = proposal_std

    def adapt_step_size(self, t, y, error, order, accept, controller_state, **kwargs):
        # Generate proposal
        noise = jax.random.normal(jax.random.PRNGKey(0), shape=y.shape) * self.proposal_std
        y_proposal = y + noise

        # Compute acceptance probability (this is a simplified version)
        acceptance_prob = jnp.minimum(1.0, jnp.exp(-error))

        # Accept or reject
        accept = jax.random.uniform(jax.random.PRNGKey(1)) < acceptance_prob

        # Adjust step size based on acceptance
        dt_factor = jnp.where(accept, 1.1, 0.9)

        # Signal to resample Brownian increments if rejected
        if not accept:
            kwargs['resample_brownian'] = True

        return dt_factor, accept, controller_state
    
class ResampleableVirtualBrownianTree(VirtualBrownianTree):
    _resample: bool
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resample = False

    def resample(self):
        self._resample = True

    def evaluate(self, *args, **kwargs):
        if self._resample:
            self.key = split_by_tree(self.key, self.shape)
            self._resample = False
        return super().evaluate(*args, **kwargs)
    


# Use this key when creating the VirtualBrownianTree
brownian = ResampleableVirtualBrownianTree(t0=0, t1=1, tol=1e-3,shape=(), key=jax.random.PRNGKey(0))
target_ctmc.get_terms(brownian)

# Set up the solver and controller
solver = Euler()
controller = MetropolisHastingsController(rtol=1e-3, atol=1e-6)
ctmc_terms = target_ctmc.get_terms(brownian)
saveat = SaveAt(ts=jnp.arange(t0, t1, dt))

# Solve the SDE
solution = diffeqsolve(
    terms=ctmc_terms,
    solver=solver,
    t0=0,
    t1=1,
    dt0=0.01,
    y0=jnp.array([1.0]),
    saveat=saveat
    stepsize_controller=controller,
)
#%%
import matplotlib.pyplot as plt
plt.plot(solution.ts, solution.ys)
plt.show()
#%%
import matplotlib.pyplot as plt
