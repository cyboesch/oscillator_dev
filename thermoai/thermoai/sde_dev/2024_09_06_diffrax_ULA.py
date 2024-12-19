# %%
import math
from typing import cast, Optional, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax.internal as lxi
from jaxtyping import Array, PRNGKeyArray, PyTree
from lineax.internal import complex_to_real_dtype

from diffrax._custom_types import (
    AbstractBrownianIncrement,
    BrownianIncrement,
    levy_tree_transpose,
    RealScalarLike,
    SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea,
)
from diffrax._misc import (
    force_bitcast_convert_type,
    is_tuple_of_ints,
    split_by_tree,
)
from diffrax import AbstractBrownianPath
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, PyTree
from typing import Optional, Union

from diffrax._custom_types import RealScalarLike, AbstractBrownianIncrement
from typing import Optional, Union
import jax.numpy as jnp
import jax.random as jrnd
from diffrax import (
    AbstractBrownianIncrement,
    BrownianIncrement,
    MultiTerm,
    ODETerm,
    ControlTerm,
    Euler,
    ItoMilstein,
    SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea,
    UnsafeBrownianPath,
    AbstractBrownianPath,
)
from diffrax._custom_types import RealScalarLike, Y, Args
from jaxtyping import PyTree
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from jax import Array, jit
import matplotlib as mpl
from malareparam.mala import sample as mala_sample
from jax.scipy.stats import norm as Normal
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import diffrax
from jax import lax
jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
# %%
# Define initial conditions in a square around (0,0)
state_dim = 2
num_trajectories = 4

# Observe a dataset S = {x_i}_{i=1}^N
rng = random.PRNGKey(42)
num_data = 100
state_dim = 2
pop_loc = 0.0
pop_scale = 1.0
loc = jnp.zeros(state_dim)
scale = jnp.ones(state_dim)

t0 = 0.0
tau = 0.01
n_steps = 10
t1 = n_steps * tau


def plot_isocontours(
    ax, U_fn, xlim, ylim, levels=20, cmap="viridis", alpha=0.5, colorbar=True, label=""
):
    x = jnp.linspace(xlim[0], xlim[1], 100)
    y = jnp.linspace(ylim[0], ylim[1], 100)
    X, Y = jnp.meshgrid(x, y)

    Z = jax.vmap(lambda x, y: U_fn(jnp.array([x, y])))(X.ravel(), Y.ravel())
    Z = Z.reshape(X.shape)

    contour = ax.contour(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)

    if colorbar:
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(label)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return contour




class UnsafeBrownianPath(AbstractBrownianPath):
    """Brownian simulation that is only suitable for certain cases.

    This is a very quick way to simulate Brownian motion, but can only be used when all
    of the following are true:

    1. You are using a fixed step size controller. (Not an adaptive one.)

    2. You do not need to backpropagate through the differential equation.

    3. You do not need deterministic solutions with respect to `key`. (This
       implementation will produce different results based on fluctuations in
       floating-point arithmetic.)

    Internally this operates by just sampling a fresh normal random variable over every
    interval, ignoring the correlation between samples exhibited in true Brownian
    motion. Hence the restrictions above. (They describe the general case for which the
    correlation structure isn't needed.)

    !!! info "Lévy Area"

        Can be initialised with `levy_area` set to `diffrax.BrownianIncrement`, or
        `diffrax.SpaceTimeLevyArea`. If `levy_area=diffrax.SpaceTimeLevyArea`, then it
        also computes space-time Lévy area `H`. This is an additional source of
        randomness required for certain stochastic Runge--Kutta solvers; see
        [`diffrax.AbstractSRK`][] for more information.

        An error will be thrown during tracing if Lévy area is required but is not
        available.

        The choice here will impact the Brownian path, so even with the same key, the
        trajectory will be different depending on the value of `levy_area`.
    """

    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: type[
        Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
    ] = eqx.field(static=True)
    key: PRNGKeyArray

    def __init__(
        self,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
        ] = BrownianIncrement,
    ):
        self.shape = (
            jax.ShapeDtypeStruct(shape, lxi.default_floating_dtype())
            if is_tuple_of_ints(shape)
            else shape
        )
        self.key = key
        self.levy_area = levy_area

        if any(
            not jnp.issubdtype(x.dtype, jnp.inexact)
            for x in jtu.tree_leaves(self.shape)
        ):
            raise ValueError("UnsafeBrownianPath dtypes all have to be floating-point.")

    @property
    def t0(self):
        return -jnp.inf

    @property
    def t1(self):
        return jnp.inf

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> Union[PyTree[Array], AbstractBrownianIncrement]:
        del left
        if t1 is None:
            dtype = jnp.result_type(t0)
            t1 = t0
            t0 = jnp.array(0, dtype)
        else:
            with jax.numpy_dtype_promotion("standard"):
                dtype = jnp.result_type(t0, t1)
            t0 = jnp.astype(t0, dtype)
            t1 = jnp.astype(t1, dtype)
        t0 = eqxi.nondifferentiable(t0, name="t0")
        t1 = eqxi.nondifferentiable(t1, name="t1")
        t1 = cast(RealScalarLike, t1)
        t0_ = force_bitcast_convert_type(t0, jnp.int32)
        t1_ = force_bitcast_convert_type(t1, jnp.int32)
        key = jr.fold_in(self.key, t0_)
        key = jr.fold_in(key, t1_)
        key = split_by_tree(key, self.shape)
        out = jtu.tree_map(
            lambda key, shape: self._evaluate_leaf(
                t0, t1, key, shape, self.levy_area, use_levy
            ),
            key,
            self.shape,
        )
        if use_levy:
            out = levy_tree_transpose(self.shape, out)
            assert isinstance(out, self.levy_area)
        return out

    @staticmethod
    def _evaluate_leaf(
        t0: RealScalarLike,
        t1: RealScalarLike,
        key,
        shape: jax.ShapeDtypeStruct,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
        ],
        use_levy: bool,
    ):
        w_std = jnp.sqrt(t1 - t0).astype(shape.dtype)
        dt = jnp.asarray(t1 - t0, dtype=complex_to_real_dtype(shape.dtype))

        if levy_area is SpaceTimeTimeLevyArea:
            key_w, key_hh, key_kk = jr.split(key, 3)
            w = jr.normal(key_w, shape.shape, shape.dtype) * w_std
            hh_std = w_std / math.sqrt(12)
            hh = jr.normal(key_hh, shape.shape, shape.dtype) * hh_std
            kk_std = w_std / math.sqrt(720)
            kk = jr.normal(key_kk, shape.shape, shape.dtype) * kk_std
            levy_val = SpaceTimeTimeLevyArea(dt=dt, W=w, H=hh, K=kk)

        elif levy_area is SpaceTimeLevyArea:
            key_w, key_hh = jr.split(key, 2)
            w = jr.normal(key_w, shape.shape, shape.dtype) * w_std
            hh_std = w_std / math.sqrt(12)
            hh = jr.normal(key_hh, shape.shape, shape.dtype) * hh_std
            levy_val = SpaceTimeLevyArea(dt=dt, W=w, H=hh)
        elif levy_area is BrownianIncrement:
            w = jr.normal(key, shape.shape, shape.dtype) * w_std
            levy_val = BrownianIncrement(dt=dt, W=w)
        else:
            assert False

        if use_levy:
            return levy_val
        return w

#%%

class TargetSDE(eqx.Module):
    H: callable
    D: callable
    Q: callable
    state_dim: int

    def __init__(self, H, D, Q, state_dim):
        self.H = H  # Target logdensity function H(z) = log \tilde{\pi}(z, args)
        self.D = D  # Positive semidefinite diffusion matrix
        self.Q = Q  # Skew-symmetric curl matrix
        self.state_dim = state_dim

    def drift(self, t, z, args):
        grad_H = jax.grad(self.H)(z, args)
        DQ = jnp.add(self.D(z), self.Q(z))
        f = -jnp.dot(DQ, grad_H) + self.Gamma(z)
        return f

    def Gamma(self, z):
        def partial_sum(i):
            return jnp.sum(jax.jacfwd(lambda z: self.D(z)[i, :] + self.Q(z)[i, :])(z))

        return jax.vmap(partial_sum)(jnp.arange(z.shape[0]))

    def diffusion(self, t, z, args):
        return jnp.sqrt(2 * self.D(z))

    def get_terms(self, brownian_path):
        return MultiTerm(ODETerm(self.drift), ControlTerm(self.diffusion, brownian_path))


"""([pdf](zotero://open-pdf/library/items/LIDT62A8?page=2&annotation=G4LHBNFY))  
([Ma et al., 2015, p. 2](zotero://select/library/items/Q4PAWG5K))"""
# Given an observed dataset S, assume iid data x \sim p(x | theta), model parameters theta \sim p(\theta|S).
# Assume p(\theta | S) \propto exp{-U(theta)} with potential function U(theta) = -\sum_{x \in Data} log p(x|theta) -log p(theta)


def U(theta, params=None):
    # Potential function U(theta) = -\sum_{x \in Data} log p(x|theta) -log p(theta)
    return jnp.sum(theta**2)


# %%
def g(theta, r):
    return 0.0


# def H(z, params):
#     # H(z) \propto log p(\theta|z, args) + log p(z)
#     theta, r = z
#     return U(theta, params) + g(theta, r)
def H(z, params):
    # H(z) \propto log p(\theta|z, args) + log p(z)
    return U(z, params)

def D(z):
    return jnp.eye(z.shape[0])  # Constant diffusion


def Q(z):
    return jnp.zeros((z.shape[0], z.shape[0]))  # No curl


Langevin_sde = TargetSDE(H, D, Q, state_dim)

# initial_conditions = 2.0*jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
theta0 = jnp.array([0.0, 0.0])

rng, solve_rng, bm_rng = random.split(rng, 3)

brownian_path = UnsafeBrownianPath(shape=(state_dim,), key=bm_rng)
Langevin_sde_terms = Langevin_sde.get_terms(brownian_path)
args = (loc, scale)

# Initialize storage for results
solution_ys = jnp.empty((num_trajectories, n_steps, state_dim))
solution_ts = jnp.empty((n_steps))

solver = diffrax.Euler()

#%%
Langevin_sde_terms.terms[1].control.evaluate(0.0, 0.1)
#%%
@jit
def Euler_integrate(t0, t1, y0):
    return solver.step(Langevin_sde_terms, t0, t1, y0, args, None, made_jump=False)[0]


def simulation_step(carry):
    prev_t, prev_y, prev_idx, solution_ts, solution_ys = carry
    cur_t = prev_t + tau
    cur_idx = prev_idx + 1

    # Integrate the SDE from prev_t to cur_t
    cur_y = Euler_integrate(prev_t, cur_t, prev_y)

    # Store trajectories and time
    solution_ts = solution_ts.at[cur_idx].set(cur_t)
    solution_ys = solution_ys.at[:, cur_idx].set(cur_y)

    # Update for next iteration
    prev_t = jnp.minimum(cur_t, t1)
    prev_y = cur_y
    prev_idx = cur_idx

    return prev_t, prev_y, prev_idx, solution_ts, solution_ys


def simulation_cond(carry):
    prev_t, _, _, _, _ = carry
    return prev_t < t1


# Initialize
prev_t, prev_y = t0, theta0
prev_idx = -1
solution_ys = jnp.zeros((num_trajectories, n_steps, state_dim))
solution_ts = jnp.zeros(n_steps)

# Run the simulation
_, _, _, solution_ts, solution_ys = lax.while_loop(
    simulation_cond,
    simulation_step,
    (prev_t, prev_y, prev_idx, solution_ts, solution_ys),
)
from malareparam.mala import accept_or_reject
target_logdensity_fn = lambda x: -1.0*U(x, args)
target_val_and_grad_logpdf = jax.value_and_grad(target_logdensity_fn)


def ula_step(t0, t1, cur_x):
    _, cur_grad = target_val_and_grad_logpdf(cur_x)
    # Compute the proposed state.
    z = Langevin_sde_terms.terms[1].control.evaluate(t0, t1)  # z \sim N(0, step_size*I)
    prop_x = cur_x + tau*cur_grad + jnp.sqrt(2.0)*z
    return prop_x 


# Define the inference loop for BlackJAX
def ula_sample(step, num_samples):
    def body_fn(carry, _):
        prev_x, prev_t = carry
        next_t = prev_t + tau
        next_x = step(prev_t, next_t, prev_x)
        return (next_x, next_t), (next_x, next_t)
    
    init_state = (theta0, 0.0) 
    return jax.lax.scan(body_fn, (init_state), None, num_samples)[1]

ula_samples = ula_sample(ula_step, 10)

# ensure they are exactly the same
assert jnp.allclose(jnp.array(ula_samples[0]), solution_ys[0], rtol=1e-9, atol=1e-9)

ts = jnp.arange(1, n_steps)*tau
solution = diffrax.diffeqsolve(
    terms=Langevin_sde_terms,
    solver=solver,
    t0=0.0,
    t1=n_steps*tau,
    dt0=tau,
    y0=theta0,
    args=args,
    saveat=diffrax.SaveAt(t0=True, t1=True, ts=ts),
)

print(solution.ys[1:])
print(solution_ys[0])
# print(ula_samples)
#%%
ts
#%%
n_steps*tau
#%%
solution.ts.shape
#%%
solution.ys.shape
#%%

#%%