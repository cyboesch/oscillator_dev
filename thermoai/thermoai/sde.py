# %%
import matplotlib.pyplot as plt
import diffrax
from diffrax import ControlTerm, MultiTerm, ODETerm
from jax import grad
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from typing import Callable
from jax import grad
import jax.numpy as jnp
import jax.random as jr
import diffrax
import lineax as lx
import jax
from typing import Union
import dataclasses
import jax.tree_util as jtu
from diffrax import AbstractBrownianPath, AbstractTerm, VirtualBrownianTree
import diffrax
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Shaped

@dataclasses.dataclass(frozen=True)
class SDE:
    """
    This class helps keep all the SDE boilerplate code from diffrax out-of-view and organized.
    
    It is from diffrax/test/helpers.py.
    """
    get_terms: Callable[[AbstractBrownianPath], AbstractTerm]
    args: PyTree
    y0: PyTree[Array]
    t0: float
    t1: float
    w_shape: tuple[int, ...]

    def get_dtype(self):
        return jnp.result_type(*jtu.tree_leaves(self.y0))

    def get_bm(
        self,
        bm_key: PRNGKeyArray,
        levy_area: type[Union[diffrax.BrownianIncrement, diffrax.SpaceTimeLevyArea]],
        tol: float,
    ):
        shp_dtype = jax.ShapeDtypeStruct(self.w_shape, dtype=self.get_dtype())
        return VirtualBrownianTree(self.t0, self.t1, tol, shp_dtype, bm_key, levy_area)


#%%
def get_sde(t0, t1, dtype, key, noise_dim):
    driftkey, diffusionkey, ykey = jr.split(key, 3)
    drift_onet = None 
    ## TODO: make oscillator network data str
    diffusion_onet = None
    args = (drift_onet, diffusion_onet, noise_dim)
    y0 = jr.normal(ykey, (3,), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))

    return SDE(get_terms, args, y0, t0, t1, (noise_dim,))


def sample(energy_fn: Callable, y0: jnp.ndarray, params: tuple, consts: tuple, sample_params: dict, kBT: float):
    N = y0.shape[0] // 2  # (x1, x2)
    w_shape = (2 * N,)  # state is (x1, x2, p1, p2)
    
    U_fn: Callable[[jnp.ndarray, tuple[float, float]], jnp.ndarray] = (
        lambda x, params: energy_fn(x, *params, *consts)
    )
    grad_x_U_fn = grad(U_fn)

    def diffusion_fn(t, state, args) -> lx.DiagonalLinearOperator:
        diag_B = args[1]
        diagonal = jnp.concatenate([jnp.zeros_like(diag_B), diag_B])
        return lx.DiagonalLinearOperator(diagonal)

    def drift_fn(_, state, args):
        x, p = state.at[:N].get(), state.at[N:].get()
        diag_M_inv, diag_B, params = args
        xdot = diag_M_inv * p
        pdot = -grad_x_U_fn(x, params) - diag_B * xdot
        return jnp.concatenate([xdot, pdot])

    diag_B = sample_params.get("diag_B", 0.5 * jnp.ones((N,)))
    diag_M_inv = sample_params.get("diag_M_inv", jnp.ones((N,)))
    t0 = sample_params.get("t0", 0.0)
    t1 = sample_params.get("t1", 100.0)
    num_samples = sample_params.get("num_samples", 1000)

    brownian_motion = diffrax.VirtualBrownianTree(
        t0, t1, 0.05, w_shape, jr.PRNGKey(0), diffrax.SpaceTimeLevyArea
    )

    terms = MultiTerm(ODETerm(drift_fn), ControlTerm(diffusion_fn, brownian_motion))
    ts = jnp.linspace(t0, t1, num_samples)
    saveat = diffrax.SaveAt(ts=ts)

    solution = diffrax.diffeqsolve(
        terms,
        solver=diffrax.SRA1(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        args=(diag_M_inv, diag_B, params),
        saveat=saveat,
        progress_meter=diffrax.TqdmProgressMeter(),
    )

    return solution.ys[:, :N]  # Return only position coordinates
