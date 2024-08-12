import diffrax
from diffrax import ControlTerm, MultiTerm, ODETerm
from jax import grad
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from jax import config
config.update("jax_disable_jit", True)

N = 3
w_shape = (2*N,)
args = (0.5, 1.2)


def diffusion_fn(t, state, args) -> lx.DiagonalLinearOperator:
    _, diag_B, _ = args
    diagonal = jnp.concatenate([jnp.zeros_like(diag_B), diag_B])
    return lx.DiagonalLinearOperator(diagonal)

def drift_fn(t, state, args):
    x, p = state[:N], state[N:]
    diag_M_inv, diag_B, grad_x_U_fn = args
    xdot = diag_M_inv * p 
    pdot = -grad_x_U_fn(x) - diag_B * xdot
    return jnp.concatenate([xdot, pdot])


y0 = jnp.ones((2*N,))
diag_B = jnp.ones((N,))
diag_M_inv = jnp.ones((N,))
grad_x_U_fn = lambda x: jnp.ones((N,))

brownian_motion = diffrax.VirtualBrownianTree(
    0.0, 1.0, 0.05, w_shape, jr.PRNGKey(0), diffrax.SpaceTimeLevyArea
)

terms = MultiTerm(ODETerm(drift_fn), ControlTerm(diffusion_fn, brownian_motion))

saveat = diffrax.SaveAt(t1=True)

solution = diffrax.diffeqsolve(
    terms,
    solver=diffrax.SRA1(),
    t0=0.0,
    t1=1.0,
    dt0=0.1,
    y0=y0,
    args=(diag_M_inv, diag_B, grad_x_U_fn),
    saveat=saveat,
    progress_meter=diffrax.TqdmProgressMeter(),
)

print(solution.ys)
