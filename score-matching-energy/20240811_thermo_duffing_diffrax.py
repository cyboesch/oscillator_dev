import diffrax
from diffrax import ControlTerm, MultiTerm, ODETerm
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

N = 3
w_shape = (N,)
args = (0.5, 1.2)


def diffusion(t, y, args):
    a, b = args
    return lx.DiagonalLinearOperator(jnp.array([b, t, 1 / (t + 1.0)]))


def drift(t, y, args):
    a, b = args
    return -a * y


y0 = jnp.ones(w_shape)

brownian_motion = diffrax.VirtualBrownianTree(
    0.0, 1.0, 0.05, w_shape, jr.PRNGKey(0), diffrax.SpaceTimeLevyArea
)

terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
saveat = diffrax.SaveAt(t1=True)
solution = diffrax.diffeqsolve(
    terms,
    diffrax.SRA1(),
    0.0,
    1.0,
    0.1,
    y0,
    args,
    saveat=saveat,
    progress_meter=diffrax.TqdmProgressMeter(),
)
