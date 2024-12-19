from typing import Optional, TypeVar, Union, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import ControlTerm, MultiTerm, ODETerm
from jaxtyping import ArrayLike
from diffrax._custom_types import Y, Args, RealScalarLike, BM


class ContinuousTimeMarkovChain(eqx.Module):
    H: Callable[[RealScalarLike, Y, Args], RealScalarLike]  # H in Fox et al.
    D: Callable[[RealScalarLike, Y, Args], ArrayLike]  # D >= 1 in Fox et al.
    Q: Callable[[RealScalarLike, Y, Args], ArrayLike]  # Q in Fox et al.
    Gamma: Callable[[RealScalarLike, Y, Args], ArrayLike]  # Correction in Fox et al.

    def __init__(
        self,
        H_fn: Callable[[RealScalarLike, Y, Args], RealScalarLike],
        D_fn: Union[Callable[[RealScalarLike, Y, Args], ArrayLike], None] = None,
        Q_fn: Union[Callable[[RealScalarLike, Y, Args], ArrayLike], None] = None,
        Gamma_fn: Union[Callable[[RealScalarLike, Y, Args], ArrayLike], None] = None,
    ):
        self.H = H_fn
        if D_fn is None:
            self.D = lambda t, z, args: jnp.eye(z.shape[-1])
        else:
            self.D = D_fn

        if Q_fn is None:
            self.Q = lambda t, z, args: jnp.zeros((z.shape[-1], z.shape[-1]))
        else:
            self.Q = Q_fn

        if Gamma_fn is None:
            pass  # Use self.Gamma as below
        else:
            self.Gamma = Gamma_fn

    @eqx.filter_jit
    def drift(self, t: RealScalarLike, z: Y, args: Args) -> ArrayLike:
        grad_H = jax.grad(self.H, argnums=1)(t, z, args)
        DQ = jnp.add(self.D(t, z, args), self.Q(t, z, args))
        f = -jnp.dot(DQ, grad_H) + self.Gamma(t, z, args)
        return f

    @eqx.filter_jit
    def Gamma(self, t: RealScalarLike, z: Y, args: Args):
        # TODO: faster implementation
        def partial_sum(i):
            return jnp.sum(
                jax.jacfwd(
                    lambda t, z, args: self.D(t, z, args)[i, :] + self.Q(t, z, args)[i, :]
                )(t, z, args)
            )

        return jax.vmap(partial_sum)(jnp.arange(z.shape[-1]))

    @eqx.filter_jit
    def diffusion(self, t: RealScalarLike, z: Y, args: Args) -> ArrayLike:
        return jnp.sqrt(2 * self.D(t, z, args))

    def get_terms(self, bm: BM) -> MultiTerm:
        return MultiTerm(ODETerm(self.drift), ControlTerm(self.diffusion, bm))


def create_ctmc_from_logdensity(
    logdensity_fn: Callable[[Y, Args], RealScalarLike],
    D: Union[Callable[[RealScalarLike, Y, Args], ArrayLike], None] = None,
    Q: Union[Callable[[RealScalarLike, Y, Args], ArrayLike], None] = None,
):
    H = lambda t, z, args: -1.0 * logdensity_fn(t, z, args)
    return ContinuousTimeMarkovChain(H, D, Q)



if __name__ == "__main__":
    pass
