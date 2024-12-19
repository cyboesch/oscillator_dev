from typing import Optional, TypeVar, Union, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import ControlTerm, MultiTerm, ODETerm
from jaxtyping import ArrayLike
from diffrax._custom_types import Y, Args, RealScalarLike, BM

class ContinuousTimeMarkovChain(eqx.Module):
    # Given a potential function `potential_fn,` this class represents a CTMC whose stationary distribution is \propto exp(-potential_fn(x, args))
    
    potential_fn: Callable[[Y, Args], RealScalarLike]   # H in Fox et al.
    D: Callable[[Y, Args], ArrayLike]   # D >= 1 in Fox et al.
    Q: Callable[[Y, Args], ArrayLike]        # Q in Fox et al.

    def __init__(
        self,
        target_logdensity_fn: Callable[[Y, Args], RealScalarLike],
        D: Union[Callable[[Y, Args], ArrayLike], None] = None,
        Q: Union[Callable[[Y, Args], ArrayLike], None] = None,
    ):
        self.potential_fn = lambda x, args: -1.0 * target_logdensity_fn(x, args)

        if D is None:
            self.D = lambda z, args: jnp.eye(z.shape[-1])
        else:
            self.D = D

        if Q is None:
            self.Q = lambda z, args: jnp.zeros((z.shape[-1], z.shape[-1]))
        else:
            self.Q = Q
            
    @eqx.filter_jit
    def drift(self, t: RealScalarLike, z: Y, args: Args) -> ArrayLike:
        grad_H = jax.grad(self.potential_fn)(z, args)
        DQ = jnp.add(self.D(z, args), self.Q(z, args))
        f = -jnp.dot(DQ, grad_H) + self.Gamma(z, args)
        return f

    @eqx.filter_jit
    def Gamma(self, z: Y, args: Args):
        # TODO: faster implementation
        def partial_sum(i):
            return jnp.sum(jax.jacfwd(lambda z, args: self.D(z, args)[i, :] + self.Q(z, args)[i, :])(z, args))

        return jax.vmap(partial_sum)(jnp.arange(z.shape[-1]))

    @eqx.filter_jit
    def diffusion(self, t, z, args):
        return jnp.sqrt(2 * self.D(z, args))

    def get_terms(self, bm: BM) -> MultiTerm:
        return MultiTerm(ODETerm(self.drift), ControlTerm(self.diffusion, bm))


class ContinuousTimeMarkovChain2(eqx.Module):
    # Given a potential function `potential_fn,` this class represents a CTMC whose stationary distribution is \propto exp(-potential_fn(x, args))
    
    H: Callable[[Y, Args], RealScalarLike]   # H in Fox et al.
    D: Callable[[Y, Args], ArrayLike]   # D >= 1 in Fox et al.
    Q: Callable[[Y, Args], ArrayLike]        # Q in Fox et al.

    def __init__(
        self,
        H: Callable[[Y, Args], RealScalarLike],
        D: Union[Callable[[Y, Args], ArrayLike], None] = None,
        Q: Union[Callable[[Y, Args], ArrayLike], None] = None,
    ):
        self.H = H
        if D is None:
            self.D = lambda z, args: jnp.eye(z.shape[-1])
        else:
            self.D = D

        if Q is None:
            self.Q = lambda z, args: jnp.zeros((z.shape[-1], z.shape[-1]))
        else:
            self.Q = Q
            
    @eqx.filter_jit
    def drift(self, t: RealScalarLike, z: Y, args: Args) -> ArrayLike:
        grad_H = jax.grad(self.H)(z, args)
        DQ = jnp.add(self.D(z, args), self.Q(z, args))
        f = -jnp.dot(DQ, grad_H) + self.Gamma(z, args)
        return f

    @eqx.filter_jit
    def Gamma(self, z: Y, args: Args):
        # TODO: faster implementation
        def partial_sum(i):
            return jnp.sum(jax.jacfwd(lambda z, args: self.D(z, args)[i, :] + self.Q(z, args)[i, :])(z, args))

        return jax.vmap(partial_sum)(jnp.arange(z.shape[-1]))

    @eqx.filter_jit
    def diffusion(self, t, z, args):
        return jnp.sqrt(2 * self.D(z, args))

    def get_terms(self, bm: BM) -> MultiTerm:
        return MultiTerm(ODETerm(self.drift), ControlTerm(self.diffusion, bm))

