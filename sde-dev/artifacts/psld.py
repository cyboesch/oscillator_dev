from typing import Callable
import jax.numpy as jnp
from ctmc import ContinuousTimeMarkovChain
from diffrax._custom_types import Y, Args, RealScalarLike

def psld_D(beta: float, Gamma: float, M: float, nu: float, dim: int):
    """
    Return the D matrix for the Phase Space Langevin Diffusion (PSLD) model.
    """
    D_upper = jnp.eye(dim) * Gamma
    D_lower = jnp.eye(dim) * M * nu
    return (beta / 2) * jnp.block([[D_upper, jnp.zeros((dim, dim))],
                      [jnp.zeros((dim, dim)), D_lower]])

def psld_Q(beta: float, dim: int):
    """
    Return the Q matrix for the Phase Space Langevin Diffusion (PSLD) model.
    """
    Q_block = jnp.array([[0, -1], [1, 0]]) 
    return (beta / 2) * jnp.kron(Q_block, jnp.eye(dim))

def psld_Gamma(dim: int):
    """
    Return the Gamma vector for the Phase Space Langevin Diffusion (PSLD) model.
    """
    return jnp.zeros(2 * dim)

def create_psld_ctmc(beta: float, Gamma: float, M: float, nu: float, dim: int, H: Callable[[Y, Args], RealScalarLike]):
    """
    Create a PSLD model using the ContinuousTimeMarkovChain class.
    """
    D_fn = lambda z, args: psld_D(beta, Gamma, M, nu, dim)
    Q_fn = lambda z, args: psld_Q(beta, dim)
    tau_fn = lambda z, args: psld_Gamma(dim)
    
    return ContinuousTimeMarkovChain(
        H=H,
        D=D_fn,
        Q=Q_fn,
        Gamma=tau_fn
    )
    
def create_cld_ctmc(beta: float, M: float, dim: int, H: Callable[[Y, Args], RealScalarLike]):
    """
    Create a CLD model using the ContinuousTimeMarkovChain class.
    """
    nu_bar = jnp.sqrt(4*M)
    beta_bar = beta/2.0
    return create_psld_ctmc(beta_bar, 0.0, M, nu_bar, dim, H)