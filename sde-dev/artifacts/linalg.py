import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jit, static_argnums=(3,))
def schur_inverse_2x2(S_xx, S_vv, S_xv, which_block=None):
    """
    Computes the inverse of a 2x2 block matrix using Schur complements,
    assuming 1D (scalar) inputs, and returns the inverse of the requested block.

    Args:
        S_xx: A scalar, the upper-left block (assumed to be non-zero).
        S_vv: A scalar, the lower-right block.
        S_xv: A scalar, the upper-right (and lower-left) block.

    Returns:
        Three scalars representing the blocks of the inverse matrix:
        S_inv_xx, S_inv_vv, S_inv_xv
    """
    # Calculate inverse of S_xx
    S_xx_inv = 1 / S_xx
    
    # Calculate the Schur complement of S_xx
    S_schur = S_vv - S_xv**2 * S_xx_inv
    
    S_schur_inv = 1 / S_schur
    
    S_inv_xx = S_xx_inv + S_xx_inv**2 * S_xv**2 * S_schur_inv
    S_inv_vv = S_schur_inv
    S_inv_xv = -S_schur_inv * S_xv * S_xx_inv
    
    if which_block == 'xx':
        return S_inv_xx
    elif which_block == 'vv':
        return S_inv_vv
    elif which_block == 'xv':
        return S_inv_xv
    else:
        return S_inv_xx, S_inv_vv, S_inv_xv
