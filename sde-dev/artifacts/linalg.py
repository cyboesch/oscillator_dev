import jax.numpy as jnp
from jax import jit

@jit
def schur_inverse_2x2(S_xx, S_vv, S_xv):
    """
    Computes the inverse of a 2x2 block matrix using Schur complements,
    assuming 1D (scalar) inputs.

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
    
    return S_inv_xx, S_inv_vv, S_inv_xv

# Example usage
S_xx = 4.0
S_vv = 9.0
S_xv = 2.0

S_inv_xx, S_inv_vv, S_inv_xv = schur_inverse_2x2(S_xx, S_vv, S_xv)
print(f"Inverse blocks: S_inv_xx = {S_inv_xx}, S_inv_vv = {S_inv_vv}, S_inv_xv = {S_inv_xv}")

# compare to numpy
print(jnp.linalg.inv(jnp.array([[S_xx, S_xv], [S_xv, S_vv]])))
