import pytest
import jax.numpy as jnp
from jax import jit
import numpy as np

@jit
def schur_inverse_square_blocks(S_xx, S_vv, S_xv):
    S_xx_inv = 1 / S_xx
    S_schur = S_vv - S_xv**2 * S_xx_inv
    S_schur_inv = 1 / S_schur
    S_inv_xx = S_xx_inv + S_xx_inv**2 * S_xv**2 * S_schur_inv
    S_inv_vv = S_schur_inv
    S_inv_xv = -S_schur_inv * S_xv * S_xx_inv
    return S_inv_xx, S_inv_vv, S_inv_xv

@pytest.mark.parametrize("S_xx, S_vv, S_xv", [
    (4.0, 9.0, 2.0),
    (1.0, 1.0, 0.5),
    (2.0, 3.0, 1.0),
    (10.0, 5.0, 3.0),
])
def test_schur_inverse_square_blocks(S_xx, S_vv, S_xv):
    # Compute inverse using our function
    S_inv_xx, S_inv_vv, S_inv_xv = schur_inverse_square_blocks(S_xx, S_vv, S_xv)
    
    # Compute inverse using numpy
    S = np.array([[S_xx, S_xv], [S_xv, S_vv]])
    S_inv_np = np.linalg.inv(S)
    
    # Extract components from numpy inverse
    S_inv_xx_np, S_inv_vv_np = S_inv_np[0, 0], S_inv_np[1, 1]
    S_inv_xv_np = S_inv_np[0, 1]  # or S_inv_np[1, 0], they should be the same
    
    # Assert that our results match numpy's results
    np.testing.assert_allclose(S_inv_xx, S_inv_xx_np, rtol=1e-5)
    np.testing.assert_allclose(S_inv_vv, S_inv_vv_np, rtol=1e-5)
    np.testing.assert_allclose(S_inv_xv, S_inv_xv_np, rtol=1e-5)

if __name__ == "__main__":
    pytest.main(["-v", __file__])