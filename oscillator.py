import jax.numpy as jnp


def drift(x, t, alpha=1.0):
    """
    Drift function for the Duffing van der Pol oscillator.

    Args:
        x (jnp.ndarray): State vector of shape (2,)
        t (float): Time point.
        alpha (float): Nonlinear damping.

    Returns:
        jnp.ndarray: Drift vector of shape (2, ).
    """
    return jnp.array((x[1], x[0] * (alpha - x[0] ** 2) - x[1]))


def diffusion(x, t):
    """
    Diffusion function for the Duffing van der Pol oscillator.

    Args:
        x (jnp.ndarray): State vector of shape (2,)
        t (float): Time point.

    Returns:
        jnp.ndarray: Diffusion matrix of shape (2,).
    """
    return jnp.array((jnp.zeros_like(x[0]), x[0]))
