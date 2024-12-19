import pytest
import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import equinox as eqx
from cld import CriticallyDampedLangevinDynamics


@pytest.fixture
def cld_instance():
    return CriticallyDampedLangevinDynamics(state_dim=10, M=1.0, gamma=1.0, beta=1.0)


def test_initialization(cld_instance):
    assert cld_instance.state_dim == 10
    assert cld_instance.M == 1.0
    assert cld_instance.beta == 1.0
    assert cld_instance.gamma == 1.0
    assert np.isclose(cld_instance.Gamma, np.sqrt(4 * cld_instance.M))


def test_critical_damping_condition(cld_instance):
    assert np.isclose(cld_instance.Gamma**2, 4 * cld_instance.M)


def test_B(cld_instance):
    t = 0.5
    assert np.isclose(cld_instance.B(t), cld_instance.beta * t)


def test_mean_shape(cld_instance):
    x_0 = jnp.zeros(cld_instance.state_dim)
    v_0 = jnp.zeros(cld_instance.state_dim)
    t = 0.5
    mean = cld_instance.mu_t(t, x_0, v_0)
    assert mean.shape == (2 * cld_instance.state_dim,)


def test_covariance_shape(cld_instance):
    t = 0.5
    cov = cld_instance.cov_t(t, Sigma_0_xx=0, Sigma_0_vv=0)
    assert cov.shape == (2 * cld_instance.state_dim, 2 * cld_instance.state_dim)


def test_dsm_kernel_params(cld_instance):
    u_0 = jnp.zeros(2 * cld_instance.state_dim)
    t = 0.5
    mean, cov = cld_instance.dsm_kernel_params(u_0, t)
    assert mean.shape == (2 * cld_instance.state_dim,)
    assert cov.shape == (2 * cld_instance.state_dim, 2 * cld_instance.state_dim)


def test_hsm_kernel_params(cld_instance):
    x_0 = jnp.zeros(cld_instance.state_dim)
    t = 0.5
    mean, cov = cld_instance.hsm_kernel_params(x_0, t)
    assert mean.shape == (2 * cld_instance.state_dim,)
    assert cov.shape == (2 * cld_instance.state_dim, 2 * cld_instance.state_dim)


def test_get_grad_u_t_log_p_t(cld_instance):
    Sigma_t = jnp.eye(2)
    epsilon_2d = jax.random.normal(jax.random.PRNGKey(0), (2 * cld_instance.state_dim,))
    grad = cld_instance.grad_ut_log_pt(Sigma_t, epsilon_2d)
    assert grad.shape == (2 * cld_instance.state_dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_get_grad_v_t_log_p_t(cld_instance):
    Sigma_t = jnp.eye(2)
    epsilon_d = jax.random.normal(jax.random.PRNGKey(0), (cld_instance.state_dim,))
    grad = cld_instance.grad_vt_log_pt(Sigma_t, epsilon_d)
    assert grad.shape == (cld_instance.state_dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_get_dsm_grad_u_t_log_p_t(cld_instance):
    u_0 = jnp.ones(2 * cld_instance.state_dim)
    t = 0.5
    epsilon_2d = jax.random.normal(jax.random.PRNGKey(0), (2 * cld_instance.state_dim,))
    grad = cld_instance.dsm_grad_ut_log_pt(u_0, t, epsilon_2d)
    assert grad.shape == (2 * cld_instance.state_dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_get_hsm_grad_u_t_log_p_t(cld_instance):
    x_0 = jnp.ones(cld_instance.state_dim)
    t = 0.5
    epsilon_2d = jax.random.normal(jax.random.PRNGKey(0), (2 * cld_instance.state_dim,))
    grad = cld_instance.hsm_grad_ut_log_pt(x_0, t, epsilon_2d)
    assert grad.shape == (2 * cld_instance.state_dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_get_dsm_grad_v_t_log_p_t(cld_instance):
    u_t = jnp.zeros(2 * cld_instance.state_dim)
    u_0 = jnp.ones(2 * cld_instance.state_dim)
    t = 0.5
    epsilon_d = jax.random.normal(jax.random.PRNGKey(0), (cld_instance.state_dim,))
    grad = cld_instance.dsm_grad_vt_log_pt(u_t, u_0, t, epsilon_d)
    assert grad.shape == (cld_instance.state_dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_get_hsm_grad_v_t_log_p_t(cld_instance):
    x_0 = jnp.ones(cld_instance.state_dim)
    t = 0.5
    epsilon_d = jax.random.normal(jax.random.PRNGKey(0), (cld_instance.state_dim,))
    grad = cld_instance.hsm_grad_vt_log_pt(x_0, t, epsilon_d)
    assert grad.shape == (cld_instance.state_dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_get_ell_t(cld_instance):
    Sigma_t = cld_instance.cov_t(1.0, Sigma_0_xx=0, Sigma_0_vv=0)
    l_t = cld_instance.ell_t(Sigma_t[:2, :2])
    assert jnp.isscalar(l_t)
    assert jnp.isfinite(l_t)


def test_get_L_t_inv_T(cld_instance):
    Sigma_t = jnp.array([[1.0, 0.5], [0.5, 2.0]])
    L_t_inv_T = cld_instance.L_t_inv_T(Sigma_t)
    assert L_t_inv_T.shape == (2, 2)
    assert jnp.all(jnp.isfinite(L_t_inv_T))


def test_compute_Sigma_t_inv(cld_instance):
    Sigma_t = jnp.array([[1.0, 0.5], [0.5, 2.0]])
    Sigma_t_inv = cld_instance.Sigma_t_inv(Sigma_t)
    assert Sigma_t_inv.shape == (2, 2)
    assert jnp.all(jnp.isfinite(Sigma_t_inv))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
