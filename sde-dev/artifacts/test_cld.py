import pytest
import jax
import diffrax
import numpy as np
import equinox as eqx
from cld import CriticallyDampedLangevinDiffusion


@pytest.fixture
def cld_instance():
    return CriticallyDampedLangevinDiffusion(d=10, M=1.0, beta=1.0)


def test_initialization(cld_instance):
    assert cld_instance.d == 10
    assert cld_instance.M == 1.0
    assert cld_instance.beta == 1.0
    assert cld_instance.gamma == 1.0
    assert np.isclose(cld_instance.Gamma, np.sqrt(4 * cld_instance.M))


def test_critical_damping_condition(cld_instance):
    assert np.isclose(cld_instance.Gamma**2, 4 * cld_instance.M)


def test_drift_shape(cld_instance):
    t = 0.5
    u = np.zeros(2 * cld_instance.d)
    drift = cld_instance.drift(t, u, None)
    assert drift.shape == (2 * cld_instance.d,)


def test_diffusion_shape(cld_instance):
    t = 0.5
    u = np.zeros(2 * cld_instance.d)
    diffusion = cld_instance.diffusion(t, u, None)
    assert diffusion.shape == (2 * cld_instance.d, 2 * cld_instance.d)


def test_get_terms(cld_instance):
    rng = jax.random.PRNGKey(0)
    bm = diffrax.UnsafeBrownianPath(shape=(cld_instance.d,), key=rng)
    sde_terms = cld_instance.get_terms(bm)
    assert isinstance(sde_terms, diffrax.MultiTerm)


def test_B_function(cld_instance):
    t = 0.5
    assert np.isclose(cld_instance.B(t), cld_instance.beta * t)


def test_mean_shape(cld_instance):
    x_0 = np.zeros(cld_instance.d)
    v_0 = np.zeros(cld_instance.d)
    t = 0.5
    mean = cld_instance.mean(x_0, v_0, t)
    assert mean.shape == (2 * cld_instance.d,)


def test_covariance_shape(cld_instance):
    t = 0.5
    cov = cld_instance.covariance(t, Sigma_0_xx=0, Sigma_0_vv=0)
    assert cov.shape == (2 * cld_instance.d, 2 * cld_instance.d)


def test_get_dsm_kernel_params(cld_instance):
    u_0 = np.zeros(2 * cld_instance.d)
    t = 0.5
    mean, cov = cld_instance.get_dsm_kernel_params(u_0, t)
    assert mean.shape == (2 * cld_instance.d,)
    assert cov.shape == (2 * cld_instance.d, 2 * cld_instance.d)


def test_get_hsm_kernel_params(cld_instance):
    x_0 = np.zeros(cld_instance.d)
    t = 0.5
    mean, cov = cld_instance.get_hsm_kernel_params(x_0, t)
    assert mean.shape == (2 * cld_instance.d,)
    assert cov.shape == (2 * cld_instance.d, 2 * cld_instance.d)


def test_compute_grad_u_t_log_p_t(cld_instance):
    Sigma_t = np.eye(2)
    epsilon_2d = np.random.normal(size=(2 * cld_instance.d,))
    grad = cld_instance.compute_grad_u_t_log_p_t(Sigma_t, epsilon_2d)
    assert grad.shape == (2 * cld_instance.d,)
    assert np.all(np.isfinite(grad))


def test_compute_grad_v_t_log_p_t(cld_instance):
    Sigma_t = np.eye(2)
    epsilon_d = np.random.normal(size=(cld_instance.d,))
    grad = cld_instance.compute_grad_v_t_log_p_t(Sigma_t, epsilon_d)
    assert grad.shape == (cld_instance.d,)
    assert np.all(np.isfinite(grad))


def test_compute_dsm_grad_u_t_log_p_t(cld_instance):
    u_0 = np.ones(2 * cld_instance.d)
    t = 0.5
    epsilon_2d = jax.random.normal(
        key=jax.random.PRNGKey(0), shape=(2 * cld_instance.d,)
    )
    grad = cld_instance.compute_dsm_grad_u_t_log_p_t(u_0, t, epsilon_2d)
    assert grad.shape == (2 * cld_instance.d,)
    assert np.all(np.isfinite(grad))


def test_compute_hsm_grad_u_t_log_p_t(cld_instance):
    x_0 = np.ones(cld_instance.d)
    t = 0.5
    epsilon_2d = jax.random.normal(
        key=jax.random.PRNGKey(0), shape=(2 * cld_instance.d,)
    )
    grad = cld_instance.compute_hsm_grad_u_t_log_p_t(x_0, t, epsilon_2d)
    assert grad.shape == (2 * cld_instance.d,)
    assert np.all(np.isfinite(grad))


def test_compute_dsm_grad_v_t_log_p_t(cld_instance):
    u_t = np.zeros(2 * cld_instance.d)
    u_0 = np.ones(2 * cld_instance.d)
    t = 0.5
    epsilon_d = jax.random.normal(key=jax.random.PRNGKey(0), shape=(cld_instance.d,))
    grad = cld_instance.compute_dsm_grad_v_t_log_p_t(u_t, u_0, t, epsilon_d)
    assert grad.shape == (cld_instance.d,)
    assert np.all(np.isfinite(grad))


def test_compute_hsm_grad_v_t_log_p_t(cld_instance):
    x_0 = np.ones(cld_instance.d)
    t = 0.5
    epsilon_d = jax.random.normal(key=jax.random.PRNGKey(0), shape=(cld_instance.d,))
    grad = cld_instance.compute_hsm_grad_v_t_log_p_t(x_0, t, epsilon_d)
    assert grad.shape == (cld_instance.d,)
    assert np.all(np.isfinite(grad))


def test_compute_l_t(cld_instance):
    Sigma_t = cld_instance.covariance(1.0, Sigma_0_xx=0, Sigma_0_vv=0)
    l_t = cld_instance.compute_l_t(Sigma_t)
    assert np.isscalar(l_t)
    assert np.isfinite(l_t)


def test_compute_L_t_inv_T(cld_instance):
    Sigma_t = np.array([[1.0, 0.5], [0.5, 2.0]])
    L_t_inv_T = cld_instance.compute_L_t_inv_T(Sigma_t)
    assert L_t_inv_T.shape == (2, 2)
    assert np.all(np.isfinite(L_t_inv_T))


def test_invalid_parameters():
    with pytest.raises(ValueError):
        CriticallyDampedLangevinDiffusion(d=-1, M=1.0, beta=1.0)
    with pytest.raises(ValueError):
        CriticallyDampedLangevinDiffusion(d=10, M=-1.0, beta=1.0)
    with pytest.raises(ValueError):
        CriticallyDampedLangevinDiffusion(d=10, M=1.0, beta=-1.0)


def test_jit_compilation():
    cld = CriticallyDampedLangevinDiffusion(d=10, M=1.0, beta=1.0)
    jitted_cld = eqx.filter_jit(lambda: cld)()
    assert isinstance(jitted_cld, CriticallyDampedLangevinDiffusion)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
