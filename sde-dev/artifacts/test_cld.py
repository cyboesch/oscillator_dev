import pytest
import jax
import diffrax
import numpy as np
import equinox as eqx
from cld import CriticallyDampedLangevinDiffusion


@pytest.fixture
def cld_instance():
    return CriticallyDampedLangevinDiffusion(d=10, M=1.0, beta=1.0, T=1.0)
  

def test_initialization(cld_instance):
    assert cld_instance.d == 10
    assert cld_instance.M == 1.0
    assert cld_instance.beta == 1.0
    assert cld_instance.T == 1.0
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


def test_perturbation_kernel_dsm_shape(cld_instance):
    u_0 = np.zeros(2 * cld_instance.d)
    t = 0.5
    mean, cov = cld_instance.perturbation_kernel_dsm(u_0, t)
    assert mean.shape == (2 * cld_instance.d,)
    assert cov.shape == (2 * cld_instance.d, 2 * cld_instance.d)


def test_perturbation_kernel_hsm_shape(cld_instance):
    x_0 = np.zeros(cld_instance.d)
    t = 0.5
    mean, cov = cld_instance.perturbation_kernel_hsm(x_0, t)
    assert mean.shape == (2 * cld_instance.d,)
    assert cov.shape == (2 * cld_instance.d, 2 * cld_instance.d)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        CriticallyDampedLangevinDiffusion(d=-1, M=1.0, beta=1.0)
    with pytest.raises(ValueError):
        CriticallyDampedLangevinDiffusion(d=10, M=-1.0, beta=1.0)
    with pytest.raises(ValueError):
        CriticallyDampedLangevinDiffusion(d=10, M=1.0, beta=-1.0)
    with pytest.raises(ValueError):
        CriticallyDampedLangevinDiffusion(d=10, M=1.0, beta=1.0, T=-1.0)


def test_jit_compilation():
    cld = CriticallyDampedLangevinDiffusion(d=10, M=1.0, beta=1.0)
    jitted_cld = eqx.filter_jit(lambda: cld)()
    assert isinstance(jitted_cld, CriticallyDampedLangevinDiffusion)


if __name__ == "__main__":
    pytest.main(["-v", __file__])