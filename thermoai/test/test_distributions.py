import pytest
import jax.numpy as jnp
import jax
from jax import random
from thermoai.distributions import sample_mog, mog_pdf, mog_logpdf, mog_energy

@pytest.fixture
def distribution_params():
    return {
        'means': jnp.array([
            [1.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0]
        ]),
        'covariances': jnp.array([
            [[0.01, 0.01], [0.01, 0.4]],
            [[0.1, -0.03], [-0.03, 0.1]],
            [[0.2, 0.0], [0.0, 0.2]]
        ]),
        'weights': jnp.array([0.3, 0.3, 0.4]),
        'n_samples': 1_000_000  # Increased for better statistical accuracy
    }

@pytest.fixture
def key():
    return random.PRNGKey(0)

def test_sample_mog_shape(distribution_params, key):
    samples = jax.vmap(lambda k: sample_mog(k, 
                                            distribution_params['means'], 
                                            distribution_params['covariances'], 
                                            distribution_params['weights']))(
        random.split(key, distribution_params['n_samples'])
    )
    assert samples.shape == (distribution_params['n_samples'], 2)

def test_sample_mog_finite(distribution_params, key):
    samples = jax.vmap(lambda k: sample_mog(k, 
                                            distribution_params['means'], 
                                            distribution_params['covariances'], 
                                            distribution_params['weights']))(
        random.split(key, distribution_params['n_samples'])
    )
    assert jnp.all(jnp.isfinite(samples))

def test_sample_mog_range(distribution_params, key):
    samples = jax.vmap(lambda k: sample_mog(k, 
                                            distribution_params['means'], 
                                            distribution_params['covariances'], 
                                            distribution_params['weights']))(
        random.split(key, distribution_params['n_samples'])
    )
    assert jnp.all((samples >= -5) & (samples <= 5))

def test_sample_mog_mean(distribution_params, key):
    samples = jax.vmap(lambda k: sample_mog(k, 
                                            distribution_params['means'], 
                                            distribution_params['covariances'], 
                                            distribution_params['weights']))(
        random.split(key, distribution_params['n_samples'])
    )
    mean = jnp.mean(samples, axis=0)
    expected_mean = jnp.sum(distribution_params['weights'][:, None] * distribution_params['means'], axis=0)
    assert jnp.allclose(mean, expected_mean, atol=0.1)

def test_mog_pdf(distribution_params):
    x = jnp.array([0.0, 0.0])
    pdf_value = mog_pdf(x, 
                        distribution_params['means'], 
                        distribution_params['covariances'], 
                        distribution_params['weights'])
    # assert jnp.isscalar(pdf_value)
    assert pdf_value >= 0

def test_mog_energy(distribution_params):
    x = jnp.array([0.0, 0.0])
    energy_value = mog_energy(x, 
                              distribution_params['means'], 
                              distribution_params['covariances'], 
                              distribution_params['weights'])
    # assert jnp.isscalar(energy_value)
    assert energy_value >= 0

def test_pdf_logpdf_consistency(distribution_params):
    x = jnp.array([0.0, 0.0])
    pdf_value = mog_pdf(x, 
                        distribution_params['means'], 
                        distribution_params['covariances'], 
                        distribution_params['weights'])
    logpdf_value = mog_logpdf(x, 
                              distribution_params['means'], 
                              distribution_params['covariances'], 
                              distribution_params['weights'])
    assert jnp.allclose(jnp.log(pdf_value), logpdf_value, atol=1e-6)

def test_logpdf_energy_consistency(distribution_params):
    x = jnp.array([0.0, 0.0])
    logpdf_value = mog_logpdf(x, 
                              distribution_params['means'], 
                              distribution_params['covariances'], 
                              distribution_params['weights'])
    energy_value = mog_energy(x, 
                              distribution_params['means'], 
                              distribution_params['covariances'], 
                              distribution_params['weights'])
    assert jnp.allclose(-logpdf_value, energy_value, atol=1e-6)

@pytest.fixture
def distribution_params_1d():
    return {
        'means': jnp.array([1.0, -1.0, 0.0]),
        'covariances': jnp.array([0.5, 0.3, 0.2]),
        'weights': jnp.array([0.3, 0.3, 0.4]),
        'n_samples': 1_000_000
    }

def test_sample_mog_1d(distribution_params_1d, key):
    samples = jax.vmap(lambda k: sample_mog(k,
                                            distribution_params_1d['means'],
                                            distribution_params_1d['covariances'],
                                            distribution_params_1d['weights']))(
        random.split(key, distribution_params_1d['n_samples'])
    )
    
    assert samples.shape == (distribution_params_1d['n_samples'],)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all((samples >= -5) & (samples <= 5))
    
    mean = jnp.mean(samples)
    expected_mean = jnp.sum(distribution_params_1d['weights'] * distribution_params_1d['means'])
    assert jnp.allclose(mean, expected_mean, atol=0.1)
    
    # Test PDF, logPDF, and energy functions for 1D case
    x = jnp.array(0.0)
    pdf_value = mog_pdf(x, 
                        distribution_params_1d['means'], 
                        distribution_params_1d['covariances'], 
                        distribution_params_1d['weights'])
    logpdf_value = mog_logpdf(x, 
                              distribution_params_1d['means'], 
                              distribution_params_1d['covariances'], 
                              distribution_params_1d['weights'])
    energy_value = mog_energy(x, 
                              distribution_params_1d['means'], 
                              distribution_params_1d['covariances'], 
                              distribution_params_1d['weights'])
    
    assert pdf_value >= 0
    assert jnp.allclose(jnp.log(pdf_value), logpdf_value, atol=1e-6)
    assert jnp.allclose(-logpdf_value, energy_value, atol=1e-6)

if __name__ == "__main__":
    pytest.main(["-v", __file__])