import os

import pytest
import chex
import jax.numpy as jnp
from jax import random
from thermoai.distributions import sample_from_superposition, gaussian_pdf

@pytest.fixture
def distribution_params():
    return {
        'mu1': jnp.array([1, 1]),
        'sigma1': jnp.array([[0.05, 0.01], [0.01, 0.25]]),
        'mu2': jnp.array([-1, 0]),
        'sigma2': jnp.array([[0.1, -0.03], [-0.03, 0.1]]),
        'p1': 0.5,
        'n_samples': 1_000_000  # Increased for better statistical accuracy
    }

def test_sample_from_superposition(distribution_params):
    key = random.PRNGKey(0)
    samples = sample_from_superposition(
        distribution_params['mu1'],
        distribution_params['sigma1'],
        distribution_params['mu2'],
        distribution_params['sigma2'],
        distribution_params['p1'],
        distribution_params['n_samples'],
        key
    )

    # Check the shape of the output
    assert samples.shape == (distribution_params['n_samples'], 2)

    # Check that all values are finite
    chex.assert_tree_all_finite(samples)

    # Check that samples are within expected range (increased range)
    assert jnp.all((samples >= -5) & (samples <= 5))

    # Statistical checks
    mean = jnp.mean(samples, axis=0)
    expected_mean = distribution_params['p1'] * distribution_params['mu1'] + (1 - distribution_params['p1']) * distribution_params['mu2']
    chex.assert_trees_all_close(mean, expected_mean, atol=0.1)

    cov = jnp.cov(samples.T)
    p1 = distribution_params['p1']
    p2 = 1-p1
    S1 = distribution_params['sigma1']
    S2 = distribution_params['sigma2']
    m1 = distribution_params['mu1']
    m2 = distribution_params['mu2']
    expected_cov = p1 * S1 + p2 * S2 + p1*p2*jnp.outer(m1-m2, m1-m2)
    chex.assert_trees_all_close(cov, expected_cov, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])