import pytest
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.scipy.stats import norm as Normal
import diffrax
from diffrax_mala_helpers import (
    make_sampler_SDE,
    BlackjaxEquivalentMetropolisHastingsController,
    AuditableUnsafeBrownianPath,
)
from malareparam import langevin
from diffrax import UnsafeBrownianPath
import blackjax
import numpy as np

jax.config.update("jax_enable_x64", True)

@pytest.fixture
def setup_parameters():
    key = jrnd.PRNGKey(42)
    loc = jnp.array([4])
    scale = jnp.array([1])
    target_logdensity_fn = lambda x: Normal.logpdf(x, loc, scale)[0]
    y0 = jnp.array([0.0])
    t0, t1 = 0.0, 1.0
    dt0 = 0.1
    ts = jnp.array([dt0 * i for i in range(1, int((t1 - t0) / dt0) + 1)])
    n_steps = len(ts)
    bm = AuditableUnsafeBrownianPath(shape=(loc.shape[0],), key=key)
    Langevin_sde = make_sampler_SDE(target_logdensity_fn, loc.shape[0])
    args = (loc, scale)
    return (
        key,
        loc,
        scale,
        target_logdensity_fn,
        y0,
        t0,
        t1,
        dt0,
        ts,
        n_steps,
        bm,
        Langevin_sde,
        args,
    )

def _langevin_mean(x, score_x, step_size):
    return x + step_size * score_x

def _langevin_std_dev(step_size):
    return jnp.sqrt(2 * step_size)

def proposal_logdensity_fn(cur_x, prop_x, step_size, target_logdensity_fn):
    cur_grad = jax.grad(target_logdensity_fn)(cur_x)
    return langevin.logpdf(
        prop_x,
        _langevin_mean(cur_x, cur_grad, step_size),
        _langevin_std_dev(step_size),
    )

def test_mala_diffrax_equals_blackjax(setup_parameters):
    (
        key,
        loc,
        scale,
        target_logdensity_fn,
        y0,
        t0,
        t1,
        dt0,
        ts,
        n_steps,
        bm,
        Langevin_sde,
        args,
    ) = setup_parameters

    mh = BlackjaxEquivalentMetropolisHastingsController(
        target_logdensity_fn,
        lambda cur_x, prop_x, step_size: proposal_logdensity_fn(
            cur_x, prop_x, step_size, target_logdensity_fn
        ),
        bm=bm,
    )

    solution = diffrax.diffeqsolve(
        terms=Langevin_sde.get_terms(bm),
        solver=diffrax.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        adjoint=diffrax.DirectAdjoint(),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=mh,
    )

    blackjax_mala = blackjax.mala(target_logdensity_fn, dt0)
    blackjax_mala_step = blackjax_mala.step
    init_state = blackjax_mala.init(y0)

    def mala_sample(num_samples):
        def body_fn(carry, _):
            prev_state, prev_t = carry
            next_t = prev_t + dt0
            rng = bm.evaluate(prev_t, next_t, rng_only=True)
            next_state, info = blackjax_mala_step(rng, prev_state)
            return (next_state, next_t), (next_state, next_t)

        return jax.lax.scan(body_fn, (init_state, t0), None, num_samples)[1]

    blackjax_samples, _ = mala_sample(n_steps)
    assert jnp.allclose(solution.ys, blackjax_samples.position, atol=1e-9)

def test_mala_convergence(setup_parameters):
    (
        key,
        loc,
        scale,
        target_logdensity_fn,
        y0,
        t0,
        t1,
        dt0,
        ts,
        n_steps,
        bm,
        Langevin_sde,
        args,
    ) = setup_parameters

    # Increase the number of steps for better convergence
    n_steps = 10000
    ts = jnp.linspace(t0, t1, n_steps)

    mh = BlackjaxEquivalentMetropolisHastingsController(
        target_logdensity_fn,
        lambda cur_x, prop_x, step_size: proposal_logdensity_fn(
            cur_x, prop_x, step_size, target_logdensity_fn
        ),
        bm=bm,
    )

    solution = diffrax.diffeqsolve(
        terms=Langevin_sde.get_terms(bm),
        solver=diffrax.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        adjoint=diffrax.DirectAdjoint(),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=mh,
    )

    samples = solution.ys.squeeze()
    
    # Calculate running means
    running_means = jnp.cumsum(samples) / jnp.arange(1, n_steps + 1)
    
    # Check if the final mean is close to the true mean
    true_mean = loc[0]
    final_mean = running_means[-1]
    
    assert jnp.abs(final_mean - true_mean) < 0.1, f"Final mean {final_mean} is not close enough to true mean {true_mean}"
    
    # Check if the running means converge
    last_thousand_means = running_means[-1000:]
    mean_diff = jnp.max(last_thousand_means) - jnp.min(last_thousand_means)
    
    assert mean_diff < 0.05, f"Running means have not converged. Difference in last 1000 means: {mean_diff}"

def test_blackjax_mala_convergence(setup_parameters):
    (
        key,
        loc,
        scale,
        target_logdensity_fn,
        y0,
        t0,
        t1,
        dt0,
        ts,
        n_steps,
        bm,
        Langevin_sde,
        args,
    ) = setup_parameters

    # Increase the number of steps for better convergence
    n_steps = 10000

    blackjax_mala = blackjax.mala(target_logdensity_fn, dt0)
    blackjax_mala_step = blackjax_mala.step
    init_state = blackjax_mala.init(y0)

    def mala_sample(num_samples):
        @jax.jit
        def body_fn(carry, _):
            prev_state, prev_t, carry_rng = carry
            carry_rng, sample_rng = jrnd.split(carry_rng)
            next_t = prev_t + dt0
            # rng = bm.evaluate(prev_t, next_t, rng_only=True)
            next_state, info = blackjax_mala_step(sample_rng, prev_state)
            return (next_state, next_t, sample_rng), (next_state, next_t)

        return jax.lax.scan(body_fn, (init_state, t0, key), None, num_samples)[1]

    blackjax_samples, _ = mala_sample(n_steps)
    samples = blackjax_samples.position.squeeze()

    # Calculate running means
    running_means = jnp.cumsum(samples) / jnp.arange(1, n_steps + 1)

    # Check if the final mean is close to the true mean
    true_mean = loc[0]
    final_mean = running_means[-1]

    assert jnp.abs(final_mean - true_mean) < 0.1, f"Final mean {final_mean} is not close enough to true mean {true_mean}"

    # Check if the running means converge
    last_thousand_means = running_means[-1000:]
    mean_diff = jnp.max(last_thousand_means) - jnp.min(last_thousand_means)

    assert mean_diff < 0.05, f"Running means have not converged. Difference in last 1000 means: {mean_diff}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
