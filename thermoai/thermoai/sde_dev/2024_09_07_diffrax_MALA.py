import pytest
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.scipy.stats import norm as Normal
import diffrax
from dist_sde import make_sampler_SDE
from diffrax_helpers import BlackjaxEquivalentMetropolisHastingsController, AuditableVirtualBrownianTree
from malareparam import langevin
from malareparam.mala import mala_step_z

jax.config.update("jax_enable_x64", True)

# Set random seed for reproducibility
key = jrnd.PRNGKey(42)

def run_diffrax_mala(loc):
    scale = jnp.array([1.])
    target_logdensity_fn = lambda x: Normal.logpdf(x, loc, scale)[0]

    y0 = jnp.array([0.0])
    t0, t1 = 0.0, 0.05  # Time range
    dt0 = 0.01  # Step size
    ts = jnp.array([dt0*i for i in range(1, int((t1-t0)/dt0)+1)])
    n_steps = len(ts)

    bm = AuditableVirtualBrownianTree(t0=t0, t1=t1, tol=1e-3, shape=(loc.shape[0],), key=key)
    Langevin_sde = make_sampler_SDE(target_logdensity_fn, loc.shape[0])
    args = (loc, scale)

    def _langevin_mean(x, score_x, step_size):
        return x + step_size * score_x

    def _langevin_std_dev(step_size):
        return jnp.sqrt(2 * step_size)

    def proposal_logdensity_fn(cur_x, prop_x, step_size):
        cur_grad = jax.grad(target_logdensity_fn)(cur_x)
        return langevin.logpdf(
            prop_x,
            _langevin_mean(cur_x, cur_grad, step_size),
            _langevin_std_dev(step_size),
        )

    mh = BlackjaxEquivalentMetropolisHastingsController(target_logdensity_fn, proposal_logdensity_fn, bm=bm)

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
        stepsize_controller=mh
    )

    target_val_and_grad_logpdf = jax.value_and_grad(target_logdensity_fn)

    def mala_sample(num_samples):
        def body_fn(carry, _):
            prev_state, prev_t = carry
            next_t = prev_t + dt0
            rng = bm.evaluate(prev_t, next_t, rng_only=True)
            prop_z = bm.evaluate(prev_t, next_t, rng_only=False)
            next_state, _, info = mala_step_z(prev_state, rng, prop_z, target_val_and_grad_logpdf, dt0)
            return (next_state, next_t), (next_state, next_t)

        return jax.lax.scan(body_fn, (init_state, t0), None, num_samples)[1]
    init_state = (y0, *target_val_and_grad_logpdf(y0))
    (malareparam_samples,_,_), _ = mala_sample(n_steps-1)
    assert jnp.allclose(malareparam_samples, solution.ys[:-1], atol=1e-9)
    
    return malareparam_samples, solution.ys[:-1]

@pytest.fixture
def sample_data():
    loc = jnp.array([4.])
    return run_diffrax_mala(loc)

@pytest.fixture
def gradient_data():
    loc = jnp.array([4.])
    return jax.jacrev(run_diffrax_mala)(loc)

def test_samples_are_identical(sample_data):
    malareparam_samples, diffrax_samples = sample_data
    assert jnp.allclose(malareparam_samples, diffrax_samples, atol=1e-9), "Samples are not identical"

def test_gradients_are_identical(gradient_data):
    mr_grads, diff_grads = gradient_data
    assert jnp.allclose(mr_grads, diff_grads, atol=1e-9), "Gradients are not identical"

if __name__ == "__main__":
    # pytest
    pytest.main(["-v", __file__])