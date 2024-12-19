from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.lax import stop_gradient
import langevin


def _langevin_mean(x, score_x, step_size):
    return x + step_size * score_x


def _langevin_std_dev(step_size):
    return jnp.sqrt(2 * step_size)


def sample_proposal(rng, state, step_size):
    cur_x, cur_val, cur_grad = state

    # Make the Langevin proposal.
    prop_x = langevin.sample(
        rng,
        mu=_langevin_mean(cur_x, cur_grad, step_size),
        std_dev=_langevin_std_dev(step_size),
    )

    return prop_x  # matches up to here

def sample_proposal_z(z, state, step_size):
    cur_x, _, cur_grad = state

    # Make the Langevin proposal.
    return cur_x + step_size * cur_grad + jnp.sqrt(2) * z


def sample_proposal_sg(rng, state, step_size):
    cur_x, cur_val, cur_grad = state

    # Make the Langevin proposal.
    prop_x = langevin.sample(
        rng,
        mu=_langevin_mean(cur_x, cur_grad, step_size),
        std_dev=_langevin_std_dev(step_size),
    )

    return prop_x

def compute_log_rho(prop_state, cur_state, step_size):
    prop_x, prop_x_unn_target_logdensity_val, prop_grad = prop_state
    cur_x, cur_x_unn_target_logdensity_val, cur_grad = cur_state

    cur_x_g_prop_x_proposal_logdensity_val = langevin.logpdf(
        cur_x,
        _langevin_mean(prop_x, prop_grad, step_size),
        _langevin_std_dev(step_size),
    )

    new_energy = -(
        cur_x_g_prop_x_proposal_logdensity_val + prop_x_unn_target_logdensity_val
    )

    prop_x_g_cur_x_proposal_logdensity_val = langevin.logpdf(
        prop_x, _langevin_mean(cur_x, cur_grad, step_size), _langevin_std_dev(step_size)
    )

    prev_energy = -(
        prop_x_g_cur_x_proposal_logdensity_val + cur_x_unn_target_logdensity_val
    )

    # Compute log accept ratio
    log_accept_ratio = prev_energy - new_energy
    
    # From BlackJAX: safe_energy_diff
    log_accept_ratio = jnp.where(jnp.isnan(log_accept_ratio), -jnp.inf, log_accept_ratio)
    return log_accept_ratio


def accept_or_reject(rng, prop_state, cur_state, step_size):
    prop_x, prop_val, prop_grad = prop_state
    cur_x, cur_val, cur_grad = cur_state

    # Compute the log acceptance ratio
    log_accept_ratio = compute_log_rho(prop_state, cur_state, step_size)

    # Do accept/reject step
    # p_accept = jnp.clip(jnp.exp(log_accept_ratio), max=1)
    
    is_accepted = jnp.log(jrnd.uniform(rng)) < log_accept_ratio
    # is_accepted = jrnd.bernoulli(rng, p_accept)
    next_state = (
        jnp.where(is_accepted, prop_x, cur_x),  # next_x
        jnp.where(is_accepted, prop_val, cur_val),  # next_val
        jnp.where(is_accepted, prop_grad, cur_grad),  # next_grad
    )

    info = (log_accept_ratio, cur_x, is_accepted)
    return next_state, info


# @partial(jax.jit, static_argnums=(2,))
def mala_step(cur_state, rng, target_val_and_grad_logpdf, step_size):
    prop_rng, accept_rng = jrnd.split(rng)

    # Compute the proposed state.
    prop_x = sample_proposal(prop_rng, cur_state, step_size)  # Integrator
    prop_val, prop_grad = target_val_and_grad_logpdf(prop_x)
    prop_state = (prop_x, prop_val, prop_grad)

    # Accept or reject the proposed state.
    next_state, info = accept_or_reject(accept_rng, prop_state, cur_state, step_size)

    return next_state, prop_x, info

# @partial(jax.jit, static_argnums=(2,))
def mala_step_z(cur_state, rng, prop_z, target_val_and_grad_logpdf, step_size):
    # Helper function for comparison with diffrax
    _, accept_rng = jrnd.split(rng)
    cur_x, _, cur_grad = cur_state

    # Compute the proposed state.
    prop_x = cur_x + step_size * cur_grad + jnp.sqrt(2) * prop_z
    prop_val, prop_grad = target_val_and_grad_logpdf(prop_x)
    prop_state = (prop_x, prop_val, prop_grad)

    # Accept or reject the proposed state.
    next_state, info = accept_or_reject(accept_rng, prop_state, cur_state, step_size)

    return next_state, prop_x, info


def sample(
    init_rng, target_logdensity_fn, init_x, num_steps, step_size, return_proposals=False
):

    target_val_and_grad_logpdf = jax.value_and_grad(target_logdensity_fn)

    def scan_body(carry, _):
        state, rng = carry
        rng, step_rng = jrnd.split(rng)
        (next_state, prop_state, info) = mala_step(
            state, step_rng, target_val_and_grad_logpdf, step_size
        )
        return (next_state, rng), (next_state, prop_state, info)

    init_val, init_grad = target_val_and_grad_logpdf(init_x)
    init_state = (init_x, init_val, init_grad)

    _, ((xs, _, _), prop_xs, info) = jax.lax.scan(
        scan_body, (init_state, init_rng), None, length=num_steps
    )

    if return_proposals:
        return xs, prop_xs, info
    else:
        return xs, info


def sample_with_stop_grad_every_n(
    init_rng,
    target_logdensity_fn,
    init_x,
    num_steps,
    step_size,
    stop_grads_every_nth,
    return_proposals=False,
):
    # This function samples using MALA, but every exact_n steps, it samples exactly from the distribution.

    value_and_grad_logpdf = jax.value_and_grad(target_logdensity_fn)

    def scan_body(carry, _):
        state, rng = carry
        rng, step_rng = jrnd.split(rng)
        (next_state, prop_state, info) = mala_step(
            state, step_rng, value_and_grad_logpdf, step_size
        )
        return (next_state, rng), (next_state, prop_state, info)

    init_val, init_grad = value_and_grad_logpdf(init_x)
    init_state = (init_x, init_val, init_grad)

    _, ((xs, _, _), prop_xs, info) = jax.lax.scan(
        scan_body, (init_state, init_rng), None, length=num_steps
    )

    (log_accept_ratio, cur_x, is_accepted) = info

    # --- Stop grads every nth step
    xs_sg = jnp.where(
        jnp.arange(num_steps) % stop_grads_every_nth == 0, xs, stop_gradient(xs)
    )
    prop_xs_sg = jnp.where(
        jnp.arange(num_steps) % stop_grads_every_nth == 0,
        prop_xs,
        stop_gradient(prop_xs),
    )
    log_accept_ratio_sg = jnp.where(
        jnp.arange(num_steps) % stop_grads_every_nth == 0,
        log_accept_ratio,
        stop_gradient(log_accept_ratio),
    )
    cur_x_sg = jnp.where(
        jnp.arange(num_steps) % stop_grads_every_nth == 0, cur_x, stop_gradient(cur_x)
    )

    info_sg = (log_accept_ratio_sg, cur_x_sg, is_accepted)

    if return_proposals:
        return xs_sg, prop_xs_sg, info_sg
    else:
        return xs_sg, info_sg


def sample_with_exact_every_n(
    init_rng,
    target_logdensity_fn,
    exact_sample_fn,
    init_x,
    num_steps,
    step_size,
    exact_every_nth,
    return_proposals=False,
):
    # This function samples using MALA, but every exact_n steps, it samples exactly from the distribution.
    value_and_grad_logpdf = jax.value_and_grad(target_logdensity_fn)
    init_state_fn = lambda x, rng: ((x, *value_and_grad_logpdf(x)), rng)

    # --- Each exact sample initializes its own chain of length `exact_every_nth`
    num_exact_init_chains = (
        num_steps // exact_every_nth
    ) - 1  # -1 because we start with init_x
    total_num_chains = num_exact_init_chains + 1

    # --- Use provided function to do exact samples
    rng, exact_rng = jax.random.split(init_rng)
    exact_samples = jax.vmap(exact_sample_fn)(
        jax.random.split(exact_rng, num_exact_init_chains)
    )

    def mala_scan_step(carry, _):
        state, rng = carry
        rng, step_rng = jrnd.split(rng)
        (next_state, prop_state, info) = mala_step(
            state, step_rng, value_and_grad_logpdf, step_size
        )
        return (next_state, rng), (next_state, prop_state, info)

    init_states = jnp.concatenate([init_x[None, :], exact_samples])
    init_rngs = jax.random.split(rng, total_num_chains)

    run_mala = lambda init_x, init_rng: jax.lax.scan(
        mala_scan_step, init_state_fn(init_x, init_rng), None, length=exact_every_nth
    )

    _, ((xs, _, _), prop_xs, info) = jax.vmap(run_mala)(init_states, init_rngs)

    if return_proposals:
        return xs, prop_xs, info
    else:
        return xs, info


def sample_with_exacts_randomly(
    init_rng,
    target_logdensity_fn,
    exact_sample_fn,
    init_x,
    num_steps,
    step_size,
    return_proposals=False,
    exact_sample_percent=0.1,
):

    value_and_grad_logpdf = jax.value_and_grad(target_logdensity_fn)

    def scan_body_with_exacts(carry, _):
        state, rng = carry
        cur_x, _, _ = state
        rng, step_rng, exact_rng = jrnd.split(rng, 3)
        should_sample_exact = jrnd.bernoulli(exact_rng, exact_sample_percent)
        exact_sample = exact_sample_fn(exact_rng)
        exact_val, exact_grad = value_and_grad_logpdf(exact_sample)
        (
            (mala_x, mala_val, mala_grad),
            mala_prop_x,
            (mala_log_accept_ratio, _, mala_is_accepted),
        ) = mala_step(state, step_rng, value_and_grad_logpdf, step_size)
        next_x = jnp.where(should_sample_exact, exact_sample, mala_x)
        next_val = jnp.where(should_sample_exact, exact_val, mala_val)
        next_grad = jnp.where(should_sample_exact, exact_grad, mala_grad)
        log_accept_ratio = jnp.where(should_sample_exact, 0.0, mala_log_accept_ratio)
        is_accepted = jnp.where(should_sample_exact, True, mala_is_accepted)
        info = (log_accept_ratio, cur_x, is_accepted)
        prop_x = jnp.where(should_sample_exact, exact_sample, mala_prop_x)
        next_state = (next_x, next_val, next_grad)
        return (next_state, rng), (next_state, prop_x, info)

    init_val, init_grad = value_and_grad_logpdf(init_x)
    init_state = (init_x, init_val, init_grad)

    _, ((xs, _, _), prop_xs, info) = jax.lax.scan(
        scan_body_with_exacts, (init_state, init_rng), None, length=num_steps
    )

    if return_proposals:
        return xs, prop_xs, info
    else:
        return xs, info


if __name__ == "__main__":
    import jax.random as jrnd
    from .gaussian import logpdf

    rng = jrnd.PRNGKey(0)
    dist_rng, sample_rng = jrnd.split(rng)
    decay_exponent = 0.99
    dim = 4
    mean = jnp.zeros(dim)
    # %%
    logpdf_x = lambda x: logpdf(
        x, mean, decay_exponent, dist_rng, scale=1, unnormalized=True
    )
    init_x = jnp.zeros(dim)

    states, (compute_log_rho, cur_x, is_accepted) = sample(
        rng, logpdf_x, init_x, 5, 0.01
    )
    # %%
    states2, (log_accept_ratio2, cur_x, is_accepted2) = sample(
        rng, logpdf_x, init_x, 5, 0.01
    )

    states3, prop_states, (log_accept_ratio3, cur_x3, is_accepted3) = sample(
        rng, logpdf_x, init_x, 5, 0.01, return_proposals=True
    )

    # %%
    # assert they are the same
    assert jnp.all(states == states2)
    assert jnp.all(is_accepted == is_accepted2)
    assert jnp.all(compute_log_rho == log_accept_ratio2)
    assert jnp.all(states == states3)
    assert jnp.all(is_accepted == is_accepted3)
    assert jnp.all(compute_log_rho == log_accept_ratio3)
    # %%
    is_accepteds = []
    states_list = []
    cur_xs = []
    log_accept_ratios = []
    prop_states_list = []

    for num_steps in range(1, 6):
        states, prop_state, (compute_log_rho, cur_x, is_accepted) = sample(
            rng, logpdf_x, init_x, num_steps, 0.05, return_proposals=True
        )
        is_accepteds.append(is_accepted)
        states_list.append(states)
        cur_xs.append(cur_x)
        log_accept_ratios.append(compute_log_rho)
        prop_states_list.append(prop_state)
