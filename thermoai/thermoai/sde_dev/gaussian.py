import argparse
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.scipy.stats import multivariate_normal as mvn


def get_eigs(dim, decay_exponent):
    return decay_exponent ** jnp.arange(dim)

@partial(jax.jit, static_argnums=(3,5))
def get_cov(rng, dim, decay_exponent, inverse=False, scale=1.0, no_rotation=False):
    if not no_rotation:
        Q = get_rotation(rng, dim)
    else:
        Q = jnp.eye(dim)
    eigs = get_decayed_eigvals(dim, decay_exponent, scale)
    if inverse:
        eigs = 1.0 / eigs
        Q = Q.T
    return (Q * eigs) @ Q.T


def get_rotation(rng, dim):
    return jnp.linalg.qr(jrnd.normal(rng, (dim, dim)))[0]

def get_decayed_eigvals(dim, decay_exponent, inverse=False):
    eigs = decay_exponent ** jnp.arange(dim)
    if inverse:
        return 1.0 / eigs
    else:
        return eigs


@jax.custom_vjp
@partial(jax.jit, static_argnums=(5, 6))
def logpdf(
    x,
    mean,
    decay_exponent,
    dist_rng,
    std_dev=1.0,
    no_rotation=False,
    unnormalized=False,
):
    # Remark: for the MALA Langevin proposal with step size tau, we set
    #   - decay_exponent = 1 (no decay)
    #   - no_rotation = True (Q = I)
    #   - std_dev = jnp.sqrt(2/tau)
    #   - mean = y + tau* grad_y_logdensity_target
    #   - unnormalized = True

    D = mean.shape[0]
    Q = jnp.where(no_rotation, jnp.eye(D), get_rotation(dist_rng, D))
    var = std_dev**2
    eigvals = var * get_decayed_eigvals(
        D, decay_exponent
    )  # S = Q * jnp.diag(eigvals) * Q.T
    QTx_m_mu = Q.T @ (x - mean)

    quad = QTx_m_mu.T @ (
        (1 / eigvals) * QTx_m_mu
    )  # (x-mu).T @ jnp.inv(S) @ (x-mu) = (x-mu).T @ (Q @ jnp.diag(eigvals) * Q.T) @ (x-mu)
    
    # if unnormalized:
    #     return -0.5 * quad
    # else:
    #     logdet_S = jnp.sum(jnp.log(eigvals))
    #     const = mean.shape[0] * jnp.log(2 * jnp.pi)
    #     return -0.5 * (quad + const + logdet_S)
    return jnp.where(unnormalized, 
                    -0.5 * quad, 
                    -0.5 * (quad + D*jnp.log(2 * jnp.pi) + np.sum(jnp.log(eigvals))))
    


def grad_x_logpdf(
    x,
    mean,
    decay_exponent,
    dist_rng,
    std_dev=1.0,
    no_rotation=False,
    unnormalized=False,
):
    D = mean.shape[0]
    Q = jnp.where(no_rotation, jnp.eye(D), get_rotation(dist_rng, D))

    eigvals = (std_dev**2) * get_decayed_eigvals(D, decay_exponent)
    S_inv = (Q * (1 / eigvals)) @ Q.T
    return -S_inv @ (x - mean)


def grad_mu_logpdf(x, mean, decay_exponent, dist_rng, std_dev=1.0, no_rotation=False, unnormalized=False):
    D = mean.shape[0]
    Q = jnp.where(no_rotation, jnp.eye(D), get_rotation(dist_rng, D))

    eigvals = (std_dev**2) * get_decayed_eigvals(D, decay_exponent)
    S_inv = (Q * (1 / eigvals)) @ Q.T
    return S_inv @ (x - mean)

def _logpdf_fwd(*args):
    return logpdf(*args), args


def _logpdf_bwd(args, g):
    gx_out = g * grad_x_logpdf(*args)
    gmean_out = g * grad_mu_logpdf(*args)
    return gx_out, gmean_out, None, None, None, None, None


logpdf.defvjp(_logpdf_fwd, _logpdf_bwd)


def sample(sample_rng, mu, decay_exponent, dist_rng, std_dev=1.0, no_rotation=False):
    z = jrnd.normal(sample_rng, mu.shape)
    D = mu.shape[0]
    Q = jnp.where(no_rotation, jnp.eye(D), get_rotation(dist_rng, D))
    sqrt_eigs = std_dev * (get_decayed_eigvals(D, decay_exponent) ** 0.5)
    L = Q * sqrt_eigs
    return L @ z + mu


# %%
def _test_logpdf_val_and_grad():

    # Generate a test case
    rng = jrnd.PRNGKey(0)
    dist_rng, sample_rng = jrnd.split(rng)
    decay_exponent = 0.99
    dim = 2
    mean = jnp.zeros(dim)
    cov_diag = decay_exponent ** jnp.arange(0, dim)
    Q = get_rotation(dist_rng, dim)
    cov = Q @ jnp.diag(cov_diag) @ Q.T

    x = sample(sample_rng, mu=mean, decay_exponent=decay_exponent, dist_rng=dist_rng)

    # Compute the log-pdf using scipy
    log_pdf_scipy = mvn.logpdf(x, mean=mean, cov=cov)
    log_pdf_jax = logpdf(x, mean, decay_exponent, dist_rng)

    # Test that the logpdfs are the same
    assert jnp.allclose(
        log_pdf_jax, log_pdf_scipy
    ), "Logpdfs do not match: {} vs {}".format(log_pdf_jax, log_pdf_scipy)

    # test that the gradients match
    from jax.test_util import check_grads

    # Checking gradient with respect to x
    check_grads(
        lambda x: logpdf(x, mean, decay_exponent, rng), (x,), order=1, modes=("rev")
    )
    # Checking gradient with respect to params.mean
    check_grads(
        lambda mean: logpdf(x, mean, decay_exponent, rng),
        (mean,),
        order=1,
        modes=("rev"),
    )


def _sample_and_save(args):
    mean = jnp.array(args.mean)
    decay_exponent = args.decay
    dist_rng = jrnd.PRNGKey(args.dist_seed)
    sample_rng = jrnd.PRNGKey(args.sample_seed)

    samples = jax.vmap(lambda rng: sample(rng, mean, decay_exponent, dist_rng))(
        jrnd.split(sample_rng, args.num_samples)
    )

    np.savez(
        args.samples_fp,
        samples=np.asarray(samples),
        mean=mean,
        decay_exponent=decay_exponent,
        dist_seed=args.dist_seed,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mean", nargs="+", type=float, default=[0.0, 0.0, 0.0], help="Mean value(s)"
    )
    parser.add_argument("--decay", type=float, default=0.99, help="Decay parameter")
    parser.add_argument(
        "--dist_seed", type=int, default=0, help="Random seed for distribution rotation"
    )
    parser.add_argument(
        "--sample_seed", type=int, default=1, help="Random seed for sampling"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--samples_fp", type=str, default="samples", help="File path to save samples"
    )
    parser.add_argument("--test", type=bool, default=True, help="Whether to run tests")

    args = parser.parse_args()
    if args.test:
        _test_logpdf_val_and_grad()
    else:
        _sample_and_save(args)
