import jax.random as jrnd
import gaussian 

# --- These parameters define an Isotropic Gaussian distribution 
isotropic_config = {
    "decay_exponent": 1.0,  # Isotropic Gaussian with no decay
    "no_rotation": True,  # No random rotation (Isotropic)
    "dist_rng": jrnd.PRNGKey(0),  # not used
}

sample = lambda rng, mu, std_dev: gaussian.sample(
    rng, mu, std_dev=std_dev, **isotropic_config
)

logpdf = lambda x, mu, std_dev: gaussian.logpdf(
    x, mu, std_dev=std_dev, unnormalized=True, **isotropic_config
)