import jax
import jax.numpy as jnp
import jax.random as jrnd
import optax
import equinox as eqx
from cld import CriticallyDampedLangevinDynamics
from resnet import ResNet  # Import ResNet

from utils.toy_data import get_diamond_sample

class ScoreModel(eqx.Module):
    """
    “Based on (i) and (ii), we parameterize sθ(ut, t) = −\`tαθ(ut, t) with αθ(ut, t) = \`t−1vt/Σtvv + α′θ(ut, t),” ([Dockhorn et al., 2022, p. 5]) ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=5&annotation=UFEVT4QU)) 
    """
    resnet: ResNet
    cld: CriticallyDampedLangevinDynamics

    def __call__(self, u_t, t):
        alpha_prime_theta = self.resnet(u_t, t)
        ell_t = self.cld.B(t)
        ell_t_inv = self.cld.L_t_inv_T(t)
        ell_t_inv_vt = ell_t_inv @ u_t
        vt = ell_t_inv_vt[..., :self.cld.dim]
        return -ell_t * vt + alpha_prime_theta
#%%

# Data
sample_x0 = get_diamond_sample()

# Hyperparameters
learning_rate = 1e-3
batch_size = 128
num_steps = 800000
loss_eps = 1e-5  # From config.loss_eps

# Initialize model and optimizer
key = jrnd.PRNGKey(0)
key_model, key_training = jrnd.split(key)
dim = 2  # Adjust based on data dimensionality

def compute_loss(key, score_model, x0, t, cld):
    """
    Compute the HSM loss for CLD.

    Args:
        model: The score model (eqx.Module).
        x0_batch: A batch of initial positions x0 (shape: [batch_size, dim]).
        t_batch: A batch of time samples (shape: [batch_size]).
        key: PRNG key.
        cld: An instance of the CriticallyDampedLangevinDynamics module.

    Returns:
        loss: The scalar loss value.
    """

    # Generate random epsilon
    epsilon = jrnd.normal(key, shape=(batch_size, dim * 2))

    # Compute mean and covariance for each sample in the batch
    mu_t, cov_t = jax.vmap(cld.get_hsm_kernel_params)(x0, t)

    # Simulate u_t via reparameterization: u_t = μ_t + L_t ε
    u_t = mu_t + jnp.einsum('bij,bj->bi', jnp.linalg.cholesky(cov_t), epsilon)

    # Extract ε_{d:2d} (noise in velocity component)
    epsilon_v = epsilon[:, dim:]  # Shape: [batch_size, dim]

    # Predict α_θ(u_t, t) using ResNet
    alpha_theta = score_model(u_t, t_batch)  # Shape: [batch_size, dim]

    # Compute loss
    loss = jnp.mean(jnp.sum((epsilon_v - alpha_theta) ** 2, axis=1))

    return loss


cld = CriticallyDampedLangevinDynamics(state_dim=2, M=0.25, beta=1.0, gamma=0.04)

score_model = ResNet(
    input_dim=dim,
    index_dim=1,
    hidden_dim=64,
    n_hidden_layers=20
)
# Pull out the parameters of the model for optimization
params = eqx.filter(score_model, eqx.is_inexact_array)

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Training step function
@eqx.filter_jit
def opt_step(model, x0_batch, t_batch, opt_state, key, cld):
    loss_value, grads = eqx.filter_value_and_grad(compute_loss)(model, x0_batch, t_batch, key, cld)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

# Training loop
for step in range(num_steps):
    key_training, subkey = jrnd.split(key_training)
    batch_keys = jrnd.split(key_training, batch_size)
    
    # Sample x0s from data distribution
    x0s = jax.vmap(sample_x0)(batch_keys)  # Define your data sampler
    
    ts = jrnd.uniform(subkey, shape=(batch_size,), minval=loss_eps, maxval=1.0 - loss_eps)

    model, opt_state, loss_value = opt_step(model, x0s, ts, opt_state, subkey, cld)

    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss_value}")
