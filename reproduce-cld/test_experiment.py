import jax
import jax.numpy as jnp
import jax.random as random
import optax
import equinox as eqx

# Import energy functions from oscillators_new.py
from oscillators_new import energy_network

# Import CTMC helper functions from ctmc.py
from ctmc import create_ctmc_from_logdensity

# Define the score model
class ScoreModel(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, input_dim, output_dim, hidden_dim=64, num_layers=2):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.relu,
            final_activation=lambda x: x,  # Identity function
            key=key,
        )

    def __call__(self, x, t):
        # x: shape (data_dim,)
        # t: scalar
        t_expanded = jnp.array([t])  # Shape (1,)
        x_input = jnp.hstack([x, t_expanded])
        return self.mlp(x_input)  # Output shape: (data_dim,)

# Define the loss function for a single data point
def loss_fn_single(score_model, x0, t, key):
    """
    Computes the loss for a single data point.

    Args:
        score_model: The score model to be trained.
        x0: Single data sample from the dataset (shape: (data_dim,))
        t: Scalar time value.
        key: PRNG key.

    Returns:
        Scalar loss value.
    """
    # Sample Gaussian noise
    epsilon = random.normal(key, shape=x0.shape)  # Shape: (data_dim,)

    # Compute alpha_bar_t
    alpha_bar_t = jnp.exp(-t)

    # Generate noisy data x_t
    sqrt_alpha_bar = jnp.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = jnp.sqrt(1 - alpha_bar_t)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon  # Shape: (data_dim,)

    # Score model estimation
    score_estimate = score_model(x_t, t)  # Shape: (data_dim,)

    # Compute the loss
    # The true score is -(epsilon) / sqrt(1 - alpha_bar_t)
    true_score = -epsilon / sqrt_one_minus_alpha_bar
    loss = jnp.sum((score_estimate - true_score) ** 2)  # Scalar

    return loss

# Vectorized loss function over batches using vmap
def loss_fn(score_model, x0_batch, t_batch, key_batch):
    """
    Computes the loss over a batch of data.

    Args:
        score_model: The score model to be trained.
        x0_batch: Batch of data samples (shape: (batch_size, data_dim))
        t_batch: Batch of time samples (shape: (batch_size,))
        key_batch: Batch of PRNG keys (shape: (batch_size, 2))  # PRNGKeyArray is size 2

    Returns:
        Scalar loss value.
    """
    # Map over the batch using vmap
    losses = jax.vmap(loss_fn_single, in_axes=(None, 0, 0, 0))(
        score_model, x0_batch, t_batch, key_batch
    )  # Shape: (batch_size,)

    # Return the mean loss over the batch
    return jnp.mean(losses)

# Training procedure
def train_score_model(
    key,
    score_model,
    optimizer,
    x0_data,
    num_steps=10000,
    batch_size=128,
):
    opt_state = optimizer.init(eqx.filter(score_model, eqx.is_inexact_array))

    @eqx.filter_jit
    def make_step(score_model, opt_state, x0_batch, t_batch, key_batch):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
            score_model, x0_batch, t_batch, key_batch
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        score_model = eqx.apply_updates(score_model, updates)
        return loss_value, score_model, opt_state

    for step in range(num_steps):
        # Sample a batch of data indices
        key, subkey = random.split(key)
        idx = random.choice(subkey, x0_data.shape[0], (batch_size,), replace=False)
        x0_batch = x0_data[idx]  # Shape: (batch_size, data_dim)

        # Sample times uniformly in [0, 1]
        key, subkey = random.split(key)
        t_batch = random.uniform(subkey, shape=(batch_size,))  # Shape: (batch_size,)

        # Generate a batch of PRNG keys for vmap
        key, *subkeys = random.split(key, num=batch_size + 1)
        key_batch = jnp.stack(subkeys)  # Shape: (batch_size, 2)

        # Perform a training step
        loss_value, score_model, opt_state = make_step(
            score_model, opt_state, x0_batch, t_batch, key_batch
        )

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss_value.item()}")

    return score_model

# Sampling procedure for a single data point
def sample_single(key, score_model, num_steps=1000, dt=1e-3, x_shape=(2,)):
    """
    Generates a single sample using Langevin dynamics.

    Args:
        key: PRNG key.
        score_model: Trained score model.
        num_steps: Number of time steps.
        dt: Time step size.
        x_shape: Shape of the data.

    Returns:
        Generated sample (shape: x_shape)
    """
    # Start from standard Gaussian noise
    key, subkey = random.split(key)
    x_t = random.normal(subkey, shape=x_shape)  # Shape: x_shape

    # Time discretization
    t_values = jnp.linspace(1.0, 0.0, num_steps)  # Shape: (num_steps,)

    def body_fun(i, x_t):
        t = t_values[i]
        key_i = random.fold_in(key, i)
        key_i, subkey = random.split(key_i)
        z = random.normal(subkey, shape=x_t.shape)
        score = score_model(x_t, t)
        x_t = x_t + 0.5 * dt * score + jnp.sqrt(dt) * z
        return x_t

    x_t = jax.lax.fori_loop(0, num_steps, body_fun, x_t)
    return x_t

# Vectorized sampling procedure over batch using vmap
def sample_with_score_model(
    key,
    score_model,
    num_samples=1000,
    num_steps=1000,
    dt=1e-3,
    x_shape=(2,),
):
    # Generate a batch of PRNG keys
    key, *subkeys = random.split(key, num=num_samples + 1)
    keys = jnp.stack(subkeys)  # Shape: (num_samples, 2)

    # Vectorize the sampling over the batch
    samples = jax.vmap(sample_single, in_axes=(0, None, None, None, None))(
        keys, score_model, num_steps, dt, x_shape
    )  # Shape: (num_samples, *x_shape)

    return samples

if __name__ == "__main__":
    # Set random seed
    key = random.PRNGKey(0)

    # Generate synthetic dataset using the energy function
    def sample_data(key, num_samples=10000):
        """
        Samples data from the target distribution defined by the oscillator network.
        """
        # For simplicity, we'll sample from a Gaussian approximation
        mean = jnp.zeros(2)
        cov = jnp.eye(2) * 0.5

        key, subkey = random.split(key)
        samples = random.multivariate_normal(subkey, mean, cov, (num_samples,))
        return samples

    # Create dataset
    num_samples = 10000
    x0_data = sample_data(key, num_samples)

    # Initialize the score model
    key, model_key = random.split(key)
    input_dim = x0_data.shape[1] + 1  # Including time t
    output_dim = x0_data.shape[1]     # Output dimension matches data dimension
    score_model = ScoreModel(
        model_key,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=64,
        num_layers=4
    )

    # Define optimizer
    optimizer = optax.adam(learning_rate=1e-3)

    # Train the score model
    score_model = train_score_model(key, score_model, optimizer, x0_data)

    # Generate samples using the trained score model
    key, subkey = random.split(key)
    generated_samples = sample_with_score_model(
        subkey, score_model, num_samples=1000, num_steps=1000, dt=1e-3, x_shape=(2,)
    )

    # Plot results (optional, requires matplotlib)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Data")
    plt.scatter(x0_data[:1000, 0], x0_data[:1000, 1], alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.title("Generated Samples")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5)
    plt.show()