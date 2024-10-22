import equinox as eqx
import jax
import jax.numpy as jnp


class ResNet(eqx.Module):
    layers: list
    activation_fn: callable = eqx.static_field()
    n_hidden_layers: int
    x_input: bool
    z_input: bool

    def __init__(self, input_dim=2, index_dim=1, hidden_dim=64, n_hidden_layers=20):

        self.activation_fn = jax.nn.silu
        self.n_hidden_layers = n_hidden_layers

        in_dim = input_dim * 2 + index_dim
        out_dim = input_dim

        layers = [eqx.nn.Linear(in_dim, hidden_dim)]
        for _ in range(n_hidden_layers):
            layers.append(eqx.nn.Linear(hidden_dim + index_dim, hidden_dim))
        layers.append(eqx.nn.Linear(hidden_dim + index_dim, out_dim))

        self.layers = layers

    def _append_time(self, h, t):
        time_embedding = jnp.log(t)
        return jnp.concatenate([h, time_embedding.reshape(-1, 1)], axis=1)

    def __call__(self, u, t):
        h0 = self.layers[0](self._append_time(u, t))
        h = self.activation_fn(h0)

        # Original for loop:
        # for i in range(self.n_hidden_layers):
        #     h_new = self.layers[i + 1](self._append_time(h, t))
        #     h = self.activation_fn(h + h_new)

        # Rewritten using scan
        def body_fn(h, i):
            h_new = self.layers[i + 1](self._append_time(h, t))
            h = self.activation_fn(h + h_new)
            return h, None

        h, _ = jax.lax.scan(body_fn, h, jnp.arange(self.n_hidden_layers))

        return self.layers[-1](self._append_time(h, t))
