import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, lax
import tqdm

@partial(jit, static_argnums=(0))
def rk4_step(f, x_prev, t_prev, dt, f_kwargs):

        # Stages
        dx1 = f(x_prev, t_prev, **f_kwargs) * dt
        dx2 = f(x_prev + dx1 / 2, t_prev + dt / 2, **f_kwargs) * dt
        dx3 = f(x_prev + dx2 / 2, t_prev + dt / 2, **f_kwargs) * dt
        dx4 = f(x_prev + dx3, t_prev + dt, **f_kwargs) * dt

        # Step
        return x_prev + (1/6)*(dx1 + 2 * dx2 + 2 * dx3 + dx4)

def rk4_solve(f, tspan, x0, f_kwargs={}):
    x = jnp.zeros((len(tspan), x0.shape[0]))
    x = x.at[0].set(x0)
    
    def _rk4_step(carry, t_cur):
        t_prev, x_prev = carry
        x_cur = rk4_step(f, x_prev, t_prev, t_cur-t_prev, f_kwargs)
        return (t_cur, x_cur), x_cur

    return lax.scan(_rk4_step, (tspan[0], x0), tspan[1:])[1]


if __name__ == "__main__":
    from oscillator import drift

    tspan = jnp.linspace(0, 20, int(20 / (2**-5)))
    x0 = jnp.array([-2 - 0.2 * 1j, 0])
    x0s = jnp.array([[-2 - 0.2*(j+1), 0] for j in range(10)])
    print(x0s.shape)
    xs = vmap(rk4_solve, (None, None, 0))(drift, tspan, x0s)
    print(xs.shape)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for j in range(10):
        x = xs[j, :, 0]
        y = xs[j, :, 1]
        u = drift(jnp.stack([x, y]), tspan)
        u_x = u[0]
        u_y = u[1]
        ax.quiver(x[15::25], y[15::25], u_x[15::25], u_y[15::25], angles='xy', scale_units='xy', width=0.005, headlength=5, headaxislength=5, alpha=0.5, scale=20)
        ax.plot(x, y, '-k', lw=0.5)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


# %%
