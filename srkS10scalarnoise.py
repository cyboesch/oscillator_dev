import jax.numpy as jnp
from jax import jit, random, vmap, lax
from functools import partial

@partial(jit, static_argnums=(1, 2))
def srk_s10_scalar_noise_step(rng, f, L, x, t, dt, Q, alpha=1.0):
    
    # Increment
    db = jnp.sqrt(dt * Q) * random.normal(rng, shape=(1,))
    dbb = 1 / 2 * (db ** 2 - Q * dt)

    # Evaluate only once
    fx = f(x, t, **{'alpha': alpha})
    Lx = L(x, t)

    # Supporting values
    x2 = x + fx * dt
    tx2 = x2 + Lx * dbb / jnp.sqrt(dt)
    tx3 = x2 - Lx * dbb / jnp.sqrt(dt)

    # Evaluate the remaining values
    fx2 = f(x2, t + dt, **{'alpha': alpha})
    Lx2 = L(tx2, t + dt)
    Lx3 = L(tx3, t + dt)

    # Step
    x_next = x + (fx + fx2) * dt / 2 + Lx * db + jnp.sqrt(dt) / 2 * (Lx2 - Lx3)
    return x_next

def srk_s10_scalar_noise_solve(key, f, L, tspan, x0, Q=None, alpha=1.0): 
    if Q is None:
        Q = 1.0  # Standard Brownian motion

    steps = len(tspan)
    # x = jnp.zeros((steps, x0.shape[0]))
    # x = x.at[0].set(x0)
    @jit
    def scan_body(carry, t_next):
        t_cur, x_cur, carry_key = carry
        carry_key, subkey = random.split(carry_key)
        dt = t_next - t_cur 
        x_cur = srk_s10_scalar_noise_step(subkey, f, L, x_cur, t_cur, dt, Q, alpha)
        return (t_next, x_cur, carry_key), x_cur

    return lax.scan(scan_body, (tspan[0], x0, key), tspan[1:])[1]

    # for i in tqdm.tqdm(range(steps - 1)):
    #     key, subkey = random.split(key)
    #     dt = tspan[i + 1] - tspan[i]
    #     x = x.at[i + 1].set(srk_s10_scalar_noise_step(subkey, f, L, x[i], tspan[i], dt, Q, alpha))

    # return x

if __name__ == "__main__":
    from python.oscillator import drift, diffusion

    tspan = jnp.linspace(0, 20, int(20 / (2**-5)))
    x0s = jnp.array([[-2 - 0.2*(j+1), 0] for j in range(10)])
    # alpha=-2.0
    alpha = -1.0
    print(x0s.shape)
    # f_kwargs = {'alpha': 1.0}
    #%%
    keys = random.split(random.PRNGKey(0), 10)
    xs = vmap(srk_s10_scalar_noise_solve, (None, None, None, None, 0, None, None))(keys[0], drift, diffusion, tspan, x0s, 1.0, alpha)
    #%%
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for j in range(10):
        x = xs[j, :, 0]
        y = xs[j, :, 1]
        ax.plot(x, y, '-k', lw=0.5)
    ax.set_xlim(-4.4, 4.4)
    ax.set_ylim(-4.8, 10)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_aspect('equal')
  
    plt.tight_layout()
    plt.show()
    
    # save
    import os
    import datetime

    # Save the figure
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{datetime.datetime.now().strftime("%Y%m%d")}_srkS10_scalar_noise_test.png'))
    
    

# %%
