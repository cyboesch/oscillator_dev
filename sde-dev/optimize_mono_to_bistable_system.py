import os
import argparse
import datetime
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import logging
import optax
from tqdm.auto import tqdm

from oscillator import drift, diffusion
from jax import vmap, random, value_and_grad, jit
from srkS10scalarnoise import srk_s10_scalar_noise_solve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tstart", type=float, default=0.0, help="Start time for simulation"
    )
    parser.add_argument(
        "--tfinal", type=float, default=20.0, help="Final time for simulation"
    )
    parser.add_argument(
        "--stepsize", type=float, default=2**-5, help="Step size for simulation"
    )
    parser.add_argument(
        "--alpha_target",
        type=float,
        default=1.0,
        help="Target alpha value for optimization",
    )
    parser.add_argument(
        "--alpha_init",
        type=float,
        default=-1.0,
        help="Initial alpha value for optimization",
    )
    parser.add_argument(
        "--state_init",
        type=float,
        nargs=2,
        default=[-2.0, 1.0],
        help="Initial state for simulation",
    )
    parser.add_argument(
        "--adam_lr", type=float, default=0.01, help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--sim_seed", type=int, default=42, help="Random seed for simulation"
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=50,
        help="Print frequency for optimization iterations",
    )
    return parser.parse_args()


def loss_fn(key, test_alpha, tspan, target_trajectory):
    key, sample_key = random.split(key)
    
    # --- Run Strong order 1.0 Stochastic Runge-Kutta ---

    # Sample random initial state for test trajectory.
    init_x = random.normal(sample_key, shape=(2,))

    # Simulate SDE test trajectory.
    test_trajectory = srk_s10_scalar_noise_solve(
        key, drift, diffusion, tspan, init_x, 1.0, test_alpha
    )


    # --- Compute loss function ---

    # Squared difference of means.
    mean_test = jnp.mean(test_trajectory)
    mean_target = jnp.mean(target_trajectory)
    mean_loss = (mean_test - mean_target) ** 2

    # Squared difference of variances.
    var_test = jnp.var(test_trajectory)
    var_target = jnp.var(target_trajectory)
    var_loss = (var_test - var_target) ** 2

    return mean_loss + var_loss


def optimize_alpha(
    tspan, alpha_target, alpha_init, state_init, adam_lr, sim_seed, print_freq
):
    
    # Sample random initial state for target trajectory.
    init_x = jnp.array(state_init)

    # Simulate target trajectory.
    target_key = random.PRNGKey(sim_seed)
    target_trajectory = srk_s10_scalar_noise_solve(
        target_key, drift, diffusion, tspan, init_x, 1.0, alpha_target
    )

    # Initialize optimizer.
    opt_adam = optax.adam(learning_rate=adam_lr)
    opt_state = opt_adam.init(alpha_init)

    loss_and_grad = value_and_grad(loss_fn, argnums=1)

    @jit
    def step(rng, opt_state, alpha):
        # Simulate trajectory with current alpha and compute gradient w.r.t. alpha:
        loss, grads = loss_and_grad(rng, alpha, tspan, target_trajectory)
        
        # Update alpha using gradients:
        updates, opt_state = opt_adam.update(grads, opt_state)
        updated_alpha = optax.apply_updates(alpha, updates)
        
        return loss, opt_state, updated_alpha

    alpha = alpha_init
    key = random.PRNGKey(sim_seed)
    num_opt_iters = int(1e5)
    alpha_values = []
    loss_values = []
    pbar = tqdm(total=num_opt_iters)
    
    # Running gradient descent:
    for i in range(num_opt_iters):
        key, subkey = random.split(key)
        loss, opt_state, alpha = step(subkey, opt_state, alpha)
        alpha_values.append(alpha)
        loss_values.append(loss)
        abs_diff = abs(alpha - alpha_target)
        if i % print_freq == 0:
            log_msg = f"iter {i}: loss = {loss:.4f}, abs_diff_to_true = {abs_diff:.4f}"
            pbar.set_postfix_str(log_msg)
        pbar.update(1)
    pbar.close()

    return alpha, alpha_values, loss_values


def main(
    tstart,
    tfinal,
    stepsize,
    alpha_target,
    alpha_init,
    state_init,
    adam_lr,
    sim_seed,
    print_freq,
):
    tspan = jnp.linspace(tstart, tfinal, int((tfinal - tstart) / stepsize))
    x0s = jnp.array([state_init for _ in range(10)])

    alpha_optimized, alpha_values, loss_values = optimize_alpha(
        tspan, alpha_target, alpha_init, state_init, adam_lr, sim_seed, print_freq
    )
    logging.info(f"Optimized alpha: {alpha_optimized}")

    xs_target = vmap(
        srk_s10_scalar_noise_solve, (None, None, None, None, 0, None, None)
    )(random.PRNGKey(sim_seed), drift, diffusion, tspan, x0s, 1.0, alpha_target)
    xs_optimized = vmap(
        srk_s10_scalar_noise_solve, (None, None, None, None, 0, None, None)
    )(random.PRNGKey(sim_seed), drift, diffusion, tspan, x0s, 1.0, alpha_optimized)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Target plot
    ax = axs[0]
    for j in range(10):
        x = xs_target[j, :, 0]
        y = xs_target[j, :, 1]
        ax.plot(x, y, "-k", lw=0.5)
    ax.set_title(f"Target (alpha={alpha_target})", fontsize=16)
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Optimized plot
    ax = axs[1]
    for j in range(10):
        x = xs_optimized[j, :, 0]
        y = xs_optimized[j, :, 1]
        ax.plot(x, y, "-k", lw=0.5)
    ax.set_title(f"Optimized (alpha={alpha_optimized:.4f})", fontsize=16)
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()

    # Save plot as PDF
    plot_filename = f'{datetime.datetime.now().strftime("%Y%m%d")}_DvdP_optalpha.pdf'
    plot_filepath = os.path.join(os.path.dirname(__file__), plot_filename)
    plt.savefig(plot_filepath, bbox_inches="tight")
    logging.info(f"Plot saved to {plot_filepath}")

    # Save data as .npz
    data_filename = f'{datetime.datetime.now().strftime("%Y%m%d")}_DvdP_optalpha.npz'
    data_filepath = os.path.join(os.path.dirname(__file__), data_filename)
    np.savez(
        data_filepath,
        xs_target=xs_target,
        xs_optimized=xs_optimized,
        tspan=tspan,
        alpha_target=alpha_target,
        alpha_optimized=alpha_optimized,
        alpha_values=alpha_values,
        loss_values=loss_values,
    )
    logging.info(f"Data saved to {data_filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(
        args.tstart,
        args.tfinal,
        args.stepsize,
        args.alpha_target,
        args.alpha_init,
        args.state_init,
        args.adam_lr,
        args.sim_seed,
        args.print_freq,
    )
