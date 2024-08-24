import os
import argparse
import datetime
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import logging

from oscillator import drift, diffusion
from srkS10scalarnoise import srk_s10_scalar_noise_solve
from jax import vmap, random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha_mono', type=float, default=-1.0, help='Alpha value for monostable simulation')
    parser.add_argument('--alpha_bi', type=float, default=1.0, help='Alpha value for bistable simulation')
    parser.add_argument('--Q', type=float, default=1.0, help='Scalar value for Q (positive or zero)')
    return parser.parse_args()

def main(alpha_mono, alpha_bi, Q):
    logging.info(f"Running simulations with alpha_mono={alpha_mono}, alpha_bi={alpha_bi}, Q={Q}")

    tspan = jnp.linspace(0, 20, int(20 / (2**-5)))
    x0s = jnp.array([[-2 - 0.2*(j+1), 0] for j in range(10)])

    logging.info("Simulating monostable system...")
    xs_mono = vmap(srk_s10_scalar_noise_solve, (None, None, None, None, 0, None, None))(random.PRNGKey(0), drift, diffusion, tspan, x0s, Q, alpha_mono)

    logging.info("Simulating bistable system...")
    xs_bistable = vmap(srk_s10_scalar_noise_solve, (None, None, None, None, 0, None, None))(random.PRNGKey(0), drift, diffusion, tspan, x0s, Q, alpha_bi)

    logging.info("Generating plots...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Monostable plot
    ax = axs[0]
    for j in range(10):
        x = xs_mono[j, :, 0]
        y = xs_mono[j, :, 1]
        ax.plot(x, y, '-k', lw=0.5)
    ax.set_title(f'Monostable (alpha={alpha_mono})', fontsize=16)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Bistable plot
    ax = axs[1]
    for j in range(10):
        x = xs_bistable[j, :, 0]
        y = xs_bistable[j, :, 1]
        ax.plot(x, y, '-k', lw=0.5)
    ax.set_title(f'Bistable (alpha={alpha_bi})', fontsize=16)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    # Save plot as PDF
    plot_filename = f'{datetime.datetime.now().strftime("%Y%m%d")}_DvdP_SDE.png'
    plot_filepath = os.path.join(os.path.dirname(__file__), plot_filename)
    logging.info(f"Saving plot to {plot_filepath}...")
    plt.savefig(plot_filepath, bbox_inches='tight')

    # Save data as .npz
    data_filename = f'{datetime.datetime.now().strftime("%Y%m%d")}_DvdP_SDE.npz'
    data_filepath = os.path.join(os.path.dirname(__file__), data_filename)
    logging.info(f"Saving data to {data_filepath}...")
    np.savez(data_filepath, xs_mono=xs_mono, xs_bistable=xs_bistable, tspan=tspan, alpha_bi=alpha_bi, alpha_mono=alpha_mono, Q=Q)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    logging.info("Starting script with the following settings:")
    logging.info(f"alpha_mono: {args.alpha_mono}")
    logging.info(f"alpha_bi: {args.alpha_bi}")
    logging.info(f"Q: {args.Q}")
    main(args.alpha_mono, args.alpha_bi, args.Q)