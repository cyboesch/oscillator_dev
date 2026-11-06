#!/usr/bin/env python3
"""Control for ode_noise_window_fine.py: does splitting the reverse SDE into 0.5-long pieces matter?

Runs the plain reverse SDE the paper's way (one solve and one VirtualBrownianTree on [0, 5] per
trajectory, knots forced, same tolerances) on the same 500 trajectories and base keys as the
fine-grid run (job FINE_JOB, default 4183779).  Trajectories 0-39 use the paper's keys, so they should reproduce the
saved paper SDE samples up to solver tolerance.  Compares, distribution-wise, with the split SDE of
the fine-grid run (same initial states, different noise).  Also re-renders the fine-grid curve.

Read-only on the checkpoint and on the fine-grid output.  Writes out/diagnostics/sde_split_control/<job id>/.
Run under Slurm (cd physical-diffusion-models/ode_vs_sde && sbatch run_sde_split_control_neuronic.sh).
"""

import json
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import flatten_util, vmap  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import ode_noise_window_fine as fine  # noqa: E402  (constants and helpers; its main() is not run)
from ode_noise_window_fine import (  # noqa: E402
    ABLATION_DIR, ATOL_SDE, BROWNIAN_TOL, CHECKPOINT_PATH, DATA_PATH, DT0, GRID, KEY_SEED, MAX_ABS_FREEZE,
    N_NEIGHBOUR_COUPLINGS, N_TIME_STEPS, NEW_SEED, OUT_OF_RANGE_MARGIN, PKG, REF_SEED, RTOL_SDE, SAVED_SDE,
    SIGMA_FORWARD, STD_ADDED_NOISE, SWITCH_TAUS, T_FORWARD, TEMP, nn_distance,
)
from physical_diffusion_fns.helper_fns import interpolate_parameters, normalize_samples  # noqa: E402
from physical_diffusion_fns.network_fns import (  # noqa: E402
    create_2d_square_lattice_connectivity,
    setup_energy_fn,
    setup_overdamped_SDE,
    solve_SDE,
)

N_TRAJ = 500
CHUNK = int(os.environ.get("CHUNK", 125))
FINE_JOB = os.environ.get("FINE_JOB", "4183779")  # job id of the full fine-grid run to compare with
FINE_DIR = os.path.join(PKG, "out", "diagnostics", "ode_noise_window_fine", FINE_JOB)
OUT_DIR = os.path.join(PKG, "out", "diagnostics", "sde_split_control", os.environ.get("SLURM_JOB_ID", "manual"))


def report(line=""):
    print(line, flush=True)
    with open(os.path.join(OUT_DIR, "report.txt"), "a") as f:
        f.write(line + "\n")


def replot_fine_curve():
    with open(os.path.join(FINE_DIR, "metrics.json")) as f:
        rows = json.load(f)["variants"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    taus = [5.0] + SWITCH_TAUS + [0.0]
    curves = [
        ("SDE above τ, ODE below", "#2a78d6", ["ODE"] + [f"SDE>ODE@{t:g}" for t in SWITCH_TAUS] + ["SDE"]),
        ("ODE above τ, SDE below", "#eb6834", ["SDE"] + [f"ODE>SDE@{t:g}" for t in SWITCH_TAUS] + ["ODE"]),
    ]
    mid = 50 * (rows["SDE"]["frac_far"] + rows["ODE"]["frac_far"])
    for name, color, labs in curves:
        y = np.array([100 * rows[l]["frac_far"] for l in labs])
        e = np.array([100 * rows[l]["se_far"] for l in labs])
        ax.fill_between(taus, y - e, y + e, color=color, alpha=0.15, linewidth=0)
        ax.plot(taus, y, color=color, linewidth=2, marker="o", markersize=5, label=name)
        ax.annotate(name, (taus[0], y[0]), textcoords="offset points", xytext=(4, 10 if y[0] >= mid else -16),
                    fontsize=8, color="#333333")
    ax.axhline(10, color="#999999", linewidth=1, linestyle="--")
    ax.text(4.95, 11, "held-out data (10% by construction)", fontsize=8, color="#666666")
    ax.set_xlim(5.05, -0.05)
    ax.set_xlabel("switch point τ (forward time; generation runs left to right)")
    ax.set_ylabel("% samples far from every training image")
    ax.set_title(f"Where the noise matters ({rows['SDE']['n']} trajectories, band = ±1 SE)", fontsize=10)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"noise_window_curve_{FINE_JOB}.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()
    report(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; jax {jax.__version__}; devices {jax.devices()}; "
           f"default float {jnp.zeros(1).dtype}; N_TRAJ {N_TRAJ}; CHUNK {CHUNK}")
    replot_fine_curve()
    report(f"re-rendered the fine-grid curve -> noise_window_curve_{FINE_JOB}.png")

    # Data (for metrics only), as in the fine-grid run
    raw = jnp.array(np.load(DATA_PATH))
    _, _, image_noise_added_key = jr.split(jr.PRNGKey(KEY_SEED), 3)
    _, subkey = jr.split(image_noise_added_key)
    data, mean_px, std_px = normalize_samples(raw + jr.normal(subkey, raw.shape) * STD_ADDED_NOISE)
    data, mean_px, std_px = np.asarray(data), float(mean_px), float(std_px)
    raw = np.asarray(raw)
    blank = np.all(raw == 0.0, axis=0)
    pix_min, pix_max = (0.0 - mean_px) / std_px, (1.0 - mean_px) / std_px
    px_train = data * std_px + mean_px
    rng = np.random.default_rng(REF_SEED)
    held = rng.choice(data.shape[0], size=200, replace=False)
    nn_ref = nn_distance(px_train[held], px_train[np.setdiff1d(np.arange(data.shape[0]), held)])
    blend_threshold = float(np.percentile(nn_ref, 90))
    report(f"reference: held-out nearest-image distance p90 {blend_threshold:.2f}")

    # Model and the paper's interpolation
    ckpt = np.load(CHECKPOINT_PATH, mmap_mode="r")
    params_history = jnp.asarray(np.array(ckpt, dtype=np.float64))
    forward_time_pts = jnp.linspace(0.0, T_FORWARD, N_TIME_STEPS)[::-1]
    interp = interpolate_parameters(params_history, forward_time_pts)
    n = GRID * GRID
    connectivity = create_2d_square_lattice_connectivity(grid_size=GRID, n_neighbour_couplings=N_NEIGHBOUR_COUPLINGS)
    p = int(connectivity.shape[0])
    template = (jnp.ones(n), jnp.zeros(n), jnp.ones(n), jnp.ones(p), jnp.zeros(p), jnp.zeros(p), jnp.zeros(n))
    theta_init, unflatten = flatten_util.ravel_pytree(template)
    energy_fn = setup_energy_fn(connectivity, unflatten)
    theta_lin = jnp.zeros_like(theta_init).at[0:n].set(1.0)
    theta_rev = lambda t: 2.0 * interp(T_FORWARD - t) - theta_lin
    drift_sde, diffusion_sde = setup_overdamped_SDE(energy_fn, theta_rev, n, time_dependent_parms=True, Temp=TEMP)
    knots = jnp.asarray(np.sort(np.asarray(T_FORWARD - forward_time_pts))[1:-1])

    # Same initial states and base keys as the fine-grid run
    x_paper = np.load(os.path.join(ABLATION_DIR, "shared_initial_states.npy"))
    keys_paper = np.load(os.path.join(ABLATION_DIR, "shared_brownian_keys.npy"))
    n_new = N_TRAJ - x_paper.shape[0]
    k_init, k_brown = jr.split(jr.PRNGKey(NEW_SEED))
    x_new = np.asarray(jnp.sqrt(TEMP) * SIGMA_FORWARD * jr.normal(k_init, (n_new, n)))
    keys_new = np.asarray(jr.split(k_brown, n_new))
    x_init = np.concatenate([x_paper, x_new]).astype(np.float64)
    keys_traj = np.concatenate([keys_paper, keys_new.astype(keys_paper.dtype)])
    fine_states = np.load(os.path.join(FINE_DIR, "final_states.npz"))
    report("initial states and base keys rebuilt with the same code and seed as the fine-grid run")

    fn = vmap(lambda y, k: solve_SDE(drift_sde, diffusion_sde, y, k, 0.0, T_FORWARD, 2, DT0, rtol=RTOL_SDE,
                                     atol=ATOL_SDE, brownian_tolerance=BROWNIAN_TOL, step_ts=knots))
    x_sde = np.empty_like(x_init)
    for c0 in range(0, N_TRAJ, CHUNK):
        t_c = time.time()
        idx = np.arange(c0, min(c0 + CHUNK, N_TRAJ))
        pad = np.concatenate([idx, np.repeat(idx[-1:], CHUNK - idx.size)])
        sol = fn(jnp.asarray(x_init[pad]), jnp.asarray(keys_traj[pad]))
        x_sde[idx] = np.asarray(sol.ys[:, -1, :])[: idx.size]
        np.save(os.path.join(OUT_DIR, "final_states_sde_unsplit.npy"), x_sde[: idx[-1] + 1])
        report(f"chunk {c0}-{idx[-1]} done in {time.time() - t_c:.0f} s")

    saved_sde = np.load(os.path.join(ABLATION_DIR, SAVED_SDE))
    d = np.max(np.abs(x_sde[:40] - saved_sde), axis=1)
    report("")
    report(f"reproduction: unsplit SDE vs saved paper SDE, trajectories 0-39: median max|diff| {np.median(d):.2e}, "
           f"max {np.max(d):.2e}; {int(np.sum(d < 1e-3))}/40 below 1e-3")

    def stats(x):
        nn = nn_distance(x * std_px + mean_px, px_train)
        div = np.max(np.abs(x), axis=1) > MAX_ABS_FREEZE
        far = (nn > blend_threshold) | div
        out = (x < pix_min - OUT_OF_RANGE_MARGIN) | (x > pix_max + OUT_OF_RANGE_MARGIN)
        return far, div, {
            "n": int(x.shape[0]), "nn_median": float(np.median(nn)), "frac_far": float(far.mean()),
            "se_far": float(np.sqrt(far.mean() * (1 - far.mean()) / x.shape[0])),
            "frac_diverged": float(div.mean()), "frac_out": float(out.mean()),
            "bg_within": float(np.median(x[~div][:, blank].std(axis=1))),
        }

    rows, fars = {}, {}
    for lab, x in [("SDE unsplit (this job)", x_sde), (f"SDE split ({FINE_JOB})", fine_states["SDE"]),
                   (f"ODE ({FINE_JOB})", fine_states["ODE"]),
                   ("SDE unsplit, traj 0-39", x_sde[:40]), ("SDE split, traj 0-39", fine_states["SDE"][:40]),
                   ("SDE paper (saved, 40)", saved_sde)]:
        fars[lab], _, rows[lab] = stats(x)
    report("")
    report("far = nearest-image distance above the data's p90, or diverged (max|x| > 15)")
    report(f"{'sample set':<26}{'n':>5}{'NN med':>8}{'% far':>7}{'+-':>5}{'%div':>6}{'%out':>7}{'bg within':>10}")
    for lab, r in rows.items():
        report(f"{lab:<26}{r['n']:>5}{r['nn_median']:>8.2f}{100 * r['frac_far']:>7.1f}{100 * r['se_far']:>5.1f}"
               f"{100 * r['frac_diverged']:>6.1f}{100 * r['frac_out']:>7.2f}{r['bg_within']:>10.4f}")
    a, b = fars["SDE unsplit (this job)"], fars[f"SDE split ({FINE_JOB})"]
    diff = a.mean() - b.mean()
    se = np.sqrt(a.mean() * (1 - a.mean()) / a.size + b.mean() * (1 - b.mean()) / b.size)
    report("")
    report(f"unsplit - split: {100 * diff:+.1f} +- {100 * se:.1f} points far (independent noise, so unpaired SE); "
           f"z = {diff / se:+.2f}")
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"blend_threshold": blend_threshold, "rows": rows, "unsplit_minus_split": diff, "se": se}, f, indent=2)
    report(f"total time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
