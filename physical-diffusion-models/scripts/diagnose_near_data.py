#!/usr/bin/env python3
"""Read-only diagnostics for the near-data (t -> 0) end of the paper checkpoint and for the
existing interpolation-ablation samples.

Part A - checkpoint, slice by slice.
    Always-blank pixels (raw value 0 in every training image) carry only the added pixel noise,
    independent of everything else. Their exact score-matching optimum is therefore known in
    closed form: an independent Gaussian with mean mu(t) = e^{-t} m and variance
    v(t) = e^{-2t} s^2 + T (1 - e^{-2t}), where m and s^2 are the pixel's empirical mean and
    variance after preprocessing. The learned conditional stiffness d^2E/dx_i^2 should equal
    k*(t) = T / v(t), and the learned score -dE/dx_i / T should equal -(x_i - mu(t)) / v(t).
    Both are measured on forward samples x ~ p_t. Each slice is also checked with a global
    scale test, R = E||grad E||^2 / (T E[tr Hess E]), which equals 1 at the score-matching
    optimum of any energy that is linear in theta (R < 1: too soft, R > 1: too stiff or held
    by the k_6 floor). Finally, each slice's parameter movement is compared with the most Adam
    can move one coordinate under the training schedule (the "step budget").

Part B - existing outputs (no new sampling).
    Background level and noise, out-of-range pixels, gray-pixel fraction, energies at theta(0),
    the same scale test on the samples, and solver step counts for the 8 ablation variants,
    plus sample grids plotted with fixed colour limits in pixel space.

Writes only to out/diagnostics/near_data_check/<job id>/. Opens the checkpoint read-only.
Run under Slurm (container/run_near_data_check_neuronic.sh), never on the login node.
"""

import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)  # must precede any array creation or RNG call
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import flatten_util  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.append(PKG)

from physical_diffusion_fns.helper_fns import normalize_samples, sample_forward_process  # noqa: E402
from physical_diffusion_fns.network_fns import (  # noqa: E402
    create_2d_square_lattice_connectivity,
    setup_energy_fn,
)
from physical_diffusion_fns.network_fns_memory_efficient import (  # noqa: E402
    setup_duffing_network_analytical_derivatives_memory_efficient,
)

#####################################
# Configuration: must match scripts/physical_diffusion_model_MNIST.py
#####################################
KEY_SEED = 1
STD_ADDED_NOISE = 0.01
TEMP = 0.02
SIGMA_FORWARD = 1.0
T_FORWARD = 5.0
N_TIME_STEPS = 100
GRID = 28
N_NEIGHBOUR_COUPLINGS = 14
LR_START, LR_END = 2e-5, 0.002
LR_DECAY_RATE, LR_DECAY_STEPS = 0.95, 6000
WINDOW_SIZE = 1000
PATIENCE_START, PATIENCE_END = 300, 100
K6_FLOOR = 0.01
EXPECTED_SHA256 = "37cfe46ef805a39129c265a0c17aaf6cf2b4e3b2ec7d8f55725e1cd66c227249"
EXPECTED_PIXEL_RANGE = (-0.4096, 2.8727)  # printed by the ablation run (job 3746389)

#####################################
# Diagnostic settings
#####################################
BATCH_PER_SLICE = 128      # forward samples per slice for stiffness, score and scale test
CHUNK = 32                 # samples evaluated together (bounds memory)
BATCH_DIRECTIONAL = 32     # samples per slice for the background-wide directional stiffness
DIR_CHUNK = 8
N_RANDOM_DIRECTIONS = 4    # random-sign background directions per slice
N_DATA_REFERENCE = 256     # data images used as the reference in Part B
OUT_OF_RANGE_MARGIN = 0.2  # normalised units, about 6x the background noise std
DIAG_SEED = 2026
TABLE_SLICES = [0, 25, 50, 75, 85] + list(range(88, N_TIME_STEPS))

#####################################
# Paths
#####################################
DATA_PATH = os.path.join(PKG, "data", "MNIST", "mnist_labels_[0, 1]_resolution_(28, 28)_balanced_True.npy")
RUN_DIR = os.path.join(
    PKG, "out", "2026_07_08", "problems", "MNIST_generation",
    "labels_[0, 1]_resolution_28_x_28",
    "n_couplings_14_Temp_0.02_energy_fn_type_6th_order_duffing_coupling",
    "added_gaussian_noise_std_0.01_key_seed_1_additional_rescaling_1",
    "training_method_SM_at_kbT_minimal_memory_solve_opt_in_reverse_True_exp_time_pts_False_"
    "t_forward_5.0_n_timesteps_100_sigma_forward_1.0_lr_2e-05_to_0.002_epochs_100000_batch_512_"
    "window_1000_tol_0.001_to_0.001_patience_300_to_100_",
    "opt_per_time_plots",
)
CHECKPOINT_PATH = os.path.join(RUN_DIR, "params_history.npy")
ABLATION_DIR = os.path.join(RUN_DIR, "interpolation_ablation_v1")
OUT_DIR = os.path.join(
    PKG, "out", "diagnostics", "near_data_check", os.environ.get("SLURM_JOB_ID", "manual")
)

REPORT_LINES = []


def report(line=""):
    print(line, flush=True)
    REPORT_LINES.append(line)


def save_report():
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(REPORT_LINES) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(x, digits=3):
    if x is None:
        return "-"
    x = float(x)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}g}"


#####################################
# Setup
#####################################
def load_data():
    """Reproduce the training data exactly as the MNIST script does (same keys, same order)."""
    raw = jnp.array(np.load(DATA_PATH))
    master_key = jr.PRNGKey(KEY_SEED)
    _, _, image_noise_added_key = jr.split(master_key, 3)
    _, subkey_image_noise_added = jr.split(image_noise_added_key)
    noisy = raw + jr.normal(subkey_image_noise_added, raw.shape) * STD_ADDED_NOISE
    data, mean_px, std_px = normalize_samples(noisy)
    return np.asarray(raw), data, float(mean_px), float(std_px)


def setup_network():
    n = GRID * GRID
    connectivity = create_2d_square_lattice_connectivity(
        grid_size=GRID, n_neighbour_couplings=N_NEIGHBOUR_COUPLINGS
    )
    p = int(connectivity.shape[0])
    template = (
        jnp.ones(n), jnp.zeros(n), jnp.ones(n),   # k_lin, k_duff, k_6
        jnp.ones(p), jnp.zeros(p), jnp.zeros(p),  # c_lin, c_optomech, c_duff
        jnp.zeros(n),                             # bias (last in the flatten order)
    )
    theta_init, unflatten = flatten_util.ravel_pytree(template)
    energy_fn = setup_energy_fn(connectivity, unflatten)
    gradient_fn, trace_hessian_fn, _, _ = setup_duffing_network_analytical_derivatives_memory_efficient(
        connectivity, unflatten
    )
    return n, p, connectivity, theta_init, unflatten, energy_fn, gradient_fn, trace_hessian_fn


def group_slices(n, p):
    names = ["k_lin", "k_duff", "k_6", "c_lin", "c_optomech", "c_duff", "bias"]
    sizes = [n, n, n, p, p, p, n]
    offsets = np.cumsum([0] + sizes)
    return {name: slice(int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names)}


def make_hess_diag(connectivity, unflatten):
    """Per-pixel Hessian diagonal; same formulas as network_fns_memory_efficient, not summed."""
    conn_i = connectivity[:, 0]
    conn_j = connectivity[:, 1]

    def hess_diag(x, theta):
        k_lin, k_duff, k_6, c_lin, c_om, c_duff, _bias = unflatten(theta)
        h_self = k_lin + 3.0 * k_duff * x**2 + 5.0 * k_6 * x**4
        xi = x[conn_i]
        xj = x[conn_j]
        d2 = (xi - xj) ** 2
        h = h_self.at[conn_i].add(2.0 * c_lin + 2.0 * c_om * xj + 3.0 * c_duff * d2)
        h = h.at[conn_j].add(2.0 * c_lin + 3.0 * c_duff * d2)
        return h, h_self

    return hess_diag


def make_batch_eval(gradient_fn, hess_diag):
    @jax.jit
    def batch_eval(theta, xs):
        g = jax.vmap(lambda x: gradient_fn(x, theta))(xs)
        h, h_self = jax.vmap(lambda x: hess_diag(x, theta))(xs)
        return g, h, h_self

    return batch_eval


def make_directional(gradient_fn):
    @jax.jit
    def directional(theta, xs, dirs):
        """(B, D) array of v^T Hess E(x) v for each sample x and unit direction v."""

        def one(x, v):
            _, hv = jax.jvp(lambda y: gradient_fn(y, theta), (x,), (v,))
            return jnp.dot(v, hv)

        return jax.vmap(lambda x: jax.vmap(lambda v: one(x, v))(dirs))(xs)

    return directional


def eval_in_chunks(fn, theta, xs, chunk, *extra):
    outs = [fn(theta, xs[i:i + chunk], *extra) for i in range(0, xs.shape[0], chunk)]
    return jax.tree_util.tree_map(
        lambda *parts: np.concatenate([np.asarray(part) for part in parts], axis=0), *outs
    )


def step_budgets():
    """Most that Adam can move one coordinate per slice (about lr per step), under the
    training schedule and the fixed epoch counts caused by the early-stopping NaN."""
    lr_sched = np.asarray(jnp.linspace(LR_START, LR_END, N_TIME_STEPS)[::-1])
    pat_sched = np.asarray(jnp.linspace(PATIENCE_START, PATIENCE_END, N_TIME_STEPS)[::-1])
    first_check = int(math.ceil((WINDOW_SIZE - 1) / 100.0) * 100)
    budgets, n_steps = [], []
    for k in range(N_TIME_STEPS):
        stop_epoch = first_check + 100 * (int(math.ceil(float(pat_sched[k]))) - 1)
        steps = stop_epoch + 1
        decay = LR_DECAY_RATE ** (np.arange(steps) / LR_DECAY_STEPS)
        budgets.append(float(lr_sched[k]) * float(decay.sum()))
        n_steps.append(steps)
    return lr_sched, np.array(budgets), np.array(n_steps)


def self_tests(theta, x, energy_fn, gradient_fn, trace_hessian_fn, hess_diag, test_pixels):
    """Check the analytic gradient and the per-pixel Hessian diagonal against autodiff."""
    grad_e = jax.grad(lambda y: energy_fn(y, theta))
    g_ad = grad_e(x)
    g_lib = gradient_fn(x, theta)
    err_g = float(jnp.max(jnp.abs(g_lib - g_ad)) / (jnp.max(jnp.abs(g_ad)) + 1e-300))
    h, _ = hess_diag(x, theta)
    tr_lib = trace_hessian_fn(x, theta)
    err_tr = float(jnp.abs(jnp.sum(h) - tr_lib) / (jnp.abs(tr_lib) + 1e-300))
    errs = []
    for i in test_pixels:
        e_i = jnp.zeros_like(x).at[int(i)].set(1.0)
        _, hv = jax.jvp(grad_e, (x,), (e_i,))
        errs.append(abs(float(hv[int(i)]) - float(h[int(i)])) / (abs(float(hv[int(i)])) + 1e-12))
    return err_g, err_tr, max(errs)


#####################################
# Part B: existing ablation outputs
#####################################
def sample_stats(x_norm, blank, bg_level, mean_px, std_px, pix_min, pix_max):
    xb = x_norm[:, blank]
    px = x_norm * std_px + mean_px
    out = (x_norm < pix_min - OUT_OF_RANGE_MARGIN) | (x_norm > pix_max + OUT_OF_RANGE_MARGIN)
    return {
        "bg_mean": float(xb.mean()),
        "bg_std_all": float(xb.std()),
        "bg_std_within_sample_median": float(np.median(xb.std(axis=1))),
        "bg_offset_abs_median": float(np.median(np.abs(xb.mean(axis=1) - bg_level))),
        "frac_pixels_out_of_range": float(out.mean()),
        "frac_samples_with_out_of_range": float(out.any(axis=1).mean()),
        "max_abs": float(np.max(np.abs(x_norm))),
        "frac_gray_pixels": float(((px > 0.1) & (px < 0.9)).mean()),
        "mean_intensity_px": float(px.mean()),
    }


def part_b(data_np, blank, mean_px, std_px, theta0, batch_eval, energy_batch, rng):
    report()
    report("=" * 100)
    report("PART B: existing interpolation-ablation outputs (no new sampling)")
    report("=" * 100)
    manifest_path = os.path.join(ABLATION_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        report(f"SKIPPED: no manifest at {manifest_path}")
        return {}
    with open(manifest_path) as f:
        manifest = json.load(f)

    pix_min = (0.0 - mean_px) / std_px
    pix_max = (1.0 - mean_px) / std_px
    bg_level = float(data_np[:, blank].mean())
    ref_idx = rng.choice(data_np.shape[0], size=N_DATA_REFERENCE, replace=False)
    data_ref = data_np[ref_idx]

    def scale_and_energy(x_norm):
        g, h, _ = eval_in_chunks(batch_eval, theta0, jnp.asarray(x_norm), CHUNK)
        r_scale = float(np.mean(np.sum(g**2, axis=1)) / (TEMP * np.mean(np.sum(h, axis=1))))
        e_over_t = eval_in_chunks(energy_batch, theta0, jnp.asarray(x_norm), CHUNK) / TEMP
        score_norm = np.sqrt(np.sum(g**2, axis=1)) / TEMP
        return r_scale, float(np.median(e_over_t)), float(np.median(score_norm))

    rows = []
    ref = sample_stats(data_np, blank, bg_level, mean_px, std_px, pix_min, pix_max)
    ref["R_scale_theta0"], ref["E_over_T_median"], ref["score_norm_median"] = scale_and_energy(data_ref)
    ref.update({"label": "DATA", "sha_ok": True, "steps_median": None, "steps_max": None})
    rows.append(ref)

    variant_states = {}
    for v in manifest.get("variants", []):
        sampler = "SDE" if v["sampler"] == "sde" else "ODE"
        params_label = "raw" if v["parameter_variant"] == "raw" else "SG"
        label = f"{sampler}/{params_label}/{v['interpolation']}"
        path = os.path.join(ABLATION_DIR, v["final_states_file"])
        if not os.path.exists(path):
            report(f"  missing: {path}")
            continue
        sha_ok = sha256_file(path) == v.get("final_states_sha256")
        x_norm = np.asarray(np.load(path), dtype=np.float64)  # additional_rescaling = 1
        stats = sample_stats(x_norm, blank, bg_level, mean_px, std_px, pix_min, pix_max)
        stats["R_scale_theta0"], stats["E_over_T_median"], stats["score_norm_median"] = scale_and_energy(x_norm)
        steps = None
        stats_path = os.path.join(ABLATION_DIR, v.get("solver_stats_file", ""))
        if v.get("solver_stats_file") and os.path.exists(stats_path):
            with np.load(stats_path) as npz:
                if "num_steps" in npz.files:
                    steps = np.asarray(npz["num_steps"]).reshape(-1)
        stats.update({
            "label": label,
            "sha_ok": sha_ok,
            "steps_median": None if steps is None else float(np.median(steps)),
            "steps_max": None if steps is None else float(np.max(steps)),
        })
        rows.append(stats)
        variant_states[label] = (x_norm, steps)

    report(f"Pixel range of the data (normalised): [{pix_min:.4f}, {pix_max:.4f}]; out-of-range margin "
           f"{OUT_OF_RANGE_MARGIN}. Background = always-blank pixels; data level {bg_level:.4f}.")
    report("bg std: std over all samples x blank pixels; within: median per-sample std; offset: median |sample "
           "background mean - data level|.")
    report("R(theta0): E||grad E||^2 / (T E[tr Hess E]) at theta(t=0); 1 for samples from exp(-E/T).")
    header = (f"{'variant':<18}{'sha':>4}{'bg mean':>9}{'bg std':>8}{'within':>8}{'offset':>8}"
              f"{'%out':>7}{'%smp out':>9}{'max|x|':>8}{'%gray':>7}{'R(th0)':>8}{'E/T med':>11}"
              f"{'steps med/max':>15}")
    report(header)
    for r in rows:
        steps_txt = "-" if r["steps_median"] is None else f"{r['steps_median']:.0f}/{r['steps_max']:.0f}"
        report(f"{r['label']:<18}{('ok' if r['sha_ok'] else 'NO'):>4}{r['bg_mean']:>9.4f}{r['bg_std_all']:>8.4f}"
               f"{r['bg_std_within_sample_median']:>8.4f}{r['bg_offset_abs_median']:>8.4f}"
               f"{100 * r['frac_pixels_out_of_range']:>7.2f}{100 * r['frac_samples_with_out_of_range']:>9.1f}"
               f"{r['max_abs']:>8.2f}{100 * r['frac_gray_pixels']:>7.2f}{fmt(r['R_scale_theta0']):>8}"
               f"{fmt(r['E_over_T_median'], 5):>11}{steps_txt:>15}")

    # Trajectories with many solver steps: are they the out-of-range ones?
    for label, (x_norm, steps) in variant_states.items():
        if steps is None or len(steps) != x_norm.shape[0]:
            continue
        slow = np.where(steps > 3 * np.median(steps))[0]
        if len(slow) == 0:
            continue
        out = (x_norm < pix_min - OUT_OF_RANGE_MARGIN) | (x_norm > pix_max + OUT_OF_RANGE_MARGIN)
        details = ", ".join(
            f"#{i}: {int(steps[i])} steps, max|x| {np.max(np.abs(x_norm[i])):.2f}, {100 * out[i].mean():.1f}% out"
            for i in slow
        )
        report(f"  {label}: {len(slow)} slow trajectories (> 3x median steps): {details}")

    with open(os.path.join(OUT_DIR, "partB_variants.json"), "w") as f:
        json.dump(rows, f, indent=2)
    plot_samples(data_np, variant_states, mean_px, std_px, rng)
    save_report()
    return {r["label"]: r for r in rows}


def plot_samples(data_np, variant_states, mean_px, std_px, rng):
    """Sample grids in pixel space with fixed colour limits [0, 1] (no per-image autoscaling)."""
    if not variant_states:
        return
    n_col = 10
    labels = list(variant_states)
    fig, axes = plt.subplots(len(labels) + 1, n_col, figsize=(1.2 * n_col, 1.25 * (len(labels) + 1)))
    data_rows = data_np[rng.choice(data_np.shape[0], size=n_col, replace=False)]
    row_sets = [("DATA", data_rows)] + [(lab, variant_states[lab][0][:n_col]) for lab in labels]
    for r, (lab, xs) in enumerate(row_sets):
        for c in range(n_col):
            ax = axes[r, c]
            if c < xs.shape[0]:
                ax.imshow((xs[c] * std_px + mean_px).reshape(GRID, GRID), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("First 10 samples per variant, pixel space, fixed limits [0, 1]", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "samples_fixed_limits.png"), dpi=150)
    plt.close(fig)

    for lab in [lab for lab in labels if "/raw/linear" in lab]:
        xs = variant_states[lab][0]
        n_rows = int(math.ceil(xs.shape[0] / n_col))
        fig, axes = plt.subplots(n_rows, n_col, figsize=(1.2 * n_col, 1.25 * n_rows), squeeze=False)
        for idx in range(n_rows * n_col):
            ax = axes[idx // n_col, idx % n_col]
            if idx < xs.shape[0]:
                ax.imshow((xs[idx] * std_px + mean_px).reshape(GRID, GRID), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{lab}: all samples, pixel space, fixed limits [0, 1]", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"samples_{lab.replace('/', '_')}_fixed_limits.png"), dpi=150)
        plt.close(fig)


#####################################
# Part A: checkpoint, slice by slice
#####################################
def part_a(ckpt, data, data_np, blank, theta_init, n, p, connectivity, gs, batch_eval, directional):
    report()
    report("=" * 100)
    report("PART A: checkpoint, slice by slice (always-blank pixels against their exact optimum)")
    report("=" * 100)
    forward_time_pts = np.asarray(jnp.linspace(0.0, T_FORWARD, N_TIME_STEPS)[::-1])
    lr_sched, budgets, n_steps = step_budgets()
    report("Step budget (most one coordinate can move per slice, about sum of lr over the slice's steps):")
    report("  " + ", ".join(f"t={forward_time_pts[k]:.3f}: {budgets[k]:.3g}" for k in [0, 50, 90, 95, 96, 97, 98, 99]))
    report("  assumes the current lr schedule, decay 0.95 per 6000 steps and the fixed epoch counts of the "
           "early-stopping NaN; the stored theta is the best-epoch iterate, so actual movement can be smaller.")

    blank_idx = np.where(blank)[0]
    n_blank = len(blank_idx)
    m_blank = data_np[:, blank].mean(axis=0)
    s2_blank = data_np[:, blank].var(axis=0)
    conn_np = np.asarray(connectivity)
    dir_rng = np.random.default_rng(DIAG_SEED + 1)

    rows = []
    t0 = time.time()
    for k in range(N_TIME_STEPS):
        t = float(forward_time_pts[k])
        theta_np = np.array(ckpt[k], dtype=np.float64)
        theta = jnp.asarray(theta_np)
        xs = sample_forward_process(t, BATCH_PER_SLICE, TEMP, SIGMA_FORWARD, data, jr.fold_in(jr.PRNGKey(DIAG_SEED), k))
        g, h, h_self = eval_in_chunks(batch_eval, theta, xs, CHUNK)
        xs_np = np.asarray(xs)

        # Exact optimum for always-blank pixels at time t.
        e1, e2 = math.exp(-t), math.exp(-2.0 * t)
        mu = e1 * m_blank
        v = e2 * s2_blank + TEMP * (1.0 - e2)
        kstar = TEMP / v

        xb, gb, hb, hsb = xs_np[:, blank], g[:, blank], h[:, blank], h_self[:, blank]
        kappa = hb.mean(axis=0)
        kappa_self = hsb.mean(axis=0)
        ratio = kappa / kstar
        ratio_self = kappa_self / kstar
        ratio_coupling = (kappa - kappa_self) / kstar
        s_model = -gb / TEMP
        s_star = -(xb - mu) / v
        score_rel_err = float(np.sum((s_model - s_star) ** 2) / np.sum(s_star**2))
        positive = hb > 0
        mode_est = np.where(positive, xb - gb / np.where(positive, hb, 1.0), np.nan)
        with np.errstate(invalid="ignore"):
            mode_offset = np.abs(np.nanmean(mode_est, axis=0) - mu) / np.sqrt(v)

        # Global scale test on all pixels: 1 at the optimum of a linear-in-theta energy.
        r_scale = float(np.mean(np.sum(g**2, axis=1)) / (TEMP * np.mean(np.sum(h, axis=1))))

        # Background-wide directions: uniform shift of all blank pixels, and random-sign patterns.
        u = np.zeros(n)
        u[blank_idx] = 1.0 / math.sqrt(n_blank)
        dirs = [u]
        for _ in range(N_RANDOM_DIRECTIONS):
            r = np.zeros(n)
            r[blank_idx] = dir_rng.choice([-1.0, 1.0], size=n_blank) / math.sqrt(n_blank)
            dirs.append(r)
        q = eval_in_chunks(directional, theta, xs[:BATCH_DIRECTIONAL], DIR_CHUNK, jnp.asarray(np.stack(dirs)))
        kstar_mean = float(np.mean(kstar))
        q_uniform = float(np.mean(q[:, 0])) / kstar_mean
        q_random = float(np.mean(q[:, 1:])) / kstar_mean

        # Raw self parameters of blank pixels, and the sample-independent c_lin stiffness.
        c_lin = theta_np[gs["c_lin"]]
        two_sum_c_lin = 2.0 * (np.bincount(conn_np[:, 0], weights=c_lin, minlength=n)
                               + np.bincount(conn_np[:, 1], weights=c_lin, minlength=n))
        k6_all = theta_np[gs["k_6"]]

        # Movement since the previous slice versus the step budget.
        prev = np.asarray(theta_init) if k == 0 else np.array(ckpt[k - 1], dtype=np.float64)
        delta = np.abs(theta_np - prev) / budgets[k]
        movement = {}
        for name, sl in gs.items():
            d = delta[sl]
            movement[name] = {
                "median": float(np.median(d)),
                "p90": float(np.percentile(d, 90)),
                "frac_ge_0.5": float(np.mean(d >= 0.5)),
                "frac_ge_0.9": float(np.mean(d >= 0.9)),
                "max": float(np.max(d)),
            }

        row = {
            "k": k, "t": t, "budget": float(budgets[k]), "n_steps": int(n_steps[k]),
            "kstar_median": float(np.median(kstar)),
            "stiffness_ratio_median": float(np.median(ratio)),
            "stiffness_ratio_p10": float(np.percentile(ratio, 10)),
            "stiffness_ratio_p90": float(np.percentile(ratio, 90)),
            "self_ratio_median": float(np.median(ratio_self)),
            "coupling_ratio_median": float(np.median(ratio_coupling)),
            "score_rel_err": score_rel_err,
            "mode_offset_median_sd": float(np.nanmedian(mode_offset)),
            "frac_nonconvex": float(1.0 - positive.mean()),
            "uniform_dir_ratio": q_uniform,
            "random_dir_ratio": q_random,
            "R_scale_all_pixels": r_scale,
            "k_lin_blank_median": float(np.median(theta_np[gs["k_lin"]][blank])),
            "bias_blank_median": float(np.median(theta_np[gs["bias"]][blank])),
            "k_duff_blank_median": float(np.median(theta_np[gs["k_duff"]][blank])),
            "k_6_blank_median": float(np.median(k6_all[blank])),
            "bias_star_median": float(np.median(-kstar * mu)),
            "two_sum_c_lin_blank_median": float(np.median(two_sum_c_lin[blank])),
            "frac_k6_at_floor_all": float(np.mean(k6_all <= K6_FLOOR * (1 + 1e-9))),
        }
        for name in gs:
            for key_name, value in movement[name].items():
                row[f"move_{name}_{key_name}"] = value
        rows.append(row)
        if k % 10 == 9 or k == N_TIME_STEPS - 1:
            report(f"  ... slice {k + 1}/{N_TIME_STEPS} done ({time.time() - t0:.0f} s)")

    with open(os.path.join(OUT_DIR, "partA_per_slice.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report()
    report("Table A1. Always-blank pixels vs exact optimum (ratios: 1 = optimal).")
    report("  stiff = mean d2E/dx_i^2 / k*, median [p10, p90] over blank pixels; self/coup = its self-term and "
           "coupling parts; score err = E(s_model - s*)^2 / E s*^2;")
    report("  mode = |conditional mode - mu| in target sd; unif/rand = stiffness along a uniform / random-sign "
           "background-wide direction, / mean k*; R = global scale test (all pixels).")
    report(f"{'k':>3}{'t':>7}{'k*':>8}{'stiff':>7}{'[p10,':>7}{'p90]':>6}{'self':>7}{'coup':>7}"
           f"{'score err':>10}{'mode':>7}{'unif':>7}{'rand':>7}{'R':>7}")
    for k in TABLE_SLICES:
        r = rows[k]
        report(f"{k:>3}{r['t']:>7.3f}{r['kstar_median']:>8.3g}{r['stiffness_ratio_median']:>7.3f}"
               f"{r['stiffness_ratio_p10']:>7.3f}{r['stiffness_ratio_p90']:>6.2f}{r['self_ratio_median']:>7.3f}"
               f"{r['coupling_ratio_median']:>7.3f}{r['score_rel_err']:>10.3g}{r['mode_offset_median_sd']:>7.2f}"
               f"{r['uniform_dir_ratio']:>7.3f}{r['random_dir_ratio']:>7.3f}{r['R_scale_all_pixels']:>7.3f}")

    report()
    report("Table A2. Parameters of blank pixels (medians) and movement since the previous slice.")
    report("  bias* = -k* mu is the self-only optimum; move = |theta_k - theta_k-1| / budget; sat = fraction of the "
           "group's coordinates with move >= 0.5.")
    groups = list(gs)
    report(f"{'k':>3}{'t':>7}{'budget':>8}{'k_lin':>8}{'k*':>8}{'bias':>8}{'bias*':>8}{'2sum c':>8}{'k6@floor':>9}  "
           + " ".join(f"{('sat ' + name):>13}" for name in groups))
    for k in TABLE_SLICES:
        r = rows[k]
        sat = " ".join(f"{r[f'move_{name}_frac_ge_0.5']:>13.3f}" for name in groups)
        report(f"{k:>3}{r['t']:>7.3f}{r['budget']:>8.3g}{r['k_lin_blank_median']:>8.3g}{r['kstar_median']:>8.3g}"
               f"{r['bias_blank_median']:>8.3g}{r['bias_star_median']:>8.3g}{r['two_sum_c_lin_blank_median']:>8.3g}"
               f"{r['frac_k6_at_floor_all']:>9.3f}  {sat}")
    report()
    report("Largest movement ratios in the last 5 slices (max over each group; > 1 would exceed the budget):")
    for k in range(N_TIME_STEPS - 5, N_TIME_STEPS):
        r = rows[k]
        report(f"  k={k} t={r['t']:.3f}: " + ", ".join(f"{name} {r[f'move_{name}_max']:.2f}" for name in groups))
    plot_part_a(rows, groups)
    save_report()
    return rows


def plot_part_a(rows, groups):
    t = np.array([r["t"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, xmax in zip(axes[0], [T_FORWARD, 0.5]):
        sel = t <= xmax + 1e-12
        ax.fill_between(t[sel], [r["stiffness_ratio_p10"] for r, s in zip(rows, sel) if s],
                        [r["stiffness_ratio_p90"] for r, s in zip(rows, sel) if s], alpha=0.25, label="p10-p90")
        ax.plot(t[sel], [r["stiffness_ratio_median"] for r, s in zip(rows, sel) if s], "k.-", label="total")
        ax.plot(t[sel], [r["self_ratio_median"] for r, s in zip(rows, sel) if s], "b.-", label="self terms")
        ax.plot(t[sel], [r["coupling_ratio_median"] for r, s in zip(rows, sel) if s], "r.-", label="couplings")
        ax.plot(t[sel], [r["uniform_dir_ratio"] for r, s in zip(rows, sel) if s], "g--", label="uniform background mode")
        ax.plot(t[sel], [r["R_scale_all_pixels"] for r, s in zip(rows, sel) if s], "m:", label="R (all pixels)")
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_xlabel("forward time t")
        ax.set_ylabel("learned / optimal")
        ax.set_title(f"Always-blank pixels: stiffness / k*(t), t <= {xmax:g}")
        ax.legend(fontsize=7)
    ax = axes[1, 0]
    ax.semilogy(t, [r["score_rel_err"] for r in rows], "k.-")
    ax.set_xlabel("forward time t")
    ax.set_ylabel("relative score error (blank pixels)")
    ax.set_title("E(s_model - s*)^2 / E(s*)^2")
    ax = axes[1, 1]
    for name in groups:
        ax.plot(t, [r[f"move_{name}_frac_ge_0.5"] for r in rows], ".-", label=name)
    ax.set_xlabel("forward time t")
    ax.set_ylabel("fraction moving >= 0.5 x budget")
    ax.set_title("Step-budget saturation per parameter group")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "partA_near_data.png"), dpi=150)
    plt.close(fig)


#####################################
# Main
#####################################
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()
    report(f"near-data diagnostics; jax {jax.__version__}; devices {jax.devices()}; default float {jnp.zeros(1).dtype}")
    report(f"output directory: {OUT_DIR}")

    for path in (DATA_PATH, CHECKPOINT_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    sha = sha256_file(CHECKPOINT_PATH)
    report(f"checkpoint sha256 {sha} ({'matches the paper checkpoint' if sha == EXPECTED_SHA256 else 'DOES NOT MATCH'})")
    ckpt = np.load(CHECKPOINT_PATH, mmap_mode="r")

    raw, data, mean_px, std_px = load_data()
    data_np = np.asarray(data)
    pix_min, pix_max = (0.0 - mean_px) / std_px, (1.0 - mean_px) / std_px
    ok_range = abs(pix_min - EXPECTED_PIXEL_RANGE[0]) < 5e-4 and abs(pix_max - EXPECTED_PIXEL_RANGE[1]) < 5e-4
    report(f"data {raw.shape}; mean {mean_px:.6f}, std {std_px:.6f}; pixel range [{pix_min:.4f}, {pix_max:.4f}] "
           f"({'matches' if ok_range else 'DOES NOT MATCH'} the paper run)")

    n, p, connectivity, theta_init, unflatten, energy_fn, gradient_fn, trace_hessian_fn = setup_network()
    gs = group_slices(n, p)
    if ckpt.shape != (N_TIME_STEPS, int(theta_init.shape[0])):
        raise RuntimeError(f"unexpected checkpoint shape {ckpt.shape}; expected {(N_TIME_STEPS, int(theta_init.shape[0]))}")
    report(f"network: {n} oscillators, {p} pairs, {int(theta_init.shape[0])} parameters; checkpoint {ckpt.shape}")

    blank = np.all(raw == 0.0, axis=0)
    n_blank = int(blank.sum())
    sd_pre = float(np.sqrt(np.mean(data_np[:, blank].var(axis=0)))) if n_blank else float("nan")
    report(f"always-blank pixels: {n_blank} of {n}; their data level {data_np[:, blank].mean():.4f}, "
           f"noise sd {sd_pre:.4f} (expected about -0.4096 and 0.0328)")
    if n_blank < 10:
        raise RuntimeError("too few always-blank pixels for Part A")

    hess_diag = make_hess_diag(connectivity, unflatten)
    batch_eval = make_batch_eval(gradient_fn, hess_diag)
    directional = make_directional(gradient_fn)

    @jax.jit
    def energy_batch(theta, xs):
        return jax.vmap(lambda x: energy_fn(x, theta))(xs)

    rng = np.random.default_rng(DIAG_SEED)
    test_pixels = list(np.where(blank)[0][:4]) + list(rng.choice(n, size=4, replace=False))
    for k, x in ((N_TIME_STEPS - 1, data[0]), (0, data[1] * math.exp(-T_FORWARD))):
        err_g, err_tr, err_h = self_tests(jnp.asarray(np.array(ckpt[k])), x, energy_fn, gradient_fn,
                                          trace_hessian_fn, hess_diag, test_pixels)
        passed = err_g < 1e-9 and err_tr < 1e-9 and err_h < 1e-7
        report(f"self-test slice {k}: gradient vs autodiff {err_g:.1e}, sum(diag) vs trace {err_tr:.1e}, "
               f"diag vs autodiff HVP {err_h:.1e} -> {'PASS' if passed else 'FAIL'}")
        if not passed:
            report("WARNING: analytic derivatives disagree with autodiff; Part A numbers are unreliable.")
    save_report()

    theta0 = jnp.asarray(np.array(ckpt[N_TIME_STEPS - 1], dtype=np.float64))  # row 99 is t = 0
    part_b_rows = part_b(data_np, blank, mean_px, std_px, theta0, batch_eval, energy_batch, rng)
    rows = part_a(ckpt, data, data_np, blank, theta_init, n, p, connectivity, gs, batch_eval, directional)

    report()
    report("=" * 100)
    report("SUMMARY")
    report("=" * 100)
    for k in (N_TIME_STEPS - 1, N_TIME_STEPS - 2, N_TIME_STEPS - 3):
        r = rows[k]
        report(f"t={r['t']:.3f}: blank-pixel stiffness {r['stiffness_ratio_median']:.2f}x optimal "
               f"(self {r['self_ratio_median']:.2f}, couplings {r['coupling_ratio_median']:.2f}); uniform background "
               f"mode {r['uniform_dir_ratio']:.2f}x; score error {r['score_rel_err']:.3g}; scale test R "
               f"{r['R_scale_all_pixels']:.3f}")
    lag = rows[-1]["stiffness_ratio_median"]
    if lag < 0.7:
        verdict = ("the near-data slices LAG: background stiffness at t=0 is well below its optimum. Retraining the "
                   "last slices with a larger, annealed learning rate is worth trying.")
    elif lag < 0.9:
        verdict = "a MILD lag at t=0; retraining the last slices may help a little."
    elif lag <= 1.1:
        verdict = "NO lag for background pixels at t=0: the model reached the optimal stiffness there."
    else:
        verdict = "background pixels are OVER-stiff at t=0 (above the optimum)."
    report(f"Verdict (background pixels): {verdict}")
    data_row = part_b_rows.get("DATA")
    if data_row:
        for label in ("SDE/raw/linear", "ODE/raw/linear", "SDE/raw/pchip", "ODE/raw/pchip"):
            r = part_b_rows.get(label)
            if r:
                report(f"{label}: background noise within samples {r['bg_std_within_sample_median'] / data_row['bg_std_within_sample_median']:.2f}x "
                       f"the data's; background offset {r['bg_offset_abs_median']:.3f}; out-of-range pixels "
                       f"{100 * r['frac_pixels_out_of_range']:.2f}%; R(theta0) {fmt(r['R_scale_theta0'])}")
    report(f"total time {time.time() - start:.0f} s")
    save_report()


if __name__ == "__main__":
    main()
