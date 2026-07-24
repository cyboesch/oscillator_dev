"""Denoising-score-matching (DSM) variant of the convex-optimization physical diffusion model.

Same lattice-of-Duffing-oscillators energy and the same matrix-free ridge/prox conjugate-gradient
solver as physical_diffusion_model_MNIST.py, but the per-slice objective is DENOISING score
matching instead of Hyvarinen implicit SM. Motivation: implicit SM matched each mode's local score
but got the RELATIVE mass of the well-separated 0 / 1 modes badly wrong (learned energy put real 0s
~8663 kT above real 1s -> generation was ~8% zeros / 92% ones). DSM regresses the model score onto
the analytic forward-kernel score, which pins down the relative mode weights.

Because the energy is linear in theta, the DSM loss is still an exact convex quadratic in theta with
the SAME Hessian as implicit SM (only the linear term differs), so the whole solver stack — exact
diagonal preconditioner, ridge_vec, prox chain, KKT active set, chunked gradient — is reused
unchanged.

DSM's regression target -(x_t - m_t x0)/v_t diverges as t->0, so the time grid is capped at a small
t_min (the forward noise floor) and the reverse SDE stops there. That is standard for diffusion
models and is exactly the noise floor we established earlier.

Resume/nohup notes carried over from the ISM driver: output dir is date-stamped (RUN_DATE pins it),
start_from_scratch=False resumes if the dir exists, CG_CHUNK=512 to fit the 12 GB card.
"""
print("Script started (DSM).")
import os
import sys
import gc
import shutil
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import flatten_util, vmap
import jax.numpy as jnp
import jax.random as jr

from physical_diffusion_fns.helper_fns import (
    normalize_samples,
    sample_forward_process,
    sample_forward_pairs,
    interpolate_parameters,
)
from physical_diffusion_fns.learning_fns import solve_score_matching_cg, exact_score_matching_diag, make_ggn_hvp
from physical_diffusion_fns.plotting_fns import (
    plot_parameter_as_fn_of_time,
    visualize_connectivity_with_non_local_couplings,
)
from physical_diffusion_fns.network_fns import (
    setup_energy_fn,
    setup_overdamped_SDE,
    solve_SDE,
    create_2d_square_lattice_connectivity,
)
from physical_diffusion_fns.network_fns_memory_efficient import (
    setup_duffing_network_analytical_derivatives_memory_efficient,
)
from physical_diffusion_fns.learning_fns_memory_efficient import (
    setup_denoising_score_matching_loss_chunked,
)

jax.config.update("jax_enable_x64", True)


def print_memory_usage(label=""):
    rss = psutil.Process(os.getpid()).memory_info().rss
    print(f"Memory usage {label}: {rss / 1024 / 1024:.2f} MB")


def optimize_memory():
    gc.collect()
    jax.clear_caches()


print_memory_usage("at start")
print(f"Number of devices: {jax.local_device_count()}")

#####################################
# Configuration
#####################################
key_seed = 1
key_seed_reverse = None
master_key = jax.random.PRNGKey(key_seed)
optimization_key, _default_reverse_sde_key, image_noise_added_key = jr.split(master_key, 3)
reverse_sde_key = jax.random.PRNGKey(key_seed_reverse) if key_seed_reverse is not None else _default_reverse_sde_key

labels = [0, 1]
resolution = (28, 28)
balanced = True
std_of_added_noise = 0.01
additional_rescaling = 1

Temp = 0.02
sigma_forward = 1.0
t_forward = 5.0
n_time_steps = 100

# DSM grid is capped at t_min (>0): the DSM target -(x_t - m_t x0)/v_t diverges as t->0.
# 30 points log-spaced in the noise variance s(t) on [t_min, 0.5], 70 uniform on (0.5, 5].
t_min = float(os.environ.get("T_MIN", 0.01))
_a = std_of_added_noise**2 + Temp * sigma_forward**2 * (1 - np.exp(-2 * t_min / sigma_forward**2))  # s(t_min)
_a = float(np.clip(_a, 1e-6, Temp * sigma_forward**2 * 0.999))
_b = Temp * sigma_forward**2
_t_of_s = lambda sv: -0.5 * sigma_forward**2 * jnp.log((_b - sv) / (_b - _a))
_s_low = jnp.geomspace(_a, _a * jnp.exp(-1.0) + _b * (1.0 - jnp.exp(-1.0)), 30)
_t_low = _t_of_s(_s_low).at[0].set(t_min)
_t_high = jnp.linspace(0.5, t_forward, n_time_steps - 29)[1:]
forward_time_pts = jnp.concatenate([_t_low, _t_high])[::-1]
print("forward_time_pts (DSM, capped at t_min):", forward_time_pts)

# --- Convex solver knobs (same regularization strategy as the ISM v2 driver) ---
cg_sample_size = int(os.environ.get("CG_SAMPLE_SIZE", 8192))
cg_heldout_size = int(os.environ.get("CG_HELDOUT_SIZE", cg_sample_size))
cg_ridge_rel = float(os.environ.get("CG_RIDGE_REL", 0.0))
cg_prox_abs = float(os.environ.get("CG_PROX_ABS", 20.0))
cg_group_ridge = float(os.environ.get("CG_GROUP_RIDGE", 2.0))
cg_chunk = int(os.environ.get("CG_CHUNK", 512))   # GGN Hv fits at 512 on the 12 GB card
cg_maxiter = int(os.environ.get("CG_MAXITER", 500))
cg_maxiter_smallt = int(os.environ.get("CG_MAXITER_SMALLT", 1500))
max_slices = int(os.environ.get("MAX_SLICES", 0))
cg_tol = 1e-5
cg_active_set_iters = 8
cg_gate_rel = 0.01
heldout_image_fraction = 0.10
k6_floor = 0.01
uniform_rate_hard_fail = -0.5
uniform_rate_warn = 0.3

n_neighbour_couplings = 14

n_trajectories = 40
rtol_sde = 1e-4
atol_sde = 1e-6
brownian_tolerance = 1e-12

start_from_scratch = False

training_method = "SM_at_kbT_DSM_convex_cg_v2"   # distinct from ISM -> separate output dir
energy_fn_type = "6th_order_duffing_coupling"

#####################################
# Output directories (RUN_DATE pins the date so a later-day rerun resumes the same checkpoint)
#####################################
problem_type_folder = "MNIST_generation"
system_specifics = f"n_couplings_{n_neighbour_couplings}_Temp_{Temp}_energy_fn_type_{energy_fn_type}"
MNIST_specifics = f"labels_{labels}_resolution_{resolution[0]}_x_{resolution[1]}"
data_folder = f"added_gaussian_noise_std_{std_of_added_noise}_key_seed_{key_seed}_additional_rescaling_{additional_rescaling}"
optimization_folder = (
    f"training_method_{training_method}_solve_opt_in_reverse_True_"
    f"t_forward_{t_forward}_t_min_{t_min}_"
    f"n_timesteps_{n_time_steps}_"
    f"sigma_forward_{sigma_forward}_"
    f"cg_sample_{cg_sample_size}_"
    f"ridge_rel_{cg_ridge_rel}_prox_abs_{cg_prox_abs}_gridge_{cg_group_ridge}_"
    f"heldout_{heldout_image_fraction}_cg_tol_{cg_tol}_cg_maxiter_{cg_maxiter}_{cg_maxiter_smallt}_"
)

here = os.path.dirname(os.path.abspath(__file__))
current_date = os.environ.get("RUN_DATE", datetime.now().strftime("%Y_%m_%d"))
base_dir = os.path.join(here, "..", "out", current_date, "problems")
output_dir_root = os.path.join(base_dir, problem_type_folder, MNIST_specifics, system_specifics, data_folder, optimization_folder)
output_dir = os.path.join(output_dir_root, "opt_per_time_plots")
plot_folder = os.path.join(output_dir_root, "final_plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)
print(f"Output directory: {output_dir}")

#####################################
# Load MNIST data
#####################################
path_to_data = os.path.join(here, "..", "data", "MNIST", f"mnist_labels_{labels}_resolution_{resolution}_balanced_{balanced}.npy")
images_flat_raw = jnp.array(np.load(path_to_data))
n_samples = images_flat_raw.shape[0]
print("Data shape:", images_flat_raw.shape)

_, subkey_image_noise_added = jr.split(image_noise_added_key)
images_flat_true = images_flat_raw + jr.normal(subkey_image_noise_added, images_flat_raw.shape) * std_of_added_noise
samples_target_unscaled, mean_MNIST, std_MNIST = normalize_samples(images_flat_true)
pixel_min_normalized = (0.0 - mean_MNIST) / std_MNIST
pixel_max_normalized = (1.0 - mean_MNIST) / std_MNIST
samples_target = samples_target_unscaled * additional_rescaling

#####################################
# Network
#####################################
N_osc = resolution[0] ** 2
connectivity = create_2d_square_lattice_connectivity(grid_size=resolution[0], n_neighbour_couplings=n_neighbour_couplings)
num_connections = connectivity.shape[0]
visualize_connectivity_with_non_local_couplings(
    connectivity, grid_size_x=resolution[0], grid_size_y=resolution[1],
    n_neighbour_couplings=n_neighbour_couplings, save_fig=True, path=output_dir,
)
print(f"Network size: {N_osc} oscillators, {num_connections} connections")

# Analytic Gaussian anchor at t=t_forward (same as ISM v2), biases LAST in the layout.
k_lin_0 = jnp.full(N_osc, 1.0 / (1.0 - jnp.exp(-2.0 * t_forward / sigma_forward**2)))
k_duff_0 = jnp.zeros(N_osc)
k_6_0 = jnp.full(N_osc, k6_floor)
c_lin_0 = jnp.zeros(num_connections)
c_optomech_0 = jnp.zeros(num_connections)
c_duff_0 = jnp.zeros(num_connections)
biases_0 = jnp.zeros(N_osc)
params_initial = (k_lin_0, k_duff_0, k_6_0, c_lin_0, c_optomech_0, c_duff_0, biases_0)
params_names = ["k_lin", "k_duff", "k_6", "c_lin", "c_optomech", "c_duff", "biases"]
params_flattened_initial, unflatten = flatten_util.ravel_pytree(params_initial)
constraint_indices = jnp.arange(2 * N_osc, 3 * N_osc)

ridge_vec, _ = flatten_util.ravel_pytree((
    jnp.zeros(N_osc), cg_group_ridge * jnp.ones(N_osc), jnp.zeros(N_osc),
    cg_group_ridge * jnp.ones(num_connections), cg_group_ridge * jnp.ones(num_connections),
    cg_group_ridge * jnp.ones(num_connections), jnp.zeros(N_osc),
))

energy_fn = setup_energy_fn(connectivity, unflatten)
gradient_fn, trace_hessian_fn, _, _ = setup_duffing_network_analytical_derivatives_memory_efficient(connectivity, unflatten)
print(f"Total parameters: {len(params_flattened_initial)}")

#####################################
# DSM objective (chunked-vmap gradient); batch is a (x_t, target) tuple
#####################################
loss_fn_per_batch, gradient_fn_per_batch = setup_denoising_score_matching_loss_chunked(
    gradient_fn, k_b=1.0, T=Temp, chunk=cg_chunk,
)
print(f"Denoising score matching (chunked-vmap, chunk={cg_chunk}) at k_bT = {Temp}")

#####################################
# Optimization loop (resume-aware)
#####################################
params_history_path = f"{output_dir}/params_history.npy"
time_index_path = f"{output_dir}/current_time_index.txt"

if start_from_scratch:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir); os.makedirs(output_dir, exist_ok=True); os.makedirs(plot_folder, exist_ok=True)
    start_t_idx = 0; params_history_all_t = []; current_params = params_flattened_initial
elif os.path.exists(params_history_path):
    full_params_history = jnp.load(params_history_path)
    start_t_idx = int(full_params_history.shape[0])
    params_history_all_t = [full_params_history[i] for i in range(start_t_idx)]
    current_params = params_history_all_t[-1] if start_t_idx > 0 else params_flattened_initial
    print(f"Resuming from time index {start_t_idx} ({len(params_history_all_t)} parameter sets loaded)")
else:
    start_t_idx = 0; params_history_all_t = []; current_params = params_flattened_initial

# Disjoint image split for the held-out diagnostic.
_perm = jr.permutation(jr.fold_in(master_key, 12345), samples_target.shape[0])
_n_hold = int(round(heldout_image_fraction * samples_target.shape[0]))
samples_fit = samples_target[_perm[_n_hold:]]
samples_heldout_src = samples_target[_perm[:_n_hold]]
print(f"Image split: {samples_fit.shape[0]} fit / {samples_heldout_src.shape[0]} held-out")

cg_diag = {k: [] for k in [
    "t", "loss_fit", "loss_heldout", "free_residual", "rhs_norm", "accepted", "n_active",
    "active_set_iters", "lambda_min", "uniform_rate", "mean_klin", "mean_abs_clin", "solve_seconds",
]}

_connectivity_np = np.asarray(connectivity)

def quadratic_sector_lambda_min(theta):
    th = np.asarray(theta)
    H = np.diag(th[:N_osc].astype(np.float64))
    c_lin = th[3 * N_osc:3 * N_osc + num_connections].astype(np.float64)
    ii, jj = _connectivity_np[:, 0], _connectivity_np[:, 1]
    np.add.at(H, (ii, ii), 2.0 * c_lin); np.add.at(H, (jj, jj), 2.0 * c_lin)
    np.add.at(H, (ii, jj), -2.0 * c_lin); np.add.at(H, (jj, ii), -2.0 * c_lin)
    return float(np.linalg.eigvalsh(H)[0])

slices_this_session = 0
if start_t_idx < len(forward_time_pts):
    for t_idx in range(start_t_idx, len(forward_time_pts)):
        t_curr = forward_time_pts[t_idx]
        optimization_key, subkey_sample, subkey_solve, subkey_heldout = jr.split(optimization_key, 4)

        # DSM fit batch: (x_t, target) pairs from the forward kernel.
        x_t, target = sample_forward_pairs(
            t_curr, cg_sample_size, D=Temp, sigma_final=sigma_forward, samples0=samples_fit, key=subkey_sample, beta=1,
        )
        batch = (x_t, target)
        diag_exact = exact_score_matching_diag(x_t, connectivity, N_osc, k_b=1.0, T=Temp)
        # Memory-light Gauss-Newton H-vector product (the default jvp-through-DSM-gradient OOMs the
        # 12 GB card at usable chunks); c = -g(0) still comes from gradient_fn_per_batch (DSM).
        hvp = make_ggn_hvp(gradient_fn, x_t, k_b=1.0, T=Temp, chunk=cg_chunk)

        t_solve0 = time.time()
        mi = cg_maxiter_smallt if float(t_curr) < 1.0 else cg_maxiter
        solve_kwargs = dict(
            ridge_rel=cg_ridge_rel, ridge_vec=ridge_vec, prox_abs=cg_prox_abs, prox_center=current_params,
            diag_exact=diag_exact, hvp_fn=hvp, constraint_indices=constraint_indices, epsilon=k6_floor,
            cg_tol=cg_tol, active_set_iters=cg_active_set_iters, loss_fn_per_batch=loss_fn_per_batch, key=subkey_solve,
        )
        theta, info = solve_score_matching_cg(gradient_fn_per_batch, current_params, batch, cg_maxiter=mi, **solve_kwargs)
        if info["residual"] > cg_gate_rel * info["rhs_norm"]:
            print(f"  gate: residual {info['residual']:.2e} > {cg_gate_rel}*||rhs|| {info['rhs_norm']:.2e}; retry {4*mi} iters")
            theta, info = solve_score_matching_cg(gradient_fn_per_batch, theta, batch, cg_maxiter=4 * mi, **solve_kwargs)
        accepted = info["residual"] <= cg_gate_rel * info["rhs_norm"]
        solve_seconds = time.time() - t_solve0

        lam_min = quadratic_sector_lambda_min(theta)
        theta_uniform_rate = 2.0 * float(jnp.mean(theta[0:N_osc])) - 1.0 / sigma_forward**2
        if accepted and theta_uniform_rate <= uniform_rate_hard_fail:
            raise RuntimeError(
                f"Slice {t_idx} (t={float(t_curr):.4f}): reverse uniform-mode rate {theta_uniform_rate:+.3f} "
                f"<= {uniform_rate_hard_fail} — DC collapse, aborting."
            )
        if accepted and theta_uniform_rate < uniform_rate_warn:
            print(f"  stability WARNING: uniform-mode reverse rate {theta_uniform_rate:+.3f} < {uniform_rate_warn}; lam_min={lam_min:.3e}")

        if accepted:
            current_params = theta
        else:
            print(f"  WARNING slice {t_idx} (t={float(t_curr):.4f}) NOT ACCEPTED (residual {info['residual']:.2e}); keeping previous.")

        # Held-out DSM loss on unseen images.
        xh, th = sample_forward_pairs(
            t_curr, cg_heldout_size, D=Temp, sigma_final=sigma_forward, samples0=samples_heldout_src, key=subkey_heldout, beta=1,
        )
        heldout_loss = float(loss_fn_per_batch(current_params, (xh, th)))
        loss_fit = float(loss_fn_per_batch(current_params, batch))
        mean_klin = float(jnp.mean(current_params[0:N_osc]))
        mean_abs_clin = float(jnp.mean(jnp.abs(current_params[3 * N_osc:3 * N_osc + num_connections])))
        uniform_rate = 2.0 * mean_klin - 1.0 / sigma_forward**2
        print(
            f"t_idx {t_idx:3d}  t={float(t_curr):.3f}  loss_fit {loss_fit:.3e}  loss_heldout {heldout_loss:.3e}  "
            f"resid/rhs={info['residual']/max(info['rhs_norm'],1e-30):.2e}  accepted={accepted}  "
            f"k6_floor={info['n_active']}/{N_osc}  mean_klin={mean_klin:.4f}  unif_rate={uniform_rate:+.3f}  "
            f"lam_min={lam_min:.3e}  mean|c_lin|={mean_abs_clin:.2e}  {solve_seconds:.0f}s"
        )
        for k, v in [
            ("t", float(t_curr)), ("loss_fit", loss_fit), ("loss_heldout", heldout_loss),
            ("free_residual", info["residual"]), ("rhs_norm", info["rhs_norm"]), ("accepted", bool(accepted)),
            ("n_active", info["n_active"]), ("active_set_iters", info["active_set_iters_used"]),
            ("lambda_min", lam_min), ("uniform_rate", uniform_rate), ("mean_klin", mean_klin),
            ("mean_abs_clin", mean_abs_clin), ("solve_seconds", solve_seconds),
        ]:
            cg_diag[k].append(v)

        params_history_all_t.append(current_params)
        optimize_memory()
        _tmp = params_history_path + ".tmp.npy"
        np.save(_tmp, np.array(params_history_all_t)); os.replace(_tmp, params_history_path)
        with open(time_index_path, "w") as f:
            f.write(str(t_idx + 1))

        slices_this_session += 1
        if max_slices and slices_this_session >= max_slices:
            print(f"MAX_SLICES={max_slices} reached; stopping after slice {t_idx} (resumable).")
            break

    if len(params_history_all_t) == len(forward_time_pts):
        print("DSM optimization complete; saved parameters to", output_dir)

if cg_diag["t"]:
    np.savez(f"{output_dir}/dsm_diagnostics_from_{start_t_idx}.npz", **{k: np.array(v) for k, v in cg_diag.items()})

if len(params_history_all_t) < len(forward_time_pts):
    print(f"Only {len(params_history_all_t)}/{len(forward_time_pts)} slices trained (smoke run) — skipping generation.")
    sys.exit(0)

params_history_all_t = jnp.array(params_history_all_t)

#####################################
# Interpolate + reverse SDE (stops at t_min)
#####################################
params_interpolator = interpolate_parameters(params_history_all_t, forward_time_pts)
plot_parameter_as_fn_of_time(
    params_names, forward_time_pts, forward_time_pts, params_history_all_t,
    params_interpolator, unflatten, N_osc, save_fig=True, path=plot_folder, log_scale=False,
)

# Confinement alarm on the assembled reverse potential (same check as ISM v2).
_th = np.asarray(params_history_all_t)
_k6_rev = 2.0 * _th[:, 2 * N_osc:3 * N_osc]
_c4_rev = 2.0 * _th[:, 3 * N_osc + 2 * num_connections:3 * N_osc + 3 * num_connections]
_cn = np.asarray(connectivity)
_u = np.sqrt(16.0 * np.maximum(-_c4_rev, 0.0) / np.maximum(_k6_rev[:, _cn[:, 0]] + _k6_rev[:, _cn[:, 1]], 1e-12))
print(f"Confinement: min k6_rev={_k6_rev.min():.4f}, max escape radius u*={_u.max():.2f}")
if _k6_rev.min() <= 0:
    raise RuntimeError("Reverse potential unbounded below (k6_rev<=0).")

def run_reverse_process_SDE(params_interpolator, suffix, key):
    """Reverse-time Langevin SDE, integrated from the noise end down to forward-time t_min."""
    tau = lambda t: t_forward - t
    forward_params = jnp.zeros_like(params_flattened_initial).at[0:N_osc].set(1.0)
    params_reverse = lambda t: 2 * params_interpolator(tau(t)) - forward_params / sigma_forward**2
    drift_fn, diffusion_fn = setup_overdamped_SDE(energy_fn, params_reverse, N_osc, time_dependent_parms=True, Temp=Temp)

    t0, t1 = 0.0, t_forward - t_min          # stop at forward-time t_min
    ts = jnp.linspace(t0, t1, 100)
    key, subkey_init = jr.split(key)
    initial_states = jnp.sqrt(Temp) * sigma_forward * jr.normal(subkey_init, shape=(n_trajectories, N_osc))
    key, subkey_brownian = jr.split(key)
    keys_brownian = jr.split(subkey_brownian, n_trajectories)
    print(f"Running reverse SDE for {n_trajectories} trajectories (to t_min={t_min})...")
    solve_one = lambda init_state, key_b: solve_SDE(
        drift_fn, diffusion_fn, init_state, key_b, t0, t1, ts.shape[0], 1e-8,
        brownian_tolerance=brownian_tolerance, rtol=rtol_sde, atol=atol_sde,
    )
    solutions = vmap(solve_one, in_axes=(0, 0))(initial_states, keys_brownian)
    final_samples_scaled = solutions.ys[:, -1, :]
    jnp.save(f"{output_dir}/final_states_{suffix}.npy", final_samples_scaled)
    del solutions; optimize_memory()
    return (final_samples_scaled / additional_rescaling).reshape(-1, resolution[0], resolution[1])

reverse_suffix = f"dsm_tmin_{t_min}_atol_{atol_sde}_rtol_{rtol_sde}"
images_generated = run_reverse_process_SDE(params_interpolator, reverse_suffix, reverse_sde_key)

#####################################
# Plot true vs generated
#####################################
def plot_true_vs_generated_grid(true_imgs, generated_imgs, generated_title, save_path, clip=None):
    n_show = n_trajectories; ncol = min(10, n_show); n_rows = max(1, -(-n_show // ncol))
    fig, axes = plt.subplots(2 * n_rows, ncol, figsize=(1.5 * ncol, 3 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for idx in range(n_show):
        for row_block, imgs in ((0, true_imgs), (n_rows, generated_imgs)):
            ax = axes[idx // ncol + row_block, idx % ncol]
            img = np.array(imgs[idx])
            if clip:
                img = np.clip(img, *clip)
            ax.imshow(img, cmap="gray")
    axes[0, 0].set_title("True images", loc="left", fontsize=16)
    axes[n_rows, 0].set_title(generated_title, loc="left", fontsize=16)
    plt.subplots_adjust(wspace=0.01, hspace=0.5)
    plt.savefig(save_path); plt.close()

images_true = images_flat_true.reshape(-1, resolution[0], resolution[1])
plot_true_vs_generated_grid(images_true, images_generated, f"DSM generated (t_min={t_min})", f"{plot_folder}/samples_{reverse_suffix}.png")
clip = (float(pixel_min_normalized), float(pixel_max_normalized))
plot_true_vs_generated_grid(
    np.clip(np.array(samples_target_unscaled.reshape(-1, resolution[0], resolution[1])), *clip),
    images_generated, f"DSM generated clipped (t_min={t_min})", f"{plot_folder}/samples_{reverse_suffix}_clipped.png", clip=clip,
)
print("DSM script completed successfully!")
