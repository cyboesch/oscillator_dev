print("Script started.")
import os
import sys
import gc
import shutil
import time
from datetime import datetime

# Make the physical_diffusion_fns package importable (parent dir of scripts/)
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
    smooth_parameters,
    interpolate_parameters,
)
from physical_diffusion_fns.learning_fns import solve_score_matching_cg, exact_score_matching_diag
from physical_diffusion_fns.plotting_fns import (
    plot_forward_marginals,
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
    setup_score_matching_kbT_loss_chunked,
)

jax.config.update("jax_enable_x64", True)


def print_memory_usage(label=""):
    """Print current resident memory usage."""
    rss = psutil.Process(os.getpid()).memory_info().rss
    print(f"Memory usage {label}: {rss / 1024 / 1024:.2f} MB")


def optimize_memory():
    """Force garbage collection and clear JAX caches."""
    gc.collect()
    jax.clear_caches()


print_memory_usage("at start")
print(f"Number of devices: {jax.local_device_count()}")

#####################################
# Configuration
#####################################
# --- RNG ---
key_seed = 1
key_seed_reverse = None  # None -> derived from master key; set an int for an independent reverse-sampling seed
master_key = jax.random.PRNGKey(key_seed)
optimization_key, _default_reverse_sde_key, image_noise_added_key = jr.split(master_key, 3)
reverse_sde_key = jax.random.PRNGKey(key_seed_reverse) if key_seed_reverse is not None else _default_reverse_sde_key

# --- Data ---
labels = [0, 1]
resolution = (28, 28)
balanced = True
std_of_added_noise = 0.01     # Gaussian noise added to the data for regularization
additional_rescaling = 1      # extra scalar applied to the (normalized) target samples

# --- Forward (noising) Ornstein-Uhlenbeck process ---
Temp = 0.02                   # bath temperature = diffusion constant D
sigma_forward = 1.0
t_forward = 5.0
n_time_steps = 100
# The energy is trained separately at each forward time. We optimize from the noisy
# (large-t) end down to the data (t=0), warm-starting each slice from the previous one,
# so the time points and every per-slice schedule below are stored in reverse order.
#
# Grid spacing (EXP_TIME_PTS=1, default): the marginal noise variance s(t) =
# a e^{-2t} + b (1 - e^{-2t}) (a = added-data-noise variance, b = Temp sigma^2) sets the
# score's stiffness ~ 1/s(t), which changes fastest near t=0. A uniform grid makes the
# first gap span a ~5.7x stiffness change (interpolation there injects ~7.5 spurious
# e-folds of contraction into the reverse SDE's final approach), so we place 30 points
# uniform in ln s(t) on [0, 0.5] and 70 uniform points on (0.5, 5].
exp_time_pts = bool(int(os.environ.get("EXP_TIME_PTS", 1)))
if exp_time_pts:
    _a, _b = std_of_added_noise**2, Temp * sigma_forward**2
    _t_of_s = lambda sv: -0.5 * sigma_forward**2 * jnp.log((_b - sv) / (_b - _a))
    _s_low = jnp.geomspace(_a, _a * jnp.exp(-1.0) + _b * (1.0 - jnp.exp(-1.0)), 30)
    _t_low = _t_of_s(_s_low).at[0].set(0.0)                        # 30 pts on [0, 0.5]
    _t_high = jnp.linspace(0.5, t_forward, n_time_steps - 29)[1:]  # 70 pts on (0.5, 5]
    forward_time_pts = jnp.concatenate([_t_low, _t_high])[::-1]
else:
    forward_time_pts = jnp.linspace(0.0, t_forward, n_time_steps)[::-1]
print("forward_time_pts", forward_time_pts)

# --- Convex solver: matrix-free regularized CG (one exact solve per forward-time slice) ---
# The per-slice implicit-score-matching loss is an exact convex quadratic in theta (the energy is
# linear in theta and all units are visible), so instead of SGD we solve regularized
# normal equations directly and matrix-free (H is never formed; P ~ 5e5).
# See physical_diffusion_fns/learning_fns.py:solve_score_matching_cg.
#
# Regularization (v2 — the 2026_07_10 run showed a GLOBAL ridge-to-zero collapses the on-site
# stiffness k_lin into the ~461-per-site edge couplings, leaving the reverse drift's uniform
# mode unstable at rate ~ -0.9 for the whole generation; blobs, not digits):
#   - PROX CHAIN  (mu/2)||theta - theta_prev||^2 with an ABSOLUTE weight CG_PROX_ABS:
#     pins weakly identified directions to the previous slice (slice 0: analytic Gaussian
#     anchor), self-anneals as data curvature grows toward t=0. The convex analogue of the
#     old warm-started early-stopped SGD.
#   - GROUP RIDGE  CG_GROUP_RIDGE on {k_duff, c_lin, c_optomech, c_duff} only (k_lin, k_6,
#     bias unshrunk): tilts the k_lin<->c_lin degenerate valley toward the physical on-site
#     representation and controls stationary noise in the coupling groups.
#   - GLOBAL ridge CG_RIDGE_REL defaults to 0 — with a zero-centered global ridge the valley
#     fixed point is h/(h+lambda)*theta_true no matter how strong the prox, i.e. the
#     collapse returns. Keep it 0 unless you know why you need it.
# Sweep knobs are env-overridable; each combo lands in its own output directory:
#   CG_PROX_ABS=10 CG_GROUP_RIDGE=0.5 python3 physical_diffusion_model_MNIST.py
cg_sample_size = int(os.environ.get("CG_SAMPLE_SIZE", 8192))          # fixed sample of p_t per slice used to FIT (>~ P/N_osc for full rank)
cg_heldout_size = int(os.environ.get("CG_HELDOUT_SIZE", cg_sample_size))  # independent sample for the held-out-loss diagnostic
cg_ridge_rel = float(os.environ.get("CG_RIDGE_REL", 0.0))            # global ridge lambda = cg_ridge_rel * mean-eig(H); see WARNING above
cg_prox_abs = float(os.environ.get("CG_PROX_ABS", 20.0))             # absolute prox weight mu (window 10-30)
cg_group_ridge = float(os.environ.get("CG_GROUP_RIDGE", 2.0))        # per-coordinate ridge on {k_duff, c_lin, c_optomech, c_duff}
cg_chunk = int(os.environ.get("CG_CHUNK", 1024))                     # vmap chunk for the gradient/Hv (512 if the GPU is shared)
cg_maxiter = int(os.environ.get("CG_MAXITER", 500))                  # CG iteration cap for t >= 1
cg_maxiter_smallt = int(os.environ.get("CG_MAXITER_SMALLT", 1500))   # CG iteration cap for t < 1 (sharper slices)
max_slices = int(os.environ.get("MAX_SLICES", 0))                    # >0: stop after this many slices (smoke test); run resumes later
cg_tol = 1e-5             # conjugate-gradient relative-residual tolerance
cg_active_set_iters = 8   # max active-set passes for the k_6 >= epsilon box constraint
cg_gate_rel = 0.01        # accept a slice only if free residual <= gate * ||rhs|| (retry once at 4x maxiter)
heldout_image_fraction = 0.10  # disjoint image split for the held-out diagnostic (holdout in DATA, not just noise)
k6_floor = 0.01           # epsilon of the k_6 >= epsilon constraint (keeps the potential confining)

# --- Network: 2D lattice of coupled 6th-order Duffing oscillators, one oscillator per pixel ---
n_neighbour_couplings = 14

# --- Reverse (generation) SDE ---
n_trajectories = 40
rtol_sde = 1e-4
atol_sde = 1e-6
brownian_tolerance = 1e-12
# Savitzky-Golay-smoothed theta(t) for generation — OFF by default in v2: the prox chain makes
# theta(t) smooth at the source, savgol is nearly inert at the t=0 endpoint (where it would
# matter most), it silently violates the k_6 >= floor constraint at spikes, and it assumes a
# UNIFORM time grid (incompatible with EXP_TIME_PTS=1). Env-overridable (USE_SMOOTHED_REVERSE=1).
use_smoothed_params_for_reverse = bool(int(os.environ.get("USE_SMOOTHED_REVERSE", 0)))

# --- Checkpointing ---
start_from_scratch = False

# Fixed labels used only for output-directory naming.
training_method = "SM_at_kbT_convex_cg_v2"   # note: still starts with "SM_at_kbT" -> reverse-SDE scaling unchanged
energy_fn_type = "6th_order_duffing_coupling"

#####################################
# Output directories
#####################################
problem_type_folder = "MNIST_generation"
system_specifics = f"n_couplings_{n_neighbour_couplings}_Temp_{Temp}_energy_fn_type_{energy_fn_type}"
MNIST_specifics = f"labels_{labels}_resolution_{resolution[0]}_x_{resolution[1]}"
data_folder = f"added_gaussian_noise_std_{std_of_added_noise}_key_seed_{key_seed}_additional_rescaling_{additional_rescaling}"
# Every knob that changes the OPTIMIZATION must appear here: start_from_scratch=False resumes
# whenever this directory already exists, so a missing token silently continues an old run.
optimization_folder = (
    f"training_method_{training_method}_solve_opt_in_reverse_True_"
    f"exp_time_pts_{exp_time_pts}_"
    f"t_forward_{t_forward}_"
    f"n_timesteps_{n_time_steps}_"
    f"sigma_forward_{sigma_forward}_"
    f"cg_sample_{cg_sample_size}_"
    f"ridge_rel_{cg_ridge_rel}_"
    f"prox_abs_{cg_prox_abs}_"
    f"gridge_{cg_group_ridge}_"
    f"heldout_{heldout_image_fraction}_"
    f"cg_tol_{cg_tol}_"
    f"cg_maxiter_{cg_maxiter}_{cg_maxiter_smallt}_"
)

here = os.path.dirname(os.path.abspath(__file__))
current_date = datetime.now().strftime("%Y_%m_%d")
base_dir = os.path.join(here, "..", "out", current_date, "problems")
output_dir_root = os.path.join(base_dir, problem_type_folder, MNIST_specifics, system_specifics, data_folder, optimization_folder)
output_dir = os.path.join(output_dir_root, "opt_per_time_plots")
plot_folder = os.path.join(output_dir_root, "final_plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)
print(f"Output directory: {output_dir}")
print(f"Plot directory: {plot_folder}")

#####################################
# Load MNIST data
#####################################
path_to_data = os.path.join(here, "..", "data", "MNIST", f"mnist_labels_{labels}_resolution_{resolution}_balanced_{balanced}.npy")
images_flat_raw = jnp.array(np.load(path_to_data))
n_samples = images_flat_raw.shape[0]
print("Data shape:", images_flat_raw.shape)

# Add small Gaussian noise for regularization, then standardize to zero-mean / unit-variance.
_, subkey_image_noise_added = jr.split(image_noise_added_key)
images_flat_true = images_flat_raw + jr.normal(subkey_image_noise_added, images_flat_raw.shape) * std_of_added_noise
samples_target_unscaled, mean_MNIST, std_MNIST = normalize_samples(images_flat_true)
# Original pixel range [0, 1] mapped through the normalization (x - mean) / std
pixel_min_normalized = (0.0 - mean_MNIST) / std_MNIST
pixel_max_normalized = (1.0 - mean_MNIST) / std_MNIST
print(f"Normalized pixel range: [{pixel_min_normalized:.4f}, {pixel_max_normalized:.4f}]")
samples_target = samples_target_unscaled * additional_rescaling

#####################################
# Setup network (energy function + analytical derivatives)
#####################################
N_osc = resolution[0] ** 2
connectivity = create_2d_square_lattice_connectivity(grid_size=resolution[0], n_neighbour_couplings=n_neighbour_couplings)
num_connections = connectivity.shape[0]
visualize_connectivity_with_non_local_couplings(
    connectivity, grid_size_x=resolution[0], grid_size_y=resolution[1],
    n_neighbour_couplings=n_neighbour_couplings, save_fig=True, path=output_dir,
)
print(f"Network size: {N_osc} oscillators, {num_connections} connections")

# Initial energy parameters = the ANALYTIC stationary solution of the forward OU process at
# t = t_forward: p ~ N(0, Temp*sigma^2*(1-e^{-2t/sigma^2})), i.e. E/kT = x^2/(2 s(t)) so
# k_lin = 1/(1-e^{-2 t_forward/sigma^2}) ~ 1.0000454 and every other group is 0 (k_6 sits at
# its confining floor). This is both the slice-0 warm start and the slice-0 prox center: the
# prox chain then carries the physically-extrapolating representation down through the
# stationary tail instead of letting the solver pick a minimum-norm one.
# Layout (flattened by ravel_pytree in this order):
#   k_lin[N], k_duff[N], k_6[N], c_lin[E], c_optomech[E], c_duff[E], biases[N]  (biases LAST)
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
# Keep the k_6 (sextic) self-coefficients >= epsilon during training so the potential stays confining.
constraint_indices = jnp.arange(2 * N_osc, 3 * N_osc)

# Per-coordinate zero-centered ridge: shrink only the groups whose true value is ~0 in the
# Gaussian tail ({k_duff, c_lin, c_optomech, c_duff}); k_lin (the physical stiffness carrier),
# k_6 (floor-constrained) and biases stay unshrunk. Built through ravel_pytree so it can never
# drift out of sync with the parameter layout.
ridge_vec, _ = flatten_util.ravel_pytree((
    jnp.zeros(N_osc),                            # k_lin
    cg_group_ridge * jnp.ones(N_osc),            # k_duff
    jnp.zeros(N_osc),                            # k_6
    cg_group_ridge * jnp.ones(num_connections),  # c_lin
    cg_group_ridge * jnp.ones(num_connections),  # c_optomech
    cg_group_ridge * jnp.ones(num_connections),  # c_duff
    jnp.zeros(N_osc),                            # biases
))

energy_fn = setup_energy_fn(connectivity, unflatten)
# The minimal-memory score-matching loss only needs the energy gradient and Hessian trace;
# it obtains the parameter gradient by autodiff, so the analytical param-Jacobians are unused.
gradient_fn, trace_hessian_fn, _, _ = setup_duffing_network_analytical_derivatives_memory_efficient(connectivity, unflatten)
print(f"Total parameters: {len(params_flattened_initial)}")
print_memory_usage("after network setup")

#####################################
# Learning objective: implicit score matching at temperature Temp (minimal-memory autodiff)
#####################################
loss_fn_per_batch, gradient_fn_per_batch = setup_score_matching_kbT_loss_chunked(
    gradient_fn, trace_hessian_fn, k_b=1.0, T=Temp, chunk=cg_chunk,
)
print(f"Score matching (chunked-vmap autodiff, chunk={cg_chunk}) at k_bT = {Temp}")

#####################################
# Optimization loop (with resume-from-checkpoint support)
#####################################
params_history_path = f"{output_dir}/params_history.npy"
time_index_path = f"{output_dir}/current_time_index.txt"

if start_from_scratch:
    print("Starting from scratch")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plot_folder, exist_ok=True)
    start_t_idx = 0
    params_history_all_t = []
    current_params = params_flattened_initial
elif os.path.exists(params_history_path):
    print("Loading existing parameters from", params_history_path)
    full_params_history = jnp.load(params_history_path)
    # Completed slices == rows in the history (the .txt is advisory only: deriving the index
    # from the data itself makes a crash between the two checkpoint writes harmless).
    start_t_idx = int(full_params_history.shape[0])
    if os.path.exists(time_index_path):
        with open(time_index_path, "r") as f:
            txt_idx = int(f.read())
        if txt_idx != start_t_idx:
            print(f"WARNING: time index file says {txt_idx} but history has {start_t_idx} rows; trusting the history.")
    params_history_all_t = [full_params_history[i] for i in range(start_t_idx)]
    current_params = params_history_all_t[-1] if start_t_idx > 0 else params_flattened_initial
    print(f"Resuming from time index {start_t_idx} ({len(params_history_all_t)} parameter sets loaded)")
else:
    print("No existing parameters found, starting from beginning")
    start_t_idx = 0
    params_history_all_t = []
    current_params = params_flattened_initial

# Disjoint image split for the held-out diagnostic: holding out only forward NOISE (as before)
# nearly coincides with the fit batch at small t, hiding true overfitting exactly where p_t
# approaches the data. The fit never sees the held-out images.
_perm = jr.permutation(jr.fold_in(master_key, 12345), samples_target.shape[0])
_n_hold = int(round(heldout_image_fraction * samples_target.shape[0]))
samples_fit = samples_target[_perm[_n_hold:]]
samples_heldout_src = samples_target[_perm[:_n_hold]]
print(f"Image split: {samples_fit.shape[0]} fit / {samples_heldout_src.shape[0]} held-out")

# Per-slice diagnostics accumulated this session.
cg_diag = {k: [] for k in [
    "t", "loss_fit", "loss_heldout", "free_residual", "rhs_norm", "accepted", "n_active",
    "active_set_iters", "lambda_min", "uniform_rate", "mean_klin", "mean_abs_clin", "solve_seconds",
]}

# Stability guard. The ONLY mode that must never go unstable in the reverse drift is the
# uniform (DC) mode: the c_lin edge couplings are a graph Laplacian (blind to the uniform
# mode), so the uniform-mode forward stiffness is exactly mean(k_lin) and its reverse rate is
# 2*mean(k_lin) - 1/sigma^2. A negative uniform rate is the blob failure (old run: -0.91).
# Non-uniform eigenvalues of the forward Hessian at x=0 MAY go negative as t -> 0: that is the
# 0-vs-1 class-separation direction developing a double well (negative curvature at the origin
# between the two digit clusters), which is exactly what the reverse SDE needs to split samples
# into 0s and 1s. We therefore do NOT require full positive-definiteness (an earlier version did
# and aborted at t~2.8 on this physical structure), and we do NOT lift k_lin (that would erase
# the class-separation mode). lambda_min is kept purely as a logged diagnostic.
uniform_rate_hard_fail = 0.0   # abort if reverse uniform-mode rate 2*mean(k_lin)-1/sig^2 <= this (real DC collapse)
uniform_rate_warn = 0.3        # warn below this (healthy tail ~ +1.0; structure-region min ~ +0.88)

_connectivity_np = np.asarray(connectivity)  # (E, 2) edge index pairs, host-side, built once

def quadratic_sector_lambda_min(theta):
    """Smallest eigenvalue of the 784x784 Hessian of E at x=0. Assembled analytically on the
    host: at the origin only the quadratic terms survive, so H = diag(k_lin) + weighted graph
    Laplacian with edge weight 2*c_lin (the c_optomech x^2 y and c_duff (x-y)^4 couplings are
    cubic/quartic and contribute nothing to the Hessian at x=0). Cheap and GPU-free — the
    autodiff Hessian vmaps 784 basis vectors through the 180810-edge network and OOMs."""
    th = np.asarray(theta)
    H = np.diag(th[:N_osc].astype(np.float64))
    c_lin = th[3 * N_osc:3 * N_osc + num_connections].astype(np.float64)
    ii, jj = _connectivity_np[:, 0], _connectivity_np[:, 1]
    np.add.at(H, (ii, ii), 2.0 * c_lin)
    np.add.at(H, (jj, jj), 2.0 * c_lin)
    np.add.at(H, (ii, jj), -2.0 * c_lin)
    np.add.at(H, (jj, ii), -2.0 * c_lin)
    return float(np.linalg.eigvalsh(H)[0])

slices_this_session = 0
if start_t_idx < len(forward_time_pts):
    for t_idx in range(start_t_idx, len(forward_time_pts)):
        t_curr = forward_time_pts[t_idx]
        optimization_key, subkey_sample, subkey_solve, subkey_heldout = jr.split(optimization_key, 4)

        # On the first slice, sanity-check that the forward process reaches the target Gaussian.
        if t_idx == 0:
            samples_0 = sample_forward_process(t_forward, n_samples, D=Temp, sigma_final=sigma_forward, samples0=samples_target, key=optimization_key)
            plot_forward_marginals(samples_0, t_forward, sigma_forward, Temp=Temp, beta=1.0, path=plot_folder, save_fig=True, fontsize=16, plot_show=True)

        # One large fixed sample of the noised marginal p_{t_curr}, held constant across the CG solve.
        batch = sample_forward_process(
            t_curr, n_samples=cg_sample_size, D=Temp, sigma_final=sigma_forward, samples0=samples_fit, key=subkey_sample, beta=1,
        )

        # Exact per-coordinate diag(H) (from batch moments): Jacobi preconditioner + scale.
        diag_exact = exact_score_matching_diag(batch, connectivity, N_osc, k_b=1.0, T=Temp)

        # Convex solve, prox-chained to the previous ACCEPTED slice. current_params is both the
        # CG warm start and the prox center (slice 0: the analytic Gaussian anchor).
        t_solve0 = time.time()
        mi = cg_maxiter_smallt if float(t_curr) < 1.0 else cg_maxiter
        solve_kwargs = dict(
            ridge_rel=cg_ridge_rel,
            ridge_vec=ridge_vec,
            prox_abs=cg_prox_abs,
            prox_center=current_params,
            diag_exact=diag_exact,
            constraint_indices=constraint_indices,
            epsilon=k6_floor,
            cg_tol=cg_tol,
            active_set_iters=cg_active_set_iters,
            loss_fn_per_batch=loss_fn_per_batch,
            key=subkey_solve,
        )
        theta, info = solve_score_matching_cg(gradient_fn_per_batch, current_params, batch, cg_maxiter=mi, **solve_kwargs)
        if info["residual"] > cg_gate_rel * info["rhs_norm"]:
            print(f"  gate: residual {info['residual']:.2e} > {cg_gate_rel} * ||rhs|| {info['rhs_norm']:.2e}; retrying at {4 * mi} iters")
            theta, info = solve_score_matching_cg(gradient_fn_per_batch, theta, batch, cg_maxiter=4 * mi, **solve_kwargs)
        accepted = info["residual"] <= cg_gate_rel * info["rhs_norm"]
        solve_seconds = time.time() - t_solve0

        # Stability panel. Guard the uniform (DC) mode only -- that is the mode that produced
        # the old run's blobs when it went unstable. lam_min (full-spectrum min eigenvalue of
        # the forward Hessian at x=0) is logged as a diagnostic but NOT required positive: as
        # t -> 0 the 0-vs-1 class-separation direction legitimately develops negative curvature
        # at the origin (the double well the reverse SDE uses to split digits), and lifting it
        # would erase that structure.
        lam_min = quadratic_sector_lambda_min(theta)
        theta_mean_klin = float(jnp.mean(theta[0:N_osc]))
        theta_uniform_rate = 2.0 * theta_mean_klin - 1.0 / sigma_forward**2
        if accepted and theta_uniform_rate <= uniform_rate_hard_fail:
            raise RuntimeError(
                f"Slice {t_idx} (t={float(t_curr):.4f}): reverse uniform-mode rate "
                f"{theta_uniform_rate:+.3f} <= {uniform_rate_hard_fail} (mean k_lin {theta_mean_klin:.4f}) "
                f"— DC mode unstable in reverse (the blob failure), aborting. Raise CG_PROX_ABS or CG_GROUP_RIDGE."
            )
        if accepted and theta_uniform_rate < uniform_rate_warn:
            print(f"  stability WARNING: uniform-mode reverse rate {theta_uniform_rate:+.3f} < {uniform_rate_warn} "
                  f"(healthy ~ +0.9); lam_min={lam_min:.3e}")

        if accepted:
            current_params = theta
        else:
            print(
                f"  WARNING slice {t_idx} (t={float(t_curr):.4f}) NOT ACCEPTED "
                f"(residual {info['residual']:.2e} vs gate {cg_gate_rel * info['rhs_norm']:.2e}); "
                f"keeping previous slice's parameters for this row — chain not poisoned."
            )

        # Held-out diagnostic: SM loss on an independent sample built from held-out IMAGES.
        heldout_batch = sample_forward_process(
            t_curr, n_samples=cg_heldout_size, D=Temp, sigma_final=sigma_forward, samples0=samples_heldout_src, key=subkey_heldout, beta=1,
        )
        heldout_loss = float(loss_fn_per_batch(current_params, heldout_batch))
        loss_fit = float(loss_fn_per_batch(current_params, batch))
        mean_klin = float(jnp.mean(current_params[0:N_osc]))
        mean_abs_clin = float(jnp.mean(jnp.abs(current_params[3 * N_osc:3 * N_osc + num_connections])))
        uniform_rate = 2.0 * mean_klin - 1.0 / sigma_forward**2  # reverse stiffness of the uniform mode; must stay > 0
        print(
            f"t_idx {t_idx:3d}  t={float(t_curr):.3f}  loss_fit {loss_fit:.2f}  loss_heldout {heldout_loss:.2f}  "
            f"resid/rhs={info['residual'] / max(info['rhs_norm'], 1e-30):.2e}  accepted={accepted}  "
            f"k6_at_floor={info['n_active']}/{N_osc}  as_iters={info['active_set_iters_used']}  "
            f"mean_klin={mean_klin:.4f}  unif_rate={uniform_rate:+.3f}  lam_min={lam_min:.3e}  "
            f"mean|c_lin|={mean_abs_clin:.2e}  {solve_seconds:.0f}s"
        )
        for k, v in [
            ("t", float(t_curr)), ("loss_fit", loss_fit), ("loss_heldout", heldout_loss),
            ("free_residual", info["residual"]), ("rhs_norm", info["rhs_norm"]),
            ("accepted", bool(accepted)), ("n_active", info["n_active"]),
            ("active_set_iters", info["active_set_iters_used"]), ("lambda_min", lam_min),
            ("uniform_rate", uniform_rate), ("mean_klin", mean_klin),
            ("mean_abs_clin", mean_abs_clin), ("solve_seconds", solve_seconds),
        ]:
            cg_diag[k].append(v)

        params_history_all_t.append(current_params)
        optimize_memory()

        # Checkpoint after each slice so the run is resumable (atomic: tmp + rename).
        _tmp_path = params_history_path + ".tmp.npy"
        np.save(_tmp_path, np.array(params_history_all_t))
        os.replace(_tmp_path, params_history_path)
        with open(time_index_path, "w") as f:
            f.write(str(t_idx + 1))  # advisory; resume derives the index from the history rows

        slices_this_session += 1
        if max_slices and slices_this_session >= max_slices:
            print(f"MAX_SLICES={max_slices} reached; stopping after slice {t_idx} (resumable).")
            break

    if len(params_history_all_t) == len(forward_time_pts):
        print("Optimization complete; saved parameters to", output_dir)

# Save + plot the per-slice diagnostics for slices run this session.
if cg_diag["t"]:
    diag_path = f"{output_dir}/cg_loss_diagnostics_from_{start_t_idx}.npz"
    np.savez(diag_path, **{k: np.array(v) for k, v in cg_diag.items()})
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(cg_diag["t"])
    ax.plot(np.array(cg_diag["t"])[order], np.array(cg_diag["loss_fit"])[order], "-o", ms=3, label="fit (in-sample)")
    ax.plot(np.array(cg_diag["t"])[order], np.array(cg_diag["loss_heldout"])[order], "-o", ms=3, label="held-out (unseen images)")
    ax.set_xlabel("forward time t"); ax.set_ylabel("score-matching loss")
    ax.set_title(f"Fit vs held-out SM loss (prox_abs={cg_prox_abs}, gridge={cg_group_ridge}, sample={cg_sample_size})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_folder}/cg_fit_vs_heldout_loss_from_{start_t_idx}.png", dpi=150)
    plt.close(fig)
    print(f"Saved CG diagnostics to {diag_path}")

if len(params_history_all_t) < len(forward_time_pts):
    print(f"Only {len(params_history_all_t)}/{len(forward_time_pts)} slices trained (MAX_SLICES smoke run) — skipping reverse SDE. Re-run without MAX_SLICES to continue.")
    sys.exit(0)

params_history_all_t = jnp.array(params_history_all_t)

#####################################
# Interpolate the learned parameters as a function of forward time
#####################################
print(f"params_history_all_t shape: {params_history_all_t.shape}, forward_time_pts shape: {forward_time_pts.shape}")

params_interpolator_non_smoothed = interpolate_parameters(params_history_all_t, forward_time_pts)

if use_smoothed_params_for_reverse:
    # savgol assumes uniformly spaced samples — refuse the combination with the log grid.
    assert not exp_time_pts, "USE_SMOOTHED_REVERSE=1 requires EXP_TIME_PTS=0 (savgol needs a uniform grid)"
    smoothed_params = smooth_parameters(
        params_history_all_t,
        window_lengths=[10] * params_history_all_t.shape[1],
        poly_orders=[3] * params_history_all_t.shape[1],
    )
    # savgol undershoots at spikes and silently breaks the k_6 >= floor guarantee (the only
    # confining term of the reverse potential) — re-clamp after smoothing.
    smoothed_params = smoothed_params.at[:, 2 * N_osc:3 * N_osc].set(
        jnp.maximum(smoothed_params[:, 2 * N_osc:3 * N_osc], k6_floor)
    )
    params_interpolator_smoothed = interpolate_parameters(smoothed_params, forward_time_pts)
else:
    params_interpolator_smoothed = params_interpolator_non_smoothed

plot_parameter_as_fn_of_time(
    params_names, forward_time_pts, forward_time_pts, params_history_all_t,
    params_interpolator_smoothed, unflatten, N_osc, save_fig=True, path=plot_folder, log_scale=False,
)

#####################################
# Reverse (generation) SDE
#####################################
def run_reverse_process_SDE(params_interpolator, suffix, key):
    """Integrate the reverse-time Langevin SDE to generate samples.

    For the OU forward process the reverse drift is (1/sigma^2) x - 2 grad_x E_theta(tau(t)).
    Exploiting the energy's linearity in the parameters, this is assembled purely by parameter
    arithmetic: theta_reverse(t) = 2 * theta(tau(t)) - theta_linear, where theta_linear is the
    pure harmonic reference (k_lin = 1) and tau(t) = t_forward - t. The initial condition is the
    asymptotic forward Gaussian N(0, Temp * sigma_forward^2 I). Only final states are kept.
    """
    tau = lambda t: t_forward - t
    forward_params = jnp.zeros_like(params_flattened_initial).at[0:N_osc].set(1.0)
    params_reverse = lambda t: 2 * params_interpolator(tau(t)) - forward_params / sigma_forward**2

    drift_fn, diffusion_fn = setup_overdamped_SDE(energy_fn, params_reverse, N_osc, time_dependent_parms=True, Temp=Temp)

    t0, t1 = 0.0, t_forward
    n_time_steps_sde = 100          # number of saved output points (does not affect solver accuracy)
    ts = jnp.linspace(t0, t1, n_time_steps_sde)
    dt0 = 1e-8

    # Initial states ~ N(0, Temp * sigma_forward^2 I), and one Brownian key per trajectory.
    key, subkey_init = jr.split(key)
    initial_states = jnp.sqrt(Temp) * sigma_forward * jr.normal(subkey_init, shape=(n_trajectories, N_osc))
    key, subkey_brownian = jr.split(key)
    keys_brownian = jr.split(subkey_brownian, n_trajectories)

    print(f"Running reverse SDE for {n_trajectories} trajectories...")
    solve_one = lambda init_state, key_b: solve_SDE(
        drift_fn, diffusion_fn, init_state, key_b, t0, t1, ts.shape[0], dt0,
        brownian_tolerance=brownian_tolerance, rtol=rtol_sde, atol=atol_sde,
    )
    solutions = vmap(solve_one, in_axes=(0, 0))(initial_states, keys_brownian)
    final_samples_scaled = solutions.ys[:, -1, :]  # (n_trajectories, N_osc), final states only

    final_states_path = f"{output_dir}/final_states_{suffix}.npy"
    jnp.save(final_states_path, final_samples_scaled)
    print(f"Final states saved to {final_states_path}")

    del solutions
    optimize_memory()
    final_samples = final_samples_scaled / additional_rescaling
    return final_samples.reshape(-1, resolution[0], resolution[1])


#####################################
# Confinement alarm (alarm-first, never silently repairs): check the assembled REVERSE
# potential theta_rev = 2*theta - theta_harmonic over all slices before integrating.
#  - k6_rev = 2*k_6 must stay >= 2*k6_floor (> 0 or the potential is unbounded below);
#  - a negative reverse quartic edge coupling c4_rev = 2*c_duff opens a pair-escape channel
#    that only the sextic terms close, at radius u* ~ sqrt(16*|c4_rev| / (k6_i + k6_j)):
#    if u* exceeds ~4.5 (data lives at |x| <~ 3), trajectories can be ejected far outside
#    the data range (the old run produced |x| up to 21.6 through exactly this channel).
#####################################
_theta_used = np.asarray(smoothed_params if use_smoothed_params_for_reverse else params_history_all_t)
_k6_rev = 2.0 * _theta_used[:, 2 * N_osc:3 * N_osc]
_c4_rev = 2.0 * _theta_used[:, 3 * N_osc + 2 * num_connections:3 * N_osc + 3 * num_connections]
_conn_np = np.asarray(connectivity)
_k6_pair = _k6_rev[:, _conn_np[:, 0]] + _k6_rev[:, _conn_np[:, 1]]
_u_star = np.sqrt(16.0 * np.maximum(-_c4_rev, 0.0) / np.maximum(_k6_pair, 1e-12))
print(f"Confinement check: min k6_rev={_k6_rev.min():.4f} (floor {2 * k6_floor}), max pair-escape radius u*={_u_star.max():.2f}")
if _k6_rev.min() <= 0:
    raise RuntimeError("Reverse potential unbounded below (k6_rev <= 0) — refusing to generate.")
if _u_star.max() > 4.5:
    print(f"WARNING: pair-escape radius u*={_u_star.max():.2f} > 4.5 — expect outlier trajectories; inspect c_duff before trusting samples.")

print(f"Running reverse SDE with final time: {t_forward}")
reverse_interpolator = params_interpolator_smoothed if use_smoothed_params_for_reverse else params_interpolator_non_smoothed
params_tag = "smoothed" if use_smoothed_params_for_reverse else "non_smoothed"
reverse_sde_suffix = (
    f"{params_tag}_sde_memory_efficient_atol_{atol_sde}_rtol_{rtol_sde}_brownian_tolerance_{brownian_tolerance}"
    f"_init_method_asymptotic_gaussian"
    f"_use_same_initial_condition_for_all_trajectories_False"
    + (f"_reverse_rng_seed_{key_seed_reverse}" if key_seed_reverse is not None else "")
)
images_generated_sde = run_reverse_process_SDE(reverse_interpolator, reverse_sde_suffix, reverse_sde_key)
print_memory_usage("after reverse SDE")

#####################################
# Plotting: true vs. generated digit grids
#####################################
def plot_true_vs_generated_grid(true_imgs, generated_imgs, generated_title, save_path):
    """Top block: true images; bottom block: generated images (<=10 columns per row).
    Robust to any n_trajectories (not only multiples of 10)."""
    n_show = n_trajectories
    ncol = min(10, n_show)
    n_rows = max(1, -(-n_show // ncol))  # ceil(n_show / ncol)
    fig, axes = plt.subplots(2 * n_rows, ncol, figsize=(1.5 * ncol, 3 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")  # blank any unused cells (partial last row)
    for idx in range(n_show):
        for row_block, imgs in ((0, true_imgs), (n_rows, generated_imgs)):
            ax = axes[idx // ncol + row_block, idx % ncol]
            img = imgs[idx]
            if img.shape[-1] == 1:  # drop singleton channel if present
                img = img.squeeze(-1)
            ax.imshow(np.array(img), cmap="gray")
    axes[0, 0].set_title("True images", loc="left", fontsize=16)
    axes[n_rows, 0].set_title(generated_title, loc="left", fontsize=16)
    plt.subplots_adjust(wspace=0.01, hspace=0.5)
    plt.savefig(save_path)
    plt.close()


images_true = images_flat_true.reshape(-1, resolution[0], resolution[1])
plot_true_vs_generated_grid(
    images_true,
    images_generated_sde,
    f"SDE sampled images (Memory Efficient), rtol={rtol_sde}, atol={atol_sde}, brownian_tolerance={brownian_tolerance}",
    f"{plot_folder}/samples_{reverse_sde_suffix}.png",
)

# Same comparison, but with true and generated images clipped to the normalized pixel range.
clip_min, clip_max = pixel_min_normalized, pixel_max_normalized
images_true_clipped = np.clip(np.array(samples_target_unscaled.reshape(-1, resolution[0], resolution[1])), clip_min, clip_max)
images_generated_clipped = np.clip(np.array(images_generated_sde), clip_min, clip_max)
plot_true_vs_generated_grid(
    images_true_clipped,
    images_generated_clipped,
    f"SDE sampled images (range:[{clip_min:.4f},{clip_max:.4f}]), rtol={rtol_sde}, atol={atol_sde}, brownian_tolerance={brownian_tolerance}",
    f"{plot_folder}/samples_{reverse_sde_suffix}_clipped_{clip_min:.4f}_to_{clip_max:.4f}.png",
)

print("Script completed successfully!")
