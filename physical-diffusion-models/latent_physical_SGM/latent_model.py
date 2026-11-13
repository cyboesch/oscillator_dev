#!/usr/bin/env python3
"""Shared code for the latent physical SGM prototype (branch latent_physical_SGM; see DESIGN.md).

Model at forward time tau:
    F(x) = E^v(x; theta_v) - T * sum_j log 2cosh(a_j(x) / T),     a_j(x) = b_j + W_j . x
E^v is the paper's visible energy (28x28 lattice of 6th-order Duffing oscillators, pair couplings within
Chebyshev radius 14, linear in its 545,566 parameters). The K Ising hidden units are averaged out exactly.
Trained per slice: theta_v(tau), and the hidden units' couplings W(tau) (K x 784) and fields b(tau) (K).

Implementation note: the paper computes grad E^v and the Laplacian per sample with a scan over the 180,810 pairs.
Here the three pair-coupling groups are scattered into dense 784x784 matrices once per step, and gradient and
Laplacian are batched matrix products. Same model, same loss; test_latent.py checks both against the paper's
functions. Everything is float64, as in the paper.

Optional (train_latent.py --harmonic R, 2026-11-12): R harmonic hidden modes, linear oscillators integrated out
exactly, F_h = -1/2 sum_j (v_j . x + c_j)^2 (harmonic_terms; parameters 'V' (R x N) and 'c' (R) in the params dict),
with a closed-form reference per slice time from the data's principal components (harmonic_reference: the Gaussian
part of the forward distribution), or from those of the Potts prototypes' within-cluster covariance (--harmonic-ref
within; prototype_reference(..., return_stats=True), harmonic_within_spectrum). Without the 'V' key every function
below is exactly as before.

Optional (train_latent.py --patch SIDE:STRIDE:STATES, 2026-11-13): P per-patch Potts hidden units, unit p coupled only
to the SIDE x SIDE pixels x_p of patch p of a regular grid of overlapping patches (PatchGeometry), integrated out
exactly (the units are independent given x): F_patch = -T sum_p log sum_s exp(a_ps / T), a_ps = bp_ps + Wp_ps . x_p
(+ Qp_ps . x_p^2 with --quad) (patch_terms; parameters 'Wp' (P, S, d), 'bp' (P, S) and 'Qp' (P, S, d) in the params
dict, d = SIDE^2). The geometry follows from the shapes (patch_geometry), so the 'Wp' key alone switches the term on,
added after every other term; strided slices and pads only (no gather or scatter), deterministic on the GPU.
Reference per patch: prototype_reference's pooled branch on the patch vectors (patch_clusters, patch_reference).
Without the 'Wp' key every function below is exactly as before.
"""

import hashlib
import json
import math
import os
import re
import sys
from functools import lru_cache, partial

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import flatten_util  # noqa: E402
from jax import lax  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if PKG not in sys.path:
    sys.path.append(PKG)

from physical_diffusion_fns.helper_fns import normalize_samples  # noqa: E402
from physical_diffusion_fns.network_fns import create_2d_square_lattice_connectivity  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# Paper settings (scripts/physical_diffusion_model_MNIST.py)
# ---------------------------------------------------------------------------------------------------------------
KEY_SEED = 1
STD_ADDED_NOISE = 0.01
TEMP = 0.02                      # bath temperature = forward diffusion constant D
SIGMA_FORWARD = 1.0
T_FORWARD = 5.0
N_TIME_STEPS = 100
GRID = 28
N = GRID * GRID
N_NEIGHBOUR_COUPLINGS = 14
BATCH_SIZE = 512
LR_AT_T0, LR_AT_T5 = 2e-5, 2e-3  # the paper's learning_rate_start / _end
LR_DECAY_RATE, LR_DECAY_STEPS = 0.95, 6000
WINDOW = 1000
PATIENCE_AT_T0, PATIENCE_AT_T5 = 300, 100
K6_FLOOR = 0.01                  # run_optimization projects k_6 >= 0.01 after every step

# Sampler settings of the paper / ode_vs_sde study
RTOL_SDE, ATOL_SDE, BROWNIAN_TOL = 1e-4, 1e-6, 1e-12
RTOL_ODE, ATOL_ODE = 1e-7, 1e-9
DT0 = 1e-8
NEW_SEED = 2026                  # trajectories 40..499 of the ode_vs_sde 500-trajectory set
EXT_SEED = 2027                  # further trajectories (500..)

DATA_PATH = os.path.join(PKG, "data", "MNIST", "mnist_labels_[0, 1]_resolution_(28, 28)_balanced_True.npy")
ONES_PATH = os.path.join(PKG, "data", "MNIST", "mnist_labels_[1]_resolution_(28, 28).npy")
PAPER_RUN_DIR = os.path.join(
    PKG, "out", "2026_07_08", "problems", "MNIST_generation",
    "labels_[0, 1]_resolution_28_x_28",
    "n_couplings_14_Temp_0.02_energy_fn_type_6th_order_duffing_coupling",
    "added_gaussian_noise_std_0.01_key_seed_1_additional_rescaling_1",
    "training_method_SM_at_kbT_minimal_memory_solve_opt_in_reverse_True_exp_time_pts_False_"
    "t_forward_5.0_n_timesteps_100_sigma_forward_1.0_lr_2e-05_to_0.002_epochs_100000_batch_512_"
    "window_1000_tol_0.001_to_0.001_patience_300_to_100_",
    "opt_per_time_plots",
)
PAPER_CKPT = os.path.join(PAPER_RUN_DIR, "params_history.npy")
PAPER_CKPT_SHA256 = "37cfe46ef805a39129c265a0c17aaf6cf2b4e3b2ec7d8f55725e1cd66c227249"
ABLATION_DIR = os.path.join(PAPER_RUN_DIR, "interpolation_ablation_v1")
UNSPLIT_4187940 = os.path.join(PKG, "out", "diagnostics", "sde_split_control", "4187940", "final_states_sde_unsplit.npy")
OUT_ROOT = os.path.join(PKG, "out", "latent_physical_SGM")


# ---------------------------------------------------------------------------------------------------------------
# Schedules (identical expressions to the paper's script, so the float values match)
# ---------------------------------------------------------------------------------------------------------------
def forward_time_pts():
    """Slice times, t = 5 first (the paper trains from the noisy end towards the data)."""
    return np.asarray(jnp.linspace(0.0, T_FORWARD, N_TIME_STEPS)[::-1])


def lr_schedule():
    return np.asarray(jnp.linspace(LR_AT_T0, LR_AT_T5, N_TIME_STEPS)[::-1])


def patience_schedule():
    return np.asarray(jnp.linspace(PATIENCE_AT_T0, PATIENCE_AT_T5, N_TIME_STEPS)[::-1])


def paper_steps(k):
    """Number of Adam steps the paper's run_optimization takes at slice k.

    Early stopping never registers an improvement (CODE_NOTES 8.1): the patience counter grows by one at every
    check (epochs 1000, 1100, ...), and the loop breaks after epoch 1000 + 100*(ceil(patience) - 1).
    """
    return WINDOW + 1 + 100 * (int(math.ceil(float(patience_schedule()[k]))) - 1)


# ---------------------------------------------------------------------------------------------------------------
# Time grids (train_latent.py --time-grid): the slice times, and each slice's learning rate and step count
# ---------------------------------------------------------------------------------------------------------------
TIME_GRID_TOL = 1e-12            # largest |difference| allowed between a run's recorded slice times and its grid's
MIN_KNOT_GAP = 1e-9              # smallest interval between neighbouring knots a power grid may have (parse_time_grid)


def parse_time_grid(spec):
    """Canonical form of a time-grid spec (train_latent.py --time-grid; config.json 'time_grid', see
    config_time_grid). Every grid keeps N_TIME_STEPS = 100 slices, slice 0 at t = 5 and slice 99 at t = 0 exactly,
    strictly decreasing in between:
      'uniform'    the paper's grid, forward_time_pts(): t_k = 5 (1 - k/99), spacing 5/99 = 0.0505 (the default)
      'power:P'    t_k = 5 (1 - k/99)^P with P > 0 (time_grid_pts). P > 1 puts the knots denser near t = 0 and sparser
                   near t = 5: P = 1.5 has the last interval 5/99^1.5 = 0.00508 and the first 5 (1 - (98/99)^1.5) =
                   0.0756. P = 1 is the uniform grid up to rounding, but a different spec. Canonical form
                   'power:' + repr(float(P)), e.g. 'power:1.5', 'power:2.0'.
      'power:P@K'  the uniform grid's slices 0..K (K = 1..97: their times, learning rates and step counts bit for bit)
                   and the power law below t_K = 5 (1 - K/99): t_k = t_K ((99 - k) / (99 - K))^P for k = K..99. With
                   100 slices the knots below t_K are only redistributed: P > 1 refines near t = 0 and widens the
                   intervals just below t_K (power:1.5@40: slices 0-40 at t >= 2.98 as on the uniform grid, then
                   0.0754 below the junction down to 2.98/59^1.5 = 0.0066 at t = 0). Canonical 'power:1.5@40'.
    Case and blanks (anywhere) are ignored. P is a plain decimal number (digits, an optional point and exponent; no
    underscores or other digit forms, so a typo such as 'power:1_5' is refused rather than read as 15), K a plain
    integer. Raises ValueError for anything else (another family, P missing, not a number > 0, K not an integer in
    1..97, extra fields) and for a P so extreme that two neighbouring knots would be closer than MIN_KNOT_GAP = 1e-9
    in forward or reverse time s = 5 - t (the samplers' step_ts and diffrax's 1e-10 clipping need distinct knots;
    power:P alone allows P up to about 4.8)."""
    if not isinstance(spec, str):
        raise ValueError(f"time grid {spec!r}: expected a string, 'uniform', 'power:P' or 'power:P@K' (e.g. 'power:1.5')")
    s = re.sub(r"\s+", "", spec).lower()
    if s == "uniform":
        return "uniform"
    kind, sep, arg = s.partition(":")
    if kind != "power" or not sep:
        raise ValueError(f"time grid {spec!r}: expected 'uniform', 'power:P' or 'power:P@K' with a number P > 0 "
                         f"(e.g. 'power:1.5')")
    p_arg, at, k_arg = arg.partition("@")
    if not re.fullmatch(r"\+?([0-9]+\.?[0-9]*|\.[0-9]+)(e[-+]?[0-9]+)?", p_arg):   # ASCII digits only
        raise ValueError(f"time grid {spec!r}: P in 'power:P' must be one plain number > 0, e.g. 'power:1.5'")
    p = float(p_arg)
    if not (math.isfinite(p) and p > 0.0):
        raise ValueError(f"time grid {spec!r}: P in 'power:P' must be a finite number > 0")
    k0 = 0
    if at:
        if not re.fullmatch(r"\+?[0-9]+", k_arg) or not 1 <= int(k_arg) <= N_TIME_STEPS - 3:
            raise ValueError(f"time grid {spec!r}: K in 'power:P@K' must be a slice index 1..{N_TIME_STEPS - 3}")
        k0 = int(k_arg)
    t = _power_time_pts(p, k0)
    gap = min(float(np.min(-np.diff(t))), float(np.min(np.diff(T_FORWARD - t))))
    if not gap > MIN_KNOT_GAP:
        raise ValueError(f"time grid {spec!r}: with P = {p!r} two neighbouring knots are {gap:.3g} apart (forward or "
                         f"reverse time; at least {MIN_KNOT_GAP:g} needed); choose a less extreme P")
    return f"power:{p!r}" + (f"@{k0}" if k0 else "")


def _power_params(spec):
    """(P, K) of a canonical 'power:P' (K = 0) or 'power:P@K' spec."""
    p_arg, _, k_arg = spec.split(":", 1)[1].partition("@")
    return float(p_arg), int(k_arg or 0)


def _power_time_pts(p, k0=0):
    """In numpy float64. k0 = 0: 5 s^p for s = (99 - k) / 99, k = 0..99; s is exactly 1 at k = 0 and exactly 0 at
    k = 99, so t_0 = 5 and t_99 = 0 exactly. k0 > 0: forward_time_pts() for k <= k0, and t_k0 r^p below with
    r = (99 - k) / (99 - k0), exactly 1 at k0 (t_k0 unchanged) and exactly 0 at k = 99.
    The powers are taken one by one with math.pow (the C library's pow), not numpy's array power, whose vectorised
    kernels differ by an ulp between CPU types (AVX-512 or not): so a power grid is the same bit for bit on every
    node, e.g. when a run is resumed or sampled on another node type."""
    if k0 == 0:
        s = np.arange(N_TIME_STEPS - 1, -1, -1, dtype=np.float64) / (N_TIME_STEPS - 1)
        return T_FORWARD * np.array([math.pow(float(x), p) for x in s], dtype=np.float64)
    t = np.array(forward_time_pts(), dtype=np.float64)
    m = N_TIME_STEPS - 1 - k0
    r = np.arange(m, -1, -1, dtype=np.float64) / m
    t[k0:] = t[k0] * np.array([math.pow(float(x), p) for x in r], dtype=np.float64)
    return t


def time_grid_pts(spec="uniform"):
    """Slice times of a time grid (parse_time_grid) in slice order (t = 5 first), float64 (N_TIME_STEPS,).
    'uniform' returns forward_time_pts() itself, i.e. the same float64 values as always; 'power:P' and 'power:P@K'
    are computed in numpy (_power_time_pts)."""
    spec = parse_time_grid(spec)
    if spec == "uniform":
        return forward_time_pts()
    return _power_time_pts(*_power_params(spec))


def time_grid_lr(spec="uniform"):
    """Learning rate of the visible parameters per slice (before --lr-scale): lr_schedule() itself for 'uniform';
    otherwise the paper's linear function of t, LR_AT_T0 = 2e-5 at t = 0 to LR_AT_T5 = 2e-3 at t = 5, at the grid's
    slice times (np.interp: exactly 2e-5 at t = 0 and 2e-3 at t = 5); 'power:P@K' keeps lr_schedule()'s values for
    its uniform slices 0..K."""
    spec = parse_time_grid(spec)
    if spec == "uniform":
        return lr_schedule()
    lr = np.interp(time_grid_pts(spec), [0.0, T_FORWARD], [LR_AT_T0, LR_AT_T5])
    k0 = _power_params(spec)[1]
    if k0:
        lr[:k0 + 1] = lr_schedule()[:k0 + 1]
    return lr


def time_grid_patience(spec="uniform"):
    """The paper's early-stopping patience per slice: patience_schedule() itself for 'uniform'; otherwise its linear
    function of t, PATIENCE_AT_T0 = 300 at t = 0 to PATIENCE_AT_T5 = 100 at t = 5, at the grid's slice times (np.interp:
    exactly 300 and 100 at the two ends); 'power:P@K' keeps patience_schedule()'s values for slices 0..K."""
    spec = parse_time_grid(spec)
    if spec == "uniform":
        return patience_schedule()
    pat = np.interp(time_grid_pts(spec), [0.0, T_FORWARD], [float(PATIENCE_AT_T0), float(PATIENCE_AT_T5)])
    k0 = _power_params(spec)[1]
    if k0:
        pat[:k0 + 1] = patience_schedule()[:k0 + 1]
    return pat


def time_grid_steps(k, spec="uniform"):
    """paper_steps(k) on a time grid: paper_steps(k) itself for 'uniform' (and for the uniform slices k <= K of
    'power:P@K'); otherwise the same rule with the grid's patience, WINDOW + 1 + 100 (ceil(patience_k) - 1), where the
    ceil ignores rounding errors below 1e-9, so that a patience that should be an integer (100 at t = 5, 300 at t = 0)
    cannot round up to one more 100-step check (10,901 steps at t = 5 and 30,901 at t = 0, as in the paper's
    schedule)."""
    spec = parse_time_grid(spec)
    if spec == "uniform":
        return paper_steps(k)
    k = range(N_TIME_STEPS)[k]                    # the slice index (IndexError outside the grid, as paper_steps)
    k0 = _power_params(spec)[1]
    if k0 and k <= k0:
        return paper_steps(k)
    return WINDOW + 1 + 100 * (int(math.ceil(float(time_grid_patience(spec)[k]) - 1e-9)) - 1)


def config_time_grid(cfg):
    """A run's time grid from its config dict (train_latent.py's config.json): the canonical form of cfg['time_grid'],
    or 'uniform' for the runs from before the option, whose config has no such key."""
    return parse_time_grid(cfg.get("time_grid", "uniform"))


def checked_time_pts(spec, t_saved=None, where="run"):
    """time_grid_pts(spec). With t_saved (the slice times a run recorded, e.g. the 't' of its hidden_history.npz)
    raises ValueError unless it has the same shape and agrees to TIME_GRID_TOL everywhere: a run's samples and
    diagnostics must use the slice times it was trained at."""
    taus = time_grid_pts(spec)
    if t_saved is not None:
        t_saved = np.asarray(t_saved, dtype=np.float64)
        if t_saved.shape != taus.shape or not np.all(np.abs(t_saved - taus) <= TIME_GRID_TOL):
            diff = (f"max |difference| {float(np.max(np.abs(t_saved - taus))):.3g}" if t_saved.shape == taus.shape
                    else f"shape {t_saved.shape} instead of {taus.shape}")
            raise ValueError(f"{where}: the slice times it recorded (hidden_history.npz 't') do not match its time grid "
                             f"{spec!r} ({diff}); refusing to use other slice times than its training did")
    return taus


def run_time_grid(run_dir, cfg=None):
    """(spec, taus) of a run folder of train_latent.py: config_time_grid of its config.json (or of cfg, if given) and
    its slice times, checked (checked_time_pts) against the 't' of its hidden_history.npz if that file exists (it is
    written when the last slice is done; a run in progress is checked against its config only)."""
    if cfg is None:
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
    spec = config_time_grid(cfg)
    t_saved = None
    path = os.path.join(run_dir, "hidden_history.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            if "t" in z.files:
                t_saved = z["t"]
    return spec, checked_time_pts(spec, t_saved, run_dir)


def interp_history(hist, taus, t):
    """hist[k] (an array per slice k at forward time taus[k], e.g. the paper's params_history.npy as a memmap) at
    forward time t: linear in t between the two slices whose times enclose t, as helper_fns.interpolate_parameters
    (jnp.interp) interpolates; a slice time returns that slice itself. Reads only those one or two slices."""
    taus = np.asarray(taus, dtype=np.float64)
    hit = np.flatnonzero(taus == t)
    if hit.size:
        return np.array(hist[int(hit[0])], dtype=np.float64)
    order = np.argsort(taus)
    xp = taus[order]
    if not xp[0] < t < xp[-1]:
        raise ValueError(f"t = {t!r} is outside the slice times [{xp[0]}, {xp[-1]}]")
    i = int(np.searchsorted(xp, t))                 # xp[i - 1] < t < xp[i]
    lo = np.array(hist[int(order[i - 1])], dtype=np.float64)
    hi = np.array(hist[int(order[i])], dtype=np.float64)
    return lo + ((t - xp[i - 1]) / (xp[i] - xp[i - 1])) * (hi - lo)


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def load_data():
    """Normalised training images exactly as the paper's script builds them, plus exact 0/1 labels.

    Returns (data float64 jnp array (11846, 784), labels np.int64 (1 = one), mean, std).
    Labels: exact pixel match against the ones-only file (as in blend_check.py; 5923 / 5923).
    """
    raw_np = np.load(DATA_PATH)
    raw = jnp.array(raw_np)
    _, _, image_noise_added_key = jr.split(jr.PRNGKey(KEY_SEED), 3)
    _, subkey = jr.split(image_noise_added_key)
    data, mean, std = normalize_samples(raw + jr.normal(subkey, raw.shape) * STD_ADDED_NOISE)
    ones = np.load(ONES_PATH)

    def as_bytes(a):
        return [r.tobytes() for r in np.round(np.asarray(a) * 255).astype(np.uint8)]

    ones_set = set(as_bytes(ones))
    labels = np.array([r in ones_set for r in as_bytes(raw_np)], dtype=np.int64)
    if labels.sum() != 5923 or (labels == 0).sum() != 5923:
        raise RuntimeError(f"label matching failed: {labels.sum()} ones, {(labels == 0).sum()} zeros")
    return data, labels, float(mean), float(std)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Visible model
# ---------------------------------------------------------------------------------------------------------------
NEIGHBOURHOODS = ("chebyshev", "von_neumann")


def connectivity(grid=GRID, radius=N_NEIGHBOUR_COUPLINGS, neighbourhood="chebyshev"):
    """Sorted unique (i, j), i < j, pairs within Chebyshev distance `radius` (default: the paper's 28x28, 14).
    neighbourhood = "von_neumann": only the pairs within Manhattan distance `radius` (a subset of the Chebyshev
    list; radius 1 = the 4 nearest neighbours of each pixel)."""
    conn = np.asarray(create_2d_square_lattice_connectivity(grid_size=grid, n_neighbour_couplings=radius))
    if neighbourhood == "chebyshev":
        return conn
    if neighbourhood != "von_neumann":
        raise ValueError(f"neighbourhood must be one of {NEIGHBOURHOODS}, not {neighbourhood!r}")
    ri, ci = np.divmod(conn[:, 0], grid)
    rj, cj = np.divmod(conn[:, 1], grid)
    return conn[np.abs(ri - rj) + np.abs(ci - cj) <= radius]


def paper_theta_layout(n_pairs, n=N):
    """Paper's initial visible parameters and its unflatten (ravel_pytree of the 7-group tuple).

    Flat order: [k_lin (n) | k_duff (n) | k_6 (n) | c_lin (P) | c_optomech (P) | c_duff (P) | bias (n)].
    Initial values: k_lin = 1, k_6 = 1, c_lin = 1 on every pair, all else 0.
    """
    template = (jnp.ones(n), jnp.zeros(n), jnp.ones(n), jnp.ones(n_pairs), jnp.zeros(n_pairs), jnp.zeros(n_pairs),
                jnp.zeros(n))
    theta0, unflatten = flatten_util.ravel_pytree(template)
    return theta0, unflatten


def theta_lin(n_theta, n=N):
    """Pure harmonic energy 1/2 |x|^2 (k_lin = 1, all else 0): the forward process's stationary energy / T."""
    return jnp.zeros(n_theta).at[0:n].set(1.0)


def k6_slice(n=N):
    return slice(2 * n, 3 * n)


def kduff_slice(n=N):
    return slice(n, 2 * n)


def cduff_slice(n, n_pairs):
    return slice(3 * n + 2 * n_pairs, 3 * n + 3 * n_pairs)


def clin_slice(n, n_pairs):
    return slice(3 * n, 3 * n + n_pairs)


def com_slice(n, n_pairs):
    return slice(3 * n + n_pairs, 3 * n + 2 * n_pairs)


# The three pair-coupling groups of the visible energy, by name (--pair-terms):
#   lin  c_lin (x_i - x_j)^2   (linear spring),   om  c_om x_i^2 x_j   (optomechanical, asymmetric),
#   duff c_duff (x_i - x_j)^4 / 4   (Duffing pair term).
PAIR_TERMS = {"lin": clin_slice, "om": com_slice, "duff": cduff_slice}


def parse_pair_terms(spec):
    """'all' / 'none' / a comma list of lin, om, duff -> sorted tuple of kept pair-coupling groups."""
    spec = spec.strip().lower()
    if spec == "all":
        return tuple(sorted(PAIR_TERMS))
    if spec == "none":
        return ()
    terms = tuple(sorted({t.strip() for t in spec.split(",") if t.strip()}))
    if not terms or any(t not in PAIR_TERMS for t in terms):
        raise ValueError(f"pair terms must be 'all', 'none' or a comma list of {sorted(PAIR_TERMS)}, not {spec!r}")
    return terms


def dropped_pair_slices(n, n_pairs, kept):
    """Slices of the flat visible parameters of the pair-coupling groups NOT in `kept` (held at exactly zero)."""
    return tuple(PAIR_TERMS[t](n, n_pairs) for t in sorted(PAIR_TERMS) if t not in kept)


K4_FLOOR = 0.01                  # quartic visible energy (--visible quartic): k_duff >= 0.01, c_duff >= 0, k_6 = 0


def quartic_initial_theta(theta0, n, n_pairs):
    """The paper's initial parameters with the sextic term replaced by a quartic one (k_6 = 0, k_duff = 1)."""
    return theta0.at[k6_slice(n)].set(0.0).at[kduff_slice(n)].set(1.0)


# ---------------------------------------------------------------------------------------------------------------
# Problems: the paper's 28x28 0/1 MNIST, and two small test problems
# ---------------------------------------------------------------------------------------------------------------
TOY_SEED = 11
PROBLEM_GEOMETRY = {"mnist28": (GRID, N_NEIGHBOUR_COUPLINGS), "mnist14": (14, 7), "gauss2_8": (8, 7)}  # (grid, radius)
DIGITS_DIR = os.path.join(OUT_ROOT, "data")      # digit subsets written by make_digit_data.py
_DIGIT_PROBLEM = re.compile(r"mnist(28|14)_d(\d{2,})")


def problem_geometry(name):
    """(grid, default radius) of a problem: PROBLEM_GEOMETRY, or the digit subsets mnist28_d<digits> (28x28, radius
    14) and mnist14_d<digits> (2x2-pooled, radius 7), e.g. mnist14_d14 = the digits 1 and 4."""
    if name in PROBLEM_GEOMETRY:
        return PROBLEM_GEOMETRY[name]
    m = _DIGIT_PROBLEM.fullmatch(name)
    if m is None:
        raise ValueError(f"unknown problem {name!r}")
    return (GRID, N_NEIGHBOUR_COUPLINGS) if m.group(1) == "28" else (14, 7)


def problem_classes(name):
    """Number of classes of a problem (2 for mnist28 / mnist14 / gauss2_8)."""
    m = _DIGIT_PROBLEM.fullmatch(name)
    return 2 if m is None else len(m.group(2))


def problem_data_path(name):
    """The data file a problem is built from (recorded as data_sha256 in config.json): the digit-subset file, or the
    paper's 0/1 file for mnist28 / mnist14 (and, as before, for the synthetic gauss2_8)."""
    m = _DIGIT_PROBLEM.fullmatch(name)
    return DATA_PATH if m is None else os.path.join(DIGITS_DIR, f"digits_{m.group(2)}_28.npz")


def _normalise_like_paper(raw):
    """The paper's preprocessing: N(0, 0.01^2) pixel noise from the paper's key, then one global mean / std."""
    rawj = jnp.array(raw)
    _, _, image_noise_added_key = jr.split(jr.PRNGKey(KEY_SEED), 3)
    _, subkey = jr.split(image_noise_added_key)
    data, mean, std = normalize_samples(rawj + jr.normal(subkey, rawj.shape) * STD_ADDED_NOISE)
    return data, float(mean), float(std)


def toy_patterns(grid=8):
    """Two binary patterns on a grid: a square ring (class 0) and a vertical bar crossing it (class 1)."""
    c = (grid - 1) / 2.0
    ii, jj = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    cheb = np.maximum(np.abs(ii - c), np.abs(jj - c))
    ring = ((cheb >= 1.5) & (cheb <= 2.5)).astype(float)
    bar = ((np.abs(jj - c) <= 0.5) & (np.abs(ii - c) <= 2.5)).astype(float)
    return ring.ravel(), bar.ravel()


def make_problem(name, radius=None, neighbourhood="chebyshev"):
    """Returns dict(name, grid, n, radius, neighbourhood, conn, data (jnp, normalised), labels (np, 1 = class 1),
    mean, std).
    radius: Chebyshev coupling radius of the visible pairs instead of the problem's default (PROBLEM_GEOMETRY);
    grid - 1 couples all pairs. neighbourhood "von_neumann": Manhattan instead of Chebyshev distance (connectivity).

    mnist28  : the paper's problem (11,846 images, radius 14).
    mnist14  : the same images, 2x2 average-pooled to 14x14 before the paper's noise and normalisation (radius 7).
    gauss2_8 : 8x8 lattice, all-to-all couplings; class 0 = square ring, class 1 = vertical bar crossing it,
               each with i.i.d. N(0, 0.15^2) pixel noise in [0, 1] pixel units; 5,000 per class. Its forward
               marginals are two-Gaussian mixtures, so the exact score has precisely the one-hidden-unit form.
    """
    grid, r0 = problem_geometry(name)
    radius = r0 if radius is None else radius
    digits = None
    m = _DIGIT_PROBLEM.fullmatch(name)
    if m is not None:                        # digit subset: same noise and normalisation as the paper's data
        z = np.load(os.path.join(DIGITS_DIR, f"digits_{m.group(2)}_28.npz"))
        raw, digit_labels = z["images"], z["labels"]
        digits = sorted(int(d) for d in m.group(2))
        if sorted(np.unique(digit_labels).tolist()) != digits:
            raise RuntimeError(f"{name}: labels {np.unique(digit_labels)} do not match the digits {digits}")
        labels = np.searchsorted(digits, digit_labels)          # class index 0 .. C-1 in digit order
        if m.group(1) == "14":
            raw = raw.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196)
        data, mean, std = _normalise_like_paper(raw)
    elif name == "mnist28":
        data, labels, mean, std = load_data()
    elif name == "mnist14":
        raw = np.load(DATA_PATH)
        _, labels, _, _ = load_data()
        pooled = raw.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196)
        data, mean, std = _normalise_like_paper(pooled)
    elif name == "gauss2_8":
        ring, bar = toy_patterns(grid)
        rng = np.random.default_rng(TOY_SEED)
        labels = np.repeat([0, 1], 5000)
        rng.shuffle(labels)
        raw = np.where(labels[:, None] == 1, bar[None, :], ring[None, :]) + 0.15 * rng.standard_normal((labels.size, grid * grid))
        data, mean, std = _normalise_like_paper(raw)
        labels = labels.astype(np.int64)
    else:
        raise ValueError(name)
    if radius is not None:                   # explicit coupling radius (the data do not depend on it)
        radius = int(radius)
    conn = connectivity(grid, radius, neighbourhood)
    return {"name": name, "grid": grid, "n": grid * grid, "radius": radius, "neighbourhood": neighbourhood,
            "conn": conn, "data": data,
            "labels": np.asarray(labels, dtype=np.int64), "mean": mean, "std": std,
            "digits": digits if digits is not None else ([0, 1] if name.startswith("mnist") else None)}


class DenseVisible:
    """Batched gradient, Laplacian and energy of the paper's visible energy.

    E = sum_i [k_lin x^2/2 + k_duff x^4/4 + k_6 x^6/6 + bias x]
      + sum_{(i<j)} [c_lin (x_i - x_j)^2 + c_om x_i^2 x_j + c_duff (x_i - x_j)^4 / 4]
    With C, D the symmetric matrices of c_lin, c_duff and U the upper-triangular matrix of c_om:
      grad_k = self' + 2(d_C x_k - (Cx)_k) + 2 x_k (Ux)_k + (U^T x^2)_k
               + d_D x_k^3 - 3 x_k^2 (Dx)_k + 3 x_k (D x^2)_k - (D x^3)_k
      lap    = sum self'' + 2 sum d_C + 2 sum (Ux) + 3 sum [d_D x^2 - 2 x (Dx) + D x^2]
    where d_C, d_D are row sums.
    """

    def __init__(self, conn, n=N, terms=None, sextic=True):
        """terms None (default): every pair group and k_6, today's code path. A tuple of kept groups (--visible-impl
        dense_skip): grad_lap leaves out the other groups (and k_6 if not sextic) statically; their parameters must
        be exactly zero, as --pair-terms and the quartic projection keep them."""
        conn = np.asarray(conn)
        self.n = int(n)
        self.I = jnp.asarray(conn[:, 0])
        self.J = jnp.asarray(conn[:, 1])
        self.P = int(conn.shape[0])
        self.n_theta = 4 * self.n + 3 * self.P
        self.terms = None if terms is None else tuple(sorted(terms))
        self.sextic = bool(sextic)

    def split(self, theta):
        P, n = self.P, self.n
        return (theta[0:n], theta[n:2 * n], theta[2 * n:3 * n], theta[3 * n:3 * n + P], theta[3 * n + P:3 * n + 2 * P],
                theta[3 * n + 2 * P:3 * n + 3 * P], theta[3 * n + 3 * P:])

    def mats(self, theta):
        k_lin, k_duff, k6, c_lin, c_om, c_duff, bias = self.split(theta)
        Z = jnp.zeros((self.n, self.n), dtype=theta.dtype)
        Cu = Z.at[self.I, self.J].add(c_lin)
        Du = Z.at[self.I, self.J].add(c_duff)
        U = Z.at[self.I, self.J].add(c_om)
        return k_lin, k_duff, k6, bias, Cu + Cu.T, U, Du + Du.T

    def grad_lap(self, theta, X):
        """X: (B, N). Returns grad E (B, N) and Laplacian of E (B,)."""
        if self.terms is not None:
            return self._grad_lap_kept(theta, X)
        k_lin, k_duff, k6, bias, C, U, D = self.mats(theta)
        dC, dD = jnp.sum(C, axis=1), jnp.sum(D, axis=1)
        X2 = X * X
        X3 = X2 * X
        XC = X @ C
        XU = X @ U.T          # (Ux) per sample
        X2U = X2 @ U          # (U^T x^2) per sample
        XD = X @ D
        X2D = X2 @ D
        X3D = X3 @ D
        g = (k_lin * X + k_duff * X3 + k6 * X3 * X2 + bias
             + 2.0 * (dC * X - XC)
             + 2.0 * X * XU + X2U
             + dD * X3 - 3.0 * X2 * XD + 3.0 * X * X2D - X3D)
        lap = (jnp.sum(k_lin + 3.0 * k_duff * X2 + 5.0 * k6 * X2 * X2, axis=1)
               + 2.0 * jnp.sum(dC)
               + 2.0 * jnp.sum(XU, axis=1)
               + 3.0 * jnp.sum(dD * X2 - 2.0 * X * XD + X2D, axis=1))
        return g, lap

    def _grad_lap_kept(self, theta, X):
        """grad_lap with only the kept pair groups (and k_6 only if sextic): the products of the dropped groups
        multiply exact zeros (with --pair-terms lin, 10 of the 12 N x N products of a training step)."""
        k_lin, k_duff, k6, c_lin, c_om, c_duff, bias = self.split(theta)
        X2 = X * X
        X3 = X2 * X
        g = k_lin * X + k_duff * X3
        s = k_lin + 3.0 * k_duff * X2
        if self.sextic:
            g = g + k6 * X3 * X2
            s = s + 5.0 * k6 * X2 * X2
        g = g + bias
        lap = jnp.sum(s, axis=1)
        Z = jnp.zeros((self.n, self.n), dtype=theta.dtype)
        if "lin" in self.terms:
            Cu = Z.at[self.I, self.J].add(c_lin)
            C = Cu + Cu.T
            dC = jnp.sum(C, axis=1)
            g = g + 2.0 * (dC * X - X @ C)
            lap = lap + 2.0 * jnp.sum(dC)
        if "om" in self.terms:
            U = Z.at[self.I, self.J].add(c_om)
            XU = X @ U.T
            g = g + 2.0 * X * XU + X2 @ U
            lap = lap + 2.0 * jnp.sum(XU, axis=1)
        if "duff" in self.terms:
            Du = Z.at[self.I, self.J].add(c_duff)
            D = Du + Du.T
            dD = jnp.sum(D, axis=1)
            XD = X @ D
            X2D = X2 @ D
            g = g + dD * X3 - 3.0 * X2 * XD + 3.0 * X * X2D - X3 @ D
            lap = lap + 3.0 * jnp.sum(dD * X2 - 2.0 * X * XD + X2D, axis=1)
        return g, lap

    def energy(self, theta, X):
        k_lin, k_duff, k6, bias, C, U, D = self.mats(theta)
        dC, dD = jnp.sum(C, axis=1), jnp.sum(D, axis=1)
        X2 = X * X
        X3 = X2 * X
        e_self = jnp.sum(0.5 * k_lin * X2 + 0.25 * k_duff * X2 * X2 + k6 * X3 * X3 / 6.0 + bias * X, axis=1)
        e_lin = jnp.sum(dC * X2, axis=1) - jnp.sum(X * (X @ C), axis=1)
        e_om = jnp.sum(X2 * (X @ U.T), axis=1)
        e_duff = 0.25 * jnp.sum(dD * X2 * X2, axis=1) - jnp.sum(X3 * (X @ D), axis=1) + 0.75 * jnp.sum(X2 * (X2 @ D), axis=1)
        return e_self + e_lin + e_om + e_duff


class StencilVisible:
    """The visible energy of DenseVisible evaluated per lattice offset instead of with dense N x N matrices
    (--visible-impl stencil). Same flat parameters and interface (n, P, n_theta, grad_lap, energy); equal to
    DenseVisible up to rounding. Only the kept pair groups `terms` (and k_6 if `sextic`) are evaluated: the others
    must be exactly zero in theta, as --pair-terms and the quartic projection keep them.

    The pairs must form complete lattice offsets on a square grid, as connectivity() makes them (every pixel whose
    partner at offset (dy, dx) is on the grid has that pair). For pair (i, j), i < j (i = source), d = x_i - x_j,
    a = c_lin, u = c_om, c = c_duff:
        g_i += 2 a d + 2 u x_i x_j + c d^3,     g_j += -2 a d + u x_i^2 - c d^3
        lap += 4 a + 2 u x_j + 6 c d^2,         E += a d^2 + u x_i^2 x_j + c d^4 / 4
    Only slices, pads, elementwise products, reductions and one gather with unique indices (no scatter-add with
    colliding indices), so the forward and the reverse pass are deterministic on the GPU.
    """

    def __init__(self, conn, n=N, terms=tuple(sorted(PAIR_TERMS)), sextic=True):
        conn = np.asarray(conn)
        G = math.isqrt(int(n))
        if G * G != int(n):
            raise ValueError(f"StencilVisible needs a square grid, not n = {n}")
        self.n, self.G, self.P = int(n), G, int(conn.shape[0])
        self.n_theta = 4 * self.n + 3 * self.P
        self.terms, self.sextic = tuple(sorted(terms)), bool(sextic)
        if any(t not in PAIR_TERMS for t in self.terms):
            raise ValueError(f"unknown pair terms {self.terms}")
        if self.P and np.any(conn[:, 0] >= conn[:, 1]):
            raise ValueError("StencilVisible needs pairs (i, j) with i < j")
        ri, ci = np.divmod(conn[:, 0], G)
        rj, cj = np.divmod(conn[:, 1], G)
        dy, dx = rj - ri, cj - ci
        self.offsets, wins = [], []
        for oy, ox in sorted(set(zip(dy.tolist(), dx.tolist()))):
            sel = np.flatnonzero((dy == oy) & (dx == ox))
            win = np.full((G - oy, G - abs(ox)), -1, dtype=np.int64)
            win[ri[sel], ci[sel] - max(0, -ox)] = sel
            if (win < 0).any():
                raise ValueError(f"pair offset ({oy}, {ox}) is not a complete lattice offset")
            self.offsets.append((int(oy), int(ox)))
            wins.append(win.ravel())
        perm = np.concatenate(wins) if wins else np.zeros(0, dtype=np.int64)
        if not np.array_equal(np.sort(perm), np.arange(self.P)):
            raise ValueError("the pair list is not a set of lattice offsets")
        self.perm = perm                                 # pair index of each window slot, offset by offset
        self.cuts = np.cumsum([0] + [w.size for w in wins]).tolist()

    def _windows(self, c):
        """(P,) pair couplings -> one (G - dy, G - |dx|) array per offset, indexed by the source pixel."""
        cw = c[self.perm]
        return [cw[s:e].reshape(self.G - oy, self.G - abs(ox))
                for (oy, ox), s, e in zip(self.offsets, self.cuts[:-1], self.cuts[1:])]

    def _views(self, Xg, oy, ox):
        """Sources x_i and targets x_j of offset (oy, ox) as (B, G - oy, G - |ox|) views, and the pads that put
        per-source / per-target values back on the (G, G) grid."""
        G = self.G
        lo, hi = max(0, -ox), max(0, ox)
        xs = Xg[:, :G - oy, lo:G - hi]
        xt = Xg[:, oy:, hi:G - lo]
        return xs, xt, ((0, 0), (0, oy), (lo, hi)), ((0, 0), (oy, 0), (hi, lo))

    def _split_self(self, theta):
        n, P = self.n, self.P
        return theta[0:n], theta[n:2 * n], theta[2 * n:3 * n], theta[3 * n + 3 * P:]

    def grad_lap(self, theta, X):
        """X: (B, N). Returns grad E (B, N) and Laplacian of E (B,)."""
        n, P, G, B = self.n, self.P, self.G, X.shape[0]
        k_lin, k_duff, k6, bias = self._split_self(theta)
        X2 = X * X
        X3 = X2 * X
        g = k_lin * X + k_duff * X3
        s = k_lin + 3.0 * k_duff * X2
        if self.sextic:
            g = g + k6 * X3 * X2
            s = s + 5.0 * k6 * X2 * X2
        g = g + bias
        lap = jnp.sum(s, axis=1)
        if not self.terms or not self.offsets:
            return g, lap
        cw = {t: self._windows(theta[PAIR_TERMS[t](n, P)]) for t in self.terms}
        Xg = X.reshape(B, G, G)
        acc = jnp.zeros_like(Xg)
        for o, (oy, ox) in enumerate(self.offsets):
            xs, xt, pad_s, pad_t = self._views(Xg, oy, ox)
            d = xs - xt
            gs, gt = 0.0, 0.0
            if "lin" in cw:
                f = 2.0 * cw["lin"][o] * d
                gs, gt = gs + f, gt - f
            if "om" in cw:
                u = cw["om"][o]
                gs = gs + 2.0 * u * xs * xt
                gt = gt + u * xs * xs
                lap = lap + 2.0 * jnp.sum(u * xt, axis=(1, 2))
            if "duff" in cw:
                c = cw["duff"][o]
                d2 = d * d
                f = c * d2 * d
                gs, gt = gs + f, gt - f
                lap = lap + 6.0 * jnp.sum(c * d2, axis=(1, 2))
            acc = acc + jnp.pad(gs, pad_s) + jnp.pad(gt, pad_t)
        if "lin" in cw:
            lap = lap + 4.0 * jnp.sum(theta[clin_slice(n, P)])        # independent of x
        return g + acc.reshape(B, n), lap

    def energy(self, theta, X):
        n, P, G, B = self.n, self.P, self.G, X.shape[0]
        k_lin, k_duff, k6, bias = self._split_self(theta)
        X2 = X * X
        e = jnp.sum(0.5 * k_lin * X2 + 0.25 * k_duff * X2 * X2 + bias * X, axis=1)
        if self.sextic:
            e = e + jnp.sum(k6 * X2 * X2 * X2 / 6.0, axis=1)
        if not self.terms or not self.offsets:
            return e
        cw = {t: self._windows(theta[PAIR_TERMS[t](n, P)]) for t in self.terms}
        Xg = X.reshape(B, G, G)
        for o, (oy, ox) in enumerate(self.offsets):
            xs, xt, _, _ = self._views(Xg, oy, ox)
            d = xs - xt
            if "lin" in cw:
                e = e + jnp.sum(cw["lin"][o] * d * d, axis=(1, 2))
            if "om" in cw:
                e = e + jnp.sum(cw["om"][o] * xs * xs * xt, axis=(1, 2))
            if "duff" in cw:
                d2 = d * d
                e = e + 0.25 * jnp.sum(cw["duff"][o] * d2 * d2, axis=(1, 2))
        return e


VISIBLE_IMPLS = ("dense", "dense_skip", "stencil")


def make_visible(conn, n=N, impl="dense", terms=None, sextic=True):
    """The visible energy of the training loss (--visible-impl). 'dense': DenseVisible(conn, n), all groups and k_6,
    the default and the path that matches the paper's code. 'dense_skip': the same products without the dropped
    groups (and k_6 unless sextic). 'stencil': StencilVisible, per lattice offset, only the kept groups. All three
    are equal up to rounding as long as the dropped groups (and k_6 of the quartic model) are exactly zero.
    terms: kept pair groups (None = all)."""
    if impl == "dense":
        return DenseVisible(conn, n)
    kept = tuple(sorted(PAIR_TERMS)) if terms is None else tuple(terms)
    if impl == "dense_skip":
        return DenseVisible(conn, n, terms=kept, sextic=sextic)
    if impl == "stencil":
        return StencilVisible(conn, n, terms=kept, sextic=sextic)
    raise ValueError(f"--visible-impl must be one of {VISIBLE_IMPLS}, not {impl!r}")


# ---------------------------------------------------------------------------------------------------------------
# Hidden units (Ising limit, averaged out)
# ---------------------------------------------------------------------------------------------------------------
def _part(draws, part):
    """All S draws (part None), the first / second half (part 0 / 1), or the i-th of k equal consecutive parts
    (part (k, i); (2, 0) and (2, 1) are the halves), for the jackknife."""
    if part is None:
        return draws
    if isinstance(part, tuple):
        k, i = part
        n = draws.shape[0] // k
        return draws[i * n:(i + 1) * n]
    h = draws.shape[0] // 2
    return draws[:h] if part == 0 else draws[h:2 * h]


def sampled_spin_mean(A, T, key, S, part=None):
    """Plug-in estimate of m = tanh(a / T) from S independent +-1 spins per entry, as a device measures it by a
    finite time average. Its value is the sample mean m_hat; its derivative is the plug-in fluctuation response
    d m_hat / d a = (1 - m_hat^2) / T (the exact rule with m -> m_hat), so autodiff of a loss built on it gives the
    force-matching gradient with measured statistics. No gradient flows through the sampling itself.
    part = 0 / 1 uses only the first / second half of the same S draws (same key and fields: same draws)."""
    up = _part(jr.bernoulli(key, 0.5 * (1.0 + jnp.tanh(A / T)), shape=(S,) + A.shape), part)
    mh = jax.lax.stop_gradient(jnp.mean(2.0 * up.astype(A.dtype) - 1.0, axis=0))
    u = (A - jax.lax.stop_gradient(A)) / T                 # zero in value, dA / T in derivative
    return mh + (1.0 - mh * mh) * u


def sampled_state_mean(A, T, key, S, part=None):
    """Plug-in estimate of p = softmax(a / T) (rows) from S independent Potts states per row: the value is the
    occupation frequency p_hat; the derivative is the plug-in covariance (diag(p_hat) - p_hat p_hat^T) dA / T.
    part = 0 / 1: the first / second half of the same S draws."""
    idx = _part(jr.categorical(key, A / T, axis=-1, shape=(S,) + A.shape[:-1]), part)
    ph = jax.lax.stop_gradient(jnp.mean(jax.nn.one_hot(idx, A.shape[-1], dtype=A.dtype), axis=0))
    u = (A - jax.lax.stop_gradient(A)) / T
    return ph + ph * u - ph * jnp.sum(ph * u, axis=-1, keepdims=True)


def loss_draw_occupation(params, X, T, key, S):
    """Each Potts state's visit fraction among the S hidden draws per configuration that the measured-statistics loss
    (make_loss_per_sample with samples = S, replicates = 1, called with `key`) takes: the same fields A (summed in the
    same order as hidden_terms_quad / potts_terms), the same key (k_h, the first of jr.split(key)) and the same
    categorical draws. A device can count the visits of the reads it already makes. Returns (M,) fractions."""
    W, b = params["W"], params["b"]
    A = X @ W.T + (X * X) @ params["Q"].T + b if "Q" in params else X @ W.T + b
    k_h = jr.split(key)[0]
    idx = jr.categorical(k_h, A / T, axis=-1, shape=(S,) + A.shape[:-1])
    return jnp.bincount(idx.ravel(), length=A.shape[-1]).astype(A.dtype) / idx.size


def hidden_terms(W, b, X, T, stat=None):
    """Contributions of K Ising hidden units to grad F and to the Laplacian of F (DESIGN.md section 1).

    W: (K, N), b: (K,), X: (B, N). With a = b + W x, m = tanh(a/T), v = 1 - m^2:
      grad F_h = -sum_j m_j W_j,   lap F_h = -(1/T) sum_j v_j |W_j|^2.
    stat(A), if given, replaces m = tanh(A / T) (e.g. a sampled estimate; v = 1 - m^2 follows it).
    Returns (grad (B, N), lap (B,), a (B, K), m (B, K)).
    """
    A = X @ W.T + b
    m = jnp.tanh(A / T) if stat is None else stat(A)
    v = 1.0 - m * m
    return -(m @ W), -(v @ jnp.sum(W * W, axis=1)) / T, A, m


def hidden_free_energy(W, b, X, T):
    """F_h(x) = -T sum_j log 2cosh(a_j/T), computed stably."""
    A = X @ W.T + b
    return -T * jnp.sum(jnp.logaddexp(A / T, -A / T), axis=1)


def potts_terms(W, b, X, T, stat=None):
    """One Potts hidden unit with M states (rows of W: (M, N), b: (M,)), averaged out:
      F_h = -T log sum_m exp(a_m / T),  a_m = b_m + W_m . x,  p = softmax(a / T)
      grad F_h = -sum_m p_m W_m,   lap F_h = -(1/T) [sum_m p_m |W_m|^2 - |sum_m p_m W_m|^2]
    stat(A), if given, replaces p = softmax(A / T).
    Returns (grad (B, N), lap (B,), a (B, M), p (B, M)). M = 2 is equivalent to one Ising unit."""
    A = X @ W.T + b
    p = jax.nn.softmax(A / T, axis=1) if stat is None else stat(A)
    mW = p @ W
    lap = -(p @ jnp.sum(W * W, axis=1) - jnp.sum(mW * mW, axis=1)) / T
    return -mW, lap, A, p


def potts_free_energy(W, b, X, T):
    A = X @ W.T + b
    return -T * jax.scipy.special.logsumexp(A / T, axis=1)


def local_terms(w, c, X, T, stat=None):
    """One Ising hidden unit per pixel, coupled only to its own pixel (a_i = c_i + w_i x_i), averaged out:
      F_loc = -T sum_i log 2cosh(a_i / T)   (a self-energy term: an asymmetric double well per pixel)
      grad_i = -m_i w_i,   lap = -(1/T) sum_i (1 - m_i^2) w_i^2,   m = tanh(a / T) (or stat(A))."""
    A = X * w + c
    m = jnp.tanh(A / T) if stat is None else stat(A)
    return -m * w, -jnp.sum((1.0 - m * m) * (w * w), axis=1) / T, A, m


def hidden_terms_quad(W, Q, b, X, T, hidden="ising", stat=None):
    """K Ising units (or one Potts unit with K states) whose input also contains the squared pixels:
      a_j = b_j + W_j . x + Q_j . x^2      (Q_j x_i^2: unit j modulates the spring constant of pixel i)
    With G_j = W_j + 2 Q_j * x (the gradient of a_j):
      Ising:  grad F_h = -sum_j m_j G_j,  lap F_h = -sum_j [ (1 - m_j^2) |G_j|^2 / T + 2 m_j sum_i Q_ji ]
      Potts:  grad F_h = -sum_m p_m G_m,  lap F_h = -(1/T) [sum_m p_m |G_m|^2 - |sum_m p_m G_m|^2] - 2 sum_m p_m sum_i Q_mi
    Returns (grad (B, N), lap (B,), a (B, K), m or p (B, K))."""
    X2 = X * X
    A = X @ W.T + X2 @ Q.T + b
    GG = jnp.sum(W * W, axis=1) + 4.0 * X @ (W * Q).T + 4.0 * X2 @ (Q * Q).T     # |G_j|^2, (B, K)
    sQ = jnp.sum(Q, axis=1)
    if hidden == "ising":
        m = jnp.tanh(A / T) if stat is None else stat(A)
        grad = -(m @ W) - 2.0 * X * (m @ Q)
        lap = -jnp.sum((1.0 - m * m) * GG, axis=1) / T - 2.0 * (m @ sQ)
        return grad, lap, A, m
    p = jax.nn.softmax(A / T, axis=1) if stat is None else stat(A)
    gbar = p @ W + 2.0 * X * (p @ Q)
    lap = -(jnp.sum(p * GG, axis=1) - jnp.sum(gbar * gbar, axis=1)) / T - 2.0 * (p @ sQ)
    return -gbar, lap, A, p


HARMONIC_KEYS = ("V", "c")       # parameters of the harmonic hidden modes (train_latent.py --harmonic)


def harmonic_terms(V, c, X):
    """Contributions of R harmonic hidden modes (train_latent.py --harmonic) to grad F and to the Laplacian of F.

    Mode j is a linear oscillator, a continuous hidden variable h_j with energy kappa_j h_j^2 / 2 - h_j (u_j . x + c'_j).
    Integrated out exactly (a Gaussian integral over h_j, so no hidden statistics are needed):
      F_h(x) = -1/2 sum_j (v_j . x + c_j)^2 + const,    v_j = u_j / sqrt(kappa_j),  c_j = c'_j / sqrt(kappa_j)
    (V (R, N) with rows v_j and c (R,) are the free parameters; the mode's mean displacement is (v_j . x + c_j) /
    sqrt(kappa_j)). With s = V x + c:
      grad F_h = -sum_j s_j v_j,    lap F_h = -sum_j |v_j|^2   (constant in x).
    The modes only add variance along their v_j (anti-springs, a negative-definite quadratic form); the visible
    springs supply the isotropic part, and the quartic (k_duff >= K4_FLOOR) or sextic visible self terms keep F
    confining at large |x|. X: (B, N). Returns (grad (B, N), lap (a scalar: the same for every sample), s (B, R))."""
    S = X @ V.T + c
    return -(S @ V), -jnp.sum(V * V), S


def harmonic_free_energy(V, c, X):
    """F_h(x) = -1/2 sum_j (v_j . x + c_j)^2 of the harmonic modes (harmonic_terms), per sample (B,)."""
    S = X @ V.T + c
    return -0.5 * jnp.sum(S * S, axis=1)


# ---------------------------------------------------------------------------------------------------------------
# Per-patch Potts hidden units (train_latent.py --patch SIDE:STRIDE:STATES; 2026-11-13)
# ---------------------------------------------------------------------------------------------------------------
PATCH_KEYS = ("Wp", "bp", "Qp")  # parameters of the per-patch Potts units (train_latent.py --patch)
PATCH_SEED0 = 20000              # k-means seed of patch p for --seed s: PATCH_SEED0 + 1000 (s - KEY_SEED) + p


def parse_patch_spec(spec, grid):
    """train_latent.py --patch 'SIDE:STRIDE:STATES' on a grid x grid lattice -> (side, stride, S); '' -> None (no patch
    units). Valid if and only if 2 <= SIDE <= grid, 1 <= STRIDE <= SIDE, (grid - SIDE) % STRIDE == 0 and STATES >= 2;
    anything else raises ValueError. The canonical form is patch_spec_str((side, stride, S))."""
    spec = (spec or "").strip()
    if not spec:
        return None
    rule = (f"--patch {spec!r}: expected SIDE:STRIDE:STATES (integers) with 2 <= SIDE <= {grid}, 1 <= STRIDE <= SIDE, "
            f"({grid} - SIDE) % STRIDE == 0 and STATES >= 2 on the {grid}x{grid} grid (e.g. 7:3:32 on 28x28)")
    parts = spec.split(":")
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        raise ValueError(rule) from None
    if len(vals) != 3:
        raise ValueError(rule)
    side, stride, S = vals
    if not (2 <= side <= grid and 1 <= stride <= side and (grid - side) % stride == 0 and S >= 2):
        raise ValueError(rule)
    return side, stride, S


def patch_spec_str(patch):
    """Canonical form of a parsed --patch spec ('' for None), as recorded in config.json 'patch'."""
    return "" if patch is None else f"{patch[0]}:{patch[1]}:{patch[2]}"


class PatchGeometry:
    """Overlapping square patches on a G x G grid (train_latent.py --patch): side s, stride r, n_p = (G - s) / r + 1
    patches per row and column, P = n_p^2. Patch p = pi n_p + pj; its entry j = u s + v (row-major inside the patch)
    is pixel (pi r + u) G + pj r + v. Attributes G, s, r, n_p, P, d = s^2, N = G^2, idx (P, d) int64 (the pixels of
    each patch), coverage (N,) (the number of patches containing each pixel), central (P,) bool (the patch centre in
    rows and columns 6-21 of 28, scaled for other grids).
    unfold / fold use strided slices and pads only (no gather, no scatter-add): deterministic on the GPU in the forward
    and the reverse pass (the transpose of a slice is a pad, and of a pad a slice)."""

    def __init__(self, grid, side, stride):
        G, s, r = int(grid), int(side), int(stride)
        if not (2 <= s <= G and 1 <= r <= s and (G - s) % r == 0):
            raise ValueError(f"invalid patch geometry side {s}, stride {r} on a {G}x{G} grid: needs 2 <= side <= {G}, "
                             f"1 <= stride <= side and ({G} - side) % stride == 0")
        self.G, self.s, self.r = G, s, r
        self.n_p = (G - s) // r + 1
        self.P, self.d, self.N = self.n_p * self.n_p, s * s, G * G
        pi, pj = np.divmod(np.arange(self.P), self.n_p)
        u, v = np.divmod(np.arange(self.d), s)
        self.idx = ((pi[:, None] * r + u[None, :]) * G + pj[:, None] * r + v[None, :]).astype(np.int64)
        self.coverage = np.bincount(self.idx.ravel(), minlength=self.N)
        ci, cj = pi * r + (s - 1) / 2.0, pj * r + (s - 1) / 2.0
        lo, hi = 6.0 * G / 28.0, 21.0 * G / 28.0
        self.central = (ci >= lo) & (ci <= hi) & (cj >= lo) & (cj <= hi)

    def unfold(self, X):
        """X (B, N) -> the patch vectors (P, B, d), patch-major: one strided slice per entry (u, v) of the patch."""
        B, G, s, r, n_p = X.shape[0], self.G, self.s, self.r, self.n_p
        Xg = X.reshape(B, G, G)
        hi = r * (n_p - 1) + 1
        cols = [lax.slice(Xg, (0, u, v), (B, u + hi, v + hi), (1, r, r)) for u in range(s) for v in range(s)]
        st = jnp.stack(cols, axis=-1)                                     # (B, n_p, n_p, d)
        return jnp.transpose(st, (1, 2, 0, 3)).reshape(self.P, B, self.d)

    def fold(self, Gp):
        """Gp (P, B, d) -> (B, N): the adjoint of unfold (the sum over the patches that contain a pixel), one pad per
        entry (u, v) of the patch."""
        B, G, s, r, n_p = Gp.shape[1], self.G, self.s, self.r, self.n_p
        Gq = jnp.transpose(Gp.reshape(n_p, n_p, B, self.d), (2, 0, 1, 3))    # (B, n_p, n_p, d)
        zero = jnp.zeros((), Gp.dtype)
        acc = None
        for u in range(s):
            for v in range(s):
                t = lax.pad(lax.index_in_dim(Gq, u * s + v, axis=3, keepdims=False), zero,
                            ((0, 0, 0), (u, G - u - r * (n_p - 1) - 1, r - 1), (v, G - v - r * (n_p - 1) - 1, r - 1)))
                acc = t if acc is None else acc + t
        return acc.reshape(B, G * G)


@lru_cache(maxsize=None)
def patch_geometry(n, P, d):
    """The PatchGeometry of P patches of d pixels on n pixels, from the shapes alone (n = G^2, d = side^2, P = n_p^2;
    stride = (G - side) / (n_p - 1), = side for n_p = 1), so that the presence of 'Wp' in params switches the patch
    term on without new arguments. ValueError if the shapes describe no valid geometry."""
    n, P, d = int(n), int(P), int(d)
    G, side, n_p = math.isqrt(n), math.isqrt(d), math.isqrt(P)
    if G * G != n or side * side != d or n_p * n_p != P or n_p < 1:
        raise ValueError(f"patch units of shapes P = {P}, d = {d} on n = {n} pixels: not a square patch grid")
    if n_p == 1:
        stride = side
        ok = side == G
    else:
        stride = (G - side) // (n_p - 1)
        ok = stride >= 1 and stride * (n_p - 1) == G - side
    if not ok:
        raise ValueError(f"patch units of shapes P = {P}, d = {d} on a {G}x{G} grid: no stride gives {n_p} x {n_p} "
                         f"patches of side {side}")
    return PatchGeometry(G, side, stride)


def _pdot(a, w):
    """(P, B, d) x (P, S, d) -> (P, B, S): a batched product over the patches (one dot_general)."""
    return lax.dot_general(a, w, (((2,), (2,)), ((0,), (0,))))


def _pdot_t(p, w):
    """(P, B, S) x (P, S, d) -> (P, B, d)."""
    return lax.dot_general(p, w, (((2,), (1,)), ((0,), (0,))))


def patch_terms(Wp, Qp, bp, X, T, with_lap=True):
    """Contributions of P per-patch Potts units (train_latent.py --patch) to grad F and to the Laplacian of F: the
    Potts branch of hidden_terms_quad applied to each patch. Unit p has S states with fields
      a_ps = bp_ps + Wp_ps . x_p (+ Qp_ps . x_p^2),   pi_ps = softmax_s(a_ps / T),   G_ps = Wp_ps + 2 Qp_ps * x_p,
      gbar_p = sum_s pi_ps G_ps:
      grad F_patch = -fold(gbar)   (fold: the sum over the patches that contain a pixel; PatchGeometry.fold),
      lap F_patch  = sum_p [ -(sum_s pi_ps |G_ps|^2 - |gbar_p|^2) / T - 2 sum_s pi_ps sum_j Qp_psj ],
      |G_ps|^2 = |Wp_ps|^2 + 4 x_p . (Wp_ps * Qp_ps) + 4 x_p^2 . Qp_ps^2.
    The Hessian is a sum of per-patch blocks, so overlapping patches add no cross terms. Wp (P, S, d), Qp (P, S, d) or
    None (no squared-pixel couplings), bp (P, S), X (B, N); the geometry follows from the shapes (patch_geometry).
    Returns (grad (B, N), lap (B,) or None if not with_lap, fields A (P, B, S), posteriors pi (P, B, S)).
    Strided slices, pads, batched dot_generals, softmax and reductions only (no gather, no scatter)."""
    geom = patch_geometry(X.shape[1], Wp.shape[0], Wp.shape[2])
    Xp = geom.unfold(X)                                                   # (P, B, d)
    A = _pdot(Xp, Wp) + bp[:, None, :]
    if Qp is not None:
        X2p = Xp * Xp
        A = A + _pdot(X2p, Qp)
    p = jax.nn.softmax(A / T, axis=-1)
    gbar = _pdot_t(p, Wp)
    if Qp is not None:
        gbar = gbar + 2.0 * Xp * _pdot_t(p, Qp)
    grad = -geom.fold(gbar)
    if not with_lap:
        return grad, None, A, p
    GG = jnp.sum(Wp * Wp, axis=-1)[:, None, :]
    if Qp is not None:
        GG = GG + 4.0 * _pdot(Xp, Wp * Qp) + 4.0 * _pdot(X2p, Qp * Qp)
    lap = -(jnp.sum(p * GG, axis=(0, 2)) - jnp.sum(gbar * gbar, axis=(0, 2))) / T
    if Qp is not None:
        lap = lap - 2.0 * jnp.sum(p * jnp.sum(Qp, axis=-1)[:, None, :], axis=(0, 2))
    return grad, lap, A, p


def patch_free_energy(Wp, Qp, bp, X, T):
    """F_patch(x) = -T sum_p logsumexp_s(a_ps / T) of the per-patch units (patch_terms), per sample (B,)."""
    geom = patch_geometry(X.shape[1], Wp.shape[0], Wp.shape[2])
    Xp = geom.unfold(X)
    A = _pdot(Xp, Wp) + bp[:, None, :]
    if Qp is not None:
        A = A + _pdot(Xp * Xp, Qp)
    return -T * jnp.sum(jax.scipy.special.logsumexp(A / T, axis=-1), axis=0)


def in_compute_dtype(fn, dtype):
    """--precision mixed: fn(params, X, key) evaluated with the parameters and X cast to `dtype` (float32) and every
    matrix product at full float32 precision (HIGHEST: XLA may otherwise use TF32 for float32 products, whose
    11-bit significand is far too coarse once fields are divided by T). The precision is fixed when the products
    are traced and carries over to their transposes in the backward pass. The casts are differentiable, so the
    parameter gradient comes back in the parameters' own dtype (float64); the result stays in `dtype`."""
    def wrapped(params, X, key=None):
        p = {k: v.astype(dtype) for k, v in params.items()}
        with jax.default_matmul_precision("float32"):
            return fn(p, X.astype(dtype), key)

    return wrapped


def make_loss_per_sample(vis, T, K, hidden="ising", local=False, samples=0, jackknife=False, replicates=1,
                         compute_dtype=None):
    """Per-sample implicit score-matching loss at k_B T = T for p ~ exp(-F/T):
    -lap F / T + |grad F|^2 / (2 T^2). hidden = 'ising' (K units) or 'potts' (one unit with K states);
    if params contains 'Q', the hidden units also couple to the squared pixels (hidden_terms_quad);
    local = True adds one Ising unit per pixel (params 'wl', 'bl').
    samples = S > 0: the hidden statistics (m, p) are plug-in estimates from S sampled hidden states per sample
    (sampled_spin_mean / sampled_state_mean, drawn with `key`), so the parameter gradient is the force-matching rule
    with the hidden units' statistics measured by finite averages; the visible force includes the same estimates.
    The plug-in loss is biased at O(1/S) (by -V/(2 S T^2) on average, V the hidden force variance).
    jackknife = True (or 1) removes that order: 2 L(all S draws) - [L(first S/2) + L(second S/2)] / 2, from the same
    draws (a device combines the updates of a full and of two half averaging windows); the remaining bias is O(1/S^2).
    jackknife = 2 (S divisible by 4) also removes the O(1/S^2) order, by Richardson extrapolation over the full set,
    its halves and its quarters: (8/3) L(S) - 2 mean L(halves) + (1/3) mean L(quarters). The loss is quadratic and
    its parameter gradient at most cubic in the sampled hidden means of one configuration, so their expectations over
    n draws are exactly a + b/n + c/n^2 and this combination is unbiased for both (to first order in the fluctuations
    it has the same noise as the plug-in estimate; a device combines a full, two half and four quarter windows).
    replicates = R > 1 (diagnostic, needs samples > 0): the mean of R independent estimates, each from its own S
    draws (keys split from `key`). It keeps the bias of one S-draw estimate and divides its variance by R, which
    separates bias from noise (R S draws pooled into one estimate would have the smaller bias of S' = R S).
    compute_dtype = jnp.float32 (--precision mixed): the loss and its parameter gradient in float32 with full-float32
    products (in_compute_dtype); None (default) leaves the function exactly as before. Exact statistics only.
    If params contains 'V' (and 'c'), the R harmonic hidden modes are added last (harmonic_terms; train_latent.py
    --harmonic): exact Gaussian integrals, the same in every sampled-statistics variant; in --precision mixed their two
    products are float32 at HIGHEST like the hidden units'. Without 'V' every path is exactly as before.
    If params contains 'Wp' (and 'bp', optionally 'Qp'), the per-patch Potts units (patch_terms; train_latent.py
    --patch) are added after everything else, with exact statistics only (samples = 0; ValueError otherwise); in
    --precision mixed their products are float32 at HIGHEST too. Without 'Wp' every path is exactly as before."""
    terms = hidden_terms if hidden == "ising" else potts_terms
    if replicates < 1 or (replicates > 1 and samples < 1):
        raise ValueError("replicates must be >= 1, and > 1 only with samples > 0")
    if compute_dtype is not None and samples > 0:
        raise ValueError("compute_dtype needs exact hidden statistics (samples = 0): the sampled draws depend on the "
                         "dtype of the fields")
    if jackknife not in (False, True, 0, 1, 2) or (jackknife == 2 and (samples < 4 or samples % 4)):
        raise ValueError("jackknife must be False / 0, True / 1, or 2 (2 needs samples divisible by 4)")

    def loss(params, X, key=None):
        if "Wp" in params and samples > 0:       # trace time only: the default jaxpr is unchanged
            raise ValueError("patch units need exact hidden statistics")
        gv, lapv = vis.grad_lap(params["theta"], X)

        def per_sample(st_h, st_l):              # summation order as before (exact path bit-identical)
            g, lap = gv, lapv
            if K > 0 and "Q" in params:
                gh, laph, _, _ = hidden_terms_quad(params["W"], params["Q"], params["b"], X, T, hidden, stat=st_h)
                g = g + gh
                lap = lap + laph
            elif K > 0:
                gh, laph, _, _ = terms(params["W"], params["b"], X, T, stat=st_h)
                g = g + gh
                lap = lap + laph
            if local:
                gl, lapl, _, _ = local_terms(params["wl"], params["bl"], X, T, stat=st_l)
                g = g + gl
                lap = lap + lapl
            if "V" in params:                    # harmonic modes (--harmonic), last: the terms above keep their order
                gm, lapm, _ = harmonic_terms(params["V"], params["c"], X)
                g = g + gm
                lap = lap + lapm
            if "Wp" in params:                   # per-patch Potts units (--patch), after the modes
                gp, lapp, _, _ = patch_terms(params["Wp"], params.get("Qp"), params["bp"], X, T)
                g = g + gp
                lap = lap + lapp
            return -lap / T + jnp.sum(g * g, axis=1) / (2.0 * T * T)

        if samples == 0:
            return per_sample(None, None)

        def estimate(key):
            k_h, k_l = jr.split(key)

            def stats(part):
                sh = partial(sampled_spin_mean if hidden == "ising" else sampled_state_mean, T=T, key=k_h,
                             S=samples, part=part)
                return sh, partial(sampled_spin_mean, T=T, key=k_l, S=samples, part=part)

            if not jackknife:
                return per_sample(*stats(None))
            if jackknife == 2:                    # second order: Richardson over S, S/2 and S/4 draws (same draws)
                halves = 0.5 * (per_sample(*stats(0)) + per_sample(*stats(1)))
                quarters = 0.25 * sum(per_sample(*stats((4, q))) for q in range(4))
                return (8.0 / 3.0) * per_sample(*stats(None)) - 2.0 * halves + (1.0 / 3.0) * quarters
            return 2.0 * per_sample(*stats(None)) - 0.5 * (per_sample(*stats(0)) + per_sample(*stats(1)))

        if replicates == 1:                       # unchanged key use (runs before --hidden-replicates reproduce)
            return estimate(key)
        return jnp.mean(jax.vmap(estimate)(jr.split(key, replicates)), axis=0)

    if compute_dtype is None:
        return loss
    return in_compute_dtype(loss, compute_dtype)


def make_loss(vis, T, K, hidden="ising", local=False, samples=0, jackknife=False, replicates=1, compute_dtype=None):
    """Batch mean of make_loss_per_sample (the paper's loss for K = 0). With compute_dtype (float32), the per-sample
    losses are averaged in float64."""
    per_sample = make_loss_per_sample(vis, T, K, hidden, local, samples, jackknife, replicates, compute_dtype)

    if compute_dtype is None:
        def loss(params, X, key=None):
            return jnp.mean(per_sample(params, X, key))
    else:
        def loss(params, X, key=None):
            return jnp.mean(per_sample(params, X, key).astype(jnp.float64))

    return loss


# ---------------------------------------------------------------------------------------------------------------
# Local (CD-1, short-release) learning rule for the hidden parameters (DESIGN section 2)
# ---------------------------------------------------------------------------------------------------------------
HIDDEN_KEYS = ("W", "b", "Q", "wl", "bl")


def make_force(vis, T, K, hidden="ising", local=False):
    """grad F (B, N) of the model of make_loss_per_sample: visible part plus the averaged-out hidden units (and the
    harmonic modes, if params contains 'V', and the per-patch Potts units, if it contains 'Wp')."""
    terms = hidden_terms if hidden == "ising" else potts_terms

    def force(params, X):
        g, _ = vis.grad_lap(params["theta"], X)
        if K > 0 and "Q" in params:
            g = g + hidden_terms_quad(params["W"], params["Q"], params["b"], X, T, hidden)[0]
        elif K > 0:
            g = g + terms(params["W"], params["b"], X, T)[0]
        if local:
            g = g + local_terms(params["wl"], params["bl"], X, T)[0]
        if "V" in params:                        # harmonic modes (--harmonic)
            g = g + harmonic_terms(params["V"], params["c"], X)[0]
        if "Wp" in params:                       # per-patch Potts units (--patch)
            g = g + patch_terms(params["Wp"], params.get("Qp"), params["bp"], X, T, with_lap=False)[0]
        return g

    return force


def make_hidden_free_energy(T, K, hidden="ising", local=False):
    """Batch sum of the hidden part of F(x); hp holds the hidden parameters only (HIDDEN_KEYS present in the model).
    Its parameter gradient is what a hidden node measures locally (m -> p_m for a Potts unit):
      dF/dW_ji = -m_j x_i,  dF/db_j = -m_j,  dF/dQ_ji = -m_j x_i^2;  per-pixel units: dF/dwl_i = -m_i x_i,  dF/dbl_i = -m_i."""

    def fh(hp, X):
        f = 0.0
        if K > 0:
            A = X @ hp["W"].T + hp["b"]
            if "Q" in hp:
                A = A + (X * X) @ hp["Q"].T
            if hidden == "ising":
                f = f - T * jnp.sum(jnp.logaddexp(A / T, -A / T))
            else:
                f = f - T * jnp.sum(jax.scipy.special.logsumexp(A / T, axis=1))
        if local:
            Al = X * hp["wl"] + hp["bl"]
            f = f - T * jnp.sum(jnp.logaddexp(Al / T, -Al / T))
        return f

    return fh


def make_hidden_hebb_sampled(T, K, hidden="ising", local=False):
    """jax.grad of make_hidden_free_energy with the hidden statistics MEASURED: m_j (or p_m) and the per-pixel m_i are
    replaced by the mean of S sampled hidden states per configuration (+-1 spins, one-hot Potts states), as a device
    reads them. Returns hebb(hp, X, key, S) -> dict like dfh: dF/dW_ji = -mhat_j x_i, dF/db_j = -mhat_j,
    dF/dQ_ji = -mhat_j x_i^2, dF/dwl_i = -mhat_i x_i, dF/dbl_i = -mhat_i (batch sums). Linear in the sampled means, so
    unbiased for any S >= 1; as S -> infinity it equals jax.grad(make_hidden_free_energy(...))."""

    def surrogate(hp, X, key, S):
        if "Wp" in hp:                           # per-patch units (--patch): exact rule only (trace time)
            raise NotImplementedError("make_hidden_hebb_sampled does not support per-patch Potts units (--patch)")
        k_h, k_l = jr.split(key)
        f = 0.0
        if K > 0:
            A = X @ hp["W"].T + hp["b"]
            if "Q" in hp:
                A = A + (X * X) @ hp["Q"].T
            Ad = jax.lax.stop_gradient(A)
            if hidden == "ising":
                up = jr.bernoulli(k_h, 0.5 * (1.0 + jnp.tanh(Ad / T)), shape=(S,) + A.shape)
                mh = jnp.mean(2.0 * up.astype(A.dtype) - 1.0, axis=0)
            else:
                idx = jr.categorical(k_h, Ad / T, axis=-1, shape=(S,) + A.shape[:-1])
                mh = jnp.mean(jax.nn.one_hot(idx, A.shape[-1], dtype=A.dtype), axis=0)
            f = f - jnp.sum(mh * A)
        if local:
            Al = X * hp["wl"] + hp["bl"]
            up = jr.bernoulli(k_l, 0.5 * (1.0 + jnp.tanh(jax.lax.stop_gradient(Al) / T)), shape=(S,) + Al.shape)
            f = f - jnp.sum(jnp.mean(2.0 * up.astype(Al.dtype) - 1.0, axis=0) * Al)
        return f

    return jax.grad(surrogate)


def make_cd1_hidden_grad(vis, T, K, hidden="ising", local=False, samples=0):
    """CD-1 estimate of the score-matching gradient dJ_T/d(hidden parameters) (DESIGN section 2):
      x' = x - (dt/2) grad F(x) + sqrt(T dt) xi      (a short release of the pixels; the hidden units stay in equilibrium)
      g  = 2 / (T^2 dt) * [ mean_b dF/dtheta_h (x_b) - mean_{b,r} dF/dtheta_h (x'_{b,r}) ]
    In expectation over xi this is dJ_T/dtheta_h + O(dt) (Hyvarinen 2007, with F in place of E). Each sample gets
    `releases` independent releases (static). Returns a dict over the hidden keys present in params.
    samples = S > 0: the hidden statistics in dF/dtheta_h are measured (make_hidden_hebb_sampled): S sampled hidden
    states per released configuration and releases * S at the clamped data (the variance-optimal split of the reads).
    The estimate stays unbiased (it is linear in the sampled means); only its noise grows. samples = 0: exact
    averages, bit-identical to before (the key is used for the release noise only)."""
    force = make_force(vis, T, K, hidden, local)
    dfh = jax.grad(make_hidden_free_energy(T, K, hidden, local))
    hebb = make_hidden_hebb_sampled(T, K, hidden, local) if samples > 0 else None

    def grad(params, X, key, dt, releases=1):
        if "Wp" in params:                       # per-patch units (--patch): exact rule only (trace time)
            raise NotImplementedError("the CD-1 rule (make_cd1_hidden_grad) does not support per-patch Potts units "
                                      "(--patch)")
        hp = {k: params[k] for k in HIDDEN_KEYS if k in params}
        B = X.shape[0]
        drift = X - 0.5 * dt * force(params, X)
        if hebb is None:
            Xr = drift[None] + jnp.sqrt(T * dt) * jr.normal(key, (releases,) + X.shape, dtype=X.dtype)
            g_data = dfh(hp, X)
            g_rel = dfh(hp, Xr.reshape((releases * B, X.shape[1])))
        else:
            k_rel, k_data, k_read = jr.split(key, 3)
            Xr = drift[None] + jnp.sqrt(T * dt) * jr.normal(k_rel, (releases,) + X.shape, dtype=X.dtype)
            g_data = hebb(hp, X, k_data, releases * samples)
            g_rel = hebb(hp, Xr.reshape((releases * B, X.shape[1])), k_read, samples)
        scale = 2.0 / (T * T * dt)
        return {k: scale * (g_data[k] / B - g_rel[k] / (releases * B)) for k in hp}

    return grad


def force_matching_local_grad(params, X, T, vis, local=False):
    """The force-matching (score-matching) gradient of the hidden parameters of one Potts unit with spring couplings
    (W, b, Q) and of the per-pixel units (wl, bl), written as a local rule (DESIGN section 2.1). It must equal
    jax.grad of make_loss (checked in cd1_check.py --local-rule). Per clamped sample x:
      Potts state m: p_m (occupation), G_m = W_m + 2 Q_m x (gradient of its input field), s_m = sum_i Q_mi;
      Gbar = sum_k p_k G_k (the unit's mean force on the pixels), V = sum_k p_k |G_k - Gbar|^2, sbar = sum_k p_k s_k;
      psi = grad F (the clamp force, including the hidden units' mean force);
      c_m = |G_m - Gbar|^2 - V + 2 T (s_m - sbar) - (G_m - Gbar) . psi            (one scalar per state)
      dJ/db_m  = < p_m c_m > / T^3
      dJ/dW_mi = < p_m [ x_i c_m / T^3 + (2 (G_mi - Gbar_i) - psi_i) / T^2 ] >
      dJ/dQ_mi = < p_m [ x_i^2 c_m / T^3 + 2 x_i (2 (G_mi - Gbar_i) - psi_i) / T^2 + 2 / T ] >
    per-pixel unit i (a_i = bl_i + wl_i x_i, m_i = tanh(a_i / T), v_i = 1 - m_i^2, k3_i = -2 m_i v_i; DESIGN 2):
      dJ/dwl_i = < -m_i psi_i + 2 v_i wl_i + (x_i / T)(k3_i wl_i^2 - v_i wl_i psi_i) > / T^2
      dJ/dbl_i = < k3_i wl_i^2 - v_i wl_i psi_i > / T^3
    The non-local inputs of c_m are the projections G_m . psi and G_m . Gbar (the change of state m's input field
    when the pixels are nudged along psi or Gbar) and |G_m|^2; everything else is a product of quantities at one
    state, one pixel and their coupler. Returns a dict over W, b, Q (if present), wl, bl (if local)."""
    if "Wp" in params:                           # per-patch units (--patch): their force is not in psi here
        raise NotImplementedError("force_matching_local_grad does not support per-patch Potts units (--patch)")
    W, b = params["W"], params["b"]
    Q = params.get("Q", jnp.zeros_like(W))
    X2 = X * X
    A = X @ W.T + X2 @ Q.T + b                                  # (B, M)
    p = jax.nn.softmax(A / T, axis=1)
    Gbar = p @ W + 2.0 * X * (p @ Q)                            # (B, N)
    GG = jnp.sum(W * W, axis=1) + 4.0 * X @ (W * Q).T + 4.0 * X2 @ (Q * Q).T     # |G_m|^2 (B, M)
    GGbar = Gbar @ W.T + 2.0 * (Gbar * X) @ Q.T                 # G_m . Gbar (B, M)
    psi = vis.grad_lap(params["theta"], X)[0] - Gbar            # clamp force grad F (B, N)
    if local:
        Al = X * params["wl"] + params["bl"]
        ml = jnp.tanh(Al / T)
        psi = psi - ml * params["wl"]
    Gpsi = psi @ W.T + 2.0 * (psi * X) @ Q.T                    # G_m . psi (B, M)
    Gbar2 = jnp.sum(Gbar * Gbar, axis=1, keepdims=True)
    V = jnp.sum(p * GG, axis=1, keepdims=True) - Gbar2
    s = jnp.sum(Q, axis=1)
    sbar = (p @ s)[:, None]
    dist2 = GG - 2.0 * GGbar + Gbar2                            # |G_m - Gbar|^2
    c = dist2 - V + 2.0 * T * (s[None, :] - sbar) - (Gpsi - jnp.sum(Gbar * psi, axis=1, keepdims=True))
    pc = p * c                                                  # (B, M)
    B = X.shape[0]
    # sum_b p_bm (2 (G_bmi - Gbar_bi) - psi_bi), with G_bmi = W_mi + 2 Q_mi x_bi
    R = 2.0 * Gbar + psi                                        # (B, N)
    pW = 2.0 * jnp.sum(p, axis=0)[:, None] * W + 4.0 * Q * (p.T @ X) - p.T @ R   # (M, N)
    out = {"b": jnp.sum(pc, axis=0) / (B * T ** 3),
           "W": (pc.T @ X / T ** 3 + pW / T ** 2) / B}
    if "Q" in params:
        pQ = 2.0 * (2.0 * W * (p.T @ X) + 4.0 * Q * (p.T @ X2) - p.T @ (R * X))  # sum_b p 2 x (2 (G - Gbar) - psi)
        out["Q"] = (pc.T @ X2 / T ** 3 + pQ / T ** 2 + 2.0 * jnp.sum(p, axis=0)[:, None] / T) / B
    if local:
        wl = params["wl"]
        vl = 1.0 - ml * ml
        k3 = -2.0 * ml * vl
        e = k3 * wl * wl - vl * wl * psi                        # (B, N)
        out["wl"] = jnp.mean(-ml * psi + 2.0 * vl * wl + X * e / T, axis=0) / T ** 2
        out["bl"] = jnp.mean(e, axis=0) / T ** 3
    return out


def hidden_field_scale(params, X, T, K, hidden="ising", local=False):
    """Typical squared gradient of the hidden input fields, S = max(<sum_m p_m |G_m|^2>_batch, max_i wl_i^2, 1), with
    G_m = W_m + 2 Q_m x. A release of length dt moves the fields by about sqrt(T dt S); dt = kappa^2 T / S keeps that
    at kappa T (DESIGN section 4: the fields' own diffusion time)."""
    s = 1.0
    if K > 0:
        Q = params.get("Q", jnp.zeros_like(params["W"]))
        X2 = X * X
        A = X @ params["W"].T + X2 @ Q.T + params["b"]
        GG = (jnp.sum(params["W"] ** 2, axis=1) + 4.0 * X @ (params["W"] * Q).T + 4.0 * X2 @ (Q * Q).T)
        p = jax.nn.softmax(A / T, axis=1) if hidden == "potts" else jnp.ones_like(A)
        s = jnp.maximum(s, jnp.mean(jnp.sum(p * GG, axis=1)))
    if local:
        s = jnp.maximum(s, jnp.max(params["wl"] ** 2))
    return s


# ---------------------------------------------------------------------------------------------------------------
# Reference direction for the hidden unit: the Gaussian-mixture (Fisher) solution at each slice (DESIGN 1.1)
# ---------------------------------------------------------------------------------------------------------------
def fisher_reference(data, labels, taus, T):
    """W*(tau), b*(tau) for one Ising unit separating the classes in the Gaussian-mixture approximation.

    At forward time tau the class means are e^-tau mu_c and the pooled within-class covariance is
    S(tau) = e^-2tau S_w + T(1 - e^-2tau) I. With the visible precision A = T S(tau)^-1, one unit with
    W* = A (mu_1(tau) - mu_0(tau)) / 2 and b* = -W* . (mu_0(tau) + mu_1(tau)) / 2 puts the two tilted copies at
    the class means with equal weights; h = +1 is class 1.
    Returns W (len(taus), N), b (len(taus),).
    """
    X = np.asarray(data, dtype=np.float64)
    X0, X1 = X[labels == 0], X[labels == 1]
    mu0, mu1 = X0.mean(0), X1.mean(0)
    Sw = ((X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)) / (X0.shape[0] + X1.shape[0] - 2)
    Ws, bs = [], []
    for tau in np.asarray(taus, dtype=np.float64):
        e = math.exp(-tau)
        S = e * e * Sw + T * (1.0 - e * e) * np.eye(X.shape[1])
        w = 0.5 * T * np.linalg.solve(S, e * (mu1 - mu0))
        Ws.append(w)
        bs.append(-w @ (0.5 * e * (mu0 + mu1)))
    return np.stack(Ws), np.array(bs)


def local_reference(data, taus, T, pixel_threshold, min_count=20):
    """w*(tau), c*(tau) of one Ising unit per pixel: the two-mode (background / ink) Gaussian-mixture solution of
    each pixel's own marginal. Values above pixel_threshold (normalised units) are 'ink'. With mode means mu_bg, mu_ink,
    pooled within-mode variance s^2, ink share pi and e = exp(-tau), s^2(tau) = e^2 s^2 + T (1 - e^2):
      w = T e (mu_ink - mu_bg) / (2 s^2(tau)),   c = -w e (mu_ink + mu_bg) / 2 + (T/2) log(pi / (1 - pi)).
    Pixels with fewer than min_count ink (or background) values get w = c = 0. Returns w, c of shape (len(taus), N)."""
    X = np.asarray(data, dtype=np.float64)
    ink = X > pixel_threshold
    n_ink = ink.sum(0)
    n_bg = X.shape[0] - n_ink
    use = (n_ink >= min_count) & (n_bg >= min_count)
    mu_ink = np.where(use, np.sum(np.where(ink, X, 0.0), 0) / np.maximum(n_ink, 1), 0.0)
    mu_bg = np.where(use, np.sum(np.where(~ink, X, 0.0), 0) / np.maximum(n_bg, 1), 0.0)
    ss = np.sum(np.where(ink, (X - mu_ink) ** 2, (X - mu_bg) ** 2), 0)
    s2 = np.where(use, ss / np.maximum(X.shape[0] - 2, 1), 1.0)
    pi = np.clip(n_ink / X.shape[0], 1e-6, 1 - 1e-6)
    ws, cs = [], []
    for tau in np.asarray(taus, dtype=np.float64):
        e = math.exp(-tau)
        s2t = e * e * s2 + T * (1.0 - e * e)
        w = np.where(use, T * e * (mu_ink - mu_bg) / (2.0 * s2t), 0.0)
        c = np.where(use, -w * e * (mu_ink + mu_bg) / 2.0 + 0.5 * T * np.log(pi / (1.0 - pi)), 0.0)
        ws.append(w)
        cs.append(c)
    return np.stack(ws), np.stack(cs)


# ---------------------------------------------------------------------------------------------------------------
# Closed-form reference of the harmonic hidden modes (train_latent.py --harmonic R; 2026-11-12)
# ---------------------------------------------------------------------------------------------------------------
def harmonic_reference_strengths(lam, t, R, T=TEMP, s2=None):
    """The strengths w_i = |v_i|^2 (i = 1..R) of the closed-form harmonic modes at forward time t (harmonic_reference)
    and the isotropic variance iso_t of the visible springs they assume.

    lam: the eigenvalues of the data covariance C, descending (sample_latent.data_stats). The forward distribution at
    t has the covariance Sigma_t = e^2 C + sigma_t^2 I (e = exp(-t / s2), sigma_t^2 = T s2 (1 - e^2), s2 =
    SIGMA_FORWARD^2 by default), with eigenvalues e^2 lam_i + sigma_t^2. For any isotropic level iso_t no larger than
    the smallest of them,
        T Sigma_t^-1 = (T / iso_t) I - sum_i w_i e_i e_i^T,     w_i = T / iso_t - T / (e^2 lam_i + sigma_t^2) >= 0.
    The reference takes the largest level, iso_t = sigma_t^2 + e^2 lam_0, lam_0 = max(min_i lam_i, 0) (for the MNIST
    problems: the variance of the N(0, 0.01^2) pixel noise the paper adds, in normalised units), so that
        w_i = T e^2 (lam_i - lam_0) / (iso_t (e^2 lam_i + sigma_t^2)),
    the weakest axis needs no mode, and iso_t > 0 down to t = 0. With lam_0 = 0 this is the plain formula
    w_i^plain = T e^2 lam_i / (sigma_t^2 (e^2 lam_i + sigma_t^2)), which diverges at t = 0 (slice 99), where
    sigma_0 = 0; for t > 0, w_i = w_i^plain (1 - lam_0 / lam_i) / (1 + b) with b = e^2 lam_0 / sigma_t^2, a relative
    difference of (lam_0 / lam_i + b) / (1 + b), at most lam_0 / lam_i + b. Returns (w (R,), iso_t)."""
    s2 = SIGMA_FORWARD ** 2 if s2 is None else float(s2)
    lam = np.asarray(lam, dtype=np.float64)
    if not 1 <= int(R) <= lam.size:
        raise ValueError(f"R = {R}: expected 1 <= R <= {lam.size} harmonic modes")
    e2 = math.exp(-2.0 * float(t) / s2)
    sig2 = T * s2 * (1.0 - e2)
    lam0 = max(float(lam.min()), 0.0)
    iso = sig2 + e2 * lam0
    if not iso > 0.0:
        raise ValueError(f"harmonic reference at t = {t}: the isotropic level is {iso} (a singular data covariance at "
                         f"t = 0)")
    top = lam[:int(R)]
    return T * e2 * np.maximum(top - lam0, 0.0) / (iso * (e2 * top + sig2)), iso


def harmonic_reference(mu, lam, U, R, t, T=TEMP, s2=None):
    """The closed-form harmonic modes at forward time t: the Gaussian part of the forward distribution p_t.

    p_t has the mean m_t = e mu and the covariance Sigma_t = e^2 C + sigma_t^2 I (mu and C = U diag(lam) U^T of the
    run's training data, sample_latent.data_stats; e = exp(-t / s2)). Its Gaussian approximation N(m_t, Sigma_t) has
    the energy (p ~ exp(-F / T))
        F_G(x) = T/2 (x - m_t)^T Sigma_t^-1 (x - m_t)
               = T / (2 iso_t) |x - m_t|^2 - 1/2 sum_i (v_i . (x - m_t))^2,       v_i = sqrt(w_i) e_i
    (w_i, iso_t: harmonic_reference_strengths). The reference modes are its R strongest: the rows of V_ref are
    sqrt(w_i) e_i (i = 1..R) and c_ref = -V_ref m_t, so that v_i . x + c_i = v_i . (x - m_t). The isotropic and linear
    parts, T / (2 iso_t) |x - m_t|^2, belong to the visible energy (per-pixel springs k_lin = T / iso_t, bias
    -T m_t / iso_t), which the model already has. With R = N the decomposition is exact. With R < N the precision is
    exact on the span of e_1..e_R, T / (e^2 lam_i + sigma_t^2), and T / iso_t on the rest (where p_t's is
    T / (e^2 lam_i + sigma_t^2) <= T / iso_t). The reference energy is always confining: its precision is positive
    along every axis. mu (N,), lam (N,) descending, U (N, N) with the eigenvectors of C as columns.
    Within-cluster reference (train_latent.py --harmonic-ref within): with lam, U of the Potts prototypes' shared
    component covariance S_w instead of C (prototype_reference(..., return_stats=True), harmonic_within_spectrum) the
    same formulas give the modes of S(t) = e^2 S_w + sigma_t^2 I, and then the prototype reference's Potts unit
    (W*(t), b*(t)) plus these modes (R = N) plus the isotropic visible part T / (2 iso_t) |x - m_t|^2 is, up to a
    constant, exactly the energy -T log sum_m pi_m N(x; e c_m, S(t)) of the prototypes' Gaussian mixture, with mu the
    mixture's mean sum_m pi_m c_m (which equals the data mean up to the k-means stopping tolerance; any difference is
    a linear term, which the visible bias absorbs): the modes carry only the covariance the prototypes do not, instead
    of counting the between-cluster part twice (the full reference's modes carry all of C, and the prototypes carry
    their spread).
    Returns (V_ref (R, N), c_ref (R,), iso_t), float64."""
    s2 = SIGMA_FORWARD ** 2 if s2 is None else float(s2)
    w, iso = harmonic_reference_strengths(lam, t, R, T, s2)
    V = np.ascontiguousarray((np.asarray(U, dtype=np.float64)[:, :int(R)] * np.sqrt(w)).T)
    return V, -(V @ (math.exp(-float(t) / s2) * np.asarray(mu, dtype=np.float64))), iso


def harmonic_lr_scales(mu, lam, U, R, taus, T=TEMP, s2=None):
    """Per-slice magnitudes of the harmonic reference, which set the modes' learning rates (train_latent.py --harmonic,
    as the hidden units' sW / sb set theirs): the root-mean-square entry of V_ref(t_k), sqrt(sum_i w_i / (R N)), and
    that of c_ref(t_k), floored at T (as sb is). Returns (sV, sc), each of shape (len(taus),). train_latent.py
    --harmonic-lr-tmin t0 passes max(t_k, t0) for t_k, so that below t0 the scales stay at their value at t0."""
    s2 = SIGMA_FORWARD ** 2 if s2 is None else float(s2)
    mu = np.asarray(mu, dtype=np.float64)
    proj = np.asarray(U, dtype=np.float64)[:, :int(R)].T @ mu            # e_i . mu
    sV, sc = [], []
    for t in np.asarray(taus, dtype=np.float64):
        w, _ = harmonic_reference_strengths(lam, t, R, T, s2)
        sV.append(math.sqrt(float(np.sum(w)) / (int(R) * mu.size)))
        sc.append(max(math.sqrt(float(np.mean(w * (math.exp(-t / s2) * proj) ** 2))), T))
    return np.asarray(sV), np.asarray(sc)


def harmonic_within_spectrum(Sw):
    """The eigen-decomposition of the Potts prototypes' within-cluster covariance S_w (prototype_reference(...,
    return_stats=True)) in the form harmonic_reference takes for the within-cluster reference (train_latent.py
    --harmonic-ref within): (lam (N,) descending, U (N, N) with the eigenvectors as columns), float64. S_w is
    symmetrised first (its two triangles agree to rounding)."""
    S = np.asarray(Sw, dtype=np.float64)
    lam, U = np.linalg.eigh(0.5 * (S + S.T))
    return lam[::-1].copy(), np.ascontiguousarray(U[:, ::-1])


def visible_hessian_at_zero(theta, conn, n):
    """Hessian of the visible energy at x = 0 (n x n, numpy): diag(k_lin) + 2 (diag(C 1) - C), C the symmetric matrix of
    the linear pair springs c_lin (the k_duff, k_6, c_om and c_duff terms are of order >= 3 in x: no curvature at 0)."""
    th = np.asarray(theta, dtype=np.float64)
    conn = np.asarray(conn)
    C = np.zeros((n, n))
    np.add.at(C, (conn[:, 0], conn[:, 1]), th[clin_slice(n, conn.shape[0])])
    C = C + C.T
    return np.diag(th[:n] + 2.0 * C.sum(axis=1)) - 2.0 * C


def harmonic_unit_stats(V, c, V_ref, c_ref, U_R, iso, T=TEMP, theta=None, conn=None):
    """Per-slice diagnostics of the harmonic modes (train_latent.py --harmonic) against the slice's closed-form
    reference (harmonic_reference: V_ref, c_ref, iso; U_R (N, R): its principal axes). numpy, float64:
      harm_V_norm, harm_Vref_norm, harm_c_norm, harm_cref_norm: Frobenius / Euclidean norms;
      harm_cos_V_Vref: cosine of the flattened V and V_ref (row by row: meaningful while the modes keep the reference's
        order, which they start from);
      harm_cos_M_Mref: cosine of the anti-spring matrices V^T V and V_ref^T V_ref, on which F_h's curvature depends
        (invariant under rotations of the modes among themselves);
      harm_in_ref_subspace: the share of |V|^2 inside the span of the reference's R axes;
      harm_strength_top5, harm_strength_ref_top5: the largest eigenvalues mu_j of V^T V (the anti-spring constants
        along the modes' principal directions) and the reference's w_i = |v_i|^2;
      harm_added_var, harm_added_var_ref: the variance the modes add along their principal directions on top of the
        reference's isotropic springs T / iso, sum_j [T / (T / iso - mu_j) - iso] over the modes with mu_j < T / iso
        (for the reference e^2 sum_i (lam_i - lam_0): the excess variance of p_t along its R axes);
        harm_added_var_top, harm_added_var_ref_top: the strongest mode alone (None beyond T / iso);
        harm_n_beyond_iso: modes with mu_j >= T / iso (they cancel the reference's isotropic springs at second order;
        T / iso is the reference's, 1 at t = 5 but 24.5 at t = 0 for mnist28_d0123, far above a trained run's springs,
        so at low t this count cannot flag modes that outgrow the run's own springs: harm_n_beyond_vis does);
      harm_iso_spring: T / iso;
      with theta and conn (the visible parameters and pairs): harm_vis_curv_min, the smallest eigenvalue of the visible
        Hessian H at x = 0 (visible_hessian_at_zero), harm_curv_min, that of H - V^T V (the springs plus the modes at
        x = 0; negative: along some direction the modes outgrow the springs, and only the higher-order visible terms
        and the hidden units confine there), and harm_n_beyond_vis, the modes whose strength mu_j reaches the springs'
        curvature u_j^T H u_j along their own principal direction u_j. H alone is already indefinite at low t in
        trained runs (the plain seed-1 study run: smallest eigenvalue -0.09 at slice 70, -0.24 at slice 99; the
        quartic, Q and hidden-unit terms confine there), so judge harm_curv_min against harm_vis_curv_min (the modes'
        own effect is the difference) and against plain runs, not against 0.
    Returns a dict of floats (lists for the top-5 values)."""
    V, V_ref = np.asarray(V, dtype=np.float64), np.asarray(V_ref, dtype=np.float64)
    nV, nR = float(np.linalg.norm(V)), float(np.linalg.norm(V_ref))
    G, Gr, M = V @ V.T, V_ref @ V_ref.T, V @ V_ref.T
    mu = np.sort(np.linalg.eigvalsh(G))[::-1]
    w = np.sort(np.diag(Gr))[::-1]               # the reference's rows are orthogonal: their |v_i|^2 are the eigenvalues
    k_iso = T / iso

    def added(m):
        ok = m < k_iso
        return T / (k_iso - m[ok]) - iso, ok

    a, ok = added(mu)
    ar, okr = added(w)
    out = {"harm_V_norm": nV, "harm_Vref_norm": nR, "harm_c_norm": float(np.linalg.norm(np.asarray(c, np.float64))),
           "harm_cref_norm": float(np.linalg.norm(np.asarray(c_ref, np.float64))),
           "harm_cos_V_Vref": float(np.sum(V * V_ref) / (nV * nR + 1e-300)),
           "harm_cos_M_Mref": float(np.sum(M * M) / (np.linalg.norm(G) * np.linalg.norm(Gr) + 1e-300)),
           "harm_in_ref_subspace": float(np.sum((V @ np.asarray(U_R, dtype=np.float64)) ** 2) / (nV * nV + 1e-300)),
           "harm_strength_top5": [float(v) for v in mu[:5]], "harm_strength_ref_top5": [float(v) for v in w[:5]],
           "harm_iso_spring": float(k_iso),
           "harm_added_var": float(np.sum(a)), "harm_added_var_ref": float(np.sum(ar)),
           "harm_added_var_top": float(a[0]) if ok[0] else None, "harm_added_var_ref_top": float(ar[0]) if okr[0] else None,
           "harm_n_beyond_iso": int(np.sum(~ok))}
    if theta is not None:
        H = visible_hessian_at_zero(theta, conn, V.shape[1])
        out["harm_vis_curv_min"] = float(np.linalg.eigvalsh(H)[0])
        out["harm_curv_min"] = float(np.linalg.eigvalsh(H - V.T @ V)[0])
        mu_j, Q_j = np.linalg.eigh(G)            # principal directions of the modes, u_j = V^T q_j / sqrt(mu_j)
        pos = mu_j > 1e-12 * max(float(np.max(mu_j, initial=0.0)), 1e-300)
        U_j = (V.T @ Q_j[:, pos]) / np.sqrt(mu_j[pos])
        out["harm_n_beyond_vis"] = int(np.sum(mu_j[pos] >= np.einsum("ij,ik,kj->j", U_j, H, U_j)))
    return out


def kmeans(X, k, seed, iters=50):
    """Plain Lloyd's k-means with k-means++ seeding (numpy). Returns (centres (k, d), assignment (n,))."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(X.shape[0])]]
    d2 = np.sum((X - C[0]) ** 2, axis=1)
    for _ in range(1, k):
        C.append(X[rng.choice(X.shape[0], p=d2 / d2.sum())])
        d2 = np.minimum(d2, np.sum((X - C[-1]) ** 2, axis=1))
    C = np.array(C)
    for _ in range(iters):
        assign = np.argmin(np.sum(X**2, 1)[:, None] - 2 * X @ C.T + np.sum(C**2, 1)[None, :], axis=1)
        newC = np.array([X[assign == j].mean(0) if np.any(assign == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, assign


@jax.jit
def _kmeanspp_device(X, i0, us):
    """k-means++ seeding of kmeans_jax on the device: the indices of the k centres, i0 first, then one per uniform
    u of `us`, drawn as Generator.choice(n, p=d2 / d2.sum()) draws it: the number of entries <= u of the normalised
    cumulative distribution (searchsorted 'right')."""
    d2 = jnp.sum((X - X[i0]) ** 2, axis=1)

    def draw(d2, u):
        cdf = jnp.cumsum(d2 / jnp.sum(d2))
        i = jnp.sum(cdf / cdf[-1] <= u)
        return jnp.minimum(d2, jnp.sum((X - X[i]) ** 2, axis=1)), i

    _, idx = jax.lax.scan(draw, d2, us)
    return jnp.concatenate([jnp.reshape(i0, (1,)).astype(idx.dtype), idx])


@partial(jax.jit, static_argnums=2)
def _lloyd_device(X, C, iters):
    """Lloyd's iterations of kmeans() on the device, with its stopping rule (np.allclose of the new and the old
    centres; the old centres and their assignment are returned then) and its empty-cluster rule (the centre stays).
    Cluster sums are one-hot products (deterministic), not scatter-adds."""
    k = C.shape[0]
    x2 = jnp.sum(X * X, axis=1)
    cols = jnp.arange(k)

    def body(state):
        C, _, it, _ = state
        a = jnp.argmin(x2[:, None] - 2.0 * (X @ C.T) + jnp.sum(C * C, axis=1)[None, :], axis=1)
        onehot = (a[:, None] == cols[None, :]).astype(X.dtype)
        counts = jnp.sum(onehot, axis=0)
        newC = jnp.where(counts[:, None] > 0, (onehot.T @ X) / jnp.maximum(counts, 1.0)[:, None], C)
        same = jnp.all(jnp.abs(newC - C) <= 1e-8 + 1e-5 * jnp.abs(C))      # np.allclose(newC, C)
        return jnp.where(same, C, newC), a, it + 1, same

    a0 = jnp.argmin(jnp.zeros((X.shape[0], 1), X.dtype), axis=1)
    C, a, _, _ = jax.lax.while_loop(lambda s: (s[2] < iters) & ~s[3], body, (C, a0, jnp.int32(0), jnp.bool_(False)))
    return C, a


def kmeans_jax(X, k, seed, iters=50):
    """kmeans() on the default JAX device (--ref-kmeans gpu): the same k-means++ seeding and Lloyd iterations, the
    same stopping rule and the same return values, in float64. The random draws are numpy's, from the same
    Generator(seed) stream as kmeans() (one integer for the first centre, then one uniform per further centre, which
    is what Generator.choice(n, p=...) draws), so the seeding picks the same centres unless rounding moves a
    cumulative-probability boundary across a draw. Distances, the cumulative distribution, the assignments and the
    cluster means are computed on the device (summation orders differ from numpy's), so the result equals kmeans()
    up to rounding: an assignment can flip where two centres are equally near to rounding, and the iterations can
    then end in a different local optimum. Deterministic. Returns numpy (centres (k, d), assignment (n,))."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    i0 = int(rng.integers(X.shape[0]))
    us = rng.random(k - 1)
    Xd = jnp.asarray(X)
    idx = _kmeanspp_device(Xd, jnp.asarray(i0), jnp.asarray(us))
    C, assign = _lloyd_device(Xd, Xd[idx], int(iters))
    return np.asarray(C), np.asarray(assign)


def prototype_reference(data, labels, taus, T, M, seed=0, pooled=False, kmeans_fn=None, return_stats=False):
    """W*(tau), b*(tau) of one Potts unit with M states: the Gaussian-mixture solution for M k-means prototypes
    (M/2 per class; pooled=True: M prototypes of all images together, without labels) with their pooled
    within-cluster covariance S_w and weights pi_m:
      S(tau) = e^-2tau S_w + T(1 - e^-2tau) I,  mu_m(tau) = e^-tau c_m,
      W_m = T S(tau)^-1 mu_m(tau),  b_m = T log pi_m - W_m . mu_m(tau) / 2.
    The common part of the W_m (their weighted mean) is moved out of the hidden unit, since a uniform linear tilt
    belongs to the visible bias. Returns W (len(taus), M, N), b (len(taus), M), state class (M,); with pooled=True
    the state class is the majority label of the cluster and is used for diagnostics only.
    kmeans_fn: the k-means, kmeans (numpy; default) or kmeans_jax (--ref-kmeans gpu).
    return_stats=True (train_latent.py --harmonic-ref within, 2026-11-12) also returns a fourth item, the dict
    {'Sw': S_w (N, N), 'pi': pi (M,), 'centres': c_m (M, N), 'counts': cluster sizes (M,)}: S_w is the shared
    covariance of the mixture's components, the part of the data covariance the prototypes do not carry (the same
    array the W_m above are computed from). The default (False) returns the three items above, computed as before."""
    km = kmeans if kmeans_fn is None else kmeans_fn
    classes = np.unique(np.asarray(labels))
    n_cls = classes.size
    if not pooled and M % n_cls:
        raise ValueError(f"M = {M} must be a multiple of the {n_cls} classes (M/{n_cls} prototypes per class)")
    X = np.asarray(data, dtype=np.float64)
    if pooled:
        C, assign = km(X, M, seed)
        R = X - C[assign]
        counts = np.bincount(assign, minlength=M)
        lab = np.asarray(labels)
        if n_cls == 2:                       # as before: majority label, ties to class 1
            cls = [int(lab[assign == j].mean() >= 0.5) if counts[j] else 0 for j in range(M)]
        else:
            cls = [int(np.bincount(lab[assign == j], minlength=n_cls).argmax()) if counts[j] else 0 for j in range(M)]
    else:
        cents, cls, resid, cnts = [], [], [], []
        for c in classes.tolist():
            Xc = X[labels == c]
            Cc, assign = km(Xc, M // n_cls, seed + c)
            cents.append(Cc)
            cls += [c] * (M // n_cls)
            resid.append(Xc - Cc[assign])
            cnts.append(np.bincount(assign, minlength=M // n_cls))
        C = np.concatenate(cents)
        R = np.concatenate(resid)
        counts = np.concatenate(cnts)
    Sw = R.T @ R / (R.shape[0] - M)
    pi = np.maximum(counts, 1) / X.shape[0]   # an empty cluster gets weight 1/n (log 0 would give NaN)
    Ws, bs = [], []
    for tau in np.asarray(taus, dtype=np.float64):
        e = math.exp(-tau)
        S = e * e * Sw + T * (1.0 - e * e) * np.eye(X.shape[1])
        mu = e * C
        W = T * np.linalg.solve(S, mu.T).T
        b = T * np.log(pi) - 0.5 * np.sum(W * mu, axis=1)
        W = W - pi @ W                       # remove the weighted-mean tilt (a visible bias)
        Ws.append(W)
        bs.append(b - (b @ pi))
    if return_stats:                         # --harmonic-ref within: the components' shared covariance as well
        return np.stack(Ws), np.stack(bs), np.array(cls), {"Sw": Sw, "pi": pi, "centres": C, "counts": counts}
    return np.stack(Ws), np.stack(bs), np.array(cls)


# ---------------------------------------------------------------------------------------------------------------
# Reference, learning-rate scales and diagnostics of the per-patch Potts units (train_latent.py --patch)
# ---------------------------------------------------------------------------------------------------------------
def patch_clusters(data, geom, S, seed0, kmeans_fn=None):
    """The k-means half of the per-patch prototype reference: for every patch p of geom (PatchGeometry), the pooled
    branch of prototype_reference on the patch vectors Xp = data[:, idx_p] (no labels), in its order of operations:
      C, assign = km(Xp, S, seed0 + p);  R = Xp - C[assign];  counts = bincount(assign, S);  Sw = R^T R / (n - S);
      pi = max(counts, 1) / n,
    km = kmeans (numpy; default) or kmeans_fn (kmeans_jax with --ref-kmeans gpu). Returns dict(C (P, S, d), Sw (P, d,
    d), pi (P, S), counts (P, S), explained_variance (P,) (1 - |R|^2 / |Xp - mean|^2), seeds (P,), seconds)."""
    import time as _time
    km = kmeans if kmeans_fn is None else kmeans_fn
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[0]
    t0 = _time.time()
    Cs, Sws, pis, cnts, evs = [], [], [], [], []
    for p in range(geom.P):
        Xp = X[:, geom.idx[p]]
        C, assign = km(Xp, S, seed0 + p)
        R = Xp - C[assign]
        counts = np.bincount(assign, minlength=S)
        Sws.append(R.T @ R / (R.shape[0] - S))
        pis.append(np.maximum(counts, 1) / n)
        Cs.append(C)
        cnts.append(counts)
        evs.append(1.0 - float(np.sum(R * R)) / max(float(np.sum((Xp - Xp.mean(axis=0)) ** 2)), 1e-300))
    return {"C": np.stack(Cs), "Sw": np.stack(Sws), "pi": np.stack(pis), "counts": np.stack(cnts),
            "explained_variance": np.asarray(evs), "seeds": seed0 + np.arange(geom.P), "seconds": _time.time() - t0}


def _patch_ref_one(C, Sw, pi, tau, T, scale):
    """One patch's reference at forward time tau: prototype_reference's formula and order of operations (scale 1),
    or the tempered expert of scale lam (patch_reference)."""
    e = math.exp(-tau)
    Sm = e * e * Sw + T * (1.0 - e * e) * np.eye(C.shape[1])
    mu = e * C
    Ws = T * np.linalg.solve(Sm, mu.T).T
    if scale == 1.0:
        b = T * np.log(pi) - 0.5 * np.sum(Ws * mu, axis=1)
        W = Ws - pi @ Ws
    else:
        b = T * np.log(pi) - 0.5 * scale * np.sum(Ws * mu, axis=1)
        W = scale * (Ws - pi @ Ws)
    return W, b - (b @ pi)


def patch_reference(clusters, taus, T, scale=1.0):
    """The per-patch Gaussian-mixture reference (prototype_reference's pooled branch per patch) from patch_clusters, at
    each forward time tau of taus. Per patch, with e = exp(-tau), S(tau) = e^2 S_w + T (1 - e^2) I_d, mu = e C:
      W* = T S(tau)^-1 mu;  scale 1: b = T log pi - W*.mu / 2, W = W* - pi W*, b <- b - b . pi (prototype_reference's
      operations in its order, so equal to it bit for bit);
      scale lam != 1 (train_latent.py --patch-init-scale): b = T log pi - lam W*.mu / 2, W = lam (W* - pi W*),
      b <- b - b . pi, a tempered expert (lam times the clusters' log-likelihood, the prior kept).
    Returns W (len(taus), P, S, d), b (len(taus), P, S), float64."""
    C, Sw, pi = clusters["C"], clusters["Sw"], clusters["pi"]
    Ws, bs = [], []
    for tau in np.asarray(taus, dtype=np.float64):
        Wt, bt = zip(*[_patch_ref_one(C[p], Sw[p], pi[p], float(tau), T, float(scale)) for p in range(C.shape[0])])
        Ws.append(np.stack(Wt))
        bs.append(np.stack(bt))
    return np.stack(Ws), np.stack(bs)


def patch_lr_scales(clusters, taus, T):
    """Per-slice, per-unit magnitudes of the patch reference at scale 1, which set the units' learning rates
    (train_latent.py --patch): sWp[k, p] = the root-mean-square entry of patch p's W*(t_k) over (S, d), sbp[k, p] =
    max(rms of its b*(t_k), T). Computed time by time (no (100, P, S, d) array is kept). Returns (sWp, sbp), each
    (len(taus), P)."""
    sW, sb = [], []
    for tau in np.asarray(taus, dtype=np.float64):
        W, b = patch_reference(clusters, [tau], T)
        sW.append(np.sqrt(np.mean(W[0] ** 2, axis=(1, 2))))
        sb.append(np.maximum(np.sqrt(np.mean(b[0] ** 2, axis=1)), T))
    return np.stack(sW), np.stack(sb)


def patch_unit_stats(Wp, bp, Qp, X, T, W_ref, geom):
    """Per-slice diagnostics of the per-patch Potts units on the batch X (numpy, one product per patch). W_ref: the
    scale-1 reference (P, S, d) at the slice's t, or None (no reference entries). Returns (dict, arrays):
      patch_W_norm, patch_Wref_norm, patch_cos_W_Wref (flattened), patch_Q_norm (with Qp), patch_b_rms;
      patch_eff_states_median / _min / _central_median: exp of the entropy of each patch's batch-mean occupation
        (over all patches, and over the central ones);
      patch_states_below_1e-6: the states with batch-mean occupation below 1e-6, summed over the patches;
      patch_mean_max_p: the mean over patches and samples of the largest posterior;
    arrays: {'patch_eff': (P,) float32}."""
    Wp, bp = np.asarray(Wp, dtype=np.float64), np.asarray(bp, dtype=np.float64)
    Qp = None if Qp is None else np.asarray(Qp, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    eff, dead, maxp = np.zeros(geom.P), 0, 0.0
    for p in range(geom.P):
        Xp = X[:, geom.idx[p]]
        A = Xp @ Wp[p].T + bp[p]
        if Qp is not None:
            A = A + (Xp * Xp) @ Qp[p].T
        Z = A / T
        Pp = np.exp(Z - Z.max(axis=1, keepdims=True))
        Pp /= Pp.sum(axis=1, keepdims=True)
        pbar = Pp.mean(axis=0)
        eff[p] = float(np.exp(-np.sum(pbar * np.log(pbar + 1e-300))))
        dead += int(np.sum(pbar < 1e-6))
        maxp += float(np.mean(Pp.max(axis=1)))
    nW = float(np.linalg.norm(Wp))
    out = {"patch_W_norm": nW}
    if W_ref is not None:
        W_ref = np.asarray(W_ref, dtype=np.float64)
        nR = float(np.linalg.norm(W_ref))
        out.update(patch_Wref_norm=nR, patch_cos_W_Wref=float(np.sum(Wp * W_ref) / (nW * nR + 1e-300)))
    if Qp is not None:
        out["patch_Q_norm"] = float(np.linalg.norm(Qp))
    cen = eff[geom.central]
    out.update(patch_b_rms=float(np.sqrt(np.mean(bp * bp))), patch_eff_states_median=float(np.median(eff)),
               patch_eff_states_min=float(np.min(eff)),
               patch_eff_states_central_median=float(np.median(cen)) if cen.size else None,
               **{"patch_states_below_1e-6": int(dead)}, patch_mean_max_p=maxp / geom.P)
    return out, {"patch_eff": eff.astype(np.float32)}


# ---------------------------------------------------------------------------------------------------------------
# K Ising hidden units: a split tree of the data, its Gaussian-mixture reference, the local rule and diagnostics
# ---------------------------------------------------------------------------------------------------------------
SPLIT_CLASS, SPLIT_KMEANS, SPLIT_MEDIAN = 0, 1, 2      # how a node of split_tree was split (its "kind")


def split_tree(X, labels, K, seed=0, pooled=False, n_min=20, iters=50):
    """A binary tree over the rows of X with K internal nodes (node j = Ising unit j, in creation order) and K + 1
    leaves; split_reference makes each node the Gaussian-mixture unit that separates its two children.

    perclass (pooled=False): first the class tree. A node holding c >= 2 classes is split by the bipartition of its
    class set with the largest Ward gain n_L n_R / (n_L + n_R) |mu_L - mu_R|^2 among all 2^(c-1) - 1 (exhaustive);
    its right child holds the node's highest class (2 classes: the right child is class 1). Multi-class leaves are
    split largest gain first until every class is a leaf: C - 1 nodes (needs K >= C - 1).
    pooled=True: no class tree; the root is the first 2-means split of all images. Labels are read only for
    unit_class / unit_purity (diagnostics; labels may be None).
    Then greedy bisecting 2-means. Every leaf with >= 2 n_min images gets one candidate split,
    kmeans(X_leaf, 2, seed + 1000 + region) (region ids: root 0; node j's children are 2j + 1 (left) and 2j + 2
    (right)). Balance guard: if the candidate's smaller side has fewer than max(n_min, 0.1 n_leaf) images, the leaf is
    cut at the median of its projection on c_2 - c_1 instead. The right child is the larger side (ties: the side of
    c_2, i.e. the upper half). The leaf with the largest Ward gain is split until there are K nodes; ValueError if no
    leaf can be split before that. Order and sign of the units are fixed by construction, so the reference W_j(t)
    is smooth in t and needs no gauge alignment.

    Returns a dict of numpy arrays (np.savez-able): parent (K,) (-1: root), depth (K,), child_left / child_right (K,)
    (a node id >= 0, or -(leaf + 1) for a leaf), n_left / n_right (K,) and mu_left / mu_right (K, N) (images and means
    of the two children), leaf_of (n,) (leaves numbered by region id), leaf_means (K + 1, N), H (K + 1, K) in
    {-1, 0, +1} (+1 / -1: the leaf lies in the right / left subtree of node j; 0: not below node j), unit_class (K, 2)
    (majority class of the left and right child; -1 without labels), unit_purity (K,) (share of the node's images in
    their child's majority class; nan without labels), gain (K,) (Ward gain), kind (K,) (SPLIT_CLASS, SPLIT_KMEANS or
    SPLIT_MEDIAN), and the settings seed, pooled, n_min (0-d)."""
    X = np.asarray(X, dtype=np.float64)
    n, N = X.shape
    K = int(K)
    if K < 1:
        raise ValueError(f"split_tree needs K >= 1 nodes, not {K}")
    lab = None if labels is None else np.asarray(labels)
    classes = None
    if not pooled:
        if lab is None:
            raise ValueError("split_tree: the class tree (pooled=False) needs labels")
        classes = tuple(int(c) for c in np.unique(lab))
        if K < len(classes) - 1:
            raise ValueError(f"split_tree: the class tree of {len(classes)} classes needs K >= {len(classes) - 1} "
                             f"nodes, not {K}")
    reg_idx, reg_cls, reg_up = [], [], []    # per region: rows (ascending), class set (class tree), (node, side)
    cand = {}                                 # region -> (gain, left rows, right rows, kind, left classes, right classes)

    def new_region(idx, cls, up):
        reg_idx.append(idx)
        reg_cls.append(cls)
        reg_up.append(up)
        return len(reg_idx) - 1

    def class_candidate(r):
        cls, idx = reg_cls[r], reg_idx[r]
        if cls is None or len(cls) < 2:
            return None
        li = lab[idx]
        cnt = {c: int(np.sum(li == c)) for c in cls}
        tot = {c: X[idx[li == c]].sum(axis=0) for c in cls}
        best = None
        for mask in range(2 ** (len(cls) - 1) - 1):          # class cls[0] in group A; A != all classes
            grp = (cls[0],) + tuple(c for i, c in enumerate(cls[1:]) if mask >> i & 1)
            oth = tuple(c for c in cls if c not in grp)
            na, nb = sum(cnt[c] for c in grp), sum(cnt[c] for c in oth)
            d = sum(tot[c] for c in grp) / na - sum(tot[c] for c in oth) / nb
            g = na * nb / (na + nb) * float(d @ d)
            if best is None or g > best[0]:
                best = (g, grp, oth)
        g, grp, oth = best
        right, left = (grp, oth) if cls[-1] in grp else (oth, grp)     # the right child holds the highest class
        in_r = np.isin(li, right)
        return g, idx[~in_r], idx[in_r], SPLIT_CLASS, tuple(sorted(left)), tuple(sorted(right))

    def kmeans_candidate(r):
        idx = reg_idx[r]
        nl = idx.size
        if nl < 2 * n_min:
            return None
        Xl = X[idx]
        C, assign = kmeans(Xl, 2, seed + 1000 + r, iters)
        up = assign == 1
        kind = SPLIT_KMEANS
        n1 = int(up.sum())
        if min(n1, nl - n1) < max(n_min, 0.1 * nl):         # balance guard: median cut along c_2 - c_1
            order = np.argsort(Xl @ (C[1] - C[0]), kind="stable")
            up = np.zeros(nl, dtype=bool)
            up[order[nl // 2:]] = True                       # the upper half (towards c_2)
            kind = SPLIT_MEDIAN
            n1 = int(up.sum())
        right = up if n1 >= nl - n1 else ~up                 # the larger side (ties: c_2's side)
        il, ir = idx[~right], idx[right]
        d = X[ir].mean(axis=0) - X[il].mean(axis=0)
        return il.size * ir.size / nl * float(d @ d), il, ir, kind, None, None

    new_region(np.arange(n), classes, (-1, -1))
    nodes, split_regions = [], set()           # nodes: (region, left region, right region, kind)
    class_phase = not pooled
    while len(nodes) < K:
        leaves = [r for r in range(len(reg_idx)) if r not in split_regions]
        if class_phase:
            pool = [r for r in leaves if len(reg_cls[r]) > 1]
            if not pool:
                class_phase = False
                continue
            get = class_candidate
        else:
            pool, get = leaves, kmeans_candidate
        for r in pool:
            if r not in cand:
                cand[r] = get(r)
        options = [r for r in pool if cand[r] is not None]
        if not options:
            raise ValueError(f"split_tree: no leaf can be split after {len(nodes)} of {K} nodes (every leaf has fewer "
                             f"than 2 n_min = {2 * n_min} images)")
        r = max(options, key=lambda q: cand[q][0])           # largest Ward gain (ties: the lowest region id)
        _, il, ir, kind, cl, cr = cand[r]
        j = len(nodes)
        rl = new_region(il, cl, (j, 0))
        rr = new_region(ir, cr, (j, 1))
        nodes.append((r, rl, rr, kind))
        split_regions.add(r)

    leaf_regions = [r for r in range(len(reg_idx)) if r not in split_regions]      # ascending region id
    leaf_id = {r: i for i, r in enumerate(leaf_regions)}
    node_of = {r: j for j, (r, _, _, _) in enumerate(nodes)}
    parent, depth = np.zeros(K, dtype=np.int64), np.zeros(K, dtype=np.int64)
    child_left, child_right = np.zeros(K, dtype=np.int64), np.zeros(K, dtype=np.int64)
    n_left, n_right = np.zeros(K, dtype=np.int64), np.zeros(K, dtype=np.int64)
    mu_left, mu_right = np.zeros((K, N)), np.zeros((K, N))
    gain, kinds = np.zeros(K), np.zeros(K, dtype=np.int64)
    unit_class, unit_purity = -np.ones((K, 2), dtype=np.int64), np.full(K, np.nan)
    for j, (r, rl, rr, kind) in enumerate(nodes):
        parent[j] = reg_up[r][0]
        depth[j] = 0 if parent[j] < 0 else depth[parent[j]] + 1
        child_left[j] = node_of[rl] if rl in node_of else -(leaf_id[rl] + 1)
        child_right[j] = node_of[rr] if rr in node_of else -(leaf_id[rr] + 1)
        il, ir = reg_idx[rl], reg_idx[rr]
        n_left[j], n_right[j] = il.size, ir.size
        mu_left[j], mu_right[j] = X[il].mean(axis=0), X[ir].mean(axis=0)
        d = mu_right[j] - mu_left[j]
        gain[j] = il.size * ir.size / (il.size + ir.size) * float(d @ d)
        kinds[j] = kind
        if lab is not None:
            hit = 0
            for side, rows in ((0, il), (1, ir)):
                vals, cnts = np.unique(lab[rows], return_counts=True)
                unit_class[j, side] = int(vals[np.argmax(cnts)])
                hit += int(cnts.max())
            unit_purity[j] = hit / (il.size + ir.size)
    leaf_of = np.zeros(n, dtype=np.int64)
    H = np.zeros((K + 1, K), dtype=np.int64)
    for i, r in enumerate(leaf_regions):
        leaf_of[reg_idx[r]] = i
        q = r
        while reg_up[q][0] >= 0:                              # walk up to the root
            j, side = reg_up[q]
            H[i, j] = 1 if side == 1 else -1
            q = nodes[j][0]
    leaf_means = np.stack([X[reg_idx[r]].mean(axis=0) for r in leaf_regions])
    return {"parent": parent, "depth": depth, "child_left": child_left, "child_right": child_right,
            "n_left": n_left, "n_right": n_right, "mu_left": mu_left, "mu_right": mu_right, "leaf_of": leaf_of,
            "leaf_means": leaf_means, "H": H, "unit_class": unit_class, "unit_purity": unit_purity, "gain": gain,
            "kind": kinds, "seed": np.int64(seed), "pooled": np.bool_(pooled), "n_min": np.int64(n_min)}


def tree_n_min(n, K, n_min=20):
    """Smallest child size of split_tree for K nodes on n images: n_min = 20, lowered for trees so large that 20 would
    stop the splitting (K + 1 leaves of >= 2 n_min images to split need about 2 n_min (K + 1) images); never below 3
    (5 before 2026-11-09 18:30; no run reached the floor). Equals 20 for every run before 2026-11-09 17:20."""
    return int(min(n_min, max(3, n // (2 * (K + 1)))))


def split_reference(data, labels, taus, T, K, seed=0, pooled=False, n_min=20):
    """W*(tau), b*(tau) of K Ising units, one per node of split_tree(data, labels, K, seed, pooled): unit j is the
    Gaussian-mixture unit that separates the two children of node j (h_j = +1: the right child),
      S(tau) = e^-2tau S_w + T(1 - e^-2tau) I     (S_w: pooled within-leaf covariance of the K + 1 leaves / (n - K - 1)),
      d_j = (mu_R - mu_L) / 2,   m_j = (mu_R + mu_L) / 2,
      W_j = T S(tau)^-1 e^-tau d_j,   b_j = -W_j . e^-tau m_j + (T/2) log(n_R / n_L).
    At K = 1 this is fisher_reference(data, labels) (perclass; the same n - 2 normalisation, and the prior term is
    log(5923 / 5923) = 0 on 0/1 MNIST), or fisher_reference on the root's 2-means assignment plus the prior term
    (pooled). Returns W (len(taus), K, N), b (len(taus), K), tree (the dict of split_tree)."""
    X = np.asarray(data, dtype=np.float64)
    tree = split_tree(X, labels, K, seed=seed, pooled=pooled, n_min=n_min)
    n, N = X.shape
    Sw = np.zeros((N, N))
    for leaf in range(K + 1):                 # leaf by leaf (at K = 1 the sum of fisher_reference, in its order)
        Xl = X[tree["leaf_of"] == leaf]
        R = Xl - Xl.mean(0)
        Sw = Sw + R.T @ R
    Sw = Sw / (n - K - 1)
    D = 0.5 * (tree["mu_right"] - tree["mu_left"])            # (K, N)
    M = 0.5 * (tree["mu_right"] + tree["mu_left"])
    prior = 0.5 * T * np.log(tree["n_right"] / tree["n_left"])
    Ws, bs = [], []
    for tau in np.asarray(taus, dtype=np.float64):
        e = math.exp(-tau)
        S = e * e * Sw + T * (1.0 - e * e) * np.eye(N)
        W = T * np.linalg.solve(S, e * D.T).T
        bs.append(np.array([-(W[j] @ (e * M[j])) for j in range(K)]) + prior)
        Ws.append(W)
    return np.stack(Ws), np.stack(bs), tree


def random_orthogonal_rows(K, n, norm, rng):
    """K mutually orthogonal rows of length `norm` in R^n, Haar distributed: the Q factor of the QR decomposition of
    an n x K standard normal matrix drawn from the numpy Generator rng, with the column signs fixed by diag R.
    W W^T = norm^2 I. Needs K <= n. Returns (K, n)."""
    if K > n:
        raise ValueError(f"random_orthogonal_rows: {K} orthogonal rows do not fit in dimension {n}")
    Qm, R = np.linalg.qr(rng.standard_normal((n, K)))
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return norm * (Qm * s).T


def ising_force_matching_local_grad(params, X, T, vis, local=False):
    """The force-matching (score-matching) gradient of the parameters of K Ising units with spring couplings (W, b, Q)
    and of the per-pixel units (wl, bl), written as a local rule (the Ising counterpart of force_matching_local_grad;
    it must equal jax.grad of make_loss, checked in cd1_check.py --local-rule and test 20). Per clamped sample x and
    unit j:
      a_j = b_j + W_j . x + Q_j . x^2,  m_j = tanh(a_j / T),  v_j = 1 - m_j^2,  k3_j = -2 m_j v_j,
      G_j = W_j + 2 Q_j x (the gradient of a_j),  s_j = sum_i Q_ji,
      psi = grad F (the clamp force, including the hidden and per-pixel units' mean forces),
      c_j = k3_j |G_j|^2 - v_j G_j . psi + 2 T v_j s_j                       (one scalar per unit)
      dJ/db_j  = < c_j > / T^3
      dJ/dW_ji = < x_i c_j / T^3 + (2 v_j G_ji - m_j psi_i) / T^2 >
      dJ/dQ_ji = < x_i^2 c_j / T^3 + 2 x_i (2 v_j G_ji - m_j psi_i) / T^2 + 2 m_j / T >
    per-pixel units as in force_matching_local_grad. The only non-local inputs are G_j . psi (the change of unit j's
    field when the pixels are nudged along psi) and |G_j|^2; there is no Gbar term, because Ising units do not
    compete. Returns a dict over W, b, Q (if present), wl, bl (if local)."""
    if "Wp" in params:                           # per-patch units (--patch): their force is not in psi here
        raise NotImplementedError("ising_force_matching_local_grad does not support per-patch Potts units (--patch)")
    W, b = params["W"], params["b"]
    Q = params.get("Q", jnp.zeros_like(W))
    X2 = X * X
    A = X @ W.T + X2 @ Q.T + b                                  # (B, K)
    m = jnp.tanh(A / T)
    v = 1.0 - m * m
    k3 = -2.0 * m * v
    GG = jnp.sum(W * W, axis=1) + 4.0 * X @ (W * Q).T + 4.0 * X2 @ (Q * Q).T     # |G_j|^2 (B, K)
    s = jnp.sum(Q, axis=1)                                      # (K,)
    psi = vis.grad_lap(params["theta"], X)[0] - (m @ W + 2.0 * X * (m @ Q))      # clamp force grad F (B, N)
    if local:
        Al = X * params["wl"] + params["bl"]
        ml = jnp.tanh(Al / T)
        psi = psi - ml * params["wl"]
    Gpsi = psi @ W.T + 2.0 * (psi * X) @ Q.T                    # G_j . psi (B, K)
    c = k3 * GG - v * Gpsi + 2.0 * T * v * s                    # (B, K)
    B = X.shape[0]
    vX = v.T @ X                                                # sum_b v_bj x_bi (K, N)
    # sum_b (2 v_bj G_bji - m_bj psi_bi), with G_bji = W_ji + 2 Q_ji x_bi
    pW = 2.0 * (jnp.sum(v, axis=0)[:, None] * W + 2.0 * Q * vX) - m.T @ psi
    out = {"b": jnp.sum(c, axis=0) / (B * T ** 3),
           "W": (c.T @ X / T ** 3 + pW / T ** 2) / B}
    if "Q" in params:
        pQ = 2.0 * (2.0 * (W * vX + 2.0 * Q * (v.T @ X2)) - m.T @ (psi * X))   # sum_b 2 x (2 v G - m psi)
        out["Q"] = (c.T @ X2 / T ** 3 + pQ / T ** 2 + 2.0 * jnp.sum(m, axis=0)[:, None] / T) / B
    if local:
        wl = params["wl"]
        vl = 1.0 - ml * ml
        k3l = -2.0 * ml * vl
        e = k3l * wl * wl - vl * wl * psi                       # (B, N)
        out["wl"] = jnp.mean(-ml * psi + 2.0 * vl * wl + X * e / T, axis=0) / T ** 2
        out["bl"] = jnp.mean(e, axis=0) / T ** 3
    return out


def ising_fields(W, b, Q, X):
    """Input fields A = X W^T + X^2 Q^T + b (B, K) of K Ising units (Q None: no spring term), numpy float64, summed in
    the order of hidden_terms_quad."""
    X = np.asarray(X, dtype=np.float64)
    A = X @ np.asarray(W, dtype=np.float64).T
    if Q is not None:
        A = A + (X * X) @ np.asarray(Q, dtype=np.float64).T
    return A + np.asarray(b, dtype=np.float64)


def _row_basis(W):
    """Orthonormal basis (N, r) of the row space of W (rank cut at 1e-10 of the largest singular value)."""
    U, sv, _ = np.linalg.svd(np.asarray(W, dtype=np.float64).T, full_matrices=False)
    if sv.size == 0 or sv[0] <= 0:
        return U[:, :0]
    return U[:, sv > 1e-10 * sv[0]]


def ising_unit_stats(W, b, Q, X, T, W_ref=None, W_prev=None, tree=None, leaf=None, labels=None):
    """Diagnostics of K Ising units on a batch X (numpy): fields A = X W^T + X^2 Q^T + b (Q may be None),
    m = tanh(A / T), v = 1 - m^2. A unit's minority share is the share of the time its spin spends in its rarer
    state, (1 - |<m_j>|) / 2 (so a unit whose field sits at a constant 0.5 T is not dead: its spin still flips).
    Returns (rec, arrays):
      rec (JSON scalars): W_norm; with W_ref (K, N): Wref_norm, cos_W_Wref (Frobenius), overlap_ref
        (|U_W^T U_ref|_F^2 / K, U orthonormal bases of the row spaces), norm_ratio_min / _median (|W_j| / |W_ref_j|);
        max_pair_cos (max |cos| between two units); with W_prev: min_cos_prev (min_j cos(W_j, W_prev_j));
        n_dead (minority share < 0.005), n_weak (std a_j < 0.1 T), n_decided (<|m_j|> > 0.9 and not dead);
        mean_abs_m, mean_v; softening_over_T = sum_j <v_j |G_j|^2> / T (G_j = W_j + 2 Q_j x); n_eff_m (participation
        ratio of cov(m)); with tree (split_tree) and leaf (the true leaf of each sample): leaf_acc (descend the tree
        by sign a_j), straddle_off (mean over units of the share of the samples outside the unit's subtree with
        |a_j| < 2T; None if no unit has such samples), own_balance_err (mean over units of |<(1 + m_j) / 2> on the
        unit's own subtree - n_R / (n_L + n_R)|), and with labels as well class_acc (unit 0, mapped through the
        majority labels of the root's children); with labels: n_gated (units with a minority share below 0.02 in one
        class and above 0.1 in another).
      arrays (K,): unit_norm, unit_cos_ref, unit_minority, unit_abs_m, unit_straddle (nan where not defined)."""
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    K, B = W.shape[0], X.shape[0]
    A = ising_fields(W, b, Q, X)
    m = np.tanh(A / T)
    v = 1.0 - m * m
    minority = 0.5 * (1.0 - np.abs(m.mean(axis=0)))
    abs_m = np.abs(m).mean(axis=0)
    dead = minority < 0.005
    weak = A.std(axis=0) < 0.1 * T
    decided = (abs_m > 0.9) & ~dead
    GG = np.sum(W * W, axis=1)[None, :]
    if Q is not None:
        Qa = np.asarray(Q, dtype=np.float64)
        GG = GG + 4.0 * X @ (W * Qa).T + 4.0 * (X * X) @ (Qa * Qa).T
    norms = np.linalg.norm(W, axis=1)
    Wn = W / np.maximum(norms, 1e-300)[:, None]
    cosm = np.abs(Wn @ Wn.T)
    cm = np.atleast_2d(np.cov(m, rowvar=False))
    tr, tr2 = float(np.trace(cm)), float(np.sum(cm * cm))
    rec = {"W_norm": float(np.linalg.norm(W))}
    unit_cos_ref = np.full(K, np.nan)
    if W_ref is not None:
        Wr = np.asarray(W_ref, dtype=np.float64)
        rn = np.linalg.norm(Wr, axis=1)
        unit_cos_ref = np.sum(W * Wr, axis=1) / (norms * rn + 1e-300)
        ratio = norms / np.maximum(rn, 1e-300)
        Ua, Ub = _row_basis(W), _row_basis(Wr)
        rec.update({"Wref_norm": float(np.linalg.norm(Wr)),
                    "cos_W_Wref": float(np.sum(W * Wr) / (np.linalg.norm(W) * np.linalg.norm(Wr) + 1e-300)),
                    "overlap_ref": float(np.sum((Ua.T @ Ub) ** 2) / K),
                    "norm_ratio_min": float(ratio.min()), "norm_ratio_median": float(np.median(ratio))})
    rec["max_pair_cos"] = float(cosm[~np.eye(K, dtype=bool)].max()) if K > 1 else 0.0
    if W_prev is not None:
        Wp = np.asarray(W_prev, dtype=np.float64)
        rec["min_cos_prev"] = float(np.min(np.sum(W * Wp, axis=1) / (norms * np.linalg.norm(Wp, axis=1) + 1e-300)))
    rec.update({"n_dead": int(dead.sum()), "n_weak": int(weak.sum()), "n_decided": int(decided.sum()),
                "mean_abs_m": float(abs_m.mean()), "mean_v": float(v.mean()),
                "softening_over_T": float(np.sum(np.mean(v * GG, axis=0)) / T),
                "n_eff_m": tr * tr / tr2 if tr2 > 0 else 0.0})
    unit_straddle = np.full(K, np.nan)
    if tree is not None and leaf is not None:
        leaf = np.asarray(leaf)
        cl, cr = np.asarray(tree["child_left"]), np.asarray(tree["child_right"])
        node, pred, rows = np.zeros(B, dtype=np.int64), np.full(B, -1, dtype=np.int64), np.arange(B)
        for _ in range(K + 1):                                # descend from the root by the sign of a_j
            act = rows[pred < 0]
            if act.size == 0:
                break
            j = node[act]
            nxt = np.where(A[act, j] > 0, cr[j], cl[j])
            fin = nxt < 0
            pred[act[fin]] = -nxt[fin] - 1
            node[act[~fin]] = nxt[~fin]
        on = np.asarray(tree["H"])[leaf] != 0                 # (B, K): the sample's leaf lies below node j
        n_off, n_on = (~on).sum(axis=0), on.sum(axis=0)
        und = np.abs(A) < 2.0 * T
        unit_straddle = np.where(n_off > 0, (und & ~on).sum(axis=0) / np.maximum(n_off, 1), np.nan)
        share = (0.5 * (1.0 + m) * on).sum(axis=0) / np.maximum(n_on, 1)
        target = np.asarray(tree["n_right"]) / (np.asarray(tree["n_left"]) + np.asarray(tree["n_right"]))
        rec.update({"leaf_acc": float(np.mean(pred == leaf)),
                    "straddle_off": float(np.nanmean(unit_straddle)) if np.any(n_off > 0) else None,
                    "own_balance_err": float(np.mean(np.abs(share - target)[n_on > 0])) if np.any(n_on > 0) else None})
        if labels is not None:
            uc = np.asarray(tree["unit_class"])[0]
            rec["class_acc"] = float(np.mean(np.where(A[:, 0] > 0, uc[1], uc[0]) == np.asarray(labels)))
    if labels is not None:
        lab = np.asarray(labels)
        mins = np.stack([0.5 * (1.0 - np.abs(m[lab == c].mean(axis=0))) for c in np.unique(lab)])     # (C, K)
        rec["n_gated"] = int(np.sum((mins.min(axis=0) < 0.02) & (mins.max(axis=0) > 0.1)))
    arrays = {"unit_norm": norms, "unit_cos_ref": unit_cos_ref, "unit_minority": minority, "unit_abs_m": abs_m,
              "unit_straddle": unit_straddle}
    return rec, arrays


def force_misfit(X, psi, T):
    """G = sym(X^T psi) / (B T) - I (N, N) on a batch X ~ p_t with psi = grad F(X) (numpy). By Stein's identity
    E[x psi^T] / T = I for the exact score, so G = 0 there; for a small new unit, J ~ J0 - sum_j W_j^T G W_j / T^2
    (a direction with a positive eigenvalue of G lowers the loss). Diagnostic only."""
    X = np.asarray(X, dtype=np.float64)
    M = X.T @ np.asarray(psi, dtype=np.float64) / (X.shape[0] * T)
    return 0.5 * (M + M.T) - np.eye(X.shape[1])


# ---------------------------------------------------------------------------------------------------------------
# Initial states and Brownian keys of the sampler (same as the ode_vs_sde 500-trajectory runs)
# ---------------------------------------------------------------------------------------------------------------
def trajectory_set(name, n=None, problem="mnist28", dim=N):
    """'base': the 500 trajectories of jobs 4183779 / 4187940 (0-39 = the paper's shared states and keys,
    40-499 from seed 2026). 'ext': 500 further trajectories from seed 2027. Returns (x0 (n, N), keys (n, 2)).
    Other problems: 'base' / 'ext' are n draws (default 500) from seeds 2026 / 2027 in dimension `dim`."""
    if problem != "mnist28":
        n = 500 if n is None else n
        k_init, k_brown = jr.split(jr.PRNGKey({"base": NEW_SEED, "ext": EXT_SEED}[name]))
        x0 = np.asarray(jnp.sqrt(TEMP) * SIGMA_FORWARD * jr.normal(k_init, (n, dim)), dtype=np.float64)
        return x0, np.asarray(jr.split(k_brown, n))
    if name == "base":
        x_paper = np.load(os.path.join(ABLATION_DIR, "shared_initial_states.npy"))
        keys_paper = np.load(os.path.join(ABLATION_DIR, "shared_brownian_keys.npy"))
        n_new = 500 - x_paper.shape[0]
        k_init, k_brown = jr.split(jr.PRNGKey(NEW_SEED))
        x_new = np.asarray(jnp.sqrt(TEMP) * SIGMA_FORWARD * jr.normal(k_init, (n_new, N)))
        keys_new = np.asarray(jr.split(k_brown, n_new))
        x0 = np.concatenate([x_paper, x_new]).astype(np.float64)
        keys = np.concatenate([keys_paper, keys_new.astype(keys_paper.dtype)])
    elif name == "ext":
        k_init, k_brown = jr.split(jr.PRNGKey(EXT_SEED))
        x0 = np.asarray(jnp.sqrt(TEMP) * SIGMA_FORWARD * jr.normal(k_init, (500, N)), dtype=np.float64)
        keys = np.asarray(jr.split(k_brown, 500))
    else:
        raise ValueError(name)
    if n is not None:
        x0, keys = x0[:n], keys[:n]
    return x0, keys
