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
"""

import hashlib
import math
import os
import re
import sys
from functools import partial

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import flatten_util  # noqa: E402

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
def connectivity(grid=GRID, radius=N_NEIGHBOUR_COUPLINGS):
    """Sorted unique (i, j), i < j, pairs within Chebyshev distance `radius` (default: the paper's 28x28, 14)."""
    return np.asarray(create_2d_square_lattice_connectivity(grid_size=grid, n_neighbour_couplings=radius))


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


def make_problem(name, radius=None):
    """Returns dict(name, grid, n, radius, conn, data (jnp, normalised), labels (np, 1 = class 1), mean, std).
    radius: Chebyshev coupling radius of the visible pairs instead of the problem's default (PROBLEM_GEOMETRY);
    grid - 1 couples all pairs.

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
    conn = connectivity(grid, radius)
    return {"name": name, "grid": grid, "n": grid * grid, "radius": radius, "conn": conn, "data": data,
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

    def __init__(self, conn, n=N):
        conn = np.asarray(conn)
        self.n = int(n)
        self.I = jnp.asarray(conn[:, 0])
        self.J = jnp.asarray(conn[:, 1])
        self.P = int(conn.shape[0])
        self.n_theta = 4 * self.n + 3 * self.P

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


def make_loss_per_sample(vis, T, K, hidden="ising", local=False, samples=0, jackknife=False, replicates=1):
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
    separates bias from noise (R S draws pooled into one estimate would have the smaller bias of S' = R S)."""
    terms = hidden_terms if hidden == "ising" else potts_terms
    if replicates < 1 or (replicates > 1 and samples < 1):
        raise ValueError("replicates must be >= 1, and > 1 only with samples > 0")
    if jackknife not in (False, True, 0, 1, 2) or (jackknife == 2 and (samples < 4 or samples % 4)):
        raise ValueError("jackknife must be False / 0, True / 1, or 2 (2 needs samples divisible by 4)")

    def loss(params, X, key=None):
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

    return loss


def make_loss(vis, T, K, hidden="ising", local=False, samples=0, jackknife=False, replicates=1):
    """Batch mean of make_loss_per_sample (the paper's loss for K = 0)."""
    per_sample = make_loss_per_sample(vis, T, K, hidden, local, samples, jackknife, replicates)

    def loss(params, X, key=None):
        return jnp.mean(per_sample(params, X, key))

    return loss


# ---------------------------------------------------------------------------------------------------------------
# Local (CD-1, short-release) learning rule for the hidden parameters (DESIGN section 2)
# ---------------------------------------------------------------------------------------------------------------
HIDDEN_KEYS = ("W", "b", "Q", "wl", "bl")


def make_force(vis, T, K, hidden="ising", local=False):
    """grad F (B, N) of the model of make_loss_per_sample: visible part plus the averaged-out hidden units."""
    terms = hidden_terms if hidden == "ising" else potts_terms

    def force(params, X):
        g, _ = vis.grad_lap(params["theta"], X)
        if K > 0 and "Q" in params:
            g = g + hidden_terms_quad(params["W"], params["Q"], params["b"], X, T, hidden)[0]
        elif K > 0:
            g = g + terms(params["W"], params["b"], X, T)[0]
        if local:
            g = g + local_terms(params["wl"], params["bl"], X, T)[0]
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


def prototype_reference(data, labels, taus, T, M, seed=0, pooled=False):
    """W*(tau), b*(tau) of one Potts unit with M states: the Gaussian-mixture solution for M k-means prototypes
    (M/2 per class; pooled=True: M prototypes of all images together, without labels) with their pooled
    within-cluster covariance S_w and weights pi_m:
      S(tau) = e^-2tau S_w + T(1 - e^-2tau) I,  mu_m(tau) = e^-tau c_m,
      W_m = T S(tau)^-1 mu_m(tau),  b_m = T log pi_m - W_m . mu_m(tau) / 2.
    The common part of the W_m (their weighted mean) is moved out of the hidden unit, since a uniform linear tilt
    belongs to the visible bias. Returns W (len(taus), M, N), b (len(taus), M), state class (M,); with pooled=True
    the state class is the majority label of the cluster and is used for diagnostics only."""
    classes = np.unique(np.asarray(labels))
    n_cls = classes.size
    if not pooled and M % n_cls:
        raise ValueError(f"M = {M} must be a multiple of the {n_cls} classes (M/{n_cls} prototypes per class)")
    X = np.asarray(data, dtype=np.float64)
    if pooled:
        C, assign = kmeans(X, M, seed)
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
            Cc, assign = kmeans(Xc, M // n_cls, seed + c)
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
    return np.stack(Ws), np.stack(bs), np.array(cls)


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
