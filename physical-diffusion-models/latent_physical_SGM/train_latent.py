#!/usr/bin/env python3
"""Train the physical diffusion model with K Ising hidden units (K = 0 reproduces the paper's model).

Training procedure = the paper's (scripts/physical_diffusion_model_MNIST.py + learning_fns.run_optimization):
  * 100 slices t = 5 -> 0, each warm-started from the previous slice's best parameters;
  * Adam, lr 2e-3 (t = 5) -> 2e-5 (t = 0) linear in the slice index, times 0.95^(step/6000);
  * fresh batches of 512 from the exact forward marginal p_t, with the paper's RNG chain (seed 1);
  * the paper's effective step count per slice (10,901 at t = 5 ... 30,901 at t = 0; CODE_NOTES 8.1);
  * k_6 projected onto >= 0.01 after every step;
  * slice result = the post-update iterate at the step with the lowest pre-update mini-batch loss.
New for K > 0:
  * the hidden couplings W (K x 784) and fields b (K) are trained jointly with theta_v at every slice, by their
    own Adam with lr = HIDDEN_LR_SCALE x (their reference magnitude at that slice), same decay;
  * K = 1 starts along the Gaussian-mixture (Fisher) direction W*(t = 5), b*(t = 5) (DESIGN 1.1; W = 0 is a
    degenerate stationary point); the first FREEZE slices keep W, b fixed while theta_v relaxes from its start.
  * K > 1 Ising units (and K = 1 with --proto-init pooled or random) start from the split-tree reference
    (latent_model.split_reference: a bisecting tree of the data, one unit per node, each the Gaussian-mixture unit that
    separates the node's two children; label-free by default) or, with --proto-init random, from Haar-orthogonal rows;
    their learning-rate scales come from the k-means seed 0 tree, as for Potts units.
Option --cls 0|1 trains on one class only (idea 1), normalised with the joint data's statistics.
Option --select ema: the slice result is an exponential moving average of the iterates (time constant --ema-steps)
instead of the paper's luckiest-mini-batch iterate (which, in most slices, is one of the first ~100 steps).
Option --problem mnist14 | gauss2_8: small test problems (latent_model.make_problem).
Option --hidden potts: one Potts unit with K states (K/2 k-means prototypes per class, Gaussian-mixture initialisation;
latent_model.prototype_reference) instead of K Ising units.
Option --quad: the hidden units also couple to the squared pixels (a_j = b_j + W_j.x + Q_j.x^2), i.e. they modulate the
pixels' spring constants, so the tilted copies of the visible model can differ in shape, not only in position.
Option --local: one Ising hidden unit per pixel, coupled only to its pixel (an asymmetric double well per pixel;
latent_model.local_reference), trained like W, b.
Option --visible quartic: visible self energy up to x^4 only (k_6 = 0); k_duff >= 0.01 and c_duff >= 0 keep the energy
confining, as k_6 >= 0.01 does in the paper. Initial k_duff = 1 (the paper starts with k_6 = 1).
Option --hidden-fixed: the hidden couplings are held at each slice's closed-form reference (no hidden learning at all).
Option --hidden-samples S: force matching with measured hidden statistics (S sampled hidden states per training sample
and step instead of the exact equilibrium averages), as a device with fast but finite hidden units would measure them.
--hidden-jackknife removes the plug-in bias at O(1/S) (with --hidden-jackknife-order 2 also at O(1/S^2));
--hidden-samples-late S2 with --hidden-samples-switch k uses S2
draws from slice k on; --hidden-replicates R (a diagnostic) averages R independent S-draw estimates per step.
Option --hidden-norm: how the hidden parameters' steps are normalised. element: per-element Adam (default); group: one
root-mean-square per parameter group (plain gradient steps with one step size per group, as a simple device would
take); state: one root-mean-square per hidden state (scale_by_adam_per_state; a gain control per unit); occupancy:
each Potts state divides a leaky average of its gradient by a leaky average of its own occupation, over one
occupation-weighted step scale per group (OccupancyTransform; a per-state learning rate from the state's visit count);
--occ-same-draws counts the visits among the draws of the gradient statistics instead of fresh draws.
Option --pair-terms (default all): which visible pair-coupling groups exist, a comma list of lin (c_lin (x_i - x_j)^2),
om (c_om x_i^2 x_j, optomechanical) and duff (c_duff (x_i - x_j)^4 / 4), or none; the others start at zero and are
held at exactly zero after every step. Option --neighbourhood von_neumann: couple only pairs within Manhattan distance
--radius (radius 1: the 4 nearest neighbours) instead of Chebyshev distance.
Option --hidden-rule cd1: the hidden parameters are trained with the local CD-1 short-release rule of DESIGN 2 instead
of the exact gradient (release length --cd1-dt, or kappa^2 T / S per slice; --cd1-releases per sample).
--cd1-hidden-samples S measures the hidden statistics of the CD-1 terms from S sampled hidden states per released
configuration; with --hidden-samples as well, the visible gradient's clamp force is measured too.
Every slice file also stores the per-sample losses on the fixed validation batch (and the paper checkpoint's, for
mnist28), for paired comparisons between runs.

Writes out/latent_physical_SGM/<name>/ (never the paper's folder, never deletes anything); resumes from the last
finished slice. Run under Slurm: cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_train_latent_neuronic.sh
"""

import argparse
import fcntl
import json
import os
import sys
import time
from functools import partial

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402  (enables float64)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import optax  # noqa: E402

from physical_diffusion_fns.helper_fns import sample_forward_process  # noqa: E402

VAL_SEED = 7
VAL_N = 4096


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--cls", choices=["none", "0", "1"], default="none")
    ap.add_argument("--hidden-lr-scale", type=float, default=5e-4)
    ap.add_argument("--freeze-hidden-slices", type=int, default=1)
    ap.add_argument("--steps-scale", type=float, default=1.0, help="fraction of the paper's steps (smoke tests)")
    ap.add_argument("--max-slices", type=int, default=lm.N_TIME_STEPS)
    ap.add_argument("--init-from", default=None,
                    help="start from another run's slices: with --init-slices N, slices 0..N-1 of that run (a folder "
                         "under --out-root with the same model config) are linked into this run, and training resumes "
                         "at slice N with this run's --steps-scale / --lr-scale (late-slice convergence)")
    ap.add_argument("--init-slices", type=int, default=0)
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiplies every learning rate (pixel couplings and hidden units) of the trained slices")
    ap.add_argument("--out-root", default=lm.OUT_ROOT)
    ap.add_argument("--problem", default="mnist28",
                    help="mnist28, mnist14, gauss2_8, or a digit subset mnist28_d<digits> / mnist14_d<digits> "
                         "(e.g. mnist14_d14; data from make_digit_data.py)")
    ap.add_argument("--hidden", choices=["ising", "potts"], default="ising",
                    help="ising: K bistable units (K = 1: the Fisher unit; K > 1: one unit per node of a split tree, "
                         "see --proto-init); potts: one unit with K states (K even)")
    ap.add_argument("--quad", action="store_true",
                    help="hidden units also couple to the squared pixels, a_j = b_j + W_j.x + Q_j.x^2 (Q starts at 0)")
    ap.add_argument("--local", action="store_true",
                    help="add one Ising hidden unit per pixel, coupled only to that pixel (params wl, bl)")
    ap.add_argument("--visible", choices=["sextic", "quartic"], default="sextic",
                    help="quartic: no x^6 self term (k_6 = 0); confinement from k_duff >= 0.01 and c_duff >= 0")
    ap.add_argument("--select", choices=["best", "ema"], default="best",
                    help="slice result: the paper's luckiest-mini-batch iterate, or an EMA of the iterates")
    ap.add_argument("--ema-steps", type=float, default=1000.0)
    ap.add_argument("--val-every", type=int, default=0,
                    help="learning curves: every N steps (a multiple of 100; 0 = off) the validation loss of the current "
                         "slice result (EMA or best iterate, as --select) on the slice's fixed validation batch; per-sample "
                         "losses saved in <run>/val_curves/slice_<k>.npz (paired comparisons between checkpoints)")
    ap.add_argument("--proto-init", choices=["perclass", "pooled", "random"], default=None,
                    help="Potts initial prototypes: K/2 k-means prototypes per class (uses the labels), or K prototypes "
                         "of all images together (pooled: no labels are used anywhere in training), or none (random: "
                         "W starts as Gaussian noise with the rms of the pooled reference at t = 5, b = 0; the pooled "
                         "reference only sets the hidden learning-rate scales and the diagnostics). Ising units "
                         "(latent_model.split_reference): perclass = the class tree first, then bisecting 2-means "
                         "(needs K >= classes - 1; K = 1 is the Fisher unit of the labels), pooled = bisecting 2-means "
                         "of all images (no labels), random = Haar-orthogonal rows with the rms row norm of the pooled "
                         "tree reference at t = 5, b = 0. Default: pooled for Ising units with K > 1, perclass "
                         "otherwise")
    ap.add_argument("--radius", type=int, default=None,
                    help="Chebyshev radius of the visible pair couplings instead of the problem's default (mnist28 14, "
                         "mnist14 7); grid - 1 couples all pairs")
    ap.add_argument("--neighbourhood", choices=list(lm.NEIGHBOURHOODS), default="chebyshev",
                    help="von_neumann: couple only the pairs within Manhattan distance --radius (radius 1: the 4 "
                         "nearest neighbours) instead of Chebyshev distance (radius 1: 8 neighbours)")
    ap.add_argument("--pair-terms", default="all",
                    help="visible pair-coupling groups that exist: 'all' (default), 'none', or a comma list of lin "
                         "(c_lin (x_i - x_j)^2), om (c_om x_i^2 x_j) and duff (c_duff (x_i - x_j)^4 / 4); the others "
                         "start at zero and are held at exactly zero")
    ap.add_argument("--visible-impl", choices=list(lm.VISIBLE_IMPLS), default="dense",
                    help="how the training loss evaluates the visible energy: dense (default: dense N x N products "
                         "of all pair groups, as before), dense_skip (the same without the groups dropped by "
                         "--pair-terms and without k_6 in the quartic model), stencil (per lattice offset, kept "
                         "groups only; about 0.1 ms instead of 7 ms per step at radius 1). Equal up to rounding; "
                         "validation and the paper comparison always use the dense path")
    ap.add_argument("--precision", choices=["f64", "mixed"], default="f64",
                    help="mixed: the training loss and its gradient in float32 with full-float32 (HIGHEST) matrix "
                         "products; parameters, Adam, EMA, the data stream and validation stay float64. Exact "
                         "hidden statistics only (not with --hidden-samples, --hidden-rule cd1, --hidden-norm "
                         "occupancy or --hidden-floor)")
    ap.add_argument("--hidden-fixed", action="store_true",
                    help="hold the hidden couplings at the closed-form reference of each slice (Fisher or prototype "
                         "W*, b*; per-pixel w*, c*) and train only the visible parameters. Not with --quad (no "
                         "reference for Q)")
    ap.add_argument("--hidden-jackknife", action="store_true",
                    help="with --hidden-samples S (even): remove the O(1/S) bias of the plug-in statistics, 2 L(S) - "
                         "[L(first S/2) + L(second S/2)] / 2 from the same draws (latent_model.make_loss_per_sample)")
    ap.add_argument("--ising-reseed", action="store_true",
                    help="K Ising units (split tree): at the start of every slice, reset the units whose field hardly "
                         "varies over the slice's data (std a_j < 0.1 T) to that slice's split-tree reference (Q_j = 0)")
    ap.add_argument("--ising-weak-boost", type=float, default=1.0,
                    help="K Ising units (split tree): step-size factor, for one slice, for the units that are weak at "
                         "its start (1 = off)")
    ap.add_argument("--occ-same-draws", action="store_true",
                    help="with --hidden-norm occupancy and --hidden-samples S: count each Potts state's visits among the "
                         "S draws per sample that the gradient statistics already use (latent_model."
                         "loss_draw_occupation) instead of S fresh draws: S reads per configuration and step, not 2S")
    ap.add_argument("--hidden-jackknife-order", type=int, choices=[1, 2], default=1,
                    help="with --hidden-jackknife: 2 also removes the O(1/S^2) bias (Richardson over S, S/2 and S/4 "
                         "of the same draws: 8/3 L(S) - 2 L(halves) + 1/3 L(quarters); S divisible by 4)")
    ap.add_argument("--hidden-samples", type=int, default=0,
                    help="S > 0: force matching with measured hidden statistics. Every hidden mean (Potts occupation p, "
                         "spin magnetisation m) is a plug-in estimate from S sampled hidden states per training sample "
                         "and step, in the clamp force and in all gradient terms (latent_model.sampled_state_mean); "
                         "0: exact equilibrium statistics")
    ap.add_argument("--hidden-replicates", type=int, default=1,
                    help="diagnostic, with --hidden-samples S: average R independent S-draw estimates per step (the "
                         "bias of S draws, the variance of R S draws; latent_model.make_loss_per_sample)")
    ap.add_argument("--hidden-samples-late", type=int, default=0,
                    help="with --hidden-samples S: use S_late draws instead from slice --hidden-samples-switch on "
                         "(same jackknife setting); 0: S in every slice")
    ap.add_argument("--hidden-samples-switch", type=int, default=0,
                    help="first slice that uses --hidden-samples-late")
    ap.add_argument("--hidden-floor", type=float, default=0.0,
                    help="Potts units with --hidden-samples S: homeostatic revival. Every step, each state that none "
                         "of B S fresh hidden draws visits raises its bias by FLOOR times the hidden bias learning "
                         "rate, after the Adam update (local: a state needs only its own visit count); 0: off")
    ap.add_argument("--hidden-norm", choices=["element", "group", "state", "occupancy"], default="element",
                    help="Adam normalisation of the hidden parameters: per element (optax.scale_by_adam, the default); "
                         "group: one root-mean-square per group W, b, Q, wl, bl (a gradient step with one adaptive "
                         "step size per group, no per-element rescaling); state: one root-mean-square per hidden state "
                         "(row of W and of Q; b, wl, bl per element: one per unit); occupancy (Potts units): each "
                         "state's leaky-averaged gradient divided by its leaky-averaged occupation (exact: the batch "
                         "mean of p(m|x); with --hidden-samples S: its visits among B S fresh hidden draws, floored at "
                         "one visit in the latest step), over one occupation-weighted step scale per group, clipped "
                         "at 1 per element (OccupancyTransform); wl, bl as group. The visible parameters stay per "
                         "element")
    ap.add_argument("--occ-beta", type=float, default=0.999,
                    help="--hidden-norm occupancy: factor of the leaky averages of the gradient and the occupation "
                         "(time constant 1 / (1 - beta) steps)")
    ap.add_argument("--hidden-rule", choices=["exact", "cd1"], default="exact",
                    help="gradient of the hidden parameters: exact (autodiff of the score-matching loss) or the local "
                         "CD-1 short-release estimate of DESIGN 2 (latent_model.make_cd1_hidden_grad); the visible "
                         "parameters get the gradient of the loss (with --hidden-samples: the measured clamp force)")
    ap.add_argument("--cd1-hidden-samples", type=int, default=0,
                    help="cd1: the hidden statistics of the CD-1 terms are measured from S sampled hidden states per "
                         "released configuration (releases * S at the clamped data) instead of being exact averages; "
                         "with --hidden-samples as well, every hidden statistic of the training is measured")
    ap.add_argument("--cd1-kappa", type=float, default=0.3,
                    help="cd1: release length dt = kappa^2 T / S per slice, S = hidden_field_scale at the slice's "
                         "start, so that a release moves the hidden fields by about kappa T")
    ap.add_argument("--cd1-dt", type=float, default=0.0, help="cd1: a fixed release length instead (if > 0)")
    ap.add_argument("--cd1-releases", type=int, default=1, help="cd1: independent releases per sample")
    ap.add_argument("--seed", type=int, default=lm.KEY_SEED,
                    help="training seed (>= 1 for Potts units and for Ising units other than the K = 1 Fisher unit): "
                         "the optimisation key chain jr.PRNGKey(seed) and, for Potts units, the k-means seeds "
                         "100 (seed - 1) + class of the initial prototypes; for Ising split trees the k-means seeds "
                         "100 (seed - 1) + 1000 + leaf, and with --proto-init random the start draw "
                         "default_rng(1000 + seed). The default 1 is the paper's. The data, the validation batches and "
                         "the hidden learning-rate scales (always from the seed-1 reference) do not depend on it")
    return ap.parse_args()


def log(path, line):
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def save_atomic(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def scale_by_adam_grouped(group_keys, b1=0.9, b2=0.999, eps=1e-8):
    """Adam's direction (as optax.scale_by_adam) with the second moment of every group in group_keys pooled over the
    group: each element of such a group moves by its own momentum divided by the group's root-mean-square gradient,
    i.e. a gradient step with one adaptive step size per group, as a device without per-element normalisation would
    take. Other groups: per element, as optax.scale_by_adam."""
    def init(params):
        mu = {k: jnp.zeros_like(v) for k, v in params.items()}
        nu = {k: (jnp.zeros((), v.dtype) if k in group_keys else jnp.zeros_like(v)) for k, v in params.items()}
        return mu, nu, jnp.zeros((), jnp.int32)

    def update(grads, state, params=None):
        mu, nu, count = state
        count = count + 1
        mu = {k: b1 * mu[k] + (1.0 - b1) * g for k, g in grads.items()}
        nu = {k: b2 * nu[k] + (1.0 - b2) * (jnp.mean(g * g) if k in group_keys else g * g) for k, g in grads.items()}
        c1, c2 = 1.0 - b1 ** count, 1.0 - b2 ** count
        return {k: (mu[k] / c1) / (jnp.sqrt(nu[k] / c2) + eps) for k in grads}, (mu, nu, count)

    return optax.GradientTransformation(init, update)


def scale_by_adam_per_state(row_keys, b1=0.9, b2=0.999, eps=1e-8):
    """--hidden-norm state: Adam's direction with the second moment of every row of the 2-d groups in row_keys (W, Q:
    one row per hidden state) pooled over the row, i.e. one adaptive step size per hidden state (a gain control per
    unit) instead of one per coupling. 1-d groups (b: one entry per state; wl, bl: one per pixel unit) and theta stay
    per element, as optax.scale_by_adam. The first moment stays per element."""
    def init(params):
        mu = {k: jnp.zeros_like(v) for k, v in params.items()}
        nu = {k: (jnp.zeros((v.shape[0], 1), v.dtype) if k in row_keys and v.ndim == 2 else jnp.zeros_like(v))
              for k, v in params.items()}
        return mu, nu, jnp.zeros((), jnp.int32)

    def update(grads, state, params=None):
        mu, nu, count = state
        count = count + 1
        mu = {k: b1 * mu[k] + (1.0 - b1) * g for k, g in grads.items()}
        nu = {k: b2 * nu[k] + (1.0 - b2) * (jnp.mean(g * g, axis=1, keepdims=True) if k in row_keys and g.ndim == 2
                                            else g * g) for k, g in grads.items()}
        c1, c2 = 1.0 - b1 ** count, 1.0 - b2 ** count
        return {k: (mu[k] / c1) / (jnp.sqrt(nu[k] / c2) + eps) for k in grads}, (mu, nu, count)

    return optax.GradientTransformation(init, update)


class OccupancyTransform:
    """--hidden-norm occupancy: a per-state learning rate set by the state's own occupation, one global step scale.

    Potts keys (W, Q: one row per state; b: one entry per state). With p_m the state's occupation in this step
    (update's argument occ: the batch mean of p(m|x) with exact statistics, or the fraction of B S fresh hidden draws
    that visit m with measured ones), <.> a leaky average with factor beta_o (bias-corrected), and
        d_m = max(<p_m>, w p_unit, 1e-30)
    (w = the weight of the latest step in <.>: 1/count at the start of a slice, 1 - beta_o later; p_unit = 1/(B S),
    so the floor is one visit in the latest step; p_unit = 0 with exact statistics), the state's direction is
        u_m = (<g_m> / d_m) / sqrt(<sum_m' <p_m'> mean((g_m' / d_m')^2)>_b2),   clipped to [-clip, clip] per element.
    The numerator is the state's gradient per unit occupation, averaged over about 1 / (1 - beta_o) steps, so that a
    rarely visited state moves steadily, about as fast as an occupied one (a state's gradient carries its occupation).
    The denominator is one step scale per group: the occupation-weighted mean square of this step's gradient per unit
    occupation (leaky average with factor b2), noise included, so that near an optimum the steps shrink as Adam's do,
    and the visits of rare states, weighted by their small occupation, hardly change it. Other group keys (wl, bl): as
    scale_by_adam_grouped; theta per element (optax.scale_by_adam)."""

    def __init__(self, potts_keys, group_keys, beta_o=0.999, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        self.potts_keys, self.group_keys = tuple(potts_keys), tuple(group_keys)
        self.beta_o, self.b1, self.b2, self.eps, self.clip = beta_o, b1, b2, eps, clip

    def init(self, params):
        mu = {k: jnp.zeros_like(v) for k, v in params.items()}
        nu = {k: (jnp.zeros((), v.dtype) if k in self.group_keys else jnp.zeros_like(v)) for k, v in params.items()}
        return mu, nu, jnp.zeros_like(params["b"]), jnp.zeros((), jnp.int32)

    def update(self, grads, state, occ, p_unit):
        mu, nu, pbar, count = state
        bo, b1, b2, eps = self.beta_o, self.b1, self.b2, self.eps
        count = count + 1
        pbar = bo * pbar + (1.0 - bo) * occ
        co, c1, c2 = 1.0 - bo ** count, 1.0 - b1 ** count, 1.0 - b2 ** count
        phat = pbar / co
        den = jnp.maximum(phat, jnp.maximum((1.0 - bo) / co * p_unit, 1e-30))     # one value per state
        wts = phat / jnp.maximum(jnp.sum(phat), 1e-30)                           # occupation weights of the scale
        new_mu, new_nu, out = {}, {}, {}
        for k, g in grads.items():
            if k in self.potts_keys:
                dd = den[:, None] if g.ndim == 2 else den
                m = bo * mu[k] + (1.0 - bo) * g
                gt = g / dd                                   # this step's gradient per unit occupation
                ms = jnp.mean(gt * gt, axis=1) if g.ndim == 2 else gt * gt
                n = b2 * nu[k] + (1.0 - b2) * jnp.sum(wts * ms)
                out[k] = jnp.clip(((m / co) / dd) / (jnp.sqrt(n / c2) + eps), -self.clip, self.clip)
            else:
                m = b1 * mu[k] + (1.0 - b1) * g
                n = b2 * nu[k] + (1.0 - b2) * (jnp.mean(g * g) if k in self.group_keys else g * g)
                out[k] = (m / c1) / (jnp.sqrt(n / c2) + eps)
            new_mu[k], new_nu[k] = m, n
        return out, (new_mu, new_nu, pbar, count)


def build_runner(loss_fn, data_train, n_vis=lm.N, ema_steps=1000.0, visible="sextic", n_pairs=None,
                 cd1_grad=None, cd1_releases=1, loss_key=False, floor=0.0, floor_samples=0, hidden_norm="element",
                 occ_beta=0.999, occ_same_draws=False, zero_slices=()):
    """Jitted scan of n training steps, identical in effect to the paper's training_step loop.
    zero_slices (--pair-terms): slices of the flat visible parameters set to exactly zero after every step (dropped
    pair-coupling groups); empty by default, which leaves every step unchanged.

    Carry: (params, adam_state, key, step, best_params, best_loss, best_epoch, ema_params). best_* implement the
    paper's slice result (the post-update iterate at the lowest pre-update mini-batch loss); ema_params is an
    exponential moving average of the iterates with time constant ema_steps (alternative slice result, --select ema).
    cd1_grad (latent_model.make_cd1_hidden_grad): if given, the hidden parameters get the CD-1 short-release estimate
    with release length dt (an argument of run) and cd1_releases releases per sample, drawn with the third key of
    the paper's per-step split (unused otherwise), so the data stream is the same as with exact gradients.
    loss_key: the loss takes that third key as well (sampled hidden statistics, --hidden-samples). With both, the
    third key is split into one for the loss and one for CD-1 (every other mode is unchanged).
    floor > 0 (--hidden-floor, Potts units): homeostatic revival. Each step draws floor_samples hidden states per
    training sample (B floor_samples visits in all) and raises the bias of every state that none of them visited by
    floor * lr_b (after the Adam update), so that no state sinks below about one visit per step. Only then is the
    third key split (one more key for these draws); without it every mode is unchanged.
    hidden_norm: "element" optax.scale_by_adam; "group" scale_by_adam_grouped; "state" scale_by_adam_per_state;
    "occupancy" (Potts units) OccupancyTransform, with each state's occupation per step from the exact posteriors
    (floor_samples = 0) or from floor_samples fresh hidden draws per training sample (the third key is then split as
    for the floor). Only "occupancy" with floor_samples > 0 changes the key stream. occ_same_draws (with "occupancy"
    and sampled statistics, floor_samples = S): count the visits among the S draws the loss itself takes
    (latent_model.loss_draw_occupation with the loss key; no extra key, S reads instead of 2S).
    """
    value_and_grad = jax.value_and_grad(loss_fn)
    adam = optax.scale_by_adam()          # b1 0.9, b2 0.999, eps 1e-8: the direction of optax.adam
    if hidden_norm == "group":            # --hidden-norm group: the hidden groups without per-element normalisation
        adam = scale_by_adam_grouped(lm.HIDDEN_KEYS)
    if hidden_norm == "state":            # --hidden-norm state: one adaptive step size per hidden state
        adam = scale_by_adam_per_state(("W", "Q"))
    occupancy = hidden_norm == "occupancy"
    if occupancy:                         # --hidden-norm occupancy: a per-state learning rate from the occupation
        adam = OccupancyTransform(("W", "b", "Q"), lm.HIDDEN_KEYS, beta_o=occ_beta)
    p_unit = 1.0 / (lm.BATCH_SIZE * floor_samples) if floor_samples > 0 else 0.0
    if occ_same_draws and not (occupancy and loss_key and floor_samples > 0 and cd1_grad is None):
        raise ValueError("occ_same_draws needs hidden_norm occupancy, sampled statistics (loss_key, floor_samples > 0) "
                         "and no CD-1")
    extra_key = floor > 0 or (occupancy and floor_samples > 0 and not occ_same_draws)
    k6 = lm.k6_slice(n_vis)
    w_ema = 1.0 / ema_steps
    if visible == "quartic":
        kd, cd = lm.kduff_slice(n_vis), lm.cduff_slice(n_vis, n_pairs)

    def project(theta):
        for s in zero_slices:             # --pair-terms: dropped pair-coupling groups stay exactly zero
            theta = theta.at[s].set(0.0)
        if visible == "sextic":           # the paper's constraint
            return theta.at[k6].set(jnp.maximum(theta[k6], lm.K6_FLOOR))
        theta = theta.at[k6].set(0.0)
        theta = theta.at[kd].set(jnp.maximum(theta[kd], lm.K4_FLOOR))
        return theta.at[cd].set(jnp.maximum(theta[cd], 0.0))

    def unvisited(params, X, key):
        """1 for every Potts state that none of floor_samples draws per sample (at these fields) visits, else 0."""
        A = X @ params["W"].T + params["b"]
        if "Q" in params:
            A = A + (X * X) @ params["Q"].T
        idx = jr.categorical(key, A / lm.TEMP, axis=-1, shape=(floor_samples,) + A.shape[:-1])
        return (jnp.bincount(idx.ravel(), length=A.shape[-1]) == 0).astype(A.dtype)

    def occupation(params, X, key):
        """Each Potts state's occupation in this batch: exact (mean of p(m|x)) or the visit fraction of
        floor_samples draws per sample."""
        A = X @ params["W"].T + params["b"]
        if "Q" in params:
            A = A + (X * X) @ params["Q"].T
        if floor_samples == 0:
            return jnp.mean(jax.nn.softmax(A / lm.TEMP, axis=-1), axis=0)
        idx = jr.categorical(key, A / lm.TEMP, axis=-1, shape=(floor_samples,) + A.shape[:-1])
        return jnp.bincount(idx.ravel(), length=A.shape[-1]).astype(A.dtype) / idx.size

    def one_step(carry, t_curr, lrs, dt):
        params, adam_state, key, step, best_params, best_loss, best_epoch, ema = carry
        key, k_epoch, k_release = jr.split(key, 3)                # same split as run_optimization (third key unused there)
        X = sample_forward_process(t_curr, lm.BATCH_SIZE, lm.TEMP, lm.SIGMA_FORWARD, data_train, k_epoch, beta=1)
        k_floor = k_release
        if extra_key:                                             # one more key, for the visit count
            k_release, k_floor = jr.split(k_release)
        k_loss = k_cd1 = k_release
        if loss_key and cd1_grad is not None:                     # both draw randomness: split the third key
            k_loss, k_cd1 = jr.split(k_release)
        if loss_key:                                              # hidden statistics sampled with the third key
            loss, grads = value_and_grad(params, X, k_loss)
        else:
            loss, grads = value_and_grad(params, X)               # loss before the update, as the paper records
        if cd1_grad is not None:                                  # local rule for the hidden parameters
            grads = dict(grads, **cd1_grad(params, X, k_cd1, dt, cd1_releases))
        if visible == "quartic":                                  # k_6 is not a parameter of the quartic model
            grads = dict(grads, theta=grads["theta"].at[k6].set(0.0))
        if occupancy:
            occ = (lm.loss_draw_occupation(params, X, lm.TEMP, k_loss, floor_samples) if occ_same_draws
                   else occupation(params, X, k_floor))
            direction, adam_state = adam.update(grads, adam_state, occ, p_unit)
        else:
            direction, adam_state = adam.update(grads, adam_state)
        decay = lm.LR_DECAY_RATE ** (step / lm.LR_DECAY_STEPS)   # optax.exponential_decay(staircase=False)
        new = {g: params[g] - lrs[g] * decay * direction[g] for g in params}
        if floor > 0:                                             # homeostatic revival of unvisited states
            new["b"] = new["b"] + floor * lrs["b"] * decay * unvisited(params, X, k_floor)
        new["theta"] = project(new["theta"])
        better = loss < best_loss
        best_params = jax.tree_util.tree_map(lambda a, b: jnp.where(better, a, b), new, best_params)
        best_loss = jnp.where(better, loss, best_loss)
        best_epoch = jnp.where(better, step, best_epoch)
        ema = jax.tree_util.tree_map(lambda e, p: e + w_ema * (p - e), ema, new)
        return (new, adam_state, key, step + 1, best_params, best_loss, best_epoch, ema), loss

    @partial(jax.jit, static_argnums=3)
    def run(carry, t_curr, lrs, n, dt=0.0):
        return jax.lax.scan(lambda c, _: one_step(c, t_curr, lrs, dt), carry, None, length=n)

    return run, adam


def init_carry(params, adam, key):
    return (params, adam.init(params), key, jnp.int32(0), params, jnp.asarray(jnp.inf), jnp.int32(0), params)


def main():
    args = parse()
    if args.proto_init is None:              # resolved here, so that every invocation from before keeps its value
        args.proto_init = "pooled" if args.hidden == "ising" and args.K > 1 else "perclass"
    fisher = args.hidden == "ising" and args.K == 1 and args.proto_init == "perclass"   # the original K = 1 path
    tree = None                              # split tree of K Ising units (latent_model.split_tree), if any
    lm.problem_geometry(args.problem)                       # unknown problem names fail here
    n_cls = lm.problem_classes(args.problem)
    if (args.ising_reseed or args.ising_weak_boost != 1.0) and not (args.hidden == "ising" and args.K > 0 and not fisher
                                                                    and not args.hidden_fixed):
        raise ValueError("--ising-reseed and --ising-weak-boost need K > 0 trained Ising units from the split tree "
                         "(--hidden ising, not the K = 1 Fisher unit, not --hidden-fixed)")
    if args.ising_weak_boost <= 0:
        raise ValueError(f"--ising-weak-boost {args.ising_weak_boost}: must be > 0")
    if fisher and n_cls != 2:
        raise NotImplementedError("the Fisher initialisation of one Ising unit needs two classes")
    if args.hidden == "ising" and args.K > 0 and not fisher:          # K Ising units from the split tree
        n_pix = lm.problem_geometry(args.problem)[0] ** 2
        if args.seed < lm.KEY_SEED:
            raise ValueError(f"--seed {args.seed}: Ising split-tree units need --seed >= {lm.KEY_SEED} "
                             f"(k-means seeds 100 (seed - {lm.KEY_SEED}) + 1000 + leaf must be >= 0)")
        if args.K > n_pix and args.proto_init == "random":     # Haar-orthogonal rows need K <= pixels; the tree does not
            raise ValueError(f"--K {args.K}: --proto-init random allows at most {n_pix} Ising units (orthogonal rows) "
                             f"on {args.problem}")
        if args.proto_init == "perclass" and args.K < n_cls - 1:
            raise ValueError(f"--proto-init perclass with Ising units needs K >= {n_cls - 1} (the class tree of the "
                             f"{n_cls} classes), not {args.K}")
    if args.hidden == "potts" and args.K > 0 and (args.K < 2 or (args.proto_init == "perclass" and args.K % n_cls)):
        raise ValueError(f"--hidden potts needs K >= 2 states, a multiple of the {n_cls} classes with "
                         f"--proto-init perclass (K/{n_cls} prototypes per class)")
    if args.K > 0 and args.cls != "none":
        raise NotImplementedError("K > 0 with --cls: the Fisher initialisation needs both classes; "
                                  "no one-class initialisation is defined")
    if args.hidden == "potts" and args.K > 0 and args.seed < lm.KEY_SEED:
        raise ValueError(f"--seed {args.seed}: Potts units need --seed >= {lm.KEY_SEED} "
                         f"(k-means seeds 100 (seed - {lm.KEY_SEED}) + class must be >= 0)")
    if args.hidden_fixed and (args.quad or not (args.K > 0 or args.local)):
        raise ValueError("--hidden-fixed needs hidden units (--K > 0 or --local) and excludes --quad "
                         "(Q has no closed-form reference)")
    if args.proto_init == "random" and (args.K == 0 or args.hidden_fixed):
        raise ValueError("--proto-init random needs hidden units (--hidden potts or ising with --K > 0) and trained "
                         "couplings (not --hidden-fixed)")
    if args.hidden_jackknife and (args.hidden_samples < 2 or args.hidden_samples % 2):
        raise ValueError("--hidden-jackknife needs --hidden-samples S >= 2, S even")
    if args.hidden_jackknife_order == 2 and (not args.hidden_jackknife or args.hidden_samples < 4
                                             or args.hidden_samples % 4 or args.hidden_samples_late % 4):
        raise ValueError("--hidden-jackknife-order 2 needs --hidden-jackknife and --hidden-samples (and "
                         "--hidden-samples-late, if used) divisible by 4")
    if args.hidden_samples < 0 or (args.hidden_samples > 0 and not (args.K > 0 or args.local)):
        raise ValueError("--hidden-samples needs hidden units")
    if args.val_every < 0 or args.val_every % 100:
        raise ValueError("--val-every must be 0 or a positive multiple of 100")
    if args.hidden_rule == "cd1" and args.hidden_jackknife:
        raise ValueError("--hidden-jackknife is for force matching: with --hidden-rule cd1 only the visible gradient "
                         "uses the loss, and it is linear in the measured statistics (no plug-in bias)")
    if args.cd1_hidden_samples < 0 or (args.cd1_hidden_samples > 0 and args.hidden_rule != "cd1"):
        raise ValueError("--cd1-hidden-samples needs --hidden-rule cd1")
    if args.hidden_replicates < 1 or (args.hidden_replicates > 1 and (args.hidden_samples < 1
                                                                       or args.hidden_rule != "exact")):
        raise ValueError("--hidden-replicates R > 1 needs --hidden-samples S > 0 and --hidden-rule exact")
    if args.hidden_samples_late < 0 or (args.hidden_samples_late > 0 and (
            args.hidden_samples < 1 or args.hidden_samples_switch < 1 or args.hidden_rule != "exact"
            or (args.hidden_jackknife and args.hidden_samples_late % 2))):
        raise ValueError("--hidden-samples-late needs --hidden-samples S > 0, --hidden-samples-switch >= 1, "
                         "--hidden-rule exact, and an even value with --hidden-jackknife")
    if args.hidden_samples_switch and not args.hidden_samples_late:
        raise ValueError("--hidden-samples-switch needs --hidden-samples-late")
    if args.hidden_floor < 0 or (args.hidden_floor > 0 and (
            args.hidden != "potts" or args.K < 2 or args.hidden_samples < 1 or args.hidden_rule != "exact"
            or args.hidden_fixed or args.hidden_replicates > 1)):
        raise ValueError("--hidden-floor needs trained Potts units (--hidden potts --K >= 2, not --hidden-fixed), "
                         "--hidden-samples S > 0, --hidden-rule exact, and no --hidden-replicates (its visit count "
                         "would see only S of the R S draws)")
    if args.hidden_norm == "occupancy" and (
            args.hidden != "potts" or args.K < 2 or args.hidden_fixed or args.hidden_rule != "exact"
            or args.hidden_replicates > 1 or args.hidden_floor > 0 or not 0.0 < args.occ_beta < 1.0):
        raise ValueError("--hidden-norm occupancy needs trained Potts units (--hidden potts --K >= 2, not "
                         "--hidden-fixed), --hidden-rule exact, no --hidden-replicates, no --hidden-floor, and "
                         "0 < --occ-beta < 1")
    if args.occ_same_draws and (args.hidden_norm != "occupancy" or args.hidden_samples < 1):
        raise ValueError("--occ-same-draws needs --hidden-norm occupancy and --hidden-samples S > 0")
    if args.precision == "mixed" and (args.hidden_samples > 0 or args.hidden_rule != "exact"
                                      or args.hidden_norm == "occupancy" or args.hidden_floor > 0):
        raise ValueError("--precision mixed needs exact hidden statistics: not with --hidden-samples, --hidden-rule "
                         "cd1, --hidden-norm occupancy or --hidden-floor (their draws and occupations would change "
                         "with the float32 fields)")
    pair_terms = lm.parse_pair_terms(args.pair_terms)          # raises on an invalid list
    if args.neighbourhood != "chebyshev" and args.radius is None:
        raise ValueError("--neighbourhood von_neumann needs an explicit --radius")
    if args.hidden_norm == "state" and (args.K < 1 or args.hidden_fixed):
        raise ValueError("--hidden-norm state needs trained hidden units (--K >= 1, not --hidden-fixed)")
    if args.hidden_rule == "cd1" and (args.hidden_fixed or not (args.K > 0 or args.local) or args.cd1_releases < 1
                                      or (args.cd1_dt <= 0 and args.cd1_kappa <= 0)):
        raise ValueError("--hidden-rule cd1 needs trained hidden units (--K > 0 or --local, not --hidden-fixed), "
                         "--cd1-releases >= 1, and --cd1-dt > 0 or --cd1-kappa > 0")
    out_dir = os.path.join(args.out_root, args.name)
    slice_dir = os.path.join(out_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)
    logf = os.path.join(out_dir, "train_log.txt")
    # one job per run directory: an exclusive lock, released automatically when the process ends
    lock_fd = os.open(os.path.join(out_dir, "LOCK"), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError):
        raise SystemExit(f"{out_dir} is locked by another job; not starting a second one")
    except OSError as e:                                    # e.g. no lock support on this filesystem
        print(f"warning: could not lock {out_dir} ({e}); continuing without a lock", flush=True)

    config = {
        "name": args.name, "K": args.K, "cls": args.cls, "hidden_lr_scale": args.hidden_lr_scale,
        "freeze_hidden_slices": args.freeze_hidden_slices, "steps_scale": args.steps_scale,
        "max_slices": args.max_slices, "temp": lm.TEMP, "batch": lm.BATCH_SIZE, "val_seed": VAL_SEED, "val_n": VAL_N,
        "init": "none" if args.K == 0 else ("fisher" if fisher else ("split-tree" if args.hidden == "ising"
                                                                     else "prototypes")),
        "hidden": args.hidden, "local": args.local, "quad": args.quad,
        "problem": args.problem, "select": args.select, "ema_steps": args.ema_steps, "visible": args.visible,
        "seed": args.seed, "proto_init": args.proto_init, "hidden_fixed": args.hidden_fixed,
        "hidden_rule": args.hidden_rule, "radius": args.radius, "hidden_samples": args.hidden_samples,
        "hidden_jackknife": args.hidden_jackknife, "hidden_replicates": args.hidden_replicates,
        "hidden_samples_late": args.hidden_samples_late, "hidden_samples_switch": args.hidden_samples_switch,
        "hidden_floor": args.hidden_floor, "hidden_norm": args.hidden_norm, "occ_beta": args.occ_beta,
        "hidden_jackknife_order": args.hidden_jackknife_order, "occ_same_draws": args.occ_same_draws,
        "pair_terms": ",".join(pair_terms) if pair_terms else "none", "neighbourhood": args.neighbourhood,
        "ising_reseed": args.ising_reseed, "ising_weak_boost": args.ising_weak_boost, "val_every": args.val_every,
        "init_from": args.init_from, "init_slices": args.init_slices, "lr_scale": args.lr_scale,
        "visible_impl": args.visible_impl, "precision": args.precision,
    }
    if args.hidden_rule == "cd1":
        config.update(cd1_kappa=args.cd1_kappa, cd1_dt=args.cd1_dt, cd1_releases=args.cd1_releases,
                      cd1_hidden_samples=args.cd1_hidden_samples)
    defaults = {"problem": "mnist28", "select": "best", "ema_steps": 1000.0, "visible": "sextic",
                "hidden": "ising", "local": False, "quad": False,
                "seed": lm.KEY_SEED, "proto_init": "perclass",
                "hidden_fixed": False, "hidden_rule": "exact", "radius": None,
                "hidden_samples": 0, "hidden_jackknife": False,
                "cd1_hidden_samples": 0, "hidden_replicates": 1,
                "hidden_samples_late": 0, "hidden_samples_switch": 0,
                "hidden_floor": 0.0, "hidden_norm": "element", "occ_beta": 0.999,
                "hidden_jackknife_order": 1, "occ_same_draws": False,
                "pair_terms": "duff,lin,om", "neighbourhood": "chebyshev",
                "ising_reseed": False, "ising_weak_boost": 1.0, "val_every": 0,
                "init_from": None, "init_slices": 0, "lr_scale": 1.0,
                "visible_impl": "dense", "precision": "f64"}   # runs started before these options existed
    code_sha = {f: lm.sha256_file(os.path.join(HERE, f)) for f in ("latent_model.py", "train_latent.py")}
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        diff = {k: (old.get(k, defaults.get(k)), v) for k, v in config.items() if old.get(k, defaults.get(k)) != v}
        if diff:
            raise RuntimeError(f"existing run {out_dir} has a different config: {diff}")
        if old.get("sha256") != code_sha:
            log(logf, f"WARNING: code differs from the run's first start: {old.get('sha256')} -> {code_sha}; "
                      f"slices from here on are made with the new code (recorded per slice)")
    else:
        config.update({
            "jax": jax.__version__, "devices": str(jax.devices()), "git_commit": os.environ.get("GIT_COMMIT", "?"),
            "slurm_job": os.environ.get("SLURM_JOB_ID", "manual"), "sha256": code_sha,
            "data_sha256": lm.sha256_file(lm.problem_data_path(args.problem)),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2)
    if os.path.exists(os.path.join(out_dir, "DONE")):
        log(logf, f"{out_dir} is complete (DONE marker); nothing to do")
        return
    log(logf, f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
              f"jax {jax.__version__}; devices {jax.devices()}; code {code_sha}; config "
              f"{json.dumps({k: config[k] for k in ('problem', 'visible', 'hidden', 'K', 'cls', 'select', 'hidden_lr_scale', 'freeze_hidden_slices', 'steps_scale', 'max_slices', 'seed', 'hidden_fixed', 'hidden_rule', 'radius', 'proto_init', 'hidden_samples', 'hidden_jackknife', 'hidden_replicates', 'hidden_samples_late', 'hidden_samples_switch', 'hidden_floor', 'hidden_norm', 'occ_beta', 'hidden_jackknife_order', 'occ_same_draws', 'pair_terms', 'neighbourhood', 'ising_reseed', 'ising_weak_boost', 'visible_impl', 'precision')})}"
              + (f"; cd1 kappa {args.cd1_kappa}, dt {args.cd1_dt}, releases {args.cd1_releases}, "
                 f"hidden samples {args.cd1_hidden_samples}" if args.hidden_rule == "cd1" else ""))

    # ---- data -------------------------------------------------------------------------------------------------
    prob = lm.make_problem(args.problem, args.radius, args.neighbourhood)
    data, labels, n_vis = prob["data"], prob["labels"], prob["n"]
    if args.cls == "none":
        data_train, labels_train = data, labels
    else:
        sel = labels == int(args.cls)
        data_train, labels_train = data[np.where(sel)[0]], labels[sel]
    log(logf, f"data ({args.problem}): {data.shape[0]} images of {n_vis} pixels, normalisation mean {prob['mean']:.5f} "
              f"std {prob['std']:.5f}; training on {data_train.shape[0]} (cls {args.cls}); {prob['conn'].shape[0]} pairs")

    # ---- model ------------------------------------------------------------------------------------------------
    vis_dense = lm.DenseVisible(prob["conn"], n_vis)          # validation and the paper comparison: always dense
    vis = vis_dense if args.visible_impl == "dense" else lm.make_visible(
        prob["conn"], n_vis, args.visible_impl, pair_terms, sextic=args.visible == "sextic")   # the training loss
    cdt = jnp.float32 if args.precision == "mixed" else None   # --precision mixed: float32 training loss
    theta0, _ = lm.paper_theta_layout(vis.P, n_vis)
    if args.visible == "quartic":
        theta0 = lm.quartic_initial_theta(theta0, n_vis, vis.P)
    zero_slices = lm.dropped_pair_slices(n_vis, vis.P, pair_terms)    # --pair-terms: groups held at exactly zero
    for s in zero_slices:
        theta0 = theta0.at[s].set(0.0)
    K = args.K
    taus = lm.forward_time_pts()
    lr_v = lm.lr_schedule()
    n_slices = min(args.max_slices, lm.N_TIME_STEPS)
    jk = (2 if args.hidden_jackknife_order == 2 else True) if args.hidden_jackknife else False
    loss_fn = lm.make_loss(vis, lm.TEMP, K, args.hidden, args.local, samples=args.hidden_samples,
                           jackknife=jk, replicates=args.hidden_replicates, compute_dtype=cdt)
    loss_ps = jax.jit(lm.make_loss_per_sample(vis_dense, lm.TEMP, K, args.hidden, args.local))
    loss_ps_k0 = jax.jit(lm.make_loss_per_sample(vis_dense, lm.TEMP, 0))
    cd1_grad = None
    if args.hidden_rule == "cd1":
        cd1_grad = lm.make_cd1_hidden_grad(vis, lm.TEMP, K, args.hidden, args.local, samples=args.cd1_hidden_samples)
        field_scale = jax.jit(lambda p, X: lm.hidden_field_scale(p, X, lm.TEMP, K, args.hidden, args.local))
    run, adam = build_runner(loss_fn, data_train, n_vis, args.ema_steps, args.visible, vis.P,
                             cd1_grad=cd1_grad, cd1_releases=args.cd1_releases, loss_key=args.hidden_samples > 0,
                             floor=args.hidden_floor, floor_samples=args.hidden_samples, hidden_norm=args.hidden_norm,
                             occ_beta=args.occ_beta, occ_same_draws=args.occ_same_draws, zero_slices=zero_slices)
    run_late = None
    if args.hidden_samples_late:                  # another draw count from slice --hidden-samples-switch on
        run_late, _ = build_runner(
            lm.make_loss(vis, lm.TEMP, K, args.hidden, args.local, samples=args.hidden_samples_late,
                         jackknife=jk, replicates=args.hidden_replicates, compute_dtype=cdt),
            data_train, n_vis, args.ema_steps, args.visible, vis.P, loss_key=True,
            floor=args.hidden_floor, floor_samples=args.hidden_samples_late, hidden_norm=args.hidden_norm,
            occ_beta=args.occ_beta, occ_same_draws=args.occ_same_draws, zero_slices=zero_slices)

    state_cls = None
    if fisher:
        W_ref, b_ref = lm.fisher_reference(data, labels, taus, lm.TEMP)
        sW = np.sqrt(np.mean(W_ref**2, axis=1))                  # reference magnitudes for the hidden lr
        sb = np.maximum(np.abs(b_ref), lm.TEMP)
        log(logf, f"Fisher reference: |W*| {np.linalg.norm(W_ref[0]):.4g} (t=5) -> {np.linalg.norm(W_ref[-1]):.4g} (t=0); "
                  f"b* {b_ref[0]:.4g} -> {b_ref[-1]:.4g}; rms W* per pixel {sW[0]:.3g} -> {sW[-1]:.3g}")
    elif K > 0 and args.hidden == "potts":
        km_seed = 100 * (args.seed - lm.KEY_SEED)
        pooled = args.proto_init in ("pooled", "random")       # random: the reference only sets lr scales/diagnostics
        W_ref, b_ref, state_cls = lm.prototype_reference(data, labels, taus, lm.TEMP, K, seed=km_seed, pooled=pooled)
        if not (np.isfinite(W_ref).all() and np.isfinite(b_ref).all()):
            raise RuntimeError(f"prototype reference is not finite for --seed {args.seed} (k-means seeds {km_seed} + class)")
        # The hidden learning-rate scales always come from the default-seed reference (k-means seed 0), so that
        # --seed changes the initial prototypes and the key chain but not the step sizes (rms b* is sensitive to the
        # clustering: at t = 0 it is 0.59 for k-means seed 0 and 1.5 for seed 100 with 128 states at 14x14).
        W_lr, b_lr = (W_ref, b_ref) if km_seed == 0 else \
            lm.prototype_reference(data, labels, taus, lm.TEMP, K, seed=0, pooled=pooled)[:2]
        sW = np.sqrt(np.mean(W_lr**2, axis=(1, 2)))
        sb = np.maximum(np.sqrt(np.mean(b_lr**2, axis=1)), lm.TEMP)
        rb = np.maximum(np.sqrt(np.mean(b_ref**2, axis=1)), lm.TEMP)
        rW = np.sqrt(np.mean(W_ref**2, axis=(1, 2)))
        log(logf, f"prototype reference ({K} states, " + ("pooled k-means without labels, seed " + str(km_seed) if pooled
                  else f"{K // n_cls} per class, k-means seeds {km_seed} + class") + "): |W*| "
                  f"{np.linalg.norm(W_ref[0]):.4g} (t=5) -> {np.linalg.norm(W_ref[-1]):.4g} (t=0); rms b* {rb[0]:.3g} -> "
                  f"{rb[-1]:.3g}; rms W* per entry {rW[0]:.3g} -> {rW[-1]:.3g}; hidden lr scales (k-means seed 0): "
                  f"b {sb[0]:.3g} -> {sb[-1]:.3g}, W {sW[0]:.3g} -> {sW[-1]:.3g}"
                  + ("; initial W random (rms of W* at t = 5), b = 0" if args.proto_init == "random" else ""))
        np.save(os.path.join(out_dir, "state_class.npy"), state_cls)
    elif K > 0 and args.hidden == "ising":       # K Ising units, one per node of a split tree (latent_model.split_reference)
        km_seed = 100 * (args.seed - lm.KEY_SEED)
        pooled = args.proto_init in ("pooled", "random")       # random: the reference only sets lr scales/diagnostics
        W_ref, b_ref, tree = lm.split_reference(data, labels, taus, lm.TEMP, K, seed=km_seed, pooled=pooled,
                                               n_min=lm.tree_n_min(len(data), K))
        if not (np.isfinite(W_ref).all() and np.isfinite(b_ref).all()):
            raise RuntimeError(f"split-tree reference is not finite for --seed {args.seed} (k-means seed {km_seed})")
        # As for Potts units, the hidden learning-rate scales always come from the k-means seed 0 tree, so that --seed
        # changes the tree, the start and the key chain but not the step sizes.
        W_lr, b_lr = (W_ref, b_ref) if km_seed == 0 else \
            lm.split_reference(data, labels, taus, lm.TEMP, K, seed=0, pooled=pooled,
                               n_min=lm.tree_n_min(len(data), K))[:2]
        sW = np.sqrt(np.mean(W_lr**2, axis=(1, 2)))
        sb = np.maximum(np.sqrt(np.mean(b_lr**2, axis=1)), lm.TEMP)
        ru = np.sqrt(np.mean(W_lr**2, axis=2))                 # rms per unit (slices, K)
        rb = np.sqrt(np.mean(b_ref**2, axis=1))
        sizes = np.bincount(tree["leaf_of"], minlength=K + 1)
        log(logf, f"split-tree reference ({K} Ising units; " + ("bisecting 2-means of all images without labels" if pooled
                  else f"class tree of the {n_cls} classes, then bisecting 2-means") + f", k-means seed {km_seed}; "
                  f"depth <= {int(tree['depth'].max())}, leaf sizes {int(sizes.min())}-{int(sizes.max())}, root purity "
                  f"{tree['unit_purity'][0]:.3f}): |W*| {np.linalg.norm(W_ref[0]):.4g} (t=5) -> "
                  f"{np.linalg.norm(W_ref[-1]):.4g} (t=0); rms b* {rb[0]:.3g} -> {rb[-1]:.3g}; hidden lr scales "
                  f"(k-means seed 0): b {sb[0]:.3g} -> {sb[-1]:.3g}, W {sW[0]:.3g} -> {sW[-1]:.3g}; rms W per unit "
                  f"{ru[0].min():.3g}-{ru[0].max():.3g} (t=5), {ru[-1].min():.3g}-{ru[-1].max():.3g} (t=0)"
                  + ("; initial W random Haar-orthogonal rows (rms row norm of W* at t = 5), b = 0"
                     if args.proto_init == "random" else ""))
        save_atomic(os.path.join(out_dir, "split_tree.npz"), **tree)
    else:
        W_ref, b_ref = np.zeros((lm.N_TIME_STEPS, 0, n_vis)), np.zeros((lm.N_TIME_STEPS, 0))
        sW = sb = np.zeros(lm.N_TIME_STEPS)

    if args.local:
        thr = (0.5 - prob["mean"]) / prob["std"]               # pixel intensity 0.5 in normalised units
        wl_ref, bl_ref = lm.local_reference(data, taus, lm.TEMP, thr)
        used = np.abs(wl_ref[-1]) > 0
        swl = np.sqrt(np.mean(wl_ref[:, used] ** 2, axis=1))
        sbl = np.maximum(np.sqrt(np.mean(bl_ref[:, used] ** 2, axis=1)), lm.TEMP)
        log(logf, f"local units: {int(used.sum())} of {n_vis} pixels have both modes; rms w* {swl[0]:.3g} (t=5) -> "
                  f"{swl[-1]:.3g} (t=0); rms c* {sbl[0]:.3g} -> {sbl[-1]:.3g}")

    # Paper checkpoint, for a per-slice comparison of the validation loss (paper problem, joint data only)
    paper = None
    if args.problem == "mnist28" and args.cls == "none" and os.path.exists(lm.PAPER_CKPT):
        paper = np.load(lm.PAPER_CKPT, mmap_mode="r")
        if paper.shape[1] != vis.n_theta:                    # another coupling radius: no per-slice comparison
            paper = None

    # ---- --init-from: link the first slices of another run (same model), then resume after them ---------------
    if args.init_from and args.init_slices > 0 and not os.path.exists(os.path.join(slice_dir, "slice_000.npz")):
        src = os.path.join(args.out_root, args.init_from)
        with open(os.path.join(src, "config.json")) as f:
            scfg = json.load(f)
        free = {"name", "steps_scale", "val_every", "lr_scale", "init_from", "init_slices", "max_slices",
                "visible_impl", "precision"}         # numerics of the new slices only: a continuation may differ
        bad = {k: (scfg.get(k, defaults.get(k)), v) for k, v in config.items()
               if k not in free and k in defaults and scfg.get(k, defaults.get(k)) != v}
        bad.update({k: (scfg.get(k), config[k]) for k in ("K", "cls", "hidden_lr_scale", "freeze_hidden_slices")
                    if scfg.get(k) != config[k]})
        if bad:
            raise RuntimeError(f"--init-from {src}: different model config {bad}")
        if scfg.get("init_from"):
            raise RuntimeError(f"--init-from {src}: that run is itself initialised from another; use the original")
        recs = {}
        with open(os.path.join(src, "train_log.jsonl")) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "slice" in r:
                        recs[int(r["slice"])] = r
        for k in range(args.init_slices):
            s_src = os.path.join(src, "slices", f"slice_{k:03d}.npz")
            if not os.path.exists(s_src) or k not in recs:
                raise RuntimeError(f"--init-from {src}: slice {k} missing")
        for k in range(args.init_slices):             # relative links: valid on the host and in the container
            os.symlink(os.path.relpath(os.path.join(src, "slices", f"slice_{k:03d}.npz"), slice_dir),
                       os.path.join(slice_dir, f"slice_{k:03d}.npz"))
        with open(os.path.join(out_dir, "train_log.jsonl"), "a") as f:
            for k in range(args.init_slices):
                f.write(json.dumps(dict(recs[k], copied_from=args.init_from)) + "\n")
        log(logf, f"--init-from: linked slices 0-{args.init_slices - 1} of {src}; training resumes at slice "
                  f"{args.init_slices}")

    # ---- resume -----------------------------------------------------------------------------------------------
    done = 0
    while os.path.exists(os.path.join(slice_dir, f"slice_{done:03d}.npz")):
        done += 1
    optimization_key, _reverse_key, _noise_key = jr.split(jr.PRNGKey(args.seed), 3)   # the paper's master split (seed 1)
    if done == 0:
        if K == 0:
            W0, b0 = np.zeros((0, n_vis)), np.zeros((0,))
        elif fisher:
            W0, b0 = W_ref[0:1], b_ref[0:1]
        elif args.hidden == "ising" and args.proto_init == "random":   # Haar-orthogonal rows of the reference's size
            rng0 = np.random.default_rng(1000 + args.seed)
            r0 = float(np.sqrt(np.mean(np.sum(W_ref[0] ** 2, axis=1))))   # rms row norm of the tree reference at t = 5
            W0 = lm.random_orthogonal_rows(K, n_vis, r0, rng0)
            b0 = np.zeros_like(b_ref[0])
        elif args.proto_init == "random":         # no prototypes: noise of the reference's size, no fields
            rng0 = np.random.default_rng(1000 + args.seed)
            W0 = rng0.normal(0.0, float(np.sqrt(np.mean(W_ref[0] ** 2))), size=W_ref[0].shape)
            b0 = np.zeros_like(b_ref[0])
        else:
            W0, b0 = W_ref[0], b_ref[0]
        params = {"theta": theta0, "W": jnp.asarray(W0), "b": jnp.asarray(b0)}
        if args.quad and K > 0:
            params["Q"] = jnp.zeros_like(params["W"])
        if args.local:
            params.update(wl=jnp.asarray(wl_ref[0]), bl=jnp.asarray(bl_ref[0]))
    else:
        prev = np.load(os.path.join(slice_dir, f"slice_{done - 1:03d}.npz"))
        params = {"theta": jnp.asarray(prev["theta"]), "W": jnp.asarray(prev["W"]), "b": jnp.asarray(prev["b"])}
        if args.quad and K > 0:
            params["Q"] = jnp.asarray(prev["Q"])
        if args.local:
            params.update(wl=jnp.asarray(prev["wl"]), bl=jnp.asarray(prev["bl"]))
        optimization_key = jnp.asarray(prev["key"])
        where = f"t = {taus[done]:.4f}" if done < n_slices else "all slices present; assembling"
        log(logf, f"resuming at slice {done} ({where}) from {slice_dir}")

    # ---- slices -----------------------------------------------------------------------------------------------
    for k in range(done, n_slices):
        W_prev = np.asarray(params["W"]) if tree is not None else None    # the previous slice's units (diagnostics)
        t0 = time.time()
        t_curr = float(taus[k])
        optimization_key, optimization_subkey = jr.split(optimization_key)
        n_steps = max(1, int(round(lm.paper_steps(k) * args.steps_scale)))
        if args.hidden_fixed:                    # hidden couplings = this slice's closed-form reference, not trained
            params = dict(params)
            if K > 0:
                one = fisher
                params["W"] = jnp.asarray(W_ref[k:k + 1] if one else W_ref[k])
                params["b"] = jnp.asarray(b_ref[k:k + 1] if one else b_ref[k])
            if args.local:
                params["wl"], params["bl"] = jnp.asarray(wl_ref[k]), jnp.asarray(bl_ref[k])
        train_hidden = K > 0 and k >= args.freeze_hidden_slices and not args.hidden_fixed
        lrs = {"theta": float(lr_v[k]),
               "W": float(args.hidden_lr_scale * sW[k]) if train_hidden else 0.0,
               "b": float(args.hidden_lr_scale * sb[k]) if train_hidden else 0.0}
        if "Q" in params:
            lrs["Q"] = float(args.hidden_lr_scale * sW[k]) if train_hidden else 0.0
        if args.local:
            on = k >= args.freeze_hidden_slices and not args.hidden_fixed
            lrs.update(wl=float(args.hidden_lr_scale * swl[k]) if on else 0.0,
                       bl=float(args.hidden_lr_scale * sbl[k]) if on else 0.0)
        lrs = {g: v * args.lr_scale for g, v in lrs.items()}
        lr_log = {"W": lrs["W"], "b": lrs["b"]}       # scalars for the log (--ising-weak-boost makes lrs per unit)
        n_boost = None
        if tree is not None and train_hidden and (args.ising_reseed or args.ising_weak_boost != 1.0):
            # --ising-reseed / --ising-weak-boost: units whose field hardly varies over this slice's data (std a_j <
            # 0.1 T on a batch of p_tau, the training log's "weak") are reset to this slice's split-tree reference
            # (Q_j = 0) and/or get a larger step size for this slice.
            Xs = np.asarray(sample_forward_process(t_curr, lm.BATCH_SIZE, lm.TEMP, lm.SIGMA_FORWARD, data_train,
                                                   jr.fold_in(jr.PRNGKey(VAL_SEED + 2), k), beta=1))
            def weak_units(p):
                A = lm.ising_fields(np.asarray(p["W"]), np.asarray(p["b"]),
                                    np.asarray(p["Q"]) if "Q" in p else None, Xs)
                return A.std(axis=0) < 0.1 * lm.TEMP
            weak = weak_units(params)
            n_weak0, n_weak1 = int(weak.sum()), int(weak.sum())
            if args.ising_reseed and weak.any():
                params = dict(params)
                params["W"] = jnp.where(jnp.asarray(weak)[:, None], jnp.asarray(W_ref[k]), params["W"])
                params["b"] = jnp.where(jnp.asarray(weak), jnp.asarray(b_ref[k]), params["b"])
                if "Q" in params:
                    params["Q"] = jnp.where(jnp.asarray(weak)[:, None], 0.0, params["Q"])
                weak = weak_units(params)            # still weak after the reset (the reference can be weak too)
                n_weak1 = int(weak.sum())
            if args.ising_weak_boost != 1.0:         # per-unit step sizes (always arrays, so the runner traces once)
                g = 1.0 + (args.ising_weak_boost - 1.0) * weak.astype(np.float64)
                for name in ("W", "Q"):
                    if name in lrs:
                        lrs[name] = jnp.asarray(lrs[name] * g[:, None])
                lrs["b"] = jnp.asarray(lrs["b"] * g)
                n_boost = int(weak.sum())
            log(logf, f"slice {k}: {n_weak0} weak units at the start"
                      + (f", {n_weak1} still weak after reseeding" if args.ising_reseed else "")
                      + (f"; step size x{args.ising_weak_boost:g} for {int(weak.sum())}"
                         if args.ising_weak_boost != 1.0 else ""))
        dt_k = 0.0                                # CD-1 release length of this slice (unused with exact gradients)
        if cd1_grad is not None:
            if args.cd1_dt > 0:
                dt_k = float(args.cd1_dt)
            else:                                 # from the slice's starting parameters, on a batch of its own
                Xs = sample_forward_process(t_curr, lm.BATCH_SIZE, lm.TEMP, lm.SIGMA_FORWARD, data_train,
                                            jr.fold_in(jr.PRNGKey(VAL_SEED + 1), k), beta=1)
                dt_k = float(args.cd1_kappa ** 2 * lm.TEMP / field_scale(params, Xs))
        carry = init_carry(params, adam, optimization_subkey)
        losses = []
        run_k = run_late if run_late is not None and k >= args.hidden_samples_switch else run
        n_full, rem = divmod(n_steps, 100)
        curve_steps, curve_ps = [], []
        if args.val_every:                      # learning curve on the slice's fixed validation batch (as below)
            kc_idx, kc_noise = jr.split(jr.fold_in(jr.PRNGKey(VAL_SEED), k))
            src_c = data if args.cls == "none" else data_train
            idx_c = np.asarray(jr.choice(kc_idx, src_c.shape[0], shape=(VAL_N,), replace=True))
            e_c = np.exp(-t_curr)
            Xc = e_c * src_c[idx_c] + np.sqrt(lm.TEMP * (1.0 - e_c * e_c)) * jr.normal(kc_noise, (VAL_N, n_vis))
            curve_steps.append(0)
            curve_ps.append(np.asarray(loss_ps(carry[4] if args.select == "best" else carry[7], Xc), np.float32))
        for i_chunk in range(n_full):
            carry, l = run_k(carry, t_curr, lrs, 100, dt_k)
            losses.append(l)
            if args.val_every and ((i_chunk + 1) * 100) % args.val_every == 0:
                curve_steps.append((i_chunk + 1) * 100)
                curve_ps.append(np.asarray(loss_ps(carry[4] if args.select == "best" else carry[7], Xc), np.float32))
        if rem:
            carry, l = run_k(carry, t_curr, lrs, rem, dt_k)
            losses.append(l)
        losses = np.concatenate([np.asarray(l) for l in losses])
        params = carry[4] if args.select == "best" else carry[7]
        best_loss, best_epoch = float(carry[5]), int(carry[6])
        t_train = time.time() - t0
        if not np.isfinite(best_loss) or not all(np.all(np.isfinite(np.asarray(params[g]))) for g in params):
            log(logf, f"slice {k}: non-finite loss or parameters; stopping without saving the slice "
                      f"(a resubmission retries it)")
            sys.exit(3)

        # ---- validation and diagnostics on a fixed batch (same for every run on the same data) -------------
        kv_idx, kv_noise = jr.split(jr.fold_in(jr.PRNGKey(VAL_SEED), k))
        src = data if args.cls == "none" else data_train
        src_labels = labels if args.cls == "none" else labels_train
        idx = np.asarray(jr.choice(kv_idx, src.shape[0], shape=(VAL_N,), replace=True))
        e = np.exp(-t_curr)
        Xv = e * src[idx] + np.sqrt(lm.TEMP * (1.0 - e * e)) * jr.normal(kv_noise, (VAL_N, n_vis))
        yv = src_labels[idx]
        val_ps = np.asarray(loss_ps(params, Xv))
        val = float(np.mean(val_ps))
        rec = {"slice": k, "t": t_curr, "steps": n_steps, "lr_v": lrs["theta"], "lr_W": lr_log["W"], "lr_b": lr_log["b"],
               **({"n_weak_boosted": n_boost} if n_boost is not None else {}),
               **({"cd1_dt": dt_k} if cd1_grad is not None else {}),
               **({"hidden_samples": args.hidden_samples_late if run_k is run_late else args.hidden_samples}
                  if run_late is not None else {}),
               "first_loss": float(losses[0]), "last1000_mean": float(np.mean(losses[-1000:])),
               "best_loss": best_loss, "best_epoch": best_epoch, "val_loss": val, "val_median": float(np.median(val_ps)),
               "select": args.select, "seconds": t_train, "steps_per_s": n_steps / t_train,
               "job": os.environ.get("SLURM_JOB_ID", "manual"), "code_sha256": code_sha}
        if args.val_every:                      # learning curve: add the final slice result, save per-sample losses
            if curve_steps[-1] != n_steps:
                curve_steps.append(n_steps)
                curve_ps.append(val_ps.astype(np.float32))
            os.makedirs(os.path.join(out_dir, "val_curves"), exist_ok=True)
            np.savez(os.path.join(out_dir, "val_curves", f"slice_{k:03d}.npz"), steps=np.asarray(curve_steps),
                     val_per_sample=np.stack(curve_ps), t=t_curr)
            cm = [float(np.mean(v)) for v in curve_ps]
            half = int(np.searchsorted(curve_steps, n_steps // 2))      # improvement over the second half of the slice
            dd = curve_ps[-1].astype(np.float64) - curve_ps[half].astype(np.float64)
            rec.update(val_curve_steps=[int(s) for s in curve_steps], val_curve_mean=cm,
                       val_gain_second_half=float(np.mean(dd)), val_gain_second_half_se=float(np.std(dd) / np.sqrt(dd.size)))
        extra = {}
        if paper is not None:
            pp = {"theta": jnp.asarray(np.array(paper[k], dtype=np.float64)), "W": jnp.zeros((0, n_vis)), "b": jnp.zeros((0,))}
            val_ps_paper = np.asarray(loss_ps_k0(pp, Xv))
            d = val_ps - val_ps_paper
            rec.update({"val_loss_paper": float(np.mean(val_ps_paper)), "val_diff_paper": float(np.mean(d)),
                        "val_diff_paper_se": float(np.std(d) / np.sqrt(d.size)),
                        "val_diff_paper_median": float(np.median(d))})
            extra["val_per_sample_paper"] = val_ps_paper.astype(np.float32)
        if K > 0 and args.hidden == "potts":
            W, b = np.asarray(params["W"]), np.asarray(params["b"])
            A = np.asarray(Xv) @ W.T + b
            if "Q" in params:                    # the spring term of the input field (missing before 2026-11-08 09:15:
                A = A + np.asarray(Xv) ** 2 @ np.asarray(params["Q"]).T   # earlier logs of --quad runs omit it)
            Z = A / lm.TEMP
            p = np.exp(Z - Z.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            pbar = p.mean(axis=0)
            rec.update({
                "W_norm": float(np.linalg.norm(W)), "Wref_norm": float(np.linalg.norm(W_ref[k])),
                "cos_W_Wref": float(np.sum(W * W_ref[k]) / (np.linalg.norm(W) * np.linalg.norm(W_ref[k]) + 1e-300)),
                "class_acc": float(np.mean(state_cls[np.argmax(p, axis=1)] == yv)),
                "mean_max_p": float(np.mean(p.max(axis=1))),
                "eff_states": float(np.exp(-np.sum(pbar * np.log(pbar + 1e-300)))),
                "states_below_1e-6": int(np.sum(pbar < 1e-6)),   # mean occupation: (nearly) lost states
            })
            if args.hidden_samples > 0:          # states below about one visit per step at this slice's draw count
                s_k = args.hidden_samples_late if run_k is run_late else args.hidden_samples
                rec["states_below_1_visit"] = int(np.sum(pbar < 1.0 / (lm.BATCH_SIZE * s_k * args.hidden_replicates)))
        elif K > 0 and fisher:
            W, b = np.asarray(params["W"]), np.asarray(params["b"])
            A = np.asarray(Xv) @ W.T + b
            if "Q" in params:
                A = A + np.asarray(Xv) ** 2 @ np.asarray(params["Q"]).T
            m = np.tanh(A / lm.TEMP)
            j = 0
            rec.update({
                "W_norm": float(np.linalg.norm(W[j])), "Wref_norm": float(np.linalg.norm(W_ref[k])),
                "cos_W_Wref": float(W[j] @ W_ref[k] / (np.linalg.norm(W[j]) * np.linalg.norm(W_ref[k]) + 1e-300)),
                "b": float(b[j]), "b_ref": float(b_ref[k]),
                "class_acc": float(np.mean((A[:, j] > 0) == (yv == 1))),
                "mean_abs_m": float(np.mean(np.abs(m[:, j]))), "mean_v": float(np.mean(1.0 - m[:, j] ** 2)),
                "softening_over_T": float(np.mean(1.0 - m[:, j] ** 2) * np.sum(W[j] ** 2) / lm.TEMP),
            })
        elif K > 0:                              # K Ising units of a split tree (latent_model.ising_unit_stats)
            stats, arrays = lm.ising_unit_stats(
                np.asarray(params["W"]), np.asarray(params["b"]), np.asarray(params["Q"]) if "Q" in params else None,
                np.asarray(Xv), lm.TEMP, W_ref=W_ref[k], W_prev=W_prev, tree=tree, leaf=tree["leaf_of"][idx],
                labels=yv)
            rec.update(stats)
            extra.update(arrays)
        if "Q" in params:
            Qn = np.asarray(params["Q"])
            rec["Q_norm"] = float(np.linalg.norm(Qn))
            extra["Q"] = Qn
        if args.local:
            wl, bl = np.asarray(params["wl"]), np.asarray(params["bl"])
            ml = np.tanh((np.asarray(Xv) * wl + bl) / lm.TEMP)[:, used]
            rec.update({"wl_norm": float(np.linalg.norm(wl)), "wl_ref_norm": float(np.linalg.norm(wl_ref[k])),
                        "cos_wl": float(wl @ wl_ref[k] / (np.linalg.norm(wl) * np.linalg.norm(wl_ref[k]) + 1e-300)),
                        "local_decided": float(np.mean(np.abs(ml) > 0.9))})
            extra.update(wl=wl, bl=bl)
        save_atomic(os.path.join(slice_dir, f"slice_{k:03d}.npz"),
                    theta=np.asarray(params["theta"]), W=np.asarray(params["W"]), b=np.asarray(params["b"]),
                    key=np.asarray(optimization_key), loss_trace=losses[::10].astype(np.float32),
                    val_per_sample=val_ps.astype(np.float32), val_idx=idx.astype(np.int32), record=json.dumps(rec),
                    **extra)
        with open(os.path.join(out_dir, "train_log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        msg = (f"slice {k:3d} t {t_curr:.4f}: {n_steps} steps in {t_train:.0f} s ({rec['steps_per_s']:.0f}/s); "
               f"loss first {rec['first_loss']:.4g}, last-1000 mean {rec['last1000_mean']:.6g}, best {best_loss:.6g} "
               f"@ {best_epoch}; val {val:.6g}")
        if args.val_every:
            msg += (f"; val change over the 2nd half of the slice {rec['val_gain_second_half']:.4g} "
                    f"+- {rec['val_gain_second_half_se']:.2g}")
        if "val_loss_paper" in rec:
            msg += f" (paper {rec['val_loss_paper']:.6g}; diff {rec['val_diff_paper']:.4g} +- {rec['val_diff_paper_se']:.2g})"
        if K > 0 and args.hidden == "potts":
            msg += (f"; |W| {rec['W_norm']:.4g} (ref {rec['Wref_norm']:.4g}), cos {rec['cos_W_Wref']:.3f}, "
                    f"class acc of top state {rec['class_acc']:.3f}, <max p> {rec['mean_max_p']:.3f}, "
                    f"effective states {rec['eff_states']:.2f}")
        elif K > 0 and fisher:
            msg += (f"; |W| {rec['W_norm']:.4g} (ref {rec['Wref_norm']:.4g}), cos {rec['cos_W_Wref']:.3f}, "
                    f"b {rec['b']:.4g} (ref {rec['b_ref']:.4g}), acc {rec['class_acc']:.3f}, <|m|> {rec['mean_abs_m']:.3f}, "
                    f"softening/T {rec['softening_over_T']:.3g}")
        elif K > 0:
            st = rec["straddle_off"]
            msg += (f"; |W| {rec['W_norm']:.4g} (ref {rec['Wref_norm']:.4g}), overlap {rec['overlap_ref']:.3f}, "
                    f"dead/weak/decided {rec['n_dead']}/{rec['n_weak']}/{rec['n_decided']}, max cos "
                    f"{rec['max_pair_cos']:.3f}, unit-0 acc {rec['class_acc']:.3f}, leaf acc {rec['leaf_acc']:.3f}, "
                    f"<|m|> {rec['mean_abs_m']:.3f}, softening/T {rec['softening_over_T']:.3g}, straddle "
                    + ("n/a" if st is None else f"{st:.3f}"))
        if "Q_norm" in rec:
            msg += f"; |Q| {rec['Q_norm']:.4g}"
        if args.local:
            msg += (f"; local |w| {rec['wl_norm']:.4g} (ref {rec['wl_ref_norm']:.4g}), cos {rec['cos_wl']:.3f}, "
                    f"decided {rec['local_decided']:.3f}")
        log(logf, msg)

    # ---- assemble ---------------------------------------------------------------------------------------------
    if n_slices == lm.N_TIME_STEPS:
        thetas, Ws, bs, wls, bls, Qs = [], [], [], [], [], []
        for k in range(n_slices):
            s = np.load(os.path.join(slice_dir, f"slice_{k:03d}.npz"))
            thetas.append(s["theta"])
            Ws.append(s["W"])
            bs.append(s["b"])
            if args.local:
                wls.append(s["wl"])
                bls.append(s["bl"])
            if args.quad and K > 0:
                Qs.append(s["Q"])
        tmp = os.path.join(out_dir, "params_history.tmp.npy")
        np.save(tmp, np.stack(thetas))
        os.replace(tmp, os.path.join(out_dir, "params_history.npy"))
        tmp = os.path.join(out_dir, "hidden_history.tmp.npz")
        loc = {"wl": np.stack(wls), "bl": np.stack(bls)} if args.local else {}
        if Qs:
            loc["Q"] = np.stack(Qs)
        np.savez(tmp, W=np.stack(Ws), b=np.stack(bs), t=taus, **loc)
        os.replace(tmp, os.path.join(out_dir, "hidden_history.npz"))
        with open(os.path.join(out_dir, "DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        log(logf, f"assembled params_history.npy {np.stack(thetas).shape} and hidden_history.npz; DONE")


if __name__ == "__main__":
    main()
