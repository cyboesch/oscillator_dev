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
Option --time-grid power:P: slice times t_k = 5 (1 - k/99)^P instead of the paper's t_k = 5 (1 - k/99) (P > 1: knots
denser near t = 0; latent_model.parse_time_grid), each slice's learning rate and step count the paper's linear
functions of t at t_k (latent_model.time_grid_lr, time_grid_steps; --steps-scale, --steps-schedule and --lr-scale
act on top). Everything per slice follows the grid: the data, the validation batch and the hidden-unit references
at t_k, the logs, hidden_history.npz 't', and the paper comparison (the paper's slices interpolated in t). Recorded in
config.json as 'time_grid' ('uniform' by default and for runs without the key): a run resumes only on its own grid,
--init-from takes a source run on another grid only if the linked slices sit at identical times (e.g. slices 0..K of
a uniform run for power:P@K, init slices <= K + 1), and the samplers read the knots from it.
Option --harmonic R (2026-11-12): R harmonic hidden modes, linear oscillators integrated out exactly, add
F_h = -1/2 sum_j (v_j . x + c_j)^2 to the free energy (latent_model.harmonic_terms; parameters V (R x N) and c (R) in
the params dict, so Adam, the EMA and the best iterate handle them like every other group). They only add variance
along their v_j, so the model can carry the near-Gaussian covariance of the high-t slices directly instead of building
it from the spread of the Potts states. Reference (latent_model.harmonic_reference): the R strongest modes of the
Gaussian approximation N(e^-t mu, e^-2t C + T (1 - e^-2t) I) of each slice's forward distribution, mu and C the
training data's (sample_latent.data_stats; the cache out/latent_physical_SGM/data_stats/ shared with sample_latent.py
--init-dist gauss). Slice 0 starts from the reference at t_0 = 5 (held there, like the hidden units, during the first
--freeze-hidden-slices slices); every later slice warm-starts from the previous one. Learning rates: --hidden-lr-scale
times the reference's root-mean-square entry of V and of c (floored at T) at the slice (latent_model.harmonic_lr_scales),
times --harmonic-lr-scale. --harmonic-fixed: the modes are this slice's reference in every slice, not trained.
--harmonic-fixed-slices N: the reference in slices 0..N-1 only, trained from slice N on (warm-started from slice N-1's
reference); below t of about 3 the reference's anti-springs exceed a trained run's visible springs (harmonic_analysis.py,
job 4230722), so fixing them everywhere asks the springs to grow by up to T / iso_0 = 24 at t = 0 (mnist28_d0123).
--harmonic-ref within (opt-in; default 'full', the above): the reference is built from the shared component
covariance S_w of the Potts prototype init (latent_model.prototype_reference(..., return_stats=True): the part of C
the prototypes do not carry) instead of from C. The full reference's modes also carry the prototypes' between-cluster
spread, which the Potts unit carries already (along PC 1 at t = 5 the plain seed-1 run's Potts layer supplies 93% of
the full reference's anti-spring; harmonic_analysis.py), so slice 0's start and every fixed slice count it twice;
with 'within' the prototype Potts unit plus the modes plus isotropic springs is exactly the prototypes' Gaussian
mixture. Its spectrum is saved in harmonic_within.npz at the first start and reused on resume. --harmonic-lr-tmin t0
(opt-in; default 0, off): below t0 the modes' learning-rate scales keep their value at t0 (the reference's isotropic
spring T / iso_t rises from 1.15 at t = 1 to 24.5 at t = 0, far above a trained run's springs, so the reference's
scale overstates the modes' size there).
Diagnostics per slice (record keys harm_*: norms, cosines and added variance against the reference, the curvature of
the visible springs minus the modes at x = 0, the modes beyond the springs along their own directions;
latent_model.harmonic_unit_stats); V and c are saved in every slice file and in hidden_history.npz (members V, c after
all others). Recorded in config.json (harmonic, harmonic_fixed, harmonic_lr_scale, harmonic_fixed_slices,
harmonic_ref, harmonic_lr_tmin; 0, False, 1.0, 0, 'full', 0.0 for runs from before); the reference's data statistics
in harmonic_reference.json. A run resumes only with the same values (strict, as every option); --init-from takes only a
source with the same modes (harmonic, harmonic_fixed, harmonic_fixed_slices, harmonic_ref; harmonic_lr_scale and
harmonic_lr_tmin are free, they act on the new slices only). Needs exact hidden statistics and the exact rule (not
with --hidden-samples, --hidden-rule cd1 or --hidden-norm other than element).
Option --patch SIDE:STRIDE:STATES (2026-11-13; e.g. 7:3:32): per-patch Potts hidden units. P = ((G - SIDE)/STRIDE + 1)^2
units with STATES states each, unit p coupled only to the SIDE x SIDE pixels x_p of its patch (fields a_ps = bp_ps +
Wp_ps . x_p, plus Qp_ps . x_p^2 with --quad, Qp starting at 0), integrated out exactly: F_patch = -T sum_p log sum_s
exp(a_ps / T) (latent_model.patch_terms; parameters Wp (P, S, d), bp (P, S), Qp in the params dict, added after every
other term, so Adam, the EMA and the best iterate handle them like every other group). Reference per patch
(latent_model.patch_clusters / patch_reference): prototype_reference's pooled branch on the patch vectors, k-means
seeds 20000 + 1000 (seed - 1) + p; written to patch_reference.json at the first start. Start: --patch-init-scale LAM
(default 0.25) times the reference at the start slice (a tempered expert: W = LAM (W* - pi W*), b = T log pi - LAM
W*.mu / 2, centred), held during --freeze-hidden-slices like every hidden unit. --patch-on-slice K (opt-in variant):
no patch units in slices 0..K-1 (exactly the plain computation; their slice files carry zero patch arrays), switched
on at slice K at LAM x the reference at t_K (LAM must then be given; 0.06 is the measured safe value at slice 31, job
4231616) and trained from slice K on. Learning rates per unit: --hidden-lr-scale x --patch-lr-scale x the
root-mean-square entry of that unit's scale-1 reference at the slice (W and Q; for b floored at T), always from the
seed-1 clustering (latent_model.patch_lr_scales), times --lr-scale. Diagnostics per slice (record keys patch_*: norms,
cosine to the reference, effective states, dead states, <max p>; force_rms_vis / _glob / _loc / _patch, the force
decomposition on the validation batch in float64; latent_model.patch_unit_stats); Wp, bp, Qp and patch_eff are saved in
every slice file, Wp, bp, Qp in hidden_history.npz after every other member. Recorded in config.json (patch,
patch_init_scale, patch_lr_scale, patch_on_slice; '', 0.25, 1.0, 0 for runs from before); a run resumes only with the
same values, --init-from takes only a source with the same patch, init scale and on-slice (patch_lr_scale is free).
Needs exact hidden statistics, the exact rule with per-element Adam, trained units and both classes (not with
--hidden-samples, --hidden-rule cd1, --hidden-norm other than element, --hidden-fixed or --cls).
Every slice file also stores the per-sample losses on the fixed validation batch (and the paper checkpoint's, for
mnist28), for paired comparisons between runs (the validation batch of slice k is drawn with the key of k at the
slice's t, so runs on different grids share it only where their slice times agree: slices 0 and 99 for power:P
against uniform, 0..K and 99 for power:P@K).

Writes out/latent_physical_SGM/<name>/ (never the paper's folder, never deletes anything); resumes from the last
finished slice. Run under Slurm: cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_train_latent_neuronic.sh
"""

import argparse
import fcntl
import hashlib
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
    ap.add_argument("--step-impl", choices=["default", "lean"], default="default",
                    help="lean: the same training steps with less overhead (build_runner track_best, theta_masks, "
                         "chunk_batches): under --select ema best_params is not updated (it is never read there); "
                         "each chunk's 100 batches are drawn before the scan from the same keys; the theta "
                         "projection and k_6 gradient zeroing are constant masks. The same arithmetic, but with "
                         "hidden units XLA compiles the changed step with other fusions (fused multiply-adds), so "
                         "results differ from default at rounding level (on the GPU bitwise equal for pixels-only "
                         "models; speed_check.py section 6). 2-11%% faster per step (K 784-6144)")
    ap.add_argument("--steps-schedule", default="",
                    help="step factors per slice range, 'k0:f0,k1:f1,...': slice k >= k_i (the last such entry) runs "
                         "paper_steps(k) x --steps-scale x f_i steps, rounded to the nearest 100 m + 1 (no "
                         "recompilation); slices before the first entry factor 1. With a schedule every slice is "
                         "rounded that way, so at --steps-scale != 1 the factor-1 slices change too (by up to 50 "
                         "steps, or up to at least 101). E.g. '30:0.5,70:1,80:2'. Empty (default): the paper's schedule, rounded as before")
    ap.add_argument("--time-grid", default="uniform",
                    help="slice times: 'uniform' (default: the paper's t_k = 5 (1 - k/99), k = 0..99) or 'power:P', "
                         "t_k = 5 (1 - k/99)^P with P > 0 (P > 1 puts the knots denser near t = 0: power:1.5 has the "
                         "last interval 0.0051 instead of 0.0505), or 'power:P@K': the uniform slices 0..K, then the "
                         "power law from t_K down to 0 (latent_model.parse_time_grid; e.g. power:1.5@40 keeps slices "
                         "0-40, t >= 2.98, as on the uniform grid). Each slice's learning rate and step count are the "
                         "paper's linear functions of t at t_k (2e-5 and 30,901 steps at t = 0, 2e-3 and 10,901 at "
                         "t = 5; latent_model.time_grid_lr / time_grid_steps), with --steps-scale, --steps-schedule "
                         "and --lr-scale on top. Recorded in config.json ('time_grid'; runs without it are uniform): "
                         "a run resumes only on its own grid, --init-from takes a source on another grid only if the "
                         "linked slices sit at identical times (e.g. a uniform run's slices 0..K for power:P@K), and "
                         "the samplers take the knots from it")
    ap.add_argument("--ref-kmeans", choices=["cpu", "gpu"], default="cpu",
                    help="Potts units: the k-means of the prototype reference (latent_model.prototype_reference) in "
                         "numpy (cpu, default) or on the GPU (latent_model.kmeans_jax: the same seeding draws and "
                         "iterations in float64, equal up to rounding; minutes -> seconds at K >= 784)")
    ap.add_argument("--split-dead", default="",
                    help="Potts units: at the start of the listed slices (comma-separated, e.g. '20,25,30,35,40'), "
                         "every state whose mean occupation on a batch of the slice's data is below 1e-6 (the "
                         "training log's states_below_1e-6 rule) becomes a copy of a busy state (split_dead_states): "
                         "donor j (busiest first, one dead state each) and the dead state get W_j +- eps v, Q_j and "
                         "b_j - T log 2 each, v the leading principal direction of the donor's posterior-weighted "
                         "data, eps = s T / sqrt(lambda_1) with s = --split-spread. The free energy changes only at "
                         "second order (by -T log cosh of a field difference of SD s), and the two copies can then "
                         "specialise. Recorded in config.json; in the --init-from free set. Default: none")
    ap.add_argument("--split-spread", type=float, default=0.5,
                    help="--split-dead: SD s of the field difference (in units of T) between the two copies over the "
                         "donor's data; 0 makes exact copies (which stay identical: no symmetry breaking)")
    ap.add_argument("--assemble", choices=["files", "memory"], default="files",
                    help="final params_history.npy / hidden_history.npz from this job's slices kept in memory "
                         "(memory; slices of an earlier job are read from their files) instead of re-reading every "
                         "slice file (files, default). The same file contents")
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
    ap.add_argument("--harmonic", type=int, default=0,
                    help="R > 0: add R harmonic hidden modes (linear oscillators integrated out exactly), F_h = -1/2 "
                         "sum_j (v_j . x + c_j)^2, parameters V (R x N) and c (R); slice 0 starts from the closed-form "
                         "reference (the R strongest modes of the Gaussian approximation of the forward distribution, "
                         "from the training data's mean and covariance; latent_model.harmonic_reference), later slices "
                         "warm-start. 0 (default): none, every path as before")
    ap.add_argument("--harmonic-fixed", action="store_true",
                    help="with --harmonic R: the modes are each slice's closed-form reference, not trained")
    ap.add_argument("--harmonic-fixed-slices", type=int, default=0,
                    help="with --harmonic R (not --harmonic-fixed): the modes are the closed-form reference in slices "
                         "0..N-1 only (t >= t_{N-1}, where p_t is nearly Gaussian) and trained from slice N on, "
                         "warm-started from slice N-1's reference; 0 (default): not used")
    ap.add_argument("--harmonic-lr-scale", type=float, default=1.0,
                    help="with --harmonic R: multiplies the modes' learning rates (--hidden-lr-scale x the reference's "
                         "root-mean-square entry of V and of c at the slice; latent_model.harmonic_lr_scales)")
    ap.add_argument("--harmonic-ref", choices=("full", "within"), default="full",
                    help="with --harmonic R: the closed-form reference of the modes (slice 0's start, the fixed slices, "
                         "the learning-rate scales and the diagnostics). 'full' (default): the R strongest modes of "
                         "the data covariance C. 'within': those of the shared component covariance S_w of the Potts "
                         "prototype init (latent_model.prototype_reference: C minus the prototypes' spread), so that "
                         "the prototype Potts unit plus the modes plus isotropic springs is exactly the prototypes' "
                         "Gaussian mixture and the modes do not count the between-cluster covariance a second time; "
                         "needs --hidden potts with K >= 2, --proto-init pooled or perclass, --cls none")
    ap.add_argument("--harmonic-lr-tmin", type=float, default=0.0,
                    help="with --harmonic R: below forward time t0 the modes' learning-rate scales stay at their value at "
                         "t0 (latent_model.harmonic_lr_scales at max(t_k, t0)): the reference's isotropic spring T / iso_t "
                         "grows from 1.15 at t = 1 to 24.5 at t = 0 (mnist28_d0123's pixel noise), far above a trained "
                         "run's springs, so its scale overstates the modes' there. 0 (default): no floor")
    ap.add_argument("--patch", default="",
                    help="per-patch Potts hidden units SIDE:STRIDE:STATES, e.g. 7:3:32: one unit with STATES states per "
                         "SIDE x SIDE patch of a grid of overlapping patches with stride STRIDE, coupled only to its "
                         "patch (Wp, bp, plus Qp with --quad), integrated out exactly (latent_model.patch_terms); start "
                         "--patch-init-scale x the per-patch prototype reference. '' (default): none, every path as "
                         "before")
    ap.add_argument("--patch-init-scale", type=float, default=None,
                    help="with --patch: the initial templates are LAM x the per-patch reference at the start slice "
                         "(default 0.25; must be given with --patch-on-slice > 0, where 0.06 is the measured safe value "
                         "at slice 31)")
    ap.add_argument("--patch-lr-scale", type=float, default=1.0,
                    help="with --patch: multiplies the per-unit learning rates (--hidden-lr-scale x the rms of each "
                         "unit's scale-1 reference at the slice; latent_model.patch_lr_scales)")
    ap.add_argument("--patch-on-slice", type=int, default=0,
                    help="with --patch: 0 (default) the units take part from slice 0 (held during "
                         "--freeze-hidden-slices); K > 0: no patch units in slices 0..K-1 (exactly the plain "
                         "computation), switched on at slice K at --patch-init-scale x the reference at t_K and trained "
                         "from slice K on")
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


def parse_steps_schedule(spec):
    """--steps-schedule 'k0:f0,k1:f1,...' (first slice : step factor; slices ascending in 0..99, factors > 0) ->
    tuple of (k_i, f_i); '' -> () (the paper's schedule, as before). Slice k gets the factor f_i of the last entry
    with k_i <= k, and factor 1 before the first entry; e.g. '30:0.5,70:1,80:2' halves slices 30-69 and doubles
    80-99."""
    spec = (spec or "").strip()
    if not spec:
        return ()
    out = []
    for part in spec.split(","):
        try:
            k, f = part.split(":")
            k, f = int(k), float(f)
        except ValueError:
            raise ValueError(f"--steps-schedule {spec!r}: entries must be 'slice:factor', e.g. '30:0.5,80:2'") from None
        if not (0 <= k < lm.N_TIME_STEPS and np.isfinite(f) and f > 0) or (out and k <= out[-1][0]):
            raise ValueError(f"--steps-schedule {spec!r}: slices must ascend within 0..{lm.N_TIME_STEPS - 1} and "
                             f"factors must be > 0")
        out.append((k, f))
    return tuple(out)


def steps_schedule_str(schedule):
    """Canonical form of a parsed --steps-schedule (as recorded in config.json; '' = none)."""
    return ",".join(f"{k}:{f!r}" for k, f in schedule)


def steps_factor(schedule, k):
    """Step factor of slice k under a parsed --steps-schedule (1 without one)."""
    f = 1.0
    for k0, fk in schedule:
        if k >= k0:
            f = fk
    return f


def slice_steps(k, steps_scale=1.0, schedule=(), grid="uniform"):
    """Adam steps of slice k. Without a schedule, as before: max(1, round(paper_steps(k) * steps_scale)). With one
    (--steps-schedule): paper_steps(k) * steps_scale * factor(k), rounded to the nearest count of the form 100 m + 1
    with m >= 1 (at least 101), the form of every paper_steps(k), so that every slice runs m chunks of 100 steps and
    one single step, the two lengths the jitted run is compiled for (each further remainder length costs one more
    compilation of a few seconds; e.g. --steps-scale 2 alone gives 2 x (100 m + 1) = 100 (2m) + 2 steps).
    grid (--time-grid): paper_steps(k) is that grid's, latent_model.time_grid_steps(k, grid) (the paper's rule at the
    slice's t; also of the form 100 m + 1); 'uniform' (default) is lm.paper_steps(k) itself."""
    base = lm.time_grid_steps(k, grid)            # 'uniform': lm.paper_steps(k), as before
    if not schedule:
        return max(1, int(round(base * steps_scale)))
    x = base * steps_scale * steps_factor(schedule, k)
    return 100 * max(1, int(round((x - 1.0) / 100.0))) + 1


def parse_split_slices(spec):
    """--split-dead 'k1,k2,...' -> sorted tuple of distinct slice indices in 1..99 ('' -> ())."""
    if not spec.strip():
        return ()
    try:
        ks = sorted({int(s) for s in spec.split(",") if s.strip()})
    except ValueError:
        raise ValueError(f"--split-dead {spec!r}: expected slice indices, e.g. '20,25,30'") from None
    if not ks or ks[0] < 1 or ks[-1] > lm.N_TIME_STEPS - 1:
        raise ValueError(f"--split-dead {spec!r}: slice indices must lie in 1..{lm.N_TIME_STEPS - 1}")
    return tuple(ks)


def split_dead_states(params, X, spread, temp=lm.TEMP, dead_below=1e-6, n_power=30):
    """--split-dead: re-seed the dead Potts states as copies of busy ones, keeping the free energy (nearly) unchanged.

    On the batch X (the slice's data): posteriors p(m | x) of the Potts fields a_m = W_m x + Q_m x^2 + b_m and the
    mean occupation pbar_m. Dead: pbar_m < dead_below. Donors: the living states, busiest first, one dead state each.
    For donor j and dead state d: v = the leading principal direction of the data weighted by p(j | x) (power
    iteration), lambda its variance, eps = spread T / sqrt(lambda); then
        W_j <- W_j + eps v,  W_d <- W_j - eps v,  Q_d <- Q_j,  b_j, b_d <- b_j - T log 2.
    The pair contributes exp(a_j / T) cosh(delta) with delta = eps v.x / T (SD `spread` over the donor's data) instead
    of exp(a_j / T): the free energy changes by -T log cosh(delta), second order in delta (exactly 0 at spread 0, where
    the copies are identical and stay so). Returns (new params, info)."""
    W = np.asarray(params["W"], dtype=np.float64)
    b = np.asarray(params["b"], dtype=np.float64)
    Q = np.asarray(params["Q"], dtype=np.float64) if "Q" in params else None
    X = np.asarray(X, dtype=np.float64)
    A = X @ W.T + b
    if Q is not None:
        A = A + (X * X) @ Q.T
    Z = A / temp
    P = np.exp(Z - Z.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    pbar = P.mean(axis=0)
    dead = np.flatnonzero(pbar < dead_below)
    alive = np.flatnonzero(pbar >= dead_below)
    n = int(min(dead.size, alive.size))
    info = {"dead": int(dead.size), "split": n}
    if n == 0:
        return params, info
    donors = alive[np.argsort(-pbar[alive], kind="stable")][:n]
    dead = dead[:n]
    W2, b2 = W.copy(), b.copy()
    Q2 = Q.copy() if Q is not None else None
    eps_all = []
    log2 = temp * np.log(2.0)
    for d, j in zip(dead, donors):
        w = P[:, j]
        sel = w > 1e-12 * w.max()                       # the data the donor carries (the rest adds nothing)
        Xs, ws = X[sel], w[sel] / w[sel].sum()
        Xc = Xs - ws @ Xs
        v = np.ones(X.shape[1]) / np.sqrt(X.shape[1])
        for _ in range(n_power):
            u = Xc.T @ (ws * (Xc @ v))
            nu = np.linalg.norm(u)
            if nu == 0.0:
                break
            v = u / nu
        lam = float(np.sum(ws * (Xc @ v) ** 2))
        eps = spread * temp / np.sqrt(lam) if lam > 0.0 else 0.0
        W2[j], W2[d] = W[j] + eps * v, W[j] - eps * v
        b2[j] = b2[d] = b[j] - log2
        if Q2 is not None:
            Q2[d] = Q[j]
        eps_all.append(eps)
    new = dict(params)
    new["W"], new["b"] = jnp.asarray(W2), jnp.asarray(b2)
    if Q2 is not None:
        new["Q"] = jnp.asarray(Q2)
    info.update(donor_occ_max=float(pbar[donors].max()), donor_occ_min=float(pbar[donors].min()),
                eps_median=float(np.median(eps_all)), eps_rel_median=float(np.median(
                    [e / (np.linalg.norm(W[j]) + 1e-300) for e, j in zip(eps_all, donors)])))
    return new, info


def write_history(out_dir, slice_dir, n_slices, taus, local, quad, held=None, harmonic=False, patch=False,
                  patch_quad=False):
    """The sampler's input: params_history.npy (theta per slice) and hidden_history.npz (W, b, t, then wl, bl with
    --local, then Q with --quad and K > 0, then V, c with --harmonic, then Wp, bp with --patch and Qp with --patch and
    --quad), stacked in slice order. Slices in `held`
    (slice -> the arrays as saved in its slice file; --assemble memory) are taken from memory, all others are read
    from their slice files (every slice with --assemble files, and slices made by an earlier job: resume,
    --init-from). The same arrays in the same order either way, so the files hold the same bytes (the zip entries of
    the .npz carry their write time). Returns the shape of the theta history."""
    held = held or {}
    thetas, Ws, bs, wls, bls, Qs, Vs, cs = [], [], [], [], [], [], [], []
    Wps, bps, Qps = [], [], []
    for k in range(n_slices):
        s = held[k] if k in held else np.load(os.path.join(slice_dir, f"slice_{k:03d}.npz"))
        thetas.append(s["theta"])
        Ws.append(s["W"])
        bs.append(s["b"])
        if local:
            wls.append(s["wl"])
            bls.append(s["bl"])
        if quad:
            Qs.append(s["Q"])
        if harmonic:
            Vs.append(s["V"])
            cs.append(s["c"])
        if patch:
            Wps.append(s["Wp"])
            bps.append(s["bp"])
            if patch_quad:
                Qps.append(s["Qp"])
    tmp = os.path.join(out_dir, "params_history.tmp.npy")
    theta_hist = np.stack(thetas)
    np.save(tmp, theta_hist)
    os.replace(tmp, os.path.join(out_dir, "params_history.npy"))
    tmp = os.path.join(out_dir, "hidden_history.tmp.npz")
    loc = {"wl": np.stack(wls), "bl": np.stack(bls)} if local else {}
    if Qs:
        loc["Q"] = np.stack(Qs)
    if harmonic:
        loc.update(V=np.stack(Vs), c=np.stack(cs))
    if patch:                                    # per-patch units: after every other member
        loc.update(Wp=np.stack(Wps), bp=np.stack(bps))
        if patch_quad:
            loc["Qp"] = np.stack(Qps)
    np.savez(tmp, W=np.stack(Ws), b=np.stack(bs), t=taus, **loc)
    os.replace(tmp, os.path.join(out_dir, "hidden_history.npz"))
    return theta_hist.shape


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
                 occ_beta=0.999, occ_same_draws=False, zero_slices=(), track_best=True, theta_masks=False,
                 chunk_batches=False):
    """Jitted scan of n training steps, identical in effect to the paper's training_step loop.
    track_best, theta_masks, chunk_batches (--step-impl lean; the defaults True, False, False leave every step as
    before): track_best False (lean with --select ema, whose slice result is the EMA): best_params keeps its value
    from init_carry, which saves reading and writing one copy of every parameter per step; best_loss and best_epoch
    are still tracked and the carry keeps its structure. theta_masks: the projection of theta and the zeroing of the
    k_6 gradient as elementwise selects with constant masks instead of indexed updates. chunk_batches: the n steps'
    batches drawn before the scan from the same keys (the per-step split of the key chain, then one vmapped
    sample_forward_process). The arithmetic is the same, but XLA compiles a changed graph with other fusions and
    fused multiply-adds: with hidden units (GEMMs) the results differ at rounding level, without them they are
    bitwise equal on the L40 (speed_check.py section 6; XLA:CPU differs at rounding level there too).
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
    if theta_masks:                       # project() and the k_6 gradient zeroing below as constant masks
        if n_pairs is None:
            raise ValueError("theta_masks needs n_pairs (the visible model's number of pairs, vis.P)")
        n_theta = 4 * n_vis + 3 * n_pairs
        zero_m, floor_v, k6_m = np.zeros(n_theta, bool), np.full(n_theta, -np.inf), np.zeros(n_theta, bool)
        k6_m[k6] = True
        for s in zero_slices:
            zero_m[s] = True
        if visible == "sextic":
            floor_v[k6] = lm.K6_FLOOR
        else:
            zero_m[k6] = True
            floor_v[kd] = lm.K4_FLOOR
            floor_v[cd] = 0.0             # dropped c_duff entries are zeroed as before (max(0, 0) = 0 there)

    def project(theta):
        if theta_masks:                   # the same values as the indexed updates below (max(x, -inf) = x)
            return jnp.where(zero_m, 0.0, jnp.maximum(theta, floor_v))
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

    def one_step(carry, t_curr, lrs, dt, xs=None):
        params, adam_state, key, step, best_params, best_loss, best_epoch, ema = carry
        if xs is None:
            key, k_epoch, k_release = jr.split(key, 3)            # same split as run_optimization (third key unused there)
            X = sample_forward_process(t_curr, lm.BATCH_SIZE, lm.TEMP, lm.SIGMA_FORWARD, data_train, k_epoch, beta=1)
        else:                                                     # chunk_batches: this step's batch and third key, drawn
            X, k_release = xs                                     # before the scan (the carry's key is already advanced)
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
            grads = dict(grads, theta=(jnp.where(k6_m, 0.0, grads["theta"]) if theta_masks
                                       else grads["theta"].at[k6].set(0.0)))
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
        if track_best:                    # --select ema never reads best_params (the slice result is the EMA)
            best_params = jax.tree_util.tree_map(lambda a, b: jnp.where(better, a, b), new, best_params)
        best_loss = jnp.where(better, loss, best_loss)
        best_epoch = jnp.where(better, step, best_epoch)
        ema = jax.tree_util.tree_map(lambda e, p: e + w_ema * (p - e), ema, new)
        return (new, adam_state, key, step + 1, best_params, best_loss, best_epoch, ema), loss

    @partial(jax.jit, static_argnums=3)
    def run(carry, t_curr, lrs, n, dt=0.0):
        if not chunk_batches:
            return jax.lax.scan(lambda c, _: one_step(c, t_curr, lrs, dt), carry, None, length=n)

        def keys(key, _):                 # one_step's per-step split of the key chain, ahead of the scan
            key, k_epoch, k_release = jr.split(key, 3)
            return key, (k_epoch, k_release)

        key, (k_epochs, k_releases) = jax.lax.scan(keys, carry[2], None, length=n)
        Xs = jax.vmap(lambda k: sample_forward_process(t_curr, lm.BATCH_SIZE, lm.TEMP, lm.SIGMA_FORWARD, data_train,
                                                       k, beta=1))(k_epochs)
        return jax.lax.scan(lambda c, xs: one_step(c, t_curr, lrs, dt, xs), carry[:2] + (key,) + carry[3:],
                            (Xs, k_releases))

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
    steps_sched = parse_steps_schedule(args.steps_schedule)    # raises on an invalid schedule
    time_grid = lm.parse_time_grid(args.time_grid)             # canonical spec; raises on an invalid one
    split_slices = parse_split_slices(args.split_dead)         # raises on an invalid list
    if split_slices and not (args.hidden == "potts" and args.K >= 2 and not args.hidden_fixed):
        raise ValueError("--split-dead needs trained Potts units (--hidden potts, --K >= 2, not --hidden-fixed)")
    if split_slices and not args.split_spread >= 0.0:
        raise ValueError("--split-spread must be >= 0")
    patch = lm.parse_patch_spec(args.patch, lm.problem_geometry(args.problem)[0])   # None, or (side, stride, S)
    if args.ref_kmeans == "gpu" and not (args.hidden == "potts" and args.K > 0) and patch is None:   # --patch: also
        raise ValueError("--ref-kmeans gpu is the k-means of the Potts prototype reference (--hidden potts, --K > 0)")
    if args.neighbourhood != "chebyshev" and args.radius is None:
        raise ValueError("--neighbourhood von_neumann needs an explicit --radius")
    if args.hidden_norm == "state" and (args.K < 1 or args.hidden_fixed):
        raise ValueError("--hidden-norm state needs trained hidden units (--K >= 1, not --hidden-fixed)")
    if args.hidden_rule == "cd1" and (args.hidden_fixed or not (args.K > 0 or args.local) or args.cd1_releases < 1
                                      or (args.cd1_dt <= 0 and args.cd1_kappa <= 0)):
        raise ValueError("--hidden-rule cd1 needs trained hidden units (--K > 0 or --local, not --hidden-fixed), "
                         "--cd1-releases >= 1, and --cd1-dt > 0 or --cd1-kappa > 0")
    n_pix = lm.problem_geometry(args.problem)[0] ** 2
    if not 0 <= args.harmonic <= n_pix:
        raise ValueError(f"--harmonic {args.harmonic}: the number of harmonic modes must lie in 0..{n_pix} (the pixels "
                         f"of {args.problem})")
    if args.harmonic_fixed and not args.harmonic:
        raise ValueError("--harmonic-fixed needs --harmonic R > 0")
    if args.harmonic_fixed_slices and not (args.harmonic and not args.harmonic_fixed
                                           and 1 <= args.harmonic_fixed_slices < lm.N_TIME_STEPS):
        raise ValueError(f"--harmonic-fixed-slices {args.harmonic_fixed_slices}: needs --harmonic R > 0, not "
                         f"--harmonic-fixed, and a slice count in 1..{lm.N_TIME_STEPS - 1}")
    if not (np.isfinite(args.harmonic_lr_scale) and args.harmonic_lr_scale > 0.0):
        raise ValueError(f"--harmonic-lr-scale {args.harmonic_lr_scale}: must be > 0")
    if args.harmonic and (args.hidden_samples > 0 or args.hidden_rule != "exact" or args.hidden_norm != "element"):
        raise ValueError("--harmonic needs exact hidden statistics and the exact rule with per-element Adam (not with "
                         "--hidden-samples, --hidden-rule cd1 or --hidden-norm group / state / occupancy)")
    if args.harmonic_ref != "full" and not (args.harmonic and args.hidden == "potts" and args.K >= 2
                                            and args.proto_init in ("pooled", "perclass") and args.cls == "none"):
        raise ValueError(f"--harmonic-ref {args.harmonic_ref}: the within-cluster reference needs --harmonic R > 0 and a "
                         f"Potts prototype init (--hidden potts, --K >= 2, --proto-init pooled or perclass) of the "
                         f"training data (--cls none)")
    if args.harmonic_lr_tmin != 0.0 and not (args.harmonic and np.isfinite(args.harmonic_lr_tmin)
                                             and 0.0 < args.harmonic_lr_tmin <= lm.T_FORWARD):
        raise ValueError(f"--harmonic-lr-tmin {args.harmonic_lr_tmin}: needs --harmonic R > 0 and a time in "
                         f"(0, {lm.T_FORWARD:g}]")
    if patch is None and (args.patch_init_scale is not None or args.patch_lr_scale != 1.0 or args.patch_on_slice != 0):
        raise ValueError("--patch-init-scale, --patch-lr-scale and --patch-on-slice need --patch SIDE:STRIDE:STATES")
    if patch is not None:
        if (args.hidden_samples > 0 or args.hidden_rule != "exact" or args.hidden_norm != "element" or args.hidden_fixed
                or args.cls != "none"):
            raise ValueError("--patch needs exact hidden statistics, the exact rule with per-element Adam, trained hidden "
                             "units and both classes (not with --hidden-samples, --hidden-rule cd1, --hidden-norm group / "
                             "state / occupancy, --hidden-fixed or --cls)")
        if args.seed < lm.KEY_SEED:
            raise ValueError(f"--seed {args.seed}: --patch needs --seed >= {lm.KEY_SEED} (k-means seeds "
                             f"{lm.PATCH_SEED0} + 1000 (seed - {lm.KEY_SEED}) + patch must be >= 0)")
        if not (np.isfinite(args.patch_lr_scale) and args.patch_lr_scale > 0.0):
            raise ValueError(f"--patch-lr-scale {args.patch_lr_scale}: must be finite and > 0")
        if args.patch_init_scale is not None and not (np.isfinite(args.patch_init_scale) and args.patch_init_scale > 0.0):
            raise ValueError(f"--patch-init-scale {args.patch_init_scale}: must be finite and > 0")
        if not 0 <= args.patch_on_slice <= lm.N_TIME_STEPS - 2:
            raise ValueError(f"--patch-on-slice {args.patch_on_slice}: must lie in 0..{lm.N_TIME_STEPS - 2}")
        if args.patch_on_slice > 0 and args.patch_init_scale is None:
            raise ValueError(f"--patch-on-slice {args.patch_on_slice} needs an explicit --patch-init-scale: 0.06 is the "
                             f"measured safe value at slice 31, where 0.25 costs +27 loss units (job 4231616)")
    lam_p = 0.25 if args.patch_init_scale is None else float(args.patch_init_scale)   # resolved (0.25 when inactive)
    K_on = int(args.patch_on_slice)              # --patch-on-slice (0: from slice 0)
    geom = None
    if patch is not None:
        geom = lm.PatchGeometry(lm.problem_geometry(args.problem)[0], patch[0], patch[1])
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
        "visible_impl": args.visible_impl, "precision": args.precision, "step_impl": args.step_impl,
        "steps_schedule": steps_schedule_str(steps_sched), "ref_kmeans": args.ref_kmeans, "assemble": args.assemble,
        "time_grid": time_grid,
        "split_dead": ",".join(map(str, split_slices)), "split_spread": args.split_spread,
        "harmonic": args.harmonic, "harmonic_fixed": args.harmonic_fixed, "harmonic_lr_scale": args.harmonic_lr_scale,
        "harmonic_fixed_slices": args.harmonic_fixed_slices, "harmonic_ref": args.harmonic_ref,
        "harmonic_lr_tmin": args.harmonic_lr_tmin,
        "patch": lm.patch_spec_str(patch), "patch_init_scale": lam_p, "patch_lr_scale": args.patch_lr_scale,
        "patch_on_slice": K_on,
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
                "visible_impl": "dense", "precision": "f64", "step_impl": "default",
                "steps_schedule": "", "ref_kmeans": "cpu", "assemble": "files",
                "time_grid": "uniform", "split_dead": "", "split_spread": 0.5,
                "harmonic": 0, "harmonic_fixed": False, "harmonic_lr_scale": 1.0,
                "harmonic_fixed_slices": 0, "harmonic_ref": "full",
                "harmonic_lr_tmin": 0.0,
                "patch": "", "patch_init_scale": 0.25, "patch_lr_scale": 1.0,
                "patch_on_slice": 0}                                            # runs from before these options
    code_sha = {f: lm.sha256_file(os.path.join(HERE, f)) for f in ("latent_model.py", "train_latent.py")}
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        diff = {k: (old.get(k, defaults.get(k)), v) for k, v in config.items()
                if k != "assemble" and old.get(k, defaults.get(k)) != v}   # --assemble: the same bytes either way
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
              f"{json.dumps({k: config[k] for k in ('problem', 'visible', 'hidden', 'K', 'cls', 'select', 'hidden_lr_scale', 'freeze_hidden_slices', 'steps_scale', 'max_slices', 'seed', 'hidden_fixed', 'hidden_rule', 'radius', 'proto_init', 'hidden_samples', 'hidden_jackknife', 'hidden_replicates', 'hidden_samples_late', 'hidden_samples_switch', 'hidden_floor', 'hidden_norm', 'occ_beta', 'hidden_jackknife_order', 'occ_same_draws', 'pair_terms', 'neighbourhood', 'ising_reseed', 'ising_weak_boost', 'visible_impl', 'precision', 'step_impl', 'steps_schedule', 'ref_kmeans', 'assemble', 'time_grid', 'split_dead', 'split_spread')})}"
              + (f"; cd1 kappa {args.cd1_kappa}, dt {args.cd1_dt}, releases {args.cd1_releases}, "
                 f"hidden samples {args.cd1_hidden_samples}" if args.hidden_rule == "cd1" else "")
              + (f"; harmonic modes {args.harmonic}" + (" (fixed at the reference)" if args.harmonic_fixed else
                                                        f", lr scale {args.harmonic_lr_scale:g}")
                 + (f", the reference in slices 0-{args.harmonic_fixed_slices - 1}" if args.harmonic_fixed_slices
                    else "")
                 + (f", {args.harmonic_ref} reference" if args.harmonic_ref != "full" else "")
                 + (f", lr scales held at their t = {args.harmonic_lr_tmin:g} value below it" if args.harmonic_lr_tmin
                    else "") if args.harmonic else "")
              + (f"; patch units {lm.patch_spec_str(patch)}: P {geom.P}, d {geom.d}, S {patch[2]}, coverage "
                 f"{int(geom.coverage.min())}-{int(geom.coverage.max())} (mean {geom.coverage.mean():.2f}), W+Q couplings "
                 f"{geom.P * patch[2] * geom.d * (2 if args.quad else 1):,}, initial scale {lam_p:g}, on slice {K_on}, lr "
                 f"scale {args.patch_lr_scale:g}" if patch is not None else ""))

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
    taus = lm.time_grid_pts(time_grid)           # slice times; 'uniform': lm.forward_time_pts(), as before
    lr_v = lm.time_grid_lr(time_grid)            # 'uniform': lm.lr_schedule(), as before
    n_slices = min(args.max_slices, lm.N_TIME_STEPS)
    if time_grid != "uniform":                   # --time-grid: the grid against the paper's (uniform) one
        t_uni, dts = lm.forward_time_pts(), -np.diff(taus)
        n_grid = [lm.time_grid_steps(k, time_grid) for k in range(lm.N_TIME_STEPS)]
        log(logf, f"time grid {time_grid}: slice times {taus[0]:.4f}, {taus[1]:.4f}, {taus[2]:.4f}, ..., {taus[-3]:.5f}, "
                  f"{taus[-2]:.5f}, {taus[-1]:.4f}; intervals {dts[0]:.4g} (first) to {dts[-1]:.4g} (last; uniform "
                  f"{t_uni[0] - t_uni[1]:.4g}); {int(np.sum(taus >= 3.0))} slices at t >= 3 (uniform "
                  f"{int(np.sum(t_uni >= 3.0))}); paper steps {n_grid[0]:,} (t = 5) to {n_grid[-1]:,} (t = 0), "
                  f"{sum(n_grid):,} in all (uniform {sum(lm.paper_steps(k) for k in range(lm.N_TIME_STEPS)):,}), "
                  f"before --steps-scale / --steps-schedule; lr {lr_v[0]:.4g} (t = 5) to {lr_v[-1]:.4g} (t = 0)")
    jk = (2 if args.hidden_jackknife_order == 2 else True) if args.hidden_jackknife else False
    loss_fn = lm.make_loss(vis, lm.TEMP, K, args.hidden, args.local, samples=args.hidden_samples,
                           jackknife=jk, replicates=args.hidden_replicates, compute_dtype=cdt)
    loss_ps = jax.jit(lm.make_loss_per_sample(vis_dense, lm.TEMP, K, args.hidden, args.local))
    loss_ps_k0 = jax.jit(lm.make_loss_per_sample(vis_dense, lm.TEMP, 0))
    cd1_grad = None
    if args.hidden_rule == "cd1":
        cd1_grad = lm.make_cd1_hidden_grad(vis, lm.TEMP, K, args.hidden, args.local, samples=args.cd1_hidden_samples)
        field_scale = jax.jit(lambda p, X: lm.hidden_field_scale(p, X, lm.TEMP, K, args.hidden, args.local))
    lean = {"theta_masks": True, "chunk_batches": True, "track_best": args.select == "best"} \
        if args.step_impl == "lean" else {}       # default: build_runner's defaults, every step as before
    run, adam = build_runner(loss_fn, data_train, n_vis, args.ema_steps, args.visible, vis.P,
                             cd1_grad=cd1_grad, cd1_releases=args.cd1_releases, loss_key=args.hidden_samples > 0,
                             floor=args.hidden_floor, floor_samples=args.hidden_samples, hidden_norm=args.hidden_norm,
                             occ_beta=args.occ_beta, occ_same_draws=args.occ_same_draws, zero_slices=zero_slices,
                             **lean)
    run_late = None
    if args.hidden_samples_late:                  # another draw count from slice --hidden-samples-switch on
        run_late, _ = build_runner(
            lm.make_loss(vis, lm.TEMP, K, args.hidden, args.local, samples=args.hidden_samples_late,
                         jackknife=jk, replicates=args.hidden_replicates, compute_dtype=cdt),
            data_train, n_vis, args.ema_steps, args.visible, vis.P, loss_key=True,
            floor=args.hidden_floor, floor_samples=args.hidden_samples_late, hidden_norm=args.hidden_norm,
            occ_beta=args.occ_beta, occ_same_draws=args.occ_same_draws, zero_slices=zero_slices, **lean)

    state_cls = None
    proto_stats = None                           # --harmonic-ref within: the prototype init's S_w (prototype_reference)
    if fisher:
        W_ref, b_ref = lm.fisher_reference(data, labels, taus, lm.TEMP)
        sW = np.sqrt(np.mean(W_ref**2, axis=1))                  # reference magnitudes for the hidden lr
        sb = np.maximum(np.abs(b_ref), lm.TEMP)
        log(logf, f"Fisher reference: |W*| {np.linalg.norm(W_ref[0]):.4g} (t=5) -> {np.linalg.norm(W_ref[-1]):.4g} (t=0); "
                  f"b* {b_ref[0]:.4g} -> {b_ref[-1]:.4g}; rms W* per pixel {sW[0]:.3g} -> {sW[-1]:.3g}")
    elif K > 0 and args.hidden == "potts":
        km_seed = 100 * (args.seed - lm.KEY_SEED)
        pooled = args.proto_init in ("pooled", "random")       # random: the reference only sets lr scales/diagnostics
        t_ref = time.time()
        km_fn = lm.kmeans_jax if args.ref_kmeans == "gpu" else None    # None: numpy k-means (lm.kmeans), as before
        want_sw = bool(args.harmonic) and args.harmonic_ref == "within"
        ref_out = lm.prototype_reference(data, labels, taus, lm.TEMP, K, seed=km_seed, pooled=pooled,
                                         kmeans_fn=km_fn, return_stats=want_sw)
        W_ref, b_ref, state_cls = ref_out[:3]
        proto_stats = ref_out[3] if want_sw else None
        if not (np.isfinite(W_ref).all() and np.isfinite(b_ref).all()):
            raise RuntimeError(f"prototype reference is not finite for --seed {args.seed} (k-means seeds {km_seed} + class)")
        # The hidden learning-rate scales always come from the default-seed reference (k-means seed 0), so that
        # --seed changes the initial prototypes and the key chain but not the step sizes (rms b* is sensitive to the
        # clustering: at t = 0 it is 0.59 for k-means seed 0 and 1.5 for seed 100 with 128 states at 14x14).
        W_lr, b_lr = (W_ref, b_ref) if km_seed == 0 else \
            lm.prototype_reference(data, labels, taus, lm.TEMP, K, seed=0, pooled=pooled, kmeans_fn=km_fn)[:2]
        t_ref = time.time() - t_ref
        sW = np.sqrt(np.mean(W_lr**2, axis=(1, 2)))
        sb = np.maximum(np.sqrt(np.mean(b_lr**2, axis=1)), lm.TEMP)
        rb = np.maximum(np.sqrt(np.mean(b_ref**2, axis=1)), lm.TEMP)
        rW = np.sqrt(np.mean(W_ref**2, axis=(1, 2)))
        log(logf, f"prototype reference ({K} states, " + ("pooled k-means without labels, seed " + str(km_seed) if pooled
                  else f"{K // n_cls} per class, k-means seeds {km_seed} + class") + "): |W*| "
                  f"{np.linalg.norm(W_ref[0]):.4g} (t=5) -> {np.linalg.norm(W_ref[-1]):.4g} (t=0); rms b* {rb[0]:.3g} -> "
                  f"{rb[-1]:.3g}; rms W* per entry {rW[0]:.3g} -> {rW[-1]:.3g}; hidden lr scales (k-means seed 0): "
                  f"b {sb[0]:.3g} -> {sb[-1]:.3g}, W {sW[0]:.3g} -> {sW[-1]:.3g}"
                  + ("; initial W random (rms of W* at t = 5), b = 0" if args.proto_init == "random" else "")
                  + f"; computed in {t_ref:.1f} s (k-means on the {args.ref_kmeans})")
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

    R_h = args.harmonic                          # harmonic modes (--harmonic R; 0: none, nothing below changes)
    if R_h:
        import sample_latent                     # data_stats: the statistics cache shared with --init-dist gauss
        st_h = sample_latent.data_stats(args.problem, args.cls)
        mu_h, lam_h, U_h = (np.asarray(st_h[k], dtype=np.float64) for k in ("mu", "lam", "U"))
        mu_err = float(np.max(np.abs(mu_h - np.asarray(data_train, dtype=np.float64).mean(axis=0))))
        if mu_h.shape != (n_vis,) or not mu_err <= sample_latent.STATS_MU_TOL:
            raise RuntimeError(f"--harmonic: the data statistics {st_h['path']} do not describe this run's training data "
                               f"(shape {mu_h.shape}, largest mean difference {mu_err:.3g})")
        lam_r, U_r, within_info = lam_h, U_h, None   # the reference's spectrum: C's ('full', the default) ...
        if args.harmonic_ref == "within":        # ... or that of the prototype init's S_w, fixed at the run's first start
            w_path = os.path.join(out_dir, "harmonic_within.npz")
            lam_now, U_now = lm.harmonic_within_spectrum(proto_stats["Sw"])
            if os.path.exists(w_path):           # resume: the saved spectrum (a k-means on other hardware could differ)
                with np.load(w_path) as zw:
                    lam_r, U_r = np.asarray(zw["lam"], dtype=np.float64), np.asarray(zw["U"], dtype=np.float64)
                dev = float(np.max(np.abs(lam_now - lam_r)) / lam_r[0]) if lam_r.shape == lam_now.shape else float("inf")
                if not dev <= 1e-9:
                    log(logf, f"WARNING: the within-cluster covariance S_w computed now differs from the run's first "
                              f"start (largest eigenvalue difference {dev:.3g} relative to the top one); the modes keep "
                              f"the saved reference {w_path}")
            else:
                lam_r, U_r = lam_now, U_now
                save_atomic(w_path, lam=lam_r, U=U_r, counts=np.asarray(proto_stats["counts"]),
                            pi=np.asarray(proto_stats["pi"]))
            Sw_e = np.einsum("ij,ik,kj->j", U_h[:, :3], np.asarray(proto_stats["Sw"]), U_h[:, :3])
            within_info = {"file": w_path, "file_sha256": lm.sha256_file(w_path), "kmeans_seed": km_seed,
                           "pooled": pooled, "states": K, "trace_Sw": float(lam_r.sum()), "trace_C": float(lam_h.sum()),
                           "Sw_lam_top5": [float(v) for v in lam_r[:5]], "C_lam_top5": [float(v) for v in lam_h[:5]],
                           "Sw_share_along_C_axes_1_3": [float(v) for v in Sw_e / lam_h[:3]]}
        t_lr = taus if args.harmonic_lr_tmin == 0.0 else np.maximum(taus, args.harmonic_lr_tmin)
        sV, sc = lm.harmonic_lr_scales(mu_h, lam_r, U_r, R_h, t_lr)
        w_first, iso_first = lm.harmonic_reference_strengths(lam_r, taus[0], R_h)
        w_last, iso_last = lm.harmonic_reference_strengths(lam_r, taus[-1], R_h)
        ref_info = {"R": R_h, "fixed": args.harmonic_fixed, "fixed_slices": args.harmonic_fixed_slices,
                    "lr_scale": args.harmonic_lr_scale,
                    "stats_file": st_h["path"], "stats_sha256": st_h["file_sha256"], "stats_recomputed": st_h["recomputed"],
                    "problem": args.problem, "cls": args.cls, "n_images": st_h["n_images"],
                    "data_file_sha256": st_h["data_file_sha256"], "lam_top5": [float(v) for v in lam_r[:5]],
                    "lam_0": max(float(lam_r.min()), 0.0), "lam_R": float(lam_r[R_h - 1]),
                    "iso_spring_first_last": [lm.TEMP / iso_first, lm.TEMP / iso_last],
                    "strength_first_top5": [float(v) for v in w_first[:5]],
                    "strength_last_top5": [float(v) for v in w_last[:5]],
                    "lr_scale_V": [float(v) for v in sV], "lr_scale_c": [float(v) for v in sc],
                    "slice_t": [float(v) for v in taus], "slurm_job": os.environ.get("SLURM_JOB_ID", "manual"),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        if args.harmonic_ref != "full" or args.harmonic_lr_tmin:     # recorded only when used (the defaults: as before)
            ref_info.update(reference=args.harmonic_ref, lr_tmin=args.harmonic_lr_tmin, within=within_info)
        ref_path = os.path.join(out_dir, "harmonic_reference.json")
        if not os.path.exists(ref_path):
            with open(ref_path + ".tmp", "w") as f:
                json.dump(ref_info, f, indent=2)
            os.replace(ref_path + ".tmp", ref_path)
        else:
            with open(ref_path) as f:
                old_ref = json.load(f)
            if old_ref.get("stats_sha256") != ref_info["stats_sha256"]:
                log(logf, f"WARNING: the data statistics differ from the run's first start ({old_ref.get('stats_file')} "
                          f"sha256 {str(old_ref.get('stats_sha256'))[:16]} -> {st_h['path']} "
                          f"{st_h['file_sha256'][:16]}); the slices from here on use the new ones")
        log(logf, f"harmonic modes: {R_h}" + (" fixed at the reference" if args.harmonic_fixed else "")
                  + (f", the reference in slices 0-{args.harmonic_fixed_slices - 1} and trained from slice "
                     f"{args.harmonic_fixed_slices} on" if args.harmonic_fixed_slices else "")
                  + f"; reference from {st_h['path']} ({st_h['n_images']} images, sha256 {st_h['file_sha256'][:16]}, "
                  f"{'computed now' if st_h['recomputed'] else 'cached'})"
                  + (f", within-cluster: the Potts prototype init's S_w ({K} states, k-means seed {km_seed}; trace "
                     f"{within_info['trace_Sw']:.5g} of C's {within_info['trace_C']:.5g}; along C's axes 1-3 it keeps "
                     f"{'/'.join(f'{v:.3f}' for v in within_info['Sw_share_along_C_axes_1_3'])} of the variance; "
                     f"{within_info['file']})" if within_info else "")
                  + f": lam top {lam_r[0]:.4g}, lam_R "
                  f"{lam_r[R_h - 1]:.4g}, lam_0 {ref_info['lam_0']:.3g}; isotropic spring T/iso {lm.TEMP / iso_first:.4g} "
                  f"(t = {taus[0]:g}) -> {lm.TEMP / iso_last:.4g} (t = {taus[-1]:g}); mode strengths |v_1|^2 "
                  f"{w_first[0]:.4g}, |v_R|^2 {w_first[-1]:.3g} (t = {taus[0]:g}) -> {w_last[0]:.4g}, {w_last[-1]:.3g} "
                  f"(t = {taus[-1]:g}); lr scales V {sV[0]:.3g} -> {sV[-1]:.3g}, c {sc[0]:.3g} -> {sc[-1]:.3g} (x "
                  f"{args.hidden_lr_scale:g} x {args.harmonic_lr_scale:g}"
                  + (f"; held at their t = {args.harmonic_lr_tmin:g} value below it" if args.harmonic_lr_tmin else "")
                  + ")")

    if patch is not None:                        # per-patch Potts units (--patch; None: none, nothing below changes)
        S_p = patch[2]
        km_fn_p = lm.kmeans_jax if args.ref_kmeans == "gpu" else None      # as for the global reference
        seed0_p = lm.PATCH_SEED0 + 1000 * (args.seed - lm.KEY_SEED)
        data64 = np.asarray(data, dtype=np.float64)
        clus = lm.patch_clusters(data64, geom, S_p, seed0_p, km_fn_p)
        # the learning-rate scales always come from the seed-1 clustering, as the global unit's from k-means seed 0
        clus_lr = clus if seed0_p == lm.PATCH_SEED0 else lm.patch_clusters(data64, geom, S_p, lm.PATCH_SEED0, km_fn_p)
        del data64
        sWp, sbp = lm.patch_lr_scales(clus_lr, taus, lm.TEMP)
        if not (np.all(np.isfinite(sWp)) and np.all(sWp > 0) and np.all(np.isfinite(sbp)) and np.all(sbp > 0)):
            raise RuntimeError(f"--patch {lm.patch_spec_str(patch)}: the per-patch learning-rate scales are not all finite "
                               f"and positive")
        Wp0, bp0 = (a[0] for a in lm.patch_reference(clus, taus[K_on:K_on + 1], lm.TEMP, scale=lam_p))   # the start
        if not (np.all(np.isfinite(Wp0)) and np.all(np.isfinite(bp0))):
            raise RuntimeError(f"--patch {lm.patch_spec_str(patch)}: the start templates are not finite (--seed "
                               f"{args.seed}, k-means seeds {seed0_p} + patch)")

        def _sha(*arrs):
            h = hashlib.sha256()
            for a in arrs:
                h.update(np.ascontiguousarray(a).tobytes())
            return h.hexdigest()

        ev, cnt = clus["explained_variance"], clus["counts"]
        mmm = lambda a: [float(np.min(a)), float(np.median(a)), float(np.max(a))]     # noqa: E731
        pref = {"spec": lm.patch_spec_str(patch), "P": geom.P, "d": geom.d, "S": S_p,
                "coverage_min_max_mean": [int(geom.coverage.min()), int(geom.coverage.max()), float(geom.coverage.mean())],
                "kmeans": "numpy" if km_fn_p is None else "gpu",
                "kmeans_seeds": [int(clus["seeds"][0]), int(clus["seeds"][-1])],
                "kmeans_seconds": float(clus["seconds"]),
                "lr_kmeans_seeds": [int(clus_lr["seeds"][0]), int(clus_lr["seeds"][-1])],
                "lr_kmeans_seconds": 0.0 if clus_lr is clus else float(clus_lr["seconds"]),
                "explained_variance_median_min_central_median": [
                    float(np.median(ev)), float(np.min(ev)), float(np.median(ev[geom.central])) if geom.central.any()
                    else None],
                "empty_clusters": int(np.sum(cnt == 0)), "clusters_below_20_images": int(np.sum(cnt < 20)),
                "lr_scale_W_min_median_max": [mmm(v) for v in sWp], "lr_scale_b_min_median_max": [mmm(v) for v in sbp],
                "init_scale": lam_p, "on_slice": K_on, "start_t": float(taus[K_on]), "lr_scale": args.patch_lr_scale,
                "start_sha256": _sha(Wp0, bp0), "lr_scales_sha256": _sha(sWp, sbp),
                "slice_t": [float(v) for v in taus], "slurm_job": os.environ.get("SLURM_JOB_ID", "manual"),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        pref_path = os.path.join(out_dir, "patch_reference.json")
        if not os.path.exists(pref_path):
            with open(pref_path + ".tmp", "w") as f:
                json.dump(pref, f, indent=2)
            os.replace(pref_path + ".tmp", pref_path)
        else:
            with open(pref_path) as f:
                old_pref = json.load(f)
            if (old_pref.get("start_sha256"), old_pref.get("lr_scales_sha256")) != (pref["start_sha256"],
                                                                                      pref["lr_scales_sha256"]):
                log(logf, f"WARNING: the per-patch reference differs from the run's first start (start arrays sha256 "
                          f"{str(old_pref.get('start_sha256'))[:16]} -> {pref['start_sha256'][:16]}, lr scales "
                          f"{str(old_pref.get('lr_scales_sha256'))[:16]} -> {pref['lr_scales_sha256'][:16]}); the slices "
                          f"from here on use the new ones")
        log(logf, f"patch units {pref['spec']}: P {geom.P}, d {geom.d}, S {S_p}; k-means ({pref['kmeans']}) seeds "
                  f"{pref['kmeans_seeds'][0]}-{pref['kmeans_seeds'][1]} in {clus['seconds']:.0f} s"
                  + ("" if clus_lr is clus else f" (lr scales from seeds {pref['lr_kmeans_seeds'][0]}-"
                                                f"{pref['lr_kmeans_seeds'][1]}, {clus_lr['seconds']:.0f} s)")
                  + f"; explained variance median {np.median(ev):.3f} (min {np.min(ev):.3f}); empty clusters "
                  f"{pref['empty_clusters']}, < 20 images {pref['clusters_below_20_images']}; lr scales W median "
                  f"{np.median(sWp[0]):.3g} (t = {taus[0]:g}) -> {np.median(sWp[-1]):.3g} (t = {taus[-1]:g}), b median "
                  f"{np.median(sbp[0]):.3g} -> {np.median(sbp[-1]):.3g} (x {args.hidden_lr_scale:g} x "
                  f"{args.patch_lr_scale:g}); start {lam_p:g} x the reference at slice {K_on} (t = {taus[K_on]:.4f}), "
                  f"|Wp| {np.linalg.norm(Wp0):.4g}" + (f"; no patch units in slices 0-{K_on - 1}" if K_on else ""))

        @jax.jit
        def force_rms(params, X):
            """Force decomposition on the validation batch (float64): sqrt(mean over samples and pixels of g^2) of
            the visible force and of each hidden part's (global unit(s), per-pixel units, harmonic modes, patches)."""
            def rms(g):
                return jnp.sqrt(jnp.mean(g * g))
            out = {"force_rms_vis": rms(vis_dense.grad_lap(params["theta"], X)[0])}
            if K > 0 and "Q" in params:
                out["force_rms_glob"] = rms(lm.hidden_terms_quad(params["W"], params["Q"], params["b"], X, lm.TEMP,
                                                                 args.hidden)[0])
            elif K > 0:
                out["force_rms_glob"] = rms((lm.hidden_terms if args.hidden == "ising" else lm.potts_terms)(
                    params["W"], params["b"], X, lm.TEMP)[0])
            if args.local:
                out["force_rms_loc"] = rms(lm.local_terms(params["wl"], params["bl"], X, lm.TEMP)[0])
            if "V" in params:
                out["force_rms_harm"] = rms(lm.harmonic_terms(params["V"], params["c"], X)[0])
            out["force_rms_patch"] = rms(lm.patch_terms(params["Wp"], params.get("Qp"), params["bp"], X, lm.TEMP,
                                                        with_lap=False)[0])
            return out

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
        src_grid = lm.config_time_grid(scfg)          # 'uniform' for runs from before --time-grid
        if src_grid != time_grid:                     # allowed only where the linked slices sit at the same times
            n_link = min(args.init_slices, lm.N_TIME_STEPS)
            if not np.array_equal(lm.time_grid_pts(src_grid)[:n_link], np.asarray(taus)[:n_link]):
                raise RuntimeError(f"--init-from {src}: its slices were trained on the time grid {src_grid!r}, this run "
                                   f"uses {time_grid!r}, and the {n_link} linked slices would sit at other times than "
                                   f"this run's (power:P@K shares slices 0..K with the uniform grid)")
            log(logf, f"--init-from {src}: source grid {src_grid!r}, this run {time_grid!r}; the {n_link} linked slices "
                      f"sit at identical times (t >= {float(taus[n_link - 1]):.4f})")
        free = {"name", "steps_scale", "val_every", "lr_scale", "init_from", "init_slices", "max_slices",
                "visible_impl", "precision", "step_impl", "steps_schedule", "ref_kmeans",
                "assemble", "time_grid", "split_dead", "split_spread",
                "harmonic_lr_scale", "harmonic_lr_tmin",
                "patch_lr_scale"}                    # numerics of the new slices only (time_grid: checked above;
                                                     # harmonic, harmonic_fixed(_slices) and harmonic_ref must match:
                                                     # the model's modes and their reference; so must patch,
                                                     # patch_init_scale and patch_on_slice)
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
        if R_h:                                  # harmonic modes: the reference at slice 0's t (t = 5)
            V0, c0, _ = lm.harmonic_reference(mu_h, lam_r, U_r, R_h, float(taus[0]))
            params.update(V=jnp.asarray(V0), c=jnp.asarray(c0))
        if patch is not None and K_on == 0:      # patch units: LAM x the reference at slice 0's t
            params.update(Wp=jnp.asarray(Wp0), bp=jnp.asarray(bp0))
            if args.quad:
                params["Qp"] = jnp.zeros_like(params["Wp"])
    else:
        prev = np.load(os.path.join(slice_dir, f"slice_{done - 1:03d}.npz"))
        params = {"theta": jnp.asarray(prev["theta"]), "W": jnp.asarray(prev["W"]), "b": jnp.asarray(prev["b"])}
        if args.quad and K > 0:
            params["Q"] = jnp.asarray(prev["Q"])
        if args.local:
            params.update(wl=jnp.asarray(prev["wl"]), bl=jnp.asarray(prev["bl"]))
        if R_h:
            if "V" not in prev.files or prev["V"].shape != (R_h, n_vis):
                raise RuntimeError(f"slice {done - 1} of {slice_dir} has no {R_h} harmonic modes to continue from")
            params.update(V=jnp.asarray(prev["V"]), c=jnp.asarray(prev["c"]))
        if patch is not None and done - 1 >= K_on:   # patch units trained up to slice done - 1 (before K_on: none)
            need = ("Wp", "bp") + (("Qp",) if args.quad else ())
            shp = {"Wp": (geom.P, patch[2], geom.d), "bp": (geom.P, patch[2]), "Qp": (geom.P, patch[2], geom.d)}
            if any(g not in prev.files or prev[g].shape != shp[g] for g in need):
                raise RuntimeError(f"slice {done - 1} of {slice_dir} has no {lm.patch_spec_str(patch)} patch units to "
                                   f"continue from")
            params.update({g: jnp.asarray(prev[g]) for g in need})
        optimization_key = jnp.asarray(prev["key"])
        where = f"t = {taus[done]:.4f}" if done < n_slices else "all slices present; assembling"
        log(logf, f"resuming at slice {done} ({where}) from {slice_dir}")

    # ---- slices -----------------------------------------------------------------------------------------------
    held = {} if args.assemble == "memory" and n_slices == lm.N_TIME_STEPS else None   # this job's slices (assembly)
    for k in range(done, n_slices):
        W_prev = np.asarray(params["W"]) if tree is not None else None    # the previous slice's units (diagnostics)
        t0 = time.time()
        t_curr = float(taus[k])
        optimization_key, optimization_subkey = jr.split(optimization_key)
        n_steps = slice_steps(k, args.steps_scale, steps_sched, time_grid)   # no schedule, uniform grid: as before
        if args.hidden_fixed:                    # hidden couplings = this slice's closed-form reference, not trained
            params = dict(params)
            if K > 0:
                one = fisher
                params["W"] = jnp.asarray(W_ref[k:k + 1] if one else W_ref[k])
                params["b"] = jnp.asarray(b_ref[k:k + 1] if one else b_ref[k])
            if args.local:
                params["wl"], params["bl"] = jnp.asarray(wl_ref[k]), jnp.asarray(bl_ref[k])
        fixed_h = R_h and (args.harmonic_fixed or k < args.harmonic_fixed_slices)   # modes held at the reference
        if R_h:                                  # this slice's harmonic reference (fixed slices: the modes themselves)
            V_rk, c_rk, iso_k = lm.harmonic_reference(mu_h, lam_r, U_r, R_h, t_curr)
            if fixed_h:
                params = dict(params)
                params["V"], params["c"] = jnp.asarray(V_rk), jnp.asarray(c_rk)
        if patch is not None and K_on > 0 and k == K_on:   # --patch-on-slice: switched on at LAM x the reference at t_K
            params = dict(params, Wp=jnp.asarray(Wp0), bp=jnp.asarray(bp0))
            if args.quad:
                params["Qp"] = jnp.zeros_like(params["Wp"])
            log(logf, f"slice {k}: patch units {lm.patch_spec_str(patch)} switched on at {lam_p:g} x the reference at "
                      f"t = {t_curr:.4f}")
        split_info = None
        if k in split_slices and k >= args.freeze_hidden_slices:   # --split-dead: dead states become copies
            Xsp = sample_forward_process(t_curr, VAL_N, lm.TEMP, lm.SIGMA_FORWARD, data_train,
                                         jr.fold_in(jr.PRNGKey(VAL_SEED + 3), k), beta=1)
            params, split_info = split_dead_states(params, Xsp, args.split_spread)
            log(logf, f"slice {k}: --split-dead: {split_info['dead']} states below 1e-6 at the start; "
                      f"{split_info['split']} re-seeded as copies of the busiest states"
                      + (f" (donor occupation {split_info['donor_occ_min']:.2e}-{split_info['donor_occ_max']:.2e}; "
                         f"median eps {split_info['eps_median']:.3g}, relative to |W_j| "
                         f"{split_info['eps_rel_median']:.3g})" if split_info["split"] else ""))
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
        if R_h:                                  # harmonic modes: held during the first freeze slices, as the units
            on = k >= args.freeze_hidden_slices and not fixed_h
            f_h = args.hidden_lr_scale * args.harmonic_lr_scale
            lrs.update(V=float(f_h * sV[k]) if on else 0.0, c=float(f_h * sc[k]) if on else 0.0)
        if "Wp" in params:                       # patch units: per-unit arrays of fixed shape (the runner traces once)
            on = k >= max(K_on, args.freeze_hidden_slices)
            f_p = args.hidden_lr_scale * args.patch_lr_scale
            lrs["Wp"] = jnp.asarray(f_p * sWp[k][:, None, None] if on else np.zeros((geom.P, 1, 1)))
            lrs["bp"] = jnp.asarray(f_p * sbp[k][:, None] if on else np.zeros((geom.P, 1)))
            if "Qp" in params:
                lrs["Qp"] = lrs["Wp"]
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
               **({"lr_V": lrs["V"], "lr_c": lrs["c"]} if R_h else {}),
               **({"lr_Wp": [float(v) for v in (np.min(lrs["Wp"]), np.median(lrs["Wp"]), np.max(lrs["Wp"]))],
                   "lr_bp": [float(v) for v in (np.min(lrs["bp"]), np.median(lrs["bp"]), np.max(lrs["bp"]))]}
                  if "Wp" in params else {}),
               **({"n_weak_boosted": n_boost} if n_boost is not None else {}),
               **({"split": split_info} if split_info is not None else {}),
               **({"cd1_dt": dt_k} if cd1_grad is not None else {}),
               **({"hidden_samples": args.hidden_samples_late if run_k is run_late else args.hidden_samples}
                  if run_late is not None else {}),
               "first_loss": float(losses[0]), "last1000_mean": float(np.mean(losses[-1000:])),
               "best_loss": best_loss, "best_epoch": best_epoch, "val_loss": val, "val_median": float(np.median(val_ps)),
               "select": args.select, "seconds": t_train, "steps_per_s": n_steps / t_train,
               **({"steps_factor": steps_factor(steps_sched, k)} if steps_sched else {}),
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
            # the paper's model at this slice's t: its slice k on the uniform grid (as before); on another --time-grid
            # its slices interpolated linearly in t, as its sampler does (exactly its slice at t = 5 and t = 0)
            th_paper = (np.array(paper[k], dtype=np.float64) if time_grid == "uniform"
                        else lm.interp_history(paper, lm.forward_time_pts(), t_curr))
            pp = {"theta": jnp.asarray(th_paper), "W": jnp.zeros((0, n_vis)), "b": jnp.zeros((0,))}
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
        if R_h:                                  # harmonic modes against this slice's reference (harmonic_unit_stats)
            Vn, cn = np.asarray(params["V"]), np.asarray(params["c"])
            rec.update(lm.harmonic_unit_stats(Vn, cn, V_rk, c_rk, U_r[:, :R_h], iso_k, lm.TEMP,
                                              theta=np.asarray(params["theta"]), conn=prob["conn"]))
            extra.update(V=Vn, c=cn)
        if "Wp" in params:                       # patch units against this slice's scale-1 reference, forces
            Wpn, bpn = np.asarray(params["Wp"]), np.asarray(params["bp"])
            Qpn = np.asarray(params["Qp"]) if "Qp" in params else None
            W_ref_p = lm.patch_reference(clus, [t_curr], lm.TEMP)[0][0]
            st_p, arr_p = lm.patch_unit_stats(Wpn, bpn, Qpn, np.asarray(Xv), lm.TEMP, W_ref_p, geom)
            rec.update(st_p)
            rec.update({q: float(v) for q, v in force_rms(params, jnp.asarray(Xv)).items()})
            extra.update(Wp=Wpn, bp=bpn, **({"Qp": Qpn} if Qpn is not None else {}), **arr_p)
        elif patch is not None:                  # --patch-on-slice, before the switch-on: zero arrays, no record keys
            extra.update(Wp=np.zeros((geom.P, patch[2], geom.d)), bp=np.zeros((geom.P, patch[2])),
                         **({"Qp": np.zeros((geom.P, patch[2], geom.d))} if args.quad else {}))
        save_atomic(os.path.join(slice_dir, f"slice_{k:03d}.npz"),
                    theta=np.asarray(params["theta"]), W=np.asarray(params["W"]), b=np.asarray(params["b"]),
                    key=np.asarray(optimization_key), loss_trace=losses[::10].astype(np.float32),
                    val_per_sample=val_ps.astype(np.float32), val_idx=idx.astype(np.int32), record=json.dumps(rec),
                    **extra)
        if held is not None:                     # --assemble memory: the arrays just saved, for write_history
            held[k] = {"theta": np.asarray(params["theta"]), "W": np.asarray(params["W"]),
                       "b": np.asarray(params["b"]),
                       **{g: extra[g] for g in ("Q", "wl", "bl", "V", "c", "Wp", "bp", "Qp") if g in extra}}
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
        if R_h:
            msg += (f"; modes |V| {rec['harm_V_norm']:.4g} (ref {rec['harm_Vref_norm']:.4g}), cos "
                    f"{rec['harm_cos_V_Vref']:.3f} (V^T V {rec['harm_cos_M_Mref']:.3f}), added variance "
                    f"{rec['harm_added_var']:.4g} (ref {rec['harm_added_var_ref']:.4g}; {rec['harm_n_beyond_iso']} modes "
                    f"beyond T/iso {rec['harm_iso_spring']:.4g}), curvature at 0: springs {rec['harm_vis_curv_min']:.3g}, "
                    f"with the modes {rec['harm_curv_min']:.3g}")
        if "patch_W_norm" in rec:
            fr = lambda q: f"{rec[q]:.3g}" if q in rec else "n/a"      # noqa: E731
            msg += (f"; patches |Wp| {rec['patch_W_norm']:.4g} (ref {rec['patch_Wref_norm']:.4g}), cos "
                    f"{rec['patch_cos_W_Wref']:.3f}, eff states {rec['patch_eff_states_median']:.2f}/"
                    f"{rec['patch_eff_states_min']:.2f} (median/min), dead {rec['patch_states_below_1e-6']}, <max p> "
                    f"{rec['patch_mean_max_p']:.3f}; force patch/glob/loc {fr('force_rms_patch')}/{fr('force_rms_glob')}/"
                    f"{fr('force_rms_loc')}")
        log(logf, msg)

    # ---- assemble ---------------------------------------------------------------------------------------------
    if n_slices == lm.N_TIME_STEPS and held is not None:      # --assemble memory: the same files (write_history)
        t_asm = time.time()
        shape = write_history(out_dir, slice_dir, n_slices, taus, args.local, args.quad and K > 0, held,
                              harmonic=bool(R_h), patch=patch is not None, patch_quad=patch is not None and args.quad)
        with open(os.path.join(out_dir, "DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        log(logf, f"assembled params_history.npy {shape} and hidden_history.npz in {time.time() - t_asm:.1f} s "
                  f"({len(held)} slices from memory, {n_slices - len(held)} from their files); DONE")
    elif n_slices == lm.N_TIME_STEPS:
        thetas, Ws, bs, wls, bls, Qs, Vs, cs = [], [], [], [], [], [], [], []
        Wps, bps, Qps = [], [], []
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
            if R_h:
                Vs.append(s["V"])
                cs.append(s["c"])
            if patch is not None:
                Wps.append(s["Wp"])
                bps.append(s["bp"])
                if args.quad:
                    Qps.append(s["Qp"])
        tmp = os.path.join(out_dir, "params_history.tmp.npy")
        np.save(tmp, np.stack(thetas))
        os.replace(tmp, os.path.join(out_dir, "params_history.npy"))
        tmp = os.path.join(out_dir, "hidden_history.tmp.npz")
        loc = {"wl": np.stack(wls), "bl": np.stack(bls)} if args.local else {}
        if Qs:
            loc["Q"] = np.stack(Qs)
        if R_h:                                  # harmonic modes: after every other member (write_history)
            loc.update(V=np.stack(Vs), c=np.stack(cs))
        if patch is not None:                    # patch units: after every other member (write_history)
            loc.update(Wp=np.stack(Wps), bp=np.stack(bps))
            if args.quad:
                loc["Qp"] = np.stack(Qps)
        np.savez(tmp, W=np.stack(Ws), b=np.stack(bs), t=taus, **loc)
        os.replace(tmp, os.path.join(out_dir, "hidden_history.npz"))
        with open(os.path.join(out_dir, "DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        log(logf, f"assembled params_history.npy {np.stack(thetas).shape} and hidden_history.npz; DONE")


if __name__ == "__main__":
    main()
