# Frozen code for the convergence study (2026-11-11)

A copy of the latent_physical_SGM code as validated on 2026-11-11 (jobs 4222239, 4222254, 4222240, 4222241, 4222242,
4222313; see `../latent_physical_SGM/CODE_AND_PERFORMANCE.md` section 9): `--visible-impl stencil` and
`--precision mixed` in training, the old (unchanged) sampler. Copied so that this study does not depend on the code
being edited in `latent_physical_SGM/` at the same time. **Do not edit.** Runs write to the shared
`out/latent_physical_SGM/` as usual. Job scripts here call the code in this folder; submit from this folder.

- `latent_model.py`, `train_latent.py`, `sample_latent.py`, `fid_score.py`, `evaluate_multiclass.py`: the snapshot
  taken before the parallel sampler / training-step work started.
- `evaluate_samples.py`, `blend_check.py`, `make_gallery.py`, `plot_learning_curves.py`, `summarize_training.py`:
  unchanged copies.

## Questions

1. How many steps, at which learning rate, do the late slices 80-99 need? (They were still improving at 4x steps;
   with float64, 1x steps at lr x4 beat 2x steps on validation loss, and lr x16 collapsed state use to 116
   effective states.)
2. Does the 2x-steps FID gain come from the early slices (state survival is decided in slices 7-30)?
3. How large is the seed-to-seed spread of FID and of the validation loss (needed to judge FID differences of 2-3)?

## Runs (four digits, Potts 784, --quad --local, quartic, radius 1, lin; stencil + mixed; --val-every 1000)

Run names `mnist28_d0123_K784_ema_potts_quad_local_quartic_r1_lin_pooled<suffix>`; each pipeline trains, samples 500
base trajectories with the old sampler (same initial states and Brownian keys as every other run) and evaluates.

| suffix | what | job |
|---|---|---|
| `_late_s1_lr4_mixst` | slices 80-99 from the paper-schedule run's slices 0-79, 1x steps, lr x4 (float32 twin of 4221036) | 4222697 |
| `_late_s1_lr2_mixst` | the same, lr x2 | 4222698 |
| `_late_s1_lr8_mixst` | the same, lr x8 | 4222699 |
| `_late_s2_lr1_mixst` | late slices at 2x steps, lr x1 | 4222700 |
| `_late_s2_lr4_mixst` | late slices at 2x steps, lr x4 | 4222706 |
| `_late_s4_lr1_mixst` | late slices at 4x steps, lr x1 (float32 twin of 4221035) | 4222707 |
| `_mixst_seed1` | full run, paper schedule, seed 1 (rounding twin of the float64 run) | 4222701 |
| `_mixst_seed2`, `_mixst_seed3` | full runs, seeds 2 and 3 | 4222702, 4222705 |
| `_mixst_early2x_part` | slices 0-30 at 2x steps (training only) | 4222703 |
| `_mixst_early2x` | slices 31-99 at 1x on top of `_early2x_part` | 4222704 |

Chains (afterany): 4222697 -> 98 -> 99 -> 4222700; 4222701 -> 02; 4222703 -> 04 -> 05; 4222706 -> 07.
Analysis (after all): FID / KID paired against the paper-schedule run's 500 samples (fid_score.py), galleries
(make_gallery.py), learning curves (plot_learning_curves.py).
