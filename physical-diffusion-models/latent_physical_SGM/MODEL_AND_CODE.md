# Latent physical SGM: the model, the current recipe, and the code changes

Reference for the current state of the code (written 2026-11-16, updated 2026-11-18 15:40). It says *what* the model is and *which* settings
we use and why; the evidence is in the dated entries of `CODE_AND_PERFORMANCE.md` (the lab notebook, cited below as
"notebook, date: heading"). Older results (14x14, Potts vs Ising, hidden-unit variants) are in `README.md`; the
original design and derivations in `DESIGN.md`. Keep this file current when a default or the recipe changes.

## 1. The model

A score-based diffusion model whose score is the force of a physical energy (paper code: arXiv 2506.19136, in
`../physical_diffusion_fns/`). This folder adds hidden units, integrated out exactly, so they never appear as
variables in the simulation.

- **Data.** Pixels in [0, 1], plus N(0, 0.01²) pixel noise, then one global mean / std normalisation (the paper's
  preprocessing, `latent_model._normalise_like_paper`). Problems: `mnist28_d0123456789` (54,210 images),
  `fashion28_c0123456789` (60,000), `celeba32g` (162,770 aligned faces, centre crop 140x140, grey, 32x32),
  `celeba64g` (the same at 64x64; data not kept on disk at present, see 5).
- **Forward process.** Ornstein-Uhlenbeck, t from 0 to 5, temperature T = 0.02:
  x_t = e^{-t} x_0 + sqrt(T (1 - e^{-2t})) eps.
- **Free energy at time t** (one parameter set per time slice; 100 slices, t = 5 -> 0):

  F(x) = F_vis(x) + F_glob(x) + F_loc(x) + sum over patch sets F_patch(x)

  - F_vis (`DenseVisible`; `--visible quartic --radius 1 --pair-terms lin`): per-pixel quadratic and quartic terms
    and a bias, plus linear springs c (x_i - x_j)² between the 8 nearest neighbours.
  - F_glob (`--hidden potts --quad`): one Potts unit with K states,
    F_glob = -T log sum_m exp(a_m / T), a_m = b_m + W_m . x + Q_m . x².
  - F_loc (`--local`): one Ising unit per pixel, coupled to that pixel only.
  - F_patch (`--patch SIDE:STRIDE:S`, optionally `--patch2`): one Potts unit with S states per overlapping
    SIDE x SIDE patch, coupled to its patch only, same form as F_glob. The patches lie on a grid with the given
    stride.
- **Loss.** Implicit score matching per sample, L = -lap F / T + |grad F|² / (2 T²), with an analytic Laplacian.
  Each slice is trained separately (Adam, 512 fresh forward samples per step, the paper's step counts: 10,901 at
  t = 5 to 30,901 at t = 0), warm-started from the previous slice; the slice result is an EMA of the iterates
  (`--select ema`, 1000 steps).
- **Sampling.** Reverse SDE dx = [x - 2 grad F(x; params(5 - t))] dt + sqrt(2T) dW (diffrax SRA1, adaptive,
  parameters linearly interpolated between the slice knots), then the **quench** (below).

### Parameter counts (couplings W + Q, i.e. 2 x units x states x pixels; biases and the visible model are small)

| Part | MNIST / Fashion 28x28 | CelebA 32x32 |
|---|---|---|
| global K784 | 1.23 M | 1.61 M |
| global K1536 | 2.41 M | 3.15 M |
| global K3072 / K6144 | 4.82 M / 9.63 M | 6.29 M / - |
| patches 7x7 every 3 px, 32 states (64 units) | 0.20 M | - |
| patches 8x8 every 2 px, 32 states | 0.50 M (121 units) | 0.69 M (169 units) |
| patches 4x4 every 2 px, 32 states (225 units) | - | 0.23 M |
| patches 16x16 every 4 px, 32 states (25 units) | - | 0.41 M |

Per time slice; the sampler interpolates 100 such slices. For comparison, Wang & Achour's KuraSHIL model has 326k
parameters in all (notebook, 2026-11-15: "Comparison: Wang & Achour"). Reducing the count: see "Parameter reduction" below (2026-11-19/20).

## 2. The current recipe

Common training flags (all runs since 2026-11-13):

```
--select ema --hidden potts --quad --local --visible quartic --radius 1 --pair-terms lin --proto-init pooled
--seed S --val-every 1000 --visible-impl stencil --precision mixed --ref-kmeans gpu
--patch SIDE:STRIDE:32 --patch-init-scale 0.06 --patch-on-slice 55          [--patch2 SIDE:STRIDE:32]
```

| Choice | Value | Why (evidence) |
|---|---|---|
| hidden units | one Potts unit + per-pixel units, quadratic couplings | README Results 7-9 (2026-11-08) |
| global start | k-means prototypes of all images (`--proto-init pooled`), no labels | README (28x28: labels not needed) |
| hidden learning rate | 5e-4 x the reference scale (default `--hidden-lr-scale`), held in slice 0 (`--freeze-hidden-slices 1`) | defaults since 2026-11-08 |
| speed | `--visible-impl stencil --precision mixed`: 16-38x per step, float32 products with TF32 off; validation stays float64 | notebook 2026-11-11, sections 9 ff. |
| patch units | 32 states each; patches switched on at slice 55 (t = 2.2) at 0.06 x their k-means reference | 2026-11-13: patches halve the FID; switching on at t = 2.2 keeps the global shapes; 0.06 is the measured safe start scale (0.25 costs +27 loss units, job 4231616) |
| patch layout | MNIST 7x7 every 3 px; Fashion and CelebA 8x8 every 2 px | 2026-11-13 22:15: 8x8 every 2 px beats 7x7 every 3 px (-0.7 to -1.0 FID) |
| second patch scale (CelebA) | 16x16 every 4 px | 2026-11-16: -1.6 to -1.8 Inception FID after the quench; 4x4 every 2 px gives -1.4 |
| K | MNIST 6144, Fashion 1536, CelebA 1536 | MNIST gains about 3.4 FID per doubling; Fashion and CelebA saturate at 1536 (2026-11-15, 2026-11-16) |
| start of the reverse process | `--init-dist equil:3`: the N(0, T I) states relaxed for 3 time units of Langevin dynamics under the t = 5 energy; no data needed | 2026-11-12: worth about 1.5 FID; after the quench it adds only 0-1 (CelebA, 2026-11-15) |
| quench | `--quench 1`: after the SDE, dx/ds = -grad F(x; t = 0 params) for s = 1 (zero temperature) | 2026-11-15: the largest single gain; removes the speckle; no memorisation (nearest-neighbour check) |
| corrector | `--corrector 2:0.16`: 2 unadjusted Langevin steps at every knot (SNR step size, Song et al.) | 2026-11-17: -1.1 Inception FID after the quench on CelebA-32 (22.64 -> 21.50), KID 14.9 -> 12.5; 4 steps no better; +4% sampling time. Physically: with s = -grad F / T a step is x <- x - (eps / T) grad F + sqrt(2 eps) z, i.e. overdamped Langevin relaxation in F(x; theta_k) at the device temperature T for a dwell time ~ eps / T with the parameters frozen at the trained slice: plain thermalisation, the most native operation of a thermal device (the predictor needs 2 grad F and the expanding +x term). It acts at the slice times, where theta is exact, and resets the error that interpolation and the predictor accumulate between slices. Caveat: the SNR rule gives each sample its own eps; a device would use one dwell time per slice (untested). Gains (2026-11-21/23): CelebA-64 -5.0 (43.15 -> 38.16), MNIST K16 -2.4, CelebA-32 -0.25 exact / -1.0 on device, Fashion K784 none. **Physical version (2026-11-23, being tested): `--corrector dwell:TAU`**, the same segmented sampler, but at every interior knot continuous Langevin dynamics dx = -grad F(x; theta_k) ds + sqrt(2T) dW for one fixed time TAU (all samples, all knots); see CODE_AND_PERFORMANCE.md, "Physical corrector" |

**On-device training** (2026-11-17/18; every hidden statistic measured from sampled hidden states, every hidden
state on the visit-count rule). Add to the common flags:

```
--hidden-samples 64 --hidden-jackknife --hidden-norm occupancy --occ-same-draws --patch-occupancy
```

CelebA-32 K784 + 8x8 + 16x16: 23.59 Inception FID after the quench, against 22.85 with exact statistics and Adam
(gap 0.7; bootstrap sd 0.3-0.4). Training takes about 2x longer at S = 64 (6.4 h vs 3.3 h). Without patch units
the gap is about 3 (two seeds), and S = 256 does not reduce it.

### LOCKED RECIPE FOR THE PAPER (2026-11-21)

The same model for every data set (MNIST ten digits, Fashion ten classes, CelebA-32, CelebA-64): K16 global Potts
unit, 64 states per patch unit, squared-pixel couplings Q and per-pixel units, 100 slices, linear interpolation,
seed 1. Runs: exact statistics (Adam); on device with S = 64; on device with S = 16. Sampling for every model: noise
start, reverse SDE, quench 1, n = 2000, and a second set with `--corrector 2:0.16`. Evaluation: Inception FID vs train
with FID_inf; MNIST / Fashion also classifier FID and class shares (`fid_score.py`). Results: CODE_AND_PERFORMANCE.md,
section "LOCK-IN FOR THE PAPER".

Training (`jobs/run_pipeline_neuronic.sh PROBLEM TAG "EXTRA" ema 16 2000 "SAMPLE_ARGS"`), EXTRA =
```
--hidden potts --quad --local --visible quartic --radius 1 --pair-terms lin --proto-init pooled --seed 1
--val-every 1000 --visible-impl stencil --precision mixed --ref-kmeans gpu --patch-init-scale 0.06
--patch-on-slice 10 --patch2-on-slice 55
  28x28 / 32x32: --patch 16:4:64 --patch2 8:2:64      (16x16 every 4 px from t = 4.49, 8x8 every 2 px from t = 2.22)
  64x64:         --patch 32:8:64 --patch2 16:4:64     (32x32 every 8 px from t = 4.49, 16x16 every 4 px from t = 2.22)
on device, add: --hidden-norm occupancy --patch-occupancy --hidden-samples S --hidden-jackknife --occ-same-draws
  (S = 64 or 16)
```
PROBLEM: `mnist28_d0123456789`, `fashion28_c0123456789`, `celeba32g`, `celeba64g`. SAMPLE_ARGS: `--quench 1` (64x64:
`--chunk 25 --quench 1` with `XLA_PYTHON_CLIENT_ALLOCATOR=platform` exported before sbatch and `--mem 160G`; the
first 64x64 run with `--chunk 100` and the default allocator sampled, but ran out of GPU memory in the quench, 2026-11-21). Corrector set:
`sample_latent.py --run RUN --n 2000 --chunk 250 --corrector 2:0.16 --quench 1` (64x64: `--chunk 25`, same
allocator, 160 G; K784 models: `--mem 64G`).
MNIST / Fashion also with `--K 784` (the user, 2026-11-21: on-device runs for both K16 and K784), since the global unit
carries the class identity there (K784: MNIST 10.01 vs 14.39, Fashion 21.32 vs 26.00).

**Visible-model ablation (CelebA-32, exact, 2026-11-21/23; CODE_AND_PERFORMANCE.md "Results 2026-11-23"):** the
radius-1 coupling springs c_lin (x_i - x_j)^2 carry the visible model (without them +5.6 FID, grainy images); the
Duffing self term k_duff x^4 is nearly dispensable: a fully harmonic visible layer (`--visible harmonic
--pair-terms lin`: linear self spring + linear coupling springs) gives 20.98 vs 20.20 and looks the same, with no
diverging samples. Without springs: quartic self term 25.83, linear 27.42. Springs alone, no hidden units: 218.

### Best models and command lines (seed 1)

Training + sampling (pipeline, from `latent_physical_SGM/`):

```
sbatch [--mem M --time T] jobs/run_pipeline_neuronic.sh PROBLEM TAG "EXTRA FLAGS" ema K N_SAMPLES ["SAMPLE ARGS"]
```

| Data | Model | Inception FID, quench 1 (n = 2000) | Classifier FID |
|---|---|---|---|
| MNIST, ten digits | **K784 + 16x16 every 4 px from t = 4.49 + 8x8 every 2 px from t = 2.22, 64 patch states, noise start** (locked patches with a larger global unit; `--K 784 --patch 16:4:64 --patch-on-slice 10 --patch2 8:2:64 --patch2-on-slice 55`; 2026-11-21) | 10.01 (4.44); **with `--corrector 2:0.16`: 9.19 (3.57), best** | 6.23; corrector 5.06 |
| MNIST, ten digits | locked recipe K16 + the same patches, noise start (2026-11-21) | 14.39 (8.13); corrector 12.00 (4.83) | 14.23; corrector 12.27 |
| MNIST, ten digits | K784 + the same patches, **on device** S = 64 / S = 16 (2026-11-22; Inception 4290780) | 11.37 (5.64) / 11.80 (6.33); corrector 9.79 (3.85) / 10.17 (4.08) | 6.60 / 7.86; corrector 6.46 / 6.55 |
| MNIST, ten digits | K16 + the same patches, **on device** S = 64 / S = 16 (2026-11-22) | 15.22 / 15.38; corrector 12.56 / 12.79 | 13.72 / 13.80 |
| MNIST, ten digits | K6144 + 7x7 every 3 px, equil start (previous best) | 18.8 (FID_inf 15.9) | 3.87 |
| Fashion, ten classes | **K784 + 16x16 every 4 px from t = 4.49 + 8x8 every 2 px from t = 2.22, 64 patch states, noise start** (2026-11-21) | **21.32 (9.78), best**; corrector 21.82 (9.79) | 2.77; corrector 2.32 |
| Fashion, ten classes | locked recipe K16 + the same patches, noise start (2026-11-21) | 26.00 (15.66); corrector 25.80 (14.62) | 5.97; corrector 4.88 |
| Fashion, ten classes | K784 + the same patches, **on device** S = 64 / S = 16 (2026-11-22; Inception 4290789) | 23.44 (11.92) / 22.93 (11.51); corrector 23.50 / 23.52 | 3.54 / 3.51; corrector 3.01 / 2.97 |
| Fashion, ten classes | K16 + the same patches, **on device** S = 64 / S = 16 (2026-11-22) | 28.29 / 27.54; corrector 27.44 / 26.08 | 6.45 / 6.08 |
| Fashion, ten classes | K1536 + 8x8 every 2 px, equil start, 3 seeds (previous best) | 25.9 +- 0.04 (21.3) | 2.71 +- 0.07 |
| CelebA 32x32 grey | **locked recipe: K16 + 16x16 every 4 px from t = 4.49 + 8x8 every 2 px from t = 2.22, 64 patch states**, noise start (2026-11-20; Inception 4287721) | **20.20 (12.3), best single model without label** | - |
| CelebA 32x32 grey | locked recipe, **on device** S = 64 / S = 16 (2026-11-21/22; Inception 4290856) | 21.20 (13.55) / 21.23 (13.45); corrector 20.23 (11.83) / 20.49 (11.93); ODE + quench (S = 64) 22.93; knots 5/9/9 (S = 64) 23.65 | - |
| CelebA 32x32 grey | K784 + the locked patches, exact (control, 2026-11-21; Inception 4290734) | 19.43 (11.59); corrector 19.52 (10.71) | - |
| CelebA 32x32 grey | **K784 + 16x16 every 4 px from t = 4.49 + 8x8 every 2 px from t = 2.22 + 8-cluster label** (`--patch 16:4:32 --patch-on-slice 10 --patch2 8:2:32 --patch2-on-slice 55 --cond 8 --cond-on global,patch`; 2026-11-19) | 21.06 (17.3); with `--corrector 2:0.16`: 20.57 (16.8) (2026-11-20) | - |
| CelebA 32x32 grey | K64 + 16x16 from t = 4.49 + 8x8 from t = 2.22, `--corrector 2:0.16` (1.23 M couplings per slice; 2026-11-20) | 20.73 (17.1) | - |
| CelebA 32x32 grey | K64 + 16x16 from t = 4.49 + 8x8 from t = 2.22, 3 seeds (1.23 M couplings per slice) | 21.86 ± 0.15 | - |
| CelebA 32x32 grey | K1536 + 8x8 every 2 px + 16x16 every 4 px, noise start, `--corrector 2:0.16` (best, 2026-11-17) | 21.50 (18.0) | - |
| CelebA 32x32 grey | K1536 + 8x8 every 2 px + 16x16 every 4 px, noise start | 22.4 (18.6) | - |
| CelebA 32x32 grey | K784 + 8x8 every 2 px + 16x16 every 4 px, noise start, exact statistics, Adam | 22.85 (19.1) | - |
| CelebA 32x32 grey | the same, **fully on-device training** (S = 64 sampled statistics, visit-count rule; 2026-11-18) | 23.59 (20.0) | - |
| CelebA 32x32 grey | K784 + 8x8 every 2 px + 4x4 every 2 px + 16x16 every 4 px, noise start (ties the above with half the global states, 2026-11-17) | 22.3 (18.7) | - |
| CelebA 64x64 grey | **locked recipe: K16 + 32x32 every 8 px from t = 4.49 + 16x16 every 4 px from t = 2.22, 64 patch states**, exact (2026-11-22; Inception 4297027) | 43.15 (34.34); **with `--corrector 2:0.16`: 38.16 (26.67), best** | - |
| CelebA 64x64 grey | locked recipe, **on device** S = 64 / S = 16 (2026-11-23; Inception 4297028) | 46.00 (38.53) / 46.58 (39.84); **with `--corrector 2:0.16`: 41.30 (29.79) / 41.14 (30.15)**; S = 64 post-hoc knots 5/9/9: 47.59 (41.63) | - |
| CelebA 64x64 grey | K1536 + 16x16 every 4 px, noise start (first run, 2026-11-17; floor 5.9) | 58.5 (55.9) | - |
| CelebA 64x64 grey | K3072 + 16x16 every 4 px, noise start (2026-11-18): no gain over K1536 | 58.3 (55.7) | - |
| CelebA 64x64 grey | K64 + 16x16 every 4 px + 32x32 every 8 px + clamped 8-cluster label (`--cond 8 --cond-on global,patch`; 4.98 M couplings per slice, a third of K1536 + 16x16; 2026-11-19; without the label 59.25) | 57.8 (55.0) | - |
| CelebA 64x64 grey | **K64 + 32x32 every 8 px switched on early, at t = 4.49, + 16x16 every 4 px from t = 2.22** (`--patch 32:8:32 --patch-on-slice 10 --patch2 16:4:32 --patch2-on-slice 55`; 2026-11-20; with the 8-cluster label 49.25; history compressed to 5 / 9 / 9 knots, 42.7 M instead of 331.8 M stored: 47.99) | 49.21 (46.8) | - |
| CelebA 64x64 grey | the same + 2 corrector steps per slice (`--corrector 2:0.16`; 2026-11-20) | 43.10 (40.0) | - |
| CelebA 64x64 grey | **the same, history compressed to 5 / 9 / 9 knots (42.7 M stored), + 2 corrector steps** (2026-11-20; at 32x32 this combination is worse: 24.69 vs 20.73) | **40.88 (37.9)** | - |

**Small models** (CelebA-32, 2026-11-18; exact statistics, Adam, noise start, quench 1; couplings W + Q per slice):

| Model | couplings per slice | Inception FID, quench 1 (FID_inf) |
|---|---|---|
| K784 + 8x8 + 16x16 | 2.71 M | 22.85 (19.1) |
| K64 + 8x8 + 16x16 | 1.23 M | 24.61 (21.0) |
| K64 + 8x8 + 16x16, trained on clean images (`--data-noise 0`) | 1.23 M | 23.91 (20.4) |
| K64 + 8x8 + 16x16 + clamped label on the hidden biases, 8 clusters (`--cond 8 --cond-on global,patch`) | 1.28 M | 23.57 (20.1) |
| K64 + 8x8 + 16x16 + clamped label on the hidden biases (`--cond 32 --cond-on global,patch`) | 1.43 M | 23.52 (20.0) |
| K64 + 8x8 + 16x16 + clamped label on the hidden biases, 128 clusters | 2.03 M | 23.64 (20.1) |
| **K64 + 8x8 (from t = 2.22) + 16x16 switched on early, at t = 4.49** (`--patch 16:4:32 --patch-on-slice 10 --patch2 8:2:32 --patch2-on-slice 55`; 2026-11-19) | 1.23 M | **21.72 (17.9)** |
| K64 + 8x8 + 16x16 + clamped label on the hidden biases and a visible field (`--cond 32`) | 1.46 M | 23.42 (19.9) |
| K32 + 8x8 + 16x16 | 1.17 M | 25.36 (21.7) |
| K784 + 8x8 | 2.30 M | 24.66 (20.8) |
| K64 + 8x8 | 0.82 M | 26.41 (22.7) |

The global unit can shrink 12-25x for about 2-2.5 FID; at small K the patch units are almost the whole budget.

### Parameter reduction (CelebA-32, 2026-11-19/20; notebook "Overnight 2026-11-19/20" and "Time-basis compression")

**2026-11-22: the post-hoc knots 5/9/9 recipe does not transfer to MNIST / Fashion.** On the on-device S = 64
models it gives Inception 234 (MNIST, K16 and K784) and 88-98 (Fashion), vs +2.5 on CelebA-32 (23.65 vs 21.20).
Relative error of the visible group at 5 knots: 0.35 on MNIST vs 0.12 on CelebA-32; global W / b / Q 0.30-0.55 vs
0.22-0.30. The visible and global parameters of the ten-class models change too much in time for 5 knots.

Starting point: K64 + 16x16 from t = 4.49 + 8x8 from t = 2.22, 1.23 M couplings per slice, 21.72. Stored over the
100 slices: **83.2 M** (the patch sets exist only from their switch-on slice). For scale: Wang & Achour 326 k in
all; a CIFAR-10 DDPM about 36 M.

*Per slice* (retrained, seed 1, quench 1): **K32 1.16 M 21.70** (= K64; 11-20); K16 1.13 M 22.11; 16 patch states
0.68 M 25.75 (with K32: 26.26); no patch Q 0.68 M 29.85; 8x8 every 4 px 0.74 M 28.89; 8 states 0.41 M 34.42; 16x16
only 0.54 M 46.69; everything reduced 0.12 M 66.1. The global unit can shrink to K16-K32 almost for free; the patch
units carry the model and cannot shrink without loss (at equal size, Q with 16 states beats 32 states without Q:
25.75 vs 29.85). Contrast is unchanged in all variants. At 64x64 (11-20, early 32x32 + 16x16): K16 48.63, K32 49.31, K64 49.21; 16 patch states
57.67, 8 states 66.38: the same picture. Running (11-20): K16 with 64 patch states, no Q anywhere.

*In time* (post-hoc, `compress_history.py`, no retraining): replace each parameter group's 100-slice history by a
few kept slices ("knots") with linear interpolation in t, as the sampler already does between slices.

| compression | stored | Inception FID, quench 1 |
|---|---|---|
| none | 83.2 M | 21.72 |
| **even knots per set: global set 5, both patch sets 9, per-pixel units wl / bl at their time mean** | **10.7 M (7.8x less)** | **21.67** |
| even knots 9 everywhere, wl / bl constant | 11.1 M | 22.03 |
| even knots 7 everywhere, wl / bl constant | 8.6 M | 22.75 |
| even knots 5 everywhere | 6.3 M | 25.03 |
| rank-8 SVD in time per group | 11.3 M | 22.20 |
| 16-state model (0.68 M per slice), knots 9 | 6.3 M | 25.80 (uncompressed 25.75) |

Rules found (rounds 1-5, 30 variants):
- **Move the groups of a set together.** The parameters of a slice are co-adapted (biases compensate couplings);
  knots placed per group (by path length) give 35-73, compressing only the coupling matrices is worse than
  compressing everything, and per-group SVD is no better than knots despite far lower reconstruction error.
- **Even spacing in t beats path-length placement** (also when shared per set: 25.9 vs 22.35 at 9 knots): the
  parameters move most at small t, where errors hardly matter (the quench repairs them), while faces and contrast
  are set at t ~ 2-4.
- **The global set needs 3-5 knots** (5 = as good as 9; a linear ramp, 2 knots, collapses to one saturated
  silhouette: 241), and more global knots with coarse patches hurt (17 / 5 / 5: 29.8). The early 16x16 set needs the
  most knots (at 5: +4.4; 8x8 at 5: +1.8). The per-pixel units need no time dependence.
- The reconstruction error does not predict the FID; losses show up as lost contrast.
**At 64x64 the same recipe, unchanged** (K64 + early 32x32 + 16x16; global 5, patch sets 9, wl / bl constant): 42.7 M
instead of 331.8 M stored (7.8x less), FID 49.21 -> **47.99** (2026-11-20, Inception 4285410).
Hardware reading: B coupling sets per patch set, crossfaded linearly in t.

*Corrector on the compressed 5 / 9 / 9 histories* (11-20): 64x64 **40.88** (best at 64x64, 42.7 M stored), but 32x32
24.69 (worse than 21.67 without, 20.73 for 100 slices with). The corrector settles at each slice to what the
interpolated parameters say, which were never trained between knots; the samples get brighter and harsher. That
helps the flat 64x64 models and overshoots at 32x32.

*Step-constant instead of linear* (11-20, post hoc, `compress_history.py --mode steps:B`: each piece held at the
mean of its slices): 34 / 17 / 9 / 5 sets give 22.36 / 23.49 / 26.96 / 37.84 against linear 21.71 / 21.87 / 22.35 /
25.03; the 5 / 9 / 9 recipe 33.22 against 21.67. **Steps need about 4x the sets of linear** (step 34 = linear 9),
as expected for a zeroth-order approximation. Trained step pieces are running.

*Trained knots* (11-20, `train_latent.py --time-knots`, one grid for all sets with the switch-on slices forced in):

| knots (global / 16x16 / 8x8) | stored | trained | post hoc, same knots |
|---|---|---|---|
| 9 / 8 / 4 | 7.4 M | **24.83** | 29.21 |
| 6 / 5 / 3 | 5.0 M | **30.23** | 42.40 |
| 5 / 4 / 2 | 3.8 M | 109 (blotchy texture) | 86.7 (one grey face) |

Training keeps the contrast that thinning loses (-4.4 at 9, -12 at 6), but a shared grid starves the 8x8 set.
Per-set grids (`--time-knots-sets`, each set's own even knots; linear 5/9/9 and 3/5/5, step 5/9/9 and 9/17/17) are
running.

Floors (real held-out images at n = 2000): MNIST 3.7, Fashion 7.2, CelebA 6.4. Wang & Achour: MNIST 27.6,
Fashion 80.8 (their n and FID code are not stated). **The model's training data are not these images**: the
paper's preprocessing adds N(0, 0.01^2) pixel noise to every training image, and 2000 such CelebA-32 training images
score 11.07 (raw test images + 1% noise: 10.75). So a perfect model of its training data would score about 11. In
practice this matters little after the quench: training on clean images gains only 0.7 there (and 12.5 as sampled,
where the remaining grain is the model's own end-of-SDE noise).

Sampling with the equilibrated start and the quench (default sampler):

```
sbatch --mem 64G --time 04:00:00 jobs/run_gpu_neuronic.sh sample_latent.py --run RUN --n 2000 --chunk 125 --init-dist equil:3
sbatch --mem 64G --time 02:00:00 jobs/run_gpu_neuronic.sh sample_latent.py --run RUN --n 2000 --chunk 125 --init-dist equil:3 --quench 1
```

(The quench reads the existing samples and writes `<stem>_q1.npy`.) For K >= 3072 see 4.

### Evaluation

| Script | What | Notes |
|---|---|---|
| `inception_fid.py --problem P LABEL=PATH ...` | standard Inception FID (TTUR graph, pool_3), FID_inf, KID; real-test floor; pixel-noise diagnostics | GPU; train statistics cached per dataset (`celeba` and `celeba64` apart) |
| `evaluate_multiclass.py` / `fid_score.py` | classifier FID, precision / recall, class shares (MNIST, Fashion) | run by the pipeline; skipped for CelebA |
| `make_gallery.py` | side-by-side galleries with a README per folder | descriptive names, e.g. `K784+patch8x8-every2px_start-noise_quench1` |
| `nn_check.py` | nearest training / test image distances (memorisation) | |
| `plot_inception_fid.py` | the summary figure `figures/inception_fid_mnist_fashion.{png,pdf}`: a 2 x 2 grid of MNIST, Fashion, CelebA 32x32, CelebA 64x64 (since 2026-11-19) | CPU job |
| `contrast_check.py` | pixel histograms, per-image contrast and brightness, dark / bright shares, maps of the mean and the spread over images | CPU; 2026-11-17 |
| `tol_check.py` | per-sample differences between sample files of the same trajectories (solver convergence) | CPU; 2026-11-17 |
| `sample_latent.py --oracle-start K --quench 1` | the reverse process started at slice K from exactly noised training images, then quenched (the oracle-start scan) | GPU; sampling and quench in one job (an existing oracle file is not quenched again) |
| `oracle_source_check.py --run R --slices ... --out DIR` | what the end of the reverse process and the quench do to real images: the oracle sets and the quenched source images against their sources (RMS change, contrast, power in three frequency bands), example and spectrum figures | GPU (the quench); 2026-11-18 |
| `forward_marginal_check.py [--snapshots]` | how much of a face is left at time t (PCA SNR, kurtosis, posterior effective number of training faces; Part B: the model's reverse trajectory against the forward marginal per PC group) | CPU / GPU; 2026-11-19 |
| `compress_history.py --run R --mode knots:B [--group-modes g=mode,...] [--knot-share group\|set\|all] [--dry-run]` | post-hoc compression of a run's 100-slice parameter history in time (knots, knotsarc, knotsopt, hat, cheb, svd:R); writes `<run>_time-<tag>` that the sampler reads unchanged, with report.txt | CPU; 2026-11-19/20; best recipe (CelebA): `--mode knots:9 --group-modes theta=knots:5,W=knots:5,b=knots:5,Q=knots:5,wl=svd:0,bl=svd:0`; it fails on MNIST / Fashion (FID 234 / 88-98, 2026-11-22) |

Sample specs `PATH+PATH` concatenate files; `latent_model.split_set_spec` keeps run names that contain a '+'.

## 3. Code changes since the paper code, and how each was verified

Backups of the files before each change: `../latent_physical_SGM_pre_ising_patch/<file>_pre_<change>.py`.

| Date | Change | Files | Verification |
|---|---|---|---|
| 11-06 to 11-09 | hidden units: Ising / Potts, `--quad`, `--local`; quartic visible model; references by k-means / split trees | `latent_model`, `train_latent`, `sample_latent` | README Results 1-15; `test_latent.py` |
| 11-11 | `--visible-impl stencil`, `--precision mixed`, `--step-impl`, `--steps-schedule`, `--ref-kmeans gpu`, `--assemble`, `sample_latent --fast` | as above, `sample_fast` | notebook 11-11 "Build results"; defaults unchanged |
| 11-12 | `--init-dist {set, gauss, equil:S}`, `--oracle-start` | `sample_latent` | notebook 11-12 |
| 11-13 | patch Potts units: `--patch`, `--patch-init-scale`, `--patch-on-slice`, `--patch-lr-scale` | `latent_model`, `train_latent`, `sample_latent` | `test_latent.py` block 19 (gradient 2e-16, Laplacian 4e-15 vs autodiff); default path bit for bit unchanged |
| 11-14 | `--patch-kind ising` | as above | notebook 11-14 |
| 11-15 | `--quench`, `--sampler ode`, `--fixed-langevin` | `sample_latent` | notebook 11-15 |
| 11-15 | host-memory leak fixed: the training data is passed to the jitted runner as an argument, not embedded as a constant (each slice used to recompile and copy it) | `train_latent.build_runner` | 54 slice arrays bit-identical before / after; CelebA MaxRSS 67 -> 13 GB, 34 -> 3 s per slice |
| 11-15 | CelebA problems `celeba32g`, `celeba64g`; `make_celeba_data.py` | `latent_model`, `fid_score`, `nn_check`, `inception_fid` | previews, split sizes as the official partition |
| 11-15 | Inception FID, nearest neighbours, chain switching, summary figure | `inception_fid.py`, `nn_check.py`, `chain_switch.py`, `plot_inception_fid.py` | real-vs-real floors, noise diagnostics |
| 11-15 | second patch scale `--patch2` (`latent_model.PATCH_SETS`) | `latent_model`, `train_latent`, `sample_latent` | job 4256601: `--patch` alone bit-identical to the old code; two-scale run trains, samples, quenches |
| 11-16 | `split_set_spec` for '+' in run names; `patch2` in gallery READMEs | `latent_model`, `evaluate_samples`, `make_gallery`, `fid_score`, `inception_fid` | rerun of the two-scale evaluation (4261206) |
| 11-16 | block unfold / fold for patch layouts with side and grid multiples of the stride (`PatchGeometry.impl = 'blocks'`); `LATENT_PATCH_IMPL=slices` forces the old path | `latent_model` | job 4262849: patch term and its gradients bitwise for every layout; training agrees to rounding (1e-10 to 2e-6 relative after 4 slices, not bitwise: the fused graph rounds differently); CelebA-32 8x8 + 16x16 training 3x faster |
| 11-16 | third patch scale `--patch3` (`PATCH3_KEYS`); the extra sets handled in one loop in `train_latent` | `latent_model`, `train_latent`, `sample_latent`, `make_gallery` | job 4262952: two-scale run bit-identical to the pre-change code; three-scale run trains, samples, quenches |
| 11-16 | `--hidden-fixed` with Potts patch units: every patch set held at its k-means reference from the switch-on slice | `train_latent` | job 4263528: frozen sets equal their reference (cos 1.000, equal norms) in every slice |
| 11-17 | quench-only mode no longer builds the unused SDE drift (a second device copy of the parameter history; CelebA-64 K1536 ran out of GPU memory) | `sample_latent` | the quench path itself is unchanged |
| 11-17 | `--patch-fixed-scale S` (with `--hidden-fixed`): frozen patch sets at S x their reference; 1/coverage = 0.0625 counts each pixel once (the full reference over-counts 16-fold) | `train_latent` | frozen faces lose the stripe artefacts (job 4266722) |
| 11-17 | `--history-dtype float32`: the parameter history on the device in float32 (half the memory; needed for CelebA-64 K3072) | `sample_latent` | vs float64, same trajectories: samples within 4.2e-4, quench within 3.7e-4 (jobs 4266800 / 4266809) |
| 11-17 | `--force-scale LAMBDA` (sampling at T / LAMBDA) and `--corrector M:SNR` (segmented predictor-corrector sampler, unadjusted Langevin, SNR step size) | `sample_latent` | test 4266899; full scan 4267202-12 |
| 11-17 | `--sde-tol-scale F` (solver tolerances x F) | `sample_latent` | convergence check: x0.1 moves samples by ~1e-6 of the sample spacing (jobs 4266909 / 4266910) |
| 11-17 | sampled hidden statistics (`--hidden-samples`, `--hidden-norm occupancy`) allowed with `--precision mixed`: the draws in float32, products in float32 with TF32 off | `latent_model.loss_draw_occupation`, `train_latent` | test 4267283: mixed vs float64 over 4 slices, validation losses to the printed digits, couplings to 1e-6-2e-5 (different draws) |
| 11-17 | sampled statistics for the patch units (`--hidden-samples` with Potts patch units; draws with key fold_in(key, 1000 + set)) | `latent_model.patch_terms(stat=)`, `make_loss_per_sample`, `train_latent` | test 4267958: unbiased loss, gradient error 1/sqrt(S) (`check_sampledpatch.py`); exact path bitwise; runs without patches bitwise |
| 11-17 | visit-count rule for the patch states, `--patch-occupancy` (one step scale per patch unit) | `latent_model.patch_occupation`, `OccupancyTransform`, `train_latent` | test 4268021: occupations reproduce the loss's draws exactly; without the flag bitwise |
| 11-17 | histories above 2^31 - 1 elements are interpolated in column blocks (`interpolate_history`; XLA crashed on CelebA-64 K3072 + 16x16, 2.79e9 elements) | `sample_latent` | test 4268605: 3 blocks vs the single call, 1e-16 relative; smaller histories take the old call |
| 11-17 | gallery panel of real images named `01_real_images.png` (was `01_real_digits.png`) | `make_gallery` | |
| 11-18 | `--data-noise SD`: the training images' pixel noise instead of the paper's 0.01 (0: clean images; same normalisation constants, so sampling and evaluation are unchanged) | `latent_model.make_problem` / `_normalise_like_paper`, `train_latent` | test 4272439: default data bitwise as before (CelebA, Fashion); 0 maps back to the raw images (6e-17) |
| 11-18 | `--cond C --cond-on global,patch,visible`: a clamped per-image label (k-means cluster of the training image, seed 31000 + seed) that shifts the global and patch state biases (`Bc` (C, K), `Bpc*` (C, P, S)) and / or adds a visible field (`hc` (C, N)); all learned, time dependent, starting at zero. Sampling draws one label per sample from the cluster frequencies and holds it through the SDE and the quench | `latent_model` (loss / force take (X, one-hot labels)), `train_latent`, `sample_latent` | tests 4272529 / 4272680: without labels bitwise as before (loss, force, 3 training slices vs the old code); label offsets = shifted biases per label (1e-14); the field term exact; the sampler drift = the training force per sample with its label (1e-15); end to end train, sample, quench |
| 11-18 | `oracle_source_check.py` (new tool) | | job 4272175 |
| 11-18 | `sample_latent --cond-scale LAMBDA` (a --cond run): every label parameter x LAMBDA in the drift and the quench (guidance by label amplification); suffix `_cs<LAMBDA>` | `sample_latent` | jobs 4273311-16: the same drawn labels as x1 (label files identical) |
| 11-19 | `--patch2-on-slice K2`: the second patch set's own switch-on slice (the coarse set early, the fine set late); config `patch2_on_slice` only when set | `train_latent` | test 4280068: regression bitwise; switch-on slices as asked (resume before the second switch-on untested) |
| 11-19 | corrector (`--corrector`) allowed for `--cond` runs (labels drawn per sample and held) | `sample_latent.sample_corrector(es=)` | test 4282746: labelled corrector runs; its unlabelled regression is not bitwise because the sampler itself is not (test 4284079, see 4) |
| 11-19 | `--no-patch-quad`: patch units couple linearly only (no Qp); config `patch_quad: false` only when set | `train_latent` | test 4284150: regression bitwise; history without Qp, switch-on slices right |
| 11-19/20 | `compress_history.py` (new tool; knots / svd / least-squares bases, per-group modes, `--knot-share`) | | checks 4284210, 4284925, 4285185: full count reproduces the run exactly (svd to 5e-13); default reproduces earlier outputs bitwise; shared knots identical within a set |
| 11-19 | `plot_inception_fid.py`: 2 x 2 grid (MNIST, Fashion, CelebA-32, CelebA-64); 11-20 early-patch, corrector and knots 5/9/9 rows (figure widened to 15.5 in) | | |
| 11-20 | `--time-knots B\|list`: only the knots' parameters exist (one grid for all sets, switch-on slices forced in); each knot trained jointly on the slices of its interval through the linear crossfade with the frozen previous knot (100-step chunks at one slice each, shuffled cycles, the slices' usual steps, lr decay stretched); a switch-on knot trains the new set alone; resume restarts at the last knot; config `time_knots` | `train_latent` (`parse_time_knots`, `slice_lrs`, `train_interval`, `build_runner(knot=True)`) | test 4285924 (its test 2 wrong: Q is zero at slice 0), re-check by srun: crossfades to 1.4e-15; regression bitwise; resume test 4286495 bitwise |
| 11-20 | `--knot-interp step`: piecewise-constant pieces instead of the crossfade (w = 1); config `knot_interp` only for step | `train_latent` | test 4286513: linear path old vs new bitwise; every slice equals its piece |
| 11-20 | `--time-knots-sets global=B,patch1=B,patch2=B`: own even knots per parameter set; segments from the union of the knots; every open knot trains over its whole interval so far (slices behind a set's finished knots enter with that set fixed); slices written after all knots; resume retrains from slice 1; config `time_knots_sets` | `train_latent` (`train_all_knots`, `build_runner(knot="perset")`) | test 4287622: shared and plain runs bitwise; per-set crossfades 2.8e-15, step pieces exact; resume bitwise |
| 11-20 | `compress_history.py --knot-slices LIST` (explicit kept slices for all groups) and `--mode steps:B` (piecewise constant, mean per piece) | | jobs 4285926, 4286502: full count reproduces the run exactly; synthetic pieces |
| 11-21 | `--visible harmonic`: only the linear self spring k_lin x^2 / 2 + bias per pixel (k_duff = k_6 = 0, held at exactly zero, their gradients zeroed); k_lin starts at 1 and is not floored, so confinement against positive squared-pixel couplings Q of the hidden units is not guaranteed (watch the samples' divergence count); refused unless `--pair-terms none` or `lin` | `train_latent` | test 4290730: quartic path old vs new bitwise (3 slices, both patch sets); refusals; k_duff / k_6 / pair groups exactly 0 in all 100 slices; confinement margins; sampling. Backup `train_latent_pre_harmonic.py` |
| 11-23 | `sample_latent --corrector dwell:TAU` (the physical corrector): the segmented predictor of `--corrector M:SNR` (same segments and keys), and at every interior knot the device's overdamped Langevin dynamics with the parameters frozen at that slice, dx = -grad F(x; theta_k) ds + sqrt(2T) dW, for a fixed time TAU, solved continuously (solve_SDE, SRA1, the predictor's tolerances; Brownian key fold_in(key, 200000 + knot)); suffix `_dwell<TAU>`. `--corrector-log` (with M:SNR): eps / T of every SNR step (its equivalent dwell time) to `<stem>_epsT.npy` + a summary in the .json, samples unchanged. `--stem-suffix` now also without `--fast` | `sample_latent` (backup `../latent_physical_SGM_pre_ising_patch/sample_latent_pre_dwell.py`) | test 4305032: refusals pass; the dwell ran out of host memory (closure over the knot time recompiled per knot), fixed by passing the knot time as diffrax args; test 4306625: dwell -> 0 limit within the amplified noise, dwell 0.1 finite incl. quench, no memory growth; 4306626 + 4306691: old vs new M:SNR equal up to run-to-run GPU noise (2e-5; the same code twice on the same node differs by as much) |

## 4. Known pitfalls

- **Corrector sampling of K784 models needs 64 GB host memory** (2026-11-21: `--corrector 2:0.16` on MNIST / Fashion
  K784 + 16x16/8x8 patches ran out of memory at the default 32 GB of `run_gpu_neuronic.sh`, jobs 4290138 / 4290145;
  64 GB ran in 1 h). K16 / K128 fit in 32 GB.
- **The job wrappers must be submitted with `sbatch`.** `bash jobs/run_cpu_neuronic.sh ...` runs the job on the login
  node (it did once for two small gallery renders, 2026-11-21). Inside an evaluation script that is itself submitted
  with sbatch, `bash jobs/run_*_neuronic.sh` is correct (it runs in the allocation).

- **GPU memory for K >= 3072 (sampling).** The default sampler holds the float64 parameter history several times
  (K3072 on CelebA-32: 5.0-5.6 GB constants). Use `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, `--chunk 25`, 160 GB host
  memory (MNIST K6144, CelebA K3072). `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` alone was not enough.
  The same applies to the locked 64x64 recipe with 64 patch states (hidden history 7.2 GB; job 4289731 sampled with
  `--chunk 100` but its quench failed allocating 7.2 GB with the default allocator, 2026-11-21).
  CelebA-64: K1536 needs the same (12 GB per copy); K3072 (22 GB per copy) additionally needs
  `--history-dtype float32`. Quench-only no longer builds a second copy (11-17).
- **Parameter histories above 2^31 elements.** One array of 2^31 or more elements crashes XLA with a segmentation
  fault (CelebA-64 K3072 + 16x16: 2.79e9, job 4266810). Since 11-17 `sample_latent` interpolates such histories in
  column blocks. Each block is a separate constant in the compiled drift, so forcing many small blocks
  (`LATENT_INTERP_MAX_ELEMS`) costs a lot of host memory: 547 blocks of a 2.7e8 history exceeded 32 GB.
- **Solver tolerance.** The reverse SDE is adaptive (SRA1, rtol 1e-4, atol 1e-6, steps onto the knots). It is
  converged at these values (2026-11-17): 10x tighter changes each sample by ~1e-6 of the distance between samples,
  at 3-4x the cost. Image defects are not solver artefacts.
- **Frozen baselines with overlapping patches.** Holding patch units at their full k-means reference counts each
  pixel about 16 times (stripes, saturation); use `--patch-fixed-scale 1/coverage`.
- **Disk.** Each run stores its parameters twice (`slices/` and `hidden_history.npz` + `params_history.npy`):
  CelebA-32 K3072 about 10 GB per run, CelebA-64 K1536 about 25 GB. The share is 1 TB since 2026-11-16; check
  `df -h /n/fs/cyrills-part` before large runs. Full-K smoke tests write full-size slices.
- **Patch speed.** The 16x16 patch set was the slow part (CelebA-32 K784: 1.3 h -> 5.3 h of training; CelebA-64:
  9x slower per step). The block unfold (11-16) makes the CelebA-32 patch slices 3x faster (patch term alone: 8x8
  every 2 px 4.6x, 16x16 every 4 px 2.3x). **On 64x64 it does not help:** 16x16 every 4 px gains only 1.2x, and
  8x8 every 2 px is 1.7x slower with blocks than with slices. MNIST's 7x7 every 3 px keeps the slice path.
  Runs that must continue bit for bit from older slices: `LATENT_PATCH_IMPL=slices`.
- **Run names with two patch scales contain a '+'.** Fine for the scripts since 11-16; take care in shell globs.
- **The equilibrated start and the quench.** `--init-dist` writes its own stem; quench it with the same
  `--init-dist` and `--quench` (it reads the existing file).
- **Code that runs while a job is queued.** Queued jobs read the code when they start; hold them
  (`scontrol hold`) while changing the code they use.

- **The GPU sampler is not bitwise reproducible run to run** (2026-11-19, test 4284079): the same command twice
  gives samples that differ by up to about 6e-5 (rms 2e-6) against sample spacings of about 25; harmless for every
  metric, but sampler regression tests must use a tolerance (about 1e-4), not bitwise equality. (Training is
  bitwise reproducible: every training regression so far matched bit for bit.)

- **Host memory at 64x64.** Sampling or quenching a CelebA-64 run needs 55-60 G of host memory (quench 4282764:
  53 G peak); at 48 G it is killed within minutes (corrector 4285143, 2026-11-20). Use `--mem 96G`.
- **Queued Slurm jobs keep the batch script of their submission.** Editing the script afterwards has no effect
  (the size-round Inception had to be cancelled and resubmitted after a script fix, 11-20). The Python code is read
  when the job starts (see above).
- **Test scripts can fail by themselves** (11-20, three times): `set -o pipefail` turns an intended error exit into a
  failed pipe; `set --` inside a loop overwrites the script's own arguments; a copied run keeps its name in
  config.json, which the config check refuses. Check the test's logic before the code.
- **Stiff quench after strong time compression.** A rank-2 SVD history (`compress_history.py --mode svd:2`) makes
  the t = 0 quench ODE stall (2 h and 40 min timeouts, reproducible; 2026-11-20); the sampling itself finishes.
- **GPU sampling is not bitwise reproducible run to run** (2026-11-23, job 4306691): the same code twice on the same
  node differs by ~2e-5 at n = 16 and ~2e-4 at n = 250 (CelebA-32, SNR corrector), small GPU differences amplified
  by the reverse SDE. Sampling regressions compare old vs new code against that noise, not bitwise.
- **Trajectory sets depend on n** for every problem except mnist28 (`lm.trajectory_set`: `jr.normal(k, (n, dim))`,
  `jr.split(k, n)`): the first 250 trajectories of an n = 2000 set are not those of an n = 250 set. Compare samples
  only at the same n (11-23: an invalid regression check).
- **Do not close over a traced value in a diffrax drift** (11-23): a closure over the knot time made the solver
  recompile at every knot until host memory ran out (50 GB in 5 min); pass such values as the solver's `args`.

## 5. Data on disk

- `data/CelebA/celeba_gray32_{train,valid,test}.npz` and `celeba_gray64_{train,valid,test}.npz` (uint8), and the
  parquet download cache `data/CelebA/parquet/` (about 12 GB).
  - The parquet cache and the 64x64 files were deleted on 2026-11-15 (disk full) and rebuilt on 2026-11-16
    (job 4263036; identical to the first build).
  - To rebuild: `make_celeba_data.py` downloads missing shards, then `CELEBA_SIZE=64 python make_celeba_data.py`
    builds the 64x64 files (4 min on 8 CPUs).

## 6. Open items

- 64x64 patch speed: the block unfold does not help there (see 4); try a convolution-based unfold, or choose the
  implementation per layout by measurement. (CelebA-64 with 16x16 patches turned out to train in 6.9 h anyway.)
- Contrast: the samples regress backgrounds and hair towards mid-grey (contrast -13% at 32x32, -23% at 64x64).
  - Colder sampling makes it worse; warmer restores it but costs 13 FID (11-17).
  - Global factor units do not help (11-18): a full-image Potts unit or 24x24 patches leave the contrast at 0.197
    (real 0.229) and the FID within 0.4.
  - **Where it forms (oracle-start scan, 11-18):** started from real noised faces at t <= 2.22, the samples keep a
    real-like dark-pixel share and coarse-scale power; started at t >= 2.98 or from noise, they lose them. So the
    deficit is made in the global stage (t > 2.2), before the patches act. Above t = 3.5 nothing is lost (an oracle
    start at t = 4 gives the normal FID); below t = 3 the FID falls steadily with later starts (21.8 at 2.98, 19.9 at
    2.22, 17.6 at 1.46, 14.5 at 0.71, 12.6 at 0.20).
  - The clamped per-image label (`--cond 32`, 11-18) gains 1.1-1.2 FID on K64 + 8x8 + 16x16 (24.61 -> 23.52 on the
    hidden biases, 23.42 with the visible field; about half of K784's advantage at about half its parameters), but
    hardly moves the contrast (dark pixels 6.1% -> 6.4% / 7.0%; real 12.7%). The field's large global-stage loss gain
    does not carry over to the samples (label drawn at sampling). The gain is the same with 8, 32 or 128 clusters
    (23.57 / 23.52 / 23.64): 8 clusters (0.05 M extra per slice) give all of it. Guidance by label amplification
    (`sample_latent --cond-scale`) only hurts (x1.5: 36.3 / 30.9; amplified hidden offsets wash the images out to
    flat grey, an amplified field pushes backgrounds to extremes). Untested: exact guidance and label dropout
    (theory note, section 6).
  - **Early coarse patches recover about half of it (2026-11-19/20):** the coarsest patch set switched on at
    t = 4.49 instead of 2.22: 32x32 image std 0.193 -> 0.213 (with corrector 0.217; real 0.229), dark pixels
    6.1% -> 9.3% (9.9%; real 12.7%); 64x64 0.177 -> 0.201 (real 0.234). It saturates from t ~ 3.5; an early 8x8
    set adds nothing. Time compression that starves t ~ 2-4 of knots loses contrast again.
- Model size: K has saturated with patch units (CelebA-32 K784 to K3072 about 1 FID, CelebA-64 K1536 = K3072), and
  the global unit can be cut to K64 for about 2 FID. The limit appears to be the model's form (one layer of
  independent template units whose forces add), not its size. Per slice, the patch units cannot shrink without loss
  (2026-11-20: 16 states +4.0, no Qp +8.1, sparser patches +7 to +12); the global unit can go to K16-K32 for free
  (K32 21.70 = K64). In time, 5 / 9 / 9 knots store 7.8x less at the same FID (see 2, "Parameter reduction");
  trained knots beat thinned ones at the same knots. Running (11-20 evening): trained per-set knots (5/9/9, 3/5/5,
  step), K16 with 64 patch states, no Q anywhere, K32 / K16 / 16 / 8 states at 64x64.
- Memorization (proposed 11-20, not started): nearest-neighbour distances of samples to the training set vs held-out
  images, and to the model's own templates (template reproduction), on existing MNIST and CelebA samples; then an
  MNIST training-set-size sweep and two models on disjoint halves (Kadkhodaie et al. 2024). Waits for the user.
- On-device training: the 3-point gap without patches (reproduced on seed 2, not reduced by S = 256) is unexplained;
  with patches it is 0.7. Candidates: the float32 draws, the same-draw visit counts.
- CelebA-64: K3072 gives 58.3, the same as K1536 (58.5), with the same contrast deficit (-23%), so K is not the
  bottleneck at 64x64 either. **Switching the 32x32 set on early (t = 4.49) gives 49.21 (2026-11-20)**, contrast
  0.177 -> 0.201 (real 0.234); the label adds nothing on top. The corrector takes it to **43.10** (-6.1; at 32x32 only -1.0), and
  the 5 / 9 / 9 time compression to 47.99 at 7.8x fewer stored numbers; both together **40.88** (11-20). Candidates: an earlier 16x16 set, finer patches (8x8 at 64x64 needs a faster patch code).
- Frozen vs trained (the dense-associative-memory question): done for MNIST ten digits and CelebA-32, with and
  without patches; learning the couplings is essential (CODE_AND_PERFORMANCE.md, 2026-11-16/17).
- Seeds 2 and 3 of the best CelebA-32 model (done for K64 + early 16x16: 21.86 +- 0.15 over 3 seeds).
- Metropolis-adjusted corrector (MALA): needs the free energy F in the sampler; only if the corrector helps.
