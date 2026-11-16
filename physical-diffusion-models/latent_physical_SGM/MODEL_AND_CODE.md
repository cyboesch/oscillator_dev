# Latent physical SGM: the model, the current recipe, and the code changes

Reference for the current state of the code (written 2026-11-16). It says *what* the model is and *which* settings
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
parameters in all (notebook, 2026-11-15: "Comparison: Wang & Achour"). Reducing the count is parked until the paper
is written (notebook, 2026-11-15: "Parked: reducing the number of parameters").

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

### Best models and command lines (seed 1)

Training + sampling (pipeline, from `latent_physical_SGM/`):

```
sbatch [--mem M --time T] jobs/run_pipeline_neuronic.sh PROBLEM TAG "EXTRA FLAGS" ema K N_SAMPLES ["SAMPLE ARGS"]
```

| Data | Model | Inception FID, quench 1 (n = 2000) | Classifier FID |
|---|---|---|---|
| MNIST, ten digits | K6144 + 7x7 every 3 px, equil start | 18.8 (FID_inf 15.9) | 3.87 |
| Fashion, ten classes | K1536 + 8x8 every 2 px, equil start, 3 seeds | 25.9 +- 0.04 (21.3) | 2.71 +- 0.07 |
| CelebA 32x32 grey | K1536 + 8x8 every 2 px + 16x16 every 4 px, noise start | 22.4 (18.6) | - |

Floors (real held-out images at n = 2000): MNIST 3.7, Fashion 7.2, CelebA 6.4. Wang & Achour: MNIST 27.6,
Fashion 80.8 (their n and FID code are not stated).

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
| `plot_inception_fid.py` | the summary figure `figures/inception_fid_mnist_fashion.{png,pdf}` (MNIST, Fashion, CelebA) | |

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

## 4. Known pitfalls

- **GPU memory for K >= 3072 (sampling).** The default sampler holds the float64 parameter history several times
  (K3072 on CelebA-32: 5.0-5.6 GB constants). Use `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, `--chunk 25`, 160 GB host
  memory (MNIST K6144, CelebA K3072). `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` alone was not enough.
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

## 5. Data on disk

- `data/CelebA/celeba_gray32_{train,valid,test}.npz` (uint8). The parquet download cache and the 64x64 files were
  deleted on 2026-11-15 (disk full); `make_celeba_data.py` downloads missing shards again, then
  `CELEBA_SIZE=64 python make_celeba_data.py` rebuilds the 64x64 files (4 min on 8 CPUs).

## 6. Open items

- 64x64 patch speed: the block unfold does not help there (see 4); try a convolution-based unfold, or choose the
  implementation per layout by measurement.
- Three patch scales (4x4 + 8x8 + 16x16) on CelebA-32: needs a third patch set.
- Seeds 2 and 3 of the best CelebA-32 model.
- CelebA 64x64 once the patch code is fast enough.
