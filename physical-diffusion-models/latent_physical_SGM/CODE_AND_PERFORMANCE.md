# How the code works, and where the time goes

Written 2026-11-11 as the starting point for speeding up training and sampling. All timings below are measured
from the run logs (`out/latent_physical_SGM/<run>/train_log.jsonl`, Slurm logs in `logs/`), not estimated, unless
marked as an estimate. Folder: `physical-diffusion-models/latent_physical_SGM/`.

## 1. What is being computed

A score-based diffusion model whose score is the force of a physical energy, so that it could in principle run on
hardware (paper: arXiv 2506.19136; this folder adds hidden units).

- **Data:** MNIST digits, 28x28 = 784 pixels, normalised (`latent_model.make_problem`, problems `mnist28_d0123`
  for digits 0-3, `mnist28_d0123456789` for all ten). 23,692 training images (four digits) or 54,210 (ten).
- **Forward process:** Ornstein-Uhlenbeck, t from 0 to 5, temperature T = 0.02. A training batch at time t is drawn
  exactly: x_t = e^{-t} x_0 + sqrt(T (1 - e^{-2t})) eps (`physical_diffusion_fns/helper_fns.sample_forward_process`).
- **Model at each time t:** a free energy F(x) of the pixels x (784 numbers):
  - visible part, `latent_model.DenseVisible`: per-pixel terms (k_lin x^2/2, k_duff x^4/4, bias x) plus pair
    terms between neighbouring pixels. Current setting (`--visible quartic --radius 1 --pair-terms lin`): the
    8 nearest neighbours, only linear springs c_lin (x_i - x_j)^2. P = 2,970 pairs; 4N + 3P = 12,046 parameters.
  - hidden units, averaged out exactly (so they never appear as variables in the simulation):
    - Potts (`--hidden potts`): one unit with K states, F_h = -T log sum_m exp(a_m / T),
      a_m = b_m + W_m.x + Q_m.x^2 (`--quad` adds Q). Parameters W, Q: (K, 784) each, b: (K,).
    - Ising (`--hidden ising`): K binary units, F_h = -T sum_j log 2cosh(a_j / T). Same shapes.
    - `--local`: one extra Ising unit per pixel coupled only to that pixel (784 + 784 parameters; cheap).
  - Formulas for grad F and the Laplacian of F: `potts_terms`, `hidden_terms`, `hidden_terms_quad`, `local_terms`
    in `latent_model.py` (derivations in `DESIGN.md` section 1).
- **Loss:** implicit score matching per sample, L = -lap F / T + |grad F|^2 / (2 T^2)
  (`latent_model.make_loss_per_sample`). The Laplacian is analytic (no Hessian, no Hutchinson estimator).
- **Training schedule (the paper's):** 100 time slices, t = 5 down to 0. Each slice is a separate optimisation,
  warm-started from the previous slice's parameters, so slices are strictly sequential. Slice k gets
  `lm.paper_steps(k)` Adam steps: 10,901 at t = 5 rising to 30,901 at t = 0, **2,095,000 steps per run** in all.
  Batch 512 fresh forward-process samples per step. Adam (optax `scale_by_adam`), per-group learning rates,
  decay 0.95^(step / 6000) within a slice. The slice result is an exponential moving average of the iterates
  (`--select ema`, time constant 1000 steps).
- **Sampling:** reverse SDE dx = [x - 2 grad F(x; params(5 - t))] dt + sqrt(2T) dW, solved with diffrax SRA1
  (adaptive, rtol 1e-4, atol 1e-6, VirtualBrownianTree tolerance 1e-12), parameters linearly interpolated between
  the 100 slice knots, the solver forced to step onto the knots (`sample_latent.py`).
- **Precision:** float64 everywhere (`jax.config.update("jax_enable_x64", True)` in `latent_model.py`), to match
  the paper's code exactly.

## 2. Code map (the parts that matter for speed)

| file | role |
|---|---|
| `latent_model.py` | constants (T, batch, schedules), data, `DenseVisible` (visible energy, gradient, Laplacian), hidden-unit terms, the loss, reference couplings (k-means / split tree, computed once at start) |
| `train_latent.py` | training. `build_runner` (line ~314) builds `one_step` and a jitted `jax.lax.scan` over n steps; the slice loop (line ~800) runs it in chunks of 100 steps, then validation, diagnostics and a checkpoint per slice (`slices/slice_kkk.npz`); at the end `params_history.npy` + `hidden_history.npz` |
| `sample_latent.py` | sampling. `drift` (line ~89) = visible force by autodiff of the paper's `energy_fn` + analytic hidden force; `vmap` over trajectories, chunks of 250 (`--chunk`) |
| `../physical_diffusion_fns/` | the paper's code: `helper_fns.sample_forward_process`, `network_fns.solve_SDE` / `solve_ODE` (diffrax), `setup_energy_fn` |
| `evaluate_multiclass.py`, `fid_score.py`, `make_gallery.py`, `prototype_test.py` | evaluation, CPU only, minutes |
| `jobs/run_pipeline_neuronic.sh` | train, then sample, then evaluate, then summarise, on one GPU |
| `jobs/run_gpu_neuronic.sh`, `jobs/run_cpu_neuronic.sh` | run any script of this folder on a GPU / CPU node (container) |

One training step (`one_step` in `build_runner`):
1. draw 512 forward-process samples X (512 x 784) at the slice's t;
2. `jax.value_and_grad` of the batch-mean loss with respect to theta, W, Q, b, wl, bl;
3. Adam direction, learning-rate decay, parameter update, projection (k_duff >= 0.01, dropped pair terms = 0);
4. best-loss and EMA bookkeeping (a tree_map over all parameters).

## 3. Hardware

Every GPU partition of this cluster (`all`, `interactive`, `lowprio`) has only NVIDIA **L40** GPUs, 8 per node.
Jobs use one GPU, 4 CPUs, 64 GB (`jobs/run_pipeline_neuronic.sh`). Container: apptainer image
`../../container/container_image.sif`, jax 0.4.33.

The L40 is a graphics/inference card: about 90 TFLOP/s in float32 but only about 1.4 TFLOP/s in float64
(1/64 of float32; NVIDIA data sheet figures).

## 4. Measured costs

### Training (100 slices, 2,095,000 steps; ten digits unless noted)

| run | steps/s | ms per step | pure training time |
|---|---|---|---|
| pixels only (no hidden units at all) | 130 | 7.7 | 4.5 h |
| K = 0 with per-pixel units (`--local`) | 126 | 8.0 | 4.6 h |
| Potts 64 | 114 | 8.8 | 5.1 h |
| Potts 784 (four digits: the same) | 61.7 | 16.2 | 9.4 h |
| Ising 784 | 60.6 | 16.5 | 9.6 h |
| Potts 3072 | 24.8 | 40.3 | 23.5 h |
| Ising 3136 | 24.8 | 40.3 | 23.4 h |
| Potts 6144 (running, job 4214805) | about 14 | about 71 | about 42 h |

- **Cost per step = 8.0 ms + 10.5 us x K**, for Potts states and Ising units alike, independent of the data set.
- Overhead outside the jitted steps (start-up, compilation, validation, diagnostics, checkpoints) is about 1-2% of
  the training wall time (Potts 3072: 23.75 h wall for 23.45 h of steps).
- `--val-every 1000` (learning curves) costs nothing measurable (61 steps/s with and without).

### Sampling (reverse SDE, `sde_base_n<N>`)

| run | 2,000 samples | per chunk of 250 |
|---|---|---|
| Potts 3072 (with Q) | 2.6 h | about 20 min |
| Ising 3136 (with Q) | 2.7 h | |
| Potts 1536 | 1.3 h | |
| frozen Potts 3072 (no Q) | 0.6 h | |

A pipeline with 500 samples spends about 40 min sampling at K = 3072. Evaluation and FID are CPU jobs of 0.5-6 min.

### Recompilation: checked, not a cost

- **Training.** The jitted functions are built once, before the slice loop: the training scan
  (`build_runner.run`, `static_argnums=3` = chunk length; only two lengths occur, 100 and the remainder 1, since
  every slice has 100 m + 1 steps), plus `loss_ps`, `loss_ps_k0` and `field_scale`. The slice time t, the learning
  rates and dt are traced arguments, so changing them does not recompile.
  - The time per step is identical to 0.1% from slice 1 to slice 99 although the step count rises from 11,201 to
    30,901: 40.27 ms (Potts 3072), 7.689 ms (pixels only). A fixed cost per slice would show up as falling time per
    step; it is below about 0.5 s per slice.
  - Only slice 0 carries the one-time compilation: about 19 s (Potts 3072), 24 s (pixels only).
- **Sampling.** The first chunk of 250 trajectories takes 70-80 s longer (Potts 3072) or 8.5 s longer
  (pixels only), and the following chunks are constant (1,131-1,153 s; 49.2-49.6 s). So compilation happens
  once per job.

**Sampling scales worse with K than training does.** A chunk takes 23x longer for Potts 3072 than for pixels only
(1,140 s vs 49 s), while a training step takes only 5x longer. This points at the per-trajectory interpolation of
W and Q in the drift (section 5), which grows with K and has no counterpart in training.

### Totals per model

Training dominates: about 95% of a pipeline's GPU time. A full ten-digit sweep (pixels only, K = 0, Potts 64-3072,
Ising 64-3136, 2,000 samples each) is about 130 GPU hours.

## 5. Where the time goes (estimates that match the measurements)

**Training is float64 matrix-multiply bound.** Rough FLOP counts per step (B = 512, N = 784):

- **Visible part, 7.7 ms per step.** `DenseVisible.grad_lap` builds three dense N x N matrices (C, U, D) from the
  pair list on every call and does six dense (B x N) @ (N x N) products: about 3.8 GFLOP forward, about 7.6 GFLOP
  with the backward pass. That is about 1.0 TFLOP/s, near the L40's float64 peak.
  - The pair graph is sparse: 2,970 pairs out of 307,000 possible, so the dense products are about 99% zeros.
  - With `--pair-terms lin`, U and D (the c_om and c_duff couplings) are identically zero but still multiplied:
    4 of the 6 products compute nothing.
- **Hidden part, 10.5 us per state per step.** `hidden_terms_quad` does six (B x N) @ (N x K)-sized products:
  A = X W^T + X^2 Q^T, the Laplacian's |G_j|^2 = |W_j|^2 + 4 X (W*Q)^T + 4 X^2 (Q*Q)^T, and the force p W + 2 X (p Q).
  That is about 4.8 MFLOP per state forward, about 14 MFLOP with the backward pass, about 1.4 TFLOP/s:
  again the float64 peak, to within the accuracy of this estimate.
- The per-pixel units, the Adam update and the EMA are elementwise and negligible at these sizes.

This is an estimate from FLOP counts. A profile (JAX profiler or Nsight on one short job) should confirm it before
optimising.

**Sampling** (not profiled): per drift evaluation and trajectory, `sample_latent.py`
- interpolates the *whole* parameter vector between two knots (at K = 3072 with Q: 2 x 3072 x 784 = 4.8 million
  numbers, per trajectory, since adaptive steps give every trajectory its own t);
- computes the visible force by autodiff of the paper's `energy_fn`, per trajectory (vmapped);
- uses a dense diffusion matrix `sqrt(2T) * eye(784)`;
- draws Brownian increments from a VirtualBrownianTree with tolerance 1e-12 (deep tree queries);
- runs in float64.

## 6. Candidate speed-ups, roughly in order of expected gain

1. **Float32 (or mixed precision).** Up to tens of times the matrix throughput on the L40.
   - Risks: T = 0.02 enters as 1/T and 1/T^2 (fields a/T of order 10^2-10^3 in the softmax, loss values about
     10^5 at t = 0), so check the loss, the gradients and the Laplacian against float64 on real batches at several t.
   - Possible middle ground: float32 only for the large hidden products, float64 for the accumulation of the loss.
   - TF32 tensor cores (lower mantissa) need their own check.
2. **Sparse visible energy.** Replace the dense N x N products by a neighbour stencil or gather (8 neighbours), and
   skip the pair groups that `--pair-terms` drops. This removes about 7.7 ms per step: about 2x for Potts 784
   (16.2 -> about 8.5 ms) and 1.2x for Potts 3072, and makes the K = 0 / pixels-only runs nearly free.
3. **Fewer wasted steps.** The convergence check (`README.md`, "Is training converged?") found slices 0-79
   (t > 1) flat for most of their budget, while slices 80-99 are still improving at the end. An adaptive stopping
   rule for the early and middle slices could save about half of the steps there. A test of more steps vs a larger
   learning rate on the late slices is running (jobs 4221035-38, `--init-from ... --init-slices 80`,
   `--steps-scale`, `--lr-scale`).
4. **Hidden products.** Fuse A and |G|^2 into one product of [X, X^2] (B x 2N) with stacked [W; Q]-derived
   matrices, and check whether JAX's backward pass recomputes or duplicates products.
5. **Sampling.**
   - Compute the fields from the two neighbouring knots and mix them (a = (1 - l) W_k x + l W_{k+1} x) instead of
     interpolating W and Q for every trajectory and evaluation.
   - Use diagonal noise instead of the dense identity matrix.
   - Revisit the 1e-12 Brownian tolerance.
   - Use an analytic visible force (`DenseVisible.grad_lap` already has it) instead of autodiff of `energy_fn`.
6. **Not useful:** splitting one run over several GPUs. Slices are sequential and the batch is only 512; the 8 GPUs
   per node are better used for independent runs (the current practice).

## 7. Constraints for any change

- **Exactness.** `test_latent.py` (53 checks: against the paper's code, the paper checkpoint's samples and autodiff)
  and `check_hidden_unit_formulas.py` must keep passing for the default path. A faster numeric path (float32,
  sparse visible energy) should be an opt-in flag first, validated against the float64 path. Then decide whether it
  becomes the default.
- **Reproducibility.** Runs resume from their last slice and must give the same result as an uninterrupted run;
  key streams follow the paper's per-step split. Keep the existing run folders readable: the trained sweep runs are
  the paper's data.
- **Cluster rules** (shared login node): no Python on the login node, not even a quick check; all computation via
  Slurm (`sbatch jobs/run_gpu_neuronic.sh SCRIPT ARGS` for a GPU, `jobs/run_cpu_neuronic.sh` for a CPU), submitted
  from `latent_physical_SGM/`; logs go to `logs/`.
- **Benchmarking.** A short GPU job is enough, e.g. `train_latent.py --problem mnist28_d0123 --K 784
  <the usual flags> --max-slices 2 --steps-scale 0.1 --name bench_<variant>`; the training log prints steps/s per
  slice. Compare loss and gradients of a new path against float64 on the same batch inside one job.
- **Disk.** Every slice checkpoint stores W and Q in float64: 38.7 MB per slice at K = 3072, so 3.9 GB per run
  (about 7.7 GB at 6144). Check `df` on the share before large sweeps.

## 8. Usual flags of the current model (for reproducing timings)

**Superseded (2026-11-16):** the current recipe (patches, equilibrated start, quench, two patch scales), the command
lines and the list of code changes are in `MODEL_AND_CODE.md`. The flags below are those of the 2026-11-11 timing
runs.

```
--problem mnist28_d0123 --K 784 --select ema --hidden potts --quad --local --visible quartic --radius 1 \
  --pair-terms lin --proto-init pooled            # Potts; Ising: --hidden ising, no --proto-init
```
Run names: `<problem>_K<K>_ema_potts_quad_local_quartic_r1_lin_pooled` (Potts),
`<problem>_K<K>_ema_ising_quad_local_quartic_r1_lin` (Ising), under `out/latent_physical_SGM/`.

## 9. Code analysis of the training step (2026-11-11) and the opt-in fast paths

Read from the code and the logs; the numbers marked "estimate" are not measured yet (jobs below). Only the
visible-energy part was checked by an independent second reading.

### What one step really costs

- **Visible energy: 12 float64 N x N products** (6 in `grad_lap`, 6 in the backward pass forming the cotangents of
  C, U, D), 7.55 GFLOP, about 7.0 ms of the 7.7 ms pixels-only step (about 0.7 ms is RNG, Adam, EMA, loop).
  With `--pair-terms lin`, **5 of the 6 forward products** multiply exact zeros (not 4 as section 5 says), and so do
  their 5 backward products: 10 of 12 are wasted. The cost does not depend on the pair count (radius 14: 129 steps/s,
  radius 1: 130/s).
- **Hidden part: 14 products of size B x N x K** (6 forward: A, |G|^2, gbar; 8 backward: the four products with X or
  X^2 on the left need only the parameter cotangent). 28 B N = **11.2 MFLOP per state**, not 14. At the measured
  10.2-10.5 us per state this is 1.07-1.11 TFLOP/s, 76-79% of the float64 peak. 14 is the minimum for the exact loss
  and gradient: stacking [X, X^2] or a custom VJP saves kernel launches, not FLOPs.
- **Outside the products** (Adam, update, EMA, the `best_params` select, which runs every step even under
  `--select ema` and is never read there; RNG; `--local` costs 0.25 ms of kernel latency): 5-8% of a step today.
  They become most of the step once the products are fast.
- **Per-run overhead** (Potts 3072: 1,054 s = 1.2%) is mostly one-time: 677 s numpy k-means before slice 0
  (1,734 s at K = 6144), 277 s final assembly (re-reading 100 slices, writing `hidden_history.npz`), about 1 s per
  slice. At 10-20x faster steps it becomes 10-20% of a run.

### Measured (job 4222239, `speed_check.py` section 5; L40, full training step incl. sampling, Adam, EMA)

| ms per step | dense f64 (today) | dense_skip f64 | stencil f64 | dense mixed | **stencil mixed** | 100-slice run, stencil mixed |
|---|---|---|---|---|---|---|
| pixels only | 7.70 | 1.70 | 0.38 | 0.50 | **0.20** (38x) | 4.5 h -> 7 min |
| Potts 784 | 16.2 | 11.1 | 9.8 | 1.13 | **0.79** (20x) | 9.4 h -> 28 min |
| Potts 3072 | 40.3 | 34.1 | 32.9 | 2.53 | **2.22** (18x) | 23.5 h -> 1.3 h |
| Potts 6144 | 71.5 | 63.8 | 62.6 | 4.86 | **4.56** (16x) | 42 h -> 2.7 h |

The dense f64 column reproduces section 4 (7.7 / 16.2 / 40.3 / 71.4 ms). Loss + gradient alone is 0.59 / 1.73 /
3.48 ms (K 784 / 3072 / 6144): about 22-25% of a fast step is sampling and optimizer bookkeeping (Adam, EMA and the
unused `best_params` select), the next thing to trim. End to end through `train_latent.py` (2 slices, slice 1):
Potts 784 stencil + mixed 1,270 steps/s (was 62/s), pixels only stencil f64 2,684 steps/s (was 130/s).
Plus the fixed per-run costs (k-means start-up 1.5 / 11 / 29 min at K 784 / 3072 / 6144, final assembly).

**Accuracy (job 4222254, `speed_check.py 1,4`).** float32 (stencil + mixed, HIGHEST) vs float64 on 16 batches of
512 at trained slices of three runs (four-digit Potts 784 and Ising 784, ten-digit Potts 3072):
- error / batch-to-batch gradient noise: visible theta about 2e-6 (late Ising slices up to 1.2e-4); W, Q, b at
  most 3e-6 for t >= 2.5, 1e-4 at t ~ 0.5, 5-7e-4 (max 4e-3) at t = 0 for Potts 3072; rarely visited Potts states
  (occupation < 1e-4) at most 9e-4; local units about 1e-5. Cosines >= 0.99998.
- The errors of W, Q, b are random (bias z about 1). Those of the local units and of theta in late Ising slices are
  systematic (bias z about 4 of a possible 4) but only 1e-5 to 1e-4 of the noise.
- Per-sample loss relative error median 1e-7 to 6e-7.
- Only failure of the preset criteria: Potts 3072 slice 99, relative error of b 1.0e-3 (limit 1e-3); its
  error / noise is 7e-4, 14x inside the limit.
- **TF32 is XLA's default for float32 products on the L40**: without HIGHEST the field errors in a/T are about 1000x
  larger, up to 3.0 at t = 0 (HIGHEST: 0.016). `in_compute_dtype` sets HIGHEST; all 17 (stencil) / 29 (dense)
  products of the mixed loss are float32 at HIGHEST (section 3 of the check).
- Stencil vs dense (section 2): 1e-16 to 1e-15 on six lattices (Chebyshev r 1, 2, von Neumann r 1, 3, 14x14, 8x8
  all-to-all), loss equal, gradients 1e-16 to 1e-13, deterministic over recompilation. `dense_skip` is equal up to
  rounding but not bit-identical to dense.
- `test_latent.py` (job 4222242): 53 checks, 0 failed, so the default path is unchanged.
- Training: the first two slices of Potts 784 with stencil + mixed (`bench_mixst_K784`) log the same numbers as the
  float64 run to every printed digit (validation -19552.9 / -19564.2, |W| 3.759, |Q| 0.1813, 596.10 effective
  states, best step 4160).
- **Late slices (job 4222313, `..._pooled_late_mixst`):** slices 80-99 of the four-digit Potts 784 run retrained
  with stencil + mixed on top of its float64 slices 0-79, at 1,259 steps/s (float64: 62/s). Per-slice validation
  loss vs the float64 slices: mean difference +4.8 units over the 20 slices, largest +32 (slice 82), slice 99
  -146,298.5 vs -146,312.0 (+13.5, 0.009%; doubling the steps moved slice 99 by 17,000). The differences appear
  right after the warm start and shrink to about +-3 units, as rounding-level divergence does. Effective states
  agree within 1-4 (slice 99: 451.8 vs 447.9), states below 1e-6: 126-137 vs 127-140.

### Speed-ups, in order

1. **`--visible-impl stencil`** (`latent_model.StencilVisible`): the visible energy per lattice offset (4 offsets
   at radius 1), only the kept pair groups, no N x N matrices. Equal to the dense path up to rounding; only slices,
   pads, elementwise products and one unique-index gather, so deterministic in both passes. `dense_skip` keeps the
   dense products but leaves out the dropped groups (about 1.2 ms instead of 7.0). For the paper's radius 14
   (420 offsets) dense stays the better choice.
2. **`--precision mixed`** (`latent_model.in_compute_dtype`): the loss and its gradient in float32 with every
   product at HIGHEST precision; parameters, Adam, EMA, the data stream (drawn in float64 with the paper's keys, then
   cast) and validation stay float64. TF32 must be excluded explicitly: XLA may use it for float32 products by
   default, it is no faster than float32 on the L40 (both about 90 TFLOP/s), and its 11-bit significand gives field
   errors of about 0.1-0.3 in a/T at t <= 0.2. Expected float32 errors (estimate): field error in a/T about 1e-5
   at t = 2.5 and up to about 1e-2 at t = 0; per-sample hidden-gradient errors about 1e-5 for t >= 1, up to 0.5-4%
   at t ~ 0 (from the softmax-backward cancellation, |G|^2 / T^2 about 7e5), random rather than systematic, so a
   few % of the batch-to-batch noise at most. The forward cancellation in the Potts Laplacian
   (sum p |G|^2 - |gbar|^2) only affects the logged loss: reverse mode never uses that value. Fallback if the last
   slices fail: float64 for slices >= 95.
3. **Step schedule** (changes results): slices 7-30 decide which Potts states survive (four digits, Potts 784:
   lost states peak at 192 with 1x steps vs 88 with 2x, 140 vs 71 at t = 0; flat from slice 30 to 79 in both), so
   only **slices 30-69** can be cut. At 1/3 of their steps a run needs 1,536,307 steps (-27%); spending the saving
   on 2x steps in slices 80-99 costs +1% in total and should recover most of the 2x FID gain (20.3 -> 17.7).
   Slices 80-99 still improve at 4x steps (job 4221035, slice 80).
4. After 1-2: cache the k-means reference, assemble the history from memory, skip `best_params` under
   `--select ema`, run per-slice diagnostics on the GPU. `--val-every 1000` (float64 validation on 4,096 samples)
   would then cost about 4-5% of a run. *(Built 2026-11-11, see "Build results" below: k-means on the GPU, assembly
   from memory, and the `best_params` skip. The skip changes results at rounding level whenever there are hidden
   units, because XLA then fuses the step differently. So it is opt-in, inside `--step-impl lean`. Per-slice
   diagnostics were measured at 1.9% of a fast K = 784 run and stay in numpy.)*
5. Sampling becomes the larger part of a pipeline (2,000 samples at K = 3072: 2.6 h); see section 6, item 5.
   *(Built 2026-11-11: `sample_latent.py --fast`, 12x faster at K = 784 and 51-61x at K = 3072; see "Build results".)*
6. Option paths: float64 Gumbel-max Potts draws cost about 140 ps x S x B x K per draw site (S = 64, K = 3072: about
   14 ms per step); an inverse-CDF sampler would remove most of it. The jackknife rebuilds A and |G|^2 per part.

### Corrections to sections 4-7

- Section 7's benchmark (`--steps-scale 0.1`) recompiles inside the timed slice: slices 0 and 1 then have
  remainders 90 and 20, and `run` is jitted with a static step count. Use `--steps-scale 1 --max-slices 2` and read
  slice 1 (11,201 = 112 x 100 + 1 steps).
- Section 4, recompilation: "only two lengths occur" holds for `--steps-scale 1` only.
- Section 7, disk: a run also writes `hidden_history.npz`, a second copy of W and Q: about 7.7 GB per run at
  K = 3072 and 15.5 GB at K = 6144.
- Section 6.6: packing several runs on one GPU gives nothing in float64 (the products already run near the float64
  peak); it may help after the float32 path, mainly for K <= 784.

### New flags (defaults unchanged)

- `--visible-impl {dense, dense_skip, stencil}` (default dense) and `--precision {f64, mixed}` (default f64), both
  recorded in `config.json`; old runs resume as dense / f64. Both are in the `--init-from` free set, so late slices
  can be retrained with them on top of float64 early slices. `--precision mixed` refuses `--hidden-samples`,
  `--hidden-rule cd1`, `--hidden-norm occupancy` and `--hidden-floor` (their draws would change with float32 fields).
- Validation (`loss_ps`) and the paper comparison (`loss_ps_k0`) always use the dense float64 path, so validation
  losses stay comparable with every existing run.
- Added by the build work (2026-11-11 evening; details and job ids under "Build results" below). All are opt-in and
  recorded in `config.json` or the samples' `.json`:
  - `train_latent.py --step-impl {default, lean}`. lean skips the `best_params` select under `--select ema`, applies
    the theta projection and the k_6 zeroing as constant masks, and draws each chunk's batches before the scan. It
    changes results at rounding level when there are hidden units.
  - `--steps-schedule 'k0:f0,k1:f1,...'` sets a step factor per slice range.
  - `--ref-kmeans {cpu, gpu}` chooses where the Potts reference k-means runs.
  - `--assemble {files, memory}` chooses where the final history is assembled from.
  - `sample_latent.py --fast`, with `--precision {f64, mixed}`, `--brownian {tree, cached}`, `--brownian-tol`,
    `--visible-impl {auto, stencil, dense}` and `--stem-suffix`.
  - `jobs/run_pipeline_neuronic.sh` takes an optional 7th argument, SAMPLE_ARGS, which it passes to `sample_latent.py`.

### Checks submitted 2026-11-11

| job | what |
|---|---|
| 4222238 | CPU: compile and import the edited modules |
| 4222239 | `speed_check.py`: stencil / dense_skip vs dense on 6 lattices (gradient, Laplacian, energy, autodiff, Hessian trace; loss and gradient; determinism); HLO of the mixed loss (all products float32 at HIGHEST); float32 vs float64 loss and gradients on trained slices (four-digit Potts 784 slices 0/50/80/90/95/99, ten-digit Potts 3072 and four-digit Ising 784 slices 0/50/80/95/99; 16 batches each; with HIGHEST and DEFAULT precision); step timings for dense / dense_skip / stencil x f64 / mixed at K = 0, 784, 3072, 6144. Report: `out/latent_physical_SGM/speed_check/4222239/report.txt` |
| 4222240 | `train_latent.py` 2 slices, Potts 784, stencil + mixed (`bench_mixst_K784`): end-to-end plumbing and steps/s of slice 1 |
| 4222241 | `train_latent.py` 2 slices, pixels only, stencil (`bench_stencil_K0`) |
| 4222242 | `test_latent.py`: the default path is unchanged |
| 4222254 | `speed_check.py 1,4`: the float32 comparison again, with the theta gradient restricted to the entries training uses (4222239 compared the dropped c_om / c_duff / k_6 entries too, where the dense path computes discarded gradients and the stencil gives 0) |
| 4222313 | slices 80-99 of the four-digit Potts 784 run retrained with stencil + mixed on top of its slices 0-79 (`..._pooled_late_mixst`, about 20 min); compare the per-slice validation loss and state counts with the original slices 80-99 (replaces 4222243, which waited on 4222239) |

Float32 acceptance (per slice and parameter group, mixed vs float64 on the same batches): median relative gradient
error <= 1e-3, cosine >= 0.9999, error / batch-to-batch gradient noise median <= 1e-2 and max <= 5e-2.

### In progress (2026-11-11, 17:45)

- **Convergence study** (11 jobs, 4222697-4222707) on a frozen copy of the validated code:
  `physical-diffusion-models/latent_frozen_20261111/` (its README lists the questions, runs and job ids). First results:
  - Slices 0-30 at 2x steps on the fast path (`..._mixst_early2x_part`) reproduce the float64 2x run's state counts
    exactly: lost states 28 / 88 / 71 / 35 / 31 and effective states 402 / 445 / 490 / 548 / 562 at slices
    9 / 15 / 19 / 25 / 30 (1x steps: 29 / 179 / 192 / 147 / 116 lost). Slices 31-99 at 1x follow
    (`..._mixst_early2x`); its FID tells how much of the 2x gain (FID 20.3 -> 17.7) comes from the early slices.
  - Late slices at 1x steps, lr x4 (`..._late_s1_lr4_mixst`, 9 min) reproduce the float64 run 4221036 (2.6 h): slice
    99 validation -173,048 vs -173,151, 401 vs 399 effective states. Slice 99 still improves by -5,141 +- 110 over the
    second half of the slice: not converged even at lr x4.
  - From the float64 late runs: 1x steps at lr x4 beat 2x steps on validation (-173,151 vs -163,474 at slice 99);
    lr x16 is better still (-180,426) but collapses state use (116 effective states vs 448). FIDs pending.
  - When all jobs have ended: FID / KID paired against the paper-schedule run, galleries, learning curves.
- **Build work** (two parallel tracks, each tested on the GPU, then reviewed; finished and reviewed 20:40, see
  "Build results" at the end):
  a fast reverse-SDE sampler (`sample_fast.py`, opt-in: integration knot interval by knot interval with the knot
  matrices shared by all trajectories, analytic visible force, diagonal noise, no per-step progress callback; options
  for the Brownian tolerance, float32 hidden products and the chunk size), and training trims (`best_params` skipped
  under `--select ema`, further per-step trims, `--steps-schedule`, k-means of the prototype reference on the GPU,
  history assembly from memory, `test_latent.py` block 13).
- **Next (agreed):** a test of knots denser near t = 0 (t_k = 5 (1 - k/99)^1.5: last interval 0.005 instead of 0.05;
  learning rate and steps per slice as the paper's linear functions of t, +9% steps), after the build tracks; then a
  pilot pipeline and a plan for the new sweep from scratch.
- **Not planned:** continuous-time (joint-knot) training. Estimate: at best 1.5-3x fewer GPU-hours (no per-slice
  restarts, bigger batches; against that two knots per sample and no warm start), possibly none; runs 5-8x shorter
  only through multi-GPU data parallelism; its main promise would be quality between the knots.

### Update (2026-11-11, 18:50)

- **Late slices, first results** (four-digit Potts 784; slices 80-99 retrained on top of the paper-schedule run's
  slices 0-79, except where noted; validation loss of slice 99, effective and lost states at t = 0):

| slices 80-99 | val slice 99 | eff. states | lost states |
|---|---|---|---|
| paper schedule (float64) | -146,312 | 447 | 140 |
| lr x2, float32 | -162,470 | 430 | 144 |
| 2x steps in all slices (float64 run `_steps2`) | -163,475 | 478 | 71 |
| lr x4, float32 (float64 twin: -173,151 / 399 / 150) | -173,048 | 401 | 151 |
| lr x8, float32 | -180,203 | 299 | 239 |
| lr x16, float64 | -180,427 | 115 | 480 |
| 2x steps and lr x4, float32 | **-185,926** | 399 | 147 |
| slices 0-30 at 2x steps, 31-99 at 1x (float32) | -147,025 | 483 | 72 |

  - lr x2 equals 2x steps on the validation loss: the late slices are limited by how far the parameters move.
  - Beyond lr x4, Potts states die (239 lost at x8, 480 at x16).
  - 2x steps at lr x4 gives the lowest loss of all, without extra state loss.
  - 2x steps in slices 0-30 alone keeps the states of the full 2x run (72 lost vs 140) but not its late loss.
  - Whether loss or state survival matters more for the samples: FID (4223118), galleries
    (`out/latent_physical_SGM/convergence_study/gallery_partial`, 4223119) and learning curves (4223120) are queued.
- **Remaining study runs** (seeds 1-3, late 2x steps lr x1, late 4x steps) wait in the queue: the cluster is busy and
  the fair-share is low after today's GPU use; estimated start about 00:20.
- **Build work resumed** (workflow run wf_590fb894-7e5): the first attempt was cut off at 17:52 when the session ended;
  its files compiled and imported cleanly (job 4223109). The new builders continue from that state, then review and fix.
- **Galleries (19:00, job 4223119, `out/latent_physical_SGM/convergence_study/gallery_partial/`; compare.png = the
  same 16 trajectories in every model).** The late-slice variants (lr x2 / x4 / x8 / x16, 2x steps at lr x4) look the
  same as the paper schedule: same digit per trajectory, same grain, although their slice-99 validation loss is up to
  27% lower (-185,926 vs -146,312). The runs that changed the early slices (full 2x steps; 2x in slices 0-30 only)
  change the digit of some trajectories (e.g. column 12: 2 -> 3), consistent with state survival being decided early.
  So far the late-slice loss does not show in the samples; FID (4223118) pending.
- **FID (19:05, job 4223118, `out/latent_physical_SGM/fid/4223118/report.txt`; 500 samples each, paired against the
  paper schedule, FID 20.29):**

| run | FID | paired diff [95%] | recall (features) |
|---|---|---|---|
| 2x steps, all slices (`_steps2`) | 17.69 | -2.60 [-4.10, -1.28] | 0.472 |
| **2x steps in slices 0-30 only** (`_mixst_early2x`) | **18.56** | **-1.74 [-2.91, -0.51]** | **0.470** |
| late lr x16 (float64) | 19.35 | -0.94 [-1.14, -0.73] | 0.434 |
| late lr x8 | 19.70 | -0.59 [-0.73, -0.39] | 0.430 |
| late 2x steps + lr x4 | 19.76 | -0.54 [-0.66, -0.38] | 0.431 |
| late lr x4 (float32; float64 twin: 19.91, -0.38) | 19.89 | -0.41 [-0.52, -0.30] | 0.428 |
| late lr x2 | 20.08 | -0.22 [-0.27, -0.16] | 0.420 |
| paper schedule | 20.29 | | 0.420 |

  - *(Corrected 2026-11-12 00:10, with three seeds: the recall gain is seed noise. The FID gain of slices 0-30 alone
    is about 60% of the 2x gain and suggestive, not established. See "final seed spread" below.)*
  - **Two thirds of the 2x-steps gain comes from slices 0-30 alone**, and with it the whole recall gain
    (0.420 -> 0.470): more Potts states survive the early slices (72 vs 140 lost), so the samples are more diverse.
    Precision is 0.93-0.95 everywhere.
  - **The late slices are not converged, and it hardly matters:** every late variant is still falling at the end of
    slices 95 and 99 (learning curves, job 4223120: late slopes -6.5 to -13.4 SE at slice 99), and their validation
    loss is up to 27% lower, but FID moves by 0.2-0.9 and recall not at all. lr x16 collapses the t = 0 occupation
    (115 effective states) without hurting FID: the digit is decided earlier in the reverse process.
  - The float32 + stencil path gives the same FID as float64 (lr x4: 19.89 vs 19.91).
  - Implication for the new sweep (to be confirmed against the seed-to-seed spread, seeds pending): spend steps on the
    early slices (state survival), not on converging the late ones; late lr x2-x4 is a free small gain.
- **Late 2x steps at lr x1 (19:25, job 4222700, `_late_s2_lr1_mixst`):** slice 99 validation -162,299, 446 effective /
  141 lost states, the same as late lr x2 at 1x steps (-162,470) and as the full 2x run's late slices (-163,475):
  for the late slices, doubling the learning rate is worth doubling the steps. Its FID will be computed together with
  the seed runs (still queued).

### Build results (2026-11-11, 20:40; workflow run wf_590fb894-7e5)

Both tracks were built, tested on the GPU, and reviewed independently ("ship"; only low-severity findings, fixed at
20:55, see below). Defaults are unchanged everywhere. All numbers are from the L40.

**Fast sampler** (`sample_fast.py`, run as `sample_latent.py --fast`). It solves one knot interval at a time. Within an
interval, the hidden fields and forces come from one stacked knot matrix shared by all trajectories, mixed with each
trajectory's own interpolation weight. The visible force is analytic, the noise is diagonal, and there is no
progress meter. The step-size controller state is carried from interval to interval, so the step rule is the same
as before.

| seconds per chunk of 250 | old sampler | fast, exact (f64, tol 1e-12, cached) | **recommended** (mixed, cached, tol 1e-8) | recommended, chunk 1000 |
|---|---|---|---|---|
| four-digit Potts 784 | 233 | 54.6 (4.3x) | **19.7 (11.8x)** | 14x per sample |
| ten-digit Potts 3072 | 1,140 | 107 (10.7x) | **22.2 (51x)** | 74.4 ms per sample (61x) |
| pixels only | 49 | 24.5 (2.0x) | 17.2 (2.9x) | 3.6x |

- The old times come from the runs' sampling manifests (4206669, 4209952, 4209942); the fast times are from jobs
  4223374, 4223294 and 4223431.
- 2,000 samples at K = 3072 take about 2.6 min instead of 2.6 h (about 14 min at the exact settings). Compilation
  takes 8-10 s instead of 70-80 s.
- Solver steps are unchanged: about 4,140 per trajectory, 2.4% rejected. The Brownian tree is now 80-90% of a step.
  A tolerance of 1e-6 instead of 1e-8 is also validated and 17% faster.

Validation of the fast sampler:
- **Drift.** It equals the old drift to 3.2e-14 relative or better over 5 runs (4222745). The review added the dense
  radius-14 Ising run (4.4e-13) and Potts without Q (1.8e-14) in job 4223512.
- **Trajectories.** Fast vs old final states differ by about 1e-4 (median, p784). This is rounding amplified by
  the step-size control, not a different step rule:
  - scaling the old drift by 1 + 1e-15 does the same to all 16 of 16 trajectories, while the old sampler run twice
    agrees to about 1e-15 in most of them (4223512);
  - the interval loop run with the old drift reproduces the old single solve (4223512).
  - One of 500 trajectories at K = 3072 ends on another digit.
- **Samples.** The statistics of the samples are unchanged:
  - Potts 784, 500 samples (4223298): FID 20.294 for the old sampler and for all three fast settings, paired
    differences at most 1e-4.
  - Potts 3072 (4223460): FID 22.740 vs 22.741 / 22.742.
  - Multiclass rates are identical in both.
- **Reproducibility.** `--brownian cached` gives bit-identical Brownian values and solves (4223374). The fast sampler
  is bit-reproducible across GPU jobs (4223419); the old one is not.
- **Default path.** Without `--fast`, results are bit for bit as before on the CPU (4223250). On the GPU the paper
  checkpoint reproduces job 4187940 (4223294). Imports of all modules pass (4223293, 4223432).
- **Not yet checked:**
  - the recommended settings at K = 6144;
  - Ising runs at the distribution level;
  - chunk 1000 for anything but speed.

  So the first fast pipeline of the new sweep is a pilot: check its manifest (n_nonfinite, steps_mean,
  chunk_seconds), and compare its FID with an exact-settings sample set of the same model.
- **Limits.** The PF-ODE, `--quench` and `--snapshots` stay on the old path. `--brownian cached` uses diffrax 0.6.0
  internals and refuses other versions.
- **Recommended:** `--fast --precision mixed --brownian cached --brownian-tol 1e-8 --chunk 500` for 500 samples,
  `--chunk 1000` for 2,000. In the pipeline this is the 7th argument:
  `sbatch jobs/run_pipeline_neuronic.sh P TAG "EXTRA" "ema" "K" 2000 "--fast --precision mixed --brownian cached --brownian-tol 1e-8 --chunk 1000"`.

**Training trims** (stencil + mixed, `--step-impl default -> lean`, ms per step, job 4223241):

| K | 0 | 784 | 3072 | 6144 |
|---|---|---|---|---|
| default | 0.203 | 0.785 | 2.208 | 4.579 |
| lean | 0.191 (-6%) | 0.768 (-2%) | 2.068 (-6%) | 4.100 (-10%) |

- **How much lean gains.** Most of the gain is the `best_params` skip. The mask and chunk trims are within run-to-run
  noise at K = 784 and 6144 and help only at K = 0 (about 6%) and K = 3072 (about 2%).
- **lean is not bitwise equal to default.** With hidden units XLA fuses the changed step differently. A 2-slice run
  differs by 1e-8 relative in theta / W / b and 8e-7 in Q, with every logged digit equal. For scale, float64 vs mixed
  differ by 2-9e-8 / 3e-6. Pixels-only runs stay bitwise equal on the GPU.
- **Fixed costs per run:**
  - k-means of the Potts reference, numpy -> GPU: 77 s -> 2.2 s (K = 784), 655 s -> 21 s (K = 3072), about
    1,734 s -> 29 s (K = 6144). These are compiled-call times; the first call includes about 3-31 s of compilation.
    Assignments are identical and the objective agrees to 2e-16.
  - Final assembly: 65-175 s -> 3.4 s (K = 784) and 115-277 s -> 20 s (K = 3072).
    `params_history.npy` is byte-identical and `hidden_history.npz` member-identical (the zip time stamps differ).
- **End to end.**
  - 2-slice bench: 125.6 s -> 47.4 s wall.
  - Full 100-slice four-digit Potts 784 run with every trim: 29 min 12 s (4222895), against an estimated 32-34 min
    without them.
  - Estimated saving per run: about 21% at K = 3072 and 27% at K = 6144.
- **Tests.**
  - `test_latent.py` passes 71/71 on the GPU (4223241); block 13 alone passes 18/18 on the CPU (4223490).
  - With `--step-impl default`, a 2-slice run reproduces `bench_mixst_K784` bitwise (4222847, 4223241, 4223404).
- **Tried and reverted:** float32 Adam moments. The first version rounded b2 = 0.999 to 0.99900001, which biased
  every step by 6.5e-6. The unbiased version gave no gain (4223427).
- **Left as is:**
  - The Ising reference's 2-means stays in numpy (63-66 s of the 72-90 s Ising start-up).
  - The Potts reference's per-tau solves stay in numpy (4.5 / 16 / 35 s at K = 784 / 3072 / 6144).
  - Caching the reference per (data, K, seed) would remove both.

**Review findings, fixed at 20:55.** CPU check 4223539 passed: all 9 modules import, the pipeline script passes
`bash -n`, and block 13 passes 21/21. The five lean-vs-default cases agree to 4e-12 relative or better (mixed) and
4e-16 (float64).
- Sampler:
  - `--fast --quench` frees the fast sampler's GPU history (7.7 GB at K = 6144) before building the ODE table.
  - Docstrings now say that any rounding-level drift change moves nearly every trajectory by about 1e-4.
  - `sampler_check.py` section 6 now passes the GPU default-path check only if the median per-trajectory difference
    is at most 1e-3. Before, it passed whenever the original was not reproducible.
- `train_latent.py`:
  - Resume no longer refuses a different `--assemble`, which does not change results.
  - `build_runner(theta_masks=True)` without `n_pairs` raises a clear error.
  - The `--steps-schedule` help now says that with a schedule every slice is rounded to 100 m + 1.
- `test_latent.py` block 13:
  - The mixed lean-vs-default tolerance is 1e-5 instead of 1e-2.
  - Three new lean checks cover paths never run before: Ising units, `--select best`, and the sextic paper visible
    energy with its k_6 floor.
- `speed_check.py` creates its output folder in `main()`, no longer at import. Earlier import checks left five empty
  folders, `out/latent_physical_SGM/speed_check/{4223295,4223375,4223401,4223425,4223446}`; they can be deleted.

### Non-uniform time grid (built 2026-11-11 21:10-22:40; workflow run wf_b3987b57-654; CPU tests only)

**What it does.** `train_latent.py --time-grid SPEC` is opt-in; the default is `uniform`, bit for bit as before.
- **Specs:**
  - `power:P`: t_k = 5 (1 - k/99)^P.
  - `power:P@K`: the uniform slices 0..K exactly, then the power law below t_K.
- **Per-slice learning rate and steps.** Each slice uses the paper's linear functions of t at t_k (np.interp, exact
  at t = 0 and 5). `--steps-scale`, `--steps-schedule` and `--lr-scale` act on top, by slice index.
- **Everything else per slice follows the grid:** data, validation, hidden-unit references, logs, hidden_history
  't', and the paper comparison (interpolated in t).
- **Recording and checks:**
  - The grid is recorded as `config.json` 'time_grid' (missing = uniform).
  - Resume refuses another grid. `--init-from` accepts another grid only where the linked slices sit at identical
    times (added after review, see below).
- **Samplers.** Both samplers (default and `--fast`, the SDE, the PF-ODE and `--quench`) take the knots from the
  run's config. They cross-check them against hidden_history 't' and refuse a difference beyond 1e-12. The samples'
  .json records `time_grid` and `knots_t`.
- **Analysis scripts.** `summarize_training.py` and `plot_learning_curves.py` pair runs only at equal t, and
  `summarize_training.py --states` tabulates effective and lost states against t. `inspect_states.py` and
  `inspect_units.py` use the run's grid.

| | uniform (paper) | power:1.5 | power:1.5@40 |
|---|---|---|---|
| total steps | 2,095,000 | 2,292,900 (+9.5%) | 2,165,600 (+3.4%) |
| first / last interval | 0.0505 / 0.0505 | 0.0756 / 0.0051 | 0.0505 / 0.0066 |
| slice 30 at t | 3.485 | 2.909 | 3.485 |
| state-survival window t 3.49-4.65 | 24 slices, 352,824 steps | 17 slices, 250,417 steps (-29%) | as uniform |
| slices at t <= 1 / t <= 0.1 | 20 / 2 | 34 / 8 | 29 / 7 |

Validation (CPU, toy problem gauss2_8 and synthetic histories):
- **Unit tests.** `test_latent.py` block 14 passed 19/19 (4223624) and 22/22 (4223680). Uniform equals the old
  arrays bit for bit; power:1 equals uniform to 9e-16; power:1.5 matches hand values; the fast sampler's interval
  weights at power knots match jnp.interp, with drift equal to 7e-15.
- **Default paths.** Default training and sampling are bit for bit the code from before the change (4223634, parts
  U1 / U2).
- **Every existing run still loads.** All 186 finished runs under `out/latent_physical_SGM/` pass the new 't' check,
  bit for bit (4223647).
- **Resume and --init-from.** A power-grid run killed mid-way resumes at the grid's t. Resuming on another grid is
  refused, and so is --init-from across different knots (4223634, part P).
- **Samplers on a power run.** Both samplers use the grid's knots: the start lines and manifests show it, and fast
  vs default agree to 8.5e-7. A sampler that ignores the grid lands 0.05 off at a sample scale of 0.33
  (4223680, part S).
- **power:1.5@40.** Its slices 0-40 equal a uniform run's bit for bit (4223680, part T).
- **Analysis scripts.** Their text output on uniform runs is unchanged (4223697, part R).

**Reviews.** Two independent reviews (default-path safety; numerics and tests) both said ship: no high or medium
issues and 12 low ones. Fixed at 22:40. CPU job 4223837 checked the fixes: all modules import, block 14 passes 22/22
with the new spec cases, and the inspect_states header is correct on a power run and a uniform one. Job 4223844
checked the new `--init-from` rule, 6/6: identical linked knots are accepted, a fifth linked slice at another t is
refused, and slice 0 alone links on any grid. The fixes:
- `inspect_states.py` and `inspect_units.py` printed ", time grid 28" in every header: the lattice size had
  overwritten the grid variable. Renamed.
- The spec parser accepted typos such as `power:1_5` as P = 15, and fullwidth digits. It now takes ASCII decimal P
  only and ignores blanks anywhere.
- Knots closer than 1e-9 in forward or reverse time are now refused (MIN_KNOT_GAP; power:P alone allows P up to
  about 4.8). The fast sampler also checks the reverse knots.
- Power-grid knots are now computed with the C library's `pow`. numpy's vectorised power differs by an ulp between
  CPU types, so a run resumed or sampled on another node type could have recorded 1-ulp-different knots.
- **`--init-from` across grids is now allowed where the linked slices sit at identical times.** Example: a uniform
  run's slices 0..K for `power:P@K` with `--init-slices <= K+1`. This lets a `power:1.5@40` pilot reuse seed 1's
  slices 0-40, which makes its early slices identical by construction. It saves only about 28% of the steps
  (1.55 M of 2.17 M remain), because slices 0-40 are the short ones.
- Pairing messages now say how many slices are paired at equal t, instead of "t = 5 and t = 0".

**Guard.** Code that knows no grid would silently sample or analyse a grid run on the uniform knots. That includes
the frozen copy `latent_frozen_20261111/` and any older script. So: sample, summarise and plot grid runs only from
`latent_physical_SGM/`, and check that the samples' .json has `time_grid` before trusting any FID or gallery.

**Pilot (GPU; not submitted, waits for the seed spread and the user's go-ahead).** Compare against seeds 1-3:
- The paired validation loss at t = 0 (slice 99).
- Lost and effective states against t (`summarize_training.py --states`).
- FID and recall against the seed-to-seed spread. One pilot seed cannot resolve FID differences below about 0.9 or
  recall differences below about 0.06.

The options:
- **(a) `--time-grid power:1.5`** (as agreed). This tests the whole grid change, but it thins the state-survival
  window by 29% of its steps. A worse result could come from that thinning rather than from the late refinement, so
  watch the lost states. About 46 min per pipeline.
- **(b) `--time-grid power:1.5@40 --init-from <seed 1> --init-slices 41`.** This isolates "denser knots below
  t = 2.98" with slices 0-40 identical to seed 1 by construction. It trains only slices 41-99: about 23 min, plus
  about 9 min of sampling.
- **Expected effect.** The convergence study found late-slice changes move FID by only 0.1-0.9. So the main gain to
  look for is in the t = 0 validation loss and in sample sharpness (galleries), rather than FID.

### Convergence study, seed 1 (2026-11-11 21:21, job 4222701, `_mixst_seed1`)

- **What it is.** Seed 1 is the paper-schedule run itself (same seed 1), retrained on the float32 + stencil path. So
  it measures how far rounding alone moves a run; it is not an independent seed.
- **Same as float64 through slice 30.** The lost-state counts are identical at slices 9 / 15 / 19 / 25 / 30 (29 /
  179 / 192 / 147 / 116), and the validation losses agree to 1e-9 relative.
- **Small drift late in the run:**
  - slice 95: validation -36,954 vs -36,949, 130 vs 129 lost states;
  - slice 99: validation -146,392 vs -146,312, 441 vs 448 effective and 136 vs 140 lost states.
- **Timing.** Training took 0.52 h and the whole pipeline (training, 500 samples with the old sampler, evaluation)
  42 min.
- **What comes next.** Seeds 2 and 3 give the seed-to-seed spread. Seed 3 and the late 4x-steps run started at
  21:12; seed 2 is queued. One FID / gallery / curves job follows when all have ended.

### Interim seed spread (2026-11-11 22:00; seed 3 and late 4x steps done; superseded by "final seed spread" below)

Jobs: FID 4223675 (`out/latent_physical_SGM/fid/4223675/report.txt`), galleries 4223676
(`out/latent_physical_SGM/convergence_study/gallery_seeds/compare.png`) and learning curves 4223677
(`out/latent_physical_SGM/learning_curves/4223677/`). 500 samples each, with the same starts and noise in every
model.

| run | FID | paired vs paper schedule [95%] | recall (features) | lost states at t = 0 |
|---|---|---|---|---|
| paper schedule (seed 1, float64) | 20.29 | | 0.420 | 140 |
| seed 1, float32 + stencil (rounding twin) | 20.25 | -0.04 [-0.10, +0.02] | 0.418 | 136 |
| **seed 3**, float32 + stencil | **19.36** | **-0.93 [-2.88, +0.72]** | **0.482** | 120 |
| 2x steps in all slices (seed 1) | 17.69 | -2.60 [-3.92, -1.13] | 0.472 | 71 |
| 2x steps in slices 0-30 (seed 1) | 18.56 | -1.74 [-2.91, -0.51] | 0.470 | 72 |
| late 2x steps, lr x1 | 20.17 | -0.12 [-0.18, -0.07] | 0.416 | 141 |
| late 4x steps, lr x1 | 20.03 | -0.27 [-0.35, -0.16] | 0.422 | 134 |

Paired against seed 3:
- 2x steps: -1.67 [-3.47, -0.08];
- 2x steps in slices 0-30: -0.80 [-2.59, +0.91];
- seed 3 vs seed 1: -0.89 [-2.86, +0.67].

**Late 4x steps:** slice-99 validation -173,041, the same as lr x4 at 1x steps (-173,048). In the late slices, steps
and learning rate trade one for one.

What this means:
- **Rounding alone moves FID by 0.04** (seed 1 vs the float64 run). The fast training path is FID-equivalent.
- **One other seed moves FID by 0.9 and recall from 0.42 to 0.48.** That is as much as the early-slice 2x steps do
  (-1.74 against seed 1, but only -0.80 against seed 3; recall 0.470).
  - The paired bootstrap intervals cover only the sampling noise of fixed models.
  - Any two separately trained runs differ by a training draw. A different schedule decorrelates the training
    trajectory too.
  - So schedule effects on FID must be judged against the seed-to-seed spread.
  - Earlier claim not established: that two thirds of the 2x-steps FID gain comes from slices 0-30.
- **What stays clear: state survival.**
  - Lost states at t = 0 are 120-140 in the three 1x runs, against 71-72 with 2x steps early.
  - At slices 15 / 19 they are 166-192 against 71-88.
  - 2x steps early robustly keeps more Potts states alive. Whether that improves the samples beyond the seed spread
    is still open.
- **Late-slice changes are small but real.** They keep the seed and the early slices, so they compare like with
  like: -0.12 to -0.94 FID, no recall change.
- **Galleries.** Seed 3 changes the digit of several trajectories, as the early-2x runs do (for example column 12:
  2 -> 3 in seed 3, 2x steps and early 2x). The late variants keep every digit and shape of the paper schedule.
- **Learning curves.** Seeds 1 and 3 converge the same way: the validation gain from N/2 to N steps, averaged over
  the last fifth of t, is -864 vs -872. 83-84 of 100 slices are still improving by more than 2 SE.
- **Next.** Seed 2 adds one more independent run. A firm FID answer for a schedule likely needs 3-5 seeds per
  schedule, which is affordable now at 42 min per run. Not queued: GPU jobs wait for the user's go-ahead.

### Late 4x steps in float64 finished (2026-11-11 23:49, pipeline 4221035, `_late_s4_lr1`)

- **First real use of the edited sampler.** Its sampling stage was the first to use `sample_latent.py` after
  today's edits. It ran the default path ("time grid uniform"), wrote 500 samples in 505 s, and its manifest
  records time_grid 'uniform' and 100 knots. Evaluation and summary passed (exit code 0).
- **The float64 run against its float32 + stencil twin** (`_late_s4_lr1_mixst`, job 4222707):

| | float64 (17.3 h of training) | float32 + stencil (45 min) |
|---|---|---|
| slice 95: validation, lost states | -38,491, 128 | -38,494, 130 |
| slice 99: validation, effective / lost states | -173,168, 443.7 / 137 | -173,041, 449.8 / 134 |

  The fast path reproduces float64 to 0.07% of the validation loss, as the seed-1 twin did. Its FID goes into the
  final FID job, together with seed 2.

### Convergence study: final seed spread (2026-11-12 00:10; seeds 1-3 done)

Jobs: FID 4224086 (`out/latent_physical_SGM/fid/4224086/report.txt`), galleries 4224087
(`out/latent_physical_SGM/convergence_study/gallery_final/compare.png`) and learning curves 4224088
(`out/latent_physical_SGM/learning_curves/4224088/`). 500 samples per model, with the same starts and noise in every
model.

| run | FID | KID x 1e3 | recall (features) | lost states at t = 0 (slice 19) |
|---|---|---|---|---|
| seed 1, float32 + stencil (same seed as the float64 paper run: 20.29 / 148.2 / 0.420 / 140) | 20.25 | 147.9 | 0.418 | 136 (192) |
| seed 2 | 20.14 | 148.8 | 0.464 | 140 (205) |
| seed 3 | 19.36 | 136.2 | 0.482 | 120 (190) |
| **seeds 1-3, mean +- SD** | **19.92 +- 0.49** | 144.3 +- 7.0 | **0.455 +- 0.033** | 132 +- 11 (196 +- 8) |
| 2x steps in all slices (seed 1) | **17.69** | 128.8 | 0.472 | 71 (71) |
| 2x steps in slices 0-30 only (seed 1) | **18.56** | 133.7 | 0.470 | 72 (71) |
| late 4x steps, lr x1, float32 (float64 twin: 20.05, 0.424, 137) | 20.03 | 147.1 | 0.422 | 134 |

Paired differences [95%] (sampling noise of fixed models only):
- 2x steps against seeds 1 / 2 / 3: -2.56 / -2.45 / -1.67, all three intervals below 0.
- 2x steps in slices 0-30 against seeds 1 / 2 / 3: -1.69 / -1.58 / -0.80 (the last interval contains 0).
- Seeds against each other: 2 - 1: -0.11; 3 - 1: -0.89; 3 - 2: -0.78 (all intervals contain 0).
- Late 4x steps, float64 vs float32: +0.02 [-0.01, +0.04].

Conclusions:
- **2x steps improves FID beyond the seed spread.** 17.69 is 1.7 below the best of the three seeds and about 4 seed
  SDs below their mean. The SD comes from only 3 seeds, so its own error is large.
- **2x steps in slices 0-30 alone gives about 60% of that** (-1.4 against the seed mean, about 2.4 SDs; below every
  seed, but only 0.8 below the best). This is suggestive, not established. One or two more seeds of this schedule
  would settle it, at about 42 min each.
- **Recall: no effect is established.** The three 1x seeds span 0.418-0.482, which covers both 2x runs (0.470,
  0.472). The earlier statement that the early slices carry "the whole recall gain" was seed noise; this corrects
  the 19:05 entry. Precision is the same everywhere (0.934-0.947).
- **State survival is the robust mechanism.** 2x steps early roughly halve the lost Potts states: 71-72 vs 120-140
  at t = 0, and 71 vs 190-205 at slice 19.
- **Late slices.** Every late variant moves FID by only -0.1 to -0.9 (paired, same seed and early slices), with no
  recall change. Float64 and float32 agree.
- **Galleries.** Seeds 2 and 3 change the digit of a few trajectories against seed 1 (columns 11 and 12: 2 -> 3), as
  much as the 2x runs do. The 2x runs keep most of seed 1's digits and clean up some malformed ones (column 12). The
  late variants and the float64 / float32 twins look the same.
- **Learning curves.** All three seeds converge the same way: the N/2 -> N gain over the last fifth of t is -864 /
  -869 / -872, and 81-84 of 100 slices are still improving by more than 2 SE.
- **For the new sweep:**
  - Spend extra steps on the early slices, for state survival.
  - Late lr x2-x4 is a free small gain.
  - Judge every schedule against the seed spread: at 500 samples the SD is about 0.5 FID and 0.03 recall. Use 2-3
    seeds per configuration for differences below about 1.5 FID.

### Late 4x steps at lr x4, float64 (2026-11-12 00:55, pipeline 4221038, `_late_s4_lr4`; FID job 4221039)

- **Sampling.** The second pipeline to sample with the edited sampler: default path, "time grid uniform", 491 s.
  Exit code 0.
- **Validation loss.** Slice 99 reaches -198,407, the lowest of all runs (paper schedule -146,312, i.e. 36% lower),
  with 409 effective / 150 lost states.
- **FID.** 19.63, paired vs the paper schedule -0.67 [-0.81, -0.50], against late_s4_lr1 -0.42 [-0.51, -0.30];
  recall 0.432.
- **So the best-converged late slices still move FID by less than one seed SD (0.5).** This confirms the final
  seed-spread conclusion: late-slice convergence is not where sample quality is decided.

### Next experiments (2026-11-12, submitted at the user's go-ahead)

**Why.** The study puts the quality bottleneck at high t (about 3.5-4.7): the start of the reverse process, where
Potts states survive or die. It is not at t near 0. So the late-knot pilot (`--time-grid power:1.5`, which also
removes 29% of the steps from t = 3.5-4.7) is parked. Two cheaper experiments come first:

1. **Early 2x steps with seeds 2 and 3** (jobs 4226676, 4226677):
   - `--steps-schedule 0:2,31:1` with `--seed 2` / `--seed 3`. This is the same schedule as `_mixst_early2x` (seed 1)
     in one job: slices 0-30 run 200m + 1 instead of 200m + 2 steps, a negligible difference.
   - Same flags and code path as the study (stencil + mixed, `--step-impl default`, old sampler, 500 samples).
   - Runs `..._pooled_mixst_early2x_seed2` / `_seed3`.
   - They pair with the 1x seeds 2 and 3, so early 2x vs 1x can be compared seed by seed for seeds 1-3.
   - About 50 min each once started.
2. **Knot-spacing check without training** (`sample_latent.py --thin-knots LO:HI`, new and opt-in on the default
   sampler):
   - Every odd slice with LO < t < HI is dropped, so the knot spacing there doubles (0.0505 -> 0.101). The parameters
     are interpolated across the gap and the solver no longer steps onto the dropped knots.
   - Samples go to `<stem>_thin<LO>-<HI>.npy`; the normal samples are never overwritten.
   - Applied to the paper-schedule run (`..._pooled`), same 500 starts and noise, in three ranges: `0:1` (9 slices
     dropped), `1:3.4` (24) and `3.4:5` (16). GPU job 4226682, after the CPU check 4226681.
   - **Reading it:** if doubling the spacing below t = 1 leaves FID and the galleries unchanged, a finer grid there
     cannot help either (linear-interpolation error scales as h^2). The other two ranges show where the knot spacing
     does matter.

### Idea parked for later (2026-11-12, the user's): Potts units on overlapping patches

The idea: instead of one Potts unit coupled to all 784 pixels, use several Potts units, each coupled to one patch,
with the patches overlapping. Not started; it waits for the knot-spacing and early-2x results.

- **Still exact.** The units are independent given x, so F = E_vis - T sum_j log sum_m exp(a_jm(x_Pj)/T).
- **Gradient and Laplacian.** They become sums over patches, with scatter-adds where patches overlap. That fits the
  stencil / mixed / fast-sampler machinery.
- **Cheaper than K = 784.** Example: 9 patches of 14x14 at stride 7 with 64 states each.
- **Open questions** (see the chat of 2026-11-12):
  - Global digit identity can now only arise through the overlaps. Keep one global Potts unit as well?
  - Weights per patch or shared?
  - Patch size, stride and states per patch.
  - Start on a smaller problem first, stepwise.

### Results: knot spacing and early 2x for seeds 1-3 (2026-11-12 11:00)

Jobs: GPU 4226682 (thinned knots), 4226676 / 4226677 (early 2x, seeds 2 / 3); FID 4226688
(`out/latent_physical_SGM/fid/4226688/report.txt`), gallery 4226689
(`out/latent_physical_SGM/convergence_study/gallery_next/compare.png`), curves 4226690.

**1. Knot spacing does not limit quality anywhere.** The paper-schedule run was sampled with doubled knot spacing in
one range at a time, paired against the full knots:

| knots dropped (spacing 0.0505 -> 0.101) | FID | paired vs full knots [95%] | recall |
|---|---|---|---|
| none (paper schedule) | 20.29 | | 0.420 |
| t < 1 (9 slices) | 20.36 | +0.06 [+0.04, +0.08] | 0.419 |
| 1 < t < 3.4 (24 slices) | 20.55 | +0.26 [+0.03, +0.58] | 0.413 |
| 3.4 < t < 5 (16 slices) | 20.51 | +0.21 [-0.05, +0.59] | 0.434 |

- **What a finer grid could gain.** The interpolation error scales as h^2, so halving the spacing everywhere would
  gain at most about 0.1-0.2 FID in total. Below t = 1 it would gain about 0.02. That is far below the seed SD (0.5).
- **Galleries.** All three thinned versions keep every digit and shape.
- **Decision.** The time-grid direction is closed in both directions. `--time-grid` stays as an option, but no pilot
  is needed.

**2. Early 2x steps (slices 0-30) helps for all three seeds.**

| | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| FID, 1x | 20.25 | 20.14 | 19.36 | 19.92 +- 0.49 |
| FID, early 2x | 18.56 | 18.74 | 18.29 | **18.53 +- 0.23** |
| paired early 2x - 1x [95%] | -1.69 [-2.79, -0.30] | -1.40 [-2.56, -0.50] | -1.07 [-2.53, +0.11] | **-1.39** |
| recall, 1x / early 2x | 0.418 / 0.470 | 0.464 / 0.470 | 0.482 / 0.510 | +0.029 |
| lost states at t = 0, 1x / early 2x | 136 / 72 | 140 / 73 | 120 / 72 | |
| lost states at slice 19, 1x / early 2x | 192 / 71 | 205 / 79 | 190 / 76 | |
| validation at slice 99, 1x / early 2x | -146,392 / -147,024 | -146,488 / -147,121 | -146,826 / -147,265 | |

- **The FID gain is established.** All three seed-paired differences are negative, -1.39 +- 0.31 on average
  (t = -7.8 with 2 dof). That is about 55-65% of the full 2x-steps gain (seed 1: -1.69 of -2.56), for about 21%
  more steps instead of 100%.
- **Recall gains a little,** +0.03 on average with all three positive, but this is within the 1x seed spread.
- **State survival is remarkably reproducible:** 72-73 lost states at t = 0 with early 2x, against 120-140 with 1x.
- **Less seed dependence.** The early-2x FIDs scatter half as much as the 1x ones (SD 0.23 vs 0.49). In the gallery
  the three early-2x runs agree on more digits than the three 1x seeds do (columns 11-12): with more states alive,
  the outcome depends less on the seed.

### How Potts states die in the early slices (2026-11-12 13:40; CPU job 4228312, `state_death.py`)

Script `state_death.py` (new). The data: 4096 fixed training images, noised with fixed noise at each slice's t.
Every state is followed through slices 0-40 and 99. Runs: the three 1x seeds and the three early-2x seeds. Report:
`out/latent_physical_SGM/state_death/4228312/report.txt`; pictures `dying_<run>.png` (dying states, their W before
death, and the states that take over).

| | 1x seeds 1 / 2 / 3 | early 2x seeds 1 / 2 / 3 |
|---|---|---|
| dead at slices 10 / 15 / 19 / 30 / 99 | 56/179/192/116/132, 49/193/206/122/141, 55/167/190/107/118 | 49/88/71/31/58, 42/92/80/34/68, 44/92/76/36/72 |
| alive at slice 5, dead at slice 30 | 116 / 122 / 107 | 31 / 34 / 36 |
| death slice, p10 / p50 / p90 | 8-9 / 11 / 14 (t about 4.3-4.6) | 6-7 / 9 / 11-14 |

- **Starving, not merging.**
  - No dying state is a near-copy of its winner. Its W has cos 0.4-0.65 (median) with the state that takes its
    data, below the typical nearest-neighbour cos of living states (0.79-0.82).
  - The data near it go to many states: the top one takes only about 10%.
  - It is beaten by about 9 T on all data, a fairly uniform gap (SD about 1.4 T).
  - Not one dying state used the merging/shadowing route (concentration >= 0.8 and cos >= 0.9).
- **The dying states never had a niche.** At slice 5 (t = 4.85) none of them wins a single one of the 4096 images
  (argmax). Their mean occupation is already about 10x below the others' (log10 -3.96 vs -3.03, seed 1). At that
  noise only about 200 of the 784 states win any image at all.
- **In the pictures,** the dying states are sharp, specific digit shapes (a particular 2, a particular 0). The states
  that take their data are smooth, generic class templates. At t of about 4.5 the data only support broad templates.
- **They are not frozen.** After death their W still moves as fast as the living states' (relative change 0.08 per
  slice vs 0.084). Per-element Adam divides out their tiny gradients, so they keep taking full-size steps but never
  win data back. So a per-state step rule (the visit-count rule) is unlikely to help: the dead states already get
  normal-size steps.
- **Revival happens in one window.** Between slices 19 and 30 (t 4.0 -> 3.5) at least 76 of the dead states come back
  by themselves (dead 192 -> 116, seed 1), as the data begin to support more modes. Almost none come back after
  slice 30 (2-6 by slice 99).
- **Early 2x saves the same states.** With the same seed and initialisation, 85 / 88 / 71 of the 116 / 122 / 107
  states that die at 1x are alive at slice 30 with early 2x. The ones it cannot save were the most starved at slice 5
  (median log10 occupation -4.28 vs -3.86).
- **Interpretation.** At high noise the data support only a few hundred distinguishable modes. The other states are
  redundant there. They do not become harmless copies; they starve, because each starts as a distinct, specific
  prototype. Most of them would be needed at lower t, but by then few can come back. This looks like deterministic
  annealing in reverse: mixture components should coincide at high "temperature" (here t) and split as it falls.
- **Candidate fixes** (not built):
  1. **Free-energy-preserving splits.** Re-seed dead states as copies of busy states, both with bias b - T log 2.
     That leaves F exactly unchanged, so training is not disturbed, and the copies then specialise. Do this in or
     after the revival window (slices 20-40). This differs from the two earlier fixes that failed: the homeostatic
     bias floor, which diverged, and Ising re-seeding, a different unit type and mechanism.
  2. **Coarse-to-fine initialisation.** Start the redundant states as near-copies of the high-t winners, so that they
     share occupation instead of starving, and let them split as t falls.
  3. **More early steps.** Proven, but it costs steps.

### Free-energy-preserving splits of dead states (2026-11-12 14:00-14:40)

**What `train_latent.py --split-dead K1,K2,...` does** (opt-in; with `--split-spread s`, default 0.5):
- At the start of each listed slice, every state with mean occupation below 1e-6 becomes a copy of one of the busiest
  states, one dead state per donor.
- Both copies get the donor's bias minus T log 2, and W_j +- eps v.
  - v is the leading principal direction of the donor's posterior-weighted data.
  - eps is chosen so that the two copies' field difference has SD s T over the donor's data.
- The free energy changes only at second order (by -T log cosh).
- Recorded in `config.json`; in the `--init-from` free set.

**Unit test** (`test_latent.py` block 15, CPU job 4228381, 6/6):
- At spread 0 the score-matching loss is unchanged to every digit, and the free energy to 7e-16.
- At spread 0.5 all dead states come back, the donors are the busiest states, the loss changes by 5e-6 relative,
  and F only decreases.

**Test on the GPU** (jobs 4228382-85). Each run reuses a 1x seed run's slices 0-19, then trains slices 20-40 with
splits at 20 / 25 / 30 / 35 / 40.

| slice 40 | 1x seeds 1 / 2 / 3 | split, spread 0.5, seeds 1 / 2 / 3 | split, spread 2, seed 1 |
|---|---|---|---|
| dead states | 114 / 116 / 101 | **0 / 0 / 0** | 0 |
| effective states | 519 / 523 / 539 | **696 / 701 / 697** | 665 |
| validation loss | -19,135.0 / -19,133.6 / -19,137.9 | -19,138.2 / -19,134.1 / -19,139.5 | -19,139.9 |

- **States stay alive.** Splitting re-seeds 192 states at slice 20 (seed 1). Between splits some die again (22-55 by
  slice 25-26 at spread 0.5, 0-8 at spread 2). The split at slice 30 re-seeds them (68 for seed 1), and after slice
  30 (t < 3.5) almost none die any more (0-3).
- **The validation loss is unchanged or slightly better** (slice 40: 0.5-3 units better for all three seeds; seed
  spread about 4 units). Slice 30 is 1.5 worse.
- **Next** (submitted 14:40): full runs with splits every 5 slices from 20 to 95, seeds 1-3 (pipelines 4228447-49:
  `--init-from` the 1x seed runs at slice 20, then sampling and evaluation). They are compared with the 1x seeds
  (paired by seed) and with early 2x: FID job 4228450, gallery 4228451
  (`out/latent_physical_SGM/convergence_study/gallery_split/`).

### Results: full runs with splits every 5 slices from 20 to 95 (2026-11-12 15:30)

Jobs: pipelines 4228447-49 (`..._mixst_seed{1,2,3}_split`; slices 0-19 linked from the 1x seed runs), FID 4228450
(`out/latent_physical_SGM/fid/4228450/report.txt`), gallery 4228451
(`out/latent_physical_SGM/convergence_study/gallery_split/compare.png`).

| | 1x seeds 1 / 2 / 3 | split seeds 1 / 2 / 3 | early 2x seeds 1 / 2 / 3 |
|---|---|---|---|
| FID | 20.25 / 20.14 / 19.36 (mean 19.92) | 19.60 / 19.75 / 19.07 (mean **19.47**) | 18.56 / 18.74 / 18.29 (mean **18.53**) |
| paired vs 1x, same seed [95%] | | -0.65 [-1.82, +0.30] / -0.40 [-1.85, +1.06] / -0.29 [-1.42, +0.94] | -1.69 / -1.40 / -1.07 |
| recall (features) | 0.418 / 0.464 / 0.482 | 0.476 / 0.478 / 0.480 | 0.470 / 0.470 / 0.510 |
| dead / effective states at t = 0 | 136/441, 140/436, 120/434 | 41/594, 43/597, 50/564 | 72/483, 73/479, 72/479 |
| validation at slice 99 | -146,392 / -146,488 / -146,826 | -145,839 / -145,901 / -146,141 | -147,024 / -147,121 / -147,265 |

Split against early 2x, paired by seed: +1.04 / +1.01 / +0.78 FID (split worse, all three).

What this means:
- **The split works mechanically but barely helps the samples.**
  - It keeps about 150 more states alive at t = 0 than the 1x runs, and about 100 more than early 2x.
  - FID improves by only 0.45 on average, every interval containing 0. Recall rises a little, within the seed spread.
  - **So the number of live states is not the bottleneck by itself.**
- **The late splits hurt the fit at t = 0.** The slice-99 validation loss is worse for all three seeds (+553 / +587
  / +685, against a seed spread of about 230). If splits are used at all, they should stop well before the last
  slices.
- **The gallery shows why.** Each split run keeps exactly the digits of its own 1x seed (split_s1 = seed1, and so
  on), while early 2x changes them. The digit a trajectory becomes is already decided by slices 0-19 (t >= 4.0),
  which the split runs take over unchanged.
- **Early 2x changes those slices.** Its gain cannot come from keeping states alive later. It must come from how
  slices 0-19/30 are trained. There the validation loss is already flat (second-half gains about 0 SE), yet more
  steps still change the state allocation: the loss is flat along how the states are allocated, and extra steps move
  along those flat directions.

### Next (2026-11-12 17:30): more steps in the decisive window, and what they change there

1. **GPU pipelines, seeds 1-3 each** (same flags as the study, old sampler, 500 samples):
   - `--steps-schedule 0:4,31:1` (4x steps in slices 0-30; `..._mixst_early4x_seed{1,2,3}`, jobs 4229131-33).
   - `0:2,20:1` (2x steps in slices 0-19 only; `..._mixst_early2x20_seed{1,2,3}`, jobs 4229134-36). This tests
     whether the window is slices 0-19 or 0-30.
2. **CPU analysis `early_slices.py`** (job 4229149). It compares plain, early 2x and full 2x runs at slices 0-30 (and a
   few later ones). The data: 2048 held-out test digits, noised at each slice's t with the same noise for every run.
   - The model's score (from the sampler drift) against the exact score of the noised training distribution, as a
     relative error. The validation loss hides this error behind a large constant.
   - The denoised (Tweedie) means, and which digit the FID classifier reads from them, against the exact posterior
     mean.
   - How the states are spread over the digit classes.

### Results: what more steps change in the early slices (2026-11-12 17:25, CPU job 4229149, `early_slices.py`)

The test data are 2048 held-out test digits, noised at each slice's t with the same noise for every run. The models
are compared with the exact posterior of the noised training distribution, computed from all 23,692 training images.
Report: `out/latent_physical_SGM/early_slices/4229149/report.txt`.

The denoising error is the relative error of the model's Tweedie mean (x + sigma^2 score) / e^-t against the exact
posterior mean. "Digit = exact" is how often the classifier reads the same digit from the two. Values are means over
three seeds.

| slice (t) | denoising error, plain / early 2x | digit = exact, plain / early 2x | states winning an image, plain / early 2x |
|---|---|---|---|
| 0 (5.00) | **1.351 / 0.139** | **0.829 / 0.952** | 324-352 (same init) |
| 3 (4.85) | 0.095 / 0.087 | 0.938 / 0.944 | 209-230 / 228-246 |
| 6 (4.70) | 0.070 / 0.061 | 0.928 / 0.941 | 163-188 / 204-227 |
| 9 (4.55) | 0.058 / 0.045 | 0.935 / 0.945 | 163-177 / 234-247 |
| 12 (4.39) | 0.052 / 0.039 | 0.926 / 0.938 | 202-204 / 296-305 |
| 15 (4.24) | 0.052 / 0.039 | 0.934 / 0.946 | 262-273 / 373-384 |
| 19 (4.04) | 0.064 / 0.052 | 0.938 / 0.944 | 360-386 / 486-490 |
| 25 (3.74) | 0.110 / 0.100 | 0.929 / 0.935 | 500-512 / 586-591 |
| 30 (3.49) | 0.163 / 0.154 | 0.945 / 0.948 | 550-566 / 611-617 |
| 40 (2.98) | 0.245 / 0.241 | 0.978 / 0.980 | about 575 / about 610 |

- **Slice 0 (t = 5) is badly underfitted with the paper's 10,901 steps.** Its denoising error is 10x larger than with
  2x steps (1.35 vs 0.14), and the digit read from its denoised mean disagrees with the exact one in 17% of cases
  instead of 5%. All three seeds agree to 1%.
  - Slice 0 is the only slice that starts from the reference initialisation instead of a trained predecessor.
  - Every later slice warm-starts from it, so its shortfall carries down the chain.
- **In slices 6-25 the error is 10-26% lower with 2x steps** in every seed, and the digit read agrees with the exact
  one about 1 point more often. By slice 30-40 the difference has faded (5%, 1.5%).
- **The validation loss cannot show this.** Most of the score-matching loss is the constant E|s*|^2, so the relative
  score error is about 1e-4 of it.
- **States.** With 2x steps, 30-45% more states win at least one image in slices 9-19. Class 1 gets the fewest states
  in every run (about 60-90 of 784).
- **Where the exact class information appears.** The exact class posterior goes from almost uninformative at t = 5
  (max p 0.40) to decided at t = 3.5 (max p 0.99). This is the window where the samples' digits are chosen, and the
  same window where the plain runs are underfitted.
- **Full 2x and early 2x are identical in slices 0-30** (as designed). Full 2x's extra FID gain (17.69 vs 18.56,
  seed 1) comes from slices 31-99.
- **Not meaningful below t of about 2.** There the exact score of the training set "memorises" individual training
  images, while the models generalise.

**Follow-up submitted (17:30):** 8x steps on slice 0 only (`--steps-schedule 0:8,1:1`, about 76k extra steps, 4% of a
run), seeds 1-3, pipelines 4229174-76 (`..._mixst_slice0x8_seed{1,2,3}`). It tests whether fixing slice 0 alone
recovers much of the early-2x gain.

### Oracle-start test (the user's idea, 2026-11-12 17:45)

**Option.** `sample_latent.py --oracle-start K` (new, opt-in, default sampler only) starts the reverse process at
slice K. The initial states come from the exact forward distribution, x = e^-t x0 + sqrt(T (1 - e^-2t)) xi:
- x0 are 500 training images with a fixed choice and noise (rng 12345), the same for every run and every K;
- from t_K down to 0 the model integrates; the Brownian keys are the usual ones;
- output goes to `<stem>_oracle<K>.npy`, never the normal samples.

**Reading it.** For a perfect model the samples would not depend on K, so FID against K shows how much error the
reverse process before t_K adds. Comparing plain with early 2x at the same K shows where early 2x gains.

**Jobs.**
- Starts K = 5 / 10 / 19 / 30 / 45 / 70 (t = 4.75 / 4.49 / 4.04 / 3.48 / 2.73 / 1.46).
- Runs: plain seeds 1-3 and early 2x seeds 1-3. GPU jobs 4229209-14, after the import check 4229208.
- FID 4229215: the 36 oracle sets plus the 6 normal ones, with early 2x vs plain paired per seed and K.
- Gallery 4229216 (`convergence_study/gallery_oracle/`, seed 1).

### Is t = 5 long enough? (2026-11-12 17:50, CPU job 4229465)

**The mismatch.** The sampler starts at t = 5 from N(0, T I) (`trajectory_set`). The true forward distribution there
still carries e^-5 of the data.

**Data** (normalised four-digit training set, 23,692 images): |mu|^2 = 234.5, trace C = 549.5, top eigenvalues 92.4 /
43.5 / 40.4 / 30.7 / 24.5; T = 0.02.

| t | mean offset along mu | variance excess, PC1 / PC10 | KL(p_t \|\| N(0, T)) |
|---|---|---|---|
| 4 | 1.98 sigma | 155% / 18% | 1.97 + 0.69 nats |
| **5 (used)** | **0.73 sigma** | **21% / 2.5%** | **0.266 + 0.019 nats** |
| 6 | 0.27 sigma | 2.8% / 0.3% | 0.036 nats |
| 7 | 0.10 sigma | 0.4% / 0.04% | 0.005 nats |

- **At t = 5 the start is measurably off:** 0.73 noise SDs along the mean image, and 21% too little variance along
  the top principal direction. KL is 0.29 nats, almost all of it from the mean.
- **Why it can matter.** The reverse dynamics shrink this error only by about e^-1 per unit time. So roughly a third
  of it is still there at t of about 4, where the digits are decided.
- **Direct test, sampling only:** oracle start at slice 1 (exact forward states at t = 4.95) against the normal start.
  GPU jobs 4229466 / 67, FID 4229468.
- **Fixes if it matters:**
  - Start from the exact Gaussian approximation N(e^-5 mu, e^-10 C + T (1 - e^-10) I). No retraining.
  - Or set T_forward to 6-7 (needs retraining).

### Overnight plan (2026-11-12 18:15; the user is away and asked to continue with the ideas)

- **K = 6144 ten-digit run** (pipeline 4214805):
  - Training finished, but its sampling was killed for host memory (64 GB; the old sampler holds the 6144-state
    history several times).
  - Resampling with the fast sampler, 2000 samples (GPU job 4229560). It also re-samples the K = 3072 run with the
    fast sampler as a cross-check against its old-sampler samples.
  - Then evaluate_multiclass + summary (4229561), FID with all ten-digit runs (4229562), and the gallery (4229563).
- **Is the Gaussian start "cheating"? No.** It needs only the data mean (784 numbers) and covariance (or its top ~20
  principal directions), computed once like any trained parameter. Sampling is then x_5 = e^-5 mu + L z: no data, and
  no forward process, at generation time. The equilibrated start needs nothing beyond the model: thermalise the
  oscillators under the learned t = 5 energy, then run the protocol. The oracle start (real training images) is a
  diagnostic only.
- **Build workflow wf_dd119390-5b4** (two tracks, one after the other; each built and CPU-tested, reviewed twice,
  fixed):
  1. `sample_latent.py --init-dist gauss | equil:S`: the exact Gaussian start, paired with the normal samples, plus a
     low-rank variant; or equilibration under the model's own slice-0 energy.
  2. Harmonic hidden modes, `train_latent.py --harmonic R`: R linear oscillators integrated out exactly,
     F_h = -1/2 sum_j (v_j.x + c_j)^2, with a closed-form per-t reference from the data's principal components (the
     exact Gaussian part of p_t). Plus `--harmonic-fixed`, sampler support and tests.
- **When results come in:** analyse the early-steps runs (FID 4229181), the oracle starts (FID 4229215, 4229468) and
  K = 6144. Then submit the start-distribution sampling (6 models) and the harmonic-mode pipelines (seeds 1-3), and
  record everything here.

### K = 6144 ten-digit run sampled with the fast sampler (2026-11-12 18:55)

**Jobs.** GPU 4229560, evaluation 4229561, FID 4229562 (`out/latent_physical_SGM/fid/4229562/report.txt`), gallery
4229563 (`gallery/28x28_all_ten_digits_sweep_number_of_hidden_elements/Potts6144_8nb.png`).

**Sampling.** The old sampler had been killed at 67 GB of host memory (pipeline 4214805, after 42 h of training). The
fast sampler (`--fast --precision mixed --brownian cached --brownian-tol 1e-8 --chunk 1000`) took **191 s for 2000
samples**: no non-finite samples, max |x| 3.86, about 4,150 solver steps per trajectory as at smaller K.

**Fast-sampler check on the ten-digit problem.** K = 3072, 2000 samples, fast vs old sampler: FID 22.48 vs 22.48,
paired -0.00 [-0.00, +0.00]; precision and recall identical. This is the first check at ten digits and 2000 samples,
after four digits at K = 784 / 3072.

| ten digits, 2000 samples | FID | KID x 1e3 | precision / recall (features) |
|---|---|---|---|
| Potts 784 | 31.10 | 204.8 | 0.904 / 0.296 |
| Potts 1536 | 25.95 | 173.9 | 0.912 / 0.340 |
| Potts 3072 | 22.48 | 153.8 | 0.924 / 0.384 |
| **Potts 6144** | **19.75** | **138.7** | **0.930 / 0.445** |
| real training images (floor) | 1.69 | 1.9 | 0.870 / 0.882 |

- **Doubling the states still pays at K = 6144:** -2.73 [-3.53, -1.72] vs 3072, and -6.20 vs 1536. Recall rises
  steadily with K (0.30 -> 0.45).
- **Gallery.** Most trajectories end on the same digit as with K = 3072; some malformed digits become clean (for
  example row 5, last column: a broken "5" becomes a "3").
- **Recommendation.** Sample every K >= 3072 with the fast sampler. The old sampler's per-trajectory parameter
  interpolation does not fit in memory at K = 6144.

### The start distribution matters: oracle start at slice 1 (2026-11-12 19:00; GPU 4229466/67, FID 4229468)

**Setup.** Each model was sampled from the exact forward states at t = 4.95 (slice 1): 500 training images
x e^-4.95 plus noise, i.e. a correct draw from p_4.95 (rng 12345, the same for every model). This was compared with
its normal start from N(0, T I) at t = 5. FID report: `out/latent_physical_SGM/fid/4229468/report.txt`.

| model | FID, normal start | FID, exact start at slice 1 | change | KID x 1e3, normal -> exact |
|---|---|---|---|---|
| plain seed 1 / 2 / 3 | 20.25 / 20.14 / 19.36 | 18.14 / 19.21 / 18.79 | -2.11 / -0.93 / -0.57 | 148 -> 112, 149 -> 118, 136 -> 115 |
| early 2x seed 1 / 2 / 3 | 18.56 / 18.74 / 18.29 | 17.04 / 17.29 / 17.58 | -1.52 / -1.45 / -0.71 | 134 -> 98, 139 -> 112, 129 -> 108 |
| mean, plain / early 2x | 19.92 / 18.53 | 18.71 / 17.30 | **-1.20 / -1.23** | |

- **The wrong start costs about 1.2 FID, as much as early 2x gains (1.39), and the two add up.** Early 2x with the
  correct start reaches 17.30 on average, close to the full 2x-steps run (17.69, seed 1) and below every plain
  configuration. KID drops by 20-35 units in all six models.
- **Caveat.** Normal vs exact start is unpaired (different initial states), and single changes have SD of about 1.5
  FID. But all six models move the same way, and the KID drop is large.
- **Starting at slice 5 instead of slice 1 changes nothing.** These two are paired (same images, noise and keys):
  +0.19 / +0.32 / -0.99 / -0.30 / -0.17 / +0.21, about 0 on average. So the model's own dynamics in t = 4.75-4.95
  are fine there; what matters is the distribution it starts from. The mismatch at t = 5 is 0.73 noise SDs along
  the mean image, 21% variance excess along PC1 and KL 0.29 nats (job 4229465).
- **Early 2x keeps its advantage with the correct start:** paired -1.10 / -1.92 / -1.21, mean -1.41, the same as
  with the normal start (-1.39). So the start and the early-slice training are two independent levers.
- **The legitimate version of this start** is the exact Gaussian start being built (`--init-dist gauss`: data mean
  and covariance only, no images at generation time), with a mean-only variant to separate the mean shift from the
  covariance. The equilibrated start is the other version. Both will be run on the same six models.

### Oracle starts across the whole reverse process: most error builds up at t of about 1.5-2.7 (2026-11-12 19:10)

**Jobs.** GPU 4229209-14, FID 4229215 (`out/latent_physical_SGM/fid/4229215/report.txt`), gallery 4229216
(`convergence_study/gallery_oracle/compare.png`, seed 1).

**Setup.** Exact forward states at slice K (the same 500 training images and noise for every model and K), then the
model down to t = 0. FID means over the three seeds:

| start | normal (t = 5) | K = 5 (4.75) | 10 (4.49) | 19 (4.04) | 30 (3.48) | 45 (2.73) | 70 (1.46) |
|---|---|---|---|---|---|---|---|
| plain (seeds 1/2/3) | 19.92 | 18.87 | 18.60 | 18.58 | 18.89 | 17.53 | **6.82** |
| early 2x (seeds 1/2/3) | 18.53 | 17.39 | 17.32 | 17.91 | 18.08 | 16.55 | **6.79** |
| paired early 2x - plain | -1.39 | -1.48 | -1.28 | -0.68 | -0.81 | -0.98 | -0.02 |

- **How to read it.** A perfect model would give FID near the floor (4.6) at every K. The curve shows where the
  model's own dynamics add error.
  - t > 2.7 adds only about 1.3 FID (K = 5 -> 45).
  - **t of about 2.7 -> 1.5 adds about 10.7** (K = 45 -> 70).
  - Below 1.5 only about 2.2 is left above the floor.
- **The gallery confirms it.** Starts at K <= 45 end in the familiar speckled, blotchy strokes; starts at K = 70 end in
  smooth digits. The grain and the malformed strokes form in slices 45-70.
- **Early 2x helps there too.** It is better by about 1 FID up to K = 45 (paired), because its later slices warm-start
  from better early ones. At K = 70 the two are equal.
- **The study never varied this window.** The step experiments covered slices 0-30 (early) and 80-99 (late). The
  earlier suggestion to cut slices 30-69 rested on state counts only and should not be followed until mid-slice
  quality is tested.
- **Submitted 19:15:**
  - 2x steps in slices 31-79 (`--steps-schedule 0:1,31:2,80:1`, `..._mixst_mid2x_seed{1,2,3}`, pipelines
    4230084-86). FID against plain and early 2x: 4230097, gallery 4230098.
  - Finer oracle starts K = 50 / 55 / 60 / 65 for plain seed 1 and early 2x seed 1 (4230087 / 88), FID 4230089,
    gallery 4230090 (`convergence_study/gallery_oracle_mid/`).

### Results: early 4x, early 2x in slices 0-19, 8x on slice 0 only (2026-11-12 19:25)

**Jobs.** Pipelines 4229131-36 and 4229174-76; FID 4229181 (`out/latent_physical_SGM/fid/4229181/report.txt`);
gallery 4229182 (`convergence_study/gallery_early/compare.png`).

| schedule (extra steps) | FID seeds 1 / 2 / 3 | mean | paired vs plain, mean | recall mean | lost states at slice 19 / t = 0 |
|---|---|---|---|---|---|
| plain (+0%) | 20.25 / 20.14 / 19.36 | 19.92 | | 0.455 | 190-205 / 120-140 |
| 8x on slice 0 only (+4%) | 20.29 / 20.25 / 19.30 | 19.95 | **+0.03** | 0.449 | as plain |
| 2x in slices 0-19 (+13%) | 18.64 / 18.94 / 18.98 | 18.85 | -1.06 | 0.483 | 71-79 / 64-79 |
| 2x in slices 0-30 (+21%) | 18.56 / 18.74 / 18.29 | 18.53 | -1.39 | 0.483 | 71-79 / 72-73 |
| **4x in slices 0-30 (+62%)** | **17.85 / 18.26 / 17.89** | **18.00** | **-1.92** | **0.493** | **19-21 / 51-55** |
| 2x in all slices (seed 1 only, +100%) | 17.69 | | -2.56 (seed 1) | 0.472 | 71 / 71 |

- **Slice 0 alone does nothing.** It is badly underfitted at the paper's steps (early_slices.py), yet 8x steps there
  changes neither FID nor the gallery: those rows are identical to plain. This fits the oracle starts, where the
  start distribution, not slice 0's fit, mattered. The early-steps gain comes from slices 1-30.
- **4x in slices 0-30 beats 2x** by 0.71 / 0.48 / 0.41 (paired, same seed), with diminishing returns: -1.39 for the
  first doubling, -0.53 for the second. It keeps almost every state alive through slice 30 (4-7 lost) and loses
  51-55 at t = 0.
- **2x in slices 0-19 gives about 75% of the 0-30 effect** for 60% of its extra steps. States lost in slices 20-30
  are partly regained by t = 0.
- **Gallery.** Early 4x keeps the digits of early 2x of the same seed, with a few strokes cleaner; the grain stays in
  every model. That fits the oracle curve: the grain forms at t of about 1.5-2.7 (mid slices), which these schedules
  do not touch directly.

### Finer oracle starts, K = 45-70 (2026-11-12 19:45; GPU 4230087/88, FID 4230089, gallery 4230090)

Seed 1, plain (p1) and early 2x (e1); exact forward states at slice K, then the model:

| K (t) | 45 (2.73) | 50 (2.47) | 55 (2.22) | 60 (1.97) | 65 (1.72) | 70 (1.46) |
|---|---|---|---|---|---|---|
| FID plain | 18.05 | 15.81 | 13.25 | 10.69 | 8.39 | 6.87 |
| FID early 2x | 16.22 | 14.72 | 12.60 | 10.26 | 8.21 | 6.84 |
| recall plain | 0.463 | 0.520 | 0.601 | 0.676 | 0.736 | 0.785 |
| paired early 2x - plain | -1.82 | -1.10 | -0.65 | -0.43 | -0.18 | -0.02 |

- **The error builds up steadily over t of about 2.7 -> 1.5,** about 2.3-2.6 FID per 5 slices. Over t > 2.7 it is
  about 0.03 per slice, over 2.7 -> 1.5 about 0.45 per slice.
- **Gallery** (`convergence_study/gallery_oracle_mid/compare.png`).
  - The later the model takes over, the more of the true digit's thin strokes survive.
  - With a handover at t = 2.7 the strokes come out thick, blotchy and speckled; at t = 1.5 they come out smooth
    and close to the source digit. Example: column 4, a cursive "2" with a loop: blob -> loop with speckles ->
    clean loop.
  - So at t of about 1.5-2.7 the model cannot hold the fine stroke structure the true state still carries. It pulls
    samples towards thick, template-like shapes plus pixel noise.
- **Likely cause: a missing kind of structure, not too few steps.**
  - At these t the digit class is fixed and the shape detail is being formed.
  - Each Potts state contributes one template (W_m) with per-pixel quadratic terms (Q_m).
  - Variation of strokes within a state can come only from the nearest-neighbour visible energy and the per-pixel
    bistable units, which give independent pixel noise: the grain.
- **Pointers this gives:**
  - More states help: ten digits, K = 3072 -> 6144: -2.7 FID.
  - The user's idea of Potts units on overlapping patches would add exactly the missing local, compositional stroke
    structure.
  - Harmonic modes add only global Gaussian directions.
  - The mid-slice 2x runs (4230084-86, queued) test the "too few steps" alternative.

### Start distributions built and submitted (2026-11-12 20:00; workflow wf_dd119390-5b4, track 1)

**Option.** `sample_latent.py --init-dist` (opt-in; default 'set' = today's start):
- **'gauss'** starts at t = 5 from N(e^-5 mu, e^-10 C + T (1 - e^-10) I). mu and C are cached once per problem in
  `out/latent_physical_SGM/data_stats/<problem>.npz`. The trajectory set's own draws are mapped onto it, so the
  samples stay paired with the normal ones. `--init-rank r` keeps the extra variance only along the top r
  principal directions; r = 0 is the mean shift alone (784 numbers).
- **'equil:S'** first runs S time units of Langevin dynamics under the model's own frozen slice-0 free energy. It
  needs no data at all.

**Tests.** `test_latent.py` block 16, 26 checks, passed on the candidate (4230022) and on the installed files
(4230110):
- the default path is bit for bit unchanged through the command line;
- mapped draws hit the target mean and covariance;
- the rank-r starts match the full start on the kept axes;
- the Langevin routine reproduces the exact Ornstein-Uhlenbeck transition on a quadratic energy;
- 20 invalid command lines are refused.

Installed atomically 19:07, import checks 4230100 and 4230160. Both reviewers: ship (low issues only).

**Experiment.** gauss / gauss with `--init-rank 0` / equil:3 for the six models (plain seeds 1-3, early 2x seeds 1-3).
GPU 4230539-44, FID 4230545 (paired against the normal start; the oracle slice-1 sets for reference), gallery 4230546
(`convergence_study/gallery_initdist/`).

### Results: the start distribution is worth about 1.5 FID, and no data is needed to fix it (2026-11-12 21:20; GPU 4230539-44, FID 4230545, gallery 4230546)

500 samples per set, paired against the normal N(0, T I) start (`t5`). Mean FID over three seeds (KID x 1e3 in brackets):

| start | plain | early 2x |
|---|---|---|
| N(0, T I) (today) | 19.92 (144) | 18.53 (134) |
| Gaussian, full covariance (`gs`) | 18.32 (122) | 17.22 (114) |
| Gaussian, mean shift only, 784 numbers (`gs0`) | 18.54 (124) | 17.34 (114) |
| equilibrated, 3 time units under the model's slice-0 energy (`eq3`) | **17.98** (114) | 17.36 (107) |
| exact forward states at slice 1 (oracle, `o1`) | 18.71 (115) | 17.30 (106) |

- **Every corrected start removes the whole start error.** Each one matches or beats the oracle start at slice 1.
  - Paired gs - t5: -1.76, -1.60, -1.43 (plain) and -1.38, -1.18, -1.38 (early 2x). KID p = 0.005-0.01 in all six.
  - eq3 - t5: -2.30, -1.89, -1.62 and -1.38, -1.57, -0.57. The intervals are wide because Langevin with fresh noise
    re-draws the trajectories, so the pairing with t5 is lost.
- **The mean shift is almost all of it.** gs - gs0 = -0.19, -0.35, -0.13, -0.04, -0.06, -0.27: every sign is
  negative, but each value is within noise. The covariance adds about 0.2.
- **It adds to early 2x.** e_gs - p_t5 = -2.70 = (-1.31 start) + (-1.39 early 2x). The two fixes act on different
  things.
- **What it changes: class balance, not strokes.**
  - The N(0, T I) start draws 27-28 % ones in all six models; every corrected start draws 23-25 % (the reference
    has 25 %).
  - Precision (0.94) and recall (0.45-0.46 in features) are unchanged.
  - In the gallery (`convergence_study/gallery_initdist/compare.png`), the Gaussian starts change the digit of 2 of
    16 trajectories (2 -> 3). Every stroke looks the same: still thick, blotchy and speckled.
  - The stroke problem sits at t of about 1.5-2.7 (oracle curve above). The start cannot reach it.
- **Cost.**
  - gs and gs0 need only the cached data mean, plus the covariance for gs. They cost nothing at sampling time.
  - eq3 needs nothing beyond the trained model. It adds 102 s to 497 s for 500 samples on the old adaptive sampler,
    about 2,500 accepted Langevin steps per trajectory. The trajectories have fully decorrelated by then (median
    displacement 5.5, against a typical norm |x| of about 4).
- **Answer to the question "is t = 5 long enough?"** For the forward process, yes: it reaches N(e^-5 mu, ...)
  closely. The sampler's N(0, T I) start ignores the e^-5 mu offset (0.73 sd along the mean direction), and that
  offset costs about 1.5 FID through class balance.
  - Either fix is legitimate. One stores 784 numbers (the data mean). The other runs the model's own slice-0
    dynamics, which needs no data.
  - Recommendation: make eq3 (or gs0) the default start for all further comparisons. Report it as part of the
    sampler.

### More global states: K = 1536 beats K = 784 by 3.5 FID on four digits (2026-11-12 21:45; FID 4230842/43, gallery 4230857)

- The four-digit K = 1536 run (`mnist28_d0123_K1536_ema_potts_quad_local_quartic_r1_lin_pooled`, float64, job
  4206670) is the same recipe as the K = 784 runs. It had been sampled on 11-10 but never scored.
- Paired against K = 784 (same start and noise):

| set | FID | KID x 1e3 | precision / recall (features) |
|---|---|---|---|
| K = 784, old float64 run | 20.29 | 148 | 0.941 / 0.420 |
| K = 784, mixst seed 1 | 20.25 | 148 | 0.942 / 0.418 |
| **K = 1536** | **16.84** | 121 | 0.948 / **0.534** |

- K1536 - K784: **-3.46 [-4.93, -1.74]**, p = 0.005. K1536 - mixst seed 1: -3.41.
- This is the largest single gain so far (early 2x: -1.39; start fix: about -1.5). It is not yet saturated:
  ten digits gained -2.7 from 3072 -> 6144.
- The mixst code reproduces the old float64 run: K784 (old) - mixst seed 1 = +0.04 [-0.02, +0.11].
- Gallery (`convergence_study/gallery_k1536/compare.png`):
  - K = 1536 strokes are somewhat thinner and less blotchy (e.g. the rings of the zeros), and the shapes vary more
    (recall +0.11).
  - Most trajectories keep their digit.
  - The pixel grain remains.
- Consequence for the patch-Potts test: the baseline is not only K = 784. Patch units must be compared with simply
  adding global states at similar cost.
  - Global units cost K x 784 couplings each for W and Q (1.2 M at K = 784).
  - Patch units on 7 x 7 windows at stride 3 with 32 states cost 64 x 32 x 49 = 0.1 M each.
- **Submitted:** K = 3136 with the same recipe on the mixst code, seed 1 (pipeline 4230861, resubmitted as 4231231:
  training, 500 samples, evaluation). This gives the next point on the K curve.

### Results: 2x steps in the middle slices barely help (2026-11-12 21:45; pipelines 4230084-86, FID 4230097, gallery 4230098)

2x steps in slices 31-79 (`--steps-schedule 0:1,31:2,80:1`, `..._mixst_mid2x_seed{1,2,3}`). This covers the window
where the oracle curve shows the error building up. Normal N(0, T I) start, 500 samples:

| | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| plain | 20.25 | 20.14 | 19.36 | 19.92 |
| early 2x (slices 0-30) | 18.56 | 18.74 | 18.29 | 18.53 |
| mid 2x (slices 31-79) | 19.66 | 19.16 | 18.96 | **19.26** |

- mid2x - plain: -0.59, -0.98 (p = 0.005), -0.40, a mean of -0.66. mid2x - early2x: +1.11, +0.42, +0.66.
  - Mid 2x spends more extra steps than early 2x (49 slices against 31) and gains half as much.
  - Training: 56-60 min per run, against 42 for plain.
- Recall 0.47 (plain 0.45, early 2x 0.48). Class shares unchanged.
- Gallery (`convergence_study/gallery_mid2x/compare.png`): the strokes look the same as plain. They are still thick,
  blotchy and speckled.
- **Conclusion: the error that builds up at t of about 1.5-2.7 is not a training-steps problem.**
  - The model lacks the structure to hold thin strokes there.
  - This leaves more states (K = 1536: -3.5) and local structure (the patch-Potts units) as the levers.

### Overnight status and plan (2026-11-12 22:00)

- **Harmonic modes** (workflow wf_dd119390-5b4, track 2):
  - Built and installed 21:17.
  - Tests: block 17 58/58 (4230780). Blocks 13-16 pass on the installed files (4230877, 4230781). Import check
    4230775.
  - Two reviewers are running.
  - GPU runs once the reviews pass, in the builder's order: `--harmonic 64` seeds 1-3, then `--harmonic 64
    --harmonic-fixed-slices 31` (modes held at the closed-form reference over t >= 3.48), then R = 16, then fully
    fixed modes (the builder expects these to be non-confining below t of about 3).
  - Each run is sampled with the normal start and with `--init-dist equil:3`.
- **Builder's analysis of the plain seed-1 run** (harmonic_analysis.py, 4230722):
  - At t = 5 the Potts layer already carries 93 / 53 / 23 % of the exact Gaussian anti-spring along PCs 1 / 16 / 64.
  - At slice 30 it carries 31 / 28 / 13 %.
  - So the Gaussian structure is missing mostly along the weaker principal directions, and more so as t falls.
- **Patch Potts** (the user's idea):
  - Design panel running (workflow wf_899ae7cd-383, read-only): three designers (a faithful integration, the learning
    dynamics including shared templates, and evaluation and physics including a fair comparison with more global
    states), then a judge who writes one spec.
  - The build starts after the harmonic track is closed, because both edit the same files.
- **Queued GPU work:**
  - K = 3136 pipeline 4230861, resubmitted as 4231231 with a 4 h limit (this site forbids changing a walltime).
  - K = 1536 with the gs0 / eq3 starts 4230862, FID 4230863.

### The start fix stacks with more states: K = 1536 + equilibrated start = FID 15.0 (2026-11-12 22:40; GPU 4230862, FID 4230863)

| start | K = 784 (mixst seed 1) | K = 1536 |
|---|---|---|
| N(0, T I) | 20.25 (KID 148) | 16.84 (121) |
| mean shift only, `gs0` | 18.69 (127) | 15.67 (100) |
| equilibrated, `eq3` | 17.95 (113) | **15.01** (91) |

- Paired K1536_gs0 - K1536_t5: FID -1.17 [-2.91, +0.71], KID -21 (p = 0.010).
- K1536_eq3 - K1536_t5: FID -1.82, KID -30 (p = 0.020). The interval is wide because equilibration re-draws the
  trajectories.
- The two fixes add up: K = 1536 with the eq3 start is 4.9 FID better than the plain seed-1 run with today's start.
  This is the best four-digit result so far. Recall is 0.54 (0.42 for plain).
- The start gain at K = 1536 (-1.2 to -1.8) is about the same as at K = 784 (-1.6 to -2.3). The start fixes class
  balance (ones 27 -> 24 %), more states fix shapes. They act on different things.

### Mid-t error: the model is converged there and capacity-limited (2026-11-12 22:50; CPU job 4231198, `early_slices.py`)

Relative denoising error against the exact posterior of the noised training set, on 2048 held-out images, seed 1
of each:

| slice (t) | plain | mid 2x | early 2x | K = 1536 |
|---|---|---|---|---|
| 45 (2.73) | 0.2715 | 0.2710 | 0.2684 | **0.2587** |
| 55 (2.22) | 0.2978 | 0.2986 | 0.2954 | **0.2878** |
| 65 (1.72) | 0.3147 | 0.3146 | 0.3122 | **0.3070** |
| 70 (1.47) | 0.3224 | 0.3225 | 0.3201 | **0.3166** |
| effective states at slice 70 | 430 | 433 | 450 | 612 (of 1536; 652 dead) |

- **Mid 2x changes nothing in slices 40-90** (differences of at most 0.001). Training at these t has converged; the
  limit is the model's capacity there.
- **K = 1536 lowers the error by 3-5 % (relative) in every mid slice** and keeps about 45 % more effective states.
- The relative change is much smaller than the FID gain (-3.46).
  - Caveat: this reference is the training set's posterior. At low t it is the memorising score, so the measure
    contains a floor that a generalising model must not reach.
  - It is a consistent ruler for comparing models at the same slice, not an absolute quality measure.
- Early 2x fixes slice 0 (denoising error 0.14 against 1.35) and gains about 0.003 in the mid slices.
- **Implication:** anything that improves mid-t quality must add capacity of a useful form there: more global states,
  or the patch units. This table is the baseline the patch runs will be measured against, with the same job script.

### Harmonic modes: review round closed, stage 1 submitted (2026-11-12 23:15; workflow wf_dd119390-5b4)

- **Review findings, fixed.** The maths reviewer found that modes built from the full data covariance C double-count
  what the Potts prototypes already carry: the 784 prototypes hold 95 % of the variance along PC 1.
  - Adding the full-C reference modes to the trained plain run makes the curvature along PC 1 at the data negative
    from slice 19 on.
  - **New opt-in `--harmonic-ref within`:** the reference is built from the within-cluster covariance S_w, the part
    the prototypes do not carry. It keeps 4.9 % of C's variance along PC 1, 48 % along PC 16 and 79 % along PC 64.
    The prototype unit plus these modes is then exactly the prototypes' Gaussian mixture (block 18: force to
    1e-14).
  - **New opt-in `--harmonic-lr-tmin 1`:** the modes' learning rate stops rising below t = 1.
- **Tests on the installed files (23:03):** blocks 13-18 pass (21, 22, 6, 26, 59 and 16 checks; jobs 4231443,
  4231503, 4231387, 4231388). Import check 4231375.
- **Stage 1 (seed 1 only, as the reviewer advised):**
  - `--harmonic 64 --harmonic-ref within --harmonic-lr-tmin 1`: pipeline 4231671.
  - The same with the modes held at the reference in slices 0-30 (`--harmonic-fixed-slices 31`): pipeline 4231672.
  - R = 16: pipeline 4231673.
  - Each is followed by equil:3 sampling (4231674-76), then FID 4231677 against plain and early 2x seed 1.
  - Seeds 2-3 come only if stage 1 is clean (finite, no runaway modes in train_log).

### Overnight go-ahead (2026-11-12 23:25, the user): build patch Potts; run the best recipe; then decide

- **Best recipe (option 3):** K = 1536 + early 2x (`--steps-schedule 0:2,31:1`) on the mixst code, seed 1
  (`mnist28_d0123_K1536_..._mixst_early2x_seed1`).
  - Pipeline 4231701, then equil:3 sampling 4231702, FID 4231703 (against K = 1536 float64 and plain / early 2x
    seed 1, each with the normal and the equilibrated start), gallery 4231704, redone as 4236962 (`galleries/2026-11-13_best-global-recipe_K1536+early2x_seed1/`).
- **Patch Potts (option 1):** build after the design panel's spec (wf_899ae7cd-383), with build, two reviewers and
  fixes.
- **If 1 and 3 are not promising, in this order:** `--split-dead` at large K (42 % of states are dead at K = 1536 by
  t = 1.5), sampler-side checks (PF-ODE, `--quench`), radius-2 visible couplings.

### Patch Potts: design panel result and build started (2026-11-13 00:00; workflow wf_899ae7cd-383)

**Panel.** Three designers worked from different angles (faithful integration; learning dynamics; evaluation and
physics). A judge checked their measurements and wrote one spec:
`out/latent_physical_SGM/patch_design/judge/final_spec.md` (scores, flags, plan and rejected ideas in `judge.json`).
Measurement jobs: 4231029, 4231053, 4231028, 4231064, 4231229, 4231394; the judge's own is 4231616.

**Key measurements:**
- **Potential.** Exact closed-form denoisers on held-out images, change in DSM against the 784-prototype mixture:
  - Additive global + per-patch residual templates (7 x 7 tiles, 64 states): -152 / -949 / -1650 at
    t = 2.73 / 1.97 / 1.46.
  - Doubling K (1536): -40 / -103 / -120.
  - So local templates carry far more of what is missing at mid t than more global ones.
  - Patch structure matters only below t of about 3.5. Above it, the patch templates cannot be told apart in the
    noise (d' < 0.5).
- **The physical form (product of experts) constrains the start.** It has no explaining-away:
  - residual templates are harmful (+930 DSM at t = 1.97);
  - the raw per-patch reference at full strength double-counts (2-11x the exact Gaussian anti-spring at t = 5);
  - the start must be weak: 0.25 x reference at slice 0, or 0.06 x at a switch-on at slice 31;
  - a small random start never breaks the symmetry.
  - The weak start only sets the direction. Each patch unit gets its own learning rate, from the scale of its full
    reference.
- **Grain metric (4231229).** Specks (ink components of at most 2 px) and ink components per image:
  - plain 0.52-0.54 and 1.84-1.90;
  - K = 1536 0.47 and 1.74;
  - oracle start at slice 70 0.05 and 1.09;
  - real digits 0.01 and 1.03.
  - Along the oracle curve they track FID with r = 0.99-1.00. More global states mainly raise coverage (recall);
    patches target the grain.
- **Cost.** 7:3:32 adds 200,704 W + Q couplings, the same as going from K = 784 to K = 912. On the CPU the patch
  path costs 3.5-5x the plain loss; the GPU cost is unknown until the first GPU test (E0).

**Spec:**
- `--patch SIDE:STRIDE:STATES` (7:3:32); `--patch-init-scale` (0.25); `--patch-lr-scale`; `--patch-on-slice K`
  (switch-on variant).
- Deterministic slices and pads, no scatter. Terms are added last.
- `load_params` refuses a patch run unless the caller takes the units; `--fast` refuses patch runs.
- Test block 19, plus a de-risk step: the integrated code must reproduce the judge's loss changes to 1e-6.

**Rejected:** shared (convolutional) templates, the hierarchical coupling, a gather/scatter implementation, a
full-strength or residual start, a small random start, one shared learning rate.

**Plan:**
- Build: a builder agent in `patch_dev/` (started 00:00).
- Then two independent reviews.
- Then GPU:
  - E0: cost and reproducibility;
  - E1 pilot seed 1: P-A (from slice 0) and P-B (from slice 31);
  - E2 seeds 2-3 only if the pilot meets the go rules (spec section 10): FID -1, or specks <= 0.40 with FID no
    worse.
- **Coupling-matched baseline submitted:** K = 912 seed 1 (pipeline 4231752, then equil:3 sampling 4231753).

### Patch Potts: built, tested, installed; E0 and the pilot submitted (2026-11-13 01:45)
**Build (builder agent, 00:05-01:40; installed 01:22:17 by `patch_dev/install.sh`).**
- Install method: each file checked against the pre-install snapshot `out/latent_physical_SGM/patch_dev/pkg/latent_orig`
  and against the tested sha256 list, then moved atomically into place.
- **Changed files:**
  - `latent_model.py`: `patch_terms`, `patch_free_energy`, `patch_reference`, `patch_lr_scales`, `patch_unit_stats`,
    `PatchGeometry`.
  - `train_latent.py`: the four flags; switch-on; per-patch learning-rate arrays; diagnostics.
  - `sample_latent.py`: `load_params(with_patch=)`, `make_drift(patch=)`, `init_start(patch=)`.
  - `early_slices.py`: patch rows.
  - The three check scripts.
  - `test_latent.py`: block 19.
- **New:** `grain_stats.py`, `patch_bench_check.py`.
- **Tests:**
  - Block 19 passed 50/50: gradient 2e-16 and Laplacian 4e-15 vs autodiff; mixed-precision gradient error/noise
    <= 6e-6; scatter and gather counts unchanged.
  - Blocks 13-18 all passed, both on the candidate and again on the installed files (jobs 4232044/45/100,
    4232192/97/98).
- **De-risk step:** the installed code reproduces the design judge's four loss changes (4231616) to 8e-13 relative or
  better.
- **Default path:** bit for bit unchanged, with and without harmonic modes (losses, gradients, jaxprs, runner, drift,
  sampler, history files).
- **Patch reference on `mnist28_d0123` 7:3:32:**
  - k-means: 62 s; median explained variance 0.80 (min 0.66); no empty clusters.
  - `lr_Wp` medians: 1.6 / 2.7 / 4.0e-5 at slices 31 / 45 / 60, as specified.
  - The per-patch spread of the learning-rate scale (max/min) grows at low t: 2.0 / 2.3 / 8.4 / 130 at slices
    0 / 60 / 80 / 99. These are border patches. To watch.
- **Toy run** (gauss2_8, 100 slices): at slice 99 the patch units collapse to 2 effective states (11 dead).
  Watch `patch_states_below_1e-6` on MNIST too.
- **Still open:** the two independent reviews (spec 8(3)) were launched at 01:45 and run in parallel with the GPU
  queue. If they find a real bug, the pilot is cancelled or re-run.

**Submitted (01:44):**
- **E0** (GPU cost and reproducibility, 3 slices each):
  - runs: plain K = 784 twice; patch twice; patch switch-on at slice 1; K = 912; K = 1536;
  - job: 4232341, followed by the CPU check 4232342 (`patch_bench_check.py`).
- **E1 pilot, seed 1:**
  - **P-A**, `--patch 7:3:32` from slice 0: pipeline 4232343.
  - **P-B**, `--patch 7:3:32 --patch-on-slice 31 --patch-init-scale 0.06`: pipeline 4232344.
  - Time limit 6 h each, because the GPU cost is unknown; on the CPU a patch step costs 3.5-5x a plain one, and the
    plain pipeline takes 42 min.
  - equil:3 sampling: 4232345 / 4232346.
  - FID 4232347, compared with plain seed 1, K = 912 seed 1 and K = 1536, each with both starts.
  - Grain 4232348; `early_slices` 4232349 (slices 30, 45, 55, 65, 70, 80); training summary 4232350; gallery 4232351
    (redone as 4236960: `galleries/2026-11-13_patch7x7_start-noise_vs_start-equil_seed1/`).
- **Decision rules (spec section 10):** go on to seeds 2-3 if FID improves by at least 1 against plain seed 1 (and is
  not worse than K = 912), or if specks are <= 0.40 with FID no worse.

**Maths review (independent agent, 02:00): no blocker, no major issue.**
- **Verified correct:**
  - gradient and Laplacian, including the per-pixel sums over overlapping patches;
  - how they enter the loss (same T and ½ factors and sign as `potts_terms`);
  - the drift sign;
  - unfold/fold, which are exact adjoints (last row and column included);
  - mixed precision: softmax with max subtraction, float32 at HIGHEST, loss cast to float64;
  - `patch_reference`, the same pooled formulas as `prototype_reference` at the right τ: t = 5 for P-A, t_31 = 3.43
    for P-B;
  - the learning-rate rule;
  - Adam re-initialised per slice, so keys that appear at the switch-on are handled.
- **Risk to watch, not changed: border patches in slices about 85-99.**
  - Near t = 0, S(τ) → S_w. In mostly-background border patches (pixel variance about 1e-3) the reference W is
    amplified about 19x, set by the rarest cluster's template. So the per-patch learning rate of a few border units is
    10-20x the median (max/min 130 at slice 99).
  - Adam bounds each entry's step, and a coherent field step of about 1 T is the same regime the global unit already
    handles. So divergence is unlikely, but border units could get noisy where specks form.
  - Decision: no code change under the queued pilot.
  - Check in the pilot's train_log: `lr_Wp`/`lr_bp` max and loss spikes in slices 85-99, and specks near the borders
    in the gallery.
  - Fix if needed: cap the per-patch scale at 8x that slice's median. This leaves slices <= 80 unchanged and could be
    applied with `--init-from` at slice 85.
- **Diagnostics are skewed (records only):**
  - `patch_cos_W_Wref` and the norms are flattened over all patches, so from slice 85 on they mostly report the border
    patches;
  - `patch_b_rms` contains the per-patch gauge offset.
  - Read the per-patch medians instead.
- **Note:** `--patch-init-scale` λ scales the likelihood part, not T log π. At small λ the anti-spring grows like λ², so
  0.25 gives about 1/16 of the reference anti-spring at t = 5. The spec measured this directly, so it is not an error.

**Integration review (independent agent, 02:05): no blocker. Review round closed; the pilot runs as submitted.**
- **Default path:** unchanged, apart from four inactive keys in config.json. This was checked in:
  - the loss, CD-1, hebb and force-matching guards (trace time);
  - every training branch;
  - resume and `--init-from` of older runs;
  - every sampler mode, and `sample_fast.py` (unchanged).
- **Patch lifecycle:** all correct.
  - Slice 0 is 0.25 x the reference, with no learning rate at slice 0.
  - P-B saves zero arrays before slice 31, then switches on at slice 31 before the learning rates and carry are built.
  - The warm start, EMA and best selection go over all keys, patches included; resume is shape-checked.
  - equil:3 sampling uses the slice-0 patch energy (0.25 x the reference for P-A, zero for P-B), as in training.
  - Patch drift columns: offset, sign and the factor of 2 are correct.
  - Tools that load without `with_patch` (sampler_check, time_grid_check) refuse patch runs loudly, so none silently
    drops the term.
- **Minor findings:**
  - `early_slices.py` showed all-zero units before the switch-on as 32 healthy states.
    - Fixed at 02:10: such slices are now marked "absent" (`patch_absent` in results.json).
    - Block 19G stays valid: it compares only the keys of the run without units.
  - P-B's slice-31 timing includes a recompile, so leave slice 31 (like slice 0) out of the cost reading.
  - `--ref-kmeans gpu --patch` with Ising or K = 0 is now accepted, but only the patch k-means uses the GPU. Not
    relevant here.
- **Scheduling risk:** E0 runs alongside the pilot, not before it, because the GPU queue is long. The 6 h limits allow
  about a 10x slower patch step. If a run times out, the follow-ups fail loudly and a resubmit resumes.

### Morning status (2026-11-13 10:00): queue stalled overnight; harmonic modes are a dead end; patch Potts cuts the mid-t loss by about 8x what K = 1536 does
**Queue.**
- The GPU queue did not move between 01:45 and 08:36, so no GPU job ran overnight.
- Since 08:36: K = 3136 (4231231) and the best recipe (4231701) are running; K = 912 (4231752) is done; E0 is done; both
  patch pilots are running (09:06, nearly finished training).

**E0, GPU cost and reproducibility (4232341, check 4232342): ALL OK.**
- **Cost:** a patch step (7:3:32) costs 1.57x a plain K = 784 step on the GPU: 745 vs 1170 steps/s. That is the same
  as K = 1536 (1.57x); K = 912 costs 1.06x. Much cheaper than the CPU estimate of 3.5-5x.
- **Reproducibility:** plain and patch runs are each bit for bit reproducible across processes.
- **Switch-on:** `--patch-on-slice 1` gives slice 0 identical to plain, bit for bit, with zero patch arrays.

**Harmonic modes, stage 1 (FID 4231677): no gain; dropped.**

| run (seed 1) | FID, plain start | paired vs plain_1 | FID, equil:3 start | recall F |
|---|---|---|---|---|
| plain_1 | 20.25 | - | 17.95 | 0.42 |
| harm64w (64 modes, within-cluster reference) | 19.98 | -0.27 [-1.61, +0.86], p 0.61 | 18.84 | 0.45 |
| harm64wfs31 (fixed for slices 0-30) | 20.54 | +0.29 [-0.53, +1.34], p 0.61 | 19.25 | 0.46 |
| harm16w (16 modes) | 20.19 | -0.06 [-1.33, +1.05], p 0.86 | 19.77 | 0.47 |

- The validation loss barely moves: -40 at slice 60, against -99 for K = 1536.
- With the equilibrated start the harmonic runs are worse than plain (18.8-19.8 vs 17.95).
- A Gaussian (rank-64) correction to the global Potts mixture does not carry what is missing. No seeds 2-3.

**Patch Potts pilot, validation loss per slice (train_log; lower is better).**

| slice (t) | plain | K = 912 | K = 1536 | harm64w | P-A (from slice 0) | P-B (on at slice 31) |
|---|---|---|---|---|---|---|
| 31 (3.43) | -19334 | -19335 | -19338 | -19336 | -19341 | -19339 |
| 45 (2.73) | -18957 | -18966 | -19007 | -18984 | -19131 | -19133 |
| 60 (1.97) | -18333 | -18352 | -18432 | -18373 | -19088 | -19091 |
| 70 (1.47) | -18312 | -18343 | -18433 | -18330 | -19341 | -19344 |
| 80 (0.96) | -19375 | -19383 | -19552 | -19366 | -20778 | -20774 |
| 90 (0.45) | -24622 | -24736 | -25121 | -24624 | -27989 | -27987 |
| 97 (0.10) | -53100 | -53751 | -55602 | -53319 | - | -71226 |

- **Mid t:** at slice 60 the patch units improve the loss by 755, against 99 for K = 1536 (7.6x) and 19 for the
  coupling-matched K = 912 (40x). This is in line with the design panel's estimate of about 10x K-doubling.
- **Start variants:** P-A and P-B are equal to within 5 from slice 45 on. Starting at slice 0 brings nothing, which
  matches the panel (patch structure is invisible above t of about 3.5).
- **Low t:** the gain grows (slice 90: -3367 vs -500 for K = 1536).
- **Watch:** the patch units' occupation narrows at low t.
  - P-A at slice 90: effective states 19.0 median and 1.7 minimum per patch, 20 dead states, <max p> 0.80.
  - P-B at slice 97: 48 dead, <max p> 0.95.
  - The patch force dominates there (patch/global/local rms 0.63/0.05/0.32 at slice 97).
- The validation loss is score matching, not sample quality. FID, grain and gallery follow once the pilots finish
  sampling (pipeline default sampler, then equil:3 4232345/46, FID 4232347, grain 4232348, early_slices 4232349,
  gallery 4232351).

### Results: patch Potts halves the FID (20.3 -> 10.6) and removes the grain (2026-11-13 10:20; pilots 4232343/44, FID 4236364, grain 4236365, gallery 4236366, early_slices 4232349)
**Pilot, seed 1, default start (sde_base_n500), 500 samples each:**

| run | FID (sd) | paired vs plain_1 | prec F | recall F | confident | specks / image | ink components / image |
|---|---|---|---|---|---|---|---|
| real training images (n = 500 floor) | 4.57 | - | 0.88 | 0.89 | 98.4 % | 0.01 | 1.03 |
| plain_1 (K = 784) | 20.25 (1.68) | - | 0.94 | 0.42 | 98.8 % | 0.52 | 1.84 |
| K = 912 (same couplings as the patches) | 19.12 (1.67) | - | 0.94 | 0.45 | 97.4 % | 0.51 | 1.82 |
| K = 1536 | 16.84 (1.38) | - | 0.95 | 0.53 | 98.0 % | 0.47 | 1.74 |
| **P-A, patches from slice 0** | **10.70** (1.42) | **-9.55** [-10.84, -7.42] | 0.76 | 0.85 | 88.4 % | **0.03** | 1.28 |
| **P-B, patches from slice 31** | **10.55** (1.37) | **-9.70** [-10.72, -7.65] | 0.79 | 0.85 | 88.2 % | **0.05** | 1.26 |

- **Comparisons:** P-A vs K = 912 -8.4 FID; vs K = 1536 -6.1 FID (all p < 0.005). P-B vs P-A -0.15
  [-0.80, +0.64], so the two are equal. Both pass the go rule (spec section 10) by far.
- **Grain** (grain_stats, deviation from real test images in units of their per-image SD):
  - specks +4.8 (plain) -> +0.17 (P-A);
  - ink components +3.3 -> +1.0;
  - speckle energy +2.0 -> +0.5;
  - the mid-grey fraction is now normal.
- **Gallery** (`galleries/2026-11-13_patch7x7_vs_more-global-states_seed1_start-noise/00_side_by_side.png`, same start and noise per column):
  - Each trajectory keeps its digit; the high-t decision is unchanged.
  - The noise cloud around the strokes is gone: solid strokes, clean background, closed rings.
- **What is left:** some shapes are locally clean but globally wrong:
  - a '0' drawn as an open 'C';
  - '2's with crossed 'x'-like strokes;
  - '3's that look like '5';
  - a '1' with a hook and a detached dot;
  - a few blobs.
- **Reading the trade-off:**
  - Recall roughly doubles (0.42 -> 0.85, real 0.89): the samples now cover the shape variety.
  - Precision drops below real (0.94 -> 0.78, real 0.88), and 12 % are not confidently one of 0-3 (the classifier
    sees a few 4/5/7/8).
  - The patches are independent experts, so nothing ties a stroke to the digit the global state chose. Local strokes
    are right; global consistency now limits.
- **Pipeline evaluation (LDA):**
  - %far 14.8 vs 5.4 (K = 912) and 10.0 (data); clean, class-averaged, 80 vs 93 (data 87).
  - The higher %far is consistent with the new variety and the off-class shapes.
  - Copy rate 0.6 %, so no memorisation.
- **early_slices (4232349):**
  - At slices 30-55 the patch runs are closer to the exact score (slice 45: 0.0316 vs 0.0348).
  - At slices 65-80 they are 1-2 % further from it.
  - Digit agreement and states per class are unchanged.
  - Caution: the "exact score" there is the score of the training-set mixture. At t < 1.7 it points to the nearest
    training image (memorisation), so it does not measure the generalising gain that the validation loss and FID
    show. Do not read the earlier "mid-t is capacity-limited" verdict (4231198) from this metric alone.
  - Patch occupation at slices 45-80: effective states 18-27 median (min 1.3-9), dead 0-8. Healthy; the collapse seen
    in the toy run does not happen on MNIST at mid t (at slice 99 the train_log shows dead 127 of 2048).
- **Cost:** training 1:04 / 0:59 h vs 0:42 h for plain; sampling 546 s vs about 500 s.

**Decision and next runs (10:25):**
- E2: P-B seeds 2 and 3, pipelines 4236383/84, equil:3 sampling 4236386/87.
- Stack test: K = 1536 + patches (P-B) seed 1, pipeline 4236385, equil:3 4236388.
- Follow-ups: FID 4236389 (each against plain of the same seed; equilibrated vs default start; K = 1536 + patches vs
  K = 1536), grain 4236390, gallery 4236923 (`galleries/2026-11-13_patch7x7-from-t3.4_seeds1-3_and_with-K1536/`; replaces 4236391).
- Still queued from the pilot: equil:3 for P-A and P-B (4232345/46), with FID, grain and gallery 4232347/48/51.
- Next design step if seeds 2-3 hold: global consistency.
  - Several patch sizes (e.g. 14:7 added to 7:3) keep the single bipartite layer: exact, instant equilibration,
    local learning.
  - Only if that fails: the tree coupling (global state biases the patch states). It is exact in simulation, but a
    device would equilibrate slowly.

### Patches clean the strokes but override the global shape; three shape tests submitted (2026-11-13 10:50)
**Gallery reading (the user, B1 vs K = 1536):**
- B1 has no speckle, but some samples are not proper digits.
- K = 1536 keeps consistent shapes but is speckled, yet its FID is worse (16.8 vs 10.6).
- So FID here mostly measures speckle: speckle shifts every image's classifier features a little, while broken shapes
  shift about 12 % of them a lot.
- From now on, report FID together with precision (0.94 -> 0.78) and the confident share (98.8 % -> 88 %), which do
  show the shape defect.

**Why the shape breaks.** Force decomposition in B1's train_log (rms per pixel):

| slice (t) | 31 (3.43) | 40 (2.98) | 50 (2.47) | 60 (1.97) | 70 (1.47) | 80 (0.96) | 99 (0) |
|---|---|---|---|---|---|---|---|
| patch | 0.004 | 0.029 | 0.065 | 0.105 | 0.137 | 0.168 | 1.12 |
| global Potts (K = 784) | 0.016 | 0.019 | 0.016 | 0.013 | 0.015 | 0.021 | 0.090 |
| local | 0.005 | 0.004 | 0.004 | 0.003 | 0.006 | 0.046 | 0.85 |

- The patch force overtakes the global force at t of about 3 and is 8x larger by t = 2.
- From mid t on, the independent 7x7 experts decide the strokes, and the whole-image template is only a weak hint. A
  faint part of a ring can be cleaned away (a '0' becomes a 'C').

**Three shape tests, all seed 1, default start:**
1. **Stronger global unit:** K = 1536 + 7:3:32 on at 31. Already queued (4236385).
2. **Bigger patches:** 14:7:64, i.e. 9 patches of 14x14 with 64 states, 225,792 couplings, about the same as 7:3:32's
   200,704. On at 31, init 0.06. Pipeline 4236523.
3. **Later switch-on:** 7:3:32 on at slice 55 (t = 2.22), init 0.06, so the global unit fixes the shape before the
   patches act. Pipeline 4236524.

- Follow-ups: FID 4236525 (pairs vs B1 and plain; read precision and the confident share too), grain 4236526,
  gallery 4236924 (`galleries/2026-11-13_shape-tests_stronger-global_bigger-patches_later-patches_seed1/`; replaces 4236527).
- If none of the three keeps both clean strokes and the shapes:
  - multi-scale patches (7:3 + 14:7 in one layer; needs `--patch` to take several specs);
  - then the tree coupling.

### K = 3136: another -3.4 FID per doubling, shapes stay clean (2026-11-13 11:20; pipeline 4231231, FID 4236581, grain 4236582, gallery 4236596)
| run (seed 1, default start) | FID (sd) | prec F | recall F | confident | specks | ink components |
|---|---|---|---|---|---|---|
| plain K = 784 | 20.25 (1.68) | 0.94 | 0.42 | 98.8 % | 0.52 | 1.84 |
| K = 1536 | 16.84 (1.53) | 0.95 | 0.53 | 98.0 % | 0.47 | 1.74 |
| **K = 3136** | **13.48** (1.43) | **0.95** | 0.63 | **98.8 %** | 0.31 | 1.51 |
| 7x7 patches (B1) | 10.55 (1.55) | 0.79 | 0.85 | 88.2 % | 0.05 | 1.26 |

- **FID:** K = 3136 vs K = 1536 -3.35 [-4.71, -1.74]. Each doubling of K so far gives about -3.4 FID (784 -> 1536 -> 3136),
  with no sign of saturation yet.
- **B1 vs K = 3136:** -2.93 [-4.58, -0.51], p 0.02.
- **Gallery** (`galleries/2026-11-13_global-states-K784-K1536-K3136_vs_patch7x7_seed1_start-noise/`): every sample is a clear digit, with no broken or off-class shapes. Precision
  stays at 0.95 and the confident share at 98.8 %.
- **Grain:** less than K = 1536 (specks 0.47 -> 0.31) but still clearly visible: speckle along the strokes and filled-in
  ring interiors.
- **Pipeline evaluation:** %far 0.4 (data 10.0), clean (class-averaged) 97.2, median NN 4.12 (data 4.41), copy 0.4 %.
  The samples sit closer to the training set than held-out digits do; this is the prototype effect.
- **Cost:** 2:23 h pipeline (K = 784: 0:42 h).
- **Reading:** global states buy shape and precision; 7x7 patches buy texture and recall. This is the case for combining
  them. K = 1536 + patches (4236385) tests whether the global shape holds against the patch force. If it does,
  K = 3136 + patches is the next candidate.
- equil:3 sampling for K = 3136 queued: 4236583.

### The equilibrated start stacks with the patches: B1 + equil:3 = FID 8.65 (2026-11-13 12:00; sampling 4232345/46, 4231753; FID 4232347; gallery 4232351)
| run (seed 1) | FID, default start | FID, equil:3 | change | precision F, equil:3 | confident, equil:3 |
|---|---|---|---|---|---|
| plain K = 784 | 20.25 | 17.95 | -2.3 | 0.94 | 98.6 % |
| K = 912 | 19.12 | 17.16 | -2.0 | 0.94 | 96.6 % |
| K = 1536 | 16.84 | 15.01 | -1.8 | 0.95 | 97.8 % |
| P-A (patches from slice 0) | 10.70 | 9.74 | -1.0 | 0.78 | 89.8 % |
| **P-B (patches from slice 31)** | 10.55 | **8.65** | **-1.9** | 0.78 | 89.6 % |

- **Best FID so far: 8.65.** The floor at n = 500 is 4.57.
- **B1_eq vs plain_1eq:** -9.30 [-10.33, -7.55]; vs K = 1536 + equil:3 -6.36 (both p < 0.005).
- **B1_eq vs A1_eq:** -1.09 [-1.89, -0.30], p 0.01. With the equilibrated start, P-B is better than P-A, presumably because
  P-A's slice-0 energy contains the weak patch units, which the start relaxes under.
- **Shapes:** precision and the confident share do not change with the start. The gallery (`galleries/2026-11-13_patch7x7_start-noise_vs_start-equil_seed1/08_K784+patch7x7-from-t3.4_start-equil_seed1.png`)
  shows the same defects: open 'C'-shaped zeros, '1's with a detached dot ('!'), crossed '2's. The start fixes class
  balance, not shape.
- **Recipe so far:** `--patch 7:3:32 --patch-on-slice 31 --patch-init-scale 0.06` + `--init-dist equil:3`.

### Best global-only recipe: K = 1536 + early 2x + equilibrated start = FID 13.55 (2026-11-13 12:10; pipeline 4231701, equil:3 4231702, FID 4231703, gallery 4231704)
| run (seed 1) | FID default | FID equil:3 | prec F (eq) | recall F (eq) | confident (eq) |
|---|---|---|---|---|---|
| plain K = 784 | 20.25 | 17.95 | 0.94 | 0.45 | 98.6 % |
| early 2x, K = 784 | 18.56 | 17.18 | 0.94 | 0.49 | 98.2 % |
| K = 1536 | 16.84 | 15.01 | 0.95 | 0.54 | 97.8 % |
| **K = 1536 + early 2x** | 14.43 | **13.55** | 0.94 | 0.59 | 98.8 % |

- **Early 2x at K = 1536:** -2.40 FID with the default start [-3.59, -1.12] and -1.46 with equil:3 [-2.61, -0.49]. That is more
  than early 2x at K = 784 (-1.7).
- **Stacking:** all three global-side fixes add up, 20.25 -> 13.55.
- **Gallery** (`galleries/2026-11-13_best-global-recipe_K1536+early2x_seed1/05_K1536+early2x_start-equil_seed1.png`): every sample is a clean digit shape, but the strokes are
  still speckled.
- **Status of the options:**
  - option 3 (best recipe) worked: -6.7 FID with shapes intact, but still about 5 FID behind the patch run with
    equil:3 (8.65);
  - option 1 (patches) is ahead on FID, behind on shape.
- **Natural combination:** K = 1536 + early 2x + patches (on at 31) + equil:3. Wait for K = 1536 + patches (4236385)
  first, to see whether the stronger global unit protects the shapes.

### Gallery naming (2026-11-13 12:30, the user's request: names must say what was done)
**Folders.** New galleries go to `out/latent_physical_SGM/galleries/<date>_<what is compared>/`. Today's four were redone
there (jobs 4236959-62) and the old short-named folders were removed:

| new folder | replaces |
|---|---|
| `2026-11-13_patch7x7_vs_more-global-states_seed1_start-noise/` | `patch_pilot/gallery_s1_default/` |
| `2026-11-13_patch7x7_start-noise_vs_start-equil_seed1/` | `patch_pilot/gallery_s1/` |
| `2026-11-13_global-states-K784-K1536-K3136_vs_patch7x7_seed1_start-noise/` | `convergence_study/gallery_K3136/` |
| `2026-11-13_best-global-recipe_K1536+early2x_seed1/` | `convergence_study/gallery_best/` |

Two pending galleries were resubmitted with the new names (4236923 replaces 4236391; 4236924 replaces 4236527). Older
galleries (before 2026-11-13) keep their names.

**Files** (`make_gallery.py`, changed 12:15):
- `00_side_by_side.png`: one row per model; column j = the same start and noise in every model.
- `01_real_digits.png`.
- `NN_<name>.png`: in row order.
- `README.txt`: the legend, plus for every set its sample file with the run's training settings (K, seed, patch,
  switch-on, init scale, steps schedule) and the sampling settings (sampler, n, start distribution).
- The left margin of the side-by-side image grows with the label length.

**Names** (file name = row label):
- `K784` = global Potts with 784 whole-image states (base model).
- `+patch7x7` / `+patch14x14` = per-patch Potts units: 7x7 every 3 px with 32 states, or 14x14 every 7 px with 64
  states.
- `-from-tX` = patch units only at diffusion times t <= X (`from-t5` = always; `from-t3.4` = `--patch-on-slice 31`;
  `from-t2.2` = slice 55).
- `early2x` = 2x steps for t > 3.4.
- `start-noise` = N(0, T I).
- `start-equil` = `--init-dist equil:3`.
- `seedN` = training seed.
- Example: `K784+patch7x7-from-t3.4_start-equil_seed1` (FID 8.65; earlier called "B1_eq").

### Results: switching the 7x7 patches on later (t = 2.2) keeps the global shapes: FID 7.7 from the noise start, precision 0.90 (2026-11-13 15:15; pipelines 4236383-85, 4236523/24; FID 4236389, 4236525; grain 4236390, 4236526; galleries 4236923/24)
**Patch gain reproduces over seeds** (`K784+patch7x7-from-t3.4`, FID; paired vs `K784` of the same seed):

| seed | K784 noise start | +patch from t3.4, noise | paired | K784 equil start | +patch from t3.4, equil | paired |
|---|---|---|---|---|---|---|
| 1 | 20.25 | 10.55 | -9.70 | 17.95 | 8.65 | -9.30 |
| 2 | 20.14 | 10.52 | -9.63 | 18.25 | 9.03 | -9.22 |
| 3 | 19.36 | 9.72 | -9.64 | 17.74 | 9.42 | -8.32 |
| mean | 19.92 | **10.26** | -9.66 | 17.98 | **9.03** | -8.95 |

- All p < 0.005.
- Precision 0.77-0.79 and confident share 88-90 % in every seed: the shape defect is systematic, not a seed effect.

**Shape tests (seed 1, noise start):**

| model | FID | paired vs from-t3.4 | prec F | recall F | confident | specks | speckle (SD) |
|---|---|---|---|---|---|---|---|
| K784 | 20.25 | | 0.94 | 0.42 | 98.8 % | 0.52 | +2.00 |
| K1536 | 16.84 | | 0.95 | 0.53 | 98.0 % | 0.47 | +1.83 |
| K784+patch7x7-from-t3.4 | 10.55 | - | 0.79 | 0.85 | 88.2 % | 0.05 | +0.45 |
| K1536+patch7x7-from-t3.4 | 8.82 | -1.73 [-3.23, -0.44] | 0.80 | 0.86 | 91.8 % | 0.02 | +0.39 |
| K784+patch14x14-from-t3.4 | 9.17 | -1.39 [-2.68, -0.66] | 0.87 | 0.79 | 94.0 % | 0.06 | +0.98 |
| **K784+patch7x7-from-t2.2** | **7.68** | **-2.87** [-4.77, -1.37] | **0.90** | 0.80 | **97.4 %** | 0.03 | +0.51 |

- **K784+patch7x7-from-t2.2 vs K784:** -12.57 [-13.40, -10.78]. This is the best FID from the noise start so far, with
  precision and confident share almost back to the global-only models.
- **Gallery** (`galleries/2026-11-13_shape-tests_stronger-global_bigger-patches_later-patches_seed1/`):
  - Column by column, the from-t2.2 samples keep the K784 shapes: the open 'C' is a '0' again, the hooked '1' is
    straight. The strokes are clean.
  - A few defects remain: a '3' like a '5', two messy '2's, one blob.
- **Stronger global unit** (K1536 + patches from t3.4): the 'C' and the crossed '2' stay. More global states alone do
  not hold the shape against the patch force.
- **Bigger patches** (14x14): better shapes than 7x7 from t3.4, but more speckle and mid-grey (speckle +0.98,
  mid-grey +0.60 SD). Still a 'C'.
- **K1536+patch7x7-from-t3.4 with equil:3:** 7.65 (vs K1536 equil -7.37); precision 0.82, confident 93.2 %.
- **Reading:** what matters is when the patches act, not how strong the global unit is.
  - From t = 3.4 the patch force overrides the global template while the shape is still being formed (t of about
    3.4-2.2).
  - From t = 2.2 the global unit has set the shape and the patches only clean the strokes.
  - This fits the earlier finding: the digit choice is made above t of about 3.5, and the grain forms at t of about
    2.7-1.5.

**Submitted (15:15):**
- equil:3 for from-t2.2 seed 1: 4237615.
- from-t2.2 seeds 2 and 3: pipelines 4237616 / 4237618, equil:3 4237617 / 4237619.
- Switch-on scan, seed 1:
  - from-t2.7 (slice 45): pipeline 4237620, equil:3 4237621;
  - from-t1.5 (slice 70): pipeline 4237622, equil:3 4237623.
- K3136 equil:3, redone (4236583 hit OOM at 48 GB; now 64 GB, chunk 125): 4237624.
- Follow-ups: FID 4237627, grain 4237628, galleries 4237629 (`galleries/2026-11-13_patch7x7-switch-on-time-scan_seed1_start-noise/`)
  and 4237630 (`galleries/2026-11-13_patch7x7-from-t2.2_seeds1-3_start-equil/`).

**Partial results (16:20; quick FID on the finished samples, job 4237732; full batch FID 4237627 still pending):**

| model | FID | prec F | recall F | confident | paired |
|---|---|---|---|---|---|
| K784_start-equil_seed1 | 17.95 | 0.94 | 0.45 | 98.6 % | |
| K784+patch7x7-from-t3.4_start-equil_seed1 | 8.65 | 0.78 | 0.85 | 89.6 % | |
| K784+patch7x7-from-t2.2_start-noise_seed1 | 7.68 | 0.90 | 0.80 | 97.4 % | |
| **K784+patch7x7-from-t2.2_start-equil_seed1** | **6.42** | 0.90 | 0.79 | 96.2 % | vs K784 equil -11.53 [-12.55, -9.66]; vs from-t3.4 equil -2.23 [-3.56, -1.34] |
| K784+patch7x7-from-t2.2_start-noise_seed2 | 7.53 | 0.89 | 0.80 | 95.4 % | vs K784 noise seed 2 -12.62 [-13.40, -10.80] |
| K3136_start-equil_seed1 | 13.04 | 0.95 | 0.61 | 98.6 % | vs K3136 noise -0.44 (n.s.) |

- **New best: FID 6.42** (floor 4.57 at n = 500), with precision 0.90.
- Seed 2 reproduces the from-t2.2 gain from the noise start.
- The equilibrated start helps K = 3136 hardly at all (-0.4); its class balance is already good.

### Results: 7x7 patches from t = 2.2 + equilibrated start = FID 6.24 (mean of 3 seeds), precision 0.90 (2026-11-13 16:40; pipelines 4237616/18/20/22, equil:3 4237615/17/19/21/23, K3136 equil:3 4237624; FID 4237627, grain 4237628, galleries 4237629/30)
**Seeds 1-3, `K784+patch7x7-from-t2.2`** (paired vs `K784` of the same seed and start, all p < 0.005):

| seed | noise start | paired | equil start | paired | prec F (equil) | confident (equil) |
|---|---|---|---|---|---|---|
| 1 | 7.68 | -12.57 | 6.42 | -11.53 | 0.90 | 96.2 % |
| 2 | 7.53 | -12.62 | 6.02 | -12.23 | 0.90 | 95.6 % |
| 3 | 6.78 | -12.58 | 6.29 | -11.45 | 0.89 | 96.0 % |
| **mean** | **7.33** | -12.59 | **6.24** | -11.74 | 0.90 | 95.9 % |

- `K784` means for comparison: 19.92 (noise), 17.98 (equil).
- The floor at n = 500 is 4.57. The gap to the floor shrank from 13.4 to 1.7 FID in one day.

**When to switch the patches on (seed 1):**

| switch-on (slice) | FID noise | FID equil | prec F | recall F | confident | specks (SD) | speckle (SD) |
|---|---|---|---|---|---|---|---|
| never (K784) | 20.25 | 17.95 | 0.94 | 0.42 | 98.8 % | +4.76 | +2.00 |
| t = 3.4 (31) | 10.55 | 8.65 | 0.79 | 0.85 | 88.2 % | +0.36 | +0.45 |
| t = 2.7 (45) | 8.17 | 7.10 | 0.84 | 0.83 | 92.6 % | +0.21 | +0.47 |
| **t = 2.2 (55)** | **7.68** | **6.42** | 0.90 | 0.80 | 97.4 % | +0.17 | +0.51 |
| t = 1.5 (70) | 9.66 | 7.67 | 0.96 | 0.69 | 99.2 % | +1.07 | +0.85 |

- **Paired vs t = 2.2:**
  - t = 2.7: +0.49 (noise, n.s.) and +0.68 (equil, p 0.06);
  - t = 1.5: +1.98 and +1.25 (both p < 0.01);
  - t = 3.4: +2.87 and +2.23 (both p < 0.01).
- **Trade-off:** a later switch-on raises precision and the confident share monotonically, and lowers recall.
  - Switching on too early (t >= 2.7): the patches override the global shape ('C' instead of '0', hooked '1').
  - Switching on too late (t = 1.5): the grain is already formed and partly stays (specks +1.07 SD).
  - Optimum near t = 2.2-2.7, consistent with the shape being fixed above t of about 2.5 and the grain forming at
    t = 2.7-1.5.
- **Gallery** (`galleries/2026-11-13_patch7x7-switch-on-time-scan_seed1_start-noise/00_side_by_side.png`):
  - the 'C' appears at t = 3.4 and 2.7 and is a '0' at t = 2.2 and 1.5;
  - the speckle returns at t = 1.5.
- **Equil-start gallery** (`galleries/2026-11-13_patch7x7-from-t2.2_seeds1-3_start-equil/`): almost all samples are clean,
  proper digits. A few remain: a '7'-like shape with a dot, one squiggle, a '1' with a dot, an open '0', some messy
  '3's.
- **K3136 equil:3** (redone with 64 GB): 13.04, vs 13.48 from noise (n.s.).

**Recipe now:** `--patch 7:3:32 --patch-on-slice 55 --patch-init-scale 0.06` + `--init-dist equil:3`.

**Submitted (16:45): stacking with the global-side fixes, seed 1.**
- K1536 + patches from t2.2: pipeline 4237903, equil:3 4237905.
- K1536 + early 2x + patches from t2.2: pipeline 4237904, equil:3 4237906.
- Follow-ups: FID 4237907, grain 4237908, gallery 4237909
  (`galleries/2026-11-13_patch7x7-from-t2.2_on-top-of-K1536-and-K1536+early2x_seed1/`).
- Expected: each doubling of K gave -3.4 FID for the global-only model. With the patches acting late, the global unit
  sets the shape, so a better global unit should now carry through.

### All ten digits: pilot of the patch recipe (2026-11-13 17:00, the user: "go for all 10 digits")
**Why now.**
- The recipe has no class-specific ingredient: 7x7 patches switched on late, plus the data-free equilibrated start.
- Ten-digit MNIST 28x28 is the paper's target. The sweep of 2026-11-10 (n = 2000, floor 1.69) gave FID 31.1 for K784,
  22.5 for K3072 and 26.0 for K1536.

**Open questions.**
- With ten digits a global state covers more variety: about 78 states per class at K784, against 196 on four digits.
  So the global unit sets a weaker shape, and K may matter more.
- Whether t = 2.2 is still the right switch-on time.

**Pilot, seed 1:**
- 2000 samples per model and both starts, as in the sweep.
- Current code (stencil, mixed precision, `--val-every 1000`).
- Equilibrated sampling at 64 GB with chunk 125, because K3136 ran out of memory at 48 GB.

| run | pipeline | equil:3 |
|---|---|---|
| K784 (baseline on today's code, against the sweep's K784 from 2026-11-10) | 4237994 | 4237998 |
| K784 + patches from t2.2 (slice 55) | 4237995 | 4237999 |
| K784 + patches from t2.7 (slice 45): switch-on check | 4237996 | 4238000 |
| K3072 + patches from t2.2 (against the sweep's K3072) | 4237997 | 4238001 |

- Follow-ups: FID 4238002, grain 4238003, gallery 4238004
  (`galleries/2026-11-13_ten-digits_patch7x7-late_vs_global-only_seed1/`).
- Four-digit stacking runs (K1536 +/- early 2x + patches from t2.2, 4237903-09) keep running.

**Update 18:20: K3072 time limit, and an early K784-only evaluation.**
- The K3072 equilibrated-sampling job 4238001 had a 2.5 h limit, but needs about 3 h.
  - Reference: K3136 four-digit (4237624) took 670 s per chunk of 125.
  - 2000 samples = 16 chunks, so about 3 h plus equilibration.
- Cancelled it and resubmitted as **4238364** with a 4 h 45 min limit (same arguments).
- FID 4238002, grain 4238003 and gallery 4238004 now depend on 4238364 instead.
  Expected around 01:00-02:00, since K3072 + patches training is at slice 57/100 at 18:05.
- So the K784 results are not held up by K3072:
  - early FID **4238365** for the K784 runs and the sweep's K784;
  - early gallery **4238366** (`galleries/2026-11-13_ten-digits_K784-only_patch7x7-late_vs_global-only_seed1/`).
  - Both depend only on the K784 equilibrated-sampling jobs 4237998-4238000. Expected around 20:00.
- Status at 18:05:
  - ten-digit K784 baseline trained and sampled (4237994, 230 s per 250 samples);
  - K784 + patch runs sampling;
  - four-digit K1536 + patches from t2.2 sampled (4237903); its equilibrated sampling 4237905 is running.

### Patch variants on four digits: 64 states, and 8x8 every 2 px (submitted 2026-11-13 18:40)
**Question.** Does more patch capacity or more overlap help the current recipe (7x7 every 3 px, 32 states, from
t = 2.2, start-equil: FID 6.42 at seed 1)? The two tests:

1. **More states per patch:** `--patch 7:3:64`. Same windows; W+Q couplings double to 401,408.
2. **Denser, more overlapping windows:** `--patch 8:2:32`.
   - 121 patches of 8x8; each pixel is in about 10 patches (now 4), and neighbouring windows overlap by 6 px.
   - W+Q couplings 495,616.
   - Stride 2 on 7x7 windows is not possible: the code needs (28 - side) divisible by the stride, so side 7 allows
     strides 1, 3 or 7 only.
   - Side and stride both change against the recipe, so this tests "denser" as a whole, not stride alone.

**Common settings.** K784, patches from slice 55 (t = 2.2) with init scale 0.06, seed 1, n = 500, both starts.
The reference is the recipe's seed 1 (`..._mixst_patch7s3S32on55_seed1`).

**Fair share at submission.**
- We had used about 2.7x our share (LevelFS 0.36), and there are no hard limits on the account or job class.
- About 28 of 256 L40s were free and about 220 GPU jobs from others were waiting on priority, but our single-GPU
  jobs had started within about 15 min all day.

| run | pipeline | equil:3 |
|---|---|---|
| K784 + patch7x7-64states from t2.2 (`..._mixst_patch7s3S64on55_seed1`) | 4238574 | 4238576 |
| K784 + patch8x8-every2px from t2.2 (`..._mixst_patch8s2S32on55_seed1`) | 4238575 | 4238577 |

- Follow-ups: FID 4238578, grain 4238579, gallery 4238580
  (`galleries/2026-11-13_patch-variants_7x7-64states_8x8-every2px_vs_7x7-32states_from-t2.2_seed1/`).

### Results: 7x7 patches from t = 2.2 on top of K = 1536 and K = 1536 + early 2x (2026-11-13 18:50)
**Jobs.**
- Pipelines 4237903 and 4237904, equilibrated sampling 4237905 and 4237906.
- FID 4237907 (`fid/4237907`), grain 4237908, gallery 4237909
  (`galleries/2026-11-13_patch7x7-from-t2.2_on-top-of-K1536-and-K1536+early2x_seed1/`).
- Seed 1, n = 500, four digits; FID floor 4.57.

| set (seed 1, n = 500) | FID | prec F | rec F | confident | specks | speckle |
|---|---|---|---|---|---|---|
| K784 + patch7x7 from t2.2, start-noise | 7.68 | 0.900 | 0.802 | 97.4% | +0.17 | +0.51 |
| K784 + patch7x7 from t2.2, start-equil | 6.42 | 0.897 | 0.791 | 96.2% | +0.29 | +0.55 |
| K1536, start-equil (no patches) | 15.01 | 0.946 | 0.536 | 97.8% | +4.82 | +1.87 |
| K1536 + early2x, start-equil (no patches) | 13.55 | 0.944 | 0.593 | 98.8% | +3.77 | +1.78 |
| K1536 + patch7x7 from t2.2, start-noise | 6.60 | 0.909 | 0.796 | 98.0% | +0.25 | +0.44 |
| K1536 + patch7x7 from t2.2, start-equil | 5.74 | 0.907 | 0.797 | 95.8% | +0.17 | +0.46 |
| K1536 + early2x + patch7x7 from t2.2, start-noise | 6.44 | 0.921 | 0.831 | 97.4% | +0.15 | +0.40 |
| **K1536 + early2x + patch7x7 from t2.2, start-equil** | **5.62** | 0.913 | 0.803 | 96.8% | +0.23 | +0.46 |

The grain columns are deviations from the real test images, in units of their per-image SD.

**Paired bootstrap (95% intervals):**
- **K1536 vs K784, both with patches:** FID -1.09 [-2.76, +0.35] (p = 0.14) from noise, -0.68 [-1.96, +0.62]
  (p = 0.27) from equilibrated. Not significant at one seed and n = 500.
- **Early 2x on top:** -0.15 (p = 0.82) and -0.13 (p = 0.78). Nothing.
- **Patches on K1536 (equilibrated start):** -9.27 [-10.1, -7.8] without early 2x and -7.94 with it, both p < 0.01.
  As on K784, the patches account for most of the gain.

**Images.**
- *First reading (18:50, withdrawn):* "K1536 + early2x + patches and K784 + patches look almost the same". This was
  too quick.
- *Recount (19:05), after the user found the K1536 runs visibly better.*
  - Count of clearly malformed digits among the first 100 samples of each grid (start-equil).
  - Malformed: broken or open loops, a 2 or 3 with a gap, a squiggle that is no digit, a blob.
  - Borderline shapes (thick or blotchy but readable) not counted.
  - One person counting one seed, so about +/- 3.

  | grid (start-equil, samples 0-99) | clearly malformed | FID |
  |---|---|---|
  | `03_K784+patch7x7-from-t2.2` | about 8 | 6.42 |
  | `09_K1536+patch7x7-from-t2.2` | about 9 | 5.74 |
  | `11_K1536+early2x+patch7x7-from-t2.2` | about 3 | 5.62 |

  - The visible gain comes with K1536 + early2x. Several of K784's broken digits are clean in the early-2x run:
    samples 6, 26, 55, 61 and 84.
  - In my count, K1536 without early 2x is no cleaner than K784; to be checked against the user's reading.
  - FID barely registers the difference (early 2x: -0.13, p = 0.78), and neither does the confident share
    (95.8% / 96.8%). This confirms the user's point from this morning: **the FID here measures grain, not shape**.
- Without patches, K1536 is grainy (specks +4.8 SD); with patches it is as clean as K784 + patches.

**Conclusions (revised 19:05).**
1. FID and the classifier confidence are blind to the remaining shape errors, at least at n = 500. Recipe decisions
   between patch models need a shape measure: a blind count on more samples, or a metric checked against one.
2. K1536 + early2x + patches is visibly the best four-digit model (about 3% malformed against about 8%). Whether that
   comes from early 2x, from K1536 + early 2x together, or from this seed is open. The shape is set at t > 2.2, where
   early 2x doubles the steps (t > 3.4) and the global unit acts, so a real effect is plausible.
3. Best FID so far on four digits: **5.62** (K1536 + early2x + patches from t2.2, start-equil), against the 4.57
   floor.

### Results: all ten digits, K784, with 7x7 patches from t2.2 / t2.7 (2026-11-13 19:35)
**Jobs.**
- Pipelines 4237994-96, equilibrated sampling 4237998-4238000.
- Early K784-only FID 4238365 (`fid/4238365`) and gallery 4238366
  (`galleries/2026-11-13_ten-digits_K784-only_patch7x7-late_vs_global-only_seed1/`).
- Seed 1, n = 2000; FID floor 1.69 (real training images at n = 2000); real images 97.8% confident.

| set (ten digits, seed 1, n = 2000) | FID | prec F | rec F | confident |
|---|---|---|---|---|
| real training images (floor / target) | 1.69 | 0.870 | 0.882 | 97.8% |
| K784, sweep 2026-11-10 (old code), start-noise | 31.10 | 0.904 | 0.296 | 89.1% |
| K784, today's code, start-noise | 31.08 | 0.904 | 0.298 | 89.1% |
| K784, today's code, start-equil | 30.12 | 0.904 | 0.284 | 89.2% |
| K784 + patch7x7 from t2.2, start-noise | 10.22 | 0.837 | 0.746 | 88.4% |
| **K784 + patch7x7 from t2.2, start-equil** | **8.94** | 0.844 | 0.733 | 88.6% |
| K784 + patch7x7 from t2.7, start-noise | 12.12 | 0.765 | 0.808 | 82.7% |
| K784 + patch7x7 from t2.7, start-equil | 10.99 | 0.769 | 0.781 | 84.0% |

**Paired bootstrap:**
- Patches from t2.2: -20.9 [-21.6, -20.0] from noise and -21.2 [-21.8, -20.3] from equilibrated (both p < 0.01).
- t2.7 against t2.2: +1.9 and +2.05 (both p < 0.01).
- Today's code against the sweep run: -0.02 [-0.06, +0.04].

**Findings.**
1. **The recipe transfers to ten digits.**
   - K784 FID 31.1 -> 8.94 with patches from t2.2 and the equilibrated start, a factor 3.5.
   - That is far below the sweep's best global-only model (K3072, 22.5) with a quarter of its global states.
   - Recall goes 0.28 -> 0.73.
2. **t2.2 is still the better switch-on time.** As on four digits, earlier switch-on (t2.7) gives more recall but
   lower precision (0.77 against 0.84) and more unclassifiable digits (84% against 89% confident).
3. **The fast code is exact enough** (stencil visible term, mixed precision): today's K784 gives the same samples as
   the float64 sweep run (FID 31.08 against 31.10, the same class shares).
4. **Shapes are the remaining problem, and more so than on four digits.**
   - Gallery `06_K784+patch7x7-from-t2.2_start-equil_seed1.png`: textures are clean, but about 16 of the first 100
     are clearly malformed (my count, +/- 4).
   - Malformed: a 5 or 3 as a squiggle, a 4 drawn as a thin cross, 4/9 and 4/8 hybrids, scribbles.
   - About 12 more are borderline. This matches the 88.6% confident share (real 97.8%).
   - The side-by-side shows the malformed shapes are already there in the global-only model with the same start and
     noise (column 1: the same squiggle 5 in every row). The shape is decided before t = 2.2, by the global unit
     with about 78 states per digit; the patches only clean it.
   - A new, small artefact: an isolated dot next to a clean digit ("1.", "8.", "2."), a few per 100.
5. **What should fix the shapes:**
   - More global states: K3072 + patches, 4238364 and FID 4238002 overnight.
   - Early 2x: on four digits, K1536 + early2x + patches had about 3 malformed per 100 against about 8. Not yet
     tested on ten digits.

### Shape tests: early 2x on ten digits, and a blind shape count (submitted 2026-11-13 19:50)
**Why.**
- FID does not see the shape errors that are left once the patches remove the grain.
- Early 2x looked visibly better on four digits (about 3 against 8 malformed per 100) at an FID change of only -0.13.

**1. Ten digits, K784 + early2x + patch7x7 from t2.2, seed 1** (`..._mixst_early2x_patch7s3S32on55_seed1`).
- Pipeline 4238946 (n = 2000), equilibrated sampling 4238947.
- FID 4238948, grain 4238949, gallery 4238950
  (`galleries/2026-11-13_ten-digits_K784+early2x+patch7x7-from-t2.2_vs_K784+patch7x7-from-t2.2_seed1/`).
- Compared with the K784 + patch7x7 from t2.2 run above (FID 8.94 from the equilibrated start, about 16% malformed).

**2. Blind shape count, new script `blind_gallery.py`.**
- **Sheets.** Each sheet has 100 tiles, all from one model in a random order. The sheets of all models are
  shuffled together, and `KEY.txt` (sheet -> model) is opened only after counting.
- **Paired.** Every model shows the same trajectories (same starts and noise).
- **Control.** `real` random training images, to give the rate at which real digits look malformed by the same rule.
- **What counts as malformed** (in the README): a broken or open stroke that changes the digit, a merged digit or
  hybrid, a scribble or blob, disconnected pieces. Stray dots are counted separately.
- **Four-digit job 4238951.** 200 samples per set, start-equil: real, K784 + patches from t2.2 seeds 1-3 (gives the
  seed-to-seed spread), K1536 + patches, K1536 + early2x + patches. 12 sheets in
  `galleries/2026-11-13_blind-shape-count_four-digits_K784+patch-seeds1-3_K1536+patch_K1536+early2x+patch_real_start-equil/`.
- **Plan.**
  - I count the sheets before opening the key; the user can count independently.
  - The same for ten digits once the early 2x and K3072 runs are in.
  - Then check whether an automatic metric (pixel precision, nearest-image distance, classifier confidence) follows
    the counts.

**Blind count, four digits (job 4238951, Claude's count, 2026-11-13 20:05, written down BEFORE opening KEY.txt).**
- Rule as in the README. Borderline tiles not counted: a 1 with serifs, a 3 with a closed but readable loop, thick
  blobs that are still readable.
- Counted as malformed: a 3/8 hybrid (an 8 is not among 0-3).
- Stray dots were not counted consistently, so no numbers for them.

| sheet | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| malformed of 100 | 9 | 0 | 7 | 8 | 2 | 9 | 7 | 5 | 1 | 6 | 9 | 8 |

**Guesses before the key:**
- 02 and 09 are the real images: smooth strokes, no patch texture, no stray dots.
- 05 is the cleanest model sheet.

**Unblinded (key opened after the counts above).**
- Both real sheets were guessed right (02, 09).
- 05 was K1536 + early2x, but so was 04, which got 8.

| set (four digits, start-equil, 200 samples each) | sheets | malformed | rate | FID (n = 500) |
|---|---|---|---|---|
| real training images | 02, 09 | 0 + 1 | **0.5%** | 4.57 (floor) |
| K784 + patch7x7 from t2.2, seed 1 | 10, 11 | 6 + 9 | 7.5% | 6.42 |
| K784 + patch7x7 from t2.2, seed 2 | 01, 06 | 9 + 9 | 9.0% | 6.02 |
| K784 + patch7x7 from t2.2, seed 3 | 03, 07 | 7 + 7 | 7.0% | 6.29 |
| K1536 + patch7x7 from t2.2, seed 1 | 08, 12 | 5 + 8 | 6.5% | 5.74 |
| K1536 + early2x + patch7x7 from t2.2, seed 1 | 04, 05 | 8 + 2 | 5.0% | 5.62 |

**Reading.**
1. **Gap to real.** All patch models produce about 5-9% clearly malformed digits, against 0.5% for real ones. That
   gap, a factor 10-15, is the main remaining defect on four digits.
2. **K1536 and early 2x.**
   - The order agrees with the user's visual impression: K784 (mean 7.8% over three seeds) > K1536 (6.5%) >
     K1536 + early2x (5.0%).
   - The steps are small and within the noise. The seed spread of K784 alone is 7.0-9.0%. Two sheets of the same
     model gave 2 and 8.
   - 10 of 200 against the K784 mean of 15.7 is p of about 0.08 (Poisson, one-sided).
   - My earlier non-blind "about 3 against 8 per 100" (first 100 samples) overstated the difference.
3. **FID.** Across models, FID follows the same order (6.2 -> 5.7 -> 5.6), but it does not follow the seeds: K784
   seed 2 has the best FID of the three seeds and the most malformed digits.
4. **Precision.** A difference of 2 percentage points needs about 1000+ counted samples per model. A blind count
   suits large differences (ten digits at about 16%, real against model). Smaller ones need an automatic shape
   measure checked against these counts.

### Results: patch variants, 64 states and 8x8 every 2 px (2026-11-13 22:15)
**Jobs.**
- Pipelines 4238574 and 4238575 (0:59 and 1:12), equilibrated sampling 4238576 and 4238577.
- FID 4238578 (`fid/4238578`), grain 4238579, gallery 4238580
  (`galleries/2026-11-13_patch-variants_7x7-64states_8x8-every2px_vs_7x7-32states_from-t2.2_seed1/`).
- Four digits, K784, patches from t2.2, seed 1, n = 500.

| set | FID noise | FID equil | prec F (equil) | rec F (equil) | confident (equil) | speckle (equil) |
|---|---|---|---|---|---|---|
| patch7x7, 32 states (recipe) | 7.68 | 6.42 | 0.897 | 0.791 | 96.2% | +0.55 |
| patch7x7, 64 states | 7.31 | 6.06 | 0.889 | 0.806 | 94.6% | +0.45 |
| **patch8x8 every 2 px, 32 states** | 6.72 | **5.73** | 0.892 | 0.810 | 96.4% | +0.36 |

**Paired against the recipe:**
- 64 states: -0.37 from noise, -0.36 from equilibrated (both p = 0.01).
- 8x8 every 2 px: -0.96 from noise, -0.69 from equilibrated (both p < 0.01).

**Images.**
- In `00_side_by_side.png`, every column has the same digit with the same shape in all three models, including the
  same defects: the broken 1 in column 7, the odd 2 in column 12, the open 0s.
- The 8x8 every-2-px grid has the same malformed tiles as the 7x7 grid (samples 6, 26, 55, 61, 84).
- Only the stroke texture changes: 8x8 every 2 px gives slightly smoother strokes, with speckle +0.36 against +0.55.

**Conclusion.**
- The patch layout is a texture setting. Denser, larger windows clean the strokes a bit more (-0.7 FID), but **no
  patch variant changes a single shape**.
- Together with the ten-digit side-by-side (the malformed shapes are already there in the global-only model), the
  shape of a sample is fixed before t = 2.2, by the global unit, the early slices and the start.
- So shape fixes have to act at t > 2.2: more global states, better-trained early slices (early 2x), or the sampler
  at early times.
- 8x8 every 2 px is a cheap texture gain (1.2x the training time of 7x7) to fold into the recipe later. It is not
  the bottleneck.

### Overnight 2026-11-13/14: what is running, what to look at in the morning (submitted 22:25)
**Already running.**

| what | jobs | expected |
|---|---|---|
| ten digits, K3072 + patch7x7 from t2.2 (pipeline 4237997 finished 21:59) | equil 4238364 -> FID 4238002, grain 4238003, gallery 4238004 | about 00:30-01:30 |
| ten digits, K784 + early2x + patch7x7 from t2.2 | equil 4238947 -> FID 4238948, grain 4238949, gallery 4238950 | about 23:30 |

**New tonight.**
- **Ten digits, K784 + patch7x7 from t2.2, seeds 2 and 3.** Gives the headline number (8.94 at seed 1) an error bar.
  - Pipelines 4240128 and 4240130, equilibrated sampling 4240129 and 4240131.
  - FID over seeds 1-3: 4240132. Expected about 01:30-02:30.
- **Blind shape count, ten digits, job 4240133.** 200 per set, start-equil: real, K784 + patches seeds 1-3,
  K784 + early2x + patches, K3072 + patches. 12 sheets in
  `galleries/2026-11-13_blind-shape-count_ten-digits_K784+patch-seeds1-3_K784+early2x+patch_K3072+patch_real_start-equil/`.
  It waits for all the equilibrated-sampling jobs, so it lands at about 01:30-02:30.

**Morning questions.**
1. Does K3072 reduce the ten-digit shape errors (about 16% malformed at K784)? FID against K784 + patches, and the
   blind count.
2. Does early 2x? Same comparison.
3. What is the seed spread of the ten-digit number?
- **Ten digits, K6144 + patch7x7 from t2.2, seed 1** (the user, 22:40: "maybe already also submit 6k potts").
  Completes the K series 784 / 3072 / 6144 with patches for the paper figure, and asks whether more global states
  fix the shapes.
  - **Timing estimate** from K3072 + patches (4237997): training 2 h 10 min, sampling 2000 from noise 2 h 40 min
    (1190 s per 250). K6144 should take about twice that.
  - **Split to finish by morning.**
    - Pipeline 4240138 (10 h limit, 96 GB): training plus a 250-sample check only.
    - Then two parallel jobs: 2000 samples from noise (4240139) and from the equilibrated start (4240140), each
      9 h limit, 96 GB.
    - The noise job uses the same command as the pipeline's sampling (`--chunk 250`, defaults otherwise).
  - **Follow-ups.** FID 4240141, paired against the 2026-11-10 sweep's global-only K6144, K3072 + patches and
    K784 + patches. Grain 4240142, gallery 4240143
    (`galleries/2026-11-13_ten-digits_patch7x7-from-t2.2_on-K784-K3072-K6144_seed1/`).
  - **Expected about 08:30-10:00** if training starts soon (about 4-4.5 h), then about 5 h of sampling in parallel.
  - The ten-digit blind sheets (4240133) do not include K6144. Add it in the morning.

### Results overnight 2026-11-13/14: ten digits, K3072 + patches, early 2x, seeds (written 2026-11-14 12:40)
**Jobs.**
- FID: 4238002 (`fid/4238002`, all ten-digit sets incl. K3072), 4238948 (early 2x), 4240132 (seeds 1-3).
- Grain: 4238003 and 4238949.
- Galleries: 4238004 (`galleries/2026-11-13_ten-digits_patch7x7-late_vs_global-only_seed1/`, now with K3072) and
  4238950 (`galleries/2026-11-13_ten-digits_K784+early2x+patch7x7-from-t2.2_vs_K784+patch7x7-from-t2.2_seed1/`).
- n = 2000; FID floor 1.69; real images 97.8% confident, precision 0.870, recall 0.882.

| ten digits, n = 2000 | FID noise | FID equil | prec F (eq) | rec F (eq) | confident (eq) | ncomp (eq) |
|---|---|---|---|---|---|---|
| K784, global only (today's code) | 31.08 | 30.12 | 0.904 | 0.284 | 89.2% | +4.62 |
| K3072, global only (sweep 2026-11-10, noise start) | 22.48 | - | 0.924 (noise) | 0.384 (noise) | 92.2% (noise) | +3.43 (noise) |
| K784 + patch7x7 from t2.2, seed 1 | 10.22 | 8.94 | 0.844 | 0.733 | 88.6% | +1.24 |
| K784 + patch7x7 from t2.2, seed 2 | 10.12 | 8.76 | 0.843 | 0.733 | 86.8% | |
| K784 + patch7x7 from t2.2, seed 3 | 10.27 | 8.92 | 0.843 | 0.718 | 89.3% | |
| **K784 + patches, mean of seeds 1-3** | **10.20 +/- 0.08** | **8.87 +/- 0.10** | 0.843 | 0.728 | 88.2% | |
| K784 + early2x + patch7x7 from t2.2, seed 1 | 9.89 | 8.34 | 0.847 | 0.728 | 89.5% | +1.24 |
| **K3072 + patch7x7 from t2.2, seed 1** | **7.03** | **6.37** | 0.879 | 0.758 | **92.6%** | **+0.99** |

- The +/- values are the SD over seeds.
- ncomp = number of ink components, as a deviation from the real test images in their SD. Too many components
  means disconnected pieces and stray dots.

**Paired bootstrap:**
- K3072 + patches against K784 + patches: -3.19 [-3.94, -2.49] from noise and -2.56 [-3.25, -1.85] from
  equilibrated (both p < 0.01).
- K3072 + patches against the sweep's K3072 global-only: -15.45.
- Early 2x: -0.33 (p = 0.15) from noise and -0.60 [-1.00, -0.14] (p = 0.01) from equilibrated.
- Seeds 2 and 3 against seed 1: -0.18 (p = 0.66) and -0.02 (p = 0.96).

**Findings.**
1. **The ten-digit number is stable across seeds:** K784 + patches = 8.87 +/- 0.10 from equilibrated, 10.20 +/- 0.08
   from noise, at 2000 samples. The seed spread (0.1) is much smaller than the effects compared here.
2. **More global states help the shapes on ten digits** (unlike four digits, where K1536 was within the noise).
   - K3072 + patches: FID 6.37 against 8.87, precision 0.879 (real 0.870, so no longer too low), 92.6% confidently
     classified against 88.2%, fewer disconnected pieces (ncomp +0.99 against +1.24).
   - Gallery `00_side_by_side.png`, same trajectories: several of K784's malformed digits are proper digits at
     K3072 (column 1: squiggle 5 -> 9; column 13: scribbled 8/3 -> 8; column 14: S-squiggle -> 2).
   - In `11_K3072+patch7x7-from-t2.2_start-equil_seed1.png` I see about 8-10 clearly malformed of 100, against
     about 16 at K784 (non-blind; the blind count is running).
   - Stray dots next to digits remain ("1.", "8.").
3. **Early 2x is a small effect on ten digits** (-0.6 FID). The side-by-side shows almost the same shapes as without
   it, column by column. Not worth its training cost compared with more global states.
4. **Paper figure** (FID against K, ten digits, noise start, n = 2000):
   - global only: 31.1 (K784), 26.0 (K1536), 22.5 (K3072);
   - with patches: 10.20 +/- 0.08 (K784, 3 seeds), 7.03 (K3072);
   - K6144 + patches pending (below).
   The patches shift the whole curve down by about 15-20 FID. Both curves keep improving with K.

**K6144 + patches: sampling failed overnight (GPU memory), rerun 12:30.**
- Training finished (pipeline 4240138, 3 h 55 min; slices at 166-168 s each). The 250-sample check and both 2000-sample
  jobs (4240139, 4240140) failed with RESOURCE_EXHAUSTED on the 48 GB L40.
- **Cause.** The default sampler vmaps diffrax over the trajectories of a chunk. Each trajectory interpolates its own
  copy of the hidden parameters at its own time, so memory is about chunk x K x 784 x 8 B x 4. At chunk 250 that is
  38.7 GB, the failed allocation. K3072 needs half and fit.
- The fast sampler avoids this (one shared knot matrix), but it refuses patch units and `--init-dist`.
- **Rerun with smaller chunks**, the same per-trajectory load as the K3072 settings that worked: noise start chunk
  100 (**4243060**), equilibrated start chunk 50 (**4243061**). Follow-ups: FID 4243062, grain 4243063, gallery 4243064.
  - Sampling time should be about twice K3072's (2.6-2.7 h), so about 5.5 h: results around 18:30.
- **12:50: the smaller chunks failed too (4243060, 4243061). Chunk was not the main problem.**
  - The noise-start job (chunk 100) failed on 15.6 GB, the chunk-scaled per-trajectory interpolation.
  - The equilibrated-start job (chunk 50) failed on the same 7,885,041,600 bytes as at chunk 125. That is the full
    hidden-parameter history in float64: 100 slices x (2 x 6144 x 784 + patches) x 8 B. It does not depend on the
    chunk.
  - **Diagnosis.** JAX preallocates only 75% of the L40's 48 GB (about 36 GB), and the default sampler holds the
    7.7 GB history several times. At K6144 the remaining pool is too small. The old sweep's K6144 was also only ever
    sampled with the fast sampler (4229560), after its default-sampler run was killed for host memory.
  - **Rerun 12:55.** `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` in the job environment (apptainer passes it on; no
    numerical change) and chunk 25 for both starts:
    - noise start **4243109**, equilibrated start **4243110**, 12 h limits;
    - FID 4243062, grain 4243063 and gallery 4243064 now depend on them (afterok).
  - **If this still fails**, the clean fix is in the code: pass only slice 0's parameters to the equilibration drift,
    and avoid the duplicate history copies in `make_drift` / `interpolate_parameters`.
- **13:05: the 95% pool failed differently (4243109, 4243110).**
  - Both failed with `INTERNAL: Failed to allocate 7885041600 bytes for new constant`.
  - The compiled diffeqsolve embeds the closed-over parameter history (7.9 GB at K6144) as an XLA **constant**, and
    constants are allocated outside the BFC pool. With the default 75% pool, the BFC pool was too small for the batch
    and history buffers. With a 95% pool, too little memory was left outside it for the constant.
  - **Rerun 13:08: `XLA_PYTHON_CLIENT_ALLOCATOR=platform`** (on-demand cudaMalloc from the whole 46 GB for buffers
    and constants alike; no numerical change), chunk 25: noise start **4243157**, equilibrated start **4243158**.
    FID, grain and gallery (4243062-64) depend on them.
  - **Proper fix for later.** Pass the history to the jitted solver as an argument instead of a closure: no
    per-executable constant copies. That would also cut the K3072 memory footprint.

### Ising patch units: build started (2026-11-14 13:15, the user: "replace the Potts patch units by Ising patch units ... could you submit those?")
**Why.**
- Global Ising units lost clearly to Potts: ten digits, FID saturated at about 51-53 against 22.5 for Potts at
  K3072; four digits at 28x28, 66% clean against 92% (README Result 15).
- But the Ising rows had smoother strokes, and a bistable element per unit is far simpler hardware than a K-way
  winner-take-all.
- The patches only set the texture (they never changed a shape), so Ising patch units are a cheap test. Does stroke
  quality hold with independent on/off local features instead of 32-state local mixtures?

**Build** (workflow wf_d1cf272e-031): `train_latent.py --patch-kind ising`, with M Ising units per patch.
- F = -T sum log 2cosh(a/T), new parameter keys Wpi/bpi/Qpi, the split-tree reference per patch, Ising
  learning-rate scales and diagnostics, sampler support.
- Test block 20, plus block 19 rerun to show the Potts path unchanged. GPU smoke run.
- Two reviewers (maths/tests; integration), then a fixer.
- Built in `candidates/ising_patch/`; installed only after the tests pass.

**Planned runs** once installed:
- four digits, K784 + Ising patches 7x7 every 3 px, 32 units per patch (the same number of bistable elements as 32
  Potts states), from t2.2, seed 1, both starts;
- two start scales (chosen from the smoke run's field sizes; small Ising fields are nearly degenerate);
- then ten digits;
- compared with the Potts patch runs on FID, grain (speckle, specks, components), precision/recall and the
  galleries.

**Also 13:20.** The adversarial verification of the on-device-training table (workflow wf_594cc60d-19b) did not run:
the service's safeguards stopped all three agents. That table (steps 1-3 against the local learning rule) stays
unverified.

### Results: blind shape count by three independent raters, four and ten digits (2026-11-14 13:20; workflow wf_eb635ec0-643)
**Method.**
- Three independent rater agents per problem saw only the 12 sheets and README.txt (no key). Each marked every
  malformed tile by position, plus borderline tiles and stray dots.
- A tile counts as malformed if at least 2 of 3 raters marked it (the "majority" column). Per sheet, the majority
  count is close to the unanimous count (in 24 sheets, unanimous is at most 4 lower). So the raters agree well.
- 200 samples per set (2 sheets), start-equil, the same trajectories in every model set.
- Real training images are included as a control.

**Ten digits** (sheets: real 07/08; K3072 01/03; K784 seed 1 10/11, seed 2 04/05, seed 3 02/09; early2x 06/12):

| set (ten digits, start-equil) | malformed (majority) | rate | stray dots (majority) | FID (n = 2000) |
|---|---|---|---|---|
| real training images | 0 + 1 | **0.5%** | 3 + 2 | 1.69 (floor) |
| K784 + patch7x7 from t2.2, seed 1 | 9 + 5 | 7.0% | 4 + 6 | 8.94 |
| K784 + patch7x7 from t2.2, seed 2 | 8 + 8 | 8.0% | 5 + 4 | 8.76 |
| K784 + patch7x7 from t2.2, seed 3 | 9 + 8 | 8.5% | 3 + 6 | 8.92 |
| K784 + early2x + patch7x7 from t2.2 | 10 + 5 | 7.5% | 4 + 5 | 8.34 |
| **K3072 + patch7x7 from t2.2** | 5 + 1 | **3.0%** | 2 + 5 | 6.37 |

- **K3072 cuts the malformed rate by more than half:** 6 against 15.7 per 200 for the K784 mean (p of about 0.006,
  Poisson). The seeds agree (7.0-8.5%).
- **Early 2x: no effect on shapes** (7.5%), although FID moves by -0.6.
- **Stray dots are only mildly above real** (about 4.5% against 2.5%); the raters also flag specks in real digits.

**Four digits** (key from 2026-11-13: real 02/09; K784 seed 1 10/11, seed 2 01/06, seed 3 03/07; K1536 08/12;
K1536 + early2x 04/05):

| set (four digits, start-equil) | malformed, raters (majority) | rate | Claude's count (2026-11-13, one rater) |
|---|---|---|---|
| real training images | 0 + 1 | 0.5% | 0.5% |
| K784 + patches, seeds 1 / 2 / 3 | 10 / 9 / 10 of 200 | 5.0 / 4.5 / 5.0% (mean 4.8%) | 7.5 / 9.0 / 7.0% |
| K1536 + patches | 4 + 6 | 5.0% | 6.5% |
| K1536 + early2x + patches | 4 + 4 | 4.0% | 5.0% |

- **The panel finds no K1536 effect on four digits** (5.0% against 4.8%). Early 2x: 4.0%, within noise.
- My single-rater count from yesterday was more lenient: about 1.4x more tiles counted malformed. It was also
  noisier (I gave the two early-2x sheets 8 and 2; the panel gives 4 and 4). So the K784 > K1536 > K1536 + early2x
  ordering I reported was mostly noise.

**Reading.**
1. All patch models are about 10x above real digits in malformed shapes (4-8% against 0.5%). That is the remaining
   defect.
2. More global states fix it where states per digit are scarce:
   - ten digits, K784 (about 78 states per digit): 7.8%;
   - ten digits, K3072 (about 307 per digit): 3.0%;
   - four digits, K784 or K1536 (196-384 per digit): 4-5%, flat.
   A plausible reading: the malformed rate falls with the number of templates per class until about 200-300, then
   flattens at a few percent. That flat part is what a conceptual change has to attack.
3. FID agrees with the counts across K but not for early 2x (FID -0.6, shapes unchanged) and not across seeds. Use
   both.
- **13:21: K6144 sampling runs** with the platform allocator and chunk 25 (4243157 noise, 4243158 equilibrated).
  - First noise chunk: 381 s for 25, including compilation. Equilibration: 20 s per 25 after the first chunk.
  - About 13 s per sample against 4.8 s at K3072, so about 7-7.5 h per 2000. Expected about 21:00, then FID 4243062,
    grain 4243063, gallery 4243064.
- **14:10: status after the session break.**
  - Noise start 4243157 is running: 245 s per chunk of 25 (9.8 s per sample), 12 of 80 chunks done; expected
    about 18:40.
  - The equilibrated start 4243158 equilibrated (1689 s) but was then killed for **host** memory (96 GB). Resubmitted
    with 160 GB as **4243472** (platform allocator, chunk 25). Expected about 21:00.
  - FID 4243062, grain 4243063 and gallery 4243064 now depend on 4243157 and 4243472.
  - The user stopped all agent workflows: the design panel (wf_d71aa8a7-d23) and the Ising patch build
    (wf_d1cf272e-031) did not finish. The build had written nothing: no candidates/ folder, and the main code files
    are unchanged since 11-13.

### Where we stand, the conceptual options, and next steps (2026-11-14 14:20)
**1. The "most coherent" K6144 gallery: K, not the absence of patches.**
- The user compared the grids in `galleries/2026-11-13_ten-digits_patch7x7-from-t2.2_on-K784-K3072-K6144_seed1/`.
  That folder holds only the grids written before its gallery job failed overnight (02-05: K784 / K3072 + patches;
  06: **K6144 global-only from the 2026-11-10 sweep**). Its side-by-side is still to come (4243064).
- 06 differs from the patch runs in two ways at once: twice the states of K3072, and no patches.
- At equal K the patches never changed a shape. In the side-by-sides, K784 and K3072 give the same digit with the
  same shape column by column, with and without patches. So the coherence comes from K = 6144.
- The K6144 + patches run (sampling now) checks that the patches keep it.

**2. The global unit uses few of its states, and decides the shape between t of about 3.4 and 2.7.**
Seed-1 training logs, ten digits, with patches (4237995, 4237997, 4240138). "eff" = exp of the entropy of the batch
state occupation; "max p" = mean largest posterior.

| slice (t) | K784: eff / max p | K3072: eff / max p | K6144: eff / max p |
|---|---|---|---|
| 0 (5.0) | 701 / 0.015 | 2472 / 0.006 | 4668 / 0.004 |
| 20 (3.99) | **456** / 0.16 | **914** / 0.12 | **1269** / 0.11 |
| 31 (3.43) | 620 / 0.47 | 1441 / 0.39 | 2003 / 0.36 |
| 45 (2.73) | 616 / 0.83 | 1471 / 0.82 | 1980 / 0.83 |
| 55 (2.22) | 612 / 0.91 | 1400 / 0.94 | 1827 / 0.95 |
| 99 (0.0) | 576 / 1.00 | 1350 / 1.00 | 1736 / 1.00 |

- **Few of the states are used:** about 600 of 784, 1400 of 3072 and 1800 of 6144. Going from 3072 to 6144 adds
  only about 400 used states.
- All three collapse at t of about 4 (slice 20) and recover only partly. The slices are trained from t = 5 down, each
  warm-started from the previous one, so the collapse is plausibly passed on downward.
- The top state's class accuracy is 0.93-0.95 from slice 45 on.
- The posterior goes from max p of about 0.4 (t 3.4) to about 0.8 (t 2.7): the shape decision window, before the
  patches switch on at t 2.2.

**3. Noise start against equilibrated start** (the user's question; details in the 2026-11-12 sections on the start
distribution).
- The forward process is OU, dx = -x dt + sqrt(2T) dW. At t = 5 the true distribution still carries e^-5 of the data:
  0.73 noise SDs along the mean digit and about 21% extra variance along PC1 (four digits). The noise start
  N(0, T I) ignores this.
- `--init-dist equil:3` takes the same draws and relaxes them for 3 time units under the model's own t = 5 energy
  (dx = -grad F(x; t=5) ds + sqrt(2T) dW). No data are used. The shift along the mean digit goes from 0 to 0.63 SD
  (true 0.73).
- The gain is about 1 FID everywhere: four digits 7.33 -> 6.24 (3 seeds); ten digits 10.20 -> 8.87 (3 seeds);
  K3072 7.03 -> 6.37.
- What it changes is the class balance: from the noise start too many trajectories end as 1s (too little "ink" along
  the mean digit). Strokes are unchanged.
- The relaxation moves each start state far (median 5.5 against |x| of about 4). So noise and equil rows of a
  gallery show different digits in the same column. Cost: about 2,500 Langevin steps per sample, about +20%
  sampling time.

**4. Assessment: is there a conceptual step?** Hyperparameter changes now give small FID gains and no shape gains
(patch size, states, stride, early 2x). The evidence points at the global unit.
- At low T the global Potts unit is a mixture of K templates (template plus a per-pixel quadratic term), the same
  family as dense associative memories / modern Hopfield networks. That form has two limits:
  - It covers the space of handwritten shapes with K separate "blobs". Coherence then needs very many templates per
    digit; as K approaches the number of training images (60k) the model starts to copy.
  - The hybrids (4/9, 3/5, 8/3) look like the spurious mixture states of an associative memory: in the decision
    window, while two templates are about equally likely, the force points at their average.
- The blind count supports this: the malformed rate falls with templates per digit (78 per digit: 7.8%; about 300:
  3.0%), and at 200-384 per digit (four digits) it is flat at 4-5%, against 0.5% for real digits.
- **Candidate conceptual steps** (my proposal; the design panel that was to critique them was stopped before it
  finished):

| step | idea | why it targets shapes | cost |
|---|---|---|---|
| A. tree hierarchy | the global Potts state sets the patch units' priors: F = -T log sum_g e^{a_g(x)/T} prod_p sum_s e^{(J_p(g,s) + a_ps(x))/T}. Still exactly summable. | each global state becomes a "shape program" over local stroke patterns instead of one pixel template; the patches could act early without overriding the shape | K*P*S per force evaluation (784*64*32 = 1.6M, like a K of about 2000 global unit); hidden-hidden couplings |
| B. per-state soft modes | each global state gets a few low-rank directions of its own (slant, thickness, small shifts): a mixture of factor analysers. The cheaper variant: shared modes with a state-dependent stiffness. | tangent planes instead of blobs. The earlier harmonic modes were shared by all states, plausibly why they gave nothing. | a gated (three-body) or dispersive coupling in hardware |
| C. use the states we pay for | fix the collapse at t of about 4: revive or split dead states (`--hidden-floor`, `--split-dead` exist), or train the slices jointly or in increasing t (DESIGN.md section 5 suggested that order) | if K3072 then behaves like K6144 or better, more shape coverage at no sampling cost | none at sampling time |

- **Cheap diagnostics without retraining:**
  - lock the global state after the decision window (a conditional reverse process);
  - sharpen the global unit inside the window.
  If the hybrids disappear, part of the problem is in the dynamics rather than in capacity.
- **Caution:** at K6144 each state covers about 10 training images. Check nearest-image distances, so that coherence
  is not copying.

**5. Are the steps compatible with on-device training?** My derivation. The adversarial verification of it
(workflow wf_594cc60d-19b) was stopped by the service's safeguards, so it is **unverified**.
- Training is force/score matching: the hidden-unit statistics are measured while the pixels are clamped at data,
  plus small nudges. There is no free-running negative phase, so depth alone does not rule out on-device training.

| step | learning rule on the device | hidden settling while clamped | main cost |
|---|---|---|---|
| C. use the states | unchanged. `--hidden-floor` is local (a state needs only its own visit count); the slice order is a schedule; time-continuous parameters give local gradient x a broadcast basis weight | unchanged | `--split-dead` needs an occasional controller copy of parameters |
| B. per-state modes | almost unchanged: integrating out Gaussian oscillators keeps each state's field quadratic in x, so the DESIGN.md 2.1 rule (occupations, nudge projections) carries over; the oscillator amplitude is an extra local factor | fast (given the state, the oscillators are linear) | gated (three-body) or dispersive coupling |
| A. tree | possible, but the hidden units are no longer independent given x. Each local update then also needs a few scalars broadcast from the clamps (instantaneous force projected on psi, its norm, the total stiffness): a three-factor rule | can be slow: the global unit and the patches lock exactly when two shapes compete | broadcast signal; longer clamped sampling per update |

**6. Ising against Potts** (the user asked whether to investigate Ising units first).
- The record (README Result 15 and the ten-digit table): a flat layer of Ising units loses clearly.
  - Ten digits, global only: FID 53.2 / 51.2 / 52.3 at 784 / 1568 / 3136 units (saturated), against Potts 31.1 /
    26.0 / 22.5.
  - Four digits at 28x28: 66% clean against 92%. Skewed class shares.
  - The idle-unit fixes (reseed, step boost) did not help.
- The likely reason: independent on/off features add up, and nothing makes them form a consistent combination; that
  needs interactions between hidden units. A Potts state encodes a whole digit.
- But Ising units had smoother strokes, and one bistable element per unit is much simpler hardware than a K-way
  winner-take-all.
- **Proposal:** Ising units as the lower level, not the shape selector. The cheapest test: Ising patch units in
  place of the Potts patch units (`--patch-kind ising`), to see whether stroke quality holds. The build started
  (13:15) but was stopped with the other agents before it wrote any code.

**7. Agents stopped (between 13:21 and 14:05, the user's request).**
- The design panel (wf_d71aa8a7-d23) and the Ising patch build (wf_d1cf272e-031) did not finish.
- The blind count (wf_eb635ec0-643) had finished; its results are above.

**8. Next steps (proposed 14:15; waiting for the user's choice).**
1. **K6144 + patches** (running; noise start about 18:40, equilibrated start 4243472 about 21:00). Then FID, the
   gallery and a blind count against K784 / K3072: does the flat part hold at about 600 states per digit?
2. **Is the slow gain from K a training artefact?** Ten-digit K784 + patches with state revival (`--hidden-floor` or
   `--split-dead` at the collapse slices). No new code; 2 runs; FID plus a blind count.
3. **Ising patch units:** restart the build (agents, or directly in the session).
4. **A conceptual step (A or B)**, chosen after 1 and 2 show whether more and better-used states close the gap.

**Correction to next step 2 (14:35).** State revival is not untested.
- `--split-dead` was run on 2026-11-12: four digits, K784, global only, 3 seeds, splits every 5 slices from 20 to 95
  (section "Results: full runs with splits every 5 slices from 20 to 95").
  - It kept about 150 more states alive (41-50 dead at t = 0 against 120-140).
  - FID changed by only -0.45 (n.s.), and the late splits hurt the slice-99 fit.
  - Reading then: the digit is decided at t >= 4, before the first split, and "the number of live states is not the
    bottleneck by itself".
- `--hidden-floor` only works with `--hidden-samples`, and it diverged when tried.
- **Still open:**
  - ten digits, where states per digit are scarce and K clearly reduces malformed shapes;
  - shapes measured by the blind count (11-12 measured FID only);
  - splits or a coarse-to-fine start before t of about 4: near-copies of the high-t winners that share occupation
    instead of starving (candidate fix 2 of 2026-11-12).

### Ising patch units: built directly in the session (2026-11-14 15:00-15:40; the user: "lets do this!", agents stopped)
**What it does.** `train_latent.py --patch SIDE:STRIDE:M --patch-kind ising`.
- The M units of a patch are independent Ising units instead of one M-state Potts unit:
  F_patch = -T sum_{p,s} log 2cosh(a_ps / T).
- The same parameter keys (Wp, bp, Qp), geometry, switch-on (`--patch-on-slice`, `--patch-init-scale`) and
  per-patch learning-rate rules as the Potts patches.

**Code** (candidate in `../latent_physical_SGM_ising_candidate/`; originals frozen in
`../latent_physical_SGM_pre_ising_patch/` for the bit-for-bit tests):
- `latent_model.py`:
  - `patch_terms` / `patch_free_energy` take `kind='potts'|'ising'`. The Ising branch is hidden_terms_quad's Ising
    formulas per patch.
  - `make_loss` / `make_loss_per_sample` / `make_force` take `patch_kind`, default 'potts', so every existing call is
    unchanged.
  - New:
    - `patch_ising_trees`: split_tree per patch, label-free, seeds = the Potts k-means seeds.
    - `patch_ising_reference`: split_reference's formula per patch, bit for bit. Scale lam multiplies W and b.
    - `patch_ising_lr_scales`, `patch_ising_unit_stats`: decided share |m| > 0.9, weak units, <|m|>, rms |a|/T.
- `train_latent.py`:
  - The option and its guard; `patch_kind` recorded in config.json for ising only (Potts and patch-free configs
    unchanged).
  - Resume and `--init-from` refuse another kind.
  - The split-tree reference, start and learning-rate scales; patch_reference.json gains the start field sizes |a|/T.
  - patchi_* diagnostics and an Ising line in the slice log.
- `sample_latent.py`: `load_params` gives the patch tuple as (Wp, bp, Qp, kind) (a 3-tuple still means Potts), and
  the manifest records `patch_kind`. `make_drift` uses the kind; `--fast` still refuses patch runs.
- `early_slices.py`: Ising rows. `make_gallery.py`: patch_kind in the provenance.

**Tests** (new block 20 in `test_latent.py`, plus block 19 rerun on the candidate as the Potts patch regression):
- 4243705: block 20 A-F (formulas against autodiff, loss / force / gradient, mixed precision, bit-for-bit default
  paths, reference = split_reference, sampler drift).
- 4243706: block 20 G (command lines: guard, 100-slice toy run, resume refused, --patch-on-slice, default sampler
  with the normal and equil start, --fast refused, a Potts patch run old vs new code bit for bit).
- 4243707 / 4243708: block 19 A-F / G.
- **Install only if all pass**, then a GPU smoke run and the real runs.

### Ising patch units: tests passed, installed, four-digit runs submitted (2026-11-14 15:20)
**Tests: all 88 checks pass** (block 20 A-F 4243705: 28/28; block 20 G 4243706: 9/9; block 19 A-F 4243707: 37/37;
block 19 G 4243708: 14/14).
- **Formulas.** Ising patch gradient and Laplacian against autodiff: 5.5e-16 / 6.4e-16. One patch covering the grid
  = the global Ising unit, 1.1e-16. Loss against a full-autodiff reference: <= 2.1e-16, four model configurations.
- **Mixed precision.** Gradient error 2-6e-7 relative, cosine 1.00000000 at t = 5 ... 0.
- **Bit for bit unchanged:** the Potts and no-patch paths (loss value, gradient and jaxpr, force, sampler drift), and
  a 3-slice Potts patch training run old against new code.
- **Reference.** The per-patch split-tree reference equals split_reference bit for bit.
- **Command lines.** The 100-slice toy run, resume refusal, `--patch-on-slice`, the default sampler (normal and equil
  start) and the `--fast` refusal all work.
- **Start field sizes** of 7:3 Ising units (M = 32) at the t = 2.22 switch-on (2048 forward samples, four digits):

| --patch-init-scale | median abs(a)/T | 90% | share > 1 |
|---|---|---|---|
| 0.06 (the Potts patch value) | 0.12 | 0.30 | 0.000 |
| 0.25 | 0.49 | 1.24 | 0.18 |
| 1 | 1.95 | 4.97 | most |

  At 0.06 the Ising units would start in the near-degenerate regime (DESIGN.md section 2: a weak unit gets almost no
  gradient), so the runs use 0.25 and 1.

**Installed 15:15:** latent_model, train_latent, sample_latent, early_slices, make_gallery, test_latent.
- The live files were checked unchanged against the frozen copies before the swap. Each file was copied in, then
  renamed into place, latent_model first.
- `../latent_physical_SGM_pre_ising_patch/` keeps the pre-change modules (needed by block 20 D). The candidate
  folder was removed.

**Runs** (four digits, K784, patches 7:3 with 32 Ising units each, from slice 55 = t 2.2, seed 1, n = 500):

| start scale | pipeline | equil:3 | run |
|---|---|---|---|
| 0.25 | 4244038 | 4244039 | `..._mixst_isingpatch7s3M32on55s025_seed1` |
| 1 | 4244040 | 4244041 | `..._mixst_isingpatch7s3M32on55s1_seed1` |

- Follow-ups: FID 4244042, paired against the Potts patch run (7:3:32 from t2.2, seed 1: 7.68 / 6.42) and the
  global-only K784. Grain 4244043 (stroke texture: speckle, specks, components). Gallery 4244044
  (`galleries/2026-11-14_four-digits_K784+isingpatch7x7-32units_vs_pottspatch7x7_from-t2.2_seed1/`).
- **Question:** do 32 independent on/off units per patch clean the strokes as well as one 32-state Potts unit? If
  yes, the patch level can be built from bistable elements without a winner-take-all.
- **15:35: switch-on at slice 55 (training logs 4244038 / 4244040).**
  - The start fields measured on clean training images are larger than on the forward samples of test 20E: median
    |a|/T 2.78 at scale 0.25 and 11.1 at scale 1 (patch_reference.json).
  - **Scale 0.25: healthy.**
    - The Ising patches grow over slices 55-57: |Wp| 2.13 -> 2.67 (reference 7.9); decided share 0.24 -> 0.38;
      22-27 weak units of 2048; <|m|> 0.60-0.64.
    - The global unit is unchanged: top-state class accuracy 0.987, about 510 effective states, <max p> 0.93.
    - The patch force (0.10-0.14) already exceeds the global one (0.02-0.03), as with Potts patches.
  - **Scale 1: the global unit collapses.**
    - From slice 55 on, the global Potts unit has 1.00 effective states and class accuracy 0.25 (chance for four
      digits). The Ising patches (decided 0.79-0.80, force 0.43 against 0.018) take over the late-time score, and the
      slice-55 validation loss improves by -234.
    - The run is kept: the digit is decided by t 2.2, so its samples show whether the patch units alone can hold a
      shape after that.

### Results: Ising patch units get within 1 FID of Potts patch units (2026-11-14 16:40; pipelines 4244038/40, equil 4244039/41, FID 4244042, grain 4244043, gallery 4244044)
Four digits, K784, 7x7 patches every 3 px, from t = 2.2, seed 1, n = 500 (floor 4.57); real images: speckle 0, etc.

| set | FID noise | FID equil | prec F (eq) | rec F (eq) | confident (eq) | ncomp | specks | speckle |
|---|---|---|---|---|---|---|---|---|
| K784, global only | 20.25 | 17.95 | 0.936 | 0.452 | 98.6% | +3.37 | +4.72 | +2.10 |
| K784 + Potts patches (32 states) | 7.68 | 6.42 | 0.897 | 0.791 | 96.2% | +0.78 | +0.29 | +0.55 |
| **K784 + Ising patches (32 units), start scale 0.25** | 8.59 | **7.35** | 0.892 | 0.782 | 95.8% | +1.31 | +0.60 | +0.45 |
| K784 + Ising patches, start scale 1 | 13.30 | 11.99 | 0.820 | 0.783 | 90.8% | +1.75 | +0.94 | +0.65 |

The grain columns are for the equilibrated start, as deviations from the real test images in their SD.

**Paired bootstrap:**
- Ising at scale 0.25 against Potts: +0.91 [+0.48, +1.34] from noise and +0.93 [+0.56, +1.36] from equilibrated.
- Against the global-only model: -10.61 (p < 0.01).
- Ising at scale 1 against Potts: +5.6.

**Findings.**
1. **Independent Ising units recover about 92% of the Potts patch gain** (global only 17.95 -> Potts 6.42 -> Ising
   7.35, from the equilibrated start). Precision, recall and the confident share are within 0.01 of Potts.
   - In the side-by-side, the scale-0.25 rows show the same digits and shapes as the Potts rows, column by column,
     with equally clean strokes.
   - The difference is small isolated specks: specks +0.60 against +0.29, components +1.31 against +0.78. Speckle on
     the ink is even slightly lower (+0.45 against +0.55).
   - So the patch level does not need a winner-take-all. 32 bistable elements per patch without lateral inhibition
     do almost the same job, a much simpler hardware story.
2. **Start scale matters.**
   - At scale 1 the strong Ising patches took over the late-time score at the switch-on, and the global unit
     collapsed to one state (effective states 1.00, class accuracy 0.25).
   - The samples still mostly keep their digit, because it is decided before t 2.2, so the patches alone hold the
     shape after that. But more strokes break (open 0s, dotted 1s), precision drops to 0.82, the confident share to
     90.8%, and FID is +5.6 worse than Potts.
   - Scale 0.25 avoids this: the global unit stays intact (about 510 states, class accuracy 0.99).
3. **Next possible steps:**
   - A smaller scale such as 0.12 (closer to the Potts setting, but still with non-degenerate fields) and seeds 2-3
     for the error bar.
   - Fewer Ising units per patch (8, 16) for the hardware count.
   - Ten digits.

### Review: is a deep (tree) hidden architecture trainable on the device? (2026-11-14 17:00, the user's question)
Claude's reasoning; not yet checked numerically. The automated verification (wf_594cc60d-19b) was stopped by the
service's safeguards.

**Gradient identities.** For any hidden system, F(x) = -T log sum_h exp(-E(x,h)/T) gives the following, with
<.> and Cov over p(h | x), x held at the data:
- grad F = <grad_x E>;
- Delta F = <Delta_x E> - Var(grad_x E)/T;
- d_theta grad F = <d_theta grad E> - Cov(d_theta E, grad E)/T, and similar third-order terms for d_theta Delta F.

**1. No negative phase, for any architecture.** The score-matching gradient needs only these conditional statistics
at clamped data, never the model's distribution over x. So depth alone does not kill on-device training, unlike
Boltzmann-machine likelihood training.

**2. Locality depends on the form of the rule.**
- **Force-matching form** (DESIGN.md 2.1, what the code uses):
  - In one bipartite layer the hidden units are independent given x, so each covariance reduces to the unit's own
    terms, and the rule is local.
  - In a tree (global Potts unit -> patch units -> pixels), the units are correlated through the global unit. Each
    update then needs, besides its local activity, a few scalars broadcast from the clamps: the instantaneous force
    f(t) projected on psi, |f(t)|^2, and the instantaneous total stiffness. That is a three-factor rule with a global
    signal.
- **Short-release (CD-1) form** (DESIGN.md 2, Hyvaerinen 2007; valid for any smooth F):
  dtheta ∝ <d_theta E>_data - <d_theta E>_after release.
  - Every parameter needs only its own average activity before and after a short release. For a global-patch
    coupling that is the co-activation 1[g, s_p].
  - So **a tree stays fully local**, at the known price of noise (signal-to-noise ratio of order sqrt(dt) per
    release).
  - This corrects the earlier table (section "Where we stand ... 5."), which listed only the broadcast variant for the
    tree.

**3. The real cost of depth is hidden equilibration.**
- Both forms need the hidden units at equilibrium given clamped x (and x').
- In a tree, the global and patch units can lock each other exactly when two shapes compete. That means longer
  sampling per update, and lag (friction; heating with two temperatures, DESIGN.md 4) during generation.
- In simulation the exact sum removes this.
- The replica construction for generation (DESIGN.md 3: duplicate the hidden system) gives 2F for any hidden system,
  so it carries over.

**Bottom line.** A tree is trainable on the device in principle: with a broadcast signal (force-matching form) or
fully locally (release form, noisier). The binding constraint is how fast the clamped hidden tree mixes, which grows
with the strength of the global-patch couplings.

**Proposed check (CPU, no GPU).** A small exactly summable tree on 8x8 (K = 4 global, 4:2 patches): verify the
broadcast-form gradient and the release-form identity (finite dt -> 0) against autodiff of the loss.

### HANDOFF: deep tree build spec (2026-11-14 17:20; to be built in a NEW session; exact training only)
**The user's decision.**
- Build the tree (global Potts unit -> patch units) now and run it overnight.
- Exact training only: ignore on-device learning for now. First find out whether the tree raises expressivity
  (shape coherence).
- Continue in a new chat. The old session's /tmp was full, so start Claude Code with
  `CLAUDE_CODE_TMPDIR=/n/fs/cyrills-part/oscillator_dev/.claude_tmp`.

**Model.**
- The global Potts unit (K states; a_g = b_g + W_g.x + Q_g.x^2) and the per-patch Potts units (P patches, S states;
  a_ps = bp_ps + Wp_ps.x_p + Qp_ps.x_p^2) are coupled by a table J (P, K, S): J_p[g, s] is the energy bonus of patch
  state s when the global state is g.
- Hidden part of the free energy (exact; tree):
  F_tree(x) = -T logsumexp_g [ a_g/T + sum_p logsumexp_s ((J_p[g,s] + a_ps)/T) ]
- **J = 0 must reproduce today's model** (hidden_terms_quad 'potts' + patch_terms) to rounding: the first test, and the
  natural start.

**Posterior.**
- p(g | x) = softmax_g(Phi_g / T), Phi_g = a_g + T sum_p log Z_p(g), with Z_p(g) = sum_s exp((J_p[g,s] + a_ps) / T).
- p(s | g, x) = softmax_s((J_p[g,s] + a_ps) / T).
- Patch marginal: q_ps = sum_g pi_g p(s | g).

**Force.** With G_g = W_g + 2 Q_g * x and G_ps = Wp_ps + 2 Qp_ps * x_p:
  grad F = -[ sum_g pi_g G_g + fold(sum_p sum_s q_ps G_ps) ].

**Laplacian.** Delta F = <Delta E> - Var(f)/T, with f the instantaneous hidden force.
- <Delta E> = -[ sum_g pi_g 2 sum_i Q_gi + sum_p sum_s q_ps 2 sum_j Qp_psj ].
- Var(f) = E_g[Var(f | g)] + Var_g(E[f | g]). Given g the patches are independent, so:
  - E_g Var(f | g) = sum_p sum_s q_ps |G_ps|^2 - sum_g pi_g sum_p |gbar_p(g)|^2, with
    gbar_p(g) = sum_s p(s | g) G_ps (P, d per g);
  - Var_g(E[f | g]) = sum_g pi_g |m_g|^2 - |sum_g pi_g m_g|^2, with m_g = G_g + fold(sum_p gbar_p(g)) (N per g).
- Check the whole Laplacian against autodiff of F_tree on 8x8.

**Cost and memory.**
- Per sample: K P S (784 * 64 * 32 = 1.6M) for the logsumexps, K P d S for gbar, K N for m_g.
- At batch 512 in float32: about 1.3 GB for m_g plus 3.3 GB for the (B, K, P, S) posteriors, several times that with
  the backward pass. It should fit the 46 GB L40.
- Expect training about 3-5x slower than K784 + patches (which takes about 1 h on four digits).

**Code changes** (mirror the --patch-kind ising build of today; same safety pattern):
- **latent_model.py:**
  - `tree_terms(W, Q, b, Wp, Qp, bp, J, X, T, with_lap)` -> (grad, lap, ...) and `tree_free_energy(...)`.
  - In `make_loss_per_sample` / `make_force`: if 'J' in params, use tree_terms in place of the global
    hidden_terms_quad + patch_terms (local units, harmonic modes and the visible part unchanged).
  - Potts global unit with Q and Potts patches only (refuse other combinations).
  - Default paths bit for bit unchanged (no 'J' key: as before).
- **train_latent.py:**
  - `--tree` (needs --hidden potts, --quad, K > 0, --patch with --patch-kind potts); config key 'tree': true only when
    set; resume and --init-from refuse a mismatch.
  - J (P, K, S) starts at zeros at the patch switch-on slice; zero before, as Wp.
  - Learning rate of J per patch = the patch bias scale sbp[k][:, None, None] x hidden lr scale x patch lr scale
    (0 before the switch-on).
  - Diagnostics: |J|, the force decomposition, the mutual information between g and the patch states (optional).
  - History: hidden_history.npz member J (100, P, K, S). That is about 1.3 GB in float64: consider float32.
- **sample_latent.py:** load J (load_params with_patch: a 5th tuple entry, or a separate prov['tree']); make_drift
  with the tree force (J interpolated like the other parameters); --fast refuses; manifest records tree.
- early_slices.py / make_gallery.py: tolerate or report the tree.
- **test_latent.py block 21:**
  - tree_terms grad and Laplacian against autodiff of tree_free_energy (8x8, 4:2 patches, K 3, S 3, with and without
    Qp);
  - J = 0 = today's terms;
  - the loss against a full-autodiff reference and the gradient against central differences;
  - mixed precision;
  - the default paths bit for bit against the pre-change modules (freeze copies as for block 20);
  - make_drift against autodiff;
  - command lines on gauss2_8 (toy 100-slice run, sampler, refusals).
- Build in a sibling candidate folder; CPU tests via Slurm; install only if green.

**Overnight runs** (four digits, K784, seed 1, n = 500, both starts; base flags of the recipe):
1. **Tree, patches from slice 31 (t 3.4)**: `--patch 7:3:32 --patch-on-slice 31 --patch-init-scale 0.06 --tree`.
   The decisive one, because shapes are decided at t of about 3.4-2.7. Compare with plain patches from t 3.4
   (`..._mixst_patch7s3S32on31_seed1`: FID about 10.3 noise / 9.0 equil, broken shapes, precision 0.78) and plain
   patches from t 2.2 (`..._mixst_patch7s3S32on55_seed1`: 7.68 / 6.42).
2. **Tree, patches from slice 55 (t 2.2)**: the control, whether J helps at all late.
3. Then FID, grain, gallery, and a **blind 3-rater count** (blind_gallery.py plus the rater workflow if agents are
   allowed again; otherwise counted directly).

**Success criterion:** fewer malformed digits in the blind count than plain patches from t 2.2 (4.8% four digits)
without losing FID; or plain-from-t3.4-level recall with t2.2-level precision.

**State at handoff (17:20).**
- K6144 + patches sampling is still running (noise start 4243157, about 18:40; equilibrated start 4243472, about
  21:00). FID 4243062, grain 4243063 and gallery 4243064 depend on them. Then a blind count against K784 / K3072.
- The Ising patch results are above.
- The current best recipe is unchanged.

### Deep tree: built in a candidate folder, tests running (2026-11-14 18:50)
**Built from the handoff spec** in `../latent_physical_SGM_tree_candidate/`; the live modules are frozen in
`../latent_physical_SGM_pre_tree/` for the bit-for-bit tests (block 21 D, G).
- `latent_model.py`:
  - `tree_terms` (grad, Laplacian, F, pi, q), `tree_free_energy`, `tree_stats`, `check_tree`.
  - The Laplacian's between-g part is computed centred, as Var_g = sum_g pi_g |m_g - E f|^2.
  - **Memory fix:** the spec's estimate left out gbar_p(g), (B, K, P, d), about 5 GB at batch 512 in float32. So
    tree_terms runs over chunks of TREE_CHUNK = 64 samples (lax.map, each chunk rematerialised in the backward pass by
    jax.checkpoint). Memory is then that of one chunk.
  - `make_loss` / `make_loss_per_sample` / `make_force`: with 'J' in params the tree replaces the global unit's and the
    patches' terms. Without 'J' nothing changes.
- `train_latent.py`:
  - `--tree`: needs --hidden potts, --quad, K > 0 and Potts --patch; refused with --split-dead.
  - J starts at zero at the patch switch-on. Learning rate per patch = the patch biases' rate.
  - config 'tree': true only when set. Resume and --init-from refuse a mismatch.
  - Slice files hold J in float64 (zeros before the switch-on). hidden_history.npz holds J in float32, as the last
    member (0.64 GB at K784).
  - Diagnostics tree_*: |J|, rms J/T, I(g; s_p), effective global states, the two parts of the coupled force.
- `sample_latent.py`: the patch tuple becomes a 5-tuple (Wp, bp, Qp, 'potts', J). make_drift uses tree_terms in place of
  the global and patch forces. The manifest records `tree`. `--fast` still refuses patch runs. Also fixed: --thin-knots
  crashed on the patch kind string.
- `early_slices.py` reports rms J/T and I(g; s_p); `make_gallery.py` lists `tree` in the README.

**Tests** (block 21 of the candidate's test_latent.py, CPU):
- 4245123: parts A-E (formulas against autodiff, chunking, J = 0 = today's terms, loss / force / gradient along J,
  mixed precision, default paths bit for bit, sampler drift).
- 4245124: part G (command lines).

**GPU benchmark** 4245127 (`tree_bench.py`): step time and memory at K784, 7:3:32, batch 512, chunks 32/64/128;
mixed against float64 on the real slice-31/55/80 parameters; the drift at --chunk 250; the validation loss; the
estimated training time.

### Deep tree: tests passed, installed, overnight runs submitted (2026-11-14 19:20)
**Tests: all 37 checks pass** (block 21 A-E 4245123: 28/28; block 21 G 4245128: 9/9; the first G job 4245124 failed on
a log line that used K before it was defined; fixed).
- **Formulas.** tree_terms gradient and Laplacian against autodiff of tree_free_energy: 9.6e-15 / 1.5e-14. Chunked =
  one chunk 3.9e-15. J = 0 = the global unit plus the patches 2.8e-14. Loss against a full-autodiff reference:
  <= 2.6e-16, three configurations (with/without Qp, local, harmonic modes). Gradient along J against central
  differences: 3e-9 to 2e-8.
- **Mixed precision** (7:3:8 patches at the scale-1 reference, J rms 2 T, 96 samples, so the chunked path): gradient
  error 1e-6 to 8e-5 relative, cosine 1.00000000 at t = 5 ... 0.
- **Bit for bit unchanged:** make_loss (value, gradient, jaxpr; no patches, Potts and Ising patches; float64 and
  mixed), make_force, make_drift (3/4/5-tuples), and a 3-slice Potts patch training run old against new code.
- **Command lines.** Guards, 100-slice toy run (J float32 (100, 9, 8, 4), last slice = the slice file), resume without
  --tree refused, --patch-on-slice 50 (J zero in 0-49), the sampler with the normal and equil start (manifest tree),
  --fast refused.
- **Note on the diagnostic** I(g; s_p | x): on the toy run it is about 1e-15 although rms J/T = 0.34, because the
  global unit is decided there (pi a delta given x, so no mutual information given x). On the real runs it is non-zero
  only where the global state is uncertain. The coupled force parts (force_rms_tree_glob / _patch) show what J does.

**Installed 19:10:** latent_model, train_latent, sample_latent, early_slices, make_gallery, test_latent, tree_bench (new).
The live files were checked unchanged against `../latent_physical_SGM_pre_tree/` before the swap. That folder keeps
the pre-tree modules (block 21 D, G). The candidate folder is kept until the runs are through.

**Runs** (four digits, K784, Potts patches 7:3:32, start scale 0.06, J = 0 at the switch-on, seed 1, n = 500; the base
flags of the recipe plus `--tree`; pipelines with a 24 h limit because the step cost was not measured yet):

| patches from | pipeline | equil:3 | run |
|---|---|---|---|
| slice 31 (t 3.4), the decisive one | 4245704 | 4245705 | `..._mixst_treepatch7s3S32on31_seed1` |
| slice 55 (t 2.2), the control | 4245706 | 4245707 | `..._mixst_treepatch7s3S32on55_seed1` |

- Follow-ups after all four: FID 4245708 (pairs: tree against plain patches at the same switch-on and start; tree
  t3.4 against plain t2.2, equil), grain 4245709, gallery 4245710
  (`galleries/2026-11-14_four-digits_K784+treepatch7x7_vs_pottspatch7x7_from-t3.4-and-t2.2_seed1/`).
- Then the blind count. **Success:** fewer malformed digits than plain patches from t 2.2 (4.8%) without losing FID
  (7.68 noise / 6.42 equil).
- GPU benchmark 4245127 still queued: step time, memory, chunk size, drift cost at --chunk 250.

### Deep tree: far too slow to train; overnight runs cancelled (2026-11-14 19:50)
**GPU benchmark 4245127** (`tree_bench.py`, L40, K784, 7:3:32, batch 512, mixed, parameters of the plain t3.4 run):
- One training step: **621 ms with the tree, 1.2 ms with plain patches (540 x)**. The temporary memory is only 1.1 GB,
  so memory is not the limit. The chunk size (32/64/128) made no difference (the benchmark may have reused one
  compiled trace; 64 is what training uses).
- Estimated training time: **about 340 h from slice 31, 240 h from slice 55** (the plain run takes 0.74 h).
- Sampler drift at --chunk 250: 50.8 ms against 2.2 ms per call (23 x).
- Validation loss on 4096 samples in float64: 11.3 s (15.5 GB).
- Precision is fine: mixed against float64, J/W/Wp gradient error 5e-6 to 2e-4 at t 3.4 ... 1. (The theta numbers in the
  report compare the stencil with the dense visible energy without masking the dropped pair terms: not a defect.)
- What J does at the real parameters (random J): with rms J = 1 T, I(g; s_p) = 0.18 nats at t 3.4 and the effective
  global states drop from 5.8 to 3.5. So J at this size would change the model at the times that matter.
- **Why:** per sample the tree touches K P S = 1.6 M posteriors and K P d = 2.5 M gbar entries, against K N = 0.6 M for
  the global unit. At batch 512 these are tensors of about 1e9 elements, read and written many times (forward,
  backward, recomputation). The spec's estimate of 3-5 x was off by two orders of magnitude.

**Cancelled** before they started: pipelines 4245704 / 4245706, equil 4245705 / 4245707, FID / grain / gallery
4245708-10. The code stays installed (unused without --tree; default paths bit for bit unchanged).

**Options** (decision for the user):
1. **Tied J:** the 784 global states share J rows within C groups (for example the prototypes' k-means, C = 32-64), so
   J_p[g, s] = J_p[c(g), s]. Exact. The patch part costs C instead of K, about 12-25 x less: perhaps 25-50 ms per
   step, so still about 10-20 h from slice 31. Needs a benchmark.
2. **Top-k truncation:** use only each sample's k most probable global states (k about 8-16; the benchmark shows 1-6
   effective states at t <= 3.4). Exact up to the dropped posterior mass, which can be measured. Choosing the top k
   still needs Z_p(g) for every g.
3. **A small separate root:** a new unit with C = 16-32 states coupled to the patches; the K784 global unit stays
   independent. Exact and the cheapest, but the coupled unit is not the one that decides the shapes now.

**Parked (2026-11-14 23:00; the user: a different research question; first see what the RBM structure gives; maybe
the paper's outlook).** The tree code lives on branch `deep_tree` (commit ddc9df6 on top of 2bc2691, with these
sections and test block 21; not pushed yet: this shell has no GitHub credentials). latent_physical_SGM no longer
carries it: the five tree-only files are back at 2bc2691, and latent_model.py keeps only the Fashion-MNIST additions.

### Results: ten digits, K6144 + patches (2026-11-14 21:30; sampling 4243157 / 4243472, FID 4243062, grain 4243063, gallery 4243064)
n = 2000, floor 1.69. Gallery: `galleries/2026-11-13_ten-digits_patch7x7-from-t2.2_on-K784-K3072-K6144_seed1/`.

| ten digits, patches 7x7 from t2.2, seed 1 | FID noise | FID equil | prec F (eq) | rec F (eq) | confident (eq) | ncomp (eq) | speckle (eq) |
|---|---|---|---|---|---|---|---|
| K784 + patches | 10.22 | 8.94 | 0.844 | 0.733 | 88.6% | +1.24 | +0.71 |
| K3072 + patches | 7.03 | 6.37 | 0.879 | 0.758 | 92.6% | +0.99 | +0.62 |
| **K6144 + patches** | **6.53** | **5.35** | 0.887 | 0.780 | 92.5% | +0.86 | +0.56 |
| K6144 global only (sweep 2026-11-10) | 19.75 | - | 0.930 (noise) | 0.445 (noise) | 92.3% (noise) | +3.02 (noise) | +1.86 (noise) |

**Paired bootstrap:**
- K6144 + patches against K3072 + patches: -0.50 [-1.06, +0.06] (p = 0.08) from noise and -1.03 [-1.59, -0.49] from
  equilibrated.
- Against K784 + patches: -3.59 (equilibrated).
- Against the K6144 global-only: -13.2 (noise).

**Findings.**
1. **Best ten-digit number so far: 5.35** (equilibrated start). The patches keep K6144's shapes: in the
   side-by-side, the K6144 + patches rows show the same digits and shapes as K6144 global-only, with clean strokes.
2. **Returns from K are shrinking.**
   - 784 -> 3072 gave -2.6 FID; 3072 -> 6144 only -1.0.
   - The confident share is flat (92.6% -> 92.5%), as are precision (0.879 -> 0.887) and components (+0.99 -> +0.86).
   - On this evidence the shape gain from K is mostly used up by K3072 on ten digits. The blind count would confirm
     it (not yet run).
3. **Copying check.**
   - The nearest training images of 12 samples are at distances 1.7-5.1 (median about 4). The 1s are closest (1.7-2.2;
     simple shapes).
   - A few samples look like close variants of a training image (the 5 at 3.7, the 7 at 3.1), but none is a copy.
   - For comparison, the median nearest-image distance of real probes was about 4.4 on four digits.
   - A proper copy rate needs evaluate_multiclass on these samples.

### Fashion-MNIST, all ten classes: set-up and first runs (2026-11-14 21:50; the user: "go straight to all 10 classes")
**Data.**
- The official raw Zalando files are in `physical-diffusion-models/data/FashionMNIST/` (md5 match the repository's).
- `make_fashion_data.py` (new) writes `out/latent_physical_SGM/data/fashion_0123456789_28.npz`: the train split,
  /255, 6,000 per class (balanced), shuffled with default_rng(42), as make_digit_data. Job **4246689**.

**Code** (additive; MNIST paths unchanged):
- `latent_model.py`: the problem family `fashion28_c<classes>` (28x28, radius 14, the digit subsets' pixel noise and
  normalisation), plus `load_fashion_raw`, `problem_dataset`, `FASHION_CLASSES`.
- `fid_score.py`: `--dataset fashion` trains and caches `fid/classifier_fashion_28.npz` (the same small CNN, on the
  Fashion train split). Scoring a `fashion28_*` problem uses that classifier and the Fashion **test** images as the
  reference. Job **4246690**.
- `grain_stats.py`: the Fashion test images as its real reference.
- `jobs/run_pipeline_neuronic.sh`: Fashion problems use evaluate_multiclass.
- Backups of the old fid_score / grain_stats are in `../latent_physical_SGM_pre_ising_patch/*_pre_fashion.py`.

**Runs** (problem `fashion28_c0123456789`, the MNIST recipe untuned, seed 1, n = 2000, both starts):

| run | pipeline | equil:3 |
|---|---|---|
| K784, global only | 4246691 | 4246694 |
| K784 + patch7x7 from t2.2 (slice 55, init 0.06) | 4246692 | 4246695 |
| K3072 + patch7x7 from t2.2 | 4246693 | 4246696 |

- Then FID 4246697 (paired: patches against none, K3072 against K784), grain 4246698, gallery 4246699
  (`galleries/2026-11-14_fashion-mnist-10-classes_K784-K3072_patch7x7-from-t2.2_vs_global-only_seed1/`).
- Expected: K784 runs about 1.5 h plus 40 min of equilibrated sampling. K3072 about 5 h plus 3 h. So the K784
  comparison around 01:00 and K3072 in the morning; the FID job waits for all.
- **Caveat.** FID here is in the features of a Fashion classifier, so the numbers are not comparable with the MNIST
  FIDs; compare within Fashion only. The floor is computed in the same way (real training images against the test
  reference).
- **22:50: set-up checks passed.**
  - Data 4246689: 60,000 images, 6,000 per class.
  - Classifier 4246690: Fashion test accuracy 0.9075 after 5 epochs (MNIST's was 0.98-0.99; about 0.91 is usual for
    this small CNN on Fashion), cached as `fid/classifier_fashion_28.npz`.
  - The K784 global-only pipeline 4246691 trains normally (slices 0-1 fine). The other two pipelines start as GPUs
    free up.
- **23:20: early K784-only comparison:** FID 4247127 and gallery 4247128
  (`galleries/2026-11-14_fashion-mnist-10-classes_K784-only_patch7x7-from-t2.2_vs_global-only_seed1/`), after the
  K784 equilibrated sampling; not held up by K3072.

### Fashion-MNIST sweep: patch onset, patch size, K (submitted 2026-11-14 23:35; the user: "play with the hyperparameters")
All ten classes, seed 1, n = 2000, both starts. Compared, paired, with K784 + patch7x7 from t2.2 (4246692 / 4246695).

| run | pipeline | equil:3 |
|---|---|---|
| K784 + patch7x7 from t2.7 (slice 45) | 4247140 | 4247141 |
| K784 + patch7x7 from t1.5 (slice 70) | 4247142 | 4247143 |
| K784 + patch8x8 every 2 px from t2.2 (121 patches) | 4247144 | 4247145 |
| K784 + patch14x14 every 7 px, 64 states, from t2.2 (9 patches; for Fashion's larger textures) | 4247146 | 4247147 |
| K1536 + patch7x7 from t2.2 (with K784 and K3072: three points in K) | 4247148 | 4247149 |

- Follow-ups: FID 4247150, grain 4247151, gallery 4247152
  (`galleries/2026-11-15_fashion-mnist-10-classes_sweep_patch-onset_patch-size_K_seed1/`).
- K6144 is held back unless K3072 shows a large gain over K784 (5.5 h of sampling and memory workarounds).

### Results: Fashion-MNIST, K784 with and without patches (2026-11-15 01:00; FID 4247127, gallery 4247128)
- n = 2000. Fashion-classifier features: the FID scale differs from MNIST's.
- Floor: real training images 0.40. Real images are only 73.8% confidently classified (Fashion classes overlap:
  shirt / T-shirt / pullover / coat).

| set | FID noise | FID equil | prec F (eq) | rec F (eq) | confident (eq) |
|---|---|---|---|---|---|
| K784, global only | 6.67 | 6.39 | 0.867 | 0.457 | 58.6% |
| **K784 + patch7x7 from t2.2** | 3.91 | **3.49** | 0.921 | 0.633 | 61.1% |

- **Patches:** -2.76 from noise and -2.90 from equilibrated (both p < 0.01); recall 0.46 -> 0.63.
- **Images** (gallery `2026-11-14_fashion-mnist-10-classes_K784-only_patch7x7-from-t2.2_vs_global-only_seed1/`):
  - Items of every class are recognisable: T-shirts, trousers, pullovers, dresses, coats, bags, sneakers, ankle boots.
  - As on MNIST, the patches remove the speckle and keep each item's shape, column by column.
  - Defects:
    - **sandals** (and some sneaker soles) come out as fragmented, speckled shapes;
    - textures are washed out: real items have fine stripes, prints and grey levels, while the samples are smoother,
      often brighter and blotchy;
    - a few dark items (dark pullovers, bags) are noisy.
- **Reading.** The recipe transfers. The remaining gap looks like texture and fine-structure detail (sandal straps,
  prints), the regime the patch-size sweep tests (8x8 every 2 px; 14x14 with 64 states).

### Results: Fashion-MNIST sweep, onset / patch size / K (2026-11-15 03:40; FID 4247150, grain 4247151, gallery 4247152)
All ten classes, seed 1, n = 2000; floor 0.40. Paired differences are against K784 + patch7x7 from t2.2 from the
equilibrated start (3.49).

| set (equilibrated start) | FID | paired diff | prec F | rec F | confident |
|---|---|---|---|---|---|
| K784 + patch7x7 from t2.2 (reference) | 3.49 | | 0.921 | 0.633 | 61.1% |
| from t2.7 | 3.65 | +0.16 [-0.01, +0.32] | 0.894 | 0.702 | 60.0% |
| from t1.5 | 3.53 | +0.04 [-0.13, +0.20] | 0.954 | 0.539 | 61.9% |
| patch8x8 every 2 px | 3.62 | +0.13 [+0.02, ...] | 0.931 | 0.638 | 61.5% |
| patch14x14 every 7 px, 64 states | 3.54 | about +0.05 | 0.928 | 0.590 | 61.6% |
| **K1536 + patch7x7 from t2.2** | **3.22** | **-0.27 [-0.54, -0.03]** | 0.917 | 0.660 | 61.6% |

**Findings.**
1. **The patch hyperparameters are near their optimum.** Every onset and size variant is within 0.16 FID of the
   reference.
   - Onset trades precision against recall exactly as on MNIST: earlier (t2.7) gives more recall (0.70) and less
     precision (0.89); later (t1.5) the reverse (0.95 / 0.54). t2.2 is the balance.
2. **Patch size changes texture only.** In the side-by-side, every column shows the same item and shape in all
   variants. 8x8 every 2 px gives the smoothest fills; 14x14 with 64 states gives visibly noisier, speckled fills.
3. **K is the lever, as on MNIST.** K1536 gives -0.27 (significant), and K3072 + patches (4246693, in the morning)
   gives the next point.
4. **Sandals stay fragmented in every variant.** Their thin straps are a shape problem decided before the patches
   act (global unit), not a texture problem. The same holds for the washed-out prints.

**Next.** Wait for K3072. If it gains clearly over K1536, run K6144 (costly: 5.5 h of sampling at chunk 25 with the
platform allocator). Otherwise the Fashion result is: the recipe transfers, and the remaining gap is fine shape
structure.
- **Grain** (4247151, equilibrated start, deviation from the Fashion test images in their SD): speckle on the items is
  +0.75 global only, +0.25 for patch7x7, **+0.09 for patch8x8 every 2 px** (closest to real), +0.49 for patch14x14 with
  64 states. The 8x8 layout gives the cleanest texture at almost the same FID (+0.13), as on MNIST.

### Results: Fashion-MNIST, K3072 + patches; K saturates early (2026-11-15 06:40; pipeline 4246693, equil 4246696, FID 4246697, gallery 4246699)
| Fashion, 10 classes, patch7x7 from t2.2 | FID noise | FID equil | prec F (eq) | rec F (eq) | confident (eq) |
|---|---|---|---|---|---|
| K784 | 3.91 | 3.49 | 0.921 | 0.633 | 61.1% |
| K1536 | 3.49 | **3.22** | 0.917 | 0.660 | 61.6% |
| K3072 | 3.46 | 3.34 | 0.926 | 0.672 | 62.1% |

- **K3072 against K784:** -0.15 [-0.39, +0.13], not significant. K3072 is no better than K1536 (3.34 against 3.22).
- **On Fashion the gain from K is used up at about K1536**, unlike ten-digit MNIST (where it kept paying up to K6144).
  So **K6144 is not run**.
- **Images** (`galleries/2026-11-14_fashion-mnist-10-classes_K784-K3072_patch7x7-from-t2.2_vs_global-only_seed1/`):
  - The same picture as K784: recognisable items, smooth fills.
  - Sandals still fragmented; prints and stripes still missing.
- **Fashion summary so far.**
  - The MNIST recipe transfers: patches -2.9 FID. The best is K1536 + patch7x7 from t2.2 at 3.22 (floor 0.40), and
    8x8 every 2 px is the cleanest texture.
  - The remaining defects (sandal straps, prints) are not fixed by K, onset or patch size: fine structure that a
    template mixture plus local patches does not produce.
- **Possible next steps:**
  - K1536 + patch8x8 every 2 px (combines the best K and the cleanest texture);
  - seeds 2-3 for error bars;
  - the conceptual question (tree, or per-state modes) as for MNIST.
- **06:50: submitted K1536 + patch8x8 every 2 px from t2.2, seeds 1-3** (the best K with the cleanest texture; error
  bars). Pipeline / equil: seed1: 4250490/4250491 seed2: 4250492/4250493 seed3: 4250494/4250495. FID 4250496 (paired with K1536 + patch7x7, seed 1), grain 4250497, gallery 4250498
  (`galleries/2026-11-15_fashion-mnist-10-classes_K1536+patch8x8-every2px-from-t2.2_seeds1-3/`). Expected about
  12:00-13:00.

### Standard Inception FID for MNIST and Fashion-MNIST (2026-11-15 10:10; the user: "could we calculate the real FID?")

- **What:** new script `inception_fid.py`, the FID papers report. It complements the small-classifier FID of
  `fid_score.py`.
  - **Network:** the original TTUR Inception-v3 graph (`inception-2015-12-05.tgz`), downloaded to
    `out/latent_physical_SGM/fid/inception/`, sha256 `009d6814...`. These are the weights pytorch-fid and clean-fid
    reproduce. The input is remapped to a free batch size; features come from `pool_3` (2048-d). TF32 is off.
  - **Images:** grey replicated to 3 channels, 0..255; the graph resizes to 299 itself. Samples are clipped to [0, 1]
    and quantised to 8 bit.
  - **Reference:** train split (60k, the paper convention) and test split (10k).
  - **Per set:** FID at n = 2000; FID_inf (fit FID(n) = FID_inf + b/n over n = 400-2000, Chong & Forsyth 2020), the
    number to compare with 50k-sample FIDs in papers; KID; real-test floor.
- **Smoke test (GPU 4252540):**
  - TF sees the L40; 500 images/s.
  - Batch size 1 vs 8: largest feature difference 3e-3 (rms 0.47).
  - Real vs real at n = 1000: FID 11.1, i.e. the strong small-n bias, hence FID_inf.
- **First results (MNIST 4252541; Fashion 4252542 timed out before its K3072 row):**

  | MNIST, ten digits (n = 2000) | FID train | FID_inf | KID x1e3 |
  |---|---|---|---|
  | real test images (floor) | 3.70 | 1.12 | - |
  | K784, equil start | 112.8 | 110.5 | 126.7 |
  | K784 + patch7x7 from t2.2, equil | 66.2 | 63.4 | 76.8 |
  | K3072 + patch7x7 from t2.2, equil | 58.3 | 56.0 | 66.4 |
  | K6144 (no patches), noise start | 89.1 | 86.6 | 97.3 |
  | K6144 + patch7x7 from t2.2, noise | 55.3 | 52.3 | 63.7 |
  | K6144 + patch7x7 from t2.2, equil | 55.9 | 53.3 | 64.2 |

  | Fashion, ten classes (n = 2000) | FID train | FID_inf | KID x1e3 |
  |---|---|---|---|
  | real test images (floor) | 7.17 | 1.90 | - |
  | K784, equil | 131.1 | 126.6 | 113.1 |
  | K784 + patch7x7 from t2.2, equil | 70.0 | 65.4 | 49.7 |
  | K784 + patch7x7 from t2.7, equil | 70.4 | 66.2 | 50.9 |
  | K784 + patch7x7 from t1.5, equil | 76.8 | 72.9 | 55.9 |
  | K784 + patch8x8 every 2 px from t2.2, equil | **56.7** | **51.7** | **38.2** |
  | K784 + patch14x14 64 states from t2.2, equil | 95.0 | 90.0 | 73.6 |
  | K1536 + patch7x7 from t2.2, noise | 70.3 | 65.1 | 50.9 |
  | K1536 + patch7x7 from t2.2, equil | 68.6 | 64.3 | 48.4 |

- **Reading:**
  - The ranking mostly agrees with the classifier FID: patches help a lot, and more K helps on MNIST.
  - But Inception weighs texture much more:
    - 8x8 every 2 px (the cleanest grain) is clearly best on Fashion, 14 FID below 7x7. In the classifier FID it was
      level.
    - 14x14 patches (the noisiest) are much worse than 7x7.
    - K1536 vs K784 is nearly level.
  - The absolute values (about 50-65) are far above the floor (1-2) and above what published diffusion models reach on
    MNIST (single digits). Suspect: faint pixel noise and grain in the background, to which Inception is known to be
    very sensitive (real MNIST backgrounds are exactly 0).
- **10:45: submitted diagnostics** (MNIST 4253087, Fashion 4253088): real test images + N(0, 0.01^2) and
  N(0, 0.02^2) noise; each sample set with faint pixels (< 0.1) set to 0; faint-pixel share per set. The report is now
  written after every row (no loss on timeout).

### Comparison: Wang & Achour, "Generative Models on Analog Hardware with Dynamics", arXiv 2606.27294 (2026-11-15 11:10)

- **Their model:**
  - "AIS": KuraSHIL oscillators (Kuramoto + second-harmonic injection locking).
  - Deterministic ODE, 80 Euler steps; start N(0, 0.25).
  - 56x56 grid = 3136 elements: 784 visible (the pixels) + 2352 hidden (visible nodes surrounded by hidden ones).
  - 24 couplings per node (5x5 neighbourhood), asymmetric.
  - c = 4 time-piecewise weight sets; 326k parameters.
  - No encoder / decoder: the visible oscillators are the pixels.
  - Trained by WGAN-GP: a digital DCGAN discriminator on the terminal state only, backpropagation through the ODE
    solver (not local, not on-device).
- **Their FID, MNIST / Fashion** (average over 4 seeds; n and implementation not stated):

  | Model | MNIST | Fashion |
  |---|---|---|
  | KuraSHIL c4 (best) | 27.6 | 80.8 |
  | 4 bit + noise 0.025 | 29.1 | 75.1 |
  | no hidden states | 43.0 | 84.3 |
  | c = 1 | 62.7 | 84.4 |
  | DTM (their re-run of Extropic's denoising thermodynamic model) | 107.8 | 112.8 |
  | NLM (Neural Langevin Machine) | 230.5 | 200.8 |

- **Ours (Inception FID against train; job 4252541 / 4252542; n = 2000, so biased upward by about 3-7 at that n):**

  | Model | MNIST | Fashion |
  |---|---|---|
  | best model | 55.3 (FID_inf 52.3; K6144 + patch7x7) | 56.7 (FID_inf 51.7; K784 + patch8x8 every 2 px) |

- **Reading:**
  - MNIST: they are about 2x better.
  - Fashion: we are clearly better (about 52-57 vs 75-81).
  - Their Fashion number is unusually poor relative to their MNIST one.
- **Differences in kind:**
  - Ours: score matching with a local rule, stochastic; hidden Potts units summed out.
  - Ours is far larger and dense: per time slice about 0.6M (K784) to 4.8M (K6144) global couplings + about 0.1M patch
    couplings, 100 slices. Theirs: 326k parameters in total, nearest-neighbour.
- **Pending:** noise diagnostics (4253087 / 4253088). If faint background noise costs much, the MNIST gap may shrink.

### Results: Inception FID noise diagnostics (2026-11-15 11:05; MNIST 4253087, Fashion 4253088)

- **Faint pixel noise is the larger part of our Inception FID on MNIST.**
  - Real test images + N(0, 0.01^2) noise alone: FID 38.5 (MNIST) and 16.2 (Fashion) at n = 2000. With
    N(0, 0.02^2): 61.1 and 27.7.
  - Our samples have the same faint-pixel share as noisy real images: 36% (MNIST) and 27% (Fashion), against 1.7% and
    4.6% for clean real images. The models reproduce the sigma = 0.01 pixel noise of the training data, as they should.
  - Inception counts that noise as a large distribution difference.
- **Faint pixels (< 0.1) set to 0** (diagnostic, not the model's number; FID against train, n = 2000):

  | Model | as sampled | faint zeroed |
  |---|---|---|
  | MNIST K784 | 112.8 | 90.8 |
  | MNIST K784 + patch7x7 | 66.2 | 35.3 |
  | MNIST K3072 + patch7x7 | 58.3 | 27.7 |
  | MNIST K6144 (no patches) | 89.1 | 64.4 |
  | MNIST K6144 + patch7x7, equil | 55.9 | **24.1** |
  | Fashion K784 + patch7x7 | 70.0 | 58.8 |
  | Fashion K784 + patch8x8 every 2 px | 56.7 | **45.8** |
  | Fashion K1536 + patch7x7 | 68.6 | 57.5 |
  | Fashion K3072 + patch7x7 | 68.0 | 56.7 |
  | Fashion K784 + patch14x14 | 95.0 | 77.9 |

- **Reading:**
  - With the background cleaned, the best MNIST model (24.1 at n = 2000, of which about 3 is small-n bias) is at or
    below Wang & Achour's 27.6.
  - Fashion (45.8) stays well below their 80.8.
  - Rankings are unchanged; the patches gain the most from cleaning (the global unit alone stays poor: shapes).
  - Fashion has real faint texture, so thresholding is cruder there.
- **The principled version:** not thresholding, but a final denoising step at the end of sampling, x <- x +
  sigma^2 * score(x, t_end) (Tweedie), or training on noise-free pixels. Both are physically natural (one more
  relaxation step / no extra noise in the data). Not done yet.
- **K3072 vs K1536 on Fashion (Inception):** 68.0 vs 68.6, level, as in the classifier FID.

### Parked: reducing the number of parameters (2026-11-15 11:40; the user: "note this down, but let's not go there for now")

- **Decision:** write the paper first; try parameter reduction at the end. If it works, it becomes a paper section; if
  not, the paper stands without it.
- **Motivation:**
  - Wang & Achour (arXiv 2606.27294): 326k stored parameters and 75k nearest-neighbour couplers in total.
  - Ours: K x 784 global couplers per slice (615k at K784, 4.8M at K6144) + about 100k patch couplers, over 100
    independent slices.
  - Two costs: couplers (energy and area; their power model scales with them) and stored parameters (memory, DAC
    programming, every time copy).
- **Options, roughly by payoff:**
  1. **Share parameters across time.** One template matrix with per-slice per-state gains and biases (an exact mixture
     has time-independent templates; only the temperature changes), or 5-10 knots with interpolation. Stored
     parameters down 10-100x, couplers unchanged. Caveat: state identity across slices (DESIGN.md: "align the
     gauge").
  2. **Prune dead global states.** Effective states are only about 600/784, 1400/3072, 1800/6144; Fashion saturates at
     K1536. Couplers down 1.5-3.5x.
  3. **Small K + patches** (K128 / K256 + patches, untested). Patches gave the biggest gain; shapes may need fewer
     templates. Couplers down 3-25x; the existing code can run it.
  4. **Low-rank global unit.** W = U V, with r ~ 64 linear nodes as an intermediate layer; still exactly summable;
     K6144: about 0.44M instead of 4.8M. The local rule becomes three-layer.
  5. **Sparse or local templates.** Magnitude pruning, or a window per state. Couplers down 3-10x.
  6. **Shared patch dictionary.** One 32 x 49 dictionary plus a bias per position: patch parameters down 64x, couplers
     unchanged.
  7. **Low-bit weights.** 4 / 3 / 2 bits post hoc; their 4-bit model loses nothing.
- **Cheap first batch when we get there:**
  - Sampling-only jobs on the best models (MNIST K6144 + patch7x7, Fashion K784 + patch8x8 every 2 px):
    - (a) every 5th / 10th / 25th slice, interpolated;
    - (b) dead states pruned;
    - (c) 4 / 3 / 2-bit weights;
    - (d) templates sparsified to 10% / 25% of weights.
  - Needs a small sampler option to load modified parameters.
  - Plus retrain K128 / K256 + patches on MNIST and Fashion.
  - Output: a curve of quality vs parameters (couplers and stored).
- **Also still open (not parameter reduction):** a final denoising step at the end of sampling,
  x <- x + sigma^2 score(x, t_end), to remove the sigma = 0.01 pixel noise that inflates the Inception FID (see the
  noise diagnostics above).

### Results: Fashion-MNIST, K1536 + patch8x8 every 2 px from t2.2, seeds 1-3 (2026-11-15 11:25; FID 4250496, grain 4250497, gallery 4250498)

- **Classifier FID (equil / noise start):**

  | Model | equil | noise |
  |---|---|---|
  | K1536 + patch8x8 every 2 px, seed 1 | 3.43 | 3.65 |
  | seed 2 | 3.44 | 3.92 |
  | seed 3 | 3.31 | 3.77 |
  | **mean +- sd over seeds** | **3.39 +- 0.06** | 3.78 +- 0.11 |
  | K1536 + patch7x7, seed 1 (reference) | 3.22 | - |

  - Paired, seed 1: 8x8 - 7x7 = +0.22 [+0.08, +0.33], p = 0.005. In the classifier FID, 7x7 stays slightly better.
  - The seed spread (0.06) is small: one seed is enough to rank settings at differences above about 0.15.
  - Equil start beats noise start in every seed (about -0.4).
- **Grain:** 8x8 is closer to real than 7x7 on every grain measure.

  | Measure | 8x8, seed 1 | 7x7 | real test |
  |---|---|---|---|
  | speckle | 0.0368 | 0.0391 | 0.0357 |
  | holes | 1.52 | 1.67 | 1.26 |
  | components | 3.27 | 3.40 | 3.01 |

- **Gallery** (`galleries/2026-11-15_fashion-mnist-10-classes_K1536+patch8x8-every2px-from-t2.2_seeds1-3/`):
  - The three seeds give nearly the same item in each column (shared start and noise): training is reproducible.
  - Sandals are still fragmented and prints still missing, as in every variant.
- **Inception FID** for these sets submitted (4254741) to settle 7x7 vs 8x8. The two metrics disagree: the classifier
  prefers 7x7 slightly, while Inception preferred 8x8 strongly at K784 (56.7 vs 70.0).
- **Inception FID result (4254741, 11:31; against train, n = 2000):**

  | Model | FID | FID_inf | faint zeroed |
  |---|---|---|---|
  | K1536 + patch7x7, seed 1 | 68.6 | 64.1 | 57.5 |
  | K1536 + patch8x8, seed 1 | 55.6 | 51.2 | 44.8 |
  | seed 2 | 55.8 | 51.5 | 45.6 |
  | seed 3 | 56.2 | 51.5 | 44.8 |
  | **8x8, mean over seeds** | **55.9 +- 0.3** | **51.4** | 45.1 |
  | K784 + patch8x8, seed 1 (earlier) | 56.7 | 51.7-52.2 | 45.8 |

  - Inception prefers 8x8 by about 13 points, far beyond the seed spread (0.3). That matches the grain measures; the
    classifier FID's slight preference for 7x7 (+0.2) is small by comparison.
  - K1536 vs K784 with 8x8 patches: level (55.9 vs 56.7).
  - **Best Fashion setting: K784 or K1536 + patch8x8 every 2 px from t2.2, equil start.** Inception FID about 56
    (FID_inf about 51), against Wang & Achour's 80.8.

### Figure: Inception FID across models, MNIST and Fashion (2026-11-15 11:40)

- **File:** `out/latent_physical_SGM/figures/inception_fid_mnist_fashion.png` (and `.pdf`), made by `plot_inception_fid.py`
  (CPU jobs 4254806 and later re-renders).
- **Content:** two panels; for each model, the FID against train at n = 2000 as sampled (bars), and the faint-zeroed
  diagnostic (dots). Lines: real test images and Wang & Achour.
- **Data source:** the plot reads `report.txt`, not `metrics.json`. In the `metrics.json` of 4253087 / 4253088 /
  4254741, every 'faint zeroed' row had the same key, so only the last one survived. Fixed in `inception_fid.py`: the
  key is now `<label> [faint zeroed]`.

### CelebA 32x32 grey: set-up (2026-11-15 11:50; the user: "lets try CelebA 32x32, gray")

- **Why CelebA first** (not CIFAR-10): faces are aligned, which suits the global template mixture, and they bring
  natural-image texture for the patches. CIFAR-10 (objects at any position and scale) is parked until hidden units
  with memory exist.
- **Data** (`make_celeba_data.py`, CPU job 4254815):
  - Source: Hugging Face parquet export `flwrlabs/celeba`, config `img_align+identity+attr` (the official aligned
    178x218 JPEGs and the official train / valid / test partition, 162,770 / 19,867 / 19,962).
  - Shards in `physical-diffusion-models/data/CelebA/parquet/`.
  - Preprocessing: centre crop 140x140 (box 19, 39, 159, 179), PIL `convert('L')` (luma), antialiased bicubic resize
    to 32x32, 8 bit.
  - Output: `data/CelebA/celeba_gray32_<split>.npz` (uint8, n x 1024), preview `data/CelebA/preview_gray32.png`.
- **Problem `celeba32g`** (`latent_model.py`):
  - 32x32 = 1024 visible units, default radius 16 (runs use radius 1 as before).
  - Full train split; one class, so labels are all 0 (the pooled prototype initialisation uses no labels).
  - Same sigma = 0.01 pixel noise and global normalisation as MNIST / Fashion, for consistency (the noise inflates the
    Inception FID; see the diagnostics above).
  - Backup before the edit: `../latent_physical_SGM_pre_ising_patch/latent_model_pre_celeba.py`.
- **Evaluation:**
  - The classifier FID, multiclass and grain measures are digit / class specific and are not used.
  - `fid_score.load_mnist` loads CelebA (`dataset='celeba'`) for `inception_fid.py`, now grid-generic (28 or 32).
    Reference: the train split (162,770); floor: the official test split.
  - The pipeline skips the class-based evaluation for `celeba*`.
- **Patches:** 8:2:32 gives 13 x 13 = 169 patches of 8x8 on the 32 grid (121 on 28x28).
- **12:16: data done** (job 4254921; the first job 4254815 was cancelled at 4 min per shard, so the conversion now runs
  in parallel over shards on 8 CPUs).
  - Splits: train 162,770 / valid 19,867 / test 19,962 (as the official partition). Pixel mean 0.441 / 0.444 / 0.435.
  - The parquet shards take 12 GB in `data/CelebA/parquet/`. They are a download cache only and can be deleted (ask
    the user first).
  - Preview (`data/CelebA/preview_gray32.png`): centred grey faces, as intended. Rows of the parquet are grouped by
    identity, so never use "the first N images" as a subset.
- **12:20: smoke pipeline submitted** (4254924): K64 + patch8:2:32 from slice 55, `--steps-scale 0.01`, 100 samples.

### Control: Langevin dynamics at one frozen slice, no time evolution (2026-11-15 12:00; the user's idea)

- **Question:** the t = 0 slice's free energy, run as Langevin dynamics at T, has the model's data distribution as its
  stationary distribution. If that mixed in practice, the time evolution would be unnecessary.
- **Expectation:** it does not mix. At t = 0 the basins of different templates and digits are separated by barriers
  far above T = 0.02, and far from the data (where the noise start is) the t = 0 force was never trained. This is the
  standard motivation for annealing (Song & Ermon 2019). Either outcome is useful for the paper: failure argues for
  time-dependent parameters (cf. Wang & Achour, 1 -> 4 weight sets: MNIST FID 62.7 -> 27.6); success would mean a
  fixed-parameter equilibrium sampler suffices (much simpler hardware).
- **Code:** `sample_latent.py --fixed-langevin t:S1,S2,...` reuses `equilibrate()` with the parameters frozen at
  forward time t. It starts from the trajectory set's N(0, T I) states (the diffusion's own start), saves after the
  cumulative durations S, and each segment has its own Brownian keys (fold FIXED_KEY_FOLD = 100 + j). Output:
  `<stem>_langevin-t<t>_S<S>.npy / .json`.
- **New:** `chain_switch.py` classifies the same chains at each checkpoint and reports class shares, confident share,
  nearest-training-image distance, and the share of chains that change class between checkpoints.
- **Run** (GPU 4254826): MNIST ten digits, K784 + patch7x7 seed 1, n = 500, t = 0, S = 1 / 5 / 25.
  - 1617 / 3818 / 14432 accepted steps per segment (111 / 209 / 820 s).
  - No non-finite chains.
  - Median displacement per segment 13.4 / 47.9 / 40.9: the chains keep moving.
- **Queued after it:**
  - gallery 4254856 (`galleries/2026-11-15_mnist-ten-digits_K784+patch7x7_fixed-slice-langevin-at-t0_S1-S5-S25_vs_annealed/`);
  - chain_switch 4254857;
  - classifier FID 4254858;
  - Inception FID 4254859.
- **Results (12:18; chain_switch 4254857, FID 4254858, Inception 4254859, gallery 4254856): the frozen t = 0 dynamics
  fail completely.**

  | Samples | n | classifier FID | Inception FID (train) | prec / rec (features) | class shares |
  |---|---|---|---|---|---|
  | annealed diffusion, equil start | 2000 | 8.94 | 66.2 | 0.84 / 0.73 | 8-12% each |
  | annealed diffusion, noise start | 2000 | 10.22 | 66.1 | 0.84 / 0.75 | 8-13% each |
  | fixed t = 0 Langevin, S = 1 | 500 | 567 | 266 | 0.40 / 0.00 | none confident |
  | fixed t = 0 Langevin, S = 5 | 500 | 456 | 282 | 0.05 / 0.00 | '2' 45%, '3' 31%, '8' 21% |
  | fixed t = 0 Langevin, S = 25 | 500 | 791 | 325 | 0.00 / 0.00 | '2' 85% |

  - **Gallery:** every chain collapses onto an almost blank (black) image, often with a small blob in the top-left
    corner (a corner patch unit). No chain forms a digit at any S. The "class shares" are the classifier's verdict on
    blank images, not digits.
  - The chains keep moving (median displacement 41-48 per segment) but within this blank basin: faint pixels drop from
    45% (S = 1) to 6% (S = 25). They relax into the blank, deeper over time.
  - **Interpretation:** from the diffusion's noise start, the t = 0 force pushes every pixel towards the background
    (most pixels in the data are background). The global Potts unit never selects a template, because the overlap of
    noise with any template is tiny. The model was never trained there, so its t = 0 energy has a spurious "blank"
    basin that captures every chain, and no digit basin is reached in 25 time units (5x the diffusion's duration).
  - **Conclusion for the paper:** the time evolution is essential. Equilibrium sampling at the final parameters does not
    work from noise; the annealed schedule is what carries the state from noise into the digit basins.
- **Possible follow-ups (not run):**
  - t = 0 dynamics started from real training images: are they stationary, and do chains change digit (barrier test)?
  - Frozen parameters at intermediate t (1, 2) followed by a short anneal or quench: how much of the schedule is
    needed?

### Temperature on the sampling side: zero-temperature quench and probability-flow ODE (submitted 2026-11-15 12:50; the user: "start 1 and 2 alongside CelebA")

- **Why:** temperature was never varied.
  - The training temperature T mostly reparametrises time (a uniform shift of the log-SNR), and the time-grid direction
    is closed (11-11), so we do not retrain at another T.
  - Sampling-side tests cost little:
    1. **Zero-temperature quench after sampling** (`--quench`, built earlier, never evaluated). It should remove the
       sigma = 0.01 pixel noise that costs about 30 Inception FID on MNIST. Caveat from the fixed-slice control: the
       t = 0 energy has a spurious blank basin, so long quenches may wash digits out, hence three durations.
    2. **Probability-flow ODE sampler** (`--sampler ode`, deterministic). In theory the same marginals as the SDE;
       with an imperfect model it may differ either way.
  - Only if these show sensitivity: a sampling-temperature knob (force scaled 0.8-1.25).
- **Models** (cheap; best K784 settings):
  - MNIST: K784 + patch7x7 from t2.2, seed 1.
  - Fashion: K784 + patch8x8 every 2 px from t2.2, seed 1.
  - All from the noise start, so the ODE pairs with the existing SDE samples (same start states). The equilibrated
    start cannot be quenched by `sample_latent` (`--init-dist` refuses an existing stem).
- **Jobs:**

  | | quench 0.05 / 0.25 / 1 | ODE | classifier FID | Inception FID | gallery |
  |---|---|---|---|---|---|
  | MNIST | 4254960 | 4254962 | 4254964 | 4254965 | 4254966 |
  | Fashion | 4254961 | 4254963 | 4254967 | 4254968 | 4254969 |

  - Galleries: `galleries/2026-11-15_<mnist|fashion>_<model>_SDE-vs-ODE-sampler_and_zero-temperature-quench-0.05-0.25-1/`.
- **Results (13:05; FID 4254964 / 4254967, Inception 4254965 / 4254968, galleries 4254966 / 4254969).**
  All from the noise start, n = 2000.

  **MNIST, K784 + patch7x7:**

  | Sampler | classifier FID | Inception FID (train) | Inception FID_inf | faint % | prec / rec (features) |
  |---|---|---|---|---|---|
  | SDE (normal) | 10.22 | 66.1 | 63.5 | 36 | 0.84 / 0.75 |
  | SDE + quench 0.05 | 9.79 | 48.9 | 46.6 | 32 | 0.84 / 0.75 |
  | SDE + quench 0.25 | 9.11 | 35.2 | 32.7 | 14 | 0.83 / 0.76 |
  | SDE + quench 1 | **7.66** | **27.9** | **25.4** | 12 | 0.81 / 0.80 |
  | ODE (deterministic) | 22.12 | 76.4 | 73.8 | 37 | 0.85 / 0.56 |

  **Fashion, K784 + patch8x8 every 2 px:**

  | Sampler | classifier FID | Inception FID (train) | Inception FID_inf | faint % | prec / rec (features) |
  |---|---|---|---|---|---|
  | SDE (normal) | 4.09 | 57.6 | 52.7 | 28 | 0.93 / 0.65 |
  | SDE + quench 0.05 | 4.05 | 51.8 | 47.2 | 27 | 0.92 / 0.66 |
  | SDE + quench 0.25 | 3.81 | 37.8 | 33.5 | 24 | 0.92 / 0.67 |
  | SDE + quench 1 | **3.17** | **26.4** | **21.5** | 21 | 0.82 / 0.73 |
  | ODE (deterministic) | 6.73 | 78.3 | 73.3 | 28 | 0.92 / 0.60 |

  - **Quench: a large gain on every metric, growing with the duration up to 1 (the longest tried).**
    - Paired, quench 0.25 vs SDE: MNIST -1.11 [-1.17, -1.04], Fashion -0.28 [-0.34, -0.20] classifier FID.
    - Inception FID roughly halves: MNIST 66 -> 28, Fashion 58 -> 26.
    - With quench 1 we are below Wang & Achour on both data sets as legitimate model numbers, not the
      thresholding diagnostic: MNIST FID_inf 25.4 vs 27.6, Fashion 21.5 vs 80.8.
    - Quench 1 beats even "faint pixels zeroed" (MNIST 26.1 zeroed vs 27.9 as is; Fashion 27.3 vs 26.4), so it
      improves more than the background.
    - The classifier FID improves too (Fashion 3.17: the best Fashion value so far, below K1536 equil 3.22), and recall
      rises (MNIST 0.75 -> 0.80) while feature precision falls slightly.
  - **Gallery:** shapes are unchanged by the quench. Stroke speckle and background dots disappear; Fashion fills
    become smooth. No sign of the blank-basin collapse at these short durations.
  - **ODE sampler: clearly worse** (paired vs SDE: MNIST +11.9 [+9.9, +14.0], Fashion +2.64 [+2.03, +3.43]). Gallery:
    noisy, broken shapes; other items than the SDE for the same start. The stochastic sampler corrects the model's
    errors, and the ODE has no such correction. Do not use the ODE.
  - **Interpretation:** the end of the SDE leaves every sample with the bath's thermal fluctuation (T = 0.02) around
    its local minimum. Inception reads this as noise, and partly the classifier does too. Cooling the bath at the end
    removes it. Physically this is free: switch off the noise source (the bath) for a moment at the end of the protocol.
- **13:10: longer quench and the equilibrated start** (quench-only on `--init-dist equil:3` samples now works:
  `sample_latent` quenches the existing file instead of refusing).

  | | quench, noise start (2, 5, 10) | quench, equil start (0.25, 1, 2, 5, 10) | classifier FID | Inception FID | gallery |
  |---|---|---|---|---|---|
  | MNIST | 4255002 | 4255003 | 4255004 | 4255005 | 4255006 |
  | Fashion | 4255007 | 4255008 | 4255009 | 4255010 | 4255011 |

  - Galleries: `galleries/2026-11-15_<mnist|fashion>_<model>_zero-temperature-quench-durations-up-to-10_noise-and-equil-start/`.

### Bug fix: host-memory leak in training, found with CelebA (2026-11-15 13:20)

- **Symptom:** the CelebA smoke (4254973) was OOM-killed at slice 17 with 64 GB host memory (MaxRSS 67 GB). Fashion
  K1536 runs peaked at 35 GB, so the leak was there before, unnoticed.
  - The first smoke (4254924) was cancelled for a different reason: the patch k-means on the CPU (169 patches x
    162,770 images) took over 30 min. `--ref-kmeans gpu` takes 82 s.
- **Cause:** `train_latent.build_runner`'s jitted `run` has the step count n as a static argument, and n differs from
  slice to slice, so every slice compiles anew. The training data were a closure constant, so XLA embedded the whole
  data set in every executable. Host memory grew by about the data size per slice.
- **Fix:** the data enter `_run` as an argument (`data_dev = jnp.asarray(data_train)`, passed by a thin wrapper `run`);
  `one_step` takes `data=`. Backup of the old version: `../latent_physical_SGM_pre_ising_patch/train_latent_pre_dataarg.py`.
- **Check (job 4255015, then the corrected comparison):** old vs new training, mnist28_d0123, K64, patch 7:3:32 from
  slice 2, 4 slices at 2% steps. All 54 numeric arrays of the 4 slice files are bitwise identical (W, b, theta, Wp,
  bp, Qp, EMA, losses, keys, validation). Only the `record` metadata strings (timings, names) differ. Temporary
  scripts and check runs deleted.
- **13:25: CelebA smoke resubmitted (4255023)** with the fix and `--ref-kmeans gpu`.

### Memorisation check by nearest neighbours: quench does not memorise (2026-11-15 13:35; jobs 4255018 / 4255019; the user: "check whether it's not just memorizing")

- **New script:** `nn_check.py`.
  - For each sample: the Euclidean pixel distance to its nearest training image (raw images the model saw) and to its
    nearest test image.
  - Baseline: 2000 real test images and their nearest training image, i.e. how close a genuinely new real image comes.
  - Figures: per set, triplets (sample | nearest train | nearest test), 12 random samples plus the 6 closest to the
    training set; and a histogram against the baseline.
  - Folders: `galleries/2026-11-15_<mnist|fashion>_<model>_nearest-neighbours-train-and-test_SDE-vs-zero-temperature-quench/`.
- **MNIST** (K784 + patch7x7; baseline real test -> train: median 4.30, 5th percentile 1.69, 1st 1.26):

  | Set | d_train median | d_test median | closer than baseline p5 | d_train < 1 |
  |---|---|---|---|---|
  | SDE, noise start | 4.65 | 5.06 | 0.8% | 0% |
  | + quench 1 | 4.59 | 5.03 | 1.1% | 0% |
  | + quench 10 | 4.69 | 5.09 | 1.1% | 0% |
  | equil start + quench 1 | 4.59 | 5.02 | 1.1% | 0% |

- **Fashion** (K784 + patch8x8 every 2 px; baseline: median 3.48, p5 1.95, p1 1.55):

  | Set | d_train median | d_test median | closer than baseline p5 | d_train < 1 |
  |---|---|---|---|---|
  | SDE, noise start | 3.54 | 3.82 | 1.1% | 0% |
  | + quench 1 | 3.39 | 3.66 | 3.2% | 0% |
  | + quench 10 | 3.46 | 3.76 | 3.3% | 0% |
  | equil start + quench 1 | 3.35 | 3.67 | 2.6% | 0% |

- **Reading: no memorisation, with or without the quench, even at quench 10.**
  - Samples sit about as far from the training set as real unseen images do (median 4.6 vs 4.3; Fashion 3.4 vs 3.5).
  - Fewer samples than real test images are unusually close: 1-3% below the baseline's 5th percentile, where real
    images by definition have 5%.
  - No near-copy (d < 1) at all.
  - The gap d_test > d_train holds for real images too: the training set is 5-6x larger, so its nearest neighbour is
    closer.
  - The quench moves samples slightly towards the data (Fashion median 3.54 -> 3.39) but not onto training images.
- **Pictures:**
  - The random samples differ visibly from their nearest training image (other slant, thickness, proportions).
  - The "closest" samples are simple shapes, upright 1s and trousers, where real test images also have training
    neighbours at 1.5-2.0. Simple classes are dense in the data, so this is not copying.
- **Results, longer quench (13:40; FID 4255004 / 4255009, Inception 4255005 / 4255010, galleries 4255006 / 4255011).**
  Classifier FID raw / clipped; Inception FID against train at n = 2000 (FID_inf in brackets).

  **MNIST, K784 + patch7x7, equilibrated start:**

  | Quench | raw | clip | Inception | rec F |
  |---|---|---|---|---|
  | 0 | 8.94 | 9.62 | 66.2 (63.4) | 0.73 |
  | 0.25 | 7.88 | 8.40 | 34.8 (31.9) | 0.76 |
  | 1 | 6.58 | 7.52 | 27.7 (24.9) | 0.79 |
  | **2** | **5.81** | **7.46** | 24.0 (21.3) | 0.83 |
  | 5 | 18.35 | 9.14 | **21.1 (18.8)** | 0.85 |
  | 10 | 45.72 | 14.00 | 22.5 (20.1) | 0.86 |

  **Fashion, K784 + patch8x8 every 2 px, equilibrated start:**

  | Quench | raw | clip | Inception | rec F |
  |---|---|---|---|---|
  | 0 | 3.62 | 5.48 | 56.7 (52.1) | 0.64 |
  | 0.25 | 3.39 | 4.01 | 37.5 (32.4) | 0.66 |
  | **1** | **3.00** | 3.54 | **26.9 (22.4)** | 0.71 |
  | 2 | 6.06 | **3.45** | 27.5 (23.0) | 0.73 |
  | 5 | 35.92 | 3.46 | 32.0 (27.8) | 0.75 |
  | 10 | 114.55 | 3.80 | 37.2 (31.8) | 0.75 |

  (The noise start gives the same pattern, about 0.5-1 worse on the classifier FID.)
  - **Paired:**
    - MNIST equil: quench 1 vs none -2.36 [-2.49, -2.12]; quench 5 vs quench 1 +11.8.
    - Fashion equil: quench 1 vs none -0.62 [-0.78, -0.44]; quench 5 vs quench 1 +32.9.
  - **Why the raw classifier FID explodes at long quenches:** pixel values overshoot [0, 1]. Clipped, Fashion stays at
    3.5-3.8 up to quench 10, and Inception (which clips) sees no blow-up. Zero-temperature relaxation under the t = 0
    energy drives pixels past the data range: the quartic wells sit outside [0, 1] in pixel units, and the thermal
    noise normally keeps samples inside. On MNIST there is also a mild real decline at quench 10 (clip 14.0, confident
    88% -> 82%).
  - **Gallery (MNIST):** long quenches remove all stray dots and thicken strokes slightly. A few thin parts erode.
    Shapes stay recognisable; no collapse to blank at durations up to 10 (unlike the noise-start fixed-slice control,
    because here the start is already a digit).
  - **Recall rises monotonically with the quench** (MNIST 0.73 -> 0.86, real 0.88), precision falls slightly. The
    quench makes the samples' feature distribution broader and more data-like.
  - **Best recipe:** quench 1-2 after the SDE.
    - MNIST equil + quench 2: classifier FID 5.81 (the best K784 MNIST so far; K6144 + patches without quench was
      5.35) and Inception 24.0.
    - Fashion equil + quench 1: classifier FID 3.00 (best Fashion so far) and Inception 26.9 (FID_inf 22.4, vs Wang &
      Achour 80.8).
    - For the paper, report quench 1 (or 2) on all models; check that the choice does not depend on K (K6144 / K1536
      next).
- **13:34: CelebA smoke passed** (4255023, 27 min, MaxRSS 13 GB instead of 67 GB).
  - Training at 1% steps: about 3-6 s per slice (about 34 s before the fix; the per-slice recompile with the embedded
    data was most of it).
  - Patch units active from slice 55; 100 samples written; summary ran; pipeline exit code 0.
  - Sampling 100 trajectories (padded to a chunk of 250) took 952 s on the barely trained smoke model.
- **13:40: real CelebA runs submitted** (Fashion recipe; seed 1; 2000 samples from the noise start each):

  | Pipeline | Model |
  |---|---|
  | 4255078 | K784, no patches |
  | 4255079 | K784 + patch8x8 every 2 px from t2.2 |
  | 4255080 | K1536 + patch8x8 every 2 px from t2.2 |

  - Afterwards: equilibrated start, quench 1-2, Inception FID (vs CelebA train 162,770; floor = test), gallery,
    nearest-neighbour check.

### Quench 1 on the largest models: MNIST K6144, Fashion K1536 (submitted 2026-11-15 13:50; the user: "also run quench 1 on K6144 MNIST and K1536 Fashion")

- **MNIST K6144 + patch7x7, seed 1** (platform allocator, chunk 25, 160 GB as for its sampling):
  - quench 1 on the noise start 4255371 and the equil start 4255372;
  - classifier FID 4255377, Inception FID 4255378, gallery 4255379
    (`galleries/2026-11-15_mnist_K6144+patch7x7-from-t2.2_zero-temperature-quench1_noise-and-equil-start/`).
- **Fashion K1536 + patch8x8 every 2 px:**
  - quench 1 on the noise start, seed 1: 4255373;
  - quench 1 on the equil start, seeds 1-3: 4255374 / 4255375 / 4255376;
  - classifier FID 4255380 (paired quench vs none per seed), Inception 4255381, gallery 4255382
    (`galleries/2026-11-15_fashion_K1536+patch8x8-every2px-from-t2.2_zero-temperature-quench1_seeds1-3/`).
- **Results (16:45; FID 4255377 / 4255380, Inception 4255378 / 4255381, galleries 4255379 / 4255382): quench 1 helps
  the largest models as much as K784. New bests on both data sets.**

  | Model | classifier FID raw (clip) | Inception FID train (FID_inf) |
  |---|---|---|
  | MNIST K6144 + patch7x7, equil | 5.35 (5.86) | 55.9 (53.4) |
  | MNIST K6144 + patch7x7, equil + quench 1 | **3.87 (4.42)** | **18.8 (15.9)** |
  | MNIST K6144 + patch7x7, noise | 6.53 (7.12) | 55.3 (53.2) |
  | MNIST K6144 + patch7x7, noise + quench 1 | 4.95 (5.64) | 18.0 (15.3) |
  | Fashion K1536 + patch8x8, equil, seeds 1-3 | 3.39 +- 0.06 | 55.9 +- 0.3 (51.2) |
  | Fashion K1536 + patch8x8, equil + quench 1, seeds 1-3 | **2.71 +- 0.07** (2.75 / 2.78 / 2.61) | **25.9 +- 0.04 (21.3)** |

  - **Paired, quench 1 vs none:** MNIST K6144 equil -1.48 [-1.62, -1.29], noise -1.59. Fashion K1536 per seed -0.68 /
    -0.66 / -0.69, all p = 0.005. The gain is very reproducible across seeds.
  - **Inception FID vs Wang & Achour:** MNIST 15.9 vs 27.6, Fashion 21.3 vs 80.8 (FID_inf; theirs: n and code not
    stated).
  - Recall rises as with K784 (MNIST 0.78 -> 0.82, Fashion 0.66 -> 0.73).
  - **Best results so far:**
    - MNIST: K6144 + patch7x7 from t2.2, equil start + quench 1 (classifier FID 3.87).
    - Fashion: K1536 + patch8x8 every 2 px from t2.2, equil + quench 1 (2.71 +- 0.07).

### CelebA 32x32 grey: first full runs (2026-11-15; pipelines 4255078 / 4255079 / 4255080, all exit code 0)

- **Timing:**

  | Model | training | sampling 2000 (noise start) | MaxRSS |
  |---|---|---|---|
  | K784 | 0.62 h | 2745 s | 20 GB |
  | K784 + patch8x8 | 1.29 h | 2240 s | 28 GB |
  | K1536 + patch8x8 | 1.72 h | 7154 s | 45 GB |

  The memory fix keeps host memory bounded.
- **17:55: submitted the evaluation.**
  - quench 1 on the noise samples: 4256031 / 4256033 / 4256035;
  - equilibrated start + quench 1: 4256032 / 4256034 / 4256036;
  - Inception FID of the noise sets (with / without quench): 4256037;
  - gallery: 4256038;
  - nearest-neighbour check: 4256039;
  - after the equil runs: Inception FID of all sets 4256040, equil gallery 4256041;
  - galleries: `galleries/2026-11-15_celeba32-grey_*`.
- **19:05: first look (gallery 4256038, nearest neighbours 4256039; Inception FID 4256037 still running).**
  - The samples are recognisable faces: hair, eyes, mouth, pose and lighting all vary. As sampled they carry the
    familiar speckle, and the quench removes it, as on MNIST and Fashion.
  - The 8x8 patches give smoother, more coherent faces than K784 alone. K1536 + patches looks only slightly better
    than K784 + patches.
  - Weak spots: some faces are smeared or asymmetric, a few have a washed-out background, and fine detail (teeth,
    eyes) is blurred.
  - **No memorisation.**
    - Real test faces lie a median 4.39 from their nearest training face (5th percentile 3.14). Samples lie
      4.0-4.3 away, and none is closer than 1.
    - After the quench, 6-7% of samples are closer than the 5th percentile of the real faces, against 2-3% before it.
    - The closest samples are dark, low-contrast faces. Every dark face is near every other dark face, so these are
      not copies (`nn_05_*.png`).
- **19:15: Inception FID, noise start** (job 4256037; against CelebA train 162,770; n = 2000; floor = real test faces):

  | Model | as sampled: FID (FID_inf) | quench 1: FID (FID_inf) | quench 1: KID x1e3 |
  |---|---|---|---|
  | real test faces (floor) | 6.43 | | 2.2 |
  | K784 | 238.0 (232.9) | 104.4 (99.6) | 120 |
  | K784 + patch8x8 | 98.4 (93.3) | 24.7 (21.3) | 17.1 |
  | K1536 + patch8x8 | 98.2 (94.1) | **23.9 (20.3)** | **16.7** |

  - Inception sees the speckle most clearly on faces: the quench cuts the FID by a factor of 4. Zeroing faint pixels
    helps only by 1-10, so the speckle is not just faint pixels.
  - The patches are essential on CelebA: they cut the quenched FID from 104 to 24. Doubling K adds only about 0.8.
  - Sensitivity check: N(0, 0.01²) pixel noise on real test faces gives 10.8, and N(0, 0.02²) gives 28.1.
  - (The report's footer still says "60k train statistics"; that is a stale string. The statistics are those of the
    CelebA train set.)
- **19:30: more Potts states, and a large K without patches** (the user: "run with more potts and also one without
  patches").
  - Two pipelines, same recipe as before:

    | Pipeline | Model |
    |---|---|
    | **4256247** | K3072 + patch8x8 every 2 px from t2.2 |
    | **4256248** | K3072, no patches |

  - Each: seed 1, 2000 samples from the noise start, `--chunk 100`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`,
    128 GB, 20 h limit.
  - Expected time: K1536 + p8 took 3.8 h (45 GB MaxRSS), so K3072 should take about 8 h.
  - K6144 is held back until K3072 shows whether K still helps on faces: at K1536, doubling K gave only -0.8.
  - Afterwards: quench 1, Inception FID, gallery.

### Quench on K6144 without patches (submitted 2026-11-15 18:05; the user: "without the patches the digits looked kind of good but a bit noisy")

- **Question:** does the quench remove the noise of the global-unit-only model well enough that patches are not
  needed?
- **Run:** `mnist28_d0123456789_K6144_ema_potts_quad_local_quartic_r1_lin_pooled` (the 2026-11-10 sweep; Potts,
  quartic, radius 1; trained with the older float64 path; noise-start samples only).
- **Jobs:** quench 1 and 2 (4256046; platform allocator, chunk 25, 160 GB).
- **Evaluation:** classifier FID 4256047, Inception 4256048, gallery 4256049
  (`galleries/2026-11-15_mnist_K6144-no-patches_vs_K6144+patch7x7_zero-temperature-quench-1-2_noise-start/`).
  - Compared with K6144 + patch7x7 from the same start states, with and without quench 1.
  - Paired: quench 1 vs none; no-patches + quench 1 vs patches + quench 1.
- **Result (19:55): the quench helps, but does not replace the patches.** n = 2000, noise start, test reference:

  | Model | classifier FID | Inception FID (FID_inf) | recall (features) |
  |---|---|---|---|
  | K6144, no patches | 19.75 | 89.1 (87.0) | 0.45 |
  | K6144, no patches, quench 1 | 15.33 | 57.8 (54.9) | 0.54 |
  | K6144, no patches, quench 2 | 14.10 | 52.1 (49.1) | 0.57 |
  | K6144 + patch7x7 | 6.53 | 55.3 (53.5) | 0.76 |
  | K6144 + patch7x7, quench 1 | **4.95** | **18.0 (15.3)** | **0.80** |

  - Paired, no patches: quench 1 vs none -4.42 [-4.54, -4.23]. No patches + quench 1 vs patches + quench 1: +10.38
    [+9.64, +10.88]. Both p = 0.005.
  - **The gallery explains it.** Without patches the strokes are ragged: broken edges, holes and stray dots inside
    the stroke, and blotchy fills (the 0s, the 4s). The quench clears the background speckle, but the stroke-internal
    defects stay; they are local minima of the t = 0 energy. With patches the strokes are smooth and solid.
  - The quench moves the no-patch model only to where the patch model starts *before* its quench (Inception 52-58
    vs 55).
  - Caveat: the no-patch run is from the 2026-11-10 sweep (older float64 training path, noise start only), so it is
    not a perfectly matched control. The gap is far larger than that difference would explain.
  - **Conclusion:** the patches fix the stroke texture, which the global units cannot. On CelebA this is the
    K3072-without-patches run (4256248).

### Figure update: Inception FID with the quench (submitted 2026-11-15 18:20; the user: "replace the red circles with the quenched results")

- `plot_inception_fid.py` now shows, per model:
  - bars: the as-sampled FID;
  - orange dots: the same samples after quench 1 (instead of the faint-pixels-zeroed diagnostic);
  - labels "as sampled -> quenched".
- **Quench 1 for the plotted models that had none** (equil start; the K3072 runs with the platform allocator):
  - MNIST K784 4256052, MNIST K3072 + 7x7 4256054;
  - Fashion K784 4256055, K784 + 14x14 4256058, K784 + 7x7 4256057, K1536 + 7x7 4256056, K3072 + 7x7 4256053;
  - Inception FID: MNIST 4256059, Fashion 4256060.
- **Other sources:**
  - K6144 without patches: noise start + quench 1 (4256048);
  - MNIST K784 + 7x7: 4255005; MNIST K6144 + 7x7: 4255378;
  - Fashion K784 + 8x8: 4255010; Fashion K1536 + 8x8, seeds 1-3: 4255381.
- **Plot job:** 4256062, after 4256048 / 4256059 / 4256060. Output: `out/latent_physical_SGM/figures/inception_fid_mnist_fashion.{png,pdf}`.
- **Done 20:00 (4256062, exit code 0). Inception FID against train, n = 2000, as sampled -> quench 1:**

  | MNIST | | Fashion | |
  |---|---|---|---|
  | K784 | 112.8 -> 86.3 | K784 | 131.1 -> 96.0 |
  | K6144, no patches (noise start) | 89.1 -> 57.8 | K784 + 14x14 (64 states) | 95.0 -> 54.9 |
  | K784 + 7x7 | 66.2 -> 27.7 | K784 + 7x7 | 70.0 -> 36.5 |
  | K3072 + 7x7 | 58.3 -> 21.1 | K1536 + 7x7 | 68.6 -> 35.4 |
  | K6144 + 7x7 | 55.9 -> **18.8** | K3072 + 7x7 | 68.0 -> 34.1 |
  | | | K784 + 8x8 every 2 px | 56.7 -> 26.9 |
  | | | K1536 + 8x8 every 2 px (3 seeds) | 55.9 -> **25.9 ± 0.04** |

  - The quench helps every model, by 25-40 points with patches and less without them.
  - The model order is unchanged.
  - With patches + quench 1, all MNIST models beat Wang & Achour's 27.6 (K784 + 7x7 only just). Every Fashion model
    with patches beats their 80.8, with or without the quench.
  - Floors (real test images): 3.7 (MNIST), 7.2 (Fashion).

### CelebA 64x64 grey and two patch scales (2026-11-15 21:00; the user: "I cannot afford to wait", two patch scales "a great idea")

- **Data (job 4256480, 4 min):** the same crop at 64x64 (4096 visible units). The preview looks right
  (`data/CelebA/preview_gray64.png`).
  - `make_celeba_data.py` takes `CELEBA_SIZE=64` and writes `celeba_gray64_{train,valid,test}.npz`
    (162,770 / 19,867 / 19,962; train pixel mean 0.435).
- **Problem `celeba64g`.**
  - `latent_model`: `celeba_grid(name)`, and `load_celeba_raw(split, grid)`.
  - Dataset name `celeba64`, so its Inception statistics are cached apart from CelebA-32's.
  - `fid_score.load_mnist`, `nn_check` and `inception_fid` accept 64x64.
  - Backups: `../latent_physical_SGM_pre_ising_patch/*_pre_celeba64.py`.
- **Estimates (scaled from CelebA-32, not measured yet):**
  - K1536 at 64x64 has 6.3M global couplings per slice, comparable to MNIST K6144 (4.8M).
  - Training about 7 h; sampling 2000 about 8 h at chunk 50; host memory 100-150 GB.
- **Submitted:**
  - smoke test **4256481**: K1536 + patch16x16 every 4 px, 1% of the steps, 100 samples at chunk 25, 160 GB;
  - full run **4256482**: K1536 + patch16x16 every 4 px (16:4:32 from slice 55, init 0.06), 2000 samples from
    the noise start at chunk 50, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, 160 GB, 2-day limit.
  - Why 16x16 every 4 px: at 64x64 it covers the same share of the face as 8x8 every 2 px at 32x32.
  - Both are held (`scontrol hold`) until the --patch2 refactor below passes its regression test: they would start
    with the changed code.
- **Two patch scales: `train_latent.py --patch2 SIDE:STRIDE:STATES`.**
  - A second set of Potts patch units with keys `Wp2`, `bp2`, `Qp2`, next to `--patch`.
  - Each set has its own k-means reference (seeds PATCH_SEED0 + 100000 + 1000 (seed - 1) + p;
    `patch2_reference.json`) and its own learning-rate scales.
  - Shared with `--patch`: init scale, switch-on slice and lr scale.
  - Diagnostics: record keys `patch2_*` and `force_rms_patch2`. The second set is stored in the slice files and,
    last, in `hidden_history.npz`.
  - `latent_model.PATCH_SETS`: the loss and the force sum over the sets present, the first set exactly as before.
  - `sample_latent`: `prov['patch']` becomes a list of both sets; `make_drift` sums them; the manifest's
    `patch_units` reads e.g. `16:4:32+8:2:32`. Fixed on the way: `--thin-knots` on a patch run sliced the kind
    string.
  - Backups: `*_pre_patch2.py`.
- **Test job 4256601.**
  1. Regression: old vs new code, `--patch` only, 4 slices, must be bit-identical.
  2. A two-scale MNIST run (7:3:16 + 4:3:8) over all 100 slices: sampling, quench and equilibrated start.
  3. Three slices of celeba64g with 16:4:32 + 8:2:32.
  - Temporary old-code copy: `../latent_physical_SGM_oldcheck/` (deleted after).
- **21:50: test 4256601 PASSED (33 min).**
  - Regression: `--patch` only, old vs new code, 4 slices. Every array is bit-identical, and so is every record value
    except the timings.
  - Two scales on MNIST: 7:3:16 + 4:3:8, all 100 slices.
    - Wp/bp/Qp and Wp2/bp2/Qp2 are zero before slice 55 and trained after it.
    - The `patch2_*` diagnostics are recorded.
    - Sampling, quench 1 and the equilibrated start give finite samples. The manifest records `7:3:16+4:3:8`; the
      quench's manifest records no `patch_units`, which is cosmetic.
  - celeba64g: K16 reference k-means 119 s. Patch k-means: 16:4:32 in 315 s (explained variance median 0.73),
    8:2:32 in 347 s (0.85). 3 slices trained.
  - The test runs and `../latent_physical_SGM_oldcheck/` are deleted.
- **Holds released** (4256481 smoke, 4256482 K1536 + 16x16/4).
- **Submitted 4256661:** K1536 + patch16x16 every 4 px + patch8x8 every 2 px (`--patch 16:4:32 --patch2 8:2:32`, from
  slice 55, init 0.06). Same settings as 4256482: 2000 samples at chunk 50, 160 GB, 2-day limit.
- **Watch item:** the 64x64 test slices ran at about 1 s per step at K16, with only 22 steps per slice, so compile
  time probably dominates. The smoke test (1% of the steps, about 110 per slice) gives the real per-step time. If it
  is about 1 s, the full runs (11,000 steps per slice) would not fit, and the step count or the steps schedule has to
  change.

### CelebA 32x32: equilibrated start (2026-11-15 22:30; sampling 4256032 / 34 / 36, Inception 4256040, gallery 4256041)

- **Inception FID against train, n = 2000 (FID_inf in brackets); floor 6.43:**

  | Model | noise start | equil:3 start | noise + quench 1 | equil + quench 1 |
  |---|---|---|---|---|
  | K784 | 238.0 | 232.8 | 104.4 | 99.1 (96.2) |
  | K784 + patch8x8 | 98.4 | 94.6 | 24.7 | **23.8 (19.9)** |
  | K1536 + patch8x8 | 98.2 | 94.5 | 23.9 | **23.9 (20.6)** |

  - The equilibrated start gains about 4 as sampled, but after the quench it gains only 0-1. The quench removes
    most of what the start fixes.
  - K784 and K1536 with patches are equal within error after the quench, so on CelebA-32, K beyond 784 buys
    nothing so far. The K3072 runs (4256247/8) test this further.
- **Gallery** (`galleries/2026-11-15_celeba32-grey_K784-vs-K784+patch8x8-vs-K1536+patch8x8_equil-start_with-and-without-quench1/`):
  - The same picture as from the noise start. The quench gives clean, smooth faces, a little blurry.
  - The patch models have more coherent hair and face outlines than K784 alone.
  - Some faces have a dark smudge over one eye, or an unclear face/background boundary.

### Change of plan: two patch scales on CelebA 32 first (2026-11-15 22:45; the user: "maybe then stop the 64x64 after all...? and first investigate the two scale on 32?")

- **Cancelled before they started:** CelebA-64 full runs 4256482 (K1536 + 16x16/4) and 4256661 (K1536 + 16x16/4 +
  8x8/2). The smoke test 4256481 keeps running for the speed and memory numbers (27-29 steps/s at K1536 before
  the patches).
- **Two-scale runs on CelebA-32.**
  - Both are K784, the size that ties K1536 after the quench.
  - Both keep the proven `--patch 8:2:32` (identical start to the K784 + 8x8 run, so the comparison is paired) and
    add a second scale from slice 55:

    | Pipeline | Second scale |
    |---|---|
    | **4256794** | `--patch2 4:2:32`, finer: 15x15 = 225 patches of 4x4 (fine detail: eyes, teeth) |
    | **4256795** | `--patch2 16:4:32`, coarser: 5x5 = 25 patches of 16x16 (face parts, outline, the smudges) |

  - 2000 samples from the noise start each.
- **Evaluation chain:**
  - quench 1: 4256796 / 4256797;
  - Inception FID of both runs and the one-scale K784 + 8x8 baseline, with and without the quench: 4256798;
  - gallery 4256799 (`galleries/2026-11-15_celeba32-grey_K784+patch8x8_vs_+patch4x4_vs_+patch16x16_two-scale-patches_noise-start_with-and-without-quench1/`).

### Disk full: /n/fs/cyrills-part at 99% (2026-11-15 23:00; CS disk-monitor email at 95%)

- **df:** 492 of 500 GB used, 9 GB free. That was not enough for the running jobs.
  - Each slice file of the CelebA-64 K1536 smoke test was 123 MB, and its assembled history would have added about
    12 GB more.
  - Writes stop at 102%, or after 7 days above 100%.
- **Done:**
  - Cancelled the smoke test 4256481 at slice 87. Its speed numbers are above.
  - Held the two-scale jobs, then released them once there was room.
  - Deleted, with the user's approval:
    - the CelebA parquet download cache (11.7 GB);
    - the smoke run (10.7 GB);
    - `celeba_gray64_*.npz` (0.8 GB). To rebuild them: re-download (`make_celeba_data.py` fetches missing
      shards), then `CELEBA_SIZE=64`.
  - Now 470 GB used, 31 GB free (94%).
- **Storage per run:** the per-slice files in `slices/` and the assembled `hidden_history.npz` hold the same
  parameters, so finished runs store them twice. CelebA-32: about 5 GB per K3072 run each way. 64x64: about 12 GB
  per K1536 run each way.
- **CelebA-32 K3072 without patches (4256248): training done, sampling FAILED.** GPU memory: "Failed to allocate
  5049912000 bytes for new constant", the history in float64, as with MNIST K6144. This was not the disk.
  - Rerun of the sampling only, **4256809**: `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, chunk 25, 160 GB.
  - The same for K3072 + patches as **4256810**, which runs only if 4256247 fails (`afternotok`).
- **CelebA-64 smoke 4256481, the speed from its log (before the cancel):**
  - K1536 without patches: 27-46 steps/s.
  - From slice 55, with the 16x16 patches every 4 px: **about 5 steps/s, nine times slower**.
  - Peak host memory: 29.5 GB.
  - At full steps (about 22,000-28,000 per patch slice, 45 slices) that is about 60 h of training for the patch
    slices alone. The 2-day runs would not have fit; the cancelled 4256482 / 4256661 would have timed out.
  - Likely cause: `PatchGeometry.unfold` / `fold` build one strided slice or pad per patch entry, 256 for 16x16, so
    the graph and the memory traffic grow with side². 64x64 needs a cheaper unfold (a convolution with an identity
    kernel, or a reshape when stride = side / 4) before it is worth running.
  - The same may slow the CelebA-32 run with `--patch2 16:4:32` (4256795); its log will show it.
- **23:15: CelebA-32 K3072 + patch8x8 (4256247): training done (all 100 slices), sampling failed** on GPU memory
  ("Failed to allocate 5608017600 bytes for new constant"), like the run without patches.
  - The backup sampling **4256810** is now active.
  - Evaluation chain after the two sampling reruns:
    - quench 1: 4256859 (no patches) / 4256860 (patches), platform allocator, chunk 25;
    - Inception FID **4256861**: K3072 with and without patches, against the quenched K784 / K784 + p8 /
      K1536 + p8;
    - gallery **4256862** (`galleries/2026-11-16_celeba32-grey_K784-vs-K3072_with-and-without-patch8x8_noise-start_quench1/`).
  - Disk at 23:15: 476 of 500 GB (96%, 25 GB free). The two-scale runs 4256794 / 4256795 are training.
- **23:40: K1536 with two patch scales** (the user: "maybe submit one with two scale patches on K15..").
  - Pipeline **4257324**: K1536 + `--patch 8:2:32 --patch2 16:4:32` from slice 55, seed 1, 2000 samples from the
    noise start, 80 GB, 16 h limit.
  - The coarse second scale was chosen because the remaining face defects are mid-scale (eye smudges, outlines);
    the quench already removes the fine speckle.
  - quench 1: 4257325; Inception FID against the one-scale K1536 + 8x8, with and without the quench: 4257326.
  - Disk: 478 of 500 GB (23 GB free). This run adds about 5 GB, and the jobs queued before it about 7 GB.

### 2026-11-16 morning: overnight results

- **Disk:** the share is now 1 TB (the user asked for more space): 490 GB used, 535 GB free, 48%.
- **CelebA-32: more Potts states** (sampling 4256809 / 4256810, quench 4256859 / 4256860, Inception **4256861**,
  gallery 4256862). Inception FID against train, n = 2000, noise start; floor 6.43:

  | Model | as sampled | quench 1 (FID_inf) |
  |---|---|---|
  | K784 | 238.0 | 104.4 (100.6) |
  | K3072 | 231.3 | 98.5 (94.1) |
  | K784 + patch8x8 | 98.4 | 24.7 (21.3) |
  | K1536 + patch8x8 | 98.2 | 23.9 (20.5) |
  | K3072 + patch8x8 | 97.7 | **23.8 (20.1)** |

  - With patches, K784 -> K1536 -> K3072 gains 0.8, then 0.1 more: K is saturated on CelebA-32. Without patches
    K3072 gains 6, but stays about 75 above the patch models.
  - The gallery (`galleries/2026-11-16_celeba32-grey_K784-vs-K3072_with-and-without-patch8x8_noise-start_quench1/`)
    agrees.
    - K3072 + patches after the quench has a little more contrast and definition in some faces (sunglasses, hair
      edges), but the same failure modes as K784 + patches: an eye smudge, blur.
    - Without patches even K3072 stays mottled after the quench.
  - Sampling time: K3072 at chunk 25 with the platform allocator took 3.5 h (no patches) and 3.8 h (patches).
- **Two-scale runs: all trained and sampled, but their evaluation failed** (Inception 4256798 / 4257326 skipped
  the sets; gallery 4256799 failed).
  - Training: K784 + 8x8 + 4x4 in 2.4 h (4256794); K784 + 8x8 + 16x16 in 5.3 h (4256795); K1536 + 8x8 + 16x16 in
    6.2 h (4257324). Quench 1 done for all three.
  - **Cause:** the `+` in the run names (`patch8s2S32+patch16s4S32`) clashed with the `PATH+PATH` concatenation
    syntax of `evaluate_samples.load_set`, `make_gallery`, `fid_score` and `inception_fid`, which split the path
    at the `+`.
  - **Fix:** `latent_model.split_set_spec`, which joins the pieces until each ends in `.npy`, is now used in all
    four; `make_gallery` also lists `patch2` in its README. Backups `*_pre_plusname.py`.
  - **Rerun:** Inception FID **4261206** for all five models (K784 + 8x8, + 4x4, + 16x16; K1536 + 8x8,
    + 16x16), with and without the quench; gallery **4261207**
    (`galleries/2026-11-16_celeba32-grey_K784-and-K1536_patch8x8-alone-vs-two-scale-patches_noise-start_with-and-without-quench1/`).
- **Two-scale results (Inception 4261206, gallery 4261207).** Inception FID against train, n = 2000, noise start
  (bootstrap sd about 0.3-0.4); floor 6.43:

  | Model | as sampled | quench 1 (FID_inf) | KID x1e3 (q1) |
  |---|---|---|---|
  | K784 + 8x8 | 98.4 | 24.7 (20.9) | 17.5 |
  | K784 + 8x8 + 4x4 every 2 px | **78.3** | 23.3 (19.7) | 15.9 |
  | K784 + 8x8 + 16x16 every 4 px | 93.6 | 22.9 (19.2) | 15.4 |
  | K1536 + 8x8 | 98.2 | 23.9 (20.4) | 16.6 |
  | K1536 + 8x8 + 16x16 every 4 px | 92.6 | **22.4 (18.6)** | **14.6** |

  - **A second scale helps, modestly:** -1.4 (4x4) and -1.8 (16x16) at K784, and -1.6 (16x16) at K1536, after the
    quench. That is 4-6 bootstrap sd, from the same start and noise in every run.
  - **The two scales act differently.** The fine 4x4 scale removes much of the speckle during sampling (98 -> 78 as
    sampled), but the quench does most of that anyway. The coarse 16x16 scale changes less before the quench and
    gains more after it, so it adds structure that the quench cannot make.
  - **New best on CelebA-32:** K1536 + 8x8 + 16x16, quench 1: 22.4 (FID_inf 18.6).
  - **Gallery:** the differences are small, as the numbers say.
    - With 16x16, some faces are more coherent: column 1's dark eye smudge is lighter and more face-like, and
      column 3's hat and face are better separated.
    - With 4x4, the as-sampled faces are visibly cleaner, while the quenched ones look like the baseline.
    - The main defects (blur, the occasional smudge) remain in every variant.
  - Cost: the 16x16 set doubles to quadruples training time (K784: 1.3 h -> 5.3 h; the slow unfold / fold, see the
    64x64 smoke). The 4x4 set costs little (2.4 h).
- **Figure update (2026-11-16 14:30; the user: "update the inception_fid_mnist_fashion.png figure and include the
  celebA runs").**
  - `plot_inception_fid.py` has a third panel, CelebA 32x32 grey, noise start: K784, K3072, K784 / K1536 / K3072
    + 8x8, and the three two-scale runs. Sources: 4256037, 4256861, 4261206.
  - The CelebA panel has no Wang & Achour line (they report no CelebA result).
  - Rows are now sorted by the quenched FID (the dots), best at the bottom.
  - The footnote takes three lines and the x range is wider, so the value labels fit.
  - Jobs 4262571 and 4262581; same file name, `figures/inception_fid_mnist_fashion.{png,pdf}`.

### Faster patch unfold / fold with blocks (2026-11-16 15:00; the user: "lets do it")

- **Problem:** `PatchGeometry.unfold` / `fold` use one strided slice / pad per patch entry: 256 for 16x16 patches.
  - Training differentiates through them twice, so each step becomes thousands of small GPU operations.
  - The 16x16 set's arithmetic is small (64x64: about 1.4M multiply-adds per image vs 6.3M for K1536), yet it
    slowed training 9x on 64x64, and from 1.3 h to 5.3 h for K784 on CelebA-32. So the overhead, not the
    arithmetic, is the suspect.
- **Change (`latent_model.PatchGeometry`, `impl = 'blocks'`):** when the side and the grid are multiples of the
  stride (8:2, 16:4, 4:2; not MNIST's 7:3):
  - the image is cut into stride x stride blocks, and each patch is an m x m group of blocks (m = side / stride);
  - unfold / fold take m² contiguous slices / pads of the block array: 16 for 16:4 instead of 256, 16 for 8:2
    instead of 64, 4 for 4:2 instead of 16;
  - still slices and pads only, so still deterministic.
  - **The same numbers:** unfold only copies, and fold adds each pixel's patch contributions in the same order (by
    patch offset u, then v) as the slice version; the slice version's extra terms are exact zeros.
  - Other layouts keep the slice version. `LATENT_PATCH_IMPL=slices` forces it everywhere.
  - Backup: `*_pre_blockunfold.py`.
- **Test job 4262849:**
  - `check_blockunfold_tmp.py`: unfold / fold, the adjoint identity, and the loss and its gradients with respect to
    Wp, Qp, bp and X, blocks vs slices, for 8:2, 4:2, 16:4 on 32x32, 16:4 and 8:2 on 64x64, and 7:3 on 28x28
    (slices both), in float32 and float64; plus the time per loss-and-gradient call.
  - Then CelebA-32 K784 + 8:2:32 + 16:4:32, 5 slices at 2% of the steps (patches from slice 1), trained once with
    each implementation: slice files compared, steps/s from the logs.
- **`MODEL_AND_CODE.md` written** (2026-11-16 15:45; the user: "do we actually have some kind of write up where
  those code changes and choices are listed?").
  - The model, parameter counts, the current recipe with the reason for each choice and links to the dated entries,
    the best models with their command lines, the evaluation scripts, the code changes since the paper code with
    their verification, known pitfalls, and data on disk.
  - Section 8 above now points to it, and the `train_latent.py` header now documents `--patch2`.
- **Block unfold: test 4262849 PASSED (12 min).**
  - **Exactness** (same random inputs, both implementations): unfold, fold, the loss and its gradients (dX, dWp, dQp,
    dbp) are bitwise identical for 8:2, 4:2, 16:4 on 32x32 and 16:4 on 64x64, in float32 and float64. One
    exception: on 64x64 8:2 the loss differs at 8.7e-8 relative (float32), the gradients still bitwise.
  - **Time per loss-and-gradient of the patch term alone** (B = 256, float32, L40):

    | Layout | slices | blocks | speed-up |
    |---|---|---|---|
    | 32x32, 8x8 every 2 px | 3.95 ms | 0.86 ms | **4.6x** |
    | 32x32, 16x16 every 4 px | 0.70 ms | 0.30 ms | 2.3x |
    | 32x32, 4x4 every 2 px | 0.34 ms | 0.31 ms | 1.1x |
    | 64x64, 16x16 every 4 px | 5.72 ms | 4.75 ms | 1.2x |
    | 64x64, 8x8 every 2 px | 8.70 ms | 14.51 ms | **0.6x (slower)** |
    | 28x28, 7x7 every 3 px | slices in both | | |

  - **Training, CelebA-32 K784 + 8x8 + 16x16** (5 slices at 2% of the steps, patches from slice 1): the patch
    slices run at **13-14 steps/s with slices and 39-41 with blocks, 3x faster**. Slice 0, without patches, is
    the same (18/s, including compile).
  - **The trained parameters agree to rounding, not bitwise:**
    - slice 0 (no patches) is bitwise;
    - from slice 1 the relative differences are 1e-10 to 2e-6 and grow slowly through the Adam iterations
      (W 1.4e-8 at slice 1, 2.3e-6 at slice 4);
    - the losses agree to 1e-7.
    - The isolated patch term is bitwise, so the difference arises in the fused training graph (XLA fuses the
      larger graph differently). Same model, a different rounding path.
  - **Consequences:**
    - All CelebA-32 layouts use blocks: training with 8x8 + 16x16 is about 3x faster.
    - 64x64 is not solved. 16x16 every 4 px gains only 1.2x there, and 8x8 every 2 px is slower with blocks. On
      64x64 the patch term needs another approach (a convolution, or choosing the implementation per layout by
      measurement) before 64x64 runs are worth it.
    - `LATENT_PATCH_IMPL=slices` reproduces the old path exactly, for runs that must continue bit for bit.
  - The temporary test runs and scripts are deleted.

### Three patch scales on CelebA-32 (2026-11-16 16:30; the user: "yes go ahead with the three-scale run")

- **Code: `--patch3 SIDE:STRIDE:STATES`** (keys Wp3, bp3, Qp3; `latent_model.PATCH3_KEYS`, `PATCH_SETS` now has three
  entries).
  - In `train_latent.py` the second-set code is now one loop over the extra sets (`xsets`: `--patch2`, `--patch3`),
    instead of a copy per set: reference, learning rates, start, resume, switch-on, diagnostics `patch<n>_*` and
    `force_rms_patch<n>`, slice files, and history (`write_history(extra_keys=...)`).
  - Set 2 keeps its seeds (+100000), file names and keys. Set 3: k-means seeds +200000, `patch3_reference.json`.
  - `sample_latent.load_params` loops over sets 2 and 3; `make_gallery` lists `patch3`.
  - Backups `*_pre_patch3.py`.
- **Test 4262952:**
  1. A `--patch` + `--patch2` run, old vs new code (4 slices). Must be bit-identical, since the arithmetic is
     unchanged.
  2. CelebA-32 K64 + 8x8 + 4x4 + 16x16 over all 100 slices, then sampling and quench 1. All nine arrays must be zero
     before slice 55 and trained after it.
- **The run, only if the test passes (afterok):** pipeline **4262953**, K784 + `--patch 8:2:32 --patch2 4:2:32
  --patch3 16:4:32` from slice 55, seed 1, 2000 samples from the noise start.
  - The 8x8 set starts as in every K784 run, and the 4x4 set as in the K784 + 8x8 + 4x4 run (same seeds); the 16x16
    set uses the new seed range.
  - quench 1: 4262954; Inception FID **4262955** against K784 + 8x8, + 4x4, + 16x16 and K1536 + 8x8 + 16x16;
    gallery **4262956** (`galleries/2026-11-16_celeba32-grey_one-two-three-patch-scales_K784-8x8-4x4-16x16_noise-start_with-and-without-quench1/`).
  - Couplings per slice: global 1.61M + patches 0.69M + 0.23M + 0.41M = 2.94M (about K1536 + 8x8: 3.84M).

### CelebA 64x64: one run submitted after all (2026-11-16 17:00; the user: "should we submit the 64x64 as well?" ... "yes please")

- **Config:** K1536 + 16x16 patches every 4 px (16:4:32) from slice 55, init 0.06, seed 1. At 64x64 this covers the
  same share of the face as 8x8 every 2 px at 32x32. Single patch scale; layout to be revisited after the CelebA-32
  three-scale result.
- **Expected cost** (from smoke 4256481 and test 4262849): slices 0-54 at 27-46 steps/s (about 7 h); the patch slices
  at about 6 steps/s (blocks gain only 1.2x for 16:4 at 64x64), about 50-55 h; sampling about 10 h. **About 3 days.**
- **Jobs:**
  - data **4263036**: `CELEBA_SIZE=64 make_celeba_data.py`; re-downloads the parquet shards (about 12 GB);
  - training **4263037**: `train_latent.py` directly via `run_gpu_neuronic.sh` (not the pipeline), so training keeps
    JAX's default allocator; 160 GB, 5-day limit;
  - sampling **4263038**: 2000 samples, noise start, chunk 25, `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (history about
    10 GB in float64), 1-day limit;
  - quench 1: **4263039**; Inception FID **4263040** (dataset `celeba64`, its own statistics);
  - gallery **4263041** (`galleries/2026-11-16_celeba64-grey_K1536+patch16x16-every4px_noise-start_with-and-without-quench1/`).
- Disk at submission: 492 GB of 1 TB used. The run adds about 25 GB, the parquet cache 12 GB.
- **In parallel (open):** find why the 64x64 patch term is slow, so that a better-configured second run can overtake
  this one.
- **Test 4262952 PASSED (26 min).**
  - Regression: the `--patch` + `--patch2` run (4 slices) is bitwise identical with the old and the new code, every
    array and the same record keys.
  - Three scales (K64 + 8x8 + 4x4 + 16x16, 100 slices):
    - all nine arrays (Wp/bp/Qp, Wp2/bp2/Qp2, Wp3/bp3/Qp3) are zero before slice 55 and trained after it;
    - `patch3_*` and `force_rms_patch3` are recorded; set 3's k-means seeds are 220000-220024;
    - sampling and the quench give finite samples; the manifest records `8:2:32+4:2:32+16:4:32`.
  - The test runs and the old-code copy are deleted. Pipeline 4262953 is released by the dependency and waits in the
    queue.
- **16:15: 64x64 data rebuilt** (4263036, 5 min). The test file's sha256 (ea0bde3b...) equals the first build's
  (4256480), so the data are identical. Training **4263037** started at 16:15.

### Are the hidden couplings learned beyond a dense associative memory? Frozen vs trained (2026-11-16 17:45)

- **Question (the user):** compared with dense associative memories / modern Hopfield networks, whose stored patterns
  are set to data rather than learned: how far do our learned weights move from the prototypes, and does learning
  them matter?
  - Summing out the global Potts unit gives F = -T log sum_m exp(a_m / T), the log-sum-exp energy of a DAM with the
    rows W_m as patterns. The exact score of the empirical training distribution is such a softmax over all N
    training images, which only memorises.
- **What we had:**
  - The global cosine of W with the prototype reference falls from 1 (t = 5) to about 0.95 / 0.9 at t = 3, then to
    0.18-0.21 at t = 0. That holds for MNIST K784 + 7x7, CelebA K784 + 8x8 and CelebA K3072 without patches, so the
    weights move far from the prototypes at low t. This is a global number only; no per-state analysis yet.
  - README Result 10 (2026-11-08, digits 0/1, Potts 128, no Q): couplings frozen at the closed-form reference
    (`--hidden-fixed`) were as good as trained ones (clean yield 80.2% vs 80.1% at 28x28), with more variety. Only
    the learned spring Q added clean yield (+6 points).
- **New test, step 1 (no code change):** K784 + per-pixel units, no Q (`--hidden-fixed` refuses Q), no patches,
  seed 1, 2000 samples from the noise start, quench 1:

  | Data | Trained | Frozen at the reference | Quench (trained / frozen) |
  |---|---|---|---|
  | MNIST, ten digits | 4263126 | 4263127 | 4263131 / 4263132 |
  | CelebA-32 grey | 4263128 | 4263129 | 4263133 / 4263134 |

  - MNIST: Inception FID **4263135**; classifier FID **4263139**, paired frozen vs trained, as sampled and
    quenched; gallery 4263137.
  - CelebA: Inception FID **4263140**; gallery 4263141.
  - Galleries: `galleries/2026-11-16_<data>_K784-hidden-frozen-at-prototype-reference-vs-trained_no-Q_no-patches_noise-start_with-and-without-quench1/`.
- **Next steps:**
  - step 2, the full recipe frozen (global and patch couplings at their references, Q = 0) vs trained: needs a code
    change;
  - step 3, a per-state analysis of the drift from each state's prototype.
- **MNIST result (ten digits; training 32 min each; Inception 4263135, classifier 4263139, gallery 4263137): trained
  clearly beats frozen.** n = 2000, noise start; Inception floor 3.70:

  | MNIST K784 + per-pixel, no Q, no patches | Inception FID (FID_inf) | far | clean, class-standardised |
  |---|---|---|---|
  | frozen at the reference | 204.9 (203.3) | 30.7% | 67.4% |
  | frozen, quench 1 | 178.9 (177.6) | 42.4% | (classes too unbalanced) |
  | trained | 151.4 (148.9) | 14.0% | 84.6% |
  | trained, quench 1 | 135.6 (133.2) | 10.5% | |

  - Paired frozen vs trained (same starts and noise): clean -17.2 points [-19.2, -15.2], far +17.8 [+15.8, +19.8],
    both p < 2e-4.
  - With the quench the frozen model collapses onto a few classes (821 of 2000 samples classed as 4s, 9 as 6s),
    so its t = 0 energy has spurious, wide basins. The trained model keeps balanced classes (152-257 per class).
  - **Unlike digits 0/1 (README Result 10, frozen = trained), on ten digits learning the hidden couplings is worth
    17 points of clean yield and about 50 Inception FID.** A DAM with K784 k-means prototype memories plus a trained
    visible model is far from enough here.
  - Both models are rough (no Q, no patches; Inception 136-179 against 86 for K784 with Q). In the gallery the
    frozen strokes are more broken and speckled, the trained ones more solid. The quench helps the trained model
    less than with Q (151 -> 136, against 113 -> 86).
- **CelebA-32 result (training 38 min each; Inception 4263140, gallery 4263141): trained clearly beats frozen here
  too.** n = 2000, noise start; floor 6.43:

  | CelebA K784 + per-pixel, no Q, no patches | Inception FID (FID_inf) | KID x1e3 | faint pixels |
  |---|---|---|---|
  | frozen at the reference | 294.7 (289.9) | 433 | 3.5% |
  | frozen, quench 1 | 178.3 (172.7) | 264 | 2.9% |
  | trained | 249.2 (245.3) | 329 | 7.2% |
  | trained, quench 1 | **120.2 (116.5)** | **146** | 7.6% |
  | (for scale: K784 with Q, trained, quench 1) | 104.4 | 120 | 7.5% |

  - Trained beats frozen by 45 as sampled and 58 after the quench.
  - Gallery: the frozen faces are low-contrast, mid-grey and uniform, with few dark or bright backgrounds (faint
    pixels 3% vs 7-8% trained and 12% real). The trained faces vary in lighting and background like the real ones,
    with more defined hair and face outlines.
  - Q adds 16 more on top of training (120 -> 104 after the quench).
- **Step 1 conclusion:** on both hard datasets, learning the hidden couplings by force matching beats holding them
  at the k-means prototype reference (the DAM-like model) by a wide margin. The 0/1 result (frozen = trained) does
  not carry over. Learning matters for MNIST shapes (17 points of clean yield) and for CelebA's contrast and
  variety (about 58 Inception FID).

### Step 2 set-up: frozen patch units (2026-11-16 17:50; the user: "the best K784 multipatch model compared to its analogue hidden structure with frozen")

- **Code (`train_latent.py`; backup `train_latent_pre_fixedpatch.py`):** `--hidden-fixed` now works with Potts patch
  units (`--patch`, `--patch2`, `--patch3`).
  - From the switch-on slice, every patch set is held at its full closed-form k-means reference for that slice's t
    (`patch_reference`, scale 1), and its learning rates are zero, as for the global unit.
  - Still without Q, which has no reference. Ising patch units are refused.
  - Trained runs are unaffected: the change only touches the `--hidden-fixed` branches.
- **Test 4263528 (2.5 min), PASSED:** CelebA-32 K64 frozen + 8:2:32 + 4:2:32, patches from slice 1, 3 slices.
  - Global |W| = reference with cos 1.000 in every slice.
  - Patches: |Wp| 3.362 = reference, then 3.535 = reference; |Wp2| 2.154, then 2.265, both = reference, cos 1.000.
    So the frozen sets follow the reference in t.
  - The test run is deleted.
- **Planned (after the three-scale result picks the best K784 multi-patch structure):** frozen (no Q) vs trained
  without Q vs the trained recipe with Q, same patches; quench 1, Inception FID, gallery.
- **19:40: step 2 submitted for the best CelebA-32 structure, K1536 + 8x8 every 2 px + 16x16 every 4 px** (22.4
  with Q, trained). Submitted now because the GPU queue is congested: Slurm puts the three-scale quench (4262954)
  at about 00:20.
  - If the three-scale K784 run comes in below 22.4, its frozen variants follow.
  - Pipelines: trained without Q **4264051**; frozen **4264052** (`--hidden-fixed`: global and both patch sets at
    their references, no Q). Both 80 GB, 16 h limit, 2000 samples from the noise start.
  - quench 1: 4264054 / 4264055; Inception FID **4264056** against the trained recipe with Q; gallery **4264057**
    (`galleries/2026-11-16_celeba32-grey_K1536+patch8x8+patch16x16_hidden-frozen-vs-trained-without-Q-vs-trained-with-Q_noise-start_with-and-without-quench1/`).
- **19:55: the three-scale structure's frozen comparison is also submitted** (the user is away overnight, so both
  comparisons are ready in the morning whichever structure wins):
  - pipelines: trained without Q **4264067**, frozen **4264068** (K784 + 8x8 + 4x4 + 16x16);
  - quench 4264070 / 4264071; Inception FID **4264072** (also after the recipe's own quench 4262954); gallery
    **4264073**.
- **Queue and fair share (sshare / sprio, 19:55):**
  - Our share of the seas account is 0.31% (NormShares 0.0031), but we used 0.82% of it recently (EffectvUsage
    0.0082), so Slurm's fair-share factor is 0.10 (LevelFS 0.38).
  - The fair-share term dominates our job priority (1031 of about 1100); age adds a few points per day.
  - That is why short jobs wait hours when the cluster is busy.
- **20:00: the three-scale frozen comparison is cancelled before it started** (the user: "remove the frozen
  three-scale structure. we will run it in the morning if we need it"): 4264067 / 68 / 70 / 71 / 72 / 73. The
  K1536 + 8x8 + 16x16 comparison (4264051 / 52 / 54 / 55 / 56 / 57) stays.

### 2026-11-17 morning: overnight results

- **Disk:** 540 GB of 1 TB used (53%).
- **Three scales on CelebA-32** (pipeline 4262953: training 1.8 h for K784 with three patch sets, about 230 steps/s
  in the patch slices, so the block unfold pays off; quench 4262954; Inception **4262955**; gallery 4262956).
  Inception FID against train, n = 2000, noise start; floor 6.43:

  | Model | as sampled | quench 1 (FID_inf) | KID x1e3 (q1) |
  |---|---|---|---|
  | K784 + 8x8 | 98.4 | 24.7 (20.9) | 17.5 |
  | K784 + 8x8 + 4x4 | 78.3 | 23.3 (19.7) | 15.9 |
  | K784 + 8x8 + 16x16 | 93.6 | 22.9 (19.2) | 15.4 |
  | K1536 + 8x8 + 16x16 | 92.6 | 22.4 (18.7) | 14.5 |
  | **K784 + 8x8 + 4x4 + 16x16** | **75.4** | **22.3 (18.7)** | 15.1 |

  - The three scales add up as sampled: the 4x4 set removes speckle (98 -> 78), and the 16x16 set adds 3 more.
  - After the quench, three scales at K784 tie the best model, K1536 + 8x8 + 16x16 (22.3 vs 22.4; FID_inf 18.7
    both), with half the global states.
  - Gallery: the quenched rows of all multi-scale models look alike. Three scales give slightly cleaner details in
    places (column 1's eye smudge is lighter), but blur remains in every variant.
  - Since the two structures are equal, the K1536 + 8x8 + 16x16 frozen comparison below stands for both; the
    three-scale frozen comparison (cancelled last night) is not needed.
- **Step 2: frozen vs trained, full structure K1536 + 8x8 + 16x16** (frozen 4264052 and trained without Q 4264051,
  1.6 h each; quench 4264054 / 4264055; Inception **4264056**; gallery 4264057):

  | Hidden couplings (K1536 + 8x8 + 16x16) | as sampled | quench 1 (FID_inf) | KID x1e3 (q1) | faint pixels |
  |---|---|---|---|---|
  | frozen at the k-means references, no Q | 225.6 | 213.6 (209.4) | 264 | 17.9% |
  | trained, no Q | 128.8 | 34.4 (30.9) | 31 | 6.5% |
  | trained, with Q (the recipe) | 92.6 | **22.4 (18.8)** | 14.5 | 6.9% |

  - **Learning is worth 179 Inception FID after the quench (213.6 -> 34.4), and Q 12 more (-> 22.4).**
  - The frozen multi-patch model is worse than the frozen model without patches (213.6 vs 178.3 after the quench).
    The patch prototypes at full reference strength make the faces worse: in the gallery they are mosaics with
    horizontal bars across the eyes, blocky stripes and saturated black or white regions (faint pixels 18% vs 7%
    trained and 12% real). The quench does not repair them.
  - The trained models give normal faces; without Q they are a little more speckled and less defined than with Q.
  - **Conclusion for the paper:** for the full model, the frozen associative-memory analogue (global and patch
    prototypes as memories, only the physical visible model trained) is far worse. Learning the couplings is what
    makes the model work, and that holds even more with patches than without.
- **CelebA 64x64:** training **4263037 finished in 6.9 h** (not 2.5 days; the patch slices ran at 66-69 steps/s);
  sampling 4263038 took 8.2 h (chunk 25, platform allocator).
  - **The quench 4263039 FAILED:** RESOURCE_EXHAUSTED, 11.5 GiB.
  - **Cause:** in quench-only mode `sample_latent` built the sampler's SDE drift (a device copy of the 12 GB
    parameter history) although only the quench drift is used, and then a second copy for the quench.
  - **Fix (`sample_latent.main`):** quench-only no longer builds the SDE drift. The quench itself is unchanged.
  - Resubmitted: quench **4266706** (platform allocator, chunk 25, 160 GB); Inception **4266707**; gallery
    **4266708**. The old 4263040 / 41 could never start after the failure and were cancelled.
- **Correction (2026-11-17 09:30; the user, looking at the frozen faces: "are you sure there is no bug in the frozen
  code?"). Not a code bug, but an unfair frozen baseline.**
  - The code holds each patch set exactly at its reference (test 4263528). But each patch reference is the complete
    mixture model of its own patch, as if that patch unit alone described it.
  - With 8x8 every 2 px and 16x16 every 4 px, each interior pixel lies in 16 patches of each set. The frozen model
    therefore counts each pixel's evidence about 16 times per set: a product of over-confident experts.
  - That explains what the faces show: saturated black and white, pixels snapped to patch prototypes (dark
    horizontal bars at the eyes and mouth), and a regular grid with the patch stride. The frozen model without
    patches has none of this.
  - The trained runs start the patches at 0.06 x the reference (chosen 11-13 because 0.25 already cost +27 loss
    units); 0.06 is close to 1/coverage = 1/16.
  - **Fix: `--patch-fixed-scale S`** (`train_latent.py`, default 1; backup `train_latent_pre_patchfixedscale.py`):
    with `--hidden-fixed`, the patch sets are held at S x their reference. It is recorded in config.json only when
    not 1, and resume checks it.
  - **Rerun:** frozen K1536 + 8x8 + 16x16 at S = 0.0625 = 1/coverage (each pixel counted once; also the trained
    runs' start). Pipeline **4266722**, quench 4266723, Inception **4266724** against the scale-1 frozen run and
    the trained models (without and with Q), gallery **4266725**.
  - The scale-1 result above (213.6) stays in the notebook as the over-counted variant, not as the frozen
    baseline.
- **CelebA 64x64, first result** (quench 4266706, 6.6 min with the fix; Inception **4266707**; gallery 4266708;
  K1536 + 16x16 every 4 px from t = 2.2, seed 1, noise start, n = 2000; floor = real test faces 5.91):

  | CelebA 64x64 | Inception FID (FID_inf) | KID x1e3 | faint pixels |
  |---|---|---|---|
  | as sampled | 130.3 (127.0) | 174 | 4.0% |
  | quench 1 | **58.5 (55.9)** | 54 | 3.3% (real 12.6%) |

  - Gallery (quenched): recognisable faces with more detail than at 32x32 (eyes, mouths, teeth, hair strands), and
    varied identities and poses.
    - Defects: some faces are distorted or asymmetric (one eye, a smeared half).
    - Contrast is too low: grey backgrounds, few dark pixels (3% vs 13% real), so dark hair and backgrounds come out
      as mid-grey.
  - First run, single patch scale; for comparison, CelebA-32 with K1536 + 8x8 gives 23.9. Next options: a second
    patch scale (8x8 every 2 px, still slow with blocks at 64x64), or K3072.

### CelebA 64x64: next steps (2026-11-17 10:15; the user: "lets 3, then 2 and maybe already submit 1 since this will take long")

- **(1) Two patch scales at 64x64, submitted:** K1536 + 16x16 every 4 px + 8x8 every 2 px (`--patch 16:4:32
  --patch2 8:2:32`), seed 1.
  - `LATENT_PATCH_IMPL=slices` for both sets: on 64x64, 8x8 every 2 px is 1.7x faster with slices than with blocks,
    and 16x16 every 4 px only 1.2x slower (test 4262849), so slices are faster for the pair.
  - Training **4266791** (3-day limit); sampling 4266792 (platform allocator, chunk 25); quench 4266793;
    Inception **4266794** against the single-scale quenched run; gallery 4266795.
  - History in float64: about 14 GB (global 12.6M + patches 4.8M couplings per slice).
- **(3) Low contrast, first look:** new `contrast_check.py` (CPU). For real test images and each sample set it reports:
  - the pixel-value histogram and percentiles;
  - per image: contrast (pixel std), mean brightness, and the share of dark (< 0.1) and bright (> 0.9) pixels;
  - per pixel position: maps of the mean and of the spread over images.

  Jobs: 64x64 (K1536 + 16x16, as sampled and quenched) **4266796**; 32x32 (K1536 + 8x8 + 16x16 as sampled and
  quenched, K784 + 8x8 equilibrated start + quench) **4266797**; output in `galleries/2026-11-17_celeba<32|64>-grey_contrast-check_*`.
- **(2) K3072 at 64x64:** after (3).
- **(3) Contrast results (4266796 / 4266797, 1 min each):**

  | | real | 64x64 sampled | 64x64 quench 1 | 32x32 best (K1536 + 8x8 + 16x16), quench 1 |
  |---|---|---|---|---|
  | per-image contrast (pixel std), mean | 0.234 | 0.182 | 0.180 | 0.198 (real 0.229) |
  | dark pixels (< 0.1) | 13.5% | 4.1% | 3.3% | 6.9% (real 12.7%) |
  | bright pixels (> 0.9) | 3.5% | 0.5% | 0.4% | 1.3% (real 3.3%) |
  | pixel percentiles 1 / 99 | 0.004 / 0.980 | 0.045 / 0.871 | 0.064 / 0.863 | 0.038 / 0.913 |
  | spread of the images' mean brightness | 0.114 | 0.095 | 0.097 | 0.115 (real 0.114) |

  - The pixel range is compressed: the samples lack the extremes near 0 and 1, which real faces have plenty of
    (black hair and backgrounds, saturated highlights).
  - Per-image contrast is 23% too low at 64x64 and 13% at 32x32.
  - **Where (maps):** the mean image is reproduced almost exactly. The variety over images is right in the face
    interior but much too low in the background and the hair ring. Real backgrounds range from black to white
    (highest spread in the corners); ours stay near mid-grey.
  - The quench does not cause it (sampled 4.1% dark, quenched 3.3%), and neither does the start (32x32 equilibrated
    start: the same).
  - **Reading:** the model regresses large, smooth regions (background, hair) towards the mean. Their brightness is
    a long-range, image-wide choice. The 16x16 patches are local, and the global unit apparently does not carry
    enough of it, more so at 64x64.
  - So K (step 2) and a coarser or global brightness mechanism are the candidates. The second patch scale (8x8)
    adds detail, not range.
- **(2) K3072 at 64x64: training submitted, 4266799** (K3072 + 16x16 every 4 px from slice 55, seed 1; only K differs
  from the first 64x64 run; 200 GB, 3-day limit).
  - **Sampling needs a change first.** The history in float64 is about 22 GB, and the sampler holds 2-3 device
    copies (`helper_fns.interpolate_parameters` sorts a copy, and jit embeds it as a constant). That does not fit
    the 46 GB L40 (the only GPU type here). K1536 at 64x64 was 12.3 GB per copy.
  - Plan: a sampler option that holds the history in float32 (halves it), tested against float64 on a small run
    before use. Rounding the parameters to float32 is about 1e-7 relative, far below the SDE tolerance (1e-4).
- **`sample_latent --history-dtype float32`** (2026-11-17; backup `sample_latent_pre_hist32.py`):
  - holds the stacked parameter history in float32 on the device, half the memory;
  - default float64, every path unchanged;
  - float32 samples get the suffix `_hist32`, and the manifest records `history_dtype`; `--fast` refuses it.
- **Test** (CelebA-32 K1536 + 8x8 + 16x16, 50 samples, same starts and noise; jobs 4266800 / 4266809):
  - as sampled: max |float32 - float64| 4.2e-4, median 2.8e-6 (normalised units; sample std 0.885);
  - quench 1: max 3.7e-4, median 2.5e-6.
  - Negligible. The test files were removed.
  - Side note: 4266800's float64 sample + quench in one process was OOM-killed at 48 GB of host memory during the
    quench; quench-only with 64 GB was fine.
- **K3072 64x64 evaluation chain (after training 4266799):**
  - sampling **4266810** (`--history-dtype float32`, platform allocator, chunk 25, 200 GB, 2-day limit);
  - quench 4266811;
  - Inception FID **4266812** against K1536 + 16x16 (quenched);
  - gallery 4266813 (`galleries/2026-11-17_celeba64-grey_K1536-vs-K3072_patch16x16-every4px_noise-start_quench1/`);
  - contrast check 4266814: does K3072 restore the background and hair range?
- **Corrected frozen baseline: result** (pipeline 4266722, 1.5 h; quench 4266723; Inception **4266724**; gallery
  4266725, `galleries/2026-11-17_celeba32-grey_K1536+patch8x8+patch16x16_frozen-patches-full-vs-over-coverage_vs_trained_noise-start_quench1/`).
  CelebA-32, K1536 + 8x8 + 16x16, n = 2000, noise start; floor 6.43:

  | Hidden couplings | as sampled | quench 1 (FID_inf) | KID x1e3 (q1) | faint pixels |
  |---|---|---|---|---|
  | frozen, patches at the full reference (over-counted) | 225.6 | 213.6 (210.6) | 265 | 17.9% |
  | **frozen, patches at reference / coverage (0.0625)** | 276.7 | **147.0 (143.8)** | 212 | 2.x% |
  | (frozen without patches, 11-16) | 294.7 | 178.3 | 264 | 2.9% |
  | trained, no Q | 128.8 | 34.4 (30.9) | 31 | 6.5% |
  | trained, with Q (the recipe) | 92.6 | **22.4 (18.7)** | 14.6 | 6.9% |

  - Counting each pixel once removes the stripes and the saturation. The frozen faces are now face-like but
    mottled, blotchy and low in contrast, as the frozen model without patches was.
  - Frozen patches help the frozen model (178 -> 147 after the quench), but **learning is worth about 113 Inception
    FID (147 -> 34), and Q another 12 (-> 22)**.
  - **This replaces the 179 quoted this morning** (that baseline over-counted the patches). Conclusion unchanged:
    for the full model, the frozen associative-memory analogue is far worse, and learning the couplings by force
    matching is essential.

### Image quality on the sampling side: corrector steps and force scale (2026-11-17 11:00; the user: "lets do 1 and 2 first, then 3")

- **Code (`sample_latent.py`; backup `sample_latent_pre_corrector.py`); both default off, every existing path
  unchanged:**
  - `--force-scale LAMBDA`: the sampling drift is x - 2 LAMBDA grad F, i.e. sampling at temperature T / LAMBDA
    (LAMBDA > 1 colder and sharper, at a likely cost in variety). The quench is not scaled. Suffix `_fs<LAMBDA>`.
  - `--corrector M:SNR` (`sample_corrector`): predictor-corrector sampling.
    - The reverse SDE is solved from knot to knot (100 segments; Brownian key fold_in(key, segment)).
    - After each segment, M unadjusted Langevin steps at that slice: x += eps s + sqrt(2 eps) z, with
      s = -LAMBDA grad F / T and grad F = x - the PF-ODE drift; eps = 2 (SNR |z| / |s|)^2 per sample (the SNR rule
      of Song et al. 2021).
    - M = 0 is the segmented sampler without corrector, the right baseline (different Brownian paths from the
      standard sampler). Suffix `_corr<M>x<SNR>`.
  - Metropolis-adjusted steps (MALA) would need the free energy F itself, which the sampler does not build yet.
    That is the next step if the corrector helps.
- **Test 4266899:** CelebA-32 K1536 + 8x8 + 16x16, 50 samples each, with `--corrector 0:0.16`, `--corrector
  2:0.16` and `--force-scale 1.2`. It reports run time, finiteness, contrast, and dark and bright shares.

### Solver convergence check of the reverse sampler (2026-11-17 11:30; the user: "are you using a fixed step size for the reverse sampler? ... make sure that reducing does not change the images")

- **The reverse sampler is adaptive, not fixed-step:** diffrax SRA1 with a PID controller, rtol 1e-4 and atol 1e-6
  (the paper code's values), forced to step onto the 98 interior knots; Brownian increments from a
  VirtualBrownianTree (tolerance 1e-12); initial dt 1e-8. The quench uses the ODE tolerances (rtol 1e-7, atol 1e-9).
- **This was never checked for convergence.** All earlier checks compared sampler variants at the same tolerance.
- **New `--sde-tol-scale F`** (`sample_latent.py`; backup `sample_latent_pre_tolscale.py`): multiplies rtol and atol
  by F (default 1, unchanged). Suffix `_tol<F>`; the manifest records the actual rtol and atol.
  - The Brownian path of a trajectory does not depend on the steps, so samples of the same trajectories can be
    compared one by one: pathwise (strong) convergence.
  - Note: for CelebA, `trajectory_set` draws the start states and keys for the whole set at once, so the first 200 of
    an n = 2000 set differ from an n = 200 set; the reference is therefore resampled at the same n.
- **New `tol_check.py`:** per trajectory, the max and RMS of |x_tight - x_ref| against the typical distance between
  two different samples (median nearest-neighbour RMS), plus contrast and dark share per set.
- **Jobs:**
  - **4266909:** CelebA-32 K1536 + 8x8 + 16x16, n = 200 at tolerance x 1, x 0.1, x 0.01, each as sampled and after
    quench 1;
  - **4266910:** CelebA-64 K1536 + 16x16, n = 50 at x 1 and x 0.1 (platform allocator).
- **Test 4266899 (+ contrast 4266936), CelebA-32 K1536 + 8x8 + 16x16, n = 50 each:** all three variants run.
  - Time per 50: segmented without corrector 311 s, + 2 corrector steps 339 s (+20%), force scale 1.2 315 s.
  - Contrast (n = 50, preliminary):

    | | contrast (pixel std) | dark < 0.1 | bright > 0.9 |
    |---|---|---|---|
    | real | 0.229 | 12.7% | 3.3% |
    | standard sampler (n = 2000) | 0.200 | 7.4% | 1.5% |
    | segmented, no corrector | 0.197 | 6.2% | 1.6% |
    | + 2 corrector steps (SNR 0.16) | 0.209 | 6.5% | 1.4% |
    | force scale 1.2 (colder) | 0.188 | 5.4% | 0.6% |

  - The corrector raises contrast a little (+6%).
  - **Colder sampling lowers it.** The learned modes themselves sit at mid-grey, so concentrating on them compresses
    the range further. The contrast deficit is in where the model puts its mass, not in thermal smearing, which
    favours the model-side fix (3, global factors).
  - Full scan (n = 2000, Inception FID): corrector M = 2 and 4, and force scale 0.9 (warmer). Submitted after the
    tolerance check settles the solver tolerance.
- **Tolerance convergence: the sampler is converged at the default tolerance.**

  | Check | max abs diff | median RMS diff / NN distance | contrast, dark share |
  |---|---|---|---|
  | CelebA-64 K1536 + 16x16, x0.1 vs x1, n = 50 (4266910: 1338 s vs 4030 s) | 5.2e-4 | 4.9e-6 | identical (0.1794, 3.31%) |
  | same, after quench 1 | 4.4e-4 | 6.1e-6 | identical (0.1767, 2.45%) |
  | CelebA-32 K1536 + 8x8 + 16x16, x0.1 vs x1, n = 200 (4266909: 791 s vs 3156 s; CPU 4267198) | 1.2e-3 | 7.2e-6 | identical (0.2008, 6.01%) |

  - A 10x tighter tolerance moves each sample by about a millionth of the distance between two different samples,
    and costs 3-4x the time. The images do not change, so their defects are not solver artefacts. The x0.01 run on
    CelebA-32 is still running and adds a third point.
- **Corrector / temperature scan submitted at the default tolerance** (CelebA-32 K1536 + 8x8 + 16x16, n = 2000,
  noise start, chunk 250):
  - variants: `--corrector 0:0.16` (segmented baseline) 4267202, `2:0.16` 4267204, `4:0.16` 4267206,
    `--force-scale 0.9` (warmer) 4267208;
  - their quenches 4267203 / 05 / 07 / 09;
  - Inception FID **4267210** with the standard sampler as reference; contrast 4267211; gallery 4267212
    (`galleries/2026-11-17_celeba32-grey_K1536+patch8x8+patch16x16_corrector-and-force-scale-scan_*`).

### On-device training: sampled hidden statistics in mixed precision (2026-11-17 14:30; the user: "lets do B")

- **Why:** on a device the hidden statistics (Potts state probabilities) can only be estimated from S sampled hidden
  states per training image. Earlier (README, 11-08/10) this matched exact training on two or three digits (S = 16-64
  with the visit-count rule and the jackknife), but fell 10 points short on four digits with Potts 512. Never tried
  on faces.
- **Code (backups `*_pre_mixedsamples.py`):**
  - `latent_model.make_loss_per_sample`: no longer refuses `compute_dtype` (mixed precision) with samples > 0. The
    states are drawn from the float32 fields: other draws than float64 for the same key, but the same distribution.
    The refusal was for reproducibility only.
  - `latent_model.loss_draw_occupation(..., dtype)`: with `--occ-same-draws` the visit counts recompute the fields
    in the loss's precision, so they count the loss's own draws.
  - `train_latent.py`: `--precision mixed` now allows `--hidden-samples` and `--hidden-norm occupancy`
    (`build_runner(occ_dtype=)`). It still refuses `--hidden-rule cd1` and `--hidden-floor` (untested in float32).
  - Previously only the float64 path (1-2 days per run on the L40s) could train with sampled statistics.
- **Test 4267283:** CelebA-32 K64 + Q + per-pixel units, S = 16, jackknife, visit-count rule with same-draw counts,
  4 slices at 1% of the steps, mixed vs f64. Both must run, with similar losses.
- **Runs (only if the test passes):** CelebA-32 K784 + Q + per-pixel units, no patches (patch units need exact
  statistics), mixed precision, seed 1, 2000 samples from the noise start:

  | Pipeline | Hidden statistics | Optimiser for the hidden couplings |
  |---|---|---|
  | 4267284 | exact | visit-count rule (`--hidden-norm occupancy`) |
  | **4267285** | **S = 64 sampled**, jackknife | visit-count rule, same-draw counts |
  | **4267286** | **S = 256 sampled**, jackknife | visit-count rule, same-draw counts |
  | (existing) | exact | Adam per element (Inception FID after the quench 104.4) |

  - quench 4267287 / 88 / 89; Inception FID **4267290** (all four, quenched; sampled runs also as sampled); gallery
    **4267291** (`galleries/2026-11-17_celeba32-grey_K784_on-device-training_exact-vs-S64-vs-S256-sampled-hidden-statistics_noise-start_quench1/`).
- **Test 4267283 PASSED (1.7 min):**
  - mixed and f64 both run at 22-24 steps/s (1% steps, K64);
  - validation losses agree to the printed digits (6.70209e6, 6.54446e6, 6.3629e6, 6.18925e6);
  - the hidden couplings train: Q_norm 0 -> 0.0035 -> 0.0072 -> 0.0113 and W_norm 1.136290 -> 1.136446;
  - mixed vs f64: W_norm agrees to 1e-6 relative and Q_norm to 2e-5 (different draws, same distribution).
  - The test runs are deleted; pipelines 4267284-86 are released by the dependency.
- **14:25: two-scale 64x64 run cancelled** (training 4266791 and its chain 4266792-95; the user agreed).
  - With `--patch 16:4:32 --patch2 8:2:32` (slices implementation) the first patch slice (55) took 4541 s:
    **5 steps/s**, against 69 steps/s for 16x16 alone (blocks) and 144 steps/s without patches.
  - The remaining 45 patch slices would need about 3 days, over the 3-day job limit.
  - 8x8 every 2 px at 64x64 is not viable until the patch term is faster for this layout (open item: a
    convolution-based unfold, or choosing the implementation per layout).
  - Slices 0-55 remain on disk in `celeba64g_K1536_..._patch16s4S32+patch8s2S32on55_seed1/` and can be resumed.

### 2026-11-17 evening: sampler results

- **Tolerance check, final** (4266909: x1 791 s, x0.1 3156 s, x0.01 14504 s; CelebA-32 K1536 + 8x8 + 16x16, n = 200):
  x0.1 and x0.01 differ from x1 by the same tiny amount (median RMS / NN distance 7.2e-6 and 7.1e-6; after the
  quench 9.7e-6 and 9.9e-6), and contrast and dark share are identical. **The default tolerance is converged.**
- **Corrector / temperature scan** (CelebA-32 K1536 + 8x8 + 16x16, n = 2000, noise start; Inception **4267210**,
  contrast 4267211, gallery 4267212); floor 6.43:

  | Sampler | as sampled | quench 1 (FID_inf) | KID x1e3 (q1) | contrast (q1) | dark / bright (q1) |
  |---|---|---|---|---|---|
  | standard | 92.6 | 22.37 (18.69) | 14.5 | 0.198 | 6.9% / 1.3% |
  | segmented, no corrector | 93.1 | 22.64 (19.44) | 14.9 | 0.197 | 6.4% / 1.4% |
  | **+ 2 corrector steps** (SNR 0.16) | 76.5 | **21.50 (17.98)** | **12.5** | 0.200 | 7.5% / 1.6% |
  | + 4 corrector steps | 70.5 | 21.71 (18.25) | 12.5 | 0.203 | 7.9% / 1.8% |
  | force scale 0.9 (warmer) | 140.7 | 35.24 (31.07) | 35.0 | 0.217 | 8.5% / 2.2% |
  | (real) | | | | 0.229 | 12.7% / 3.3% |

  - **The corrector helps.**
    - As sampled it cuts the FID a lot (93 -> 77 -> 71 with 2 and 4 steps): it removes much of the speckle on the
      way.
    - After the quench it gains about 1.1 over its segmented baseline (22.64 -> 21.50), 0.9 over the standard
      sampler, and KID 14.9 -> 12.5. 4 steps add nothing over 2.
    - **New best on CelebA-32: 21.50 (FID_inf 18.0).**
    - Gallery: the corrector faces are more coherent and better defined (eyes, mouth, hair outline), with fewer
      smudges. Some look more frontal and average; a recall check would tell whether variety suffers.
    - Cost: +4% sampling time for 2 steps here (2.5 h vs 2.4 h per 2000; the segmentation itself is the larger cost
      over the standard sampler, 1.8 h).
  - **Temperature trades contrast against fidelity.** Warmer (0.9) restores contrast (0.217, dark 8.5%), colder
    (1.2, n = 50) lowers it, but warmer costs 13 Inception FID after the quench (noisier textures that the quench
    only partly removes). So temperature is not the fix for the contrast deficit.
  - Next on the sampler side: a Metropolis-adjusted corrector (MALA), only if worth the code (the gain is about 1
    FID); the corrector is ready to use for the paper models (`--corrector 2:0.16`).
- **On-device training, interim** (Inception 4267845; CelebA-32 K784 + Q + per-pixel units, no patches, n = 2000,
  noise start; training 2.6 h exact + visit-count rule, 2.8 h S = 64; floor 6.43):

  | Hidden statistics, optimiser | as sampled | quench 1 (FID_inf) | KID x1e3 (q1) |
  |---|---|---|---|
  | exact, Adam (11-15) | 238.0 | 104.4 (100.6) | 119 |
  | exact, visit-count rule | 238.4 | 106.4 (102.0) | 124 |
  | **S = 64 sampled**, jackknife, visit-count rule (same-draw counts) | 246.9 | **109.1 (105.8)** | 130 |

  - **Training from 64 sampled hidden states per image comes within about 3 Inception FID of exact statistics with
    the same optimiser** (109.1 vs 106.4; bootstrap sd 1.5-2.1), and within about 5 of our standard training.
  - The visit-count rule itself costs 2.0 against Adam with exact statistics.
  - So on faces with 784 states, device-measurable statistics nearly suffice at S = 64. S = 256 (4267286, slice 85
    at 17:45) follows in the full comparison 4267290.
  - Caveat: the no-patch model is the weak one (104 against 22 with patches). The decisive test is the patch model
    with sampled statistics, which needs sampled statistics for the patch units (code).

### On-device training with patch units: measured patch statistics (2026-11-17 18:30; the user: "start coding this up to have [patches] with the learning rule")

- **Step 1 (this entry): sampled statistics for the patch units.**
  - `latent_model.patch_terms(..., stat=)`: a measured estimate replaces the posteriors, as `potts_terms(stat=)`
    does for the global unit.
  - `make_loss_per_sample`: with samples = S > 0 and Potts patch units, each patch unit's posterior is estimated
    from S sampled states per sample, with key fold_in(key, 1000 + set index) and the same jackknife parts.
    The global and per-pixel draws keep their keys, so runs without patches are bit-identical.
  - `train_latent.py`: `--patch` accepts `--hidden-samples` (Potts patch units) and `--hidden-norm occupancy`.
  - Under the visit-count rule (`OccupancyTransform`) the patch couplings get per-element Adam: they are in neither
    its Potts keys (W, b, Q) nor its group keys.
  - Backups `*_pre_sampledpatch.py`.
- **Step 2 (next):** the visit-count rule for the patch states (a visit counter per patch unit and state).
- **Test 4267958:**
  1. New `check_sampledpatch.py` on the trained CelebA-32 K784 + 8x8 run (slices 60 and 85; one batch of 128):
     loss and gradients with S = 16 / 64 / 256 sampled states (jackknife, 8 keys) against the exact ones, per
     parameter group (cosine, single-key error, error of the key mean).
  2. Regression: exact statistics with patches, old vs new code, 3 slices, must be bitwise.
  3. Training: K64 + 8x8 + 16x16 patches, S = 16, jackknife, visit-count rule with same-draw counts, 4 slices.
- **Test 4267958 PASSED (8 min).**
  1. **Estimator** (`check_sampledpatch.py`, CelebA-32 K784 + 8x8, batch 128, 8 keys, jackknife):

     | | slice 60 (t = 1.97): Wp cos / err(1 key) / err(mean of 8) | slice 85 (t = 0.71): Wp | W (global unit), slice 85 |
     |---|---|---|---|
     | S = 16 | 0.906 / 0.47 / 0.17 | 0.900 / 0.48 / 0.17 | 0.876 / 0.57 / 0.17 |
     | S = 64 | 0.976 / 0.23 / 0.08 | 0.975 / 0.23 / 0.08 | 0.964 / 0.28 / 0.10 |
     | S = 256 | 0.994 / 0.11 / 0.04 | 0.994 / 0.11 / 0.04 | 0.992 / 0.13 / 0.05 |

     - The loss is unbiased within its noise at every S. For example at slice 85: S = 16 gives -23583.9 +- 3.3,
       against exact -23583.9.
     - The single-key gradient error halves for every 4x in S (1/sqrt(S)), and the mean of 8 keys has about
       1/sqrt(8) of it. So it is pure noise, with no detectable bias.
     - The patch gradients (Wp, bp, Qp) behave like the global unit's: bp is the noisiest, as b is.
  2. **Regression:** exact statistics with patches, old vs new code, slices 0-2: all arrays bitwise.
  3. **Training** (K64 + 8x8 + 16x16, S = 16, jackknife, visit-count rule for the global unit): it runs (11
    steps/s at this size), and the patch couplings train (Qp_norm 0.010 -> 0.021 -> 0.032).
  - Temporary runs and the old-code copy are deleted. `check_sampledpatch.py` stays as a tool.
- **18:45: step-1 runs submitted** (the user: "A now and B in parallel"). CelebA-32 K784 + 8x8 every 2 px + 16x16 every
  4 px from slice 55, Q, per-pixel units, mixed precision, seed 1, 2000 samples from the noise start:

  | Pipeline | Hidden statistics | Optimiser |
  |---|---|---|
  | (existing) | exact | Adam (quench 1: 22.85) |
  | **4268000** | exact | visit-count rule for the global unit, Adam for the patches |
  | **4268001** | **S = 64 sampled** for the global unit and every patch unit, jackknife | visit-count rule (same-draw counts) for the global unit, Adam for the patches |

  - quench 4268002 / 4268003; Inception FID **4268004**; gallery **4268005**
    (`galleries/2026-11-17_celeba32-grey_K784+patch8x8+patch16x16_on-device-training_exact-vs-S64-sampled-statistics_noise-start_quench1/`).
- **Step 2 (in parallel):** the visit-count rule for the patch states.
- **Step 2, code (2026-11-17 19:15): the visit-count rule for the patch states, `train_latent.py --patch-occupancy`**
  (with `--hidden-norm occupancy` and Potts patch units).
  - `latent_model.patch_occupation(params, X, T, key, S, dtype)`: each patch state's occupation in the batch, per
    patch set. Exact (batch-mean posterior) or the visit fraction of S draws from the same fields and keys as the
    loss's patch draws (fold_in(key, 1000 + set)), so with the loss key they are the loss's own draws.
  - `OccupancyTransform(patch_sets=...)`: per patch state, the leaky-averaged gradient divided by the state's
    leaky-averaged occupation, as for the global unit's states. Each patch unit has its own step scale (the
    occupation-weighted mean square over its states), where the global unit (one unit) has one. Same clip, same
    visit floor (1 / (B S)).
  - Without `--patch-occupancy` the optimiser state and update are as before. The runner passes `{'b': global,
    patch b key: (P, S)}` only with the flag; before the switch-on slice the patch part is empty.
  - Recorded in config.json as `patch_occupancy` (only when set); resume checks it.
  - Backup of the pre-step-1 code: `*_pre_sampledpatch.py`.
  - **The step-1 runs 4268000 / 4268001 were held (`scontrol hold`) during the edit**, since they would start with
    the new code. They will be released after the regression test. Their path, without the flag, is unchanged.
- **Test 4268021:**
  1. `patch_occupation` with the loss key reproduces the loss's patch draws exactly (via `sampled_state_mean`).
  2. Regression: visit-count rule without patches, code from before 18:30 vs now, bitwise.
  3. K64 + 8x8 + 16x16 with `--patch-occupancy`, 4 slices: S = 16 with jackknife and same-draw counts, and exact
     statistics.
- **Test 4268021 PASSED (8 min):**
  1. `patch_occupation` with the loss key reproduces the loss's patch draws exactly (max difference 0 for both
     patch sets); occupations sum to 1 per patch unit.
  2. Regression: visit-count rule without patches, before vs after the change, slices 0-2: all arrays bitwise.
  3. `--patch-occupancy` trains, with S = 16 and with exact statistics. K64 + 8x8 + 16x16, slice 3: Wp_norm
     0.20208 / 0.20232, Qp_norm 0.0327 / 0.0329, median effective patch states 28.8 of 32; validation losses agree
     to 4 digits.
  - The temporary runs and the old-code copy are deleted.
- **18:35: step-1 runs released** (4268000 / 4268001; their path, without `--patch-occupancy`, is unchanged).
- **Step-2 runs submitted** (same structure and settings, plus `--patch-occupancy`):
  - exact statistics **4268085**; S = 64 sampled, jackknife, same-draw counts **4268086**;
  - quench 4268087 / 4268088;
  - **Inception FID 4268089** of all five variants after the quench: exact + Adam; exact + visit-count rule for
    the global unit; S = 64 + that; exact + visit-count rule for global unit and patch states; S = 64 + that;
  - gallery **4268090**.
- **On-device training without patches: final** (S = 256 training 4267286, 5.5 h; Inception **4267290**; gallery
  4267291, `galleries/2026-11-17_celeba32-grey_K784_on-device-training_exact-vs-S64-vs-S256-sampled-hidden-statistics_noise-start_quench1/`).
  CelebA-32 K784 + Q + per-pixel units, n = 2000, quench 1; floor 6.43:

  | Hidden statistics, optimiser | quench 1 (FID_inf) | KID x1e3 |
  |---|---|---|
  | exact, Adam | 104.4 (100.6) | 119 |
  | exact, visit-count rule | 106.4 (102.2) | 123 |
  | S = 64 sampled, jackknife, visit-count rule | 109.1 (105.3) | 129 |
  | S = 256 sampled, jackknife, visit-count rule | 110.1 (105.9) | 129 |

  - **S = 256 is no better than S = 64** (110.1 vs 109.1; bootstrap sd about 1.5). So the remaining 3-4 points
    against exact statistics with the same optimiser are not sample noise in the hidden statistics: 4x more
    samples would have shrunk them.
  - Candidates: training-seed variance (single seeds; the bootstrap sd covers only the sample noise), or a small
    systematic difference of the sampled runs (float32 draws, same-draw visit counts).
  - Either way, **S = 64 is enough**: device-measurable statistics train the model to within a few points of the
    exact computation.
  - Gallery: after the quench the rows for exact, S = 64 and S = 256 show essentially the same faces in every
    column (same start and noise), with no visible degradation.
- **19:40 status:** the patch runs are at slice 54-62 of 100. Exact statistics run at about 110 s per slice; S = 64
  runs at about 310 s per slice, so they need about 3.7 h more. Inception FID 4268004 / 4268089 is expected around
  01:00. `make_gallery.py` now names the real-image panel `01_real_images.png` (it was `01_real_digits.png`, which
  is wrong for faces); older gallery folders keep the old name.

### Overnight 2026-11-17/18: global factor units and a second seed for on-device training (submitted 20:05; the user: "yes, lets do it!")

- **Global factor units against the contrast deficit** (step 3 of the quality list; existing code via `--patch3`).
  K784 + 8x8 + 16x16 (exact statistics, Adam, seed 1) plus a third patch set from slice 55:
  - **4268394: `--patch3 32:32:32`**, one full-image patch, i.e. a second global Potts unit with 32 states (blocks
    with m = 1). It can carry an image-wide factor (lighting, background brightness) next to the K784 unit;
  - **4268395: `--patch3 24:4:32`**, 9 near-global 24x24 patches every 4 px (localised lighting, background per
    side).
  - quench 4268396 / 97; Inception FID **4268398** against the two-scale baseline (22.85 after the quench);
    contrast check **4268399**; gallery **4268400**.
- **Second seed for on-device training without patches** (is the 3-4 point gap between S = 64 and exact statistics
  run-to-run variance?): K784 + Q + per-pixel units, visit-count rule, seed 2.
  - exact **4268401**; S = 64 sampled with jackknife and same-draw counts **4268402**;
  - quench 4268403 / 4268404; Inception FID **4268405** of seeds 1 and 2 of both (quenched).

### CelebA-64 K3072: sampling crash and fix (2026-11-17 22:00)

- **Sampling 4266810 failed after 5 min with a segmentation fault** (exit 139; host memory 143 of 200 GB, so not an
  out-of-memory kill; no core file). The training (4266799, 10.4 h) is complete. The quench, Inception, gallery and
  contrast jobs (4266811-4266814) were stuck on the failed dependency.
- **Cause: the stacked parameter history is one array of 2.79e9 elements**, above 2^31 = 2.15e9. W and Q are
  1.26e9 each, the patch units 2.8e8. `make_drift` concatenates every group into one (100, 28M) array for
  `interpolate_parameters`, and XLA crashes on an array that size. Every array on its own is below the limit. The
  CelebA-64 K1536 history (1.54e9) sampled fine, and `--history-dtype float32` does not help, since it halves the
  bytes, not the element count.
- **Fix, `sample_latent.interpolate_history`:** a history above 2^31 - 1 elements is interpolated in column
  blocks of at most 2^30 elements, and the results are concatenated. Here that means 3 blocks. Each parameter is
  interpolated on its own, so the values are the same. A history up to 2^31 - 1 elements, which covers every
  earlier run, takes the single call as before. `LATENT_INTERP_MAX_ELEMS` overrides the limit for tests. Backup:
  `sample_latent_pre_chunkinterp.py`.
- **Test 4268591:** the drift of the CelebA-32 K784 + 8x8 + 16x16 run with forced blocks (10^6 elements) against
  the single call, 9 times, 3 states, float64 and float32 history; must agree to 1e-12 relative.
  - **It ran out of host memory (32 GB) after 1.5 min.** The forced limit made 547 blocks, each a separate constant in
    the compiled drift. That is a flaw of the test setting, not of the fix, which makes 3 blocks here.
  - **Retest 4268605:** a limit of 2e8, i.e. 3 blocks of at most 1e8 elements for this 2.7e8-element history, as in
    production; 96 GB. Sampling 4268592 now depends on it.
- **Resubmitted:** sampling **4268592** (after the test passes; same settings: n 2000, chunk 25, float32
  history, 200 GB). The quench 4266811 now depends on 4268592, so Inception 4266812, gallery 4266813 and contrast
  4266814 follow as before.
- **Retest 4268605 PASSED (2.4 min, 31 GB host memory):** 3 column blocks against the single call, maximum
  difference 3.6e-15 for a drift of magnitude 28 (1.3e-16 relative, i.e. float64 rounding from a different XLA
  fusion; not bitwise), in float64 and float32 history. Sampling 4268592 is released (22:04, waiting for priority).

### Overnight results 2026-11-17/18 (written 2026-11-18 09:55)

- **On-device training with patch units: final** (Inception **4268089**; gallery **4268090**,
  `galleries/2026-11-17_celeba32-grey_K784+patch8x8+patch16x16_on-device-training_exact-vs-S64_Adam-vs-visit-count-rule_quench1/`).
  CelebA-32 K784 + 8x8 every 2 px + 16x16 every 4 px from slice 55, Q, per-pixel units, mixed precision, seed 1,
  n = 2000, noise start, quench 1; floor 6.43:

  | Hidden statistics | Optimiser | quench 1 (FID_inf) | KID x1e3 | training |
  |---|---|---|---|---|
  | exact | Adam (standard) | 22.85 (19.14) | 15.2 | |
  | exact | visit-count rule, global unit (patches Adam) | 22.63 (18.87) | 15.2 | 3.3 h (4268000) |
  | S = 64 sampled, global and patch units | visit-count rule, global unit (patches Adam) | 23.46 (19.91) | 16.0 | 6.4 h (4268001) |
  | exact | visit-count rule, global unit and patch states | 23.12 (19.58) | 15.3 | 3.3 h (4268085) |
  | **S = 64 sampled, global and patch units** | **visit-count rule, global unit and patch states** | **23.59 (19.99)** | **16.1** | 6.4 h (4268086) |

  - **The fully on-device model (every hidden statistic measured from 64 samples, every hidden state on the
    visit-count rule) reaches 23.59, 0.7 Inception FID from the standard training (22.85).** Bootstrap sd 0.3-0.4.
  - Sampled statistics cost 0.5-0.8 (23.46 vs 22.63; 23.59 vs 23.12). The visit-count rule for the patch states
    costs about 0.1-0.5; for the global unit, nothing (22.63 vs 22.85).
  - With patches the cost of sampled statistics is much smaller than without them (about 3 FID at 106-111).
  - Gallery: after the quench, the five models show the same faces in every column (same start and noise), with no
    visible degradation from the sampled statistics. The as-sampled row of the fully on-device model is noisy, as
    every model before the quench (95.5 as sampled).
  - The sampled-statistics runs take about 2x longer to train (6.4 h vs 3.3 h) at S = 64.
- **Second seed, on-device training without patches** (Inception **4268405**; K784 + Q + per-pixel units, visit-count
  rule, quench 1):

  | | seed 1 | seed 2 | gap S = 64 - exact |
  |---|---|---|---|
  | exact statistics | 106.42 | 108.04 | |
  | S = 64 sampled, jackknife | 109.11 | 111.37 | seed 1: 2.7; seed 2: 3.3 |

  - The 3-point gap of sampled statistics reproduces with seed 2, so it is a small systematic cost of the sampled
    statistics in the no-patch model, not seed-to-seed variance. Seeds differ by about 1.6-2.3.
  - S = 256 did not reduce it (11-17), so it is not noise in the statistics either. With patches the cost drops to
    about 0.7. Open: the cause (the float32 draws or the same-draw visit counts); not pressing.
- **Global factor units against the contrast deficit** (Inception **4268398**; contrast **4268399**; gallery
  **4268400**, `galleries/2026-11-17_celeba32-grey_K784+patch8x8+patch16x16_plus-global-factor-units_full-image-32-states_vs_24x24-every4px_noise-start_quench1/`).
  K784 + 8x8 + 16x16 plus a third patch set from slice 55, quench 1:

  | | quench 1 (FID_inf) | image contrast (real 0.229) | dark pixels (real 12.7%) | training |
  |---|---|---|---|---|
  | baseline (two scales) | 22.85 (19.14) | 0.197 | 6.8% | |
  | + one full-image Potts unit, 32 states | 22.76 (19.27) | 0.197 | 6.7% | 2.3 h (4268394) |
  | + 24x24 every 4 px (9 patches) | 22.43 (18.95) | 0.196 | 6.7% | 3.3 h (4268395) |

  - **Neither restores contrast**: image contrast stays at 0.196-0.197 (real 0.229), and dark pixels at half the
    real share. The FID changes are within about 1 sd (0.2-0.4).
  - So the contrast deficit is not a missing image-wide factor that a global or near-global unit could carry.
    Remaining suspects: the visible model (quartic potential pulling pixels to mid-grey), or the training loss
    weighting the tails of the pixel distribution too weakly.
  - Gallery: the faces are the same in every column with and without the extra units.
- **CelebA-64 K3072 sampling 4268592**: running since 23:08 with the column-block interpolation, chunk 79 of 80
  at 09:48 (about 8 min per chunk of 25). The fix works on the real history. Quench, Inception, gallery and
  contrast follow (4266811-4266814).

### Toward a minimal model: small global units (2026-11-18 10:10; the user: "can we find a minimal model that performs just or almost as well [...] maybe we can try K32+8x8+16 [...] submit both K32 and K64 in parallel")

- **Parameter budget per time slice** (couplings W + Q): the global unit is 2 x K x 1024 (K784 1.61 M, K64 0.13 M,
  K32 0.07 M); the two patch sets are fixed at 1.10 M (8x8 every 2 px 0.69 M, 16x16 every 4 px 0.41 M). So at small
  K the patches dominate, and K alone can cut the count by at most about 2.4x (2.71 M -> 1.17 M). Further cuts must
  come from the patch states (32 each), the Q terms, or the number of time slices (100 knots).
- **Dead states in the K784 two-scale run** (training log, states with mean occupation below 1e-6 on the slice's
  batch; effective states = exp of the occupation entropy):

  | slice (t) | 0 (5.0) | 10 (4.5) | 20 (4.0) | 30 (3.5) | 40 (3.0) | 55 (2.2) | 70 (1.5) | 85 (0.7) | 99 (0) |
  |---|---|---|---|---|---|---|---|---|---|
  | dead of 784 | 0 | 284 | 217 | 72 | 31 | 56 | 231 | 271 | 307 |
  | effective states | 693 | 237 | 322 | 405 | 500 | 386 | 257 | 253 | 244 |

  - The dead set changes with t: states die at t = 4.5, revive by t = 3 (31 dead), and die again towards t = 0. A
    state can be removed from hardware only if it is dead at every t. 250-500 states are in use, but the K784 vs
    K1536 vs K3072 comparison with patches (24.7 / 23.9 / 23.8 with 8x8) suggests the patches carry most of the
    detail, so a much smaller global unit may do.
- **Submitted** (same recipe as the K784 baseline, 22.85 after the quench: exact statistics, Adam, 8x8 every 2 px +
  16x16 every 4 px from slice 55, seed 1, 2000 samples from the noise start):
  - K32: pipeline **4271874**, quench 4271876; K64: pipeline **4271875**, quench 4271877;
  - Inception FID **4271878** (with the K784 baseline); gallery **4271879**
    (`galleries/2026-11-18_celeba32-grey_K32-vs-K64-vs-K784_global-states+patch8x8+patch16x16_noise-start_quench1/`).
  - Training about 3 h each (the patch slices dominate); pending on priority at 10:10.
- **10:15: K64 + 8x8 only** (the user: "maybe directly submit K64 + 8x8 only as well"): 0.82 M couplings per slice
  (0.13 M global + 0.69 M patches), 3.3x fewer than the K784 two-scale model. Same recipe without the 16x16 set.
  - Pipeline **4271892**, quench 4271893; Inception FID **4271894** against K784 + 8x8 (24.7 after the quench, 11-16);
    gallery **4271895** (`galleries/2026-11-18_celeba32-grey_K64-vs-K784_global-states+patch8x8-only_noise-start_quench1/`).
- **CelebA-64 K3072 + 16x16 every 4 px: result** (training 4266799, 10.4 h; sampling 4268592, 10.8 h with the
  column-block interpolation and float32 history; quench 4266811; Inception **4266812**; gallery 4266813,
  `galleries/2026-11-17_celeba64-grey_K1536-vs-K3072_patch16x16-every4px_noise-start_quench1/`; contrast 4266814,
  `galleries/2026-11-17_celeba64-grey_contrast-check_K1536-vs-K3072_patch16x16_quench1/`). n = 2000, floor 5.91:

  | | as sampled | quench 1 (FID_inf) | KID x1e3 | image contrast (real 0.234) | dark pixels (real 13.5%) |
  |---|---|---|---|---|---|
  | K1536 + 16x16 (11-17) | | 58.46 (55.62) | 53.7 | 0.180 | 3.3% |
  | K3072 + 16x16 | 130.8 | **58.28 (55.74)** | 53.6 | 0.179 | 3.3% |

  - **Doubling the global unit gains nothing at 64x64** (58.3 vs 58.5, within 1 sd), and the contrast deficit
    (-23%) is unchanged. As at 32x32, K is not the bottleneck once patch units are there.
  - Gallery: blurry, low-contrast faces of the same quality for both K; backgrounds and hair at mid-grey.
  - So 64x64 needs something other than more global states: finer patches (8x8 at 64x64 was too slow, 5 steps/s),
    or a fix of the contrast / blur problem. This also supports the small-model direction.

### Where do the faces go wrong? Oracle-start scan on CelebA-32 (2026-11-18 10:55; the user: "yes")

- **Context: K has saturated.** CelebA-64 K3072 = K1536, and at 32x32 with patches K784 to K3072 gains about 1. The
  validation losses (lower is better) show why:

  | | slice 50 (t = 2.5, before the patches) | slice 60 (t = 2.0, with 8x8 + 16x16) | slice 98 |
  |---|---|---|---|
  | K32 + 8x8 + 16x16 | -23651 | -23918 | -41597 |
  | K64 + 8x8 + 16x16 | -23717 | -23921 | -41532 |
  | K784 + 8x8 + 16x16 | -23921 | -23924 | -42023 |
  | K1536 + 8x8 + 16x16 | -23954 | -23931 | -42265 |
  | K784 + 8x8 + 4x4 + 16x16 | -23921 | -23935 | -49107 |

  - Once the patches act, the global unit's size barely matters for the loss (K32 within 6 units of K784 at
    slice 60).
  - More patch scales lower the late-slice loss a lot (-49107 vs -42023) but the FID only a little (22.3 vs 22.9):
    at small t the loss is dominated by pixel-scale detail that the FID hardly sees.
  - Working hypothesis: the limit is the model's form, not its size. It is one layer of independent template units
    whose forces add, so each pixel is pulled to an average of the templates covering it (blur, mid-grey contrast
    loss). Every large gain so far was structural (patches, switch-on time, quench).
- **The scan:** start the reverse process at forward time t from exactly noised *training* images (`--oracle-start
  K`: x = e^-t x0 + sqrt(T (1 - e^-2t)) xi; a fixed choice of images and noise, the same in every job), then quench 1.
  If starting from real noised faces at some t gives much better samples than the normal start, the model errs in
  the part of the reverse process before t. MNIST (11-12): the error built up at t = 1.5-2.7.
  - Model: K784 + 8x8 every 2 px + 16x16 every 4 px from slice 55 (t = 2.22), exact statistics, Adam, seed 1
    (22.85 after the quench from the normal start). It is the reference for the on-device and small-K runs.
  - Slices 20 / 30 / 40 / 55 / 70 / 85 / 95 (t = 3.99 / 3.48 / 2.98 / 2.22 / 1.46 / 0.71 / 0.20): jobs **4271997**
    to **4272003**, sampling and quench in one job each (the sampler refuses to quench an existing oracle file).
  - Inception FID **4272004** (as sampled and quenched, with the normal start); gallery **4272005**
    (`galleries/2026-11-18_celeba32-grey_K784+patch8x8+patch16x16_oracle-start-scan_t3.99-to-0.20_quench1/`).
  - Reading: the FID at small t tends to the floor, since the samples are then nearly the training images.
- **11:10 status.** Small-K training is done: K64 + 8x8 (pipeline incl. sampling 1.1 h; its quench waits for a GPU),
  K32 / K64 + 8x8 + 16x16 (sampling). Oracle starts at t = 1.46 / 0.71 / 0.20 are done, the other four are running.
  - Validation losses of the small models: K32 + 8x8 + 16x16 at slice 70 is -23296 vs -23324 for K784 (28 units
    worse; K784 8x8-only vs two-scale differ by 101 and 1.9 FID); before the patches (slice 50) the gap is 270.
  - K32's global unit ends with 3 effective states at t = 0 (31 at slice 50).
- **Small global units: results** (Inception **4271878** and **4271894**; galleries **4271879**
  `galleries/2026-11-18_celeba32-grey_K32-vs-K64-vs-K784_global-states+patch8x8+patch16x16_noise-start_quench1/` and
  **4271895** `galleries/2026-11-18_celeba32-grey_K64-vs-K784_global-states+patch8x8-only_noise-start_quench1/`).
  CelebA-32, exact statistics, Adam, seed 1, n = 2000, noise start; floor 6.43:

  | Model | couplings per slice | as sampled | quench 1 (FID_inf) | KID x1e3 | training + sampling |
  |---|---|---|---|---|---|
  | K784 + 8x8 + 16x16 | 2.71 M | | 22.85 (19.14) | 15.2 | |
  | K64 + 8x8 + 16x16 | 1.23 M | 98.0 | **24.61 (20.97)** | 17.0 | 1.5 h (4271875) |
  | K32 + 8x8 + 16x16 | 1.17 M | 98.8 | 25.36 (21.67) | 18.1 | 1.5 h (4271874) |
  | K784 + 8x8 | 2.30 M | | 24.66 (20.83) | 17.4 | |
  | K64 + 8x8 | **0.82 M** | 104.1 | 26.41 (22.74) | 19.5 | 1.1 h (4271892) |

  - **The global unit can shrink 12-25x for 1.8-2.5 FID.** K64 + 8x8 + 16x16 (1.23 M) equals K784 + 8x8 (2.30 M),
    and K64 + 8x8 (0.82 M) is 1.75 behind K784 + 8x8. K32 vs K64 costs 0.75.
  - Gallery: after the quench K32 and K64 make coherent faces of the same style as K784 (smooth, slightly flat);
    no broken shapes. Different people in many columns (different models from the same noise).
  - The patches are now most of the budget (1.10 M of 1.23 M). Next levers for a minimal model: fewer patch states
    (32 -> 16 / 8), no Qp, fewer time knots.
  - Training is fast at small K: 1.1-1.5 h including sampling (K784 two-scale: 3.3 h).
- **Oracle-start scan: result** (Inception **4272004**; gallery **4272005**,
  `galleries/2026-11-18_celeba32-grey_K784+patch8x8+patch16x16_oracle-start-scan_t3.99-to-0.20_quench1/`).
  K784 + 8x8 + 16x16, n = 2000, quench 1; floor 6.43; dark = share of pixels < 0.1 (real test 11.8%):

  | start | as sampled | quench 1 (FID_inf) | KID x1e3 | dark (q1) |
  |---|---|---|---|---|
  | noise, t = 5 (normal) | | 22.85 (19.14) | 15.2 | 6.8% |
  | real noised, t = 3.99 | 88.8 | 22.46 (19.06) | 14.4 | 7.6% |
  | t = 3.48 | 88.2 | 22.54 (18.96) | 14.1 | 7.8% |
  | t = 2.98 | 86.5 | 21.81 (18.16) | 13.6 | 8.2% |
  | t = 2.22 | 75.6 | 19.91 (16.39) | 11.3 | 11.3% |
  | t = 1.46 | 68.3 | 17.62 (13.68) | 9.1 | 11.5% |
  | t = 0.71 | 55.4 | 14.50 (10.72) | 6.5 | 11.5% |
  | t = 0.20 | 36.7 | 12.56 (8.65) | 5.9 | 11.6% |

  - **Above t = 3.5 nothing is lost**: real noised faces at t = 4 or 3.5 give the same FID as the normal start.
  - **Below t = 3 the FID falls steadily with later starts** (about 2 per 0.75 time units), so the error builds up
    over the whole range t < 3, not in one stage.
  - **The last stretch alone costs about 6**: from nearly clean real faces (t = 0.2) the SDE and the quench give
    12.6, twice the floor. The Inception FID is very sensitive to pixel-scale texture (real test images + 1%
    Gaussian pixel noise: 6.4 -> 10.8; + 2%: 28.1), so a large part of the gap is fine texture at the very end,
    which the gallery does not show (the t = 0.2 faces look like the training images).
  - **The contrast deficit forms early**: the dark-pixel share is real-like (11.3-11.6%) for every start at
    t <= 2.22, but low for starts at t >= 2.98 (7.6-8.2%) and from noise (6.8%). So the mid-grey bias arises in
    the global stage (t > 2.2), before the patches act.
  - My gallery reading (problem between t = 3 and 1.5) was incomplete: the faces' identity and contrast are set
    there, but the FID loses as much again in fine texture over t < 1.5.
  - Implications: (1) contrast and global layout: the early stage, where a clamped per-image variable would act;
    (2) texture: the small-t model and the quench (finer patch scales lower the late loss most: 4x4 three-scale
    22.3).
  - Open diagnostic: what the last stretch and the quench do to real faces (RMS change, contrast, high-frequency
    power of the t = 0.2 samples against their source training images; the quench applied directly to clean
    training images).
- **What the last stretch and the quench do to real faces** (submitted 12:05; the user: "lets run the cheap
  diagnostics!"). New tool `oracle_source_check.py`:
  - takes the 2000 training images behind the oracle sets (indices from the samples' .json), saves them as
    `samples/oracle-source_train-x0_n2000.npy`, and quenches them directly (`sample_latent.do_quench`, quench 1)
    into `..._q1.npy`;
  - compares each image with its source: x0 quenched, and the oracle sets from t = 0.20 / 0.71 / 1.46 / 2.22 (as
    sampled and quenched). RMS pixel change, correlation, contrast ratio, brightness shift, dark share, and the
    power in low / mid / high spatial-frequency bands relative to the source (also of the change itself). The normal
    samples' band powers are compared unpaired. Figures: examples, radial spectra.
  - Job **4272175** (`galleries/2026-11-18_celeba32-grey_K784+patch8x8+patch16x16_what-the-last-stretch-and-the-quench-do-to-real-faces/`);
    Inception FID **4272176** of the sources, the sources quenched, the t = 0.20 oracle set and the normal samples.
- **Result: the last stretch and the quench on real faces** (job 4272175, 1 min; Inception **4272176**; report,
  `examples.png` and `spectra.png` in the gallery folder above). The 2000 oracle source images, pixels in [0, 1]:

  | set (paired with its source) | Inception FID (FID_inf) | RMS change | contrast ratio | power vs source: low / mid / high |
  |---|---|---|---|---|
  | **the training images themselves** | **11.07 (6.88)** | 0 | 1 | 1 / 1 / 1 |
  | training images, quench 1 only | 13.36 (9.71) | 0.023 | 0.992 | 1.005 / 0.956 / 0.789 |
  | oracle t = 0.20, as sampled | 36.7 | 0.029 | 0.999 | 0.999 / 0.990 / 0.988 |
  | oracle t = 0.20, quench 1 | 12.56 (8.62) | 0.031 | 0.992 | 1.007 / 0.955 / 0.786 |
  | oracle t = 0.71, quench 1 | 14.50 | 0.046 | 0.991 | 1.008 / 0.945 / 0.771 |
  | oracle t = 1.46, quench 1 | 17.62 | 0.073 | 0.988 | 1.006 / 0.924 / 0.742 |
  | oracle t = 2.22, quench 1 | 19.91 | 0.109 | 0.979 | 0.996 / 0.879 / 0.706 |
  | normal samples, quench 1 (unpaired) | 22.85 | | 0.865 (0.197 vs 0.228) | **0.766 / 0.705 / 0.587** |

  (bands: |k| <= 4, 4 < |k| <= 8, |k| > 8 cycles per image; image mean removed)
  - **The training images score 11.07, not the 6.43 of the raw test images.** The model is trained on the paper's
    preprocessing, which adds N(0, 0.01^2) pixel noise to every training image (`latent_model._normalise_like_paper`,
    as for MNIST). Real test images + 1% noise score 10.75. So a perfect model of its training data would score
    about 11: **the reachable floor is about 11, not 6.4**, and the normal samples are about 12 above it.
  - **The quench costs about 2.3 on real images** (11.07 -> 13.36): it removes the noise but also some fine detail
    (high-band power -21%; above |k| = 16 about 10x less power), without changing the contrast (0.992).
  - So of the "last stretch costs about 6" (oracle t = 0.2: 12.56), about 4.6 is the noise in the training data
    and about 1.5-2.3 the quench; the model's own late stage is nearly lossless.
  - **The normal samples lack power at every scale, most at the coarse scale relative to the oracle runs**: low band
    0.77 of the real faces, while every oracle start at t <= 2.22 keeps it at 0.99-1.01. So the large-scale contrast
    deficit is made in the global stage (t > 2.2), as the dark-pixel share said.
  - Options: (1) evaluate also against the statistics of the noisy training images, or train on clean images
    (the 1% noise is the paper's choice; one K64 two-scale run is 1.5 h); (2) for the coarse-scale deficit, the
    global stage (a clamped per-image variable, or global-to-patch coupling).

### Training on clean images, and a clamped per-image variable (2026-11-18 12:50; the user: "could we try both in parallel?")

- **(A) Clean training data: `train_latent.py --data-noise SD`** (code 12:40).
  - `latent_model._normalise_like_paper(raw, noise)` / `make_problem(..., data_noise=)`: the training images get
    SD x the paper's noise draw instead of 0.01 (0: clean). The global mean / std stay those of the paper's noisy
    images, so sampling and every evaluation tool map back to pixels as before. Default (None): exactly the paper's
    data. Refused for mnist28 and gauss2_8.
  - Recorded in config.json as `data_noise` only when set; resume checks it in both directions.
  - Backups `*_pre_datanoise.py`.
  - Test **4272439**: the default data bitwise as before (old module vs new); data_noise 0 maps back to the raw
    images; 0.005 is half the default noise; mnist28 refuses; a 3-slice K64 training with `--data-noise 0`
    records `data_noise: 0.0`.
  - Run (after the test): K64 + 8x8 + 16x16, `--data-noise 0`, otherwise as the K64 two-scale run (24.61 after the
    quench): pipeline **4272440**, quench 4272441, Inception **4272442** (as sampled and quenched, against the
    paper-data run), contrast 4272443, gallery **4272444**
    (`galleries/2026-11-18_celeba32-grey_K64+patch8x8+patch16x16_trained-on-paper-data-1pct-noise_vs_clean-data_noise-start_with-and-without-quench1/`).
  - Expectation: up to about 4.6 better if the samples become clean; the quench may become unnecessary (compare
    as sampled).
  - **Test 4272439 PASSED (1.5 min):** default data bitwise as before (CelebA-32 and Fashion); `--data-noise 0`
    maps back to the raw images to 6e-17 with the same mean / std; the default noise has SD 0.01000 and 0.005 gives
    exactly half; mnist28 refuses; the 3-slice training records `data_noise: 0.0`. The temporary run is deleted.
    Pipeline 4272440 released (waiting for priority at 13:50).
- **(B) A clamped per-image variable c** (the user's idea of 11-18): code next.
- **(B) code (2026-11-18 14:30): a clamped per-image label, `train_latent.py --cond C [--cond-on global,patch,visible]`.**
  - Every training image gets a fixed label c: its k-means cluster among C (normalised training images, seed
    31000 + seed; `cond_clusters.npz` in the run). The label shifts the global unit's state biases (`Bc` (C, K)),
    every patch set's state biases (`Bpc`, `Bpc2`, `Bpc3` (C, P, S)) and / or adds a visible field (energy
    -hc[c] . x, `hc` (C, N)). All start at zero (the unconditioned model) and are time dependent like every
    parameter. Given the visible state and c the hidden units still factorise (exact sums).
  - `latent_model`: the loss and the force take X = (visible states, one-hot labels); a plain array traces exactly
    as before. `patch_terms(bp_extra=)`, `cond_patch_offsets`, `COND_KEYS`; `in_compute_dtype` casts a tuple.
  - `train_latent`: the batch's labels from the same index draw as `sample_forward_process` (same key); validation
    with the labels; learning rates: hc at the visible rate, Bc at the global unit's bias rate, patch offsets at
    their units' bias rates; saved in every slice and `hidden_history.npz` after everything else; config `cond`,
    `cond_on` (only when set; resume checks). Only with exact statistics, `--hidden-norm element`, Potts units,
    `--step-impl default`, no harmonic modes / frozen units / CD-1.
  - `sample_latent`: labels drawn once per sample from the cluster frequencies (`default_rng(32000)`, saved as
    `<stem>_cond_labels.npy`, recorded in the manifest) and held through the reverse SDE and the quench (the drift's
    args). Default SDE sampler and `--quench` only; every other caller refuses a --cond run.
  - Backups `*_pre_cond.py`.
  - Note: the clean-data pipeline 4272440 started (13:58) after the edits, so it runs on the new code; its path
    (no --cond) must be unchanged, which test 4272529 checks (part 2).
  - **Test 4272529:** (1) loss and force without labels bitwise as before; zero label parameters = no labels; label
    bias offsets = shifted biases per label (loss and force); the visible field changes the loss by exactly
    (-2 g.h + |h|^2) / 2T^2 and the force by -h; mixed precision with labels; the sampler drift (a constant history)
    equals the training force per sample with its label. (2) Old vs new code without --cond, K16 + 8x8 + 16x16,
    3 slices: bitwise. (3) `--cond 8` trains 4 slices. (4) `--cond 8` end to end: 100 short slices, 16 samples,
    quench.
- **Test 4272529 (21 min): parts 2-4 PASSED, part 1 stopped at a bug in the test script** (a duplicate dict key at
  check 3, not in the code). Checks 1-2 passed: without labels the loss and the force are bitwise as before; zero
  label parameters give the unlabelled loss (2.5e-16). (2) Old vs new code without --cond, K16 + 8x8 + 16x16,
  3 slices: all slice arrays bitwise, so the clean-data run 4272440 (started on the new code) is valid. (3)
  `--cond 8`: clusters of 13149-30220 images; hc trains from slice 0, Bc from slice 1, the patch offsets from the
  switch-on (norms 0.20 / 0.0013 / 0.005 / 0.004 at slice 3). (4) End to end: 100 short slices, 16 samples (labels
  saved, manifest records them), quench: all finite. Temporary runs and the old-code copy are deleted.
  - Fixed test script, part 1 only: **4272680**.
- **Submitted (14:20; after 4272680 passes):** K64 + 8x8 + 16x16, the paper's data, `--cond 32`, otherwise as the
  K64 two-scale run (24.61 after the quench):
  - **B1** labels on the hidden biases only (`--cond-on global,patch`; the user's idea: a clamp connecting the
    global unit and the patches): pipeline **4272681**, quench 4272683;
  - **B2** also a visible field (`--cond-on global,patch,visible`): pipeline **4272682**, quench 4272684;
  - Inception **4272685**, contrast 4272686, gallery **4272687**
    (`galleries/2026-11-18_celeba32-grey_K64+patch8x8+patch16x16_clamped-per-image-label-32-clusters_on-hidden-biases_vs_plus-visible-field_noise-start_quench1/`).
  - The quench jobs wait for the pipelines with afterany (a failing summary step must not block them).
- **Test 4272680 PASSED (1.5 min; CelebA-32 K784 + 8x8 + 16x16 at slice 85, 64 noised images, C = 8):**
  (1) no labels: loss and force bitwise as before; (2) zero label parameters vs no labels: 2.5e-16; (3) label bias
  offsets vs the shifted biases of each sample's label: loss 9.3e-16, force 1.2e-14; (4) visible field: loss change
  vs (-2 g.h + |h|^2) / 2T^2 1.6e-14, force exactly g - h; (5) mixed precision with labels: 1.3e-8 of float64;
  (6) the sampler drift vs the training force per sample with its label: 1.1e-15 (vs the unconditioned force:
  0.24, so the labels act). B1 / B2 (4272681 / 4272682) released; waiting for priority at 14:25.
- **14:55 interim: validation losses of the clamped-label runs** (lower is better; with each image's own label;
  K64 + 8x8 + 16x16; K784 + 8x8 + 16x16 for scale):

  | slice (t) | 10 (4.5) | 30 (3.5) | 50 (2.5) | 60 (2.0) | 70 (1.5) | 75 (1.2) |
  |---|---|---|---|---|---|---|
  | K64 (no label) | -25512.5 | -25199.9 | -23716.7 | -23921.2 | -23298.0 | -23082.9 |
  | B1: label on hidden biases | -25540.4 | -25218.3 | -23716.1 | -23943.0 | -23316.9 | -23108.7 |
  | B2: + visible field | -25541.6 | -25265.5 | **-23892.6** | -23947.5 | -23325.9 | -23116.6 |
  | K784 (no label) | -25513.4 | -25240.5 | -23920.6 | -23923.6 | -23323.6 | |

  - B2's field lifts the global stage: at slice 50 (t = 2.5) K64 + label field is 176 units better than K64 and
    close to K784 (-23921). B1 (biases only) gains nothing there, but 20-26 units once the patches act.
  - Label parameter norms at slice 75: Bc 20 / 18, Bpc 33 / 31, Bpc2 16 / 13 (B1 / B2), hc 4.2 (B2): the labels are
    used.
  - The conditional loss is not the unconditional FID (sampling draws the label); the FIDs decide.
  - Clean-data run 4272440 at slice 97 (sampling next).
- **Related work for the clamped label** (checked 2026-11-18):
  - Bao, Li, Sun, Zhu, "Why Are Conditional Generative Models Better Than Unconditional Ones?", NeurIPS 2022
    workshop, arXiv 2212.00362. Self-conditioned diffusion models (SCDM): k-means on self-supervised features, the
    cluster index as the condition, labels drawn at sampling; the gain comes from partitioning the data into simpler
    conditional distributions. ImageNet 64x64 FID 3.94 without labels.
  - Hu, Zhang, Asano, Burghouts, Snoek, "Self-Guided Diffusion Models", CVPR 2023, arXiv 2210.06462. Self-labelled
    guidance: k-means on self-supervised image features, used with classifier-free guidance; beats unguided models
    and can match or beat ground-truth-label guidance.
  - Ours differs: k-means on raw pixels (coarse brightness layout, not semantics), and the label enters as clamped
    bias patterns of an energy model rather than a network input; no guidance (a mixture, as SCDM).
- **(A) Clean training data: result** (pipeline 4272440, 1.5 h; quench 4272441; Inception **4272442**; contrast
  4272443; gallery **4272444**). K64 + 8x8 + 16x16, seed 1, n = 2000, noise start; floor 6.43:

  | Training data | as sampled (FID_inf) | quench 1 (FID_inf) | KID x1e3 (q1) | contrast q1 (real 0.229) | dark q1 (real 12.7%) |
  |---|---|---|---|---|---|
  | the paper's (1% pixel noise) | 98.0 (93.8) | 24.61 (20.82) | 17.0 | 0.193 | 6.1% |
  | clean (`--data-noise 0`) | **85.5 (81.4)** | **23.91 (20.36)** | 16.2 | 0.191 | 6.0% |

  - **Clean data help the raw samples (-12.5) but hardly the quenched ones (-0.7).** The as-sampled images stay
    grainy (85.5; gallery: the same speckle as with noisy data), so the grain is the model's own end-of-SDE noise,
    not the data's 1%; the quench is still needed and removes it either way.
  - So the "reachable floor of about 11" applies to a perfect model of the noisy data, but our models are not limited
    by it: after the quench the data noise costs only 0.7. The contrast deficit is unchanged.
  - Gallery: quenched faces of both runs are near-identical column by column (same start and noise).
  - Decision for the paper: either keep the paper's preprocessing (consistency) or use clean data (-0.7); minor.
- **Theory note** (the user: "could you add the guiding to the theory note?"): `writeups/latent_physical_SGM_writeup/theory.tex`
  has a new Section 6, "A clamped per-image label: conditioning and guidance" (backup
  `theory_pre_cond_2026-11-18.tex.bak`; PDF rebuilt, 11 pages, compile jobs 4273223 / 4273225, no errors or warnings):
  the label, the conditional free energy (exact sums given x and c), conditional score matching with the label-routed
  bias rule and the field's stationarity condition E_{p_t(.|c)}[grad F] = 0, generation with a drawn and held label,
  and guidance in three realisations (label amplification B, Bp, h -> (1+w) x: one physical knob, exact CFG for the
  field only; exact CFG against the label mixture with calibrated normalisers log Z_t(c), C force evaluations per
  step; label dropout with a null label, two readouts per step), plus related work (Bao et al. 2022, Hu et al. 2023,
  Ho and Salimans 2022). `compile.sh` pointed at the folder's old location; its `cd` path is fixed.
- **(B) Clamped label: result** (pipelines 4272681 / 4272682, 1.6 h each; quench 4272683 / 4272684; Inception
  **4272685**; contrast 4272686; gallery **4272687**). CelebA-32, K64 + 8x8 + 16x16, the paper's data, seed 1,
  n = 2000, noise start; floor 6.43:

  | Model | couplings + label params per slice | as sampled | quench 1 (FID_inf) | KID x1e3 | contrast q1 (real 0.229) | dark q1 (real 12.7%) |
  |---|---|---|---|---|---|---|
  | K64 (no label) | 1.23 M | 98.0 | 24.61 (20.78) | 16.9 | 0.193 | 6.1% |
  | **B1**: label on hidden biases (C = 32) | 1.43 M | 92.4 | **23.52 (19.96)** | 15.9 | 0.194 | 6.4% |
  | **B2**: + visible field | 1.46 M | 92.3 | **23.42 (19.90)** | 15.9 | 0.195 | 7.0% |
  | K784 (no label), for scale | 2.71 M | | 22.85 (19.14) | 15.2 | 0.197 | 6.8% |

  - **The clamped label gains about 1.1-1.2 Inception FID** (bootstrap sd 0.3), with 16-19% more parameters; K64 +
    label sits between K64 and K784 (half of the K784 gain at about half of its parameters).
  - The visible field adds only 0.1 beyond the hidden-bias offsets, despite its large global-stage loss gain
    (slice 50: -176): the conditional loss improvement does not carry over to the unconditional samples. This is
    where conditional training differs from unconditional sampling: the label is drawn at random, and getting the
    right images per cluster is not the same as good images overall.
  - Contrast improves a little (dark pixels 6.1% -> 6.4% / 7.0%; real 12.7%): the label does not fix the
    contrast deficit.
  - Gallery: coherent faces of the same style, no artefacts; different people in many columns (the label changes
    the trajectory).
  - Next, cheap: guidance by label amplification (sampling only, B, Bp, h x lambda; theory note section 6), e.g.
    lambda = 1.5 / 2 / 3; and a finer partition (C = 128, or labels from features).

### Clamped label: guidance and a finer partition (2026-11-18 16:20; the user: "okay lets do both 1 and 2")

- **Code: `sample_latent.py --cond-scale LAMBDA`** (a --cond run only): guidance by label amplification, every label
  parameter (Bc, Bpc*, hc) multiplied by LAMBDA >= 0 in the drift and the quench; the same drawn labels, start and
  noise as LAMBDA = 1, so the comparison is per sample. Samples get `_cs<LAMBDA>`; the manifest records `cond_scale`.
  Theory note section 6 (exact classifier-free guidance for the visible field relative to the zero-offset model; a
  tempered label prior for the hidden offsets). Backup `sample_latent_pre_condscale.py`. Default path unchanged.
- **(1) Guidance scan:** B1 (label on hidden biases) and B2 (+ visible field), LAMBDA = 1.5 / 2 / 3: sampling and
  quench jobs **4273311-4273313** (B1), **4273314-4273316** (B2); Inception **4273321** (with LAMBDA = 1 and K64
  without label); contrast 4273322; gallery **4273323**
  (`galleries/2026-11-18_celeba32-grey_K64+patch8x8+patch16x16_label32_guidance-by-label-amplification-x1-x1.5-x2-x3_noise-start_quench1/`).
- **(2) Partition:** the B1 model with C = 8 and C = 128 clusters (32 done): pipelines **4273317** (C128),
  **4273318** (C8); quench 4273319 / 4273320; Inception **4273324** (C 8 / 32 / 128 and no label); contrast 4273325;
  gallery **4273326**. Label parameters per slice C x (64 + 194 x 32): C8 0.05 M, C32 0.20 M, C128 0.80 M.
  The visible field is left out (it added 0.1 at C = 32).
- **(1) Guidance by label amplification: result** (sampling 4273311-4273316, 31-32 min each; Inception **4273321**;
  contrast 4273322; gallery **4273323**). Same labels, start and noise as x1 (checked: the label files are
  identical); quench 1; floor 6.43; real: contrast 0.229, dark 12.7%, bright 3.3%, sd of image brightness 0.114:

  | label settings x | B1 (hidden biases): FID (FID_inf) | contrast / dark | B2 (+ field): FID (FID_inf) | contrast / dark / bright / sd(brightness) |
  |---|---|---|---|---|
  | 1 | **23.52** (20.07) | 0.194 / 6.4% | **23.42** (20.13) | 0.195 / 7.0% / 1.3% / 0.111 |
  | 1.5 | 36.33 (32.96) | 0.143 / 1.3% | 30.85 (27.31) | 0.192 / 7.9% / 2.9% / 0.139 |
  | 2 | 49.54 (46.54) | 0.108 / 0.3% | 38.81 (35.82) | 0.195 / 9.5% / 4.3% / 0.161 |
  | 3 | 70.86 (68.17) | 0.072 / 0.1% | 53.01 (49.44) | 0.204 / 13.0% / 6.3% / 0.187 |

  - **Guidance by amplification only hurts**, at every factor tried, and steeply.
  - **Amplifying the hidden-bias offsets collapses the images to flat grey** (B1 x3: contrast 0.072; gallery: the
    same washed-out mean-like face in every column). A sharper label prior over the hidden states pushes each unit
    to its most frequent, average-like templates: the opposite of what classifier-free guidance does.
  - **Amplifying the visible field restores the dark / bright tails** (B2 x3: dark 13.0%, real 12.7%; bright 6.3%,
    real 3.3%) but exaggerates the brightness differences between images (sd 0.187 vs 0.114: backgrounds too dark
    or too bright; gallery: identities kept, backgrounds pushed to extremes, faces smoother).
  - So the label amplification of the theory note (option 1) is not a usable guidance here; exact guidance (option
    2, calibrated normalisers) or label dropout (option 3) would be the proper tests. A field-only amplification at
    small factors (1.1-1.2) might buy contrast cheaply, but the FID trend makes a gain unlikely.
- **(2) Partition: result** (pipelines 4273317 C128, 1.7 h, and 4273318 C8, 1.6 h; quench 4273319 / 4273320;
  Inception **4273324**; contrast 4273325; gallery **4273326**). Label on the hidden biases (B1 type):

  | clusters C | cluster sizes | label params per slice | FID (FID_inf) | KID x1e3 | contrast / dark |
  |---|---|---|---|---|---|
  | none | | 0 | 24.61 (20.78) | 16.9 | 0.193 / 6.1% |
  | 8 | 13149-30220 | 0.05 M | 23.57 (20.07) | 16.1 | 0.192 / 6.1% |
  | 32 | 2941-9373 | 0.20 M | 23.52 (19.96) | 15.9 | 0.194 / 6.4% |
  | 128 | 303-2791 | 0.80 M | 23.64 (20.10) | 15.7 | 0.195 / 6.5% |

  - **The gain does not depend on the number of clusters**: 8, 32 and 128 give the same 1.0-1.1 (within the bootstrap
    sd 0.3). Eight clusters (0.05 M extra) already give all of it; a finer pixel-space partition adds nothing.
  - So the label's benefit is a coarse, early decision (about the 8 broadest image types: background / lighting /
    hair mass), not a fine partition of the data. Contrast is unchanged at every C.
  - Gallery: the same quality for all C; different people per column.
- **Summary of the clamped label (11-18):** a cheap, physically simple addition (C bias patterns, loaded per sample)
  worth about 1 FID at K64 with 8 labels (0.05 M parameters); it does not fix the contrast; amplification guidance
  hurts. Best small model so far: K64 + 8x8 + 16x16 + 8-cluster label, 1.28 M couplings per slice, 23.57.

### CelebA-64: the small model with an 8-cluster label (2026-11-18 18:35; the user: "K64 + 8x8 + 16x16 with 8 labels, can you run this for the 64x64?")

- The 32x32 model scaled to 64x64: K64 + **16x16 every 4 px + 32x32 every 8 px** (patch sizes doubled; 8x8 every 2 px
  at 64x64 ran at 5 steps/s, about 3 days of training, 11-17), 32 states each, from slice 55; with and without
  `--cond 8 --cond-on global,patch`. Reference: K1536 + 16x16 every 4 px (58.46 after the quench; floor 5.91).
  - Couplings per slice: global 0.52 M, 16x16 set 2.77 M, 32x32 set 1.64 M; label 8 x (64 + 194 x 32) = 0.05 M.
  - Pipelines **4274099** (no label) and **4274100** (label 8), sampling `--chunk 100`; quench 4274101 / 4274102;
    Inception **4274103**; contrast 4274104; gallery **4274105**
    (`galleries/2026-11-18_celeba64-grey_K64+patch16x16-every4px+patch32x32-every8px_with-and-without-8-cluster-label_vs_K1536+patch16x16_noise-start_quench1/`).
  - Expected: about 6-8 h training (K1536 + 16x16 ran 66-69 steps/s in the patch slices), then sampling; results
    tomorrow morning.
- **Result (2026-11-19 08:25):** training + sampling 6.9 h (no label) / 7.3 h (label 8); quench 4274101 / 4274102;
  Inception **4274103**; contrast 4274104; gallery **4274105**. CelebA-64, n = 2000, noise start; floor 5.91;
  real: contrast 0.234, dark 13.5%:

  | Model | couplings per slice | as sampled | quench 1 (FID_inf) | KID x1e3 | contrast q1 | dark q1 |
  |---|---|---|---|---|---|---|
  | K1536 + 16x16 (11-17) | 15.4 M | | 58.46 (55.62) | 53.7 | 0.180 | 3.3% |
  | K64 + 16x16 + 32x32 | 4.93 M | 130.9 | 59.25 (56.98) | 54.4 | 0.175 | 2.7% |
  | **K64 + 16x16 + 32x32 + 8-cluster label** | 4.98 M | 127.8 | **57.77 (55.04)** | 52.8 | 0.177 | 3.2% |

  - **The small model with an 8-cluster label is the best 64x64 model so far** (57.77), 0.7 better than K1536 + 16x16
    with a third of the couplings. Without the label the small model is 0.8 worse than K1536.
  - **The label gains 1.5 at 64x64** (59.25 -> 57.77; bootstrap sd 0.3-0.4), somewhat more than at 32x32 (1.0).
  - Contrast stays at the 64x64 deficit (0.175-0.180 vs 0.234; dark 2.7-3.3% vs 13.5%).
  - Gallery: all three models give the same style (smooth, mid-grey, blurry); no broken faces.
  - Clusters: 13325-28803 images each (8 clusters, seed 31001).

### Performance overview figure as a 2 x 2 grid (2026-11-19 08:45; the user: "make it 2x2 plot: MNIST, fashion MNIST, 32x32 Celeb and 64x64 celeb")

- `plot_inception_fid.py` now draws MNIST, Fashion-MNIST, CelebA 32x32 and CelebA 64x64 in a 2 x 2 grid
  (`out/latent_physical_SGM/figures/inception_fid_mnist_fashion.{png,pdf}`, same file name; backup
  `plot_inception_fid_pre_2x2.py`; plot jobs 4280009 / 4280033 / 4280039).
  - CelebA 32x32 updated: K784 / K3072 without patches; K784 / K1536 + 8x8; K784 / K1536 + 8x8 + 16x16; K784 +
    8x8 + 4x4 + 16x16; K1536 + 8x8 + 16x16 with 2 corrector steps (best, 21.5); K784 + 8x8 + 16x16 fully on-device
    (23.6); K64 + 8x8 + 16x16 (24.6) and with the 8-cluster label (23.6). Dropped: K3072 + 8x8 and K784 + 8x8 + 4x4
    (intermediate).
  - CelebA 64x64 (new panel): K1536 / K3072 + 16x16; K64 + 16x16 + 32x32 with and without the 8-cluster label
    (best, 57.8).
  - The as-sampled FID of the K64 + 8-cluster-label model at 32x32 was missing; Inception **4280007**: 93.93 as
    sampled, 23.57 after the quench (as before).
  - Layout: the same row spacing in both panels of a grid row (short panels keep their bars at the top), reference
    lines over the rows only; footnote with the patch layouts, the corrector, on-device training, the label and the
    1%-noise floor.

### Where contrast is lost: earlier coarse patches, the reverse trajectory, and the forward marginal (2026-11-19 09:10; the user: "yes please. do both. but also: at 2.2 how does the distribution look?")

- **Code: `train_latent.py --patch2-on-slice K2`** (with `--patch2`, K2 >= `--patch-on-slice`): the second patch set's
  own switch-on slice (its start reference at that slice's t); default -1 = with the first set, as before.
  Recorded in config.json as `patch2_on_slice` only when it differs; resume checks it. Backup
  `train_latent_pre_patch2on.py`. Test **4280068**: regression without the option (old vs new code, 3 slices,
  bitwise); 16x16 on slice 1 and 8x8 on slice 3, 5 slices with a resume in between.
- **New tool `forward_marginal_check.py`** (CPU): Part A, the data only: the SNR of x_t per principal component,
  the number of components still above the noise, the non-Gaussianity (excess kurtosis, KL to the moment-matched
  Gaussian) of the leading components, and how many training faces a noised test face is still consistent with
  (effective number of the posterior over 20000 training images); example noised faces. Part B (`--snapshots`):
  the model's signal variance along the principal components at the snapshot times, against the forward
  process's.
  - Part A: job **4280069** (`galleries/2026-11-19_celeba32-grey_forward-marginal_how-much-face-is-left-at-time-t/`).
  - Snapshots of the K784 + 8x8 + 16x16 model (n = 1000, 11 times): **4280070**; Part B: **4280071**
    (`galleries/2026-11-19_celeba32-grey_K784+patch8x8+patch16x16_reverse-trajectory-vs-forward-marginal/`).
- Test 1 (submitted after 4280068 passes): K64 with the 16x16 set switched on at slice 40 (t = 2.98) and the 8x8
  set at slice 55 (t = 2.22), against K64 + 8x8 + 16x16 both at slice 55 (24.61).
- **Test 4280068:** (1) regression without the option, old vs new code, 3 slices: all slice arrays bitwise;
  (2) 16x16 switched on at slice 1 and 8x8 at slice 3 exactly as asked (|Wp| 0 / 0.131 / 0.131 / 0.131, |Wp2|
  0 until slice 3, then 0.223; config `patch2_on_slice: 3`). The job is marked FAILED only because my resume step
  changed `--max-slices` (4 -> 5), which the resume check refuses as a config change, as it should; the resume
  path with the second set not yet switched on is therefore untested (it matters only if a run restarts before
  its switch-on). Temporary runs deleted.
- **Test 1 submitted (09:20):** K64 + 16x16 every 4 px from slice 40 (t = 2.98) + 8x8 every 2 px from slice 55
  (t = 2.22): pipeline **4280302**, quench 4280303, Inception **4280304** (against both sets from t = 2.22),
  contrast 4280305, gallery **4280306**
  (`galleries/2026-11-19_celeba32-grey_K64+patch8x8+patch16x16_16x16-switched-on-at-t2.98-vs-t2.22_noise-start_quench1/`).
- **Part A result: the forward marginal** (job 4280069, 22 s; report and figures in its gallery folder).
  CelebA-32: 1024 components, total variance 874 (normalised units), leading eigenvalues 257, 88, 80, 54, 51.

  | t | noise SD | SNR of PC1 | components with SNR > 1 (> 10) | their share of the face variance | training faces consistent with a noised test face (median of 20000) | PC2 excess kurtosis (clean: +0.87) |
  |---|---|---|---|---|---|---|
  | 0.20 | 0.081 | 26125 | 768 (264) | 99.8% | 1.0 | +0.87 |
  | 0.71 | 0.123 | 4096 | 340 (83) | 98.5% | 1.0 | +0.87 |
  | 1.46 | 0.138 | 733 | 125 (23) | 94.9% | 1.0 | +0.86 |
  | **2.22** | 0.141 | **153** | **41 (7)** | **87.7%** | **1.0** (top-1 posterior 0.996) | **+0.84** |
  | 2.98 | 0.141 | 33 | 13 (3) | 76.3% | 2.3 | +0.73 |
  | 3.48 | 0.141 | 12 | 7 (1) | 68.1% | 14.6 | +0.56 |
  | 3.99 | 0.141 | 4.4 | 3 (0) | 48.6% | 281 | +0.30 |
  | 5.00 | 0.141 | 0.6 | 0 | 0% | 9050 | +0.03 |

  - **At t = 2.2 the distribution is far from Gaussian.** 41 components are still above the noise and carry 88% of
    the face variance; along the leading components the distribution is exactly as non-Gaussian as the clean
    faces (kurtosis +0.84 vs +0.87); and a noised held-out face is still consistent with essentially one training
    face out of 20000 (posterior 0.996 on the nearest). Only the fine detail (components beyond about 40) is
    drowned. It becomes Gaussian-like only at t >= 4.
  - **So when the patches switch on at 2.2, the global stage must already have produced the whole coarse face**, at
    the level of an individual face (41 components). A mixture of K global templates (K = 64 ... 3072, against
    162770 faces) cannot represent that: its force there is an average over templates, which shrinks towards the
    mean. This fits the contrast loss forming at t = 3 -> 2.2, and the saturation in K.
  - The window t = 3 -> 2.2 is exactly where the data go from "2 faces consistent" to "1" and from 13 to 41
    components above the noise, and where the oracle starts gain their contrast (dark 8.2% -> 11.3%).
  - Implication: something richer than global templates must act above t = 2.2: local units earlier (test 1), or
    a structure that composes the coarse face (a hierarchy) rather than choosing one template.
- **Switch-on sweep of the coarse set (09:35; the user: "maybe we directly run a little sweep of turn on times"):**
  K64 + 16x16 every 4 px from slice k + 8x8 every 2 px from slice 55 (t = 2.22), seed 1, 2000 samples, quench 1:
  - k = 10 (t = 4.49): pipeline **4280385**, quench 4280386; k = 20 (t = 3.99): **4280387** / 4280388;
    k = 30 (t = 3.48): **4280389** / 4280390; k = 48 (t = 2.58): **4280391** / 4280392; k = 40 (t = 2.98):
    4280302 / 4280303 (above); k = 55 (t = 2.22): the existing K64 + 8x8 + 16x16 run (24.61).
  - Inception **4280393** (all six), contrast 4280394, gallery **4280395**
    (`galleries/2026-11-19_celeba32-grey_K64+patch8x8-from-t2.22+patch16x16_switch-on-sweep_t4.49-to-t2.22_noise-start_quench1/`).
  - The risk at the early end: on MNIST, patches switched on at t = 3.4 broke shapes (2026-11-13).
- **The 8x8 set earlier too (09:45; the user: "maybe also submit two jobs where even the 8x8 turn on earlier"):** with
  16x16 from slice 40 (t = 2.98), 8x8 from slice 48 (t = 2.58) and from slice 40 (t = 2.98, both sets together); with
  the running 8x8-from-2.22 run a three-point scan at fixed 16x16 time. 8x8 from slice 48: pipeline **4280421**,
  quench 4280422; from slice 40: pipeline **4280423**, quench 4280424 (Inception **4280425**, contrast 4280426, gallery **4280427**,
  `galleries/2026-11-19_celeba32-grey_K64+patch16x16-from-t2.98+patch8x8-from-t2.22-vs-t2.58-vs-t2.98_noise-start_quench1/`).
- **Results (2026-11-19 16:30).**
  - **Part B, the model's trajectory** (snapshots 4280070, 26 min; analysis 4280071; K784 + 8x8 + 16x16 with
    both sets from t = 2.22; 1000 samples). Signal variance along the principal components, model over data
    (1 = as the forward process):

    | t | PC 1 | PC 2-5 | PC 6-20 | PC 21-100 |
    |---|---|---|---|---|
    | 4.5 | 0.67 | 0.75 | 0.68 | 1.10 |
    | 3.5 | 0.84 | 0.82 | 0.60 | 1.09 |
    | 3.0 | 0.85 | 0.78 | **0.56** | 1.30 |
    | 2.5 | 0.89 | 0.78 | **0.55** | 1.26 |
    | 2.0 | 0.91 | 0.79 | 0.57 | 1.01 |
    | 1.0 | 0.91 | 0.79 | 0.59 | 0.80 |
    | 0 | 0.93 | 0.80 | **0.59** | 0.74 |

    - **Components 6-20 lose about 40% of their variance between t = 4.5 and 3, and never recover** (0.59 at t = 0);
      components 2-5 sit at about 0.8 throughout. The coarse structure the samples lack is fixed in the global stage,
      as the oracle starts said. Components 21-100 overshoot at t = 3-2.5 and end at 0.74.
    - Leading-component kurtosis is too low at t = 0 (PC2 +0.62 vs +0.87, PC4 +0.19 vs +0.69): too few extreme
      images (dark backgrounds, strong lighting).
    - (Components beyond 100 carry almost no signal above t = 2, so their ratio is a ratio of tiny numbers there and
      not meaningful; at t = 5 every ratio is noise.)
  - **16x16 switch-on sweep** (pipelines 4280302 / 4280385 / 4280387 / 4280389 / 4280391, 1.6 h each; Inception
    **4280393**; contrast 4280394; gallery **4280395**). K64 + 8x8 from t = 2.22, quench 1; real: contrast 0.229, dark
    12.7%, bright 3.3%:

    | 16x16 on at t | FID (FID_inf) | KID x1e3 | contrast | dark | bright |
    |---|---|---|---|---|---|
    | 2.22 (baseline) | 24.61 (20.78) | 16.9 | 0.193 | 6.1% | 1.1% |
    | 2.58 | 23.68 (20.10) | 15.8 | 0.194 | 6.4% | 1.2% |
    | 2.98 | 22.91 (19.43) | 14.6 | 0.199 | 7.4% | 1.5% |
    | 3.48 | 21.98 (18.44) | 13.8 | 0.207 | 8.6% | 2.1% |
    | 3.99 | 21.89 (18.45) | 14.2 | 0.212 | 9.1% | 2.4% |
    | **4.49** | **21.72 (17.86)** | 13.9 | **0.213** | **9.3%** | 2.4% |

    - **Switching the coarse 16x16 set on earlier gains up to 2.9 FID and a large part of the contrast**: contrast
      0.193 -> 0.213 (real 0.229), dark pixels 6.1% -> 9.3% (real 12.7%). The gain grows down to t = 3.5 and
      saturates there (3.99 and 4.49 within the noise of 3.48).
    - **K64 + early 16x16 (21.72) beats K1536 + 8x8 + 16x16 (22.37)** and comes within 0.2 of the best so far
      (K1536 + 8x8 + 16x16 with the corrector, 21.50), with 1.23 M couplings per slice.
    - Gallery: coherent faces, no broken shapes (unlike MNIST at t = 3.4); darker hair and backgrounds, more
      lighting contrast; the same people in most columns.
  - **8x8 earlier as well** (pipelines 4280421 / 4280423; Inception **4280425**; contrast 4280426; gallery 4280427),
    16x16 from t = 2.98: 8x8 from t = 2.22 / 2.58 / 2.98 -> 22.91 / 23.08 / 23.10 (contrast 0.199 / 0.200 / 0.199):
    **no gain** from early fine patches. Only the coarse set matters early, as the forward marginal suggests (early,
    only coarse components carry signal).
  - **Conclusion:** the contrast and coarse-structure deficit is a capacity gap of the global stage (t > 2.2), and
    local coarse units acting there close much of it. Next: combine (early 16x16 + 8-cluster label + corrector),
    seeds, snapshots of the early model (does PC 6-20 recover?), and the 64x64 version (32x32 set early).

### Next steps after the early-coarse-patch result (2026-11-19 16:50; the user: "yes please do all of the next steps. in particular combining the gains")

- **Code: the corrector with labels.** `sample_latent.sample_corrector(..., es=)` holds each sample's one-hot label in
  the predictor segments and the corrector steps; the sampler no longer refuses `--corrector` for a --cond run, and
  the corrector's grad_ode gets the label parameters. Unlabelled runs keep the exact code path. Backup
  `sample_latent_pre_condcorr.py`. Test **4282746**: (1) corrector without labels, old vs new code, 16 samples,
  bitwise; (2) corrector with labels, 16 samples + quench: finite, the same labels as the plain sampler. The
  corrector jobs below start only after it passes.
- **(A) Combining the gains**, CelebA-32, 16x16 from slice 10 (t = 4.49), 8x8 from slice 55, seed 1:
  - K64 + 8-cluster label: pipeline **4282748**, quench 4282749, corrector + quench **4282750**;
  - K784 + 8-cluster label: pipeline **4282751**, quench 4282752, corrector + quench **4282753**;
  - K64 without label (the 21.72 run): corrector + quench **4282754**;
  - Inception **4282767** (with K64 all-from-2.22 and the previous best, K1536 + 8x8 + 16x16 + corrector 21.50),
    contrast 4282768, gallery **4282769**
    (`galleries/2026-11-19_celeba32-grey_combined-gains_early16x16-from-t4.49_8-cluster-label_corrector_K64-and-K784_noise-start_quench1/`).
- **(B) Seeds 2 and 3** of K64 + early 16x16: pipelines **4282755** / **4282757**, quench 4282756 / 4282758; Inception
  of seeds 1-3 **4282770**.
- **(C) Snapshots** (1000 samples, 11 times) and the trajectory analysis for K64 with 16x16 from t = 2.22 (sampling
  4282759, analysis 4282760) and from t = 4.49 (4282761 / 4282762): do components 6-20 recover with early coarse
  patches?
- **(D) CelebA-64:** K64 + 32x32 every 8 px from slice 10 + 16x16 every 4 px from slice 55, without (pipeline
  **4282763**, quench 4282764) and with the 8-cluster label (**4282765** / 4282766); Inception **4282771** (with K1536 +
  16x16 58.46 and K64 + 16x16 + 32x32 + label from t = 2.22, 57.77), contrast 4282772, gallery **4282773**. About 7 h.
- 27 jobs in the queue at 16:50; disk 67%.
- **Interim results (2026-11-19 22:10).**
  - **Corrector-with-labels test 4282746 FAILED its regression check only:** the unlabelled corrector gave non-bitwise
    samples old vs new code (size not recorded; the test deleted its files). The unlabelled path is the same code
    in both versions (the new branch runs only with labels), so the likely cause is run-to-run nondeterminism of the
    GPU sampler (XLA's per-process GEMM algorithm choice, amplified by the adaptive solver). The labelled corrector
    worked (finite, same labels as the plain sampler, rms 0.88 from the plain trajectories). The blocked corrector
    jobs 4282750 / 4282753 / 4282754 were released by hand (22:00). **Reproducibility test 4284079:** the same
    command twice, plain and corrector, 16 samples, with the size of the difference.
  - **Seeds** (Inception **4282770**), K64 + 16x16 from t = 4.49: seed 1 / 2 / 3 = 21.72 / 22.07 / 21.80, **mean 21.86 ±
    0.15** (sd over seeds), against 24.61 with both sets from t = 2.22 (seed 1). The gain (about 2.8) is robust.
  - **Trajectories with early coarse patches** (snapshots 4282759 / 4282761, analyses 4282760 / 4282762; K64; signal
    variance over data):

    | t | PC 2-5: 16x16 from 2.22 / from 4.49 | PC 6-20: from 2.22 / from 4.49 | PC 21-100: from 2.22 / from 4.49 |
    |---|---|---|---|
    | 4.0 | 0.80 / 0.90 | 0.51 / 0.88 | 1.09 / 0.65 |
    | 3.0 | 0.74 / 0.95 | **0.44 / 0.97** | 1.50 / 1.01 |
    | 2.0 | 0.72 / 0.94 | 0.47 / 0.92 | 1.13 / 0.91 |
    | 0 | **0.72 / 0.95** | **0.51 / 0.89** | 0.80 / 0.81 |

    - **Early coarse patches restore the coarse structure almost fully**: components 6-20 end at 0.89 of the data's
      variance instead of 0.51, components 2-5 at 0.95 instead of 0.72. The trajectory confirms the mechanism.
    - Remaining: the leading components are still too light-tailed (PC2 kurtosis +0.35 vs +0.87, PC4 +0.26 vs +0.69):
      too few extreme images; and components 21-100 end at 0.81 in both.
  - **Combined, plain sampler** (interim Inception **4284080**; quench 1):

    | Model | as sampled | quench 1 (FID_inf) |
    |---|---|---|
    | K64 + early 16x16 | | 21.72 (17.95) |
    | K64 + early 16x16 + 8-cluster label | 85.4 | 21.35 (17.93) |
    | **K784 + early 16x16 + 8-cluster label** | 83.6 | **21.06 (17.31)** |

    - **K784 + early 16x16 + label (21.06) is the new best without a corrector**, below the previous best with one
      (K1536 + 8x8 + 16x16 + corrector, 21.50). The label adds only 0.4 on top of early patches (1.0 without
      them): both act on the global stage, so their gains overlap. K784 adds 0.3 over K64.
    - The corrector versions (4282750 / 4282753 / 4282754) and the full comparison 4282767 follow.
  - CelebA-64 pipelines 4282763 / 4282765 running (4.6 h / 2.9 h at 22:00).
- **Reproducibility test 4284079 (4 min):** the same `sample_latent.py` command twice on the same node (K64 + early
  16x16, 16 samples): **neither the plain sampler nor the corrector is bitwise reproducible run to run**, but the
  differences are rounding-sized: max |diff| per sample median 6.0e-6 (plain) / 1.1e-5 (corrector), largest 6.5e-5 /
  4.1e-5, rms 2e-6, against a median nearest-neighbour distance between samples of 27 / 24 (relative 2e-6, the same
  size as the 11-17 solver-tolerance check). So the failed regression of 4282746 was this nondeterminism, not the
  code change; the corrector-with-labels code stands. Regression tests of the sampler must compare with a tolerance
  (about 1e-4 absolute), not bitwise.

### Overnight 2026-11-19/20: how simple can the model be? (submitted 22:40; the user: "I would like to see how much simpler we can make those models and still perform similarly. to reduce the number of parameters")

- Starting point: the best small recipe, K64 + 16x16 every 4 px from t = 4.49 + 8x8 every 2 px from t = 2.22, 32 states
  per patch unit, patch Q on: **1.23 M couplings per slice** (global 0.13 M, 8x8 set 0.69 M, 16x16 set 0.41 M);
  21.72 (seed 1), 21.86 ± 0.15 over 3 seeds.
- **Code: `train_latent.py --no-patch-quad`** (with --quad and --patch): the patch units couple to the pixels
  linearly only (no Qp, half the patch couplings); the global unit keeps Q. Config `patch_quad: false` only when set;
  resume checks it. Backup `train_latent_pre_nopatchquad.py`. Test **4284150**: regression without the flag (old vs
  new, 3 slices, bitwise); 100 short slices with the flag (no Qp in the history, switch-on slices right), 16 samples
  and a quench.
- **Sweep** (seed 1, noise start, quench 1; couplings per slice):

  | variant | couplings | pipeline / quench |
  |---|---|---|
  | 16 states per patch unit | 0.68 M | 4284152 / 4284153 |
  | 8 states per patch unit | 0.41 M | 4284154 / 4284155 |
  | 8x8 every 4 px (49 instead of 169 units) | 0.74 M | 4284156 / 4284157 |
  | 8x8 every 4 px + 16x16 every 8 px (9 units) | 0.48 M | 4284158 / 4284159 |
  | early 16x16 only, no 8x8 set | 0.54 M | 4284160 / 4284161 |
  | K16 global states | 1.13 M | 4284162 / 4284163 |
  | no patch Q (after the test) | 0.68 M | 4284164 / 4284165 |
  | K16 + 8x8 every 4 px + 16x16 every 8 px, 16 states, no patch Q (after the test) | **0.12 M** | 4284166 / 4284167 |

  - Inception **4284168** (with the 1.23 M baseline), contrast 4284169, gallery **4284170**
    (`galleries/2026-11-19_celeba32-grey_parameter-reduction-sweep_from-K64+early16x16+8x8_1.23M-down-to-0.12M_noise-start_quench1/`).
- Still running from before: the corrector samplings 4282750 / 4282753 / 4282754 (combined comparison 4282767) and
  CelebA-64 with early 32x32 (4282763 / 4282765; Inception 4282771).
- **Time-basis compression (23:00; the user: "can we implement the time basis functions to reduce parameters?").**
  Context: our models store one parameter set per slice (100), so the stored total is 100 x the per-slice count
  (1.23 M -> 123 M for the K64 recipe; a CIFAR-scale DDPM has about 36 M; Wang & Achour 326k).
  - **New tool `compress_history.py`** (CPU): replaces each parameter group's history (over its active slices; patch
    sets stay zero before their switch-on) by `knots:B` (B kept slices, linear interpolation in t; stored B x P) or
    `svd:R` (mean + rank-R expansion sum_r a_r(t) V_r: R coupling patterns driven by R scalar control signals; stored
    (R + 1) x P + R x slices), and writes a new run directory that the sampler reads unchanged
    (`<run>_time-<mode>`, config `time_compression`, report.txt with per-group errors and stored counts). Post-hoc,
    no retraining; training with the basis built in comes only if this shows how far one can go.
  - Job **4284210**: check (knots:all reproduces the history exactly, svd:full to rounding), then knots 34 / 17 / 9 /
    5 and svd 16 / 8 / 4 / 2 / 1 of the K64 + early 16x16 run (21.72). Sampling + quench **4284211-4284219**;
    Inception **4284220**; gallery **4284221**
    (`galleries/2026-11-19_celeba32-grey_K64+early16x16_time-basis-compression_knots-and-low-rank_noise-start_quench1/`).
- **Compression job 4284210 done (3 min):** check PASSED (knots:all reproduces the history exactly; svd:full to 4.9e-13).
  The stored total of the original is **83.2 M** (not 123 M: the 16x16 set exists for 90 slices, the 8x8 set for 45);
  1.26 M per slice. Stored after compression: knots 34 / 17 / 9 / 5 -> 42.7 / 21.4 / 11.3 / 6.3 M; rank 16 / 8 / 4 / 2
  / 1 -> 21.4 / 11.3 / 6.3 / 3.8 / 2.5 M.
  - Reconstruction errors (relative, per group): with knots the per-pixel units (wl, bl) are not representable
    (error > 1 at 9 knots: their histories jump from slice to slice); the couplings lose 6-20% at 9 knots (W 10%, Q
    20%, Wp 11%, Qp 19%, Wp2 6%). Rank 4 does better at the same storage for wl / bl (13%) and theta (2%), similar
    for the couplings (W 12%, Q 21%, Wp 12%, Qp 19%, Wp2 8%).
  - Sampling of the 9 compressed runs (4284211-4284219) and Inception 4284220 running.
- **compress_history.py extended (23:05; backup `compress_history_v1.py`):** `knotsopt:B` (B kept slices chosen
  greedily where the interpolation error is largest), `hat:B` / `cheb:B` (least-squares fits with B piecewise-linear
  hats on uniform knots in t, or B Chebyshev polynomials; least squares also averages slice-to-slice jitter),
  `--group-modes` (per-group overrides, e.g. `wl=svd:4,bl=svd:4`), `--dry-run` (error report only). Dry-run
  comparison of all bases at 5 / 9 / 17 stored sets: **4284247**.
- **Test 4284150 PASSED (22 min):** `--no-patch-quad` - regression without the flag (old vs new code, 3 slices):
  all slice arrays bitwise; with the flag, 100 short slices: config `patch_quad: false`, history members without Qp /
  Qp2 (the global Q kept), 16x16 zero before slice 10 and 8x8 zero before slice 55, 16 samples and quench finite.
  Temporary runs deleted. The two sweep runs that use the flag (4284164, 4284166) are released (waiting for
  priority at 23:10).
- **Dry-run comparison 4284247** (relative reconstruction error per group at equal storage: 5 / 9 / 17 stored sets;
  svd:R stores R + 1 sets):

  | basis (9 sets) | theta | W | Q | wl | bl | Wp | Qp | Wp2 | Qp2 |
  |---|---|---|---|---|---|---|---|---|---|
  | knots:9 (even) | 0.062 | 0.099 | 0.201 | 1.210 | 1.180 | 0.110 | 0.187 | 0.061 | 0.099 |
  | knotsopt:9 (greedy) | 0.041 | 0.105 | 0.193 | 0.121 | 0.101 | 0.097 | 0.171 | 0.062 | 0.099 |
  | hat:9 (least squares) | 0.036 | 0.073 | 0.156 | 0.646 | 0.611 | 0.082 | 0.138 | 0.042 | 0.072 |
  | cheb:9 (least squares) | 0.034 | 0.079 | 0.150 | 0.369 | 0.355 | 0.098 | 0.165 | 0.038 | 0.071 |
  | **svd:8** | **0.012** | **0.061** | **0.111** | **0.044** | **0.038** | **0.060** | **0.106** | **0.036** | **0.062** |

  - **Low rank in time is the best basis at equal storage for every group**, at 5, 9 and 17 sets alike. Least-squares
    fits beat interpolation for the couplings; greedy knots fix the per-pixel units but not the couplings.
  - The per-pixel units (wl, bl) are noise-like in time: from t ~ 4 on about half of the 1024 pixels' wl change sign
    from one slice to the next (30266 sign flips over 101376 steps; relative slice-to-slice change 0.52, the same
    after a joint sign fix of (w, c), so not a gauge flip but w wandering around 0). Smooth bases cannot follow that;
    the rank expansion averages it.
  - Next (after the FIDs of round 1): per-group ranks (cheap groups at higher rank, the big coupling groups at the
    lowest rank that keeps the FID).
- **Sweep results (2026-11-20 00:40; Inception 4284168, contrast 4284169, gallery 4284170; all 8 pipelines done
  without errors, 37-96 min each):**

  | variant | couplings per slice | Inception FID (quench 1) | change |
  |---|---|---|---|
  | baseline K64 + early 16x16 + 8x8, 32 states, patch Q | 1.23 M | **21.72** | |
  | K16 global states | 1.13 M | **22.11** | +0.4 |
  | 16 states per patch unit | 0.68 M | 25.75 | +4.0 |
  | 8x8 every 4 px | 0.74 M | 28.89 | +7.2 |
  | no patch Q | 0.68 M | 29.85 | +8.1 |
  | 8x8 every 4 px + 16x16 every 8 px | 0.48 M | 33.44 | +11.7 |
  | 8 states per patch unit | 0.41 M | 34.42 | +12.7 |
  | early 16x16 only, no 8x8 set | 0.54 M | 46.69 | +25.0 |
  | K16 + sparse patches + 16 states + no patch Q | 0.12 M | 66.10 | +44.4 |

  - **The global unit can shrink (K64 -> K16 costs 0.4), the patch units cannot.** But the global unit holds only
    0.13 M of the 1.23 M, so K16 saves little. Every cut of the patch couplings costs at least 4 FID; at equal size
    (0.68 M) halving the states (25.75) is much better than dropping the patch Q (29.85). The 8x8 set is essential
    (without it 46.69).
  - **Contrast is unchanged in every variant** (image std 0.212-0.214, dark pixels 8.4-9.4%, as the baseline): the
    early 16x16 set fixes contrast at any size; the cuts cost local detail.
  - Gallery (same noise per column): every variant draws the same face per column (pose, hair, lighting set early).
    With 8 states or sparse patches the texture turns blotchy and blocky (patch-sized tiles, smeared eyes and mouths);
    0.12 M gives grainy, softer faces that are still faces.
  - So per slice the 32x32 recipe needs about 0.7-1.2 M couplings. The larger saving should come from time (100
    stored slices): see the time-basis compression below.

### Combined gains on CelebA-32: early 16x16 + label + corrector (results 2026-11-20 01:25)

Corrector samplings 4282750 (K64 + label8), 4282753 (K784 + label8), 4282754 (K64, no label); Inception
**4282767**, contrast 4282768, gallery **4282769**
(`galleries/2026-11-19_celeba32-grey_combined-gains_early16x16-from-t4.49_8-cluster-label_corrector_K64-and-K784_noise-start_quench1/`).
All with 16x16 patches from t = 4.49 and 8x8 from t = 2.22; 2 corrector steps per slice where marked; quench 1.

| model | plain | 2 corrector steps |
|---|---|---|
| K64, both patch sets from t = 2.22 (reference) | 24.61 | |
| K64 + early 16x16 | 21.72 | **20.73** |
| K64 + early 16x16 + label8 | 21.35 | **20.71** |
| K784 + early 16x16 + label8 | 21.06 | **20.57** (new best at 32x32) |
| K1536 + 8x8 + 16x16 (previous best, patches from t = 2.22) | | 21.50 |

- **The gains do not add up fully.** The corrector gives -1.0 on K64 + early 16x16, but on top of it the label adds
  nothing (20.73 -> 20.71) and K784 instead of K64 only -0.14 (within the spread of 0.2-0.5). The early 16x16 set,
  the label and the corrector all fix partly the same thing: the contrast loss during the global stage.
- **The small K64 model with corrector (20.73, 1.23 M couplings per slice) beats the previous best K1536 with
  corrector (21.50).**
- Contrast: corrector + early 16x16 reach image std 0.216-0.217 and 9.7-9.9% dark pixels (real 0.229 / 12.7%;
  previous best 0.200 / 7.5%).
- Gallery: the corrector rows draw different faces (the corrector uses its own noise), clean and well lit, fewer
  sunglasses-like dark blotches than the plain rows; K64 and K784 with corrector look alike.

### Time-basis compression, round 1: FIDs (2026-11-20 02:15; the user: "explore different time knotting methods")

- The first Inception job 4284220 never ran: the rank-2 sampling 4284218 saved its samples (1464 s) but its quench
  ran into the 2 h limit (the other quenches take 24-29 s). A quench-only retry (**4284715**, 40 min) hung the same
  way, so it is reproducible: the rank-2 t = 0 energy makes the quench ODE stiff (the adaptive solver stalls). Not
  chased; rank 2 is scored without quench. Inception redone without it: **4284716**, gallery **4284717** (same
  folder as above).

| compression | stored parameters (original 83.2 M) | Inception FID (quench 1) |
|---|---|---|
| none: 100 slices | 83.2 M | **21.72** |
| knots 34 (every 3rd slice) | 42.7 M | **21.71** |
| knots 17 | 21.4 M | 21.87 |
| knots 9 | 11.3 M | **22.35** |
| knots 5 | 6.3 M | 25.03 |
| rank 16 | 21.4 M | 21.87 |
| rank 8 | 11.3 M | **22.20** |
| rank 4 | 6.3 M | 26.08 |
| rank 2 | 3.8 M | (quench hangs; 100.2 without quench) |
| rank 1 | 2.5 M | 67.78 |

- **The time dependence compresses about 7x at almost no cost:** 9 stored sets (knots 9 or rank 8, 11.3 M stored)
  cost 0.5-0.6 FID; 17 sets cost 0.15; 34 sets nothing. Below 9 sets the FID rises fast (5 sets +3.3 to +4.4).
- **Knots do as well as low rank in FID, although their reconstruction error is far larger** (wl / bl errors > 1
  with knots; see the dry run). At 5 sets knots are better (25.03 vs 26.08). The reconstruction error is not the
  right measure: the per-pixel units' slice-to-slice jitter does not matter for the samples. For hardware, knots are
  simpler (B coupling sets switched in turn with linear crossfades, no control signals).
- Gallery: knots 34 / 17 / 9 and rank 16 / 8 are hard to tell from the original (same face per column, same
  detail); knots 5 softer and a little blotchy; rank 4 grainy; rank 1 changes the faces (darker, grainy, other
  identities); rank 2 unquenched shows the usual pixel speckle over intact faces.
- Next, round 2 (submitted 02:35): where to put the knots (equal steps of the parameters' path length instead of
  even in t), whether only the coupling matrices need few knots, whether the per-pixel units need time at all, and
  knots 9 on the smaller K16 and 16-state models.

#### Design proposal (not implemented, not run): training with the knots built in (2026-11-20 02:50)

Post-hoc knots lose 0.6 FID at 9 knots and 3.3 at 5 because the kept slices were trained to be good at their own t,
not to be good as the end points of a crossfade. Training the knots directly should recover most of that.
- **Parametrisation.** Knot times t_0 = 5 > t_1 > ... > t_B = 0, one parameter set theta_j per knot; on the slice
  grid theta(t_k) = (1 - w_k) theta_j + w_k theta_{j+1} for t_j >= t_k >= t_{j+1} (exactly what the sampler sees
  after compress_history.py). Patch sets keep their own switch-on slice as their first knot.
- **Sequential training, as now.** Training already runs t = 5 -> 0 with every slice warm-started from the previous
  one. With knots: when the walk enters the interval (t_j, t_{j+1}], theta_j is frozen (already trained) and only
  theta_{j+1} is trained, on the sum of the per-slice ISM losses of all slices in the interval at their interpolated
  parameters: grad theta_{j+1} = sum_k w_k grad L_k(theta(t_k)). Same number of slice visits and noise draws as now;
  only B + 1 parameter sets are kept.
- **On device:** the slice statistics are already accumulated per slice; with knots each slice's update is scaled by
  w_k and applied to the next knot's couplings, i.e. the hardware holds two coupling sets and crossfades between them.
- **Cheaper test first:** fine-tune after compression (start from the post-hoc knots and run the knot training above
  for a fraction of the steps). If that brings 5 knots to <= 22.4, the full version is worth implementing.
- Per-slice size and time compression multiply: K16 (1.13 M per slice) with 9 knots stores about 10 M, the 16-state
  model (0.68 M) about 6 M (both in round 2).

### CelebA-64: early 32x32 patches (results 2026-11-20 03:10)

Pipelines 4282763 (no label, 7.5 h) and 4282765 (label8, 7.8 h); quench 4282764 / 4282766; Inception **4282771**,
contrast 4282772, gallery **4282773**
(`galleries/2026-11-19_celeba64-grey_K64+patch32x32-from-t4.49+patch16x16-from-t2.22_with-and-without-label8_noise-start_quench1/`).
K64 + 32x32 patches every 8 px from t = 4.49 + 16x16 every 4 px from t = 2.22 (the 32x32 recipe scaled by 2).

| model (64x64, quench 1) | Inception FID | image std | dark pixels |
|---|---|---|---|
| real test images (floor 5.91) | | 0.234 | 13.5% |
| K1536 + 16x16 | 58.46 | 0.180 | 3.3% |
| K64 + 16x16 + 32x32 from t = 2.22 + label8 (previous best) | 57.77 | 0.177 | 3.2% |
| **K64 + 32x32 from t = 4.49 + 16x16 from t = 2.22** | **49.21** | **0.201** | **6.5%** |
| same + label8 | 49.25 | 0.200 | 6.7% |

- **Switching the coarse patches on early gives -8.6 FID at 64x64** (-2.9 at 32x32): the biggest single gain at
  64x64 so far, and with the small K64 global unit. Contrast recovers about half the gap (std 0.177 -> 0.201 of
  0.234; dark pixels 3.2% -> 6.5% of 13.5%).
- **The label adds nothing on top** (49.25 vs 49.21), as at 32x32 (where it added 0.4 without corrector, 0 with).
- Gallery: crisper faces with darker hair and backgrounds, more varied lighting, fewer washed-out grey faces than the
  previous best; some faces still distorted (eyes, mouths).
- Next candidates at 64x64: the corrector on this model (at 32x32 -1.0), and switching the 16x16 set on earlier too.
- **Submitted 03:25:** corrector (2 steps, snr 0.16) on K64 + early 32x32 at 64x64: sampling + quench **4285143**,
  Inception **4285144** (about 2.6 h + 10 min).
- **Overview figure update (03:20):** `plot_inception_fid.py` gets the early-patch rows (32x32: K64 + early 16x16,
  with corrector, K784 + label + corrector; 64x64: K64 + early 32x32, with label). Their as-sampled FIDs were never
  scored, so Inception **4285111** (32x32) and **4285112** (64x64) score raw and quenched sets; plot **4285132**
  after them. Footnote: "early" = coarsest set from t = 4.49. Backup `plot_inception_fid_pre_early.py`.

### Time-basis compression, round 2 (results 2026-11-20 03:20)

Compression + tests **4284925** (2 min; PASSED: new mode `knotsarc:B` reproduces the run exactly at B = all slices,
puts knots where a synthetic history moves, keeps a switched-on set zero before its switch-on; `svd:0` = time mean;
`knots:all` unchanged). Sampling 4284926-4284933; Inception **4284934**; gallery **4284935**
(`galleries/2026-11-20_celeba32-grey_K64+early16x16_time-basis-round2_even-vs-arclength-knots_small-models_noise-start_quench1/`).

| variant (K64 + early 16x16 unless noted) | stored | Inception FID (quench 1) |
|---|---|---|
| original, 100 slices | 83.2 M | 21.72 |
| knots 9, even in t | 11.3 M | 22.35 |
| knots 7, even | 8.8 M | 23.31 |
| knots 5, even | 6.3 M | 25.03 |
| knots 9 at equal path length (per group) | 11.3 M | **35.54** |
| knots 5 at equal path length (per group) | 6.3 M | **73.44** |
| knots 3 at equal path length (per group) | 3.8 M | 46.39 |
| knots 5 for the six coupling matrices only, the rest (theta, b, wl, bl, bp, bp2) at all 100 slices | 8.8 M | **28.37** |
| knots 9, per-pixel units (wl, bl) constant at their time mean | 11.1 M | **22.03** |
| K16, 100 slices | 74.6 M | 22.11 |
| K16, knots 9 | 10.3 M | 23.09 |
| 16 states per patch unit, 100 slices | 46.3 M | 25.75 |
| 16 states, knots 9 | 6.3 M | **25.80** |

- **Placing the knots by path length per group fails badly** (35.5 / 73.4 / 46.4; not even monotone in B), although
  its reconstruction errors are lower than with even knots. Gallery: grey, low-contrast faces with blotchy
  patch-sized tiles. With per-group path lengths each group gets its own knot times (e.g. at 5 knots W at t = 3.64
  / 2.68 / 1.52, b at 2.22 / 0.96 / 0.30, wl at 2.63 / 0.20 / 0.05), so at most t the sampler combines couplings
  interpolated between one pair of slices with biases interpolated between another.
- **The same with exact biases:** compressing only the coupling matrices (5 knots) and keeping theta, b, wl, bl, bp,
  bp2 exact is worse (28.37) than compressing everything at the same 5 knots (25.03).
- **Reading: the parameters of a slice are co-adapted** (e.g. the biases compensate the couplings' mean field); a
  compression must move all groups of a set together. Even knots did that by accident (every group of a set has the
  same active range, hence the same knots). This also explains why per-group low rank (independent errors per
  group) was no better than knots although its reconstruction errors were much lower.
- **The per-pixel units need no time dependence:** wl, bl fixed at their time mean improve knots 9 (22.35 -> 22.03).
  Their slice-to-slice jitter (half the pixels flip sign per slice) is noise that averaging removes.
- **The 16-state model compresses without loss** (25.75 -> 25.80 at 9 knots, 6.3 M stored): about the same as
  the 32-state model at 5 knots (25.03 at 6.3 M). K16 gains nothing over K64 when compressed (23.09 vs 22.35).
- Next (round 3): knots shared within each set (global / 16x16 / 8x8) placed by the set's joint path length, and
  one knot schedule for all sets; even knots with constant wl, bl at 5 and 7.
- **Overview figure updated (03:35; `out/latent_physical_SGM/figures/inception_fid_mnist_fashion.png/.pdf`, plot
  4285132, footnote line split in 4285195).** As sampled -> quenched (Inception 4285111 / 4285112): 32x32 K64 +
  early 16x16 86.3 -> 21.72; with 2 corrector steps 72.3 -> 20.73; K784 + early 16x16 + label + corrector 69.8 ->
  **20.57** (bottom row of the CelebA-32 panel); 64x64 K64 + early 32x32 120.0 -> **49.21**, with label 116.1 ->
  49.25 (bottom two rows of the CelebA-64 panel, 8-9 below the previous rows).

### Time-basis compression, round 3: one knot schedule per set (submitted 03:30)

- **Code: `compress_history.py --knot-share {group,set,all}`** (backup `compress_history_v3.py`): `set` chooses the
  kept slices once per parameter set (global = theta, W, b, wl, bl, Q, [Bc, hc]; each patch set = Wp / bp / Qp /
  Bpc with its suffix), for `knotsarc` from the set's joint path length (sum over its groups of the squared step
  lengths, switch-on steps excluded); `all` one schedule for every group (each patch set keeps its switch-on slice
  plus the schedule's later knots). Groups with a `--group-modes` override keep their own mode. Default `group` =
  round 1-2 behaviour.
- Job **4285185**: tests (default reproduces round-2 knotsarc:5 and knots:5 bitwise; set / all at full count
  reproduce the run exactly; within each set all groups get the same knot times) and 7 variants of the K64 run:
  knotsarc 5 / 9 shared per set, even knots 5 / 9 with one schedule for all, even knots 5 / 7 with wl, bl constant,
  knotsarc 5 shared per set with wl, bl constant. Sampling 4285186-4285192, Inception **4285193**, gallery
  **4285194**. 24 compressed variants on disk after this round (limit 30; disk 71%).
- **Round 3 results (04:15; tests PASSED in 4285185; Inception 4285193, gallery 4285194
  `galleries/2026-11-20_celeba32-grey_K64+early16x16_time-basis-round3_knots-shared-per-set-or-for-all_per-pixel-units-constant_noise-start_quench1/`):**

  | variant | stored | Inception FID (quench 1) |
  |---|---|---|
  | original, 100 slices | 83.2 M | 21.72 |
  | even knots per set 9 / 5 (rounds 1-2) | 11.3 / 6.3 M | 22.35 / 25.03 |
  | path-length knots per group 9 / 5 (round 2) | 11.3 / 6.3 M | 35.54 / 73.44 |
  | **path-length knots shared per set** 9 / 5 | 11.3 / 6.3 M | 25.88 / 31.70 |
  | path-length knots shared per set 5, wl / bl constant | 6.1 M | 31.93 |
  | even knots, one schedule for all sets 9 / 5 (8x8 set gets only 5 / 3 knots) | 8.5 / 4.9 M | 25.45 / 32.04 |
  | **even knots, wl / bl constant** 9 / 7 / 5 | 11.1 / 8.6 / 6.1 M | **22.03 / 22.75** / 25.51 |

  - **Sharing the knots within a set rescues the path-length placement** (35.5 -> 25.9 at 9, 73.4 -> 31.7 at 5):
    the co-adaptation reading holds. But it is still worse than even knots by 3.5-6.7.
  - **Why even beats path length:** the joint path length puts the knots at the switch-on slices (the global set
    gets knots at t = 4.49 and 2.22: it jumps when a patch set switches on) and densely at small t (patch sets:
    0.35 / 0.15 / 0.05), leaving t ~ 1-3.5 with few knots. The parameters move most at small t, but there the
    errors hardly matter (the quench at the end relaxes the sample under the t = 0 energy anyway), while the face is
    shaped at t ~ 2-4 (the global stage and the early 16x16 set). Knots belong where errors cost samples, not where
    parameters move.
  - One schedule for all sets is worse, but confounded: it gives the 8x8 set fewer knots (and wastes one in the
    16x16 set: knots at 4.49 and 4.39).
  - Constant wl / bl: 9 knots 22.35 -> 22.03, 7 knots 23.31 -> 22.75, 5 knots 25.03 -> 25.51 (spread 0.2-0.4): no
    loss at 7-9 knots; keep them constant (one fewer time-dependent group, 2 k numbers instead of 2 k per knot).
  - **Best so far: even knots per set with wl / bl constant: 9 knots 22.03 (11.1 M stored, 7.5x smaller), 7 knots
    22.75 (8.6 M, 9.7x).**
- Next (round 4, 04:20): unequal knot budgets per set (the global set is cheap: 0.15 M per knot; 16x16: 0.41 M;
  8x8: 0.70 M): global 17 + patches 5; global 5 + patches 9; 16x16 9 + 8x8 5; 16x16 5 + 8x8 9 (wl / bl constant).
  - Gallery round 3: the worse variants (one schedule for all, path length per set) are visibly greyer, lower in
    contrast, with the same faces per column; even knots with wl / bl constant at 9 and 7 keep the original's
    contrast. So too few knots at t ~ 2-4 cost contrast, the same deficit the early 16x16 set had fixed.
- Round 4 submitted 04:25: compression **4285308**, sampling 4285309-4285312, Inception **4285313**, gallery
  **4285314** (`galleries/2026-11-20_celeba32-grey_K64+early16x16_time-basis-round4_unequal-knot-budgets-per-set_per-pixel-units-constant_noise-start_quench1/`).
  28 compressed variants after this round (limit 30).
- **Round 4 results (05:10; Inception 4285313, gallery 4285314):** even knots, wl / bl constant, knots per set
  (global = theta, W, b, Q / 16x16 set / 8x8 set):

  | knots global / 16x16 / 8x8 | stored | Inception FID (quench 1) |
  |---|---|---|
  | 100 slices (original) | 83.2 M | 21.72 |
  | 9 / 9 / 9 | 11.1 M | 22.03 |
  | **5 / 9 / 9** | **10.7 M** | **21.67** |
  | 7 / 7 / 7 | 8.6 M | 22.75 |
  | 9 / 9 / 5 | 8.5 M | 23.83 |
  | 9 / 5 / 9 | 9.7 M | 26.41 |
  | 5 / 5 / 5 | 6.1 M | 25.51 |
  | 17 / 5 / 5 | 8.0 M | **29.77** |

  - **The global set needs only 5 knots:** 5 / 9 / 9 equals the original (21.67 vs 21.72) at 7.8x fewer stored
    numbers. More global knots do not help and can hurt: 17 / 5 / 5 is far worse than 5 / 5 / 5 (29.77 vs 25.51).
    The global set's fine time structure is co-adapted with the patch sets' fine structure; once the patches are
    smoothed, a detailed global history no longer matches them. Matched (or coarser) resolution is better.
  - **Of the patch sets, the early 16x16 set needs the knots:** 16x16 at 5 costs +4.4 (26.41), 8x8 at 5 +1.8
    (23.83). The 16x16 set acts at t = 4.49-2.2, where the faces are shaped.
- **Round 5 submitted 05:15** (last 2 variants; 30 in total): (a) the global set as a linear ramp (2 knots: t = 5
  and 0) with patches at 9, 32x32; (b) the 5 / 9 / 9 recipe unchanged on the 64x64 early-32x32 model (49.21).
  Compression **4285406**; sampling 4285407 (32x32) / 4285408 (64x64, ~2.5 h); Inception **4285409** / **4285410**;
  galleries 4285411 / 4285412.
  - Gallery round 4: 5 / 9 / 9 cannot be told from the original (same faces, same contrast); 17 / 5 / 5 and
    9 / 5 / 9 are greyer and flatter, 9 / 9 / 5 in between. Again, the losses show as lost contrast.
- **64x64 corrector 4285143 failed (03:20):** out of host memory at 48 G after 3 min (64x64 sampling and quench need
  ~55-60 G: the quench-only job 4282764 peaked at 53 G of 96 G). Resubmitted with 96 G: **4285413**; Inception
  4285144 now waits on it. The compressed 64x64 sampling 4285408 raised to 96 G before it started.
- **Round 5, 32x32 (06:00; Inception 4285409, gallery 4285411
  `galleries/2026-11-20_celeba32-grey_K64+early16x16_time-basis-round5_global-set-5-or-2-knots_patch-sets-9_noise-start_quench1/`):**
  the global set as a linear ramp (2 knots, t = 5 and 0; patches 9; 10.3 M stored) fails completely: **241.2**
  (5 knots: 21.67). Gallery: every sample is the same saturated black-and-white face silhouette (white face, black
  hair outline), the template without contrast control. The global set's history is strongly nonlinear in t; it
  needs 3-5 knots (3 not tested), 5 suffice.
- **Round 5, 64x64, and the 64x64 corrector (07:50; Inception 4285410 and 4285144):** both on CelebA-64 K64 +
  32x32 from t = 4.49 + 16x16 from t = 2.22 (100 slices: 49.21).

  | 64x64 variant | stored | Inception FID train (sd) | FID test | KID x1e3 |
  |---|---|---|---|---|
  | 100 slices, quench 1 | 331.8 M (3.32 M / slice) | 49.21 (0.61) | 48.10 | 42.8 |
  | **knots 5 / 9 / 9, wl / bl constant, quench 1** | **42.7 M** (0.129) | **47.99** (0.42) | 47.21 | 41.8 |
  | 100 slices, 2 corrector steps (snr 0.16), no quench | 331.8 M | 98.88 (0.79) | 100.03 | 127.6 |
  | **100 slices, 2 corrector steps, quench 1** | 331.8 M | **43.10** (0.33) | 41.95 | 32.8 |

  - **The 32x32 compression recipe carries over to 64x64 unchanged and even helps:** 7.8x fewer stored numbers,
    FID 49.21 -> 47.99 (2 sd). Knots: global set at t = 5.00 / 3.74 / 2.47 / 1.26 / 0; 32x32 set at 9 even
    knots from t = 4.49; 16x16 set at 9 from t = 2.22. Relative reconstruction errors are larger than at 32x32
    (W 0.22, Q 0.27) and still cost nothing, so the slice-to-slice detail of the global set is noise from training.
  - **The corrector gives -6.1 at 64x64** (49.21 -> 43.10, new best at 64x64); at 32x32 it gave -1.0
    (21.72 -> 20.73). Without the quench it doubles the FID (the Langevin noise of the last step stays in the
    image); the quench removes it.
  - Not yet run: corrector on the compressed 64x64 history (expected ~42-43, at 42.7 M stored).
  - The gallery 4285412 ran out of memory at 16 G (CPU wrapper default). Resubmitted with 48 G as **4285857**,
    with the two corrector rows added:
    `galleries/2026-11-20_celeba64-grey_K64+early32x32_time-basis-5-9-9-knots_and_2-corrector-steps_noise-start_quench1/`.
- **Submitted 07:55: corrector on the compressed 5 / 9 / 9 histories** (both gains together, no new compressed
  variant): sampling 32x32 **4285859**, 64x64 **4285860** (96 G, ~2.7 h); Inception **4285861** / **4285862**
  (100 slices vs knots, each with and without corrector); galleries **4285863** / **4285864**
  (`galleries/2026-11-20_celeba{32,64}-grey_..._time-basis-5-9-9-knots_with-and-without-2-corrector-steps_noise-start_quench1/`).
- **Overview figure (08:11, job 4285865):** the 64x64 panel gets "K64 + early 32x32 + 16x16, 2 corrector steps"
  (98.9 -> 43.1, best at 64x64). `out/latent_physical_SGM/figures/inception_fid_mnist_fashion.png/.pdf`, checked.
- **Gallery 4285857 (08:20, 22 G peak;
  `galleries/2026-11-20_celeba64-grey_K64+early32x32_time-basis-5-9-9-knots_and_2-corrector-steps_noise-start_quench1/`):**
  the compressed 5 / 9 / 9 row shows the same faces as the 100-slice row, column by column, at the same contrast.
  Both are mostly bright, smiling, light-haired faces on light backgrounds. The corrector changes the population:
  after the quench there are many more men, dark hair, dark backgrounds, sunglasses, so the spread of brightness is
  much closer to the real row (that is where its -6 FID comes from); some faces are distorted. Without the quench
  the corrector rows carry visible grain and are darker. (The gallery title is cut at the left edge; the full text
  is in README.txt.)

### Morning summary, 2026-11-20 (overnight 2026-11-19/20)

All CelebA numbers: Inception FID vs train, n = 2000, quench 1 (floor 6.43 at 32x32, 5.91 at 64x64).

**1. Best models now**

| | model | FID | was (evening 11-19) |
|---|---|---|---|
| CelebA-32 | K784 + 16x16 from t = 4.49 + 8x8 from t = 2.22 + 8-cluster label, 2 corrector steps | **20.57** | 21.50 (K1536 + corrector) |
| CelebA-32 | K64 + early 16x16 + 8x8, 2 corrector steps (1.23 M couplings per slice) | 20.73 | |
| CelebA-64 | **K64 + 32x32 from t = 4.49 + 16x16 from t = 2.22, 2 corrector steps** | **43.10** | 57.8 |
| CelebA-64 | the same without corrector | 49.21 | |

The early coarse set is the big step at 64x64 (57.8 -> 49.2; contrast 0.177 -> 0.201, real 0.234), and the
corrector adds much more there than at 32x32 (-6.1 vs -1.0): its samples are far more varied (men, dark hair, dark
backgrounds), closer to the real brightness spread. The 8-cluster label adds nothing once the coarse set is
early. The overview figure has all of these.

**2. Fewer parameters per slice: no free lunch.** Retrained variants of K64 + early 16x16 (21.72): only the global
unit shrinks cheaply (K16: 22.11). Every cut in the patch units costs 4-25 FID (16 states 25.75, no patch Q 29.85,
sparser patches 28.9-33.4, everything reduced 66). The patch units are the model.

**3. Fewer parameters in time: 7.8x for free.** `compress_history.py` (post-hoc, no retraining; 30 variants in 5
rounds) keeps a few slices per parameter set ("knots") and interpolates linearly in t, as the sampler already does.
- Recipe: even knots in t, **global set (theta, W, b, Q) 5 knots, each patch set 9, per-pixel units wl / bl
  constant** (`--mode knots:9 --group-modes theta=knots:5,W=knots:5,b=knots:5,Q=knots:5,wl=svd:0,bl=svd:0`).
- CelebA-32: 83.2 M -> **10.7 M stored, 21.72 -> 21.67**. CelebA-64, recipe unchanged: 331.8 M -> **42.7 M,
  49.21 -> 47.99**. The galleries cannot tell the compressed models from the originals.
- Rules: (a) the groups of a set must share their knots (the parameters of a slice are co-adapted; per-group
  placement gives 35-73); (b) even spacing in t beats placement by path length (the parameters move most at small
  t, where the quench repairs errors; faces and contrast form at t ~ 2-4); (c) the global set needs 3-5 knots, 2
  (a linear ramp) collapses (241), more global knots with coarse patches hurt; (d) the early coarse set needs the most
  knots; (e) the reconstruction error does not predict the FID: losses show up as lost contrast.
- Hardware reading: per patch set a handful of coupling sets crossfaded linearly in t; the per-pixel units fixed.

**4. Running** (submitted 07:55): the corrector on the compressed 5 / 9 / 9 models, 32x32 (4285859 -> Inception
4285861) and 64x64 (4285860 -> 4285862, ~2.7 h, ready ~11:00). If it holds, the best 64x64 model stores 42.7 M.

**5. Suggested next steps**
- Train with the knots built in (design: "Time-basis compression, round 1"): the slices between knots share
  parameters during training, so the knots are fitted rather than picked; this could also allow fewer knots.
- The global set at 3 knots (not tested; 5 is safe, 2 fails).
- At 64x64: an earlier 16x16 set (the coarsest set early was the big gain twice).
- Not pursued: shrinking the patch units per slice (every cut tested costs >= 4 FID).

### Training directly at a few time knots (submitted 2026-11-20 ~09:00; the user: "just directly train at the fewer time steps")

- **Code:** `train_latent.py --time-knots B|list`, backup `../latent_physical_SGM_pre_ising_patch/train_latent_pre_knots.py`.
  - Only the knots' parameters exist; slice k between knots uses the linear crossfade, which is the sampler's own
    interpolation.
  - Slice 0 trains as before.
  - Each interval (k_j, k_j+1]: P_j frozen; P_j+1 (all sets, started from P_j) trained jointly on the interval's slices.
    100-step chunks at one slice each, shuffled cycles, the slice's learning rates, gradient through the crossfade;
    total steps = the slices' usual sum; lr decay stretched over the interval; result = EMA.
  - At a switch-on knot (10, 55) only the new patch set trains, alone.
  - Slices saved as usual, so sampling and evaluation are unchanged. Resume restarts at the last knot.
  - `compress_history.py --knot-slices` (backup `compress_history_v4.py`): the post-hoc counterpart at the same knots.
- **Test job 4285924** must pass before anything trains:
  - (0) knot placement;
  - (1) old vs new code without the option, bitwise;
  - (2) 100-slice knot run: slices = exact crossfades, switch-on knots;
  - (3) killed-and-resumed run = uninterrupted, bitwise;
  - (4) sampling.

  The trainings wait on it with afterok: **if 4285924 failed, cancel 4285927 / 4285929 / 4285931**.
- **Knots** (slice indices; 0 = t 5, 99 = t 0; switch-on slices fixed, the rest spread to minimise the largest gap):

  | knots | slices | stored (approx.) |
  |---|---|---|
  | 9 | 0,10,21,32,44,55,70,84,99 | ~7.4 M |
  | 6 | 0,10,32,55,77,99 | ~5.0 M |
  | 5 | 0,10,32,55,99 | ~3.8 M; the 8x8 set a single linear ramp |

  Stored for the 100-slice model: 83.2 M (21.72); post-hoc 5 / 9 / 9 recipe: 10.7 M (21.67).
- **Jobs:**
  - pipelines (train + 2000 samples + quench 1) **4285927** (9), **4285929** (6), **4285931** (5);
  - post-hoc counterparts: compression **4285926** (tests itself first), sampling 4285928 / 4285930 / 4285932;
  - Inception **4285934** and gallery **4285935** over whatever exists
    (`galleries/2026-11-20_celeba32-grey_K64+early16x16_trained-at-time-knots-9-6-5_vs_posthoc-at-same-knots_noise-start_quench1/`).
  - Run names: `celeba32g_K64_ema_..._patch16s4S32on10+patch8s2S32on55_knots{9,6,5}_seed1` and
    `<100-slice run>_time-posthoc-knots_<slices>`.
  - Scripts: scratchpad `compress_knots_counterpart.sh`, `eval_knots.sh`, `test_knots.sh`.
- **Also still running:** the corrector on the compressed 5 / 9 / 9 models; Inception **4285861** (32x32) and
  **4285862** (64x64, ~11:00).
- **Test 4285924 FAILED, because of the test, not the code (found 11:10).**
  - Test 2 treated every group that is all zero at the start knot and non-zero at the end knot as a set switching
    on there, and required zeros in between. Q is exactly zero at slice 0 (hidden units frozen at slice 0), so the
    crossfade of Q over slices 1-9 failed that check.
  - Re-checked on the saved run (short srun): every group, Q included, equals the crossfade to 1.4e-15; the patch
    sets are zero before their switch-on.
  - The other checks passed: (0) knot placement; (1) old vs new code bitwise; (2) switch-on knots train only the new
    set; (4) sampling finite.
  - Test (3) proved nothing: the killed run had saved no slices after 100 s.
  - Released the three trainings at 11:12 (`scontrol update Dependency=`). They have run since about 11:12.
  - Proper resume test, mid-interval (slices 40-99 deleted from a copy, resume at 33, compare bitwise):
    **4286222**.
- **Corrector on the compressed 5 / 9 / 9 histories (Inception 4285861 / 4285862, galleries 4285863 / 4285864):**

  | | 100 slices | 5 / 9 / 9 knots | 100 slices + corrector | 5 / 9 / 9 + corrector |
  |---|---|---|---|---|
  | CelebA-32 (83.2 M / 10.7 M stored) | 21.72 | 21.67 | 20.73 | **24.69** (worse) |
  | CelebA-64 (331.8 M / 42.7 M stored) | 49.21 | 47.99 | 43.10 | **40.88** (new best at 64x64) |

  - Opposite signs at the two sizes. Galleries: with the corrector, the compressed models draw the same faces as
    the 100-slice ones, but brighter: whiter backgrounds, harsher dark features. The share of near-black pixels
    drops: 32x32 4.2% vs 9.8% (real 11.8); 64x64 3.9% vs 9.0% (real 12.6).
  - Reading: the corrector equilibrates at each t to what the parameters say, and between knots the interpolated
    parameters were never trained, so it amplifies the interpolation error. At 64x64, where the models lack
    contrast, the push helps; at 32x32 it overshoots. Trained knots should not have this problem: test the corrector
    on them too.
- **Overview figure updated (2026-11-20 ~11:45; plot jobs 4286280, then the replot).** New rows:
  - 32x32 and 64x64: "knots 5/9/9" and "knots 5/9/9 + corrector".
  - Their as-sampled FIDs had never been measured: Inception **4286278** (32x32) and **4286279** (64x64) score all
    four sets as sampled and quenched.
  - Labels shortened and the figure widened to 15.5 in: the right panels' axis labels had been cut off. Checked.
    Backup `plot_inception_fid_pre_knots.py`.

  | set | as sampled | quench 1 |
  |---|---|---|
  | 32x32 100 slices | 86.3 | 21.72 |
  | 32x32 knots 5/9/9 | **53.7** | 21.67 |
  | 32x32 100 slices + corrector | 72.3 | 20.73 |
  | 32x32 knots 5/9/9 + corrector | **42.8** | 24.69 |
  | 64x64 100 slices | 120.0 | 49.21 |
  | 64x64 knots 5/9/9 | 103.9 | 47.99 |
  | 64x64 100 slices + corrector | 98.9 | 43.10 |
  | 64x64 knots 5/9/9 + corrector | 89.8 | **40.88** |

  - **New observation: the compressed models are much better before the quench** (32x32: 86 -> 54; with corrector
    72 -> 43). As-sampled FID is dominated by the grain the reverse SDE leaves at t = 0. A guess, not tested: the
    last slices before t = 0 are the least converged (slice 99 still improves by ~600 loss units over the second
    half of its steps). Their slice-to-slice noise is averaged away by the knots, and the per-pixel units are held
    constant. The quench removes the grain either way, so the quenched FIDs are close.

### Model size, round 2: K32, and fewer patch states at 64x64 (submitted 2026-11-20 ~12:00; the user: "I want to understand how much smaller we can make this")

- Already known at 32x32 (overnight sweep, early 16x16, quench 1): K64 21.72, K16 22.11, 16 patch states 25.75,
  8 patch states 34.42. New runs (seed 1, best setup = early coarse patches, plain sampler + quench 1, no corrector):

  | size | run | pipeline (train + 2000 samples + quench) |
  |---|---|---|
  | 32x32 | K32, 32 patch states | **4286481** |
  | 32x32 | K32, 16 patch states | **4286482** |
  | 64x64 | K32, 32 patch states | **4286483** (~7.5 h, 96 G) |
  | 64x64 | K16, 32 patch states | **4286484** |
  | 64x64 | K64, 16 patch states | **4286485** |
  | 64x64 | K64, 8 patch states | **4286486** |

- Evaluation over the sets that exist, including the earlier runs (scratchpad `eval_size2.sh`):
  - Inception: **4286487** (32x32) / **4286489** (64x64);
  - galleries: 4286488 / 4286490
    (`galleries/2026-11-20_celeba{32,64}-grey_model-size_K64-K32-K16_patch-states-32-16-8_..._noise-start_quench1/`).

### Step-wise constant parameters instead of linear interpolation (submitted 2026-11-20 12:45; the user: "instead of using a linear interpolator, use just step wise constant parameters ... in practice this would be easier")

- Hardware motivation: B coupling sets switched B times (program, hold, switch: a staircase annealing schedule)
  instead of a crossfade, which needs two banks with complementary gains or continuous reprogramming.
- Expected: a step is a zeroth-order approximation of theta(t), error ~ dt |d theta/dt|; linear is first order,
  error ~ dt^2 |d^2 theta/dt^2|. So steps need more pieces for the same accuracy, and every switch kicks the samples
  off the new distribution.
- **Code:** `compress_history.py --mode steps:B` (backup `compress_history_v5.py`). The same B knot slices as
  knots:B. The first active slice is its own piece; each later piece (k_j-1, k_j] is held at the mean of its slices
  (the least-squares constant). In the 100-slice format the sampler switches between pieces over one slice interval
  (dt = 0.05): a fast switch.
- **Jobs:**
  - compression **4286502**: tests steps:3 on synthetic histories, a set switched on mid-way, and steps:100
    reproducing the run exactly; then steps 5 / 9 / 17 / 34 and the 5 / 9 / 9 per-set recipe with wl / bl
    constant;
  - sampling 4286503-4286507 (after the knot pipelines, to keep about 10 GPUs);
  - Inception **4286508**, gallery **4286509**
    (`galleries/2026-11-20_celeba32-grey_K64+early16x16_time-basis_linear-knots-vs-step-constant-pieces_noise-start_quench1/`).
- Linear counterparts (re-scored in the same job): knots 34 / 17 / 9 / 5 = 21.71 / 21.87 / 22.35 / 25.03;
  5 / 9 / 9 recipe = 21.67.
- **Compression 4286502 done (12:36):** tests passed:
  - steps:3 pieces on synthetic data;
  - a set switched on mid-way;
  - steps:100 and the run's own steps:100 reproduce exactly.

  Stored: steps 5 / 9 / 17 / 34 = 6.3 / 11.3 / 21.4 / 42.7 M; 5 / 9 / 9 recipe 10.7 M (the same as linear).
- **Trained step version:** `train_latent.py --knot-interp step` (with --time-knots; backup
  `train_latent_pre_knotinterp.py`).
  - Every slice of an interval (k_j, k_j+1] uses the knot P_j+1 itself (w = 1), so each piece is one parameter set
    trained on the plain losses of its slices.
  - Recorded in config.json as 'knot_interp' only for step. The linear path differs by two conditionals.
- **Test 4286513:**
  - (0) step without --time-knots is refused;
  - (1) linear knot run, old vs new code, bitwise over 100 slices;
  - (2) step run: every slice equals its piece, the pieces differ;
  - (3) sampling.
- **Pipelines** (after the test with afterok, and after the post-hoc step samplings to keep about 10 GPUs):
  **4286514** (9 pieces), **4286515** (6), **4286516** (5). Run names `..._knots{9,6,5}step_seed1`.
- **Evaluation:** Inception **4286517**, gallery **4286518**: trained linear vs trained step at the same knots
  (`galleries/2026-11-20_celeba32-grey_K64+early16x16_trained-at-time-knots-9-6-5_linear-vs-step-constant_noise-start_quench1/`).
- Resume test of the linear knots, second try **4286495**. The first try 4286222 failed in the test itself: the
  copied run kept its name in config.json, which the config check refuses.

### Results (2026-11-20 evening): trained knots, step-constant pieces, tests

- **Tests:**
  - **Resume test 4286495 PASSED:** resumed at slice 33 after slices 40-99 were deleted, bitwise equal to the
    uninterrupted run.
  - **Step test 4286513:** tests 1-3 PASSED (linear path old vs new bitwise over 100 slices; every step slice equals
    its piece; sampling). Test 0 "failed" in the test itself: under `pipefail` the intended exit code 1 of the
    refused run failed the pipe although grep found the message. A direct check (srun) shows the refusal works.
  - Trained step pipelines released at 18:20 (`scontrol update Dependency=`).
- **Eval-script bug:** `eval_size2.sh` overwrote its own mode argument with `set --` in a loop, so Inception 4286487
  made the 32x32 gallery instead of scoring. Fixed (MODE saved first). 32x32 Inception resubmitted as **4287589**;
  the queued 64x64 Inception 4286489 was cancelled and resubmitted as **4287590**. Gallery 4286490 was not affected.
- **Trained knots vs post hoc at the same knots (Inception 4285934, gallery 4285935).** One grid for all sets with
  the switch-on slices forced in, so per set: 9 -> 9/8/4, 6 -> 6/5/3, 5 -> 5/4/2.

  | knots (per set) | stored | trained | post hoc, same knots | near-black px % (trained / post hoc; real 11.8) |
  |---|---|---|---|---|
  | 100 slices | 83.2 M | 21.72 | - | 9.2 |
  | 5/9/9 per set + wl/bl constant (post hoc) | 10.7 M | - | **21.67** | 5.9 |
  | 9 (9/8/4) | 7.4 M | **24.83** | 29.21 | 10.8 / 1.8 |
  | 6 (6/5/3) | 5.0 M | **30.23** | 42.40 | 9.1 / 0.1 |
  | 5 (5/4/2) | 3.8 M | 109.1 | 86.7 | 4.6 / 0.0 |

  - **Training the knots keeps the contrast; post hoc at the same knots loses it.**
    - Gallery: trained 9 and trained 6 show the same faces as the 100-slice model at full contrast.
    - Post hoc 9 is greyer, post hoc 6 washed out, post hoc 5 a uniform grey mean face.
    - Training gains 4.4 FID at 9 knots and 12 at 6.
  - **The shared grid is the limit, not the training.** These grids give the 8x8 set only 4 / 3 / 2 knots.
    - Post hoc, 9 even knots per group score 22.35; the shared grid at 9 scores 29.21.
    - Trained 5 (8x8 set a single ramp from t = 2.22 to 0) keeps faces and contrast but is covered in blotchy
      texture artifacts. FID 109 ranks it below the grey blobs of post hoc 5 (86.7).
  - Next for trained knots: give the 8x8 set enough knots. Either a shared grid denser at small t (global knots
    are cheap, 147 k each against 698 k for an 8x8 knot), or per-set grids, which need intervals that do not line
    up (more code).
- **Step-constant vs linear, post hoc (Inception 4286508, gallery 4286509):**

  | sets (B) | stored | linear | step | step - linear |
  |---|---|---|---|---|
  | 34 | 42.7 M | 21.71 | 22.36 | +0.6 |
  | 17 | 21.4 M | 21.87 | 23.49 | +1.6 |
  | 9 | 11.3 M | 22.35 | 26.96 | +4.6 |
  | 5 | 6.3 M | 25.03 | 37.84 | +12.8 |
  | 5/9/9 per set, wl/bl constant | 10.7 M | 21.67 | 33.22 | +11.6 |

  - As expected for a zeroth-order approximation: **steps need about 4x the sets of linear for the same FID** (step
    34 = linear 9: 22.36 vs 22.35). The near-black share falls with fewer pieces (step 5: 0.0%), again lost contrast.
  - Trained step pieces (pipelines 4286514-4286516, Inception 4286517) will show how much of this training recovers.

### Trained knots with one knot grid per parameter set (submitted 2026-11-20 ~19:00; the user: "yes" to per-set grids)

- **Code:** `train_latent.py --time-knots-sets global=B1,patch1=B2,patch2=B3` (backup
  `train_latent_pre_knotsets.py`; also with `--knot-interp step`).
  - Each set's knots are spaced evenly over its own range, the same knots as compress_history knots:B per group:
    - global (theta, W, b, Q, wl, bl) 5: slices 0 25 50 74 99;
    - 16x16 9: 10 21 32 43 54 66 77 88 99;
    - 8x8 9: 55 61 66 71 77 83 88 93 99.
  - The union of the knots cuts t into segments. In each segment, every set's open knot (the next after its last
    finished one) trains jointly through the crossfades. Training runs on all slices from the start of the
    earliest open interval to the segment end; slices behind a set's finished knots enter with that set fixed. Each
    knot thus trains over its whole interval, and the intervals need not line up.
  - Steps per segment = the usual steps of its new slices. A patch set's first knot trains alone at its switch-on
    slice. Slices are written after all knots are trained, so a resumed run retrains from slice 1.
  - The runner gets a per-group base + w (P - anchor) (`knot="perset"`). The shared-grid path only gained a
    `base=None` argument.
- **Test 4287622:**
  - (1) old vs new code bitwise for a shared --time-knots run and a plain run;
  - (2) per-set linear: every slice equals its set's crossfade (< 1e-12), with the expected knots; per-set step:
    every slice equals its piece exactly; patch sets zero before switch-on; slice 60's knots recorded;
  - (3) a run cut at slice 40 retrains from slice 1, bitwise equal;
  - (4) sampling.
- **Trainings** (afterok on the test):
  - linear 5/9/9 **4287623** (~10.7 M; post-hoc 21.67 with wl/bl constant);
  - linear 3/5/5 **4287624** (~6.0 M);
  - step 5/9/9 **4287625** (post-hoc step 33.22);
  - step 9/17/17 **4287626** (~20 M).
  - Run names `..._knots-g5-p9-p9[step]_seed1` etc.
- **Evaluation** with the post-hoc references: Inception **4287627**, gallery **4287628**
  (`galleries/2026-11-20_celeba32-grey_K64+early16x16_trained-per-set-time-knots_linear-and-step_vs_posthoc_noise-start_quench1/`).
- **Test 4287622 PASSED (19:15):**
  - (1) shared --time-knots and plain runs, old vs new code bitwise;
  - (2) per-set linear slices = crossfades to 2.8e-15; step slices = pieces exactly; patch sets zero before
    switch-on; slice 60 between global 50-74, 16x16 54-66, 8x8 55-61;
  - (3) a run cut at slice 40 resumes at slice 1 and ends bitwise equal;
  - (4) sampling.
  - The four per-set trainings 4287623-4287626 are now queued for a GPU.
- **Model size, 32x32 (Inception 4287589; gallery 4286487/4286488):** all with early 16x16 + 8x8, seed 1, quench 1.

  | global states K | patch states S | Inception FID |
  |---|---|---|
  | 64 | 32 | 21.72 |
  | **32** | 32 | **21.70** |
  | 16 | 32 | 22.11 |
  | 64 | 16 | 25.75 |
  | **32** | 16 | **26.26** |
  | 64 | 8 | 34.42 |

  - **K32 = K64** (21.70 vs 21.72), and K16 costs only +0.4: the global unit is not where the model's capacity lies.
  - Halving the patch states costs +4 at any K, a quarter +12.7. The patch units carry the model, so further
    reductions belong in time (knots) rather than in the patch states.

### K16 with 64 patch states, and no Q couplings (submitted 2026-11-20 ~19:30; the user: "run 16 global and 64 for the patches ... set Q = 0 again with the new setup. what is the price we pay?")

- Both at 32x32 with early 16x16 + 8x8, seed 1, quench 1.
  - **K16, 64 patch states:** pipeline **4287719**. About 2.4 M couplings per slice, against 1.23 M for K64 with 32
    states.
  - **No Q anywhere** (K64, 32 states, without --quad, so no global Q and no patch Qp): pipeline **4287720**. About
    0.61 M per slice. Known already: patch Q off, global Q kept: 29.85 (+8.1); at equal size, 16 states with Q:
    25.75.
- Inception **4287721** and gallery **4287722** with K64 / K32 / K16 at 32 states and the patch-Q-off run as references
  (`galleries/2026-11-20_celeba32-grey_early16x16+8x8_K16-with-64-patch-states_and_without-Q-couplings_noise-start_quench1/`).
- **Model size, 64x64 (Inception 4287590; gallery 4286490).** All with 32x32 patches from t = 4.49 + 16x16 from
  t = 2.22, seed 1, quench 1, no corrector.

  | global states K | patch states S | Inception FID |
  |---|---|---|
  | 64 | 32 | 49.21 |
  | 32 | 32 | 49.31 |
  | **16** | 32 | **48.63** |
  | 64 | 16 | 57.67 |
  | 64 | 8 | 66.38 |

  - The same picture as at 32x32. **K16 = K32 = K64** within about 1 sd (48.63 / 49.31 / 49.21), so the global unit
    can go to K16.
  - The patch states cannot be cut: 16 states +8.5, 8 states +17 (at 32x32: +4.0 / +12.7).

### Locked-in small models trained on device at 9 step-constant knots per set (submitted 2026-11-20 ~21:00; the user: "lock in K16, 32 patches ... train the two best small models on device at few knots. maybe do 9/9/9 ... the staircase time evolution ... is more relevant")

- **Models:** K16 + early coarse patches, 32 states. 32x32: 16x16 from t = 4.49 + 8x8 from 2.22 (100 slices: 22.11).
  64x64: 32x32 from 4.49 + 16x16 from 2.22 (100 slices: 48.63).
- **Time:** `--time-knots-sets global=9,patch1=9,patch2=9 --knot-interp step`, i.e. 9 coupling arrays per set,
  switched as a staircase, each trained on the slices it covers. 9/9/9 instead of 5/9/9: global knots are cheap
  (147 k each vs 410 k / 698 k for a patch knot), so about 11.4 M stored instead of 10.7 M (32x32), with equal
  counts per set.
- **On device:** `--hidden-samples 64 --hidden-jackknife --hidden-norm occupancy --patch-occupancy --occ-same-draws`
  (as the 11-18 fully on-device K784 run: 23.59 vs 22.85 exact).
- **Code change:** the visit-count rule is now allowed with --time-knots / --time-knots-sets, with the visit counts
  measured on the parameters in use at the slice (crossfade or piece), not on the trained knot alone. Backup
  `train_latent_pre_knotocc.py`. **Test 4287881:**
  - old vs new code bitwise for a plain on-device run and an exact per-set step run;
  - on-device per-set step 9/9/9: every slice equals its piece;
  - on-device shared-grid linear: exact crossfade;
  - sampling.
- **Runs** (afterok on the test; exact-statistics twins to separate the on-device cost from the knot cost):

  | size | statistics | pipeline | time limit |
  |---|---|---|---|
  | 32x32 | on device | **4287882** | 1 day (K784 on device took 6.4 h) |
  | 32x32 | exact | **4287883** | |
  | 64x64 | on device | **4287884** | 3 days, 96 G (exact 64x64 trains ~5 h; on device several times slower) |
  | 64x64 | exact | **4287885** | 96 G |

  Run names `celeba{32,64}g_K16_..._[S64jk_occ-patchocc_]knots-g9-p9-p9step_seed1`.
- **Evaluation:** Inception **4287886** / **4287888**, galleries 4287887 / 4287889
  (`galleries/2026-11-20_celeba{32,64}-grey_K16_patch-states32_early-coarse-patches_on-device_step-9-9-9-time-knots_noise-start_quench1/`).
  References: K16 100 slices, and K16 post-hoc linear knots 9 (32x32: 23.09).
- Caveat: per-set runs write their slices only after all knots are trained, and a resubmitted run retrains from
  slice 1. Fine within the time limits, but a node failure in the 64x64 on-device run would cost the whole run.

### Probability-flow ODE, second try (submitted 2026-11-20 ~21:15; the user: "the flow ODE ... so much simpler hardware wise. are we sure that we integrate the flow ODE correctly with correct factor 2 in comparison with the SDE?")

- **Check of the factor 2: correct.**
  - Forward process (paper `sample_forward_process`, beta = sigma_final = 1, D = T = 0.02):
    dx = -x dt + sqrt(2T) dW.
  - The model's score is -grad F / T, so: reverse SDE dx = [x - 2 grad F] dtau + sqrt(2T) dW; PF-ODE
    dx/dtau = x - grad F.
  - `sample_latent.make_drift` uses `scale = 2` (SDE) / `1` (ODE) on every force term. The visible part is
    -grad E(x; scale theta - theta_lin), with E(x; theta_lin) = |x|^2 / 2 giving the +x. No diffusion for the ODE.
  - Gaussian check: v_t = e^{-2t} s0^2 + T (1 - e^{-2t}); the ODE drift x (1 - T / v_t) gives dv/dt = -2 (v - T),
    exactly the forward process. Tsit5 at rtol 1e-7: integration error negligible.
- **History:** the ODE was tried only on 11-15 (MNIST / Fashion K784 + patches, no quench): classifier FID +11.9 /
  +2.6 worse than the SDE, noisy broken shapes. Never on CelebA, never quenched, never with knots.
- **Why it can be worse with a correct drift:**
  - The SDE = ODE transport + Langevin at T, and the Langevin part keeps the samples on the model's p_t; the ODE
    accumulates every score error.
  - Our slices are fitted independently, so the family p_t is not exactly consistent with its own transport.
  - The flow expands from variance T = 0.02 to the data's ~1 (normalised pixels).
- **Runs** (32x32, noise start = the SDE runs' start states, `--sampler ode --quench 1`):

  | model | ODE sampling |
  |---|---|
  | K64 early, 100 slices | 4287931 |
  | K64 early, post-hoc 5/9/9 | 4287932 |
  | K16 early, 100 slices | 4287933 |
  | K64 trained 9 knots, linear | 4287934 |
  | K64 trained 9 pieces, step | 4287935 |

  The idea behind the knot rows: a smoother, jointly trained family may suit the ODE better.
- **Evaluation:** Inception **4287936** (SDE and ODE, each as sampled and quenched); gallery **4287937** (ODE q1 vs
  SDE q1 per model; `galleries/2026-11-20_celeba32-grey_early16x16+8x8_PF-ODE-vs-SDE_K64-K16-knots_noise-start_quench1/`).
- **Next (proposed, after these):** a force scale lambda for the ODE (x - lambda grad F, a sampling temperature);
  the reduced-noise family dx = [x - (1 + g^2) grad F] dtau + g sqrt(2T) dW, g = 0 / 0.25 / 0.5 / 1; an ODE
  predictor with corrector holds.
- **PF-ODE results (22:30; Inception 4287936, gallery 4287937).** ODE sampling takes 2-3 min per 2000 samples (SDE
  ~25 min).

  | model (32x32) | SDE | SDE + quench | ODE | ODE + quench | near-black % SDE q1 / ODE q1 (real 11.8) |
  |---|---|---|---|---|---|
  | K64 early, 100 slices | 86.3 | 21.72 | 121.0 | **25.22** | 9.2 / 6.6 |
  | K64 early, post-hoc 5/9/9 | 53.7 | 21.67 | 76.3 | 25.51 | 5.9 / 4.4 |
  | K16 early, 100 slices | 87.8 | 22.11 | 121.8 | 25.86 | 9.1 / 6.4 |
  | K64 trained 9 knots, linear | 97.8 | 24.83 | 130.5 | 30.25 | 10.8 / 7.8 |
  | K64 trained 9 pieces, step | 114.4 | 32.12 | 133.9 | 38.47 | 6.7 / 5.5 |

  - **With the quench the ODE is only +3.5 to +3.8 behind the SDE** for the 100-slice and post-hoc models. That is
    no longer "do not use": ODE + quench at 25.2 is about where the SDE models with late patches were (24.6-24.7).
  - **Gallery:** real faces, not the broken shapes of MNIST on 11-15 (which had no quench). The ODE is
    deterministic, so every model turns the same start into nearly the same face (one face per column across the
    ODE rows). Greyer and softer than the SDE: fewer deep blacks, mid-grey backgrounds, some smudged features. The
    near-black share is lower in every row (6.6 vs 9.2%).
  - **The knot hypothesis does not hold:** trained knots widen the gap (+5.4 linear, +6.4 step), and post-hoc
    knots do not narrow it. The ODE's deficit is mainly lost contrast, not inconsistency between slices.
  - Next (proposed): a force scale lambda > 1 for the ODE (colder transport against the grey shift); the
    reduced-noise family g = 0.25 / 0.5 (how much noise the correction needs); an ODE predictor with corrector
    holds.
- **Trained step pieces, shared grid (Inception 4286517, gallery 4286518):**

  | knots (shared grid) | trained linear | trained step | post-hoc step, even knots per group |
  |---|---|---|---|
  | 9 (9/8/4 per set) | 24.83 | 32.12 | 26.96 (9 per group) |
  | 6 (6/5/3) | 30.23 | 40.68 | - |
  | 5 (5/4/2) | 109.1 | 61.06 | 37.84 (5 per group) |

  Trained steps stay well behind trained linear (+7 at 9, +10 at 6), except at 5, where the linear 8x8 ramp
  collapses. The shared grid starves the 8x8 set (4 knots at "9"), so the per-set step runs (5/9/9, 9/17/17, and
  the K16 9/9/9 on-device runs) are the relevant test.
- **On-device knot test 4287881 PASSED:**
  - old vs new code bitwise (plain on-device run; exact per-set step run);
  - on-device per-set step 9/9/9: every slice equals its piece; global knots 0 12 25 37 50 62 74 87 99;
  - on-device shared linear: crossfade to 1.8e-15;
  - sampling.
  - The four K16 runs (4287882-4287885) are queued for GPUs.

### ODE temperature and the reduced-noise sampler family (submitted 2026-11-20 ~22:50; the user: "yes, implement both and test them")

- **Code** (`sample_latent.py`, backup `sample_latent_pre_noisescale.py`):
  - `--force-scale LAMBDA` now also with `--sampler ode` (drift x - LAMBDA grad F).
  - New `--noise-scale G` (default SDE sampler, not with --corrector, --fast or --cond): drift
    x - (1 + G^2) LAMBDA grad F, noise G sqrt(2T). Built from the PF-ODE drift (`reduced_noise`). G = 1 is the
    reverse SDE, G -> 0 the PF-ODE; for exact scores every G has the same marginals. Suffix `_ns<G>`.
  - The default paths call the same functions as before (make_drift gets the ODE drift only when G != 1).
- **Test 4288048:**
  - (1) exact Gaussian (d = 8, 4000 samples, s0^2 = 1.5): G = 0.25 / 0.5 / 1 and the ODE must reach the data
    variance within 4%; lambda = 1.2 must shrink it;
  - (2) on the K16 model: G = 1, lambda = 1 equals the reverse-SDE drift (rel. < 1e-10); ODE lambda scaling and
    G = 0.5 drift exact; diffusion G sqrt(2T);
  - (3) end to end on a scratch copy of the run (links): ODE at lambda 1.2 and G = 0.5, n 100 + quench, finite;
    refusals with --sampler ode and with --corrector.
- **Sweep on K16 early 32x32** (after the test; noise start = the same start states as the SDE/ODE samples so far):
  - ODE lambda 0.8 / 0.9 / 1.1 / 1.25 + quench;
  - SDE G = 0.25 / 0.5 + quench.
  - Sampling  4288049 4288050 4288051 4288052 4288053 4288054; Inception **4288055**, gallery **4288056**
    (`galleries/2026-11-20_celeba32-grey_K16_early16x16+8x8_SDE-reduced-noise-and-PF-ODE-force-scale_noise-start_quench1/`).
  - References: K16 SDE + quench 22.11, ODE + quench 25.86.
- **Test 4288048 (22:22):**
  - (1) PASSED. Exact Gaussian: final variance 1.515 / 1.515 / 1.519 at G = 0.25 / 0.5 / 1 and 1.517 for the ODE
    (target 1.5, errors 1.0-1.3%); ODE at lambda 1.2: 0.49 (colder, less expansion).
  - (2) PASSED. On the K16 model the G = 1, lambda = 1 drift equals the reverse-SDE drift to 2e-16; lambda and G
    scaling exact.
  - (3) failed in the test itself. The scratch run linked the model files by absolute host paths, which do not
    exist inside the container, so the sampler refused a run without DONE (fourth test-script bug today).
  - Rerun of part 3 with relative links: **4288058**. The sweep jobs 4288049-4288054 now wait on it (afterok).
- **Test part 3, rerun 4288058 (22:27):** end to end PASSED.
  - ODE at lambda 1.2 and G = 0.5, n 100, each + quench: files written, finite. Std 0.70 for the cold ODE against
    0.91 at G = 0.5: colder means less spread, as expected.
  - The two refusal checks "failed" in the test itself (pipefail again); checked directly by srun: both refused
    with the --noise-scale message, no files written.
  - The sweep jobs 4288049-4288054 were released at 22:35.

### Overnight plan 2026-11-20/21 (the user: "I let you work on this overnight")

- Collect and log as results arrive, looking at every gallery:
  - ODE / noise sweep on K16 (~01:00);
  - per-set trained knots on K64 (linear 5/9/9, 3/5/5; step 5/9/9, 9/17/17);
  - K16 with 64 patch states; no Q anywhere;
  - the K16 staircase runs (on device and exact, 32x32 by morning, 64x64 later).
- Follow-ups only within these questions:
  - if G = 0.25 comes close to the SDE, also G = 0.1;
  - if a force scale helps the ODE, its neighbours;
  - the best ODE / low-noise setting also on the K16 64x64 model (sampling only).
- Fix broken chains (test-script bugs, failed dependencies). No new research directions; code changes only to fix
  today's new options, each with a test; at most about 10 GPUs; no deletions.
- In the morning: a summary at the end of this notebook, MODEL_AND_CODE.md updated, the overview figure redrawn
  with the new rows.

### Results collected 2026-11-21 10:10 (the overnight runs; no overnight follow-up)

- **Note:** the overnight plan above was not carried out. My last turn ended without a watcher or wake-up armed,
  so the session was never woken; the jobs ran on their own and are evaluated here.
- **K16 with 64 patch states, no Q (Inception 4287721, gallery 4287722), 32x32 early, quench 1:**

  | model | couplings per slice | FID |
  |---|---|---|
  | K64, 32 states (reference) | 1.23 M | 21.72 |
  | K16, 32 states | 1.13 M | 22.11 |
  | **K16, 64 patch states** | ~2.4 M | **20.20** (best without corrector at 32x32) |
  | K64, patch Q off (global Q kept) | 0.68 M | 29.85 |
  | K64, no Q anywhere | ~0.61 M | 31.09 |

  - More patch states help (-1.9); the global unit stays small.
  - Q: the patch Q is worth +8.1, the global Q only +1.2 more. Gallery: same faces; without Q the texture is
    grainier and harsher.
- **Per-set trained knots, K64 (Inception 4287627, gallery 4287628):** trained linear 5/9/9 25.11 (post-hoc
  21.67); trained linear 3/5/5 31.76 (post-hoc 5/5/5: 25.51); trained step 5/9/9 31.28 (post-hoc step 33.22);
  trained step 9/17/17 25.59.
  - All knot-trained models sit about 3.5 or more behind the 100-slice model, whatever the grid (shared 9:
    24.83; per-set 5/9/9: 25.11; step 9/17/17: 25.59).
  - Per-set training does not beat post-hoc thinning at 5/9/9. Training only helped where post-hoc knots were
    badly placed (shared grid).
- **ODE temperature / reduced noise, K16 32x32 (Inception 4288055, gallery 4288056), all + quench:**
  - SDE G = 1: 22.11; G = 0.5: 24.19; G = 0.25: 24.94; ODE (G = 0): 25.86. The noise helps gradually: half the
    noise recovers half the gap.
  - ODE force scale: lambda 0.8 / 0.9 / 1.1 / 1.25 give 264 / 163 / 49.3 / 81.4. **The ODE is extremely sensitive
    to the force scale; lambda = 1 is far the best.** No temperature knob helps; the flow has to be exact.
- **K16 staircase 9/9/9 (per-set step), 32x32 (Inception 4287886, gallery 4287887):**

  | K16 32x32 | FID |
  |---|---|
  | 100 slices, exact | 22.11 |
  | post-hoc linear knots 9 | 23.09 |
  | trained step 9/9/9, exact (4287883, 1.6 h) | 25.88 |
  | trained step 9/9/9, **on device** (4287882, 5.4 h) | **42.61** |

  - Gallery: the exact staircase is close to 100 slices, slightly blockier. On device: the same faces, but blurred
    and flat in detail.
  - Per-slice validation loss against the 100-slice model:
    - exact staircase: +0 to +6 down to t = 3.5, +12 to +25 at t = 3-2.5, +265 at the 8x8 switch-on (t = 2.22),
      +50 to +60 down to t = 1, then **+379 at t = 0.45, +855 at 0.2, +12,092 at t = 0**. The last piece of each
      set cannot follow the parameters, which change fastest near t = 0.
    - on device: like exact down to t ~ 2, then increasingly worse: +192 (t = 1.46), +502 (0.96), +1,962 (0.45),
      +4,412 (0.2), +18,773 (0).
  - **Missing control:** K16 on device with 100 slices (no knots). Without it the on-device loss cannot be split
    into on-device statistics as such vs on-device together with knot training. (K784 on device, 100 slices, lost
    only 0.7 FID on 11-18.)
- **64x64 K16 staircase:** exact (4287885) done at 05:50, Inception **4289662** submitted now. On device
  (4287884): at segment 83-87 of 22 after 11.3 h, about 2-3 h of training left, then sampling (~3 h); Inception
  4287888 and gallery 4287889 follow automatically.
- **Is the on-device loss a knot problem?** Per-slice validation loss, on device minus exact:

  | t | K784, 100 slices (11-18; FID cost only +0.7) | K16, staircase 9/9/9 |
  |---|---|---|
  | 0.96 | +28 | +455 |
  | 0.45 | +93 | +1,583 |
  | 0.2 | +274 | +3,557 |
  | 0 | +1,004 | +6,681 |

  About 15x more loss with knots. Suspect: on-device statistics together with the knot training (visit counts and
  sampled losses while chunks cycle through the slices of a window), or K16 itself.
- **Submitted 10:40** (the user: "our whole punch line is on device training ... K16 with 64 patch states -> can we
  run this with on device training as well?"), both 32x32, 100 slices, on device:
  - K16 32 states: **4289675** (the control for the staircase);
  - K16 64 states: **4289676**.
  - Inception **4289677**, gallery **4289678**
    (`galleries/2026-11-21_celeba32-grey_K16_32-and-64-patch-states_on-device-vs-exact_100-slices-vs-step-9-9-9_noise-start_quench1/`).
- **64x64 K16 exact staircase 9/9/9 (Inception 4289662, 10:15):** 61.44 against 48.63 for 100 slices (+12.8; at
  32x32 +3.8). The staircase costs far more at 64x64. With the near-t = 0 losses at 32x32 this argues for more
  arrays at small t, or linear crossfades instead of steps, before the step version is used at 64x64.

## LOCK-IN FOR THE PAPER (2026-11-21 ~10:50; the user's lock-in list. No new exploratory runs after this.)

- **Model** (the same for every data set): K16 global Potts unit, 64 states per patch unit, Q and per-pixel units,
  100 slices, linear interpolation.
  - 28x28 / 32x32: 16x16 every 4 px from t = 4.49 + 8x8 every 2 px from t = 2.22 (CelebA-32 exact: 20.20,
    Inception 4287721).
  - 64x64: 32x32 every 8 px from t = 4.49 + 16x16 every 4 px from t = 2.22, 64 states.
- **Runs**, seed 1, for MNIST ten digits, Fashion ten classes, CelebA-32, CelebA-64: (1) exact statistics, Adam;
  (2) on device, S = 64; (3) on device, S = 16. On device = the configuration of 4289676 (every hidden unit sampled,
  jackknife, visit-count rule for global and patch states, same-draw counts). CelebA-32 exact = the existing 20.20
  run; CelebA-32 on device S = 64 = 4289676 (checked: --hidden-samples 64).
- **Queue cleaned:** cancelled 4287888 / 4287889 (64x64 staircase evaluation) and 4289678 (extra gallery). Left
  running: 4287884 (64x64 on-device staircase) and 4289675 (K16 32-state on device, 100 slices); its result shows
  up in the gate job 4289677 as context.
- **Gates:**
  - (a) On device for the other data sets only if Inception 4289677 shows K16 on device within about 1 FID of exact
    on CelebA-32 (20.20).
  - (b) MNIST on device only if the exact MNIST model is not clearly worse than K784 + 7x7. Reference: 27.7 after
    the quench, equilibrated start; the same model from the noise start, as the lock-in samples: 27.86.
  - No tuning.
- **Stage 1, submitted 10:50:**

  | data set | pipeline (train + 2000 SDE samples + quench 1) | corrector set (2:0.16, quench 1) | Inception | classifier FID |
  |---|---|---|---|---|
  | MNIST | **4289729** | 4289732 | 4289737 | 4289738 |
  | Fashion | **4289730** | 4289733 | 4289739 | 4289740 |
  | CelebA-64 | **4289731** (160 G, 2 days: 64 states double the 60 G / 7.8 h of the 32-state run) | 4289734 | 4289741 | - |

  - CelebA-64 update (2026-11-22): 4289731 trained (6 h) and sampled (4 h), but the quench ran out of GPU memory;
    4289734 / 4289741 cancelled; quench done by 4297019 (platform allocator) -> corrector 4297020 -> Inception 4297027. See
    "LOCK-IN RESULTS, first wave".
  | CelebA-32 (exact exists) | - | 4289735 | 4289742 (with ODE 4289736 and on-device 4289676) | - |

  - Run names: `{mnist28_d0123456789,fashion28_c0123456789,celeba32g}_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1`,
    `celeba64g_K16_..._patch32s8S64on10+patch16s4S64on55_seed1`; on device: `..._S64jk_occ-patchocc_seed1` /
    `..._S16jk_occ-patchocc_seed1`.
  - Evaluation script: scratchpad `eval_lock.sh DATASET inc|cls|gal`. It takes every set that exists (exact,
    S64, S16; SDE as sampled and quenched, corrector, knots, ODE), plus the old best model of the data set as a
    reference row.
- **Stage 1 results, MNIST and Fashion exact (2026-11-21 12:45):** all n = 2000 from the noise start, seed 1.
  Inception = FID against train, Inception 4289737 (MNIST) / 4289739 (Fashion). Classifier FID, confident share and
  k-NN precision / recall: 4289738 / 4289740. Old best = K784 + 7x7 patch (MNIST) or K784 + 8x8 every 2 px
  (Fashion), both 32 states from t = 2.22, same sampler.

  | MNIST ten digits | Inception (sd) | FID_inf | classifier FID | confident | precision F | recall F |
  |---|---|---|---|---|---|---|
  | real images (floor) | 3.70 | 1.11 | 1.69 | 97.8% | 0.87 | 0.88 |
  | old best K784 + 7x7, SDE + quench 1 | 27.86 (0.36) | 24.13 | **7.66** | **88.8%** | **0.81** | 0.80 |
  | K16 / 64 states exact, SDE as sampled | 50.56 | 58.93 | 15.74 | 71.4% | 0.63 | 0.87 |
  | K16 / 64 states exact, SDE + quench 1 | 14.39 (0.61) | 8.13 | 14.23 | 71.5% | 0.61 | 0.89 |
  | K16 / 64 states exact, corrector 2:0.16 + quench 1 | **12.00** (0.27) | **4.83** | 12.27 | 73.4% | 0.61 | **0.90** |

  | Fashion ten classes | Inception (sd) | FID_inf | classifier FID | confident | precision F | recall F |
  |---|---|---|---|---|---|---|
  | real images (floor) | 7.17 | 2.09 | 0.40 | 73.8% | 0.88 | 0.87 |
  | old best K784 + 8x8, SDE + quench 1 | 26.35 (0.36) | 13.71 | **3.17** | **62.7%** | **0.82** | 0.73 |
  | K16 / 64 states exact, SDE as sampled | 53.26 | 37.28 | 7.38 | 47.2% | 0.79 | 0.75 |
  | K16 / 64 states exact, SDE + quench 1 | 26.00 (0.51) | 15.66 | 5.97 | 49.9% | 0.70 | 0.80 |
  | K16 / 64 states exact, corrector 2:0.16 + quench 1 | **25.80** (0.46) | 14.62 | 4.88 | 50.3% | 0.72 | **0.81** |

  - **Gate (b), MNIST:** on the gate's metric (Inception, 27.7 / 27.86) K16 exact is far better (14.39; 12.00
    with the corrector). On the classifier it is clearly worse: FID 14.2 vs 7.7, 72% vs 89% confident digits,
    precision 0.61 vs 0.81 (recall a little higher). Galleries
    (`galleries/2026-11-21_mnist_interim_K16-64-patch-states-exact-vs-K784-patch7x7-old-best_noise-start_quench1`,
    `..._fashion_interim_...`): K16 strokes are cleaner (no specks or grain, which is what Inception rewards), but
    more samples are not a clear digit. Fashion: Inception equal (26.00 vs 26.35), classifier worse (5.97 vs 3.17).
    Reported to the user; MNIST on device goes with gate (a) unless the user says to hold it.
  - **Decision (the user, 2026-11-21 ~13:00):** "lets stick to the Inception FID. there is no point of satisfying
    a metric that only us uses". Inception FID is the gate and headline metric (it is what Wang & Achour and the
    literature report). The classifier FID and class shares stay in the tables as a supplementary column only.
    Gate (b) passed; MNIST on device goes with gate (a).
  - Class shares stay balanced (MNIST 6-15% per digit; the '3' and '5' a little high, '6' low).
- **MNIST and Fashion on device submitted before gate (a) (2026-11-21 13:30; the user: "why not submit the Fashion
  MNIST and MNIST training with K16 and the 64 patches? why wait?").** If gate (a) fails, these are reported and can
  be cancelled. Each chain runs by Slurm dependencies: pipeline (train + 2000 SDE samples + quench 1; 64 G, 24 h) ->
  corrector set; S = 64 also -> post-hoc knots 5/9/9 (`compress_history.py --mode knots:9 --group-modes
  theta=knots:5,W=knots:5,b=knots:5,Q=knots:5,wl=svd:0,bl=svd:0`) -> SDE + quench 1 of the compressed run; then
  Inception, classifier FID and gallery over all sets (exact, S64, S16, corrector, knots). Submission script:
  scratchpad `lock_submit_ondevice.sh` (+ `lock_compress.sh`).

  | data set | S | pipeline | corrector | knots 5/9/9: compress / sample | Inception / classifier / gallery |
  |---|---|---|---|---|---|
  | MNIST | 64 | **4290098** | 4290099 | 4290100 / 4290101 | 4290104 / 4290105 / 4290106 |
  | MNIST | 16 | **4290102** | 4290103 | - | (same) |
  | Fashion | 64 | **4290107** | 4290108 | 4290109 / 4290110 | 4290113 / 4290114 / 4290115 |
  | Fashion | 16 | **4290111** | 4290112 | - | (same) |

  Resources: the CelebA-32 on-device run (4289676) uses 8.8 G RSS and about 7-8 h; MNIST / Fashion have fewer patch
  units (16 + 121 vs 25 + 169). Disk: 203 G free (81% of 1 T), about 3 G per run.
- **Global-unit test, MNIST and Fashion (submitted 2026-11-21 13:30; the user: "K16 locked model: lower Inception FID
  but fewer clear digits than the old K784 model (71.5% vs 88.8% confident). Test whether this is the global
  unit.")** Exact statistics, seed 1, locked patch units unchanged (16x16 every 4 px from t = 4.49 + 8x8 every 2 px
  from t = 2.22, 64 states), global K = 784 and K = 128. Same sampling and evaluation as the K16 exact runs (noise
  start, SDE + quench 1, corrector set; Inception, classifier FID and confident share; gallery on the same
  trajectories with the old best, K16, K128, K784). Evaluation script: scratchpad `eval_kglobal.sh`.

  | data set | K | pipeline (12 h, 64 G) | corrector | Inception / classifier / gallery |
  |---|---|---|---|---|
  | MNIST | 784 | **4290137** | 4290138 | 4290141 / 4290142 / 4290143 |
  | MNIST | 128 | **4290139** | 4290140 | (same) |
  | Fashion | 784 | **4290144** | 4290145 | 4290148 / 4290149 / 4290150 |
  | Fashion | 128 | **4290146** | 4290147 | (same) |

  - Gallery: `galleries/2026-11-21_{mnist,fashion}_global-unit-test_K16-vs-K128-vs-K784_locked-patches-16x16-8x8-64-states_exact_noise-start_quench1`.
  - **Results (2026-11-21 16:00; Inception 4290141 / 4290148, classifier 4290142 / 4290149):** n = 2000, noise
    start, seed 1, quench 1 unless noted. The K784 corrector sets ran out of memory at 32 G (4290138 / 4290145);
    rerun with 64 G as 4290595 / 4290598, re-scored by 4290596 / 4290597 (MNIST) and 4290599 / 4290600 (Fashion).

    | MNIST ten digits | Inception (FID_inf) | + corrector | classifier FID | confident | precision F / recall F |
    |---|---|---|---|---|---|
    | old best K784 + 7x7 (one 32-state set from t = 2.22) | 27.86 (24.13) | - | 7.66 | **88.8%** | **0.81** / 0.80 |
    | locked patches + K16 | 14.39 (8.13) | 12.00 (4.83) | 14.23 | 71.5% | 0.61 / 0.89 |
    | locked patches + K128 | 12.03 (6.29) | 10.62 (4.20) | 9.44 | 76.3% | 0.67 / 0.88 |
    | locked patches + K784 | **10.01 (4.44)** | **9.19 (3.57)** | **6.23**; corrector 5.06 | 83.0%; corrector 82.2% | 0.74 / 0.88 |

    | Fashion ten classes | Inception (FID_inf) | + corrector | classifier FID | confident | precision F / recall F |
    |---|---|---|---|---|---|
    | old best K784 + 8x8 every 2 px (32 states from t = 2.22) | 26.35 (13.71) | - | 3.17 | **62.7%** | **0.82** / 0.73 |
    | locked patches + K16 | 26.00 (15.66) | 25.80 (14.62) | 5.97 | 49.9% | 0.70 / 0.80 |
    | locked patches + K128 | 23.53 (11.80) | 23.30 (11.14) | 3.57 | 58.5% | 0.78 / 0.81 |
    | locked patches + K784 | **21.32 (9.78)** | 21.82 (9.79) | **2.77**; corrector 2.32 | 62.0%; corrector 62.6% | 0.81 / 0.80 |

  - **Reading: yes, it is the global unit.** Every metric improves monotonically with K at fixed patches (Inception,
    classifier FID, confident share, precision; recall stays at 0.80-0.89). With K784 the locked patches beat the
    old best on both data sets in Inception (MNIST 10.0 vs 27.9, Fashion 21.3 vs 26.4) and in classifier FID (6.2 vs
    7.7, 2.8 vs 3.2). The confident share is back at the old level on Fashion (62.0 vs 62.7%) and closes most of
    the gap on MNIST (83.0 vs 88.8%; K16 was 71.5%). Contrast with CelebA-32, where K hardly mattered (K16 / 32 / 64
    within about 0.4 FID): on the ten-class data the global unit carries the class / shape identity, which the
    local patches cannot.
  - **Gallery** (`galleries/2026-11-21_mnist_global-unit-test_...`): K784 + locked patches draws the same digit as
    the old best in most columns (same trajectory), without the grain; K16 / K128 more often an unclear shape.
    Nearest training images: similar digits, not copies.
  - Run times (exact, train + 2000 samples): K16 1.4 h, K128 1.5 h, K784 2.2 h.
  - **K784 corrector sets (17:15, 4290596 / 4290599, classifier 4290597 / 4290600):** MNIST 9.19 (FID_inf 3.57), the
    best MNIST number so far, classifier 5.06; Fashion 21.82 vs 21.32 without (FID_inf equal, 9.79 vs 9.78): on Fashion
    the corrector no longer helps Inception at K784 (it gave -0.2 at K16 / K128), while the classifier FID still
    improves (2.32 vs 2.77).
  - **K16 on device for MNIST and Fashion held until this is in** (the user: "Hold the pending K16 on-device runs").
    The four trainings had started 5-6 min earlier; `scontrol requeuehold` put them back to pending with their ids,
    so their corrector / knots / evaluation jobs stay chained. Release: `scontrol release 4290098 4290102 4290107
    4290111` (they resume from the saved slices; slices are written atomically).
- **Ablation: do the visible-visible springs matter in the locked model? (submitted 2026-11-21 ~14:00; the user:
  "Control ... This is an ablation row, not a change of the locked model.")** CelebA-32, exact, seed 1, n = 2000,
  noise start, SDE + quench 1, Inception FID, gallery on the same trajectories. No corrector set (not asked).

  | row | flags | run | pipeline |
  |---|---|---|---|
  | locked model with springs (the 20.20 run) | `--pair-terms lin` (radius-1 c_lin (x_i - x_j)^2, the only pair group used) | `celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1` | exists |
  | locked model, no springs | identical flags except `--pair-terms none` (on-site quartic terms k_lin x^2 / 2 + k_duff x^4 / 4 + bias kept; global, patch and per-pixel units unchanged) | `celeba32g_K16_ema_potts_quad_local_quartic_r1_nopairs_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1` | **4290231** (12 h) |
  | pixels only (optional row) | the same visible model with springs (`--visible quartic --radius 1 --pair-terms lin`), K = 0, no `--local`, no patches: no hidden units at all | `celeba32g_K0_ema_pixels-only_quartic_r1_lin_mixst_seed1` | **4290232** (6 h) |
  | no springs, linear self spring only (`--visible harmonic`, added 2026-11-21 17:30) | `--pair-terms none`, k_duff = 0 | `celeba32g_K16_ema_potts_quad_local_harmonic_r1_nopairs_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1` | 4290731: **27.42** (FID_inf 22.83), div 0 (Inception 4290732) |
  | springs kept, linear self spring (`--visible harmonic --pair-terms lin`; no x^4 term anywhere; added 2026-11-22 10:40, the user: "only linear local and coupling stiffness. no duffing for the visible ones") | k_duff = 0, c_lin kept | `celeba32g_K16_ema_potts_quad_local_harmonic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1` | **4297259** (12 h); Inception 4297260 (all 5 rows), gallery 4297261 (`galleries/2026-11-21_celeba32_spring-ablation_locked-K16-64-states_self-term-quartic-vs-linear_x_springs-vs-none_plus-pixels-only_exact_noise-start_quench1`) |

  - Inception 4290233, gallery 4290234
    (`galleries/2026-11-21_celeba32_spring-ablation_locked-K16-64-states_with-vs-without-visible-springs_vs-pixels-only_exact_noise-start_quench1`);
    scratchpad `eval_springs.sh`.
  - Checked before submitting: the stencil energy handles an empty pair-group list (on-site terms only); K = 0
    refuses `--ref-kmeans gpu` and the patch flags, so the pixels-only row drops them (and `--quad`, `--proto-init`,
    which need hidden units).
  - Reading of "the same visible model" for the pixels-only row: with springs (without them the pixels would be
    independent); "no hidden units" includes the per-pixel units of `--local`.
  - **Results (2026-11-21 16:20; Inception 4290233; floor 6.43 at n = 2000, FID_inf 2.22):**

    | CelebA-32, exact, seed 1 | Inception, SDE + quench 1 (sd) | FID_inf | as sampled (no quench) | train time |
    |---|---|---|---|---|
    | locked model with springs (K16, 64 states) | **20.20** (0.19) | **12.34** | 76.70 | - |
    | the same, no springs (`--pair-terms none`) | 25.83 (0.35) | 20.32 | 99.31 | 2.2 h |
    | pixels only: springs, no hidden units | 218.38 (1.44) | 337.93 | 317.32 | 0.3 h |

  - **Reading:** the springs matter, but as a texture term: +5.6 FID (+8.0 FID_inf) without them. In the gallery
    the no-spring samples are the same faces on every trajectory (identity, pose, hair, lighting set by the
    hidden units), only grainier, with pixel-to-pixel speckle and blockier edges. The quench removes much less of
    that grain without springs. Pixels only (springs alone) gives generic blurry face-like texture, no structure:
    all content comes from the hidden units; the springs only smooth neighbouring pixels.
  - Gallery: `galleries/2026-11-21_celeba32_spring-ablation_locked-K16-64-states_with-vs-without-visible-springs_vs-pixels-only_exact_noise-start_quench1`.
- **Two more CelebA-32 controls and more on-device MNIST / Fashion (submitted 2026-11-21 17:00-17:30; the user: "no
  coupling and only linear self spring on each pixel (no duffing)"; "K784 for the 32x32 CelebA. to see whether this
  also improves"; "then I want start running some more on-device training for the MNIST and the Fashion MNIST").**
  - **Harmonic visible model, no springs:** new option `--visible harmonic` (only k_lin x^2 / 2 + bias per pixel;
    k_duff = k_6 = 0 exactly; k_lin starts at 1, no floor). Caveat: without the x^4 term, positive squared-pixel
    couplings Q of the hidden units can make F unbounded at large |x|; the test prints the per-pixel margin
    k_lin / 2 - (max Q+ of the global unit + folded max Q+ of the patch units), also for the trained 20.20 quartic
    model, and the samples' divergence count. Test **4290730** (quartic bitwise regression, refusals, constraints in
    all 100 slices, margins, sampling) -> full run **4290731** (only if the test passes; locked flags with
    `--visible harmonic --pair-terms none`; run `celeba32g_K16_ema_potts_quad_local_harmonic_r1_nopairs_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1`)
    -> Inception 4290732 and gallery 4290733 with all spring-ablation rows (scratchpad `eval_springs.sh`, now 4
    rows; gallery `..._spring-ablation_locked-K16-64-states_with-vs-without-springs_vs-harmonic-self-spring-only_vs-pixels-only_...`).
  - **Harmonic test 4290730 PASSED (17:41):** quartic path old vs new bitwise; refusals of --pair-terms all / duff /
    lin,om; k_duff, k_6 and all pair groups exactly 0 in all 100 slices; 0 of 200 samples diverged (1%-steps run).
    Confinement margin min_i [k_lin_i / 2 - Q+_i] (Q+ = max positive squared-pixel coupling of the global unit +
    folded patch units): in the trained **quartic 20.20 model** it turns negative from t ~ 1.5 on (slice 70: 983 of
    1024 pixels; slices 90 and 99: all pixels; k_lin median 0.66 -> 0.04, patch Q+ up to 1.3). So at small t the
    x^4 term, not k_lin, confines that model; a fully trained harmonic model may become unbounded there (watch the
    div column of 4290732). Full run 4290731 started 17:43.
  - **CelebA-32 K784 with the locked patches, exact:** pipeline **4290714**, corrector set 4290715 (64 G); Inception
    4290734 (K784 vs K16 vs the previous K64), gallery 4290735
    (`galleries/2026-11-21_celeba32_global-unit-test_K16-vs-K784_locked-patches-16x16-8x8-64-states_exact_noise-start_quench1`);
    scratchpad `eval_c32_kglobal.sh`.
    **Result (Inception 4290734, 23:05):** K784 19.43 (11.59), corrector 19.52 (10.71) vs K16 20.20 (12.34) / 19.95
    (11.62): -0.8 / -0.4, a small gain (MNIST / Fashion: -4.4 / -4.7).
  - **MNIST / Fashion on device, K16 and K784** (the user chose "K784 + K16"): the held K16 runs released at 17:25
    (4290098, 4290102, 4290107, 4290111; chains as above). K784 on device with the same flags, `--K 784`:

    | data set | S | pipeline (36 h) | corrector (64 G) | knots 5/9/9: compress / sample | Inception / classifier / gallery |
    |---|---|---|---|---|---|
    | MNIST | 64 | **4290774** | 4290775 | 4290776 / 4290777 | 4290780 / 4290781 / 4290782 |
    | MNIST | 16 | **4290778** | 4290779 | - | (same) |
    | Fashion | 64 | **4290783** | 4290784 | 4290785 / 4290786 | 4290789 / 4290790 / 4290791 |
    | Fashion | 16 | **4290787** | 4290788 | - | (same) |

    The evaluation (scratchpad `eval_k784od.sh`) waits for both the K16 and the K784 chains and puts K16 and K784,
    exact / S64 / S16 / corrector / knots, in one table and one gallery
    (`galleries/2026-11-21_{mnist,fashion}_locked-patches-64-states_K16-and-K784_exact-vs-on-device-S64-vs-S16_noise-start_quench1`).
  - Disk at submission: 858 G of 1 T used (84%), 167 G free.
- **CelebA-32 on device S = 16 started before gate (a) (2026-11-21 17:55; the user: "yes start CelebA-32 S=16 now").**
  Pipeline **4290850** (24 h), corrector set 4290851. Also queued behind 4289676 (CelebA-32 on device S = 64), the
  lock-in sampling extras that were not yet submitted: corrector set 4290852, post-hoc knots 5/9/9 4290853 -> SDE +
  quench 1 4290854, ODE + quench 1 4290855. Full CelebA-32 lock-in table (exact, S64, S16, corrector, knots, ODE):
  Inception **4290856**; gallery (real, exact, S64, S16) 4290857. The earlier Inception 4289742 still runs after
  4289676 with the sets that exist then. CelebA-64 on device (S = 64, S = 16) still waits for gate (a).
- **Best-MNIST gallery (2026-11-21 17:30, job 4290806; the user: "where can I see that samples"):**
  `galleries/2026-11-21_mnist_K784+16x16-8x8-patches-64-states_exact_corrector2x0.16_quench1_best-MNIST/` (the 9.19
  set; rows on the same draws: K784 with corrector, K784 quench only, K16 with corrector, old best). Clean strokes,
  mostly clear digits, a handful of failures. Several samples are close to their nearest training image (not pixel
  copies); a proper memorisation check (sample-to-train distances vs held-out test-to-train) was offered, the user:
  "I think this is fine for now".
- **Comparing with Wang & Achour (discussion 2026-11-21 ~13:00):** both report Inception FID; theirs states no n or
  implementation. Inception FID on MNIST weighs faint pixel noise heavily (real test images + N(0, 0.01^2) noise:
  38.5; our K16 before / after the quench 50.6 / 14.4 with images that look alike) and is biased at small n (real vs
  real 11.1 at n = 1000, 3.7 at n = 2000), so their 27.6 and our numbers compare only roughly; report n, the
  reference set (train), the implementation (TTUR graph) and FID_inf. An independent cross-check (pytorch-fid on the
  same files) was offered, not run.
- **Status 2026-11-21 18:00 (what is still to come):**
  - Gate (a) (Inception 4289677, about 19:30); after it, CelebA-64 on device S = 64 / S = 16 (the only lock-in runs
    not yet submitted).
  - Running: MNIST / Fashion K16 and K784 on device (8 runs), CelebA-32 S = 16, CelebA-64 exact (sampling), CelebA-32
    K784 exact, harmonic no-spring run; all evaluation jobs chained.
  - Second wave, after the first results: seeds 2 and 3 of S = 64 for MNIST, Fashion, CelebA-32; 10,000 samples of the
    S = 64 models. Reporting: one table per data set (exact, S64, S16, corrector, knots, ODE).

### LOCK-IN RESULTS, first wave (collected 2026-11-22 10:00; jobs finished 2026-11-21 19:00 - 2026-11-22 01:30)

All: seed 1, n = 2000, noise start, Inception FID against train at n = 2000 with FID_inf in brackets; "div" = samples
that diverged (left out). K16 = the locked recipe (K16 + 16x16 every 4 px from t = 4.49 + 8x8 every 2 px from t =
2.22, 64 patch states; 64x64: 32x32 / 16x16); K784 = the same patches with a 784-state global unit. On device = every
hidden unit sampled with S states per sample, jackknife, visit-count rule for global and patch states, same-draw
counts. Knots = post-hoc 5/9/9 (global 5, each patch set 9, wl / bl constant) of the S = 64 model, SDE + quench 1.

- **Gate (a) PASSED at the edge (Inception 4289677, 19:14):** CelebA-32 K16 on device S = 64 21.20 (13.43) vs exact
  20.20 (12.29): +1.00 ("within about 1 FID"); with the corrector +0.28 (below). Context rows: K16 32 states exact
  22.11 / on device 22.91 (+0.80); the 9/9/9 staircase runs 25.88 / 42.61. CelebA-64 on device submitted on that basis
  (2026-11-22 10:10, below).

**CelebA-32** (Inception 4290856; K784 4290734; floor 6.43, FID_inf 2.22)

| model | SDE + quench 1 | corrector 2:0.16 + quench 1 | ODE + quench 1 | knots 5/9/9 + quench 1 | as sampled (no quench) |
|---|---|---|---|---|---|
| K16 exact (the 20.20 run) | **20.20** (12.34) | **19.95** (11.62) | 22.14 (16.31) | - | 76.70 |
| K16 on device S = 64 (4289676) | 21.20 (13.55) | 20.23 (11.83) | 22.93 (17.08) | 23.65 (15.10) | 79.59 |
| K16 on device S = 16 (4290850) | 21.23 (13.45) | 20.49 (11.93) | - | - | 81.28 |
| K784 exact (4290714; control) | 19.43 (11.59) | 19.52 (10.71) | - | - | 74.49 |
| previous K64 + early patches, 32 states | 21.72 (14.05) | - | - | - | - |

**MNIST ten digits** (Inception 4290780, classifier 4290781; floor 3.70, FID_inf 1.11; classifier FID in the last
two columns)

| model | SDE + quench 1 | corrector + quench 1 | knots 5/9/9 | classifier FID, quench 1 | classifier FID, corrector |
|---|---|---|---|---|---|
| K16 exact | 14.39 (8.13) | 12.00 (4.83) | - | 14.23 | 12.27 |
| K16 on device S = 64 | 15.22 (8.86), div 2 | 12.56 (5.69), div 5 | 234.1 | 13.72 | 11.88 |
| K16 on device S = 16 | 15.38 (9.51), div 9 | 12.79 (5.89), div 4 | - | 13.80 | 12.68 |
| K784 exact | 10.01 (4.49) | **9.19 (3.55)** | - | 6.23 | **5.06** |
| K784 on device S = 64 | 11.37 (5.64) | 9.79 (3.85) | 234.4 | 6.60 | 6.46 |
| K784 on device S = 16 | 11.80 (6.33), div 1 | 10.17 (4.08), div 6 | - | 7.86 | 6.55 |
| old best K784 + 7x7, 32 states | 27.86 (24.13) | - | - | 7.66 | - |

**Fashion ten classes** (Inception 4290789, classifier 4290790; floor 7.17, FID_inf 2.09)

| model | SDE + quench 1 | corrector + quench 1 | knots 5/9/9 | classifier FID, quench 1 | classifier FID, corrector |
|---|---|---|---|---|---|
| K16 exact | 26.00 (15.66) | 25.80 (14.62) | - | 5.97 | 4.88 |
| K16 on device S = 64 | 28.29 (17.78) | 27.44 (15.39) | 98.0 | 6.45 | 5.38 |
| K16 on device S = 16 | 27.54 (17.06) | 26.08 (14.10) | - | 6.08 | 4.80 |
| K784 exact | **21.32 (9.58)** | 21.82 (9.88) | - | 2.77 | **2.32** |
| K784 on device S = 64 | 23.44 (11.92) | 23.50 (11.08) | 88.0 | 3.54 | 3.01 |
| K784 on device S = 16 | 22.93 (11.51) | 23.52 (11.67) | - | 3.51 | 2.97 |
| old best K784 + 8x8, 32 states | 26.35 (13.71) | - | - | 3.17 | - |

- **On device vs exact (SDE + quench 1 / corrector):** CelebA-32 +1.0 / +0.3 (S = 64), +1.0 / +0.5 (S = 16); MNIST K16
  +0.8 / +0.6 and +1.0 / +0.8, K784 +1.4 / +0.6 and +1.8 / +1.0; Fashion K16 +2.3 / +1.6 and +1.5 / +0.3, K784 +2.1 /
  +1.7 and +1.6 / +1.7. **S = 16 is as good as S = 64 everywhere** (within the noise; on Fashion slightly better). On
  device the MNIST models produce a few diverged samples (1-9 of 2000; none exact).
- **Post-hoc knots 5/9/9 fail on MNIST and Fashion** (234 on MNIST for both K16 and K784, classifier FID 431 / 485 with
  almost no '0's; Fashion 98 / 88), while CelebA-32 costs +2.5 (23.65 vs 21.20). The compression reports
  (4290100 / 4290776 / 4290853) show why: the visible group (theta) at 5 knots has relative error 0.35 on MNIST vs 0.12
  on CelebA-32, the global W / b / Q 0.30-0.55 vs 0.22-0.30. The recipe was tuned on CelebA; on the ten-class data the
  visible and global parameters change too much in time for 5 knots. Not tuned (lock-in rule).
- **ODE + quench (CelebA-32):** exact 22.14, on device 22.93: about +2 over the SDE, as before.
- **K784 on CelebA-32 (control):** 19.43 vs 20.20 (-0.8), with the corrector 19.52 vs 19.95 (-0.4): a small gain, far
  less than on MNIST / Fashion (-4.4 / -4.7): on faces the global unit matters little.
- **Spring ablation, harmonic row (Inception 4290732):** no springs and only the linear self spring (--visible
  harmonic, run 4290731) 27.42 (22.83), div 0; vs no springs with the quartic self term 25.83 (20.32) and the locked
  model 20.20 (12.34). Dropping the x^4 self term costs another +1.6 and, despite the negative margins seen in the
  quartic model, no sample diverged. Gallery
  `galleries/2026-11-21_celeba32_spring-ablation_locked-K16-64-states_with-vs-without-springs_vs-harmonic-self-spring-only_vs-pixels-only_exact_noise-start_quench1`.
- **CelebA-64 exact (4289731): training (about 6 h) and the 2000 SDE samples (20 chunks of 100, 4 h) done; the
  quench after them FAILED** (GPU out of memory allocating 7.2 GB, the size of hidden_history.npz with 64 patch
  states). Corrector 4289734 and Inception 4289741 never started (cancelled). 2026-11-22 10:05, job **4297019**
  (`XLA_PYTHON_CLIENT_ALLOCATOR=platform`, `--chunk 25`, 160 G) found the saved SDE samples and ran only the quench
  (177 s; corrected 10:20, I first read the failure as a sampling failure). Corrector 4297020 (running), early
  Inception (exact only) 4297027.
- **CelebA-64 on device (submitted 2026-11-22 10:10, after gate (a)):** S = 64 pipeline **4297021** (4 days; on device
  S = 64 took about 4x exact on CelebA-32, exact 64x64 trained about 6 h), corrector 4297022, knots 4297023 / 4297024; S =
  16 pipeline **4297025** (3 days), corrector 4297026; all with `--chunk 25`, platform allocator, 160 G. Full CelebA-64
  table: Inception 4297028, gallery 4297029. Scratchpad `c64_lock_ids.txt`.
- **Galleries** (real, exact, S = 64, S = 16 on the same trajectories): `galleries/2026-11-21_celeba32_locked-recipe_K16_64-patch-states_exact-vs-on-device-S64-vs-S16_noise-start_quench1`,
  `..._mnist_locked-recipe_...`, `..._fashion_locked-recipe_...`; K16 + K784 together:
  `galleries/2026-11-21_{mnist,fashion}_locked-patches-64-states_K16-and-K784_exact-vs-on-device-S64-vs-S16_noise-start_quench1`.
- **Disk 2026-11-22 10:00:** 907 G of 1 T used (89%), 118 G free; a 64x64 run takes about 14.5 G (hidden_history 7.2 G
  **Looked at (2026-11-22):** in all three galleries, on device S = 64 and S = 16 draw nearly the same image as exact
  training on the same trajectory in almost every column (same face / digit / garment, small differences in detail;
  a few columns differ, e.g. MNIST K784 S = 16 column 11). On-device training learns essentially the same model;
  a candidate paper figure.
  + slices/ 7.2 G, which duplicate it).
