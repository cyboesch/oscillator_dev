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
