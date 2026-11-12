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
