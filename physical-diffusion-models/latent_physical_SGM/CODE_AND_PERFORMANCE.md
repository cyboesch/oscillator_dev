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
