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
   would then cost about 4-5% of a run.
5. Sampling becomes the larger part of a pipeline (2,000 samples at K = 3072: 2.6 h); see section 6, item 5.
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
- **Build work** (two parallel tracks, each tested on the GPU, then reviewed; not finished, not yet reviewed):
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
