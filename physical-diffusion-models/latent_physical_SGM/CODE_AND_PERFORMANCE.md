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
