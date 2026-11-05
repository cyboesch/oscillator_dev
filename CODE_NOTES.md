# Code notes: `physical-diffusion-models`

These notes come from a line-by-line read of every tracked source file on branch `paper_code_v2` at commit `98793e0` (read on 2026-11-04). An independent multi-agent review of the same code then checked them (§12). Notebooks were read from their code cells and the text outputs saved in them. Nothing was executed, because cluster policy forbids computation on the login node. Statements about runs therefore rest on the code plus the outputs already saved in the notebooks.

**Markers:**
- ✔ checked, by derivation, by tracing the code, or from saved output
- ◐ hypothesis worth testing
- *(review)*: first found by the independent review and confirmed by its verifier (§12)
- *latent*: wrong, but not on any current code path
- *cosmetic*: docs or naming only

**Conventions:**
- **Notebook cells** are numbered from 0 by position, counting markdown cells.
- **Always-blank pixels** are pixels that are 0 in every training image. After preprocessing they sit at −0.4096 ± 0.0328 in normalised units.

---

## TL;DR

- **What it is.** This is the code for *"Local Learning Rules for Out-of-Equilibrium Physical Generative Models"*.
  - The model is a 28×28 lattice of coupled 6th-order Duffing oscillators, one per pixel. Its energy is **linear in its 545,566 parameters**.
  - Parameters are fit separately at each of 100 forward-diffusion times, warm-started from the neighbouring time. The fit uses score matching, or CD-1, at the bath temperature.
  - Samples come from the physical reverse-time Langevin SDE, or from the probability-flow ODE, with the parameters interpolated in time.
- **The core math is right** (§9). This covers the analytic derivatives, all four score-matching implementations, the CD-1 sign convention, the OU transition, the reverse-SDE and PF-ODE parameter arithmetic, PCHIP, and the RNG pairing. The independent review re-derived all of it and found no error.
- **Early stopping never fires on improvement** ✔ (§8.1). This affects every training run.
  - `best_moving_avg` starts at `inf`, so the first relative improvement is `nan`, and patience grows at every check.
  - Each slice therefore trains for a fixed `window + 100·(⌈patience⌉−1)` epochs: 10,900–30,900 for MNIST. `tolerance` is never used.
  - In the 2D score-matching notebook all five slices "converge" at exactly epoch 1000. That includes t = 1.25, whose loss is still +107 against ≈ −10 for its neighbours.
- **Near the data, training does keep up: the suspected lag was tested and is not there** (§8.13, measured by Slurm job 4181431 on 2026-11-05).
  - Background pixels reach their optimal stiffness at every slice, within a few percent (1.03× at t = 0). The couplings supply 97% of it, so the per-parameter step budget never binds.
  - The one shortfall is the stiffness of a uniform shift of the whole background: 0.57× optimal at t = 0. That shows up as a small background offset in both samplers.
- **The CD-1 notebook picks per-slice parameters with the wrong-temperature loss** ✔ (§8.2).
  - The monitored loss is score matching at `T = 1`, while the CD-1 gradient targets `T = Temp = 0.01`. Since J₁(θ) = J₀.₀₁(0.01·θ), the monitor ranks iterates by their distance to 100× the CD-1 solution.
  - In 8 of 10 slices the "best" epoch is ≤ 398 of ~2,000, and in four of them it is ≤ 21. The stored θ(t) is therefore mostly the warm start.
- **The stored per-slice parameters are the iterate right after the luckiest mini-batch** ✔ (§8.3). The one-step offset is negligible. What matters is that the choice rests on mini-batch noise, and the learning rate is never annealed.
- **ODE vs SDE, and smoothing** (§10).
  - Your ablation already rules out integration error, interpolation kinks and parameter jitter as the main cause.
  - **Leading explanation.** The review reached it independently: the SDE contains a built-in Langevin corrector toward each slice's model, while the ODE carries every error along.
  - **In numbers.** Suppose a background pixel's learned precision is off by a relative ε. The ODE's final background noise then changes by a factor e^{−6.5ε}, the SDE's only by about e^{−ε}. At ε = −20% that is 3.6× against 1.3×.
  - **Weak evidence for "worse".** The verdict rests on 40 samples, shown with per-image autoscaling and no metric (§8.15).
  - **Out-of-range samples.** An older MNIST SDE run produced pixel values up to ±86 in normalised units, against a data span of −0.41 to 2.87. The clipped plots hide this (§8.5).
  - Cheap diagnostics are in §10. Several need only the outputs you already have.
- **Do not retrain with the current flags** ✔ (§8.12) *(review)*.
  - `checkpoint_date` is hard-wired, so `start_from_scratch=True` would `rmtree` the folder holding the SHA-pinned paper checkpoint.
  - `out/` is not in git, so nothing could be recovered.
  - Set `checkpoint_date = None` first.

---

## 1. Pipeline

```
MNIST digits {0,1}, 28×28, class-balanced (11,846 images)            scripts/preprocess-mnist.py
  │  + N(0, 0.01²) pixel noise, then ONE global mean/std (μ = 0.1248, σ = 0.3047)
  ▼
forward OU   dx = −x dt + √(2D) dW,   D = Temp = 0.02,   t ∈ [0, 5]   (exact transition, nothing simulated)
  │  100 uniform slices, trained t = 5 → 0 with warm start:
  │  Adam on implicit score matching at k_BT = Temp, fresh mini-batches from p_t
  ▼
θ(t_k), k = 0…99  →  out/2026_07_08/…/opt_per_time_plots/params_history.npy   (100 × 545,566, float64)
  │  interpolate in t (linear | PCHIP); optionally Savitzky–Golay-smooth over k
  ▼
reverse SDE   θ_rev(t) = 2·θ(5−t) − θ_lin      physical overdamped Langevin at k_BT    (diffrax SRA1)
PF-ODE        θ_pf(t)  =   θ(5−t) − θ_lin      deterministic                           (diffrax Tsit5)
  │  x(0) ~ N(0, Temp·I), 40 trajectories, initial states and Brownian keys shared by every variant
  ▼
final states (normalised space) → image grids, unclipped and clipped to [−0.41, 2.87]
```
θ_lin is the pure harmonic energy: k_lin = 1, all other parameters 0.

## 2. File map

| File | What it holds | Used by |
|---|---|---|
| [network_fns.py](physical-diffusion-models/physical_diffusion_fns/network_fns.py) | lattice connectivity; energy; analytic ∇E, tr∇²E and their θ-Jacobians (dense, small N only); overdamped SDE; `solve_SDE`, `solve_ODE` | script, all notebooks |
| [network_fns_memory_efficient.py](physical-diffusion-models/physical_diffusion_fns/network_fns_memory_efficient.py) | copy of analytic ∇E and tr∇²E (used), plus two θ-derivative helpers (broken, unused; §8.7) | script, Gradient_Tests |
| [learning_fns.py](physical-diffusion-models/physical_diffusion_fns/learning_fns.py) | score-matching (SM) loss (autodiff); SM gradient (autodiff and analytic); `CD1_gradient`; `run_optimization` | script (`run_optimization` only), notebooks |
| [learning_fns_memory_efficient.py](physical-diffusion-models/physical_diffusion_fns/learning_fns_memory_efficient.py) | SM loss and θ-gradient from analytic ∇E and tr∇²E, autodiff in θ, `lax.scan` over the batch | script, Gradient_Tests |
| [helper_fns.py](physical-diffusion-models/physical_diffusion_fns/helper_fns.py) | Gaussian-mixture sampling; normalisation; exact OU sampling; history reshaping; Savitzky–Golay; linear and PCHIP interpolation | script, notebooks |
| [plotting_fns.py](physical-diffusion-models/physical_diffusion_fns/plotting_fns.py) | diagnostics and paper figures; also switches on float64 at import | script, notebooks |
| [physical_diffusion_model_MNIST.py](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py) | MNIST driver; currently configured for the sampling-only interpolation ablation on the paper checkpoint | Slurm |
| [preprocess-mnist.py](physical-diffusion-models/scripts/preprocess-mnist.py) | builds the MNIST `.npy`; currently set to labels [0,1,2,3] | Slurm |
| `container/{container.def, build.def, Dockerfile, requirements.txt}` | image: jax 0.4.33 (cuda12), diffrax 0.6.0, equinox 0.11.5, optax 0.2.4, numpy 2.0.1, scipy 1.14.0, TF 2.18 + tfds | — |
| `container/run_*_neuronic.sh` | Slurm job scripts (Apptainer) | — |
| `run-mnist-generation`, `run-mnist-preprocessing`, `container/Makefile` | stale launchers; they point to scripts and paths that no longer exist | — |
| `notebooks/*.ipynb` | 2D Gaussian-mixture experiments (SM, CD-1), gradient tests, MNIST plotting and preprocessing | — |

## 3. The physical model

### 3.1 Energy ([network_fns.py:60-98](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L60-L98))

$$
E_\theta(x)=\sum_i\Big[\tfrac12 k^{\rm lin}_i x_i^2+\tfrac14 k^{\rm duff}_i x_i^4+\tfrac16 k^{(6)}_i x_i^6+b_i x_i\Big]
+\sum_{(i,j)}\Big[c^{\rm lin}_{ij}(x_i-x_j)^2+c^{\rm om}_{ij}\,x_i^2x_j+\tfrac14 c^{\rm duff}_{ij}(x_i-x_j)^4\Big]
$$

- **Linear coupling.** The code writes it as `c_lin*x*(x-y) + c_lin*y*(y-x)`, which equals $c^{\rm lin}(x-y)^2$. The spring constant is therefore $2c^{\rm lin}$.
- **Optomechanical term.** $x_i^2x_j$ is cubic and can take either sign.
  - It is also asymmetric: $i<j$ is always the raster-earlier pixel.
  - The stiffness change that is linear in a neighbour's value (Hessian term $2c^{\rm om}x_j$) therefore comes only from raster-later neighbours *(review)*.
  - The $c^{\rm duff}$ term, by contrast, changes the stiffness symmetrically.
- **Linear in θ.** $E_\theta(x)=\theta\cdot\phi(x)$, so $\nabla_xE=\Phi(x)\,\theta$. Three later results depend on this: the reverse-process parameter arithmetic (§6), convexity of the loss (§5.1), and the temperature scaling (§5.1).
- **Flatten order.** `ravel_pytree` flattens the parameter tuple ([script:237-246](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L237-L246)) in this order: `[k_lin(784) | k_duff(784) | k_6(784) | c_lin(P) | c_optomech(P) | c_duff(P) | bias(784)]`.
  - With P = 180,810 pairs this gives **545,566 parameters**.
  - `constraint_indices = arange(1568, 2352)` is the k_6 block.
  - The comment at [script:235-236](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L235-L236) puts bias fourth; that is wrong, but the code's index arithmetic is right (§8.10).
- **Connectivity** ([network_fns.py:12-51](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L12-L51)). Every pair within Chebyshev distance ≤ 14 on the 28×28 grid is coupled *(review)*:
  - that is 180,810 of the 306,936 possible pairs (59%);
  - the mean degree is ≈ 461, and the four central pixels couple to all 783 others;
  - couplings make up 542,430 of the 545,566 parameters.
- **Confinement.** Only the sextic self-term is guaranteed to confine, because `run_optimization` projects k_6 ≥ 0.01 after every Adam step ([learning_fns.py:381-385](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L381-L385)).
- **Capacity** ◐ *(review)*.
  - The self energy has no x³ or x⁵ term, so the self-force is an odd polynomial plus a constant.
  - A double well with minima at the background (−0.41) and ink (2.87) values then needs its barrier inside |x| < 0.41, right next to the background minimum.
  - With the one-sided optomechanical term, this may cap quality for both samplers. The couplings add flexibility.
- **Initial parameters.**
  - MNIST: k_lin = 1, k_duff = 0, k_6 = 1, **c_lin = 1 on all 180,810 pairs**, and c_om = c_duff = b = 0.
  - 2D notebooks: a double well, with k_lin = −1, k_duff = 1, k_6 = 1, and couplings 0.

### 3.2 Dynamics ([network_fns.py:448-562](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L448-L562))

- **`setup_overdamped_SDE`** implements $dx=-\gamma^{-1}\nabla_xE\,dt+\sqrt{2k_BT/\gamma}\,dW$ with γ = k_B = 1. Its stationary distribution is $\propto e^{-E/k_BT}$.
  - θ may be a function of t.
  - The drift is computed by autodiff of `energy_fn`, not by the analytic gradient.
  - The diffusion is the dense matrix $\sqrt{2T}\,I$.
- **`solve_SDE`** uses SRA1, which is strong order 1.5 for additive noise and requires `SpaceTimeLevyArea`. It pairs it with a `VirtualBrownianTree` (tol 1e-12) and an adaptive `PIDController(rtol, atol, step_ts)`, saving at `linspace(t0, t1, N_save)`.
- **`solve_ODE`** uses Tsit5 with the same controller and save grid, and no noise.
- **Both solvers** pass `args=()` and set `max_steps = 1e9` with no `dtmin` (§8.9).

### 3.3 Analytic derivatives ✔

- **Self term:**
  - first derivative $k^{\rm lin}x+k^{\rm duff}x^3+k^{(6)}x^5+b$
  - second derivative $k^{\rm lin}+3k^{\rm duff}x^2+5k^{(6)}x^4$
- **Pair term, first derivatives:**
  - $\partial_{x_i}=2c^{\rm lin}(x_i-x_j)+2c^{\rm om}x_ix_j+c^{\rm duff}(x_i-x_j)^3$
  - $\partial_{x_j}=-2c^{\rm lin}(x_i-x_j)+c^{\rm om}x_i^2-c^{\rm duff}(x_i-x_j)^3$
- **Pair term, diagonal of the Hessian:**
  - at $i$: $2c^{\rm lin}+2c^{\rm om}x_j+3c^{\rm duff}(x_i-x_j)^2$
  - at $j$: $2c^{\rm lin}+3c^{\rm duff}(x_i-x_j)^2$
- All of these match the energy. So do the θ-Jacobians in `network_fns.py` for the 7-group layout, including their offsets.
- **Why the learning rule is "local".** ∂E/∂c_ij involves only x_i and x_j. The SM or CD-1 update for a coupling therefore needs only the two coordinates and the forces at sites i and j.

## 4. Data and forward process

- **Data file.** `data/MNIST/mnist_labels_[0, 1]_resolution_(28, 28)_balanced_True.npy` holds 11,846 images, 5,923 per class. The filename matches what the preprocessor writes.
  - **Provenance** *(review)*. `plot_sde_generated_images` cell 5 shows that notebook writing this file on 2026-02-13, with an algorithm equivalent to `preprocess-mnist.py`: same filter, /255, linear resize, `default_rng(42)` choice and permutation.
  - The repo does not record which copy the checkpoint was trained on. A dataset SHA in the manifest would.
- **Preprocessing** ([script:207-220](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L207-L220)):
  1. Add N(0, 0.01²) pixel noise, drawn from a key-derived RNG stream.
  2. Standardise with **one global** mean and std (ddof = 0): μ = 0.12478, σ = 0.30467. The mean of x₀² over all entries is then exactly 1.
  3. As a result the normalised pixel range is [−0.4096, 2.8727], which is also the clipping window for plots. Background pixels have std ≈ 0.033 (variance ≈ 1.08e-3).
- **Forward process.** `sample_forward_process` ([helper_fns.py:80-126](physical-diffusion-models/physical_diffusion_fns/helper_fns.py#L80-L126)) samples the exact OU transition $x_t=e^{-\beta t/\sigma_f^2}x_0+\sqrt{D\sigma_f^2(1-e^{-2\beta t/\sigma_f^2})}\,\xi$.
  - With β = σ_f = 1 and D = 0.02, the stationary distribution is $p_\infty=\mathcal N(0,0.02\,I)$.
  - At t = 5 the residual signal is e⁻⁵ ≈ 0.0067 of x₀, against a noise std of 0.141.
  - Along a direction where the data variance is λ, signal and noise are equal at t ≈ ½ ln(1 + λ/D). That is ≈ 2.0 for λ = 1. It would be ≈ 4.3–4.5 for λ = 100–160, a plausible but uncomputed range for the top principal components (§10.3).
- **`Temp` plays two roles.** It is both the bath temperature k_BT and the forward diffusion constant D. Setting D = k_BT is exactly what makes the reverse process a physical Langevin process at the bath temperature (§6).
- **Time grid.** `linspace(0, 5, 100)[::-1]` gives Δt = 0.0505, stored with t = 5 first. Every per-slice schedule is indexed in the same reversed order.

## 5. Training

### 5.1 Objective

Implicit score matching for $p_\theta\propto e^{-E_\theta/k_BT}$, with model score $s_\theta=-\nabla E_\theta/k_BT$:

$$J_T(\theta)=\mathbb E_{p_t}\Big[-\frac{\Delta_xE_\theta}{k_BT}+\frac{\lVert\nabla_xE_\theta\rVert^2}{2(k_BT)^2}\Big]=\tfrac12\mathbb E\lVert s_\theta-\nabla\log p_t\rVert^2+\text{const}$$

$$\partial_\theta J_T=\mathbb E\Big[-\frac{\partial_\theta\Delta E}{k_BT}+\frac{(\partial_\theta\nabla E)^{\top}\nabla E}{(k_BT)^2}\Big]$$

| Implementation | How it computes | Used by |
|---|---|---|
| `setup_score_matching_LOSS_per_batch` | autodiff Hessian of −E | Gradient_Tests (ground truth); CD-1 notebook as the *monitored* loss, with default T = 1 (§8.2) |
| `setup_force_matching_GRADIENT_per_batch_using_AD` | nested `jacfwd` | Gradient_Tests |
| `setup_force_matching_LOSS_and_GRADIENT_per_batch_analytical` | analytic Jacobians (dense D×P) | SM notebook |
| `setup_score_matching_kbT_loss_minimal_memory` | analytic ∇E and tr∇²E, autodiff in θ, `lax.scan` over samples, `jax.checkpoint` | MNIST script |

All four agree in sign and in their k_BT factors ✔. Gradient_Tests (cells 17–23) matches them to ≈ 1e-13, within the limits listed in §8.11.

**Consequences of linearity in θ** ✔ (by derivation):
- **The loss is a convex quadratic.** $J_T(\theta)=\tfrac{1}{2T^2}\theta^\top A\theta-\tfrac1T b^\top\theta$, with $A=\mathbb E[\Phi^\top\Phi]\succeq0$ and $b=\mathbb E[\Delta\phi]$.
  - A is ill-conditioned; the `hessian` branch found it "badly conditioned". Many quite different θ therefore give nearly the same loss.
  - Two examples *(review)*: for an always-blank pixel the bias and k_lin features are 99.7% correlated (2×2 condition number ≈ 620), and the c_duff curvature is ~1e-6 of k_lin's.
  - Each mini-batch quadratic is exactly rank-deficient *(review)*: its rank is at most 512·784 = 401,408 < 545,566, so every batch leaves ≥ 144,158 directions completely free.
- **Temperature is only a reparametrisation** for a fixed target. With u = θ/T, $J_T(\theta)=J_1(u)$. The fitted *distribution* depends on T only through three things:
  1. the forward process, since D = T;
  2. the k_6 floor: k_6 ≥ 0.01 means u_6 ≥ 0.01/T. At T = 0.02 this forces at least x⁶/12 into −log p_θ, about 47 nats at the bright-pixel value 2.87;
  3. the optimiser's scale: at T = 0.02 the initial θ₀ corresponds to u₀ = 50·θ₀, and an Adam step of size lr in θ is a step of lr/T in u.
- **Corollary** *(review)*. Adam ignores the scale of the gradient. Setting lr = Temp, as the SM notebook does, therefore gives the same dynamics in u at every temperature. The start u₀ = θ₀/T and the floor 0.01/T still change with T.

### 5.2 CD-1 ([learning_fns.py:215-275](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L215-L275)) ✔

- **The step** is $x'=x-\tfrac{dt}{2}\nabla E+\sqrt{D\,dt}\,\xi$: an overdamped Langevin step of length dt/2 at temperature D.
- **What `CD1_gradient` returns.** Expanding to second order, $\langle\partial_\theta E(x)\rangle-\langle\partial_\theta E(x')\rangle\approx \tfrac{D^2dt}{2}\,\partial_\theta J_{T=D}$.
  - The function therefore returns **$-\partial_\theta J_D$**, the CD update direction of Hyvärinen (2007), eq. 1. That is a *descent* direction for J_D.
  - `run_optimization` descends whatever gradient it is given, so callers must flip the sign. (An earlier version of these notes called the return value an "ascent direction", which was misleading.)
- **Callers.** Both negate it and keep `maximize=False`: CD-1 notebook cell 15 and Gradient_Tests cell 25. The result is correct gradient descent on $J_{T=D}$, with best-loss tracking consistently in minimisation mode.
- **Hockey-stick test** (Gradient_Tests cell 25).
  - The minimum relative error is 8.7e-3, at dt ≈ 6e-3, with Temp = 0.1 and 100 noise samples.
  - The CD-1 notebook itself runs at D = 0.01, dt = 1e-4 with 3,000 noise samples, an operating point that is not tested (§8.11).
- **Deterministic option.** `num_noise_samples=0` drops the noise, so the result is no longer a score-matching gradient.

### 5.3 `run_optimization` ([learning_fns.py:285-426](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L285-L426))

- **Optimiser.** Adam with `optax.exponential_decay(lr, lr_decay_steps, lr_decay_rate)`, which decays smoothly rather than in steps. The training step is jitted, gradients are multiplied by a mask, and `maximize` flips the sign.
- **Default decay** *(review)*. The defaults (×0.99 per 10 steps) shrink the learning rate tenfold every ≈ 2,291 steps.
  - The 2D notebooks use these defaults, so their learning rate is down to 0.37× by epoch 1000 and 0.13× by epoch 2000.
  - MNIST overrides them with ×0.95 per 6,000 steps.
- **Fresh state per call.** Each call creates a new Adam state and a new jitted closure, so every slice is re-traced and re-compiled. That costs runtime only.
- **Constraint.** k_6 is projected onto its floor after each step. The floor `epsilon = .01` is hard-coded ([learning_fns.py:383](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L383)) and must match the script's `ablation_k6_floor` by hand.
- **Reported loss.** It is the **pre-update** mini-batch loss.
- **History.** A snapshot is kept on device every 100 epochs. At MNIST scale that is up to ~310 × 4.4 MB ≈ 1.4 GB per slice in float64.
- **Return value.** `(history, losses, best_params, best_loss, best_epoch)`. **Every caller uses `best_params` as the slice result** (§8.3).
- **Early stopping:** see §8.1.
- **`maximize=True` is only half supported** *(latent, review)*. It flips the gradient and the best-loss comparison, but not the moving-average test. That only matters once §8.1 is fixed.

### 5.4 MNIST schedule as configured ([script:95-119](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L95-L119), [script:325-405](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L325-L405))

- **Order.** Slices are trained from t = 5 down to t = 0, each warm-started from the previous one. Every step uses a batch of 512 fresh forward samples.
- **Learning rate.** 2e-3 at t = 5 falling linearly by slice index to 2e-5 at t = 0, multiplied by 0.95 per 6,000 steps. That factor reaches ≈ 0.77 over a 30,900-step slice. The resulting per-parameter movement budget is far too small near t = 0 (§8.13).
- **Patience.** 100 at t = 5 rising to 300 at t = 0. Because of §8.1 the slices run for exactly **10,900 to 30,900 epochs**; `n_epochs = 100000` and `tolerance = 1e-3` never come into play.
- **Checkpointing.** After every slice the script writes the stack of `best_params` and the next time index.

## 6. Generation ✔

Ingredients: forward drift $f=-x/\sigma^2$, noise $g=\sqrt{2D}$, score $-\nabla E_\theta/k_BT$, and D = k_BT. Run the reverse process in time t, with forward time τ = 5 − t.

- **Reverse SDE** ([script:504-547](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L504-L547)):
  - $dx=[x/\sigma^2-2\nabla E_{\theta(\tau)}]dt+\sqrt{2k_BT}\,dW=-\nabla E_{\theta_{\rm rev}}dt+\sqrt{2k_BT}\,dW$, with $\theta_{\rm rev}=2\theta(\tau)-\theta_{\rm lin}/\sigma^2$.
  - This is a physical overdamped system at the bath temperature, with time-programmed parameters.
- **PF-ODE** ([script:550-593](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L550-L593)):
  - $\dot x=x/\sigma^2-\nabla E_{\theta(\tau)}=-\nabla E_{\theta_{\rm pf}}$, with $\theta_{\rm pf}=\theta(\tau)-\theta_{\rm lin}/\sigma^2$.
- **The identity that links them.** drift_SDE = drift_ODE + (−∇E_θ(τ)), with noise √(2k_BT). In other words, **reverse SDE = PF-ODE + equilibrium Langevin dynamics on the model's own p_θ(τ).** The review re-derived this independently.

**Sampler settings:**
- x(0) ~ N(0, Temp·σ²·I); 40 trajectories; t ∈ [0, 5]; dt0 = 1e-8.
- SDE: SRA1, rtol 1e-4, atol 1e-6, Brownian tolerance 1e-12.
- ODE: Tsit5, rtol 1e-7, atol 1e-9.
- Only the final state of each trajectory is used and saved.
- **Prior mismatch** *(review)*. N(0, Temp·I) is not exactly p₅. The residual mean e⁻⁵x₀ puts the KL between them at most 0.89 nats over all 784 pixels.

**RNG** ([script:81-85](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L81-L85), [script:599-607](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L599-L607)):
- Master key 1 splits into three streams: optimisation, reverse sampling, and image noise.
- The reverse key is split twice and the results discarded. This reproduces the subkeys of the February 2026 samplers (commit c48963b; for how this relates to the checkpoint's date, see §8.14).
- It then provides the initial-state subkey and 40 Brownian keys, which **every variant shares**. The comparison is therefore paired.
- The image-noise stream matches the one in `plot_sde_generated_images`.

**Interpolation ablation** (the current default: `sampling_only=True`, `run_interpolation_ablation=True`):
- **Guard.** It requires a complete (100, 545,566) checkpoint whose SHA-256 is `37cfe46e…7249`, at `out/2026_07_08/problems/MNIST_generation/…/opt_per_time_plots/params_history.npy`.
- **Variants.** Parameters {raw, or Savitzky–Golay smoothed (window 9, cubic; k_6 smoothed in log space and then floored at 0.01)} × interpolation {linear, PCHIP} × sampler {SDE, ODE}: 8 runs.
- **Knots.** All 8 runs force solver steps onto the 98 interior knots via `step_ts`. The paper's sampler did not, so even raw/linear matches the paper's samples only to solver tolerance (§8.14).
- **Outputs.** Final states, solver stats, unclipped and clipped image grids, and a `manifest.json` with file hashes.

## 7. Notebooks

| Notebook | What it does | Settings | Remarks |
|---|---|---|---|
| [SM_local_R1_hyperParameterSearch](physical-diffusion-models/notebooks/sampling-from-gaussian-mixture_current_paper_SM_local_R1_hyperParameterSearch.ipynb) | 2D, 2-component Gaussian mixture (weights 0.6/0.4, means ±(1, 0.5), cov 1e-3·I); 2 oscillators + 1 coupling; analytic SM; reverse SDE vs equilibrium sampling (ES); error-vs-temperature study | Temp 0.1; 5 slices, ascending 0 → 5; lr = Temp with the default decay; batch 10,000; window 50; patience 10; reverse-SDE atol 1e-2, rtol 1e-3 | every slice stops at epoch 1000 and t = 1.25 is unconverged (§8.1); ES runs only to t₁ = 0.1 (§8.6); the temperature study pools Temp ∈ {0.01, 0.1, 0.75} and does not isolate temperature (§8.16); the forward-marginal check is plotted at D = 1 and saved (§8.8) |
| [CD1_R1](physical-diffusion-models/notebooks/sampling-from-gaussian-mixture_current_paper_CD1_R1.ipynb) | same mixture, CD-1 training | Temp 0.01; 10 slices, ascending; CD1_dt 1e-4; 3,000 noise samples; batch 2,048; lr 0.01 (cell 17 overwrites the 0.01·Temp set in cell 15) with the default decay; window 100; patience 20 | monitored loss is at T = 1 (§8.2); ES t₁ = 1000; ES cache path never matches (§8.6); reverse initial states are drawn from p₅, not p_∞; the saved parameter-evolution figure shows smoothed curves, while sampling uses raw ones (§8.8); outputs date from 2026-07-06 and predate the library cleanup (§8.16) |
| [Gradient_Tests](physical-diffusion-models/notebooks/Gradient_Tests.ipynb) | autodiff vs analytic vs memory-efficient SM gradients; CD-1 hockey stick | Temp 0.1; 2 oscillators; initial θ | errors ≈ 1e-13; coverage limits in §8.11 |
| [plot_sde_generated_images](physical-diffusion-models/notebooks/plot_sde_generated_images.ipynb) | re-plots an older MNIST SDE result in pixel space | — | the loaded states span ±86 in normalised units (§8.5); refers to a deleted script; its source checkpoint cannot be identified |
| [MNIST-preprocess](physical-diffusion-models/notebooks/MNIST-preprocess.ipynb) | old 6×6 preprocessing without balancing | — | superseded by `scripts/preprocess-mnist.py` |

**Conventions in the notebooks that differ from the MNIST script** (corrected and completed by the review):

| | 2D notebooks | MNIST script |
|---|---|---|
| Training direction | t = 0 → 5 | t = 5 → 0 |
| Learning rate | one value for all slices, decayed ×0.99 per 10 steps by `run_optimization`'s default (not constant, as these notes said earlier) | linear ramp by slice, ×0.95 per 6,000 steps |
| Initial θ | double well, couplings 0 | harmonic, c_lin = 1 on every pair |
| Temperature | 0.01 (CD-1), 0.1 (SM, Gradient_Tests) | 0.02 |
| Monitored loss | SM at T = 1 (CD-1, §8.2); SM at T = Temp (SM) | SM at T = Temp |
| Reverse initial state | p₅ of the data (CD-1); N(0, Temp) (SM) | N(0, Temp) |
| Reverse-SDE tolerances | atol 1e-2, rtol 1e-3 | atol 1e-6, rtol 1e-4 |
| σ_f | hard-coded to 1 in the SM reverse drift (*latent*) | a parameter, set to 1 |

The reverse-drift arithmetic and the parameter layout are identical. The interpolation matches only MNIST's raw/linear variant.

## 8. Issues

### 8.1 Early stopping never registers an improvement ✔ (affects every training run)

[learning_fns.py:366](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L366), [learning_fns.py:402-416](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L402-L416)

```python
best_moving_avg = float('inf')
...
relative_improvement = (best_moving_avg - current_moving_avg) / abs(best_moving_avg)   # (inf - a)/inf = nan
if relative_improvement > tolerance:      # nan > tol is always False
    best_moving_avg = current_moving_avg  # never reached
    patience_counter = 0
else:
    patience_counter += 1                 # on every check
```

**What happens.**
- Training stops at epoch e₀ + 100·(⌈patience⌉ − 1). Here e₀ is the first multiple of 100 at which the loss window is full.
- The stopping point does not depend on the loss trajectory, and `tolerance` has no effect.
- The message "Convergence reached" is misleading.

| Run | window | patience | stops at epoch |
|---|---|---|---|
| MNIST, t = 5 slice | 1000 | 100 | 10,900 |
| MNIST, t = 0 slice | 1000 | 300 | 30,900 |
| 2D SM notebook | 50 | 10 | 1,000 (seen in the saved output, all 5 slices) |
| 2D CD-1 notebook | 100 | 20 | 2,000 |

**Evidence that it matters.**
- In the SM notebook (cell 19), the t = 1.25 slice falls from 22,700 to 117 and is still falling steeply at the forced stop. Its best loss is 107, at epoch 993.
- The neighbouring slices sit at ≈ −10, and a Gaussian-mixture estimate of the optimum is ≈ −10 to −11.
- The excess loss of ≈ 120 equals ½E‖s_θ − s*‖², while E‖s*‖² ≈ 21. The score error at that slice is therefore several times the score itself.
- Two old MNIST training logs have exactly the line count these fixed budgets predict (§12.4, C4; unverified).
- For MNIST it is unknown whether 10,900–30,900 epochs are enough. The per-slice loss plots (`parameter_opt_evolution_timeidx_*.png`) would show it. Near t = 0 the step budget is the tighter limit (§8.13).

**Fix.**
- Record the first full-window average as the baseline instead of comparing against `inf`.
- Guard the division with `max(abs(best), eps)`, and flip the sign when `maximize=True`.
- Keep the old behaviour behind a flag, because the paper checkpoints were produced with the fixed budget.

### 8.2 The CD-1 notebook selects parameters with a T = 1 loss ✔

**The mismatch.** Cell 15 sets `loss_fn_per_batch = setup_score_matching_loss_per_batch(energy_fn)`, which uses the defaults k_b = T = 1. But `gradient_fn_per_batch = -CD1_gradient(…, D=Temp)` minimises $J_{T=0.01}$.

**Why it matters.**
- $J_1$ and $J_{0.01}$ have different minimisers: E ≈ −log p for the first, E ≈ −0.01·log p for the second.
- More precisely, $J_1(\theta)=J_{0.01}(0.01\,\theta)$. The monitor therefore ranks iterates by their distance to 100× the CD-1 target, and favours larger, more confining iterates.
- As CD-1 progresses, the monitored loss *rises*, yet `best_params` is chosen by it.

**Evidence (cell 19 output).** Best epoch per slice, as of the last printed epoch (the output is truncated at ~1,300 epochs; runs stop at 2,000):

| t | 0.56 | 1.11 | 1.67 | 2.22 | 2.78 | 3.33 | 3.89 | 4.44 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|
| best epoch | 113 | 11 | **0** | 1185 | 166 | 398 | 239 | **5** | 21 |

- At t = 0.56 the monitored loss climbs from −5.6 to −2.8 after epoch 113.
- For most slices the stored θ(t) is the previous slice's parameters plus 1 to ~400 Adam steps, out of ~2,000.
- The t = 0 slice starts from the initial double well and is still improving at epoch 1,200.

**Fix.** Call `setup_score_matching_LOSS_per_batch(energy_fn, k_b=1.0, T=Temp)`, or keep the final iterate. Commit 56ff44b says this notebook reproduces the paper figure, so the paper's CD-1 panel was probably produced this way.

### 8.3 "Best" parameters are the post-update iterate at the luckiest mini-batch ✔

- **One step late.** `loss_val` is evaluated before the update ([learning_fns.py:347](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L347)). But `best_params` stores the parameters after the update and projection ([learning_fns.py:379-396](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L379-L396)). History entries are paired the same way.
- **Selected on noise.** The choice is a minimum over ~10⁴ single-mini-batch estimates; for MNIST each batch is 512 fresh forward samples. That amounts to picking a random late iterate, and `best_loss` is biased low.
- **Size** *(review)*.
  - The one-step offset is negligible: one Adam step is at most about lr per coordinate.
  - The selection noise is real, but no larger than Adam's own jitter, because the learning rate is never annealed. It ends at 0.77–0.91 of its start value.
- **Effect.** Both feed slice-to-slice jitter into θ(t), which the warm start carries into the next slice.
- **Fix.** Return the final iterate or an EMA/Polyak average, or select on a fixed validation batch, and anneal the learning rate to ≈ 0 at the end of each slice.

### 8.4 The t = 1.25 slice in the SM notebook

This is a consequence of §8.1 and is described there. The final 2D sample statistics still look reasonable, which itself shows how forgiving the SDE is (§10.1):
- component weights are exact: 1200/800 of 2000;
- component means are within ~1%;
- the covariance of the whole mixture is within 2%.

However, the **per-component covariances come out 23–43% below** the target 1e-3 (cell 30), with correlations of 0.22 and 0.35 where the target has none. Part of that may be solver error: the reverse SDE runs with atol 1e-2, about a third of the component std ◐ (§8.16).

### 8.5 Out-of-range MNIST samples in an older SDE run ◐

- **Observation.** `plot_sde_generated_images` cell 9 (run on 2026-02-13) loads `out/final_states_non_smoothed_sde_memory_efficient_…npy`.
  - Its values span **[−86.2, 86.2]** in normalised units, or [−26.1, 26.4] in pixels. The data span [−0.41, 2.87].
  - The clipped grids hide this, and the unclipped grids are autoscaled per image (§8.15).
  - The file sits at a flat `out/` path that the current script never writes, and the notebook names a deleted script. Which checkpoint produced it cannot be determined *(review)*.
  - Whether the 2026_07_08 checkpoint behaves the same way is unknown. The script's check ([script:838-842](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L838-L842)) catches only non-finite values, not large finite ones.
- **Possible mechanism.**
  - Only the sextic self-term is constrained to confine.
  - The coupling terms are summed over up to 783 neighbours and can take either sign: the cubic $c^{\rm om}x_i^2x_j$, and the quartic $c^{\rm duff}$ term with either sign of $c^{\rm duff}$.
  - Balanced against $k^{(6)}x^6$, these sums can create minima far outside the data range. The nearly symmetric ±86 suggests coupled pixel pairs driven apart.
  - Score matching only evaluates scores where there is data, so it cannot see such wells. The near-null directions of the quadratic (§5.1) are exactly the ones the data leave free.
- **Diagnostics:**
  - Count the fraction of pixels outside [−0.41, 2.87] per sample, for each sampler and variant. The existing `final_states_*.npy` files are enough for this.
  - Compute ‖∇E‖ at the final states. A value near 0 means the trajectory is trapped in a well.
  - Save whole trajectories, to see when pixels leave the data range.
  - At small t, inspect θ: how many k_6 sit at the floor, the sign of k_duff, and per-pixel sums of c_om and c_duff.

### 8.6 The equilibrium-sampling baseline ✔

- **The two notebooks use very different budgets.** Each equilibrium-sampling (ES) run is a single trajectory from (0, 0) with 2,000 saved points.
  - CD-1 notebook (cell 25): t₁ = 0.1·n_traj·t_fwd = **1000**.
  - SM notebook (cell 25): t₁ = n_traj·t_fwd·1e-5 = **0.1**.
- At t₁ = 0.1 the "equilibrium sampling" row is far from equilibrium. If R1 compares SGM (sampling by the reverse diffusion) against ES, state the ES budget and make it comparable. One option is equal total simulated time, n_traj·t_fwd = 10⁴.
- **ES probably shows one mode only** ◐ *(review)*.
  - A model that reproduces the component covariance has a barrier of order 100 k_BT between the modes. The single ES trajectory then almost certainly never leaves its first well.
  - In that case no run length lets ES recover the mixture weights: the ES row shows mode trapping, not the learned weights. The normalised density panels already show those weights.
- **The CD-1 cache never reloads.** The notebook checks for `sde_samples_direct_simT_{t1}.npy` but saves `sde_samples_direct.npy`.

### 8.7 Latent bugs (no current caller) ✔

- **Wrong offsets in the memory-efficient θ-derivative helpers.**
  - `network_fns_memory_efficient.py` assumes the bias block comes *before* the couplings: `c_lin_start = 4*n_osc` ([:142-149](physical-diffusion-models/physical_diffusion_fns/network_fns_memory_efficient.py#L142-L149), [:240-244](physical-diffusion-models/physical_diffusion_fns/network_fns_memory_efficient.py#L240-L244)).
  - In the real layout bias is last and c_lin starts at 3·N, so every coupling entry would land N slots too far. That is wrong for every layout the module could be given.
  - Both helpers also loop in Python over all 180,810 pairs ([:192](physical-diffusion-models/physical_diffusion_fns/network_fns_memory_efficient.py#L192), [:286](physical-diffusion-models/physical_diffusion_fns/network_fns_memory_efficient.py#L286)).
  - Every caller discards them, and nothing tests them. They can be deleted.
  - The misconception is internal to that module, which dates from 2025-11-25. The MNIST comment that repeats it was written in 2026-07 *(review)*.
- **The 5-group branch writes into the bias block.**
  - In `network_fns.py` the 5-group branch sets `c_duff = zeros_like(c_lin)` ([:244-248](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L244-L248), [:365-369](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L365-L369)).
  - That makes `len(c_duff) > 0` true. A "c_duff" derivative is then scattered into the bias block (for pair_idx < N) or out of range, where JAX silently drops it.
  - The 6-group branch's offsets are correct *(review)*.
- **Only the 7-group layout works end to end.** `setup_energy_fn`, `gradient_duffing_network` and `trace_hessian_duffing_network` always unpack 7 groups ([:94](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L94), [:173](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L173), [:200](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L200)), even though the docstring advertises 5- and 6-group layouts.

### 8.8 Plotting ✔

- **float64 is switched on by import.** `plotting_fns.py` enables float64 globally when imported ([:14](physical-diffusion-models/physical_diffusion_fns/plotting_fns.py#L14)). This silently affects every importer, and is a big slowdown on GPUs with weak FP64. Keep it for reproduction, but make it explicit.
- **Figure 1's energy colorbar shows Ê/k_BT** *(review)*.
  - Both 2D notebooks pass θ(0)/Temp to the Figure 1 routine. Since E has no constant term, the energy panel shows Ê_θ(0)/k_BT: 100× (CD-1) or 10× (SM) the energy.
  - Yet the colorbar reads $\hat{E}_{\theta(0)}$ ([:620](physical-diffusion-models/physical_diffusion_fns/plotting_fns.py#L620)). The probability label, exp[−Ê/k_BT]/Z, is right, so the two labels are inconsistent.
  - Relabel the energy colorbar as $\hat{E}_{\theta(0)}/k_BT$.
- **Density plots use k_BT = 1.** `plot_energy_and_distributions` and `plot_sampels_energy_marginals_for_SGM_vs_ES` plot exp(−E) ([:76](physical-diffusion-models/physical_diffusion_fns/plotting_fns.py#L76), [:579](physical-diffusion-models/physical_diffusion_fns/plotting_fns.py#L579)).
  - The final 2D figures pass `params/Temp`, which is correct because E is linear in θ.
  - The "initial energy landscape" plots, however, are drawn at k_BT = 1.
  - With zero couplings the initial 2D model has four wells at (±0.786, ±0.786) *(review)*. Their density relative to the origin is 1.42 as plotted, but ≈ 33 at the SM temperature and ≈ 10¹⁵ at the CD-1 temperature.
- **Forward-marginal checks at D = 1** *(review)*.
  - The t = 5 sanity check in both 2D notebooks (cell 19) samples with D = 1 against an N(0, 1) overlay.
  - Training uses D = Temp, a process 10× (SM) or 100× (CD-1) narrower in variance.
  - SM saves this figure to `final_plots`.
- **Tautological normalisation checks** *(review)*.
  - The density panels compute exp(−E) without subtracting the minimum, and normalise over the plotting window only.
  - The printed "total prob ≈ 1" divides a sum by itself, so it can only ever reveal a NaN.
  - Overflow is *latent*: about 550 nats of margin remain for models of this family.
- **Figure 1 marginal panel** *(review)*.
  - The y-axis is fixed at (0, 13) ([:608](physical-diffusion-models/physical_diffusion_fns/plotting_fns.py#L608)).
  - p(x₂) is evaluated on the x₁ grid, which is harmless while both ranges are equal.
  - ◐ If the CD-1 equilibrium trajectory is stuck in one well (§8.6), its histogram peaks near 12.6 and can touch the cap.
- **The CD-1 parameter-evolution figure shows smoothed curves** *(review)*.
  - The saved figure plots the Savitzky–Golay curve. With window 10 → 9 on 10 knots, that is two overlapping cubic fits spliced between knots 4 and 5.
  - Sampling uses the raw linear interpolant (`use_smoothed_params = False`).
- **Very slow at MNIST scale** (training mode only):
  - `plot_forward_marginals` draws 784 histograms into a 56×56-inch figure at 300 dpi.
  - `visualize_connectivity_with_non_local_couplings` makes separate `jnp` calls for each of the 180,810 pairs.
  - `plot_parameter_as_fn_of_time` draws the (100 × 180,810) coupling histories as individual lines.
  - `reformat_optimization_results` and `plot_parameter_as_fn_of_time` fill arrays with eager `.at[i].set`, copying the whole history each time *(review)*. That is ≈ 44 GB of copies per slice.
- **Possible blank file.** `visualize_connectivity_with_non_local_couplings` calls `plt.show()` before `plt.savefig()` ([:467-470](physical-diffusion-models/physical_diffusion_fns/plotting_fns.py#L467-L470)), so the saved file can be blank when `plot_show=True`.
- **Mismatched axis.** In `plot_parameter_evolution`, `best_idx` is an epoch number, but the x-axis is the snapshot index (one snapshot per 100 epochs). The "Last 30 Steps" panel likewise counts snapshots.
- *Latent* *(review)*:
  - `visualize_connectivity` decodes the two ends of an edge with different divisors ([:377-378](physical-diffusion-models/physical_diffusion_fns/plotting_fns.py#L377-L378)). It is correct for the current 2×1 grid only by coincidence.
  - `interpolate_parameters` returns reversed axes for a 2-D `t` ([helper_fns.py:227-228](physical-diffusion-models/physical_diffusion_fns/helper_fns.py#L227-L228)).
- **MNIST image grids:** see §8.15.

### 8.9 Infrastructure and repo hygiene ✔

- **Stale launchers.** `run-mnist-generation`, `run-mnist-preprocessing` and `container/Makefile` call scripts that no longer exist (`duffing_network_learning_MNIST_sde*.py`), under `/n/fs/gen-osc/...`.
- **The CPU fallback in the MNIST Slurm script never takes effect.**
  - In [run_physical_diffusion_model_MNIST_neuronic.sh](container/run_physical_diffusion_model_MNIST_neuronic.sh), the GPU check sets a CPU fallback ([:39](container/run_physical_diffusion_model_MNIST_neuronic.sh#L39)).
  - The unconditional `export JAX_PLATFORMS=cuda,cpu` a few lines later ([:44](container/run_physical_diffusion_model_MNIST_neuronic.sh#L44)) overwrites it.
  - Moreover, the reviewer's reading of JAX's `xla_bridge` (not run here) is that an explicit platform list makes JAX fail loudly when one platform is missing, rather than fall back *(review)*.
- **Resource requests.**
  - The MNIST job requests 40 CPUs and no `--mem` for a single-GPU JAX job ([:12](container/run_physical_diffusion_model_MNIST_neuronic.sh#L12)).
  - The CPUs were raised to get memory: af3a1dc says "doubled the number of cpus requested, since it failed out of memory" *(review)*. Request `--mem` explicitly instead. `jobstats <jobid>` shows what a run actually used.
  - The preprocessing job asks for 12 CPUs, a GPU and 40 h for a few minutes of work.
- **Submit from `container/`.** The image path and the bind directory are both relative to it.
- **Label mismatch.** `preprocess-mnist.py` still builds labels [0,1,2,3] ([:114](physical-diffusion-models/scripts/preprocess-mnist.py#L114), left over from 2f73edb), while the driver uses [0,1] ([script:88](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L88)).
- **Checkpoint writes.**
  - Checkpointing rewrites the entire growing history after every slice. Over a 100-slice run that is ≈ 22 GB written to the shared filesystem; writing one file per slice would reduce it to ≈ 0.44 GB.
  - The history is written before the index file, so a kill between the two leaves them inconsistent *(review)*.
- **The resume path** (*latent*, *review*):
  - The optimisation key is re-derived from the master key at each launch and then split from `start_t_idx` onward ([script:84](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L84), [script:330](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L330)). Resumed slice k therefore gets the subkey that slice 0 got. Use `jr.fold_in(key, t_idx)`.
  - Training mode turns the loaded history into 54.6 M Python floats, and converts the whole list again after every slice.
  - If `current_time_index.txt` is missing, the script starts from scratch and overwrites `params_history.npy` at its first save.
- **Solver limits** *(review)*.
  - `max_steps = 1e9` with no `dtmin` ([network_fns.py:522](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L522), [:557](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L557)).
  - A stalled trajectory holds the whole vmapped batch of 40 until the 90 h limit. Solver stats are written only after a solve returns, so nothing would show why.
  - Use `max_steps` ≈ 1e6–1e7.
- **Compile cost** ◐ *(review)*.
  - `solve_SDE` and `solve_ODE` hard-code `args=()` ([network_fns.py:519](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L519), [:554](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L554)). The parameter table therefore enters the drift through a closure, as a trace-time constant: 436 MB for linear interpolation, 873 MB for PCHIP.
  - Each of the 8 ablation solves builds a new closure and compiles anew.
  - Passing the table through `args` should cut compile time and host memory.
- **JAX FutureWarning** *(review)*.
  - Every 2D-notebook cell that calls `solve_SDE` prints "Attempting to hash a tracer will lead to an error in a future JAX release". The hash happens inside the diffrax/equinox trace.
  - It is harmless with the pinned stack (jax 0.4.33, diffrax 0.6.0, equinox 0.11.5).
  - Before any upgrade, find the source with `python -W error::FutureWarning`.
- **`.gitignore`.**
  - The `.pdf` and `.npz` patterns lack a leading `*`.
  - `out/` and `data/` are ignored, so the paper checkpoint and the data are unversioned. The SHA-256 guard is the only record of provenance.
- **Claude permissions.** `.claude/settings.local.json` allows `Bash(nohup python3 *)`. On this cluster that would run computation on the login node; see `/etc/agents.d/PRINCETON_CS_AGENTS.md`.

### 8.10 Documentation and cosmetic issues ✔

- **Sign in two docstrings.** The `learning_fns.py` docstrings ([:61](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L61), [:137](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L137)) write $L=\frac1{kT}\mathrm{Tr}\nabla^2E-\frac1{2kT^2}\lVert\nabla E\rVert^2$. That is the *negative* of the implemented loss; the gradients they state do match the code.
- **`normalize_samples`.** Its docstring says "per dimension", but it uses one global mean and std. The clipping relies on that global behaviour.
- **`sample_forward_process`** (corrected).
  - The docstring says the final variance is σ_f² *if* D(1 − e^{−2βt/σ_f²}) = 1. That statement is correct.
  - However, the condition cannot hold for D = Temp < 1. In every training caller the final variance is D·σ_f² = 0.02.
  - An earlier version of these notes wrongly said the docstring omits σ_f².
- **Parameter-layout comment** *(review)*.
  - [script:235-236](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L235-L236) lists bias with the per-oscillator groups, but in the flatten order bias comes last.
  - Anyone slicing by the comment would get couplings instead of biases.
- **`run_optimization` docstring** *(review)*:
  - it documents `samples`, but the argument is `sampler`;
  - it says "any pytree", but the code needs a flat array;
  - it says patience counts "consecutive windows", but it counts checks, one per 100 epochs;
  - it says history is kept "at each optimization step", but it is kept every 100 epochs;
  - `lr_decay_rate`, `lr_decay_steps` and `constraint_indices` are undocumented.
- **`CD1_gradient` docstring** *(review)*. It misnames its argument and leaves out the −1/(D²dt/2) scaling and sign. Only the inline comment at [:275](physical-diffusion-models/physical_diffusion_fns/learning_fns.py#L275) explains them.
- **Output-folder names.** They encode `tol_…_patience_…_epochs_100000`, which does not describe the actual stopping rule (§8.1).
- **Key reuse.** At t_idx = 0 the forward-marginal sanity plot uses `optimization_key` and then splits the same key again ([script:330-334](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L330-L334)). This is harmless.
- **Stale comments** *(review)*.
  - `plot_sde_generated_images` cites the deleted `duffing_network_learning_MNIST_sde_memory_efficient.py`. That script de-normalised its samples; the current one does not.
  - `setup_overdamped_SDE` documents `T` for the argument `Temp` ([network_fns.py:457](physical-diffusion-models/physical_diffusion_fns/network_fns.py#L457)).
  - The MNIST-preprocess notebook writes filenames without `_balanced_`.
- **Plot label.** SM notebook cell 31 labels the means as μ_k[d,d].

### 8.11 Test coverage

Gradient_Tests checks one 2-oscillator, 1-pair network at one parameter point. It only prints norms, with no assertions. Because the energy is linear in θ, that is a decent check of the formulas, but the review found these gaps *(review)*:
- **Zero couplings and biases.** At the test point all couplings and biases are 0.
  - On the `network_fns` analytic path, the coupling part of ∇E is then multiplied by 0, so a mistake there would go unnoticed. The θ-Jacobian columns are still exercised.
  - The MNIST path differentiates the memory-efficient ∇E and tr∇²E with respect to θ, so it does see the coupling formulas, but only for one pair.
- **Gradients only.** Loss values are never compared.
- **Broken helpers untouched.** The memory-efficient outputs 3 and 4, the helpers of §8.7, are never called.
- **CD-1 operating point.** The CD-1 hockey stick runs only at D = 0.1 with 100 noise samples. Its `dt_values = logspace(-10, 0, 10)` gives 10 points from 1e-10 to 1, not the "20 points from 1e-7 to 0.1" of its comment.

Also never exercised:
- scatter accumulation over several pairs, since each node appears in only one pair;
- the 28×28 layout and its offsets;
- the reverse-drift arithmetic;
- interpolation and smoothing.

Cheap additions that would close most of this:
- On a 4×4 lattice with `n_neighbour_couplings=2` and random nonzero θ, compare the analytic ∇E and tr∇²E with `jax.grad` and `jax.hessian`. Compare loss values too, and assert thresholds.
- Check that $-\nabla E_{\theta_{\rm rev}}(x)=x/\sigma^2-2\nabla E_\theta(x)$.
- Run the hockey stick at D = 0.01.

### 8.12 Retraining with the current flags deletes the paper checkpoint ✔ *(review; latent, but destructive)*

- **How.**
  - `checkpoint_date = "2026_07_08"` ([script:146](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L146)) sets the output folder in every mode ([script:187](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L187)).
  - The guards fire only for the ablation ([script:150](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L150)), or for `sampling_only` together with `start_from_scratch` ([script:272](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L272)).
  - The natural retraining setting (`start_from_scratch=True`, `sampling_only=False`, `run_interpolation_ablation=False`) therefore reaches `shutil.rmtree(output_dir)` ([script:275-280](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L275-L280)) on the paper folder.
- **What would be lost.**
  - `params_history.npy` (SHA `37cfe46e…7249`) and `current_time_index.txt`.
  - `interpolation_ablation_v1/`, with the final states, solver stats, shared initial states and Brownian keys, and `manifest.json`.
  - Only the ablation PNGs under `final_plots/` would survive. `out/` is git-ignored, so git cannot restore anything.
- **The opposite trap.**
  - With `start_from_scratch=False` and `sampling_only=False`, the complete checkpoint makes the script skip training silently.
  - That happens even if you changed hyperparameters the folder name does not encode: `lr_decay_rate`, `lr_decay_steps`, `balanced`, the k_6 floor, or the initial values.
- **Fix.**
  - Use `checkpoint_date` only when `sampling_only=True`.
  - Refuse to `rmtree` a folder whose `params_history.npy` has the pinned SHA.
  - Keep a read-only copy of the checkpoint outside `out/`.

### 8.13 Near-data slices: step budget vs optimum — hypothesis tested and REFUTED

**Measured result** (job 4181431; `scripts/diagnose_near_data.py`; output in `out/diagnostics/near_data_check/4181431/`):

| t | 0.20 | 0.10 | 0.05 | 0 |
|---|---|---|---|---|
| blank-pixel stiffness / optimum (median) | 1.01 | 1.00 | 1.01 | 1.03 |
| share from self terms / couplings | 0.40 / 0.61 | 0.25 / 0.75 | 0.16 / 0.85 | 0.06 / 0.97 |
| uniform-background-mode stiffness / optimum | 0.82 | 0.70 | 0.63 | 0.57 |
| relative score error (blank pixels) | 0.027 | 0.030 | 0.036 | 0.050 |
| global scale test R (1 = optimum) | 1.006 | 1.005 | 0.986 | 0.976 |

- k_lin of blank pixels does stay near 0.86, as predicted. But Σ2c_lin over each pixel's ~460 pairs grows to 19.4 and supplies the stiffness.
- No parameter group comes close to its budget: at t = 0 the largest movement is 0.71× the budget (c_lin), and the 90th percentiles are far lower.
- The coupling route leaves the *uniform* background mode too soft, since (x_i − x_j)² terms cannot stiffen a common shift. That fits the background offset of ≈ 0.07 normalised units (≈ 0.02 in pixel units) seen in all samples (§8.15).
- Conclusion: retraining the last slices for the learning rate's sake is not worth it. The analysis below is kept for the record.

**The budget.**
- The learning rate is `linspace(2e-5, 2e-3, 100)[::-1]` ([script:108](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L108)): 2e-3 at t = 5, 4e-5 at t = 0.05, 2e-5 at t = 0. It decays smoothly by 0.95 per 6,000 steps, and each slice runs the fixed length from §8.1.
- Adam moves each coordinate by about lr per step at most. Summed over a slice, that gives the most any single parameter can move:

| slice t | 5.0 | 0.45 | 0.20 | 0.15 | 0.10 | 0.05 | 0 |
|---|---|---|---|---|---|---|---|
| max movement | 20.8 | 5.2 | 2.7 | 2.1 | 1.6 | 1.1 | 0.54 |

**The target.** For an always-blank pixel the exact optimum is a harmonic well with:
- precision k = T/v(t), where v(t) = 0.00108·e^{−2t} + 0.02·(1 − e^{−2t});
- bias −k·μ(t), where μ(t) = −0.4096·e^{−t}.

| t | 0.10 | 0.05 | 0 |
|---|---|---|---|
| optimal k_lin | 4.41 | 6.91 | 18.57 |
| optimal bias | 1.63 | 2.69 | 7.60 |

**The gap.**
- The last interval needs Δk_lin = +11.7, which is 21× the 0.54 budget, and Δbias = +4.9, which is 9×.
- The slice before cannot keep up either: it needs +2.5 and has 1.08.
- Even starting from the optimum at t = 0.10, k_lin can reach at most ≈ 6 at t = 0, against 18.6.

**Caveat.** The cap is per coordinate. A pixel's stiffness can also come collectively from its ~460 couplings, each moving a little (2c_lin per pair, or 2c_om·x_j). That would come at the cost of spurious structure between pixels. Whether the score near t = 0 is actually too soft is therefore empirical.

**Why it matters.**
- The lag is smooth in t, so Savitzky–Golay smoothing cannot remove it (§10.2).
- Under the ODE a precision error accumulates over the path (§10.1).
- It fits commit 34508c9 ("having more timesteps helps"): more slices near t = 0 mean more steps there.

**Check** (seconds, run under Slurm):
- For always-blank pixels, compare k_lin, the bias, and the diagonal Hessian ∂²E/∂x_i² at the data value with T/v(t), over the last 15 slices.
- Histogram |θ₉₉ − θ₉₈| per parameter group. A pile-up near 0.5 means the budget was exhausted.

**Fix** for retraining:
- Decouple the learning rate from t, or scale it to the expected change of the optimum.
- Anneal to 0 within each slice, and fix §8.1.
- A cheap test: re-fit slices 95–99 from the stored θ with lr ≈ 1e-3 and a cosine decay. Then compare the SM loss on a fixed validation batch, and the samples.

### 8.14 Reproducing the paper's samples ✔ / ◐ *(review)*

- **The ablation's sampler is not the paper's.**
  - All 8 ablation runs pass the 98 interior knots as `step_ts`, an option added in 98793e0 (2026-08-22).
  - The paper-era samplers (80d7e40, fd225e8, 411c28e) and the script's non-ablation branch pass none. Keys, initial states, tolerances, dt0 and the Brownian path are otherwise identical.
  - So the raw/linear SDE arm reproduces the paper's samples only to solver tolerance, not bit for bit.
  - There are no paper ODE samples to compare against: the ODE sampler first appeared in b0af2f3.
  - Comparisons *within* the ablation are unaffected.
  - Fix: add a raw/linear arm with `step_ts=None` and compare it with the paper's final-states file.
- **RNG and the checkpoint's date.**
  - The current splitting reproduces the subkeys of the February-2026 samplers: 80d7e40 and 925baa9, which share today's configuration (n = 14), and fd225e8.
  - The script committed on 2026-07-08 is 411c28e, used until c48963b on 2026-07-21. It drew the initial states from split #1 and the Brownian keys from split #2.
  - If `out/2026_07_08` came from a fresh run of the committed code that day, any samples saved in it used the 411c28e draws. Git cannot tell.
  - Fix: record provenance (commit, date, host) next to the checkpoint, and compare with any `final_states` file inside the 2026_07_08 folder.
- **Bitwise determinism on GPU** ◐.
  - The drift is `jax.grad` of a gather over 180,810 pairs. Its backward pass scatter-adds 361,620 contributions into 784 slots, about 461 per pixel.
  - XLA:GPU implements this with atomics, so the summation order, and hence the last bits, can vary. Under adaptive stepping that can flip accept/reject decisions.
  - The manifest's final-state SHA-256s are therefore probably not reproducible across reruns, even with identical keys. No `XLA_FLAGS` are set.
  - Test: run one variant twice and compare. Fix: `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, or compare with a tolerance.
- **Library versions are not recorded.**
  - The manifest stores file hashes, but not the jax/jaxlib/diffrax/equinox/numpy/scipy versions, the device, or PRNG flags.
  - `requirements.txt` pins jax 0.4.33, but not the `nvidia-*` wheels.
  - ◐ Per the reviewer (not checked here), JAX ≥ 0.5 changed a PRNG default that alters `jr.split`/`jr.normal` output for the same seed. The saved `shared_*.npy` files would at least reveal that.
- **Resumed training** replays the slice-0 RNG stream (§8.9).
  - A checkpoint produced across several launches therefore cannot be reproduced by an uninterrupted run.
  - The repo does not record whether that happened, and the job limit is 90 h.

### 8.15 How sample quality is judged ✔ *(review)*

- **The grids are autoscaled per image.**
  - The image grids use `imshow` without `vmin`/`vmax` ([script:798](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L798)), so every image is scaled to its own min and max.
  - That hides offset and gain errors and a uniformly higher background noise level, which is exactly the signature §10.1 predicts for the ODE.
  - One outlier pixel compresses the rest of an image. The clipped grid is also autoscaled after clipping.
- **The unclipped grid mixes scales.** It puts raw-pixel true images ([script:807](physical-diffusion-models/scripts/physical_diffusion_model_MNIST.py#L807)) next to normalised samples. Autoscaling makes that invisible, but it is confusing.
- **No metric anywhere.** The comparison uses 40 samples. The manifest stores hashes, solver stats and paths only.
- **Fix:**
  - Plot both rows in pixel space with `vmin=0, vmax=1`.
  - Use ≥ 500 paired samples.
  - Report: the pixel histogram; the fraction outside the data range; the per-pixel mean and std of always-blank pixels, against −0.4096 and 0.0328; nearest-neighbour distance to the training set; a 0/1 classifier's confidence; and E(x; θ(0))/T.

### 8.16 The 2D notebooks: further issues ✔ *(review)*

- **The temperature study (SM) does not isolate temperature.**
  - Temp was edited and the notebook re-run; the saved points are 0.01, 0.1 and 0.75.
  - lr = Temp is *not* a confound (§5.1).
  - What does change with Temp: the normalised start θ₀/T (75× across the range), the floor 0.01/T, and how far each slice gets in its fixed 1000 epochs, with the lr decayed to 0.37×. At Temp = 0.1, t = 1.25 is unconverged.
  - The JSON records hold only Temp: no lr, slice count, seed or commit.
  - The metric uses 2,000 hard-clustered samples from one seed. The weight's standard error is ≈ 0.011, and an exactly-zero error cannot be drawn on the log axis.
  - Fix §8.1 first, then repeat over seeds with error bars.
- **Reverse-SDE tolerance.**
  - atol 1e-2 and rtol 1e-3, against a component std of 0.032. MNIST uses 1e-6 and 1e-4.
  - ◐ The low per-component covariances (§8.4) may partly be solver error. Re-run cells 23–30 with atol 1e-5 to separate it from model error.
- **The CD-1 notebook was not re-run after the library cleanup.**
  - Its outputs come from 56ff44b (2026-07-06), and the 07-08 cleanup commits did not touch it.
  - Its imports resolve, and the code path it uses appears unchanged.
- **CD-1 reverse initial states** come from the data's p₅ rather than from N(0, Temp).

## 9. Checked and correct ✔

- **Analytic derivatives:** every formula in both network files, and the 7-group θ-Jacobians with their offsets.
- **Connectivity:** unique, sorted i<j pairs within Chebyshev radius 1…n *(review)*.
- **Score matching:** all four implementations agree; CD-1's sign is handled correctly; the hockey-stick test behaves as expected.
- **`run_optimization`:** Adam, mask, projection and schedule all work as intended, apart from §8.1 and §8.3.
- **Index arithmetic in the script:** the k_6 block used for the projection and the k_lin block used for θ_lin are right, despite the misleading layout comment *(review)*.
- **Forward process:** it uses the exact OU transition, with variance D·σ²·(1 − e^{−2βt/σ²}), and the initial reverse states match p_∞.
- **Reverse samplers:** the θ_rev and θ_pf arithmetic, the time map τ(t), the noise √(2k_BT) and the initial law are all correct. Two review agents re-derived them independently.
- **Interpolation:**
  - `interpolate_parameters` sorts the descending time grid and clamps outside it.
  - `interpolate_parameters_pchip` matches SciPy's `PchipInterpolator`: the weighted harmonic mean for interior slopes, the three-point endpoint rule with both limiters, the Hermite basis, the interval search, and clamping. Clamping is never exercised.
- **Smoothing:** the `smooth_parameters` rewrite in 98793e0 is equivalent to the old per-column loop. It clips to an odd window the same way and uses SciPy's default `mode='interp'`.
- **RNG:** the historical splitting is reproduced, and initial states and Brownian keys are shared across variants. The image-noise stream matches the plotting notebook.
- **Checkpoint guards:** the shape check for a complete history, and the exact SHA-256 required by the ablation.
- **diffrax usage:** SRA1 with SpaceTimeLevyArea and VirtualBrownianTree is consistent with the diffrax 0.6 API.
- **Data naming:** consistent between the preprocessor and the driver.

## 10. Open research questions: mechanisms and diagnostics

### 10.1 Why the PF-ODE does worse than the reverse SDE

Your ablation (commits b0af2f3 → 98793e0) already rules out four causes:
- ODE integration error: tolerances were tightened from 1e-4/1e-6 to 1e-7/1e-9;
- the interpolation scheme: linear vs PCHIP;
- kinks at the knots: steps forced onto them with `step_ts`;
- parameter jitter: Savitzky–Golay smoothing.

The review independently ranked the remaining candidates the same way and quantified the first two. Most plausible first (all ◐ until tested):

1. **The ODE has no corrector.**
   - By §6, the SDE is the ODE plus Langevin dynamics on p_θ(τ). That extra term keeps pulling samples toward the model's own distribution at each slice.
   - In doing so it absorbs two kinds of error: errors in θ(τ) itself, and the fact that slices fitted one by one need not be consistent with the forward Fokker–Planck equation.
   - **The consistency condition** *(review)*. A family p_τ ∝ e^{−E(x;θ(τ))/T} follows the forward process exactly when ∂_τE − x·∇E + ‖∇E‖² − TΔE does not depend on x.
     - Gaussian check: k = T/v obeys k′ = 2k − 2k².
     - Independent per-slice fits with a misspecified family break this condition, and so does interpolating linearly in θ between knots.
   - **The SDE has a bound; the ODE does not.**
     - By Girsanov, the SDE's KL to the data is bounded by T∫E‖s_θ − s‖²dτ plus the prior mismatch. That is directly the sum of the per-slice SM excess losses (Song et al., NeurIPS 2021).
     - The PF-ODE has no such bound. Its likelihood also depends on errors in the score's derivatives (Lu et al., ICML 2022).
   - Model mismatch is certain here, since a polynomial pair energy cannot represent MNIST.
2. **Errors in the learned precision accumulate along the ODE path** *(review; it absorbs this list's earlier "near-cancelling drift" item)*.
   - Take a near-Gaussian component with learned precision k̂(τ).
     - Under the ODE, d ln|y|/ds = 1 − k̂, so its final log-std error is −∫δk̂ dτ over the *whole* path.
     - Under the SDE, the relative variance error relaxes at rate ≈ 2k and ends near −ε for a relative precision error ε. It is set locally, near τ = 0.
   - For always-blank pixels ∫₀⁵ k dτ = 6.5.
     - A uniform error ε therefore scales the ODE's final std by e^{−6.5ε} and the SDE's by ≈ e^{−ε}.
     - At ε = −20%: 3.6× vs 1.3× the correct background noise.
   - **Large τ.** At large τ, k ≈ 1 and the ODE drift nearly cancels. A relative error ε there gives a velocity εx of the same size as the true one.
     - Training at t = 5 must also unlearn c_lin = 1 on 180,810 pairs, and any leftovers drive the ODE directly.
     - *Check 1:* inspect row 0 of the checkpoint, i.e. θ(5) − θ_lin.
     - *Check 2:* compare the final loss of the t = 5 slice with the exact optimum $J^*=-\tfrac12\mathbb E\lVert\nabla\log p_\infty\rVert^2=-392/T=-19{,}600$, accurate to ~0.2%. The excess equals ½E‖s_θ − s*‖².
   - **Sources of systematic precision error here:**
     - the step-budget lag near t = 0 (§8.13);
     - unconverged slices (§8.1);
     - interpolation bias in the last interval (§10.3);
     - Savitzky–Golay distortion (§10.2).
3. **Forces off the data support** (§8.5). The near-null directions of the SM quadratic (§5.1, §10.2) are not constrained by the data, but they set the forces wherever ODE trajectories drift.
4. **Prior mismatch along high-variance directions** *(review)*.
   - N(0, T) differs from p₅ by at most 0.89 nats overall (§6).
   - Along a direction with data variance λ, the residual signal-to-noise at t = 5 is 0.00227·λ.
   - If the top principal components have λ ≈ 100–160 (not computed), the terminal variance deficit at λ = 160 is 27% for the ODE against 7% for the SDE.
   - Check: the eigenvalues of the standardised training covariance.
5. **Knots too coarse near t → 0** (§10.3).
6. **The display** (§8.15) can exaggerate or hide the difference.

Solver error is excluded: you tightened the tolerances, and the review found the drift, diffusion and initial law correct.

**Diagnostics, cheapest first:**
- **Pixel statistics on existing outputs** (no new runs).
  - Build a mask of always-blank pixels.
  - For each of the 8 variants, compare those pixels' mean and std across the 40 final states with −0.4096 and 0.0328. Candidate 2 predicts an ODE std well above the SDE's.
  - Also count out-of-range pixels (§8.5).
- **Read the precision lag off the checkpoint** (§8.13). Then predict the ODE's spread as exp(−∫δk̂ dτ) and compare it with the observed spread.
- **Corrector test from the saved ODE outputs** *(review)*.
  - Run Langevin dynamics at θ(0) for 0.02, 0.1 and 0.5 time units, starting from the saved ODE final states. θ(0) is row 99 of the checkpoint, because rows run in descending t.
  - If samples reach SDE quality quickly, the ODE's problem is local inconsistency rather than global transport.
  - The in-loop version: insert a few Langevin steps at fixed τ (drift −∇E_θ(τ), noise √(2k_BT)) between ODE steps.
- **Liouville consistency** *(review)*.
  - Augment the ODE state with l, where dl/ds = tr∇²E(x; θ_pf(s)) and l(0) = −‖x₀‖²/(2T) − (N/2)ln(2πT). Then l is the exact log-density of the ODE's own law along each trajectory. `trace_hessian_fn` already exists.
  - If the slices were consistent, l + E(x; θ(τ))/T would be the same for every trajectory at each knot. How its spread grows with τ shows where the inconsistency arises.
- **Run both samplers with the exact score.**
  - The forward marginal of the finite training set is a Gaussian mixture, $p_t(x)=\frac1M\sum_m\mathcal N(x;e^{-t}x_0^{(m)},v_tI)$, with $v_t=\sigma_{\rm pre}^2e^{-2t}+D(1-e^{-2t})$ and M = 11,846. Its score can be computed exactly; a 1–2k image subset is enough.
  - If both samplers do well with it, the ODE's gap comes from the learned score rather than from the sampler.
  - Run it once continuous in τ and once interpolated linearly between the same 100 knots. That isolates the time discretisation.
  - The same computation gives the learned score's error at every slice, E‖s_θ − s_exact‖².
- **Repeat ODE vs SDE on the 2D mixture notebooks.** There p_t is known exactly, the runs are cheap, and they show whether the gap already appears in 2D.

### 10.2 Why Savitzky–Golay smoothing did not help ◐

- **On the exact optimum, the smoothing distorts the slices that matter most.**
  - Near t → 0 the optimal parameters change fastest. For an always-blank pixel k = T/v(t) has a pole just 0.55 knot spacings before t = 0.
  - A 9-knot window spans 0.40 time units, and `mode='interp'` fits *one* cubic to the first 9 knots.
  - Applied to that exact profile, SG(9,3) is off by −6.5%, +33.6%, +1.4%, −31.9% and −33.9% at the first five knots *(review)*.
- **But probably not much on the checkpoint.** Because of §8.13, the trained trajectory near t = 0 is likely rate-limited and smooth, which a cubic follows easily. The actual distortion of the checkpoint is unknown and probably smaller.
- **What the smoothing cannot remove** *(review)*.
  - *Lag:* the schedule lag of §8.13 is smooth in t, and SG reproduces any cubic trend over 9 knots.
  - *Random-walk drift:* in weakly identified directions, warm starting without restoring curvature makes errors add up from slice to slice. That drift is low-frequency, and a 9-tap filter passes it through.
  - *Not the cause of that drift:* the fresh Adam state per slice. Its reset kicks add at most 0.1 per coordinate over all 100 slices.
  - *Inconsistency:* smoothing does nothing about the Fokker–Planck inconsistency (§10.1).
- **The jitter it does remove mostly does not matter.**
  - Slice-to-slice jitter comes from the selection rule (§8.3), the fixed-length runs (§8.1) and warm starting.
  - It lies mostly along near-null directions of the quadratic (§5.1). Those barely change the score on the data, so smoothing them does not help.
  - Off the data support, however, those same directions set the forces (§10.1, item 3).
- **Diagnostics** *(review)*:
  - per slice, the SM loss of raw vs smoothed θ on one fixed large validation batch;
  - the relative score change E‖∇E_smooth − ∇E_raw‖²/E‖∇E_raw‖²;
  - the lag-1 autocorrelation of the increments θ_k − θ_{k−1} per parameter group: ≈ −0.5 for white jitter, ≈ 0 for a random walk, near +1 for a smooth trend.
- **Better alternatives:**
  - fix the selection rule (§8.3) and the step budget (§8.13);
  - average iterates within each slice (EMA);
  - use the same x₀ indices and noise for every t (common random numbers), add a ridge term, and solve each quadratic with CG. Then θ*_λ(t) = T(A_t + λI)^{−1}b_t is smooth in t by construction *(review)*;
  - couple neighbouring slices during training, either with a penalty λ‖θ_k − θ_{k+1}‖², or with a smooth-in-time parametrisation trained on random t. For example, use B-splines in log-SNR; the loss stays quadratic in the coefficients.

### 10.3 Time grid ◐

- **Interpolation error near t = 0**, for the exact profile of an always-blank pixel *(numbers checked by the review)*:
  - At mid-interval the curvature is 9.95. Linear interpolation gives 12.7 (+28%); PCHIP gives 11.2 (+13%).
  - Integrated over the last interval, ∫k dt is 0.545 exactly and 0.643 with linear interpolation. That is an extra contraction by a factor 0.906 under the ODE.
  - The checkpoint's own error is unknown, because its near-data knots are rate-limited (§8.13).
- **Correction to an earlier claim.**
  - These notes said that half of the knots lie beyond t = 2.5, "where little changes". That holds per pixel.
  - Along class-separating directions with λ ≈ 100–160, the signal-to-noise ratio at t = 3 is still 12–20, and it crosses 1 only near t ≈ 4.3–4.5 (§4). Those knots resolve the 0-vs-1 decision.
- **The old exponential grid** (fd225e8) erred the other way. It put 71 of 100 knots below t = 0.028.
- **Better options:**
  - knots uniform in log-SNR, computed from the data spectrum;
  - for a quick sampling-only test, interpolate ψ(t) = v_ref(t)·θ(t) and divide v_ref back out, which is exact for constant pixels;
  - a cheap retraining test: add 4–8 slices in (0, 0.05), warm-started from θ(0) with lr ≈ 1e-3. The CD-1 notebook already has an `exponential_time_pts` switch.

## 11. Running and reproducing

- **Where to run.** All computation goes through Slurm; the login node is only for editing and submitting.
  - Run `cd container && sbatch run_physical_diffusion_model_MNIST_neuronic.sh`.
  - The image `container/container_image.sif` is built from `container.def` or `build.def`.
- **Data.** Set `labels` in `scripts/preprocess-mnist.py`, then run `sbatch run_mnist_preprocess_data_neuronic.sh`. tfds downloads MNIST on first use.
- **Driver modes**, set by flags at the top of the script:

| Mode | `start_from_scratch` | `sampling_only` | `run_interpolation_ablation` |
|---|---|---|---|
| train from scratch | True | False | False — **set `checkpoint_date = None` first** (§8.12) |
| resume training | False | False | False — **set `checkpoint_date = None`** unless you mean to resume the paper run |
| sample from a complete checkpoint | False | True | False |
| interpolation ablation (current setting) | False | True | True (needs the exact checkpoint SHA) |

- **Output root:** `out/<checkpoint_date | today>/problems/MNIST_generation/labels_[0, 1]_resolution_28_x_28/n_couplings_14_Temp_0.02_energy_fn_type_6th_order_duffing_coupling/added_gaussian_noise_std_0.01_key_seed_1_additional_rescaling_1/training_method_…/`
  - `opt_per_time_plots/` holds the checkpoint, the per-slice plots and the ablation outputs.
  - `final_plots/` holds the sample grids.
- **Reproduction runs** (§8.14):
  - export `XLA_FLAGS=--xla_gpu_deterministic_ops=true`;
  - record library versions, device and commit in the manifest;
  - compare final states with a tolerance rather than by hash.
- **Notebooks.** The 2D notebooks write to `../out/<today>/…` and reload a cached `params_history.npy` if one exists.

## 12. Independent review

### 12.1 How it was done

- **Six readers**, one per area, read the code line by line without seeing these notes. The areas were the physics core, learning, helpers and plotting, the MNIST pipeline, the notebooks, and the research questions (ODE vs SDE, smoothing, time grid).
- **An adversarial verifier per area** then re-read the cited code and re-derived the maths. It checked whether each code path actually runs, and tried to refute every finding.
- **Claims from these notes.** Eleven of them were slipped into the verifiers' lists as hypotheses to break.
- **A completeness critic** looks for anything missed (§12.4).
- **Nothing was executed.** The numbers come from derivation, arithmetic and the outputs saved in the notebooks.

### 12.2 Outcome

| Area | Verdicts | Confirmed | Corrected in detail | Refuted |
|---|---|---|---|---|
| Physics core | 17 | 13 | 4 | 0 |
| Learning | 15 | 11 | 4 | 0 |
| Helpers and plotting | 18 | 14 | 4 | 0 |
| MNIST pipeline | 18 | 15 | 3 | 0 |
| Notebooks | 23 | 19 | 4 | 0 |
| Research questions | 19 | 12 | 7 | 0 |
| **Total** | **110** | **84** | **26** | **0** |

- **The eleven claims from these notes:** 9 confirmed, 2 corrected in detail. The corrections were the "ascent direction" wording in §5.2 and the 6-group offsets in §8.7.
- **How independent this is.**
  - The readers never saw these notes. Their prompts did point at the relevant code, for example "trace `best_moving_avg` on the first few checks" or "which loss is monitored", but never gave the answers.
  - Readers without such a pointer still converged on the main bugs. Four of the six readers found the early-stopping NaN, three of them unprompted. Five found the broken memory-efficient helpers, again three unprompted.
  - Their stop-epoch formulas match §8.1.
- **Corrected in detail** means the defect is real, but a number, scope or wording was off. The text above uses the corrected versions.

### 12.3 What changed in these notes

**Corrections to things these notes had wrong:**
- §7: the 2D notebooks do not use a constant learning rate. They inherit `run_optimization`'s default decay of ×0.99 per 10 steps.
- §3.1, §8.5: the largest degree is 783 (all other pixels), not 840.
- §5.2: `CD1_gradient` returns a descent direction for J. The sign flip is needed because `run_optimization` descends whatever it is given.
- §8.10: the `sample_forward_process` docstring is correct under its stated condition. The real problem is that the condition cannot hold for D < 1.
- §8.7: the 6-group branch's offsets are right; only the 5-group branch is wrong.
- §10.3: the large-t knots do matter, along class-separating directions.
- §8.3: the one-step offset is negligible.

**Sharpened:**
- §8.2: the monitor ranks iterates by their distance to 100× the target.
- §5.1: conditioning numbers, mini-batch rank, and why lr = Temp is not a confound.
- §10.1: the Fokker–Planck consistency condition, the Girsanov bound, and the quantified error accumulation.
- §10.2 and §10.3: the exact-profile numbers, with the caveat that the checkpoint is probably smoother.

**New:**
- §8.12–§8.16;
- the bullets marked *(review)* in §3, §4, §5.3, §6, §8.6–§8.11 and §10;
- the new diagnostics in §10.1.

### 12.4 Completeness critic (unverified)

The critic reported nine further items. The review was stopped before its last verifier finished, so these are **unverified**, except where noted.

- **C1. `.claude/settings.local.json` is committed** (*latent*; a policy risk).
  - It pre-approves `nohup python3 …` for every agent session on every clone, including on the login node (§8.9).
  - Fix: `git rm --cached` it and add it to `.gitignore`.
- **C2. The container toolchain drifted between builds.**
  - `requirements.txt` pins only top-level packages. It does not pin the `nvidia-*` CUDA wheels (which include ptxas), or diffrax's and optax's dependencies (lineax, optimistix, jaxtyping, chex).
  - The saved notebook outputs show PTX compiler 12.8.93 on 2025-06-03 and 12.9.86 on 2026-02-13, on the same GPU and driver. I confirmed this part from the outputs.
  - Bitwise reproduction therefore depends on the exact image file. Record `pip freeze` and archive the SIF with its SHA (§8.14).
- **C3. The ES baseline cannot cross between the modes.**
  - For any energy that matches the component covariance, the barrier is ≈ a²/(8σ²) ≈ 156 k_BT. One crossing within t = 1000 would need a barrier below about 8 k_BT.
  - The ES rows therefore show a single mode, whatever the run length (§8.6).
  - Check: evaluate the energy along the segment between the two means.
- **C4. Evidence for §8.1 at MNIST scale.**
  - Two untracked training logs, `container/MNIST_generation_3550618.out` and `…3550621.out`, each have exactly 21,594 lines. That is the count the fixed epoch budgets predict for the 2f73edb script, and a loss-dependent stopping rule would almost never hit it.
  - Check: `grep -c 'Convergence reached'` should give 100, and the first stop epochs should be 10,900, 11,200 and 11,400.
- **C5. Checkpoint writes are not atomic** (*latent*).
  - A kill during `jnp.save` truncates the only checkpoint.
  - A kill between the data write and the index write leaves k+1 rows with index k, so resuming appends slice k again and the history has 101 rows.
  - In training mode `interpolate_parameters` then silently uses the first 100 rows. That shifts every later slice by one knot and drops t = 0. Sampling-only mode catches it with its shape and SHA checks.
  - Fix: write to a temp file and `os.replace` it; derive the index from the row count; assert matching shapes in the interpolator.
- **C6. SM notebook caches and the temperature study's provenance.**
  - Caches are keyed by date and by a folder name that omits the seed, the initial parameters, the mixture and the code version. A same-day re-run therefore silently reuses stale parameters and samples.
  - The aggregated Temp = 0.01 and 0.75 records have no committed provenance. Git shows the 0.75 record written to `out/2026_07_06`, yet the 2026_07_07 aggregation found all three temperatures in `out/2026_07_07`.
- **C7. Turning on float64 at import also changes the random streams.**
  - Without x64, `jr.normal` draws float32 from a different bit stream. The data noise, initial states and Brownian paths would all change.
  - The ODE tolerances (rtol 1e-7, atol 1e-9) sit at or below float32 resolution.
  - Every current entry point also sets x64 itself, so current results are unaffected (§8.8).
- **C8. CD-1's gradient noise at the notebook's settings** (an analytic estimate).
  - The CD-1 estimator's extra Monte Carlo noise is ≈ 1.2× (near the data) to 3.7× (at t = 5) the exact SM gradient's mini-batch noise. Its O(dt) bias is ≈ 5e-4 relative.
  - dt ≈ 1e-3 would cut the extra noise by √10 while keeping the bias below 5e-3.
- **C9. The tracer-hashing FutureWarning comes from diffrax/equinox.**
  - The equilibrium-sampling cells warn too, with constant parameters, no vmap and no `step_ts`. So the project's drift, interpolation and vmap are not the trigger.
  - A future JAX upgrade will therefore need a diffrax/equinox upgrade.
  - By the critic's unverified recollection, diffrax 0.7 deprecates `PIDController(step_ts=…)`, which the ablation relies on.

**Open questions it raised:**
- Which tolerances did the `MNIST_ODE_tight` and `MNIST_ODE_tighter` jobs use? Their job names appear in no committed script.
- Does the 2026_07_08 checkpoint have its own training log, and was that run ever resumed?
- Which recipe built `container_image.sif`?

**Rule slip.** The critic broke the read-only ground rules once: `wc -l container/*` read through the container image and the Slurm logs. It saw only line counts.
