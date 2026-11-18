# Consistency check: the write-ups against the code (2026-11-17)

**What was compared.** I checked the two write-ups below against the code they describe:

- **Write-ups:** `MODEL_AND_CODE.md`, and `CODE_AND_PERFORMANCE.md` (all 4,433 lines).
- **Code:** `latent_model.py`, `train_latent.py`, `sample_latent.py`, `test_latent.py`, `inception_fid.py`,
  `make_gallery.py`, `jobs/run_pipeline_neuronic.sh`, and the paper's code in `../physical_diffusion_fns/`.

**How.** I only read files; nothing was run. Line numbers refer to the working tree on 2026-11-17, including the
uncommitted edits. Nothing in the code or in the existing write-ups was changed.

**Verdict.**

- **The core is consistent.** Every formula, flag, default and file name in `MODEL_AND_CODE.md` matches the code. The
  analytic gradients and Laplacians also check out (re-derived by hand).
- **No bug** was found in what the trained or sampled models do.
- **Two issues can change how a result is read:**
  - the "frozen baseline counts each pixel once" statement (section 1.1);
  - the parameter counts quoted next to Wang & Achour (section 1.2).
- **The rest** is documentation that is wrong, stale or overstated, plus help, log and README text that disagrees with
  what the code does.

---

## 1. Findings that can change how a result is read

### 1.1 The frozen patch baseline at `--patch-fixed-scale 0.0625` does not count each pixel once

**Where.** `MODEL_AND_CODE.md` lines 141 and 156-157, and the notebook, lines 4247-4258.

**Claim.** In 8x8 every 2 px and 16x16 every 4 px, each pixel lies in 16 patches, so 1/16 of the reference counts each
pixel once.

**Code.**

- One factor is applied to every patch set (`train_latent.py:1741-1744`).
- The geometry has its own coverage per pixel (`latent_model.PatchGeometry.coverage`, `latent_model.py:1116`).

Coverage per pixel, from the patch grid (n_p = (G - side) / stride + 1):

| layout | grid | coverage, centre | pixels with that coverage | mean coverage | corner |
|---|---|---|---|---|---|
| 16x16 every 4 px | 32 | 16 | 64 of 1024 (central 8x8) | 6.25 | 1 |
| 8x8 every 2 px | 32 | 16 | 400 of 1024 (central 20x20) | 10.6 | 1 |
| 4x4 every 2 px | 32 | 4 | 784 of 1024 | 3.5 | 1 |
| 7x7 every 3 px | 28 | 4-9 (alternating) | - | 4.0 | 1 |
| 16x16 every 4 px | 64 | 16 | 1600 of 4096 | 10.6 | 1 |

What this means:

- **At 0.0625, most pixels are under-counted.** The under-count reaches 1/16 at the edges, which is the background and
  hair, where the contrast deficit of the samples sits (`MODEL_AND_CODE.md` section 6).
- **"Once" is per set.** With two patch sets plus the global unit, an interior pixel is still counted about three times.
- **One factor cannot be 1/coverage for every set when the layouts differ.** Example: the K784 three-scale model, where
  8x8/2 and 16x16/4 have coverage 16 but 4x4/2 has 4. The pitfall "use `--patch-fixed-scale 1/coverage`"
  (`MODEL_AND_CODE.md:157`) does not say this.
- **"S x their reference" is not a plain rescaling.** In `latent_model._patch_ref_one` (`latent_model.py:1977-1990`),
  scale S gives W = S (W* - pi W*) and b = T log pi - S W*.mu / 2 (then centred). This is a tempered expert: S times the
  clusters' log-likelihood, with the prior kept.
- **"Also the trained runs' start" (notebook 4258) is not quite right.** The trained runs start at 0.06, not 0.0625.

**Effect on the conclusion.** "Trained is far better than frozen" stands (Inception FID about 22 against 147). The
wording "fair baseline / each pixel counted once" is overstated. A per-pixel normalisation would need weights per pixel,
which a single scale per patch set cannot give.

### 1.2 Parameter counts in the 11-15 entries leave out Q (2x too low)

**Notebook statements:**

- 3191-3192: "about 0.6M (K784) to 4.8M (K6144) global couplings + about 0.1M patch".
- 3235: "615k at K784, 4.8M at K6144 ... about 100k patch couplers".
- 3771: "K1536 at 64x64 has 6.3M global couplings".

**Code.** The recipe uses `--quad`, and Q has the same shape as W. `MODEL_AND_CODE.md` section 1 counts W + Q:

- K784: 1.23M; K6144: 9.63M; 7x7 patches: 0.20M.
- CelebA-64 K1536: 12.6M. The notebook's own line 4287 uses this value, so lines 3771 and 4287 disagree.

**Why it matters.** These numbers are the ones set against Wang & Achour's 326k parameters. Use the W + Q counts
(`MODEL_AND_CODE.md`) for the paper.

---

## 2. `MODEL_AND_CODE.md`: wrong or stale statements

| line | says | actually | evidence |
|---|---|---|---|
| 17 | `celeba64g` data "not kept on disk at present, see 5" | rebuilt 11-16 (section 5 says so); files present | `data/CelebA/celeba_gray64_{train,valid,test}.npz` |
| 131 | `--quench`, `--sampler ode` added 11-15 | both exist on 11-11 (notebook 490, 533); only `--fixed-langevin` and quenching saved `--init-dist` samples are 11-15 | backup `../latent_physical_SGM_pre_ising_patch/sample_latent.py:594-597` |
| 120-144 | the change table | leaves out `--time-grid` (11-11), `--thin-knots`, `--split-dead` / `--split-spread`, the `--harmonic` family, `--init-rank` (11-12) and the scripts `state_death.py`, `early_slices.py`, `grain_stats.py`, `blind_gallery.py`, `patch_bench_check.py`, `harmonic_analysis.py` | notebook 548-1440 |
| 122 | backups "before each change" | not every change has one (e.g. none for the 11-17 quench-only fix, none named for 11-15 `--quench`) | `ls ../latent_physical_SGM_pre_ising_patch/` |
| 62 | common training flags | do not include `--patch3`, used by the K784 three-scale CelebA model (row at line 91) | |
| 71 | "0.06 is the measured safe start scale" | measured for a switch-on at slice 31 (job 4231616); the recipe switches on at slice 55, not measured there | `train_latent.py:1047-1048` ("at slice 31") |
| 75 | equil start: "no data needed" | true for the start states; the sampler still loads the data and hashes the data file for diagnostics, so an equil job fails if the data file is missing | `sample_latent.py:624-629` |
| 137 | block unfold: patch term and gradients "bitwise for every layout" | the notebook records an exception: 64x64 8:2, loss differs by 8.7e-8 relative in float32 (gradients bitwise) | notebook 4003-4004 |
| 144, 154 | 10x tighter tolerance moves samples by "~1e-6" of the sample spacing | measured 4.9e-6 to 7.2e-6, i.e. a few millionths | notebook table near 4424; `tol_check.py:47-50` |
| 161-162 | CelebA-64 with patches "9x slower per step" | from the 1% smoke test; the full run 4263037 did the patch slices at 66-69 steps/s, 6.9 h in all (section 6 says so, section 4 not updated) | notebook 3880, 4063 vs 4236 |
| 141, 156-157 | 1/coverage = 0.0625 counts each pixel once | see 1.1 | |

---

## 3. `CODE_AND_PERFORMANCE.md`: statements to qualify

**Slice 0 (11-12 entries, lines 1000-1004, 1019-1021, 1190-1192).**

- The entries call slice 0 "badly underfitted" and conclude "8x steps on slice 0 do nothing".
- They do not say that the hidden units are held fixed in slice 0 (`--freeze-hidden-slices 1`), so both statements
  concern the visible couplings theta only. See section 6.

**Planning numbers.**

- **CelebA-64 speed** (3878-3886, 4062-4063: about 5-6 steps/s, "about 3 days"). Contradicted by the full run (4236):
  66-69 steps/s, 6.9 h. The smoke test ran 1% of the steps, so per-slice compilation probably dominated it (not
  verified). The notebook never reconciles the two.
- **History size, CelebA-64 K1536 + 16x16/4 + 8x8/2** (4287: "about 14 GB, patches 4.8M"). The patches are 2.77M +
  3.44M = 6.2M, so about 15 GB per float64 copy, and 30-45 GB with 2-3 device copies on the 46 GB L40.

**Evaluation.**

- **FID_inf, KID and bootstrap sd depend on where a set sits in the job.** One `default_rng(0)` is consumed in turn by
  the floor, the diagnostics and each set (`inception_fid.py:195-201`). The FID at fixed n is deterministic. FID_inf
  differences below about 0.5 between jobs are noise. Examples:
  - the same samples give 18.6 / 18.7 / 18.8;
  - the "ties" of 22.3 (18.7) vs 22.4 (18.6) in `MODEL_AND_CODE.md:90-91`.
- **KID is measured against the test features, not train** (`inception_fid.py:200`, `kid(F, Fte, rng)`). Tables headed
  "FID train | FID_inf | KID" read as if all three use train.
- **CelebA table, line 3667:** the floor row puts 2.2 in the KID column. 2.2 is the floor's FID_inf (report 4256037). The
  code computes no KID for the floor.

**Corrector (2026-11-17).**

- **99 segments, not 100.** It is [t0, 98 interior knots, 5], and the corrector runs after the first 98 only, so there is
  no corrector step at the t = 0 slice (`sample_latent.py:1022-1023, 1041-1048`; notebook 4369-4370 says 100 and "after
  each segment").
- **Timing (4399):** "311 s ... 339 s (+20%)" is +9%. The +20% matches the manifests' `chunk_seconds` (271 -> 325 s), not
  the quoted seconds.

**Sampling options.**

- **`--force-scale` "sampling at temperature T / LAMBDA"** is an interpretation. The drift is exactly x - 2 LAMBDA grad F
  (`sample_latent.py:917-921`), but the x term and the noise are not scaled, so it is not exact sampling of p^LAMBDA.
  It is exact only for the corrector's Langevin target exp(-LAMBDA F / T).
- **`--sde-tol-scale`** scales rtol and atol of the reverse SDE and of the corrector segments
  (`sample_latent.py:1026-1027, 1072, 1102`). It does not scale the equil start (`sample_latent.py:509-510`) or the
  quench (ODE tolerances, line 996). The convergence test used the noise start, so its result is unaffected.
- **Line 4390** ("for CelebA, `trajectory_set` draws the whole set at once") is true for every problem except the
  paper's mnist28 (0/1), including MNIST ten digits and Fashion (`latent_model.py:2576-2580`).

**Older entries.**

- **Line 903 / `--split-dead`:** "the two copies' field difference has SD s T". With W_j +- eps v and
  eps = s T / sqrt(lambda), the difference is 2 eps v.x, SD 2 s T. Each copy differs from the donor by SD s T, which is
  how the docstring at `train_latent.py:506-507` puts it.
- **Line 2363** plans the keys `Wpi/bpi/Qpi`. The code uses `Wp/bp/Qp` (superseded at 2574).
- **Deep tree (2834, 2884):** `../latent_physical_SGM_pre_tree/` and the candidate folder are described as kept. Neither
  exists any more, and no tree code is left, which matches "parked" (2930-2933).
- **Line 2851** ("Also fixed: --thin-knots" for the patch-kind string) was undone when the tree was removed. The bug came
  back until the `--patch2` edit; the current code handles it (`sample_latent.py:858`). `--thin-knots` on patch runs was
  therefore broken between about 11-14 23:00 and the `--patch2` edit.

**Sections 1-8 (written 11-11), stale.**

- The line references: `build_runner` is at line 711 (not ~314), the slice loop at about 1708 (not ~800), `drift` at 288
  (not ~89).
- "chunks of 250": the default `--chunk` is 125 now; the pipeline passes 250.
- "float64 everywhere": the recipe trains with `--precision mixed`.
- "53 checks": the tests now run to block 20.
- Section 8 says it is superseded; sections 1, 2 and 7 do not.

---

## 4. Code text that disagrees with the code's behaviour (no effect on results)

- **`train_latent.py:112-113` (header):** `--patch` "not with ... --hidden-fixed". Allowed since 11-16; the
  `--hidden-fixed` help (line 283) was updated. `--patch-fixed-scale` is not in the header.
- **`train_latent.py:1737` (log line):** "held at the full reference from now on", whatever `--patch-fixed-scale` is.
  The comment at line 1740 says "(scale 1)". So the log of run 4266722 (scale 0.0625) says "full reference".
- **`train_latent.py:271-276` (`--split-spread` help):** the factor 2 of section 3.
- **`make_gallery.py:73`** looks for the manifest key `quench`, but the quench files write `quench_duration`
  (`sample_latent.py:1006`). The gallery READMEs therefore show no sampling settings for `_q*` files.
- **The READMEs never list several settings:** `hidden_fixed`, `patch_fixed_scale`, `quad`, `force_scale`, `corrector`,
  `history_dtype`, `sde_tol_scale` (`make_gallery.py:71-73`). The frozen-vs-trained and corrector galleries rely on
  their folder names alone.
- **The quench manifest has no `patch_units`** (`sample_latent.py:1006-1009`; the notebook knows, 3803).
- **`inception_fid.py`:**
  - the docstring (lines 9, 14) mentions only 28/32 grids and 60,000 train images;
  - the footer (line 240) says "60k train statistics" also for CelebA (162,770) and the 64x64 problem.

---

## 5. Test coverage

- **`test_latent.py` ends at block 20 (2026-11-14, Ising patch units).**
- **No test block covers these features:**
  - `--patch2` and `--patch3`;
  - the block unfold / fold;
  - `--quench`, `--corrector` and `--force-scale`;
  - `--history-dtype` and `--sde-tol-scale`;
  - `--hidden-fixed` with patches, and `--patch-fixed-scale`.
- **They were checked by one-off jobs.** `MODEL_AND_CODE.md` section 3 cites those jobs correctly.
- **Notebook section 7** ("`test_latent.py` must keep passing" as the guard for every change) therefore no longer
  covers the newest code.
- **Possibly stale:** block 19's comparisons of 4:2 patch runs against `PATCH_ORIG` were written before the block
  unfold, which now makes 4:2 layouts differ at rounding level. Not checked whether any of them are bitwise.

---

## 6. Slice 0: why the hidden units are not trained there, and does it matter?

### What the code does

`--freeze-hidden-slices` defaults to 1, and all three best models use it (config.json of
`mnist28_d0123456789_K6144_..._patch7s3S32on55_seed1`, `fashion28_c0123456789_K1536_..._patch8s2S32on55_seed1`,
`celeba32g_K1536_..._patch8s2S32+patch16s4S32on55_seed1`: `freeze_hidden_slices` 1, paper steps, no step schedule). In
slice 0 (t = 5):

- **Held fixed:** the learning rates of W, b, Q and of the per-pixel units wl, bl are 0 (`train_latent.py:1756-1765`).
  - The Potts unit and the per-pixel units sit at their closed-form reference at t = 5: the k-means Gaussian-mixture
    reference (`latent_model.prototype_reference`) and `local_reference`.
  - Q stays 0.
- **Trained:** only the visible couplings theta, for the paper's 10,901 steps.
- **From slice 1 (t = 4.95) on,** everything is trained, warm-started from slice 0.

### Why

- **The stated reason** (`train_latent.py` header, lines 14-15, from the K = 1 Fisher-unit version): "the first FREEZE
  slices keep W, b fixed while theta_v relaxes from its start".
- **The situation at slice 0.** theta starts from the paper's generic initial values (quartic model: k_duff = 1), far
  from a fit. The hidden units start from a reference that is already close to the right score at t = 5.
- **My reading of why holding them helps** (not tested in the notebook): if both trained at once, the hidden units'
  gradient would at first mostly soak up theta's misfit and move them away from the reference.
- **Never varied.** `--freeze-hidden-slices` is not mentioned in the notebook, the README or the job scripts. Its value
  has been 1 throughout.

### Is slice 0 still undertrained?

**By the 11-12 measure, probably yes, and only in theta.** The measurement (`early_slices.py`, four digits, global
Potts 784):

- slice 0's denoising error is 1.35 with the paper's steps against 0.14 with 2x steps;
- the hidden units are held in both runs, so the gain is theta finishing its relaxation;
- the current best models use the same slice-0 settings, so their slice 0 is presumably in the same state. Not
  measured on them.

**It has not shown up in sample quality:**

- **The metric is extremely sensitive at t = 5.** The denoised (Tweedie) mean divides a small remainder by e^-5 = 0.0067,
  so small force errors are magnified about 150x. The validation loss cannot see them (notebook 1007-1008).
- **The gap closes within a few slices,** because slice 1 on trains everything: slice 3 error 0.095 against 0.087
  (notebook table at 987-998).
- **8x steps on slice 0 alone change nothing:** FID +0.03, the same gallery (notebook 1184, 1190-1192).
- **The equilibrated start, which relaxes under slice 0's own energy,** was as good as the exact-data starts:
  - eq3 17.98 against Gaussian 18.32 and oracle 18.71, plain runs (notebook 1262-1270);
  - so slice 0's energy is good enough for the start the recipe uses.
- **Most of the error forms at t of about 1.5-2.7,** not near t = 5 (oracle starts, notebook 1152-1169).

**Caveat.** All of this was measured on four digits, global units only, on 11-12. It was not re-measured on ten digits,
Fashion or CelebA with patches.

**If you want to close it,** one cheap check would settle it for a current model:

- the `early_slices.py` slice-0 error of a best model;
- or a run with `--steps-schedule 0:2,1:1` (2x steps on slice 0 only), or with `--freeze-hidden-slices 0`.

The evidence above suggests it is low priority.

---

## 7. Checked and correct (main items)

**Model and loss.**

- The free energy F = F_vis + F_glob + F_loc + sum F_patch, with the quartic visible model: k_6 held at 0,
  k_duff >= 0.01, only linear springs at radius 1 (8 neighbours).
- The Potts / Ising gradients and Laplacians with Q (`hidden_terms_quad`, `patch_terms`), re-derived by hand. The patch
  Laplacian as a sum of per-patch blocks.
- The loss -lap F / T + |grad F|^2 / (2 T^2).
- The data: N(0, 0.01^2) pixel noise, one global mean / std (`_normalise_like_paper`); the CelebA split sizes.

**Training.**

- Slice times t_k = 5 (1 - k/99), so slice 55 = t 2.22.
- Steps 10,901 (t = 5) to 30,901 (t = 0), 2,095,000 in all. Learning rate 2e-3 -> 2e-5.
- Adam (optax `scale_by_adam`), decay 0.95^(step/6000), EMA weight 1/1000, 512 fresh forward samples per step.
- `--hidden-lr-scale` 5e-4. Patches switch on at 0.06 x the reference at t_K.

**Precision.**

- `--precision mixed`: float32 at HIGHEST precision inside `in_compute_dtype`; validation (`loss_ps`) on the dense
  float64 path.
- It refuses `--hidden-samples`, `--hidden-rule cd1`, `--hidden-norm occupancy` and `--hidden-floor`.
- `visible_impl` and `precision` are recorded in config.json and are in the `--init-from` free set.

**Sampler.**

- The drift is x - 2 grad F: the paper's `energy_fn` at 2 theta - theta_lin, plus the hidden and patch forces.
- The paper's `solve_SDE`: SRA1, rtol 1e-4, atol 1e-6, VirtualBrownianTree tolerance 1e-12, stepping onto the 98
  interior knots, linear interpolation.
- The PF-ODE is x - grad F.
- `equil:S`: Langevin dynamics under the slice-0 energy.
- The quench: dx/ds = -grad F at the t = 0 parameters, Tsit5 at rtol 1e-7 / atol 1e-9, writing `<stem>_q<s>`.
- Quench-only mode builds no SDE drift.
- `--force-scale` gives the drift x - 2 LAMBDA grad F.
- The corrector uses the Song et al. SNR step with the knot's grad F.
- `--history-dtype` casts only the stored history.

**Tests that link training to sampling.**

- Block 2: the dense energy against the paper's `energy_fn`.
- Blocks 7, 17, 19F, 20F: `make_drift` against x - s grad F by autodiff.

**Tables, scripts and data.**

- Every row of the parameter-count table (W + Q).
- The pipeline's argument order, and its 7th argument SAMPLE_ARGS. Its evaluation always reads the noise-start
  `sde_base_n<N>.npy`, so the equil start and the quench are separate jobs, as written.
- The evaluation scripts named in `MODEL_AND_CODE.md` exist. The Inception FID is the TTUR graph, pool_3, TF32 off.
- The CelebA data on disk; the share is 1 TB (55% used on 11-17).

## 8. Not verified

- The run results themselves, apart from spot checks of `fid/4237907` and `fid_inception/4256037` (both as quoted).
- The total step counts of the power:1.5 grids (would need computation).
- The branch and commit statements (no git was used).
- The cause of the slow CelebA-64 smoke test.
