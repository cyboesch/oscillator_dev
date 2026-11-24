# Review of the latent physical SGM, and a search for a better CelebA-32 model

Started 2026-11-23 22:10 (overnight session, the user asleep). Everything of this review lives in this folder
(`latent_physical_SGM_review_2026-11-23/`): code copies, job scripts, logs, outputs (`out/`), galleries. Nothing in
`latent_physical_SGM/`, `out/latent_physical_SGM/` or the paper folders was changed; no git command was run.

**Goal (the user, 2026-11-23):** a more expressive model and better samples on CelebA-32 (grey), exact gradients;
on-device training is run later on the best models. Headline metric: Inception FID against train, n = 2000, SDE +
quench 1. Stay bipartite (no hidden-hidden couplings), so the physical learning rule keeps its form.

**Baseline to beat:** the locked recipe, K16 + 16x16 every 4 px from t = 4.49 + 8x8 every 2 px from t = 2.22, 64 patch
states: **20.20** (FID_inf 12.34); with the corrector 19.95; K784 control 19.43. Seed-to-seed sd about 0.15, bootstrap
sd 0.2-0.35, so differences below about 0.5 are noise.

## Morning summary (2026-11-24, written 06:15; updated as the last runs come in)

**Best model of the night: 17.04 Inception FID** (FID_inf 13.3, KID x 1000 9.1), against the locked baseline's
20.20 (16.5 / 12.4) in the same evaluation job. Same data, seed, 2000 trajectories, exact gradients.

| model (CelebA-32, exact, seed 1, n = 2000) | numbers per slice | as sampled | best FID (quench duration) |
|---|---|---|---|
| locked baseline (K16, 16x16 + 8x8 patches, 64 states) | 2.27 M | 76.70 | 20.20 (1) |
| the same recipe, seed 2 | 2.27 M | 76.88 | 20.25 (1) |
| better-converged late slices (4x steps in slices 85-99) | 2.27 M | 71.94 | 19.26 (0.7) |
| 128 states in both patch sets | 4.48 M | 68.10 | 18.88 (0.7) |
| K784 + 128 states in both sets | 6.06 M | 66.98 | 18.39 (0.7) |
| **+ bond units, 2 states** (new element, section 3) | 2.29 M | 59.44 | **18.40** (0.4) |
| + bond units, 4 states; seed 2 | 2.31 M | 58.72; 58.82 | 18.47 (0.4); 18.57 (0.4) |
| + bond units, 4 states, + 2x learning rate | 2.31 M | 56.90 | **18.04** (0.4) |
| **+ bond units, 2 states, + 4x steps in slices 85-99** (best small model) | 2.29 M | 58.01 | **17.96** (0.4) |
| + bond units over range 2 (pixel pairs up to 2 apart), 2 states; 4 states | 2.34 M; 2.40 M | 52.69; 52.42 | 18.09 (0.3); 18.48 (0.3) |
| **128 states + 4x4 set + bond units + 2x learning rate** | 5.01 M | **46.23** | **17.04** (0.3) |

What I found, in order of importance:

1. **A new, tiny model element gives the largest single gain: -1.8 FID for +1% parameters.** A two-state hidden unit
   on every spring between neighbouring pixels that can release the spring ("bond unit", section 3). It does nothing
   above t = 1 and takes over below: the score-matching loss at t = 0 improves by more than half (-31k of -55k), the
   samples before the quench improve by 18 FID (76.7 -> 58.7), and after its own best quench the model is at 18.4.
   Two seeds agree (18.47 / 18.57; baseline 20.20 / 20.25). **You have to judge whether it is physical for your
   hardware:** it has no hidden-hidden coupling and keeps the learning rule's form, but it couples one hidden unit to
   two pixels (a hidden state changing a coupling spring, as Q changes an on-site spring). The on-device (sampled
   statistics) path for it is not written yet.
   - It can **replace** the plain visible springs: with `--pair-terms none` the model reaches the same 18.47.
   - It is only needed at the end: switched on at t = 0.96 instead of 2.22 it gives the same 18.41.
2. **The quench duration is a real setting, and the right value depends on the model.** The quench never converges
   (displacement grows almost linearly with its duration). The baseline is best at 1 (the lock-in value; 1.5 / 2 / 3
   are worse), better-converged or larger models at 0.7, bond-unit models at 0.4, the combination at 0.3. With bond
   units, quench 1 is 8 FID worse than quench 0.4 (plateaus), so **a model must be scored at its own best duration**;
   at quench 1 the bond units look like a failure (26.4).
3. **The late slices of the locked recipe are not converged**, and the t = 0 slice, which the quench uses, is the
   worst. More steps in the last 15 slices (-0.9) or a larger learning rate (-0.8 at 4x) help at no model cost.
4. **Patch capacity is worth about 1 FID per doubling of the patch states** (64 -> 128: 18.88), at twice the
   parameters. The other capacity variants (second unit per patch, 12x12 or 4x4 sets, denser 16x16 patches) give
   0.4-0.5 at their best quench; training on clean images gives nothing (20.04).
5. **The gains add up**: bond units + 2x learning rate 18.04; bond units (2 states) + 4x late steps 17.96, the best
   model at the locked recipe's size (+1%); with 128 states and a 4x4 set on top 17.04.
6. **Audit: no bug found** in the model terms, the loss, the reverse drift, the quench or the corrector (section 1);
   my copy of the code reproduces the baseline's validation losses to the last digit and its FID exactly (20.20).

What I would do next (not started):
- Decide whether the bond unit is acceptable physically. If yes: write its sampled-statistics path and run the
  on-device version of "locked + bond units (2 states)"; rerun the lock-in tables with the quench duration chosen per
  model; try it on CelebA-64, MNIST and Fashion (the contrast / blur deficits are larger at 64x64).
- Independently of that: give the late slices more steps or a larger learning rate in the locked recipe, and rescan
  the quench duration (0.7) for it.
- All runs of the night are finished and scored (09:10); nothing is running. The last three combinations (V1-V3,
  17.73-17.99) did not beat the 17.04 model.

Things that went wrong, so you can discount accordingly:
- My first two FID jobs were wrong (baseline row skipped; "as sampled" scored the quenched files); fixed and rerun.
- Two runs with three patch sets sampled 3x slower than planned: one timed out at 1750 / 2000 samples (scored at
  n = 1750 against the baseline's first 1750), one was cancelled (8x8 patches every 1 px). Two quench-scan jobs and one
  pipeline's quench ran out of host memory and were rerun with more.
- The time-averaged read-out I tried instead of the quench is much worse than the quench (43-57); dropped.
- Disk: the share went from 80 G free to 28 G free (98% used). My runs hold about 50 G in
  `latent_physical_SGM_review_2026-11-23/out/`; nothing was deleted outside this folder. Inside it I removed only the
  duplicate `slices/` folders of finished runs and the incomplete slices of the cancelled run.

## 1. Audit: does the code do what the write-ups say?

Read in full: `MODEL_AND_CODE.md`, `DESIGN.md`, the CelebA parts of `CODE_AND_PERFORMANCE.md` (oracle scan, contrast,
early patches, small models, lock-in, results up to 2026-11-23), and in the code the model terms
(`latent_model.py`: visible model, global / per-pixel / patch units, the loss and the force), the training step
(`train_latent.build_runner`, the slice loop, switch-on, saving) and the sampler (`sample_latent.make_drift`,
`do_quench`, `sample_corrector`).

Checked by hand against the theory:

| item | where | result |
|---|---|---|
| Potts unit with squared-pixel couplings: gradient and Laplacian | `latent_model.hidden_terms_quad` vs DESIGN 2.1 | agree (grad = -sum p G, lap = -(sum p G^2 - gbar^2)/T - 2 sum p sum Q) |
| patch units: the same per patch, fold = adjoint of unfold, no cross terms between overlapping patches | `patch_terms`, `PatchGeometry` | agree |
| loss -lap F / T + grad F^2 / (2 T^2) per sample | `make_loss_per_sample` | agree |
| reverse drift x - 2 grad F at pixel temperature T; visible part as 2 theta - theta_lin, hidden part doubled directly (DESIGN 3) | `make_drift` | agree |
| quench: dx/ds = -grad F at the t = 0 parameters | `do_quench` | agree |
| corrector: x += eps s + sqrt(2 eps) z, s = -grad F / T | `sample_corrector` | agree (Langevin at temperature T) |

No bug found so far in the parts read. Not yet read line by line: the sampled-statistics / visit-count path (not used
here, exact gradients), `--time-knots`, `compress_history.py`, the evaluation scripts beyond their headers.

Observations that matter for "how to improve" (all from the existing logs and write-ups):

1. **Late slices are not converged.** The baseline's training log shows the validation loss still falling in the
   second half of the late slices (slice 98: -348 +- 5; slice 99: -775 +- 13, of a loss of about -45,000 to -55,000).
   For scale, 100 loss units at slice 70 separated two models by 1.9 FID. So more steps or a larger step size at small
   t may help. Tested here (runs I and N).
2. **Patch states are not saturated**: 16 -> 32 -> 64 states gave 25.75 -> 21.7 -> 20.20. 128 was never run. Tested
   here (A, B, L).
3. **Where the FID is lost** (oracle-start scan of 2026-11-18, older model): almost nothing above t = 3.5 once the
   coarse patches act early; about 7 between t = 2.2 and 0.2 (the patch stage); the quench itself costs about 2.3 on
   real images (it removes 21% of the high-frequency power); and the 1% pixel noise of the training data another 4.6
   against the clean-image floor of 6.4. So the patch stage and the fine scale are where capacity is missing.
4. **The visible springs are purely quadratic**, c (x_i - x_j)^2. The write-up's own diagnosis is blur ("each pixel
   is pulled to an average of the templates covering it"), low contrast and missing high-frequency power. A quadratic
   spring cannot release across an edge. This motivates the one new model element tried here (section 3).

## 2. Round 1: capacity and optimisation variants with the existing code (submitted 22:25)

All: CelebA-32, exact, seed 1, the locked flags unless noted, n = 2000 SDE samples + quench 1. Script
`jobs/submit_round1.sh`, ids in `jobs/round1_ids.txt` (the first submission with 12 h limits was cancelled and
resubmitted with shorter limits for backfill; the queue is busy: 666 pending jobs, fair share 0.07).

| id | job | change against the locked recipe | question |
|---|---|---|---|
| A | 4306827 | 128 states in both patch sets | are patch states saturated? |
| I | 4306828 | 2x training steps | are the slices converged? |
| D | 4306829 | a second 64-state unit on every 8x8 patch (from t = 2.22) | product of two mixtures per patch vs more states |
| C | 4306830 | a second 64-state unit on every 16x16 patch (from t = 4.49) | richer coarse stage |
| H | 4306831 | trained on clean images (`--data-noise 0`) | the 1% pixel noise |
| G | 4306832 | 8x8 patches every 1 px (625 units) | denser fine patches |
| F | 4306833 | 16x16 patches every 2 px (81 units) | denser coarse patches |
| E | 4306834 | a third set, 4x4 every 2 px from t = 1.21 | fine texture |
| N | 4306835 | 2x learning rate | step size |
| B | 4306836 | 128 states in the 8x8 set only | which set needs the states |
| K | 4306837 | a third set, 12x12 every 4 px from t = 4.49 | an intermediate scale |
| L | 4306838 | K784 + 128 states in both sets | the combination |

Code change needed: `--patch3-on-slice` (the third set's own switch-on slice; before, it switched on with the first).

Disk: the share had 80 G free at the start (93%). Each run is 2-4 G after its `slices/` folder (a duplicate of the
history files) is removed by the job; the job refuses to start below 20 G free.

## 3. A new model element: hidden units on the springs ("bond units")

**The element.** One Potts hidden unit with S states on every bond of the radius-1 lattice (the pairs the visible
springs already couple: 3,906 bonds at 32x32, 4 orientations), coupled only to that bond's elongation d = x_j - x_i:

    a_s = bb_s + wb_s d + qb_s d^2,        F_bond = -T sum_bonds log sum_s exp(a_s / T).

State s changes the bond's spring constant by -qb_s (the visible spring is c_lin d^2) and its rest length through
wb_s. With S = 2 and wb = 0 this is the classic "line process" of image models: a bistable element in the coupler
that releases the spring across an edge and keeps it in flat regions; with more states it is a Gaussian scale
mixture on each pixel difference, the standard heavy-tailed model of image gradients.

**Why this one.** The quadratic springs carry the texture (+5.6 FID without them) but can only smooth; the documented
defects are blur, missing high-frequency power (0.59 of the real images' above 8 cycles per image) and low contrast.
The patch units cannot represent this term: their fields contain x_i and x_i^2 but no product x_i x_j.

**Is it still the same kind of physics? (for the user to judge.)**
- No hidden-hidden couplings; the bond units are independent given the pixels, so they are summed out exactly and the
  force-matching rule keeps its form (per bond: the unit's occupations, d, and the two pixels' clamp forces).
- It is the pair analogue of the squared-pixel couplings Q: Q lets a hidden state change a pixel's on-site spring,
  qb lets a hidden state change a coupling spring. But it couples one hidden unit to *two* pixels through one term,
  y_s (x_i - x_j)^2, so in a strict "hidden unit - pixel" graph it is a new kind of coupler.
- Cost: 3 x 3,906 x S numbers per slice (46,872 for S = 4), about 2% of the locked model's 2.4 M.
- Not done yet: sampled statistics (on-device rule) for these units; the code path refuses it.

**Code** (this folder only): `latent_model.py` (`bond_terms`, `bond_free_energy`, `bond_reference`, `bond_start`,
`bond_lr_scales`; the key `wb` in the parameters switches the term on in the loss and the force), `train_latent.py`
(`--bond S --bond-on-slice K --bond-init-scale LAM --bond-lr-scale`; saved in the slice files and in
hidden_history.npz like a patch set), `sample_latent.py` (the drift adds -grad F_bond; `BOND_P`).
Start: at the switch-on slice, LAM x a zero-mean Gaussian scale mixture fitted to each bond's variance at that time
(variances x 4^(s - (S-1)/2)), centred over the states, wb = 0. Learning rates: the hidden rate x the reference
magnitudes, as for the other units.

**Checks.**
- `bond_check.py` (CPU job 4306855), all 17 pass at 1e-16 to 1e-13: gradient and Laplacian of the bond term against
  autodiff; fold = adjoint of the difference; the full training loss and force (visible + global + per-pixel + patch +
  bond) against autodiff of the total free energy; mixed precision against float64 (1.7e-7); the sampler's ODE and
  SDE drift against the training force at a knot and between knots; without bond units the drift is unchanged.
- Training smoke test (CPU job 4306863, 3 slices, bond units switched on at slice 2): trains, saves (4, 32, 32, S)
  arrays, zeros before the switch-on. At LAM = 0.25 the start force of the bond units (0.2) was larger than the visible
  force (0.14), so the runs below start at LAM = 0.05.

**Round 2 (submitted 22:36, `jobs/submit_round2.sh`, ids in `jobs/round2_ids.txt`):**

| id | job | change against the locked recipe |
|---|---|---|
| P | 4306934 | + bond units, 4 states, from t = 2.22 |
| Q | 4306936 | + bond units, 2 states (line process), from t = 2.22 |
| R | 4306937 | + bond units, 4 states, from t = 3.99 |
| O | 4306935 | 4x training steps in slices 85-99 only (t < 0.76) |

Run O follows from the baseline's log: the validation loss is flat within each slice down to t = 1 (change over the
second half of a slice: -4 or less), then -9 at t = 0.76, -23 at 0.45, -118 at 0.15, -348 at 0.05 and -775 at
t = 0. Only the last 10-15 slices are unconverged, and the t = 0 slice, which the quench uses, is the worst.

## 4. Results

### Interim, 2026-11-24 00:05: validation loss against the baseline (before any FID)

Same validation batch per slice as the baseline (`val_compare.py`); negative = better. The copy of the code
reproduces the baseline exactly: every run is at 0.0 on the slices where its configuration equals the locked one.
(The clean-image run is validated on clean images, so its numbers are not comparable.)

| run | t = 1.36 (slice 72) | t = 0.76 (84) | t = 0.56 (88) | t = 0.35 (92) | t = 0 (99) |
|---|---|---|---|---|---|
| A: 128 states in both sets | -111 | -299 | | | |
| B: 128 states in the 8x8 set | -67 | -228 | -339 | | |
| D: second unit per 8x8 patch | -105 | -331 | | | |
| C: second unit per 16x16 patch | -70 | -116 | -175 | -273 | |
| K: + 12x12 set | -69 | -140 | | | |
| E: + 4x4 set from t = 1.21 | 0 | -301 | -525 | | |
| N: 2x learning rate | -31 | -53 | -88 | -228 | **-2936** |
| P: bond units, 4 states | -0.1 | -22 | -148 | **-616** | |
| Q: bond units, 2 states | -0.6 | -20 | -87 | -537 | |

- **The late slices are step-size limited.** Doubling the learning rate changes little down to t = 1 and then gains
  -674 at t = 0.15, -1535 at t = 0.05 and -2936 at t = 0, where the quench acts.
- **The bond units do nothing above t = 1 and a lot below.** Down to t = 1.4 the optimiser shrinks them to a
  degenerate state (all states equally occupied, loss = baseline): there the pixel differences are noise and a
  Gaussian spring is right. Below t = 0.8 they switch on by themselves: -148 at t = 0.56 and -616 at t = 0.35 for 47k
  parameters, more than the 4x4 patch set (460k) or the second 8x8 unit (1.4 M) gain there.
- Loss is not FID (the 2026-11-18 three-scale run lowered the late loss by 7000 and the FID by 0.6).

### Round 3 (submitted 00:05, `jobs/submit_round3.sh`; with `--save-f32`, half the disk)

| id | job | model |
|---|---|---|
| S1 | 4307244 | locked + bond units (4 states) + 2x learning rate |
| S2 | 4307245 | locked + 4x learning rate |
| S3 | 4307246 | locked + 2x learning rate + 4x steps in slices 85-99 |
| S4 | 4307247 | 128 states in both sets + 4x4 set from t = 1.21 + bond units + 2x learning rate (first combination) |
| S5 | 4307248 | locked + bond units with 8 states |

### First FIDs, 2026-11-24 01:00-01:50 (Inception jobs 4307784 quenched, 4307785 as sampled, 4307838 quench scan)

All n = 2000, seed 1, noise start; FID against train (bootstrap sd 0.2-0.35 after the quench, about 1 as sampled).
The baseline row is the existing lock-in run scored in the same job: it reproduces 20.20 exactly. (FID_inf from this
script version is 16.49 for the baseline; the lock-in tables quote 12.34 from another evaluation script, so compare
FID_inf only within this file.)

| model | SDE as sampled | SDE + quench 1 | FID_inf (quench 1) |
|---|---|---|---|
| locked baseline | 76.70 | 20.20 | 16.49 |
| C: + second unit per 16x16 patch | 72.28 | 19.78 | 16.32 |
| K: + 12x12 set | 71.81 | 19.91 | 16.39 |
| N: 2x learning rate | 72.09 | 19.94 | 16.31 |
| H: trained on clean images | 66.79 | 20.67 | 17.03 |
| Q: + bond units, 2 states | **59.44** | 25.01 | 21.21 |
| P: + bond units, 4 states, from t = 2.22 | **58.72** | 26.36 | 22.82 |
| R: + bond units, 4 states, from t = 3.99 | **58.59** | 26.37 | 23.20 |

- **Coarse-scale capacity and the learning rate: within the noise** (-0.3 to -0.4). The -2936 loss gain of the 2x
  learning rate at t = 0 does not reach the FID after quench 1.
- **Bond units: 18 FID better as sampled, 6 worse after quench 1.** Before the quench their samples are visibly less
  speckled (gallery `galleries/2026-11-24_celeba32_spring-units-4states_vs_locked-baseline_vs_2x-learning-rate_as-sampled-and-quench1_exact_noise-start/`,
  looked at: the same faces on every trajectory in all models; bond-unit rows cleaner as sampled, slightly blockier
  after quench 1). After quench 1, 18% of neighbouring pixel pairs are equal to within 1/255 (real images 14%, baseline
  10%): the zero-temperature descent under edge-releasing springs flattens regions into plateaus (the known
  "staircase" of such priors); the model's distribution is fine, its energy minimum is not a good image.
- **The quench does not converge; its duration is a setting.** Median displacement of the bond-unit samples: 0.08 /
  0.23 / 0.69 / 1.59 / 3.09 (normalised units) for durations 0.01 / 0.03 / 0.1 / 0.3 / 1, almost linear in the
  duration (baseline at 1: 3.06). Quench scan on P (bond units, 4 states), job 4307838:

  | quench duration | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 |
  |---|---|---|---|---|---|---|
  | FID | 58.72 | 55.39 | 49.34 | 33.80 | **19.16** | 26.36 |
  | FID_inf | 54.2 | 50.6 | 45.3 | 30.0 | **15.35** | 22.8 |

  So with a shorter quench the bond-unit model is at 19.16, below the baseline's 20.20 at quench 1. Whether the
  baseline also prefers a shorter quench is being measured (linked copy of the baseline run in `out/`, jobs 4307958-60;
  durations 0.1-1 for baseline, P and Q).
- **New read-out added to the sampler copy:** `--quench=-TAU`, a time-averaged read-out (the device stays at T with
  the t = 0 parameters for a time TAU; the image is the time average of 64 snapshots). Jobs 4307844 / 4307845, FID
  4307962.

### Quench duration: baseline against bond units (2026-11-24 02:15; Inception 4307962, 4308043, 4308044)

Same 2000 trajectories; the baseline rows are the main run's saved SDE samples quenched for other durations through a
linked copy (`out/celeba32g_K16_exact_BASELINE-locked-patches-64states_links-to-main-run_seed1/`: links to the main
run's history files, a copy of its SDE samples; nothing written to the main folder).

| quench duration | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1 |
|---|---|---|---|---|---|---|---|---|
| locked baseline | 76.70 | 54.83 | 41.43 | 32.91 | 27.40 | 23.93 | 20.74 | **20.20** |
| + bond units, 4 states (P) | 58.72 | 33.80 | 22.99 | 19.16 | **18.47** | 19.12 | 21.77 | 26.36 |
| + bond units, 2 states (Q) | 59.44 | | 23.18 | 19.21 | (running) | | | 25.01 |

(FID_inf / KID x 1000 at the best duration: baseline 16.49 / 12.36; bond units 4 states at 0.4: 14.65 / 10.25.)

- **With its own best quench (0.4) the bond-unit model is at 18.47, 1.7 below the baseline's best (20.20 at 1)**, for
  47k more parameters per slice (+2%). The baseline keeps improving up to duration 1 (longer durations 1.5 / 2 / 3:
  job 4308171, running); the bond-unit model reaches a better image in less than half the time and then degrades
  (plateaus).
- **Time-averaged read-out: no.** tau = 0.03 / 0.1 / 0.3 gives 56.7 / 52.4 / 43.0 on the bond-unit model and 70.6 /
  67.4 / 59.8 on the 2x-learning-rate model, far worse than a quench of the same length. Dropped.
- Bond-unit states: the t = 0 loss gain is -28.0k with 2 states, -31.1k with 4, -31.4k with 8 (S5): 4 is enough.

### Round 4 (submitted 02:25, `jobs/submit_round4.sh`; sampled with quench 0.2-1)

| id | job | model |
|---|---|---|
| T1 | 4308189 | locked + bond units, 4 states, range 2 (pixel pairs up to 2 apart: 12 orientations, 11,160 bonds) |
| T2 | 4308192 | locked + bond units, 2 states, range 2 |
| T4a | 4308191 | the locked recipe itself, seed 2 (seed spread of the baseline in this pipeline) |
| T4b | 4308190 | locked + bond units, 4 states, seed 2 |

Code: `--bond-range 2` (the number of orientations is the first axis of wb, so the sampler needs no flag);
`bond_check.py` passes for both ranges (jobs 4308176 range 2, 4308177 range 1; 17 checks each).

### Every model at its own best quench duration (2026-11-24 03:40; Inception 4308400, 4308649, 4308650)

n = 2000, seed 1, same trajectories. "Best" = the lowest FID over the scanned durations (in brackets).

| model | as sampled | quench 0.4 | 0.5 | 0.7 | 1 | 1.5 | best (duration) | FID_inf / KID x 1000 at best |
|---|---|---|---|---|---|---|---|---|
| locked baseline | 76.70 | 27.40 | 23.93 | 20.74 | 20.20 | 22.05 | **20.20** (1) | 16.49 / 12.4 |
| I: 2x training steps | 72.37 | 25.08 | 22.17 | 19.71 | 19.78 | 21.87 | 19.71 (0.7) | 16.00 / 11.9 |
| O: 4x steps in slices 85-99 | 71.94 | 24.24 | 21.47 | 19.26 | 19.47 | 21.39 | 19.26 (0.7) | 15.43 / 11.4 |
| S2: 4x learning rate | 69.64 | 23.37 | 21.03 | 19.40 | 20.00 | 22.17 | 19.40 (0.7) | 15.52 / 11.4 |
| E: + 4x4 set from t = 1.21 | 61.32 | 20.99 | 19.76 | 19.72 | 21.14 | 23.58 | 19.72 (0.7) | 16.11 / 12.5 |
| P: + bond units, 4 states | 58.72 | 18.47 | 19.12 | 21.77 | 26.36 | | 18.47 (0.4) | 14.65 / 10.3 |
| Q: + bond units, 2 states | 59.44 | 18.40 | 18.90 | 21.16 | 25.01 | | 18.40 (0.4) | 14.64 / 10.4 |
| S5: + bond units, 8 states | 59.02 | 18.59 | 19.28 | 22.17 | 26.97 | | 18.59 (0.4) | 14.91 / 10.4 |
| **S1: + bond units, 4 states, + 2x learning rate** | 56.90 | **18.04** | 18.76 | 21.56 | 26.00 | | **18.04** (0.4) | **14.19 / 9.7** |

Baseline at quench 2 / 3: 23.94 / 26.65 (job 4308400): its optimum is at about 1, so the lock-in's choice was right
for that model, and 20.20 is the fair reference.

- **Bond units: -1.7 to -1.8** (18.4-18.6 against 20.20), the same with 2, 4 or 8 states. Two states (the plain
  "spring on / spring released" unit) are enough: 23k extra numbers per slice.
- **Better converged late slices: -0.8 to -0.9** (4x steps in the last 15 slices 19.26; 4x learning rate 19.40; 2x
  steps everywhere 19.71), with a slightly shorter best quench (0.7).
- **The two add up partly:** bond units + 2x learning rate **18.04** (-2.2 against the baseline).
- **The better a model is at small t, the shorter its best quench**: baseline 1, better-converged or 4x4 models 0.7,
  bond units 0.4. As sampled the order is the same (76.7 / 70-72 / 61 / 57-59). The quench is a repair of the
  model's small-t error, and it costs detail when run longer than needed.
- Jobs lost to memory: quench scans with five durations in one process ran out of 48 G host memory for the
  128-state and three-set models (jobs 4308409, 4308651; each duration compiles its own solver). Re-run with fewer
  durations and 96 G (4308876).

### Round 5 and 6 (submitted 02:55 and 03:40)

| id | job | model |
|---|---|---|
| U1 | 4308402 | bond units (2 states) + 4x steps in slices 85-99 |
| U5 | 4308403 | bond units (4 states), no plain visible springs (`--pair-terms none`): can the bond units replace them? |
| U3 | 4308404 | bond units (2 states) switched on only at t = 0.96 |
| V1 | 4308877 | bond units (4 states) + 2x learning rate + 4x steps in slices 85-99 |
| V2 | 4308878 | bond units (4 states) + 4x learning rate |
| V3 | 4308879 | bond units (4 states) + 2x learning rate + second unit per 16x16 patch |

Cancelled: G (8x8 patches every 1 px, 4306832) at slice 73 after 4.2 h: no better in loss than 128 states (-151 at
t = 1.36) and far too slow to sample by morning; its incomplete slice files were removed, the training log kept.
D (second unit per 8x8 patch) timed out in sampling at 1750 of 2000 samples (three patch sets sample at 15 min per
chunk); the 1750 are scored against the baseline's first 1750 (job 4308872).

### Later results (2026-11-24 04:00-06:30; Inception 4308872, 4308874, 4308875, 4309475, 4309476, 4309771, 4309772, 4310044, 4310046, 4310300, 4310302)

All n = 2000 unless noted, seed 1 unless noted; best duration in brackets.

| model | as sampled | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1 | best |
|---|---|---|---|---|---|---|---|---|
| A: 128 states in both sets | 68.10 | | | | 20.76 | 18.88 | 19.18 | 18.88 (0.7) |
| B: 128 states in the 8x8 set | 70.27 | | | | 21.53 | 19.41 | 19.64 | 19.41 (0.7) |
| F: 16x16 patches every 2 px | 70.52 | | | | 21.93 | 19.72 | 19.85 | 19.72 (0.7) |
| C: second unit per 16x16 patch | 72.28 | | | | 22.52 | 19.90 | 19.78 | 19.78 (1) |
| K: + 12x12 set | 71.81 | | | | 22.40 | 19.82 | 19.91 | 19.82 (0.7) |
| H: trained on clean images | 66.79 | | | | 21.48 | 20.04 | 20.67 | 20.04 (0.7) |
| S3: 2x learning rate + 4x late steps | 68.56 | | | | 20.54 | 19.04 | 19.61 | 19.04 (0.7) |
| L: K784 + 128 states in both sets | 66.98 | | | | | 18.39 | 18.62 | 18.39 (0.7) |
| T4a: the locked recipe, seed 2 | 76.88 | | | | 24.14 | 20.87 | 20.25 | 20.25 (1) |
| T4b: bond units, 4 states, seed 2 | 58.82 | 23.05 | 19.25 | 18.57 | 19.25 | 21.87 | 26.37 | 18.57 (0.4) |
| U5: bond units, 4 states, no plain visible springs | 58.53 | 22.91 | 19.17 | 18.47 | 19.12 | 21.83 | 26.45 | 18.47 (0.4) |
| U3: bond units, 2 states, from t = 0.96 | 59.30 | 23.26 | 19.24 | 18.41 | 18.87 | 21.10 | 24.96 | 18.41 (0.4) |
| T1: bond units, 4 states, range 2 | 52.42 | 20.62 | 18.48 | 18.65 | 19.70 | 22.47 | 26.63 | 18.48 (0.3) |
| T2: bond units, 2 states, range 2 | 52.69 | 20.22 | 18.09 | 18.48 | 19.68 | 22.71 | 26.95 | 18.09 (0.3) |
| S4: 128 states + 4x4 set + bond units (4 states) + 2x lr | 46.23 | 18.01 | **17.04** | 17.79 | 19.05 | | 25.52 | **17.04** (0.3) |

S4 at 0.25 / 0.35: 17.20 / 17.30. D (second unit per 8x8 patch, n = 1750): 21.18 / 20.53 / 22.00 at 0.5 / 0.7 / 1,
against the baseline's first 1750 at 0.7 / 1: 21.75 / 21.28 (so -0.75).

- **Seed spread:** baseline 20.20 / 20.25, bond units (4 states) 18.47 / 18.57. The gain is 1.7 on both seeds.
- **Bond units replace the plain springs** (U5 = P to the second digit at every duration) and **are needed only below
  t = 1** (U3 = Q).
- **Range 2** improves the samples before the quench by another 6 (52.5 against 58.7) and shortens the best quench to
  0.3; after it the gain over range 1 is 0 to 0.3, within the noise.
- **The first full combination (S4) is the best model: 17.04**, as sampled 46.2. Its gains are roughly the sum of its
  parts (128 states -1.3, bond units -1.8, late-slice convergence -0.3 to -0.9).
- Gallery, looked at: `galleries/2026-11-24_celeba32_best-models-of-the-review_each-at-its-best-quench_.../`. Same
  faces on every trajectory in all models; the bond-unit rows at quench 0.4 are crisper in eyes and hair than the
  baseline at quench 1; the bond-unit row at quench 1 is flatter. Nearest training images of the 18.04 model's samples
  are other people.

- **U1: bond units (2 states) + 4x steps in slices 85-99** (Inception 4310555): as sampled 58.01; quench 0.2 / 0.3 /
  0.4 / 0.5 / 0.7 / 1: 22.40 / 18.69 / **17.96** / 18.47 / 20.64 / 24.16 (FID_inf 14.14, KID x 1000 9.8 at 0.4).
  The late-step gain (-0.4 here) adds to the bond units': the best model of the locked recipe's size.

### Final combinations, round 6 (2026-11-24 09:10; Inception 4310884)

All: locked recipe + bond units (4 states) + the listed changes; n = 2000, seed 1.

| model | numbers per slice | as sampled | quench 0.3 | 0.4 | 0.5 | 1 | best |
|---|---|---|---|---|---|---|---|
| V1: + 2x learning rate + 4x steps in slices 85-99 | 2.31 M | 55.94 | 18.22 | **17.87** | 18.63 | 25.15 | 17.87 (0.4) |
| V2: + 4x learning rate | 2.31 M | 55.96 | 18.26 | 17.99 | 18.86 | 25.73 | 17.99 (0.4) |
| V3: + 2x learning rate + second unit per 16x16 patch | 3.13 M | 54.51 | 17.87 | **17.73** | 18.67 | (see report) | 17.73 (0.4) |

- At the locked recipe's size, every way of converging the late slices on top of the bond units lands at 17.9-18.0
  (2x lr 18.04, 4x lr 17.99, 2x lr + 4x late steps 17.87, 2 states + 4x late steps 17.96): that lever is used up.
- The second 16x16 unit adds 0.3 (17.73), as it did without bond units.
- **Best model of the review: S4, 17.04** (128 states in both patch sets + 4x4 set from t = 1.21 + bond units with 4
  states + 2x learning rate; quench 0.3). Gallery with it, looked at:
  `galleries/2026-11-24_celeba32_best-model-17.04_128-states+4x4-set+spring-units-4states+2x-learning-rate_quench0.3_vs_locked-baseline-20.20_vs_small-spring-unit-models_exact_noise-start/`
  (row 2 of `00_side_by_side.png`; 100 samples in `03_best_..._quench0.3.png`; nearest training images in
  `nearest_best_..._quench0.3.png`). Same faces as the baseline on every trajectory; sharper eyes, mouths and hair
  edges; before the quench clearly less speckle than the baseline (bottom two rows). The 100-sample sheet still has
  some distorted faces and smudged backgrounds, as the baseline has.
- Disk: the share was enlarged to 2 T in the morning (1.1 T free at 09:00).
