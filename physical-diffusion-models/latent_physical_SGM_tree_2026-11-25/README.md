# Tree: a direct coupling between the global unit and the patch units (started 2026-11-25)

After the meeting with Dmitry Krotov (2026-11-25). His proposal: a hierarchy as in his "Hierarchical Associative
Memory" (arXiv 2107.06446, Section 4.3): pixels <-> patch layer <-> top unit, the top unit instantaneous, the patch
layer simulated. In our thermal model the same hierarchy is a tree of Potts units and can be summed exactly, so
nothing has to be simulated: the global unit and every patch unit are coupled by a table `J_p[g, s]`, and

    F_tree(x) = -T logsumexp_g [ a_g / T + sum_p logsumexp_s ((a_ps + J_p[g, s]) / T) ].

`J = 0` is today's model. Krotov's network (top unit without pixel couplings) is the same formula with `W = Q = 0`;
his energy is the naive mean-field free energy of this Potts system. The same model was built once before
(2026-11-14, "deep tree", branch `deep_tree`, K784, one patch set) and parked for cost (540 x per step at K784).

This folder imports the locked code of `../latent_physical_SGM` read-only and writes only below `out/` here.

## Files

- `tree_model.py`: `tree_free_energy`, `tree_terms` (gradient and Laplacian), `tree_force` (the cheap force for
  the sampler), `make_tree_loss(_per_sample)`, `make_tree_force`, `hidden_posteriors`. Two or three patch sets
  (`J`, `J2`, `J3`).
- `test_tree.py`: brute-force sum, autodiff, J = 0 = the locked loss and force, dL/dJ, Krotov's variant, float32.
- `cooccurrence_diagnostic.py`: do generated images coordinate the hidden states as real images do? (no training)
- `tree_bench.py`: step cost on a trained locked run, sampler force cost, and a J-only probe.
- `train_J.py`: J trained on top of a frozen locked run (the cheap test); writes a mirror run below `out/`.
- `make_train_tree.py` -> `train_tree.py`: the locked trainer plus `--tree` (8 scripted edits, about 40 lines; all
  parameters trained together). J rides on the trainer's mechanism for the `--cond` tables (created as zeros at a
  patch set's switch-on, saved, resumed, stacked into `hidden_history.npz`); its learning rate is the patch units'
  bias rate x `--tree-lr-scale`. Output root: this folder's `out/`.
- `sample_tree.py`: the locked sampler with the tree's force for a run with J (checks at start that J = 0 gives
  the locked drift).
- `jobs/run_cpu.sh`, `jobs/run_gpu.sh`: `sbatch jobs/run_gpu.sh SCRIPT ARGS` from this folder;
  `jobs/test_trainer.sh` (toy test of the trainer), `jobs/eval_tree_inc.sh` (Inception FID next to the locked run).

## Results

### Tests (job 4317600, 2026-11-25): all passed

- F_tree = brute-force sum over all hidden configurations: 2e-16.
- Gradient and Laplacian against autodiff of F_tree: <= 1.5e-14 (with / without Q, Qp, Qp2).
- J absent or zero = the locked loss per sample and force: <= 3e-16 (dense and stencil visible model).
- dL/dJ against central differences: 3e-7. Krotov's variant (W = Q = 0): F = brute force, finite.
- float32 loss against float64: value 8e-9, gradients <= 1.4e-5.

### Cost on the locked CelebA-32 run (job 4317601; K16, patches 16:4:64 + 8:2:64, batch 512, float32, L40)

| slice | t | locked, ms / step | tree, ms / step | ratio | force at chunk 250: locked / tree (`tree_terms`), ms |
|---|---|---|---|---|---|
| 30 (one patch set) | 3.49 | 0.9 | 9.0 | 10.6 | 1.09 / 10.95 |
| 60 | 1.97 | 2.6 | 33.0 | 12.9 | 2.59 / 30.85 |
| 80 | 0.96 | 2.6 | 33.3 | 13.0 | 2.60 / 30.88 |
| 99 | 0.00 | 2.6 | 33.2 | 12.8 | 2.63 / 30.96 |

- Estimated full 100-slice training (2.1 M steps, step cost only): locked 1.2 h, **tree 14.3 h (12 x)**.
- Peak GPU memory 13 GB. J = 0 equals the locked loss on the real parameters (<= 1.3e-14).
- The force column used `tree_terms`, which computes the mean force per global state; the sampler does not need it.
  `tree_force` (added afterwards) needs only the K P S posteriors on top of today's force; to be timed.

### Scale of loss differences (stored validation losses of existing CelebA-32 runs, paired, 4096 samples)

| against the locked run | slice 30 | slice 60 | slice 80 | slice 99 |
|---|---|---|---|---|
| K784, same patches (FID 19.43 vs 20.20) | -11.6 | -30.4 | -55.9 | -744 |
| on device S = 64 (21.20) | +0.8 | +9.1 | +44.9 | +1506 |
| harmonic visible model (20.98) | 0.0 | -0.2 | +32.2 | +1246 |
| 32 patch states | +0.9 | +45.8 | +248 | +4987 |
| no springs (25.83) | -0.6 | +2.2 | +362 | +22134 |

So about 1 FID corresponds to a loss difference of order 50 at t ~ 1 and of order 1000 at t = 0.

### J-only probe, first run (job 4317601): J = 0 start, everything else frozen, 1500 Adam steps at the slice's lr

| slice | t | lr | held-out loss change (4096 test images, paired) | rms J / T | effective global states |
|---|---|---|---|---|---|
| 30 | 3.49 | 1.4e-3 | -0.7 +- 0.5 | 1.06 | 3.58 -> 3.57 |
| 60 | 1.97 | 8e-4 | +0.3 +- 1.1 | 0.95 / 0.63 | 2.74 -> 1.65 |
| 80 | 0.96 | 4e-4 | -5.6 +- 1.2 | 0.77 / 0.66 | 2.21 -> 1.43 |
| 99 | 0.00 | 2e-5 | -11.9 +- 0.9 | 0.08 / 0.09 | 1.23 -> 1.23 |

Small against the scale above (at t = 0 the learning rate is so small that J hardly moved).

Longer probes, 4000 steps, the last iterate (no averaging), held-out loss change:

| slice | t | lr 2e-4 (job 4317621) | lr 6e-4 (job 4317622) |
|---|---|---|---|
| 70 | 1.47 | -18.2 +- 1.1 | +2.4 +- 1.3 |
| 80 | 0.96 | -18.4 +- 1.2 | +4.1 +- 1.3 |
| 90 | 0.46 | -18.0 +- 1.6 | +9.8 +- 1.9 |
| 99 | 0.00 | +27.8 +- 4.5 | +62.8 +- 5.6 |

- J-only gains so far are at most about 18, a third of the "1 FID" scale at t ~ 1, and nothing at t = 0.
- Larger steps make it worse: J then random-walks (rms J / T 1.2 at lr 6e-4 against 0.7 at 2e-4). The probes
  evaluate the last iterate; the training evaluates an average of the iterates, which removes much of that noise.
  `train_J.py` does that.

### CelebA-64 cost (job 4317602; K16, patches 32:8:64 + 16:4:64; stopped by GPU memory in the probe at slice 60)

- Step: 2.9 -> 40 ms with one patch set (13.9 x), 10.8 -> 156 ms with both (14.5 x). A full tree training would
  take about 3.5 days against 6 h.

### Diagnostic: do generated images coordinate the hidden states as real images do? (CelebA-32, job 4317603)

Locked exact run, 2000 generated images per set, 100,000 real train images for the reference tables, real test
images in 9 blocks of 2000. "Cross gain" = how much one state tells about the other under the real images' rule
(nats; for real held-out images it is the real mutual information). Generated sets in % of the real value, at the
t = 0 slice:

| pair of hidden states | real, nats | no corrector | SNR corrector | dwell 0.1 | ODE |
|---|---|---|---|---|---|
| global <-> 16x16 patch | 0.286 | 68% | 74% | 78% | 58% |
| global <-> 8x8 patch | 0.211 | 77% | 84% | 86% | 68% |
| 8x8 <-> 8x8, touching (8 px apart) | 0.283 | 88% | 100% | 103% | 74% |
| 8x8 <-> 8x8, far (16 px apart) | 0.129 | 32% | 53% | 51% | 13% |
| 8x8 <-> 8x8, mirror images | 0.191 | 42% | 54% | 52% | 36% |
| 16x16 <-> 16x16, touching (16 px apart) | 0.185 | -2% | 13% | 10% | -11% |
| 16x16 <-> 16x16, mirror images | 0.195 | -6% | -1% | -7% | -10% |

- The generated images coordinate neighbouring 8x8 patches as real images do, but distant patches only about half
  as much, and the global state with the patches about 70-85%. The order of the samplers is the order of their FID
  (ODE 22.14 worst; correctors best).
- With more forward noise (the same images seen through earlier slices) the gap closes: global <-> 16x16 is 88%
  at t = 3, the 16x16 pairs 73%. So the coarse layout is right and the fine state combinations are not.
- Robustness (rerun 4317619): state combinations that none of the 100,000 reference images shows make up 0.1-0.5%
  of the generated sets' pairs against 0.1-0.2% for real held-out images. At about -10 nats each they explain a tenth
  of the gaps above at most, so the gaps are not an artefact of a few impossible combinations.

### The same diagnostic on CelebA-64 (job 4317620; locked exact run, 32x32 and 16x16 patches), t = 0, % of real

| pair of hidden states | real, nats | no corrector (FID 43.15) | SNR corrector (38.16) | dwell 0.3 (36.46) |
|---|---|---|---|---|
| global <-> 32x32 patch | 0.242 | 81% | 86% | 107% |
| global <-> 16x16 patch | 0.157 | 79% | 84% | 107% |
| 16x16 <-> 16x16, touching (16 px apart) | 0.248 | 74% | 87% | 116% |
| 16x16 <-> 16x16, far (32 px apart) | 0.115 | 19% | 39% | 81% |
| 16x16 <-> 16x16, mirror images | 0.174 | 25% | 37% | 65% |
| 32x32 <-> 32x32, touching (32 px apart) | 0.148 | -10% | 5% | 33% |
| 32x32 <-> 32x32, mirror images | 0.169 | -12% | -6% | 16% |

- The same pattern, stronger: without a corrector distant patches are coordinated a fifth to a quarter as much as in
  real faces. **The physical corrector (dwell 0.3) restores most of it** (far pairs 19% -> 81%), far more than the SNR
  corrector; this is the first mechanistic reading of why the dwell helps at 64x64: relaxing at a frozen slice lets
  distant parts agree. Values above 100% (global <-> patch, touching patches) mean the dwell images follow the real
  rule more strictly than real images do, i.e. they are less varied in these respects.
- Global-state usage also moves toward the real one: KL to real 0.165 (none), 0.043 (SNR), 0.022 (dwell 0.3).

### J on top of the locked CelebA-32 model, all slices, then sampled (submitted 2026-11-25)

`train_J.py`: everything frozen, J trained from the first patch slice (10) to the last with warm starts, 2000 Adam
steps per slice, the slice's J = the average of the iterates over 1000 steps. Two step sizes: lr 5e-5 (training
4317649, sampling 4317650) and lr 2e-4 (4317651, 4317652); Inception FID 4317653 (against the locked 20.20),
diagnostic on the new samples 4317654. `sample_tree.py` is the locked sampler with the tree's force; it checks at
every start that J = 0 reproduces the locked drift. `tree_force` now costs about as much as today's force (two small
products per patch set instead of (P, B, K, S) exponentials; tested against the direct form, job 4317648).

### Trainer test (job 4317680, 8x8 toy problem, CPU): all passed

- `train_tree.py` without `--tree` = the locked `train_latent.py`: every array of four slice files bit for bit.
- With `--tree`: J zero before its set's switch-on, trained after; validation finite; config records `tree`.
- Resume: the last slice removed and the call repeated: it resumes from the saved J and reproduces the slice exactly.

### Full tree training on CelebA-32 (submitted 2026-11-25 17:50)

The locked recipe's command line plus `--tree` (K16, patches 16:4:64 from slice 10 and 8:2:64 from slice 55, exact,
seed 1), all parameters trained together; J starts at zero when its patch set switches on. Run
`out/celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_tree_patch16s4S64on10+patch8s2S64on55_seed1`.
Training **4317684** (36 h limit; about 14 h of steps expected plus validation), sampling 4317685 (n = 2000,
quench 1), Inception 4317686 (against the locked 20.20), diagnostic 4317687.

Not started: Krotov's variant (the top unit without pixel couplings). With W = Q = 0 and J = 0 at the start all
global states are identical and stay so (the same gradient for each), so it needs a start that tells them apart
(for example J from the co-occurrence of image clusters and patch states); to be designed.

### Convergence (the user, 2026-11-25: "make sure that the training actually reaches convergence")

`convergence_report.py`: from the stored validation curves, per slice the change of the validation loss over the last
third of its steps and over the last check (paired over the 4096 validation samples); for a tree run also J per slice.

**The locked CelebA-32 run (job 4317693): converged for t > 2.5, slowly falling for 0.7 < t < 2, clearly not converged
for t < 0.6** (as the review of 11-23/24 found):

| slice | t | steps | change over the whole slice | over the last third | over the last check (~900 steps) |
|---|---|---|---|---|---|
| 40 | 2.98 | 19,001 | -4.4 | -0.2 +- 0.2 | 0.0 |
| 60 | 1.97 | 23,101 | -16.5 | -2.0 +- 0.5 | 0.0 |
| 80 | 0.96 | 27,101 | -51.5 | -4.4 +- 0.6 | 0.0 |
| 90 | 0.46 | 29,101 | -168 | -13.4 +- 0.9 | -0.2 |
| 94 | 0.25 | 29,901 | -402 | -40.7 +- 1.4 | -3.2 |
| 97 | 0.10 | 30,501 | -1148 | -124 +- 3 | -5.4 |
| 99 | 0.00 | 30,901 | -3661 | -467 +- 8 | -36.5 +- 1.2 |

(1 FID is a loss difference of about 50 at t ~ 1 and about 1000 at t = 0.) The slices with t < 0.5 still gain 1052
in their last thirds together.

**Plan for the tree run:**
- the same report for the tree run next to the locked one, when it finishes (job 4317703);
- a continuation of the tree run with **4 x the steps in slices 85-99** (`--init-from` the tree run, `--init-slices 85
  --steps-schedule 85:4`, the review's fix; `--init-from` now works with `--tree`, trainer test 4317692): training
  **4317700** (after the tree run; about 16 h), sampling 4317701, Inception 4317706, report 4317707. Its locked
  counterpart exists: the review's run `celeba32g_K16_exact_locked-patches-64states_4x-training-steps-in-slices-85-to-99_seed1`
  (19.26 at its best quench, 0.7); the Inception job scores both at quench 1.
- So the comparison is 2 x 2: locked / tree, each with the paper's steps and with 4 x the steps in the last 15 slices.

### RESULT: J on top of the frozen locked CelebA-32 model (2026-11-25 19:00; trainings 4317649 / 4317651)

Everything frozen, only J trained at every slice from 10 on (2000 Adam steps per slice, averaged iterates, warm
starts); 2000 samples with the locked sampler + quench 1 (4317650 / 4317652; the J = 0 drift check passed, 7e-16).

| model | Inception FID (sd) | FID test | FID_inf | KID x1e3 |
|---|---|---|---|---|
| locked, no J | 20.20 (0.34) | 19.59 | 16.49 | 12.39 |
| **+ J, lr 5e-5** | **19.15** (0.30) | 18.54 | 15.46 | 11.75 |
| + J, lr 2e-4 | 19.29 (0.16) | 18.67 | 15.71 | 12.01 |

(Inception 4317653; floor 6.43.) **About -1.0 FID from the coupling alone**, with nothing else re-trained and both
step sizes agreeing. The coordination diagnostic on the new samples (4317654), t = 0, % of real:

| pair of hidden states | locked | + J, lr 5e-5 | + J, lr 2e-4 |
|---|---|---|---|
| global <-> 16x16 patch | 68% | 85% | 85% |
| global <-> 8x8 patch | 77% | 91% | 92% |
| 8x8 <-> 8x8, touching | 88% | 97% | 98% |
| 8x8 <-> 8x8, far (16 px) | 32% | 50% | 52% |
| 8x8 <-> 8x8, mirror images | 42% | 58% | 58% |
| 16x16 <-> 16x16, touching | -2% | 22% | 21% |
| 16x16 <-> 16x16, mirror images | -6% | 15% | 14% |

- J does what it should: the global state and the patch states agree almost as in real images, and distant patches
  agree more (through the global unit), about as much as the correctors achieved, without a corrector.
- Held-out loss gain per slice (lr 5e-5): -1 (t = 4), -6 (t = 3), -9 (t = 2.5), -22 (t = 2), -22 to -27 for
  t < 1.5; 80 of 90 slices beyond 3 standard errors, none worse. Small against the "1 FID" loss scale at t = 0
  (about 1000), so the FID gain comes from the middle of the process, where the layout is decided.
- rms J / T grows to 2.6 (16x16 set) and 1.4 (8x8 set) at lr 5e-5, to 6.6 / 4.7 at lr 2e-4 with the same FID: part
  of the larger J is noise.

### Full tree run, state 2026-11-25 20:00

Training 4317684 at slice 52 of 100 after 2.1 h (102 steps/s with one patch set; from slice 55 both sets, about
15 min per slice): expected to finish around 07:30 on 11-26.

Gallery of the J-on-top runs (job 4318055; same start and noise per column):
`out/galleries/2026-11-25_celeba32_locked-K16-64-states_exact_with-vs-without-global-patch-coupling-J-trained-on-top_SDE_quench1_noise-start`.
Looked at: the same faces in every row; J changes details only (glasses and hat edges, mouth, hair); no change of
identity or pose.

### Overnight jobs submitted 2026-11-25 21:00 (the user: "lets run 1-4 ... but already start a tree run on CelebA 64")

All J-on-top runs: lr 5e-5, 2000 steps per slice, 2000 samples + quench 1 with `sample_tree.py`.

| what | training | sampling | Inception | diagnostic |
|---|---|---|---|---|
| 1. J on top of the locked CelebA-64 model (the largest coordination gap; validation in chunks of 128) | 4318124 | 4318125 (chunk 25, platform allocator) | 4318126 (next to locked none / SNR / dwell 0.3) | 4318127 |
| 2a. J on top of the locked CelebA-32 model, seed 2 (review folder's run) | 4318128 | 4318129 | 4318132 | |
| 2b. J on top of the review's better-converged model (4 x steps in slices 85-99) | 4318130 | 4318131 | 4318132 | |
| 3. the J-on-top CelebA-32 model (seed 1) with the SNR corrector 2:0.16 and with dwell 0.1 | | 4318133, 4318134 | 4318135 (next to the locked correctors) | |
| 4. J on top of grey CIFAR K16 | 4318136 | 4318137 | 4318138 | |
| 5. full tree training on CelebA-64 (`train_tree.py --tree`, the locked 64x64 recipe; 5-day limit, about 3.5 days expected; can be stopped) | **4318139** | | | |

Mirrors of runs from the review folder are written below this folder's `out/` (the review folder is only read).
At 21:11 all first-stage jobs had started cleanly (CelebA-64 J training about 70 s per slice with one patch set;
the corrector sampler passed the J = 0 drift check); 4318139 was waiting for a GPU.
