# K > 1 Ising hidden units: changes in the staging folder

Staging folder: `physical-diffusion-models/latent_physical_SGM_ising_dev/`, next to the live folder
`physical-diffusion-models/latent_physical_SGM/`. `PKG` and `OUT_ROOT` in `latent_model.py` resolve to the same project
folder and the same `out/latent_physical_SGM`. Specification: `k_ising_spec.md`, sections 2.1 to 2.4.

Nothing in the live folder was edited, moved or created. Nothing was executed: no Python, no JAX, no Slurm job.
The only check that ran is `bash -n` on the test script, which parses it without running it. Everything below still
has to be verified by test 20 (`run_test20_latent_neuronic.sh`).

## Files

| file | status |
|---|---|
| `latent_model.py` | copy of the live file plus new functions after `prototype_reference` (additive only) |
| `train_latent.py` | copy plus the edits of spec 2.3 (listed below) |
| `cd1_check.py` | copy; `--local-rule` also handles Ising units |
| `sample_latent.py` | unchanged copy (`make_drift` already handles K Ising units with Q and per-pixel units) |
| `inspect_states.py` | unchanged copy |
| `run_test19_latent_neuronic.sh` | unchanged copy, used as the template only (it still points at the live folder) |
| `inspect_units.py` | new |
| `run_test20_latent_neuronic.sh` | new |
| `CHANGES.md` | this file |

## latent_model.py

Nothing existing changed. The diff against the live file has no removed lines. The new section after
`prototype_reference`:

- **`SPLIT_CLASS, SPLIT_KMEANS, SPLIT_MEDIAN = 0, 1, 2`**: how a node was split (the tree's `kind`).
- **`split_tree(X, labels, K, seed=0, pooled=False, n_min=20, iters=50) -> dict`**
  - **perclass.** First the class tree:
    - an exhaustive Ward bipartition of each multi-class leaf's class set (2^(c-1) - 1 candidates);
    - leaves are split in order of largest gain, until every class is a leaf (C - 1 nodes);
    - the right child holds the node's highest class, so for C = 2 the right child is class 1.
  - **pooled.** No class tree.
  - **Then greedy bisecting 2-means:**
    - the candidate for a leaf is `kmeans(X_leaf, 2, seed + 1000 + region)`;
    - the balance guard switches to a median cut along c2 - c1;
    - the right child is the larger side;
    - the leaf with the largest Ward gain is split next.
  - Raises `ValueError` if K < C - 1 (perclass), if K < 1, or if no leaf can be split any more.
  - **Returns** everything listed in the spec, plus `kind`, and the settings `seed`, `pooled` and `n_min` as 0-d
    arrays (so that `split_tree.npz` records them). Every value is a numpy array, so `np.savez(**tree)` works.
- **`split_reference(data, labels, taus, T, K, seed=0, pooled=False) -> (W (S,K,N), b (S,K), tree)`**
  - S_w is accumulated leaf by leaf, and at K = 1 in exactly the order of `fisher_reference`.
  - One `np.linalg.solve(S, e * D.T).T` per slice.
  - b_j is a row-wise dot product, `-(W_j @ (e m_j)) + (T/2) log(n_R/n_L)`.
  - At K = 1 perclass the operations are those of `fisher_reference`, with the factor 1/2 applied before the solve
    rather than after. Halving is exact in floating point, so the result should match bit for bit. Test 0a prints
    whether it does; it requires only 1e-9.
- **`random_orthogonal_rows(K, n, norm, rng) -> (K, n)`**: the QR of an n x K normal matrix with signs fixed by
  diag R, times `norm`. Raises if K > n.
- **`ising_force_matching_local_grad(params, X, T, vis, local=False) -> dict`**: the spec's rule, vectorised over the
  batch (W, b, Q if present, wl and bl if local).
  - The per-pixel part is copied from `force_matching_local_grad`.
  - I re-derived the formulas from `hidden_terms_quad` and the loss -lap F/T + |grad F|^2/(2T^2), and they agree
    with the spec, including the 2 m_j/T term of dJ/dQ.
- **`ising_fields(W, b, Q, X)`**: new helper, not in the spec. It returns the numpy fields A in the summation order of
  `hidden_terms_quad`; check 0h compares the two.
- **`_row_basis(W)`**: a private helper for `overlap_ref` (an SVD basis with a rank cut-off).
- **`ising_unit_stats(W, b, Q, X, T, W_ref=None, W_prev=None, tree=None, leaf=None, labels=None) -> (rec, arrays)`**
  - It returns every field and array of the spec.
  - Keys whose inputs are missing are left out.
  - `straddle_off` and `own_balance_err` are `None` when no unit has the samples they need (for example K = 1:
    the root has no samples outside its subtree).
  - Two definitions are my choice; see "Uncertain points" 1 and 2.
- **`force_misfit(X, psi, T)`**: sym(X^T psi)/(B T) - I, in numpy.

## train_latent.py

Every edit, in file order:

1. **Module docstring.** A paragraph on K > 1 Ising units.
2. **Help texts.** `--hidden`, `--proto-init` (now `default=None`) and `--seed`.
3. **After `parse()`.**
   - `proto_init` is resolved: `pooled` if `hidden == "ising" and K > 1`, otherwise `perclass`.
   - `fisher = hidden == "ising" and K == 1 and proto_init == "perclass"`.
   - `tree = None`.
4. **Validation.**
   - The K > 1 refusal is deleted.
   - The two-class refusal now applies only when `fisher` is true; its message is unchanged.
   - New, for Ising units without `fisher`: `--seed >= 1`; `K <= n_vis` (from `problem_geometry`, before any folder
     is created); `perclass` needs K >= C - 1.
5. **`--proto-init random`.** Now refused only for K = 0 or with `--hidden-fixed`. The message text changed: it no
   longer says "--hidden potts".
6. **config `init`.** `"fisher"` if `fisher`; `"split-tree"` for any other Ising K > 0; otherwise unchanged. No new
   config keys.
7. **References.**
   - `if K == 1 and args.hidden == "ising":` became `if fisher:`; the body is unchanged.
   - New branch `elif K > 0 and args.hidden == "ising":`:
     - calls `split_reference` with k-means seed `100 (seed - 1)` and `pooled = proto_init in (pooled, random)`;
     - checks that the reference is finite;
     - computes W_lr and b_lr from k-means seed 0 (reused when km_seed is 0), sW over axis (1, 2) and
       sb = max(rms_j b, T);
     - writes one log line: the tree summary, |W*|, rms b*, sW and sb at t = 5 and t = 0, the per-unit rms range of
       W_lr at t = 5 and t = 0, and the random-start note;
     - saves `split_tree.npz` through the existing `save_atomic`, i.e. `np.savez` to a temporary file, then
       `os.replace`.
8. **Start.**
   - `elif args.hidden == "ising":` became `elif fisher:`.
   - New `elif args.hidden == "ising" and args.proto_init == "random":`: `r0 = sqrt(mean_j |W_ref[0, j]|^2)`,
     `W0 = random_orthogonal_rows(K, n_vis, r0, default_rng(1000 + seed))`, `b0 = 0`.
   - The Potts random branch and the final `else` are unchanged.
9. **Slice loop.**
   - First statement: `W_prev = np.asarray(params["W"]) if tree is not None else None`. On every existing path it
     does nothing.
   - In `--hidden-fixed`: `one = fisher`.
10. **Record.**
    - The old Ising `elif K > 0:` became `elif K > 0 and fisher:` (unchanged).
    - New `elif K > 0:` calls `lm.ising_unit_stats(W, b, Q, Xv, T, W_ref=W_ref[k], W_prev=W_prev, tree=tree,
      leaf=tree["leaf_of"][idx], labels=yv)`, then `rec.update(stats)` and `extra.update(arrays)`. The arrays
      `unit_norm`, `unit_cos_ref`, `unit_minority`, `unit_abs_m` and `unit_straddle` go into each slice file.
11. **Message.**
    - Same split: `elif K > 0 and fisher:` is unchanged.
    - New `elif K > 0:` appends: `|W| (ref), overlap, dead/weak/decided, max cos, unit-0 acc, leaf acc, <|m|>,
      softening/T, straddle` ("n/a" if None).
    - `|Q|` comes from the existing `Q_norm` block.

**Old paths unchanged.** For every invocation that was valid before, except the case below, the options, the
resolved `proto_init`, the config keys and values, the reference, the start, the key stream and every log line are
the same as before. The code paths for K = 0, the K = 1 Fisher unit and Potts units are textually identical.

**The one behaviour change, and it is intended.** `--hidden ising --K 1 --proto-init pooled` used to ignore the
option and run the Fisher path with labels.
- It now runs the label-free single unit (spec 2.1), with `init` "split-tree".
- If an old run with this combination were resumed, the config check would refuse it (`init` differs).
- I grepped the `config.json` of the runs in `out/latent_physical_SGM` (one level): none of the 24 Fisher runs
  uses `pooled` or `random`.

## cd1_check.py

- `--local-rule` now uses `lm.ising_force_matching_local_grad` when `hidden == "ising"`, and
  `force_matching_local_grad` otherwise.
- The condition `hidden == "potts" and K > 0` became `K > 0`.
- The help text was updated. The Potts path is unchanged.

## inspect_units.py (new, CPU)

**Usage:** `python inspect_units.py RUN[:SLICES] ... [--samples STEM] [--max-units 32] [--probes 1000] [--top-novel 32]`.

**Tree.** Rebuilt from the run's config with `split_reference`. An old K = 1 Fisher run is the perclass tree with
K = 1. If the run has a `split_tree.npz`, the rebuilt tree is compared with it.

**Per slice:**
- the stats line of `ising_unit_stats`;
- per unit (the first `--max-units`): depth, child sizes, purity, the class mix with a<0 and with a>0, the minority
  share, ⟨|m|⟩, the straddle, |W_j| and its cos with the reference;
- an image grid with the left and right child means (clean pixels), W_j and Q_j;
- the |cos| matrix.

**With `--samples STEM`** (`<run>/samples/STEM.npy`), at slice 99:
- codes sign(a) of the samples, of the probes (held-out training images, default 1,000) and of the rest of the
  training images (the code book);
- the Hamming distance to the nearest code, counted only over the units with |a| > 2T in both;
- the novel-code rates and their distribution;
- per-class activity in `<run>_activity.txt`;
- a gallery of the most novel samples.

**Output:** `out/latent_physical_SGM/units/<job>/` (`summary.txt`, `<run>_metrics.json`, PNGs).

No Slurm script: after landing, `sbatch run_cpu_neuronic.sh inspect_units.py RUN:0,25,50,99 --samples sde_base_n500`.
In the staging folder, test 20 runs it inside its GPU job.

## run_test20_latent_neuronic.sh (new)

**Setup.**
- GPU job laid out like test 19, with `--time=01:30:00`. The spec says about 1 h; the margin covers the 100-slice
  run, which recompiles the training scan for every new remainder of steps per slice.
- Submit from the staging folder. The script refuses to start anywhere else.
- `D=/physical-diffusion-models/latent_physical_SGM_ising_dev`, `OUT=../out/latent_physical_SGM`.
- All run names have the form `smoke20_*_${SLURM_JOB_ID}`.

**Exit-code bits:**

| bit | meaning |
|---|---|
| 1 | unit checks |
| 2 | regression run failed |
| 4 | smoke run failed |
| 8 | invalid options accepted |
| 16 | default path differs |
| 32 | resume failed |
| 64 | smoke checks failed |
| 128 | 100-slice run, sampler or inspect_units failed |

**0. Unit checks a-h** (one Python heredoc, bit 1), then **i** as a print only:
- **a.** K = 1 identities on mnist14 and mnist28, all 100 slices.
  - perclass equals `fisher_reference`; the check also prints whether the match is bit for bit.
  - pooled equals `fisher_reference(data, leaf_of)` once (T/2) log(n_R/n_L) is subtracted from b; see "Uncertain
    points" 6.
- **b.** `split_tree` invariants for K in {2, 8, 33, 128}, perclass and pooled, and K = 8 on mnist14_d012.
  - The leaves partition the images.
  - The children of every node are disjoint, their union is the node, and they agree with H and `leaf_of`.
  - Following H's signs from the root leads to each leaf.
  - n_L and n_R >= n_min, and the stored means and counts are correct.
  - perclass: the class tree comes first and every later node is class-pure.
  - pooled: the structure is unchanged when the labels are permuted.
  - Two calls give identical trees.
  - k-means seed 100 must give a different tree at K = 128 only; for the other K it is printed (see "Uncertain
    points" 7).
  - Also: the error is raised when the leaves run out.
- **c.** `split_reference` checks for K in {8, 32, 128}, pooled and perclass. The minimum consecutive cosine is
  printed with its slice and unit.
- **d.** Start scale on 4,096 draws of p_5; W W^T = r0^2 I to 1e-12 (relative to r0^2); seeds 1 and 2 differ.
- **e.** Local rule against `jax.grad(make_loss)`.
  - Visible model: von Neumann radius 1, quartic self terms, lin springs only, scaled by (1 + 0.05 ξ).
  - W and b from slices 1, 50 and 99 of the K = 8 pooled reference; Q random at 0.1 × rms W.
  - With and without per-pixel units, and also without Q; relative difference <= 1e-9.
- **f.** Loss invariance under a unit permutation with random sign flips, to 1e-12. The hidden gradient at
  W = b = Q = 0 must be exactly 0.
- **g.** `make_drift` with K = 8 Ising units, Q and per-pixel units, von Neumann radius 1, at knots 20 and 60.
  - The hidden drift is the drift minus the drift with all hidden histories at zero.
  - The hidden force is `make_force` with theta = 0, where the visible force is exactly 0.
  - Requirement: hidden drift = -2 × hidden force to 1e-10.
- **h.** A K = 3 constructed model: decided (a = 50T x), weak (0.01T x) and dead (10T + 0.5T x).
  - The fields equal `hidden_terms_quad`'s A with Q != 0.
  - The stats use those fields.
  - Also prints the K = 8 reference's leaf accuracy at t = 0.
- **i.** Print only: `force_misfit` at slice 40 (t = 2.98), 4 × 4096 samples.
  - Runs: `mnist14_K0_ema_quartic` (radius 7) and `mnist14_K0_ema_quartic_vn1_lin` (nearest neighbour). The latter
    was still training when this was written (61 slices); a missing slice is skipped.
  - Prints the rms of G on the diagonal, on coupled pairs and on uncoupled pairs, a noise estimate and the top
    eigenvalues.

**1. Default paths.**
- Four regressions, 2 slices at full steps: K = 1 quartic, K = 1 quad, Potts 128 + both power 4, Potts 128 random.
  The `val_loss` strings are compared with the production runs' first two lines (bit 16).
- Diagnosis only, no extra bit: if a string differs, the same flags are run once with the live code, and the script
  prints whether the live code matches the production run and the staging code. This separates an older difference
  from one introduced here.
- Resume check: `--out-root` exists in `train_latent.py`, so it runs. The script still checks for the option and
  says so if it is missing.

**2. Smokes.**
- The 10 runs of the spec. Common flags: `--hidden ising --visible quartic --quad --local --select ema --radius 1
  --neighbourhood von_neumann --pair-terms lin --steps-scale 0.02`. The `--hidden-fixed` run has no `--quad`.
- Each run is followed by a check script:
  - `init` is split-tree and `proto_init` the resolved one;
  - `split_tree.npz` equals a fresh build;
  - slice 0 equals the start exactly (the seeded Haar draw for random);
  - with `--hidden-fixed`, every slice equals the reference exactly;
  - lr_W and lr_b equal 5e-4 × the seed-0 scales (0 while frozen or fixed);
  - n_dead = 0 and min_cos_prev > 0 in every slice;
  - for the 100-slice run, the shapes in `hidden_history`.
- Then `sample_latent --n 20 --chunk 20`, which must report `n_nonfinite` 0, and `inspect_units RUN:0,50,99
  --samples sde_base_n20`.

**3. Refusals.** The seven cases of the spec. Each must exit non-zero and write no `slice_000.npz`; the script prints
the error line.

## Not implemented, and why

- **Spec 2.4 part 4 (after the first real run).** It needs a trained Ising run.
  - Commands: `cd1_check.py --run RUN --slices 1,50,90,99 --local-rule --skip-cd1` (GPU) and
    `inspect_units.py RUN:0,25,50,99` (CPU).
- **Landing.**
  - Copy to a temporary name, then `mv` into the live folder.
  - Update DESIGN §5 item 4, add DESIGN §2.2 (the Ising local rule) and update the README's options and Files table.
  - Not done here because they would edit the live folder.
- **Items the spec drops or defers.**
  - Dropped: the class gate, the ITQ and code variants, the Hungarian alignment, new CLI options and new config keys.
  - Optional remedies, deferred until seen: re-centring one-sided units, and an Ising visit-count rule for measured
    statistics.
- **The "spurious-code rate".** The review calls it ill-defined. `inspect_units` reports the novel-code rate
  against real probes instead.
- **No Slurm wrapper for `inspect_units.py`.** `run_cpu_neuronic.sh` already runs any script of the folder.

## Uncertain points (worth a look when reading test 20's output)

1. **Minority share, which sets `n_dead` and `n_gated`.**
   - I use the spin's mean minority-state probability, (1 - |<m_j>|)/2. The share of samples with sign(a_j) in the
     minority is not used.
   - Reason: at t = 5 the fields of deep tree units sit near the prior term (T/2) log(n_R/n_L), about 0.2–1.1 T,
     with a spread of about 0.1–0.2 T.
   - So sign(a) has almost no minority there, although the spins still flip (⟨v⟩ > 0.3).
   - A sign-based count would call these units dead and make "n_dead = 0" fail at slices 0–2.
   - Straddle and leaf accuracy use the fields and signs, as specified.
2. **`own_balance_err`.** For each unit: |⟨(1 + m_j)/2⟩ over the samples of its own subtree - n_R/(n_L + n_R)|,
   averaged over units. The spec gives only the name, so this definition is mine.
3. **Class-tree details.**
   - The right child holds the node's highest class (for C = 2: class 1, as the spec requires).
   - Class splits are ordered by largest gain.
   - The 2-means seed uses the region id (root 0; node j's children are 2j+1 and 2j+2) as "leaf_id".
4. **`unit_class` and `unit_purity`.** `unit_class` is (K, 2), the majority class of the left and of the right
   child; `class_acc` maps unit 0's sign through it. `unit_purity` is the share of a node's images in their child's
   majority class.
5. **Exact equality between processes.** Several checks compare arrays from two processes:
   - slice 0 = W_ref[0], `--hidden-fixed`, `split_tree.npz` = fresh build;
   - the reference is rebuilt in the check process with numpy/LAPACK, and should match exactly on the same node;
   - if BLAS threading ever made it non-deterministic, the check prints the size of the difference.
6. **Pooled K = 1.** `split_reference` includes the prior term (T/2) log(n_R/n_L), which `fisher_reference` does not
   have, so check 0a compares b - prior. On 0/1 MNIST the perclass prior is exactly 0.
7. **"k-means seed 100 gives a different tree".** 2-means on large leaves often converges to the same partition
   whatever the start, so small trees can legitimately coincide. Only K = 128 must differ.
8. **Check 0c, cos(W_ref[k, j], W_ref[k+1, j]) > 0.9.**
   - Near t = 0, S(t)^-1 changes fastest along pure-noise pixel directions (S_w eigenvalues about 1.3e-3, against
     T = 0.02).
   - Deep units with small subtrees carry more noise in d_j.
   - My estimate gives a worst case of about 0.91, so the check may be tight; it prints the minimum and where it is.
9. **Smoke check "n_dead = 0" over the 100-slice run.** It depends on training: a unit could collapse late. It is
   a real finding if it fails, not necessarily a bug.
10. **Regressions against the production runs.**
    - The two K = 1 production runs were made with older code (sha 84ab8… and ff8171…).
    - If one differs, the live-code diagnosis shows whether the difference predates this change.
11. **Python was never executed here.** I checked the Python by reading only. JAX shapes: W, Q (K, N); b (K,);
    fields (B, K); everything float64; no Python control flow on traced values in the new jitted code.
    `ising_force_matching_local_grad` branches only on the static `local` flag and on the dict keys.
