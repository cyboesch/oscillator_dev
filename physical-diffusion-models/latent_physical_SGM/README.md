# Latent physical SGM: fast bistable hidden units

Branch `latent_physical_SGM`, started 2026-11-06 from `paper_code_v2` (9d89dea).

The idea: give the physical diffusion model K hidden bistable oscillators. They couple to the 784 pixel oscillators and equilibrate much faster than the pixels do. Averaged out, they give the pixels a free-energy landscape with a global, non-Gaussian many-body part, in particular a "which digit" variable that the present two-body energy lacks. The learning rule stays local.

**Key terms** (every other abbreviation is in the Glossary at the end):
- **Pixels (visible units):** one Duffing oscillator per pixel x_i, with the paper's energy E_vis = Σ_i [k_lin x_i²/2
  + k_duff x_i⁴/4 + k_6 x_i⁶/6 + bias_i x_i] + Σ_pairs [c_lin (x_i − x_j)² + c_om x_i² x_j + c_duff (x_i − x_j)⁴/4].
  The pairs lie within a Chebyshev radius (14 at 28x28: 59% of all pixel pairs; 7 at 14x14). **Power 6** is the
  paper's x⁶ confinement, **power 4** (`--visible quartic`) uses x⁴ instead. **Springs** (`lin`), the
  **optomechanical** term (`om`, asymmetric) and the **Duffing pair** term (`duff`) are the three pair terms.
- **Slice:** the diffusion time t = 5 → 0 is cut into 100 slices; every parameter (visible and hidden) has its own
  value in each slice.
- **Hidden units** are averaged out exactly, i.e. assumed infinitely fast (Caveat 1). Three kinds:
  - **Ising unit:** one bistable unit h = ±1 coupled to all pixels, energy −h (b + W·x + Q·x²). Only one is used so
    far (K = 1).
    (Correction from ChatGPT: The K = 1 restriction is outdated. The current code supports multiple Ising units,
    and Result 15 reports trained models with K > 1.)
  - **Potts unit (Potts M):** one unit with M states, exactly one active; state m has its own couplings W_m, Q_m,
    b_m to all pixels, so each state acts as a prototype. "Potts 128" is one unit with 128 states, not 128 units. In
    hardware: M bistable units with winner-take-all inhibition, or one M-well oscillator.
  - **Per-pixel units** (`--local`): one bistable unit per pixel, coupled only to its own pixel.
  - **Spring coupling** (`--quad`, Q): the hidden unit also couples to the squared pixels, so each state changes the
    pixels' stiffness. **"both"** = spring coupling + per-pixel units. **K** = number of Ising units or Potts
    states; K = 0 is the paper's model.
- **Exact vs measured statistics:** the learning rule needs hidden-unit averages (e.g. each Potts state's
  probability given an image). "Exact" computes them analytically; "measured" estimates them from **S** samples
  of the hidden unit per training image, as a device would (**S16** = 16 samples). **Reads** = hidden readouts per
  training image and step.
- **Jackknife (jk):** plugging S-sample averages into the (quadratic) loss biases the gradient by order 1/S; the
  combination 2·L(all S samples) − mean L(the two halves) cancels that order. **jk2** (second order) also uses the
  four quarters and cancels order 1/S² as well.
- **Optimisers for the hidden couplings:** **per-element Adam** (one adaptive step size per coupling), **plain
  steps** (one per parameter group), **gain per state** (one per hidden state), **visit-count rule** (each state's
  step divided by how often the samples visit that state).
- **Effective states:** exp(entropy) of the Potts states' average occupation; **lost states:** average occupation
  below 1e-32 at t = 0.
- **Recall / precision** (k-nearest-neighbour, k = 3, per class against real images): the share of real images
  covered by the samples (variety) / the share of samples that look like real images (fidelity).
- **base / ext:** two fixed sets of 500 sampling trajectories (start and noise), the same for every model, so
  models are compared **paired** (McNemar tests, paired bootstrap; [a, b] = 95% interval). `ext` was never used to
  choose a model.

**How to read the numbers.** Each generated sample is scored by three flags (`blend_check.py`,
`evaluate_samples.py`; Result 2). "Real images" below are the 1,000 probes: training images scored leave-one-out,
not unseen test digits (Caveat 6).
- diverged: an amplitude above 15 normalised units (6.2% of the paper's samples at 28x28; none for the Potts +
  both models of Results 7 and 8);
- blend: a 0 and a 1 overlaid: one 0 template plus one 1 template fit the sample clearly better than any single
  template (a squared residual more than 25% smaller at 14x14, 16% at 28x28; the threshold lets 5% of real images
  count as blends);
- far: its nearest training image is farther away than for 90% of real images (14x14). At 28x28 a fixed distance,
  4.84, inherited from an earlier study, which 13.2% of the real images exceed;
- clean: none of these. Its class (0 or 1) comes from a linear 0-vs-1 discriminant.

The flags overlap: a far blend counts in both rates, and the raw far rate also counts diverged samples (the
paper's 35.6% far at 28x28 includes its 6.2% diverged).

**Clean yield (50/50)** is the share of clean samples among the non-diverged 0s and among the non-diverged 1s,
averaged, so drawing more of the easier 1s does not help. Diverged samples are left out, not counted as failures
(the paper's 31.9% at 28x28 is over its 469 non-diverged samples). It is a pass/fail test per sample: not
diverged, not an overlay, and about as close to some training image as real images are. Recognisability itself is
not tested (a sample between the two classes still passes if it is close to a training image). The blend cut, and
at 14x14 the far cut, are set on real images, so real images score 85.7% at 14x14 and 83% at 28x28 (lower there
because of the inherited far cut), and a model can score above them by drawing only typical digits. For three and
four digits the same tests are generalised to C classes (`evaluate_multiclass.py`). Clean yield says nothing about
variety (recall of the real images' variety, `coverage_samples.py`) or copying (the copy check), and a pixel
distance misses small flaws (a faint ghost, a small gap). So I also look at the images
(`out/latent_physical_SGM/gallery/INDEX.md`; what I saw is in Result 13).

**Status (2026-11-09, 17:30).**
- **Newest (afternoon, Result 15): the simplest pixel model works, if the hidden units carry the structure.** Pixels
  with x^4 potentials coupled only to their nearest neighbours by linear springs (8 neighbours, or just the 4
  nearest), no long-range or optomechanical pixel couplings, exact training, 14x14 0/1:
  - Without hidden units nothing is a digit (0% clean). With one Potts unit the clean yield rises with every
    doubling of the states: 32 / 128 / 256 / 512 states give 65.5 / 81.4 / 85.0 / **89.4%** (8 neighbours) and
    58.2 / 74.2 / 80.4 / **87.8%** (4 neighbours); recall of the 0s rises too (0.61 -> 0.83), copies stay <= 0.4%.
  - Hidden states replace pixel couplings: the 4-neighbour deficit shrinks from 7.2 points at 128 states to 1.6
    (not significant) at 512. So the hardware-friendly pixel model (bistable-ready x^4 pixels, nearest-neighbour
    springs) costs almost nothing once the hidden unit is large enough.
  - **K Ising units do as well as K Potts states** (32: 66.7 vs 65.5%; 128: 78.7 vs 81.4%; neither difference
    significant), from 1 unit (38.7%) up to 196 (81.1%, flattening: +2.5 over 128, p = 0.05). Ising units are the
    easier hardware story (each is one bistable element; a Potts unit with M states is M bistable elements plus a
    winner-take-all constraint), with the same wiring per element (each coupled to every pixel).
  - Ising samples differ in kind: more 0/1 blends (6-7% against 1-2%), 1s 93% clean against 98%, and a failure the
    Potts models do not have, faint dotted near-empty images (about 10 of 100 flawed at 196 units, against about 3
    for Potts 512). A likely reading, untested: a Potts unit is a mixture (each sample picks one prototype), Ising
    units a product of on/off features, which can combine features of both digits.
  - More Ising units (22:00): 512 units 83.5%, 1024 units 85.4%, still 4 points below Potts 512 (about +2 per
    doubling); on four digits Ising is flat from 256 to 512 (67%) and 12 points below Potts 512 (79.0%), with
    14-16% blends, and the Ising samples hardly change with K (the extra units seem unused; untested).
  - **Potts samples are not "prototype plus blur"** (Result 15, prototype test): each state partitions the data like
    a stored pattern of a dense associative memory, but samples sit only 5-11% closer to their state's prototype than
    held-out real images and vary within a state 90-100% as much as real members, independent of the number of
    states (32-512); a third of the samples have their nearest training image in another state, as real images do.
    (Correction from ChatGPT: These diagnostics support substantial variation around the prototypes, but do not
    rule out every prototype-plus-noise model. Here the real probes are excluded from centroid construction,
    not from training the generative model.)
  - **28x28 best model trained locally:** Potts 512 + both, power 4, S = 64 with the visit-count rule, 90.5%
    against 92.3% with exact training (-1.8 [-4.3, +0.7], not significant), 31.9% for the paper.
  - **Idle Ising units are not the limit** (2026-11-10, Result 15): resetting weak units to the split-tree formula
    every slice keeps all 512 units active but gives +1.3 points on 0/1 (n.s.) and -0.1 on four digits; a 10x step
    size for weak units does not revive them. The Ising-Potts gap is in the model form, not the optimiser.
    (Correction from ChatGPT: The tested remedies did not close the gap. This does not rule out undertraining,
    other optimisation choices, or a combination of optimisation and model limitations.)
  - **28x28, four digits, simplest pixel model (2026-11-10, Result 15):** Potts 784 92.2% clean (real probes
    86.8%), Ising 784 66.2%, frozen Potts 784 74.6%. Grainy strokes, no copies.
  - Running (18:30): 28x28, 0/1/2/3, the simplest pixel model, Ising 784 / 1568 / 3136 and Potts 784 / 1536
    (trained and frozen). Since 22:40 / 01:00: 28x28, all ten digits, Potts 1536 / 3072 (trained and frozen) and
    plain Ising 3136.
- **Hidden units improve the samples a lot, and more hidden states help, up to about 256** (Results 7 and 8).
  - At 14x14, 512 states do not differ from 256 on independent trajectories and overshoot towards the training set
    (with the full, long-range power-6 pixel model; with the simplest pixel model 512 states do help, +4.3 and +7.4
    points over 256, Result 15).
    At 28x28, 512 states with power 4 give the best samples so far: 92.3% clean yield, 1.6% blends, recall of real
    0s 0.68 (Result 9; not significantly better than 256 states on clean yield).
  - Design: Potts units (one M-state hidden unit) coupled to the pixels and to their squares (spring), plus one
    bistable unit per pixel ("Potts M + both").
  - 28x28, Potts 256 + both, against the paper on 500 paired trajectories:
    - blends 3.4% vs 43.2%; far from every training image 6.0% vs 35.6%; diverged 0% vs 6.2%;
    - share of 1s 52% vs 68%; class-standardised clean yield 90% vs 32% (real images 83%).
  - Going from 32 to 128 to 256 states at 28x28:
    - far (50/50) 19.6% -> 9.7% -> 6.2%;
    - recall of the real 0s' variety 0.26 -> 0.46 -> 0.55 (real 0.89).
    - The remaining gap to real images is variety: the samples are more typical than real images.
  - The 32- and 128-state results are confirmed on 500 independent trajectories; 256 states is `base` only.
  - Labels are not needed: k-means prototypes of all images do as well as per-class ones, at 14x14 and 28x28.
- **Power 4 instead of power 6 works once there are enough hidden states** (Result 9).
  - At 28x28 the far cost of power 4 falls from 7.3 points with 32 states to 3.0 with 128 and 1.8 (not significant)
    with 256.
  - With 256 states the clean yield is the same (90.3%), and power 4 gives more variety (recall of real 0s 0.61 vs
    0.55).
  - With 512 states and power 4: clean yield 92.3%, blends 1.6%, recall of real 0s 0.68 (`base` only).
- **The hidden units can be trained locally, with measured statistics** (Result 10).
  - **New overnight (2026-11-08/09): a device-friendly optimiser for the hidden couplings.** Steps with one step
    size per parameter group lose most hidden states, even with exact statistics. A per-state step divided by the
    state's own visit count (the "visit-count rule": one visit counter per state, one leaky average per coupling)
    keeps most of them (with 2-64 measured draws 17-41 of the 128 states are still lost, against 2 with exact
    statistics):
    - at 14x14 0/1, 16 hidden reads per configuration (S = 8 plus 8 for the visit counts) give 81.6% clean yield,
      against 75.6% for per-element Adam with the same 16 reads (+6.0 [+2.9, +9.2]); exact statistics 84.3%;
    - 8 reads (S = 4) still give 82.2%, +6.6 [+3.4, +9.8] over per-element Adam with twice the reads. With 4 reads
      (S = 2) the rule falls to 73.7%, the level of per-element Adam with 16 reads (-1.9 [-5.3, +1.4]); its 0s
      fail, and only 29 of its 102 used states win 0s;
    - a second-order jackknife (the same reads, extrapolated over all draws, their halves and their quarters)
      lifts per-element Adam at S = 16 from 75.6% to 80.7% (+5.1 [+2.3, +8.0]), as much as 16 times the reads
      did. With the visit-count rule it adds only +1.5 ([-1.1, +4.1], not significant): 82.9%, indistinguishable
      from exact training (-1.5 [-4.0, +1.1]) but below S = 256 (-2.8 [-5.3, -0.2]);
    - on three digits, S = 64 with the rule is indistinguishable from exact training (80.1% vs 81.1%, -1.0
      [-4.2, +2.2]), against 73.1% for per-element Adam with S = 64;
    - **the visit counts can come from the same draws as the statistics at no measurable cost (S = 8 and 16, one
      run each), so the rule need not read more than per-element Adam:** S = 8 with same-draw counts (8 reads)
      gives 81.7%, indistinguishable from fresh counts (+0.1 [-2.7, +2.8]); S = 16 gives 83.0% (+1.6 [-1.0, +4.3]
      against fresh counts; -1.4 [-4.2, +1.4] against exact statistics);
    - **at exactly equal reads (8 per configuration and step) the rule beats per-element Adam by 13 points:**
      81.7% against 68.4% (+13.3 [+9.9, +16.8]). Per-element Adam with 8 draws keeps 14 effective states (16 used,
      112 of 128 lost), the rule 84 (98 used, 30 lost);
    - **the main comparison replicates at a second training seed** (new k-means prototypes and minibatch stream):
      S = 8 with the rule and same-draw counts against S = 16 with per-element Adam, +7.1 [+3.9, +10.1] (seed 1:
      +6.1). Between seeds the same arm moves by 1.1 and 0.2 points;
      (Correction from ChatGPT: This replication compares S = 8 visit-count training with S = 16 Adam, giving
      about 6-7 points. The separate 13.3-point comparison at equal eight reads has one training seed here;
      the companion theory note incorrectly attributes second-seed replication to that equal-read result.)
    - by eye it removes most of the broken strokes of few-sample training (gallery `14x14_0-1_local_training_optimisers_Potts128/`).
    - **at full size (28x28, Potts 128 + both, 0/1) S = 16 with the rule matches exact training:** 87.1% against
      86.3% (+0.9 [-1.4, +3.2]) and against S = 64 with per-element Adam (87.6%, -0.5 [-3.1, +2.1]); the paper's
      model 31.9% (evaluation 4202042; S = 16 with per-element Adam still being evaluated, 4202044);
    - with per-element Adam, S = 16 at 28x28 gives 85.3% (-1.0 [-3.9, +1.8] against exact; the rule +1.9 [-0.9,
      +4.7] over it; evaluation 4202044), so at 28x28 the optimiser matters less than at 14x14;
    - S = 4 with same-draw counts (4 reads) still gives 81.5% (= S = 8, -0.2 [-2.6, +2.2]; +5.9 [+2.7, +9.1] over
      per-element Adam with 16 reads; evaluation 4203238);
    - **four digits (0/1/2/3, Potts 512 + both, S = 64 with the rule) fall short: 74.7% against 85.0% for exact
      training (-10.3 [-13.9, -6.7]; evaluation 4202165).** Local training does not yet scale to more classes.
  - **Simpler visible model (power 4, 14x14 0/1, Potts 128 + both, exact training; evaluation 4203473):**
    - radius 7 / 3 / 2 / 1 (8 neighbours): 84.7 / 83.9 / 83.8 / 81.4% (radius 1 against 7: -3.3 [-6.1, -0.6]);
    - at radius 1, springs only = all three pair terms (81.4 vs 81.4%), and the optomechanical term adds nothing
      (-0.3 [-2.6, +2.0]; with 4 neighbours +0.1 [-1.8, +2.1]);
    - 4 nearest neighbours instead of 8: 74.2%, -7.2 ([-10.1, -4.4]); no pair couplings: 60.0% (-21.4);
    - without hidden units the nearest-neighbour models produce essentially no clean samples (K = 0, radius 1:
      3.5%; 4 neighbours, springs only: 0.0%); with Potts 128 + both, 74.2%.
    - So 8 neighbours with springs only cost about 3 points; the optomechanical term is not needed; 4 neighbours
      cost 7. With more states the 4-neighbour gap closes (Result 15).
  - **Simplest pixel model (power 4, nearest-neighbour springs only), number of Potts states and of Ising units
    (Result 15):** Potts 512 reaches 89.4% (8 neighbours) and 87.8% (4 neighbours), the 4-neighbour gap closing
    with more states; K Ising units match K Potts states at K = 32 and 128 (66.7 vs 65.5, 78.7 vs 81.4%), with
    more 0/1 blends and faint samples; 196 Ising units 81.1%.
  - **At full size (28x28) S = 64 hidden samples with the jackknife train as well as exact statistics:** clean yield
    87.6% vs 86.3% ([-1.3, +4.0]), against 31.9% for the paper's model. S = 256 gives the same (86.9%, +0.6
    [-1.5, +2.6] against exact).
  - The force-matching (score-matching) gradient of the hidden couplings is a local three-factor rule. Every model
    above was trained with it, using exact hidden statistics.
  - On a device the hidden statistics are measured from S hidden samples per clamped configuration. That needs a
    bias correction (jackknife) and enough samples. At 14x14, clean yield is 75.6% / 81.1% / 85.6% for S = 16 / 64 /
    256 with the jackknife. Exact statistics give 84.3% (S = 256 vs exact: +1.3, [-1.5, +4.0]); K = 0 gives 38.6%.
  - With per-element Adam, what a smaller S costs is hidden states (with the visit-count rule it is not; below). A
    rarely occupied state is not visited by the samples, gets no
    gradient, and is lost during or just after the consolidation (slices 10-40, t ≈ 4.5-3.0). Exact training
    brings such states back.
    - A jackknifed run then samples like an exactly trained model with as many states as it kept (S = 64: 74 of 128
      states used, 62 effective; 81.1% against 80.7% for 64 exact states).
    - At S = 64 the missing visits (and noise), not the bias, cost: averaging 4 independent 64-draw estimates
      gives the clean yield of S = 256 (85.7% vs 85.6%; +4.6 [+1.9, +7.3] over S = 64 with one estimate) and keeps
      most of the states (75 effective at t = 0, against 81 for S = 256 and 62 for S = 64).
    - At S = 16 visits and noise are only half of the cost: 16 replicates of S = 16 recover about half (80.9% vs
      75.6%, +5.3 [+2.1, +8.6]) but stay 4.7 points below S = 256 ([-7.5, -2.1]). With per-element Adam the
      jackknife's O(1/S²) bias is a cost of the same size: removing it with a second-order jackknife, at the same
      16 reads, also gives +5.1 ([+2.3, +8.0]; 80.7%, against 80.9% for the 16 replicates, -0.2 [-3.2, +2.9]).
      Whether the two remedies add up is untested.
    - Most of the benefit of many draws comes early: S = 256 up to slice 29 (t ≥ 3.5, which includes the
      consolidation) and 16 afterwards matches S = 256 throughout in clean yield (84.9% vs 85.6%, -0.8 [-3.1,
      +1.6]; exact 84.3%) with about a third of the draws. It still keeps fewer states (71 vs 81 effective at
      t = 0), and its samples lie farther from the training set (median nearest-image distance 1.349 vs 1.295;
      exact 1.260).
    - A naive local fix, an unvisited state raising its own bias, makes training diverge.
    - **Gradient steps with one adaptive step size per parameter group lose the states even with exact
      statistics** (`--hidden-norm group`: Adam's momentum divided by one root-mean-square per group, the closest
      tested rule to the plain steps of simple hardware; "plain steps" below. One run per arm: 22 effective states,
      clean yield 70.4% vs 84.3%, -13.9 points [-17.7, -10.3], like a 16-32-state model). So the optimiser, not only
      the statistics, decides which states survive. Under plain steps S = 64 equals exact statistics (71.7%, 22 states).
      (Per-element Adam is itself local, two leaky averages per coupling; the question is hardware cost.)
    - Why: a state's gradient carries its occupation. Per-element Adam divides it out; plain steps do not. A
      per-state step size is enough (one run per arm, paired). With exact statistics, both per-state rules match
      per-element Adam (84.3%):
      - one Adam gain per hidden state (`--hidden-norm state`): 83.3%, -1.0 [-2.4, +0.3];
      - a step divided by the state's own visit count (`--hidden-norm occupancy`, the visit-count rule): 85.6%, +1.3
        [-1.1, +3.7];
      - against plain steps, +12.9 and +15.2 points.
    - **The visit-count rule removes most of the cost of few hidden samples** (S = draws per clamped configuration
      for the gradient statistics; the rule takes S more fresh draws for its visit counts, so it reads 2S hidden
      states per configuration and step, against S for per-element Adam):
      - S = 16: clean yield 81.4% against 75.6% with per-element Adam (+5.8 points, [+2.5, +9.0]); far (50/50)
        16.8% vs 22.4%; 78 effective states at t = 0 against 30. Still 3.0 points below exact statistics ([-5.6,
        -0.3]).
      - **At equal reads (16 per configuration): S = 8 with the visit-count rule against S = 16 with per-element
        Adam, 81.6% vs 75.6%, +6.0 ([+2.9, +9.2]).** S = 8 does as well as S = 16 with the rule (+0.3, [-2.1, +2.7]).
      - **S = 4 (8 reads) does as well: 82.2%, +6.6 ([+3.4, +9.8]) over S = 16 with per-element Adam at half its
        reads.** S = 2 (4 reads) does not: 73.7%, 8.5 points below S = 4 ([-11.8, -5.2]), the level of S = 16 with
        per-element Adam (-1.9, [-5.3, +1.4]); its 0s fail (clean yield of the 0s 52% vs 68%), and only 29 of its
        102 used states win 0s.
      - With this rule fewer draws do not cost states: S = 4 keeps 95 effective states, more than S = 16 (78) or 64
        (86), yet its clean yield is no higher and its samples lie farther from the training set (median
        nearest-image distance 1.424, against 1.359 at S = 16; one run each).
      - **With 32 reads it does what 256 reads do for per-element Adam:** S = 16 with the rule against 16 replicates
        of S = 16, 81.4% vs 80.9% (+0.5, [-2.0, +3.0]). So it removes the visit part of the cost.
      - **A gap to exact statistics remains at S = 4-16, and it does not depend on S.** Against the rule's own
        exact-statistics run (85.6%): -3.5 at S = 4 ([-6.4, -0.5]), -4.0 at S = 8 ([-6.9, -1.1]), -4.3 at S = 16
        ([-6.9, -1.6]) (paired evaluation 4203290); against exact statistics with per-element Adam (84.3%) it is
        2.2-3.0 points (significant only at S = 16). S = 4 against S = 16, both with the rule: +0.8 ([-2.1, +3.7]).
        The gap does not grow with the 16x larger O(1/S²) bias and 4x the noise variance of S = 4, so it is
        probably neither mainly bias nor mainly noise; what it is remains open. The second-order jackknife does not
        decide it: with the rule it adds +1.5 at S = 16 ([-1.1, +4.1], p = 0.25; about half of the 3.0-point gap to
        exact statistics with per-element Adam as a point estimate; 82.9%, -1.5 [-4.0, +1.1] against that run, -2.8
        [-5.3, -0.2] against S = 256), against +5.1 with per-element Adam (above).
      - S = 64: 83.4% against 81.1% (+2.3, [-0.4, +5.1], not significant); 86 effective states against 62. Against 4
        replicates of S = 64 (85.7%): -2.3 ([-5.0, +0.5], not significant).
      - **Three digits (0/1/2), S = 64: 80.1% against 73.1% with per-element Adam (+7.1, [+4.1, +10.1]),
        indistinguishable from exact training (81.1%, -1.0 [-4.2, +2.2]; a cost up to about 4 points is not excluded)
        and from S = 256 with per-element Adam (80.2%), with half its reads (128 against 256)** (Result 12).
      - One gain per state does not keep the states with measured statistics (S = 64: 65 effective states, against
        62 with per-element Adam and 86 with the visit-count rule), as expected if it moves a rarely visited state
        as slowly as per-element Adam does (step sizes not measured). In clean yield (82.8%) it is indistinguishable
        from both (-0.6 [-3.4, +2.2] against the visit-count rule).
      - Some states are still lost: at S = 8-64 mostly between slices 20 and 40, during and just after the
        consolidation (at t = 0, 34 of 128 at S = 8, 41 at S = 16 and 31 at S = 64, against 2 with exact
        statistics). S = 4 and 2 lose fewer, and later: 17 and 26 at t = 0 (one run each), of which only 7 and 8
        are below 1e-8 at slice 40 (job 4202874). A nominally 10
        times (in effect about 5 times) longer visit window loses more states, not fewer (50 vs 41; one run). That
        does not support "too few visits per window", though the longer window also lags a falling occupation, so
        it does not rule it out; the mechanism is open.
      - By eye (gallery `14x14_0-1_local_training_optimisers_Potts128/`, same start and noise): plain steps and S = 16 with
        per-element Adam break strokes and put a detached blob on top of slanted 1s; the visit-count rule with S = 16
        mostly does not (Result 13).
  - From a random start, force matching finds class-pure prototype states by itself.
  - At 14x14 and 28x28, couplings frozen at a data-statistics reference work as well as trained ones on far and
    clean yield, with more variety (28x28: recall of real 0s 0.59 vs 0.38). What training adds is the learned spring
    (+6 points of clean yield at 28x28).
  - CD-1 (64 releases per sample) also trains the hidden units as well as the exact gradient, but only with exact
    hidden averages: clean yield 85.3% (+0.9, [-2.1, +3.9]).
    - With sampled hidden states its gradient drowns in noise: 10-6,000 |g|, cosine with the true gradient ≤ 0.07.
    - CD-1 amplifies the difference of two hidden readouts by 2/(T² dt). Matching exact-average CD-1 would take
      about 10⁴-10⁸ hidden reads per configuration, against 256 for force matching.
    - **So force matching is the on-device rule.**
- **Other digits** (Results 11 and 12): with hidden units the clean yield is 81-88% on 1/4, 4/9 and 0/1/2 (real
  85-87%), against 34-60% without them.
  - On 0/1/2 the 2s are the weak class. 256 states lift their clean yield from 56% to 67% (real 73%) and the recall
    of their variety from 0.63 to 0.72 (real 0.88).
  - Local training with S = 64 and the jackknife costs 8 points on 0/1/2 (73.1% vs 81.1%), against 3 on 0/1; S = 256
    closes the gap on 0/1/2 as on 0/1 (80.2% vs 81.1%, -1.0 [-3.9, +1.9]). So does S = 64 with the visit-count
    rule (80.1%, -1.0 [-4.2, +2.2]).
- **Four digits, 0/1/2/3** (Result 14): 512 states reach 85.0% clean yield against 86.2% for real images (not
  tested), and 24.2% without hidden units. 256 states give 78.2% (512 vs 256: +6.8 points, [+3.4, +10.4]), against
  83.4% on 0/1/2 and 90.1% on 0/1 at 256 states: the states needed grow with the number of classes (one run per
  problem; not tested across problems). The 2s and 3s are the weak classes; by eye 7-11 of 100 samples are bad,
  nearly all 2s or 3s.
- **Running (2026-11-09, 17:22):**
  - 28x28, 0/1/2/3, power 4, springs only, 8 neighbours, exact training: Ising 784 / 1568 / 3136 (4206653-55) and
    Potts 784 / 1536 (4206669-70); evaluations 4206656-58, 4206671-72; galleries 4206673-77 (folder
    `28x28_0-1-2-3_four_digits_power4_springs_only_Ising_vs_Potts`). Expected: 784 early on 11-10, 1536/1568
    around 09:00, 3136 in the evening of 11-10.
  - simplest pixel model (power 4, springs only, 8 nearest neighbours, exact training), Result 15 continued:
    - 0/1 with 256 / 512 / 1024 Ising units (4206339-41; evaluations 4206346, and 4206348 with 1024). More than 196
      Ising units (the number of pixels) are now allowed with the split-tree start; the tree's smallest child size
      drops below 20 images only when the tree would not fit otherwise (`latent_model.tree_n_min`: 20 for every
      earlier run, 11 at 512 and 5 at 1024 units on 0/1). The large trees were not tested before submission.
    - 0/1/2/3 with 196 / 256 / 512 Ising units, and Potts 512 with the same pixel model as the comparison
      (4206342-45; evaluation 4206349, also against the full-pixel-model Potts 512 of Result 14);
    - gallery 4206350 (new folder `14x14_0-1-2-3_four_digits_power4_springs_only_Ising_vs_Potts`).
  - Finished since the morning (reported above and in Results 10, 14 and 15): 28x28 S = 16 with and without the
    visit-count rule, S = 4 with same-draw counts, four digits with the visit-count rule, the pair-coupling range
    and pair-term steps (`--pair-terms`, `--neighbourhood von_neumann`; test 19 passed) and the Potts and Ising
    sweeps with the simplest pixel model.
- **Idealised, not yet tested** (Caveats):
  - the hidden units are averaged out exactly (infinitely fast; Potts states are exactly one-hot);
  - the nudge measurements the local rule needs are not simulated.

## Files

Folders: `jobs/` holds the Slurm job scripts (`run_*_neuronic.sh`, `submit_sample_eval.sh`); submit them from this folder, e.g. `sbatch jobs/run_cpu_neuronic.sh fid_score.py ...` (relative paths in the scripts resolve against the submit folder). Slurm logs (`latent_<kind>_<job>.out / .err`) go to `logs/`. Both since 2026-11-10. Run outputs, samples, evaluations and galleries are under `../out/latent_physical_SGM/`.

| file | what |
|---|---|
| [DESIGN.md](DESIGN.md) | derivation: free energy and what it adds, score-matching gradient and local learning rule (including a CD-1 form), three ways to run the reverse process, conditions on physical hidden units, prototype plan |
| `check_hidden_unit_formulas.py` | compares the formulas of DESIGN §1–§3 with JAX autodiff (Ising and double-well hidden units, up to full size) |
| `blend_check.py` | classifies existing samples as clean 0 / clean 1 / blend / malformed / diverged |
| `jobs/run_latent_checks_neuronic.sh` | CPU Slurm job running both scripts (~1 min of compute); submit from this folder |
| `../out/diagnostics/{latent_checks,blend_check}/4190135/` | reports, metrics and figures of job 4190135 (Results 1 and 2) |
| `latent_model.py` | problems (mnist28, mnist14, gauss2_8), the paper's schedules, dense visible energy, hidden free energies (Ising, Potts, spring `Q`, per-pixel units) with analytic gradients and Laplacians, the score-matching loss, reference initialisations (Fisher, k-means prototypes, per-pixel) |
| `train_latent.py` | the paper's training loop (jitted scan), with hidden units: `--K --hidden ising/potts --quad --local --visible sextic/quartic --radius R --neighbourhood chebyshev/von_neumann --pair-terms all/none/lin,om,duff --select best/ema --seed --proto-init perclass/pooled/random --hidden-fixed --problem`; local training: `--hidden-samples S --hidden-jackknife` (force matching with measured hidden statistics), `--hidden-jackknife-order 2` (second-order jackknife: Richardson over S, S/2 and S/4 of the same draws, also removes the O(1/S²) bias), `--hidden-samples-late --hidden-samples-switch` (another S from a given slice on), `--hidden-floor` (homeostatic revival of unvisited Potts states; diverges), `--hidden-norm group` (hidden parameters with one step size per group instead of per-element Adam: the closest tested rule to plain gradient steps), `--hidden-norm state` (one Adam gain per hidden state), `--hidden-norm occupancy --occ-beta` (a per-state step from the state's own occupation, one step scale per group), `--occ-same-draws` (its visit counts from the draws of the gradient statistics instead of fresh draws), `--hidden-replicates R` (diagnostic: R independent S-draw estimates averaged), `--hidden-rule cd1 --cd1-dt --cd1-releases --cd1-hidden-samples` (CD-1, with exact or measured hidden statistics); `--steps-scale` (multiple of the paper's steps per slice); `--val-every N` (learning curves: validation loss of the slice result every N steps on the slice's fixed validation batch, per-sample losses in `<run>/val_curves/slice_<k>.npz`, second-half gain in `train_log.jsonl`) |
| `sample_latent.py` | reverse SDE (SRA1, the paper's settings) from a run or the paper checkpoint; trajectory sets `base` / `ext`; `--snapshots`; `--sampler ode` (PF-ODE); `--quench 0.05,0.25,1` (zero-temperature relaxation under the learned t = 0 energy after sampling; quenches the saved samples if they exist, writes `<stem>_q<s>.npy`) |
| `evaluate_samples.py` | blend check, far / copy checks, class-resolved and 50/50 rates, texture, per-class pixel statistics, McNemar and paired-bootstrap tests, example grids |
| `coverage_samples.py` | k-NN precision / recall, spread, distinct nearest images and centroid distance per class |
| `inspect_states.py` | what the Potts states of a run represent, per slice: mean data image, class mix, W_m and Q_m of the most occupied states; effective states, class purity, commitment, states per majority class, and the states' mean occupations by decade (lost states) |
| `fid_score.py` | FID and KID in the features of a small MNIST CNN (trained once per resolution with `--train-only --grid 14/28`, cached in `out/latent_physical_SGM/fid/`), against balanced MNIST test images; raw and clipped; real-image floor at the same n; paired bootstrap (`--pair A,B`); CNN class shares (CPU job; Result 15) |
| `prototype_test.py` | do Potts samples just recall their state's prototype? Distance to the own prototype, within-state spread, nearest training image, posterior sharpness, against held-out real images (CPU job; Result 15) |
| `make_gallery.py`, `jobs/run_gallery_neuronic.sh` | image gallery (Result 13): real images, unsorted samples per model, the same trajectories side by side, nearest training images; into `out/latent_physical_SGM/gallery/` (CPU job; resubmit as runs finish) |
| `make_digit_data.py`, `evaluate_multiclass.py` | other MNIST digit sets (the paper's preprocessing, verified identical for 0/1, 0/1/2, 0-3; problems `mnist14_d<digits>`, `mnist28_d<digits>`) and their evaluation for C ≥ 2 classes (identical to `evaluate_samples.py` for C = 2) |
| `cd1_check.py` | on the slices of a trained run: the local force-matching rule against autodiff (`--local-rule`), measured-statistics gradients (`--samples`, `SxR` for R replicates, suffix `o2` for the second-order jackknife) and CD-1 short-release gradients (`--cd1-samples`: with measured hidden statistics) against the exact gradient and its minibatch noise |
| `test_latent.py` | 53 checks, all passing in job 4195804: 47 against the paper's code, the paper-checkpoint samples of job 4187940 and autodiff, plus 6 sanity checks |
| `summarize_training.py`, `analyze_snapshots.py` | per-sample validation-loss comparisons; hidden-unit commitment along trajectories |
| `jobs/run_train_latent_neuronic.sh`, `jobs/run_sample_latent_neuronic.sh`, `jobs/run_pipeline_neuronic.sh`, `jobs/run_small_pipeline_neuronic.sh` | GPU Slurm jobs: training, sampling, a generic train-sample-evaluate pipeline, and the K = 0 / 1, best / EMA pipeline for mnist14 or gauss2_8; submit from this folder |
| `HARDWARE_NOTES.md` | how buildable hidden units coupled to all pixels are (electronics, photonics, mechanics), with checked references |
| `jobs/run_test_latent_neuronic.sh`, `jobs/run_test{2,...,9,9b,10,...,19}_latent_neuronic.sh` | GPU Slurm jobs: the validation rounds (`test_latent.py` plus smoke trainings and regressions of the exact path; e.g. round 6, job 4197121: `--hidden-fixed`; round 10, job 4197238: the jackknife; round 11: CD-1 with measured hidden statistics; round 12, job 4199405: replicates and the S switch; round 13, job 4199443: the homeostatic floor; round 14, job 4199773: `--hidden-norm group`; round 15, job 4200507: the first `--hidden-norm occupancy`; round 16, job 4200653: `--hidden-norm state` and the redesigned occupancy; round 17, job 4202647: `--hidden-jackknife-order 2`; round 18, job 4202937: `--occ-same-draws`) |
| `jobs/run_evaluate_neuronic.sh`, `jobs/run_summary_neuronic.sh`, `jobs/run_cpu_neuronic.sh` | CPU Slurm jobs: `evaluate_samples.py`, `summarize_training.py`, and any other script of this folder (e.g. `coverage_samples.py`) |
| `jobs/run_gpu_neuronic.sh` | GPU Slurm job running any script of this folder (e.g. `cd1_check.py`) |
| `jobs/submit_sample_eval.sh` | login-node helper that only calls sbatch: samples 500 `base` trajectories from finished full-size runs and evaluates them with the paper baseline |

Full outputs of job 4190135, including the figures `blend_examples.png` and `blend_scatter.png`, are in `physical-diffusion-models/out/diagnostics/{latent_checks,blend_check}/4190135/`. Runs, samples and evaluations of the prototype are in `physical-diffusion-models/out/latent_physical_SGM/{<run>,eval/<job>,coverage/<job>,summary/<job>}/`. Git ignores both.

## Result 1: the design formulas hold (job 4190135)

37 tests plus 4 diagnostic entries, 0 failures:
- **Free energy.** ∇F and ΔF (DESIGN §1) match autodiff to round-off (~1e-16 relative). Ising units were tested at T = 0.02 and T = 1, and at full size (784 pixels, 16 hidden units, no pair terms); continuous double-well units at T = 0.02 with 8 pixels.
- **Gradients.** The score-matching gradients with respect to W, b and the visible parameters (DESIGN §2) also match to round-off.
- **Reverse process.** Doubling the joint energy with the hidden units at 2T reproduces the required free energy 2F − ½‖x‖² exactly, for both hidden types. Doubling only W and b with the hidden units at T, the θ_rev arithmetic, gives the wrong force.
- **CD-1.** The relation between CD-1 and score matching holds for the free energy, with error ∝ dt.
  - At T = 1 the error falls from 26% to 1.2% (W; b: 35% to 1.7%) as dt goes from 0.1 to 0.003.
  - At T = 0.25 it falls from 4% to 0.1% (W) and from 12.5% to 0.4% (b).
  - The T = 0.25 run confirms the T² in the hardware rule's scale T²dt/2.

## Result 2: the samplers' failures are mostly 0/1 superpositions (job 4190135)

The first run, job 4190039, gave identical numbers.

**Method** (details in `blend_check.py`):
- Training images are labelled exactly, by matching them to the ones-only MNIST file: 5,923 + 5,923.
- **Blend:** adding a template of the other class explains the sample much better than any single template. The threshold is set so that 5% of held-out real images count as blends.
- **Between classes:** the sample's Fisher class score lies between the 99% ranges of real 0s and real 1s. The discriminant is 99.9% accurate on held-out images.
- **Clean or malformed:** the nearest training image is within the "far" threshold 4.84 of the `ode_vs_sde` study, or beyond it.
- **Sanity check:** the % far values reproduce that study exactly (35.6 / 47.6 / 34.6 / 22.5).

| sample set | n | % blend | % between classes | % clean 0 | % clean 1 | % malformed 0 / 1 | % diverged | 1s among non-blends |
|---|---|---|---|---|---|---|---|---|
| held-out real images | 1000 | 5.0 (by construction) | 0.9 | 36.2 | 47.1 | 11.3 / 0.4 | 0 | 50% |
| SDE (500, one Brownian tree) | 500 | 43.2 ± 2.2 | 12.6 | 2.4 | 35.6 | 12.0 / 0.6 | 6.2 | 72% |
| ODE (500) | 500 | 51.0 ± 2.2 | 21.8 | 1.8 | 29.2 | 14.8 / 1.6 | 1.6 | 65% |
| SDE, the paper's 40 | 40 | 35.0 ± 7.5 | 10.0 | 5.0 | 50.0 | 10.0 / 0.0 | 0 | 77% |

"1s among non-blends" counts Fisher-score 1s among the non-diverged, non-blend samples. Results 4 onwards use
evaluate_samples.py's "share of 1s", which counts them among all non-diverged samples (68% for the same SDE samples).

- **Superposition is the main failure.** 43% of SDE samples and 51% of ODE samples are flagged as 0/1 blends, against 5% of real images. Among the far, non-diverged samples, 57% (SDE) and 64% (ODE) are blends. The example grid confirms it: flagged samples are a ring with a stroke through it, or a stroke with a partial ring.
- **The 0s are what fails.** Clean 0s are almost absent: 2.4% (SDE) and 1.8% (ODE), against 36% of real images.
  - Generated 0s nearly always carry a 1 stroke inside, so they end up as blends or malformed 0s.
  - The 1s fare far better: 35.6% (SDE) and 29.2% (ODE) clean 1s, against 47%.
  - Among samples that are neither blends nor diverged, 72% (SDE) and 65% (ODE) are 1s, against 50% in the data.
- **The flag undercounts.** Many samples labelled clean 1 or malformed 0, especially from the ODE, carry a faint ghost of the other digit that the 5%-false-positive threshold does not catch.
- **ODE versus SDE.** The ODE's extra failures are superpositions: +8 points flagged as blends and +9 points between the classes. The SDE instead diverges more (6.2% vs 1.6%).
- **"% far" depends on the class.** 23–26% of *real* 0s are "far" from their nearest training image, against 1–4% of real 1s (bounds from the aggregate counts). Part of every sampler's "far" count is therefore just 0s being 0s.
- **What this means for the idea.**
  - The present model fails at exactly the global 0-vs-1 decision that hidden units supply (DESIGN §1.1). The prototype therefore targets the main failure.
  - Whether bilinear hidden units suffice is open. They add a global class variable, but the copies they mix share their shape (DESIGN §1.1), so the different pixel statistics of 0s and 1s must still come from the visible energy's anharmonic terms.
  - The prototype's metrics therefore include per-class pixel statistics. If those stay wrong, the next step is hidden units that gate pair couplings.

## Result 3 (2026-11-07): one hidden unit removes the blends of a two-pattern toy

**Setup.** Prototype code in `latent_model.py` / `train_latent.py` / `sample_latent.py` / `evaluate_samples.py`; checked
against the paper's code by `test_latent.py` (30 checks in job 4195132, 38 in 4195196; 53 after later additions). All
are within tolerance: analytic comparisons to round-off, finite-difference gradients to 2e-8 or better, 150 training
steps match the paper's own loop to 1e-15, and the sampler reproduces job 4187940 to 2-3e-5. Toy problem `gauss2_8`: an 8x8 lattice with all-to-all pair couplings; class 0 is a
square ring, class 1 a vertical bar crossing it (8 of its 12 pixels overlap the ring), each with N(0, 0.15^2) pixel noise,
5,000 per class. Its forward marginals are two-Gaussian mixtures, so the exact score has precisely the one-hidden-unit form.
Trained with the paper's full schedule (100 slices, 2.09M Adam steps), K = 0 and K = 1, with the paper's slice result
(`best`) and with an average of the iterates (`ema`); 500 reverse-SDE samples each, same initial states and noise.

| (job 4195211) | real samples | K = 0 | K = 1 |
|---|---|---|---|
| flagged blends | 5.0% (by construction) | 59.0% | 3.6% |
| far from every training sample | 10.0% (by construction) | 72.2% | 11.2% |
| off-manifold (exact likelihood below the data's 0.5% quantile) | 0.5% | 57.8% | 0.8% |
| share of class 1 | 53.7% | 53.8% | 51.6% |

- **The pair-coupled model fails at a superposition; one hidden unit fixes it.** Paired trajectories: 282 are blends only
  without the hidden unit, 5 only with it (McNemar p = 1e-76). The class means of the K = 0 samples show the bar inside the
  ring and ghost rings beside the bar; the K = 1 class means and per-class pixel statistics match the data to within
  about 1.2x the finite-sample noise floor.
- **The trained coupling converges to the theoretical optimum.** From t ~ 3 to t ~ 0.5, cos(W, W*) >= 0.997 and
  |W| / |W*| is within 3%; nearer the data |W| grows above W* (by 15% at t = 0, cos 0.989). W* is the Gaussian-mixture
  solution of DESIGN 1.1.
- **Loss is a poor proxy.** Averaging the iterates (`ema`) lowers the validation loss about as much as the hidden unit
  does, but barely changes the samples (57.8% vs 59.0% blends, paired p = 0.03; far unchanged). The hidden unit's gain
  lies along the class decision.

## Result 4 (2026-11-07): 28x28 MNIST, one Ising hidden unit

Same training procedure as the paper (`best` slice result), 500 reverse-SDE samples per model from the same initial states
and Brownian paths as job 4187940 (jobs 4195145-49, sampling 4195747-51, evaluation 4195752, snapshots 4195753).
`cls0` / `cls1` are idea 1: one model trained on 0s only and one on 1s only (no hidden unit).

| | real images | paper | K = 0 (this code) | K = 1 | K = 1, 4x hidden lr | cls0 + cls1 |
|---|---|---|---|---|---|---|
| flagged blends | 5.0% | 43.2% | 43.2% | 33.8% | 33.4% | 9.5% |
| between the classes | 0.9% | 12.6% | 12.6% | 4.0% | 3.6% | 0.6% |
| share of 1s | 50% | 68% | 68% | 57% | 57% | 50% |
| far from every training image | 13.2% | 35.6% | 35.6% | 45.0% | 44.6% | 37.2% |
| diverged | 0% | 6.2% | 6.2% | 7.4% | 7.2% | 2.3% |
| non-blend 0s that are far | 24% | 83% | 83% | 87% | 88% | 70% |
| non-blend 1s that are far | 0.8% | 1.7% | 1.7% | 0.5% | 0.5% | 0% |

- **The re-implementation reproduces the paper exactly.** K = 0 and the paper checkpoint put every one of the 500 paired
  samples in the same category; the per-slice validation losses agree to about 1e-5 in absolute terms (at most 1.2e-7
  relative).
- **The hidden unit makes the class decision, as designed.** Most trajectories commit around tau ~ 4 (the speciation
  time): |m| > 0.9 for 69% of trajectories at tau = 4, 93% at 3.5, 99% by 3. 94% never flip afterwards, and the final
  sign agrees with the final image's class for 98.7% of samples. Blends drop by 9 points (paired p = 2e-5), samples
  between the classes by two thirds, and the class balance moves from 68% to 57% 1s. Ghost rings around the 1s disappear
  (ring intensity 0.076 vs 0.115; data 0.089), and the 1s match those of a 1s-only model (mean-image RMSE 0.030 vs 0.030).
- **"Far" gets worse, because the model now draws more 0s and its 0s are poor.** Even a model trained on 0s alone draws
  70% of its non-blend 0s far from every training image. The paper's lower "far" rate is partly an artefact of drawing 68% 1s,
  which are easy.
- **The remaining blends are not indecision.** Blended samples commit as early and as firmly as clean ones. The visible
  energy draws a stroke into the ring even after the unit has chosen "0": a bilinear unit can only tilt the shared
  visible model, not make the centre pixels of a 0 stiffer (DESIGN 1.1, "tilted copies share their shape").
- **Hidden learning rate:** 4x larger changes nothing (33.4% vs 33.8% blends).

## Result 5 (2026-11-07): 14x14 MNIST, hidden-unit variants

The same images 2x2-pooled to 14x14 (pair couplings within Chebyshev radius 7), the paper's full schedule, the EMA slice
result, 500 reverse-SDE samples each from one shared set of initial states and Brownian paths (so all rows are paired).
"Far" uses the held-out images' p90 nearest-image distance (1.894 here, by construction 10% of real images).
Hidden-unit variants (all `train_latent.py` options):
- `Ising`: one bistable unit coupled to all pixels, initialised along the Fisher direction;
- `Potts M`: one unit with M states (M/2 k-means prototypes per class), F_h = -T log sum_m exp((b_m + W_m.x)/T);
- `+ spring` (`--quad`): the unit also couples to the squared pixels, a = b + W.x + Q.x^2, so it modulates the pixels'
  spring constants and the copies of the visible model it mixes can differ in width, not only in position;
- `+ per-pixel` (`--local`): one extra bistable unit per pixel, coupled only to that pixel (an asymmetric double well).

| (jobs 4195212, 4195544, 4195796, 4195820-23) | real images | K = 0 | Ising | Ising + spring | Potts 8 | Potts 8 + spring | Potts 8 + per-pixel | Potts 8 + both | Potts 16 | Potts 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| flagged blends | 5.0% | 32.6% | 21.4% | 15.2% | 11.0% | 5.6% | 9.2% | 4.2% | 9.4% | 10.0% |
| between the classes | 1.0% | 25.0% | 5.2% | 4.8% | 3.6% | 2.2% | 3.6% | 1.4% | 2.6% | 3.0% |
| clean 0s | 39.3% | 3.2% | 2.8% | 3.4% | 10.8% | 14.8% | 11.4% | 14.4% | 16.2% | 19.8% |
| far | 10.0% | 48.4% | 50.2% | 48.6% | 39.0% | 33.8% | 36.8% | 33.4% | 32.2% | 28.2% |
| diverged | 0% | 3.0% | 3.0% | 1.2% | 2.4% | 0.6% | 0% | 0% | 3.0% | 3.2% |
| share of 1s | 50.2% | 55.3% | 51.3% | 50.0% | 51.8% | 51.7% | 50.8% | 51.8% | 52.4% | 52.1% |

- **Each ingredient mainly improves a different failure (the effects overlap).** The Ising unit fixes the class
  decision (between-class samples 25% -> 5%). Prototypes (Potts states) improve the shapes: clean 0s and "far" improve
  steadily with the number of states, though Potts 32 alone stays well short of the data (19.8% clean 0s, 28.2% far).
  The spring-constant coupling halves the remaining blends with Potts units (11.0 -> 5.6) and cuts them by about 30%
  with one Ising unit (21.4 -> 15.2); it also lowers divergences and "far". Per-pixel units remove the divergences and
  also cut blends somewhat.
- **Potts 8 + both is essentially blend-free:** 4.2% flagged blends and 1.4% between the classes, against 5.0% and 1.0% for
  real images, and no divergences.
- **No memorisation:** no sample of any variant is closer to a training image than the 5% closest held-out images are
  (copy rate 0% for all variants; median nearest-image distance 1.42-1.87 vs 1.17 for held-out images; job 4196072).
- **Every blend reduction against K = 0 is paired and large** (McNemar p from 4.5e-8 for Ising to 8.9e-36 for Potts 8 +
  both); Potts 8 + both also beats plain Potts 8 (blends p = 3e-7, far p = 1e-4). Far and divergence differences are
  not uniformly significant (Ising vs K = 0: far p = 0.36; divergences of Ising / Potts 8 / 16 / 32: p = 0.45-1).
- The texture stays smoother than real images (roughness 0.32-0.38 vs 0.43); in the example grids most remaining
  "far" samples look like recognisable but soft digits.
- **Per-pixel units on their own (job 4195824) are a stabiliser, not a fix:** without a class unit, blends 30.4% (vs
  32.6%), far 52.2% (vs 48.4%); with one Ising unit, blends 17.6% (vs 21.4%), far 48.8% (vs 50.2%); in both cases 0%
  diverged (vs 3%). They raise the roughness by about 0.03 (towards the data) and the grey fraction slightly (0.108 ->
  0.117, away from the data's 0.103).

## Result 6 (2026-11-07): power-4 visible energy (`--visible quartic`)

k_6 = 0; confinement from k_duff >= 0.01 and c_duff >= 0 (the paper's k_6 >= 0.01 floor costs ~47 kT per bright pixel,
the quartic floor ~8.5 kT). EMA slice result; 28x28 jobs 4195472/73 (evaluation 4195932), 14x14 job 4195471.

| | real images | sextic K = 0 | quartic K = 0 | sextic K = 1 | quartic K = 1 |
|---|---|---|---|---|---|
| 28x28: diverged | 0% | 6.8% | 2.0% | 6.8% | 1.0% |
| 28x28: flagged blends | 5.0% | 44.4% | 38.8% | 32.2% | 32.6% |
| 28x28: between the classes | 0.9% | 12.4% | 18.8% | 4.2% | 21.6% |
| 28x28: far | 13.2% | 36.0% | 47.0% | 43.4% | 48.6% |
| 14x14: flagged blends | 5.0% | 32.6% | 33.6% | 21.4% | 24.2% |
| 14x14: far | 10.0% | 48.4% | 60.2% | 50.2% | 55.0% |

- **Power 4 trains and is more stable:** 3-7x fewer divergences (paired p ~ 1e-5 to 1e-7).
- **Without strong hidden units its samples are somewhat worse:** more ghosting at 28x28 (ghost strokes in 0s 0.17-0.21
  vs 0.11-0.12; real 0.09), more class-ambiguous images at 28x28, and 5-12 points more "far" at both sizes. At 28x28 the
  roughness is closer to real images (0.12-0.13 vs 0.08 for sextic; real 0.16); at 14x14 it is unchanged. The grey
  fraction is further from real at both sizes.
- One Ising unit still reduces blends with power 4 at both sizes (28x28 p = 0.005; 14x14 p = 3.7e-5). At 14x14 it also
  cuts the between-class samples (26.8% -> 7.4%) and far (60.2% -> 55.0%, p = 0.003); at 28x28 the between-class share
  rises (18.8% -> 21.6%).
- With the best hidden units the picture changes: power 4 is then close to power 6 (Result 9).

## Result 7 (2026-11-08): 28x28, Potts units with spring and per-pixel couplings

EMA slice result; 500 reverse-SDE trajectories from the same initial states and Brownian paths as Result 4 (`base`;
evaluations 4196504, 4196630, 4196761, 4196767), confirmed on 500 independent trajectories that were never used to
choose a model (`ext`, seed 2027; evaluation 4196783; base + ext pooled: 4196784). Two new kinds of metric:
- **50/50 rates** (`evaluate_samples.py`): clean and far rates within each Fisher class (non-diverged samples),
  averaged over the two classes. Real 0s are far from every training image 25.5% of the time and real 1s 1%, so the raw
  rates reward a model for drawing the easy 1s. Paired bootstrap CIs (10,000 resamples of trajectories).
- **Coverage** (`coverage_samples.py`, job 4196778): k-NN precision and recall (k = 3) against held-out real images of the
  same class, 150 images per class.

| (base set) | real images | paper | Ising | Potts 8 | Potts 8 + spring | Potts 32 | Potts 32 + both | Potts 128 + both | Potts 256 + both | cls0 + cls1 |
|---|---|---|---|---|---|---|---|---|---|---|
| flagged blends | 5.0% | 43.2% | 32.2% | 19.6% | 11.2% | 16.8% | 6.2% | 4.8% | 3.4% | 9.5% |
| between the classes | 0.9% | 12.6% | 4.2% | 3.0% | 1.2% | 1.6% | 0.6% | 0.4% | 0.6% | 0.6% |
| far | 13.2% | 35.6% | 43.4% | 40.0% | 35.2% | 34.8% | 19.0% | 9.4% | 6.0% | 37.2% |
| diverged | 0% | 6.2% | 6.8% | 8.4% | 1.4% | 8.4% | 0.0% | 0.0% | 0.0% | 2.3% |
| share of 1s | 50% | 68% | 57% | 57% | 53% | 56% | 52% | 52% | 52% | 50% |
| clean yield, 50/50 | 83.3% | 31.9% | 43.5% | 52.6% | 58.2% | 59.5% | 76.4% | 86.3% | 90.3% | 58.6% |
| far, 50/50 | 13.2% | 44.2% | 45.7% | 39.9% | 36.7% | 32.3% | 19.6% | 9.7% | 6.2% | 35.9% |
| far among 0s | 25.5% | 80.0% | 91.0% | 78.7% | 72.2% | 62.6% | 38.0% | 19.4% | 12.4% | 71.8% |
| recall of real 0s / 1s | 0.89 / 0.89 | 0.09 / 0.37 | | | 0.11 / 0.41 | 0.11 / 0.53 | 0.26 / 0.64 | 0.46 / 0.79 | 0.55 / 0.83 | 0.22 / 0.41 |
| roughness | 0.159 | 0.074 | 0.076 | 0.084 | 0.103 | 0.091 | 0.128 | 0.131 | 0.132 | 0.098 |

"Ising" is K1_ema (one unit), "both" = spring (`--quad`) + per-pixel units (`--local`), "real images" = the 1,000
held-out probes (for recall: one half of them against the other). Potts 128 + both: evaluation 4197068, coverage
4197069. Potts 256 + both: evaluation 4197141, coverage 4197142.

- **A second training seed reproduces Potts 32 + both** (`--seed 2`: new k-means prototypes and minibatch stream;
  evaluation 4197122, coverage 4197123).
  - Against seed 1: far (50/50) -2.1 points ([-5.0, +0.8]), clean yield (50/50) +0.5 ([-2.7, +3.6]), blends 7.6% vs
    6.2% (p = 0.3); recall of real 0s / 1s 0.24 / 0.64 vs 0.26 / 0.64.
  - Against the paper: blends 7.6% vs 43.2%, far 16.6% vs 35.6%, clean yield (50/50) +45.0 ([+40.3, +49.6]).
  - Training-seed noise is about 2 points at 28x28, as at 14x14.
- **The labels are not needed at 28x28 either** (`--proto-init pooled`: 32 k-means prototypes of all images;
  evaluation 4197118, coverage 4197119).
  - Against the labelled Potts 32 + both:
    - far 14.8% vs 19.0% (p = 0.005); far (50/50) -4.1 points ([-6.9, -1.3]); clean yield (50/50) +2.5
      ([-0.7, +5.7]);
    - blends 7.2% vs 6.2% (p = 0.46);
    - recall of real 0s / 1s 0.28 / 0.59 vs 0.26 / 0.64.
  - Against the paper: clean yield (50/50) +47.0 points ([+42.4, +51.5]).
- **128 states improve again and pass the real images on far and clean yield, though not on coverage.**
  - Against Potts 32 + both on the same trajectories: far 9.4% vs 19.0% (p = 2e-10); far (50/50) -9.9 points
    ([-12.8, -7.0]); clean yield (50/50) +9.9 ([+6.6, +13.2]); blends 4.8% vs 6.2% (p = 0.3).
  - Against the paper: blends 4.8% vs 43.2% (p = 9e-52); far 9.4% vs 35.6% (p = 3e-27); clean yield (50/50) 86.3% vs
    31.9% (+54.4, [+49.9, +58.6]).
  - Confirmed on `ext` (4197138). Against 32 states: far 8.6% vs 18.6%, clean yield (50/50) +11.6 ([+7.9, +15.5]).
    Against the paper: blends 5.6% vs 45.8%, far 8.6% vs 28.2%, clean yield 84.4% vs 30.4%.
  - Pooled over n = 1000 (4197139): blends 5.2%, far 9.0%, far (50/50) 10.0% (real 13.2%), clean yield (50/50) 85.4%
    (real 83.3%), share of 1s 55.0%.
  - Coverage: recall of real 0s 0.46 (`ext`, 4197140: 0.39); 32 states 0.26 (0.22); paper 0.09. Recall of real 1s 0.79
    (0.80); 32 states 0.64 (0.67).
  - Precision is 0.99 for the 0s and 0.95 for the 1s (real 0.87 / 0.88). The 0s' spread is 8.3 (real 9.8).
  - So the samples are now more typical than real images: fewer far samples, no near-copies (0% on `base`, 0.2% on
    `ext`, real 5%), and still only about half of the real 0s' variety covered.
- **256 states improve again, on fidelity and variety together** (`base` only so far).
  - Against 128 states on the same trajectories:
    - far 6.0% vs 9.4% (p = 0.002); far (50/50) -3.5 points ([-5.6, -1.4]);
    - clean yield (50/50) 90.3% vs 86.3% (+4.1, [+1.5, +6.7]);
    - blends 3.4% vs 4.8% (p = 0.23).
  - Coverage: recall of real 0s 0.55 vs 0.46, of real 1s 0.83 vs 0.79, with precision unchanged (0.99 / 0.95). The
    0s' spread is 8.4 (real 9.8).
  - Against the paper: blends 3.4% vs 43.2% (p = 7e-55); far 6.0% vs 35.6% (p = 3e-34); clean yield (50/50) 90.3% vs
    31.9% (+58.5, [+54.3, +62.5]).
  - The samples move closer to the training images:
    - median nearest-image distance 3.17 vs 3.44 for 128 states at the same share of 1s (held-out images 3.29 at
      50% 1s);
    - 0.6% near-copies (128 states 0%; real 5%).
    - At 14x14 this pull became an overshoot at 512 states (Result 8).
  - 512 states with power 4 (`P512_ema_quad_local_quartic`): done; the best model so far (Result 9: 92.3% clean
    yield, not significantly better than 256 states).

- **Potts 32 + both fixes superposition and divergences, halves "far" and removes most of the excess of 1s.** It draws
  52% 1s on `base` and 55% pooled with `ext` (paper 68% and 71%; real images 50%). Against the paper on the same
  trajectories: blends 6.2% vs 43.2% (McNemar p = 7e-52), far 19.0% vs 35.6% (p = 7e-11), diverged 0% vs 6.2%
  (p = 9e-10); clean yield (50/50) 76.4% vs 31.9% (+44.5 points, 95% CI [+40.0, +49.0]); real images reach 83.3%.
- **Confirmed on the independent `ext` trajectories:** blends 7.6% vs 45.8% (p = 2e-47), far 18.6% vs 28.2% (p = 7e-5),
  diverged 0% vs 5.0%, clean yield (50/50) 72.8% vs 30.4% (+42.4 [+37.7, +47.0]). Pooled base + ext (n = 1000): blends
  6.9% vs 44.5% (p = 2e-97), far 18.8% vs 31.9% (p = 5e-14), share of 1s 54.9% vs 71.2%.
  - On `ext` every model draws about 6 points more 1s than on `base` (Potts 32 + both: 58.2% vs 51.6%).
  - The trajectories are shared, so this is one draw of noise.
  - The pooled 54.9% is the best estimate: about 5 points, or 3 binomial standard errors, above 50%. The paper is 21
    points above.
- **The spring, the per-pixel units and more states each help on their own, as at 14x14.**
  - Spring alone on top of Potts 32 (evaluation 4197124, coverage 4197125):
    - blends 16.8% -> 8.8% (p = 5e-7), far 34.8% -> 22.0% (p = 2e-14), diverged 8.4% -> 0.8% (p = 7e-12);
    - clean yield (50/50) +12.1 points ([+8.4, +15.6]);
    - recall of real 0s 0.11 -> 0.17.
  - Per-pixel units on top of Potts 32 + spring:
    - blends 8.8% -> 6.2% (p = 0.01), far 22.0% -> 19.0% (p = 0.004);
    - clean yield (50/50) +4.9 ([+2.5, +7.3]);
    - recall of real 0s 0.17 -> 0.26.
  - The spring coupling also makes Potts 8 work at 28x28: blends 19.6% -> 11.2% (p = 3e-7), diverged 8.4% -> 1.4%
    (p = 6e-10).
  - More states improve the 0s: Potts 32 vs Potts 8 lowers far (50/50) by 7.7 points ([-11.0, -4.4]).
  - Spring + per-pixel units on top of 32 states: blends 16.8% -> 6.2% (p = 3e-11), far 34.8% -> 19.0% (p = 2e-17),
    diverged 8.4% -> 0% (p = 5e-13). Against Potts 8 + spring: blends p = 0.002, far p = 3e-16.
  - One Ising unit with spring is much weaker (blends 28.0%, far 41.4%, diverged 3.4%).
- **The paper's raw "far" flatters it.** It draws 68% 1s, which are easy. At 50/50 the paper is at 44.2% far, and every
  Potts model is better (Potts 8 + spring 36.7%: -7.6 points, [-11.7, -3.3]). Result 4's "far gets worse with the
  hidden unit" was this class-mix effect.
- **One model with hidden units beats two models trained on separate classes.** cls0 + cls1 (labels used at training
  time) reaches clean yield 58.6%, far 35.9% (50/50) and 9.5% blends; Potts 32 + both 76.4%, 19.6% and 6.2% (unpaired:
  the union reuses the trajectories).
- **The images look like the data's.**
  - Class mean images: RMSE against the data 0.033 for the 0s and 0.013 for the 1s (paper 0.092 and 0.042;
    finite-sample floors 0.016 and 0.010).
  - No ghost strokes in the 0s (0.081; real 0.092).
  - Roughness 0.128 (real 0.159, paper 0.074).
  - In the example grids: rings without strokes, crisp 1s.
- **Fidelity and coverage both improve, but the 0s stay less varied than real 0s.**
  - k-NN precision of the 1s 0.89 (real 0.88, paper 0.50).
  - Recall of real 1s 0.64 (paper 0.37); recall of real 0s 0.26 (paper 0.09, real-vs-real 0.89).
  - The 0s' mean pairwise distance is 8.0 (paper 6.5, real 9.8).
  - No near-copies of training images (0% on base, ext and pooled).
  - The median nearest-image distance is not a clean check here, because it falls when the share of 1s rises: 3.49 on
    base (52% 1s), 3.08 pooled (55%), 2.87 on ext (58%), against 3.29 for held-out images (50%).

## Result 8 (2026-11-08): 14x14, number of Potts states, training seeds, coverage

14x14 as in Result 5; one Potts unit with M states plus spring and per-pixel couplings ("both"), EMA slice result,
500 paired trajectories. Combined evaluation 4196913, coverage 4196914 (training: 4195823, 4196037, 4196425-26,
4196568-69; second training seed: 4196765).

| | real images | K = 0 | M = 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|
| flagged blends | 5.0% | 32.6% | 4.2% | 4.4% | 3.2% | 2.8% | 2.6% | 2.0% | 2.6% |
| far | 10.0% | 48.4% | 33.4% | 30.2% | 21.2% | 16.6% | 13.2% | 8.0% | 5.6% |
| clean 0s | 39.3% | 3.2% | 14.4% | 19.0% | 26.8% | 31.6% | 34.6% | 40.0% | 42.2% |
| clean yield, 50/50 | 85.7% | 38.6% | 63.0% | 66.4% | 75.7% | 80.7% | 84.3% | 90.1% | 92.1% |
| recall of real 0s / 1s | 0.89 / 0.90 | 0.48 / 0.79 | 0.67 / 0.73 | 0.66 / 0.77 | 0.70 / 0.82 | 0.74 / 0.87 | 0.78 / 0.87 | 0.80 / 0.90 | 0.85 / 0.87 |
| copies | 4.9% | 0% | 0% | 0% | 0% | 0% | 0.4% | 0.6% | 1.4% |
| median nearest-image distance | 1.165 | 1.83 | 1.47 | 1.48 | 1.39 | 1.37 | 1.26 | 1.175 | 1.126 |
| roughness | 0.432 | 0.302 | 0.381 | 0.393 | 0.397 | 0.403 | 0.402 | 0.409 | 0.412 |

K = 0 diverges in 3.0% of samples, no Potts + both model does. Share of 1s: 51.4-52.2% (data 50.2%, K = 0 55.3%).

- **More states give better samples, up to the data.** Every doubling from 8 to 256 states lowers "far" (paired p
  between 0.048 and 1e-9) and raises the 50/50 clean yield by 3.4-9.2 points. Blends stay at 2.0-4.4% (real images: 5%
  by construction).
- **The gain is mostly coverage, with mild averaging.**
  - Recall of the real 0s rises with M: 0.48 (K = 0), 0.67 (8 states), 0.80 (256), 0.85 (512); real 0.89. The dip from
    8 to 16 states (0.67 -> 0.66) is smaller than the subsampling spread (about 0.04).
  - The 0s' spread grows with M (3.70 -> 3.96; real 4.20).
  - Among 150 samples, 140-147 distinct nearest training images (real 146-148).
  - Precision of the 0s exceeds the real-vs-real value from 32 states on (0.93-0.95 vs 0.88), so the sampled 0s are
    slightly more typical than real 0s. For the 1s, precision is below the real value at 32 and 64 states (0.78, 0.84),
    equal at 128 (0.88) and above it only from 256 states (0.92-0.93).
  - The templating measure points the same way. From 16 states on, the 0s lie slightly closer to their class's k-means
    centroids than real 0s do (median distance 2.00-2.07 vs 2.155; 8 states 2.165, K = 0 2.51). The 1s do so from 64
    states on (1.00-1.03 vs 1.033).
  - For the 0s this distance is flat from 32 to 512 states (2.00-2.04), while recall rises from 0.70 to 0.85.
- **About 256 states match held-out images; 512 overshoot.**
  - At 256 states, far (8.0%), clean 0s (40.0%) and the median nearest-image distance (1.175 vs 1.165) match held-out
    real images.
  - At 512 states the samples are closer to the training set than held-out images are (far 5.6%, median distance
    1.126). This is the start of a kernel-density-like regime, with one state per ~23 training images.
  - Near-copies stay rarer than for real images (1.4% vs 4.9%). A near-copy is a sample whose nearest training image
    is closer than 0.445, which by construction holds for 5% of real images.
- **Training seed.** A second training run of Potts 128 + both differs from the first by -2.0 far points (50/50,
  [-4.3, +0.4]) and +1.5 clean-yield points ([-1.1, +4.2]), neither significant. It used new k-means prototypes and a new
  minibatch stream, with the same hidden learning-rate scales (`--seed 2`). Single-run differences of about 2 points are
  within training noise; every effect claimed above is larger.
- **Confirmed on 500 independent `ext` trajectories** (seed 2027, never used to choose a model; evaluation 4197062,
  coverage 4197063; K = 0 and 32, 128, 256, 512 states).
  - Potts 128 + both against K = 0: blends 2.0% vs 31.4% (p = 4e-37), far 11.0% vs 43.4% (p = 6e-38), diverged 0% vs
    3.6%, clean yield (50/50) 87.4% vs 41.2% (+46.2 points, [+41.8, +50.5]).
  - 256 states beat 128 again: far 7.2% vs 11.0% (p = 0.003), far (50/50) -3.7 points ([-6.0, -1.4]).
  - 512 and 256 states do not differ: far 6.8% vs 7.2% (p = 0.86), far (50/50) -0.4 ([-2.7, +1.9]).
  - Median nearest-image distance 1.337 / 1.254 / 1.108 at 128 / 256 / 512 states (held-out 1.165). 512 states sit
    closer to the training set than held-out images on both trajectory sets; 256 states sit at (`base`) or above
    (`ext`) the held-out value, with fewer far samples than held-out images on `ext` (7.2% vs 10.0%).
  - Recall of real 0s: 0.50 (K = 0), 0.70 (32 states), 0.82 (128, 256 and 512); real 0.89. On `ext` recall stops
    rising at 128 states, so the rise from 256 to 512 states on `base` (0.80 -> 0.85) is not confirmed.
  - Precision of the 1s: 0.76 / 0.87 / 0.89 / 0.91 at 32 / 128 / 256 / 512 states (real 0.88), as on `base`.
- **More two-body couplings do not help; hidden units do** (evaluation 4197157, coverage 4197158).
  - Setup: the paper's model (K = 0) with all 19,110 pixel pairs coupled (`--radius 13`), against the paper's radius
    7 (11,760 pairs).
  - Parameters per slice: 58,114 instead of 36,064 (+61%).
  - Result: blends 37.4% vs 32.6% (p = 0.001, i.e. more blends), clean yield (50/50) 38.6% vs 38.6%, far (50/50) -3.4
    points ([-5.7, -1.2]).
  - Coverage is no better: recall of real 0s 0.49 vs 0.48.
  - Hidden units add far fewer parameters and fix the problem:
    - Potts 8 + both adds 3,536 parameters (+10%): blends 4.2%, clean yield 63.0%;
    - Potts 32 + both adds 12,968 (+36%): blends 3.2%, clean yield 75.7%.
  - So the gain is the many-body structure the hidden units add (DESIGN §1.1), not the parameter count.
- **The labels are not needed.**
  - With `--proto-init pooled`, the 128 initial prototypes are k-means centroids of all images together. The per-pixel
    units are set from pixel intensities alone, so no label enters training. Evaluation 4197111, coverage 4197112.
  - Against the labelled Potts 128 + both on the same trajectories:
    - far 10.2% vs 13.2% (p = 0.02); far (50/50) -3.1 points ([-5.5, -0.7]);
    - clean yield (50/50) 87.1% vs 84.3% (+2.7, [+0.1, +5.3]);
    - blends 3.2% vs 2.6% (p = 0.58);
    - recall of real 0s 0.79 vs 0.79, of real 1s 0.85 vs 0.88; no near-copies.
  - On `ext` (evaluation 4197113, coverage 4197115) the two are equal: far 10.2% vs 11.0% (p = 0.63), far (50/50) -0.8
    ([-3.2, +1.5]), clean yield (50/50) -0.6 ([-3.3, +2.2]), recall of real 0s 0.81 vs 0.82. Pooled base + ext
    (4197114): far (50/50) -1.9 ([-3.6, -0.2]), blends 3.3% vs 2.3% (p = 0.09).
  - The labelled model's second seed differs from its first by -2.0 far points (50/50, `base`). So the label-free
    model is as good as the labelled one, not better.

## Result 9 (2026-11-08): power 4 with hidden units

The Potts + both design of Results 7 and 8 with `--visible quartic` (k_6 = 0):
- at 14x14, with 32 and 128 states (128 at both training seeds);
- at 28x28, Potts 32, 128 and 256 + both (Result 7).

Evaluations 4196913 (14x14) and 4196782 (28x28, 32 states); 128 and 256 states: below the table.

| power 6 -> power 4 | 14x14, Potts 32 + both | 14x14, Potts 128 + both, seed 1 / seed 2 | 28x28, Potts 32 + both | 28x28, Potts 128 + both | 28x28, Potts 256 + both |
|---|---|---|---|---|---|
| far | 21.2% -> 25.0% | 13.2% -> 13.4% / 11.4% -> 12.2% | 19.0% -> 26.0% | 9.4% -> 12.4% | 6.0% -> 7.8% |
| difference in far, 50/50 (95% CI) | +3.8 [+1.6, +6.2] | +0.3 [-1.6, +2.2] / +0.7 [-0.5, +2.0] | +7.3 [+4.8, +9.9] | +3.0 [+1.1, +5.1] | +1.8 [-0.2, +4.0] |
| flagged blends | 3.2% -> 3.6% | 2.6% -> 2.0% / 2.8% -> 2.2% | 6.2% -> 3.8% | 4.8% -> 3.4% | 3.4% -> 2.4% |
| diverged | 0% -> 0% | 0% -> 0% / 0% -> 0.2% | 0% -> 0% | 0% -> 0% | 0% -> 0% |

28x28 with 128 states: evaluation 4197068, coverage 4197069; `ext` 4197138, 4197140; pooled 4197139. 28x28 with 256 states: evaluation 4197141, coverage 4197142.

- **At 14x14, with enough hidden states, power 4 matches power 6 on far, blends, divergences and clean yield.** Its
  cost falls from 3.8 far points at 32 states to zero at 128 states, at both training seeds (clean-yield differences
  +0.3 and +0.1 points).
  - On `base` the median nearest-image distance stayed about 0.07 larger with power 4 at 128 states, at both seeds
    (1.329 vs 1.260; 1.352 vs 1.284; seed-to-seed 0.02; held-out 1.165). On the independent `ext` trajectories the gap
    is 0.007 (1.344 vs 1.337), so it is not robust. No significance test was run for this metric.
  - **Confirmed on `ext`** (evaluation 4197062): far (50/50) +2.3 points at 32 states ([+0.2, +4.4], p = 0.04) and
    +0.9 at 128 states ([-0.6, +2.4], p = 0.25); recall of real 0s 0.81 vs 0.82 at 128 states.
- **At 28x28 with 32 states power 4 costs 4-7 far points.** On `base` 26.0% vs 19.0% (p = 2e-7); on `ext` (evaluation
  4197060) 22.6% vs 18.6% (p = 0.002; far 50/50 +4.6 [+1.7, +7.5]); pooled over n = 1000 (4197061) 24.3% vs 18.8%
  (p = 2e-9; far 50/50 +6.0 [+4.2, +7.9]; clean yield 50/50 -4.3 [-6.4, -2.2]). In return it has:
  - fewer blends (3.8% vs 6.2%, p = 0.02), and no divergences either;
  - texture closer to real images (roughness 0.148 vs 0.128; real 0.159);
  - class means closer to the data's (RMSE 0.026 vs 0.033 for the 0s; 0.010 vs 0.013 for the 1s, which is at the floor).
- **At 28x28 with 128 states power 4 reaches the real images' far rate and clean yield.**
  - Its cost shrinks with the number of states, as at 14x14: far (50/50) +7.3 points at 32 states, +3.0 at 128
    ([+1.1, +5.1]; `ext` +4.3 [+1.8, +6.8]; pooled +3.6 [+2.1, +5.2]).
  - Clean yield (50/50) -1.6 ([-3.9, +0.5]); `ext` -3.3 ([-6.4, -0.3]); pooled -2.5 ([-4.4, -0.6]).
  - Pooled over n = 1000: far 12.4% and clean yield (50/50) 82.9%, against 13.2% and 83.3% for real images.
  - Its samples are more varied than with power 6: recall of real 0s 0.55 vs 0.46 (`ext` 0.48 vs 0.39), spread of
    the 0s 8.5 vs 8.3 (real 9.8).
  - Its texture is closer to real images: roughness 0.148 vs 0.131 (real 0.159).
  - Against the paper: blends 3.4% vs 43.2% (p = 7e-53), far 12.4% vs 35.6% (p = 9e-20), clean yield (50/50) 84.6% vs
    31.9%.
- **At 28x28 with 256 states power 4 costs nothing measurable** (`base` only).
  - Against power 6 with 256 states:
    - clean yield (50/50) 90.3% for both (difference -0.0, [-2.5, +2.4]);
    - far 7.8% vs 6.0% (far 50/50 +1.8, [-0.2, +4.0], p = 0.09); blends 2.4% vs 3.4% (p = 0.3).
  - It is more varied: recall of real 0s 0.61 vs 0.55 (1s 0.83 for both); spread of the 0s 8.6 vs 8.4 (real 9.8).
  - Its texture is closer to real images: roughness 0.147 vs 0.132 (real 0.159).
  - Against power 4 with 128 states: far 7.8% vs 12.4% (p = 3e-4); clean yield (50/50) +5.7 ([+3.0, +8.4]).
  - So the far cost of power 4 shrinks with the number of states: +7.3 points at 32, +3.0 at 128 and +1.8 (not
    significant) at 256.
- **Power 4 with hidden units beats power 6 without them by a wide margin.** At 28x28, Potts 32 + both with power 4
  against the paper: blends 3.8% vs 43.2% (p = 2e-55), far 26.0% vs 35.6% (p = 1e-4), diverged 0% vs 6.2%, clean yield
  (50/50) 70.9% vs 31.9%.
- **At 28x28 with 512 states, power 4 gives the best samples so far** (`base` only; training 4198215, sampling
  4198216, evaluation 4198225, coverage 4198226):
  - clean yield (50/50) 92.3%, far 6.4%, blends 1.6%, no divergences; with 256 states 90.3% / 6.0% / 3.4% (power 6)
    and 90.3% / 7.8% / 2.4% (power 4). The paired differences are not significant: clean yield +1.9 ([-0.7, +4.6])
    and +2.0 ([-0.7, +4.6]); blends against power 6 with 256 states p = 0.049.
  - Real images score 83.3% clean and 13.2% far: the samples are more typical than real digits. The recall shows
    what is still missing.
  - More variety: recall of real 0s 0.68, against 0.55 (power 6) and 0.61 (power 4) with 256 states (real 0.89);
    the 1s stay at 0.83.
  - Distance to the training set: median nearest-image distance 3.20, like 256 states with power 6 (3.17), and below
    256 states with power 4 (3.36) and held-out real images (3.29). Copy rate 1.4% (256 states: 0.6% / 0.4%;
    held-out real images 5% by construction).
  - Against the paper: clean yield +60.4 points ([+56.3, +64.3]); blends 1.6% vs 43.2%, far 6.4% vs 35.6%, diverged
    0% vs 6.2%.
- **For hardware:** the x^6 term can be dropped once hidden units supply the multi-modality. The price falls with
  the number of hidden states; with 256 states at 28x28 it is not measurable, and power 4 with 512 states is the
  best 28x28 model so far.

## Result 10 (2026-11-08): can the hidden units be trained locally, on the device?

**1. Force matching is a local rule for the hidden units too** (DESIGN §2.1).
- The score-matching ("force-matching") gradient of the Potts couplings W, Q and fields b, and of the per-pixel units,
  is a three-factor rule.
  - Coupler (m, i) uses p_m and one scalar c_m from its hidden state, and x_i, ψ_i and the hidden mean force Ḡ_i from
    its pixel.
  - c_m needs two projections, G_m·ψ and G_m·Ḡ: the change of state m's input field when the clamped pixels are nudged
    along ψ or Ḡ.
- Written out this way (`latent_model.force_matching_local_grad`), it equals the autodiff gradient used in training to
  1e-16 to 4e-11 (jobs 4197172, 4197173; trained Potts 128 + both at 14x14 and 28x28, slices 1, 50, 90 and 99).
- So every model of Results 7-9 was trained with this local rule, given exact hidden statistics.

**2. Measured hidden statistics: how many hidden samples per clamped configuration?**
- On a device, p_m is the average over S hidden states sampled while the pixels are clamped. The fluctuation terms
  are the sample covariances.
- `--hidden-samples S` uses exactly these plug-in estimates, in the clamp force and in every gradient term.
- Gradient check on the trained 14x14 Potts 128 + both (job 4197166; batch 512, 16 draws; slices 1, 50, 90, 99):

| | two exact batches differ by | S = 4 | S = 16 | S = 64 | S = 256 | S = 4096 |
|---|---|---|---|---|---|---|
| error of one estimate, hidden parameters | 1.2-1.8 | 0.6-4.6 | 0.3-2.5 | 0.14-1.5 | 0.07-0.7 | 0.02-0.17 |
| error of one estimate, visible θ | | 0.02-0.07 | 0.01-0.03 | 0.005-0.017 | 0.002-0.008 | ≤0.002 |
| bias | | visible (W, slice 50: mean of 16 draws 0.34, noise alone 0.16) | small | none detectable | none | none |

  "Error" is |ĝ − g| / |g| on the same batch.
  - The first column is |g(X₂) − g(X)| / |g(X)| for two independent batches of 512. One batch's own noise, which
    training already absorbs, is about 1/√2 of that: 0.85-1.25.
  - The errors fall as 1/√S, and the cosine with the exact gradient is 0.99-1.00 at S = 4096.
  - The largest errors are at the first and last slices (t = 5 and t = 0); at the middle slices S = 16 gives
    0.3-0.6.
- **Reading:** from S ≈ 64 hidden samples per clamped configuration, the measured-statistics rule is no noisier than
  one minibatch at any slice. S = 16 is up to about twice as noisy at the first and last slices, and quieter in
  between. In this check the plug-in bias is below the noise from S ≈ 64. In training it still costs about 4
  points of clean yield at S = 64 (plain 76.9% vs 81.1% with the jackknife, below; not a paired test).
- **Training with measured statistics** (14x14, Potts 128 + both; paired evaluations 4197439 and 4197511, coverage
  4197440 and 4197512):

| | exact | exact, seed 2 | S = 256, jackknife | S = 64, jackknife | S = 64 | S = 16, jackknife | S = 16 | S = 4 | K = 0 |
|---|---|---|---|---|---|---|---|---|---|
| far | 13.2% | 11.4% | 12.0% | 16.6% | 20.2% | 21.6% | 31.2% | 35.0% | 48.4% |
| clean yield, 50/50 | 84.3% | 85.9% | 85.6% | 81.1% | 76.9% | 75.6% | 66.4% | 63.6% | 38.6% |
| clean yield vs exact (95% CI) | | +1.5 [-1.1, +4.2] | +1.3 [-1.5, +4.0] | -3.2 [-6.2, -0.4] | -7.5 [-10.9, -4.0] | -8.8 [-12.2, -5.4] | -17.9 [-21.5, -14.4] | -20.7 [-24.3, -17.0] | -45.7 |
| flagged blends | 2.6% | 2.8% | 2.6% | 2.0% | 3.4% | 2.6% | 4.8% | 6.2% | 32.6% |
| recall of real 0s / 1s | 0.77 / 0.87 | 0.80 / 0.87 | 0.79 / 0.87 | 0.78 / 0.86 | 0.75 / 0.82 | 0.76 / 0.83 | 0.71 / 0.76 | 0.69 / 0.70 | 0.48 / 0.79 |
| precision of the 1s | 0.88 | 0.87 | 0.86 | 0.85 | 0.77 | 0.74 | 0.59 | 0.55 | 0.35 |

  - **With 256 hidden samples per clamped configuration and the jackknife, force matching with measured statistics
    trains as well as with exact statistics:**
    - clean yield (50/50) +1.3 points ([-1.5, +4.0], p = 0.37);
    - far 12.0% vs 13.2% (p = 0.42); blends 2.6% for both;
    - the same coverage of both classes.
  - **At full size (28x28, Potts 128 + both) S = 64 with the jackknife already trains as well as exact statistics**
    (training 4198213, sampling 4198214, evaluation 4198223, coverage 4198224; `base`, paired against exact training):
    - clean yield (50/50) 87.6% vs 86.3% (+1.4, [-1.3, +4.0], p = 0.33); far 8.4% vs 9.4% (p = 0.42); blends 4.0% vs
      4.8% (p = 0.52); no divergences;
    - recall of real 0s / 1s 0.52 / 0.76 against 0.46 / 0.79 (subsampling spread ±0.03-0.05); precision of the 1s
      0.93 vs 0.95;
    - against the paper: clean yield (50/50) +55.7 points ([+51.3, +59.9]); blends 4.0% vs 43.2%, far 8.4% vs
      35.6%, diverged 0% vs 6.2%.
    - It still keeps fewer states than exact training: 82 effective states at t = 0 against 112 (training log; exact:
      `inspect_states.py`, Caveat 6). But its consolidation dip is shallow: at least 75 effective states throughout
      (75.5 at slice 15), against a dip to 48 (slice 25) for S = 64 at 14x14. One run, one seed.
  - **S = 256 at 28x28 gives the same** (training 4198663, sampling 4198664, evaluation 4198665, coverage 4198666;
    `base`, paired):
    - clean yield (50/50) 86.9%: against exact statistics +0.6 ([-1.5, +2.6]), against S = 64 -0.8 ([-3.3, +1.8]);
    - far 8.2% (exact 9.4%, S = 64 8.4%; p = 0.21 against exact); blends 5.2% (4.8%, 4.0%); no divergences;
    - recall of real 0s / 1s 0.50 / 0.77 (exact 0.46 / 0.79, S = 64 0.52 / 0.76); precision of the 1s 0.95;
    - 97 effective states at t = 0, at least 84 throughout (training log; S = 64: 82 and 75).
    - So at 28x28 on 0/1, S = 64 with per-element Adam is already enough; the extra states of S = 256 buy no
      measurable clean yield. At 14x14 S = 64 still cost 3.2 points; the larger images seem to need fewer hidden
      reads (one run each, not tested across sizes).
  - Each fourfold increase of S counts. With the jackknife, S = 16 -> 64 -> 256 gives clean yield 75.6% -> 81.1% ->
    85.6% (+5.5, [+2.6, +8.4]; +4.5, [+2.0, +7.1]).
  - With plug-in statistics the loss grows as S shrinks. Part of it is bias (below), which the jackknife removes:
    S = 16 with jackknife against plain S = 16 gives far 21.6% vs 31.2% (p = 8e-9) and clean yield +9.2 points
    ([+6.0, +12.4]).
  - What finite S still costs after the jackknife (8.8 points at S = 16, 3.2 at S = 64) is hidden states lost during
    training ("What finite S costs is states", below).
    - The overall noise level is not the problem. In aggregate CD-1's hidden gradient is noisier still and loses only
      3 states (point 3). At slice 15, S = 256-64 adds 0.28-0.58 |g| to one batch's own noise of about 1.0 |g| (two
      exact batches differ by 1.4 |g|; job 4199405).
    - But these errors are norms over all states, dominated by the occupied ones. What matters is the noise on each
      rarely occupied state's own parameters. CD-1's averages are exact, so every state gets a gradient every step;
      a sampled state is not visited at all in most steps.
    - At slice 25 CD-1 and S = 64 with the jackknife each had 28 states at 1e-8-1e-4 mean occupation. CD-1 lost 3
      states in all, S = 64 lost 54 (job 4199418).
  - Finite S costs most where the hidden units are still uncommitted, in the middle of the schedule (validation-loss
    table in point 3).

- **Part of the cost is bias.** A device averaging S hidden samples estimates every product of hidden averages
  with an O(1/S) bias. For the loss, E[(sample mean)²] = μ² + σ²/S gives exactly
  E[L̂_S] = J_T − V/(2 S T²), where V is the hidden force variance. That term rewards less committed hidden states.
  - Score matching's gradient is a small difference of large terms, so this bias moves the optimum. Noise averages
    out over steps; bias does not.
  - Measured on the trained model (job 4197238, 64 draws): with S = 16 the plug-in gradient is off by 0.08-0.26 |g|
    after subtracting the noise (e.g. bl at slice 50: mean of 64 draws 0.27 |g|, noise alone 0.06).
- **Fix: the jackknife** (`--hidden-jackknife`): 2 L(S draws) − [L(first S/2) + L(second S/2)] / 2.
  - It cancels the O(1/S) bias exactly for the loss and leaves O(1/S²) for the gradient.
  - On a device it means combining the updates from one full and two half averaging windows.
  - With S = 16 the remaining bias is below the test's resolution (about 0.05 |g|), and the noise is unchanged.
- **Reading:** force matching trains the hidden units on a device with measured statistics. It needs:
  - bias correction (the jackknife);
  - about 256 hidden samples per clamped configuration. For fast hidden units that is a time average over about
    256 of their relaxation times per training sample, while the pixels are clamped.
- **What finite S costs is states** (`inspect_states.py`, jobs 4199383, 4199394, 4199418; 0/1, Potts 128 + both):

| | exact | CD-1 | S = 256, jk | S = 64, jk | S = 64 | S = 16, jk | S = 16 | S = 4 |
|---|---|---|---|---|---|---|---|---|
| states used at t = 0 (of 128) | 128 | 125 | 96 | 74 | 36 | 33 | 14 | 11 |
| effective states at t = 0 | 99 | 99 | 81 | 62 | 32 | 30 | 12 | 10 |
| clean yield, 50/50 | 84.3% | 85.3% | 85.6% | 81.1% | 76.9% | 75.6% | 66.4% | 63.6% |
| median nearest-image distance | 1.260 | | 1.295 | 1.330 | 1.438 | 1.384 | 1.548 | 1.618 |
| exact training with about as many states (Result 8): clean yield, distance | | | 64-128: 80.7-84.3%, 1.366-1.260 | 64: 80.7%, 1.366 | 32: 75.7%, 1.386 | 32: 75.7%, 1.386 | 16: 66.4%, 1.478 | 8 / 16: 63.0 / 66.4%, 1.467 / 1.478 |

  ("used": the state is the most likely one for at least one of 4,096 data points; CD-1: 64 releases, exact hidden
  averages; jk: jackknife. Exact seeds differ by about 4 used states: 128 vs 124. Distances: evaluations 4197439,
  4197511, 4196913.)
  - **In its samples, a jackknifed run behaves like an exactly trained model with about as many states as it kept**:
    clean yield within about 1 point, far within 2, and about the same distance to the training set.
    - At S = 256 the 32 lost states cost no measurable clean yield, so that column does not test this reading. Its
      distance (1.295) fits about 96 states.
    - Without the jackknife the samples are also farther from the training set than those of an exact model with as
      many states (1.62 / 1.55 / 1.44 at S = 4 / 16 / 64, against 1.47 / 1.48 / 1.39). There the plug-in bias
      distorts the model on top of the lost states.
    - In validation loss none of them is simply a smaller exact model. Through the middle of the schedule they match
      (slice 45: S = 64 with the jackknife +6.5 against +8.2 for 64 exact states). From slice 85 on they fall below
      exact training (t = 0: -283 to -1058), while smaller exact models lie above it (64 states: +763).
  - When: every run inspected, exact training included, passes through a consolidation phase from about slice 10 to
    25 (t ≈ 4.5-3.7), in which the data crowd into fewer states. The annealing phase of point 5.
    - Exact training: effective states 108 at slice 5, 79 at slice 15, 71 at slice 20, 77 at slice 25. It then
      brings the states back: 97 effective states at slice 40, all 128 used at t = 0.
    - The plug-in runs crowd earlier and deeper (S = 64: 101 -> 64 -> 32 at slices 5 / 10 / 15).
    - The sampled runs recover only part of the way (S = 64 with the jackknife: 48 at slice 25 -> 61 at slice 40 ->
      62 at t = 0).
  - Which states are lost: those whose mean occupation falls below about 1e-4 during or just after consolidation
    (slices 20-40; decades of the mean occupation over 4,096 data points, job 4199418):
    - exact training: at most 11 states fall to 1e-8-1e-4 (slice 25), none lower, and all come back;
    - S = 64 with the jackknife: 28 states at 1e-8-1e-4 at slice 25, 51 at slice 30, 55 below 1e-4 at slice 40. At
      t = 0, 46 states sit below 1e-32 and 54 never exceed 1e-6 on any data point;
    - S = 256 with the jackknife: 21 at slice 25, 35 at slice 30, 32 lost;
    - S = 16 without it: 115 states below 1e-4 by slice 25, 114 lost.
  - Bias kills states too: at equal S the jackknife keeps about twice as many (74 vs 36 at S = 64, 33 vs 14 at
    S = 16).
  - **Working explanation**, for what remains after the jackknife (tested below: visits vs bias, and the role of
    Adam):
    - A state with mean occupation p̄ is visited about λ = B S p̄ times per step (B = 512 training samples). An
      unvisited state gets exactly zero gradient (every term carries its sampled occupation).
    - So its gradient is zero in most steps and a spike in the others: unbiased with the jackknife. But Adam divides
      it by its own root-mean-square, so the state moves at most about √λ as fast as with exact statistics (about
      32 λ once visits are rarer than one per 1,000 steps, Adam's second-moment window).
      (Correction from ChatGPT: First-order jackknife leaves O(1/S²) bias in the hidden-gradient estimator.
      Second-order jackknife removes that remaining bias for the current cubic estimator under independent
      draws at fixed parameters. Correlated physical time samples need additional analysis.)
    - With exact averages the gradient, however small and noisy, is present every step, and Adam turns it into
      full-size steps (CD-1, with exact averages, shows the same).
    - States below λ ≈ 1, p̄ ≲ 1/(B S) (3e-5 at S = 64), fall behind the growing couplings of the others and are lost
      for good.
  - Tests (0/1, Potts 128 + both; the measured-statistics runs all with the jackknife; effective states per slice
    from the training logs; the logs and
    `inspect_states.py` can differ by up to about 2 effective states: 1.8 for S = 64 with the floor at slice 60,
    during its collapse, 1.2 for one gain per state at t = 0, 0.2 for S = 16 at t = 0. The per-element exact row
    comes from `inspect_states.py` because its training log predates the fix of the logged hidden diagnostics,
    Caveat 6):

| slice | 10 | 20 | 30 | 40 | 50 | 60 | 70 | t = 0 |
|---|---|---|---|---|---|---|---|---|
| exact, per-element Adam (`inspect_states.py`) | | 71 | | 97 | | | | 99 |
| S = 256 | 101 | 62 | 70 | 79 | 83 | 84 | 84 | 81 |
| S = 64 over 4 replicates | 102 | 62 | 68 | 78 | 80 | 80 | 82 | 75 |
| S = 64 | 100 | 49 | 55 | 61 | 65 | 66 | 67 | 62 |
| S = 256 to slice 29, then 16 | 101 | 62 | 70 | 74 | 75 | 76 | 77 | 71 |
| S = 16 over 16 replicates | 81 | 39 | 49 | 56 | 56 | 56 | 57 | 52 |
| S = 16 | 96 | 26 | 25 | 30 | 30 | 30 | 30 | 29 |
| S = 64 with the floor | 102 | 53 | 54 | 67 | 92 | 23 | stopped | |
| S = 16 with the floor | 98 | 29 | 30 | 41 | 11 | 2 | 2 | stopped |
| exact, group normalisation | 95 | 38 | 24 | 21 | 21 | 22 | 23 | 22 |
| S = 64, group normalisation | 92 | 36 | 24 | 23 | 23 | 23 | 23 | 22 |
| exact, one gain per state | 103 | 71 | 86 | 98 | 107 | 107 | 108 | 103 |
| exact, visit-count rule | 100 | 64 | 79 | 99 | 108 | 108 | 109 | 108 |
| S = 64, one gain per state | 100 | 49 | 55 | 63 | 65 | 67 | 68 | 65 |
| S = 64, visit-count rule | 103 | 63 | 72 | 83 | 87 | 87 | 88 | 86 |
| S = 16, visit-count rule | 102 | 49 | 59 | 75 | 79 | 79 | 80 | 78 |
| S = 8, visit-count rule | 86 | 41 | 61 | 79 | 87 | 87 | 87 | 81 |
| S = 4, visit-count rule | 45 | 54 | 70 | 85 | 95 | 99 | 100 | 94 |
| S = 2, visit-count rule | 28 | 40 | 58 | 75 | 86 | 89 | 90 | 84 |
| S = 8, visit-count rule, counts from the same draws | 86 | 41 | 60 | 78 | 89 | 88 | 89 | 84 |
| S = 16, second-order jackknife | 106 | 43 | 35 | 42 | 44 | 44 | 44 | 41 |
| S = 16, second-order jackknife, visit-count rule | 107 | 60 | 65 | 73 | 75 | 75 | 75 | 73 |

    - **Visits (and noise), not bias, at S = 64:** S = 64 averaged over 4 independent replicates
      (`--hidden-replicates`: the bias of 64 draws, the visits and noise of 256) samples like S = 256 (pipeline
      4199408; paired evaluation 4199411, coverage 4199412, states 4199413):
      - clean yield (50/50) 85.7% against 81.1% for S = 64 with one estimate (+4.6, [+1.9, +7.3]), 85.6% for
        S = 256 (+0.1, [-1.9, +2.1]) and 84.3% for exact statistics (+1.4, [-1.1, +3.9]); far 12.6% vs 16.6% for
        S = 64 (McNemar p = 0.003); median nearest-image distance 1.294 (S = 256 1.295, S = 64 1.330);
      - it keeps its states like S = 256 through slice 70, where S = 64 with one estimate has lost a fifth of them;
        at t = 0 it keeps 75 effective states (93 used), against 81 for S = 256 and 62 for S = 64;
      - the gradient check confirms that the replicates are as accurate per step as S = 256 (job 4199405);
      - so at S = 64 the cost is the missing visits and the noise of 64 draws, not their bias; the replicates
        cannot separate visits from noise.
    - **At S = 16 visits and noise are only half of the cost:** 16 replicates of S = 16 (the visits and noise of
      256 draws, the bias of 16; pipeline 4199410):
      - clean yield 80.9% against 75.6% for S = 16 with one estimate (+5.3, [+2.1, +8.6]), but 4.7 points below
        S = 256 ([-7.5, -2.1]); far 17.2% vs 21.6% and 12.0% (McNemar p = 0.005 and 7e-5);
      - 53 effective states at t = 0 by `inspect_states.py` (52 in the training log; 63 used), against 30 for
        S = 16 with one estimate and 81 for S = 256. They crowd earlier than every other jackknifed or exact run with
        per-element Adam (81 at slice 10, 39 at slice 20); only the plug-in runs, and S = 4 and 2 with the
        visit-count rule (45 and 28 at slice 10), crowd earlier;
      - so at S = 16 visits (and noise) explain about half of the cost. The rest is not visits or noise; the
        jackknife's O(1/S²) bias was the obvious candidate. With per-element Adam the second-order jackknife
        confirms that the bias costs about as much (+5.1 at the same 16 reads, below); with the visit-count rule
        its effect is smaller and not significant (S = 4-16 flat; order 2 adds +1.5 [-1.1, +4.1]); whether it is
        really smaller is untested (one run per arm).
    - **Many draws matter most early** (slices 0-29, which include the consolidation; pipeline 4199409): S = 256
      up to slice 29 and S = 16 afterwards matches S = 256 throughout and exact training in clean yield, far and
      blends with about a third of the draws, but keeps fewer states (a switch at slice 10 would be needed to tie
      the benefit to the consolidation itself):
      - clean yield (50/50) 84.9%, far 13.0%, blends 3.0%, against 85.6% / 12.0% / 2.6% for S = 256 and 84.3% /
        13.2% / 2.6% for exact statistics; paired (4199411): -0.8 ([-3.1, +1.6]) against S = 256, +0.5 ([-2.0,
        +3.1]) against exact statistics;
      - it keeps 71 effective states at t = 0 (S = 256: 81; S = 16 throughout: 29 in the log, 30 by
        `inspect_states.py`, clean yield 75.6%). The two runs are identical up to slice 29; the gap opens right after
        the switch (slice 40: 74 vs 79);
      - median nearest-image distance 1.349, against 1.295 for S = 256, 1.260 for exact statistics and 1.330 for
        S = 64 with the jackknife. That is about what exact training with 64-128 states gives (Result 8: 1.366-1.260;
        interpolated, about 70 states), in line with its 71 effective states. No significance test for the
        distance, and noisy at this level: it is farther than S = 64 although it keeps more states;
      - so the later slices still need more than 16 draws to keep as many states as S = 256 (71 vs 81 effective at
        t = 0), though not for the clean yield;
      - draws per configuration summed over the slices: 30 x 256 + 70 x 16 = 8,800, against 25,600 for S = 256
        throughout (weighted by the steps per slice, which grow towards t = 0, the saving is larger).
    - **A naive local fix fails: homeostasis** (`--hidden-floor 2`). Each state counted its own visits, and a state
      that no draw visited in a step raised its bias by twice its bias learning rate (on a second, independent set of
      B S reads).
      - From about slice 45 (t ≈ 2.7) the occupation collapses onto a few states, and training diverges. S = 16:
        effective states 43 at slice 42, 8 at slice 51, 2 at slice 60; validation loss -4751 at slice 42 and +2819
        at slice 69, against -4510 without the floor. S = 64: 88 effective states at slice 48, 23 at slice 60.
      - Both runs were stopped (jobs 4199448/49, at slices 61 and 71).
      - The floor does keep the states visible (`inspect_states.py`, job 4199762; S = 16): no state falls below 1e-8,
        and at slice 50, 91 states win some data against 33 without the floor.
      - But the data then crowd into two states. At slice 60, 2.3 effective states and mean max p 0.995, while the
        many revived states stay at 1e-8-1e-4 mean occupation. Why the revived states destabilise the occupied ones
        is open.
      - A working rule would need more than a bias push: e.g. a much gentler push, or one that also moves the revived
        state's couplings towards data it could capture.
    - **The role of Adam: steps with one step size per group lose the states even with exact statistics**
      (pipelines 4199774/75): the hidden parameters with one root-mean-square per group W, b, Q, wl, bl instead of
      per element (`--hidden-norm group`; Adam's momentum is kept; the visible parameters keep per-element Adam).
      This is the closest tested rule to the plain gradient steps of the simplest hardware ("plain steps" below);
      truly plain steps, without any adaptive scale, are untested. Per-element Adam is local as well (each coupling
      keeps two leaky averages of its own gradient), but costs that state and a division per coupling.
      - Exact statistics: 22 effective states from slice 40 on (per-element Adam: 97 at slice 40, 99 at t = 0). With
        S = 64 and the jackknife the same: 22 at t = 0. Under plain steps S = 64 costs no extra states.
      - Samples, paired (evaluation 4199776, coverage 4199777):
        - exact statistics, plain against per-element: clean yield (50/50) 70.4% vs 84.3%, -13.9 points ([-17.7,
          -10.3]); far 25.8% vs 13.2% (p = 5e-12); blends 4.4% vs 2.6% (p = 0.11). The 0s suffer (clean 45.5%,
          far 52.1%). That is what exact training with 16-32 states gives (66.4-75.7%, Result 8): again a smaller
          model;
        - S = 64 with the jackknife, plain against per-element: 71.7% vs 81.1%, -9.4 ([-12.6, -6.1]);
        - under plain steps S = 64 equals exact statistics: 71.7% vs 70.4% (+1.3, [-1.2, +4.0]); far 25.0% vs 25.8%;
        - recall of real 0s 0.70 (exact, plain) and 0.73 (S = 64, plain) against 0.77 and 0.78 with per-element Adam;
          precision of the 1s 0.71 and 0.69 against 0.88 and 0.85; median nearest-image distance 1.425 and 1.434
          against 1.260 and 1.330. (Coverage 4200972 gives recall 0.69 and 0.75 and precision 0.72 and 0.70 for the
          same samples: `coverage_samples.py` seeds its subsamples by the set's name, so renamed sets draw new
          subsamples; the spread is about ±0.03.)
      - States (`inspect_states.py`, 4199778): under plain steps 25 states (exact) and 26 (S = 64) win data at t = 0,
        14-15 for 0s and 11 for 1s. The other 102-103 sink below 1e-32 mean occupation.
        - The hard assignment narrows early. With plain steps 24 states win data at slice 10, 14 at slice 15 and 19
          at slice 25. With per-element Adam (exact statistics, job 4199394) it is 41, 33 and 68, and all 128 by t = 0
          (job 4199383).
        - Both optimisers pass through the consolidation. Only per-element Adam brings the states back afterwards.
      - So the optimiser, not only the statistics, decides which states survive (one run per arm). Group
        normalisation changes every hidden parameter's step, so this does not yet isolate the mechanism. Working
        explanation, which test 16 and its production runs test: a state's gradient carries its occupation.
        Per-element Adam divides that out when the gradient is present every step (exact statistics), and only partly
        when it comes as rare visits (measured statistics, √λ above). Plain gradient steps never divide it out.
    - **Next tests: cheaper per-state step sizes** (test 16, job 4200653, passed; production runs 4200967-70, exact and
      S = 64 with the jackknife for each rule; evaluation 4200971, coverage 4200972, states 4200973):
      - `--hidden-norm state`: Adam with one second moment per hidden state in each group (its row of W, separately
        its row of Q; its bias b keeps per-element Adam, which is already one per state) instead of one per coupling:
        a gain control per unit. With exact statistics it divides out the occupation as per-element Adam does. With
        rare visits it should give the same √λ as per-element Adam. The per-pixel units wl, bl keep per-element Adam.
      - `--hidden-norm occupancy`: each Potts state divides a leaky average of its gradient (time constant 1,000
        steps) by a leaky average of its own occupation: the exact batch mean or, on a device, its visit count among
        B S fresh draws, floored at one visit in the latest step. One step scale per group, the occupation-weighted
        root-mean-square of the per-step gradient per unit occupation, noise included. Normalised steps are clipped
        at ±1 per element, so each parameter moves by at most one learning rate per step. The per-pixel units wl, bl
        get plain steps (one root-mean-square per group, as with `--hidden-norm group`). The optimiser state,
        including the visit averages, restarts at every slice.
        - It is local: each state needs one visit counter, each coupling one leaky average; one scale per group.
        - It is the analogue of the natural-gradient (or EM) step of a mixture model, where the Fisher information of
          a component's parameters is proportional to its weight.
        - Expected: a state visited at least once per averaging time moves as fast as an occupied one, where
          per-element Adam gives √λ.
      - A first occupancy design failed its smoke test (test 15, job 4200507). Its step scale was the root-mean-square
        of the averaged direction itself, so it moved at full speed even at an optimum: |W| 0.76 -> 2.23 -> 3.90 in
        slices 0-2, against about 0.74 with per-element Adam. Not run further.
      - Test 16: both rules pass their unit checks, and in slices 1-2 at full steps they behave like per-element
        Adam: |W| 0.74-0.75, effective states at slice 2 109 (state) and 106 (occupancy) with exact statistics, 109 for
        both with S = 64 (per-element Adam with S = 64: 109; plain steps: 87 exact, 100 with S = 64).
      - Interim, exact statistics (training logs at slice 34, 00:05):
        - `state` tracks per-element Adam almost exactly: effective states 103 / 79 / 71 / 77 at slices
          10 / 15 / 20 / 25 (per-element Adam, `inspect_states.py`: 102.5 / 79.0 / 70.9 / 77.0); |W| 0.93 / 1.74 at
          slices 10 / 20 (0.92 / 1.74);
        - `occupancy`: 100 / 76 / 64 / 70, then 79 at slice 30 and 89 at slice 34;
        - plain steps: 95 / 61 / 38 / 28, then 24 and 23.
        - So with exact statistics a per-state step size is enough to bring the states back; per-element
          normalisation is not needed.
      - Production runs, one per arm (pipelines 4200967-70 and 4201364; paired evaluations 4200971 and 4201366 on
        the 500 `base` trajectories, coverage 4200972 and 4201367, states 4200973 and 4201368; effective states per
        slice in the table above):
      - **Exact statistics: a per-state step size is enough.** Against per-element Adam (84.3%): one gain per state
        83.3% (-1.0, [-2.4, +0.3]), visit-count rule 85.6% (+1.3, [-1.1, +3.7]). Against plain steps (70.4%): +12.9
        ([+9.3, +16.6]) and +15.2 ([+11.6, +18.7]).
        - Far (50/50) 14.1% and 12.2% (per-element 13.7%, plain 26.6%); blends 3.4% and 3.0% (2.6%, 4.4%); median
          nearest-image distance 1.256 and 1.266 (1.260, 1.425).
        - Recall of real 0s 0.77 and 0.76 (per-element 0.77, plain 0.69); precision of the 1s 0.89 and 0.88 (0.88,
          plain 0.72).
        - Effective states 102 and 108 at t = 0 by `inspect_states.py` (per-element 99, plain 22). During
          consolidation one gain per state dips as deep as per-element Adam (71 at slice 20), the visit-count rule
          deeper (64), and both recover.
      - **Measured statistics, S = 64 with the jackknife:**
        - one gain per state: 82.8%, +1.7 against per-element Adam with S = 64 ([-0.6, +4.1]). It loses the states
          as per-element Adam does: 65 effective states at t = 0 against 62, and the same 49 / 55 at slices 20 / 30.
          That is consistent with a rare visitor getting the same √λ speed from a per-state as from a per-coupling
          normalisation (step sizes not measured);
        - visit-count rule: 83.4%, +2.3 ([-0.4, +5.1], p = 0.10); far (50/50) 15.1% vs 17.3% (p = 0.10); 86 effective
          states at t = 0 against 62, more than S = 256 with per-element Adam (81). That is consistent with the
          design, a rarely visited state moving about as fast as an occupied one (step sizes not measured);
        - one gain per state and the visit-count rule do not differ in clean yield (-0.6 [-3.4, +2.2]), so at
          S = 64 the extra states of the visit-count rule buy no measurable clean yield;
        - the visit-count rule with S = 64 against itself with exact statistics: -2.2 ([-4.7, +0.3]); far (50/50)
          15.1% vs 12.2% (+3.0, [+0.8, +5.2]).
      - **S = 16 with the jackknife: the visit-count rule removes most of the cost** (pipeline 4201364):
        - reads: the rule's visit counts come from S fresh draws per configuration and step (`occupation` in
          `train_latent.py`), on top of the S draws behind the gradient statistics, so it reads 2S = 32 hidden states
          per configuration and step, against 16 for per-element Adam with S = 16. (Reusing the same draws for the
          visit counts halves that and, at S = 8, costs nothing measurable: "Visit counts from the same draws",
          below.)
        - clean yield 81.4% against 75.6% with per-element Adam (+5.8, [+2.5, +9.0], p < 2e-4); far (50/50) 16.8% vs
          22.4% (p = 2e-4); precision of the 1s 0.86 vs 0.73; median nearest-image distance 1.359 vs 1.384;
        - 78 effective states at t = 0 against 30;
        - still below exact statistics (-3.0, [-5.6, -0.3]) and S = 256 with per-element Adam (-4.3, [-7.0, -1.6]);
          against itself with S = 64, -2.1 ([-4.7, +0.5]);
        - the same as per-element Adam with 16 replicates of S = 16 (256 reads, with the bias of 16 draws): +0.5
          ([-2.0, +3.0]); far (50/50) 16.8% vs 17.7%; 78 effective states against 53 (paired evaluation 4202614;
          states 4201368 and 4199413). So with 32 reads the visit-count rule does what per-element Adam does with
          256: the extra noise of 16 draws costs nothing measurable here. The rest, 4-5 points against S = 256,
          matches what the 16 replicates leave (-4.7), which pointed to the bias of 16 draws, and with per-element
          Adam the second-order jackknife bears that out (+5.1, below); with this rule, though, S = 8 does as well as
          S = 16 (below), which argues against bias;
        - at S = 64 it stays below 4 replicates of S = 64 (83.4% vs 85.7%, -2.3 [-5.0, +0.5], p = 0.10, not
          significant). If real, its cause is open: probably not the noise of 64 draws alone, since at S = 16 the
          rule matches 16 replicates; the replicate arm also uses per-element Adam.
      - **A second-order jackknife against the remaining S = 16 bias** (`--hidden-jackknife-order 2`; test 17,
        job 4202647; then pipelines 4202648 with the visit-count rule and 4202649 with per-element Adam, S = 16;
        results at the end of this item):
        - the score-matching loss is quadratic, and its parameter gradient at most cubic, in the sampled hidden means
          of one configuration. So their expectations over n draws are exactly a + b/n + c/n². The split-half
          jackknife removes b and leaves -2c/S²;
        - Richardson extrapolation over the full set, its two halves and its four quarters of the same draws,
          (8/3) L(S) - 2 L(halves) + (1/3) L(quarters), removes both. To first order in the fluctuations it adds no
          noise (all three share the same linear term). On a device: a full, two half and four quarter averaging
          windows of the same reads;
        - expected if the remaining S = 16 cost is the bias: with the visit-count rule, S = 16 close to S = 256
          (85.6%).
        - Test 17 (job 4202647) passed:
          - exact enumeration (a cubic of the mean of S = 8 Bernoulli(0.3) draws, all 256 outcomes): order 2 is
            unbiased to rounding (4e-16); order 1 leaves exactly -2κ₃/S² (-2.6e-3); the plug-in is off by 7.7e-2;
          - Monte Carlo through the loss itself (Potts 3 + both, 4 configurations, S = 8, 400,000 draws): order 2 is
            consistent with zero bias in every parameter (max |z| 3.9 over all elements). The plug-in's bias is
            obvious (|z| up to 930); order 1's O(1/S²) remainder is below this test's resolution as well;
          - the order-1 path is unchanged (validation losses of slices 0-1 equal to the production run's to all
            digits);
          - on the trained Potts 128 + both (S = 16 and 64, slices 10, 20, 30, t = 0): the mean of 64 estimates is
            noise-dominated for both orders (its error is that of one estimate over 8), so the order-1 bias per step
            is at most about 0.1 |g|, against a noise of about 0.6-2.7 |g| per estimate at S = 16 (0.3-1.5 at S = 64)
            in the hidden groups. Order 2 adds up to about 10% noise at S = 16 (W 1-6%, bl 5-10%) and under 3% at
            S = 64 (W at S = 16: 0.95 -> 0.99 |g| at slice 20, 0.68 -> 0.72 at slice 30);
          - the visible couplings' gradient is linear in the sampled means and therefore the same for every order;
          - smokes; invalid combinations refused; Potts 128 + both with the visit-count rule tracks order 1 over
            slices 0-2 at full steps.
        - So per step the bias that order 2 removes is small next to the noise; whether it adds up over a training
          run (as the 16 replicates suggest) is what the production runs test.
        - **Results (paired evaluation 4202650, coverage 4202651, states 4202652; one run each, both at S = 16 and
          the same 16 hidden reads per configuration for the statistics as order 1):**
          - **per-element Adam: 80.7% against 75.6% with order 1 (+5.1, [+2.3, +8.0], p = 2e-4)**; far (50/50)
            18.1% vs 22.4% (-4.4, [-7.1, -1.7]). The same as 16 replicates of S = 16 with order 1 (80.9%, -0.2
            [-3.2, +2.9]; far (50/50) 18.1% vs 17.7%), which read 16 times as much. Still 3.6 points below exact
            statistics and 4.9 below S = 256 (unpaired differences). Precision of the 1s 0.81 (order 1 0.73,
            exact 0.88); median nearest-image distance 1.360 (order 1 1.384); 41 effective states at t = 0 (48
            used, 80 lost below 1e-32), against 30 with order 1;
          - **visit-count rule: 82.9% against 81.4% with order 1 (+1.5, [-1.1, +4.1], p = 0.25, not
            significant)**; against exact statistics -1.5 ([-4.0, +1.1], p = 0.26); against S = 256 -2.8 ([-5.3,
            -0.2], p = 0.03); against order 2 with per-element Adam +2.1 ([-0.7, +5.0], p = 0.15). Far (50/50)
            15.3%; precision of the 1s 0.87; median nearest-image distance 1.323 (order 1 1.359, exact 1.260,
            S = 256 1.295); 73 effective states at t = 0 (82 used, 46 lost), against 78 (41 lost) with order 1;
          - so the bias that order 2 removes, small next to the noise in any one step (test 17), does add up over
            training with per-element Adam, by about as much as visits and noise do (16 replicates). With the
            visit-count rule order 2 adds +1.5, not significant: as a point estimate about half of the 3.0 points
            left against exact statistics, with an interval from none of that gap to all of it, so this run neither
            shows nor rules out a bias share; the flat S = 4-16 results argue against a large one. Whether the two
            optimisers really respond differently is untested (+5.1 [+2.3, +8.0] against +1.5 [-1.1, +4.1]; the
            intervals overlap and each arm is one run);
          - by eye (gallery job 4202946; Result 13): order 2 repairs some of per-element Adam's broken 1s; the
            order-2 model with the visit-count rule draws nearly the same 100 samples as the one with per-element
            Adam.
      - **Lost states remain.** At t = 0 the visit-count rule leaves 41 (S = 16) and 31 (S = 64) of the 128 states
        below 1e-32 mean occupation, against 2 with exact statistics.
        - Most drop below 1e-8 between slices 20 and 40: states below 1e-8 at slices 20 / 30 / 40 are 1 / 27 / 38 at
          S = 16 (41 at t = 0) and 2 / 26 / 31 at S = 64 (31 at t = 0). The counts never fall at the inspected slices
          (individual states are not tracked).
        - First explanation: one visit moves a state for about one averaging time (1,000 steps, a fifteenth to a
          seventeenth of a slice at slices 20-30), so a state visited more rarely, mean occupation below about
          1/(1,000 B S) (1e-7 at S = 16, 3e-8 at S = 64), still falls behind.
        - **Not supported by its test:** S = 16 with a nominally 10 times longer window (`--occ-beta 0.9999`: 10,000
          steps, about 0.6 of a slice at slices 20-30; as the optimiser state restarts every slice, a visit then
          pushes for about 5,000 steps on average, 5 times as long as at 0.999; pipeline 4202033, evaluation
          4202035, states 4202037) loses more states, not fewer: 50 below 1e-32 at t = 0 against 41, 69 effective
          against 78 (and 26 + 8 below 1e-8 at slice 30, against 27). Clean yield 81.6% against 81.4% (+0.2, [-2.0,
          +2.5]).
        - A possible reason (untested): a long window also lags a falling occupation, so a state that is losing its
          data is divided by a too-large occupation and slows down just when it should keep up. Which mechanism
          loses the remaining states is open.
      - **S = 8 does as well as S = 16 with the visit-count rule** (pipeline 4202034, evaluation 4202035, coverage
        4202036, states 4202037):
        - clean yield 81.6%: against S = 16 with the visit-count rule +0.3 ([-2.1, +2.7]); against S = 16 with
          per-element Adam +6.0 ([+2.9, +9.2]), at the same 16 reads per configuration and step; against exact
          statistics -2.7 ([-5.6, +0.1], p = 0.06);
        - far (50/50) 15.8%; precision of the 1s 0.88, recall of real 0s 0.78; median nearest-image distance 1.385
          (S = 16: 1.359);
        - 82 effective states at t = 0 (94 used, 34 lost), against 78 for S = 16. It crowds harder early (86
          effective at slice 10, 41 at slice 20, against 102 and 49) and recovers more later (one run each).
        - So with the visit-count rule the clean yield is flat from S = 8 to 16 (81.6%, 81.4%; 16 replicates of
          S = 16 with per-element Adam: 80.9%) and a little higher at S = 64 (83.4%). The 4x larger O(1/S²) bias and
          twice the noise variance (√2 the noise) of S = 8 cost nothing measurable in clean yield, which argues
          against bias as the main remaining cost at S = 16 with this rule; the second-order jackknife is
          inconclusive (+1.5, not significant; above).
      - **S = 4 does as well as S = 8 and 16 in clean yield; S = 2 does not** (pipelines 4202870/71, paired
        evaluations 4202872 and 4203290, coverage 4202873, states 4202874; one run each):
        - S = 4 (8 reads per configuration and step): clean yield 82.2%; against S = 8 with the rule +0.5 ([-2.1,
          +3.3]); against S = 16 with the rule +0.8 ([-2.1, +3.7]); against S = 16 with per-element Adam +6.6
          ([+3.4, +9.8], p < 2e-4) at half its reads; against exact statistics with per-element Adam -2.2 ([-5.2,
          +0.8], p = 0.15), with the rule -3.5 ([-6.4, -0.5], p = 0.02). Far (50/50) 16.2%; blends 2.0%; precision
          of the 1s 0.86; recall of real 0s 0.74 (S = 8: 0.78; exact statistics with per-element Adam 0.77;
          subsampling spread ±0.04);
        - S = 2 (4 reads): clean yield 73.7%, 8.5 points below S = 4 ([-11.8, -5.2]) and below exact statistics by
          10.7 ([-14.2, -7.2]); indistinguishable from S = 16 with per-element Adam (-1.9, [-5.3, +1.4]) at a
          quarter of its reads. Far (50/50) 24.0%; blends 5.0% (S = 4: 2.0%, McNemar p = 0.008); clean yield of the
          0s 52.3% against 68.4% for S = 4 (1s: 95.0% vs 96.0%); precision of the 1s 0.81, recall of real 0s 0.71;
        - the median nearest-image distance grows steadily as S falls, also where the clean yield is flat: exact
          statistics with per-element Adam 1.260, then with the rule S = 64 1.308, 16 1.359, 8 1.385, 4 1.424, 2
          1.558 (real images 1.165). So the samples drift away from the training images with fewer draws (copies
          stay at 0-0.6%);
        - states: S = 4 and 2 lose fewer states than S = 8-16, though S = 2 loses more than S = 4 (one run each).
          At t = 0, S = 4 keeps 95 effective states (111 used as argmax, 17 below 1e-32), S = 2 84 (102 used, 26
          lost), against 82 and 78 at S = 8 and 16 (34 and 41 lost). Both crowd earlier (45 and 28 effective states
          at slice 10, against 86 at S = 8 and 102 at S = 16) and recover more later; most of their lost states drop
          out after slice 40 (only 7 and 8 are below 1e-8 there). That argues further against "too few visits" as
          the cause of the states lost at S = 8-16;
        - allocation (states used as argmax at t = 0, 0s / 1s, all with the visit-count rule): exact statistics
          65 / 61, S = 64 50 / 47, S = 16 42 / 45, S = 8 42 / 52, S = 4 40 / 71, S = 2 29 / 73. The 0s' share
          falls with S (52, 52, 48, 45, 36, 28%), mostly because the 1s take more states; the 0s' count is flat from
          S = 16 to 4 and drops only at S = 2. At S = 2 the 0s, the class with more variety, get under a third of
          the states, and the 0s are what fails (one run; whether one causes the other is untested);
        - so with the visit-count rule the clean yield is flat from S = 4 to 16 (82.2, 81.6, 81.4%) and breaks
          between 4 and 2. S = 4 has 16 times the O(1/S²) bias and 4 times the noise variance of S = 16 at no
          measurable cost in clean yield (its samples do lie farther from the training set), so the 3.5-4.3 points
          that S = 4-16 leave against the rule's exact-statistics run are probably neither mainly bias nor mainly
          noise. What they are is open.
      - **Visit counts from the same draws** (`--occ-same-draws`): the rule as above, but each state's visits are
        counted among the S draws the gradient statistics already use (`latent_model.loss_draw_occupation`), so it
        reads S hidden states per configuration and step instead of 2S. Test 18 (job 4202937) passed: the counts
        equal the batch mean of the loss's own sampled occupations exactly (three keys), the rule without the flag
        is unchanged to all digits, invalid combinations are refused, and Potts 128 + both with S = 8 tracks the
        fresh-draw run over slices 0-2.
        - **S = 8 with same-draw counts does as well as with fresh counts** (pipeline 4202938; paired evaluation
          4203199, coverage 4203200, states 4203201; one run): clean yield 81.7% against 81.6% (+0.1, [-2.7,
          +2.8]); against S = 16 with per-element Adam, which reads twice as much, +6.1 ([+2.8, +9.4], p = 2e-4);
          against S = 4 with fresh counts, the same 8 reads, -0.5 ([-3.1, +2.2]); against exact statistics -2.7
          ([-5.6, +0.2], p = 0.07). Far (50/50) 16.7%; precision of the 1s 0.87, recall of real 0s 0.76; median
          nearest-image distance 1.365 (fresh counts 1.385). 84 effective states at t = 0 (98 used, 30 lost; 0s /
          1s 42 / 56), against 82 (94 used, 34 lost; 42 / 52) with fresh counts.
        - So taking the numerator and the denominator from the same draws costs nothing measurable in the trained
          model here (one run; the bias of the step itself was not measured), and the rule then reads no more
          hidden states than per-element Adam: S per configuration and step. With 8 reads it gives 81.7%, where
          per-element Adam with twice the reads gives 75.6%.
        - **S = 16 with same-draw counts** (pipeline 4202939; evaluation 4202940, coverage 4202941, states 4202942):
          83.0% against 81.4% with fresh counts (+1.6, [-1.0, +4.3]) and 84.3% with exact statistics (-1.4, [-4.2,
          +1.4]); 74 effective states at t = 0 (83 used, 45 lost).
        - **Equal reads, 8 per configuration and step** (pipeline 4203228; evaluation 4203232, coverage 4203233,
          states 4203234): per-element Adam with S = 8 gives 68.4%, the rule with same-draw counts 81.7% (+13.3,
          [+9.9, +16.8]; far (50/50) 29.6% vs 16.7%). Per-element Adam loses almost every state: 14 effective at
          t = 0 (16 used, 112 lost), against 84 (98 used, 30 lost). With per-element Adam, halving the draws from
          16 to 8 costs 7.2 points ([-10.4, -4.2]); with the rule it costs nothing measurable.
        - **Second training seed** (`--seed 2`: new k-means prototypes and minibatch stream; pipelines 4203229/30,
          evaluation 4203235, coverage 4203236, states 4203237): S = 8 with the rule and same-draw counts 82.8%
          against S = 16 with per-element Adam 75.8% (+7.1, [+3.9, +10.1]; seed 1: +6.1 [+2.8, +9.4]). Each arm
          moves little between seeds (+1.1 [-1.4, +3.7] and +0.2 [-2.7, +3.0]); 86 against 33 effective states at
          t = 0. Against exact training at seed 2 (86.1%) the rule is -3.2 ([-6.2, -0.2]), as at seed 1. So the main
          comparison replicates, and seed-to-seed variation is about a point here.
      - Also running: at 28x28, S = 16 with the visit-count rule against per-element Adam (4202038/39).
      - By eye (gallery job 4201995, `14x14_0-1_local_training_optimisers_Potts128/`; Result 13): plain steps and S = 16 with
        per-element Adam break strokes and put a detached blob on top of slanted 1s; the visit-count rule with S = 16
        mostly does not.

**3. The CD-1 (short-release) rule: noisy per step, but it trains the hidden units as well as the exact gradient**
(`cd1_check.py`: jobs 4197145 at 14x14, 4197146 at 28x28; training: evaluation 4197251, coverage 4197252).
- What it needs per sample: the Hebbian products p_m x_i, p_m, p_m x_i² (and m_i x_i, m_i) at the data and after a
  release of length dt.
- Bias:
  - none detected for dt ≤ 3e-3 at all slices checked;
  - at dt = 1e-2 it appears at the last slice (t = 0);
  - the test averages 32 draws of 16 releases, so at dt = 3e-3 it resolves only biases larger than about 0.5 |g|.
- Noise:
  - one release per sample gives errors of 6-42 |g| at dt = 3e-3 (14x14); 16 releases bring them to 1.5-10;
  - matching one batch's own noise (about 1 |g|) would take about 600-1,700 releases per sample at most slices,
    fewer at the last.
  - That is the O(√dt) signal-to-noise of DESIGN §2. It is cheap for fast hardware, but 10-100 times more samples
    than force matching needs (S = 16-64).
- **Training** (14x14, Potts 128 + both; CD-1 for all hidden parameters with 64 releases per sample, dt = 3e-3;
  visible parameters by their usual rule):

| | exact | CD-1, 64 releases | K = 0 | real images |
|---|---|---|---|---|
| far | 13.2% | 13.0% | 48.4% | 10.0% |
| clean yield, 50/50 | 84.3% | 85.3% | 38.6% | 85.7% |
| flagged blends | 2.6% | 1.6% | 32.6% | 5.0% |
| recall of real 0s / 1s | 0.77 / 0.87 | 0.78 / 0.87 | 0.48 / 0.79 | 0.89 / 0.90 |
| precision of the 0s / 1s | 0.95 / 0.88 | 0.94 / 0.87 | 0.59 / 0.35 | 0.88 / 0.88 |

  - **CD-1 training is indistinguishable from exact training:** clean yield (50/50) +0.9 points ([-2.1, +3.9],
    p = 0.55), far (50/50) -0.3 ([-3.0, +2.4]), blends 1.6% vs 2.6% (p = 0.3); same coverage.
  - Its hidden gradient is about 1.5-4 times noisier than a batch gradient, and Adam absorbs that.
  - Idealisation: the Hebbian products use the exact hidden averages p_m, at the data and after each release (as the
    exact force-matching run does). A device would read sampled hidden states instead.
- **With sampled hidden states CD-1 drowns in noise** (`cd1_check.py --cd1-samples`, job 4198622; trained Potts 128 +
  both, dt = 3e-3, 32 draws, slices 1, 25, 50, 75 and 99).
  - Setup: R·S hidden reads at the data and S after each of the R releases (`--cd1-hidden-samples S`), against exact
    averages.

| error of one estimate (in \|g\|) / cosine with the exact gradient | W | b | Q | wl | bl |
|---|---|---|---|---|---|
| exact averages, R = 64 | 1.4-4.3 / 0.25-0.58 | 0.8-4.9 / 0.20-0.78 | 1.6-4.2 / 0.26-0.53 | 3.0-5.0 / 0.23-0.30 | 1.8-5.3 / 0.19-0.48 |
| sampled, R = 64, S = 1 | 11-813 / ≤ 0.07 | 17-2,730 / ≤ 0.02 | 15-584 / ≤ 0.03 | 108-536 / ≤ 0.05 | 181-5,890 / ≤ 0.01 |
| minibatch (two exact batches differ by) | 1.4-1.5 | 1.3-1.8 | 1.3-1.5 | 1.2-1.8 | 1.4-1.7 |

  - The estimate stays unbiased: it is linear in the sampled states, and the mean of 32 draws falls as 1/√32. Its
    error falls as 1/√(number of reads): four reads per release halve it.
  - Why: CD-1 multiplies the difference of two hidden readouts, at the data and after a release, by
    2/(T² dt) ≈ 1.7·10⁶. With exact averages that difference is O(dt). With independent samples it keeps their full
    sampling noise.
  - Reaching the noise of exact-average CD-1 (which trains as well as exact force matching) would take about
    4·10³-2·10⁷ hidden reads per configuration for the Potts unit, and up to 9·10⁷ for the per-pixel units.
  - Force matching reads the hidden units at one clamped configuration only, so its sampling noise is not
    amplified: 256 reads suffice (point 2).
  - The training run was therefore not done. **For on-device training with measured hidden statistics, force
    matching is the rule.** CD-1 needs (nearly) exact hidden averages, e.g. hidden units whose occupation
    probabilities can be read directly, or readouts that stay correlated across the release (not modelled).
- **Where measured statistics cost force matching, and CD-1 does not** (exact validation loss per slice, fixed
  validation batch, against exact training; positive = worse):

| slice (t) | 25 (3.74) | 45 (2.73) | 65 (1.72) | 85 (0.71) |
|---|---|---|---|---|
| S = 16 | +0.9 | +33.2 | +87.1 | +58.8 |
| S = 16, jackknife | +0.4 | +16.8 | +43.0 | +11.8 |
| S = 64 | +0.4 | +16.1 | +44.9 | +15.6 |
| S = 64, jackknife | +0.3 | +6.5 | +13.8 | -15.8 |
| S = 256, jackknife | +0.1 | +3.5 | +11.0 | +1.7 |
| CD-1, 64 releases | +0.6 | +4.3 | +6.8 | -14.6 |

  - The measured-statistics runs fall behind between t ≈ 3.7 and 1.7. There the hidden states are still uncommitted
    (mean max p 0.22 at slice 25, 0.59 at 45, 0.86 at 65; S = 64 jackknife run) and split into prototypes (point 5),
    so the hidden samples vary most.
  - The jackknife roughly halves the gap; S = 64 with the jackknife leaves about a sixth of it, S = 256 with the
    jackknife about an eighth.
  - CD-1, whose hidden statistics are exact here, stays within 9 units of exact training down to t ≈ 1.4 and within
    15 down to t ≈ 0.45 (mostly below exact training there).
  - From t ≈ 0.4 on the runs drift apart, by more than a thousand units at t = 0.
    - The force-matching runs end below exact training (-586 to -1058 at t = 0).
    - CD-1 falls steadily behind: +17 at t = 0.40, +225 at 0.20, +1654 at 0. That is plausibly the O(dt) bias of its
      fixed release length where the hidden fields are steepest (field scale 1 down to t ≈ 1.2, 3.4 at t = 0.45, 22
      at t = 0).
    - Neither shows a matching difference in the samples.
    - These are shifts of the typical validation sample, not of a few outliers: the medians of the per-sample
      losses move almost as much as the means (CD-1 at t ≤ 0.2: +200 to +1463 against +225 to +1654).
  - The validation loss is only a rough guide to the samples: S = 256 with the jackknife trails exact training by
    11 units at slice 65, about as much as S = 64 with the jackknife (14), yet its samples are as good as exact
    training's and those of S = 64 are not.

**4. Do the hidden couplings need to be learned at all?**
- Setup: 14x14, Potts 128 + per-pixel units, no spring (Q has no closed-form reference). Evaluations 4197128 (`base`)
  and 4197181 (`ext`); coverage 4197129 and 4197182.
- The hidden couplings are frozen at their closed-form reference in every slice (`--hidden-fixed`: k-means
  prototypes and pooled covariance, as at initialisation); only the visible couplings are trained. This is compared
  with the same model whose hidden couplings are trained by force matching.

| frozen vs trained | `base` | `ext` |
|---|---|---|
| far, 50/50 | +2.7 [-0.4, +5.9] | +0.8 [-2.3, +3.8] |
| clean yield, 50/50 | -1.3 [-4.7, +2.0] | +0.8 [-2.4, +3.9] |
| flagged blends | 1.6% vs 5.2% (p = 3e-4) | 1.4% vs 4.0% (p = 0.007) |
| recall of real 0s / 1s | 0.82 / 0.90 vs 0.72 / 0.87 | 0.83 / 0.88 vs 0.74 / 0.86 |
| precision of the 0s / 1s | 0.90 / 0.77 vs 0.94 / 0.87 | 0.92 / 0.77 vs 0.93 / 0.85 |

(real images: recall 0.89 / 0.90, precision 0.88 / 0.88)

- Frozen couplings are as good on far and clean yield, give fewer blends and more variety, and less precision for
  the 1s. The spread of the 0s is 4.04 vs 3.84 (real 4.20).
- Training moves the couplings far from the reference at late slices (at t = 0: cos 0.20, |W| 17.5 vs 12.4). It
  makes the samples more typical (higher precision, lower recall) without improving far or clean yield.
- **What training does to the couplings** (`inspect_states.py`, job 4197197; figures in
  `out/latent_physical_SGM/states/4197197/`).
  - Frozen couplings are noisy, high-frequency patterns: T S⁻¹ μ_m, with the inverse pooled covariance amplifying
    small within-class shape differences. Occupations are even (112 effective states of 128) and max p is 0.95 at
    t = 0.
  - Trained couplings become clean class-contrast templates. Every 1-state gets a strong negative ring where a 0
    would put ink, and every 0-state a positive ring. They are larger, alike across the states of a class, and
    commit harder (max p 0.996, 104 effective states).
  - So force matching spends the couplings on separating the classes (the modes) rather than on within-class shape,
    which makes the samples more typical.
  - In every trained model the states stay class-pure (≥ 0.997) and nearly all are used. At 28x28 that is 112 of
    128 and 30 of 32 at t = 0.
  - With the spring, Q_m gives each state its own stiffness pattern. A 1-state pins the pixels where a 0 would
    have ink (Q < 0); a 0-state softens the inside of its ring. The tilted copies thus differ in shape, not only in
    position (DESIGN §1.1).
- **Reading:** at 14x14 the benefit of the hidden units comes from their structure, a mixture of prototype-centred
  tilted copies of the visible model.
  - Couplings set once from data statistics deliver it, with the visible couplings adapting to them through the
    paper's local rule.
  - Learning the hidden couplings by force matching trades variety for typicality.
  - The trained spring adds -3.8 far points on top (50/50, `ext` [-6.2, -1.4]; Potts 128 + both vs + per-pixel
    units only).
- **28x28: the same, more strongly** (`P128_ema_local_fixed` vs `P128_ema_local`, Potts 128 + per-pixel units, no
  spring; 500 paired `base` trajectories; evaluation 4197195, coverage 4197196):

| | frozen | trained | Potts 128 + both (trained spring) | real images |
|---|---|---|---|---|
| far, 50/50 | 16.8% | 14.8% | 9.7% | 13.2% |
| clean yield, 50/50 | 80.2% | 80.1% | 86.3% | 83.3% |
| flagged blends | 4.6% | 7.4% | 4.8% | 5.0% |
| recall of real 0s / 1s | 0.59 / 0.86 | 0.38 / 0.74 | 0.46 / 0.79 | 0.89 / 0.89 |
| precision of the 0s / 1s | 0.98 / 0.85 | 0.99 / 0.94 | 0.99 / 0.95 | 0.87 / 0.88 |
| spread of the 0s | 8.35 | 7.85 | 8.29 | 9.77 |

  - Frozen against trained: clean yield +0.0 ([-3.5, +3.5]); far (50/50) +2.0 ([-0.9, +5.0], p = 0.19); blends 4.6% vs
    7.4% (p = 0.029).
  - Frozen couplings give much more variety: recall of the real 0s 0.59 vs 0.38, the highest of any 28x28 model with
    sextic visible energy (Potts 256 + both: 0.55; power 4: 0.61). Their precision is a little lower.
  - The trained spring adds clean yield on top (Potts 128 + both against trained Potts 128 + per-pixel units: +6.1,
    [+3.4, +8.9]; far (50/50) -5.1, [-7.5, -2.7]), at lower recall than the frozen couplings.
  - Against the paper both are far better (frozen: clean yield +48.3, [+43.7, +52.7]; blends 4.6% vs 43.2%).
- **Reading for hardware:** at both sizes, hidden couplings set once from data statistics (k-means prototypes and the
  pooled covariance) give the main benefit, with only the visible couplings trained by the paper's local rule.
  Training the hidden couplings by force matching makes the samples more typical without making them cleaner. The
  learned spring (no closed-form reference) is what training adds.

**5. Force matching finds the hidden structure from a random start** (14x14, Potts 128 + both).
- Setup: `--proto-init random`, so W starts as Gaussian noise of the reference's size at t = 5 and b = 0. No k-means
  and no labels; the per-pixel units start from their label-free reference as always.
- Result (evaluation 4197157, coverage 4197158), against the label-free k-means start:
  - far 12.4% vs 10.2% (p = 0.13); far (50/50) +2.3 ([-0.3, +4.9]); clean yield (50/50) -2.1 ([-5.0, +0.8]);
  - blends 2.4% vs 3.2%;
  - recall of real 0s / 1s 0.77 / 0.87 vs 0.79 / 0.85;
  - against K = 0: blends 2.4% vs 32.6%, clean yield +46.4 points.
- How the states form (`inspect_states.py`, job 4197234):
  - slice 1 (t = 4.95): occupation uniform, class purity 0.58 (chance);
  - slice 25 (t = 3.74): a few states have captured the coarse modes (29 effective states, purity 0.86). Their W_m
    are already clean 0-rings and 1-strokes of various tilts;
  - slice 50 (t = 2.47): they have split into class-pure prototypes (92 effective states, purity 0.999), as with the
    k-means start (98, 0.998);
  - t = 0: 94 effective states, purity 1.000.
- **Reading:** the noise schedule acts like the temperature in deterministic annealing. At high noise the data
  occupy a few states; as the noise falls, the states split into prototypes. The local force-matching rule therefore
  needs no offline initialisation.

The all-pairs control (is the gain a matter of parameter count?) is in Result 8: it is not.

## Result 11 (2026-11-08): other digit pairs (14x14)

- Data: `make_digit_data.py` builds any digit set with the paper's preprocessing; for 0/1, 0/1/2 and 0-3 it
  reproduces the paper's files exactly (job 4197174).
- Problems `mnist14_d<digits>` are pooled 2x2 like `mnist14`. `mnist14_d01` reproduces the `mnist14` training bit
  for bit (job 4197177).
- Evaluation as in Result 8; class 0 is the smaller digit. The paper's model is K = 0.

| (`base`, 500 paired trajectories) | 0/1: K = 0 | 0/1: Potts 128 + both | 1/4: K = 0 | 1/4: Potts 128 + both | 1/4: real | 4/9: K = 0 | 4/9: Potts 128 + both | 4/9: real |
|---|---|---|---|---|---|---|---|---|
| flagged blends | 32.6% | 2.6% | 22.6% | 4.0% | 5.0% | 4.4% | 1.8% | 5.0% |
| far | 48.4% | 13.2% | 21.2% | 8.4% | 10.0% | 39.2% | 17.2% | 10.0% |
| clean yield, 50/50 | 38.6% | 84.3% | 58.6% | 88.1% | 85.7% | 59.8% | 81.4% | 85.3% |
| share of the second digit | 55% | 52% | 39% | 53% | 49% | 38% | 48% | 49% |
| recall of real images (class 0 / 1) | 0.48 / 0.79 | 0.79 / 0.88 | 0.81 / 0.43 | 0.88 / 0.72 | 0.91 / 0.88 | 0.51 / 0.73 | 0.73 / 0.84 | 0.88 / 0.88 |

(0/1 real images: blends 5.0%, far 10.0%, clean yield 85.7%, recall 0.89 / 0.90.)

- **1 and 4** (evaluation 4197189, coverage 4197190):
  - The paper's model fails less than on 0/1. The classes share a vertical stroke, so a superposition is less far
    from both.
  - It still produces 22.6% blends and under-produces 4s.
  - Potts 128 + both fixes both:
    - blends 4.0% (p = 1e-21);
    - clean yield (50/50) +29.4 points ([+24.7, +34.1]), at the real images' level;
    - far below the real images' rate;
    - the 4s' recall from 0.43 to 0.72.
- **4 and 9** (similar shapes; evaluation 4197278, coverage 4197279):
  - The classes overlap: the class-score 99% bounds cross (0.70 > 0.33), so nothing is "between the classes".
  - A superposition of a 4 and a 9 looks like a real digit, so the blend test does not flag the paper's model
    (4.4%, below the 5% of real images).
  - The paper's model fails here by malformed shapes instead: far 39.2% (real 10.0%), and it under-produces 9s.
  - Potts 128 + both:
    - far 17.2% (p = 3e-23); clean yield (50/50) +21.6 points ([+17.2, +26.1]);
    - share of 9s 48% (real 49%);
    - recall of the 4s 0.51 -> 0.73 and of the 9s 0.73 -> 0.84; precision of the 9s 0.62 -> 0.90.
- **Reading:** on all three pairs the hidden units lift the clean yield to 81-88% (real 85-86%), whatever the
  visible model's failure mode: superposition (0/1, 1/4) or malformed shapes (4/9). Its weakest point remains the
  recall of the more varied class (0s, 4s).

## Result 12 (2026-11-08): three digits, 0/1/2 (14x14)

- Setup:
  - `mnist14_d012`: the paper's preprocessing, pooled 2x2; 17,769 training images, 5,923 per digit.
  - Potts 128 + both with the label-free initialisation (`--proto-init pooled`: 128 k-means prototypes of all images),
    against K = 0 (the paper's model).
  - 500 paired `base` trajectories; evaluation 4198198, coverage 4198199. With 256 states and local training (S = 64,
    jackknife): paired evaluation 4198227, coverage 4198228; S = 256 with the jackknife: 4199370, 4199371.
- Evaluation: `evaluate_multiclass.py`, the two-class evaluation generalised to C classes.
  - Class: argmax of the LDA discriminants.
  - Blend: the best pair of templates from two different classes explains the sample much better than any single
    template.
  - Far: nearest training image beyond the probes' 90th percentile, so real images are far 10% of the time by
    construction.
  - Standardised rates: the mean over the three classes of the within-class rates (the 50/50 rates for C = 2).

| (`base`, 500 paired trajectories) | real images | K = 0 | Potts 128 + both, label-free | Potts 256 + both, label-free | Potts 128 + both, local: S = 64, jackknife | Potts 128 + both, local: S = 256, jackknife | Potts 128 + both, local: S = 64, jackknife, visit-count rule |
|---|---|---|---|---|---|---|---|
| flagged blends | 5.0% | 22.0% | 2.0% | 3.4% | 3.4% | 2.8% | 3.2% |
| far | 10.0% | 54.6% | 17.4% | 13.8% | 25.0% | 17.8% | 17.2% |
| diverged | 0% | 3.4% | 0% | 0% | 0% | 0% | 0% |
| share of 0s / 1s / 2s | 33 / 34 / 33% | 32 / 44 / 25% | 33 / 34 / 33% | 32 / 36 / 32% | 33 / 34 / 33% | 33 / 35 / 32% | 33 / 35 / 31% |
| clean yield, standardised | 86.5% | 34.4% | 81.1% | 83.4% | 73.1% | 80.2% | 80.1% |
| clean yield within 0s / 1s / 2s | 89.5 / 96.7 / 73.2% | 14.5 / 65.4 / 23.3% | 87.7 / 99.4 / 56.3% | 86.9 / 96.6 / 66.7% | 78.3 / 99.4 / 41.5% | 87.3 / 96.6 / 56.6% | 86.7 / 98.9 / 54.8% |
| far within 0s / 1s / 2s | 8.1 / 0.3 / 22.0% | 83.6 / 20.4 / 71.7% | 11.7 / 0.6 / 40.1% | 10.6 / 0.6 / 31.5% | 19.9 / 0.6 / 55.5% | 12.1 / 2.3 / 40.9% | 12.0 / 0.6 / 41.4% |
| recall of real 0s / 1s / 2s | 0.90 / 0.89 / 0.88 | 0.57 / 0.81 / (too few 2s) | 0.79 / 0.83 / 0.63 | 0.84 / 0.88 / 0.72 | 0.82 / 0.83 / 0.66 | 0.80 / 0.86 / 0.62 | 0.84 / 0.85 / 0.66 |
| precision of the 0s / 1s / 2s | 0.89 / 0.89 / 0.88 | 0.50 / 0.30 / (too few 2s) | 0.88 / 0.74 / 0.95 | 0.92 / 0.80 / 0.94 | 0.83 / 0.70 / 0.93 | 0.87 / 0.72 / 0.94 | 0.88 / 0.72 / 0.94 |

Recall and precision: k-NN coverage at matched n = 150 per class, mean over 20 subsamples (subsampling spread about
±0.01-0.04; ±0.10-0.13 for K = 0's precision).

- **The hidden units work for three classes as they did for two, without labels.** Against K = 0:
  - blends 2.0% vs 22.0% (p = 4e-24); far 17.4% vs 54.6% (p = 1e-36); diverged 0% vs 3.4% (p = 2e-5);
  - clean yield (standardised) +46.7 points ([+42.0, +51.5]), against real images' 86.5%;
  - the class shares become even (K = 0 under-produces 2s: 25%).
- **The 2s are the weak class.**
  - The 2s are the most varied digit: even real 2s are far 22% of the time.
  - The model's 2s are far 40% of the time, and its samples cover 63% of the real 2s' variety.
  - Their precision is high (0.95), so the model draws typical 2s and misses the unusual ones. That is the same
    under-dispersion as for the 28x28 0s.
  - They are not short of states (`inspect_states.py`, job 4198240; state = majority class of the samples it wins):
    - at t = 0, 55 states carry 0s, 19 carry 1s and 54 carry 2s;
    - 99.1% of real 2s are won by a 2-state;
    - the less pure states are 2-states that also win some 1s: 3.8% of real 1s go to a non-1 state.
    - For comparison, two classes: 93 / 34 states for 0 / 1 at 14x14 (label-free start); 63 / 62 at 28x28 with 128
      states and 122 / 119 with 256 (per-class start). Purity ≥ 0.999 in all of these.
  - So the 2s get their share of states. What they lack is variety within each state.
- **256 states beat 128, and the gain is variety** (paired, 4198227):
  - far 13.8% vs 17.4% (McNemar p = 0.027); far, standardised, -3.2 points ([-6.2, -0.2], p = 0.035);
  - clean yield (standardised) 83.4% vs 81.1%, +2.2 points ([-1.1, +5.6], p = 0.2); blends 3.4% vs 2.0% (p = 0.19);
  - mostly the 2s: clean 56.3% -> 66.7% (real 73.2%), far 40.1% -> 31.5% (real 22.0%). The per-class differences
    are not tested separately.
  - Recall rises in all three classes: 0.79 -> 0.84 (0s), 0.83 -> 0.88 (1s), 0.63 -> 0.72 (2s), at unchanged or higher
    precision. So the extra states add variety, not only fidelity, as at 28x28 (Result 7: recall of the 0s 0.46 ->
    0.55).
  - The states split as before, about twice as many per class: 104 / 37 / 114 states carry 0s / 1s / 2s at t = 0
    (255 of 256 states win some samples; job 4198803), purity 0.994; 99.3% of the real 2s are won by a 2-state.
- **Local training with measured statistics (S = 64, jackknife) seems to cost more on three classes than on two**
  (8.1 vs 3.2 points; that difference is not tested, the confidence intervals overlap, and each run is one training
  seed; paired within 0/1/2, 4198227):
  - against exact training: clean yield (standardised) 73.1% vs 81.1%, -8.1 points ([-11.5, -4.6]); far 25.0% vs
    17.4% (p = 8e-6); blends 3.4% vs 2.0% (p = 0.17);
  - against K = 0 it still gains +38.7 points ([+33.8, +43.6]);
  - the cost is in the 0s and 2s (clean 78.3% vs 87.7% and 41.5% vs 56.3%; far 19.9% vs 11.7% and 55.5% vs 40.1%);
    the 1s are unaffected;
  - recall is unchanged (0.82 / 0.83 / 0.66 vs 0.79 / 0.83 / 0.63) and precision a little lower (0.83 / 0.70 / 0.93 vs
    0.88 / 0.74 / 0.95). So finite S costs fidelity, not variety.
  - It keeps 78 of the 128 states at t = 0 (32 / 17 / 29 for 0s / 1s / 2s, against 55 / 19 / 54 with exact training;
    job 4199364; 68 vs 102 effective states). They are lost in the same consolidation phase as on two classes
    (effective states 51 at slice 20 and 48 at slice 25, against 62 and 70 with exact training; job 4199394; Result
    10, "What finite S costs is states").
  - With two classes S = 64 cost 3.2 points ([-6.2, -0.4]) and S = 256 cost nothing (Result 10).
- **With S = 256 and the jackknife, local training matches exact training on three classes too** (pipeline 4199369;
  paired evaluation 4199370, coverage 4199371, states 4199372):
  - against exact training: clean yield (standardised) 80.2% vs 81.1%, -1.0 points ([-3.9, +1.9], p = 0.51); far
    17.8% vs 17.4% (p = 0.88); blends 2.8% vs 2.0% (p = 0.48);
  - against S = 64: +7.1 points ([+4.1, +10.1]); far 17.8% vs 25.0% (p = 7e-7). So the 8-point cost of S = 64
    closes at S = 256, as the 3-point cost did on two classes;
  - against K = 0: +45.8 points ([+40.8, +50.7]);
  - per class: clean 87.3 / 96.6 / 56.6% (exact 87.7 / 99.4 / 56.3%); recall of real 0s / 1s / 2s 0.80 / 0.86 / 0.62
    (exact 0.79 / 0.83 / 0.63);
  - it keeps 100 of the 128 states at t = 0, 41 / 18 / 41 for 0s / 1s / 2s (exact: 128, 55 / 19 / 54; S = 64: 78),
    83 effective (exact 102, S = 64 68). 28 states sink below 1e-32 mean occupation. Again the lost states cost no
    measurable clean yield at S = 256, as on two classes (96 of 128 kept there).
- **With S = 64, the visit-count rule is indistinguishable from exact training on three classes**
  (`--hidden-norm occupancy`, Result 10; pipeline 4201365, paired evaluation 4201369, coverage 4201370, states
  4201371; one run; a cost up to about 4 points is not excluded):
  - against per-element Adam with S = 64: clean yield (standardised) 80.1% vs 73.1%, +7.1 points ([+4.1, +10.1],
    p < 2e-4); far 17.2% vs 25.0% (McNemar p = 4e-8);
  - against exact training: -1.0 ([-4.2, +2.2], p = 0.53); far 17.2% vs 17.4%. Against S = 256 with per-element
    Adam: -0.0 ([-2.9, +2.9]);
  - per class: clean 86.7 / 98.9 / 54.8% (exact 87.7 / 99.4 / 56.3%; S = 64 with per-element Adam 78.3 / 99.4 /
    41.5%); recall of real 0s / 1s / 2s 0.84 / 0.85 / 0.66; median nearest-image distance 1.816 (exact 1.764,
    S = 256 1.818, S = 64 with per-element Adam 1.893);
  - it keeps 100 of the 128 states at t = 0, 40 / 19 / 41 for 0s / 1s / 2s, 89 effective, like S = 256 with
    per-element Adam (100, 41 / 18 / 41, 83 effective). As many states as there, 28, sink below 1e-32 mean
    occupation; they were lost between slices 20 and 40, during or just after the consolidation (1 below 1e-8 at
    slice 20, 28 at slice 40).
  - So on three classes the visit-count rule gives S = 64 what per-element Adam needs S = 256 for, with half the
    hidden reads per configuration and step (64 for the statistics plus 64 fresh draws for the visit counts,
    against 256); a quarter if the visit counts could reuse the same reads (untested).

## Result 13 (2026-11-08): what the samples look like

Image gallery (`make_gallery.py`, job 4199761): `out/latent_physical_SGM/gallery/<problem>/`, with `INDEX.md`
there. Per problem: `real.png` (100 random real images), `<model>.png` (samples 0-99 of each model, unsorted, in
trajectory order), `compare.png` (one row per model; column j is trajectory j, the same start and noise in every
model), and `nearest_<model>.png` (samples above their nearest training images). Read by eye, not scored:

- **Paper model against hidden units (28x28).** About a third of the paper model's samples look like acceptable
  digits. The rest show fragmented 1s with a grey halo, smeared or holed 0s, "1 inside a 0" ghosts, and 4 of 100
  diverged. With hidden units essentially every trajectory ends as a recognisable 0 or 1.
- **The trajectory decides the digit.** In `compare.png` a given trajectory gives the same digit, often with a
  similar slant and width, in every hidden-unit model. The models differ in finish, not in what they draw.
- **Texture is the most visible gap to real MNIST.** Real strokes are solid white with sharp edges on a clean
  background. Generated strokes are speckled inside, grey at the edges, with faint background noise. The samples look
  like slightly noisy, softened real digits. The metrics miss this: the roughness measure is close to the real one.
- **Typicality, as in the coverage numbers.** Generated 0s are mostly fat, round rings. Real 0s vary far more: open
  loops, filled blobs, hairline and very thick strokes, serif 1s. Frozen couplings give visibly more varied 0s, but
  the grainiest texture and a few artefacts (blocky stroke ends, a broken "!"-shaped 1).
- **Local training is invisible at a glance (14x14 0/1).** Exact training, force matching with S = 256 or 64
  (jackknife), CD-1 and the random start draw nearly the same image for each trajectory. Plain S = 16 adds small
  stray fragments.
- **Shared failures (14x14 0/1).** The same few trajectories fail in every hidden-unit model: a thin, broken 0 (7th
  column of `compare.png`) and a small 0 with a stroke inside, a residual 0/1 blend (15th column; CD-1 turns this one
  into a clean 1).
- **Three digits: the failures are malformed 2s, not blends.** The 0s and 1s look as good as with two digits. Many
  2s are blobby, with a filled-in loop, heavy speckle or extra strokes; a few samples are unrecognisable fragments.
  By eye about 8-10 of the 100 Potts 256 samples are clearly bad, mostly 2s (clean 2s: 67%).
- **Not copies.** Each sample resembles its nearest training image in slant and width, at distances 1.9-4.4 (28x28;
  a near-copy is below 1.42, and real images lie at a median 3.3 from their nearest neighbour). The 1s sit closer
  to the training set than the 0s.
- **Later additions (gallery job 4200685, 23:24):**
  - 28x28, local training with S = 64 (`local_S64jk.png`): indistinguishable at a glance from exact training. The
    same digit on each trajectory, the same speckled strokes. About 4 of 100 are flawed: a 1 crossed by a bar, a
    smudged 1, a smudged filled 0, a doubled 0 loop.
  - 28x28, local training with S = 256 (`local_S256jk.png`, gallery job 4199767, 03:49): the same picture. About
    3-4 of 100 flawed (a kinked 1 with a flare at the bottom, row 8 column 5; a speckled, filled 0, row 10 column 7;
    a 0 with a doubled loop, row 9 column 4). One serif 1 (row 10 column 1); `real.png` has 3-4 in 100. Against `real.png` the strokes are frayed and speckled at the edges where real strokes are smooth and
    solid, and very thick, filled 0s are still rare.
  - 28x28, 512 states with power 4 (`Potts512_both_power4.png`): nearly all clean. The 0s vary more in width,
    slant and stroke weight than with 256 states, consistent with the higher recall (0.68). Still missing against
    `real.png`: serif 1s, and solid, very thick 0s; its one thick 0 comes out as a speckled blob. 3-4 of 100 are
    flawed (that blob, a doubled loop, a 1 with a grey halo, a broken thin 1).
  - 14x14, plain gradient steps (`exact_groupnorm.png`, 22 effective states): by eye nearly the same images as
    per-element Adam with 128 states, trajectory by trajectory. The metric difference (clean 70.4% vs 84.3%) is in
    the 0s' fine shape (far 52% within the 0s), which these 14x14 images barely show; a few 1s get a stray blob or
    bend. (Seen side by side, below, the 1s' flaws are more frequent than "a few".)
- **The hidden-unit optimisers side by side (gallery job 4201995, 03:10; `14x14_0-1_local_training_optimisers_Potts128/`):** 0/1 at
  14x14, Potts 128 + both, the same 500 trajectories in every model. Against `real.png`:
  - In `compare.png` (16 trajectories) most columns give the same digit in the same pose in every model. The
    columns that differ are where the optimisers differ:
    - plain steps (exact or S = 64) and S = 16 with per-element Adam: a broken, 5-like 0 (column 7), slanted 1s
      with a detached blob on top, like an "i" (columns 11, 14), and a 1 broken in the middle (column 16, clear
      for exact plain steps and S = 16 per-element Adam);
    - with S = 64, per-element Adam and one gain per state still draw the blob-topped 1 in column 14, and one gain
      per state and the visit-count rule the 5-like 0 in column 7;
    - the blob-topped 1s (columns 11, 14) are absent only with exact statistics under per-element Adam, one gain
      per state or the visit-count rule, and with the visit-count rule at S = 64 or 16. With S = 16 the visit-count
      rule draws columns 11, 14 and 16 clean; its column 7 is the thin, broken 0 that exact statistics with
      per-element Adam also draw (exact plain steps draw the 5-like one);
    - column 15: a small 0 with a stroke inside (a residual 0/1 blend) in every model except the visit-count rule
      with exact statistics, which draws a wider, flatter 0.
  - 100 samples each, by my count: about 10 flawed with plain steps (exact statistics) and about 10-12 with S = 16
    and per-element Adam, mostly 1s with a detached blob or a gap. About 4-5 with the visit-count rule and exact
    statistics, about 6-7 with S = 16. The S = 16 visit-count model has two bad samples of its own (a scribble, row
    3 column 3; a bent 1, row 6 column 9); at row 2 column 9 it swaps a gapped 1 for a speckled 1 with a heavy head
    on a base; the faint speckle at row 9 column 5 is bad in every model. (A second count by the verifier: about
    13-15 for exact plain steps and S = 16 per-element Adam, 5-7 for the exact visit-count rule, 8-9 for S = 16
    with the visit-count rule: the counts are subjective, the ranking is not.)
  - The 0s look alike in all models at this resolution. Against the real 0s they are more uniform in stroke: real
    0s include very thick, filled ones and thin-stroked wide ones, which the models rarely draw (recall of real 0s
    0.76-0.78; with plain steps 0.69-0.75).
  - So the metric gap of plain steps and few samples, which is largest in the 0s (far), also shows in the 1s by
    eye.
- **The visit-count rule with S = 8, 4 and 2 (gallery job 4203152, 07:52; rows `S8jk_visit_count`,
  `S4jk_visit_count`, `S2jk_visit_count` of `14x14_0-1_local_training_optimisers_Potts128/compare.png`, and their 100-sample grids):**
  - `compare.png`: S = 8 and 4 draw the same digit in the same pose as S = 16 in most columns. S = 4 gives the
    slanted 1 of column 14 a heavy top and turns the small 0 of column 15 into a 1 with a bulge. S = 2 changes more:
    besides these two, it draws 0s where every other model draws a 1 (column 3, a thin, faint 0; column 12, an
    oval 0) and a 1 with a gap near the top (column 11).
  - `S4jk_visit_count.png`: mostly clean 0s and 1s, about as many flawed samples as with S = 16 (by my count 7-9,
    nearly all on trajectories that S = 16 also gets wrong: the scribble at row 3 column 3, the bent 1 at row 6
    column 9, the 0 with a bar through it at row 6 column 7, which every model draws, and the 1s broken into two
    pieces in rows 3 and 9). New with S = 4: a saturated sample, row 9 column 4, a mostly white square with black
    holes.
  - Saturated samples counted over all 500 trajectories of every 14x14 0/1 run (jobs 4203167 and 4203221; ink
    fraction, pixels above 0.5, over 0.35, which 6 of 11,846 training images reach; a threshold chosen after seeing
    the sample):
    - they are an old failure of weaker models, not new: K = 0 at power 6 has 1-4 per 500 (besides its 11-19
      divergences; none at power 4), one Ising unit up to 5, and Potts 8-32 without spring or per-pixel units 2-4
      (Potts 8 with spring alone: 0). They recur on the same few trajectories (for example 3, 28, 215 and 408), so
      some starts and noise paths are hard;
    - the four whose amplitude was printed (job 4203167) peak at 11.2-11.8 normalised units, and every saturated
      sample exceeds 8 units (job 4203221), against 3.1-3.3 for the brightest training pixels and the divergence
      threshold of 15: near-divergent samples, which the evaluation counts as far, not diverged;
    - among the Potts 128 + both trainings they appear with the plug-in estimates without the jackknife (S = 4: 2,
      S = 16: 1), with CD-1 (1), and in 4 of the 9 visit-count runs (exact statistics 1, S = 2: 1, S = 4: 2, S = 8
      with same-draw counts: 1; 5 of 4,500 samples), but in none of the 18 runs with exact or jackknifed statistics
      and per-element Adam, plain steps or one gain per state (0 of 9,000);
    - this is not yet shown to come from the optimiser. 4 of the visit-count rule's 5 come from S = 2, 4 and 8,
      where no jackknifed per-element Adam run exists, and plug-in per-element Adam at S = 4 has 2 as well. At
      matched settings (exact statistics, S = 16 and 64) it is 1 of 2,500 against 0 of 9,000 (p ≈ 0.2). Above 8
      units, a check that avoids the ink cut but is also chosen after looking, it is 3 of 2,500 (all from the
      exact-statistics run) against 1 of 9,000 (p ≈ 0.03). The cut was set after seeing the S = 4 sample, so
      none of these p-values is a planned test;
    - so small S (and perhaps the visit-count rule) lets through a little of a failure that the best trainings
      remove: rare, 0-0.4% per run.
  - `S2jk_visit_count.png`: clearly worse, as the metrics say, and the flaws are mostly 0s and things that are
    neither digit: scribbles or smudges at row 7 columns 3 and 9, row 9 column 7 and row 10 columns 3 and 6
    (besides the row 3 column 3 scribble of S = 16, 8 and 4 and the row 9 column 5 speckle of every model), a 0
    open on one side (row 7 column 4), 0s with a bar through them, and more broken 1s; about 12-15 of 100 by my
    count. The 1s that are fine look like those of S = 4.
  - `S8jk_visit_count_same_draws.png` (visit counts from the same 8 samples; gallery job 4203212, 08:08) against
    `S8jk_visit_count.png` (fresh counts): nearly the same samples, trajectory by trajectory (4 of 100 differ). It
    draws a near-divergent white sample at row 2 column 7 where the fresh-count run draws a slanted 0; at row 2
    column 5 the thick 1 with a bulge that S = 4 and 2 also draw (compare.png column 15), where the fresh-count run
    draws a small 0 with a stroke inside; and a speckled 1 with a flag and a base at row 9 column 7 where the
    fresh-count run draws a clean slanted 1. The fresh-count run draws the same kind of speckled shape at row 7
    column 9, where the same-draw run draws a thin 0.
- **The second-order jackknife at S = 16 (gallery job 4202946, 08:00; rows `S16jk2_per_element_Adam`,
  `S16jk2_visit_count`):**
  - `S16jk2_per_element_Adam.png` against `S16jk_per_element_Adam.png` (order 1, same trajectories): several of
    order 1's broken 1s become whole (row 1 column 6, row 7 column 5); the gapped 1 at row 2 column 9 becomes a 1
    with a flag and a base, like some real 1s; the flat 0 with a bar at row 5 column 9 becomes a clean 1. One 1
    gets bent instead (row 6 column 9, the trajectory that the visit-count rule also bends). By my count about 8-10
    flawed samples of 100, against 11-12 with order 1; the 0s look the same.
  - `S16jk2_visit_count.png` is nearly identical to `S16jk2_per_element_Adam.png`, trajectory by trajectory (it
    keeps the flat 0 at row 5 column 9); as for most hidden-unit models, the trajectory decides the digit.
- **Three digits with the visit-count rule (gallery job 4202168, 03:42; `compare.png` regenerated by job 4202921 at
  05:55, after an older gallery job had overwritten it without this row; `14x14_0-1-2_three_digits_Potts/`):**
  - `compare.png`: the S = 64 visit-count model draws the same digit in the same pose as exact training (Potts
    128) in 15 of the 16 columns, and as S = 256 in 15 (column 13: exact training draws a 1 where the local models
    draw a 2; column 12: S = 256 draws a thin 0 where the others draw a looped 2). The broken 0 with two dots on
    top (column 7) and the 2 with a crossbar (column 6) appear in every hidden-unit model.
  - `local_S64jk_visit_count.png` against `local_S64jk.png` (per-element Adam, same trajectories): most samples are
    identical by eye. Where they differ (a handful, nearly all 2s) the visit-count version is more often the
    cleaner 2 (e.g. row 6 column 2, row 9 column 1). By my count roughly 10-13 bad samples in each, nearly all
    2s: by eye the difference is smaller than the metrics' (far 17.2% vs 25.0%).
  - Against `real.png`: the 2s are the weak class in every model, speckled and often broken, with a scribbled
    loop or crossbar. The real 2s are clean strokes with or without the loop, in many slants and widths. So the
    far rate of the 2s (41% vs 22% for real 2s) is visible as noisy, broken 2s, not as wrong digits.

## Result 14 (2026-11-08): four digits, 0/1/2/3 (14x14)

- Setup:
  - `mnist14_d0123`: the paper's preprocessing, pooled 2x2; 23,692 training images, 5,923 per digit.
  - Potts 256 and 512 + both, label-free initialisation (`--proto-init pooled`), exact force matching, against K = 0.
    Pipelines 4198798 (K = 0) and 4198799 (both Potts models).
  - 500 paired `base` trajectories: evaluation 4198800 (`evaluate_multiclass.py`), coverage 4200400 (matched n = 100
    per class, so that every class is scored; the first coverage run 4198801, at n_match 150, could not score any
    model's 2s or 3s, nor K = 0's 0s, and is not used), states 4198802, gallery 4200384.

| (`base`, 500 paired trajectories) | real images | K = 0 | Potts 256 + both | Potts 512 + both |
|---|---|---|---|---|
| flagged blends | 5.0% | 21.0% | 4.0% | 2.6% |
| far | 10.0% | 66.8% | 17.8% | 12.6% |
| diverged | 0% | 0.2% | 0% | 0% |
| share of 0s / 1s / 2s / 3s | 24 / 27 / 23 / 26% | 23 / 33 / 22 / 22% | 26 / 29 / 21 / 24% | 26 / 28 / 23 / 23% |
| clean yield, standardised | 86.2% | 24.2% | 78.2% | 85.0% |
| clean yield within 0s / 1s / 2s / 3s | 92.2 / 96.2 / 71.8 / 84.5% | 7.0 / 55.7 / 17.6 / 16.5% | 85.3 / 93.0 / 62.3 / 72.1% | 93.1 / 98.6 / 64.6 / 83.8% |
| far within 0s / 1s / 2s / 3s | 5.8 / 1.5 / 22.6 / 11.2% | 93.0 / 33.5 / 80.6 / 76.1% | 13.2 / 2.1 / 34.9 / 26.2% | 6.1 / 0.7 / 31.9 / 15.4% |
| recall of real 0s / 1s / 2s / 3s | 0.88 / 0.89 / 0.90 / 0.87 | 0.51 / 0.79 / 0.32 / 0.47 | 0.86 / 0.87 / 0.67 / 0.81 | 0.86 / 0.85 / 0.79 / 0.75 |
| precision of the 0s / 1s / 2s / 3s | 0.91 / 0.89 / 0.88 / 0.86 | 0.45 / 0.40 / 0.78 / 0.73 | 0.91 / 0.81 / 0.95 / 0.92 | 0.94 / 0.86 / 0.95 / 0.95 |
| median nearest-image distance | 1.597 | 2.363 | 1.807 | 1.729 |

Recall and precision: matched n = 100 per class, mean over 20 subsamples (subsampling spread ±0.02-0.06; ±0.08-0.09
for K = 0's precision of 0s and 1s). Copy rates: 0% / 0% / 0.6% (real 5% by construction).

- **The hidden units scale to four classes, and 512 states come within 1.2 points of real images on clean yield:**
  - against K = 0 (paired): clean yield (standardised) +60.8 points ([+56.6, +65.0]); far 12.6% vs 66.8%
    (p = 9e-74); blends 2.6% vs 21.0% (p = 2e-22);
  - 512 against 256 states (paired): clean yield +6.8 points ([+3.4, +10.4], p = 0.0004); far 12.6% vs 17.8%
    (p = 0.0007); mostly the 3s (clean 72.1% -> 83.8%, far 26.2% -> 15.4%), then the 0s and 1s (clean 85.3% ->
    93.1% and 93.0% -> 98.6%). The per-class differences are not tested separately;
  - 512 states against real images (not paired, not tested): clean yield 85.0% vs 86.2%, far (standardised) 13.5%
    vs 10.3%. The 2s stay below real (clean 64.6% vs 71.8%); the 0s and 1s are above it.
- **More classes need more states** (clean yield, standardised, against real images; one training run each; the
  differences between problems are not tested):
  - at 256 states: 90.1% vs 85.7% on 0/1 (labelled start, Result 8), 83.4% vs 86.5% on 0/1/2 (Result 12), 78.2% vs
    86.2% on 0/1/2/3;
  - at 128 states: 87.1% vs 85.7% on 0/1 (label-free, Result 8), 81.1% vs 86.5% on 0/1/2;
  - tested within a problem: on 0/1/2/3, 512 states beat 256 by +6.8 points and come within 1.2 points of real
    images; on 0/1/2, 256 against 128 states +2.2 points ([-1.1, +5.6], p = 0.2).
- **The weak class is again the 2s**, now joined by the 3s at 256 states. At 512 states the recall of the 2s is
  higher (0.79 vs 0.67, real 0.90) at unchanged precision 0.95, and the 3s' recall does not rise (0.75 vs 0.81).
  Neither difference is tested: the ± values are subsampling spreads, not error bars, and the 2s' pools (106 and
  113 samples) barely exceed n = 100, so the spreads hardly reflect the samples' own noise.
- The states split by class at t = 0 (`inspect_states.py`, 4198802): 77 / 24 / 89 / 66 states carry 0s / 1s / 2s /
  3s with 256 states (all 256 used, purity 0.977, 204 effective states), 141 / 51 / 175 / 135 with 512 (502 used,
  purity 0.990, 386 effective). The 1s, the least varied digit, get the fewest states.
- **What the images show** (`gallery/14x14_0-1-2-3_four_digits_Potts/`, read by eye against `real.png`):
  - K = 0 draws almost no recognisable digit: dark, speckled superpositions of two or three digits. 4 of the first
    100 samples saturate to white noise but stay below the divergence cut (|x| > 15).
  - With 512 states the four classes come out in even shares. The 0s and 1s look like real ones, apart from the
    speckle. The 3s are recognisable but rougher than real 3s: spiky strokes, some with an extra bar or a closed,
    8-like bottom. The 2s are the weakest: often blobby, with a filled base or loop, as on 0/1/2.
  - About 7-11 of the first 100 samples are clearly bad or doubtful: fragments, a flat 0 crossed by a bar, a
    7-like 2. Nearly all are 2s or 3s.
  - In `compare.png` the 256- and 512-state models draw the same digit on each trajectory, as with two digits; 512
    states mostly clean up the same shape.

## Result 15 (2026-11-09): the simplest pixel model, number of Potts states and of Ising units (14x14, 0/1)

Pixels: power 4 (`--visible quartic`), pair couplings springs only (`--pair-terms lin`) between the 8 nearest
neighbours (`--radius 1`) or the 4 nearest (`--neighbourhood von_neumann --radius 1`). Hidden: one Potts unit with
K states, or K Ising units (each a bistable unit coupled to all pixels and their squares; label-free split-tree
start for K > 1), both "+ both" (spring and per-pixel units). Exact training, seed 0. Evaluations 4205293 (Potts),
4205473 (Ising); coverage 4205294, 4205474. Clean yield (50/50); recall/precision of the 0s (k-NN, k = 3).

| hidden | clean, 8 nb | clean, 4 nb | 0s recall / precision (8 nb) | blends (8 nb) |
|---|---|---|---|---|
| none | | 0.0% | | |
| Potts 32 | 65.5% | 58.2% | 0.61 / 0.91 | 3.2% |
| Potts 128 | 81.4% | 74.2% | 0.76 / 0.94 | 1.4% |
| Potts 256 | 85.0% | 80.4% | 0.80 / 0.95 | 1.6% |
| Potts 512 | 89.4% | 87.8% | 0.83 / 0.95 | 2.2% |
| Ising 1 | 38.7% | | 0.41 / 0.29 | 9.0% |
| Ising 8 | 51.9% | | 0.53 / 0.76 | 9.0% |
| Ising 32 | 66.7% | | 0.73 / 0.88 | 6.8% |
| Ising 128 | 78.7% | | 0.81 / 0.89 | 7.0% |
| Ising 196 | 81.1% | | 0.81 / 0.89 | 6.4% |
| real images (probes) | 85.7% | | 0.89 / 0.88 | |

- Potts: every doubling helps (paired, p <= 0.02); the gain is in the 0s (the 1s are ~98% clean from 32 states).
  The 4-neighbour deficit shrinks with K: -7.2 at 128, -4.7 at 256, -1.6 ([-3.5, +0.2], p = 0.08) at 512. Copies
  <= 0.4%, median nearest-neighbour distance 1.24-1.46 (near-copy < 0.45). 89% is above the probes' 85.7%: the
  yield saturates for typical digits; recall (0.83-0.85 against 0.89) is the fairer comparison there.
- Ising: K Ising units match K Potts states in clean yield (32: +1.2 [-2.9, +5.3]; 128: -2.7 [-6.6, +1.1]), with
  higher recall of the 0s and lower precision, and more blends (6-7% against 1-2%); the 1s are 93.5% clean
  against 98%. 128 -> 196: +2.5 ([-0.0, +5.0], p = 0.05), flattening. 196 was then the code's cap (one unit per
  pixel); the cap is only needed for the random (orthogonal) start and was lifted for the split tree at 17:20.
- By eye (`gallery/14x14_0-1_power4_springs_only_number_of_Ising_units/`): Ising 196 draws mostly clean 0s and 1s;
  about 10 of 100 are flawed, a distinct Ising failure: faint, dotted, nearly empty images (e.g. trajectory 7 in
  every Ising row of `compare.png`, where Potts draws a 0) and 1s with stray bits (an "i", an E-like shape).
  Potts 512 (4 neighbours) has about 3 of 100 flawed (a serifed 1, lumpy 0s).
- Reading: a Potts unit is a mixture (one prototype per sample), Ising units a product of features; per hidden
  element the two are equal up to 128, the product mixes 0 and 1 features more often. Untested.
- Hardware reading: per hidden element the two cost the same wiring (each element coupled to every pixel and its
  square), but an Ising unit is one bistable element, while a Potts unit with M states is M bistable elements that
  must also inhibit each other (exactly one on; HARDWARE_NOTES.md). Ising units matching Potts states one for one
  therefore makes Ising the preferred design, if it keeps scaling.
- Running (17:22, Status): 256 / 512 / 1024 Ising units on 0/1; 196 / 256 / 512 Ising units and Potts 512 on
  0/1/2/3, all with this pixel model and 8 neighbours.


**Update (2026-11-09, 22:00): more Ising units, and four digits** (evaluations 4206346, 4206348, 4206349; coverage
4206347; gallery 4206350).

| 0/1, 8 neighbours | clean | 0s clean | 1s clean | blends | 0s recall | copies |
|---|---|---|---|---|---|---|
| Ising 196 | 81.1% | 68.8 | 93.5 | 6.4% | 0.81 | 0.2% |
| Ising 256 | 80.7% | 66.8 | 94.5 | 5.6% | 0.81 | 0.8% |
| Ising 512 | 83.5% | 70.5 | 96.5 | 4.8% | 0.80 | 2.4% |
| Ising 1024 | 85.4% | 74.6 | 96.2 | 4.8% | | 2.4% |
| Potts 512 | 89.4% | 80.7 | 98.0 | 2.2% | 0.84 | 0.4% |

- 256 -> 512: +2.8 ([+0.5, +5.2]); 512 -> 1024: +1.9 ([-0.2, +4.1], p = 0.08); Ising 1024 against Potts 512: -4.0
  ([-7.5, -0.5]). Ising keeps improving slowly (about +2 points per doubling against about +4 for Potts at this
  size), mostly in the 1s and blends; the recall of the 0s does not move (0.80-0.81). Copies 2.4% at 512 and 1024
  (held-out real images 4.9%, by construction of the threshold), median nearest-image distance 1.12-1.16: closer
  to the training set than Potts 512, not memorisation.
- By eye (`Ising1024_8nb.png`): the faint, near-empty samples remain, at the same trajectories as with 196 units
  (e.g. trajectories 7, 12, 49, 85).

| 0/1/2/3, 8 neighbours | clean (std) | 0 | 1 | 2 | 3 | blends |
|---|---|---|---|---|---|---|
| Ising 196 | 64.5% | 63.1 | 87.4 | 45.0 | 62.6 | 14.3% |
| Ising 256 | 67.0% | 68.0 | 89.4 | 46.7 | 63.7 | 13.9% |
| Ising 512 | 66.9% | 67.8 | 88.0 | 48.2 | 63.4 | 15.5% |
| Potts 512 | 79.0% | 91.1 | 96.3 | 58.0 | 70.6 | 2.6% |
| Potts 512, full pixel model (Result 14) | 85.0% | 93.1 | 98.6 | 64.6 | 83.8 | 2.7% |
| real images (probes) | 86.2% | 92.2 | 96.2 | 71.8 | 84.5 | 5.1% |

- Four digits: Ising flat from 256 to 512 (-0.1 [-3.1, +3.0]) and 12.1 points below Potts 512 ([-16.4, -7.8]),
  with 14-16% blends against 2.6%. The product of features does not scale better with the number of classes than
  the Potts mixture; on this evidence it scales worse. The simplest pixel model costs Potts 512 6.0 points on four
  digits ([-9.3, -2.8]; about 3 on 0/1 at 128 states).
- By eye (`14x14_0-1-2-3_four_digits_power4_springs_only_Ising_vs_Potts/compare.png`): the Ising 196, 256 and 512
  rows are nearly the same image for image: the extra units hardly change the samples. Trajectories 6, 7, 9 and
  14 are scribbles in every Ising row where both Potts models draw a 1, a 0, a 0 and a 1.
- Reading (untested): the extra Ising units seem not to be used. Units deep in the split tree separate small leaves
  and start small; whether they stay small, or duplicate other units, is what `inspect_units.py` would show.

**Do the Potts states just recall prototypes?** (`prototype_test.py`, job 4207598; figures in
`out/latent_physical_SGM/prototype_test/4207598/`). Under the t = 0 couplings, every training image, probe and sample
is assigned to its most likely state; a state's prototype is the mean of its training members. Samples against the
1,000 probes (real images not used to build the clusters):

| run | distance to own prototype, samples / probes | within-state spread, samples / real | nearest training image in own state, samples / probes |
|---|---|---|---|
| Potts 32, 8 nb | 0.93 | 0.90 | 0.58 / 0.72 |
| Potts 128, 8 nb | 0.91 | 0.93 | 0.65 / 0.64 |
| Potts 256, 8 nb | 0.89 | 0.91 | 0.67 / 0.63 |
| Potts 512, 8 nb | 0.91 | 0.90 | 0.69 / 0.62 |
| Potts 512, 4 nb | 0.94 | 0.95 | 0.66 / 0.63 |
| Potts 128, full pixel model | 0.95 | 0.95 | 0.67 / 0.66 |
| Potts 512, 0/1/2/3, full pixel model | 0.93 | 1.02 | 0.65 / 0.59 |

- Not "prototype plus blur": samples sit only 5-11% closer to their state's prototype than real images do, and vary
  within a state about 90-100% as much as real members. The contraction does not grow with the number of states
  (0.93 at 32, 0.91 at 512).
  (Correction from ChatGPT: These distance and spread checks give evidence against near-copy or narrowly
  jittered prototypes, but a prototype-centred noise model can also reproduce such statistics. The probes were
  held out from computing the centroids, not from training the model.)
- About a third of the samples have their nearest training image in another state, as for real images: the samples
  are not confined to their state's members.
- At t = 0 every image, real or generated, belongs decisively to one state (max posterior 0.99-1.00): the states
  partition the data like a dense associative memory, and the variation within a state comes from the pixel model.
- State sizes are very uneven (Potts 512: one state of 1,146 training images, 389 states with >= 5 members, effective
  number 229), so the average of 23 images per state understates how coarse the main states are. The samples use
  fewer states than the training images (229 of 464), partly because 500 samples cannot hit every small state
  (1,000 probes use 192 effective states against the samples' 155; not corrected for sample size).
- By eye (Potts 512, 8 nb; the 8 states most used by samples are all 1s): the samples of a state are different 1s
  of the state's slant and thickness, slightly noisier than the real members, none a copy of the prototype.

**Update (2026-11-10, 01:00): idle Ising units are not what limits Ising.** The training logs of Ising 128 / 512 /
1024 on 0/1 show that at slice 50 (t about 2.5) only about 40-50 units are decided (|m| > 0.9) whatever K is; the
others are weak (field std < 0.1 T: 0 / 113 / 671 units) and switch on only late (t < 1). Two fixes in
`train_latent.py`, both for split-tree Ising units with exact training:
- `--ising-reseed`: at the start of every slice, weak units are reset to the split-tree formula (W, b from
  `split_reference`, Q = 0);
- `--ising-weak-boost B`: weak units get B times the step size (B = 10).

Ising 512, 14x14, simplest pixel model, 8 neighbours (runs 4207321-24; evaluations 4207325, 4207327; coverage 4207326;
gallery 4207328). Paired against plain Ising 512 (same initial states and noise):

| run | weak units at slices 30 / 50 / 70 | clean (0/1) | difference [95% CI] | clean std (0/1/2/3) | difference [95% CI] |
|---|---|---|---|---|---|
| Ising 512 | (113 at slice 50) | 83.5% | | 66.9% | |
| reseed | 1 / 2 / 0 before, 0 after reseeding | 84.8% | +1.3 [-1.2, +3.7], p = 0.31 | | |
| boost 10 | 278 / 278 / 250 | 84.5% | +1.0 [-1.3, +3.3] | | |
| reseed + boost 10 | 1 / 0 / 0 | 82.8% | -0.7 [-3.4, +1.9] | 66.8% | -0.1 [-3.4, +3.2] |
| Potts 512 | | 89.4% | | 79.0% | |

- Reseeding keeps every unit active through the diffusion, yet the samples do not improve: on four digits the
  blends even rise (18.2% against 15.2%, p = 0.07). The larger step size alone does not revive the weak units.
- Reseeding lowers far-off samples on 0/1 (11.5% against 14.1%, -2.7 [-4.7, -0.5]), but not the clean yield.
- Coverage (0/1) unchanged: recall of the 0s 0.83-0.84 for all Ising variants, 0.88 for Potts 512.
- By eye: the reseeded 0/1 samples look like plain Ising 512 (a few faint near-empty images remain); on four digits
  the same broken and blended 2s and 3s.
- Reading: the units were weak because, with exact training, the extra units carry little score; giving them the
  formula couplings does not make them useful. The gap to Potts is in the form of the model (a product of on/off
  features against a mixture of prototypes), not in the optimiser.
  (Correction from ChatGPT: This interpretation is not established by the tested interventions. They rule out
  those particular remedies as sufficient fixes; the relative roles of model capacity and optimisation remain
  unresolved, as the later convergence discussion acknowledges.)
- Consequence (decision rule agreed 2026-11-09): no fix beat plain Ising 512 by >= 3 points, so at 28x28 on all ten
  digits only plain Ising 3136 runs, as the comparison for Potts 1536 / 3072 (job 4208054; evaluation 4208055 against
  Potts 1536 / 3072; gallery 4208056, folder `28x28_all_ten_digits_power4_springs_only_Ising_vs_Potts`).

**Update (2026-11-10, 08:30): 28x28, four digits, simplest pixel model** (power 4, springs only, 8 neighbours,
exact training; runs 4206653, 4206669, 4207686; evaluations 4206656, 4206671, 4207688; gallery
`28x28_0-1-2-3_four_digits_power4_springs_only_Ising_vs_Potts`).

| 0/1/2/3, 28x28 | clean (std) | 0 | 1 | 2 | 3 | far (std) | blends | copies | median NN |
|---|---|---|---|---|---|---|---|---|---|
| Ising 784 | 66.2% | 64.4 | 85.0 | 52.7 | 62.7 | 24.4% | 17.0% | 0.4% | 4.90 |
| Potts 784 | **92.2%** | 93.2 | 98.6 | 85.6 | 91.3 | 5.2% | 3.4% | 0.0% | 4.49 |
| Potts 784, frozen (formula couplings) | 74.6% | 74.4 | 99.3 | 52.1 | 72.4 | 24.9% | 0.8% | 0.0% | 5.14 |
| Ising 1568 (12:20 update) | 70.8% | 68.3 | 87.2 | 61.2 | 66.7 | 19.6% | 15.2% | | |
| Potts 1536 (12:20 update) | **95.8%** | 97.6 | 99.3 | 87.0 | 99.2 | 2.1% | 2.6% | | |
| Potts 1536, frozen (12:20 update) | 87.6% | 87.9 | 100.0 | 76.4 | 86.2 | 11.7% | 0.7% | 0.0% | 4.93 |
| real images (probes) | 86.8% | 93.0 | 97.7 | 71.0 | 85.7 | 10.1% | 5.0% | 5.0% | 4.41 |

- Potts 784 against Ising 784: +26.0 points ([+21.6, +30.4]); against frozen Potts 784: +17.6 ([+13.7, +21.6]).
- 12:20 update (evaluations 4206657, 4206672, 4207681): doubling helps both, Potts 1536 - 784 +3.6 [+0.9, +6.4],
  Ising 1568 - 784 +4.6 [+1.4, +8.0]; the gap stays (Ising 1568 - Potts 1536 -25.0 [-29.5, -20.5]). Frozen
  couplings improve with size (74.6 -> 87.6%) but stay below trained (-8.2 [-11.5, -5.0]). FID of these follows in
  4209815 after Ising 3136.
- Potts 784 is above the real probes on clean yield: fewer far samples (5.2% against 10.1%), i.e. somewhat more
  typical than real digits, with no copies and a median nearest-image distance near the probes' (4.49 against
  4.41). The 2s gain most (85.6 against 71.0 for real).
- With the simplest pixel model, 28x28 is much better than 14x14 (Potts 512 at 14x14: 79.0%); Ising is the same at
  both sizes (about 66-67%).
- Training the hidden couplings matters: the frozen formula couplings give speckled, broken 2s and 3s.
- By eye (`Potts784_8nb.png`): nearly all 100 samples are clear digits with natural variety; strokes are grainy
  (white speckle on the ink, a few isolated dots around it). Ising 784 draws smoother strokes but many broken or
  blended 2s and 3s.

**FID and KID, in addition to the % measures (2026-11-10, `fid_score.py`; jobs 4209862-65, classifiers 4209809/10).**
Features: the 128 hidden units of a small CNN trained on all ten MNIST training digits (test accuracy 98.2% at
14x14, 99.1% at 28x28). Reference: balanced MNIST test images of the problem's digits (never seen by any model),
with the paper's 0.01 pixel noise. Raw pixel values (as the % measures); "clip" = clipped to [0, 1]. Floor = 500 real
training images against the same reference (FID is biased upwards at n = 500, so read models against the floor, not
against 0). Confident = the CNN assigns one of the problem's digits with max p >= 0.9. FID and KID agree throughout.

(Correction from ChatGPT: `fid_score.py` excludes nonfinite samples and samples with max |x| > 15, so these scores
describe the surviving samples. Its reference floor uses the median number of survivors across the compared
sets, rather than a separately matched sample count for each model. It also needs a guard for fewer than two
survivors; this edge case was not encountered in the reports checked during the review.)

| problem | model | FID | FID clip | confident | clean (% measure) |
|---|---|---|---|---|---|
| 14x14 0/1 | real floor | 0.85 | | 94.3% | |
| | Potts 32 / 128 / 256 / 512, 8 nb | 4.17 / 2.45 / 1.70 / 1.38 | 5.36 / 3.25 / 2.27 / 1.83 | 88-95% | 65.5-89.4% |
| | Ising 128 / 196 / 256 / 512 / 1024 | 3.56 / 3.48 / 3.47 / 3.49 / 3.75 | 3.9-4.2 | 88-90% | 78.7-85.4% |
| | Ising 512 reseed / boost10 / both | 3.20 / 4.08 / 3.29 | | 89-90% | 82.8-84.8% |
| 14x14 0/1/2/3 | real floor | 1.50 | | 94.3% | 86.2% |
| | Ising 196 / 256 / 512 / 512 both fixes | 8.99 / 8.54 / 8.28 / 7.87 | 8.6-10.1 | 71% | 64.5-67.0% |
| | Potts 512, simplest pixel model | 3.40 | 4.83 | 93.2% | 79.0% |
| | Potts 512, full pixel model | 4.15 | 5.09 | 90.6% | 85.0% |
| 28x28 0/1 | real floor | 2.80 | | 98.6% | |
| | paper (no hidden units) | 135.0 | 134.6 | 55.7% | 31.9% |
| | Potts 32 / 128 / 256, full model | 20.3 / 12.8 / 10.8 | | 96-99% | |
| | Potts 128 frozen | 21.5 | | 94.8% | |
| | Potts 256 / 512, power 4 | 9.89 / 8.17 | | 98.4% | 92.3% (512) |
| | Potts 512, power 4, local S = 64 visit-count | 9.52 | | 97.8% | 90.5% |
| 28x28 0/1/2/3 | real floor | 4.57 | | 98.4% | 86.8% |
| | Ising 784 | 37.3 | 38.1 | 71.6% | 66.2% |
| | Potts 784 | 20.3 | 22.6 | 98.6% | 92.2% |
| | Potts 784 / 1536 frozen | 57.4 / 48.3 | | 84 / 88% | 74.6% (784) |

Paired bootstrap differences (same trajectories), 95% intervals:
- Ising against Potts at equal size: 14x14 0/1 Ising 512 - Potts 512 +2.1 [+0.8, +4.3]; four digits +4.9 [+3.1, +7.1];
  28x28 four digits Ising 784 - Potts 784 +17.0 [+12.3, +23.0].
- More states: Potts 512 - 128 (14x14 0/1) -1.1 [-1.8, -0.3]; Potts 512 - 256 power 4 (28x28 0/1) -1.7 [-2.3, -1.0].
- Trained against frozen hidden couplings: 28x28 0/1 Potts 128 -8.7 [-10.3, -7.2]; four digits Potts 784 -37.1
  [-39.9, -34.1].
- Local training (S = 64, visit-count rule) against exact, Potts 512 power 4 at 28x28: +1.35 [+0.6, +2.0], p < 0.01.
  On clean yield the same comparison was -1.8 points, not significant: FID resolves a small cost of local training.
- Ising 512 reseed - plain (14x14 0/1): -0.30 [-0.63, +0.02], n.s. Full against simplest pixel model, Potts 512,
  four digits at 14x14: +0.75 [+0.01, +1.46] (KID +14 [+3, +26]).

Reading:
- FID confirms the main ranking: hidden units >> none (paper 135 against 8.2), more Potts states help, trained >>
  frozen, Potts >> Ising, and the Ising fixes do not help.
- FID shows Ising flat from 128 units on (3.5-3.8 at 14x14), where clean yield still rose a few points.
- At 14x14 on 0/1 the best Potts models are near the real floor (1.38 against 0.85) and the CNN finds as many
  confident digits as in real images. At 28x28 they are not: Potts 512 power 4 has FID 8.2 against a floor of 2.8, and
  four-digit Potts 784 20.3 against 4.6, although its clean yield (92.2%) is above the real images' (86.8%). The
  clean yield measures shape (class, blend, distance to the nearest training image); FID also sees texture, and the
  28x28 samples are grainy (speckle on the ink, stray dots). The remaining gap at 28x28 is likely the grain.
- The two measures disagree on the pixel model at 14x14, four digits: clean yield favours the full pixel model
  (85.0 against 79.0%, Result 14 / above), FID and KID the simplest (3.40 against 4.15), as does the CNN's confident
  share (93.2 against 90.6%). The claim "the simplest pixel model costs 6 points on four digits" depends on the
  measure.
- Clipping to [0, 1] changes FID little for good models (Potts 512 14x14 1.38 -> 1.83) and slightly worsens it; it
  matters for bad ones (no hidden units, four digits: 229 raw, dominated by a few out-of-range samples, 51 clipped).
- Pending (chained to the runs): 28x28 four digits with all runs (4209815) and ten digits (4209816).

**Grain test (2026-11-10): is the 28x28 grain leftover thermal noise or learned?** The 28x28 samples are grainy
(speckle on the strokes, stray dots), the likely reason why FID stays far above the real floor while the clean yield
is high. Sampling only, on two finished models, with the same 500 trajectories: (i) the SDE samples as before;
(ii) the same samples quenched, i.e. relaxed at zero temperature under the learned t = 0 free energy,
dx/ds = -grad F(x), for s = 0.05 / 0.25 / 1 (cooling the bath at the end of the protocol; `sample_latent.py
--quench`); (iii) the PF-ODE sampler (no noise at all). Jobs 4209958-67; galleries `28x28_grain_test_quench_and_ODE_*`.
If the grain were thermal, the quench would remove it and FID would drop towards the floor.

(Correction from ChatGPT: The implemented quench removes visible diffusion while retaining the original
temperature in the hidden tanh/softmax statistics: it follows -grad F_T. Cooling a shared physical bath would
also change those statistics and therefore the effective free energy. The tested finite relaxation times do not
guarantee that all thermal texture would disappear.)

| model | FID: SDE | quench 0.05 | quench 0.25 | quench 1 | PF-ODE | real floor |
|---|---|---|---|---|---|---|
| Potts 784, simplest pixel model, four digits | 20.3 | 19.9 | 18.6 | **16.4** | 35.4 | 4.6 |
| Potts 512, power 4, full pixel model, 0/1 | 8.17 | 7.89 | **7.25** | 7.33 | 15.4 | 2.8 |

- Paired against the SDE samples: four digits quench 1 -3.9 [-4.3, -3.5]; 0/1 quench 0.25 -0.9 [-1.3, -0.7]; both
  significant. The PF-ODE is much worse: +15.1 [+10.0, +21.7] and +7.2 [+4.3, +10.5].
- **The grain is mostly learned, not thermal.** The best quench closes about a quarter of the gap to the floor on four
  digits (20.3 -> 16.4 against 4.6) and about a sixth on 0/1 (8.2 -> 7.3 against 2.8). By eye the speckle on the
  strokes thins a little; the blotchy, uneven strokes remain. The remaining FID gap sits in the learned t = 0 energy
  (the pixel model and the hidden units), not in the noise of the last sampling steps.
  (Correction from ChatGPT: The experiments establish that the tested final relaxation improves FID but leaves
  a texture gap. They do not isolate thermal noise, errors accumulated earlier in the reverse process, and errors
  in the final-time energy, so the attribution above is stronger than the evidence.)
- Clean yield unchanged (four digits 92.2 -> 91.8%; 0/1 within 0.4 points). The quench pulls samples slightly
  towards the training images (copies four digits 0.0 -> 0.6%, 0/1 1.4 -> 2.2%; held-out real images 5% by
  construction) and smooths them (roughness four digits 0.171 -> 0.149, real 0.180).
- Long quenches can leave the learned energy's basin: with the full pixel model 4 of 500 samples diverged at s = 1
  (0 at s = 0.05, 1 at s = 0.25); none with the simplest pixel model. The t = 0 energy is not confining everywhere.
  (Correction from ChatGPT: Both saved quench models have confining energies: their positive local quartics and
  nonnegative pair quartics dominate the lower-order terms, including the hidden contribution. The evaluation
  label "diverged" includes finite samples with an amplitude above 15; it does not prove the energy is unbounded
  below. FID also excludes these failures: the binary model's reported SDE / quench 0.25 / quench 1 scores use
  500 / 499 / 496 surviving samples, respectively.)
- The PF-ODE gives faint, speckled samples with a different class balance (more 1s); with an imperfect learned score
  the SDE is the better sampler, as is common.
- Consequence: a short quench (s = 0.25) is a cheap, physically natural improvement (cooling at the end; about 20 s
  per 500 samples) and could be part of the sampling protocol in the paper, reported next to plain SDE sampling.
  It does not change any conclusion. Whether more hidden elements shrink the learned grain is what the ten-digit
  sweep will show; if FID levels off above the floor there, the pixel couplings are the limit.
  (Correction from ChatGPT: A plateau alone would not identify pixel couplings as the limit. Fixed training
  budgets, optimisation, hidden architecture, and sampling errors can also produce a plateau.)

**First ten-digit results, 28x28 (2026-11-10, 15:45; simplest pixel model, % measure, FID with the full sweep
tonight).** Evaluations 4207381, 4207682 (500 samples) and the sweep pipelines 4209941-44 (2,000 samples each).

| ten digits, 28x28 | clean (std) | far (std) | blends | copies |
|---|---|---|---|---|
| pixels only (no units at all) | 0.0% | 100% | 0.5% | 0.0% |
| no hidden elements (per-pixel units only, K = 0) | 4.1% | 95.8% | 0.4% | 0.0% |
| Ising 64 | 51.4% | 46.6% | 5.0% | 0.0% |
| Potts 64 | 64.1% | 35.2% | 0.9% | 0.0% |
| Potts 1536, frozen | 78.6% | 21.3% | 0.2% | 0.0% |
| **Potts 1536** | **92.2%** | 6.6% | 1.2% | 0.0% |
| real images (probes) | 86.1% | 10.4% | 5.0% | 5.0% |

- Without shared hidden elements the simplest pixel model cannot make digits at all (0-4% clean, nearly all far);
  64 hidden elements already give 51-64%; Potts 1536 is above the real probes, with balanced class shares
  (7-12% per digit), no copies and median nearest-image distance 4.59 (probes 4.28).
- By eye (`28x28_all_ten_digits_power4_springs_only_Potts/Potts1536_8nb.png`): all ten digits, clearly
  recognisable and varied, with the same grain as on four digits; a few messy 8s / 9s / 5s.

**Is training converged? (2026-11-10, running)** Every slice trains a fixed number of steps (the paper's schedule,
10,901 at t = 5 rising to 30,901 at t = 0; the paper's early stopping never triggers), with no convergence check.
Crude test from the logs: if a slice had converged, its best mini-batch would fall anywhere in the slice (about 10%
of slices in the last 10% of steps). Observed: 51 / 53 / 55 of 100 slices (28x28 four digits Potts 784 / Ising 784;
14x14 0/1 Potts 512). Mid-diffusion slices look flat (best mini-batch at 7-53% of the steps); the late slices (80-99,
the ones that set the t = 0 energy and so the grain) have it at 83-99% of the steps in every run, so they are
still improving when they stop. Risks: part of the learned grain may be undertraining, and a fixed budget could make
FID level off with K for a training reason (larger models may need more steps).
- Test (job 4211741, about 20 h; FID 4211742, eval 4211743, gallery 4211744): four-digit Potts 784 retrained with
  twice the steps (`--steps-scale 2`) and learning curves (`--val-every 1000`: EMA validation loss every 1,000
  steps, paired per-sample losses). If FID barely moves, keep the fixed schedule and report the check; if it improves
  clearly, choose a convergence criterion from the curves (e.g. stop when the validation loss improved by less than
  one standard error over the last few thousand steps, capped at about 3x) and rerun the sweep, largest K first.

## Caveats

1. **The hidden units are idealised.**
   - They are averaged out exactly, i.e. infinitely fast.
   - A Potts unit's states are exactly one-hot. In hardware that means one M-well oscillator, or M bistable units with
     strong lateral inhibition (winner-take-all).
   - The reverse process uses the doubled free energy; explicit hidden dynamics were never simulated.
2. **Local training with measured hidden statistics is shown end to end at 14x14, and at 28x28 for 0/1 only.**
   - Every model of Results 7-9 was trained with exact score-matching gradients and Adam. Result 10 shows that this
     gradient is a local force-matching rule (exact hidden statistics).
   - Trained end to end at 14x14 (Result 10): force matching with measured hidden statistics (S samples, with and
     without the jackknife; on 0/1/2 also, Result 12), with per-element Adam, plain steps or the per-state rules,
     and CD-1 with exact hidden averages. At 28x28 (Potts 128 + both only), S = 64 and S = 256 with the jackknife
     are indistinguishable from exact training (Result 10); S = 16 with the visit-count rule matches exact training too (87.1%;
     per-element Adam 85.3%); the best model (Potts 512 + both, power 4) with S = 64 and the visit-count rule is
     running. Four digits (S = 64, visit-count rule) fall short: 74.7% against 85.0% exact (Result 14). The
     simplest-pixel-model runs of Result 15 (Potts and Ising sweeps) were trained with exact statistics only.
   - CD-1 with sampled hidden states was only gradient-checked: matching the noise of exact-average CD-1 would take
     about 4·10³-9·10⁷ hidden reads per configuration (Result 10, point 3); training with fewer reads was not tried.
   - One training seed per measured-statistics run; training-seed noise is about 2 points (Caveat 4).
   - Not modelled: the nudge measurements (DESIGN §2.1; for the Potts + spring unit a symmetric pair of nudges is
     exact, so only readout noise remains), Adam's per-parameter scaling on hardware (the visit-count rule replaces
     it for the Potts couplings, with one scale per group for the per-pixel units; the visible couplings keep
     per-element Adam), and double-well (non-Ising) hidden units.
3. **Labels enter the default initialisation.** The Fisher direction and the per-class k-means prototypes use the
   labels; K = 0 does not. The fair label-using reference is cls0 + cls1, and Potts 32 + both beats it (Result 7). The
   label-free variants are at least as good as the labelled ones:
   - `--proto-init pooled` at 14x14 (Result 8) and 28x28 (Result 7);
   - a random start at 14x14 (Result 10).
4. **One training run per model**, except two seeds each for:
   - Potts 128 + both at 14x14, at power 6 (Result 8) and at power 4 (Result 9);
   - Potts 32 + both at 28x28 (Result 7).

   The seeds differ by up to about 2 points (50/50 far: -2.0 and -1.5 at 14x14, -2.1 at 28x28; none significant).
   Most p-values cover trajectory noise only.
5. **Model selection and multiple testing.**
   - Far more than 40 variants were compared on the `base` trajectories of their size, with many McNemar tests
     and paired bootstraps: about 15 at 28x28; at 14x14 over 55 sample sets for 0/1 (31 of them Potts 128 + both
     training variants), plus 0/1/2, 0/1/2/3, 1/4 and 4/9. Borderline single results (p around 0.01-0.05) should
     be read with that in mind.
   - Confirmed on the independent `ext` trajectories:
     - 28x28: Potts 32 + both against the paper (Result 7), and its power-4 version (Result 9);
     - 14x14: K = 0 and Potts + both with 32, 128, 256 and 512 states, and power 4 at 32 and 128 states (Results 8
       and 9).
   - The rest rests on `base` alone: Results 4-6, the other 28x28 variants of Result 7, and 8, 16 and 64 states and the
     second seeds at 14x14.
   - Borderline p-values (0.01-0.05) should not be over-read.
6. **Metric definitions.**
   - "Real images" are training images scored leave-one-out (the 1,000 probes), not unseen test digits.
   - "Far" depends on the class mix; the 50/50 rates correct for it.
   - The 28x28 far threshold (4.84) is inherited from the ode_vs_sde study.
   - In `train_log.jsonl` of `--quad` runs started before 2026-11-08 09:15, the hidden diagnostics `class_acc`,
     `mean_max_p` and `eff_states` omit the spring term Q·x² of the input field and are wrong. For example they show
     25 effective states at t = 0 for 28x28 Potts 128 + both, where the correct value is 112. Training was not
     affected; use `inspect_states.py`.
7. **Hardware cost** (how buildable global hidden units are, platform by platform, with references: `HARDWARE_NOTES.md`). Per time slice, on top of the visible model's 545,566 parameters:
   - Potts 32 + both adds 51,776 parameters (9.5%) and a 32-way winner-take-all;
   - Potts 128 + both adds 202,400 (37%) and a 128-way one;
   - Potts 256 + both adds 403,232 (74%) and a 256-way one.

   431 of the 784 pixels get a per-pixel unit (those with two intensity modes).

## Companion writeup corrections (2026-11-10)

(Correction from ChatGPT: The consistency review found agreement in the core free-energy, local-gradient and
sampler calculations. The following corrections concern statements in
[`theory.tex`](../latent_physical_SGM_writeup/theory.tex) and [DESIGN.md](DESIGN.md); they are recorded here for
follow-up edits to those documents.)

- **Finite hidden speed:** detailed balance establishes an equilibrium distribution for fixed parameters; it
  does not imply that the driven joint system is in an instantaneous Boltzmann state. Fast conditional hidden
  equilibration is the assumption behind the implemented effective drift. For frozen replica-generation
  parameters, the visible equilibrium energy is `2F - |x|²/2`, not `F`.
- **Reverse-time argument:** with forward-time parameters `theta(t)` and reverse time `tau = 5 - t`, the drift
  uses `x - 2 grad F_{theta(5 - tau)}(x)`. The note currently writes `theta(tau)`; the code uses the correct reversal.
- **Zero-weight degeneracy:** for bilinear Ising coupling with `b = 0`, `W = 0` is stationary; with nonzero bias,
  suitable visible-optimum conditions are needed. Absorption of the leading hidden quadratic term requires the
  corresponding visible couplings; it is not automatic for the sparse networks. The blanket stationary/degenerate claim needs these
  qualifications.
- **Bistability:** negative quadratic stiffness does not guarantee a double well in the presence of a learned
  bias. For example, `-x²/2 + x⁴/4 + 2x` has one minimum. A per-pixel hidden term also permits rather than
  guarantees bistability of the total potential.
- **CD1:** for the release `x' = x - dt grad F/2 + sqrt(T dt) xi`, the clamped-minus-released parameter derivative
  equals `(T² dt/2) grad J + O(dt²)`, with the released state held fixed in that derivative. The note omits the
  remainder and release convention; the code uses this release convention and documents the small-step approximation.
- **Results and settings:** matching the probes' clean yield does not establish matching their full distribution;
  the FID and coverage gaps remain. Keep the two optimiser comparisons and their replication evidence distinct
  (see Status). The note's EMA description matches the cited EMA runs, but the CLI default is still `--select best`.

## Next steps

1. **Running** (Status): at 28x28, the best model (Potts 512 + both, power 4) trained locally with S = 64 and
   the visit-count rule; with the simplest pixel model, 256 / 512 / 1024 Ising units on 0/1, and 196 / 256 / 512
   Ising units against Potts 512 on 0/1/2/3.
2. **A device-compatible optimiser for the hidden units** (Result 10, "The role of Adam"): steps with one step size
   per group (the closest tested rule to plain gradient steps) lose the states even with exact statistics; a
   per-state step size brings them back. The visit-count rule also keeps most of them with few measured samples
   (at equal reads, S = 8 with the rule against S = 16 with per-element Adam: +6.0 points; S = 4 and 8 as good as
   S = 16; S = 2 breaks), and on 0/1/2 it is indistinguishable from exact training with S = 64 (Result 12). Its
   visit counts can come from the same S draws as the statistics at no measurable cost (S = 8), so it reads no
   more than per-element Adam.
   - Tested at 28x28 (S = 16 matches exact training; Potts 512 + both with S = 64 running) and on four digits
     (S = 64: 74.7% against 85.0% exact, the open four-digit gap, Result 14).
   - Why at S = 2 the states recovered after the consolidation go mostly to the 1s (0s / 1s 23 / 22 at slice 20,
     29 / 73 at t = 0), whether that is why its 0s fail, and why S = 4 does not break; whether a minimum step (or
     a label-free balance of the visit counts) keeps the allocation balanced at S = 2.
   - Visit counts from the draws of the statistics (half the reads, `--occ-same-draws`, test 18 passed): S = 8 as
     good as fresh counts (81.7% vs 81.6%); S = 16 83.0% and S = 4 81.5% (4 reads), as good as
     fresh counts. Next: same-draw counts in the 28x28 and multi-digit runs.
   - The states it still loses (34-41 of 128 at S = 8-16; 17 at S = 4, 26 at S = 2): a longer visit window does
     not help (it loses more), and S = 4 and 2 lose fewer than S = 8-16.
     Untested: many reads only during consolidation (with per-element Adam, S = 256 up to slice 29 and then 16
     matched S = 256 in clean yield, paired -0.8 [-3.1, +1.6], but lost states right after the switch, 71 vs 81
     effective at t = 0; and at S = 16 a third of the visit-count rule's lost states drop out after slice 30).
   - The gap to exact statistics at S = 4-16 (3.5-4.3 points against the rule's own exact-statistics run,
     significant at S = 4, 8 and 16; flat in S) is probably neither mainly bias nor mainly noise: S = 4, with 16
     times the O(1/S²) bias and 4 times the noise variance of S = 16, does as well in clean yield (+0.8 [-2.1,
     +3.7]; one run each). The second-order jackknife's +1.5 at S = 16 is not significant and does not decide it.
     With per-element Adam the bias does matter (+5.1 from order 2). What the gap is remains open. Untested: order
     2 with fewer draws under the visit-count rule (S = 4 or 8), and whether order 2 and replicates add up for
     per-element Adam.
   - The per-pixel units already train with one step size per group in the visit-count runs (only the Potts
     couplings get the per-state rule), and those runs match per-element Adam. Still to check: no Adam-like global
     scale at all (a fixed learning-rate schedule), and the visible couplings (per-element Adam in every run).
3. **The simplest pixel model, with the expressivity in the hidden units: Ising scaling** (the strategy since
   2026-11-09: 14x14, power 4, linear springs between nearest neighbours only; quality as a function of the number
   of hidden units). Done (Result 15): Potts 32-512 at 8 and 4 neighbours (512: 89.4 / 87.8%), Ising 1-196 at 8
   neighbours (196: 81.1%, matching Potts at equal K up to 128). The optomechanical pixel term is not needed (Status).
   Ising units are the preferred hardware story (one bistable element each, no winner-take-all), so the next steps
   follow the Ising units:
   - done: more Ising units on 0/1 (512: 83.5%, 1024: 85.4%, about +2 per doubling, still below Potts 512) and four
     digits (flat at 67% from 256 to 512, 12 points below Potts 512); idle-unit fixes (reseed, step-size boost) do
     not help (Result 15, 01:00 update);
   - running: 28x28, four digits, Ising 784 / 1568 / 3136; all ten digits, plain Ising 3136 against Potts 1536 /
     3072;
   - the Ising failure modes (blends, faint near-empty samples) are in the model form, so the next candidates change
     the model: hidden-hidden coupling among Ising units, or a winner-take-all among groups of units (a step
     towards Potts with bistable elements);
     (Correction from ChatGPT: These are reasonable candidate model changes, but the experiments have not
     established that the failures are intrinsic to the model form; optimisation remains an open explanation.)
   - if Ising scales: 4 nearest neighbours, ten digits (proposed 2026-11-09, deferred until the Ising results),
     28x28, and local training with measured statistics (so far exact training only);
   - open: whether the trained pixel potentials are bistable (double wells) and whether that can be imposed.
4. **Hardware realism, at 14x14 first:**
   - replace the M-state Potts unit by M bistable units with finite lateral inhibition, and find how strong it must be;
   - simulate the hidden dynamics explicitly, at finite speed (now they are averaged out exactly);
   - model the nudge measurements of the force-matching rule with readout noise (DESIGN §2.1: symmetric nudges are
     exact otherwise);
   - CD-1 would become usable with hidden readouts that stay correlated across the release; that needs the explicit
     hidden dynamics above.
5. **Variety.** The samples are more typical than real images; the most varied class is under-covered (28x28 0s:
   recall 0.55-0.61 vs 0.89; 14x14 2s: 0.72 with 256 states vs 0.88). Options:
   - more states (they add variety at 28x28 and on 0/1/2 up to at least 256);
   - frozen couplings, which keep more variety (28x28: recall of real 0s 0.59 vs 0.38), combined with a learned
     spring, which adds clean yield (untested combination);
   - low-rank Potts-gated pair couplings, so that each state carries its own covariance, not only its own mean and
     per-pixel widths.
6. **More classes and full size:** four digits need 512 states at 14x14 (Result 14). Next all ten digits at 14x14
   (about 1,000-2,000 states by that count, or a second Potts unit), and three or four digits at 28x28.
7. **Seeds:** a second training seed of the local-training runs (S = 256 with the jackknife) and of 28x28 Potts 256 +
   both.
