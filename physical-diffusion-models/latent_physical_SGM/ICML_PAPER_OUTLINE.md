# ICML paper on the latent physical SGM: outline, evidence and missing experiments

Written 2026-11-17 in the paper chat. Job states are as of 2026-11-17 21:35.

- **Purpose.** The proposed outline of the paper (claims, figures, tables), what is already established for each
  claim, and what is still missing. **Section 6 lists the missing numerical experiments**; it is the hand-over to
  the experiments chat.
- **Sources.** `MODEL_AND_CODE.md` (model, recipe, best results); the dated entries of `CODE_AND_PERFORMANCE.md`
  (cited as "notebook, date"); `DESIGN.md` (derivations); `README.md` (14x14 results, the K sweep, the first
  on-device results); `CONSISTENCY_CHECK_2026-11-17.md`; `writeups/latent_physical_SGM_writeup/theory.tex`.
- **Not changed.** No code, no job, and neither `MODEL_AND_CODE.md` nor `CODE_AND_PERFORMANCE.md`.
- **Numbers.** "Inception FID" is against the train split at n = 2000, with FID_inf in brackets, unless stated.
  "Classifier FID" is the FID in the features of our small CNN (`fid_score.py`). The two scales are not comparable.
  "Quench 1" is `--quench 1`; "equil start" is `--init-dist equil:3`.

## 0. Current plan (2026-11-21): one short paper

This section is the plan. Sections 1 to 6 record the state of 2026-11-17 and are superseded where they differ.

- **Venue and dates.** arXiv in about two weeks; ICML 2027 (abstract 2027-01-14, paper 2027-01-22). One manuscript
  for both: 8 pages and an appendix (the user: "a short snappy paper").
- **Model (locked by the user).** K16 global Potts unit, 64 states per patch unit, a coarse patch set from t = 4.49
  and a fine one from t = 2.22, Q and per-pixel units, 100 slices with linear interpolation. Data: MNIST, Fashion,
  CelebA-32, CelebA-64. Per data set: exact statistics, on device with S = 64, on device with S = 16. The ODE as
  one ablation. Everything else is stopped.
- **Three claims.**
  1. **Model and rule.** Hidden units summed out, the score is the system's force, and the couplings are learned
     by a local rule from statistics at clamped data, without a negative phase.
  2. **It trains from sampled hidden states:** exact against S = 64 against S = 16 on four data sets.
  3. **Quality:** above earlier generative models for analog hardware on MNIST and Fashion, and the first faces.
- **Page plan.**

  | Section | Pages | Content |
  |---|---|---|
  | 1 Introduction | 0.9 | the problem, three contributions, Figure 1 |
  | 2 Model and learning rule | 1.6 | energy, hidden units, free energy, patch units; the rule in one equation block; what a device measures |
  | 3 Generation | 0.6 | reverse process, time-dependent couplings, quench, corrector |
  | 4 Results | 3.0 | Table 1, Figures 2 and 3, the ablation |
  | 5 Implementation | 0.6 | the mapping to hardware in a short table; what is not shown |
  | 6 Related work | 0.8 | the neighbours of section 12 |
  | 7 Limitations and conclusion | 0.5 | simulation, grey images, parameter count, what "on device" covers |

- **Figures and tables.**
  - Figure 1: concept (pixels, global unit, two patch sets, protocol) with a strip of samples.
  - Figure 2: samples on the four data sets next to real images.
  - Figure 3: exact, S = 64 and S = 16 on the same trajectories, and FID against S.
  - Table 1: four data sets by (exact, S = 64, S = 16), with and without corrector; Wang & Achour and DTM on MNIST
    and Fashion.
  - Table 2 (small): as sampled, quench, corrector, ODE.
- **Appendix.** Derivation of the rule including the patch units; jackknife and visit-count rule; training details
  and compute; memorisation check; the tables of the model search as they were run (K, patch states, Q, onset,
  schedule compression, staircase), each labelled with the model it was run on.
- **Dropped from the main text:** K sweep, Ising against Potts, frozen against trained, oracle starts and
  switch-on scan, compression. One sentence each where needed, pointing to the appendix.
- **Cannot be cut:** the positioning against the closest work, and the statement of what "on device" covers.
- **Open since 2026-11-21 afternoon: the size of the global unit on MNIST and Fashion.** The first locked runs
  (exact, seed 1, noise start, SDE + quench 1; notebook entry "Stage 1 results", Inception 4289737 / 4289739,
  classifier 4289738 / 4289740; galleries `2026-11-21_mnist_interim_...` and `2026-11-21_fashion_interim_...`):

  | | Inception FID | classifier FID | confident share | precision / recall |
  |---|---|---|---|---|
  | MNIST real | 3.70 | 1.69 | 97.8% | 0.87 / 0.88 |
  | MNIST old best, K784 + 7x7 (32 states, from t = 2.22) | 27.86 | 7.66 | 88.8% | 0.81 / 0.80 |
  | MNIST locked, K16 + two patch sets (64 states) | 14.39 (corrector 12.00) | 14.23 | 71.5% | 0.61 / 0.89 |
  | Fashion real | 7.17 | 0.40 | 73.8% | 0.88 / 0.87 |
  | Fashion old best, K784 + 8x8 (32 states, from t = 2.22) | 26.35 | 3.17 | 62.7% | 0.82 / 0.73 |
  | Fashion locked, K16 + two patch sets (64 states) | 26.00 (corrector 25.80) | 5.97 | 49.9% | 0.70 / 0.80 |

  - The user, looking at the galleries: "while the FID is lower with the K16, the old K784 look visibly better". I
    looked at the same grids and agree: K16 strokes are smooth and free of specks, but many more tiles are not a
    digit. On Fashion the difference is smaller: more blended objects with K16.
  - Inception FID on MNIST follows stroke texture, not digit identity. The quench alone takes the K16 model from
    50.56 to 14.39, while the confident share stays at 71.4% -> 71.5% and the classifier FID at 15.7 -> 14.2.
  - The comparison changes two things at once: the global unit (784 -> 16 states) and the patch units (one 7x7 or
    8x8 set with 32 states from t = 2.22 -> two sets with 64 states, the coarse one from t = 4.49). The notebook's
    earlier finding (2026-11-14, blind count) points at the global unit for the shapes: malformed ten-digit samples
    7.8% at K784, 3.0% at K3072, real 0.5%, and "at equal K the patches never changed a shape". K16 is 1.6 states
    per digit class. The clean strokes should then come from the new patch units. This is my reading; the run that
    separates the two (large K with the locked patch units, on MNIST and Fashion) has not been made.
  - Consequence for the paper if the reading holds: the architecture is the same for every data set and K is the
    one size set per data set (16 for faces, larger for the ten-class sets). K16 against large K on MNIST becomes
    the ablation that shows what the global unit does. Figure 3 cannot show the K16 digits next to an FID of 12-14
    without a second number: a reader checks the samples first.
  - Not decided by the user as of this entry. `results.tex` is not filled for MNIST and Fashion until it is.
  - **Result of the test (notebook, "Global-unit test", 2026-11-21 16:00; exact, seed 1, n = 2000, quench 1):**
    the reading holds. With the locked patch units, every metric improves with K.

    | | Inception FID | classifier FID | confident share |
    |---|---|---|---|
    | MNIST, K16 / K128 / K784 | 14.39 / 12.03 / **10.01** | 14.23 / 9.44 / **6.23** | 71.5 / 76.3 / **83.0%** |
    | MNIST, old best (K784 + 7x7) | 27.86 | 7.66 | 88.8% |
    | Fashion, K16 / K128 / K784 | 26.00 / 23.53 / **21.32** | 5.97 / 3.57 / **2.77** | 49.9 / 58.5 / **62.0%** |
    | Fashion, old best (K784 + 8x8) | 26.35 | 3.17 | 62.7% |

    Galleries `2026-11-21_{mnist,fashion}_global-unit-test_...`: K784 with the locked patches draws the same digit
    as the old best on most trajectories, without the grain; K16 and K128 more often an unclear shape. My
    recommendation, as the coding agent's: K784 for MNIST and Fashion, K16 for faces. Cost: about 2.7 M couplings
    per slice at 28x28 instead of about 1.55 M (my count from the unit sizes), 2.2 h instead of 1.4 h. Open risk:
    on-device training of K784 at S = 16 has not been run on these data. In the paper: K is the one size set per
    data set, and the K16 / K128 / K784 rows are the ablation of the global unit. The user's decision is pending.
- **Ablation of the visible springs (2026-11-21 16:20; notebook, "Ablation: do the visible-visible springs
  matter"; CelebA-32, exact, seed 1, n = 2000, quench 1; Inception 4290233).**

  | | Inception FID | as sampled, no quench |
  |---|---|---|
  | locked model | 20.20 | 76.70 |
  | the same without springs (`--pair-terms none`) | 25.83 | 99.31 |
  | pixels only: this paper's visible energy, no hidden units | 218.38 | 317.32 |

  - Gallery `2026-11-21_celeba32_spring-ablation_...`: without springs the same face on every trajectory, with
    speckle from pixel to pixel and blockier edges; pixels only gives one generic face-like texture on every
    trajectory. I looked at the grids and agree with the notebook's reading.
  - For the paper: one decomposition, 218 -> 25.8 -> 20.2 (hidden units give the content, springs smooth
    neighbouring pixels). Two sentences in Section 5, rows in the ablation table, the same-trajectory strip in the
    appendix. Section 6: the device is not purely bipartite; it needs nearest-neighbour couplings between the
    visible units on top of the crossbar of Bacvanski et al.
  - **Reading after the fourth row (2026-11-22; the user: "there is hardly any difference between the images").**
    With springs 20.20, without 25.83, without springs and with a harmonic on-site term only 27.42: the same faces
    on the same trajectories, the difference is pixel grain. Consequences: (i) the visible units can be linear and
    uncoupled, so the device can be purely bipartite, as the Bacvanski et al. circuit; the springs are an optional
    smoothing term (this replaces "the device is not purely bipartite" above); (ii) the oscillator network taken
    over from the first paper is not what makes the model work, and the text and Figure 1a should not suggest it
    is; (iii) 5 to 7 FID correspond to a difference one hardly sees, so FID differences of this size between our
    variants are texture, and the paper should show the same-trajectory strips; (iv) the runs share the seed and
    the k-means start of the hidden units, which is part of why the faces coincide: say so in captions. Open, not
    requested: whether a longer quench or the corrector removes the grain without springs (sampling only).
  - **Does the harmonic, uncoupled row trivialise the paper? (the user's worry, 2026-11-22; my assessment.)**
    (i) Physics: not removed, relocated. With linear visible units all nonlinearity and all content sit in the
    thermal multi-state hidden units; linear hidden units would give a Gaussian. Controls in hand: pixels only 218,
    final energy without the time-dependent protocol fails, frozen templates 147 against 22 to 34 learned.
    (ii) Training: not trivial (the k-means start is far from the trained model; states are lost without the
    visit-count rule), but the model class is classical. Without visible couplings it is a restricted Boltzmann
    machine with Gaussian visible units and categorical hidden units. Neighbour to cite and to read: Liao,
    Kornblith, Ren, Fleet, Hinton, "Gaussian-Bernoulli RBMs Without Tears", arXiv 2210.10318 (single hidden layer,
    4096 hidden units on MNIST, modified contrastive divergence with a Gibbs-Langevin sampler, MNIST FID 17.49 per
    a search summary; Fashion and CelebA-32 numbers not yet read). Added to `references.bib` and related work.
    (iii) Memorisation: capacity argues against it on faces (16 global templates for 162,770 images; the content
    is composed from patch states), but it is not measured on the locked models, and on MNIST with K784 several
    samples were close to their nearest training image (notebook, best-MNIST gallery; the proper check was offered
    and deferred). Needed for the paper, a CPU job on existing samples: distance of each sample to its nearest
    training image against the same for held-out test images, and the share of samples with nearest distance
    below one third of the second-nearest (the criterion of Yoon et al. 2023, used in the memorisation
    literature).
  - **Liao et al. read in full (2026-11-22, PDF v1 of arXiv 2210.10318, 18 pages; replaces the search summary in
    the item above).**
    - Model: E = |(v - mu)/sigma|^2 / 2 - (v/sigma^2)^T W h - b^T h; binary hidden units, each coupled to all
      pixels; one learned variance per pixel, the same for every hidden state; no visible-visible and no
      hidden-hidden couplings; not convolutional. One set of couplings. Hidden units: 4096 (MNIST), 10,000
      (Fashion, CelebA). My arithmetic: 3.2 M weights on MNIST, 7.8 M on Fashion, 30.7 M on colour CelebA-32.
    - Training: likelihood gradient, positive minus negative phase (the Boltzmann rule, local). The negative chain
      starts from Gaussian noise ("modified CD") and runs 100 sampling steps with 10 Langevin steps each, for every
      update; SGD at learning rate 0.01, gradient norm clipped to 10; 3000 epochs at batch 512. No score matching.
    - Sampling: from noise, Gibbs-Langevin (hidden units resampled, then 10 Langevin steps of the pixels at fixed
      hidden state), step size decayed by a cosine schedule "as in score-based models"; 100 steps. One energy, no
      time-dependent couplings.
    - Results: MNIST FID 17.49 (their best; plain Gibbs 47.53; sample count and reference set not stated). Fashion,
      colour CelebA-32 and "CelebA-2K-64" (2000 training images at 64x64): samples only, no FID, no memorisation
      check. Their Fashion samples lack texture (their words); the CelebA-32 samples are smooth colour faces.
    - Hardware: one sentence of motivation (real-valued data for stochastic binary Boltzmann machines); nothing on
      devices, local measurement or sampled statistics.
    - **Their rule on hardware (the user's question, 2026-11-22; my assessment).** Per update: clamp data, sample
      the hidden units once, record v_i h_j / sigma_i^2 (positive); set the pixels to Gaussian noise, run the
      Gibbs-Langevin sampler for 100 outer steps of 10 Langevin steps, record the same product averaged over the
      whole chain (burn-in 0); update by the difference (their Algorithm 3). Good for hardware: a two-factor
      Hebbian rule, simpler per coupler than ours (no sum over the unit's pixels, no clamp-force sensor); binary
      units; one coupling set; the negative phase is the device generating. Costly: one generation run per
      training example and update; the update is a difference of two sampled correlations, small near
      convergence; the chain is not at equilibrium, so the learned energy is tied to this sampler and its
      schedule; fewer chain steps are worse (their Tables 2 and 3: 50 against 100 outer steps 21.0 against 17.5;
      1 / 5 / 10 Langevin steps 35.1 / 19.0 / 17.5). My count from their Appendix B: about 350,000 updates of 512
      chains, 1.8e8 generation runs for MNIST. Ours: about 2.1e6 steps of 512 clamped images over the 100 noise
      levels, 1.1e9 clamps of 16 reads, no generation run. Which is cheaper in device time depends on how many
      hidden reads one generation run is worth; I would not put a number on it in the paper.
    - **Against ours:** same family once our visible couplings are removed. Theirs is ahead in having one coupling
      set and colour. Ours: no negative phase (theirs runs a full generation from noise per update), time-dependent
      couplings as the price; Potts and patch units with local wiring; state-dependent stiffness; MNIST 10.0 exact
      and 11.4 from samples against 17.5 (protocols differ). Stored couplings: ours 2.2 to 2.75 M per noise level
      times 100, theirs 3 to 31 M once. "First faces" only with the qualifier "among models proposed for analog
      hardware".
  - **Bipartite does not mean no communication (the user's point, 2026-11-22).** Hidden units are independent given
    the image and interact through the pixels they share; pixels are independent given the hidden states (without
    springs) and interact through the hidden units they share. Consequences used in the text: the image is the
    medium through which the hidden units agree (one sentence added to Section 3.1); independent pixels given the
    hidden states is why the samples without springs have pixel grain; distant patch units agree only through
    chains of overlapping patches or through the global unit, which fits the result that MNIST needs a large
    global unit.
  - **Can the FID be trusted? (the user, 2026-11-22: the MNIST digits of Liao et al. look better than ours at a
    higher FID, 17.5 against 10.0.)** My assessment after reading `inception_fid.py`:
    - The number is computed in the standard way (original TTUR Inception graph, pool_3, samples clipped and
      quantised to 8 bit, full train split as reference, floor from real test images). Not yet cross-checked with
      a second implementation (offered on 11-21, not run).
    - It is not a measure of how clear the digits are. Own evidence: the quench alone 50.6 -> 14.4 at an unchanged
      share of clear digits; real test images with faint noise 38.5; K16 against the old best: half the FID with
      fewer clear digits. And FID mixes fidelity with diversity: ours has recall 0.88 and precision 0.74 (classifier
      features); a grid shows only fidelity.
    - FIDs of different papers are not comparable without the protocol (Liao et al. and Wang & Achour state no
      sample count).
    - Consequences: within the paper, at one protocol, FID differences stand (exact against sampled, ablations).
      Against other papers: state the protocol, report precision / recall and the clear-class share for MNIST and
      Fashion, show uncurated samples, and do not claim better digits than Liao et al. The MNIST lead over Wang &
      Achour depends on the quench (as sampled about 50 against their 27.6; on 11-15, before the quench, ours was
      52 to 55); on Fashion we were ahead before the quench as well (52 to 57 against 80.8).
    - To close the question, not requested: clean-fid or pytorch-fid on the same sample files; precision and recall
      in Inception features; and, the only rigorous comparison, the public code of Liao et al. evaluated in our
      pipeline.
  - **Binary hidden units again? (the user, 2026-11-22, after Liao et al.) My recommendation: no new runs; frame the
    paper as a dense associative memory turned into a diffusion model trained by a single-phase local rule.**
    The notebook already has both tests (2026-11-14): global binary units lose clearly to one global Potts unit
    (ten digits, classifier FID about 51 to 53 against 22.5; four digits 66% clean against 92%); 32 independent
    bistable units per patch come within 1 classifier FID of a 32-state Potts patch unit (7.35 against 6.42, four
    digits, K784, 7x7 patches, n = 500; `--patch-kind ising`). Reading: mutually exclusive templates are needed at
    the global level, independent features suffice locally. With binary units throughout, the model would be a
    time-dependent version of the machine of Liao et al., and the link to dense associative memories and to the
    circuit of Bacvanski et al. would be lost. Units with values 0 / 1 instead of -1 / +1 are the same model
    family (the difference is absorbed by the on-site linear and quadratic terms); only the parametrisation of
    training changes. In the paper: the binary-unit results as an appendix item labelled "earlier models", and one
    sentence in the hardware section that the patch level may not need a winner-take-all. Optional, not requested:
    one run with binary patch units on the locked CelebA-32 model.
  - **Linear visible model with springs (the user's run, started 2026-11-22; no notebook entry yet when this was
    written).** CelebA-32, harmonic on-site term + linear springs, hidden units as locked. It completes a 2 x 2 on
    CelebA-32: quartic + springs 20.20; quartic, no springs 25.83; harmonic, no springs 27.42; harmonic + springs
    pending. The user's plan: if it performs similarly, keep it as the model; otherwise stay with the quartic
    visible units and discuss the harmonic case briefly. My additions: (i) fix "similar" beforehand: within about
    0.5 FID of 20.2 and the same-trajectory strip indistinguishable (the quartic term was worth 1.6 without
    springs); (ii) check that the free energy stays bounded: in the quartic model the conservative margin
    k_i / 2 - sum of the largest positive Q on pixel i is negative from t of about 1.5 on, so there the x^4 term
    confines; zero diverged samples of 2000 is not proof for a harmonic model; (iii) adopting it means repeating
    the first wave (CelebA-32, MNIST, Fashion: exact, S = 64, S = 16; CelebA-64, the long one), and training from
    sampled states and the ten-class data are untested with linear visible units: test CelebA-32 S = 16 and MNIST
    K784 exact first; (iv) the alternative without reruns: quartic stays the main model and the linear model is
    one ablation row, which already carries the statement that the visible units can be linear.
  - **Decision (the user, 2026-11-22): no memorisation test and no precision / recall metrics** ("on the
    memorization/recall. I think this is not needed"). This closes the corresponding open items above. In the
    paper: the existing panels of samples above their nearest training images, the capacity argument, and the
    existing classifier-based share as a supplementary appendix column; the text says "not copies of their nearest
    training images", never "no memorisation". Still open from that list, unless the user drops it too: the
    cross-check of the FID with a second implementation.
  - **FIDs of other papers: out of Table 1, no ranking (2026-11-22; the user: "we are comparing apples with
    bananas").** Checked in the Wang & Achour paper itself (`relevant-papers/2606.27294v1 (1).pdf`, text search):
    the only statement on the metric is "We report the average Frechet Inception Distance (FID) over four random
    seeds"; no sample count, implementation, reference set or preprocessing. In our protocol the sample count
    alone moves the FID of real MNIST test images from 3.7 (n = 2000) to 11.1 (n = 1000), which is the size of the
    MNIST gap (10 to 11 against 27.6). Their DTM and NLM numbers are their own re-runs "using their official source
    code" in an iso-parameter configuration (DTM: 80x80 grid, 24 couplings per node, 4 diffusion steps, grey
    pixels encoded by three binary units, 60 epochs), not results of the original authors. Changes made: the
    abstract sentence "improves on earlier generative models designed for analog hardware" removed; the block
    "Earlier models for analog hardware" removed from Table 1; Section 5.3 quotes Wang & Achour (and Liao et al.)
    in the text for orientation, says the numbers are not comparable, and does not quote DTM / NLM. Claim 3 is now
    stated against the floor of real images at the same sample count, plus faces "among models proposed for analog
    hardware". A like-for-like comparison would need their sample count and FID code or their samples (ask the
    authors); not planned.
  - **Which model for which data set (the user's proposal, 2026-11-22; not yet decided).** K784 for every data
    set in the main results, and one section ("Compressing the model" / "Smaller models") with K16 and the 5/9/9
    knots on CelebA. My assessment: a good structure, with two conditions.
    (i) The section must also say where the compression fails: on MNIST and Fashion K16 costs 4.4 and 4.7 FID and
    clear digits, and the knots give 234 and about 90. One or two sentences, numbers in the appendix; otherwise it
    is selective reporting.
    (ii) 64x64: K784 there would be three new runs (exact, S = 64, S = 16), the sampled ones longer than the 3 to
    4 days of the running K16 runs, at about 15 M couplings per noise level (my count) instead of about 9 M. I
    would not start them: report 64x64 with K16, inside that section or marked in the table.
    Runs the proposal needs: CelebA-32 with K784 from samples, S = 64 and S = 16, with corrector sets, and ODE +
    quench for the sampler table (exact K784 exists: 19.43, job 4290714). Costs to state: 3.8 M couplings per noise
    level at 32x32 instead of 2.2 M (my count), for 0.8 FID. The section can also hold the other simplifications
    (no springs, harmonic visible units, binary patch units): one place for "what each simplification costs".
  - **Second proposal by the user (2026-11-22): K by a coarse search, per data set; K784 for MNIST, K16 for
    Fashion and CelebA.** My reply: the per-data-set choice by a stated rule is the best option (no new runs), but
    Fashion belongs with MNIST: K784 is worth 4.7 FID there (21.32 against 26.00; clear-class share 62% against
    50%), as much as on MNIST (4.4). Rule proposed for the text: the smallest K within one FID of the largest
    tested, with the patch units fixed; this gives K784 for MNIST (K128 is 2.0 worse) and Fashion (K128 is 2.2
    worse) and K16 for CelebA-32 (0.8 worse than K784). `main.tex` is already written this way; the rule is now
    in the setup bullet. Awaiting the user's decision between this and "K784 everywhere".
  - **Label the pixels-only row correctly:** it is the simplified visible energy of this paper (quartic on-site
    terms, springs to 8 neighbours), not the model of the first paper (all pairs within distance 14, three pair
    terms). It must not be presented as the first paper's result.
  - **FID_inf is not usable as it stands.** In today's tables it exceeds the FID in two rows (pixels only 338
    against 218; MNIST as sampled 58.9 against 50.6), and its distance to the FID varies from 3.7 to 12.6 between
    similar models. A likely cause: 2000 samples are fewer than the 2048 feature dimensions, so the extrapolation
    in 1/N is not in its linear range. Report the FID at fixed n with the floor of real images at the same n, and
    FID_inf only from the 10,000-sample sets.

- **First wave of the lock-in runs (2026-11-22; notebook, "LOCK-IN RESULTS, first wave"; seed 1, n = 2000, noise
  start, Inception FID).** In the paper: K16 for CelebA, K784 for MNIST and Fashion (my assumption for the tables;
  the user ran both).

  | SDE + quench 1 (with corrector) | exact | on device S = 64 | on device S = 16 |
  |---|---|---|---|
  | CelebA-32, K16 | 20.20 (19.95) | 21.20 (20.23) | 21.23 (20.49) |
  | MNIST, K784 | 10.01 (9.19) | 11.37 (9.79) | 11.80 (10.17) |
  | Fashion, K784 | 21.32 (21.82) | 23.44 (23.50) | 22.93 (23.52) |
  | MNIST, K16 | 14.39 (12.00) | 15.22 (12.56) | 15.38 (12.79) |
  | Fashion, K16 | 26.00 (25.80) | 28.29 (27.44) | 27.54 (26.08) |

  - Galleries (`2026-11-21_celeba32_locked-recipe_..._exact-vs-on-device-S64-vs-S16_...`,
    `2026-11-21_{mnist,fashion}_locked-patches-64-states_K16-and-K784_exact-vs-on-device-...`): looked at on
    2026-11-22. Exact, S = 64 and S = 16 draw the same face, digit or garment on almost every trajectory. This strip
    is the strongest figure for claim 2.
  - **Claim 2 wording:** "within 1 to 2 FID of exact training" (0.3 to 0.6 with the corrector on CelebA and MNIST),
    not "matches": the gaps are several bootstrap standard deviations, and Fashion is about +2 in every row.
  - **S = 16 equals S = 64 everywhere.** The gap to exact training therefore does not come from the number of reads.
    Figure 2 as planned (FID against S) would show two flat points per data set. Options: make the same-trajectory
    strip the figure; or add one run with fewer reads (S = 4) to show where sampling starts to cost. Not requested.
  - **To state in the paper:** 1 to 9 of 2000 trajectories diverged for the MNIST models trained on device (none
    exact, none on other data); cause not known.
  - **K:** on CelebA-32, K784 gives 19.43 against 20.20 (control); on MNIST and Fashion it is worth 4.4 and 4.7.
    The K rows are the ablation "the global unit carries the class".
  - **Time compression does not transfer:** post-hoc knots 5/9/9 give 23.65 on CelebA-32 on device (+2.5), 234 on
    MNIST and 88 to 98 on Fashion. The paper must not say that a few coupling sets suffice (section 12, "what is
    ours", item 4, is corrected by this).
  - **Sampler ablation (CelebA-32):** as sampled 76.70 / 79.59, quench 20.20 / 21.20, corrector 19.95 / 20.23,
    ODE + quench 22.14 / 22.93 (exact / S = 64). **Spring ablation, fourth row:** no springs and harmonic on-site
    term only 27.42.
  - **Still open:** CelebA-64 (exact being resampled after an out-of-memory failure; on device S = 64 about 4 days,
    S = 16 about 3 days from 2026-11-22 10:10); the second wave (seeds 2 and 3, 10,000 samples), which should include
    seeds of the exact runs, because the gap to exact is the claim; the parameter counts (my count 2.24 M at 32x32
    with K16, the notebook says about 2.4 M; 2.75 M at 28x28 with K784).
  - `results.tex` is filled with these numbers (one decimal); `main.tex` Sections 3.1, 5 and 6 are updated.

## 1. Thesis and scope

**Thesis.** A diffusion model whose score is the force of a physical free energy becomes a competitive image
generator once fast hidden units are added in one bipartite layer and summed out exactly. Training stays a local
rule on clamped data, and generation is a physical protocol: relax, anneal, quench.

**Working title.** *Physical diffusion models with fast hidden units.*

**Relation to the first paper** (arXiv 2506.19136, no hidden units, being resubmitted separately). Its model is the
"pixels only" baseline here. For double-blind review it is cited in the third person, and the overlap is limited to
the background section.

**Proposed weight of the claims** (decision 1 in section 8): image quality from a hardware-compatible energy model
leads; training from device-measurable statistics is the supporting result, because its decisive runs are still in
the queue.

## 2. Claims: what is established, what is missing

Experiment labels (R1, A4, B5, ...) refer to section 6.

### Claim 1. The model and its learning rule

**Statement.** A global Potts unit, one Ising unit per pixel and Potts units on overlapping patches form one
bipartite layer that sums out exactly, so the pixels move in a free energy F. The score-matching gradient of every
hidden coupling is a local rule that needs only statistics measured while the pixels are clamped at data: no
free-running negative phase and no backpropagation.

**Established.**
- F, its gradient and its Laplacian in closed form (`DESIGN.md` sections 1-2, `theory.tex`). Checked against
  autodiff: `check_hidden_unit_formulas.py`; patch terms in `test_latent.py` block 19 (gradient 2e-16, Laplacian
  4e-15).
- The local force-matching rule for the global Potts unit with Q and for the per-pixel units (`DESIGN.md` 2.1): a
  three-factor rule per coupler. `latent_model.force_matching_local_grad` equals `jax.grad` of the training loss to
  1e-16 to 4e-11 on trained models (jobs 4197172, 4197173).
- Generation on a device: duplicating every hidden unit gives 2F exactly, at one temperature (`DESIGN.md` 3). The
  conditions on the hidden time scale are in `DESIGN.md` 4. Theory only.

**Missing.**
- The local rule for the patch units. `force_matching_local_grad` raises `NotImplementedError` for `--patch`, so
  the rule is neither written nor checked for the recipe (A6).
- Any numerical check of the replica construction, or of a finite hidden switching time. Every result is in the
  fast-hidden limit (B5).
- Text: for each factor of the rule, what a device measures (clamp force, occupations, nudge projections).

### Claim 2. The hidden units carry the model, and they must be learned mixtures

**Statement.** Without hidden units the physical model does not generate ten digits. One large Potts unit does, and
quality rises with its number of states. Independent global Ising units saturate early, and prototypes that are set
from data instead of learned (the dense-associative-memory analogue) are far worse.

**Established.**
- **K sweep**, ten digits, no patches, noise start, as sampled, classifier FID (README, "The main result",
  FID 4209955; K6144: notebook 11-12, FID 4229562; floor 1.69):

  | hidden units | 0 | 64 | 256 | 784 | 1536 | 3072 | 6144 |
  |---|---|---|---|---|---|---|---|
  | Potts states | 197.5 (pixels only 199.1) | 67.6 | 41.7 | 31.1 | 26.0 | 22.5 | 19.8 |
  | Ising units | | 87.5 | 63.7 | 53.2 | 51.2 (1568) | 52.3 (3136) | |
  | Potts, frozen at prototypes | | | | | 66.0 | 56.1 | |

- **Frozen against trained in Inception FID.**
  - MNIST ten digits, K784, no Q, no patches: 204.9 frozen against 151.4 trained; clean yield 67.4% against 84.6%
    (notebook 11-16; Inception 4263135, classifier 4263139).
  - CelebA-32, K1536 + 8x8 + 16x16, quench 1: 147.0 frozen (patches at reference / 16), 34.4 trained without Q,
    22.4 with the recipe (notebook 11-17; Inception 4266724).
- **Returns from K shrink once patches are present.** MNIST classifier FID 8.94 / 6.37 / 5.35 at K784 / 3072 / 6144;
  Fashion and CelebA-32 saturate at K1536 and K784. Only about 600 of 784, 1,400 of 3,072 and 1,800 of 6,144 states
  are used (notebook 11-14, "Where we stand").

**Missing.**
- The sweep in the paper's metric. With quench 1, the Inception FID exists only for K784 (86.3, equil start) and
  K6144 (57.8, noise start). The rest needs a quench and a re-score of existing samples (A4).
- The Ising rows had smoother strokes and Inception weighs texture, so the Potts margin may shrink in this metric.
  Check before the text relies on it (A4).
- One seed for every point of the sweep (A3 covers the headline model only).
- With patches on ten digits: K64, K256 and K1536 are not trained (A4).
- Frozen against trained for the full structure on MNIST (B2).
- Wording: "each pixel counted once" for the frozen patch baseline is overstated (consistency check 1.1).

### Claim 3. Patches set the texture, the global unit sets the shape, and timing decides whether they cooperate

**Statement.** Potts units on overlapping patches, in the same bipartite layer, supply the local stroke structure
that a mixture of whole-image templates lacks. They must switch on after the global unit has fixed the shape
(t of about 2.2); earlier they override it.

**Established.**
- **The gain, three data sets** (K784, quench 1, Inception FID): MNIST 86.3 to 27.7; Fashion 96.0 to 36.5 (7x7) and
  26.9 (8x8 every 2 px); CelebA-32 104.4 to 24.7 (notebook 11-15, figure update 4256062; CelebA 4256037).
- **Structure, not parameter count** (four digits, classifier FID, notebook 11-13, FID 4236364): the
  coupling-matched K912 gives 19.1 and K1536 16.8, against 10.6 with patches on K784.
- **Patches and quench are not interchangeable.** MNIST K6144 without patches, quench 1: 57.8. With patches, same
  start: 18.0 (notebook 11-15; Inception 4256048).
- **Switch-on time** (four digits, K784, seed 1, noise start; notebook 11-13, FID 4237627):

  | switch-on | never | t = 3.4 | t = 2.7 | t = 2.2 | t = 1.5 |
  |---|---|---|---|---|---|
  | classifier FID | 20.25 | 10.55 | 8.17 | 7.68 | 9.66 |
  | precision / recall | 0.94 / 0.42 | 0.79 / 0.85 | 0.84 / 0.83 | 0.90 / 0.80 | 0.96 / 0.69 |

  Ten digits agree: t = 2.2 gives 8.94, t = 2.7 gives 10.99 (equil start, FID 4238365).
- **Where the error forms** (oracle starts, four digits, K784 without patches, mean of three seeds; FID 4229215):
  FID 17.5 if the model takes over at t = 2.7, and 6.8 at t = 1.5 (floor 4.6). Above t = 2.7 the model adds about
  1.3.
- **The patch layout changes texture only.** 64 states or 8x8 every 2 px give the same shapes and the same defects
  sample by sample (FID 4238578; Fashion sweep 4247150).
- **Seeds.** Ten digits, K784 + patches: 8.87 +- 0.10 over three seeds (FID 4240132).
- **Ising patch units** come within 0.9 classifier FID of Potts patch units (7.35 against 6.42, four digits,
  FID 4244042).
- **More patch scales** on CelebA-32 (quench 1): 24.7 with 8x8; 23.3 with 4x4 added; 22.9 with 16x16 added; 22.3
  with all three (Inception 4261206, 4262955).

**Missing.**
- The mechanism evidence (switch-on scan, coupling-matched control, oracle starts) is four digits, n = 500,
  classifier FID, before the quench, mostly seed 1. Either present it as such, or repeat it on ten digits with the
  final protocol (B1).
- The malformed-shape rates (real 0.5%; ten digits K784 + patches 7.8%, K3072 + patches 3.0%) come from three AI
  raters (notebook 11-14, workflow wf_eb635ec0-643). For the paper: human raters, or a clear label, or the
  classifier's confident share instead (section 7).

### Claim 4. Generation is a three-step physical protocol, and each step is needed

**Statement.** Relax the noise state under the t = 5 energy, run the annealed reverse SDE, then switch the bath off
for a short quench. Each step is an operation a device can do, and none needs data.

**Established.**
- **The time evolution is essential.** Langevin dynamics at the fixed t = 0 energy gives classifier FID 456 to 791,
  against 8.9 to 10.2 for the annealed process; every chain falls into a blank basin (notebook 11-15, FID 4254858).
- **Noise is needed.** The probability-flow ODE gives 22.1 against 10.2 on MNIST and 6.7 against 4.1 on Fashion
  (FID 4254964, 4254967).
- **Equilibrated start.** About 1 classifier FID through class balance, with no data: four digits 7.33 to 6.24,
  ten digits 10.20 to 8.87, three seeds each (notebook 11-12 and 11-14).
- **Quench 1, the largest single gain** (Inception FID): MNIST K6144 + patches 55.9 to 18.8; Fashion K1536 + 8x8
  55.9 to 25.9; CelebA-32 K784 + 8x8 98.4 to 24.7 (Inception 4255378, 4255381, 4256037).
- **No memorisation.** No sample lies within distance 1 of a training image; 1 to 3% of the MNIST and Fashion
  samples are closer to the training set than the 5th percentile of real test images, and 6 to 7% on CelebA after
  the quench (`nn_check.py`, jobs 4255018, 4255019, 4256039).
- **Corrector.** Two unadjusted Langevin steps per slice: CelebA-32 22.37 to 21.50 (Inception 4267210).
- **Solver.** A 10x tighter tolerance moves a sample by a few millionths of the distance between samples (jobs
  4266909, 4266910).

**Missing.**
- One protocol for all headline numbers. MNIST and Fashion use the equilibrated start, the best CelebA models the
  noise start, and the corrector was tried on CelebA only (A1; decision 2).
- After the quench the equilibrated start changes the Inception FID by 0 to 1 only (MNIST K6144: 18.8 equil against
  18.0 noise). Its effect is on class balance (MNIST classifier FID 3.87 against 4.95). The text and the ablation
  table must show it in the metric that sees it.
- The quench duration was chosen on the reported metrics at K784 and then used unchanged elsewhere. Say so; a scan
  on CelebA would show it is not tuned per data set (C2).
- Text: what "switching the bath off" means at a finite cooling rate. Long quenches push pixels outside [0, 1].

### Claim 5. Quality

**Statement.** With the recipe, the model generates digits, clothing and faces, at a quality above earlier
generative models built for analog hardware.

**Established** (seed 1 unless stated):

| Data | Model | Inception FID (FID_inf) | Floor | Wang & Achour |
|---|---|---|---|---|
| MNIST, ten digits | K6144 + 7x7 every 3 px, equil start, quench 1 | 18.8 (15.9) | 3.7 | 27.6 |
| Fashion, ten classes | K1536 + 8x8 every 2 px, equil, quench 1, 3 seeds | 25.9 +- 0.04 (21.3) | 7.2 | 80.8 |
| CelebA 32x32 grey | K1536 + 8x8 + 16x16, noise start, quench 1 | 22.4 (18.6) | 6.4 | - |
| CelebA 32x32 grey | the same with 2 corrector steps | 21.5 (18.0) | 6.4 | - |
| CelebA 64x64 grey | K1536 + 16x16 every 4 px, noise, quench 1 | 58.5 (55.9) | 5.9 | - |

- Classifier FID: MNIST 3.87, Fashion 2.71 +- 0.07.
- Wang & Achour (arXiv 2606.27294) also report their re-runs of DTM (107.8 / 112.8) and NLM (230.5 / 200.8). Their
  n and FID code are not stated; theirs is trained by a GAN loss with backpropagation through the solver.
- Training cost: about 4 h on one L40 for MNIST K6144 + patches; 1.8 h for CelebA-32 K784 with three patch scales.

**Missing.**
- Sample count: everything is n = 2000, and FID_inf is an extrapolation from n <= 2000 (A2).
- Seeds for MNIST K6144 + patches and for the best CelebA-32 model (A3).
- A digital reference on the same grey data and metric (A7).
- Memorisation check on the final headline sample sets (A8).
- The parameter count. The best MNIST model stores 9.8 M couplings per slice (W + Q) over 100 slices, about 10^9
  numbers, against 326k in total for Wang & Achour. K784 + patches (1.4 M per slice) reaches 27.7, level with them
  (B7, parked).
- Contrast on faces: 13% too low at 32x32 and 23% at 64x64 (R3).
- CelebA 64x64 is one run; the K3072 evaluation failed (R4, C1).

### Claim 6. Training from device-measurable statistics

**Statement.** The exact hidden averages in the learning rule can be replaced by S sampled hidden states per
clamped image, with a jackknife correction and a visit-count step rule, at little cost in sample quality.

**Established.**
- **Estimator** (CelebA-32 K784 + 8x8, `check_sampledpatch.py`, job 4267958): the loss is unbiased; the gradient
  error falls as 1/sqrt(S); gradient cosine with the exact one at S = 16 / 64 / 256 is 0.91 / 0.98 / 0.99 for the
  patch couplings and 0.88 / 0.96 / 0.99 for the global unit.
- **CelebA-32, K784, no patches, quench 1** (Inception 4267290):

  | hidden statistics, optimiser | Inception FID (FID_inf) |
  |---|---|
  | exact, Adam | 104.4 (100.6) |
  | exact, visit-count rule | 106.4 (102.2) |
  | S = 64, jackknife, visit-count rule | 109.1 (105.3) |
  | S = 256, jackknife, visit-count rule | 110.1 (105.9) |

  The 3 to 4 points against exact statistics do not shrink with S, so they are not sampling noise.
- **Small problems** (README, Status block and Result 10; clean yield): with the visit-count rule, S = 16 matches exact training on
  digits 0/1 at 28x28 (87.1% against 86.3%), and S = 64 on digits 0/1/2 at 14x14 (80.1% against 81.1%).
- **Contrastive divergence** with sampled hidden states is pure noise; force matching reads the hidden units at one
  clamped configuration (`DESIGN.md` 2).

**Missing.**
- The recipe with patch units: running (R1).
- A second seed, to tell run-to-run variance from a systematic gap: running (R2).
- A second data set with the recipe (A5).
- The one negative result was never repeated: four digits, Potts 512 at 14x14, S = 64 gave 74.7% against 85.0%
  (README Result 14). A5 re-tests it on ten digits.
- A curve of quality against S (B3).
- Only the sampling of the hidden states is simulated. The clamp force and the nudge projections are exact (B6),
  and Adam on the patch couplings keeps two extra numbers per coupler. The text must say what "on device" covers.

### Claim 7 (optional; parked on 2026-11-15). Parameter efficiency

Nothing is run. The sampling-only tests (fewer time slices, pruning unused states, low-bit weights) and small K
with patches would answer the first objection to claim 5 (B7; decision 4).

## 3. Figures

| # | Content | Data | Status |
|---|---|---|---|
| 1 | Concept: pixels with nearest-neighbour springs; global, patch and per-pixel hidden units; the protocol (relax, anneal with late patches, quench); the local rule | - | to draw (start from `writeups/njp_uc/figures/conceptual_draft.pdf`) |
| 2 | Uncurated samples next to real images and nearest training images: MNIST, Fashion, CelebA-32 | final sample sets (A1, A2); `nn_check.py` | samples exist; composite to make |
| 3 | Ingredient ladder per data set: pixels only, global unit, patches, start, quench, corrector | values below | pixels-only point missing (A4); current version: `out/latent_physical_SGM/figures/inception_fid_mnist_fashion.pdf` |
| 4 | Quality against K: Potts, Ising, frozen; with and without patches | A4 | only in classifier FID before the quench |
| 5 | Timing: (a) oracle-start curve, (b) FID, precision and recall against switch-on time, (c) the same trajectories with patches from t = 3.4 and t = 2.2 | notebook 11-12, 11-13 | data exist on four digits; B1 for ten digits |
| 6 | Training from sampled statistics: (a) gradient error against S, (b) FID for exact, S = 64, S = 256, with and without patches, (c) the same faces | 4267958, 4267290, R1 | (b) with patches pending |

**Values available for Figure 3** (Inception FID, as sampled / quench 1):

| MNIST (equil start) | | Fashion (equil start) | | CelebA-32 (noise start) | |
|---|---|---|---|---|---|
| K784 | 112.8 / 86.3 | K784 | 131.1 / 96.0 | K784 | 238.0 / 104.4 |
| K784 + 7x7 | 66.2 / 27.7 | K784 + 7x7 | 70.0 / 36.5 | K784 + 8x8 | 98.4 / 24.7 |
| K3072 + 7x7 | 58.3 / 21.1 | K784 + 8x8 | 56.7 / 26.9 | K784 + 8x8 + 16x16 | 93.6 / 22.9 |
| K6144 + 7x7 | 55.9 / 18.8 | K1536 + 8x8, 3 seeds | 55.9 / 25.9 | K784 + 8x8 + 4x4 + 16x16 | 75.4 / 22.3 |
| K6144, no patches (noise start) | 89.1 / 57.8 | | | K1536 + 8x8 + 16x16 | 92.6 / 22.4 |

## 4. Tables

1. **Comparison with generative models for analog hardware** (MNIST, Fashion): Inception FID, number of couplings
   (W + Q counts from `MODEL_AND_CODE.md`), time slices, how the model is trained (local rule on clamped data
   against backpropagation through the solver). Rows: ours at K784 + patches and at the best K; Wang & Achour
   (best, 4 bit + noise); DTM; NLM; the digital reference (A7). Take DTM's and NLM's own published numbers as well
   as Wang & Achour's re-runs.
2. **Ablation on three data sets**, if Figure 3 does not carry it: global unit, patches, start, quench, corrector,
   with FID, precision and recall.
3. **Appendix:** the recipe and its flags; parameter counts; training and sampling cost per model.

## 5. Section plan (8 pages of main text)

| Section | Pages | Content |
|---|---|---|
| 1 Introduction | 1 | physical generative models; what limits them (pair energies, non-local training); the three contributions |
| 2 Background | 0.5 | energy-based score model of the first paper; force matching per time slice |
| 3 Model and learning rule | 1.5 | hidden units summed out; global, per-pixel and patch units; the local rule; generation with replicas (claim 1) |
| 4 Sampling protocol | 0.75 | equilibrated start, annealing with late patches, quench (claim 4) |
| 5 Results | 3.25 | quality and comparison (claim 5); what each ingredient does and capacity (claims 2, 3); timing (claim 3); sampled statistics (claim 6) |
| 6 Related work | 0.5 | analog and thermodynamic generative hardware; RBMs and score matching with latent variables; dense associative memories; local learning |
| 7 Limitations and outlook | 0.5 | section 9 below |

**Appendix:** derivations (free energy, local rule, jackknife, visit-count rule, replicas, time-scale conditions);
training details; convergence and seed studies; solver convergence; memorisation; contrast analysis; negative
results (harmonic modes, state splitting, deep tree, global Ising fixes).

## 6. Missing numerical experiments

**Common protocol for every new number.** Inception FID against train with `inception_fid.py`, FID_inf, quench 1,
and the start chosen in decision 2. Classifier FID, precision / recall and class shares as secondary measures on
MNIST and Fashion. Paired comparisons use the same trajectory set.

**Priorities.** R = already running. A = the paper needs it. B = a reviewer will probably ask. C = optional.
GPU hours are estimates from the timings in the notebook, on one L40.

### 6.1 Overview

| ID | Experiment | Serves | New training | GPU h (estimate) | Status |
|---|---|---|---|---|---|
| R1 | On-device training with patch units, CelebA-32 | claim 6, Fig. 6 | yes | running | 4268000 / 01 / 85 / 86 running |
| R2 | On-device training without patches, seed 2 | claim 6 | yes | running | 4268401 / 02 running |
| R3 | Global factor units against the contrast deficit | claim 5 | yes | running | 4268394 / 95 running |
| R4 | CelebA-64 K3072 | claim 5 (appendix) | done | about 16 | **sampling 4266810 failed; chain stuck** |
| A0 | Evaluation fixes before the final numbers | all | no | 0 (code) | not started |
| A1 | One sampling protocol for the headline models | claims 4, 5 | no | 3 to 20 | waits for decision 2 |
| A2 | Headline numbers at 10,000 samples | claim 5, Table 1 | no | 45 to 60 | not started |
| A3 | Seeds 2 and 3, MNIST and CelebA-32 | claim 5 | yes | about 35 | not started |
| A4 | Ladder and K sweep in Inception FID with quench | claims 2, 3, Figs. 3, 4 | 3 runs | about 10 | not started |
| A5 | On-device training with the recipe, MNIST ten digits | claim 6 | 3 runs | 15 to 20 | not started |
| A6 | Local rule for patch units against autodiff | claim 1 | no | CPU | not started |
| A7 | Digital reference on the same data and metric | claim 5, Table 1 | yes (new code) | 10 to 15 | not started |
| A8 | Memorisation and variety of the final sample sets | claims 4, 5 | no | CPU | not started |
| B1 | Mechanism on ten digits with the final protocol | claim 3, Fig. 5 | 3 runs | about 6 | not started |
| B2 | Frozen against trained, full structure, MNIST | claim 2 | 2 runs | about 3 | not started |
| B3 | Quality against S | claim 6, Fig. 6 | 2 runs | about 7 | not started |
| B4 | Weight precision and coupling noise, sampling only | claim 5, Table 1 | no (new code) | 6 to 8 | not started |
| B5 | Finite hidden time scale and replicas, small model | claim 1 | new code | small | not started |
| B6 | Measurement noise in the learning rule | claim 6 | 2 to 3 runs | about 8 | not started |
| B7 | Parameter efficiency, sampling only | claim 7 | no | 3 to 5 | parked 2026-11-15 |
| C1 | CelebA-64: second run and second patch scale | claim 5 (appendix) | yes | 20+ | after R4 |
| C2 | Quench duration on CelebA-32 | claim 4 | no | 1 | not started |
| C3 | Slice 0: hidden units held fixed | appendix | 1 run | 1 to 2 | not started |
| C4 | Ising patch units on ten digits, seeds | hardware argument | 3 runs | about 4 | not started |
| C5 | Colour (CelebA-32 RGB) | scope | new code | unknown | not planned |

### 6.2 Already running or submitted

**R1. On-device training with patch units** (notebook 11-17 18:30). CelebA-32, K784 + 8x8 every 2 px + 16x16 every
4 px, seed 1, n = 2000, noise start.
- Runs: exact statistics with the visit-count rule for the global unit and Adam for the patches (4268000); S = 64
  sampled for all units (4268001); the same two with the visit-count rule for the patch states as well,
  `--patch-occupancy` (4268085 exact, 4268086 S = 64).
- Evaluation: quench 4268002 / 03 and 4268087 / 88; Inception FID 4268004 and 4268089 (all five variants);
  galleries 4268005 and 4268090. Reference: exact statistics with Adam, quench 1: 22.85.
- **Reading.**
  - S = 64 within the seed spread of exact statistics with the same optimiser: claim 6 covers the recipe.
  - A gap of a few points that does not shrink with S, as without patches: state it as "within x", and let R2
    say whether it is run-to-run variance.
  - A large gap: claim 6 is limited to the global unit, and the patch units are trained with exact statistics.

**R2. Seed 2 without patches** (notebook 11-17 20:05). K784, visit-count rule: exact 4268401, S = 64 4268402;
quench 4268403 / 04; Inception FID 4268405 (seeds 1 and 2 of both).

**R3. Global factor units against the contrast deficit** (notebook 11-17 20:05). K784 + 8x8 + 16x16 plus a third
set: `--patch3 32:32:32` (4268394) or `--patch3 24:4:32` (4268395); quench 4268396 / 97; Inception FID 4268398;
contrast check 4268399; gallery 4268400.

**R4. CelebA-64, K3072 + 16x16 every 4 px.**
- Training 4266799 finished (10 h 23 min).
- **Sampling 4266810 failed after 5 minutes with exit code 139.** The quench 4266811, Inception FID 4266812,
  gallery 4266813 and contrast check 4266814 are pending with `DependencyNeverSatisfied`, so they will never start.
- Needed: find the cause, then resubmit the sampling (`--history-dtype float32`, platform allocator, chunk 25) and
  the four follow-up jobs.

### 6.3 Priority A: needed for the paper

**A0. Evaluation fixes before the final numbers** (code only; consistency check, section 3).
- `inception_fid.py` passes one `default_rng(0)` through the floor, the diagnostics and every set in turn. FID_inf,
  KID and the bootstrap sd of a set therefore depend on its position in the job: the same samples gave 18.6, 18.7
  and 18.8. Give each set its own generator.
- `N_GRID` of the FID_inf fit ends at 2000. Extend it for larger sets (A2).
- KID is computed against the test features. Label it so, or compute it against train.
- **Reading.** After this, differences in FID_inf below about 0.5 stop being job noise. Until then, treat such
  differences as ties (for example 22.3 against 22.4 on CelebA-32).

**A1. One sampling protocol for the headline models.**
- **What.** The three headline models under one protocol. Proposed: equilibrated start, SDE, quench 1 (decision 2).
- **Missing runs.**
  - CelebA-32: the best models have noise-start samples only. Equilibrated start and quench 1 for
    `celeba32g_K1536_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch8s2S32+patch16s4S32on55_seed1` and for
    the K784 run with three patch scales.
  - If the corrector joins the protocol: `--corrector 2:0.16` for MNIST K6144 + patches and for Fashion
    K1536 + 8x8 (three seeds). Otherwise it stays one ablation row on CelebA.
- **Known.** Rows that exist: MNIST K6144 noise + quench 18.0, equil + quench 18.8; CelebA K784 + 8x8 noise +
  quench 24.7, equil + quench 23.8.
- **Cost.** CelebA: 2 to 3 GPU h per model. Corrector on MNIST K6144: about 9 h per 2000 samples (the segmented
  sampler took 1.3x the standard one on CelebA).

**A2. Headline numbers at 10,000 samples.**
- **What.** 10,000 samples for each headline model under the A1 protocol; Inception FID at n = 10,000, FID_inf,
  KID, precision and recall.
- **Why.** Every number is n = 2000, where the FID is biased upward by about 3 to 7. Papers quote 10k to 50k
  samples, and Wang & Achour do not state their n.
- **How.** The sampler has two trajectory sets, `--set base` and `--set ext` (seeds 2026 and 2027), and a set of
  n = 10,000 is one draw: the existing 2,000 samples are not its first 2,000. Either run `--n 10000`, or add a seed
  argument and concatenate sets of 2,000 with `PATH+PATH`.
- **Cost.** MNIST K6144 + patches: 10 to 13 s per sample at chunk 25, so 27 to 35 GPU h. CelebA-32
  K1536 + 8x8 + 16x16: about 9 GPU h. Fashion K1536 + 8x8: about 8 to 10 GPU h (not timed in the notebook).
- **Reading.** If the FID at 10,000 agrees with the FID_inf fitted from n <= 2000 to about 1, all ablation rows can
  stay at n = 2000 with FID_inf. If not, the rows of Table 2 need more samples too.

**A3. Seeds 2 and 3 for MNIST and CelebA-32.**
- MNIST K6144 + patch7x7 from t = 2.2: training about 3.9 h per seed, sampling about 7 h per 2000 with the
  equilibrated start. About 22 GPU h.
- CelebA-32, the model that A1 keeps: training 1.8 h (K784, three scales) or 3 to 6 h (K1536 + 8x8 + 16x16; 6.2 h
  before the block unfold), sampling 2 to 2.5 h. About 10 GPU h.
- 2,000 samples per extra seed are enough; 10,000 only for seed 1.
- **Known spread.** Fashion K1536 + 8x8: 25.9 +- 0.04 over three seeds. MNIST K784 + patches: classifier FID
  8.87 +- 0.10.

**A4. The ingredient ladder and the K sweep in the paper's metric.**
- **Part 1, no retraining.** Quench 1 and Inception FID for the existing 2,000-sample sets of the 2026-11-10
  ten-digit sweep: pixels only; K = 0 with per-pixel units; Potts 64, 256, 1536, 3072; Ising 64, 256, 784, 1568,
  3136; frozen Potts 1536 and 3072. Potts 784 and 6144 are done (86.3 and 57.8).
- **Part 2, training.** The missing points with patches on ten digits: K64, K256 and K1536 + patch7x7 from
  t = 2.2. K784, K3072 and K6144 exist (27.7, 21.1, 18.8).
- **Why.** Figures 3 and 4. Everything about K, Ising against Potts, and frozen against trained on MNIST is in the
  classifier FID before the quench.
- **Starts.** The sweep has noise-start samples only, so quench those. Use one start for the whole with-patches
  curve.
- **Watch.** The global Ising rows had smoother strokes, and Inception weighs texture. The Potts margin may be
  smaller in this metric.
- **Cost.** Quench and Inception take minutes per set at small K; K3072 and K6144 need the platform allocator and
  160 GB. Training K64 / K256 / K1536 with patches: about 0.8 / 0.9 / 1.5 h, plus sampling.
- **Also gives** the small-model end of the quality-against-parameters curve (B7).

**A5. On-device training with the recipe on MNIST, ten digits.**
- **What.** K784 + patch7x7 from t = 2.2, mixed precision, seed 1, n = 2000:
  1. exact statistics, Adam (exists: 27.7 with the equilibrated start and quench 1);
  2. exact statistics, visit-count rule (`--hidden-norm occupancy`; with `--patch-occupancy` if R1 favours it);
  3. S = 64 sampled (`--hidden-samples 64`, jackknife, `--occ-same-draws`);
  4. S = 16 sampled.
- **Why.** On the recipe, claim 6 would rest on CelebA-32 alone. Ten digits also re-test the one negative result
  (four digits, Potts 512 at 14x14: 74.7% against 85.0% clean yield, README Result 14), which was never repeated
  with more states, patches or 28x28.
- **Cost.** On CelebA a patch slice with S = 64 takes about 3x the exact one (310 s against 110 s). Expect 4 to 9 h
  per run.
- **Reading.** Equal within the seed spread: the claim holds on digits and faces. A gap on MNIST only: report it as
  a dependence on the number of classes and scope the claim.

**A6. The local rule for the patch units, checked against autodiff** (code and a CPU job).
- Extend `latent_model.force_matching_local_grad` to the patch sets (it refuses `--patch` now); the clamp force
  must include every hidden force.
- Compare with `jax.grad` of the training loss through `cd1_check.py --local-rule` on a trained recipe model, for
  example CelebA-32 K784 + 8x8 at slices 60 and 85 (the slices `check_sampledpatch.py` uses), and on a model with
  two patch scales.
- **Pass:** relative differences at rounding level in float64, as for the global unit (1e-16 to 4e-11).

**A7. A digital reference on the same data and metric** (new code).
- **What.** A standard small diffusion model (a U-Net of about 1 to 2 M parameters) trained on the same images:
  MNIST and Fashion 28x28, CelebA-32 grey with the same crop and split. Sampled at the same n and scored by the
  same `inception_fid.py`.
- **Why.** It places our numbers on the scale the ICML audience knows. No published FID exists for our grey
  CelebA crop.
- **Expected.** Far better than ours. The paper's claim is about hardware-compatible models; the reference shows
  the size of the remaining gap honestly.
- **Also worth checking in the literature:** FIDs of Gaussian-Bernoulli RBMs on MNIST, Fashion and CelebA (Liao et
  al. 2022, as I recall; verify), the closest digital model with one bipartite hidden layer.

**A8. Memorisation and variety of the final sample sets** (CPU).
- `nn_check.py` on the final headline sets. For MNIST K6144 + patches there is only an informal look at 12 samples
  (notebook 11-14 21:30); the CelebA check was on the first K784 and K1536 + 8x8 runs.
- Precision and recall for the corrector rows on CelebA: the corrector faces look "more frontal and average"
  (notebook 11-17 evening).

### 6.4 Priority B: probable reviewer requests

**B1. The mechanism on ten digits with the final protocol.**
- Switch-on scan at K784: t = 3.4 and t = 1.5 are missing (slices 31 and 70); never, t = 2.7 and t = 2.2 exist.
- Coupling-matched control: K912 without patches against K784 + patch7x7.
- Oracle starts (`--oracle-start K`) for K784 and K784 + patches at slices 5, 19, 31, 45, 55, 70: where does the
  error form once the patches are in?
- All with quench 1, Inception FID, precision and recall. Three trainings of about 1 h, plus sampling.
- **Reading.** Figure 5 on the paper's data set and metric. If the quench changes the precision / recall
  trade-off, the switch-on argument is restated on shapes (confident share), not on FID.

**B2. Frozen against trained for the full structure on MNIST.**
- K784 + patch7x7, ten digits: frozen at the prototype references without Q (`--hidden-fixed`, with
  `--patch-fixed-scale`), and trained without Q. Trained with Q exists.
- Note: one scale per patch set cannot equal 1 / coverage for every pixel (7x7 every 3 px has coverage 4 to 9,
  1 at the corners); state the scale used (consistency check 1.1).
- **Why.** The full-structure comparison exists on CelebA only (147 / 34 / 22).

**B3. Quality against S.**
- CelebA-32, K784, no patches, visit-count rule: add S = 4 and S = 16 to the existing S = 64 and S = 256. If R1
  is good, do the scan on the recipe instead.
- **Reading.** Figure 6(b) as a curve. On digits 0/1 the rule held down to S = 4 and broke at S = 2 (README
  Result 10); the paper should show where it breaks on a hard data set.

**B4. Weight precision and coupling noise, sampling only** (needs a sampler option that loads modified parameters).
- On K784 + patches (MNIST) and K1536 + 8x8 (Fashion): couplings quantised to 8 / 6 / 4 bits; static relative
  errors of 1%, 3%, 10%; per-read noise.
- **Why.** `HARDWARE_NOTES.md` estimates 9 to 11 bits near t = 0, by reasoning only. Wang & Achour report a
  "4 bit + noise" row (29.1 / 75.1), and Table 1 should have ours.
- **Reading.** The measured precision requirement goes into the limitations; a large loss at 6 to 8 bits is a
  result to report, not to hide.

**B5. Finite hidden time scale and the replica construction, on a small model** (new code).
- A joint simulation with explicit hidden variables: the pixels as Langevin variables, the hidden units as
  stochastic two-state or Potts variables with a finite switching rate, two copies of each hidden unit at T for
  generation (`DESIGN.md` 3).
- On 14x14 digits 0/1 or a small ten-digit model: sample quality against the ratio of hidden to pixel rates,
  compared with the exact-marginal sampler.
- **Why.** It is the only test of the idealisation that every result rests on. `DESIGN.md` 4 predicts friction but
  no bias with replicas.
- **Reading.** A plot for the appendix and one sentence in the main text: how fast the hidden units must be.

**B6. Measurement noise in the learning rule.**
- The rule also needs the clamp force and the nudge projections, which are computed exactly now. Train CelebA-32
  K784 without patches (S = 64, visit-count rule) with Gaussian read noise on the clamp force at 1%, 3% and 10%.
- **Reading.** Either "the rule tolerates x% force noise", or a stated limit.

**B7. Parameter efficiency, sampling only** (parked on 2026-11-15; listed for decision 4).
- Fewer time slices: `--thin-knots 0:5` halves the count with the existing code; every 5th or 10th slice needs a
  small sampler change.
- Pruning the unused global states (about 1,800 of 6,144 are used).
- Low-bit weights: the same runs as B4.
- Small K with patches: the K64 and K256 points of A4.
- **Output.** One curve of quality against stored couplings, for Table 1 and the limitations.

### 6.5 Priority C: optional

- **C1. CelebA-64.** After R4: decide whether K3072 moves the result (58.5). Two patch scales ran at 5 steps/s and
  were cancelled; they need a faster patch unfold for 8x8 every 2 px at 64x64 (`MODEL_AND_CODE.md` section 6).
- **C2. Quench duration on CelebA-32:** 0.25, 1, 2 on the best model, sampling only. Shows that quench 1 is not
  tuned per data set.
- **C3. Slice 0.** The hidden units are held fixed in slice 0 (`--freeze-hidden-slices 1`), never varied. One run
  with `--freeze-hidden-slices 0` or 2x steps on slice 0 for a current model (consistency check, section 6: low
  priority).
- **C4. Ising patch units** on ten digits with seeds: the simpler-hardware variant is one four-digit run so far.
- **C5. Colour.** CelebA-32 RGB needs a new problem type. Not planned; grey images are stated as a limitation.

### 6.6 Order and total

1. A0 and decision 2, then A1: they fix the protocol for everything after.
2. A4 part 1 and A8: cheap, and they settle Figures 3 and 4.
3. A5, A3 and A4 part 2: the long trainings.
4. A2, once the models and the protocol are final.
5. A6 and A7 in parallel (code).
6. B1 to B4, then B5 and B6.

Priority A is about 120 to 150 GPU h in all, close to one ten-digit sweep (130 GPU h, notebook section 4). The
fair-share factor is already low (0.10 on 11-16), so the 10,000-sample runs are best spread over several days.

## 7. Missing work that is not a numerical experiment

- **Figure 1** (concept and protocol) and the sample composite of Figure 2.
- **Shape defects.** A human count on the existing blind sheets (12 sheets per problem, 200 samples per set), or
  the AI-rater count with a clear label, or the confident share only.
- **Derivations for the appendix:** the local rule with patch units (with A6); the jackknife and the visit-count
  rule (in `theory.tex`); the replica construction; what a device measures for each factor.
- **Wang & Achour's protocol:** their n and FID implementation (ask the authors, or state the caveat). DTM's and
  NLM's own published numbers.
- **Corrections before quoting numbers** (consistency check):
  - parameter counts must include Q; the 11-15 notebook entries are 2x too low, `MODEL_AND_CODE.md` is right;
  - the frozen patch baseline does not "count each pixel once";
  - KID is against the test features;
  - "10x tighter tolerance moves samples by ~1e-6" is 4.9e-6 to 7.2e-6.
- **Double-blind:** the first paper in the third person; no run paths or job ids in the text; anonymous code.

## 8. Open decisions

1. **Lead claim.** Proposed: quality first, on-device training second. R1 and A5 decide how strongly claim 6 can be
   stated.
2. **Sampling protocol for all headline numbers.** Proposed: equilibrated start, SDE, quench 1, with the corrector
   as one ablation row. The alternative is to add the corrector everywhere (about 1 FID on CelebA; more sampling
   time).
3. **CelebA-64.** Proposed: appendix only, unless R4 or C1 changes the picture.
4. **Parameter efficiency.** Parked on 2026-11-15 until the paper is written. Proposed: pull the sampling-only part
   (B7) forward, because the parameter count is the first objection to Table 1 and most of B7 reuses A4 and B4.
5. **Main metric.** Proposed: Inception FID with FID_inf in the main text; classifier FID, precision / recall and
   class shares where class balance or shape matters.

## 9. Limitations to state in the paper

- All results are in the fast-hidden limit, with the hidden units summed out exactly; no device and no explicit
  hidden dynamics (B5).
- Parameter count: 1.4 to 9.8 M couplings per time slice and 100 programmed slices.
- A Potts unit with thousands of states is not existing hardware; a crossbar that computes the averaged force is
  the realistic route (`HARDWARE_NOTES.md`). The patch level works with plain Ising units; the global level does
  not.
- Grey images up to 32x32 (64x64 as a first run). On faces, backgrounds and hair regress towards mid-grey.
- Malformed shapes remain: a few per cent of the digits, against 0.5% for real ones.
- The local rule needs clamp forces and nudge projections, which are computed exactly here (B6).
- The comparison with Wang & Achour is across unknown differences in n and FID code.

## 10. Target and format

- ICML 2027, from the official call (icml.cc/Conferences/2027/CallForPapers, read 2026-11-21; the page says
  updates to the submission process will be announced by late November):
  - abstract 2027-01-14, full paper 2027-01-22 (both 11:59 PM anywhere on Earth); the site opens 2027-01-08;
  - 8 pages of main text, unlimited references, impact statement and appendices; double-blind;
  - two-stage reviewing: first-stage rejections by 2027-02-23, final decisions 2027-05-03;
  - every submission needs a reciprocal reviewer among its authors, and an AI disclosure statement.
- **arXiv before submission is allowed.** "Authors are allowed to post versions of their work on preprint servers
  such as arXiv." Conditions: the submitted version must not refer to the non-anonymized version; the work must
  not be advertised as an ICML submission during review (talks and posts about the work itself are allowed).
- **No parallel journal submission:** a paper submitted "in parallel to other conferences or journals" is not
  allowed, so PRX Intelligence cannot run alongside.
- Own earlier work in the third person (2026 author instructions); differences from it must be explained.

## 11. Chances at ICML, and what moves them (assessment of 2026-11-17)

Claude's judgement, asked for by the user on 2026-11-17. It is an opinion, not a measurement.

- **Estimate.** As the evidence stands today: below the acceptance rate (about one paper in four), roughly 10 to
  15%. With priority A done, a good outcome of R1 and A5, and an answer to the parameter count: about the
  acceptance rate, 25 to 35%. The reviews will spread widely, because a generative-modelling reviewer and a
  hardware reviewer judge this paper by different standards.
- **What speaks for it.**
  - A clear step over the two direct competitors (Wang & Achour, DTM) on MNIST and Fashion, and the first faces
    from a model of this kind.
  - Training without backpropagation and without a negative phase. Wang & Achour train with a digital
    discriminator and backpropagation through the solver.
  - Careful ablations with honest negative results, and a mechanism (shape early and global, texture late and
    local) shown by an intervention, the switch-on time.
- **Objections to expect.**
  1. "Physical" is a simulation in the fast-hidden limit, and no hardware has a Potts unit with thousands of
     states. No estimate of energy or time on a device.
  2. About 10^9 stored parameters for MNIST against 326k, so Table 1 compares unlike things.
  3. By digital standards the quality is low (small diffusion models reach single-digit FID on MNIST), and the data
     are small and grey.
  4. Parts of the method are known in machine learning: score matching with hidden units summed out (Swersky et al.
     2011, from memory), a final denoising step, stochastic against deterministic samplers. They must be presented
     as the physical counterparts of known practice, not as new.
  5. Many ingredients and hyperparameters, chosen on the reported metrics.
- **What moves the odds most.**
  1. **R1 and A5.** If training from sampled statistics works on the full recipe, the paper has a second
     contribution that no competitor has. If it does not, the local rule is a derivation with exact statistics only.
  2. **The parameter count and hardware tolerance (B7, B4).** For acceptance I would treat them as priority A: a
     small model (K64 or K256 with patches, 10 to 20 time slices, 6 to 8 bit couplings) with a stated FID makes
     Table 1 defensible.
  3. **One message.** The recipe details go to the appendix. Claim 3 should be tied to the literature on when a
     diffusion model decides class and detail (Biroli et al. 2024; Sclocchi et al. 2024; Kamb & Ganguli 2024; from
     memory, to verify), which gives the machine-learning reader a reason to care beyond hardware.

### Update 2026-11-20: ICML or a physics journal?

Asked by the user on 2026-11-20. Again Claude's judgement.

- **Results since 11-17 that change the estimate.**
  - Training fully from sampled statistics works on the recipe: CelebA-32, K784 + 8x8 + 16x16, S = 64 for every
    hidden unit with the visit-count rule gives 23.59, against 22.85 with exact statistics and Adam (notebook
    11-18; Inception 4268089). Lever 1 above, on one data set.
  - The parameter count fell. K64 with an early 16x16 set, an 8x8 set and the corrector gives 20.73 at 1.23 M
    couplings per slice. Keeping a few time knots per parameter set stores 10.7 M numbers instead of 83.2 M with no
    loss (21.67 against 21.72); the same rule at 64x64 stores 42.7 M instead of 331.8 M (notebook 11-20, morning
    summary). Lever 2 above, on CelebA.
  - Revised estimate for ICML: 30 to 40%, if priority A is finished on the new recipe.
- **Recommendation: a physics journal for the full story** (PRX or PRX Intelligence first).
  1. The results that set the work apart are results about a physical system: training from measured statistics,
     generation as a protocol of physical operations, and a time schedule that reduces to a few coupling sets
     cross-faded in t. A physics referee weighs these; a machine-learning reviewer weighs the FID against digital
     models, where the work loses by construction.
  2. The weaknesses at ICML are structural (quality by digital standards, small grey data, known components). The
     weakness at a physics journal is the missing experiment, and proposals of physical learning machines tested in
     simulation have appeared in PRX (from memory, to verify).
  3. The story has six parts (model, rule, protocol, mechanism, measured statistics, compression). Eight pages
     keep three of them.
  4. The recipe still changes every night. A journal has no deadline on 22 January; ICML would freeze a recipe that
     is nine weeks from settled, and MNIST and Fashion have not been rerun with it.
  5. ICML proceedings are archival. An acceptance uses these results up, and a later journal paper must be a
     different paper.
- **The case for ICML.** A decision in about four months instead of six to ten; a deadline; the competing papers
  (Wang & Achour, DTM) are read by that community; a rejection costs four months and one rewrite. It is the right
  choice if the aim is to be noticed by the machine-learning community.
- **PRX Intelligence** is open for submissions and waives its publication charge for submissions received before
  2027-01-01 (APS pages, read 2026-11-20). Check that hardware for AI is within its scope.
- **Note.** Sections 2 to 6 of this file predate the results of 11-18 to 11-20 (small global unit, early coarse
  patches, clamped label, time compression, trained knots). They need an update before they are used.

### Update 2026-11-20 (2): a crossbar mapping, or more hidden-unit comparisons?

Asked by the user on 2026-11-20: work out a crossbar circuit that could implement the model, or keep the Potts unit
idealised and add comparisons with Ising units and exponential family harmoniums. Claude's judgement.

- **Recommendation.** The crossbar mapping, at the level of architecture plus measured tolerances, not a
  transistor-level design. It answers the first objection at either venue ("no hardware has such a unit"). Fold
  the Ising comparison into it as one experiment. Use exponential family harmoniums as the framing of the model,
  not as further experiments.
- **Is it possible? In principle yes: each operation maps onto a known primitive.**

  | Model operation | Crossbar primitive |
  |---|---|
  | fields a_m = b_m + W_m.x + Q_m.x^2 | forward read of a conductance array (signed weights as differential pairs); a second set of input lines carries x^2, from one squaring circuit per pixel |
  | hidden average p = softmax(a / T) | an analog softmax element. Elfadel & Wyatt (NIPS 1993) show a reciprocal, passive multiterminal resistive element that computes softmax; this is the summed-out Potts unit as a circuit |
  | a sampled Potts state | a winner-take-all circuit with noise |
  | force on pixel i, sum_m p_m (W_mi + 2 Q_mi x_i) | transposed read of the same array (the same conductances, so the coupling is reciprocal), plus one multiplier per pixel for the Q term |
  | factor 2 of the reverse process | a gain of 2 on the transposed read; no replicas needed when the circuit computes the average |
  | time dependence | the 11-20 result: 5 to 9 fixed arrays per parameter set, mixed by two time-dependent gains (linear interpolation of couplings = linear interpolation of fields and forces at given p). No reprogramming during generation |
  | learning rule | per clamped sample, two rank-one updates (a state signal times a pixel signal) and a decay gated by p_m; with a sampled state only the winner's row changes. The projections in c_m are extra forward reads of the nudge vectors |
  | pixels | capacitors with a cubic nonlinear conductance (the quartic well), resistors to the 8 neighbours (the springs), and an injected noise current (the bath) |

  Precedent in `HARDWARE_NOTES.md`: NeuRRAM ran a dense 794 x 120 RBM with forward and transposed reads.
- **Why the current model suits it better than the one of 11-17.** The patches are the model and they are local
  (169 patch units of 32 states at 32x32, each wired to 64 pixels); the global unit can be 16 to 64 states; the
  schedule is a handful of fixed arrays. 10.7 M stored couplings are within the size of built arrays.
- **What is not known, and decides feasibility: precision.** The softmax must resolve field differences of order T
  on fields of 10^2 to 10^3 T (`HARDWARE_NOTES.md` estimates 9 to 11 bits by reasoning); resistive cells give
  about 4 to 6. The simulation can measure it: experiment B4 (couplings at 4 / 6 / 8 bits, static errors of 1 to
  10%, read noise on the fields), plus the distribution of the gap between the two largest fields over t. Add one
  retraining with the static errors in place, since on-device training should absorb them.
- **Also for the mapping:** Ising patch units with the current recipe on CelebA-32 (`--patch-kind ising`; so far
  one four-digit run, within 0.9 classifier FID of Potts). Binary units are the simplest hardware, and the patches
  are now most of the model.
- **Exponential family harmoniums as framing.** The model is a harmonium with multinomial hidden units and visible
  statistics (x, x^2), plus visible-visible springs, trained by score matching per noise level. The local rule of
  `DESIGN.md` 2 needs only the cumulants of a hidden unit, so it holds for any hidden unit that couples through one
  statistic, including a real double-well oscillator. Experiments on further families are not worth their cost:
  Gaussian hidden units are the harmonic modes (no gain, 11-13), and the others have no hardware counterpart.

### Update 2026-11-21: how hardware-close does the paper have to be?

Asked by the user on 2026-11-21: is it enough to train the best small model with the on-device rule and add a
high-level "Implementation" section, given the risk that someone else writes this paper first? Claude's judgement.

- **Answer: yes, that is enough. Stop pushing hardware-closeness and freeze the model.** The central claim (model,
  local rule from sampled statistics, quality) does not depend on a staircase schedule, the ODE or a circuit.
  "As implementable as possible" has no stopping point, and each step tried on 11-20 / 11-21 cost quality:
  - staircase 9/9/9, trained: 25.88 against 22.11 with 100 slices; with on-device training 42.61; at 64x64 61.44
    against 48.63;
  - knots built into the training: about 3.5 behind the 100-slice model on every grid;
  - ODE: 25.86 against 22.11, and 163 at a force scale of 0.9.
- **Linear interpolation is not the less buildable choice.** A crossfade between two fixed coupling sets is two
  time-dependent gains, and it needs a quarter of the sets a staircase needs (measured post hoc: 34 steps = 9
  linear knots, 22.36 against 22.35). The result to report is the post-hoc one: train at 100 slices, keep 5 / 9 / 9
  knots, 10.7 M stored, no loss. The staircase is one sentence.
- **The ODE is an ablation, not a goal.** It needs an exact force (a 10% error in its scale destroys the samples),
  which is the property hardware does not have; the noise of the SDE is what a physical system at temperature T
  supplies.
- **Minimum for the paper.**
  1. The headline model trained with the on-device rule at 100 slices. Open: the K16 controls 4289675 / 4289676
     (Inception 4289677). So far the rule is confirmed only for K784 + two patch scales (23.59 against 22.85).
     **Decision rule:** if K16 on device does not come within about 1 FID of exact, take the model where it does
     (K64 or K784) as the headline instead of repairing K16.
  2. The post-hoc knot schedule applied to that on-device model: one sampling job.
  3. MNIST and Fashion with the same recipe and rule; they are the only data with competitors' numbers.
  4. Two or three seeds of the headline, one n, one protocol.
  5. The "Implementation" section: the mapping table of the 11-20 update above, and a list of what is not shown
     (explicit hidden dynamics, nudge measurements, device precision).
  6. Wording: "on-device rule" means sampled hidden statistics in simulation; clamp forces and projections are
     exact.
- **Changes to earlier advice in this file.** B7 (parameter efficiency) is done by the work of 11-18 to 11-20. B4
  (coupling precision) and the crossbar study drop from "priority A" to optional: one sampling-only run with
  quantised couplings if it costs a day (Wang & Deng report such a test), otherwise an open point in the text.
  B5 and B6 go to the outlook.
- **Against being scooped: post on arXiv as soon as items 1 to 6 are written.** The preprint fixes the date,
  whatever the venue takes.

### Update 2026-11-21 (2): which venue, after the lock-in?

Asked by the user on 2026-11-21 ("ICML or some physics journal? I don't think it will get into Nature without
experiments"). Claude's judgement; the user has not decided.

- **Recommendation: PRX Intelligence, with arXiv on the day of submission.**
  - Topic fit in practice: it accepted "Understanding and embracing imperfection in physical learning networks"
    (Dillavou, Guzman, Liu, Durian) on 2026-11-04, a paper of the physical-learning community.
  - Research articles have no length limit, and the scope covers "theory, computation, and experiment", so a
    simulation paper is admissible.
  - Its publication charge is waived for submissions and transfers received on or before 2026-12-31 (journal
    pages, read 2026-11-21). That date is three weeks before ICML's and forces the same pace.
  - No archival proceedings and no dual-submission bind.
- **Cautions.**
  - The written scope is "AI/ML that advances the physical sciences"; hardware for AI is not named. Send the
    editor a short presubmission inquiry.
  - The accepted paper above is an experiment; ours is a simulation. It shows the topic fits, not that a
    simulation passes. The criteria ask for an "exceptional" advance, connection, capability or insight.
  - The journal is new (first papers July 2026) and has no track record yet. If a known name matters for
    applications in the next year, that speaks for ICML (machine learning) or PRX (physics).
- **Not recommended.**
  - Nature, Nature Physics, Nature Electronics: agreed, not without hardware.
  - Nature Communications, Nature Machine Intelligence: possible without experiments, but slow and a gamble while
    close preprints appear monthly.
  - PRX first: a desk decision after 2026-12-31 would lose the waiver on a transfer, and a simulation sequel to an
    unpublished first paper is a likely desk rejection.
- **ICML stays the alternative** (trackers: abstract 2027-01-14, paper 2027-01-22; one report mentions a posted
  call with two-stage reviewing; not confirmed on icml.cc). About one chance in three, eight pages, archival.
  Choose it only if the aim is to be seen by the machine-learning community.
- **If PRX Intelligence declines:** Physical Review Research or npj Unconventional Computing, or NeurIPS in May.

### Update 2026-11-21 (3): the user's aims change the venue recommendation to ICML first

The user on 2026-11-21: the readers wanted are physical and neuromorphic computing people; the CV has many physics
papers and no machine-learning paper; "at the end of the day the thing I care most about is that it is read".

- **If accepted at both, ICML is worth more to the user.**
  - CV: a first paper at a top machine-learning conference is a new credential; one more physics paper in a
    journal without a track record adds little to a physics-heavy list.
  - Readers: the people who work on equilibrium propagation and neuromorphic learning publish and review at
    ICML / NeurIPS / ICLR. The physicists of physical learning read arXiv whatever the venue.
- **So: ICML first (paper deadline about 2027-01-22), arXiv at submission, PRX Intelligence or NeurIPS (May) if it
  is declined.** This replaces the recommendation of the update above; the earlier one assumed a physics audience
  and equal value of the two outcomes.
- **Cost of ICML first:** about one chance in three; eight pages (the rest in the appendix); a rejection costs
  four months, the waiver of the publication charge (ends 2026-12-31) and one rewrite. The arXiv posting keeps the
  date and the readers in the meantime.
- **Being read depends less on the venue than on:**
  1. arXiv on the day of submission, cross-listed (cs.LG, cs.ET, cs.NE, cond-mat.dis-nn), with a title and
     abstract that say what was done;
  2. code and sample galleries released with it;
  3. a short note to the authors of the closest papers (section 12); they are the first readers and the likely
     referees;
  4. talks: workshops on new compute paradigms at the machine-learning conferences, seminars, APS March Meeting.
- **Check in the ICML 2027 call:** preprints are allowed (ICML 2026 call); whether advertising a preprint on
  social media during review is discouraged was not confirmed (recalled from earlier calls).
- **File name:** stays `ICML_PAPER_OUTLINE.md`.

## 12. Closest prior work (check of 2026-11-21)

Asked by the user: "nobody has done what we are doing yet, right?" A check by web search on 2026-11-21, about ten
queries and four papers read (partly through summaries; marked below). Not exhaustive: three of the closest
preprints are from the last seven months.

**Result.** The combination was not found anywhere: a diffusion model whose score is the force of a physical
system with hidden units, with time-dependent couplings learned by a local rule from statistics at clamped data,
without a negative phase, trained in simulation from sampled hidden states, at the quality of faces. The search for
exactly this combination returns our own first paper. But each single ingredient has a neighbour:

| Work | Hidden units | Time dependence | Training | Second phase | Shown on |
|---|---|---|---|---|---|
| Extropic DTM, arXiv 2510.23972 | binary latent nodes | a chain of 2 to 8 Boltzmann machines, each with its own weights | likelihood gradient: clamped minus free expectations, Gibbs sampling in both | yes, a free phase per step | simulation; binarised Fashion-MNIST, FID 28 +- 1 |
| Wang & Achour, arXiv 2606.27294 | 2352 hidden oscillators | 4 weight sets | GAN loss, backpropagation through the solver | - | simulation; MNIST 27.6, Fashion 80.8 |
| Whitelam, arXiv 2506.15121 (PRL 2026) | not confirmed | fixed couplings, read out of equilibrium | likelihood of the reversed noising trajectory | no | simulation |
| Wang & Deng, arXiv 2609.15439 (Sep 2026) | 512 explicit Langevin units | fixed couplings | the same loss as an online local rule | no | simulation; 3 to 15 MNIST prototypes, no FID in the part read |
| De, arXiv 2604.23806 (Apr 2026) | a hidden block; the score is the equilibrium of an output block | one noise-conditioned parameter set | denoising score matching by equilibrium propagation, symmetric finite nudges, bias O(beta^2) | yes, free and nudged equilibria | simulation; synthetic Gaussian data, D = 64 |
| Gower, arXiv 2610.14409 (Oct 2026) | hidden layers that relax | time-independent potential | energy matching by centred equilibrium propagation at finite nudge | yes; negatives from an autograd twin (summary) | simulation; CIFAR-10 FID 3.64 against 3.54 with backpropagation |
| Our first paper, arXiv 2506.19136 | none | yes | force matching | no | simulation; digits 0/1 |

Also seen, not read: an APS March Meeting 2026 talk, "Diffusion Models in Analog Circuits" (self-adjusting resistor
networks with coupled local learning; possibly real hardware); Yu et al., neural Langevin machine; the restricted
Kirchhoff machine (arXiv 2509.15842). Wang & Deng cite our first paper; Gower reportedly calls it the closest
prior work ("no hidden units"; from a summary, to verify).

**What follows for the claims.**
- **Do not claim "trained on device".** Nobody has, and neither have we: every work above, and ours, is a digital
  simulation. Our statement is "trained in simulation from statistics a device can measure".
- **Do not claim hidden units, time dependence, a local rule or backpropagation-free training as firsts.** DTM has
  per-step models with latent nodes and a local Boltzmann rule; De and Gower train generative models with hidden
  units by nudging.
- **Do not claim "finite nudges".** Finite symmetric nudges are the equilibrium-propagation line (De, Gower). And
  our runs do not use nudges: the projections G_m.psi and G_m.Gbar are computed exactly (in `latent_model.py` and
  `train_latent.py` the word appears only in two docstrings). The nudge measurement is a derivation in
  `DESIGN.md` 2.1.
- **What is ours.**
  1. The score is the system's own force with the hidden units summed out, and the gradient needs statistics at
     clamped data only: no free phase (DTM) and no second equilibrium of the whole system (De, Gower).
  2. Our nudge is a measurement, not a carrier of the loss: a field is quadratic in the pixels, so a symmetric pair
     of nudges gives the projection exactly at any size, where equilibrium propagation has a bias of order
     beta^2. This becomes a claim only once it is simulated (experiment B6, with read noise).
  3. Training from sampled hidden states is shown at scale: 23.59 against 22.85 with exact statistics on faces.
     De has synthetic data; Wang & Deng a few prototypes; DTM binarised Fashion-MNIST.
  4. The architecture and protocol: patch units switched on by time, the quench, a schedule of a few coupling
     sets.
- **Suggested sentence.** "To our knowledge, this is the first score-based diffusion model in which the score is
  the force of a physical system with hidden units, and whose time-dependent couplings are learned by a local rule
  from statistics measured at clamped data, without a negative phase."
- **To do before writing related work:** read Wang & Deng, De and Gower in full (the reports here cover them only
  in part), the APS abstract, and search arXiv by author for follow-ups.

### Addition 2026-11-21: three papers on dense associative memories (in `relevant-papers/`)

Read on 2026-11-21 (the first two in part, the third from its abstract and introduction).

- **Bacvanski, You, Hopfield, Krotov, "Dense Associative Memories with Analog Circuits", arXiv 2512.15002.**
  The largest overlap, in architecture and hardware. Visible and hidden neurons joined by one resistive crossbar
  used in both directions; a softmax over the hidden neurons from bipolar transistors; hidden neurons about ten
  times faster than the visible ones, which leaves the energy -(1/beta) log sum_mu exp[-(beta/2) |xi_mu - v|^2] on
  the visible neurons. That is one of our Potts units summed out, in the deterministic limit, with a uniform Q.
  Theirs: relaxation to a fixed point (inference), weights set by hand or trained digitally with backpropagation
  through time, no noise, no time dependence, no generation, non-negative conductances; XOR, a Hamming code, a
  small language model, validated in SPICE. **Consequences:** the bipartite architecture with fast hidden units is
  not ours to claim (it is the Krotov-Hopfield form of a dense associative memory); and their circuit is the
  basis to cite in "Towards Hardware".
- **Pham, Raya, Negri, Zaki, Ambrogioni, Krotov, Nature Communications 2026 (in press)**, with Ambrogioni 2024
  and Hoover et al. 2023. The score of a diffusion model is the gradient of a dense-associative-memory energy
  whose memories are the training images; with more data the model passes from retrieval, through spurious
  states, to generalisation. They read trained neural diffusion models this way. We build the energy
  explicitly with K << N learned templates per noise level. Our frozen-against-trained runs are the direct test
  of the stored-memory limit (CelebA-32: 147 frozen against 34 and 22 trained).
- **Rooke, Krotov, Balasubramanian, Wolpert, "Stochastic Thermodynamics of Associative Memory", arXiv
  2601.01253.** Least overlap in results. Binary-spin polynomial memories under Glauber dynamics at finite
  temperature, Hebbian memories; work and entropy production of retrieval and of driving; a finite-temperature
  failure of higher-order memories started far from every memory. For us: the framework for the energy cost of
  the generation protocol, which we do not compute; and a qualitative parallel to our control in which Langevin
  dynamics under the final energy alone falls into a blank state.
- **In the write-up:** cited in Section 3.1 (relation to dense associative memories), 5.4 (time dependence),
  6 (the circuit), 7 (related work), 8 (outlook); eight entries added to `references.bib`.

### Addition 2026-11-21 (afternoon): two novelty statements proposed by the user

The user: "two novelty claims that we currently leave out: trained DAMs for diffusion models (in contrast to just
memorized), and the fact that the clamping rule is finite amplitude, not a nudge". Both go into the paper, each
in a narrower wording than "first". Searches of 2026-11-21 (five web searches, five arXiv abstracts; no full
texts read).

**1. Learned memories.**
- Holds as: "a dense associative memory used as a diffusion model whose memories are learned at every noise level
  (K << N templates), where earlier links between the two have the training images as memories or read trained
  neural networks in these terms" (Hoover et al. 2023, Ambrogioni 2024, Pham et al. 2026; no counter-example
  found, also not in the abstract of the ICML 2025 tutorial by Krotov, Hoover, Ram, Pham, arXiv 2507.06211).
- Does not hold as "the first trained dense associative memory": Krotov & Hopfield 2016 train the memories for
  classification.
- A machine-learning referee will read a learned log-sum-exp energy as a mixture model: the global term is a
  mixture of K Gaussians with diagonal covariance, the patch units a product of such mixtures. Relatives to cite,
  so that we say it first: Shah, Chen, Klivans (NeurIPS 2023, arXiv 2307.01178; mixtures of Gaussians learned by
  gradient descent on the denoising objective, theory); Zach, Kobler, Chambolle, Pock (J. Math. Imaging Vis. 66,
  2024, arXiv 2310.12653; products of Gaussian-mixture experts on filter responses, trained over the whole
  diffusion horizon, for denoising; no generation in the abstract); Zoran & Weiss (ICCV 2011; mixtures on
  overlapping patches as an image prior); Kamb & Ganguli (arXiv 2412.20292; mixing local training-set patches
  predicts what convolutional diffusion models generate: the stored-memory counterpart of our patch units).
- Evidence we have: templates held at data prototypes 147 against 34 and 22 learned (CelebA-32, an earlier and
  larger model). To be labelled as such or repeated on the locked model.

**2. One phase at full clamp.**
- Holds as: "the rule is evaluated in one phase with the pixels fully clamped to data; no free phase, no nudged
  phase, no small-perturbation limit; the fluctuations of the thermal hidden units carry the response that a nudge
  would measure" (`DEVICE_RULE_CHECK_2026-11-20.md`, section "Clamping, not nudging"; device emulation without any
  displacement of the pixels equals autodiff to 1e-15 on a toy model with patch units, job 4287868).
- This replaces the earlier item "do not claim finite nudges" by a sharper one: do not use the word "nudge" for
  anything we do. The clamp is a full clamp, as in the positive phase of a Boltzmann machine. The optional
  read-out of the sums over pixels by symmetric displacements is called "probe displacement" in the paper.
- Limits to state: (i) that score matching needs no samples from the model is a known property, and its gradient
  for hidden units as posterior moments at data is known mathematics (Hyvarinen 2005, 2007; Swersky et al. 2011);
  our first paper has the clamp for visible couplings. New: the single-phase rule for hidden units as fluctuation
  measurements, local to each coupler, and training from S reads. (ii) The price: thermal hidden units are
  required, and a bias of order 1/S takes the place of the finite-nudge bias (removed by the jackknife).
  (iii) The runs simulate the sampling of hidden states only; clamp forces and sums over pixels are exact
  arithmetic.
- Neighbours for the contrast: De (arXiv 2604.23806; per a search summary, unbiased only for vanishing nudge: to
  check in the full text), Gower PotentialEP (2610.14409), and new in this search **Gower, FlowEqProp, arXiv
  2604.08150** (flow matching with "gradient equilibrium propagation", small digit images, 5 pages).
- **In the write-up:** abstract, introduction (gap (iv), contributions 1 and 2, a wording note), Section 3.1
  ("Learned, not stored"), Section 4.1 ("One phase, full clamp", "Fluctuations take the place of the nudge", "The
  price"), related work (three bullets), Appendix A. Five entries added to `references.bib` (41 in total).

## 13. Updates

- 2026-11-17: first version.
- 2026-11-17: section 11 added (chances at ICML and what moves them).
- 2026-11-20: section 11, update on the venue (ICML or a physics journal) and the revised estimate.
- 2026-11-20: section 11, second update (crossbar mapping against more hidden-unit comparisons).
- 2026-11-21: section 12 added (closest prior work and what can be claimed as new).
- 2026-11-21: section 11, update on how hardware-close the paper has to be (minimum list; B4 and the crossbar
  study now optional).
- 2026-11-21: section 11, second update on the venue (PRX Intelligence recommended; ICML the alternative).
- 2026-11-21: section 11, third update: with the user's aims (readers in physical and neuromorphic computing, a
  first machine-learning paper), ICML first, PRX Intelligence or NeurIPS as fallback.
- 2026-11-21: section 10 replaced by the official ICML 2027 dates and rules (arXiv before submission allowed).
- 2026-11-21: section 0 added: the current plan for one short 8-page paper with the locked model; sections 1 to 6
  are now the record of 11-17.
- 2026-11-21: LaTeX skeleton in `/n/fs/cyrills-part/oscillator_dev/ICML-writeup/` (`main.tex`, official ICML 2027
  style; numbers in `results.tex`; compile with `sbatch compile.sh`). It follows section 0, with three figures
  (concept; FID exact against sampled statistics; generated images) and two tables.
- 2026-11-21: section 12, addition on three papers on dense associative memories (Bacvanski et al., Pham et al.,
  Rooke et al.); cited in `main.tex` sections 3.1, 5.4, 6, 7, 8 and appendix E.
- 2026-11-21 (afternoon): section 0, open item on the size of the global unit for MNIST and Fashion: locked K16
  has the lower Inception FID (14.39 against 27.86 on MNIST) but fewer clear digits (71.5% against 88.8%
  confident); the user finds the old K784 samples visibly better.
- 2026-11-21 (afternoon): section 12, addition on two novelty statements proposed by the user (learned memories;
  one phase at full clamp), with the wording that holds and the prior work that limits each; written into
  `main.tex`.
- 2026-11-21 (evening): abstract and Sections 1 to 4 of `main.tex` written as prose, at the user's request, with
  the derivation of the rule in the main text (the generator form: dJ/dtheta = -<L dF/dtheta>/T^2 with
  L = T Lap - psi . grad; L p_m = p_m c_m / T; product rule for W and Q; then the covariance form and the rule per
  read). Sections 5 to 8 and the appendices are still bullets. The earlier bullets of Sections 1 to 4 are kept in
  `ICML-writeup/main_skeleton_bullets_2026-11-21.tex`. **Length:** Sections 1 to 4 with Figure 1 end at the top of
  page 6 (about 5.0 pages, notes shown), against 3.9 planned; with the planned 4.9 pages for Sections 5 to 8 the
  main text would be about 10 pages against the limit of 8. First candidates to move to the appendix: the
  jackknife formula and visit-count rule (4.4), "what a device measures" and the rule per read (4.3), the factor
  two (3.2), equation (8) with the stored memories (3.1).
- 2026-11-21 (17:00): section 0, results of the global-unit test (K784 with the locked patches is best on MNIST
  and Fashion; decision pending) and of the spring ablation (20.20 / 25.83 / 218.4); notes on the label of the
  pixels-only row and on FID_inf.
- 2026-11-22: section 0, first wave of the lock-in runs (on device 1 to 2 FID above exact, S = 16 = S = 64, K784
  for MNIST and Fashion, post-hoc knots fail on MNIST and Fashion); `results.tex` filled, `main.tex` updated.
- 2026-11-22 (afternoon): **decision by the user: K per data set by the stated rule** ("okay lets go with that"):
  K784 for MNIST and Fashion, K16 for CelebA at both resolutions; smallest K within one FID of the largest tested.
  The user also wants the result that the global unit matters on MNIST and Fashion and hardly on faces featured;
  it is the paragraph "The global unit carries the class" in Section 5.3.
- 2026-11-22 (afternoon): **Sections 5 to 8 of `main.tex` written as prose.** 5.1 Setup; 5.2 Training from
  Sampled Hidden States (Table 1, Figure 2); 5.3 What the Parts of the Model Do (new Table 2: visible-energy
  ablation and K; Figure 3; "Other models" in the text only); 5.4 The Steps of the Sampler (Table 3; the
  time-dependence control); 5.5 Smaller Models (K16 on faces, 5/9/9 coupling sets, where both fail, binary patch
  units); 6 Towards Hardware; 7 Related Work; 8 Limitations and Conclusion. The earlier bullets are kept in
  `ICML-writeup/main_skeleton_bullets_sections5-8_2026-11-22.tex`. Figures 2 and 3 are provisional: four gallery
  images copied to `ICML-writeup/figures/provisional/` (they still carry their own titles and row labels).
  **Length:** the main text ends at the top of page 10, about 9.1 pages with the notes shown, against the limit
  of 8; CelebA-64 and the seeds will add a few lines. Open notes in the text: seeds and sample count; CelebA-64;
  the origin of the gap to exact training; the cause of the diverged MNIST samples; the fourth cell of the
  visible-energy ablation (run 4297259); coupling counts from the code; the statements about Bacvanski et al.,
  De, Gower and Wang & Deng to be checked against the full papers; the first paper's bibliography entry. Still
  bullets: Software and Data, AI Use Statement, Impact Statement, all appendices.
- 2026-11-20: independent check of the Potts learning rule (`DEVICE_RULE_CHECK_2026-11-20.md`, job 4287868). Theory and code are correct; the rule for the patch units is written and checked against autodiff in `check_device_rule.py` on a toy model (A6: done in substance, not yet inside `force_matching_local_grad` or on trained slices). The runs simulate the hidden-state sampling only; B6 should also put noise on the per-read sums over pixels, not only on the clamp force.
