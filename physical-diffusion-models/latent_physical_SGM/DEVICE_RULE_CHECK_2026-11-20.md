# Check of the on-device learning rule for the Potts units (2026-11-20)

The user's question: is the gradient rule claimed for on-device training of the Potts units really measurable
(clamp the pixels, measure the hidden statistics), is the theory correct, and does the code implement it?

Read-only check: no existing code or note was changed. New files: this note, `check_device_rule.py`, and the job
output `logs/check_device_rule_4287868.out` (CPU job **4287868**, 4 min).

## Verdict

- **Theory: correct.** The formulas of `DESIGN.md` 2.1 and `theory.tex` section 3 (Potts couplings, plug-in bias,
  both jackknife orders) hold. I re-derived them by hand and checked them numerically.
- **Code: correct.** The gradient that training uses with `--hidden-samples S` is exactly the rule a device could
  apply from its reads, for the global unit and for the patch units (agreement at rounding level, 1e-15).
- **Measurable: in principle yes, but only one of the measured quantities is simulated.** The runs sample the hidden
  states. The clamp force and the sums over each unit's pixels are computed exactly. "Fully on-device" overstates
  this (see "What the runs test").

## Theory

- For a parameter of state m, the gradient of J_T is `p_m c_m / T^3` plus two explicit terms, with
  `c_m = |G_m - Gbar|^2 - V + 2T (s_m - sbar) - (G_m - Gbar) . psi`. Derived again from
  `dp_k/da_m = p_k (delta_km - p_m) / T`, `Lap F = Lap E^v - V/T - 2 sbar`; all three gradients (b, W, Q) agree
  with `DESIGN.md` 2.1.
- With several hidden units (global, patches, per-pixel) the rule of each unit is the same formula with its own
  p, G, Gbar, V, sbar and the **total** clamp force psi. The Laplacian adds over units, so there are no cross terms.
- Only statistics conditional on the clamped pixels enter: no negative phase and no release. This is the known
  property of score matching for models with latent variables.
- **A form that shows what is measured.** `c_m` is a centred scalar: `p_m c_m = Cov(1[state = m], phi)` over the
  reads, with one scalar per read

      phi = sum_i df_i (df_i - psi_i) + 2T sum_i Q_(active state, i)

  where `df_i` is the unit's force on pixel i at that read minus its window average, and `psi_i` the window-averaged
  clamp force. The update of a read goes to the active state only:

      db_m  ~ (phi - mean phi) / T^3
      dW_mi ~ x_i (phi - mean phi) / T^3 + (2 df_i - psi_i) / T^2
      dQ_mi ~ x_i^2 (phi - mean phi) / T^3 + 2 x_i (2 df_i - psi_i) / T^2 + 2 / T

  Averaged over the S reads this is the rule with p -> p_hat everywhere (the plug-in rule), exactly.

## Clamping, not nudging (added after the user's question, 2026-11-20)

- Force matching is one experiment per training image: the pixels are held at the noised data point at full
  amplitude, the hidden units equilibrate, and everything is read at that one configuration. There is no free phase,
  no second run, no difference of two states and no small-amplitude limit.
- **Fluctuations take the place of the nudge.** The term a nudge would give as a response,
  `psi . grad_x p_m = p_m (G_m - Gbar) . psi / T`, is read as the covariance of "state m is active" with the
  force fluctuation projected on psi. This needs thermal hidden units: without their fluctuations there is nothing
  to read.
- The "nudges" of `DESIGN.md` 2.1 are something else: an optional read-out of the sums over pixels by finite
  displacements of the clamp, exact at any amplitude because a_m is quadratic in x. The force form above does not
  need them (the device emulation of check 2 never moves the pixels). A different word in the paper, for example
  "probe displacement", would avoid the confusion with nudging in Equilibrium Propagation.

## What a device must measure

Per clamped image and per window of S reads:

| Quantity | How | Simulated in the runs? |
|---|---|---|
| active state of every Potts unit at each read | read-out of the hidden unit | **yes** (S draws) |
| window-averaged force on every clamp, psi_i | one slow force sensor per pixel | no: exact arithmetic on the sampled states |
| the unit's force on each of its pixels at each read, df_i | from the coupler's own setting (W_mi + 2 Q_mi x_i), or a force sensor faster than the hidden switching | no |
| per read, the sum over the unit's pixels in phi | a second sum over the unit's fan-in, or nudges (below) | no |
| the active state's spring sum, sum_i Q_mi | from the couplings | no |

### Where G_m comes from (added 2026-11-23, the user's question: "do we need to measure grad a_m = G, and how?")

- **G is needed, in two roles.** Per coupler, `G_mi` enters the W and Q updates (`2 (G_mi - Gbar_i)`). Summed over
  the unit's pixels it enters `c_m` (`|G_m - Gbar|^2`, `(G_m - Gbar) . psi`).
- **It is a force, not a derivative to be taken.** The joint energy contains `-a_m(x)` when state m is active, so
  `G_mi = W_mi + 2 Q_mi x_i` is the force that coupler (m, i) puts on pixel i. By reciprocity the same number is the
  sensitivity of state m's field to pixel i.
- **In the runs and in `check_device_rule.py` G is not measured.** It is computed from W, Q and x for the active
  state. Only the hidden states are sampled.
- **Three ways a device could get it:**
  1. *From the coupler's own setting.* Local for the per-coupler role. The sums over pixels then need a summing line
     per unit (N products per visited state). The rule then uses the coupling values, so calibration errors of W and
     Q enter.
  2. *From the clamp forces, resolved per read.* `Cov(1[state = m], force on clamp i) = -p_m (G_mi - Gbar_i)`, with
     no knowledge of W or Q. Needs force sensors faster than the hidden switching. Every other hidden unit that
     pulls on the same pixel adds noise (zero on average, because the units are independent given x): about 34
     units per interior pixel in the CelebA-32 recipe (global, 16 + 16 patches, per-pixel). **Not tested.**
  3. *From probe displacements with field read-out.* Gives sums `G_m . u`, never the components. Covers
     `G_m . psi` and `G_m . Gbar`; `|G_m|^2` needs one more pair per visited state.
- Correction of a sentence below: computing the sums digitally does save something, but little. It costs N per
  visited state, against N per state for all fields a_m (which give the exact p_m).

### Is "local, on-device" a fair name? (2026-11-23)

Yes, with the couplings taken from the couplers (way 1 above), if both words are defined:

- **On device:** the device supplies the hidden states at the clamped image and the clamp force. Nothing simulates
  the dynamics, nothing is differentiated through it, and there is no free-running phase. A teacher forms the
  products and applies the updates.
- **Local:** a three-factor rule. Coupler (m, i) uses its own G_mi, its pixel's x_i, psi_i and Gbar_i, and its
  state's p_m and c_m. `c_m` is a sum of local products over that state's own couplers: one extra sum along
  connections that exist, no signal from other hidden units.
- **Not parameter-free.** The G terms use the coupling values. Errors of the real couplings against the assumed
  ones are absorbed in p_hat and psi (measured on the device) but not in the G terms. The outline's remark under B4
  that on-device training absorbs static coupling errors holds only in part, unless G is read from forces (way 2).
- **What is evaluated at the true hardware values** (the user's follow-up):
  - the occupations p_hat and the clamp force psi, with whatever couplings the device really has;
  - therefore the whole gradient of the **visible** couplings, `<-Lap phi_k / T + grad phi_k . psi / T^2>`: the
    phi_k are known functions of x, so no coupling value enters. For these the rule knows the model, not the
    parameters;
  - in the **hidden** couplings' gradient, every factor except G (and the spring sums s_m). A mismatch dG gives an
    error of first order in dG in those terms only.
  - Test (not started): a run in which p_hat and psi come from couplings with a static error and the G terms from
    the error-free values.
- **It assumes the model.** The response `dp/da = (diag(p) - p p^T) / T` is that of an ideal Potts unit in
  equilibrium at T. A unit with finite inhibition or other switching statistics breaks it.
- **What a referee may ask.** Given x the hidden units are independent, so the same statistics cost a few M x N
  products digitally. The device does not solve a hard problem in training. The case for it is that the training
  signal comes from the real hardware and needs no model of its dynamics.

- **Nudges (`DESIGN.md` 2.1).** `[a_m(x + eps u) - a_m(x - eps u)] / (2 eps) = G_m . u` holds for any eps (checked
  with eps = 0.5). It gives `G_m . psi` and `G_m . Gbar`, if the field a_m of every visited state can be read as an
  analogue value, not only which state is active.
- **`|G_m|^2` has no clean nudge protocol in the notes.** "Follows from the couplings" is a computation, not a
  measurement. The release version (variance of a_m under pixel noise) is a variance estimate and needs many
  repeats. Through the forces it is `sum_i df_i^2` at the clamps, or one more nudge pair per visited state
  (along that state's own force pattern).
- **The alternative through occupations is not practical.** `(G_m - Gbar) . psi = T d(log p_m)/d(nudge)` is true,
  but it is a difference of two sampled occupations at a small nudge: the weakness already found for CD-1.
- **If the sums over pixels are computed digitally from stored couplings, sampling the hidden states saves nothing:**
  the same O(M N) arithmetic gives the fields a_m and hence the exact p_m. The on-device claim therefore rests on
  the physical measurement of these sums, which is the part not simulated.

## Code

- `latent_model.sampled_state_mean`: value = occupation frequency of S categorical draws from softmax(a / T),
  derivative = `(diag(p_hat) - p_hat p_hat^T) / T`. Autodiff of the loss through it gives the plug-in rule. Correct.
- `make_loss_per_sample`: the clamp force uses the same sampled means (as a time-averaged force would). The
  jackknife parts are halves and quarters of the same draws. The patch units draw with their own keys. Correct.
- `loss_draw_occupation` / `patch_occupation` with `--occ-same-draws`: same fields, same keys, same draws as the
  loss. Correct.
- `force_matching_local_grad` still refuses `--patch`. Training does not use it; the numpy rule in
  `check_device_rule.py` covers the patch units.

## Numerical check (job 4287868, `check_device_rule.py`)

Small model in float64: 6x6 pixels, global Potts unit with 7 states and Q, per-pixel units, two patch sets (4x4 every
2 px with 3 states, 3x3 every 3 px with 4 states), a stand-in visible energy, T = 0.02, batch 5, random couplings
(largest state probability per image 0.50-0.89).

| Check | Result |
|---|---|
| 1. my numpy local rule, exact statistics, global **and patch** units, against `jax.grad(make_loss)` | relative difference <= 1.7e-15 in every group |
| 2. device emulation (states at each read, clamp forces, updates to the active state only; no softmax) against `jax.grad(make_loss(samples=S))`, same draws | plug-in S = 8: 7.9e-16; jackknife S = 8: 1.2e-15; jackknife S = 64: 2.3e-15 |
| 3a. plug-in loss bias at S = 8, 200,000 keys | -12.70 +- 0.08, predicted -V / (2 S T^2) = -12.76 |
| 3b. plug-in gradient bias, relative to the exact gradient | 2.4-14.4% per hidden group (chi2/dof 9309) |
| 3c. first-order jackknife | 0.1-0.8% (chi2/dof 29: the O(1/S^2) rest is visible at S = 8) |
| 3d. second-order jackknife | none (chi2/dof 0.97; Monte Carlo floor 0.1-0.2%) |
| 4. nudge identity, eps = 0.5 | 1.5e-13 |

- The visible gradient is linear in the sampled means and unbiased in all three variants.
- At S = 64 with the first-order jackknife (the training runs) the remaining bias is about 64x smaller than in 3c:
  negligible.
- The check uses a toy model, not a trained recipe model. The identities do not depend on the parameters; the
  outline's A6 asked for trained CelebA-32 slices, which would only test rounding at realistic field sizes.

## What the runs test, and wording

- The CelebA-32 result (23.59 against 22.85 exact, `MODEL_AND_CODE.md`) shows that **S = 64 sampled hidden states per
  clamped image are enough**. It does not test force read-out, the sums over pixels, read noise, correlated reads or
  coupling precision.
- Suggested wording: "trained with sampled hidden statistics" instead of "fully on-device training".
- Outside the gradient rule, and also digital in the runs: per-element Adam for the visible couplings, the leaky
  gradient average per hidden coupling, the per-group step scale of the visit-count rule, the average of the
  iterates per slice, and the k-means references at initialisation.

## A deterministic version: the denoising form (derivation of 2026-11-20, not tested)

The user's question: is there a deterministic version of the rule, as the flow ODE is a deterministic sampler?

- **The sampler does not change training.** SDE and flow ODE use the same F; both need the score learned on noised
  data. Noised data is part of any diffusion or flow-matching training, but it is data preparation, not device noise.
- **Why the present rule needs fluctuations.** The implicit loss contains the Laplacian of F, a trace over all pixel
  directions. The gradient is the Langevin generator applied to the Hebbian observable,
  `dJ/dtheta = -E[ T Lap(dF/dtheta) - grad F . grad(dF/dtheta) ] / T^2` (for b_m: `E[T Lap p_m - psi . grad p_m] / T^2`,
  which is `p_m c_m / T^3`). The drift part is a first-order response. The Laplacian part needs thermal
  fluctuations, or N probe displacements, or random probe patterns (sliced score matching).
- **The denoising form has no Laplacian.** With the clean image x_0 and the noised clamp x_t of slice t,

      psi* = (x_t - e^(-t) x_0) / (1 - e^(-2t)),      J_DSM = E |psi - psi*|^2 / (2 T^2) = J_T + const (Vincent 2011)

  (a spring toward the shrunk clean image; T cancels). With the force error `e = psi - psi*`:

      dJ/db_m  = -< p_m (G_m - Gbar) . e > / T^3
      dJ/dW_mi = -< p_m [ x_i (G_m - Gbar) . e / T^3 + e_i / T^2 ] >
      dJ/dQ_mi = -< p_m [ x_i^2 (G_m - Gbar) . e / T^3 + 2 x_i e_i / T^2 ] >
      dJ/dtheta_k = < e . grad phi_k > / T^2

  Against the present rule: `c_m` becomes `-(G_m - Gbar) . e`; the terms `|G_m - Gbar|^2`, `V`, `s_m - sbar` and the
  `2/T` of Q are gone.
- **No gain in sample quality expected.** The two objectives are the same function of theta up to a constant, so
  they have the same minimiser for any model family and the same expected gradient at every theta. The flow ODE run
  on a model trained with the present rule uses the same score. The denoising rule changes the noise of the gradient
  estimate and what a device must measure, nothing else.
- **What a device does.** One probe displacement of the clamp along e (a symmetric pair, exact at any amplitude)
  and a read of the fields a_m gives `G_m . e` for every state of every unit at once. Nothing else is non-local.
  The hidden units may be deterministic elements with analogue outputs p_m = softmax(a/T) (T is then a gain), or
  thermal units read by sampling; in the second case only p_hat and one covariance are needed.
- **Which combination needs a displacement of the clamp.** Deterministic hidden units force it, not the denoising
  objective: a thermal unit's fluctuations at one clamp already contain its response to a displacement.

  | | thermal hidden units | deterministic hidden units |
  |---|---|---|
  | implicit objective (present rule) | one clamp, no displacement; second- and third-order fluctuation statistics | N displacements per image, or random patterns |
  | denoising objective | one clamp, no displacement; one covariance | one pair of finite displacements along e |

  The pair is finite and exact only if the fields a_m can be read. With the outputs p_m only, it must be small
  (a nudge in the sense of Equilibrium Propagation).

  The table is for measured `G_m . e` (the user's point, 2026-11-25). If G is taken from the coupler settings, as
  route 1 of "Where G_m comes from" does for the present rule, deterministic units need no displacement either:
  read p_m and psi at the one clamp, compute `(G_m - Gbar) . e` digitally (a sum over pixels for every state with
  non-negligible p_m; at gain 1/T = 50 these are few). Price: these terms are then at the nominal couplings. The
  thermal unit keeps one advantage: the clamp force of each read contains G of the active state as the device
  realises it (route 2). A deterministic unit shows only the mean `sum_m p_m G_m` in its clamp force, so for it
  the displacement is the only measurement of `G_m . e`.
- **e is not the drawn noise** (the user's question, 2026-11-25). With `x_t = e^(-t) x_0 + sigma_t xi`,
  `sigma_t^2 = T (1 - e^(-2t))`, the target is the drawn noise rescaled, `psi* = T xi / sigma_t`. e is the model's
  clamp force minus that: `e = (T / sigma_t) (xi_predicted - xi)` with `xi_predicted = sigma_t psi / T`. At the
  optimum `psi = E[psi* | x_t]`, so e averages to zero given x_t but is not zero for one draw.
- **Deterministic hidden units, spelled out** (2026-11-25). One unit = M analogue neurons with
  `tau_h dh_m/dt = a_m(x) - h_m`, outputs `f = softmax(h / T)`, force on pixel i `sum_m f_m G_mi`. For fast hidden
  neurons the pixels move in the same F as with a thermal Potts unit. This is the hidden layer of Krotov & Hopfield
  (2021) with the log-sum-exp Lagrangian, T = 1/beta, plus the Q terms. The exact-statistics runs and the generation
  code compute exactly this; the thermal Potts unit is the other physical realisation of the same F.
  - Every coupler carries its state's force scaled by f_m all the time (thermal: one state at a time, at full force).
  - No barrier, so no conflict between bistable and fast; the settling time of the softmax circuit replaces it.
  - Generation with the SDE needs 2F: double the couplers' output, or two copies. The flow ODE needs F once.
  - No fluctuations to read: the covariance read-out of the present rule is not available, the denoising rule is.
    (The present gradient could still be computed from the read outputs f_m, the clamp force and the nominal G, with
    arithmetic over all M states and N pixels. Then every G term is a calculation from the coupler settings, not a
    measurement.)
  - The thermal Potts unit needs its noise at all times, in training and in generation (the user's question,
    2026-11-25). The noise is what makes it hop; without it the unit locks into the state of largest a_m, which is
    the T -> 0 model (max instead of log-sum-exp), a different F. So thermal and deterministic are two builds of
    the same F, not a training mode and a generation mode. What the present rule adds on top of the noise is the
    time-resolved read-out: S reads of the active state, each with its clamp force.
- **Price.** The teacher must know x_0 next to x_t (it prepares the clamp, so it does). The target psi* adds
  variance, most at small t. Equal to the present rule only in expectation over the noise. Not implemented here:
  `latent_model.py` has the implicit loss and CD-1 only.

## Follow-ups (not started)

0. The denoising rule above: check its gradient against the implicit one in expectation on a toy model, then one
   training run (CelebA-32 K784 + 8x8 + 16x16) with exact and with sampled hidden statistics.

1. Noise on the measured sums: Gaussian read noise on `phi` (per read) and on `psi`, at 1%, 3% and 10%. The
   outline's B6 lists noise on the clamp force only; `phi` enters with 1 / T^3 and is the more exposed factor.
2. State in the paper which protocol is assumed for the sums over pixels (force read-out per read, or nudges with
   analogue field read-out), and its cost per training image.
3. Optional: port the patch rule of `check_device_rule.py` into `force_matching_local_grad` (closes A6 in the code).
