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
