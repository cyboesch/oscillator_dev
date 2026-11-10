# Latent physical SGM: handoff, 2026-11-10

## Agreed publication direction

Revise the existing rejected PRL manuscript into **one combined paper**, including hidden Ising/Potts units and a demonstration that their couplings can be trained locally. Keep systematic scaling, encoder–decoder architectures, and more realistic physical implementations as future research directions and a potential group/job-talk agenda. Do not misinterpret this as a plan for two separate papers or to postpone hidden units altogether.

The intentional time-dependent, open-loop parameter protocol is central to the contribution. A requirement that parameters must be static is not a valid objection to that objective. Hardware still has to implement the prescribed protocol, but no state-dependent inference feedback is required. Venue remains undecided; do not treat the time-dependence objection as a reason to downgrade the target.

Documents read previously: `2506.19136v4.pdf`; `current-paper-prl-rejected/lh20021_report_1_c.pdf`; `current-paper-prl-rejected/Rebuttal_R1_Boesch_et_al.pdf`; extensive portions of the latent README. The folder did not contain a separate final editor decision or second-round report. Referee A wanted clearer significance, quantitative generation measures, and complexity/scaling discussion; Referee B objected to time-dependent parameters.

## Active task: consistency/error review (unfinished)

User requested another careful comparison of:

- `physical-diffusion-models/latent_physical_SGM_writeup/theory.tex`
- `physical-diffusion-models/latent_physical_SGM/latent_model.py`, `train_latent.py`, `sample_latent.py`, and evaluation/diagnostic code
- `physical-diffusion-models/latent_physical_SGM/README.md` and `DESIGN.md`

The newly visible IDE file `njp_unconventional/Generative_Oscillators_npj_2026-08-13_fixed/LaTeX_source/main.tex` has **not** been inspected in this review. No scientific/code files have been edited; no numerical tests or training jobs have been run.

## Findings so far

1. **Core algebra agrees.** Independently checked visible energy/gradient/Laplacian, Ising and Potts marginal derivatives, quadratic hidden coupling Q, and explicit local parameter gradients. No sign or temperature-factor error found in these formulas. Relevant code: `latent_model.py` approximately 338–402, 454–521, 713–767, 1125–1174. The sampler's reverse SDE uses x−2 grad F and its probability-flow ODE uses x−grad F, consistently with the intended effective model.

2. **Writeup is stale about multi-Ising support.** `theory.tex` approximately 99–100, 166, 338–340 still describes only K=1 training / multi-Ising as future work. Current code and README include large multi-Ising runs. README's opening “only one Ising unit used so far” was also stale in the previously read version; recheck its current wording.

3. **Finite hidden equilibration needs a careful correction.** `theory.tex` approximately 274–278 and DESIGN discuss finite correlation time/detailed balance. Fixed-parameter equilibrium marginals can be correct for arbitrary hidden relaxation times, but this does not establish the driven reverse process used for generation. Exact effective drift assumes fast conditional equilibration. Replica construction has stationary visible energy 2F−|x|²/2, not F. Read the exact paragraph before deciding whether this is an outright false statement or a context/qualification problem.

4. **Zero-weight degeneracy is overstated.** `theory.tex` approximately 163–166: W=0 is stationary for unbiased Ising units, but with nonzero b it need not be stationary away from a visible-bias optimum. Vanishing quadratic curvature also requires the visible model to absorb the hidden quadratic contribution. Counterexample: T=1, visible energy (x1²+x2²)/2 with no cross edge; centered data covariance [[1,c],[c,1]], c>0; unbiased hidden W=(w,w). Then J(w)=J(0)−2c w²+O(w⁴). Thus zero W is stationary but not degenerate to quadratic order. Sparse visible models make the qualification important.

5. **Bistability statements need assumptions.** Negative visible quadratic stiffness alone does not guarantee a double well with a learned bias/tilt (`theory.tex` around 59). Example: −x²/2+x⁴/4+2x has one minimum. A local −T log cosh term likewise permits but does not guarantee a double well in the total potential (around 123).

6. **CD1 equality needs its small-step remainder and convention.** Theory around 202–207 omits the finite-release remainder; DESIGN specifies x'=x−dt psi/2+sqrt(T dt) xi. Under that convention the difference of clamped and released parameter derivatives is (T² dt/2) grad J + O(dt²), with released state held fixed during the derivative.

7. **Sampled-statistic gradient implementation looks consistent so far.** Agent inspected plug-in response derivatives, independent hidden draws, replicate averaging, and jackknife coefficients. Scalar loss is quadratic in empirical means; its plug-in gradient can be cubic. First-order jackknife leaves O(S^-2) bias; second-order cancels that too under IID draws. Theory's newer detailed discussion correctly recognizes this, but around line 237 it still calls an estimator “unbiased with the jackknife” without specifying the order. Finite correlated physical readouts need additional assumptions.

8. **README causal conclusions are too strong (reporting issues).** Failed Ising reseeding/learning-rate remedies do not prove the gap is architectural rather than optimization-related. Short quenches and PF-ODE comparisons do not prove all remaining sample error lies in the final-time energy or that grain is mostly learned: earlier score errors and limited relaxation are not excluded. Prototype-distance/spread checks rule out near-copy/narrow-jitter behavior, not every possible prototype-plus-noise model.

9. **Quench interpretation needs checking.** `sample_latent.py` 158–174 implements deterministic −grad F at the original hidden temperature, with no visible diffusion. Cooling a shared physical bath would also change hidden tanh/softmax statistics. The README's “cooling the bath” language therefore needs qualification. Its statement that large-amplitude quenched samples prove the energy is nonconfining is also suspect: positive sextic confinement, or positive local quartic coefficients with nonnegative pair quartics, dominates the hidden O(|x|²) terms. Verify the actual trainer projections and distinguish amplitude >15 from mathematical divergence.

## Pending checks

- Finish full `train_latent.py` inspection: parameter projections, optimizer locality, EMA vs best selection, and configuration constraints.
- Check whether first-order jackknife validates even S>=2; `make_loss_per_sample` appears to validate only second-order divisibility, but CLI may guard it.
- Check whether `--occ-same-draws` is prohibited with `--hidden-replicates > 1`; occupation helper reproduces the loss draw key only for one replicate. These are hypotheses, not confirmed bugs.
- Finish evaluation-code audit, especially class-standardized scores for absent classes, divergence exclusion, FID/KID filtering and paired-bootstrap interpretation. No evaluator bug confirmed yet.
- Check current README claims and result artifacts against their exact source lines; much latest numerical evidence comes from one training seed and 500 paired trajectories, which does not measure training-seed uncertainty.
- Confirm exact wording and severity of findings before producing the final review. Lead with concrete errors; distinguish code errors, theoretical assumptions, stale docs, and unsupported inferences.

## Runtime/access and agents

Ordinary shell execution fails before commands run: `bwrap: Creating new namespace failed: No space left on device`. Unsandboxed reads sometimes succeed; automatic approval repeatedly timed out. User explicitly approved **unsandboxed read-only shell access** for this audit via the async question. Do not ask again. User subsequently requested this handoff file, authorizing its creation.

Review agents: `/root/theory_core` (theory/core formulas), `/root/training_audit` (trainer/estimators), `/root/readme_metrics` (README/evaluation). Inspect their messages/status on resume; several were retrying reads after approval. A root `functions.exec` cell with ID `7` was running when interrupted, containing a full numbered theory read plus sampler-helper/confinement searches and README lines 1950–2100. If still available, collect its output before repeating those reads.
