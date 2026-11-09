# Design note: fast bistable hidden units in the physical diffusion model

The idea: add K hidden (latent) bistable oscillators that couple to the 784 pixel oscillators and equilibrate much faster than they do. Seen from the pixels, the hidden units are then averaged out, and the pixels move in a *free-energy* landscape that contains a global, non-Gaussian many-body part the present two-body energy lacks.

This note derives four things:
1. the effective free energy and what it can and cannot add;
2. the score-matching objective and its gradient, and why the learning rule stays local;
3. how the reverse (generative) process has to be run;
4. the conditions on the physical hidden units.

It ends with a plan for a first in-silico prototype.

**Verification.**
- The formulas of §1–§3 are checked against autodiff by `check_hidden_unit_formulas.py` (see README).
- An independent review (a blind re-derivation, plus a physics review and a code review) confirmed every formula. It corrected several statements around them: the mixture direction and shape in §1.1, the stationary point at W = 0 in §2, the alternatives in §3, and the noise and time-scale conditions in §4. This version includes those corrections.

**Conventions** (as in CODE_NOTES §3–§6 and the `ode_vs_sde` study):
- τ is forward (noising) time;
- T = 0.02 is both the bath temperature and the forward diffusion constant;
- σ_f = γ = k_B = 1;
- the forward process is dx = −x dτ + √(2T) dW.

## 1. Model

- **Variables.** Visible units x ∈ ℝᴺ (the N = 784 pixels) and hidden units y ∈ ℝᴷ.
- **Joint energy** at slice τ:

$$E_\theta(x,y)=E^{\rm v}_{\theta_v}(x)+\sum_{j=1}^K\big[U_j(y_j)-y_j\,a_j(x)\big],\qquad a_j(x)=b_j+\sum_i W_{ji}x_i .$$

  - E^v is the present energy: self terms plus pair couplings, linear in θ_v.
  - U_j is the hidden unit's own potential, for example a double well −½κy² + ¼λy⁴.
  - The hidden units couple bilinearly to the pixels. There are no hidden–hidden couplings (bipartite, as in an RBM); that keeps everything below local (§2).
  - **Spring couplings, two hardware caveats.** A spring c(x_i − y_j)² has the bilinear form, plus self parts.
    - A passive spring only gives W_ji = 2c ≥ 0. A 0-vs-1 discriminant needs couplings of both signs, hence active couplings.
    - Absorbing Σ_i c_ji y_j² into U_j makes the hidden barrier depend on the learned couplings. Either retune κ as W changes, or use a coupling without the y² self term. This is harmless only in the strict Ising limit.
- **Ising limit.** For deep wells with minima at ±y₀, y_j → y₀h_j with h_j ∈ {±1}; y₀ is absorbed into W and b.
- **Linearity in θ.** The joint energy is linear in all parameters θ = (θ_v, W, b), as before. The *marginal* below is not.
- **Marginal.** Integrating out the hidden units gives

$$p_\theta(x)\propto\int dy\,e^{-E_\theta(x,y)/T}=e^{-F_\theta(x)/T},\qquad F_\theta(x)=E^{\rm v}(x)-T\sum_j\log Z_j\big(a_j(x)\big),\qquad Z_j(a)=\int dy\,e^{-(U_j(y)-ya)/T}.$$

  In the Ising limit, Z_j(a) = 2 cosh(a/T).
- **Conditional cumulants.** Write m_j, v_j and k3_j for the mean, variance and third cumulant of y_j under p(y_j | x) ∝ e^{−(U_j − y_j a_j)/T}.
  - log Z_j is, up to a constant, the cumulant generating function of y_j at s = a_j/T, with base measure ∝ e^{−U_j/T}. Hence

$$T\,\partial_a\log Z_j=m_j,\qquad \partial_a m_j=v_j/T,\qquad \partial_a v_j=k3_j/T .$$

  - Ising: m = tanh(a/T), v = 1 − m², k3 = −2mv.
- **Derivatives of F.** Here W_j ∈ ℝᴺ is row j of W.
  - ψ ≡ ∇F is the *mean restoring gradient*: the pixels feel the mean force −ψ, and ψ_i is the force a clamp must supply to hold pixel i.

$$\psi\equiv\nabla F=\nabla E^{\rm v}-\sum_j m_j W_j,\qquad \nabla\nabla F=\nabla\nabla E^{\rm v}-\frac1T\sum_j v_j\,W_jW_j^{\top},\qquad \Delta F=\Delta E^{\rm v}-\frac1T\sum_j v_j\lVert W_j\rVert^2 .$$

  - The hidden units exert the mean force +Σ_j m_j W_j on the pixels.
  - **Softening along W_j.** Along the unit vector Ŵ_j = W_j/‖W_j‖, a hidden unit lowers the curvature by v_j‖W_j‖²/T.
  - Near a_j = 0 (v_j ≈ 1) it turns a single well along Ŵ_j into a double well wherever ‖W_j‖²/T exceeds the visible curvature Ŵ_jᵀ∇∇E^vŴ_j.

### 1.1 What hidden units add, and what they do not

- **Mixture of tilted copies.** In the Ising limit,

$$p_\theta(x)\propto\sum_{h\in\{\pm1\}^K}\exp\Big(-\big[E^{\rm v}(x)-\textstyle\sum_j h_j a_j(x)\big]/T\Big),$$

  a mixture of 2ᴷ exponentially tilted copies of the visible model.
  - **Quadratic visible energy.** For E^v = ½xᵀAx − cᵀx, the components are Gaussians with means A⁻¹(c + Σ_j h_jW_j) and common covariance TA⁻¹. After whitening this is a Gaussian–Bernoulli RBM.
  - **Direction of the split.** One hidden unit with W_1 = ½A(μ_1 − μ_0) and b_1 = −W_1·(μ_0 + μ_1)/2 puts the two components at the class means μ_0 and μ_1.
    - This is the Fisher (LDA) direction. It is parallel to μ_1 − μ_0 only if A ∝ I, and A is strongly anisotropic here.
    - P(h = +1 | x) = σ(2a_1/T) is then the LDA posterior.
- **Tilted copies share their shape.** A bilinear coupling only *shifts* the components; for quadratic E^v it does so exactly.
  - Class-specific shape, such as the different pixel covariances of rings and strokes, must come from the anharmonic part of E^v.
  - If generated 0s and 1s keep the wrong per-class pixel statistics, the next step is hidden units that *gate* pair couplings (terms y_j x_i x_k), not more bilinear units.
- **Mode-selection force.** The mixture's score contains a force like W_{1i}·tanh(a_1(x)/T) on pixel i: a nonlinear function of a global projection of all pixels.
  - **A two-body energy cannot represent this force exactly.** Its forces satisfy ∂_k∂_l f_i = 0 for distinct k, l ≠ i; the tanh force does not.
  - **Expansion.** −T log cosh(a/T) = −a²/(2T) + a⁴/(12T³) − … (for |a| < πT/2) contains products x_i x_k x_l x_m of four distinct pixels, and the mixed fourth derivative is non-zero in general.
  - This is the thermal analogue of how fast hidden neurons produce dense interactions in Krotov & Hopfield (2021), whose model is deterministic.
- **This does not prove that the present model cannot select modes.**
  - Pair energies with bistable self terms can have collective modes (Hopfield networks, mean-field ferromagnets).
  - The a² part of the hidden free energy is itself a pair term. Ising units add the non-Gaussian many-body remainder.
  - Whether the 0/1 split needs that remainder is an empirical question. The blend check (README) shows that the present model fails at the split; the prototype tests whether hidden units fix it.
- **Wells and confinement.**
  - Hidden units cannot provide confinement. For Ising units their force is bounded by Σ_j‖W_j‖, so the sextic self term (k_6 ≥ 0.01) dominates as ‖x‖ → ∞.
  - They can shift, remove or split wells wherever ‖∇E^v‖ ≲ Σ_j‖W_j‖, including far from the data along the hyperplanes a_j = 0.
  - Whether the deep wells of E_θ(0) found in the `ode_vs_sde` study survive has to be checked on a trained model.

## 2. Training objective and the local learning rule

**Objective.** The per-slice objective is the one the code already uses (CODE_NOTES §5.1), with F in place of E:

$$J_T(\theta)=\mathbb E_{x\sim p_\tau}\Big[-\frac{\Delta F}{T}+\frac{\lVert\nabla F\rVert^2}{2T^2}\Big],\qquad \partial_\theta J_T=\mathbb E\Big[-\frac{\partial_\theta\Delta F}{T}+\frac{(\partial_\theta\nabla F)^{\top}\psi}{T^2}\Big].$$

**Gradients.** Using §1:

- **Visible parameters** θ_k (E^v = Σ θ_k φ_k): the form is unchanged, but ψ now includes the hidden-averaged part.

$$\partial_{\theta_k}J_T=\mathbb E\Big[-\frac{\Delta\phi_k}{T}+\frac{\nabla\phi_k\cdot\psi}{T^2}\Big].$$

- **Hidden couplings and biases:**

$$\partial_{W_{ji}}J_T=\frac1{T^2}\,\mathbb E\Big[-m_j\psi_i+2v_jW_{ji}+\frac{x_i}{T}\big(k3_j\lVert W_j\rVert^2-v_j\,W_j\!\cdot\!\psi\big)\Big],\qquad
\partial_{b_j}J_T=\frac1{T^3}\,\mathbb E\Big[k3_j\lVert W_j\rVert^2-v_j\,W_j\!\cdot\!\psi\Big].$$

  - **Derivation sketch.** ∂_{W_ji}ψ = −(v_j x_i/T)W_j − m_j e_i, and ∂_{W_ji}ΔF = −(k3_j x_i/T²)‖W_j‖² − 2v_jW_ji/T.
  - For b_j, set x_i → 1 and drop the two terms that come from the explicit dependence on W (−m_j e_i and −2v_jW_ji/T). That gives ∂_{b_j}ψ = −(v_j/T)W_j and ∂_{b_j}ΔF = −(k3_j/T²)‖W_j‖².

**Locality.** Each update needs:
- **at visible node i:** its position x_i, and ψ_i, the force a clamp must supply to hold it;
- **at hidden node j:** the statistics of its own fluctuations (m_j, v_j, k3_j), which are time averages of the fast unit while the pixels are held; the sum of its squared couplings ‖W_j‖², a property of the node; and W_j·ψ;
- **the coupling itself:** W_ji.

How node j gets W_j·ψ:
- It cannot sense it at a clamped configuration, because node j only sees its input field a_j.
- W_j·ψ is minus the drift of a_j when the pixels are briefly released (ȧ_j = −W_j·ψ + noise). Equivalently, it is the change of a_j under a small nudge x → x − εψ.
- So the analytic rule needs short releases from data configurations, but **no equilibrium (free-running) negative phase**.

With Ising units, v_j = 1 − m_j² and k3_j = −2m_jv_j, so both this rule and the CD-1 form below need only m_j.

**CD-1 form (simpler for hardware).** Hyvärinen's (2007) relation between CD-1 with a Langevin step and score matching holds for any smooth F, so it holds for the free energy.
- Take x′ = x − (dt/2)ψ(x) + √(T dt) ξ: a short release of the pixels, during which the fast hidden units keep equilibrating.
- Then, to O(dt²),

$$\langle\partial_\theta F(x)\rangle-\langle\partial_\theta F(x')\rangle=\frac{T^2dt}{2}\,\partial_\theta J_T .$$

- With ∂F/∂W_ji = −m_j x_i and ∂F/∂b_j = −m_j, the descent update is

$$\Delta W_{ji}\propto\langle m_j x_i\rangle_{\rm data}-\langle m_j x_i\rangle_{\rm after\ release},\qquad \Delta b_j\propto\langle m_j\rangle_{\rm data}-\langle m_j\rangle_{\rm after\ release}.$$

  This is a Hebbian term minus the same term after a short release: the Boltzmann-machine rule with a short release instead of a full negative phase. The visible parameters keep their present CD-1 rule.
- **The price is noise.** Per release, the difference has mean O(dt) but spread O(√dt). Its signal-to-noise ratio is therefore O(√dt), and accuracy at small dt costs about 1/dt releases. The release-based W_j·ψ above has the same trade-off.
- **Measured hidden states make it far worse.** The update needs m_j (or p_m) at x and at x′.
  - With exact averages their difference is O(dt).
  - A device reads sampled hidden states. Independent reads at x and x′ keep their full sampling noise, O(1/√N) for N reads, in the difference. The signal-to-noise ratio becomes O(dt·√N), against O(√(dt·R)) for R releases with exact averages.
  - Matching the exact-average rule therefore takes N ≈ R/dt reads per configuration, times model-dependent factors. For the 14x14 model at dt = 3e-3 and R = 64 (R/dt ≈ 2·10⁴), the measured requirement is 4·10³-9·10⁷ reads (README Result 10, point 3).
  - The force-matching rule reads the hidden units at one clamped configuration, so its sampling noise is not amplified (§2.1).

**What changes for the optimiser.**
- **J_T is no longer a convex quadratic** (CODE_NOTES §5.1), because F is nonlinear in (W, b).
- **There is a gauge symmetry.** For symmetric U_j, (W_j, b_j, y_j) → (−W_j, −b_j, −y_j) leaves F unchanged, and so do permutations of identical hidden units.
- **W = 0 is a degenerate stationary point.**
  - At W = 0, ∂J/∂W_ji = −(m_j/T²)E[ψ_i]. That vanishes at the visible-bias optimum, since E[ψ_i] = T²·∂J/∂(visible bias i).
  - Worse, to second order in W the hidden free energy adds only a constant, a linear and a quadratic term in x. The visible bias and pair couplings absorb these wherever E^v couples the pixels that W_j touches; 59% of all pixel pairs are coupled.
  - At the visible optimum, J therefore changes only at O(‖W‖³) (b_j ≠ 0) or O(‖W‖⁴) (b_j = 0). Small random W gets almost no gradient and can decay back to 0.
  - A saturated unit (|a_j| ≫ T with one sign on all samples) is dead as well.
  - Initialisation therefore matters (§5).
- **The temperature reparametrisation survives** in the Ising limit: F/T depends on θ only through u = θ/T. For a fixed hardware potential U_j it does not.

### 2.1 Potts units, spring couplings and per-pixel units: the local force-matching rule

The models of README Results 7-9 have one Potts unit (states m = 1…M, exactly one active) whose input field also
contains the squared pixels, plus one Ising unit per pixel:

$$a_m=b_m+W_m\cdot x+Q_m\cdot x^{2},\qquad F=E^{\rm v}-T\log\sum_m e^{a_m/T}-T\sum_i\log 2\cosh\frac{c_i+w_ix_i}{T}.$$

**Per clamped sample x, each hidden state and pixel provides:**
- **State m:**
  - its occupation p_m (time average of the fast Potts unit);
  - the gradient of its input field, G_m = W_m + 2Q_m∘x;
  - s_m = Σ_i Q_mi.
- **The Potts unit as a whole:**
  - its mean force on the pixels, Ḡ = Σ_k p_k G_k;
  - V = Σ_k p_k |G_k − Ḡ|²;
  - s̄ = Σ_k p_k s_k.
- **Pixel i:** x_i and the clamp force ψ_i = ∂_iF, which includes the hidden units' mean force.

**One scalar per state:**

$$c_m=\lvert G_m-\bar G\rvert^2-V+2T(s_m-\bar s)-(G_m-\bar G)\cdot\psi .$$

**The force-matching (score-matching) gradient of J_T** (§2) is then

$$\partial_{b_m}J_T=\frac{\langle p_m c_m\rangle}{T^3},\qquad
\partial_{W_{mi}}J_T=\Big\langle p_m\Big[\frac{x_ic_m}{T^3}+\frac{2(G_{mi}-\bar G_i)-\psi_i}{T^2}\Big]\Big\rangle,$$

$$\partial_{Q_{mi}}J_T=\Big\langle p_m\Big[\frac{x_i^2c_m}{T^3}+\frac{2x_i\,(2(G_{mi}-\bar G_i)-\psi_i)}{T^2}+\frac2T\Big]\Big\rangle .$$

**Per-pixel unit i** (a_i = c_i + w_ix_i, m_i = tanh(a_i/T), v_i = 1 − m_i², k3_i = −2m_iv_i) is the Ising rule of §2 with one coupling:

$$\partial_{w_i}J_T=\frac{\langle -m_i\psi_i+2v_iw_i+\tfrac{x_i}{T}(k3_iw_i^2-v_iw_i\psi_i)\rangle}{T^2},\qquad \partial_{c_i}J_T=\frac{\langle k3_iw_i^2-v_iw_i\psi_i\rangle}{T^3}.$$

**Derivation sketch.**
- ∂p_k/∂θ = (p_k/T)(δ_km − p_m)·∂a_m for a parameter of state m.
- ∂∇F = −∂Ḡ = −(p_m/T)(G_m − Ḡ)∂a_m − p_m∂G_m.
- ΔF = ΔE^v − V/T − 2s̄ − (1/T)Σ_iv_iw_i².
- Insert these into ∂J_T = ⟨−∂ΔF/T + ψ·∂∇F/T²⟩.

**Locality.**
- **Coupler (m, i)** needs:
  - p_m and c_m from its state;
  - x_i, ψ_i and Ḡ_i from its pixel;
  - its own G_mi.

  It is a three-factor rule: a state-level scalar times a pixel signal, plus a pixel-level term gated by p_m.
- **The bias b_m** needs only its state's p_m c_m: the state-level part of the coupler rule with x_i → 1. The
  pixel-level term drops, because b_m does not enter G_m.
- **Inside c_m, the non-local inputs** are the projections G_m·ψ and G_m·Ḡ, and |G_m|².
  - G_m·ψ and G_m·Ḡ are the changes of state m's input field when the clamped pixels are nudged along ψ or along Ḡ. This is the analogue of W_j·ψ in §2.
  - |G_m|² follows from the couplings, or from the fluctuations of a_m during a short release.
  - All other factors are averages over the Potts unit's own states.
- **Measuring the projections.**
  - a_m is quadratic in x, so a symmetric pair of nudges ±εu gives the projection exactly, for any nudge size:
    [a_m(x+εu) − a_m(x−εu)]/(2ε) = G_m·u.
  - The pixel-dependent terms of c_m can also be read from the Potts unit's own occupations. Since
    ∇_x log p_m = (G_m − Ḡ)/T:
    - (G_m − Ḡ)·ψ is T times the response of log p_m to a nudge of the clamped pixels along ψ;
    - |G_m − Ḡ|² = T²|∇_x log p_m|². A short noisy release of length dt measures it as T·Var(log p_m)/dt, and its
      p-weighted sum over the states is V.
  - What remains is s_m − s̄, which follows from the couplings: s_m = Σ_i Q_mi is minus state m's total spring
    constant (state m adds −Q_mi x_i² to pixel i's energy, so Q < 0 stiffens).
- **The per-pixel units** need only their own pixel.

**Checked.** `latent_model.force_matching_local_grad` implements these formulas. It equals jax.grad of the training loss to relative differences of 1e-16 to 4e-11 (`cd1_check.py --local-rule`):
- jobs 4197172 (14x14) and 4197173 (28x28);
- trained Potts 128 + both, slices 1, 50, 90 and 99.

Every model of README Results 7-9 was therefore trained with this local rule, given exact hidden statistics.

**Measured instead of exact statistics.**
- On a device, p_m (and m_i) are averages over S hidden states sampled while the pixels are clamped.
- The fluctuation terms (the factors (δ_km − p_m), V and the like) are the corresponding sample covariances.
- `--hidden-samples S` trains with exactly these plug-in estimates (`latent_model.sampled_state_mean`).
- The resulting gradient converges to the exact one as 1/√S (README Result 10).

## 3. The reverse (generative) process

**Target.** The reverse-time SDE (t = 5 − τ) must have drift x + 2T∇log p_τ = x − 2∇F_{θ(τ)}(x) at pixel temperature T. The pixels therefore need the effective potential 2F − ½‖x‖².

**Visible-only model.** Here F = E is linear in θ, so this is E_{θ_rev} with θ_rev = 2θ − θ_lin (CODE_NOTES §6).

**With hidden units, doubling θ is not enough.** Doubling W and b with the hidden units at T gives −T log 2cosh(2a/T), not −2T log 2cosh(a/T); the check script confirms the force is wrong.

**Three exact constructions** (fast-hidden limit). In all three the pixels use θ_rev on the visible part, as before.

1. **Replicas at the bath temperature (preferred).**
   - For generation, duplicate every hidden unit: 2K units at T, each copy with the trained W_j, b_j and U_j.
   - The hidden free energy is −T Σ_j 2 log Z_j(a_j) = −2T Σ_j log Z_j(a_j), so F′ = 2F − ½‖x‖² exactly, for any U_j.
   - Everything stays at one temperature with reciprocal couplings. At every instant the joint system has the Boltzmann state e^{−E′/T}, no separate noise source is needed, and a finite hidden correlation time causes no heating (§4).
   - The cost is 2K hidden units, of which K are decoupled during training.
2. **Doubled energy at 2T.**
   - The same K units, with the joint energy doubled (U_j, W, b) and the hidden units at T_h = 2T.
   - The hidden statistics are unchanged (e^{−2E/2T} = e^{−E/T}); only the force on the pixels doubles.
   - It needs a hidden noise source at 2T and a potential U_j that can be doubled.
3. **Amplified back-action.**
   - The same K units at T, with potential U_j and input field a_j, but a force 2W_ji y_j on pixel i (non-reciprocal).
   - Rescaling time by 2 maps construction 2 onto exactly this dynamics, with the hidden units twice as fast. Construction 2 is therefore no more "energy-based" than construction 3: neither has a joint Boltzmann state.
   - For a fixed hardware U_j that cannot be doubled, construction 3 is the exact realisation of construction 2.

**Uniqueness.**
- Within the same K two-state units, with one common T_h and a′ = c·a, the solution c = ±2, T_h = 2T is unique up to the gauge flip. Matching at small and large a forces it.
- It is not unique in general. Any set of copies at temperatures T_c, with couplings (T_c/T)(W_j, b_j), potentials (T_c/T)U_j and Σ_c T_c = 2T, works: replicas are two copies at T, construction 2 is one copy at 2T.
- For Gaussian units, T_h is not fixed at all.

**PF-ODE variant.** Noise-free pixels with E′ = E − ½‖x‖², and the hidden units at T with unchanged couplings. The `ode_vs_sde` study found the ODE worse, so this is not the default.

**In simulation.** Use the reverse drift x − 2∇F(x; θ(τ)) directly, with ∇F from §1. This is what all three constructions produce in the fast-hidden limit. Do **not** feed the hidden block through the existing θ_rev = 2θ − θ_lin arithmetic.

## 4. Physical hidden units: when the idealisation holds

**Adiabaticity.** The hidden units must follow their input field a_j. Two clocks set how fast they must be:
- **The pixel relaxation time.**
  - Near τ = 0 the slice stiffness of a background pixel is k* = T/(e^{−2τ}s² + T(1 − e^{−2τ})) ≈ T/s² ≈ 18.5, where s² ≈ 1.08e-3 is the variance of the added pixel noise in normalised units (CODE_NOTES §4).
  - In generation the pixels move in 2F − ½‖x‖², with stiffness 2k − 1 ≈ 36. So the binding relaxation time is ≈ 0.03 (γ = 1).
- **The field's own diffusion.** Pixel noise alone moves a_j by ~T in a time ≈ T/(2‖W_j‖²). That is shorter than the relaxation time whenever the unit is strong enough to split a well (‖W_j‖²/T ≳ 10).
- **Condition:** τ′_c ≪ min(1/(2k − 1), T/(2‖W_j‖²)), where τ′_c is the hidden correlation time in the generation configuration.

**Finite τ_c: friction always, heating only with two temperatures.**
- **At equal temperatures** (clamped training, CD-1 releases, replica generation) the joint dynamics obeys detailed balance with respect to e^{−E/T}. The pixel marginal stays exactly e^{−F/T} for any τ_c. A finite τ_c only adds lag friction along Ŵ_j, which slows the pixels there.
- **With two temperatures** (constructions 2 and 3) the hidden fluctuations also heat the pixels along Ŵ_j.
  - Both the heating and the loss of pixel mobility along Ŵ_j are of relative order Γ_j = 2v_jτ′_c‖W_j‖²/T, so require Γ_j ≪ 1.
  - Replicas avoid the heating.

**Bistable versus fast.** These pull in opposite directions.
- Ising-like behaviour wants a barrier ΔU ≫ T_h. Then intra-well spread is small, and the coupling field does not tilt away one of the wells.
- Fast switching wants a modest barrier, since the Kramers rate falls as e^{−ΔU/T_h}.
- **Ways out:** a much higher attempt frequency than the pixels, with damping near the Kramers turnover (γ ~ ω_b), and barriers of a few T_h. In the underdamped, energy-diffusion-limited regime, less damping *slows* switching.
- Under construction 2 the barrier-to-temperature ratio is unchanged. The switching rate doubles for overdamped units and rises less for underdamped ones; deep in the energy-diffusion regime it does not change. Replicas leave the hidden dynamics unchanged.
- **The derivation does not need the Ising limit:** §1–§2 hold for any U_j, with m, v and k3 the true conditional cumulants. In simulation, Z_j(a) for a double well is a 1D integral that can be tabulated.

**Measurement noise.**
- m_j, v_j and k3_j are time averages over a finite window, so their estimation error enters the updates as noise.
- Ising units need only m_j.
- The CD-1 form trades this for its per-release O(√dt) noise (§2).

## 5. Plan for a first in-silico prototype

1. **Parameter layout.** Append W (K × 784) and b_h (K) to the flat θ, after the visible bias. θ_lin gets zeros in the new block.
2. **Loss.** Add the hidden terms to the analytic ∇E and ΔE in `setup_score_matching_kbT_loss_minimal_memory`, using §1 with Ising cumulants. The θ-gradient stays autodiff; the analytic §2 rule is for the hardware argument and for checks.
3. **Sampler and metrics.**
   - Reverse drift x − 2∇F(x; θ(τ)) (§3).
   - Metrics: % far, the blend fraction and clean-0 fraction from `blend_check.py`, % diverged, class balance.
   - Also the per-class pixel mean and variance of generated vs real samples, which tests the shared-shape limitation of §1.1.
4. **Initialisation and continuity.**
   - **Do not start from small random W**; W = 0 is degenerate (§2). Initialise at finite scale along a bimodal direction:
     - W_j ∝ A(μ_1 − μ_0), with A ≈ ∇∇E^v of the slice's visible model and μ_0, μ_1 the class means of the slice's data;
     - scaled so that ‖W_j‖²/T is comparable to Ŵ_jᵀAŴ_j;
     - b_j = −W_j·(μ_0 + μ_1)/2.
     - For K > 1, use further bimodal directions (for example within-class modes) or perturbations of that one.
   - **Order of the slices.** Consider training in increasing τ, starting near τ = 0 where the classes are well separated and warm-starting toward larger τ: the reverse of the present order. Otherwise, re-seed W_j at any slice where ‖W_j‖ becomes small.
   - At initialisation, check that J decreases along ±W_j.
   - Monitor ‖W_j‖ per slice.
   - **Align the gauge before interpolating between knots:** fix the signs and the order of the hidden units by maximising W_j(τ_k)·W_j(τ_{k+1}). A sign flip between knots would make the interpolated W pass through 0.
5. **Runs.**
   - K = 1, 4 and 16 against the present model (35.6% far and 43% flagged blends, SDE, 500 trajectories).
   - J_T is non-convex, so repeat at least one K with a second initialisation to measure the run-to-run spread.
   - With 500 trajectories per arm, each fraction has an SE of about 2.2 points, so unpaired differences below about 6 points are not readable. Use about 2,000 trajectories for the final comparison.
   - Each training takes about 2.7 days, like the paper's; the runs can go in parallel. The hidden block adds only K × 785 parameters to the 545,566.
6. **Reference.** Idea 1, one model trained on 0s only and one on 1s only, removes superposition by construction. It shows what this architecture reaches on one class.

## References

- Hyvärinen, A. Connections between score matching, contrastive divergence, and pseudolikelihood for continuous-valued variables. IEEE Trans. Neural Networks 18(5), 1529–1531 (2007).
- Krotov, D. & Hopfield, J. J. Large associative memory problem in neurobiology and machine learning. ICLR 2021 (arXiv:2008.06996).
- Song, Y. et al. Score-based generative modeling through stochastic differential equations. ICLR 2021.
- Hinton, G. E. Training products of experts by minimizing contrastive divergence. Neural Computation 14(8), 1771–1800 (2002).
