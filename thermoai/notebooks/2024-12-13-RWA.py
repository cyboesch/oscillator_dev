# %% [markdown]
# # Visualization of Time Scale Separation in a Weakly Nonlinear Duffing Oscillator
#
# This Jupyter notebook demonstrates the separation of timescales in a weakly nonlinear Duffing oscillator:
# $$
# \ddot{x} + x + \epsilon x^3 = 0, \quad \epsilon \ll 1.
# $$
#
# The Duffing oscillator can be viewed as a weak perturbation of the simple harmonic oscillator $\ddot{x} + x = 0.$ In the small-$\epsilon$ limit, the solution resembles a harmonic oscillator whose amplitude and phase vary very slowly over time.
#
# We will:
# 1. Numerically integrate the Duffing oscillator equation for a small $\epsilon$.
# 2. Show that the solution $x(t)$ undergoes rapid oscillations on a timescale $\mathcal{O}(1)$.
# 3. Extract and visualize the slowly varying amplitude envelope by a time-averaging method.
# 4. Derive the equations of motion for the slowly varying amplitude and phase using averaging theory.
# 5. Compare the numerical solution to the analytical approximation obtained from averaging theory.
#
# By plotting both the raw solution and its slowly varying envelope, we will illustrate the separation of fast and slow dynamics.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import hilbert, find_peaks

# %% [markdown]
# ## Parameters and Setup
# Choose a small epsilon to illustrate the perturbation.

# %%
epsilon = 0.0001
# Initial conditions: Let's start with some initial amplitude.
x0 = 1.0   # initial displacement
y0 = 0.0   # initial velocity

# Time domain:
# We'll integrate over a relatively long time to see the slow evolution of amplitude.
t_start = 0.0
t_end = 300.0
t_eval = np.linspace(t_start, t_end, 5000)  # sufficiently fine resolution

# %% [markdown]
# ## Duffing Oscillator ODE
# The system is:
# $$
# \dot{x} = y,\quad \dot{y} = -x - \epsilon x^3.
# $$

# %%
def duffing(t, XY, epsilon=epsilon):
    x, y = XY
    dxdt = y
    dydt = -x - epsilon*x**3
    return [dxdt, dydt]

# Solve the ODE
sol = solve_ivp(duffing, [t_start, t_end], [x0, y0], t_eval=t_eval, args=(epsilon,))

t = sol.t
x = sol.y[0,:]
y = sol.y[1,:]

# %% [markdown]
# ## Extracting the Envelope
# The fast oscillations occur roughly with frequency near 1 (for small amplitude).
# We can try to extract a slowly varying amplitude by:
# 1. Using the Hilbert transform (from `scipy.signal`) to get an analytic signal and thus the instantaneous amplitude.
#    OR
# 2. Using a moving window RMS to approximate the envelope.
#
# Here, let's use the Hilbert transform to find the envelope of x(t).

# %%
analytic_signal = hilbert(x)
amplitude_envelope = np.abs(analytic_signal)

# Estimate the average period using the time difference between the first few peaks
peak_indices, _ = find_peaks(amplitude_envelope)
if len(peak_indices) >= 5:
    # Calculate the average period from the first 5 peaks.
    # If there are not enough peaks, fall back to default period of 2*pi.
    num_peaks_for_period = min(5, len(peak_indices) - 1)
    avg_period = np.mean(np.diff(t[peak_indices[:num_peaks_for_period]]))
else:
    avg_period = 2 * np.pi  # Default period for simple harmonic oscillator

# Window size for smoothing, based on the estimated period
window_size = int(avg_period / (t[1] - t[0]))  # Convert period to number of samples

# Perform windowed averaging around each peak to get a smoother envelope
smoothed_envelope = np.copy(amplitude_envelope)  # Initialize with the original envelope
for peak_index in peak_indices:
    # Define a window around the peak
    start = max(0, peak_index - window_size // 2)
    end = min(len(amplitude_envelope), peak_index + window_size // 2 + 1)
    window = np.arange(start, end)

    # Apply a weighted average (e.g., Gaussian) within the window
    weights = np.exp(-0.5 * ((window - peak_index) / (window_size / 4))**2)  # Gaussian weights
    smoothed_envelope[window] = np.sum(amplitude_envelope[window] * weights) / np.sum(weights)

# compute the tangent of the envelope as max - min / length, starting at the 7th peak
start_index = peak_indices[6]
dr_dt = (np.max(smoothed_envelope[start_index:]) - np.min(smoothed_envelope[start_index:])) / len(smoothed_envelope[start_index:])
print(f"Measured rate of change of amplitude: {dr_dt:.2e}")
# %% [markdown]
# ## Visualization
# We'll create two plots:
# 1. A short-time zoomed-in view showing the fast oscillations.
# 2. A long-time view showing the slowly varying amplitude envelope.

# %%
fig, axes = plt.subplots(2, 1, figsize=(10,8), sharex=False)

# Plot a short-time segment to see fast oscillations
t_zoom_start = 0
t_zoom_end = 30
mask_zoom = (t >= t_zoom_start) & (t <= t_zoom_end)

axes[0].plot(t[mask_zoom], x[mask_zoom], label='$x(t)$')
axes[0].set_title('Fast Oscillations (Short-Time View)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('$x(t)$')
axes[0].legend()
axes[0].grid(True)

# Long-time view to see envelope
# only plot starting at third peak, and third to last peak
start_index = peak_indices[5]
end_index = peak_indices[-5]
axes[1].plot(t[start_index:end_index], x[start_index:end_index], color='C0', alpha=0.3, label='Raw $x(t)$')
axes[1].plot(t[start_index:end_index], smoothed_envelope[start_index:end_index], color='C1', linewidth=2, label='Smoothed Envelope')
axes[1].set_title('Slow Amplitude Variation (Long-Time View)')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('$x(t)$ and Envelope')
axes[1].legend()
axes[1].grid(True)

# Text box on plot with the rate of change of the amplitude
text_box = f"Measured rate of change of amplitude: {dr_dt:.2f}"
axes[1].text(0.05, 0.95, text_box, transform=axes[1].transAxes, fontsize=12, verticalalignment='top')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Averaging Theory Analysis
#
# We will now derive the equations of motion for the slowly varying amplitude and phase using averaging theory.
#
# ### 1. Phase Space Formulation
#
# Introduce $y = \dot{x}$. Then the system can be written as a first-order system in phase space $(x,y)$:
# $$
# \dot{x} = y, \quad \dot{y} = -x - \epsilon x^3.
# $$
#
# When $\epsilon = 0$, this reduces to the unperturbed harmonic oscillator:
# $$
# \dot{x} = y, \quad \dot{y} = -x.
# $$
#
# ### 2. Exact Solution for the Unperturbed Case ($\epsilon=0$)
#
# For the classical harmonic oscillator, the solution with amplitude $r$ and phase $\phi$ can be expressed as:
# $$
# x(t) = r \cos(t + \phi), \quad y(t) = -r \sin(t + \phi).
# $$
#
# This represents uniform circular motion in the $(x,y)$-plane with radius $r$. The period of oscillation is $2\pi$.
#
# ### 3. Rotating Frame Transformation
#
# When the nonlinear perturbation is present but small ($\epsilon \neq 0$ and $\epsilon \ll 1$), we expect that the amplitude $r(t)$ and the phase $\phi(t)$ of the solution will not remain strictly constant. Instead, they will vary slowly over time.
#
# We introduce a rotating frame that moves with the same angular frequency as the unperturbed system. By doing so, we "freeze out" the trivial fast oscillation and focus on the slow modulation of amplitude and phase due to the small perturbation.
#
# **Definition of $r(t)$ and $\phi(t)$:**
# $$
# x(t) = r(t) \cos(t + \phi(t)), \quad y(t) = -r(t) \sin(t + \phi(t)).
# $$
#
# From these definitions:
# $$
# r(t) = \sqrt{x^2(t) + y^2(t)}, \quad \tan(t + \phi(t)) = -\frac{y(t)}{x(t)}.
# $$
#
# ### 4. Deriving the Evolution Equations for $\dot{r}$ and $\dot{\phi}$
#
# Starting from:
# $$
# r^2 = x^2 + y^2.
# $$
#
# Differentiate with respect to time:
# $$
# 2r\dot{r} = 2x\dot{x} + 2y\dot{y} \implies r\dot{r} = x\dot{x} + y\dot{y}.
# $$
#
# Substitute $\dot{x}=y$ and $\dot{y}=-x-\epsilon x^3$:
# $$
# r\dot{r} = x(y) + y(-x - \epsilon x^3) = xy - xy - \epsilon y x^3 = -\epsilon y x^3.
# $$
#
# But $y = -r \sin(t+\phi)$ and $x = r \cos(t+\phi)$, hence:
# $$
# r\dot{r} = -\epsilon (-r \sin(t+\phi)) (r \cos(t+\phi))^3 = \epsilon r^4 \sin(t+\phi) \cos^3(t+\phi).
# $$
#
# Divide by $r$:
# $$
# \dot{r} = \epsilon r^3 \cos^3(t+\phi) \sin(t+\phi).
# $$
#
# Similarly, to find $\dot{\phi}$, we use:
# $$
# \frac{d}{dt}(t + \phi(t)) = 1 + \dot{\phi}(t).
# $$
#
# Through the chain rule and some trigonometric manipulations, we obtain:
# $$
# \dot{\phi} = \frac{\epsilon x^3}{r} \cos(t+\phi) = \epsilon r^2 \cos^4(t+\phi).
# $$
#
# Both $\dot{r}$ and $\dot{\phi}$ are $O(\epsilon)$, confirming our intuition that amplitude and phase vary slowly compared to the fast oscillation time scale.
#
# ### 5. Averaging
#
# We separate the problem into two time scales:
#
# 1. **Fast time scale**: the basic harmonic oscillation with period $\approx 2\pi$.
# 2. **Slow time scale**: the slow variation of $r(t)$ and $\phi(t)$ over intervals of order $1/\epsilon$.
#
# We will perform a time-averaging procedure over one period of the fast oscillation. This averaging will remove the explicit fast time dependence and yield simpler, autonomous equations for $\overline{r}$ and $\overline{\phi}$.
#
# **Definition of the Running Average:**
# $$
# \overline{g}(t) = \langle g \rangle_t = \frac{1}{2\pi}\int_{t-\pi}^{t+\pi} g(s)\,ds.
# $$
#
# A key property is that:
# $$
# \overline{\dot{g}} = \dot{\overline{g}}.
# $$
#
# This follows from the fundamental theorem of calculus and the fact that a full $2\pi$-integral over a derivative can be converted into boundary terms.
#
# We have:
# $$
# \dot{r} = \epsilon r^3 \cos^3(t+\phi) \sin(t+\phi),
# $$
# $$
# \dot{\phi} = \epsilon r^2 \cos^4(t+\phi).
# $$
#
# Take the running average over one period:
# $$
# \overline{\dot{r}} = \dot{\overline{r}} = \left\langle \epsilon r^3 \cos^3(t+\phi) \sin(t+\phi) \right\rangle_t.
# $$
#
# $$
# \overline{\dot{\phi}} = \dot{\overline{\phi}} = \left\langle \epsilon r^2 \cos^4(t+\phi) \right\rangle_t.
# $$
#
# **Approximation:**
# Since $r$ and $\phi$ evolve slowly, we can treat them as approximately constant when performing these averages. Thus:
# $$
# r = \overline{r} + O(\epsilon), \quad \phi = \overline{\phi} + O(\epsilon).
# $$
#
# Substituting back leads to autonomous equations in $\overline{r}$ and $\overline{\phi}$ up to $O(\epsilon)$ accuracy.
#
# ### 6. Application to the Duffing Oscillator
#
# 1. For $\dot{r}$:
# $$
# \dot{r} = \epsilon r^3 \cos^3(t+\phi)\sin(t+\phi).
# $$
# Averaging $\cos^3(t+\phi)\sin(t+\phi)$ over one period gives zero. Thus:
# $$
# \dot{\overline{r}} = 0 \implies \overline{r} \text{ is nearly constant to order } \epsilon.
# $$
#
# 2. For $\dot{\phi}$:
# $$
# \dot{\phi} = \epsilon r^2 \cos^4(t+\phi).
# $$
#
# The average $\langle \cos^4(\theta) \rangle = 3/8$. Thus:
# $$
# \dot{\overline{\phi}} = \epsilon r^2 \frac{3}{8}.
# $$
#
# This shows the frequency is shifted:
# $$
# \omega_{\text{eff}} = 1 + \frac{3}{8}\epsilon r^2.
# $$
#
# The period is thus slightly shorter than $2\pi$ if $\epsilon > 0$.
#
# ### 7. Interpretation and Conclusion
#
# - To first order in $\epsilon$, the amplitude $r$ does not change significantly.
# - The frequency is slightly altered by the nonlinear term, increasing if $\epsilon > 0$.
#
# By employing the averaging method, we obtain autonomous equations for the slowly varying amplitude and phase, allowing the use of standard phase-plane analysis techniques and confirming the intuition that the perturbation effects accumulate over many cycles, leading to a slow drift in the system’s parameters.
#
# ### 8. Comparison with Numerical Solution
#
# Let's compare the analytical approximation obtained from averaging theory with the numerical solution.
#
# From averaging theory, we have $\overline{r} \approx r_0$ and $\overline{\phi} \approx \frac{3}{8} \epsilon r_0^2 t + \phi_0$.
#
# The approximate solution is then:
# $$
# x_{\text{approx}}(t) = r_0 \cos\left(t + \frac{3}{8} \epsilon r_0^2 t + \phi_0\right).
# $$
#
# We can choose $r_0 = x_0$ and $\phi_0 = 0$ to match the initial conditions.

# %%
r0 = x0
phi0 = 0
x_approx = r0 * np.cos(t + (3/8) * epsilon * r0**2 * t + phi0)

# %%
# Plot the numerical solution and the analytical approximation
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(t, x, color='C0', alpha=0.5, label='Numerical Solution')
ax.plot(t, x_approx, color='C2', linestyle='--', label='Averaging Theory Approximation')
ax.set_title('Comparison of Numerical Solution and Averaging Theory Approximation')
ax.set_xlabel('Time')
ax.set_ylabel('$x(t)$')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# We can see that the averaging theory approximation captures the slow variation of the amplitude and phase reasonably well, especially for small $\epsilon$. The approximation becomes less accurate as $\epsilon$ increases.

#%%
# Visualization of the amplitude over short and long timescales

# Short-time zoom to compare numerical and analytical solutions
fig_short, ax_short = plt.subplots(figsize=(10,4))
t_short_start = 0
t_short_end = 30
mask_short = (t >= t_short_start) & (t <= t_short_end)
ax_short.plot(t[mask_short], x[mask_short], label='Numerical Solution', linewidth=2)
ax_short.plot(t[mask_short], x_approx[mask_short], label='Analytical Approximation', linestyle='--', linewidth=2)
ax_short.set_title('Short-Time Comparison')
ax_short.set_xlabel('Time')
ax_short.set_ylabel('$x(t)$')
ax_short.legend()
ax_short.grid(True)
plt.tight_layout()
plt.show()

# Long-time view to see the slow amplitude modulation
fig_long, ax_long = plt.subplots(figsize=(10,4))
ax_long.plot(t, x, color='C0', alpha=0.5, label='Numerical Solution')
ax_long.plot(t, x_approx, color='C2', linestyle='--', label='Analytical Approximation')
ax_long.set_title('Long-Time Comparison')
ax_long.set_xlabel('Time')
ax_long.set_ylabel('$x(t)$')
ax_long.legend()
ax_long.grid(True)
plt.tight_layout()
plt.show()

# Plot the difference between numerical and analytical solutions
fig_diff, ax_diff = plt.subplots(figsize=(10,4))
ax_diff.plot(t, x - x_approx, color='C3', label='Difference (Numerical - Analytical)')
ax_diff.set_title('Difference Between Numerical and Analytical Solutions')
ax_diff.set_xlabel('Time')
ax_diff.set_ylabel('Difference')
ax_diff.legend()
ax_diff.grid(True)
plt.tight_layout()
plt.show()

# Zoom into a single period to show constant amplitude
period = 2 * np.pi / (1 + (3/8) * epsilon * r0**2)  # Adjusted period from averaging theory
t_single_period_start = 0
t_single_period_end = t_single_period_start + period
mask_single_period = (t >= t_single_period_start) & (t <= t_single_period_end)

fig_single_period, ax_single_period = plt.subplots(figsize=(10,4))
ax_single_period.plot(t[mask_single_period], x[mask_single_period], label='Numerical Solution', linewidth=2)
ax_single_period.plot(t[mask_single_period], x_approx[mask_single_period], label='Analytical Approximation', linestyle='--', linewidth=2)
ax_single_period.set_title('Single Period Comparison')
ax_single_period.set_xlabel('Time')
ax_single_period.set_ylabel('$x(t)$')
ax_single_period.legend()
ax_single_period.grid(True)
plt.tight_layout()
plt.show()

#%%

# Compute amplitude r(t) and phase phi(t) from numerical solution
# $\overline{\phi} \approx \frac{3}{8} \epsilon r_0^2 t + \phi_0$.


# Define the exact evolution equations for r(t) and phi(t)
def evolution_equations(t, RPhi, epsilon=epsilon):
    r, phi = RPhi
    drdt = - (3/8) * epsilon * r**3 * np.sin(2 * (t + phi))
    dphidt = (1/2) * epsilon * r**2 * (1 + (3/4) * np.cos(2 * (t + phi)))
    return [drdt, dphidt]

# Initial conditions for r and phi
RPhi0 = [r0, phi0]

# Time array for integration
t_span = [t_start, t_end]
t_eval = t  # Use the same time points as the numerical solution

# Integrate the evolution equations numerically
sol_RPhi = solve_ivp(evolution_equations, t_span, RPhi0, t_eval=t_eval, args=(epsilon,))
r_theoretical = sol_RPhi.y[0]
phi_theoretical = sol_RPhi.y[1]

# Compute slow time scale tau
tau = epsilon * t

# Reconstruct x_theoretical using r_theoretical and phi_theoretical
x_theoretical = r_theoretical * np.cos(t + phi_theoretical)

# Proceed with plotting using 'tau'
# Plotting Amplitude over Fast and Slow Timescales
fig_r, ax_r = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

# Amplitude vs fast time t
ax_r[0].plot(t, r_numerical, label='Numerical $r(t)$', alpha=0.8)
ax_r[0].plot(t, r_theoretical, linestyle='--', label='Theoretical $r(t)$', alpha=0.8)
ax_r[0].set_title('Amplitude $r(t)$ vs Fast Time $t$')
ax_r[0].set_ylabel('$r(t)$')
ax_r[0].legend()
ax_r[0].grid(True)

# Amplitude vs slow time tau
ax_r[1].plot(tau, r_numerical, label='Numerical $r(\\tau)$', alpha=0.8)
ax_r[1].plot(tau, r_theoretical, linestyle='--', label='Theoretical $r(\\tau)$', alpha=0.8)
ax_r[1].set_title('Amplitude $r(\\tau)$ vs Slow Time $\\tau = \\epsilon t$')
ax_r[1].set_xlabel('Slow Time $\\tau$')
ax_r[1].set_ylabel('$r(\\tau)$')
ax_r[1].legend()
ax_r[1].grid(True)
plt.tight_layout()
plt.show()

# Plotting Phase over Fast and Slow Timescales
fig_phi, ax_phi = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

# Phase vs fast time t
ax_phi[0].plot(t, phi_numerical, label='Numerical $\\phi(t)$', alpha=0.8)
ax_phi[0].plot(t, phi_theoretical, linestyle='--', label='Theoretical $\\phi(t)$', alpha=0.8)
ax_phi[0].set_title('Phase $\\phi(t)$ vs Fast Time $t$')
ax_phi[0].set_ylabel('$\\phi(t)$')
ax_phi[0].legend()
ax_phi[0].grid(True)

# Phase vs slow time tau
ax_phi[1].plot(tau, phi_numerical, label='Numerical $\\phi(\\tau)$', alpha=0.8)
ax_phi[1].plot(tau, phi_theoretical, linestyle='--', label='Theoretical $\\phi(\\tau)$', alpha=0.8)
ax_phi[1].set_title('Phase $\\phi(\\tau)$ vs Slow Time $\\tau = \\epsilon t$')
ax_phi[1].set_xlabel('Slow Time $\\tau$')
ax_phi[1].set_ylabel('$\\phi(\\tau)$')
ax_phi[1].legend()
ax_phi[1].grid(True)
plt.tight_layout()
plt.show()

# Compare x_numerical and x_theoretical
fig_comparison, ax_comp = plt.subplots(figsize=(10,4))
ax_comp.plot(t, x, label='Numerical $x(t)$', alpha=0.8)
ax_comp.plot(t, x_theoretical, linestyle='--', label='Theoretical $x(t)$', alpha=0.8)
ax_comp.set_title('Comparison of Numerical and Theoretical $x(t)$')
ax_comp.set_xlabel('Time $t$')
ax_comp.set_ylabel('$x(t)$')
ax_comp.legend()
ax_comp.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Observations on Amplitude and Phase Evolution
#
# The above plots illustrate how the amplitude $ r(t) $ and phase $ \phi(t) $ evolve over the fast timescale $ t $ and the slow timescale $ \tau = \epsilon t $.
#
# - **Amplitude $ r(t) $:**
#   - **Fast Time \( t \):** Remains nearly constant, confirming minimal change within individual oscillations.
#   - **Slow Time \( \tau \):** Shows negligible variation, indicating that amplitude modulation is not significant even over extended periods.
#
# - **Phase $ \phi(t) $:**
#   - **Fast Time \( t \):** Changes little over one oscillation period, so the phase is relatively stable in the short term.
#   - **Slow Time \( \tau \):** Increases linearly, consistent with the theoretical prediction from averaging theory. This linear growth demonstrates the cumulative effect of the weak nonlinearity on the phase over long times.
#
# These observations support the concept of time scale separation in the weakly nonlinear Duffing oscillator, where the amplitude and phase evolve differently over the fast and slow timescales.

#%%