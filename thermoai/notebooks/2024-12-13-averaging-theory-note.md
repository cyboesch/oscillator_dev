
---

**Starting Point: Weakly Nonlinear Duffing Oscillator Viewed as a Perturbation of a Harmonic Oscillator**

We consider a weakly nonlinear system described by the equation:  
$$
\ddot{x} + x + \epsilon h(x, \dot{x}) = 0, \quad \epsilon \ll 1.
$$

This equation can be thought of as a small perturbation of the standard harmonic oscillator $\ddot{x} + x = 0$.

---

**Phase Space Formulation of the Harmonic Oscillator**

Introduce $y = \dot{x}$. Then the system can be written as a first-order system in phase space $(x,y)$:  
$$
\dot{x} = y, \quad \dot{y} = -x - \epsilon h(x,y).
$$

When $\epsilon = 0$, this reduces to the unperturbed harmonic oscillator:  
$$
\dot{x} = y, \quad \dot{y} = -x.
$$

---

**Exact Solution for the Unperturbed Case ($\epsilon=0$), the harmonic oscillator**

For the classical harmonic oscillator, the solution with amplitude $r$ and phase $\phi$ can be expressed as:  
$$
x(t) = r \cos(t + \phi), \quad y(t) = -r \sin(t + \phi).
$$

This represents uniform circular motion in the $(x,y)$-plane with radius $r$. The period of oscillation is $2\pi$.

---

**When there is weak nonlinearity: $\epsilon \neq 0$**

When the nonlinear perturbation is present but small ($\epsilon \neq 0$ and $\epsilon \ll 1$), we expect that the amplitude $r(t)$ and the phase $\phi(t)$ of the solution will not remain strictly constant. Instead, they will vary slowly over time. The trajectory in the $(x,y)$-plane will remain "nearly circular," and the oscillation will have a period close to $2\pi$. Over many cycles, there will be a slow drift in both $r$ and $\phi$.

---

**Rotating Frame Analysis**

The main idea is to introduce a rotating frame that moves with the same angular frequency as the unperturbed system. By doing so, we "freeze out" the trivial fast oscillation and focus on the slow modulation of amplitude and phase due to the small perturbation.

**Definition of $r(t)$ and $\phi(t)$:**  
$$
x(t) = r(t) \cos(t + \phi(t)), \quad y(t) = -r(t) \sin(t + \phi(t)).
$$

From these definitions:  
$$
r(t) = \sqrt{x^2(t) + y^2(t)}, \quad \tan(t + \phi(t)) = -\frac{y(t)}{x(t)}.
$$

These define $r(t)$ and $\phi(t)$ uniquely. In the unperturbed case, $r$ and $\phi$ would be constants. With a small perturbation, they evolve slowly.

---

**Deriving the Evolution Equations for $\dot{r}$ and $\dot{\phi}$**

Starting from:
$$
r^2 = x^2 + y^2.
$$

Differentiate with respect to time:
$$
2r\dot{r} = 2x\dot{x} + 2y\dot{y} \implies r\dot{r} = x\dot{x} + y\dot{y}.
$$

Substitute $\dot{x}=y$ and $\dot{y}=-x-\epsilon h(x,y)$:
$$
r\dot{r} = x(y) + y(-x - \epsilon h(x,y)) = xy - xy - \epsilon y h(x,y) = -\epsilon y h(x,y).
$$

But $y = -r \sin(t+\phi)$, hence:
$$
r\dot{r} = -\epsilon (-r \sin(t+\phi)) h(x,y) = \epsilon r \sin(t+\phi) h(x,y).
$$

Divide by $r$:
$$
\dot{r} = \epsilon h(r\cos(t+\phi), -r\sin(t+\phi)) \sin(t+\phi).
$$

Thus:
$$
\boxed{\dot{r} = \epsilon h(x,y) \sin(t+\phi)}.
$$

Similarly, to find $\dot{\phi}$, we use:
$$
\frac{d}{dt}(t + \phi(t)) = 1 + \dot{\phi}(t).
$$

Through the chain rule and some trigonometric manipulations (not shown in full detail here), we obtain:
$$
\boxed{\dot{\phi} = \frac{\epsilon h(x,y)}{r} \cos(t+\phi)}.
$$

Both $\dot{r}$ and $\dot{\phi}$ are $O(\epsilon)$, confirming our intuition that amplitude and phase vary slowly compared to the fast oscillation time scale.

---

**Non-Autonomy After Transformation**

Notice that:
$$
h = h(r\cos(t+\phi), -r\sin(t+\phi)),
$$
which depends explicitly on time. The transformation to the rotating frame has introduced explicit time-dependence, making the system non-autonomous in terms of $(r,\phi)$.

However, this is expected. We started with an autonomous system in $(x,y)$, but changing to a rotating frame essentially imposes an external time dependence. Our task is to eliminate this explicit time dependence by averaging over one period.

---

**Strategy to Handle the Slow-Drift Dynamics: Averaging**

We separate the problem into two time scales:

1. **Fast time scale**: the basic harmonic oscillation with period $\approx 2\pi$.
2. **Slow time scale**: the slow variation of $r(t)$ and $\phi(t)$ over intervals of order $1/\epsilon$.

We will perform a time-averaging procedure over one period of the fast oscillation. This averaging will remove the explicit fast time dependence and yield simpler, autonomous equations for $\overline{r}$ and $\overline{\phi}$.

**Definition of the Running Average:**
$$
\overline{g}(t) = \langle g \rangle_t = \frac{1}{2\pi}\int_{t-\pi}^{t+\pi} g(s)\,ds.
$$

A key property is that:
$$
\overline{\dot{g}} = \dot{\overline{g}}.
$$

This follows from the fundamental theorem of calculus and the fact that a full $2\pi$-integral over a derivative can be converted into boundary terms.

---

**Averaging $\dot{r}$ and $\dot{\phi}$**

We have:
$$
\dot{r} = \epsilon h(r\cos(t+\phi), -r\sin(t+\phi)) \sin(t+\phi),
$$
$$
\dot{\phi} = \frac{\epsilon h(r\cos(t+\phi), -r\sin(t+\phi))}{r} \cos(t+\phi).
$$

Take the running average over one period:
$$
\overline{\dot{r}} = \dot{\overline{r}} = \left\langle \epsilon h(r\cos(t+\phi), -r\sin(t+\phi)) \sin(t+\phi) \right\rangle_t.
$$

$$
\overline{\dot{\phi}} = \dot{\overline{\phi}} = \left\langle \frac{\epsilon h(r\cos(t+\phi), -r\sin(t+\phi))}{r} \cos(t+\phi) \right\rangle_t.
$$

**Approximation:**
Since $r$ and $\phi$ evolve slowly, we can treat them as approximately constant when performing these averages. Thus:
$$
r = \overline{r} + O(\epsilon), \quad \phi = \overline{\phi} + O(\epsilon).
$$

Substituting back leads to autonomous equations in $\overline{r}$ and $\overline{\phi}$ up to $O(\epsilon)$ accuracy.

---

**Application to the Duffing Oscillator**

Consider the Duffing oscillator:
$$
\ddot{x} + x + \epsilon x^3 = 0.
$$

Here:
$$
h(x,y) = x^3.
$$

Substitute $x = r\cos(t+\phi)$:
$$
h(x,y) = (r\cos(t+\phi))^3 = r^3 \cos^3(t+\phi).
$$

1. For $\dot{r}$:
$$
\dot{r} = \epsilon r^3 \cos^3(t+\phi)\sin(t+\phi).
$$
Averaging $\cos^3(t+\phi)\sin(t+\phi)$ over one period gives zero. Thus:
$$
\dot{\overline{r}} = 0 \implies \overline{r} \text{ is nearly constant to order } \epsilon.
$$

2. For $\dot{\phi}$:
$$
\dot{\phi} = \frac{\epsilon r^3 \cos^3(t+\phi)}{r} \cos(t+\phi) = \epsilon r^2 \cos^4(t+\phi).
$$

The average $\langle \cos^4(\theta) \rangle = 3/8$. Thus:
$$
\dot{\overline{\phi}} = \epsilon r^2 \frac{3}{8}.
$$

This shows the frequency is shifted:
$$
\omega_{\text{eff}} = 1 + \frac{3}{8}\epsilon r^2.
$$

The period is thus slightly shorter than $2\pi$ if $\epsilon > 0$.

---

**Interpretation and Conclusion**

- To first order in $\epsilon$, the amplitude $r$ does not change significantly.
- The frequency is slightly altered by the nonlinear term, increasing if $\epsilon > 0$.

By employing the averaging method, we obtain autonomous equations for the slowly varying amplitude and phase, allowing the use of standard phase-plane analysis techniques and confirming the intuition that the perturbation effects accumulate over many cycles, leading to a slow drift in the system’s parameters.
