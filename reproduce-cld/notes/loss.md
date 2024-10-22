B. 3 CLD ObJECTIVE

To derive the objective for training CLD-based SGMs, we start with a derivation that targets maximum likelihood training in a similar fashion to Song et al. (2021b). Let $p_0$ and $q_0$ be two densities, then

$$
\begin{aligned}
D_{\mathrm{KL}}\left(p_0 \| q_0\right) & =D_{\mathrm{KL}}\left(p_0 \| q_0\right)-D_{\mathrm{KL}}\left(p_T \| q_T\right)+D_{\mathrm{KL}}\left(p_T \| q_T\right) \\
& =-\int_0^T \frac{\partial D_{\mathrm{KL}}\left(p_t \| q_t\right)}{\partial t} d t+D_{\mathrm{KL}}\left(p_T \| q_T\right)
\end{aligned}
$$

where $p_t$ and $q_t$ are the marginal densities of $p_0$ and $q_0$, respectively, diffused by our criticallydamped Langevin diffusion. As has been shown in Song et al. (2021b), Eq. (39) can be written as a mixture (over $t$ ) of score matching losses. To this end, let us consider the Fokker-Planck equation associated with the critically-damped Langevin diffusion:

$$
\begin{aligned}
\frac{\partial p_t\left(\mathbf{u}_t\right)}{\partial t} & =\nabla_{\mathbf{u}_t} \cdot\left[\frac{1}{2}\left(G(t) G(t)^{\top} \otimes \boldsymbol{I}_d\right) \nabla_{\mathbf{u}_t} p_t\left(\mathbf{u}_t\right)-p_t\left(\mathbf{u}_t\right)\left(f(t) \otimes \boldsymbol{I}_d\right) \mathbf{u}_t\right] \\
& =\nabla_{\mathbf{u}_t} \cdot\left[\boldsymbol{h}_p\left(\mathbf{u}_t, t\right) p_t\left(\mathbf{u}_t\right)\right], \quad \boldsymbol{h}_p\left(\mathbf{u}_t, t\right):=\frac{1}{2}\left(G(t) G(t)^{\top} \otimes \boldsymbol{I}_d\right) \nabla_{\mathbf{u}_t} \log p_t\left(\mathbf{u}_t\right)-\left(f(t) \otimes \boldsymbol{I}_d\right) \mathbf{v}
\end{aligned}
$$


Similarly, we have $\frac{\partial q_t\left(\mathbf{u}_t\right)}{\partial t}=\nabla_{\mathbf{u}_t} \cdot\left[\boldsymbol{h}_q\left(\mathbf{u}_t, t\right) q_t\left(\mathbf{u}_t\right)\right]$. Assuming $\log p_t\left(\mathbf{u}_t\right)$ and $\log q_t\left(\mathbf{u}_t\right)$ are smooth functions with at most polynomial growth at infinity, we have

$$
\lim _{\mathbf{u}_t \rightarrow \infty} \boldsymbol{h}_p\left(\mathbf{u}_t, t\right) p_t\left(\mathbf{u}_t\right)=\lim _{\mathbf{u}_t \rightarrow \infty} \boldsymbol{h}_q\left(\mathbf{u}_t, t\right) q_t\left(\mathbf{u}_t\right)=0
$$


Using the above fact, we can compute the time-derivative of the Kullback-Leibler divergence between $p_t$ and $q_t$ as

$$
\begin{aligned}
\frac{\partial D_{\mathrm{KL}}\left(p_t \| q_t\right)}{\partial t}= & \frac{\partial}{\partial t} \int p_t\left(\mathbf{u}_t\right) \log \frac{p_t\left(\mathbf{u}_t\right)}{q_t\left(\mathbf{u}_t\right)} d \mathbf{u}_t \\
= & -\int p_t\left(\mathbf{u}_t\right)\left[\boldsymbol{h}_p\left(\mathbf{u}_t, t\right)-\boldsymbol{h}_q\left(\mathbf{u}_t, t\right)\right]^{\top}\left[\nabla_{\mathbf{u}_t} \log p_t\left(\mathbf{u}_t\right)-\nabla_{\mathbf{u}_t} \log q_t\left(\mathbf{u}_t\right)\right] d \mathbf{u}_t \\
= & -\frac{1}{2} \int p_t\left(\mathbf{u}_t\right)\left[\nabla_{\mathbf{u}_t} \log p_t\left(\mathbf{u}_t\right)-\nabla_{\mathbf{u}_t} \log q_t\left(\mathbf{u}_t\right)\right]^{\top}\left(G(t) G(t)^{\top} \otimes \boldsymbol{I}_d\right)\left[\nabla_{\mathbf{u}_t} \log p_t\left(\mathbf{u}_t\right)\right. \\
& \left.-\nabla_{\mathbf{u}_t} \log q_t\left(\mathbf{u}_t\right)\right] d \mathbf{u}_t \\
= & -\beta(t) \Gamma \int p_t\left(\mathbf{u}_t\right)\left\|\nabla_{\mathbf{v}_t} \log p_t\left(\mathbf{u}_t\right)-\nabla_{\mathbf{v}_t} \log q_t\left(\mathbf{u}_t\right)\right\|_2^2 d \mathbf{u}_t
\end{aligned}
$$


Notice that due to the form of $G(t)$, we now have only gradients with respect to the velocity component $\mathbf{v}_t$. Combining the above with Eq. (39), we have

$$
\begin{aligned}
D_{\mathrm{KL}}\left(p_0 \| q_0\right) & =\mathbb{E}_{t \sim \mathcal{U}[0, T], \mathbf{u}_t \sim p_t(\mathbf{u})}\left[\Gamma \beta(t)\left\|\nabla_{\mathbf{v}_t} \log p_t\left(\mathbf{u}_t\right)-\nabla_{\mathbf{v}_t} \log q_t\left(\mathbf{u}_t\right)\right\|_2^2\right]+D_{\mathrm{KL}}\left(p_T \| q_T\right) \\
& \approx \mathbb{E}_{t \sim \mathcal{U}[0, T], \mathbf{u}_t \sim p_t(\mathbf{u})}\left[\Gamma \beta(t)\left\|\nabla_{\mathbf{v}_t} \log p_t\left(\mathbf{u}_t\right)-\nabla_{\mathbf{v}_t} \log q_t\left(\mathbf{u}_t\right)\right\|_2^2\right]
\end{aligned}
$$


Note that the approximation holds if $p_T$ is sufficiently "close" to $q_T$. We obtain a more general objective function by replacing $\Gamma \beta(t)$ with an arbitrary function $\lambda(t)$, i.e.,

$$
\mathbb{E}_{t \sim \mathcal{U}[0, T], \mathbf{u}_t \sim p_t(\mathbf{u})}\left[\lambda(t)\left\|\nabla_{\mathbf{v}_t} \log p_t\left(\mathbf{u}_t\right)-\nabla_{\mathbf{v}_t} \log q_t\left(\mathbf{u}_t\right)\right\|_2^2\right]
$$

As shown in App. C, the above can be rewritten, up to irrelevant constant terms, as either of the following two objectives:

$$
\begin{aligned}
& \operatorname{HSM}(\lambda(t)):=\mathbb{E}_{t \sim \mathcal{U}[0, T], \mathbf{x}_0 \sim p_0\left(\mathbf{x}_0\right), \mathbf{u}_t \sim p_t\left(\mathbf{u}_t \mid \mathbf{x}_0\right)}\left[\lambda(t)\left\|\nabla_{\mathbf{v}_t} \log p_t\left(\mathbf{u}_t \mid \mathbf{x}_0\right)-\nabla_{\mathbf{v}_t} \log q_t\left(\mathbf{u}_t\right)\right\|_2^2\right] \\
& \operatorname{DSM}(\lambda(t)):=\mathbb{E}_{t \sim \mathcal{U}[0, T], \mathbf{u}_0 \sim p_0\left(\mathbf{u}_0\right), \mathbf{u}_t \sim p_t\left(\mathbf{u}_t \mid \mathbf{u}_0\right)}\left[\lambda(t)\left\|\nabla_{\mathbf{v}_t} \log p_t\left(\mathbf{u}_t \mid \mathbf{u}_0\right)-\nabla_{\mathbf{v}_t} \log q_t\left(\mathbf{u}_t\right)\right\|_2^2\right]
\end{aligned}
$$


For both HSM and DSM, we have shown in App. B. 1 that the perturbation kernels $p_t\left(\mathbf{u}_t \mid \mathbf{x}_0\right)$ and $p_t\left(\mathbf{u}_t \mid \mathbf{u}_0\right)$ are Normal distributions with the following structure of the covariance matrix:

$$
\boldsymbol{\Sigma}_t=\Sigma_t \otimes \boldsymbol{I}_d, \quad \Sigma_t=\left(\begin{array}{ll}
\Sigma_t^{x x} & \Sigma_t^{x v} \\
\Sigma_t^{x v} & \Sigma_t^{v v}
\end{array}\right)
$$


We can use this fact to compute the gradient $\nabla_{\mathbf{u}_t} \log p_t\left(\mathbf{u}_t \mid \cdot\right)$

$$
\begin{aligned}
\nabla_{\mathbf{u}_t} \log p_t\left(\mathbf{u}_t \mid \cdot\right) & =-\nabla_{\mathbf{u}_t} \frac{1}{2}\left(\mathbf{u}_t-\boldsymbol{\mu}_t\right) \boldsymbol{\Sigma}_t^{-1}\left(\mathbf{u}_t-\boldsymbol{\mu}_t\right) \\
& =-\boldsymbol{\Sigma}_t^{-1}\left(\mathbf{u}_t-\boldsymbol{\mu}_t\right) \\
& =-\boldsymbol{L}_t^{-\top} \boldsymbol{L}_t^{-1}\left(\mathbf{u}_t-\boldsymbol{\mu}_t\right) \\
& =-\boldsymbol{L}_t^{-\top} \boldsymbol{\epsilon}_{2 d}
\end{aligned}
$$

where $\boldsymbol{\epsilon}_{2 d} \sim \mathcal{N}\left(\mathbf{0}, \boldsymbol{I}_{2 d}\right)$ and $\boldsymbol{\Sigma}_t=\boldsymbol{L}_t \boldsymbol{L}_t^{\top}$ is the Cholesky factorization of the covariance matrix $\boldsymbol{\Sigma}_t$. Note that the structure of $\boldsymbol{\Sigma}_t$ implies that $\boldsymbol{L}_t=L_t \otimes \boldsymbol{I}_d$, where $L_t L_t^{\top}$ is the Cholesky factorization of $\Sigma_t$, i.e,

$$
L_t=\left(\begin{array}{ll}
L_t^{x x} & L_t^{x v} \\
L_t^{x v} & L_t^{v v}
\end{array}\right)=\left(\begin{array}{lc}
\sqrt{\Sigma_t^{x x}} & 0 \\
\frac{\Sigma_t^{x v}}{\sqrt{\Sigma_t^{x x}}} & \sqrt{\frac{\Sigma_t^{x x} \Sigma_t^{v y}-\left(\Sigma_t^{x v}\right)^2}{\Sigma_t^{x x}}}
\end{array}\right)
$$


Furthermore, we have

$$
\begin{aligned}
\boldsymbol{L}_t^{-\top} & =L_t^{-\top} \otimes \boldsymbol{I}_d \\
& =\left(\begin{array}{cc}
\sqrt{\Sigma_t^{x x}} & \frac{\Sigma_t^{x v}}{\sqrt{\Sigma_t^{x x}}} \\
0 & \sqrt{\frac{\Sigma_t^{x x} \sum_t^{x v}-\left(\Sigma_t^{x v}\right)^2}{\Sigma_t^{x x}}}
\end{array}\right)^{-1} \otimes \boldsymbol{I}_d \\
& =\left(\begin{array}{cc}
\frac{1}{\sqrt{\Sigma_t^{x x}}} & \frac{-\Sigma_t^{x z}}{\sqrt{\Sigma_t^{x x}} \sqrt{\Sigma_t^{x x} \sum_t^{x z}-\left(\Sigma_t^{x v}\right)^2}} \\
0 & \sqrt{\frac{\Sigma_t^{x x}}{\Sigma_t^{x x} \sum_t^{x y}-\left(\Sigma_t^{x v}\right)^2}}
\end{array}\right) \otimes \boldsymbol{I}_d
\end{aligned}
$$


Using the above, we can compute

$$
\begin{aligned}
\nabla_{\mathbf{v}_t} \log p_t\left(\mathbf{u}_t \mid \cdot\right) & =\left[\nabla_{\mathbf{u}_t} \log p_t\left(\mathbf{u}_t \mid \cdot\right)\right]_{d: 2 d} \\
& =\left[-\boldsymbol{L}_t^{-\top} \boldsymbol{\epsilon}_{2 d}\right]_{d: 2 d} \\
& =-\ell_t \boldsymbol{\epsilon}_{d: 2 d}
\end{aligned}
$$

where

$$
\ell_t:=\sqrt{\frac{\Sigma_t^{x x}}{\Sigma_t^{x x} \Sigma_t^{v v}-\left(\Sigma_t^{x v}\right)^2}}
$$

and $\boldsymbol{\epsilon}_{d: 2 d}$ denotes those (latter) $d$ components of $\boldsymbol{\epsilon}_{2 d}$ that actually affect $\nabla_{\mathbf{v}_t} \log p_t\left(\mathbf{u}_t \mid \cdot\right)$.
Note that $\ell_t$ depends on the conditioning in the perturbation kernel, and therefore $\ell_t$ is different for DSM, which is based on $p\left(\mathbf{u}_t \mid \mathbf{u}_0\right)$, and HSM, which is based on $p\left(\mathbf{u}_t \mid \mathbf{x}_0\right)$. Therefore, we will henceforth refer to $\ell_t^{\mathrm{HSM}}$ and $\ell_t^{\mathrm{DSM}}$ if distinction of the two cases is necessary (otherwise we will simply refer to $\ell_t$ for both).
As discussed in Section 3.2, we model $\nabla_{\mathbf{v}_t} \log q_t\left(\mathbf{u}_t\right)$ as $s_{\boldsymbol{\theta}}\left(\mathbf{u}_t, t\right)=-\ell_t \alpha_{\boldsymbol{\theta}}\left(\mathbf{u}_t, t\right)$. Plugging everything back into our obiective functions, Eq. (45) and Eq. (46), we obtain

$$
\begin{aligned}
& \operatorname{HSM}(\lambda(t))=\mathbb{E}_{t \sim \mathcal{U}[0, T], \mathbf{x}_0 \sim p_0\left(\mathbf{x}_0\right), \mathbf{u}_t \sim p_t\left(\mathbf{u}_t \mid \mathbf{x}_0\right)}\left[\lambda(t)\left(\ell_t^{\mathrm{HSM}}\right)^2\left\|\boldsymbol{\epsilon}_{d: 2 d}-\alpha_{\boldsymbol{\theta}}\left(\mathbf{u}_t, t\right)\right\|_2^2\right] \\
& \operatorname{DSM}(\lambda(t))=\mathbb{E}_{t \sim \mathcal{U}[0, T], \mathbf{u}_0 \sim p_0\left(\mathbf{u}_0\right), \mathbf{u}_t \sim p_t\left(\mathbf{u}_t \mid \mathbf{u}_0\right)}\left[\lambda(t)\left(\ell_t^{\mathrm{DSM}}\right)^2\left\|\boldsymbol{\epsilon}_{d: 2 d}-\alpha_{\boldsymbol{\theta}}\left(\mathbf{u}_t, t\right)\right\|_2^2\right]
\end{aligned}
$$

Figure 6: Comparison of $\ell_t^{\mathrm{HSM}}$ (in green) and $\ell_t^{\mathrm{DSM}}$ (in orange) for our main hyperparameter setting with $M=0.25$ and $\gamma=0.04$. In contrast to $\ell_t^{\mathrm{DSM}}, \ell_t^{\mathrm{HSM}}$ is analytically bounded. Nevertheless, numerical computation can be unstable (even when using double precision) in which case adding a numerical stabilization of $\epsilon_{\text {num }}=10^{-9}$ to the covariance matrix before computing $\ell_t$ suffices to make HSM work (see App. B.4).
where $\mathbf{u}_t$ is sampled via reparameterization:

$$
\mathbf{u}_t=\boldsymbol{\mu}_t+\boldsymbol{L}_t \boldsymbol{\epsilon}=\boldsymbol{\mu}_t+\binom{L_t^{x x} \boldsymbol{\epsilon}_{0: d}}{L_t^{x v} \boldsymbol{\epsilon}_{0: d}+L_t^{v v} \boldsymbol{\epsilon}_{d: 2 d}}
$$


Note again that $L_t$ is different for HSM and DSM.
Analogously to prior work (Ho et al., 2020; Vahdat et al., 2021; Song et al., 2021b) an objective better suited for high quality image synthesis can be obtained by "dropping the variance prefactor":

$$
\begin{aligned}
& \operatorname{HSM}\left(\lambda(t)=\left(\ell_t^{\mathrm{HSM}}\right)^{-2}\right)=\mathbb{E}_{t \sim \mathcal{U}[0, T], \mathbf{x}_0 \sim p_0\left(\mathbf{x}_0\right), \mathbf{u}_t \sim p_t\left(\mathbf{u}_t \mid \mathbf{x}_0\right)}\left[\left\|\boldsymbol{\epsilon}_{d: 2 d}-\alpha_{\boldsymbol{\theta}}\left(\mathbf{u}_t, t\right)\right\|_2^2\right] \\
& \operatorname{DSM}\left(\lambda(t)=\left(\ell_t^{\mathrm{DSM}}\right)^{-2}\right)=\mathbb{E}_{t \sim \mathcal{U}[0, T], \mathbf{u}_0 \sim p_0\left(\mathbf{u}_0\right), \mathbf{u}_t \sim p_t\left(\mathbf{u}_t \mid \mathbf{u}_0\right)}\left[\left\|\boldsymbol{\epsilon}_{d: 2 d}-\alpha_{\boldsymbol{\theta}}\left(\mathbf{u}_t, t\right)\right\|_2^2\right]
\end{aligned}
$$
