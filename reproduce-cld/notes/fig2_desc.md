“E.1 SCORE AND JACOBIAN EXPERIMENTS” ([Dockhorn et al., 2022, p. 34](zotero://select/library/items/EW8U6A8H)) ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=34&annotation=E2DSJHNQ))

Implemented in: reproduce-cld/fig2_mog.py


In this section, we provide details for the experiments presented in Sec. 3.1. For both experiments, we consider a two-dimensional simple mixture of Normals of the form

$$
p_{\text {data }}(\mathbf{x})=\sum_{k=1}^9 \frac{1}{9} p^{(k)}(\mathbf{x})
$$

where $p^{(k)}(\mathbf{x})=\mathcal{N}\left(\mathbf{x} ; \boldsymbol{\mu}_k ; 0.04^2 \boldsymbol{I}_2\right)$ and

$$
\begin{aligned}
& \boldsymbol{\mu}_1=\binom{-a}{0}, \quad \boldsymbol{\mu}_2=\binom{-a / 2}{a / 2}, \quad \boldsymbol{\mu}_3=\binom{0}{a}, \\
& \boldsymbol{\mu}_4=\binom{-a / 2}{-a / 2}, \quad \boldsymbol{\mu}_5=\binom{0}{0}, \quad \boldsymbol{\mu}_6=\binom{a / 2}{a / 2}, \\
& \boldsymbol{\mu}_7=\binom{0}{-a}, \quad \boldsymbol{\mu}_8=\binom{a / 2}{-a / 2}, \quad \boldsymbol{\mu}_9=\binom{a}{0},
\end{aligned}
$$

and $a=2^{-\frac{1}{2}}$. The choice of this data distribution is not arbitrary. In fact, mixture of Normal distributions are diffused by simply diffusing the components, i.e., setting $p_0\left(\mathbf{x}_0\right)=p_{\text {data }}(\mathbf{x})$, we have

$$
p_t\left(\mathbf{x}_t\right)=\sum_{k=1}^9 \frac{1}{9} p_t^{(k)}\left(\mathbf{x}_t\right)
$$

where $p_t^{(k)}$ are the diffused components (analogously for CLD with velocity augmentation). This means that for both CLD as well as VPSDE Song et al. (2021c) we can diffuse $p_{\text {data }}(\mathbf{x})$ with analytical access to the diffused marginal $p_t\left(\mathbf{x}_t\right)$ or $p_t\left(\mathbf{u}_t\right)$. This allows us to perform interesting analyses that would be impossible when working, for example, with image data. We visualize the data distribution in Fig. 10.

Score experiment: We empirically verify the reduced complexity of the score of $p_t\left(\mathbf{v}_t \mid \mathbf{x}_t\right)$, which is learned in CLD, compared to the score of $p_t\left(\mathbf{x}_t\right)$, which is learned in VPSDE. To avoid scaling issues between VPSDE and CLD, we chose $M=\gamma=1$ for CLD in this experiment; this results in an equilibrium distribution of $\mathcal{N}\left(\mathbf{0}_2, \boldsymbol{I}_2\right)$ (for both data and velocity components, which are independent at equilibrium), which is the same as the equilibrium distribution of the VPSDE. We then measure the difference of the respective scores at time $t$ and the equilibrium (or prior) scores, i.e. (recall that the score of a Normal distribution $p(\mathbf{x})=\mathcal{N}\left(\mathbf{0}_2, \boldsymbol{I}_2\right)$ is simply $\left.\nabla_{\mathbf{x}} \log p(\mathbf{x})=-\mathbf{x}\right)$,

$$
\begin{aligned}
\xi^{\mathrm{VPSDE}}(t) & :=\mathbb{E}_{\mathbf{x}_t \sim p\left(\mathbf{x}_t\right)}\left\|\nabla_{\mathbf{x}_t} \log p_t\left(\mathbf{x}_t\right)+\mathbf{x}_t\right\|_2^2 \\
\xi^{\mathrm{CLD}}(t) & :=\mathbb{E}_{\mathbf{u}_t \sim p\left(\mathbf{u}_t\right)}\left\|\nabla_{\mathbf{v}_t} \log p_t\left(\mathbf{v}_t \mid \mathbf{x}_t\right)+\mathbf{v}_t\right\|_2^2
\end{aligned}
$$


The expectations are approximated using $10^5$ samples from $p\left(\mathbf{x}_t\right)$ and $p\left(\mathbf{u}_t\right)$ for VPSDE and CLD, respectively. As can be seen in Fig. 9a, $\xi^{\mathrm{CLD}}(t)$ is smaller than $\xi^{\mathrm{VPSDE}}(t)$ for all $t \in[0, T]$. The difference is particularly striking for small time values $t$. Other previous SDEs, such as the VESDE, sub-VPSDE, etc., are expected to behave similarly. This result implies that the ground truth scores that need to be learnt in CLD are closer to Normal scores than the ground truth scores in previous SDEs like the VPSDE. Since the score of a Normal is very simple-and indeed directly leveraged in our mixed score formulation-we would intuitively expect that the CLD training task is easier.

Complexity experiment: Therefore, to understand the above observations in terms of learning neural networks, we train a small ResNet architecture (less than 100k parameters) for each of the following four setups: both CLD and VPSDE each with and without a mixed score parameterization. The mixed score of the VPSDE simply assumes a standard Normal data distribution (which is also the equilibrium distribution of VPSDE) resulting in adding $-\mathbf{x}_t$ to the score function. Formally, $-\mathbf{x}_t$ is the score of a Normal distribution with unit variance.
We train the models for 1 M iterations using fresh data synthesized from $p_{\text {data }}$ at a batch size of 512 . The model and data distributions are visualized in Fig. 10. We see that all models have learnt good representations of the data. We measure the complexity of the trained neural networks using the squared Frobenius norm of the networks' Jacobians. For CLD, we have

$$
\mathcal{J}_{\mathrm{F}}^{\mathrm{CLD}}(t):=\mathbb{E}_{\mathbf{u}_t \sim p\left(\mathbf{u}_t\right)}\left\|\nabla_{\mathbf{u}_t} \alpha_{\boldsymbol{\theta}}^{\prime}\left(\mathbf{u}_t\right)\right\|_F^2
$$


Similarly, for the VPSDE we compute

$$
\mathcal{J}_{\mathrm{F}}^{\mathrm{VPSDE}}(t):=\mathbb{E}_{\mathbf{x}_t \sim p\left(\mathbf{x}_t\right)}\left\|\nabla_{\mathbf{x}_t} \alpha_{\boldsymbol{\theta}}^{\prime}\left(\mathbf{x}_t\right)\right\|_F^2
$$


For both CLD and VPSDE, expectations are again approximated using $10^5$ samples. As can be seen in Fig. 9b the neural network complexity is significantly lower for CLD compared to VPSDE. A mixed score formulation further helps decreasing the neural network complexity for both CLD and VPSDE. This result implies that the arguably simpler training task in CLD indeed also translates to reduced model complexity in that the neural network is smoother as measured by $\mathcal{J}_{\mathrm{F}}(t)$. In largescale experiments, this would mean that, given similar model capacity, a CLD-based SGM could potentially have a higher expressivity. Or, on the other hand, similar performance could be achieved with a smoother and potentially smaller model. Indeed these findings are in line with our strong results on the CIFAR-10 benchmark.