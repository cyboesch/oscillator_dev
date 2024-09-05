from typing import Union
import jax.numpy as jnp
import equinox as eqx
import diffrax
from linalg import schur_inverse_2x2


class CriticallyDampedLangevinDynamics(eqx.Module):
    state_dim: int
    M: float
    gamma: float
    beta: float
    Gamma: Union[float, None] = None  # Default to sqrt(4 * M)

    def __init__(
        self,
        state_dim: int,
        M: float,
        beta: float,
        gamma: float,
        Gamma: Union[float, None] = None,
    ):
        if state_dim <= 0 or M <= 0 or beta <= 0:
            raise ValueError("All parameters must be positive.")
        self.state_dim = state_dim
        self.M = M
        self.Gamma = Gamma if Gamma is not None else jnp.sqrt(4 * M)
        self.gamma = gamma
        self.beta = beta

    def drift(self, t, u, args):
        x, v = u[: self.state_dim], u[self.state_dim :]
        f_x = self.beta * v / self.M
        f_v = -self.beta * x - self.beta * self.Gamma * v / self.M
        return jnp.concatenate([f_x, f_v])

    def diffusion(self, t, u, args):
        G = jnp.zeros((2 * self.state_dim, 2 * self.state_dim))
        G = G.at[self.state_dim :, self.state_dim :].set(
            jnp.sqrt(2 * self.Gamma * self.beta) * jnp.eye(self.state_dim)
        )
        return G

    def get_terms(self, bm):
        drift_term = diffrax.ODETerm(self.drift)
        diffusion_term = diffrax.ControlTerm(self.diffusion, bm)
        return diffrax.MultiTerm(drift_term, diffusion_term)

    def B(self, t):
        return self.beta * t

    def mean(self, t, x_0, v_0):
        if x_0.shape != (self.state_dim,) or v_0.shape != (self.state_dim,):
            raise ValueError("x_0 and v_0 must have shape (state_dim,)")
        B_t = self.B(t)
        exp_term = jnp.exp(-2 * B_t / self.Gamma)

        x_t = (
            2 * B_t / self.Gamma * x_0 + 4 * B_t / self.Gamma**2 * v_0 + x_0
        ) * exp_term
        v_t = (-B_t * x_0 - 2 * B_t / self.Gamma * v_0 + v_0) * exp_term

        return jnp.concatenate([x_t, v_t])

    def Sigma_xx_t(self, t, Sigma_0_xx, Sigma_0_vv):
        """$\Sigma_t^{x x}=\Sigma_0^{x x}+e^{4 \mathcal{B}(t) \Gamma^{-1}}-1+4 \mathcal{B}(t) \Gamma^{-1}\left(\Sigma_0^{x x}-1\right)+4 \mathcal{B}^2(t) \Gamma^{-2}\left(\Sigma_0^{x x}-2\right)+16 \mathcal{B}(t)^2 \Gamma^{-4} \Sigma_0^{v v}$"""
        B_t = self.B(t)
        Sigma_xx = (
            Sigma_0_xx
            + jnp.exp(4 * B_t / self.Gamma)
            - 1
            + 4 * B_t / self.Gamma * (Sigma_0_xx - 1)
            + 4 * B_t**2 / self.Gamma**2 * (Sigma_0_xx - 2)
            + 16 * B_t**2 / self.Gamma**4 * Sigma_0_vv
        )
        return Sigma_xx

    def Sigma_vv_t(self, t, Sigma_0_xx, Sigma_0_vv):
        """$\Sigma_t^{v v}=\frac{\Gamma^2}{4}\left(e^{4 \mathcal{B}(t) \Gamma^{-1}}-1\right)+\mathcal{B}(t) \Gamma+\Sigma_0^{v v}\left(1+4 \mathcal{B}(t)^2 \Gamma^{-2}-4 \mathcal{B}(t) \Gamma^{-1}\right)+\mathcal{B}(t)^2\left(\Sigma_0^{x x}-2\right)$"""
        Sigma_vv = (
            self.Gamma**2 / 4 * (jnp.exp(4 * self.B(t) / self.Gamma) - 1)
            + self.B(t) * self.Gamma
            + Sigma_0_vv
            * (1 + 4 * self.B(t) ** 2 / self.Gamma**2 - 4 * self.B(t) / self.Gamma)
            + self.B(t) ** 2 * (Sigma_0_xx - 2)
        )
        return Sigma_vv

    def Sigma_xv_t(self, t, Sigma_0_xx, Sigma_0_vv):
        """$\Sigma_t^{x v}=-\mathcal{B}(t) \Sigma_0^{x x}+4 \mathcal{B}(t) \Gamma^{-2} \Sigma_0^{v v}-2 \mathcal{B}^2(t) \Gamma^{-1}\left(\Sigma_0^{x x}-2\right)-8 \mathcal{B}^2(t) \Gamma^{-3} \Sigma_0^{v v}$"""
        B_t = self.B(t)
        Sigma_xv = (
            -B_t * Sigma_0_xx
            + 4 * B_t / self.Gamma**2 * Sigma_0_vv
            - 2 * B_t**2 / self.Gamma * (Sigma_0_xx - 2)
            - 8 * B_t**2 / self.Gamma**3 * Sigma_0_vv
        )
        return Sigma_xv

    def Sigma_inv_xx_t(self, t, Sigma_0_xx, Sigma_0_vv):
        return schur_inverse_2x2(
            self.Sigma_xx_t(t, Sigma_0_xx, Sigma_0_vv),
            self.Sigma_vv_t(t, Sigma_0_xx, Sigma_0_vv),
            self.Sigma_xv_t(t, Sigma_0_xx, Sigma_0_vv),
            which_block="xx",
        )

    def Sigma_inv_vv_t(self, t, Sigma_0_xx, Sigma_0_vv):
        return schur_inverse_2x2(
            self.Sigma_xx_t(t, Sigma_0_xx, Sigma_0_vv),
            self.Sigma_vv_t(t, Sigma_0_xx, Sigma_0_vv),
            self.Sigma_xv_t(t, Sigma_0_xx, Sigma_0_vv),
            which_block="vv",
        )

    def Sigma_inv_xv_t(self, t, Sigma_0_xx, Sigma_0_vv):
        return schur_inverse_2x2(
            self.Sigma_xx_t(t, Sigma_0_xx, Sigma_0_vv),
            self.Sigma_vv_t(t, Sigma_0_xx, Sigma_0_vv),
            self.Sigma_xv_t(t, Sigma_0_xx, Sigma_0_vv),
            which_block="xv",
        )

    def covariance(self, t, Sigma_0_xx, Sigma_0_vv):
        Sigma_xx = self.Sigma_xx_t(t, Sigma_0_xx, Sigma_0_vv)
        Sigma_vv = self.Sigma_vv_t(t, Sigma_0_xx, Sigma_0_vv)
        Sigma_xv = self.Sigma_xv_t(t, Sigma_0_xx, Sigma_0_vv)
        Sigma_t = jnp.array([[Sigma_xx, Sigma_xv], [Sigma_xv, Sigma_vv]]) * jnp.exp(
            -4 * self.B(t) / self.Gamma
        )
        return jnp.kron(Sigma_t, jnp.eye(self.state_dim))

    def get_dsm_kernel_params(self, u_0, t):
        """“when conditioning on initial data and velocity samples x0 and v0 (as in denoising score matching (DSM)), the mean and covariance matrix of the perturbation kernel p(ut|u0) can be obtained by setting μ0 = (x0, v0)>, Σ0xx = 0, and Σ0vv = 0.” ([Dockhorn et al., 2022, p. 21](zotero://select/library/items/EW8U6A8H)) ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=21&annotation=6ZWQTEZZ))"""
        x_0, v_0 = u_0[: self.state_dim], u_0[self.state_dim :]
        mean = self.mean(t, x_0, v_0)
        cov = self.covariance(t, Sigma_0_xx=0, Sigma_0_vv=0)
        return mean, cov

    def get_hsm_kernel_params(self, x_0, t, gamma=1.0):
        """“conditioning only on initial data samples x0 and marginalizing over the full initial velocity distribution (as in our hybrid score matching (HSM), see Sec. C), the mean and covariance matrix of the perturbation kernel p(ut|x0) can be obtained by setting μ0 = (x0, 0d)>, Σ0xx = 0, and Σ0vv = γM” ([Dockhorn et al., 2022, p. 21](zotero://select/library/items/EW8U6A8H)) ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=21&annotation=CW5PUPSH))"""
        v_0_hsm = jnp.zeros_like(x_0)
        mean = self.mean(t, x_0, v_0_hsm)
        cov = self.covariance(t, Sigma_0_xx=0, Sigma_0_vv=gamma * self.M)
        return mean, cov

    def compute_l_t(self, Sigma_t):
        """Compute l_t as defined in:
        ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=23&annotation=G4NCGLIX))
        ([Dockhorn et al., 2022, p. 23](zotero://select/library/items/EW8U6A8H))"""
        Sigma_xx, Sigma_xv, Sigma_vv = Sigma_t[0, 0], Sigma_t[0, 1], Sigma_t[1, 1]
        l_t = jnp.sqrt(Sigma_xx / (Sigma_xx * Sigma_vv - Sigma_xv**2))
        return l_t.item()  # Convert to scalar

    def compute_L_t_inv_T(self, Sigma_t):
        """Compute L_t^(-T) as defined in:
        ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=23&annotation=8M85CFQL))
        ([Dockhorn et al., 2022, p. 23](zotero://select/library/items/EW8U6A8H))"""
        Sigma_xx, Sigma_xv, Sigma_vv = Sigma_t[0, 0], Sigma_t[0, 1], Sigma_t[1, 1]
        sqrt_Sigma_xx = jnp.sqrt(Sigma_xx)
        sqrt_det = jnp.sqrt(Sigma_xx * Sigma_vv - Sigma_xv**2)

        L_t_inv_T = jnp.array(
            [
                [1 / sqrt_Sigma_xx, -Sigma_xv / (sqrt_Sigma_xx * sqrt_det)],
                [0, sqrt_Sigma_xx / sqrt_det],
            ]
        )
        return L_t_inv_T

    def compute_Sigma_t_inv(self, Sigma_t):
        """Compute Sigma_t^(-1) as defined in:
        ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=23&annotation=8M85CFQL))
        ([Dockhorn et al., 2022, p. 23](zotero://select/library/items/EW8U6A8H))"""
        Sigma_xx, Sigma_xv, Sigma_vv = Sigma_t[0, 0], Sigma_t[0, 1], Sigma_t[1, 1]
        det = Sigma_xx * Sigma_vv - Sigma_xv**2
        return jnp.array([[Sigma_vv, -Sigma_xv], [-Sigma_xv, Sigma_xx]]) / det

    def compute_grad_u_t_log_p_t(self, Sigma_t, epsilon_2d):
        """Compute the gradient of log p_t(u_t | ·) with respect to u_t."""
        L_t_inv_T = self.compute_L_t_inv_T(Sigma_t)
        return -jnp.kron(L_t_inv_T, jnp.eye(self.state_dim)) @ epsilon_2d

    def compute_grad_v_t_log_p_t(self, Sigma_t, epsilon_d):
        """Compute the gradient of log p_t(u_t | ·) with respect to v_t."""
        l_t = self.compute_l_t(Sigma_t)
        return -l_t * epsilon_d

    def compute_dsm_grad_u_t_log_p_t(self, u_0, t, epsilon_2d):
        """Compute the DSM gradient of log p_t(u_t | u_0)."""
        _, Sigma_t = self.get_dsm_kernel_params(u_0, t)
        return self.compute_grad_u_t_log_p_t(Sigma_t, epsilon_2d)

    def compute_hsm_grad_u_t_log_p_t(self, x_0, t, epsilon_2d):
        """Compute the HSM gradient of log p_t(u_t | x_0)."""
        _, Sigma_t = self.get_hsm_kernel_params(x_0, t, self.gamma)
        return self.compute_grad_u_t_log_p_t(Sigma_t, epsilon_2d)

    def compute_dsm_grad_v_t_log_p_t(self, u_t, u_0, t, epsilon_d):
        """Compute the DSM gradient of log p_t(u_t | u_0) with respect to v_t."""
        _, Sigma_t = self.get_dsm_kernel_params(u_0, t)
        return self.compute_grad_v_t_log_p_t(Sigma_t, epsilon_d)

    def compute_hsm_grad_v_t_log_p_t(self, x_0, t, epsilon_d):
        """Compute the HSM gradient of log p_t(u_t | x_0) with respect to v_t."""
        _, Sigma_t = self.get_hsm_kernel_params(x_0, t, self.gamma)
        return self.compute_grad_v_t_log_p_t(Sigma_t, epsilon_d)
