import jax.numpy as jnp
import equinox as eqx
from linalg import schur_inverse_2x2


class CriticallyDampedLangevinDynamics(eqx.Module):
    state_dim: int
    M: float
    gamma: float
    beta: float
    Gamma: float

    def __init__(
        self,
        state_dim: int,
        M: float,
        beta: float,
        gamma: float,
    ):
        if state_dim < 1 or M <= 0 or beta <= 0:
            raise ValueError("All parameters must be positive.")
        self.state_dim = state_dim
        self.M = M
        self.gamma = gamma
        self.beta = beta
        self.Gamma = jnp.sqrt(4 * M)
        self.sigma_0_xx = 0
        self.sigma_0_vv = self.gamma * self.M
        
    def _u2xv(self, u_0):
      # check that u_0 is of shape (2, self.state_dim)
      assert u_0.shape == (2 * self.state_dim,), f"Expected u_0 to have shape (2 * self.state_dim,), got {u_0.shape}. Use vmap for batching."
      x0 = u_0.at[:self.state_dim].get()
      v0 = u_0.at[self.state_dim:].get()
      return x0, v0
  
    def _xv2u(self, x_0, v_0):
      return jnp.concatenate([x_0, v_0])
  
    def B(self, t):
        return self.beta * t

    def mu_t(self, u_0, t):
        # at t = 0, we have u_0 = (x_0, v_0==0)
        x_0, v_0 = self._u2xv(u_0)
        B_t = self.B(t)
        exp_term = jnp.exp(-2 * B_t / self.Gamma)

        x_t = (
            2 * B_t / self.Gamma * x_0 + 4 * B_t / self.Gamma**2 * v_0 + x_0
        ) * exp_term
        v_t = (-B_t * x_0 - 2 * B_t / self.Gamma * v_0 + v_0) * exp_term
        u_t = self._xv2u(x_t, v_t)
        assert u_t.shape == (2 * self.state_dim,)
        return u_t

    def sigma_xx_t(self, t, sigma_0_xx, sigma_0_vv):
        """$\sigma_t^{x x}=\sigma_0^{x x}+e^{4 \mathcal{B}(t) \Gamma^{-1}}-1+4 \mathcal{B}(t) \Gamma^{-1}\left(\sigma_0^{x x}-1\right)+4 \mathcal{B}^2(t) \Gamma^{-2}\left(\sigma_0^{x x}-2\right)+16 \mathcal{B}(t)^2 \Gamma^{-4} \sigma_0^{v v}$"""
        B_t = self.B(t)
        sigma_xx = (
            sigma_0_xx
            + jnp.exp(4 * B_t / self.Gamma)
            - 1
            + 4 * B_t / self.Gamma * (sigma_0_xx - 1)
            + 4 * B_t**2 / self.Gamma**2 * (sigma_0_xx - 2)
            + 16 * B_t**2 / self.Gamma**4 * sigma_0_vv
        )
        return sigma_xx

    def sigma_vv_t(self, t, sigma_0_xx, sigma_0_vv):
        """$\sigma_t^{v v}=\frac{\Gamma^2}{4}\left(e^{4 \mathcal{B}(t) \Gamma^{-1}}-1\right)+\mathcal{B}(t) \Gamma+\sigma_0^{v v}\left(1+4 \mathcal{B}(t)^2 \Gamma^{-2}-4 \mathcal{B}(t) \Gamma^{-1}\right)+\mathcal{B}(t)^2\left(\sigma_0^{x x}-2\right)$"""
        sigma_vv = (
            self.Gamma**2 / 4 * (jnp.exp(4 * self.B(t) / self.Gamma) - 1)
            + self.B(t) * self.Gamma
            + sigma_0_vv
            * (1 + 4 * self.B(t) ** 2 / self.Gamma**2 - 4 * self.B(t) / self.Gamma)
            + self.B(t) ** 2 * (sigma_0_xx - 2)
        )
        return sigma_vv

    def sigma_xv_t(self, t, sigma_0_xx, sigma_0_vv):
        """$\sigma_t^{x v}=-\mathcal{B}(t) \sigma_0^{x x}+4 \mathcal{B}(t) \Gamma^{-2} \sigma_0^{v v}-2 \mathcal{B}^2(t) \Gamma^{-1}\left(\sigma_0^{x x}-2\right)-8 \mathcal{B}^2(t) \Gamma^{-3} \sigma_0^{v v}$"""
        B_t = self.B(t)
        sigma_xv = (
            -B_t * sigma_0_xx
            + 4 * B_t / self.Gamma**2 * sigma_0_vv
            - 2 * B_t**2 / self.Gamma * (sigma_0_xx - 2)
            - 8 * B_t**2 / self.Gamma**3 * sigma_0_vv
        )
        return sigma_xv

    def Sigma_t(self, t, sigma_0_xx, sigma_0_vv):
        sigma_xx = self.sigma_xx_t(t, sigma_0_xx, sigma_0_vv)
        sigma_vv = self.sigma_vv_t(t, sigma_0_xx, sigma_0_vv)
        sigma_xv = self.sigma_xv_t(t, sigma_0_xx, sigma_0_vv)
        
        Sigma_t = jnp.array([[sigma_xx, sigma_xv], 
                             [sigma_xv, sigma_vv]]) * jnp.exp(
            -4 * self.B(t) / self.Gamma
        )
        return Sigma_t  # (2,2)

    def kernel_params(self, u_0, t):
        """ "conditioning only on initial data samples x0 and marginalizing over the full initial velocity distribution (as in our hybrid score matching (HSM), see Sec. C), the mean and covariance matrix of the perturbation kernel p(ut|x0) can be obtained by setting μ0 = (x0, 0d)>, Σ0xx = 0, and Σ0vv = γM" ([Dockhorn et al., 2022, p. 21](zotero://select/library/items/EW8U6A8H)) ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=21&annotation=CW5PUPSH))"""
        v_0_hsm = jnp.zeros_like(x_0)
        mean = self.mu_t(t, x_0, v_0_hsm)
        sigma_t = self.Sigma_t(t, sigma_0_xx=0, sigma_0_vv=self.gamma * self.M)
        return mean, jnp.kron(sigma_t, jnp.eye(self.state_dim))

    def ell_t(self, u_0, t):
        """Compute l_t for dsm as defined in:
        ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=23&annotation=G4NCGLIX))
        ([Dockhorn et al., 2022, p. 23](zotero://select/library/items/EW8U6A8H))"""
        _, sigma_t = self.kernel_params(u_0, t)
        sigma_xx, sigma_xv, sigma_vv = sigma_t[:, :self.state_dim, :self.state_dim], sigma_t[:, :self.state_dim, self.state_dim:], sigma_t[:, self.state_dim:, self.state_dim:]
        l_t = jnp.sqrt(sigma_xx / (sigma_xx * sigma_vv - sigma_xv**2))
        return l_t.item()  # Convert to scalar
      
    def L_t_blocks(self, u_0, t):
      """$L_t=\left(\begin{array}{ll}L_t^{x x} & L_t^{x v} \\ L_t^{x v} & L_t^{v v}\end{array}\right)=\left(\begin{array}{cc}\sqrt{\sigma_t^{x x}} & 0 \\ \frac{\sigma_t^{x v}}{\sqrt{\sigma_t^{x x}}} & \sqrt{\frac{\sigma_t^{x x} \sum_t^{v y}-\left(\sigma_t^{x v}\right)^2}{\sigma_t^{x x}}}\end{array}\right)$"""
      L_t_xx = jnp.sqrt(sigma_xx)
      L_t_vx = sigma_xv / jnp.sqrt(sigma_xx)
      L_t_vv = jnp.sqrt((sigma_xx * sigma_vv - sigma_xv**2) / sigma_xx)
      return L_t_xx, L_t_vx, L_t_vv

    # def L_t_inv_T(self, x_0, t):
    #     _, sigma_t = self.hsm_kernel_params(x_0, t)
    #     sigma_xx, sigma_xv, sigma_vv = sigma_t[0, 0], sigma_t[0, 1], sigma_t[1, 1]
    #     sqrt_sigma_xx = jnp.sqrt(sigma_xx)
    #     sqrt_det = jnp.sqrt(sigma_xx * sigma_vv - sigma_xv**2)

    #     L_t_inv_T = jnp.array(
    #         [
    #             [1 / sqrt_sigma_xx, -sigma_xv / (sqrt_sigma_xx * sqrt_det)],
    #             [0, sqrt_sigma_xx / sqrt_det],
    #         ]
    #     )
    #     return L_t_inv_T

    # def sigma_t_inv(self, sigma_t):
    #     """Compute sigma_t^(-1) as defined in:
    #     ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=23&annotation=8M85CFQL))
    #     ([Dockhorn et al., 2022, p. 23](zotero://select/library/items/EW8U6A8H))"""
    #     sigma_xx, sigma_xv, sigma_vv = sigma_t[0, 0], sigma_t[0, 1], sigma_t[1, 1]
    #     assert sigma_xx.shape == (self.state_dim, self.state_dim)
    #     assert sigma_xv.shape == (self.state_dim, self.state_dim)
    #     assert sigma_vv.shape == (self.state_dim, self.state_dim)
    #     det = sigma_xx * sigma_vv - sigma_xv**2
    #     return jnp.array([[sigma_vv, -sigma_xv], [-sigma_xv, sigma_xx]]) / det
      
    def sample_ut(self, epsilon_2d, x_0, t):
      mu_t = self.mu_t(x_0, t)
      L_t_xx, L_t_vx, L_t_vv = self.L_t_blocks(x_0, t)
      eps_0_d = epsilon_2d[:self.state_dim]
      eps_d_2d = epsilon_2d[self.state_dim:]
      L_t_eps = jnp.concatenate([L_t_vx * eps_0_d, L_t_vv * eps_d_2d], axis=0)
      return mu_t + L_t_eps

    def grad_ut_log_pt(self, x_0, t, epsilon_2d):
        """Compute the gradient of log p_t(u_t | ·) with respect to u_t."""
        L_t_inv_T = self.L_t_inv_T(x_0, t)
        return -jnp.kron(L_t_inv_T, jnp.eye(self.state_dim)) @ epsilon_2d

    def grad_vt_log_pt(self, x_0, t, epsilon_d):
        """Compute the gradient of log p_t(u_t | ·) with respect to v_t."""
        l_t = self.ell_t(x_0, t)
        return -l_t * epsilon_d

    def grad_ut_log_pt(self, x_0, t, epsilon_2d):
        """Compute the HSM gradient of log p_t(u_t | x_0)."""
        _, sigma_t = self.hsm_kernel_params(x_0, t, self.gamma)
        return self.grad_ut_log_pt(sigma_t, epsilon_2d)

    def grad_vt_log_pt(self, x_0, t, epsilon_d):
        """Compute the HSM gradient of log p_t(u_t | x_0) with respect to v_t."""
        _, sigma_t = self.hsm_kernel_params(x_0, t, self.gamma)
        return self.grad_vt_log_pt(sigma_t, epsilon_d)
