import jax.numpy as jnp
import equinox as eqx
import diffrax

class CriticallyDampedLangevinDiffusion(eqx.Module):
    d: int
    M: float
    Gamma: float
    beta: float
    T: float

    def __init__(self, d: int, M: float, beta: float, T: float = 1.0):
        if d <= 0 or M <= 0 or beta <= 0 or T <= 0:
          raise ValueError("All parameters must be positive.")
        self.d = d
        self.M = M
        self.Gamma = jnp.sqrt(4 * M)
        self.beta = beta
        self.T = T

    def drift(self, t, u, args):
        x, v = u[:self.d], u[self.d:]
        f_x = self.beta * self.M**-1 * v
        f_v = -self.beta * x - self.beta * self.Gamma * self.M**-1 * v
        return jnp.concatenate([f_x, f_v])

    def diffusion(self, t, u, args):
        G = jnp.zeros((2*self.d, 2*self.d))
        G = G.at[self.d:, self.d:].set(jnp.sqrt(2 * self.Gamma * self.beta) * jnp.eye(self.d))
        return G
      
    def get_terms(self, bm):
        drift_term = diffrax.ODETerm(self.drift)
        diffusion_term = diffrax.ControlTerm(self.diffusion, bm)
        return diffrax.MultiTerm(drift_term, diffusion_term)

    def B(self, t):
        return self.beta * t

    def mean(self, x_0, v_0, t):
        B_t = self.B(t)
        exp_term = jnp.exp(-2 * B_t / self.Gamma)
        
        x_t = (2 * B_t / self.Gamma * x_0 + 4 * B_t / self.Gamma**2 * v_0 + x_0) * exp_term
        v_t = (-B_t * x_0 - 2 * B_t / self.Gamma * v_0 + v_0) * exp_term
        
        return jnp.concatenate([x_t, v_t])

    def covariance(self, t, Sigma_0_xx, Sigma_0_vv):
        B_t = self.B(t)
        exp_term = jnp.exp(-4 * B_t / self.Gamma)
        exp_term_plus = jnp.exp(4 * B_t / self.Gamma)
        
        Sigma_xx = (Sigma_0_xx + exp_term_plus - 1 + 
                    4*B_t/self.Gamma*(Sigma_0_xx-1) + 
                    4*B_t**2/self.Gamma**2*(Sigma_0_xx-2) + 
                    16*B_t**2/self.Gamma**4*Sigma_0_vv) * exp_term
        
        Sigma_xv = (-B_t*Sigma_0_xx + 4*B_t/self.Gamma**2*Sigma_0_vv - 
                    2*B_t**2/self.Gamma*(Sigma_0_xx-2) - 
                    8*B_t**2/self.Gamma**3*Sigma_0_vv) * exp_term
        
        Sigma_vv = (self.Gamma**2/4*(exp_term_plus-1) + B_t*self.Gamma + 
                    Sigma_0_vv*(1+4*B_t**2/self.Gamma**2-4*B_t/self.Gamma) + 
                    B_t**2*(Sigma_0_xx-2)) * exp_term
        
        Sigma_t = jnp.array([[Sigma_xx, Sigma_xv], [Sigma_xv, Sigma_vv]])
        return jnp.kron(Sigma_t, jnp.eye(self.d))

    def perturbation_kernel_dsm(self, u_0, t):
        # “when conditioning on initial data and velocity samples x0 and v0 (as in denoising score matching (DSM)), the mean and covariance matrix of the perturbation kernel p(ut|u0) can be obtained by setting μ0 = (x0, v0)>, Σ0xx = 0, and Σ0vv = 0.” ([Dockhorn et al., 2022, p. 21](zotero://select/library/items/EW8U6A8H)) ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=21&annotation=6ZWQTEZZ))
        x_0, v_0 = u_0[:self.d], u_0[self.d:]
        mean = self.mean(x_0, v_0, t)
        cov = self.covariance(t, Sigma_0_xx=0, Sigma_0_vv=0)
        return mean, cov

    def perturbation_kernel_hsm(self, x_0, t, gamma=1.0):
        # “conditioning only on initial data samples x0 and marginalizing over the full initial velocity distribution (as in our hybrid score matching (HSM), see Sec. C), the mean and covariance matrix of the perturbation kernel p(ut|x0) can be obtained by setting μ0 = (x0, 0d)>, Σ0xx = 0, and Σ0vv = γM” ([Dockhorn et al., 2022, p. 21](zotero://select/library/items/EW8U6A8H)) ([pdf](zotero://open-pdf/library/items/NAYANTYJ?page=21&annotation=CW5PUPSH))
        v_0_hsm = jnp.zeros_like(x_0)
        mean = self.mean(x_0, v_0_hsm, t)
        cov = self.covariance(t, Sigma_0_xx=0, Sigma_0_vv=gamma*self.M)
        return mean, cov