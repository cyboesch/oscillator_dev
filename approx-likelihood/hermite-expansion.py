#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class VDPTransitionDensity:
    def __init__(self, mu=1.0, K=0.1, sigma1=0.1, sigma2=0.1, dt=0.01, K_order=2):
        self.mu = mu
        self.K = K
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dt = dt
        self.K_order = K_order
        
        # Pre-compute Hermite polynomials
        self.hermite_polys = [hermite(i) for i in range(K_order + 1)]
    
    def H(self, n, x):
        """Evaluate nth Hermite polynomial at x"""
        return self.hermite_polys[n](x)
    
    def compute_coefficients(self, x):
        """Compute all expansion coefficients up to second order"""
        coef = {}
        
        # Zero order
        coef['c0000'] = -(np.sum(x**2))/(2*self.sigma1**2*self.dt)
        
        # First order terms for oscillator 1
        coef['c1000'] = (x[0]/(self.sigma1**2*self.dt) + 
                        self.mu*x[1]/self.sigma1**2 - 
                        self.K*x[0]/self.sigma1**2 + 
                        self.K*x[2]/self.sigma1**2)
        
        coef['c0100'] = (x[1]/(self.sigma1**2*self.dt) - 
                        x[0]/self.sigma1**2 + 
                        self.mu*(1-x[0]**2)*x[1]/self.sigma1**2)
        
        # First order terms for oscillator 2
        coef['c0010'] = (x[2]/(self.sigma2**2*self.dt) + 
                        self.mu*x[3]/self.sigma2**2 - 
                        self.K*x[2]/self.sigma2**2 + 
                        self.K*x[0]/self.sigma2**2)
        
        coef['c0001'] = (x[3]/(self.sigma2**2*self.dt) - 
                        x[2]/self.sigma2**2 + 
                        self.mu*(1-x[2]**2)*x[3]/self.sigma2**2)
        
        # Mixed second order terms
        coef['c1010'] = self.K/(self.sigma1*self.sigma2*self.dt)
        coef['c0101'] = self.K/(self.sigma1*self.sigma2*self.dt)
        
        # Pure second order terms for oscillator 1
        coef['c2000'] = (-1/(2*self.sigma1**2*self.dt) + 
                        self.mu*x[1]**2/(2*self.sigma1**2) - 
                        self.K*x[0]**2/(2*self.sigma1**2))
        
        coef['c0200'] = (-1/(2*self.sigma1**2*self.dt) + 
                        self.mu*(1-x[0]**2)/(2*self.sigma1**2))
        
        # Pure second order terms for oscillator 2
        coef['c0020'] = (-1/(2*self.sigma2**2*self.dt) + 
                        self.mu*x[3]**2/(2*self.sigma2**2) - 
                        self.K*x[2]**2/(2*self.sigma2**2))
        
        coef['c0002'] = (-1/(2*self.sigma2**2*self.dt) + 
                        self.mu*(1-x[2]**2)/(2*self.sigma2**2))
        
        return coef
    
    def log_transition_density(self, y, x):
        """Compute full log transition density approximation"""
        coef = self.compute_coefficients(x)
        
        log_p = -0.5*np.log(2*np.pi*self.sigma1**2*self.dt) - 0.5*np.log(2*np.pi*self.sigma2**2*self.dt)
        
        # Add all terms
        log_p += coef['c0000']
        
        # First order
        log_p += coef['c1000'] * self.H(1, y[0])
        log_p += coef['c0100'] * self.H(1, y[1])
        log_p += coef['c0010'] * self.H(1, y[2])
        log_p += coef['c0001'] * self.H(1, y[3])
        
        # Second order mixed
        log_p += coef['c1010'] * self.H(1, y[0]) * self.H(1, y[2])
        log_p += coef['c0101'] * self.H(1, y[1]) * self.H(1, y[3])
        
        # Second order pure
        log_p += coef['c2000'] * self.H(2, y[0])
        log_p += coef['c0200'] * self.H(2, y[1])
        log_p += coef['c0020'] * self.H(2, y[2])
        log_p += coef['c0002'] * self.H(2, y[3])
        
        return log_p
    
    def simulate_trajectory(self, T, x0=None):
        """Simulate VDP trajectory using Euler-Maruyama method"""
        if x0 is None:
            x0 = np.zeros(4)
        
        steps = int(T / self.dt)
        traj = np.zeros((steps + 1, 4))
        traj[0] = x0
        
        for i in range(steps):
            # Current state
            x1, v1, x2, v2 = traj[i]
            
            # Drift terms
            dx1 = v1
            dv1 = self.mu * (1 - x1**2) * v1 - x1 + self.K * (x2 - x1)
            dx2 = v2
            dv2 = self.mu * (1 - x2**2) * v2 - x2 + self.K * (x1 - x2)
            
            # Add noise
            traj[i+1,0] = x1 + dx1 * self.dt + self.sigma1 * np.sqrt(self.dt) * np.random.randn()
            traj[i+1,1] = v1 + dv1 * self.dt + self.sigma1 * np.sqrt(self.dt) * np.random.randn()
            traj[i+1,2] = x2 + dx2 * self.dt + self.sigma2 * np.sqrt(self.dt) * np.random.randn()
            traj[i+1,3] = v2 + dv2 * self.dt + self.sigma2 * np.sqrt(self.dt) * np.random.randn()
            
        return traj


def visualize_expansion_enhanced():
    vdp = VDPTransitionDensity()
    
    # Generate trajectory
    T = 50
    traj = vdp.simulate_trajectory(T)
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Phase space trajectories with density estimation
    ax1 = fig.add_subplot(231)
    ax1.plot(traj[:,0], traj[:,1], 'b.', alpha=0.5, markersize=1)
    sns.kdeplot(x=traj[:,0], y=traj[:,1], ax=ax1, levels=20, cmap='viridis')
    ax1.set_title('Phase Space Density (Oscillator 1)')
    
    # 2. Coefficient evolution
    ax2 = fig.add_subplot(232)
    coefs_t = []
    for i in range(len(traj)-1):
        coefs_t.append(vdp.compute_coefficients(traj[i]))
    
    coef_names = ['c1000', 'c0100', 'c2000', 'c0200']
    t = np.arange(len(traj)-1)*vdp.dt
    for name in coef_names:
        values = [c[name] for c in coefs_t]
        ax2.plot(t, values, label=name)
    ax2.set_title('Coefficient Evolution')
    ax2.legend()
    
    # 3. Transition density slices
    ax3 = fig.add_subplot(233, projection='3d')
    x_grid = np.linspace(-3, 3, 30)
    y_grid = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    x0 = np.array([0, 0, 0, 0])
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            y = np.array([X[i,j], Y[i,j], 0, 0])
            Z[i,j] = np.exp(vdp.log_transition_density(y, x0))
    
    ax3.plot_surface(X, Y, Z, cmap='viridis')
    ax3.set_title('3D Transition Density')
    
    # Continue with more visualizations...
    
    plt.tight_layout()
    plt.show()

class VDPTransitionDensityVisualizer:
    def __init__(self, vdp_system):
        self.vdp = vdp_system
        self.fig = plt.figure(figsize=(15, 10))
        
    def animate_transition_density(self, n_frames=100):
        """Animate evolution of transition density from different initial conditions"""
        
        # Set up grid for density evaluation
        x_grid = np.linspace(-3, 3, 50)
        y_grid = np.linspace(-3, 3, 50)
        self.X, self.Y = np.meshgrid(x_grid, y_grid)
        
        # Initialize subplots
        gs = self.fig.add_gridspec(2, 2)
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Hermite approximation
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Numerical solution
        self.ax3 = self.fig.add_subplot(gs[1, :])  # Difference
        
        # Initialize plots
        self.im1 = self.ax1.pcolormesh(self.X, self.Y, np.zeros_like(self.X), 
                                      shading='auto', cmap='viridis')
        self.im2 = self.ax2.pcolormesh(self.X, self.Y, np.zeros_like(self.X), 
                                      shading='auto', cmap='viridis')
        self.im3 = self.ax3.pcolormesh(self.X, self.Y, np.zeros_like(self.X), 
                                      shading='auto', cmap='RdBu')
        
        # Animation
        anim = FuncAnimation(self.fig, self.update, frames=n_frames, 
                            interval=50, blit=False)
        return anim
    
    def compute_numerical_density(self, x0, n_samples=10000):
        """Compute numerical transition density using Monte Carlo"""
        samples = np.zeros((n_samples, 4))
        
        # Generate many short trajectories
        for i in range(n_samples):
            traj = self.vdp.simulate_trajectory(self.vdp.dt, x0)
            samples[i] = traj[-1]
        
        # Compute KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(samples[:, :2].T)
        Z = kde.evaluate(np.vstack([self.X.ravel(), self.Y.ravel()]))
        return Z.reshape(self.X.shape)
    
    def update(self, frame):
        """Update function for animation"""
        # Vary initial condition
        theta = 2 * np.pi * frame / 100
        x0 = np.array([np.cos(theta), np.sin(theta), 0, 0])
        
        # Compute Hermite approximation with numerical stability
        Z1 = np.zeros_like(self.X)
        for i in range(len(self.X)):
            for j in range(len(self.Y)):
                y = np.array([self.X[i,j], self.Y[i,j], 0, 0])
                # Add numerical stability by subtracting the maximum value
                log_density = self.vdp.log_transition_density(y, x0)
                # Clip very large negative values to prevent underflow
                log_density = np.clip(log_density, -500, 500)
                Z1[i,j] = np.exp(log_density)
        
        # Normalize the density
        Z1 = Z1 / (np.sum(Z1) * (self.X[1,1] - self.X[0,0]) * (self.Y[1,1] - self.Y[0,0]))
        
        # Compute numerical solution
        Z2 = self.compute_numerical_density(x0)
        
        # Compute difference
        Z3 = Z1 - Z2
        
        # Update plots
        self.im1.set_array(Z1.ravel())
        self.im2.set_array(Z2.ravel())
        self.im3.set_array(Z3.ravel())
        
        self.ax1.set_title(f'Hermite Approximation (t={frame*self.vdp.dt:.2f})')
        self.ax2.set_title('Numerical Solution')
        self.ax3.set_title('Difference')
        
        return self.im1, self.im2, self.im3

def compare_solutions():
    """Compare Hermite approximation with numerical solution"""
    vdp = VDPTransitionDensity()
    visualizer = VDPTransitionDensityVisualizer(vdp)
    
    # Create animation
    anim = visualizer.animate_transition_density()
    
    # Add colorbar
    plt.colorbar(visualizer.im1, ax=visualizer.ax1, label='Density')
    plt.colorbar(visualizer.im2, ax=visualizer.ax2, label='Density')
    plt.colorbar(visualizer.im3, ax=visualizer.ax3, label='Difference')
    
    plt.tight_layout()
    
    # Save animation
    anim.save('transition_density_evolution.gif', writer='pillow')
    plt.show()

# Error analysis
def compute_error_metrics():
    """Compute error metrics between approximation and numerical solution"""
    vdp = VDPTransitionDensity()
    x_test = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0]
    ])
    
    errors = []
    for x0 in x_test:
        # Generate grid
        x_grid = np.linspace(-3, 3, 50)
        y_grid = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute approximation
        Z1 = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(Y)):
                y = np.array([X[i,j], Y[i,j], 0, 0])
                Z1[i,j] = np.exp(vdp.log_transition_density(y, x0))
        
        # Compute numerical solution
        visualizer = VDPTransitionDensityVisualizer(vdp)
        Z2 = visualizer.compute_numerical_density(x0)
        
        # Compute errors with numerical stability
        l1_error = np.mean(np.abs(Z1 - Z2))
        l2_error = np.sqrt(np.mean((Z1 - Z2)**2))
        
        # Add small constant to prevent division by zero
        eps = 1e-10
        kl_div = np.mean(Z1 * np.log((Z1 + eps)/(Z2 + eps)))
        
        errors.append({
            'x0': x0,
            'L1': l1_error,
            'L2': l2_error,
            'KL': kl_div
        })
    
    return errors

# if __name__ == "__main__":

#%%
# Create VDP system
vdp = VDPTransitionDensity()

# Run enhanced visualization
print("Generating enhanced visualization...")
visualize_expansion_enhanced()

# # Run comparison visualization
# print("\nGenerating comparison animation...")
# compare_solutions()

# # Compute and display error metrics
# print("\nComputing error metrics...")
# errors = compute_error_metrics()
# print("\nError Analysis:")
# for e in errors:
#     print(f"\nInitial condition: {e['x0']}")
#     print(f"L1 error: {e['L1']:.6f}")
#     print(f"L2 error: {e['L2']:.6f}")
#     print(f"KL divergence: {e['KL']:.6f}")
    
#%%