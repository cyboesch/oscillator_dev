# %% [markdown]
# # Boltzmann Machine Learning for 2Dof NVS, unsuccessful

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
import matplotlib.pyplot as plt
from functools import partial

from diffrax import ControlTerm, MultiTerm, ODETerm, PIDController

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Setup Problem

# %%
# ----------------------------
# 1. Define Parameters
# ----------------------------


# Physical Constants and Parameters
M1 = 1.0
M2 = 1.0

w1 = 1.0          # Natural frequency of oscillator 1 (rad/s)
w2 = 1.0          # Natural frequency of oscillator 2 (rad/s)

wp1 = 2*(w1+0.0)  # Driving frequency for oscillator 1 (rad/s)
wp2 = 2*(w2+0.0)  # Driving frequency for oscillator 2 (rad/s)

period1 = 2*jnp.pi/w1
period2 = 2*jnp.pi/w2

Fp1 = 0.0        # Amplitude of the wp driving for oscillator 1
Fp2 = 0.00         # Amplitude of the wp driving for oscillator 2

Q1 = 100.0       # Quality factor for oscillator 1
Q2 = 100.0       # Quality factor for oscillator 2

GAMMA1 = w1/(2*Q1)
GAMMA2 = w2/(2*Q2)

gamma1 = 0.00    # Nonlinearity coefficient for oscillator 1 (Duffing term)
gamma2 = 0.00    # Nonlinearity coefficient for oscillator 2 (Duffing term)

c = .01


k_B = 1.0             # Boltzmann's constant (arbitrary units for simulation)
T = .3             # Temperature (K)
    # Noise strength

# Simulation Parameters
t0 = 0.0  
num_periods = 20000# Start time
period = 2*jnp.pi/w1
t1 = num_periods*period
dt0 = period/1000
num_pts_per_period = 100
ts_dim = jnp.linspace(t0, t1, 10000)
saveat = diffrax.SaveAt(ts=ts_dim)

y0_QP = jnp.array([0, 0, 0, 0])


# %%
def Hamiltonian(Q1, P1, Q2, P2, args):
    dw = wp1/2 - w1  # detuning term
    
    # Parameters of the Hidden oscillator
    gamma_h, wp_h, Fp_h = args
    
    # Single oscillator terms (for both oscillators)
    H1 = 3*gamma1/(16*wp1) * (Q1**2 + P1**2)**2 + dw/2 *(Q1**2 - P1**2) + Fp1/(4*M1*wp1) * (P1**2 - Q1**2)
    H2 = 3*gamma_h/(16*wp_h) * (Q2**2 + P2**2)**2 + dw/2 *(Q2**2 - P2**2) + Fp_h/(4*M2*wp_h) * (P2**2 - Q2**2)

    # Coupling term
    H_coupling = c/4 * ((Q1 - Q2)**2 + (P1 - P2)**2)
    
    return H1 + H2 + H_coupling


def drift_QP_fn(t, state, args):
    """
    Drift function for the RWA equations
    """
    Q1, P1, Q2, P2 = state  # Unpack all state variables
    
    # Calculate gradients of Hamiltonian for both oscillators
    dQ1dt =  jax.grad(Hamiltonian, 1)(Q1, P1, Q2, P2, args) - GAMMA1*Q1
    dP1dt = -jax.grad(Hamiltonian, 0)(Q1, P1, Q2, P2, args) - GAMMA1*P1
    dQ2dt =  jax.grad(Hamiltonian, 3)(Q1, P1, Q2, P2, args) - GAMMA2*Q2
    dP2dt = -jax.grad(Hamiltonian, 2)(Q1, P1, Q2, P2, args) - GAMMA2*P2
    
    return jnp.array([dQ1dt, dP1dt, dQ2dt, dP2dt])



def diffusion_QP_fn(t, state, args):
    """
    Diffusion matrix for Q and P equations for two coupled oscillators.
    Returns a 4x4 matrix with noise terms for both oscillators.
    """
    # Noise strength for oscillator 1
    noise_strength_QP1 = jnp.sqrt(2*GAMMA1*k_B * T/(M1*w1**2))
    # Noise strength for oscillator 2  
    noise_strength_QP2 = jnp.sqrt(2*GAMMA2*k_B * T/(M2*w2**2))
    
    # Create 4x4 matrix for both oscillators [Q1, P1, Q2, P2]
    noise_matrix = jnp.zeros((4, 4))
    noise_matrix = noise_matrix.at[0:2, 0:2].set(noise_strength_QP1 * jnp.eye(2))  # First oscillator
    noise_matrix = noise_matrix.at[2:4, 2:4].set(noise_strength_QP2 * jnp.eye(2))  # Second oscillator
    return noise_matrix



# %%
@jax.jit
def solve_SDE_QP(args, seed=0):
    """
    Solve the SDE for the QP equations with configurable random seed
    
    Args:
        args: Parameters for the SDE
        seed (int): Random seed for noise generation (default=0)
    
    Returns:
        Array of solution samples
    """
    # Define state shape
    w_shape = (4,)  # state is (Q, P)
    
    # Create Brownian motion with specified seed
    brownian_motion_QP = diffrax.VirtualBrownianTree(
        t0, t1, 1.e-11, w_shape, jr.PRNGKey(seed), diffrax.SpaceTimeLevyArea
    )
    
    # Define terms with the new Brownian motion
    terms_QP = MultiTerm(ODETerm(drift_QP_fn), 
                        ControlTerm(diffusion_QP_fn, brownian_motion_QP))
    
    # Solve the SDE
    solution_QP = diffrax.diffeqsolve(
        terms_QP,
        solver=diffrax.SRA1(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0_QP,
        args=args,
        saveat=saveat,
        progress_meter=diffrax.TqdmProgressMeter(),
        max_steps=10000000000,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
    )
    
    sde_samples_QP = solution_QP.ys
    return sde_samples_QP



# ### Solve to get samples from target and initial distributions (or load them from file)
args_0 = (.1, wp2, 0.3)
args_target = (0.03, wp2, 0.5)
sde_samples_QP_0 = solve_SDE_QP(args_0, seed=0)
sde_samples_QP_target = solve_SDE_QP(args_target, seed=1)


def Hamiltonian_clamped(Q2, P2, Q1_fixed, P1_fixed, args):
    dw = wp1/2 - w1
    gamma_h, wp_h, Fp_h = args
    
    H2 = 3*gamma_h/(16*wp_h) * (Q2**2 + P2**2)**2 + dw/2 *(Q2**2 - P2**2) + Fp_h/(4*M2*wp_h) * (P2**2 - Q2**2)
    H_coupling = c/4 * ((Q1_fixed - Q2)**2 + (P1_fixed - P2)**2)
    
    return H2 + H_coupling

def drift_QP_clamped(t, state, Q1_fixed, P1_fixed, args):
    Q2, P2 = state  # Now state only contains Q2 and P2
    
    dQ2dt = jax.grad(Hamiltonian_clamped, 1)(Q2, P2, Q1_fixed, P1_fixed, args) - GAMMA2*Q2
    dP2dt = -jax.grad(Hamiltonian_clamped, 0)(Q2, P2, Q1_fixed, P1_fixed, args) - GAMMA2*P2
    
    return jnp.array([dQ2dt, dP2dt])

@jax.jit
def solve_SDE_QP_clamped(Q1_fixed, P1_fixed, args, seed=0):
    """
    Solve the SDE for the QP equations with clamped Q1 and P1 values
    """
    # Define new terms for clamped system (2D instead of 4D)
    w_shape_clamped = (2,)  # state is now just (Q2, P2)
    brownian_motion_QP_clamped = diffrax.VirtualBrownianTree(
        t0, t1, 1.e-11, w_shape_clamped, jr.PRNGKey(seed), diffrax.SpaceTimeLevyArea
    )
    
    # Create partial function with fixed Q1, P1
    drift_fn = lambda t, state, args: drift_QP_clamped(t, state, Q1_fixed, P1_fixed, args)
    
    terms_QP_clamped = MultiTerm(
        ODETerm(drift_fn), 
        ControlTerm(lambda t, state, args: diffusion_QP_fn(t, state, args)[2:4, 2:4], brownian_motion_QP_clamped)
    )
    
    y0_QP_clamped = jnp.array([0, 0])  # Initial conditions for Q2, P2 only
    
    solution_QP = diffrax.diffeqsolve(
        terms_QP_clamped,
        solver=diffrax.SRA1(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0_QP_clamped,
        args=args,
        saveat=saveat,
        progress_meter=diffrax.TqdmProgressMeter(),
        max_steps=10000000000,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
    )
    
    sde_samples_QP = solution_QP.ys
    return sde_samples_QP

downsample_step = 100

# Downsample the Q1,P1 values from the zero simulation, starting halfway through
halfway_idx = len(sde_samples_QP_target) // 2  # Get the midpoint index
Q1_samples_target = sde_samples_QP_target[::downsample_step, 0]
P1_samples_target = sde_samples_QP_target[::downsample_step, 1]

Q1P1_pairs_target = jnp.stack([Q1_samples_target, P1_samples_target], axis=1)  # Shape: (n_samples, 2)

# Vectorize the clamped solver over Q1,P1 pairs
def batch_solve_clamped(Q1P1_pair, args, seed):
    return solve_SDE_QP_clamped(Q1P1_pair[0], Q1P1_pair[1], args, seed)

# Run the parallel simulation
seeds = jnp.arange(len(Q1P1_pairs_target))

# ## Setup BM training

def df_dgamma_h(Q2, P2, args):
    """Drift term derivatives with respect to gamma_h"""
    gamma_h, wp_h, Fp_h = args
    
    # For Q2: dQ2/dt = dH/dP2 - GAMMA2*Q2
    # Only dH/dP2 depends on gamma_h through: 3*gamma_h/(16*wp_h) * (Q2**2 + P2**2)**2
    dQ2_dgamma = 3/(4*wp_h) * (Q2**2 + P2**2) * P2
    
    # For P2: dP2/dt = -dH/dQ2 - GAMMA2*P2
    # Only -dH/dQ2 depends on gamma_h through: -3*gamma_h/(16*wp_h) * (Q2**2 + P2**2)**2
    dP2_dgamma = -3/(4*wp_h) * (Q2**2 + P2**2) * Q2
    
    return jnp.array([dQ2_dgamma, dP2_dgamma])

def df_dFp_h(Q2, P2, args):
    """Drift term derivatives with respect to Fp_h"""
    gamma_h, wp_h, Fp_h = args
    
    # For Q2: dQ2/dt = dH/dP2 - GAMMA2*Q2
    # dH/dP2 depends on Fp_h through: Fp_h/(4*M2*wp_h) * (P2**2 - Q2**2)
    dQ2_dFp = 1/(2*M2*wp_h) * P2
    
    # For P2: dP2/dt = -dH/dQ2 - GAMMA2*P2
    # -dH/dQ2 depends on Fp_h through: -Fp_h/(4*M2*wp_h) * (P2**2 - Q2**2)
    dP2_dFp = 1/(2*M2*wp_h) * Q2
    
    return jnp.array([dQ2_dFp, dP2_dFp])

def compute_gradients(sde_samples_free, sde_samples_clamped, args):
    """
    Compute gradients using the rule:
    grad = df/dp(Q_free,P_free).dot([Q_clamped;P_clamped]-[Q_free;P_free])
    for each clamped sample individually
    
    Args:
        sde_samples_free: samples from free running simulation (Q1,P1,Q2,P2)
        sde_samples_clamped: samples from clamped simulation (Q2,P2)
        args: (gamma_h, wp_h, Fp_h)
    
    Returns:
        gradients for gamma_h and Fp_h
    """
    # Compute df/dp for free samples (derivatives of drift terms)
    df_dgamma_free = jax.vmap(lambda sample: df_dgamma_h(sample[2], sample[3], args))(sde_samples_free)
    df_dFp_free = jax.vmap(lambda sample: df_dFp_h(sample[2], sample[3], args))(sde_samples_free)
    # Take mean of df/dp
    mean_df_dgamma = jnp.mean(df_dgamma_free, axis=0)  # Shape: (2,)
    mean_df_dFp = jnp.mean(df_dFp_free, axis=0)       # Shape: (2,)
    
    # Compute state differences for each clamped trajectory
    # sde_samples_clamped shape: (n_clamped, time, 2)
    # sde_samples_free shape: (time, 4)
    state_diffs = sde_samples_clamped - sde_samples_free[:, 2:4]  # Broadcasting handles this
    
    print(state_diffs.shape)
    
    # Take mean over time for each clamped trajectory
    expected_state_diffs = jnp.mean(state_diffs, axis=0)  # Shape: (n_clamped, 2)
    
    print(expected_state_diffs.shape)

    
    # Compute dot product for each clamped trajectory
    grad_gammas = jnp.dot(expected_state_diffs, mean_df_dgamma)  # Shape: (n_clamped,)
    grad_Fps = jnp.dot(expected_state_diffs, mean_df_dFp)        # Shape: (n_clamped,)
    
    
    return grad_gammas, grad_Fps
# %%
# Training loop
def train_step(args, learning_rate=0.01):
    """
    Perform one training step
    """
    # Get samples from current parameters
    sde_samples_free = solve_SDE_QP(args)
    clamped_samples = jax.vmap(batch_solve_clamped, in_axes=(0, None,0))(Q1P1_pairs_target, args, seeds)
    
    halfway_idx = len(sde_samples_QP_target) // 2

    grad_gamma_per_clamping, grad_Fp_per_clamping = jax.vmap(compute_gradients, in_axes=(None, 0, None))(
        sde_samples_free[halfway_idx:], 
        clamped_samples[:,halfway_idx:],  
        args
    )
    # Sum up the gradients across all clamped samples
    grad_gamma = jnp.mean(grad_gamma_per_clamping)  # or jnp.sum(grad_gamma)
    grad_Fp = jnp.mean(grad_Fp_per_clamping)        # or jnp.sum(grad_Fp)
    
    print(f"Gradients: gamma_h = {grad_gamma:.4f}, Fp_h = {grad_Fp:.4f}")
    # Update parameters
    gamma_h, wp_h, Fp_h = args
    new_gamma_h = gamma_h + learning_rate * grad_gamma
    new_Fp_h = Fp_h + learning_rate * grad_Fp
    
    return (new_gamma_h, wp_h, new_Fp_h)

def train(initial_args, n_steps=100, learning_rate=0.001):
    args = initial_args
    # Store history of parameters
    history = {
        'gamma_h': [initial_args[0]],
        'Fp_h': [initial_args[2]],
        'step': [0]
    }
    
    for step in range(n_steps):
        args = train_step(args, learning_rate)
        # Store parameters
        history['gamma_h'].append(args[0])
        history['Fp_h'].append(args[2])
        history['step'].append(step + 1)
        
        if step % 1 == 0:
            print(f"Step {step}: gamma_h = {args[0]:.4f}, Fp_h = {args[2]:.4f}")
    
    return args, history

args_target = args_target
# Starting from args_0 and training towards args_target
initial_args = args_0  

n_steps = 3
learning_rate = 10.
final_args = train(initial_args, n_steps=n_steps, learning_rate=learning_rate)

# Run simulation with final args
sde_samples_QP_final = solve_SDE_QP(final_args[0])


history = final_args[1]
# Create output directory if it doesn't exist
import os
output_dir = "../out/BMlearning"
os.makedirs(output_dir, exist_ok=True)

# Create figure with proper spacing for all plots
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[3, 2], width_ratios=[1, 1, 1, 0.05], wspace=0.2, hspace=0.3)

# Distribution plots in top row
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
cax = fig.add_subplot(gs[0, 3])

# Training history plots in bottom row
ax4 = fig.add_subplot(gs[1, 0:2])  # Spans two columns
ax5 = fig.add_subplot(gs[1, 2])    # Takes the third column

# Get global min/max for consistent axes
all_Q = jnp.concatenate([
    sde_samples_QP_target[:, 0], 
    sde_samples_QP_0[:, 0],
    sde_samples_QP_final[:, 0]
])
all_P = jnp.concatenate([
    sde_samples_QP_target[:, 1], 
    sde_samples_QP_0[:, 1],
    sde_samples_QP_final[:, 1]
])
q_lim = max(abs(all_Q.min()), abs(all_Q.max()))
p_lim = max(abs(all_P.min()), abs(all_P.max()))
lim = max(q_lim, p_lim) * 1.1

# Plot distributions
sample_step = 1
bins = 50
range_lim = [[-lim, lim], [-lim, lim]]
hist1 = ax1.hist2d(sde_samples_QP_0[::sample_step, 0], 
                   sde_samples_QP_0[::sample_step, 1], 
                   bins=bins, cmap='viridis', range=range_lim)
hist2 = ax2.hist2d(sde_samples_QP_target[::sample_step, 0], 
                   sde_samples_QP_target[::sample_step, 1], 
                   bins=bins, cmap='viridis', range=range_lim)
hist3 = ax3.hist2d(sde_samples_QP_final[::sample_step, 0], 
                   sde_samples_QP_final[::sample_step, 1], 
                   bins=bins, cmap='viridis', range=range_lim)

# Set identical axes properties for distribution plots
for ax in [ax1, ax2, ax3]:
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Q₁')
    ax.set_ylabel('P₁')
    
# Titles for distribution plots
ax1.set_title('Initial Distribution\n' + f'γ={args_0[0]:.3f}, Fp={args_0[2]:.3f}')
ax2.set_title('Target Distribution\n' + f'γ={args_target[0]:.3f}, Fp={args_target[2]:.3f}')
ax3.set_title('Final Distribution\n' + f'γ={final_args[0][0]:.3f}, Fp={final_args[0][2]:.3f}')    

# Add colorbar
plt.colorbar(hist1[3], cax=cax, label='Count')

# Add training history plots
ax4.plot(history['step'], history['gamma_h'], '-o')
ax4.axhline(y=args_target[0], color='r', linestyle='--', label='Target γ')
ax4.set_xlabel('Step')
ax4.set_ylabel('γ_h')
ax4.grid(True)
ax4.legend()

ax5.plot(history['step'], history['Fp_h'], '-o')
ax5.axhline(y=args_target[2], color='r', linestyle='--', label='Target Fp')
ax5.set_xlabel('Step')
ax5.set_ylabel('Fp_h')
ax5.grid(True)
ax5.legend()

# Compact parameter title and filename
params_str = (f"M{M1}_w{w1}_wp{wp1}_Q{Q1}_g{gamma1}_c{c}_T{T}"
             f"_Nper{num_periods}_Nclamp{P1_samples_target.shape[0]}"
             f"_Nsteps{n_steps}_lr{learning_rate}")
title = (f'Parameters: M₁,₂={M1}, ω₁,₂={w1}, ωp₁,₂={wp1}, '
         f'Q₁,₂={Q1}, γ₁,₂={gamma1}, c={c}, T={T} '
         f'N_periods={num_periods}_N_clamp_samples={P1_samples_target.shape[0]} '
         f'N_steps={n_steps}learning_rate={learning_rate}')
plt.suptitle(title, y=1.02, size=10)

# Adjust layout and save
plt.tight_layout()
filename = os.path.join(output_dir, f"BMlearning_{params_str}.png")
plt.savefig(filename, bbox_inches='tight', dpi=300)
plt.show()
