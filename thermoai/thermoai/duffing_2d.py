import jax.numpy as jnp
from jax import vmap
def _validate_input(positions: jnp.ndarray, params: dict) -> bool:
    """
    Validates that oscillator network parameters have consistent shapes
    
    Args:
        params: Dictionary containing network parameters
        
    Returns:
        True if valid, raises ValueError with description if invalid
    """
    # Extract parameters
    k_lin = params.get('k_lin')
    k_duff = params.get('k_duff') 
    c_lin = params.get('c_lin')
    c_optomech = params.get('c_optomech')
    connectivity = params.get('connectivity')
    k_b = params.get('k_b', 1.)
    T = params.get('T', 100.)
    
    # Check all required params exist
    required = ['k_lin', 'k_duff', 'c_lin', 'c_optomech', 'connectivity']
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
        
    # Get number of oscillators from k_lin
    n_oscillators = len(k_lin)
    
    # Check shapes match number of oscillators
    if len(k_duff) != n_oscillators:
        raise ValueError(f"k_duff length {len(k_duff)} must match k_lin length {n_oscillators}")
        
    # Check coupling terms
    n_connections = len(connectivity)
    if len(c_lin) != n_connections:
        raise ValueError(f"c_lin length {len(c_lin)} must match number of connections {n_connections}")
    if len(c_optomech) != n_connections:
        raise ValueError(f"c_optomech length {len(c_optomech)} must match number of connections {n_connections}")
        
    # Check connectivity matrix shape and values
    if connectivity.shape[1] != 2:
        raise ValueError(f"Connectivity must be Nx2 matrix, got shape {connectivity.shape}")
    if jnp.any(connectivity >= n_oscillators):
        raise ValueError(f"Connectivity indices must be < {n_oscillators}")
      
    # Check positions shape
    if positions.shape != (n_oscillators,):
        raise ValueError(f"Positions must be array of length {n_oscillators}, got shape {positions.shape}")
        
    return True

def potential_energy(
    x: jnp.ndarray,                    # Positions of oscillators
    params: dict = {
        'k_lin': jnp.array([1., 1., 1.]),  # Linear spring terms
        'k_duff': jnp.array([1., 1., 1.]), # Nonlinear (Duffing) terms
        'c_lin': jnp.zeros(2),             # Linear coupling between oscillators
        'c_optomech': jnp.array([1., 1.]), # Optomechanical coupling
        'connectivity': jnp.array([[0, 1], [1, 2]]), # Which oscillators are connected
        'k_b': 1.,                         # Boltzmann constant
        'T': 100.                          # Temperature
    }
) -> float:
    """
    Computes potential energy of coupled oscillator network
    
    Args:
        x: Position array of shape (n_oscillators,)
        params: Dictionary of network parameters
        
    Returns:
        Potential energy value
    """
    _validate_input(x, params)

    # Extract parameters
    k_lin = params['k_lin']
    k_duff = params['k_duff']
    c_lin = params['c_lin']
    c_optomech = params['c_optomech']
    connectivity = params['connectivity']
    k_b = params['k_b']
    T = params['T']
    
    # Single oscillator terms (linear + duffing)
    single_osc_energy = jnp.sum(0.5 * k_lin * x**2 + 0.25 * k_duff * x**4)
    
    # Coupling terms between connected oscillators
    coupling_energy = 0.
    for i, (idx1, idx2) in enumerate(connectivity):
        dx = x[idx1] - x[idx2]
        coupling_energy += 0.5 * c_lin[i] * dx**2
        coupling_energy += 0.5 * c_optomech[i] * dx**2
        
    total_energy = single_osc_energy + coupling_energy
    return total_energy

def probability_density(x: jnp.ndarray, params: dict) -> float:
    """
    Computes Boltzmann probability density at position x
    
    Args:
        x: Position array of shape (n_oscillators,)
        params: Dictionary of network parameters
        
    Returns:
        Probability density value (unnormalized)
    """
    energy = potential_energy(x, params)
    k_b = params.get('k_b', 1.)
    T = params.get('T', 100.)
    return jnp.exp(-energy / (k_b * T))