#%%
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from diffrax import (AbstractBrownianPath, AbstractStepSizeController,
                     ConstantStepSize)
from diffrax._custom_types import (VF, Args, BoolScalarLike, IntScalarLike,
                                   RealScalarLike, Y)
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm
from jaxtyping import PyTree
from mala import accept_or_reject

_MhState = tuple[RealScalarLike, RealScalarLike, RealScalarLike]
   

class MetropolisHastingsController(AbstractStepSizeController[tuple, Optional[RealScalarLike]]):
    target_logdensity_fn: Callable
    # bm: AbstractBrownianPath
    init_rng: jax.random.PRNGKey
    base_controller: AbstractStepSizeController = ConstantStepSize()
    
    def wrap(self, direction: IntScalarLike):
        return self

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: RealScalarLike,
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, _MhState]:  # TODO: not necessary scalar grad
        del terms, t1, func, error_order
        y0_val, y0_grad = jax.value_and_grad(self.target_logdensity_fn)(y0, args)
        return t0+dt0, (dt0, y0_val, y0_grad, self.init_rng)
    
    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,  # x_n
        y1_candidate: Y,  # \hat{x}_n
        args: Args,
        y_error: Optional[Y],
        error_order: RealScalarLike,
        controller_state: tuple,
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        _MhState,
        RESULTS,
    ]:
        del y_error, error_order
        
        # TODO: is this correct for a virtual brownian tree? Overkill? Just split the key?
        dt0, cur_val, cur_grad, prev_rng = controller_state
        cur_rng, accept_rng = jax.random.split(prev_rng)
        
        # Compute acceptance probability
        
        cur_state = (y0, cur_val, cur_grad)
        prop_state = (y1_candidate, *jax.value_and_grad(self.target_logdensity_fn)(y1_candidate, args))
        
        (next_y0, accepted_val, accepted_grad), (_,_, is_accepted) = accept_or_reject(accept_rng, prop_state, cur_state, dt0)
        jax.debug.print("is_accepted {x}", x=is_accepted)
        
        next_t0 = jnp.where(is_accepted, t1, t0)
        next_t1 = jnp.where(is_accepted, t1+dt0, t0+dt0)
        
        next_controller_state = (dt0, accepted_val, accepted_grad, cur_rng)
        
        return is_accepted, next_t0, next_t1, False, next_controller_state, RESULTS.successful
    


"""def compute_log_rho(prop_state, cur_state, step_size):
    prop_x, prop_x_unn_target_logdensity_val, prop_grad = prop_state
    cur_x, cur_x_unn_target_logdensity_val, cur_grad = cur_state

    cur_x_g_prop_x_proposal_logdensity_val = langevin.logpdf(
        cur_x,
        _langevin_mean(prop_x, prop_grad, step_size),
        _langevin_std_dev(step_size),
    )

    new_energy = -(
        cur_x_g_prop_x_proposal_logdensity_val + prop_x_unn_target_logdensity_val
    )

    prop_x_g_cur_x_proposal_logdensity_val = langevin.logpdf(
        prop_x, _langevin_mean(cur_x, cur_grad, step_size), _langevin_std_dev(step_size)
    )

    prev_energy = -(
        prop_x_g_cur_x_proposal_logdensity_val + cur_x_unn_target_logdensity_val
    )

    # Compute log accept ratio
    log_accept_ratio = prev_energy - new_energy
    
    # From BlackJAX: safe_energy_diff
    log_accept_ratio = jnp.where(jnp.isnan(log_accept_ratio), -jnp.inf, log_accept_ratio)
    return log_accept_ratio
"""