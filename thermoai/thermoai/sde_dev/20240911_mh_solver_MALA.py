# %%
from collections.abc import Callable
from typing import Optional

import jax


import jax.numpy as jnp
import jax.random as jr


from jaxtyping import PyTree, PRNGKeyArray

from diffrax._custom_types import Args, BoolScalarLike, RealScalarLike, VF, Y, DenseInfo
from diffrax._heuristics import is_sde
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm
from diffrax._solver.base import AbstractSolver, _SolverState, AbstractWrappedSolver

from mala import compute_log_rho


class AbstractMetropolisAdjustedSolver(AbstractSolver[_SolverState]):
    """Indicates that this solver provides log density and score estimates, and that as such it may be used with an adaptive step size controller for Metropolis-Hastings MCMC sampling."""

    logdensity_fn: Callable[[Y], RealScalarLike]


class MetropolisAdjustedSolver(
    AbstractMetropolisAdjustedSolver[_SolverState], AbstractWrappedSolver[_SolverState]
):
    """Wraps another solver and applies a Metropolis adjustment step."""

    logdensity_fn: Callable[[Y], RealScalarLike]
    solver: AbstractSolver[_SolverState]

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        return self.solver.strong_order(terms)

    def error_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        if is_sde(terms):
            order = self.strong_order(terms)
            if order is not None:
                order = order + 0.5
        else:
            order = self.order(terms)
            if order is not None:
                order = order + 1
        return order

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        key: PRNGKeyArray,
    ) -> tuple[_SolverState, PRNGKeyArray]:
        return (self.solver.init(terms, t0, t1, y0, args), key)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        mh_solver_state: tuple[_SolverState, PRNGKeyArray],
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, tuple[_SolverState, PRNGKeyArray], RESULTS]:

        solver_state, step_key = mh_solver_state

        # Step with the wrapped solver
        y1, y_error, dense_info, new_solver_state, result = self.solver.step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )

        # Compute log acceptance ratio
        cur_state = (y0, *jax.value_and_grad(self.logdensity_fn)(y0, args))
        prop_state = (y1, *jax.value_and_grad(self.logdensity_fn)(y1, args))
        log_accept_ratio = compute_log_rho(prop_state, cur_state, t1 - t0)

        # Metropolis-Hastings acceptance step
        next_step_key, accept_key = jr.split(step_key)
        u = jr.uniform(accept_key)
        do_accept = jnp.log(u) < log_accept_ratio

        # Update y1 and error estimate based on acceptance
        y1_accepted = jnp.where(do_accept, y1, y0)

        # Update y1 in dense_info without changing other keys
        dense_info["y1"] = y1_accepted

        if y_error is not None:
            raise NotImplementedError("Only Euler estimate is supported")

        return y1_accepted, y_error, dense_info, (new_solver_state, next_step_key), result

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)
