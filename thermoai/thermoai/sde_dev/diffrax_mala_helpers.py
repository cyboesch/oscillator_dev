# %%
import math
from typing import cast, Optional, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, PyTree
from lineax.internal import complex_to_real_dtype

from diffrax._custom_types import (
    AbstractBrownianIncrement,
    BrownianIncrement,
    levy_tree_transpose,
    RealScalarLike,
    SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea,
)
from diffrax._misc import (
    force_bitcast_convert_type,
    split_by_tree,
)
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree

from malareparam.mala import accept_or_reject

from diffrax import (
    AbstractBrownianPath,
    AbstractStepSizeController,
    ConstantStepSize,
    ControlTerm,
    MultiTerm,
    ODETerm,
)
from diffrax._custom_types import (
    Args,
    BoolScalarLike,
    IntScalarLike,
    RealScalarLike,
    VF,
    Y,
)
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm

from typing import Optional, Callable

from diffrax import UnsafeBrownianPath, VirtualBrownianTree
from diffrax._misc import linear_rescale
from diffrax._brownian.tree import (
    _make_levy_val,
    _levy_diff,
    _State,
    _LevyVal,
    _split_interval,
    FloatTriple,
    PRNGKeyArray,
    FloatDouble,
)


# %%
class SampleSDE(eqx.Module):
    H: callable
    D: callable
    Q: callable
    state_dim: int

    def __init__(self, H, D, Q, state_dim):
        self.H = H  # Target logdensity function H(z) = log \tilde{\pi}(z, args)
        self.D = D  # Positive semidefinite diffusion matrix
        self.Q = Q  # Skew-symmetric curl matrix
        self.state_dim = state_dim

    def drift(self, t, z, args):
        grad_H = jax.grad(self.H)(z, args)
        DQ = jnp.add(self.D(z), self.Q(z))
        f = -jnp.dot(DQ, grad_H) + self.Gamma(z)
        return f

    def Gamma(self, z: Y, args: Args):
        def partial_sum(i):
            return jnp.sum(jax.jacfwd(lambda z: self.D(z)[i, :] + self.Q(z)[i, :])(z))

        return jax.vmap(partial_sum)(jnp.arange(z.shape[0]))

    def diffusion(self, t, z, args):
        return jnp.sqrt(2 * self.D(z))

    def get_terms(self, brownian_path):
        return MultiTerm(
            ODETerm(self.drift), ControlTerm(self.diffusion, brownian_path)
        )


def make_sampler_SDE(
    target_logdensity_fn,
    state_dim,
    D=None,
    Q=None,
):
    potential_fn = lambda x, args: -1.0 * target_logdensity_fn(x)
    if D is None:
        D = lambda _: jnp.eye(state_dim)
    if Q is None:
        Q = lambda _: jnp.zeros((state_dim, state_dim))
    return SampleSDE(potential_fn, D, Q, state_dim)


class AuditableVirtualBrownianTree(VirtualBrownianTree):

    def _brownian_arch(self, _state: _State, shape, dtype) -> tuple[
        RealScalarLike,
        FloatTriple,
        FloatDouble,
        tuple[PRNGKeyArray, PRNGKeyArray],
        Optional[FloatTriple],
        Optional[FloatDouble],
        Optional[FloatTriple],
        Optional[FloatDouble],
    ]:
        r"""For `t = (s+u)/2` evaluates `w_t` and (optionally) `bhh_t`
         conditioned on `w_s`, `w_u`, `bhh_s`, `bhh_u`, where
         `bhh_st` represents $\bar{H}_{s,t} \coloneqq (t-s) H_{s,t}$.
         To avoid cancellation errors, requires an input of `w_su`, `bhh_su`
         and also returns `w_st` and `w_tu` in addition to just `w_t`. Same for `bhh`
         if it is not None.
         Note that the inputs and outputs already contain `bkk`. These values are
         there for the sake of a future extension with "space-time-time" Lévy area
         and should be None for now.

        **Arguments:**

        - `_state`: The state of the Brownian tree
        - `shape`:
        - `dtype`:

        **Returns:**

        - `t`: midpoint time
        - `w_stu`: $(W_s, W_t, W_u)$
        - `w_st_tu`: $(W_{s,t}, W_{t,u})$
        - `keys`: a tuple of subinterval keys `(key_st, key_tu)`
        - `bhh_stu`: (optional) $(\bar{H}_s, \bar{H}_t, \bar{H}_u)$
        - `bhh_st_tu`: (optional) $(\bar{H}_{s,t}, \bar{H}_{t,u})$
        - `bkk_stu`: (optional) $(\bar{K}_s, \bar{K}_t, \bar{K}_u)$
        - `bkk_st_tu`: (optional) $(\bar{K}_{s,t}, \bar{K}_{t,u})$

        """
        key_st, midpoint_key, key_tu = jr.split(_state.key, 3)
        keys = (key_st, key_tu)
        tdtype = complex_to_real_dtype(dtype)
        su = jnp.asarray(2.0**-_state.level, dtype=tdtype)
        st = su / 2
        s = jnp.asarray(_state.s, dtype=tdtype)
        t = s + st
        root_su = jnp.sqrt(su)

        w_s, w_u, w_su = _state.w_s_u_su

        with jax.numpy_dtype_promotion("standard"):
            if self.levy_area is SpaceTimeTimeLevyArea:
                raise NotImplementedError(
                    "Auditable SpaceTimeTimeLevyArea not implemented"
                )

            elif self.levy_area is SpaceTimeLevyArea:
                raise NotImplementedError("Auditable SpaceTimeLevyArea not implemented")

            elif self.levy_area is BrownianIncrement:
                assert _state.bhh_s_u_su is None
                assert _state.bkk_s_u_su is None
                mean = 0.5 * w_su
                w_term2 = (root_su / 2) * jr.normal(midpoint_key, shape, dtype)
                w_st = mean + w_term2
                w_tu = mean - w_term2
                w_st_tu = (w_st, w_tu)
                w_t = w_s + w_st
                w_stu = (w_s, w_t, w_u)
                bhh_stu, bhh_st_tu, bkk_stu, bkk_st_tu = None, None, None, None

            else:
                assert False

        return t, w_stu, w_st_tu, keys, bhh_stu, bhh_st_tu, bkk_stu, bkk_st_tu

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
        rng_only: bool = False,  # Added to allow for blackjax tests with same rng
    ) -> Union[PyTree[Array], AbstractBrownianIncrement]:
        t0 = eqxi.nondifferentiable(t0, name="t0")
        # map the interval [self.t0, self.t1] onto [0,1]
        t0 = linear_rescale(self.t0, t0, self.t1)
        levy_0 = self._evaluate(t0)
        if t1 is None:
            raise NotImplementedError("t1 cannot be None")
        else:
            t1 = eqxi.nondifferentiable(t1, name="t1")
            # map the interval [self.t0, self.t1] onto [0,1]
            t1 = linear_rescale(self.t0, t1, self.t1)

            if rng_only:
                rng = self._evaluate(t1, rng_only)
                return rng
            else:
                levy_1 = self._evaluate(t1)
            # take the difference between the output for t0 and t1 via Chen's relation
            levy_out = jtu.tree_map(_levy_diff, self.shape, levy_0, levy_1)

        levy_out = levy_tree_transpose(self.shape, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, self.levy_area)
        return levy_out if use_levy else levy_out.W

    def _evaluate_leaf(
        self,
        key,
        r: RealScalarLike,
        struct: jax.ShapeDtypeStruct,
        rng_only: bool = False,
    ) -> Union[PyTree[Array], PRNGKeyArray]:
        shape, dtype = struct.shape, struct.dtype
        tdtype = complex_to_real_dtype(dtype)
        t0 = jnp.zeros((), tdtype)
        r = jnp.asarray(r, tdtype)

        if self.levy_area is SpaceTimeTimeLevyArea:
            raise NotImplementedError("Auditable SpaceTimeTimeLevyArea not implemented")

        elif self.levy_area is SpaceTimeLevyArea:
            raise NotImplementedError("Auditable SpaceTimeLevyArea not implemented")
        elif self.levy_area is BrownianIncrement:
            state_key, init_key_w = jr.split(key, 2)
            # HACK: return the init_key_w rng if rng_only is True
            if rng_only:
                return init_key_w
            bhh = None
            bkk = None

        else:
            assert False

        w_0 = jnp.zeros(shape, dtype)
        w_1 = jr.normal(init_key_w, shape, dtype)
        w = (w_0, w_1, w_1)

        init_state = _State(
            level=0, s=t0, w_s_u_su=w, key=state_key, bhh_s_u_su=bhh, bkk_s_u_su=bkk
        )

        def _cond_fun(_state):
            """Condition for the binary search for r."""
            # If true, continue splitting the interval and descending the tree.
            return 2.0 ** (-_state.level) > self.tol

        def _body_fun(_state: _State):
            """Single-step of the binary search for r."""

            (
                _t,
                _w_stu,
                _w_st_tu,
                _keys,
                _bhh_stu,
                _bhh_st_tu,
                _bkk_stu,
                _bkk_st_tu,
            ) = self._brownian_arch(_state, shape, dtype)

            _level = _state.level + 1
            _cond = r > _t
            _s = jnp.where(_cond, _t, _state.s)
            _key_st, _key_tu = _keys
            _key = jnp.where(_cond, _key_st, _key_tu)

            _w = _split_interval(_cond, _w_stu, _w_st_tu)
            assert _w is not None
            _bhh = _split_interval(_cond, _bhh_stu, _bhh_st_tu)
            _bkk = _split_interval(_cond, _bkk_stu, _bkk_st_tu)

            return _State(
                level=_level,
                s=_s,
                w_s_u_su=_w,
                key=_key,
                bhh_s_u_su=_bhh,
                bkk_s_u_su=_bkk,
            )

        final_state = jax.lax.while_loop(_cond_fun, _body_fun, init_state)

        s = final_state.s
        su = jnp.asarray(2.0**-final_state.level, dtype=tdtype)

        sr = jax.nn.relu(r - s)
        # make sure su = sr + ru regardless of cancellation error
        ru = jax.nn.relu(su - sr)

        w_s, _, w_su = final_state.w_s_u_su

        if self.levy_area is SpaceTimeTimeLevyArea:
            raise NotImplementedError("Auditable SpaceTimeTimeLevyArea not implemented")

        elif self.levy_area is SpaceTimeLevyArea:
            raise NotImplementedError("Auditable SpaceTimeLevyArea not implemented")

        elif self.levy_area is BrownianIncrement:
            with jax.numpy_dtype_promotion("standard"):
                w_mean = w_s + sr / su * w_su
                if self._spline == "sqrt":
                    z = jr.normal(final_state.key, shape, dtype)
                    bb = jnp.sqrt(sr * ru / su) * z
                elif self._spline == "quad":
                    z = jr.normal(final_state.key, shape, dtype)
                    bb = (sr * ru / su) * z
                elif self._spline == "zero":
                    bb = jnp.zeros(shape, dtype)
                else:
                    assert False
            w_r = w_mean + bb
            return _LevyVal(dt=r, W=w_r, H=None, bar_H=None, K=None, bar_K=None)

        else:
            assert False

    def _evaluate(
        self, r: RealScalarLike, rng_only: bool = False
    ) -> Union[PyTree[Array], PRNGKeyArray]:
        """Maps the _evaluate_leaf function at time r using self.key onto self.shape"""
        r = eqxi.error_if(
            r,
            (r < 0) | (r > 1),
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        map_func = lambda key, shape: self._evaluate_leaf(key, r, shape, rng_only)
        return jtu.tree_map(map_func, self.key, self.shape)


class AuditableUnsafeBrownianPath(UnsafeBrownianPath):
    """This is a modified version of diffrax.UnsafeBrownianPath that handles
    rngs identically to blackjax in the Metropolis-adjusted Langevin algorithm (MALA).
    """

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
        rng_only: bool = False,  # Added to allow for blackjax tests with same rng
    ) -> Union[PyTree[Array], AbstractBrownianIncrement]:
        del left
        if t1 is None:
            dtype = jnp.result_type(t0)
            t1 = t0
            t0 = jnp.array(0, dtype)
        else:
            with jax.numpy_dtype_promotion("standard"):
                dtype = jnp.result_type(t0, t1)
            t0 = jnp.astype(t0, dtype)
            t1 = jnp.astype(t1, dtype)
        t0 = eqxi.nondifferentiable(t0, name="t0")
        t1 = eqxi.nondifferentiable(t1, name="t1")
        t1 = cast(RealScalarLike, t1)
        t0_ = force_bitcast_convert_type(t0, jnp.int32)
        t1_ = force_bitcast_convert_type(t1, jnp.int32)
        key = jr.fold_in(self.key, t0_)
        key = jr.fold_in(key, t1_)
        key = split_by_tree(key, self.shape)

        ########################
        # Use this to drop out the correct rng for blackjax in tests
        ########################
        if rng_only:
            return key
        else:
            out = jtu.tree_map(
                lambda key, shape: self._evaluate_leaf(
                    t0, t1, key, shape, self.levy_area, use_levy
                ),
                key,
                self.shape,
            )
            if use_levy:
                out = levy_tree_transpose(self.shape, out)
                assert isinstance(out, self.levy_area)
            return out

    @staticmethod
    def _evaluate_leaf(
        t0: RealScalarLike,
        t1: RealScalarLike,
        key,
        shape: jax.ShapeDtypeStruct,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
        ],
        use_levy: bool,
    ):
        w_std = jnp.sqrt(t1 - t0).astype(shape.dtype)
        dt = jnp.asarray(t1 - t0, dtype=complex_to_real_dtype(shape.dtype))

        ########################
        # The next line matches the diffrax noise sample to blackjax
        ########################
        key, _ = jr.split(key, 2)

        if levy_area is SpaceTimeTimeLevyArea:
            key_w, key_hh, key_kk = jr.split(key, 3)
            w = jr.normal(key_w, shape.shape, shape.dtype) * w_std
            hh_std = w_std / math.sqrt(12)
            hh = jr.normal(key_hh, shape.shape, shape.dtype) * hh_std
            kk_std = w_std / math.sqrt(720)
            kk = jr.normal(key_kk, shape.shape, shape.dtype) * kk_std
            levy_val = SpaceTimeTimeLevyArea(dt=dt, W=w, H=hh, K=kk)

        elif levy_area is SpaceTimeLevyArea:
            key_w, key_hh = jr.split(key, 2)
            w = jr.normal(key_w, shape.shape, shape.dtype) * w_std
            hh_std = w_std / math.sqrt(12)
            hh = jr.normal(key_hh, shape.shape, shape.dtype) * hh_std
            levy_val = SpaceTimeLevyArea(dt=dt, W=w, H=hh)
        elif levy_area is BrownianIncrement:
            w = jr.normal(key, shape.shape, shape.dtype) * w_std
            levy_val = BrownianIncrement(dt=dt, W=w)
        else:
            assert False

        if use_levy:
            return levy_val
        return w


@eqx.filter_jit
class BlackjaxEquivalentMetropolisHastingsController(
    AbstractStepSizeController[tuple, Optional[RealScalarLike]]
):
    target_logdensity_fn: Callable
    proposal_logdensity_fn: Callable
    bm: AbstractBrownianPath
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
    ) -> tuple[RealScalarLike, tuple[RealScalarLike, RealScalarLike, RealScalarLike]]:
        del terms, t1, args, func, error_order
        # Compute the initial gradient for MALA accept-reject step
        y0_val, y0_grad = jax.value_and_grad(self.target_logdensity_fn)(y0)
        return t0 + dt0, (dt0, y0_val, y0_grad)

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,  # x_n
        y1_candidate: Y,  # proposal sample
        args: Args,
        y_error: Optional[Y],
        error_order: RealScalarLike,
        controller_state: tuple,
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        tuple,
        RESULTS,
    ]:
        del args, y_error, error_order

        #########################################################
        # The call to `split` ensures that the diffrax acceptance rng matches the blackjax MALA acceptance rng
        # It isn't necessary in general, but it is necessary for the tests to run correctly
        #########################################################
        _, key_rmh = jax.random.split(self.bm.evaluate(t0, t1, rng_only=True))

        # Compute acceptance probability
        dt0, cur_val, cur_grad = controller_state

        cur_state = (y0, cur_val, cur_grad)
        prop_state = (
            y1_candidate,
            *jax.value_and_grad(self.target_logdensity_fn)(y1_candidate),
        )

        (_, next_val, next_grad), (_, _, is_accepted) = accept_or_reject(
            key_rmh, prop_state, cur_state, dt0
        )

        next_t0 = jnp.where(is_accepted, t1, t0)
        next_t1 = jnp.where(is_accepted, t1 + dt0, t0 + dt0)

        next_controller_state = (dt0, next_val, next_grad)

        return (
            is_accepted,
            next_t0,
            next_t1,
            False,
            next_controller_state,
            RESULTS.successful,
        )


class TargetSDE(eqx.Module):
    H: callable
    D: callable
    Q: callable
    state_dim: int

    def __init__(self, H, D, Q, state_dim):
        self.H = H  # Target logdensity function H(z) = log \tilde{\pi}(z, args)
        self.D = D  # Any positive semidefinite diffusion matrix
        self.Q = Q  # Any skew-symmetric curl matrix
        self.state_dim = state_dim

    def drift(self, t, z, args):
        grad_H = jax.grad(self.H)(z, args)
        DQ = jnp.add(self.D(z), self.Q(z))
        f = -jnp.dot(DQ, grad_H) + self.Gamma(z)
        return f

    def Gamma(self, z):
        def partial_sum(i):
            return jnp.sum(jax.jacfwd(lambda z: self.D(z)[i, :] + self.Q(z)[i, :])(z))

        return jax.vmap(partial_sum)(jnp.arange(z.shape[0]))

    def diffusion(self, t, z, args):
        return jnp.sqrt(2 * self.D(z))

    def get_terms(self, brownian_path):
        return MultiTerm(
            ODETerm(self.drift), ControlTerm(self.diffusion, brownian_path)
        )


def make_sampler_SDE(
    target_logdensity_fn,
    state_dim,
    D=None,
    Q=None,
):
    potential_fn = lambda x, args: -1.0 * target_logdensity_fn(x)
    if D is None:
        D = lambda _: jnp.eye(state_dim)
    if Q is None:
        Q = lambda _: jnp.zeros((state_dim, state_dim))
    return TargetSDE(potential_fn, D, Q, state_dim)
