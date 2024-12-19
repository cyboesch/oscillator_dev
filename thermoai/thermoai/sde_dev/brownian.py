"""This module contains a class that is identical to the VirtualBrownianTree class in diffrax as of version 0.6.0, but only for BrownianIncrement. 

The '.evaluate' method of this class has a new parameter 'rng_only'. If True, it returns the rng used to generate the Brownian increments. Otherwise, it returns the Brownian increments themselves, as usual.
"""
from typing import Optional, Union

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
from diffrax._misc import linear_rescale
from diffrax import VirtualBrownianTree
from diffrax._brownian.tree import _State, _LevyVal, _split_interval, FloatTriple, PRNGKeyArray, FloatDouble

from diffrax._custom_types import RealScalarLike

from typing import Optional

from diffrax import  VirtualBrownianTree
from diffrax._misc import linear_rescale
from diffrax._brownian.tree import _levy_diff, _State, _LevyVal, _split_interval, FloatTriple, PRNGKeyArray, FloatDouble 

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
                rng = self._evaluate(t1, rng_only=True)
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
        
        if rng_only:
            return final_state.key

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

