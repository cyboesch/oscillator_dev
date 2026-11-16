#!/usr/bin/env python3
"""Temporary check of PatchGeometry impl 'blocks' against 'slices' (2026-11-16; deleted after use).
Per layout: unfold and fold exact equality; patch_terms (grad, lap) and jax.grad of a score-matching-like loss with
respect to Wp, Qp, bp and X, both impls, float32 and float64: max |difference| and bitwise equality; then the time of
one jitted loss-and-gradient call (median of 20) for each impl."""
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import latent_model as lm  # noqa: E402

LAYOUTS = [(32, 8, 2), (32, 4, 2), (32, 16, 4), (64, 16, 4), (64, 8, 2), (28, 7, 3)]
S, B, T = 32, 256, lm.TEMP
fails = 0


def geom(G, s, r, impl):
    g = lm.PatchGeometry(G, s, r)
    if impl == "slices":
        g.impl = "slices"
    return g


def terms(g, Wp, Qp, bp, X):
    """patch_terms with an explicit geometry (the module's cached patch_geometry picks the impl from the env)."""
    Xp = g.unfold(X)
    A = lm._pdot(Xp, Wp) + bp[:, None, :] + lm._pdot(Xp * Xp, Qp)
    p = jax.nn.softmax(A / T, axis=-1)
    gbar = lm._pdot_t(p, Wp) + 2.0 * Xp * lm._pdot_t(p, Qp)
    grad = -g.fold(gbar)
    GG = jnp.sum(Wp * Wp, axis=-1)[:, None, :] + 4.0 * lm._pdot(Xp, Wp * Qp) + 4.0 * lm._pdot(Xp * Xp, Qp * Qp)
    lap = -(jnp.sum(p * GG, axis=(0, 2)) - jnp.sum(gbar * gbar, axis=(0, 2))) / T
    lap = lap - 2.0 * jnp.sum(p * jnp.sum(Qp, axis=-1)[:, None, :], axis=(0, 2))
    return grad, lap


def loss(params, X, g):
    grad, lap = terms(g, params["Wp"], params["Qp"], params["bp"], X)
    return jnp.mean(-lap / T + jnp.sum(grad * grad, axis=1) / (2.0 * T * T))


for G, s, r in LAYOUTS:
    gs, gb = geom(G, s, r, "slices"), geom(G, s, r, "blocks")
    print(f"== {G}x{G}, patch {s}:{r}: P {gs.P}, d {gs.d}; impl chosen {gb.impl}", flush=True)
    rng = np.random.default_rng(0)
    for dt in (jnp.float32, jnp.float64):
        X = jnp.asarray(rng.standard_normal((B, G * G)), dt)
        Gp = jnp.asarray(rng.standard_normal((gs.P, B, gs.d)), dt)
        params = {"Wp": jnp.asarray(0.1 * rng.standard_normal((gs.P, S, gs.d)), dt),
                  "Qp": jnp.asarray(0.01 * rng.standard_normal((gs.P, S, gs.d)), dt),
                  "bp": jnp.asarray(0.1 * rng.standard_normal((gs.P, S)), dt)}
        u_eq = bool(jnp.array_equal(jax.jit(gs.unfold)(X), jax.jit(gb.unfold)(X)))
        f_s, f_b = jax.jit(gs.fold)(Gp), jax.jit(gb.fold)(Gp)
        f_eq, f_d = bool(jnp.array_equal(f_s, f_b)), float(jnp.max(jnp.abs(f_s - f_b)))
        # adjoint check of the block version: <unfold X, Gp> = <X, fold Gp>
        adj = float(jnp.sum(gb.unfold(X) * Gp) - jnp.sum(X * f_b)) / float(jnp.sum(jnp.abs(X * f_b)))
        vg_s = jax.jit(jax.value_and_grad(lambda p, x: loss(p, x, gs), argnums=(0, 1)))
        vg_b = jax.jit(jax.value_and_grad(lambda p, x: loss(p, x, gb), argnums=(0, 1)))
        (ls, (gps, gxs)), (lb, (gpb, gxb)) = vg_s(params, X), vg_b(params, X)
        rows = [("loss", ls, lb), ("dX", gxs, gxb)] + [(f"d{k}", gps[k], gpb[k]) for k in ("Wp", "Qp", "bp")]
        msg = []
        for name, a, b in rows:
            eq = bool(jnp.array_equal(a, b))
            rel = float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-300))
            msg.append(f"{name} {'bitwise' if eq else f'rel {rel:.1e}'}")
            tol = 1e-5 if dt == jnp.float32 else 1e-12
            if not eq and rel > tol:
                fails += 1
        if not u_eq:
            fails += 1
        print(f"  {jnp.dtype(dt).name}: unfold {'bitwise' if u_eq else 'DIFFERS'}; fold "
              f"{'bitwise' if f_eq else f'max diff {f_d:.1e}'}; adjoint rel {adj:.1e}; " + ", ".join(msg), flush=True)
        if dt == jnp.float32:                      # timing, float32 (the training precision of the products)
            for name, f in (("slices", vg_s), ("blocks", vg_b)):
                jax.block_until_ready(f(params, X))
                ts = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    jax.block_until_ready(f(params, X))
                    ts.append(time.perf_counter() - t0)
                print(f"  time {name}: {1e3 * np.median(ts):.2f} ms per loss-and-gradient (B {B})", flush=True)
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})")
