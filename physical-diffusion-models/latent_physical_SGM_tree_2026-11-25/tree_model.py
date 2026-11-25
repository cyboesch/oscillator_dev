"""The hierarchical ("tree") hidden system for the latent physical SGM (2026-11-25; after the meeting with Dmitry
Krotov, his Hierarchical Associative Memory, arXiv 2107.06446, Section 4.3).

Today's model: the global Potts unit and the per-patch Potts units are each coupled to the pixels only; given the
pixels they are independent. Here they are also coupled to each other by a table J per patch set,

    hidden energy  -E(x, g, s) = a_g(x) + sum_p [ a_{p s_p}(x) + J_p[g, s_p] ],

    a_g  = b_g + W_g . x + Q_g . x^2                       (global unit, K states; as hidden_terms_quad 'potts')
    a_ps = bp_ps + Wp_ps . x_p + Qp_ps . x_p^2             (patch unit p, S states; as patch_terms)
    J_p[g, s]: the energy bonus of state s of patch p when the global unit is in state g (P, K, S).

Given g the patch units are independent, so every hidden unit is still summed out exactly (a tree):

    F_tree(x) = -T logsumexp_g [ a_g / T + sum_p logsumexp_s ((a_ps + J_p[g, s]) / T) ].

J = 0 is today's model (global unit + patch units). In Krotov's network the top layer has no pixel couplings (W = Q =
0); that is the same formula. This is the thermal, exactly summed counterpart of his deterministic network (whose
energy is the naive mean-field free energy of this Potts system).

With pi_g = p(g | x), p(s | g) = softmax_s((a_ps + J_p[g, s]) / T), q_ps = sum_g pi_g p(s | g),
G_g = W_g + 2 Q_g * x, G_ps = Wp_ps + 2 Qp_ps * x_p, gbar_p(g) = sum_s p(s | g) G_ps and
m_g = G_g + sum over sets fold(gbar(g)) (the mean hidden force given g):

    grad F = -sum_g pi_g m_g
    lap F  = -[ sum_g pi_g 2 sum_i Q_gi + sum_p sum_s q_ps 2 sum_j Qp_psj ]
             -(1 / T) [ sum_p sum_s q_ps |G_ps|^2 - sum_g pi_g sum_p |gbar_p(g)|^2 + sum_g pi_g |m_g - sum_g' pi_g' m_g'|^2 ]

(the bracket is the variance of the instantaneous hidden force: within g, the patches are independent; between g).

This module only adds functions; latent_model.py of the locked code is imported, not changed. Per sample the tree
touches K P S posteriors and K P d mean forces, so it is written for a small global unit (K16: no chunking).
"""
import os
import sys

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(os.path.dirname(HERE), "latent_physical_SGM")
if MAIN not in sys.path:
    sys.path.insert(0, MAIN)
import latent_model as lm  # noqa: E402

J_KEYS = ("J", "J2", "J3")           # the coupling table of patch set i (lm.PATCH_SETS order): (P, K, S)


def _patch_fields(Wp, Qp, bp, Xp):
    A = lm._pdot(Xp, Wp) + bp[:, None, :]
    if Qp is not None:
        A = A + lm._pdot(Xp * Xp, Qp)
    return A                                                             # (P, B, S)


def tree_sets(params):
    """[(Wp, Qp, bp, J)] of the patch sets present in params, in lm.PATCH_SETS order; a set without its J key gets
    J = 0 (exactly today's independent patch units inside the tree sum)."""
    out = []
    for (kw, kb, kq), kj in zip(lm.PATCH_SETS, J_KEYS):
        if kw in params:
            Wp = params[kw]
            J = params[kj] if kj in params else jnp.zeros((Wp.shape[0], params["W"].shape[0], Wp.shape[1]), Wp.dtype)
            out.append((Wp, params.get(kq), params[kb], J))
    return out


def tree_free_energy(W, Q, b, sets, X, T):
    """F_tree(x) per sample (B,). W (K, N), Q (K, N) or None, b (K,); sets: [(Wp, Qp, bp, J)]."""
    Phi = (X @ W.T + b) / T
    if Q is not None:
        Phi = Phi + ((X * X) @ Q.T) / T
    for Wp, Qp, bp, J in sets:
        geom = lm.patch_geometry(X.shape[1], Wp.shape[0], Wp.shape[2])
        A = _patch_fields(Wp, Qp, bp, geom.unfold(X))
        Phi = Phi + jnp.sum(logsumexp((A[:, :, None, :] + J[:, None, :, :]) / T, axis=-1), axis=0)
    return -T * logsumexp(Phi, axis=1)


def tree_terms(W, Q, b, sets, X, T, with_lap=True):
    """Contributions of the tree (global Potts unit + coupled patch units) to grad F and the Laplacian of F.
    Returns (grad (B, N), lap (B,) or None, pi (B, K), [q (P, B, S) per set], [p(s | g) (P, B, K, S) per set])."""
    B, N = X.shape
    K = W.shape[0]
    X2 = X * X
    Phi = (X @ W.T + b) / T
    if Q is not None:
        Phi = Phi + (X2 @ Q.T) / T
    work = []
    for Wp, Qp, bp, J in sets:
        geom = lm.patch_geometry(N, Wp.shape[0], Wp.shape[2])
        Xp = geom.unfold(X)                                              # (P, B, d)
        A = _patch_fields(Wp, Qp, bp, Xp)                                # (P, B, S)
        AJ = (A[:, :, None, :] + J[:, None, :, :]) / T                   # (P, B, K, S)
        Phi = Phi + jnp.sum(logsumexp(AJ, axis=-1), axis=0)
        work.append((geom, Xp, jax.nn.softmax(AJ, axis=-1)))
    pi = jax.nn.softmax(Phi, axis=1)                                     # (B, K)
    m = jnp.broadcast_to(W[None, :, :], (B, K, N))                       # G_g, (B, K, N)
    if Q is not None:
        m = m + 2.0 * X[:, None, :] * Q[None, :, :]
    lap_a = pi @ jnp.sum(Q, axis=1) * 2.0 if Q is not None else jnp.zeros((B,), X.dtype)
    within = jnp.zeros((B,), X.dtype)
    qs, pcs = [], []
    for (Wp, Qp, bp, J), (geom, Xp, pc) in zip(sets, work):
        P, S, d = Wp.shape
        pcf = pc.reshape(P, B * K, S)
        gbar = lm._pdot_t(pcf, Wp)                                       # (P, B K, d)
        if Qp is not None:
            Xpk = jnp.broadcast_to(Xp[:, :, None, :], (P, B, K, d)).reshape(P, B * K, d)
            gbar = gbar + 2.0 * Xpk * lm._pdot_t(pcf, Qp)
        m = m + geom.fold(gbar).reshape(B, K, N)
        q = jnp.einsum("bk,pbks->pbs", pi, pc)                           # (P, B, S)
        qs.append(q)
        pcs.append(pc)
        if with_lap:
            GG = jnp.sum(Wp * Wp, axis=-1)[:, None, :]
            if Qp is not None:
                GG = GG + 4.0 * lm._pdot(Xp, Wp * Qp) + 4.0 * lm._pdot(Xp * Xp, Qp * Qp)
                lap_a = lap_a + 2.0 * jnp.sum(q * jnp.sum(Qp, axis=-1)[:, None, :], axis=(0, 2))
            g2 = jnp.sum(gbar * gbar, axis=(0, 2)).reshape(B, K)         # sum_p |gbar_p(g)|^2
            within = within + jnp.sum(q * GG, axis=(0, 2)) - jnp.sum(pi * g2, axis=1)
    mbar = jnp.einsum("bk,bkn->bn", pi, m)
    if not with_lap:
        return -mbar, None, pi, qs, pcs
    dm = m - mbar[:, None, :]
    between = jnp.sum(pi * jnp.sum(dm * dm, axis=2), axis=1)
    lap = -lap_a - (within + between) / T
    return -mbar, lap, pi, qs, pcs


def make_tree_loss_per_sample(vis, T, local=True, compute_dtype=None):
    """The per-sample implicit score-matching loss of lm.make_loss_per_sample (-lap F / T + |grad F|^2 / (2 T^2)) with
    the global Potts unit and the patch units coupled by params[J_KEYS] (tree_terms); exact statistics only. Without
    any J key it is the same model as lm.make_loss_per_sample(vis, T, K, 'potts', local) (sums in another order)."""
    def loss(params, X, key=None):
        g, lap = vis.grad_lap(params["theta"], X)
        gt, lapt, _, _, _ = tree_terms(params["W"], params.get("Q"), params["b"], tree_sets(params), X, T)
        g, lap = g + gt, lap + lapt
        if local:
            gl, lapl, _, _ = lm.local_terms(params["wl"], params["bl"], X, T)
            g, lap = g + gl, lap + lapl
        return -lap / T + jnp.sum(g * g, axis=1) / (2.0 * T * T)

    return loss if compute_dtype is None else lm.in_compute_dtype(loss, compute_dtype)


def make_tree_loss(vis, T, local=True, compute_dtype=None):
    per_sample = make_tree_loss_per_sample(vis, T, local, compute_dtype)
    return lambda params, X, key=None: jnp.mean(per_sample(params, X, key).astype(jnp.float64))


def make_tree_force(vis, T, local=True):
    """grad F (B, N) of the tree model (visible part + tree + per-pixel units), as lm.make_force."""
    def force(params, X):
        g, _ = vis.grad_lap(params["theta"], X)
        g = g + tree_terms(params["W"], params.get("Q"), params["b"], tree_sets(params), X, T, with_lap=False)[0]
        if local:
            g = g + lm.local_terms(params["wl"], params["bl"], X, T)[0]
        return g

    return force


def hidden_posteriors(params, X, T):
    """Today's model (no J): pi (B, K) of the global unit and [q (P, B, S)] of the patch sets present, given X."""
    A = X @ params["W"].T + params["b"]
    if "Q" in params:
        A = A + (X * X) @ params["Q"].T
    qs = []
    for kw, kb, kq in lm.PATCH_SETS:
        if kw in params:
            Wp = params[kw]
            geom = lm.patch_geometry(X.shape[1], Wp.shape[0], Wp.shape[2])
            qs.append(jax.nn.softmax(_patch_fields(Wp, params.get(kq), params[kb], geom.unfold(X)) / T, axis=-1))
    return jax.nn.softmax(A / T, axis=1), qs
