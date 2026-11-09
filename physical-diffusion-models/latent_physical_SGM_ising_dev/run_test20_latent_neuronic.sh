#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test20
#SBATCH --output=latent_test20_%j.out
#SBATCH --error=latent_test20_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306
# Twentieth validation round, for K > 1 Ising hidden units (the split tree: latent_model.split_tree,
# split_reference, random_orthogonal_rows, ising_force_matching_local_grad, ising_unit_stats, force_misfit; the
# train_latent.py Ising path; cd1_check --local-rule for Ising units; inspect_units.py). It runs the STAGING copy in
# physical-diffusion-models/latent_physical_SGM_ising_dev (the live folder is not touched); outputs go to the usual
# out/latent_physical_SGM. mnist14 unless stated:
#   0. unit checks: a. K = 1 identities with fisher_reference (mnist14 and mnist28, all 100 slices); b. split_tree
#      invariants (K 2, 8, 33, 128, perclass and pooled, and K 8 on mnist14_d012); c. split_reference (K 8, 32, 128);
#      d. start scale on p_5 (tree and random starts, K 8 and 128); e. the Ising local rule against autodiff;
#      f. permutation / sign-flip symmetry and the zero-gradient point; g. sample_latent.make_drift with K Ising units;
#      h. ising_unit_stats on a constructed K = 3 model; i. a print of force_misfit on trained K = 0 slices (no bit);
#   1. default paths unchanged: validation loss of slices 0-1 at full steps equal to four production runs (K = 1
#      quartic, K = 1 quad, Potts 128 + both power 4, Potts 128 random); each production run's config.json and DONE,
#      copied into a fresh --out-root, resume to "complete (DONE marker)";
#   2. smokes (Ising, power 4, --quad --local --select ema, von Neumann radius 1, lin, 3 slices at 2% of the steps)
#      and their checks (config, split_tree.npz, exact start, --hidden-fixed, learning rates, n_dead, min_cos_prev);
#      one K = 8 run over all 100 slices at 2%, then sample_latent --n 20 and inspect_units;
#   3. invalid options refused before training.
# Submit from the staging folder:   cd physical-diffusion-models/latent_physical_SGM_ising_dev && sbatch run_test20_latent_neuronic.sh

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
[ "$(basename "$(pwd)")" = "latent_physical_SGM_ising_dev" ] || { echo "Error: submit from physical-diffusion-models/latent_physical_SGM_ising_dev/"; exit 1; }
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM_ising_dev/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
D=/physical-diffusion-models/latent_physical_SGM_ising_dev
LIVE=/physical-diffusion-models/latent_physical_SGM
OUT=../out/latent_physical_SGM
COUT=/physical-diffusion-models/out/latent_physical_SGM
J=${SLURM_JOB_ID}
run() { apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u "$@"; }

rc=0
echo "== 0. unit checks"
run - <<'EOF' || rc=$((rc | 1))
import math
import sys
sys.path.insert(0, "/physical-diffusion-models/latent_physical_SGM_ising_dev")
import numpy as np
import latent_model as lm
import sample_latent as sl
import jax
import jax.numpy as jnp

ok = True
def check(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(f"  {name}: {'ok' if good else 'FAIL'} {extra}", flush=True)

def rel(a, b):
    a, b = np.ravel(np.asarray(a, dtype=np.float64)), np.ravel(np.asarray(b, dtype=np.float64))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

T = lm.TEMP
taus = lm.forward_time_pts()
P14 = lm.make_problem("mnist14")
X14, L14 = np.asarray(P14["data"]), P14["labels"]
n14 = X14.shape[1]

def p_t(X, t, B, rng):
    e = math.exp(-t)
    return e * X[rng.integers(0, X.shape[0], B)] + math.sqrt(T * (1.0 - e * e)) * rng.standard_normal((B, X.shape[1]))

print("-- a. K = 1 identities (all 100 slices)", flush=True)
for pname in ("mnist14", "mnist28"):
    P = P14 if pname == "mnist14" else lm.make_problem("mnist28")
    X, L = np.asarray(P["data"]), P["labels"]
    Wf, bf = lm.fisher_reference(X, L, taus, T)
    Ws, bs, _ = lm.split_reference(X, L, taus, T, 1, seed=0, pooled=False)
    check(f"{pname}: split_reference(K=1, perclass) = fisher_reference(labels)",
          Ws.shape == (100, 1, X.shape[1]) and rel(Ws[:, 0], Wf) <= 1e-9 and rel(bs[:, 0], bf) <= 1e-9,
          f"(rel W {rel(Ws[:, 0], Wf):.1e}, b {rel(bs[:, 0], bf):.1e}; bitwise W {np.array_equal(Ws[:, 0], Wf)}, "
          f"b {np.array_equal(bs[:, 0], bf)})")
    Wp, bp, tp = lm.split_reference(X, L, taus, T, 1, seed=0, pooled=True)
    Wf2, bf2 = lm.fisher_reference(X, tp["leaf_of"], taus, T)    # K = 1: leaf 0 = left child, leaf 1 = right (h = +1)
    prior = 0.5 * T * math.log(tp["n_right"][0] / tp["n_left"][0])
    check(f"{pname}: split_reference(K=1, pooled) = fisher_reference(root 2-means) + (T/2) log(n_R/n_L)",
          rel(Wp[:, 0], Wf2) <= 1e-9 and rel(bp[:, 0] - prior, bf2) <= 1e-9,
          f"(rel W {rel(Wp[:, 0], Wf2):.1e}, b {rel(bp[:, 0] - prior, bf2):.1e}; n_L/n_R {tp['n_left'][0]}/"
          f"{tp['n_right'][0]}, prior {prior / T:.3g} T, root purity {tp['unit_purity'][0]:.3f})")
    del P, X

print("-- b. split_tree invariants", flush=True)
STRUCT = ("parent", "depth", "child_left", "child_right", "n_left", "n_right", "mu_left", "mu_right", "leaf_of",
          "leaf_means", "H", "gain", "kind", "seed", "pooled", "n_min")

def same_tree(a, b, keys=None):
    keys = sorted(a) if keys is None else keys
    return set(a) == set(b) and all(np.array_equal(a[k], b[k], equal_nan=(k == "unit_purity")) for k in keys)

def tree_ok(tag, X, L, K, pooled, n_min=20):
    tr = lm.split_tree(X, L, K, seed=0, pooled=pooled)
    n, C = X.shape[0], len(np.unique(L))
    lo, H = tr["leaf_of"], tr["H"]
    fails = []
    if not (lo.shape == (n,) and lo.min() == 0 and lo.max() == K and np.all(np.bincount(lo, minlength=K + 1) > 0)):
        fails.append("leaves do not partition the images")
    if H.shape != (K + 1, K) or not np.all(np.isin(H, (-1, 0, 1))):
        fails.append("H shape / values")
    rows_of = lambda j: H[lo, j] != 0                       # images below node j
    for j in range(K):
        left, right = H[lo, j] == -1, H[lo, j] == 1
        node = np.ones(n, bool) if tr["parent"][j] < 0 else \
            (H[lo, tr["parent"][j]] == (1 if tr["child_right"][tr["parent"][j]] == j else -1))
        if np.any(left & right) or not np.array_equal(left | right, node):
            fails.append(f"node {j}: children not disjoint or union != node")
        for side, c, mask in (("left", tr["child_left"][j], left), ("right", tr["child_right"][j], right)):
            sub = rows_of(c) if c >= 0 else lo == (-c - 1)
            if not np.array_equal(sub, mask):
                fails.append(f"node {j}: {side} child inconsistent with H / leaf_of")
        if left.sum() != tr["n_left"][j] or right.sum() != tr["n_right"][j]:
            fails.append(f"node {j}: n_left / n_right")
        if not (np.allclose(tr["mu_left"][j], X[left].mean(0)) and np.allclose(tr["mu_right"][j], X[right].mean(0))):
            fails.append(f"node {j}: child means")
        if min(tr["n_left"][j], tr["n_right"][j]) < n_min:
            fails.append(f"node {j}: a child below n_min ({tr['n_left'][j]}/{tr['n_right'][j]})")
    for leaf in range(K + 1):                               # descending by H's signs reaches the leaf
        j, steps = 0, 0
        while j >= 0 and steps <= K:
            s = H[leaf, j]
            if s == 0:
                break
            j = tr["child_right"][j] if s == 1 else tr["child_left"][j]
            steps += 1
        if j != -(leaf + 1):
            fails.append(f"leaf {leaf}: path by H does not end at the leaf")
    if not pooled:
        for j in range(K):
            below = rows_of(j)
            if j < C - 1:
                good = tr["kind"][j] == lm.SPLIT_CLASS and all(
                    len(np.unique((H[lo, j] == 1)[(L == c) & below])) <= 1 for c in np.unique(L))
            else:
                good = tr["kind"][j] != lm.SPLIT_CLASS and len(np.unique(L[below])) == 1
            if not good:
                fails.append(f"node {j}: " + ("not a class split" if j < C - 1 else "not class-pure"))
    else:
        perm = np.random.default_rng(0).permutation(n)
        trp = lm.split_tree(X, L[perm], K, seed=0, pooled=True)
        if not same_tree(tr, trp, STRUCT):
            fails.append("structure changes when the labels are permuted")
    again = lm.split_tree(X, L, K, seed=0, pooled=pooled)
    if not same_tree(tr, again):
        fails.append("two calls differ")
    other = lm.split_tree(X, L, K, seed=100, pooled=pooled)
    differs = not np.array_equal(other["leaf_of"], tr["leaf_of"])
    check(f"{tag} K={K} {'pooled' if pooled else 'perclass'}", not fails,
          f"(depth <= {tr['depth'].max()}, leaf sizes {np.bincount(lo).min()}-{np.bincount(lo).max()}, kinds "
          f"class/2-means/median {'/'.join(str(int(np.sum(tr['kind'] == q))) for q in range(3))}, root purity "
          f"{tr['unit_purity'][0]:.3f}; k-means seed 100 gives {'a different' if differs else 'THE SAME'} tree) "
          + "; ".join(fails[:4]))
    return differs

for pooled in (False, True):
    for K in (2, 8, 33, 128):
        differs = tree_ok("mnist14", X14, L14, K, pooled)
        if K == 128:
            check(f"mnist14 K=128 {'pooled' if pooled else 'perclass'}: k-means seed 100 changes the tree", differs)
P012 = lm.make_problem("mnist14_d012")
for pooled in (False, True):
    tree_ok("mnist14_d012", np.asarray(P012["data"]), P012["labels"], 8, pooled)
try:
    lm.split_tree(X14[:100], L14[:100], 8, pooled=True)
    check("split_tree raises when it runs out of splittable leaves", False)
except ValueError as e:
    check("split_tree raises when it runs out of splittable leaves", True, f"({e})")

print("-- c. split_reference", flush=True)
refs = {}
for pooled in (True, False):
    for K in (8, 32, 128):
        W, b, tr = lm.split_reference(X14, L14, taus, T, K, seed=0, pooled=pooled)
        refs[(K, pooled)] = (W, b, tr)
        shapes = W.shape == (100, K, n14) and b.shape == (100, K)
        fin = bool(np.isfinite(W).all() and np.isfinite(b).all())
        cond = min(float(s[-1] / s[0]) for s in (np.linalg.svd(W[k], compute_uv=False) for k in (0, 50, 99)))
        Wn = W / np.linalg.norm(W, axis=2, keepdims=True)
        G = np.abs(np.einsum("skn,sln->skl", Wn, Wn))
        G[:, np.arange(K), np.arange(K)] = 0.0
        adj = np.sum(Wn[:-1] * Wn[1:], axis=2)                       # cos(W[k, j], W[k + 1, j]), (99, K)
        kk, jj = np.unravel_index(np.argmin(adj), adj.shape)
        check(f"K={K} {'pooled' if pooled else 'perclass'}: shapes, finite, sigma_min/max > 1e-8 (k 0/50/99), "
              f"max pair |cos| < 0.99, cos(k, k+1) > 0.9",
              shapes and fin and cond > 1e-8 and G.max() < 0.99 and adj.min() > 0.9,
              f"(sigma ratio {cond:.2e}, max pair |cos| {G.max():.3f}, min cos(k, k+1) {adj.min():.4f} at k {kk} "
              f"unit {jj})")

print("-- d. start scale on 4096 draws of p_5", flush=True)
rng = np.random.default_rng(5)
X5 = p_t(X14, 5.0, 4096, rng)
for K in (8, 128):
    W, b, _ = refs[(K, True)]
    r0 = float(np.sqrt(np.mean(np.sum(W[0] ** 2, axis=1))))
    Wr1 = lm.random_orthogonal_rows(K, n14, r0, np.random.default_rng(1000 + 1))
    Wr2 = lm.random_orthogonal_rows(K, n14, r0, np.random.default_rng(1000 + 2))
    for tag, Ws_, bs_ in (("tree", W[0], b[0]), ("random", Wr1, np.zeros(K))):
        A = X5 @ Ws_.T + bs_
        sd = A.std(axis=0) / T
        vm = np.mean(1.0 - np.tanh(A / T) ** 2, axis=0)
        check(f"K={K} {tag} start: std(a_j)/T < 1 and <v_j> > 0.3 for every unit", sd.max() < 1 and vm.min() > 0.3,
              f"(std/T {sd.min():.3f}-{sd.max():.3f}, <v> {vm.min():.3f}-{vm.max():.3f})")
    err = float(np.max(np.abs(Wr1 @ Wr1.T - r0 ** 2 * np.eye(K))) / r0 ** 2)
    check(f"K={K} random rows: W W^T = r0^2 I", err <= 1e-12, f"(max error / r0^2 {err:.1e}, r0 {r0:.4g})")
    check(f"K={K} random rows: seeds 1 and 2 differ", not np.allclose(Wr1, Wr2),
          f"(max |diff| {np.max(np.abs(Wr1 - Wr2)):.3g})")

print("-- e. Ising local rule = autodiff (von Neumann radius 1, springs only)", flush=True)
vis = lm.DenseVisible(lm.connectivity(14, 1, "von_neumann"), n14)
th0, _ = lm.paper_theta_layout(vis.P, n14)
th0 = lm.quartic_initial_theta(th0, n14, vis.P)
for s_ in lm.dropped_pair_slices(n14, vis.P, ("lin",)):
    th0 = th0.at[s_].set(0.0)
rng = np.random.default_rng(7)
th0 = th0 * (1.0 + 0.05 * jnp.asarray(rng.standard_normal(th0.shape)))    # generic values, the same zero pattern
K = 8
W8, b8, _ = refs[(8, True)]
wl_ref, bl_ref = lm.local_reference(X14, taus, T, (0.5 - P14["mean"]) / P14["std"])
Q8 = 0.1 * np.sqrt(np.mean(W8 ** 2, axis=(1, 2)))[:, None, None] * rng.standard_normal(W8.shape)
autod = {loc: jax.jit(jax.grad(lm.make_loss(vis, T, K, "ising", loc))) for loc in (False, True)}
def rule_of(loc):                                   # loc is a Python bool closed over, never traced
    return jax.jit(lambda p, X: lm.ising_force_matching_local_grad(p, X, T, vis, loc))
rule = {loc: rule_of(loc) for loc in (False, True)}

def params_at(k, quad=True, local=True):
    p = {"theta": th0, "W": jnp.asarray(W8[k]), "b": jnp.asarray(b8[k])}
    if quad:
        p["Q"] = jnp.asarray(Q8[k])
    if local:
        p.update(wl=jnp.asarray(wl_ref[k]), bl=jnp.asarray(bl_ref[k]))
    return p

Xk = {k: jnp.asarray(p_t(X14, float(taus[k]), 512, rng)) for k in (1, 50, 99)}
for k in (1, 50, 99):
    for local in (False, True):
        for quad in (True, False):
            p = params_at(k, quad, local)
            ga, gr = autod[local](p, Xk[k]), rule[local](p, Xk[k])
            want = {"W", "b"} | ({"Q"} if quad else set()) | ({"wl", "bl"} if local else set())
            errs = {h: rel(gr[h], ga[h]) for h in gr}
            check(f"slice {k}, quad {quad}, local {local}", set(gr) == want and max(errs.values()) <= 1e-9,
                  "(" + ", ".join(f"{h} {e:.1e}" for h, e in errs.items()) + ")")

print("-- f. symmetries", flush=True)
loss8 = jax.jit(lm.make_loss(vis, T, K, "ising", True))
p = params_at(50)
L0 = float(loss8(p, Xk[50]))
perm = rng.permutation(K)
sgn = jnp.asarray(np.where(rng.random(K) < 0.5, -1.0, 1.0))
p2 = dict(p, W=p["W"][perm] * sgn[:, None], b=p["b"][perm] * sgn, Q=p["Q"][perm] * sgn[:, None])
L1 = float(loss8(p2, Xk[50]))
check("loss unchanged under a unit permutation and sign flips", abs(L1 - L0) <= 1e-12 * abs(L0),
      f"(loss {L0:.10g}, rel diff {abs(L1 - L0) / abs(L0):.1e}, flipped {int(np.sum(np.asarray(sgn) < 0))} of {K})")
pz = dict(p, W=jnp.zeros_like(p["W"]), b=jnp.zeros_like(p["b"]), Q=jnp.zeros_like(p["Q"]))
gz = autod[True](pz, Xk[50])
mx = {h: float(jnp.max(jnp.abs(gz[h]))) for h in ("W", "b", "Q")}
check("hidden gradient exactly 0 at W = b = Q = 0", all(v == 0.0 for v in mx.values()), f"({mx})")

print("-- g. make_drift (K = 8 Ising, Q, per-pixel units, von Neumann radius 1)", flush=True)
nt = len(taus)
th_hist = np.tile(np.asarray(th0), (nt, 1)) * (1.0 + 0.01 * rng.standard_normal((nt, th0.size)))
loc_hist = (wl_ref, bl_ref)
drift, _ = sl.make_drift(th_hist, W8, b8, "sde", "mnist14", "ising", loc_hist, Q8, 1, "von_neumann")
drift0, _ = sl.make_drift(th_hist, np.zeros_like(W8), np.zeros_like(b8), "sde", "mnist14", "ising",
                          (np.zeros_like(wl_ref), np.zeros_like(bl_ref)), np.zeros_like(Q8), 1, "von_neumann")
force = jax.jit(lm.make_force(vis, T, K, "ising", True))
for k in (20, 60):
    x = jnp.asarray(p_t(X14, float(taus[k]), 1, rng)[0])
    t_rev = lm.T_FORWARD - float(taus[k])
    d_full = np.asarray(drift(t_rev, x, None))
    hid_drift = d_full - np.asarray(drift0(t_rev, x, None))
    pk = {"theta": jnp.zeros(th_hist.shape[1]), "W": jnp.asarray(W8[k]), "b": jnp.asarray(b8[k]),
          "Q": jnp.asarray(Q8[k]), "wl": jnp.asarray(wl_ref[k]), "bl": jnp.asarray(bl_ref[k])}
    hid_force = np.asarray(force(pk, x[None, :]))[0]          # theta = 0: the visible force is exactly 0
    check(f"knot {k}: drift finite, hidden drift = -2 x hidden force", np.all(np.isfinite(d_full))
          and rel(hid_drift, -2.0 * hid_force) <= 1e-10,
          f"(rel {rel(hid_drift, -2.0 * hid_force):.1e}, |hidden force| {np.linalg.norm(hid_force):.3g})")

print("-- h. ising_unit_stats on a constructed K = 3 model", flush=True)
rng = np.random.default_rng(3)
Xh = rng.standard_normal((4096, n14))
W3 = np.zeros((3, n14))
W3[0, 0], W3[1, 1], W3[2, 2] = 50 * T, 0.01 * T, 0.5 * T       # decided, weak, dead (field 10 T +- 0.5 T)
b3 = np.array([0.0, 0.0, 10 * T])
rec, arr = lm.ising_unit_stats(W3, b3, None, Xh, T)
dead = arr["unit_minority"] < 0.005
check("classifies a decided, a weak and a dead unit", rec["n_dead"] == 1 and rec["n_weak"] == 1
      and rec["n_decided"] == 1 and list(dead) == [False, False, True] and arr["unit_abs_m"][0] > 0.9
      and arr["unit_abs_m"][1] < 0.1, f"(dead/weak/decided {rec['n_dead']}/{rec['n_weak']}/{rec['n_decided']}, "
      f"minority {np.round(arr['unit_minority'], 4)}, <|m|> {np.round(arr['unit_abs_m'], 3)})")
Q3 = 0.01 * T * rng.standard_normal((3, n14))
A_np = lm.ising_fields(W3, b3, Q3, Xh)
_, _, A_j, m_j = lm.hidden_terms_quad(jnp.asarray(W3), jnp.asarray(Q3), jnp.asarray(b3), jnp.asarray(Xh), T)
check("fields = A of hidden_terms_quad (Q != 0)", rel(A_np, A_j) <= 1e-12, f"(rel {rel(A_np, A_j):.1e})")
rec_q, arr_q = lm.ising_unit_stats(W3, b3, Q3, Xh, T)
m_j = np.asarray(m_j)
check("ising_unit_stats uses those fields", rel(arr_q["unit_abs_m"], np.abs(m_j).mean(0)) <= 1e-12
      and rel(arr_q["unit_minority"], 0.5 * (1.0 - np.abs(m_j.mean(0)))) <= 1e-12)
W, b, tr = refs[(8, True)]
idx = np.random.default_rng(9).integers(0, X14.shape[0], 4096)
rec0, _ = lm.ising_unit_stats(W[-1], b[-1], None, X14[idx], T, W_ref=W[-1], W_prev=W[-2], tree=tr,
                              leaf=tr["leaf_of"][idx], labels=L14[idx])
print(f"  (print) the K = 8 pooled reference at t = 0 on clean images: leaf acc {rec0['leaf_acc']:.3f}, unit-0 acc "
      f"{rec0['class_acc']:.3f}, dead/weak/decided {rec0['n_dead']}/{rec0['n_weak']}/{rec0['n_decided']}, straddle "
      f"{rec0['straddle_off']}, own balance error {rec0['own_balance_err']:.3f}, overlap {rec0['overlap_ref']:.3f}")
print("unit checks:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF

echo "-- i. (print only) force_misfit on trained K = 0 slices near t = 3: radius 7 vs the nearest-neighbour model"
run - <<'EOF' || echo "  misfit print failed (no exit-code bit)"
import json
import os
import sys
sys.path.insert(0, "/physical-diffusion-models/latent_physical_SGM_ising_dev")
import numpy as np
import latent_model as lm
import jax.numpy as jnp
import jax.random as jr
from physical_diffusion_fns.helper_fns import sample_forward_process

T, taus, k = lm.TEMP, lm.forward_time_pts(), 40
rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
for run in ("mnist14_K0_ema_quartic", "mnist14_K0_ema_quartic_vn1_lin"):
    rd = os.path.join(lm.OUT_ROOT, run)
    path = os.path.join(rd, "slices", f"slice_{k:03d}.npz")
    if not os.path.exists(path):
        print(f"  {run}: no slice {k}; skipped")
        continue
    with open(os.path.join(rd, "config.json")) as f:
        cfg = json.load(f)
    prob = lm.make_problem(cfg.get("problem", "mnist28"), cfg.get("radius"), cfg.get("neighbourhood", "chebyshev"))
    vis = lm.DenseVisible(prob["conn"], prob["n"])
    theta = jnp.asarray(np.load(path)["theta"])
    Gs = []
    for c in range(4):
        X = sample_forward_process(float(taus[k]), 4096, T, lm.SIGMA_FORWARD, prob["data"], jr.PRNGKey(50_000 + c), beta=1)
        Gs.append(lm.force_misfit(np.asarray(X), np.asarray(vis.grad_lap(theta, X)[0]), T))
    G = np.mean(Gs, axis=0)
    noise = rms(0.5 * (Gs[0] + Gs[1]) - 0.5 * (Gs[2] + Gs[3])) / 2.0
    n, conn = prob["n"], prob["conn"]
    unc = np.ones((n, n), bool)
    unc[np.diag_indices(n)] = False
    unc[conn[:, 0], conn[:, 1]] = False
    unc[conn[:, 1], conn[:, 0]] = False
    ev = np.linalg.eigvalsh(G)
    print(f"  {run} (radius {cfg.get('radius') or 'default 7'}, {cfg.get('neighbourhood', 'chebyshev')}, "
          f"{conn.shape[0]} pairs; slice {k}, t = {taus[k]:.3f}; 4 x 4096 samples): rms G on the diagonal "
          f"{rms(np.diag(G)):.3g}, on coupled pairs {rms(G[conn[:, 0], conn[:, 1]]):.3g}, on uncoupled pairs "
          f"{rms(G[unc]) if unc.any() else float('nan'):.3g} (noise about {noise:.2g}); eigenvalues top {ev[-1]:.3g}, "
          f"2nd {ev[-2]:.3g}, 3rd {ev[-3]:.3g}, bottom {ev[0]:.3g}")
EOF

echo "== 1. default paths: validation loss of slices 0-1 at full steps, staging code vs the production runs"
REG=(
  "smoke20_reg_K1quartic_${J}|mnist14_K1_ema_quartic|--problem mnist14 --K 1 --select ema --visible quartic"
  "smoke20_reg_K1quad_${J}|mnist14_K1_ema_quad|--problem mnist14 --K 1 --quad --select ema"
  "smoke20_reg_P128quartic_${J}|mnist14_K128_ema_potts_quad_local_quartic|--problem mnist14 --hidden potts --K 128 --quad --local --select ema --visible quartic"
  "smoke20_reg_P128random_${J}|mnist14_K128_ema_potts_quad_local_random|--problem mnist14 --hidden potts --K 128 --proto-init random --quad --local --select ema"
)
vals() { grep -o '"val_loss": [-0-9.e+]*' "$1" 2>/dev/null | head -2 | tr '\n' ' '; }
for r in "${REG[@]}"; do
  IFS='|' read -r name prod flags <<< "${r}"
  set -x
  run ${D}/train_latent.py --name ${name} ${flags} --max-slices 2 || rc=$((rc | 2))
  set +x
  a=$(vals ${OUT}/${name}/train_log.jsonl)
  b=$(vals ${OUT}/${prod}/train_log.jsonl)
  echo "  ${prod}: staging [${a}] production [${b}]"
  if [ "$a" = "$b" ] && [ -n "$a" ]; then
    echo "  ${prod}: identical"
  else
    echo "  ${prod}: DIFFERENT"; rc=$((rc | 16))
    # diagnosis only (no extra bit): the live code with the same flags on this node
    run ${LIVE}/train_latent.py --name ${name/smoke20_reg_/smoke20_live_} ${flags} --max-slices 2
    c=$(vals ${OUT}/${name/smoke20_reg_/smoke20_live_}/train_log.jsonl)
    echo "  ${prod}: live code [${c}]: $([ "$c" = "$b" ] && echo 'equals' || echo 'differs from') the production run," \
         "$([ "$c" = "$a" ] && echo 'equals' || echo 'differs from') the staging code"
  fi
done
echo "== 1b. resume: each production run's config.json and DONE copied into a fresh --out-root; the relaunch must exit 0 with 'complete (DONE marker)'"
if grep -q -- '"--out-root"' train_latent.py; then
  RES=test20_resume_${J}
  for r in "${REG[@]}"; do
    IFS='|' read -r name prod flags <<< "${r}"
    mkdir -p ${OUT}/${RES}/${prod}
    cp ${OUT}/${prod}/config.json ${OUT}/${prod}/DONE ${OUT}/${RES}/${prod}/
    if run ${D}/train_latent.py --name ${prod} --out-root ${COUT}/${RES} ${flags} \
       && grep -q "complete (DONE marker)" ${OUT}/${RES}/${prod}/train_log.txt; then
      echo "  ${prod}: resumes (DONE marker)"
    else
      echo "  ${prod}: RESUME FAILED"; rc=$((rc | 32))
    fi
  done
else
  echo "  train_latent.py has no --out-root option: resume check skipped"
fi

echo "== 2. smokes (Ising, power 4, --quad --local --select ema, von Neumann radius 1, lin, 2% of the steps)"
BASE="--problem mnist14 --hidden ising --visible quartic --local --select ema --radius 1 --neighbourhood von_neumann --pair-terms lin --steps-scale 0.02"
I3="${BASE} --quad --max-slices 3"
SMOKES=(
  "smoke20_i8_${J}|pooled|${I3} --K 8"
  "smoke20_i8_perclass_${J}|perclass|${I3} --K 8 --proto-init perclass"
  "smoke20_i32_random_${J}|random|${I3} --K 32 --proto-init random"
  "smoke20_i1_pooled_${J}|pooled|${I3} --K 1 --proto-init pooled"
  "smoke20_i8_S16jk_${J}|pooled|${I3} --K 8 --hidden-samples 16 --hidden-jackknife"
  "smoke20_i8_state_${J}|pooled|${I3} --K 8 --hidden-norm state"
  "smoke20_i8_fixed_${J}|pooled|${BASE} --max-slices 3 --K 8 --hidden-fixed"
  "smoke20_i8_d012_${J}|pooled|${I3} --K 8 --problem mnist14_d012"
  "smoke20_i8_seed2_${J}|pooled|${I3} --K 8 --seed 2"
  "smoke20_i8_full_${J}|pooled|${BASE} --quad --K 8"
)
for r in "${SMOKES[@]}"; do
  IFS='|' read -r name proto flags <<< "${r}"
  set -x
  run ${D}/train_latent.py --name ${name} ${flags} || { rc=$((rc | 4)); set +x; continue; }
  set +x
  grep -m 1 "split-tree reference" ${OUT}/${name}/train_log.txt | cut -c1-400
  tail -n 1 ${OUT}/${name}/train_log.txt | cut -c1-400
  run - "${COUT}/${name}" "${proto}" <<'EOF' || rc=$((rc | 64))
import glob
import json
import os
import sys
sys.path.insert(0, "/physical-diffusion-models/latent_physical_SGM_ising_dev")
import numpy as np
import latent_model as lm

run_dir, want_proto = sys.argv[1], sys.argv[2]
ok = True
def check(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(f"  {name}: {'ok' if good else 'FAIL'} {extra}", flush=True)

with open(os.path.join(run_dir, "config.json")) as f:
    cfg = json.load(f)
with open(os.path.join(run_dir, "train_log.jsonl")) as f:
    recs = [json.loads(line) for line in f]
K, seed, proto, fixed = cfg["K"], cfg["seed"], cfg["proto_init"], cfg["hidden_fixed"]
check("config: init split-tree, resolved proto_init", cfg["init"] == "split-tree" and proto == want_proto,
      f"(init {cfg['init']}, proto_init {proto})")
prob = lm.make_problem(cfg["problem"])
X, L = np.asarray(prob["data"]), prob["labels"]
taus = lm.forward_time_pts()
km, pooled = 100 * (seed - lm.KEY_SEED), proto in ("pooled", "random")
W_ref, b_ref, tree = lm.split_reference(X, L, taus, lm.TEMP, K, seed=km, pooled=pooled)
z = np.load(os.path.join(run_dir, "split_tree.npz"))
check("split_tree.npz equals a fresh build", set(z.files) == set(tree) and all(
    np.array_equal(z[k], tree[k], equal_nan=(k == "unit_purity")) for k in tree))
files = sorted(glob.glob(os.path.join(run_dir, "slices", "slice_*.npz")))
s0 = np.load(files[0])
if proto == "random":
    r0 = float(np.sqrt(np.mean(np.sum(W_ref[0] ** 2, axis=1))))
    W0, b0 = lm.random_orthogonal_rows(K, X.shape[1], r0, np.random.default_rng(1000 + seed)), np.zeros(K)
else:
    W0, b0 = W_ref[0], b_ref[0]
check("slice 0: W, b = the start exactly" + (" (the seeded Haar draw)" if proto == "random" else " (W_ref[0], b_ref[0])"),
      np.array_equal(s0["W"], W0) and np.array_equal(s0["b"], b0),
      f"(max |dW| {np.max(np.abs(s0['W'] - W0)):.2e}, max |db| {np.max(np.abs(s0['b'] - b0)):.2e})")
if fixed:
    bad = [i for i, f in enumerate(files) if not (np.array_equal(np.load(f)["W"], W_ref[i])
                                                  and np.array_equal(np.load(f)["b"], b_ref[i]))]
    check("--hidden-fixed: W, b = W_ref[k], b_ref[k] in every slice", not bad and len(files) == len(recs),
          f"({len(files)} slices; differing: {bad})")
W_lr, b_lr = (W_ref, b_ref) if km == 0 else lm.split_reference(X, L, taus, lm.TEMP, K, seed=0, pooled=pooled)[:2]
sW = np.sqrt(np.mean(W_lr ** 2, axis=(1, 2)))
sb = np.maximum(np.sqrt(np.mean(b_lr ** 2, axis=1)), lm.TEMP)
s = cfg["hidden_lr_scale"]

def same(a, w):
    return a == w if w == 0.0 else abs(a - w) <= 1e-12 * abs(w)
bad = []
for r in recs:
    k = r["slice"]
    on = k >= cfg["freeze_hidden_slices"] and not fixed
    want_W, want_b = (float(s * sW[k]), float(s * sb[k])) if on else (0.0, 0.0)
    if not (same(r["lr_W"], want_W) and same(r["lr_b"], want_b)):
        bad.append((k, r["lr_W"], want_W, r["lr_b"], want_b))
check("lr_W = 5e-4 sW[k], lr_b = 5e-4 sb[k] of the k-means seed 0 reference (0 while frozen / fixed)", not bad,
      f"(sW {sW[0]:.3g} -> {sW[-1]:.3g}, sb {sb[0]:.3g} -> {sb[-1]:.3g}; slice 1 lr_W {recs[min(1, len(recs) - 1)]['lr_W']:.3g}"
      f"{'; differing: ' + str(bad[:3]) if bad else ''})")
check("n_dead = 0 and min_cos_prev > 0 in every slice",
      all(r["n_dead"] == 0 and r["min_cos_prev"] > 0 for r in recs),
      f"(n_dead max {max(r['n_dead'] for r in recs)}, min_cos_prev min {min(r['min_cos_prev'] for r in recs):.4f}, "
      f"n_weak max {max(r['n_weak'] for r in recs)}, overlap last {recs[-1]['overlap_ref']:.3f})")
if len(recs) == lm.N_TIME_STEPS:
    h = np.load(os.path.join(run_dir, "hidden_history.npz"))
    check("hidden_history W (100, K, n), b (100, K)", h["W"].shape == (100, K, X.shape[1]) and h["b"].shape == (100, K),
          f"(W {h['W'].shape}, b {h['b'].shape}, files {h.files})")
print(os.path.basename(run_dir), "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF
done
FULL=smoke20_i8_full_${J}
echo "== 2b. the 100-slice run: sample_latent --n 20, inspect_units"
if [ -f ${OUT}/${FULL}/DONE ]; then
  set -x
  run ${D}/sample_latent.py --run ${FULL} --n 20 --chunk 20 || rc=$((rc | 128))
  set +x
  if grep -q '"n_nonfinite": 0' ${OUT}/${FULL}/samples/sde_base_n20.json 2>/dev/null; then
    echo "  samples finite: ok ($(grep -o '"max_abs_final": [-0-9.e+]*' ${OUT}/${FULL}/samples/sde_base_n20.json))"
  else
    echo "  samples finite: FAIL"; rc=$((rc | 128))
  fi
  set -x
  run ${D}/inspect_units.py ${FULL}:0,50,99 --samples sde_base_n20 || rc=$((rc | 128))
  set +x
  [ -s ${OUT}/units/${J}/summary.txt ] && ls ${OUT}/units/${J} || { echo "  inspect_units wrote no summary: FAIL"; rc=$((rc | 128)); }
else
  echo "  ${FULL} has no DONE marker: FAIL"; rc=$((rc | 128))
fi

echo "== 3. invalid options refused before training"
BAD=(
  "smoke20_bad_occ_${J}|${I3} --K 8 --hidden-norm occupancy"
  "smoke20_bad_floor_${J}|${I3} --K 8 --hidden-floor 1 --hidden-samples 8"
  "smoke20_bad_cls_${J}|${I3} --K 8 --cls 0"
  "smoke20_bad_randomfixed_${J}|${BASE} --max-slices 3 --K 8 --proto-init random --hidden-fixed"
  "smoke20_bad_fisher_d012_${J}|--problem mnist14_d012 --K 1 --proto-init perclass --select ema --max-slices 1"
  "smoke20_bad_perclass_d0123_${J}|${I3} --problem mnist14_d0123 --K 2 --proto-init perclass"
  "smoke20_bad_K200_${J}|${I3} --K 200"
)
bad=0
ERRF=$(mktemp)
for r in "${BAD[@]}"; do
  IFS='|' read -r name flags <<< "${r}"
  if run ${D}/train_latent.py --name ${name} ${flags} > /dev/null 2> ${ERRF}; then
    echo "  ${name}: ACCEPTED"; bad=1
  elif [ -e ${OUT}/${name}/slices/slice_000.npz ]; then
    echo "  ${name}: failed, but only after training a slice"; bad=1
  else
    echo "  ${name}: refused ($(grep -E 'Error' ${ERRF} | tail -n 1 | cut -c1-220))"
  fi
done
rm -f ${ERRF}
[ ${bad} -eq 0 ] || rc=$((rc | 8))
echo "== 3. invalid options refused: $([ ${bad} -eq 0 ] && echo yes || echo NO)"
echo "Finished at $(date) (exit code ${rc}: bits 1 unit checks, 2 regression run failed, 4 smoke run failed, 8 invalid accepted, 16 default path differs, 32 resume failed, 64 smoke checks failed, 128 100-slice run / sampler / inspect_units failed)"
exit ${rc}
