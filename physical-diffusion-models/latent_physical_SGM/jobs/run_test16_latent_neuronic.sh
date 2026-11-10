#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test16
#SBATCH --output=logs/latent_test16_%j.out
#SBATCH --error=logs/latent_test16_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Sixteenth validation round, after adding --hidden-norm state (scale_by_adam_per_state: one second moment per hidden
# state) and redesigning --hidden-norm occupancy (OccupancyTransform: each state's leaky-averaged gradient per unit
# occupation over one occupation-weighted step scale per group, clipped at 1). Test 15 showed that the first design
# (its step scale was the root-mean-square of the leaky-averaged direction itself) moves at full speed even at an
# optimum: |W| 0.76 -> 2.23 -> 3.90 in slices 0-2, against 0.74 with per-element Adam. mnist14, 0/1:
#   0. unit checks of both on small arrays;
#   1. the default and group paths are unchanged (Potts 128 + both, slices 0-1 at full steps; S = 64 jackknife with
#      per-element Adam, and exact statistics with group normalisation, against the production runs);
#   2. smoke: state (exact) and occupancy (exact; S = 16 jackknife) on Potts 8 + both, 3 slices at 2% of the steps;
#   3. state and occupancy on Potts 128 + both for 3 slices at full steps, exact and S = 64 jackknife, next to the
#      production runs' slices 0-2 (per-element and group normalisation).
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test16_latent_neuronic.sh

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
export GIT_COMMIT="$(git -C "${HOST_PROJECTS_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
D=/physical-diffusion-models/latent_physical_SGM
OUT=../out/latent_physical_SGM
J=${SLURM_JOB_ID}
run() { apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u "$@"; }
P128="--problem mnist14 --hidden potts --K 128 --quad --local --select ema"
P8="--problem mnist14 --hidden potts --K 8 --quad --local --select ema"

rc=0
echo "== 0. unit checks"
run - <<'EOF' || rc=$((rc | 1))
import sys
sys.path.insert(0, "/physical-diffusion-models/latent_physical_SGM")
import numpy as np
import jax.numpy as jnp
import optax
import train_latent as tl

M, n, bo, eps = 4, 3, 0.999, 1e-8
rng = np.random.default_rng(0)
params = {"theta": jnp.zeros(5), "W": jnp.zeros((M, n)), "b": jnp.zeros(M), "Q": jnp.zeros((M, n)),
          "wl": jnp.zeros(n), "bl": jnp.zeros(n)}
g = {k: jnp.asarray(rng.normal(size=v.shape)) for k, v in params.items()}
ok = True
def check(name, got, want, tol=1e-10):
    global ok
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    good = err < tol
    ok &= good
    print(f"  {name}: max error {err:.2e} {'ok' if good else 'FAIL'}")
adam_ref = optax.scale_by_adam()
ref, _ = adam_ref.update(g, adam_ref.init(params))

print(" state (scale_by_adam_per_state):")
tr = tl.scale_by_adam_per_state(("W", "Q"))
d, st = tr.update(g, tr.init(params))
for k in ("W", "Q"):
    gk = np.asarray(g[k])
    check(f"{k} step 1 = g / row rms", d[k], gk / (np.sqrt(np.mean(gk * gk, axis=1, keepdims=True)) + eps))
for k in ("b", "wl", "bl", "theta"):
    check(f"{k} step 1 = per-element Adam", d[k], ref[k])
g2 = {k: jnp.asarray(rng.normal(size=v.shape)) for k, v in params.items()}
d2, _ = tr.update(g2, st)
gk, gk2 = np.asarray(g["W"]), np.asarray(g2["W"])
m = (0.9 * 0.1 * gk + 0.1 * gk2) / (1 - 0.9 ** 2)
v = (0.999 * 0.001 * np.mean(gk * gk, axis=1, keepdims=True) + 0.001 * np.mean(gk2 * gk2, axis=1, keepdims=True)) / (1 - 0.999 ** 2)
check("W step 2 (momentum, row second moment)", d2["W"], m / (np.sqrt(v) + eps))

print(" occupancy (OccupancyTransform):")
tr = tl.OccupancyTransform(("W", "b", "Q"), ("W", "b", "Q", "wl", "bl"), beta_o=bo)
def expected(gd, occ_eff, u_num, clip=1.0):
    """occ_eff: per-state denominators; u_num: dict of numerators (already per unit occupation); scale from gd."""
    out = {}
    w = np.asarray(occ_w) / np.sum(occ_w)
    for k in ("W", "Q", "b"):
        gt = np.asarray(gd[k]) / (occ_eff[:, None] if gd[k].ndim == 2 else occ_eff)
        ms = np.mean(gt * gt, axis=1) if gt.ndim == 2 else gt * gt
        out[k] = np.clip(np.asarray(u_num[k]) / (np.sqrt(np.sum(w * ms)) + eps), -clip, clip)
    return out
# (a) first step, exact (p_unit 0)
occ = jnp.array([0.5, 0.3, 0.15, 0.05]); occ_w = occ
den = np.maximum(np.asarray(occ), 1e-30)
d, _ = tr.update(g, tr.init(params), occ, 0.0)
want = expected(g, den, {k: np.asarray(g[k]) / (den[:, None] if g[k].ndim == 2 else den) for k in ("W", "Q", "b")})
for k in ("W", "Q", "b"):
    check(f"(a) {k} step 1 = clip((g/p) / occupation-weighted rms)", d[k], want[k])
for k in ("wl", "bl"):
    gk = np.asarray(g[k])
    check(f"(a) {k} = g / group rms", d[k], gk / (np.sqrt(np.mean(gk * gk)) + eps))
check("(a) theta = per-element Adam", d["theta"], ref["theta"])
# (b) a gradient that carries the occupation (g_m = p_m f_m), constant: the direction is f / scale, any p_m
occ = jnp.array([0.9, 0.09, 9e-4, 9.9e-3]); occ_w = occ
f = {k: jnp.asarray(0.1 * rng.normal(size=params[k].shape)) for k in ("W", "b", "Q")}
gp = dict(g, W=f["W"] * occ[:, None], Q=f["Q"] * occ[:, None], b=f["b"] * occ)
st = tr.init(params)
for _ in range(50):
    d, st = tr.update(gp, st, occ, 0.0)
want = expected(gp, np.asarray(occ), {k: np.asarray(f[k]) for k in ("W", "Q", "b")})
for k in ("W", "Q", "b"):
    check(f"(b) {k} after 50 steps = clip(f / scale)", d[k], want[k], 1e-9)
print(f"     (b) row rms of the W direction, states 0-3: {np.sqrt(np.mean(np.asarray(d['W'])**2, axis=1)).round(3)} "
      f"(occupations 0.9 / 0.09 / 9e-4 / 9.9e-3; f rows {np.sqrt(np.mean(np.asarray(f['W'])**2, axis=1)).round(3)})")
# (c) measured statistics, B S = 1000 reads: state 3 visited once in the gradient draws at step 1, never in the count
# draws: its numerator is its gradient per visit, its weight in the scale is 0; then it decays as bo^(c-1)
p_unit = 1e-3
occ = jnp.array([0.5, 0.3, 0.2, 0.0]); occ_w = occ
fv = np.asarray(0.1 * rng.normal(size=n))
gv = dict(g, W=g["W"].at[3].set(jnp.asarray(fv) * p_unit))
st = tr.init(params)
d, st = tr.update(gv, st, occ, p_unit)
den = np.maximum(np.asarray(occ), p_unit)
want = expected(gv, den, {k: np.asarray(gv[k]) / (den[:, None] if gv[k].ndim == 2 else den) for k in ("W", "Q", "b")})
check("(c) step 1, W (state 3: gradient per visit / scale of states 0-2)", d["W"], want["W"])
g0 = dict(gv, W=gv["W"].at[3].set(0.0))
for _ in range(99):
    d, st = tr.update(g0, st, occ, p_unit)
mu, nu, pbar, count = st
w = np.asarray(occ) / np.sum(np.asarray(occ))
gt = np.asarray(g0["W"])[:3] / np.asarray(occ)[:3, None]
scale = np.sqrt(np.sum(w[:3] * np.mean(gt * gt, axis=1)))
check("(c) step 100, W state 3 = clip(bo^99 f_visit / scale)", d["W"][3], np.clip(fv * bo ** 99 / (scale + eps), -1, 1))
# (d) the clip
big = dict(g, W=g["W"] * 0.0 + 1.0, Q=g["Q"] * 0.0, b=g["b"] * 0.0)
occ = jnp.array([0.97, 0.01, 0.01, 0.01]); occ_w = occ
big = dict(big, W=big["W"].at[0].set(0.0))
d, _ = tr.update(big, tr.init(params), occ, 0.0)
check("(d) clip at 1 per element", d["W"][1:], np.ones((3, n)))
print("unit checks:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF

set -x
run ${D}/train_latent.py --name smoke16_m14_P128_S64jk_${J} ${P128} --hidden-samples 64 --hidden-jackknife --max-slices 2 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke16_m14_P128_group_${J} ${P128} --hidden-norm group --max-slices 2 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke16_m14_P8_state_${J} ${P8} --hidden-norm state --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke16_m14_P8_occ_${J} ${P8} --hidden-norm occupancy --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke16_m14_P8_S16jk_occ_${J} ${P8} --hidden-norm occupancy --hidden-samples 16 --hidden-jackknife --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke16_m14_P128_state_${J} ${P128} --hidden-norm state --max-slices 3 || rc=$((rc | 8))
run ${D}/train_latent.py --name smoke16_m14_P128_S64jk_state_${J} ${P128} --hidden-norm state --hidden-samples 64 --hidden-jackknife --max-slices 3 || rc=$((rc | 8))
run ${D}/train_latent.py --name smoke16_m14_P128_occ_${J} ${P128} --hidden-norm occupancy --max-slices 3 || rc=$((rc | 8))
run ${D}/train_latent.py --name smoke16_m14_P128_S64jk_occ_${J} ${P128} --hidden-norm occupancy --hidden-samples 64 --hidden-jackknife --max-slices 3 || rc=$((rc | 8))
set +x
echo "== 1. default and group paths: validation loss of slices 0-1, new vs the production runs (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke16_m14_P128_S64jk_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_S64jk/train_log.jsonl | head -2)
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke16_m14_P128_group_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_groupnorm/train_log.jsonl | head -2)
echo "== 2. smokes (Potts 8 + both)"
for r in P8_state P8_occ P8_S16jk_occ; do
  echo "-- ${r}"
  grep -o -E '"(slice|val_loss|W_norm|Q_norm|eff_states)": [-0-9.e]+' ${OUT}/smoke16_m14_${r}_${J}/train_log.jsonl | paste - - - - -
  grep -o -E '"(hidden_norm|occ_beta)": ("[a-z]*"|[0-9.]+)' ${OUT}/smoke16_m14_${r}_${J}/config.json | tr '\n' ' '; echo
done
echo "== 3. Potts 128 + both, slices 0-2: state and occupancy (exact; S = 64 jk) vs per-element and group (production)"
for r in smoke16_m14_P128_state_${J} smoke16_m14_P128_occ_${J} mnist14_K128_ema_potts_quad_local_groupnorm \
         smoke16_m14_P128_S64jk_state_${J} smoke16_m14_P128_S64jk_occ_${J} mnist14_K128_ema_potts_quad_local_S64jk \
         mnist14_K128_ema_potts_quad_local_S64jk_groupnorm; do
  echo "-- ${r}"
  head -3 ${OUT}/${r}/train_log.jsonl | grep -o -E '"(slice|val_loss|eff_states|mean_max_p|W_norm|Q_norm)": [-0-9.e]+' | paste - - - - - -
done
echo "(per-element exact: its logged eff_states omit the spring term; test 15 smoke of the first occupancy design: |W| 2.23 / 3.90, |Q| 3.20 / 6.40 at slices 1 / 2)"
echo "Finished at $(date) (exit code ${rc}: bits 1 unit checks, 2 regression, 4 smokes, 8 Potts 128)"
exit ${rc}
