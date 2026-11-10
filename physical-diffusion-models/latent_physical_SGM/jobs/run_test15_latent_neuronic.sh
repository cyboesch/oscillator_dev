#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test15
#SBATCH --output=logs/latent_test15_%j.out
#SBATCH --error=logs/latent_test15_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Fifteenth validation round, after adding --hidden-norm occupancy (train_latent.OccupancyTransform: each Potts state
# divides a leaky average of its gradient by a leaky average of its own occupation, then one root-mean-square per
# group). mnist14, 0/1:
#   0. unit checks of OccupancyTransform on small arrays (first step against hand-computed directions; a state's
#      direction does not depend on its occupation when its gradient carries it; the one-visit floor);
#   1. the default and group paths are unchanged: Potts 128 + both, slices 0-1 at full steps, S = 64 with the
#      jackknife (element) and exact statistics (group); the validation losses must equal those of the production runs
#      mnist14_K128_ema_potts_quad_local_S64jk and mnist14_K128_ema_potts_quad_local_groupnorm;
#   2. smoke: --hidden-norm occupancy with exact statistics and with S = 16 jackknife (Potts 8 + both, 3 slices at 2%
#      of the steps): finite losses, and the log line shows hidden_norm occupancy;
#   3. occupancy on Potts 128 + both for 3 slices at full steps, exact and S = 64 jackknife, next to the production
#      runs' slices 0-2 with per-element and group normalisation (slice 0 trains no hidden parameter).
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test15_latent_neuronic.sh

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
echo "== 0. unit checks of OccupancyTransform"
run - <<'EOF' || rc=$((rc | 1))
import sys
sys.path.insert(0, "/physical-diffusion-models/latent_physical_SGM")
import numpy as np
import jax.numpy as jnp
import optax
import train_latent as tl

M, n, bo = 4, 3, 0.999
rng = np.random.default_rng(0)
params = {"theta": jnp.zeros(5), "W": jnp.zeros((M, n)), "b": jnp.zeros(M), "Q": jnp.zeros((M, n)),
          "wl": jnp.zeros(n), "bl": jnp.zeros(n)}
g = {k: jnp.asarray(rng.normal(size=v.shape)) for k, v in params.items()}
tr = tl.OccupancyTransform(("W", "b", "Q"), ("W", "b", "Q", "wl", "bl"), beta_o=bo)
ok = True
def check(name, got, want, tol=1e-10):
    global ok
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    good = err < tol
    ok &= good
    print(f"  {name}: max error {err:.2e} {'ok' if good else 'FAIL'}")

# (a) first step, exact statistics (p_unit 0): u = g / occ per state, then one rms per group
occ = jnp.array([0.5, 0.3, 0.2, 1e-40])          # state 3 below the 1e-30 floor
d, _ = tr.update(g, tr.init(params), occ, 0.0)
den = np.maximum(np.asarray(occ), 1e-30)
for k in ("W", "Q"):
    u = np.asarray(g[k]) / den[:, None]
    check(f"(a) {k} = (g/p) / rms", d[k], u / (np.sqrt(np.mean(u * u)) + 1e-8))
u = np.asarray(g["b"]) / den
check("(a) b = (g/p) / rms", d["b"], u / (np.sqrt(np.mean(u * u)) + 1e-8))
for k in ("wl", "bl"):
    gk = np.asarray(g[k])
    check(f"(a) {k} = g / rms (group)", d[k], gk / (np.sqrt(np.mean(gk * gk)) + 1e-8))
ref, _ = optax.scale_by_adam().update({"theta": g["theta"]}, optax.scale_by_adam().init({"theta": params["theta"]}))
check("(a) theta = per-element Adam", d["theta"], ref["theta"])

# (b) a gradient that carries the occupation (g_m = p_m f_m): the direction does not depend on p_m
occ = jnp.array([0.9, 1e-3, 1e-6, 1e-12])
f = {k: jnp.asarray(rng.normal(size=params[k].shape)) for k in ("W", "b", "Q")}
gp = dict(g, W=f["W"] * occ[:, None], Q=f["Q"] * occ[:, None], b=f["b"] * occ)
st = tr.init(params)
for _ in range(50):                               # constant in time: leaky averages converge to g and p exactly
    d, st = tr.update(gp, st, occ, 0.0)
for k in ("W", "Q"):
    u = np.asarray(f[k])
    check(f"(b) {k} after 50 steps = f / rms(f)", d[k], u / (np.sqrt(np.mean(u * u)) + 1e-8), 1e-9)

# (c) measured statistics, B S = 1000 reads per step: state 3 is visited once in the gradient draws at step 1 and
# never in the count draws: its direction is its gradient per visit (floor = one visit), then decays as bo^(c-1)
p_unit = 1e-3
occ = jnp.array([0.5, 0.3, 0.2, 0.0])
fv = np.asarray(rng.normal(size=n))
gv = dict(g, W=g["W"].at[3].set(jnp.asarray(fv) * p_unit))
st = tr.init(params)
_, st = tr.update(gv, st, occ, p_unit)
mu, nu, pbar, count = st
u3 = (np.asarray(mu["W"][3]) / (1 - bo)) / p_unit
check("(c) step 1: <g>/floor = gradient per visit", u3, fv)
g0 = dict(gv, W=gv["W"].at[3].set(0.0))
for _ in range(99):
    _, st = tr.update(g0, st, occ, p_unit)
mu, nu, pbar, count = st
co = 1 - bo ** int(count)
den3 = max(float(pbar[3]) / co, (1 - bo) / co * p_unit)
u3 = (np.asarray(mu["W"][3]) / co) / den3
check("(c) step 100: decays as bo^99", u3, fv * bo ** 99)
print("unit checks:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF

set -x
run ${D}/train_latent.py --name smoke15_m14_P128_S64jk_${J} ${P128} --hidden-samples 64 --hidden-jackknife --max-slices 2 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke15_m14_P128_group_${J} ${P128} --hidden-norm group --max-slices 2 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke15_m14_P8_occ_${J} ${P8} --hidden-norm occupancy --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke15_m14_P8_S16jk_occ_${J} ${P8} --hidden-norm occupancy --hidden-samples 16 --hidden-jackknife --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke15_m14_P128_occ_${J} ${P128} --hidden-norm occupancy --max-slices 3 || rc=$((rc | 8))
run ${D}/train_latent.py --name smoke15_m14_P128_S64jk_occ_${J} ${P128} --hidden-norm occupancy --hidden-samples 64 --hidden-jackknife --max-slices 3 || rc=$((rc | 8))
set +x
echo "== 1. default and group paths: validation loss of slices 0-1, new vs the production runs (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke15_m14_P128_S64jk_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_S64jk/train_log.jsonl | head -2)
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke15_m14_P128_group_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_groupnorm/train_log.jsonl | head -2)
echo "== 2. occupancy smokes"
for r in P8_occ P8_S16jk_occ; do
  grep -o -E '"(slice|val_loss|W_norm|Q_norm|eff_states)": [-0-9.e]+' ${OUT}/smoke15_m14_${r}_${J}/train_log.jsonl | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g'
  echo
  grep -o -E '"(hidden_norm|occ_beta)": ("[a-z]*"|[0-9.]+)' ${OUT}/smoke15_m14_${r}_${J}/config.json | tr '\n' ' '; echo
done
echo "== 3. Potts 128 + both, slices 0-2: occupancy (exact; S = 64 jk) vs per-element and group (production runs)"
for r in smoke15_m14_P128_occ_${J} mnist14_K128_ema_potts_quad_local_groupnorm smoke15_m14_P128_S64jk_occ_${J} \
         mnist14_K128_ema_potts_quad_local_S64jk mnist14_K128_ema_potts_quad_local_S64jk_groupnorm; do
  echo "-- ${r}"
  head -3 ${OUT}/${r}/train_log.jsonl | grep -o -E '"(slice|val_loss|eff_states|mean_max_p|W_norm|Q_norm)": [-0-9.e]+' | paste - - - - - -
done
echo "(per-element exact: its logged eff_states omit the spring term; inspect_states gives 108 at slice 5)"
echo "Finished at $(date) (exit code ${rc}: bits 1 unit checks, 2 regression, 4 occupancy smokes, 8 occupancy Potts 128)"
exit ${rc}
