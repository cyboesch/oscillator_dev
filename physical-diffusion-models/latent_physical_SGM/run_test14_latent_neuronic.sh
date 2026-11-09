#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test14
#SBATCH --output=latent_test14_%j.out
#SBATCH --error=latent_test14_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Fourteenth validation round, after adding --hidden-norm group (the hidden parameters without per-element Adam
# normalisation: one root-mean-square per group W, b, Q, wl, bl) and the diagnostic states_below_1_visit (mnist14):
#   1. the default path is unchanged: Potts 128 + both, S = 64 with the jackknife, slices 0-1 at full steps; the
#      validation losses must equal those of mnist14_K128_ema_potts_quad_local_S64jk;
#   2. smoke: --hidden-norm group with exact statistics and with S = 16 jackknife (Potts 8 + both, 3 slices at 2% of
#      the steps): finite losses, and the log line shows hidden_norm group;
#   3. smoke: --hidden-norm group with exact statistics on Potts 128 + both for 3 slices at full steps, next to the
#      production run's slices 0-2 (slice 0 trains no hidden parameter, so it must agree to rounding).
# Submit from this folder:   cd physical-diffusion-models/latent_physical_SGM && sbatch run_test14_latent_neuronic.sh

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
set -x
run ${D}/train_latent.py --name smoke14_m14_P128_S64jk_${J} ${P128} --hidden-samples 64 --hidden-jackknife --max-slices 2 || rc=$((rc | 1))
run ${D}/train_latent.py --name smoke14_m14_P8_group_${J} ${P8} --hidden-norm group --max-slices 3 --steps-scale 0.02 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke14_m14_P8_S16jk_group_${J} ${P8} --hidden-norm group --hidden-samples 16 --hidden-jackknife --max-slices 3 --steps-scale 0.02 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke14_m14_P128_group_${J} ${P128} --hidden-norm group --max-slices 3 || rc=$((rc | 4))
set +x
echo "== 1. default path: validation loss of slices 0-1, new vs the production run (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke14_m14_P128_S64jk_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_S64jk/train_log.jsonl | head -2)
grep -o '"states_below_1_visit": [0-9]*' ${OUT}/smoke14_m14_P128_S64jk_${J}/train_log.jsonl | tr '\n' ' '; echo
echo "== 2. group-norm smokes"
for r in P8_group P8_S16jk_group; do
  grep -o -E '"(slice|val_loss|W_norm|Q_norm|eff_states)": [-0-9.e]+' ${OUT}/smoke14_m14_${r}_${J}/train_log.jsonl | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g'
  echo
  grep -o '"hidden_norm": "[a-z]*"' ${OUT}/smoke14_m14_${r}_${J}/config.json
done
echo "== 3. group norm, exact statistics, Potts 128 + both, slices 0-2: new vs the production run (per-element Adam)"
paste <(grep -o -E '"(val_loss|eff_states|W_norm)": [-0-9.e]+' ${OUT}/smoke14_m14_P128_group_${J}/train_log.jsonl | paste - - -) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local/train_log.jsonl | head -3)
echo "Finished at $(date) (exit code ${rc}: bits 1 regression, 2 group-norm smokes, 4 group-norm Potts 128)"
exit ${rc}
