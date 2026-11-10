#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test11
#SBATCH --output=logs/latent_test11_%j.out
#SBATCH --error=logs/latent_test11_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Eleventh validation round, after adding CD-1 with measured hidden statistics (--cd1-hidden-samples) and allowing
# --hidden-samples together with --hidden-rule cd1 (mnist14):
#   1. the CD-1 path with exact hidden averages is unchanged: Potts 128 + both, CD-1 dt 3e-3, 64 releases, slices 0-1
#      at full steps; the validation losses must equal those of mnist14_K128_ema_potts_quad_local_cd1 (slice 1:
#      -4888.736466566072);
#   2. gradient check of CD-1 with measured hidden statistics (S = 1 and 4 reads per released configuration, R * S at
#      the data) against exact averages (S = 0), dt 3e-3, R = 16 and 64, on the trained Potts 128 + both, 32 draws;
#   3. smoke training with every hidden statistic measured: --hidden-rule cd1 --cd1-hidden-samples 1 --hidden-samples 16,
#      Potts 8 + both, 3 slices at 2% of the steps.
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test11_latent_neuronic.sh

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
run ${D}/train_latent.py --name smoke11_m14_P128_cd1_${J} ${P128} --hidden-rule cd1 --cd1-dt 3e-3 --cd1-releases 64 --max-slices 2 || rc=$((rc | 1))
run ${D}/cd1_check.py --run mnist14_K128_ema_potts_quad_local --dts 3e-3 --releases 16,64 --cd1-samples 0,1,4 --draws 32 --slices 1,25,50,75,99 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke11_m14_P8_cd1_meas_${J} ${P8} --hidden-rule cd1 --cd1-dt 3e-3 --cd1-releases 4 --cd1-hidden-samples 1 --hidden-samples 16 --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
set +x
echo "== 1. CD-1 path with exact averages: validation loss of slices 0-1, new vs the production run (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke11_m14_P128_cd1_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_cd1/train_log.jsonl | head -2)
echo "== 3. measured-statistics CD-1 smoke"
grep -o -E '"(slice|val_loss|W_norm|Q_norm)": [-0-9.e]+' ${OUT}/smoke11_m14_P8_cd1_meas_${J}/train_log.jsonl | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g'
echo
grep -m1 "cd1 kappa" ${OUT}/smoke11_m14_P8_cd1_meas_${J}/train_log.txt | cut -c1-400
echo "Finished at $(date) (exit code ${rc}: bits 1 CD-1 regression, 2 gradient check, 4 measured-statistics smoke)"
exit ${rc}
