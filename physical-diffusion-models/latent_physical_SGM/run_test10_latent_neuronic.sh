#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test10
#SBATCH --output=latent_test10_%j.out
#SBATCH --error=latent_test10_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Tenth validation round, after adding the jackknife bias correction of the sampled hidden statistics (mnist14):
#   1. exact path unchanged after restructuring the loss: Potts 8 + quad + local, 3 slices at 1% of the steps; validation
#      losses equal to slices 0-2 of smoke5_m14_P8_quad_local_full_4195804;
#   2. gradient bias of the plug-in and the jackknife estimates, S = 4, 16, 64, 64 draws, on the trained Potts 128 + both;
#   3. --hidden-samples 16 --hidden-jackknife: Potts 8 + quad + local, 3 slices.
# Submit from this folder:   cd physical-diffusion-models/latent_physical_SGM && sbatch run_test10_latent_neuronic.sh

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
P8="--problem mnist14 --hidden potts --K 8 --quad --local --select ema"

rc=0
set -x
run ${D}/train_latent.py --name smoke10_m14_P8_exact_${J} ${P8} --max-slices 3 --steps-scale 0.01 || rc=$((rc | 1))
run ${D}/cd1_check.py --run mnist14_K128_ema_potts_quad_local --skip-cd1 --samples 4,16,64 --draws 64 --slices 1,50,90,99 || rc=$((rc | 2))
run ${D}/cd1_check.py --run mnist14_K128_ema_potts_quad_local --skip-cd1 --samples 4,16,64 --draws 64 --slices 1,50,90,99 --samples-jk || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke10_m14_P8_S16jk_${J} ${P8} --hidden-samples 16 --hidden-jackknife --max-slices 3 --steps-scale 0.02 || rc=$((rc | 8))
set +x
echo "== 1. exact path: validation loss of slices 0-2, new vs smoke5 (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke10_m14_P8_exact_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke5_m14_P8_quad_local_full_4195804/train_log.jsonl | head -3)
echo "== 3. jackknife smoke"
grep -o -E '"(slice|val_loss|W_norm|Q_norm)": [-0-9.e]+' ${OUT}/smoke10_m14_P8_S16jk_${J}/train_log.jsonl | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g'
echo
echo "Finished at $(date) (exit code ${rc}: bits 1 exact, 2 plug-in check, 4 jackknife check, 8 jackknife training)"
exit ${rc}
