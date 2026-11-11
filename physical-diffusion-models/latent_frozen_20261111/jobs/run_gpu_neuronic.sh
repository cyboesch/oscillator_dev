#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_gpu
#SBATCH --output=logs/latent_gpu_%j.out
#SBATCH --error=logs/latent_gpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# One GPU. Runs any script of this folder with its arguments, e.g.
#   sbatch jobs/run_gpu_neuronic.sh cd1_check.py --run mnist14_K128_ema_potts_quad_local
# Writes physical-diffusion-models/out/latent_physical_SGM/<script-specific>/<job>/. Submit from latent_physical_SGM/.

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date); args: $*"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
export MPLBACKEND=Agg
set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_frozen_20261111/"$@"
rc=$?
set +x
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
