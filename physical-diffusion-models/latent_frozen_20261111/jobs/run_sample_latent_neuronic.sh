#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_sample
#SBATCH --output=logs/latent_sample_%j.out
#SBATCH --error=logs/latent_sample_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Samples from a finished run (or --run paper); arguments are passed to sample_latent.py, e.g.
#   sbatch jobs/run_sample_latent_neuronic.sh --run K1 --set base --n 500
# Writes physical-diffusion-models/out/latent_physical_SGM/<run>/samples/. Submit from latent_physical_SGM/.

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date); args: $*"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_frozen_20261111/sample_latent.py "$@"
rc=$?
set +x
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
