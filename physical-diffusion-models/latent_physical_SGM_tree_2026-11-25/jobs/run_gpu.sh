#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=tree_gpu
#SBATCH --output=logs/tree_gpu_%j.out
#SBATCH --error=logs/tree_gpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# One GPU. Runs a script of this folder (latent_physical_SGM_tree_2026-11-25) with its arguments, e.g.
#   sbatch jobs/run_gpu.sh test_tree.py
# The scripts import the locked code of ../latent_physical_SGM read-only and write only below this folder's out/.
# Submit from latent_physical_SGM_tree_2026-11-25/.

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date); args: $*"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM_tree_2026-11-25/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
export MPLBACKEND=Agg

apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_physical_SGM_tree_2026-11-25/"$@"
rc=$?
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
