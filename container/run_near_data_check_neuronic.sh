#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=near_data_check
#SBATCH --output=near_data_check_%j.out
#SBATCH --error=near_data_check_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --exclude=neu306

# Read-only, CPU-only diagnostics of the paper checkpoint (near-data slices) and of the
# existing interpolation-ablation samples. Writes to
#   physical-diffusion-models/out/diagnostics/near_data_check/<job id>/
# Submit from container/:   cd container && sbatch run_near_data_check_neuronic.sh

set -euo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"

IMAGE="container_image.sif"
HOST_PROJECTS_DIR="$(realpath ../physical-diffusion-models)"
RUNTIME_ARGS="--bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"

ls -l "${IMAGE}" || { echo "Error: container image not found (submit from container/)"; exit 1; }
ls -l "${HOST_PROJECTS_DIR}/scripts/diagnose_near_data.py" || { echo "Error: diagnostic script not found"; exit 1; }

export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/scripts/diagnose_near_data.py
echo "Finished at $(date)"
