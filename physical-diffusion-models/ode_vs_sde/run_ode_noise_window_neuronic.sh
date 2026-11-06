#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=ode_noise_window
#SBATCH --output=ode_noise_window_%j.out
#SBATCH --error=ode_noise_window_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Hybrid ODE/SDE samplers: where along the reverse path does noise matter?
# Read-only on the paper checkpoint. Writes physical-diffusion-models/out/diagnostics/ode_noise_window/<job id>/
# Submit from this folder:   cd physical-diffusion-models/ode_vs_sde && sbatch run_ode_noise_window_neuronic.sh

set -euo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"

IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"

ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/ode_vs_sde/)"; exit 1; }
ls -l "${HOST_PROJECTS_DIR}/ode_vs_sde/ode_noise_window.py" || { echo "Error: script not found"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export JAX_PLATFORMS=cuda,cpu  # cpu needed by the solver progress-bar callback
export MPLBACKEND=Agg

set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/ode_vs_sde/ode_noise_window.py
echo "Finished at $(date)"
