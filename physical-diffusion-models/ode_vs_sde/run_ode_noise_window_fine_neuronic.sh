#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=noise_win_fine
#SBATCH --output=noise_win_fine_%j.out
#SBATCH --error=noise_win_fine_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Noise-window test on a fine grid (500 trajectories, switch points every 0.5 in tau). Resumable: export NOISE_WINDOW_OUT=/physical-diffusion-models/out/diagnostics/ode_noise_window_fine/<job id> (container path) before sbatch.
# Read-only on the paper checkpoint. Writes physical-diffusion-models/out/diagnostics/ode_noise_window_fine/<job id>/
# Submit from this folder:   cd physical-diffusion-models/ode_vs_sde && sbatch run_ode_noise_window_fine_neuronic.sh

set -euo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"

IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"

ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/ode_vs_sde/)"; exit 1; }
ls -l "${HOST_PROJECTS_DIR}/ode_vs_sde/ode_noise_window_fine.py" || { echo "Error: script not found"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export JAX_PLATFORMS=cuda,cpu  # cpu needed by the solver progress-bar callback
export MPLBACKEND=Agg

set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/ode_vs_sde/ode_noise_window_fine.py
echo "Finished at $(date)"
