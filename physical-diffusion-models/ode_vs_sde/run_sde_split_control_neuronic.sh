#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=sde_split_ctrl
#SBATCH --output=sde_split_ctrl_%j.out
#SBATCH --error=sde_split_ctrl_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Control: plain reverse SDE (one Brownian tree on [0, 5]) on the 500 trajectories of the fine-grid noise-window run.
# Read-only on the paper checkpoint. Writes physical-diffusion-models/out/diagnostics/sde_split_control/<job id>/
# Submit from this folder:   cd physical-diffusion-models/ode_vs_sde && sbatch run_sde_split_control_neuronic.sh

set -euo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"

IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"

ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/ode_vs_sde/)"; exit 1; }
ls -l "${HOST_PROJECTS_DIR}/ode_vs_sde/sde_split_control.py" || { echo "Error: script not found"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export JAX_PLATFORMS=cuda,cpu  # cpu needed by the solver progress-bar callback
export MPLBACKEND=Agg

set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/ode_vs_sde/sde_split_control.py
echo "Finished at $(date)"
