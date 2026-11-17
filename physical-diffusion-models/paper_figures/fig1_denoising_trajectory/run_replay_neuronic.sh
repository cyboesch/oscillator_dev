#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=fig1_traj_replay
#SBATCH --output=/n/fs/cyrills-part/oscillator_dev/physical-diffusion-models/paper_figures/fig1_denoising_trajectory/logs/replay_%j.out
#SBATCH --error=/n/fs/cyrills-part/oscillator_dev/physical-diffusion-models/paper_figures/fig1_denoising_trajectory/logs/replay_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Replays the paper's MNIST reverse-SDE run (Fig. 4b) and saves whole trajectories.
# Submit from anywhere:  sbatch run_replay_neuronic.sh

set -euo pipefail
echo "Starting on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

IMAGE="/n/fs/cyrills-part/oscillator_dev/container/container_image.sif"
HOST_PROJECTS_DIR="/n/fs/cyrills-part/oscillator_dev/physical-diffusion-models"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"

export JAX_PLATFORMS=cuda,cpu   # cpu is needed: diffrax's progress meter uses io_callback
export MPLBACKEND=Agg

set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} \
    python3 -u /physical-diffusion-models/paper_figures/fig1_denoising_trajectory/replay_reverse_sde_trajectories.py
echo "Finished at $(date)"
