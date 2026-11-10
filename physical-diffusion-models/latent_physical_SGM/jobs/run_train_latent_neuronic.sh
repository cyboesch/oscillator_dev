#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_train
#SBATCH --output=logs/latent_train_%j.out
#SBATCH --error=logs/latent_train_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Full training (the paper's 100-slice schedule) with K hidden units; arguments are passed to train_latent.py, e.g.
#   sbatch jobs/run_train_latent_neuronic.sh --name K1 --K 1
#   sbatch jobs/run_train_latent_neuronic.sh --name cls0 --K 0 --cls 0
# Resubmitting with the same arguments resumes from the last finished slice.
# Writes physical-diffusion-models/out/latent_physical_SGM/<name>/. Submit from latent_physical_SGM/.

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date); args: $*"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
export GIT_COMMIT="$(git -C "${HOST_PROJECTS_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_physical_SGM/train_latent.py "$@"
rc=$?
set +x
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
