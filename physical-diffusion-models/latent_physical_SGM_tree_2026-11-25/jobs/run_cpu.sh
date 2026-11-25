#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=tree_cpu
#SBATCH --output=logs/tree_cpu_%j.out
#SBATCH --error=logs/tree_cpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --exclude=neu306

# CPU only. Runs a script of this folder (latent_physical_SGM_tree_2026-11-25) with its arguments, e.g.
#   sbatch jobs/run_cpu.sh test_tree.py
# The scripts import the locked code of ../latent_physical_SGM read-only and write only below this folder's out/.
# Submit from latent_physical_SGM_tree_2026-11-25/.

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date); args: $*"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM_tree_2026-11-25/)"; exit 1; }

export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_physical_SGM_tree_2026-11-25/"$@"
rc=$?
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
