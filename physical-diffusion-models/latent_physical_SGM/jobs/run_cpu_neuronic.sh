#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_cpu
#SBATCH --output=logs/latent_cpu_%j.out
#SBATCH --error=logs/latent_cpu_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --exclude=neu306

# CPU only. Runs any script of this folder with its arguments, e.g.
#   sbatch jobs/run_cpu_neuronic.sh coverage_samples.py <args>
# Writes physical-diffusion-models/out/latent_physical_SGM/<script-specific>/<job>/. Submit from latent_physical_SGM/.

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date); args: $*"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_physical_SGM/"$@"
rc=$?
set +x
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
