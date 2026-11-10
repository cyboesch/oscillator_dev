#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_eval
#SBATCH --output=logs/latent_eval_%j.out
#SBATCH --error=logs/latent_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=neu306

# CPU only. Blend check and per-class pixel statistics of sample sets; arguments are passed to evaluate_samples.py:
#   sbatch jobs/run_evaluate_neuronic.sh "K0=K0/samples/sde_base_n500.npy" "K1=K1/samples/sde_base_n500.npy" --pair K1,K0
# Writes physical-diffusion-models/out/latent_physical_SGM/eval/<job>/. Submit from latent_physical_SGM/.

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
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_physical_SGM/evaluate_samples.py "$@"
rc=$?
set +x
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
