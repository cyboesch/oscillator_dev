#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_checks
#SBATCH --output=latent_checks_%j.out
#SBATCH --error=latent_checks_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=neu306

# CPU only. (1) Numerical checks of the formulas in DESIGN.md; (2) blend check of the existing samples.
# Read-only on all inputs. Writes physical-diffusion-models/out/diagnostics/{latent_checks,blend_check}/<job id>/
# Submit from this folder:   cd physical-diffusion-models/latent_physical_SGM && sbatch run_latent_checks_neuronic.sh

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"

IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"

ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
for s in check_hidden_unit_formulas.py blend_check.py; do
  ls -l "${HOST_PROJECTS_DIR}/latent_physical_SGM/${s}" || { echo "Error: ${s} not found"; exit 1; }
done

export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

rc=0
set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_physical_SGM/check_hidden_unit_formulas.py || rc=1
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_physical_SGM/blend_check.py || rc=$((rc + 2))
set +x
echo "Finished at $(date) (exit code ${rc}: 1 = formula checks failed to run, 2 = blend check failed to run)"
exit ${rc}
