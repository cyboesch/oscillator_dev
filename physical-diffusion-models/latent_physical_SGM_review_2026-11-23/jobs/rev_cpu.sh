#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=rev_cpu
#SBATCH --output=logs/rev_cpu_%j.out
#SBATCH --error=logs/rev_cpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --exclude=neu306

# REVIEW COPY: one GPU, runs any script of latent_physical_SGM_review_2026-11-23/ with its arguments.
#   sbatch jobs/rev_cpu.sh inception_fid.py --problem celeba32g LABEL=PATH ...
set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date); args: $*"
REVNAME=latent_physical_SGM_review_2026-11-23
[ "$(basename "$(pwd)")" = "${REVNAME}" ] || { echo "Error: submit from ${REVNAME}/"; exit 1; }
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found"; exit 1; }
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export GIT_COMMIT="review-copy-no-git"
export BOND_CHECK_NB=${BOND_CHECK_NB:-4}
set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/${REVNAME}/"$@"
rc=$?
set +x
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
