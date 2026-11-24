#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=rev_seq
#SBATCH --output=logs/rev_seq_%j.out
#SBATCH --error=logs/rev_seq_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# REVIEW COPY: one GPU, runs any script of latent_physical_SGM_review_2026-11-23/ with its arguments.
#   sbatch jobs/rev_seq.sh inception_fid.py --problem celeba32g LABEL=PATH ...
set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date); args: $*"
REVNAME=latent_physical_SGM_review_2026-11-23
[ "$(basename "$(pwd)")" = "${REVNAME}" ] || { echo "Error: submit from ${REVNAME}/"; exit 1; }
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
export MPLBACKEND=Agg
export GIT_COMMIT="review-copy-no-git"
set -x
rc=0; while IFS= read -r line; do [ -z "$line" ] && continue; echo "=== $line"; apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/${REVNAME}/$line < /dev/null || rc=1; done < "$1"

set +x
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
