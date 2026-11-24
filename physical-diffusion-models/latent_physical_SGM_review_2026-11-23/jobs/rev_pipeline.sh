#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=rev_pipe
#SBATCH --output=logs/rev_pipe_%j.out
#SBATCH --error=logs/rev_pipe_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# REVIEW COPY (2026-11-23) of latent_physical_SGM/jobs/run_pipeline_neuronic.sh: train -> sample (SDE) -> quench, one
# GPU, everything inside latent_physical_SGM_review_2026-11-23/ (code and out/). No git calls. After a finished
# training (DONE marker, history files present) the run's own slices/ folder (a duplicate of the history, needed only
# to resume) is removed to save disk; only ever inside this folder's out/.
#   sbatch jobs/rev_pipeline.sh PROBLEM RUN_NAME "TRAIN_ARGS" K N_SAMPLES "SAMPLE_ARGS" [MIN_FREE_GB]
# Submit from latent_physical_SGM_review_2026-11-23/.
set -uo pipefail
P=${1:?problem}; R=${2:?run name}; EXTRA=${3:-}; K=${4:-16}; NS=${5:-2000}; SAMPLE_ARGS=${6:-"--quench 1"}; MINFREE=${7:-20}
echo "Starting job on $(hostname) in $(pwd) at $(date); problem ${P} run '${R}' extra '${EXTRA}' K ${K} n ${NS} sample args '${SAMPLE_ARGS}'"
REVNAME=latent_physical_SGM_review_2026-11-23
[ "$(basename "$(pwd)")" = "${REVNAME}" ] || { echo "Error: submit from ${REVNAME}/"; exit 1; }
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
FREE=$(df --output=avail -BG "$(pwd)" | tail -1 | tr -dc '0-9')
echo "disk: ${FREE} G free on the share"
[ "${FREE}" -ge "${MINFREE}" ] || { echo "Error: less than ${MINFREE} G free on the share; not starting"; exit 3; }
export JAX_PLATFORMS=cuda,cpu
export MPLBACKEND=Agg
export GIT_COMMIT="review-copy-no-git"
D=/physical-diffusion-models/${REVNAME}
OUT=${HOST_PROJECTS_DIR}/${REVNAME}/out
run() { apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u "$@"; }
rc=0
set -x
run ${D}/train_latent.py --name ${R} --problem ${P} --K ${K} --select ema ${EXTRA} || rc=$((rc | 1))
set +x
RD="${OUT}/${R}"
if [ -f "${RD}/DONE" ] && [ -s "${RD}/hidden_history.npz" ] && [ -s "${RD}/params_history.npy" ] && [ -d "${RD}/slices" ]; then
  case "${RD}" in */${REVNAME}/out/*) echo "removing the duplicate ${RD}/slices (history files present)"; rm -rf -- "${RD}/slices";; esac
fi
df -h "$(pwd)" | tail -1
set -x
[ -f ${OUT}/${R}/samples/sde_base_n${NS}.npy ] && [[ "${SAMPLE_ARGS}" != *quench* || -f ${OUT}/${R}/samples/sde_base_n${NS}_q1.npy ]] || run ${D}/sample_latent.py --run ${R} --n ${NS} --chunk 250 ${SAMPLE_ARGS} || rc=$((rc | 2))
set +x
echo "Finished at $(date) (exit code ${rc}: bits 1 training, 2 sampling)"
exit ${rc}
