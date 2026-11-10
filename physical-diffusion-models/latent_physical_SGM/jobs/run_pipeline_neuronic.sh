#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_pipe
#SBATCH --output=logs/latent_pipe_%j.out
#SBATCH --error=logs/latent_pipe_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Generic train -> sample -> evaluate pipeline on one GPU.
#   sbatch jobs/run_pipeline_neuronic.sh PROBLEM TAG "EXTRA_TRAIN_ARGS" "SELECTIONS" "K_VALUES" [N_SAMPLES]
# e.g. sbatch jobs/run_pipeline_neuronic.sh mnist14 _quartic "--visible quartic" "ema" "0 1"
# Runs are named <PROBLEM>_K<K>_<SEL><TAG> (for PROBLEM mnist28: K<K>_<SEL><TAG>). For every selection rule the
# K values are compared pairwise against K = 0 (same initial states and Brownian keys).
# Resubmitting resumes unfinished trainings and skips finished trainings and samples.
# Submit from latent_physical_SGM/.

set -uo pipefail
P=${1:?problem}; TAG=${2:-}; EXTRA=${3:-}; SELS=${4:-ema}; KS=${5:-"0 1"}; NS=${6:-500}
echo "Starting job on $(hostname) in $(pwd) at $(date); problem ${P} tag '${TAG}' extra '${EXTRA}' sels '${SELS}' Ks '${KS}' n ${NS}"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
export MPLBACKEND=Agg
export GIT_COMMIT="$(git -C "${HOST_PROJECTS_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
D=/physical-diffusion-models/latent_physical_SGM
OUT=${HOST_PROJECTS_DIR}/out/latent_physical_SGM
run() { apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u "$@"; }
name() { if [ "${P}" = mnist28 ]; then echo "K${1}_${2}${TAG}"; else echo "${P}_K${1}_${2}${TAG}"; fi; }

rc=0
set -x
for SEL in ${SELS}; do for K in ${KS}; do
  run ${D}/train_latent.py --name $(name ${K} ${SEL}) --problem ${P} --K ${K} --select ${SEL} ${EXTRA} || rc=$((rc | 1))
done; done
SETS=(); PAIRS=(); RUNS=()
for SEL in ${SELS}; do for K in ${KS}; do
  R=$(name ${K} ${SEL}); RUNS+=(${R})
  [ -f ${OUT}/${R}/samples/sde_base_n${NS}.npy ] || run ${D}/sample_latent.py --run ${R} --n ${NS} --chunk 250 || rc=$((rc | 2))
  SETS+=("K${K}_${SEL}${TAG}=${R}/samples/sde_base_n${NS}.npy")
  [ "${K}" != 0 ] && [[ " ${KS} " == *" 0 "* ]] && PAIRS+=(--pair "K${K}_${SEL}${TAG},K0_${SEL}${TAG}")   # only if K = 0 is in this job
done; done
EVAL=evaluate_samples.py; [[ "${P}" =~ ^mnist(14|28)_d[0-9]{3,}$ ]] && EVAL=evaluate_multiclass.py   # three or more classes
run ${D}/${EVAL} --problem ${P} "${SETS[@]}" "${PAIRS[@]}" || rc=$((rc | 4))
run ${D}/summarize_training.py "${RUNS[@]}" --ref ${RUNS[0]} || rc=$((rc | 8))
set +x
echo "Finished at $(date) (exit code ${rc}: bits 1 training, 2 sampling, 4 evaluation, 8 summary)"
exit ${rc}
