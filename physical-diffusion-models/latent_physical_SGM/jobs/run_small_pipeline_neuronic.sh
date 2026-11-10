#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_small
#SBATCH --output=logs/latent_small_%j.out
#SBATCH --error=logs/latent_small_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Small-problem pipeline for one problem (first argument: mnist14 or gauss2_8; optional second: run-name tag):
#   train K = 0 and K = 1, each with the paper's slice result (best) and with the EMA (the paper's full schedule);
#   sample 500 trajectories from each (reverse SDE); evaluate all four, with paired comparisons K1 vs K0.
# Resubmitting resumes unfinished trainings and skips finished ones.
# Writes physical-diffusion-models/out/latent_physical_SGM/<problem>_K{0,1}_{best,ema}<tag>/ and eval/<job>/.
# Submit from latent_physical_SGM/:   sbatch jobs/run_small_pipeline_neuronic.sh gauss2_8

set -uo pipefail
P=${1:?problem}
TAG=${2:-}
echo "Starting job on $(hostname) in $(pwd) at $(date); problem ${P} tag '${TAG}'"
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

rc=0
set -x
for SEL in best ema; do
  for K in 0 1; do
    run ${D}/train_latent.py --name ${P}_K${K}_${SEL}${TAG} --problem ${P} --K ${K} --select ${SEL} || rc=$((rc | 1))
  done
done
for SEL in best ema; do
  for K in 0 1; do
    R=${P}_K${K}_${SEL}${TAG}
    [ -f ${OUT}/${R}/samples/sde_base_n500.npy ] || run ${D}/sample_latent.py --run ${R} --n 500 --chunk 250 || rc=$((rc | 2))
  done
done
run ${D}/evaluate_samples.py --problem ${P} \
  "K0_best=${P}_K0_best${TAG}/samples/sde_base_n500.npy" "K1_best=${P}_K1_best${TAG}/samples/sde_base_n500.npy" \
  "K0_ema=${P}_K0_ema${TAG}/samples/sde_base_n500.npy" "K1_ema=${P}_K1_ema${TAG}/samples/sde_base_n500.npy" \
  --pair K1_best,K0_best --pair K1_ema,K0_ema --pair K0_ema,K0_best || rc=$((rc | 4))
run ${D}/summarize_training.py ${P}_K0_best${TAG} ${P}_K1_best${TAG} ${P}_K0_ema${TAG} ${P}_K1_ema${TAG} --ref ${P}_K0_best${TAG} || rc=$((rc | 8))
set +x
echo "Finished at $(date) (exit code ${rc}: bits 1 training, 2 sampling, 4 evaluation, 8 summary)"
exit ${rc}
