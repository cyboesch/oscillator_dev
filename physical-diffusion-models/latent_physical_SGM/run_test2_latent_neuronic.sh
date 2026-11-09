#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test2
#SBATCH --output=latent_test2_%j.out
#SBATCH --error=latent_test2_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Second validation round, after adding --problem, --select ema and per-sample validation losses:
#   test_latent.py (all checks again, plus the small problems);
#   3-slice smoke trainings: mnist28 K1 with EMA; mnist14 K0, K1; gauss2_8 K0, K1;
#   end-to-end on gauss2_8: 100 slices at 1% of the steps, then 64 sampled trajectories and their evaluation.
# Writes physical-diffusion-models/out/latent_physical_SGM/{tests/<job>, smoke2_*_<job>, eval/<job>}/
# Submit from this folder:   cd physical-diffusion-models/latent_physical_SGM && sbatch run_test2_latent_neuronic.sh

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
export MPLBACKEND=Agg
export GIT_COMMIT="$(git -C "${HOST_PROJECTS_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
D=/physical-diffusion-models/latent_physical_SGM
J=${SLURM_JOB_ID}
run() { apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u "$@"; }

rc=0
set -x
run ${D}/test_latent.py || rc=1
run ${D}/train_latent.py --name smoke2_m28_K1_ema_${J} --K 1 --select ema --max-slices 3 --steps-scale 0.02 || rc=$((rc | 2))
for P in mnist14 gauss2_8; do
  for K in 0 1; do
    run ${D}/train_latent.py --name smoke2_${P}_K${K}_${J} --problem ${P} --K ${K} --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
  done
done
run ${D}/train_latent.py --name smoke2_gauss2_8_K1_full_${J} --problem gauss2_8 --K 1 --select ema --steps-scale 0.01 || rc=$((rc | 8))
run ${D}/sample_latent.py --run smoke2_gauss2_8_K1_full_${J} --n 64 --chunk 64 || rc=$((rc | 16))
set +x
echo "Finished at $(date) (exit code ${rc}: bits 1 checks, 2 EMA smoke, 4 small-problem smokes, 8 end-to-end training, 16 sampling)"
exit ${rc}
