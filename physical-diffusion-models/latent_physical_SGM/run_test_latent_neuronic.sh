#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test
#SBATCH --output=latent_test_%j.out
#SBATCH --error=latent_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Validation of the latent-SGM code against the paper's code (test_latent.py), then two short smoke trainings
# (3 slices at 2% of the paper's steps, K = 0 and K = 1). Read-only on the paper checkpoint.
# Writes physical-diffusion-models/out/latent_physical_SGM/{tests/<job>, smoke_K0_<job>, smoke_K1_<job>}/
# Submit from this folder:   cd physical-diffusion-models/latent_physical_SGM && sbatch run_test_latent_neuronic.sh

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

rc=0
set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u ${D}/test_latent.py || rc=1
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u ${D}/train_latent.py --name smoke_K0_${SLURM_JOB_ID} --K 0 --max-slices 3 --steps-scale 0.02 || rc=$((rc + 2))
apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u ${D}/train_latent.py --name smoke_K1_${SLURM_JOB_ID} --K 1 --max-slices 3 --steps-scale 0.02 || rc=$((rc + 4))
set +x
echo "Finished at $(date) (exit code ${rc}: 1 = checks failed, 2 = K0 smoke failed, 4 = K1 smoke failed)"
exit ${rc}
