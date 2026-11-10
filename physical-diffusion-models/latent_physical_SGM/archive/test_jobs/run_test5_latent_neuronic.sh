#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test5
#SBATCH --output=logs/latent_test5_%j.out
#SBATCH --error=logs/latent_test5_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Fifth validation round, after adding the squared-pixel coupling of the hidden units (--quad):
#   test_latent.py (all checks, including --quad for Ising and Potts units);
#   3-slice smoke training: mnist28 K1 + quad;
#   end-to-end on mnist14, Potts (8 states) + quad + local units: 100 slices at 1% of the steps, 16 trajectories.
# Writes physical-diffusion-models/out/latent_physical_SGM/{tests/<job>, smoke5_*_<job>, eval/<job>}/
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test5_latent_neuronic.sh

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
run ${D}/train_latent.py --name smoke5_m28_K1_quad_${J} --K 1 --quad --select ema --max-slices 3 --steps-scale 0.02 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke5_m14_P8_quad_local_full_${J} --problem mnist14 --hidden potts --K 8 --quad --local --select ema --steps-scale 0.01 || rc=$((rc | 8))
run ${D}/sample_latent.py --run smoke5_m14_P8_quad_local_full_${J} --n 16 --chunk 16 || rc=$((rc | 16))
set +x
echo "Finished at $(date) (exit code ${rc}: bits 1 checks, 2 quad smoke, 8 end-to-end training, 16 sampling)"
exit ${rc}
