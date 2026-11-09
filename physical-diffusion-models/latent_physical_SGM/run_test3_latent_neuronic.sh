#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test3
#SBATCH --output=latent_test3_%j.out
#SBATCH --error=latent_test3_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Third validation round, after adding Potts hidden units, --visible quartic and sampler snapshots:
#   test_latent.py (all checks, including the Potts unit);
#   3-slice smoke trainings: mnist28 Potts (8 states), mnist28 K1 quartic;
#   end-to-end on gauss2_8 with a 4-state Potts unit: 100 slices at 1% of the steps, 16 trajectories with snapshots.
# Writes physical-diffusion-models/out/latent_physical_SGM/{tests/<job>, smoke3_*_<job>, eval/<job>}/
# Submit from this folder:   cd physical-diffusion-models/latent_physical_SGM && sbatch run_test3_latent_neuronic.sh

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
run ${D}/train_latent.py --name smoke3_m28_potts8_${J} --hidden potts --K 8 --select ema --max-slices 3 --steps-scale 0.02 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke3_m28_K1_quartic_${J} --K 1 --visible quartic --select ema --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke3_g8_potts4_full_${J} --problem gauss2_8 --hidden potts --K 4 --select ema --steps-scale 0.01 || rc=$((rc | 8))
run ${D}/sample_latent.py --run smoke3_g8_potts4_full_${J} --n 16 --chunk 16 --snapshots || rc=$((rc | 16))
set +x
echo "Finished at $(date) (exit code ${rc}: bits 1 checks, 2 Potts smoke, 4 quartic smoke, 8 end-to-end training, 16 sampling)"
exit ${rc}
