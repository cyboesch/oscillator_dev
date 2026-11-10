#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test6
#SBATCH --output=logs/latent_test6_%j.out
#SBATCH --error=logs/latent_test6_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Sixth validation round, after adding --hidden-fixed (hidden couplings held at each slice's closed-form reference):
#   3-slice smoke trainings on mnist14: Potts (8 states) + per-pixel units, and one Ising unit, both with --hidden-fixed;
#   the per-slice diagnostics must show cos(W, W*) = 1 and |W| = |W*| (and the same for the per-pixel units);
#   --hidden-fixed together with --quad must be refused.
# Writes physical-diffusion-models/out/latent_physical_SGM/smoke6_*_<job>/
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test6_latent_neuronic.sh

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
export GIT_COMMIT="$(git -C "${HOST_PROJECTS_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
D=/physical-diffusion-models/latent_physical_SGM
OUT=../out/latent_physical_SGM
J=${SLURM_JOB_ID}
run() { apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u "$@"; }

rc=0
set -x
run ${D}/train_latent.py --name smoke6_m14_P8_local_fixed_${J} --problem mnist14 --hidden potts --K 8 --local --hidden-fixed --select ema --max-slices 3 --steps-scale 0.02 || rc=$((rc | 1))
run ${D}/train_latent.py --name smoke6_m14_K1_fixed_${J} --problem mnist14 --K 1 --hidden-fixed --select ema --max-slices 3 --steps-scale 0.02 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke6_m14_P8_quad_fixed_${J} --problem mnist14 --hidden potts --K 8 --quad --hidden-fixed --max-slices 1 --steps-scale 0.01 && rc=$((rc | 4))
set +x
for r in smoke6_m14_P8_local_fixed_${J} smoke6_m14_K1_fixed_${J}; do
  echo "== ${r}: per-slice diagnostics"
  grep -o -E '"(slice|lr_W|lr_b|W_norm|Wref_norm|cos_W_Wref|wl_norm|wl_ref_norm|cos_wl|val_loss)": [-0-9.e]+' ${OUT}/${r}/train_log.jsonl | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g'
  echo
done
[ -d ${OUT}/smoke6_m14_P8_quad_fixed_${J} ] && { echo "the refused --quad run left a directory"; rc=$((rc | 8)); }
echo "Finished at $(date) (exit code ${rc}: bits 1 Potts + local fixed, 2 Ising fixed, 4 --quad not refused, 8 directory left)"
exit ${rc}
