#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test9
#SBATCH --output=logs/latent_test9_%j.out
#SBATCH --error=logs/latent_test9_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Ninth validation round, after adding the digit-subset problems mnist14_d<digits> / mnist28_d<digits>:
#   1. mnist14_d01 must reproduce mnist14 exactly: Potts 8 + quad + local, 3 slices at 1% of the steps; validation
#      losses equal to slices 0-2 of smoke5_m14_P8_quad_local_full_4195804;
#   2. mnist14_d14 (digits 1 and 4) end to end: K = 0, all 100 slices at 1% of the steps, 64 samples, evaluation;
#   3. three classes (mnist14_d147): Potts 6 + quad + local with per-class (2 per class) and pooled prototypes, 3 slices.
# Writes physical-diffusion-models/out/latent_physical_SGM/{smoke9_*_<job>, eval/<job>}/
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test9_latent_neuronic.sh

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
run ${D}/train_latent.py --name smoke9_m14d01_P8_${J} --problem mnist14_d01 --hidden potts --K 8 --quad --local --select ema --max-slices 3 --steps-scale 0.01 || rc=$((rc | 1))
run ${D}/train_latent.py --name smoke9_m14d14_K0_${J} --problem mnist14_d14 --K 0 --select ema --steps-scale 0.01 || rc=$((rc | 2))
run ${D}/sample_latent.py --run smoke9_m14d14_K0_${J} --n 64 --chunk 64 || rc=$((rc | 4))
run ${D}/evaluate_samples.py --problem mnist14_d14 "K0smoke=smoke9_m14d14_K0_${J}/samples/sde_base_n64.npy" || rc=$((rc | 8))
run ${D}/train_latent.py --name smoke9_m14d147_P6pc_${J} --problem mnist14_d147 --hidden potts --K 6 --quad --local --select ema --max-slices 3 --steps-scale 0.01 || rc=$((rc | 16))
run ${D}/train_latent.py --name smoke9_m14d147_P6pool_${J} --problem mnist14_d147 --hidden potts --K 6 --quad --local --proto-init pooled --select ema --max-slices 3 --steps-scale 0.01 || rc=$((rc | 32))
set +x
echo "== 1. mnist14_d01 vs mnist14 (smoke5): validation loss of slices 0-2 (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke9_m14d01_P8_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke5_m14_P8_quad_local_full_4195804/train_log.jsonl | head -3)
echo "== 2. mnist14_d14 evaluation"
sed -n '1,12p' ${OUT}/eval/${J}/report.txt
echo "== 3. three classes: prototype references"
grep -h -m1 'prototype reference' ${OUT}/smoke9_m14d147_P6pc_${J}/train_log.txt ${OUT}/smoke9_m14d147_P6pool_${J}/train_log.txt | cut -c1-300
echo "Finished at $(date) (exit code ${rc}: bits 1 d01, 2 d14 training, 4 d14 sampling, 8 d14 evaluation, 16 d147 per class, 32 d147 pooled)"
exit ${rc}
