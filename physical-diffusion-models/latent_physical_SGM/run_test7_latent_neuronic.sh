#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test7
#SBATCH --output=latent_test7_%j.out
#SBATCH --error=latent_test7_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Seventh validation round, after adding --radius, --proto-init random and --hidden-rule cd1 (mnist14 throughout):
#   1. exact path unchanged: Potts 8 + quad + local, 3 slices at 1% of the steps; the validation losses must equal those
#      of slices 0-2 of smoke5_m14_P8_quad_local_full_4195804 (same seed, data stream and initialisation);
#   2. --radius 13 (all pairs) with K = 0: all 100 slices at 1% of the steps, then 16 reverse-SDE samples;
#   3. --proto-init random: Potts 8 + quad + local, 3 slices;
#   4. --hidden-rule cd1 (4 releases per sample): Potts 8 + quad + local, 3 slices; cd1_dt is logged per slice;
#   5. --proto-init random with --hidden-fixed must be refused.
# Writes physical-diffusion-models/out/latent_physical_SGM/smoke7_*_<job>/
# Submit from this folder:   cd physical-diffusion-models/latent_physical_SGM && sbatch run_test7_latent_neuronic.sh

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
P8="--problem mnist14 --hidden potts --K 8 --quad --local --select ema"

rc=0
set -x
run ${D}/train_latent.py --name smoke7_m14_P8_exact_${J} ${P8} --max-slices 3 --steps-scale 0.01 || rc=$((rc | 1))
run ${D}/train_latent.py --name smoke7_m14_K0_r13_${J} --problem mnist14 --K 0 --radius 13 --select ema --steps-scale 0.01 || rc=$((rc | 2))
run ${D}/sample_latent.py --run smoke7_m14_K0_r13_${J} --n 16 --chunk 16 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke7_m14_P8_random_${J} ${P8} --proto-init random --max-slices 3 --steps-scale 0.02 || rc=$((rc | 8))
run ${D}/train_latent.py --name smoke7_m14_P8_cd1_${J} ${P8} --hidden-rule cd1 --cd1-releases 4 --max-slices 3 --steps-scale 0.02 || rc=$((rc | 16))
run ${D}/train_latent.py --name smoke7_m14_P8_bad_${J} ${P8} --proto-init random --hidden-fixed --max-slices 1 && rc=$((rc | 32))
set +x
echo "== 1. exact path: validation loss of slices 0-2, new vs smoke5 (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke7_m14_P8_exact_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke5_m14_P8_quad_local_full_4195804/train_log.jsonl | head -3)
echo "== 2. radius 13: pairs and parameters"
grep -m2 -E 'pairs|start' ${OUT}/smoke7_m14_K0_r13_${J}/train_log.txt | cut -c1-300
ls ${OUT}/smoke7_m14_K0_r13_${J}/samples/ 2>&1
for r in smoke7_m14_P8_random_${J} smoke7_m14_P8_cd1_${J}; do
  echo "== ${r}: per-slice diagnostics"
  grep -m1 'prototype reference' ${OUT}/${r}/train_log.txt | cut -c1-400
  grep -o -E '"(slice|cd1_dt|lr_W|val_loss|W_norm|Wref_norm|cos_W_Wref|Q_norm|cos_wl)": [-0-9.e]+' ${OUT}/${r}/train_log.jsonl | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g'
  echo
done
[ -d ${OUT}/smoke7_m14_P8_bad_${J} ] && { echo "the refused run left a directory"; rc=$((rc | 64)); }
echo "Finished at $(date) (exit code ${rc}: bits 1 exact, 2 radius training, 4 radius sampling, 8 random init, 16 cd1, 32 bad combination not refused, 64 directory left)"
exit ${rc}
