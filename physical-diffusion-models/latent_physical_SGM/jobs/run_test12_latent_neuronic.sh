#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test12
#SBATCH --output=logs/latent_test12_%j.out
#SBATCH --error=logs/latent_test12_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Twelfth validation round, after adding --hidden-replicates R (mean of R independent S-draw estimates, a diagnostic
# that keeps the bias of S draws at the variance of R S draws) and --hidden-samples-late / --hidden-samples-switch
# (another draw count from a given slice on) (mnist14):
#   1. the measured-statistics path is unchanged: Potts 128 + both, S = 64 with the jackknife, slices 0-1 at full
#      steps; the validation losses must equal those of mnist14_K128_ema_potts_quad_local_S64jk;
#   2. gradient check on the trained Potts 128 + both: S = 64 and 256 (jackknife) against 64x4 and 16x16; the
#      error of one estimate should fall about as 1/sqrt(R) at fixed S;
#   3. smoke training with replicates (S = 16, jackknife, R = 4; Potts 8 + both, 3 slices at 2% of the steps);
#   4. smoke training with a draw-count switch (S = 64 -> 16 from slice 2, jackknife; 4 slices at 2%): the per-slice
#      log must show hidden_samples 64, 64, 16, 16;
#   5. invalid combinations must be refused (exit code != 0).
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test12_latent_neuronic.sh

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
P128="--problem mnist14 --hidden potts --K 128 --quad --local --select ema"
P8="--problem mnist14 --hidden potts --K 8 --quad --local --select ema"

rc=0
set -x
run ${D}/train_latent.py --name smoke12_m14_P128_S64jk_${J} ${P128} --hidden-samples 64 --hidden-jackknife --max-slices 2 || rc=$((rc | 1))
run ${D}/cd1_check.py --run mnist14_K128_ema_potts_quad_local --skip-cd1 --samples 64,256,64x4,16x16 --samples-jk --draws 16 --slices 15,25,50,99 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke12_m14_P8_S16jkR4_${J} ${P8} --hidden-samples 16 --hidden-jackknife --hidden-replicates 4 --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke12_m14_P8_S64to16jk_${J} ${P8} --hidden-samples 64 --hidden-jackknife --hidden-samples-late 16 --hidden-samples-switch 2 --max-slices 4 --steps-scale 0.02 || rc=$((rc | 8))
bad=0
run ${D}/train_latent.py --name smoke12_bad1_${J} ${P8} --hidden-replicates 2 --max-slices 1 && bad=1
run ${D}/train_latent.py --name smoke12_bad2_${J} ${P8} --hidden-samples 64 --hidden-samples-switch 3 --max-slices 1 && bad=1
run ${D}/train_latent.py --name smoke12_bad3_${J} ${P8} --hidden-samples 64 --hidden-jackknife --hidden-samples-late 15 --hidden-samples-switch 2 --max-slices 1 && bad=1
[ ${bad} = 0 ] || rc=$((rc | 16))
set +x
echo "== 1. measured-statistics path: validation loss of slices 0-1, new vs the production run (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke12_m14_P128_S64jk_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_S64jk/train_log.jsonl | head -2)
echo "== 3. replicates smoke"
grep -o -E '"(slice|val_loss|W_norm|Q_norm|eff_states)": [-0-9.e]+' ${OUT}/smoke12_m14_P8_S16jkR4_${J}/train_log.jsonl | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g'
echo
echo "== 4. draw-count switch smoke (hidden_samples per slice must read 64 64 16 16)"
grep -o -E '"(slice|hidden_samples|val_loss)": [-0-9.e]+' ${OUT}/smoke12_m14_P8_S64to16jk_${J}/train_log.jsonl | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g'
echo
echo "Finished at $(date) (exit code ${rc}: bits 1 regression, 2 gradient check, 4 replicates smoke, 8 switch smoke, 16 a bad combination was accepted)"
exit ${rc}
