#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test8
#SBATCH --output=logs/latent_test8_%j.out
#SBATCH --error=logs/latent_test8_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Eighth validation round, after adding --hidden-samples (force matching with sampled hidden statistics), mnist14:
#   1. gradient check on the trained Potts 128 + both run: plug-in gradients with S = 4 ... 4096 sampled hidden states
#      per training sample against the exact gradient (the error must fall towards the noise floor as S grows);
#   2. exact path unchanged: Potts 8 + quad + local, 3 slices at 1% of the steps; the validation losses must equal those
#      of slices 0-2 of smoke5_m14_P8_quad_local_full_4195804;
#   3. --hidden-samples 16: Potts 8 + quad + local, 3 slices;
#   4. --hidden-samples with --hidden-rule cd1 must be refused.
# Writes physical-diffusion-models/out/latent_physical_SGM/{cd1_check/<job>, smoke8_*_<job>}/
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test8_latent_neuronic.sh

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
run ${D}/cd1_check.py --run mnist14_K128_ema_potts_quad_local --skip-cd1 --samples 4,16,64,256,4096 --draws 16 --slices 1,50,90,99 || rc=$((rc | 1))
run ${D}/train_latent.py --name smoke8_m14_P8_exact_${J} ${P8} --max-slices 3 --steps-scale 0.01 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke8_m14_P8_S16_${J} ${P8} --hidden-samples 16 --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke8_m14_P8_bad_${J} ${P8} --hidden-samples 16 --hidden-rule cd1 --max-slices 1 && rc=$((rc | 8))
set +x
echo "== 2. exact path: validation loss of slices 0-2, new vs smoke5 (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke8_m14_P8_exact_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke5_m14_P8_quad_local_full_4195804/train_log.jsonl | head -3)
echo "== 3. S = 16: per-slice diagnostics"
grep -o -E '"(slice|lr_W|val_loss|W_norm|Wref_norm|cos_W_Wref|Q_norm|cos_wl)": [-0-9.e]+' ${OUT}/smoke8_m14_P8_S16_${J}/train_log.jsonl | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g'
echo
[ -d ${OUT}/smoke8_m14_P8_bad_${J} ] && { echo "the refused run left a directory"; rc=$((rc | 16)); }
echo "Finished at $(date) (exit code ${rc}: bits 1 gradient check, 2 exact, 4 S = 16 training, 8 bad combination not refused, 16 directory left)"
exit ${rc}
