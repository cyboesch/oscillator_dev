#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test13
#SBATCH --output=logs/latent_test13_%j.out
#SBATCH --error=logs/latent_test13_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

# Thirteenth validation round, after adding --hidden-floor (homeostatic revival of unvisited Potts states) and the
# per-slice diagnostic states_below_1e-6 (mnist14):
#   1. the measured-statistics path is unchanged: Potts 128 + both, S = 64 with the jackknife, slices 0-1 at full
#      steps; the validation losses must equal those of mnist14_K128_ema_potts_quad_local_S64jk;
#   2. function: Potts 128 + both with S = 4 plug-in statistics, which loses most states, for 26 slices at 10% of the
#      steps, without and with --hidden-floor 2; the floor must keep more states (states_below_1e-6, eff_states);
#   3. invalid combinations must be refused (exit code != 0).
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test13_latent_neuronic.sh

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

rc=0
set -x
run ${D}/train_latent.py --name smoke13_m14_P128_S64jk_${J} ${P128} --hidden-samples 64 --hidden-jackknife --max-slices 2 || rc=$((rc | 1))
run ${D}/train_latent.py --name smoke13_m14_P128_S4_${J} ${P128} --hidden-samples 4 --max-slices 26 --steps-scale 0.1 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke13_m14_P128_S4_floor2_${J} ${P128} --hidden-samples 4 --hidden-floor 2 --max-slices 26 --steps-scale 0.1 || rc=$((rc | 2))
bad=0
run ${D}/train_latent.py --name smoke13_bad1_${J} ${P128} --hidden-floor 2 --max-slices 1 && bad=1
run ${D}/train_latent.py --name smoke13_bad2_${J} --problem mnist14 --hidden ising --K 1 --hidden-samples 4 --hidden-floor 2 --max-slices 1 && bad=1
[ ${bad} = 0 ] || rc=$((rc | 4))
set +x
echo "== 1. measured-statistics path: validation loss of slices 0-1, new vs the production run (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke13_m14_P128_S64jk_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_S64jk/train_log.jsonl | head -2)
for r in S4 S4_floor2; do
  echo "== 2. ${r}: slice, effective states, states below 1e-6, validation loss"
  grep -o -E '"(slice|eff_states|states_below_1e-6|val_loss)": [-0-9.e]+' ${OUT}/smoke13_m14_P128_${r}_${J}/train_log.jsonl \
    | tr '\n' ' ' | sed 's/"slice"/\n"slice"/g' | awk 'NR % 5 == 1 || NR > 22'
  echo
done
echo "Finished at $(date) (exit code ${rc}: bits 1 regression, 2 floor runs, 4 a bad combination was accepted)"
exit ${rc}
