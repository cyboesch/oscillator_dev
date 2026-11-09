#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_gallery
#SBATCH --output=latent_gallery_%j.out
#SBATCH --error=latent_gallery_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=neu306

# CPU only. Image gallery (make_gallery.py): real MNIST images and unsorted samples of the main models, per problem,
# into physical-diffusion-models/out/latent_physical_SGM/gallery/<problem>/. Sets whose samples do not exist yet are
# skipped, so the job can be resubmitted as runs finish. Submit from this folder: sbatch run_gallery_neuronic.sh

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
OUT=../out/latent_physical_SGM                       # host path, to test which sample files exist
G=/physical-diffusion-models/out/latent_physical_SGM/gallery
run() { apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u /physical-diffusion-models/latent_physical_SGM/make_gallery.py "$@"; }
SETS=()
add() {   # add NAME RUN: the run's 500 base samples, if present
  local f="$2/samples/sde_base_n500.npy"
  if [ -f "${OUT}/${f}" ]; then SETS+=("$1=${f}"); else echo "skip $1 ($2: no samples yet)"; fi
}
rc=0

SETS=(paper=../diagnostics/sde_split_control/4187940/final_states_sde_unsplit.npy)
add Potts32_both P32_ema_quad_local
add Potts128_both P128_ema_quad_local
add Potts256_both P256_ema_quad_local
add Potts256_both_power4 P256_ema_quad_local_quartic
add Potts512_both_power4 P512_ema_quad_local_quartic
add Potts128_frozen P128_ema_local_fixed
add Potts128_trained_nospring P128_ema_local
add local_S64jk P128_ema_quad_local_S64jk
add local_S256jk P128_ema_quad_local_S256jk
add local_S16jk P128_ema_quad_local_S16jk
add local_S16jk_visit_count P128_ema_quad_local_S16jk_occ
add Potts512_both_power4_local_S64jk_visit_count P512_ema_quad_local_quartic_S64jk_occ
run --problem mnist28 --out ${G}/28x28_0-1_paper_vs_hidden-unit_models --title "28x28, digits 0/1" "${SETS[@]}" \
    --nearest Potts256_both,Potts128_frozen,Potts512_both_power4_local_S64jk_visit_count || rc=1

SETS=()
add K0_no_hidden mnist14_K0_ema
add Potts128_both mnist14_K128_ema_potts_quad_local
add Potts256_both mnist14_K256_ema_potts_quad_local
add local_S256jk mnist14_K128_ema_potts_quad_local_S256jk
add local_S64jk mnist14_K128_ema_potts_quad_local_S64jk
add local_S64jk_4replicates mnist14_K128_ema_potts_quad_local_S64jkR4
add local_S16jk_16replicates mnist14_K128_ema_potts_quad_local_S16jkR16
add local_S256to16jk mnist14_K128_ema_potts_quad_local_S256to16jk
add exact_groupnorm mnist14_K128_ema_potts_quad_local_groupnorm
add local_S64jk_groupnorm mnist14_K128_ema_potts_quad_local_S64jk_groupnorm
add exact_gain_per_state mnist14_K128_ema_potts_quad_local_state
add exact_visit_count mnist14_K128_ema_potts_quad_local_occ
add local_S64jk_gain_per_state mnist14_K128_ema_potts_quad_local_S64jk_state
add local_S64jk_visit_count mnist14_K128_ema_potts_quad_local_S64jk_occ
add local_S16jk mnist14_K128_ema_potts_quad_local_S16jk
add local_S16jk_visit_count mnist14_K128_ema_potts_quad_local_S16jk_occ
add local_S16_plain mnist14_K128_ema_potts_quad_local_S16
add CD1_exact_averages mnist14_K128_ema_potts_quad_local_cd1
add random_init mnist14_K128_ema_potts_quad_local_random
run --problem mnist14 --out ${G}/14x14_0-1_hidden-unit_models_and_local_training --title "14x14, digits 0/1" "${SETS[@]}" \
    --nearest Potts128_both,local_S256jk,local_S64jk_visit_count || rc=1

# the hidden-unit optimisers side by side (Potts 128 + both): per-element Adam, plain steps (group normalisation),
# one gain per state, visit-count rule; exact statistics, and measured with S = 64 and S = 16 (jackknife)
SETS=()
add exact_per_element_Adam mnist14_K128_ema_potts_quad_local
add exact_plain_steps mnist14_K128_ema_potts_quad_local_groupnorm
add exact_gain_per_state mnist14_K128_ema_potts_quad_local_state
add exact_visit_count mnist14_K128_ema_potts_quad_local_occ
add S64jk_per_element_Adam mnist14_K128_ema_potts_quad_local_S64jk
add S64jk_plain_steps mnist14_K128_ema_potts_quad_local_S64jk_groupnorm
add S64jk_gain_per_state mnist14_K128_ema_potts_quad_local_S64jk_state
add S64jk_visit_count mnist14_K128_ema_potts_quad_local_S64jk_occ
add S16jk_per_element_Adam mnist14_K128_ema_potts_quad_local_S16jk
add S16jk_visit_count mnist14_K128_ema_potts_quad_local_S16jk_occ
add S16jk2_per_element_Adam mnist14_K128_ema_potts_quad_local_S16jk2
add S16jk2_visit_count mnist14_K128_ema_potts_quad_local_S16jk2_occ
add S8jk_visit_count mnist14_K128_ema_potts_quad_local_S8jk_occ
add S4jk_visit_count mnist14_K128_ema_potts_quad_local_S4jk_occ
add S2jk_visit_count mnist14_K128_ema_potts_quad_local_S2jk_occ
add S8jk_visit_count_same_draws mnist14_K128_ema_potts_quad_local_S8jk_occsame
add S16jk_visit_count_same_draws mnist14_K128_ema_potts_quad_local_S16jk_occsame
run --problem mnist14 --out ${G}/14x14_0-1_local_training_optimisers_Potts128 --title "14x14, digits 0/1, hidden-unit optimisers" \
    "${SETS[@]}" --nearest S16jk_per_element_Adam,S16jk_visit_count || rc=1

# simpler visible model (power 4, exact training): shorter pair-coupling range (Chebyshev radius r: 8 neighbours at
# r = 1; vn1: the 4 nearest neighbours), fewer pair terms (lin: springs (x_i - x_j)^2 only; linom: + x_i^2 x_j;
# none: no pair couplings), with Potts 128 + both or without hidden units (K0)
SETS=()
add K0_r7 mnist14_K0_ema_quartic
add K0_r1 mnist14_K0_ema_quartic_r1
add K0_vn1_lin mnist14_K0_ema_quartic_vn1_lin
add Potts128_r7 mnist14_K128_ema_potts_quad_local_quartic
add Potts128_r3 mnist14_K128_ema_potts_quad_local_quartic_r3
add Potts128_r2 mnist14_K128_ema_potts_quad_local_quartic_r2
add Potts128_r1 mnist14_K128_ema_potts_quad_local_quartic_r1
add Potts128_r1_lin mnist14_K128_ema_potts_quad_local_quartic_r1_lin
add Potts128_r1_linom mnist14_K128_ema_potts_quad_local_quartic_r1_linom
add Potts128_vn1_lin mnist14_K128_ema_potts_quad_local_quartic_vn1_lin
add Potts128_vn1_linom mnist14_K128_ema_potts_quad_local_quartic_vn1_linom
add Potts128_r1_none mnist14_K128_ema_potts_quad_local_quartic_r1_none
run --problem mnist14 --out ${G}/14x14_0-1_power4_short-range_pixel_couplings_Potts128 --title "14x14, digits 0/1, power 4, simpler visible couplings" \
    "${SETS[@]}" --nearest Potts128_vn1_lin || rc=1

# number of Potts states with the simplest pixel model: power 4, springs only, 8 (r1) or 4 (vn1) nearest neighbours
SETS=()
add no_hidden_4nb mnist14_K0_ema_quartic_vn1_lin
for nb in r1 vn1; do for K in 32 128 256 512; do
  add Potts${K}_$( [ ${nb} = r1 ] && echo 8nb || echo 4nb ) mnist14_K${K}_ema_potts_quad_local_quartic_${nb}_lin
done; done
run --problem mnist14 --out ${G}/14x14_0-1_power4_springs_only_number_of_Potts_states --title "14x14, digits 0/1, power 4, nearest-neighbour springs only: number of Potts states" \
    "${SETS[@]}" --nearest Potts512_8nb,Potts512_4nb || rc=1

SETS=()
add K0_no_hidden mnist14_d012_K0_ema
add Potts128_both mnist14_d012_K128_ema_potts_quad_local_pooled
add Potts256_both mnist14_d012_K256_ema_potts_quad_local_pooled
add local_S64jk mnist14_d012_K128_ema_potts_quad_local_pooled_S64jk
add local_S256jk mnist14_d012_K128_ema_potts_quad_local_pooled_S256jk
add local_S64jk_visit_count mnist14_d012_K128_ema_potts_quad_local_pooled_S64jk_occ
run --problem mnist14_d012 --out ${G}/14x14_0-1-2_three_digits_Potts --title "14x14, digits 0/1/2" "${SETS[@]}" \
    --nearest Potts256_both || rc=1

SETS=()
add K0_no_hidden mnist14_d0123_K0_ema
add Potts256_both mnist14_d0123_K256_ema_potts_quad_local_pooled
add Potts512_both mnist14_d0123_K512_ema_potts_quad_local_pooled
add local_S64jk_visit_count mnist14_d0123_K512_ema_potts_quad_local_pooled_S64jk_occ
run --problem mnist14_d0123 --out ${G}/14x14_0-1-2-3_four_digits_Potts --title "14x14, digits 0/1/2/3" "${SETS[@]}" || rc=1

for p in 14 49; do
  SETS=()
  add K0_no_hidden mnist14_d${p}_K0_ema
  add Potts128_both mnist14_d${p}_K128_ema_potts_quad_local
  run --problem mnist14_d${p} --out ${G}/14x14_${p:0:1}-${p:1:1}_digit_pair_Potts128 --title "14x14, digits ${p:0:1}/${p:1:1}" "${SETS[@]}" || rc=1
done
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
