#!/bin/bash
# REVIEW COPY: Inception FID (and optionally a gallery) of every finished run of this folder, with the locked baseline
# (main out/ folder, read only) as the first row. Light: only builds the argument list and calls sbatch.
#   bash jobs/eval_all.sh [SUFFIX] [SBATCH_ARGS...]     SUFFIX: sample-file suffix, default _q1 (SDE + quench 1)
set -euo pipefail
SUF=${1-_q1}; shift || true
REV=$(pwd); [ "$(basename "$REV")" = latent_physical_SGM_review_2026-11-23 ] || { echo "run from the review folder"; exit 1; }
BASE_RUN='celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1'
ARGS=("BASELINE_locked_K16_64states=/physical-diffusion-models/out/latent_physical_SGM/${BASE_RUN}/samples/sde_base_n2000${SUF}.npy")
for f in out/celeba32g_*/samples/sde_base_n2000${SUF}.npy; do
  [ -f "$f" ] || continue
  r=$(basename "$(dirname "$(dirname "$f")")"); lab=${r#celeba32g_}; lab=${lab%_seed1}
  ARGS+=("${lab}=${REV}/${f}")
done
echo "${#ARGS[@]} sets"; printf '  %s\n' "${ARGS[@]%%=*}"
sbatch --parsable --time=00:40:00 --mem=32G --job-name=rev_fid "$@" jobs/rev_gpu.sh inception_fid.py --problem celeba32g "${ARGS[@]}"
