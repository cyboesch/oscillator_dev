#!/bin/bash
# REVIEW COPY: Inception FID of every sample file (all read-out variants) of the given runs. bash jobs/eval_scan.sh [SBATCH_ARGS --] RUN ...
set -euo pipefail
SB=(); if [[ " $* " == *" -- "* ]]; then while [ "$1" != "--" ]; do SB+=("$1"); shift; done; shift; fi
ARGS=()
for r in "$@"; do
  lab=${r#celeba32g_K16_exact_}; lab=${lab%_seed1}; lab=${lab#locked-patches-64states_}
  for f in $(ls out/$r/samples/sde_base_n2000*.npy | sort -V); do
    v=$(basename $f .npy); v=${v#sde_base_n2000}; ARGS+=("${lab}${v:-_as-sampled}=$(pwd)/$f")
  done
done
echo "${#ARGS[@]} sets"
sbatch --parsable --time=01:00:00 --mem=32G --job-name=rev_fid "${SB[@]}" jobs/rev_gpu.sh inception_fid.py --problem celeba32g "${ARGS[@]}"
