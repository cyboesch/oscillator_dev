#!/bin/bash
# Round 4 (2026-11-24 02:25): bond units over range 2, and seed 2 of the baseline and of the 4-state bond-unit model.
set -euo pipefail
COMMON="--hidden potts --quad --local --visible quartic --radius 1 --pair-terms lin --proto-init pooled --val-every 1000 --visible-impl stencil --precision mixed --ref-kmeans gpu --patch-init-scale 0.06 --patch-on-slice 10 --patch2-on-slice 55 --save-f32 --patch 16:4:64 --patch2 8:2:64"
PRE="celeba32g_K16_exact_locked-patches-64states"
sub() { # NAME TIME RUN "FLAGS" NICE
  local id; id=$(sbatch --parsable --time="$2" --mem=64G --nice="$5" --job-name="r4_$1" jobs/rev_pipeline.sh celeba32g "$3" "${COMMON} $4" 16 2000 "--quench=0.2,0.3,0.4,0.5,0.7,1")
  echo "$id $3" | tee -a jobs/round4_ids.txt
}
sub T1 04:30:00 "${PRE}_plus-spring-units-4states-range2-from-t2.22_seed1" "--seed 1 --bond 4 --bond-range 2 --bond-on-slice 55 --bond-init-scale 0.05" 0
sub T4b 04:00:00 "${PRE}_plus-spring-units-4states-from-t2.22_seed2" "--seed 2 --bond 4 --bond-on-slice 55 --bond-init-scale 0.05" 1
sub T4a 04:00:00 "${PRE}_no-change-baseline-recipe_seed2" "--seed 2" 2
sub T2 04:30:00 "${PRE}_plus-spring-units-2states-range2-from-t2.22_seed1" "--seed 1 --bond 2 --bond-range 2 --bond-on-slice 55 --bond-init-scale 0.05" 3
