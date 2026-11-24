#!/bin/bash
# Round 6 (2026-11-24 03:40): final combinations around the best so far (bond units, 4 states, + 2x learning rate:
# 18.04 at quench 0.4).
set -euo pipefail
COMMON="--hidden potts --quad --local --visible quartic --radius 1 --pair-terms lin --proto-init pooled --seed 1 --val-every 1000 --visible-impl stencil --precision mixed --ref-kmeans gpu --patch-init-scale 0.06 --patch-on-slice 10 --patch2-on-slice 55 --save-f32 --patch 16:4:64 --patch2 8:2:64 --bond 4 --bond-on-slice 55 --bond-init-scale 0.05"
PRE="celeba32g_K16_exact_locked-patches-64states_plus-spring-units-4states-from-t2.22"
sub() { # NAME TIME RUN "FLAGS" NICE
  local id; id=$(sbatch --parsable --time="$2" --mem=64G --nice="$5" --job-name="r6_$1" jobs/rev_pipeline.sh celeba32g "$3" "${COMMON} $4" 16 2000 "--quench=0.3,0.4,0.5,1")
  echo "$id $3" | tee -a jobs/round6_ids.txt
}
sub V1 04:00:00 "${PRE}_2x-learning-rate_4x-training-steps-in-slices-85-to-99_seed1" "--lr-scale 2 --steps-schedule 85:4" 0
sub V2 03:30:00 "${PRE}_4x-learning-rate_seed1" "--lr-scale 4" 1
sub V3 04:00:00 "${PRE}_2x-learning-rate_plus-second-unit-per-16x16-patch-64states-from-t4.49_seed1" "--lr-scale 2 --patch3 16:4:64" 2
