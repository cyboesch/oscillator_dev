#!/bin/bash
# Round 2 (2026-11-23 night): the new bond ("spring") units on the locked recipe, and extra steps in the late slices.
set -euo pipefail
BASE="--hidden potts --quad --local --visible quartic --radius 1 --pair-terms lin --proto-init pooled --seed 1 --val-every 1000 --visible-impl stencil --precision mixed --ref-kmeans gpu --patch-init-scale 0.06 --patch-on-slice 10 --patch2-on-slice 55 --patch 16:4:64 --patch2 8:2:64"
PRE="celeba32g_K16_exact_locked-patches-64states"
sub() { # NAME TIME RUN "FLAGS" NICE
  local id; id=$(sbatch --parsable --time="$2" --mem=64G --nice="$5" --job-name="r2_$1" jobs/rev_pipeline.sh celeba32g "$3" "${BASE} $4" 16 2000 "--quench 1")
  echo "$id $3" | tee -a jobs/round2_ids.txt
}
sub P 04:00:00 "${PRE}_plus-spring-units-4states-from-t2.22_seed1" "--bond 4 --bond-on-slice 55 --bond-init-scale 0.05" 3
sub O 04:00:00 "${PRE}_4x-training-steps-in-slices-85-to-99_seed1" "--steps-schedule 85:4" 4
sub Q 04:00:00 "${PRE}_plus-spring-units-2states-from-t2.22_seed1" "--bond 2 --bond-on-slice 55 --bond-init-scale 0.05" 5
sub R 04:00:00 "${PRE}_plus-spring-units-4states-from-t3.99_seed1" "--bond 4 --bond-on-slice 20 --bond-init-scale 0.05" 6
