#!/bin/bash
# Round 5 (2026-11-24 02:55): combinations around the bond units (best so far: 18.4-18.5 at quench 0.4).
set -euo pipefail
COMMON="--hidden potts --quad --local --visible quartic --radius 1 --proto-init pooled --seed 1 --val-every 1000 --visible-impl stencil --precision mixed --ref-kmeans gpu --patch-init-scale 0.06 --patch-on-slice 10 --patch2-on-slice 55 --save-f32 --patch 16:4:64 --patch2 8:2:64"
PRE="celeba32g_K16_exact_locked-patches-64states"
sub() { # NAME TIME RUN "FLAGS" NICE
  local id; id=$(sbatch --parsable --time="$2" --mem=64G --nice="$5" --job-name="r5_$1" jobs/rev_pipeline.sh celeba32g "$3" "${COMMON} $4" 16 2000 "--quench=0.2,0.3,0.4,0.5,0.7,1")
  echo "$id $3" | tee -a jobs/round5_ids.txt
}
sub U1 04:00:00 "${PRE}_plus-spring-units-2states-from-t2.22_4x-training-steps-in-slices-85-to-99_seed1" "--pair-terms lin --bond 2 --bond-on-slice 55 --bond-init-scale 0.05 --steps-schedule 85:4" 0
sub U5 03:30:00 "${PRE}_plus-spring-units-4states-from-t2.22_no-plain-visible-springs_seed1" "--pair-terms none --bond 4 --bond-on-slice 55 --bond-init-scale 0.05" 4
sub U3 03:30:00 "${PRE}_plus-spring-units-2states-from-t0.96_seed1" "--pair-terms lin --bond 2 --bond-on-slice 80 --bond-init-scale 0.05" 5
