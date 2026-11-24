#!/bin/bash
# Round 3 (2026-11-24 00:05): follow-ups on the two strongest loss signals of rounds 1-2 (late slices are step-size
# limited: 2x lr gives -2936 at t = 0; bond units give -616 at t = 0.4 with 47k parameters), before the FIDs are in.
set -euo pipefail
COMMON="--hidden potts --quad --local --visible quartic --radius 1 --pair-terms lin --proto-init pooled --seed 1 --val-every 1000 --visible-impl stencil --precision mixed --ref-kmeans gpu --patch-init-scale 0.06 --patch-on-slice 10 --patch2-on-slice 55 --save-f32"
LOCK="--patch 16:4:64 --patch2 8:2:64"
PRE="celeba32g_K16_exact_locked-patches-64states"
sub() { # NAME TIME RUN "FLAGS" NICE
  local id; id=$(sbatch --parsable --time="$2" --mem=64G --nice="$5" --job-name="r3_$1" jobs/rev_pipeline.sh celeba32g "$3" "${COMMON} $4" 16 2000 "--quench 1")
  echo "$id $3" | tee -a jobs/round3_ids.txt
}
sub S1 03:30:00 "${PRE}_plus-spring-units-4states-from-t2.22_2x-learning-rate_seed1" "${LOCK} --bond 4 --bond-on-slice 55 --bond-init-scale 0.05 --lr-scale 2" 0
sub S2 03:30:00 "${PRE}_4x-learning-rate_seed1" "${LOCK} --lr-scale 4" 1
sub S3 04:00:00 "${PRE}_2x-learning-rate_4x-training-steps-in-slices-85-to-99_seed1" "${LOCK} --lr-scale 2 --steps-schedule 85:4" 2
sub S4 05:30:00 "celeba32g_K16_exact_coarse16x16-every4px-128states-from-t4.49_fine8x8-every2px-128states-from-t2.22_plus-4x4-every2px-64states-from-t1.21_plus-spring-units-4states-from-t2.22_2x-learning-rate_seed1" "--patch 16:4:128 --patch2 8:2:128 --patch3 4:2:64 --patch3-on-slice 75 --bond 4 --bond-on-slice 55 --bond-init-scale 0.05 --lr-scale 2" 3
sub S5 03:30:00 "${PRE}_plus-spring-units-8states-from-t2.22_seed1" "${LOCK} --bond 8 --bond-on-slice 55 --bond-init-scale 0.05" 4
