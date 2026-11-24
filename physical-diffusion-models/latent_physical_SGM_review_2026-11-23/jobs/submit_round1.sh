#!/bin/bash
# Round 1 (2026-11-23 night): capacity and optimisation variants of the locked CelebA-32 recipe, exact gradients,
# seed 1, n = 2000 SDE samples + quench 1. Baseline = the locked run (20.20), in the main out/ folder (read only).
set -euo pipefail
BASE="--hidden potts --quad --local --visible quartic --radius 1 --pair-terms lin --proto-init pooled --seed 1 --val-every 1000 --visible-impl stencil --precision mixed --ref-kmeans gpu --patch-init-scale 0.06 --patch-on-slice 10 --patch2-on-slice 55"
PRE="celeba32g_K16_exact"
sub() { # NAME K TIME MEM RUN "FLAGS" NICE  (submitted most important first; time limits tight for backfill)
  local id; id=$(sbatch --parsable --time="$3" --mem="$4" --nice="$7" --job-name="r1_$1" jobs/rev_pipeline.sh celeba32g "$5" "${BASE} $6" "$2" 2000 "--quench 1")
  echo "$id $5" | tee -a jobs/round1_ids.txt
}
sub A 16 04:30:00 64G "${PRE}_coarse16x16-every4px-128states-from-t4.49_fine8x8-every2px-128states-from-t2.22_seed1" "--patch 16:4:128 --patch2 8:2:128" 0
sub I 16 05:00:00 64G "${PRE}_locked-patches-64states_2x-training-steps_seed1" "--patch 16:4:64 --patch2 8:2:64 --steps-scale 2" 1
sub D 16 04:00:00 64G "${PRE}_locked-patches-64states_plus-second-unit-per-8x8-patch-64states-from-t2.22_seed1" "--patch 16:4:64 --patch2 8:2:64 --patch3 8:2:64 --patch3-on-slice 55" 2
sub C 16 04:00:00 64G "${PRE}_locked-patches-64states_plus-second-unit-per-16x16-patch-64states-from-t4.49_seed1" "--patch 16:4:64 --patch2 8:2:64 --patch3 16:4:64" 3
sub H 16 03:00:00 64G "${PRE}_locked-patches-64states_trained-on-clean-images_seed1" "--patch 16:4:64 --patch2 8:2:64 --data-noise 0" 4
sub G 16 08:00:00 64G "${PRE}_coarse16x16-every4px-64states-from-t4.49_fine8x8-every1px-64states-from-t2.22_seed1" "--patch 16:4:64 --patch2 8:1:64" 5
sub F 16 04:30:00 64G "${PRE}_coarse16x16-every2px-64states-from-t4.49_fine8x8-every2px-64states-from-t2.22_seed1" "--patch 16:2:64 --patch2 8:2:64" 6
sub E 16 04:00:00 64G "${PRE}_locked-patches-64states_plus-4x4-every2px-64states-from-t1.21_seed1" "--patch 16:4:64 --patch2 8:2:64 --patch3 4:2:64 --patch3-on-slice 75" 7
sub N 16 03:00:00 64G "${PRE}_locked-patches-64states_2x-learning-rate_seed1" "--patch 16:4:64 --patch2 8:2:64 --lr-scale 2" 8
sub B 16 04:00:00 64G "${PRE}_coarse16x16-every4px-64states-from-t4.49_fine8x8-every2px-128states-from-t2.22_seed1" "--patch 16:4:64 --patch2 8:2:128" 9
sub K 16 04:00:00 64G "${PRE}_locked-patches-64states_plus-12x12-every4px-64states-from-t4.49_seed1" "--patch 16:4:64 --patch2 8:2:64 --patch3 12:4:64" 10
sub L 784 07:00:00 64G "celeba32g_K784_exact_coarse16x16-every4px-128states-from-t4.49_fine8x8-every2px-128states-from-t2.22_seed1" "--patch 16:4:128 --patch2 8:2:128" 11
