#!/bin/bash
# Inception FID of tree runs (mirrors below this folder's out/) next to the locked run, with the locked code's
# inception_fid.py (run from ../latent_physical_SGM; its report goes to the usual out/latent_physical_SGM/fid_inception/<job>/).
# Usage (from latent_physical_SGM_tree_2026-11-25/): sbatch <gpu options> jobs/eval_tree_inc.sh PROBLEM LOCKED_RUN LABEL=MIRROR/samples/FILE.npy ...
set -uo pipefail
P=$1; LOCKED=$2; shift 2
HERE=/physical-diffusion-models/latent_physical_SGM_tree_2026-11-25/out
SETS="locked_no-J_SDE_quench1=${LOCKED}/samples/sde_base_n2000_q1.npy"
for s in "$@"; do
  v=${s#*=}
  if [[ "$v" == /physical-diffusion-models/* ]]; then        # an absolute (container) path, e.g. a run of the review folder
    if [ -f "..${v#/physical-diffusion-models}" ]; then SETS="$SETS ${s%%=*}=${v}"; else echo "missing: $s"; fi
  elif [ -f "out/$v" ]; then SETS="$SETS ${s%%=*}=${HERE}/${v}"; else echo "missing: $s (out/$v)"; fi
done
cd ../latent_physical_SGM && bash jobs/run_gpu_neuronic.sh inception_fid.py --problem $P ${SETS}
