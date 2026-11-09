#!/bin/bash
# Login-node helper (only calls sbatch): sample 500 base trajectories (reverse SDE) from each finished full-size run,
# then evaluate them together with the paper baseline (job 4187940: same initial states and Brownian keys).
#   ./submit_sample_eval.sh LABEL RUN [RUN ...] [-- PAIR ...]
# e.g. ./submit_sample_eval.sh main K0 K1 K1_lr2e-3 cls0 cls1 -- K1,K0 K1_lr2e-3,K0 K0,paper
# A union of the two one-class models is added automatically when both cls0 and cls1 are listed.
set -euo pipefail
cd "$(dirname "$0")"
LABEL=$1; shift
RUNS=(); PAIRS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do RUNS+=("$1"); shift; done
[ $# -gt 0 ] && shift
while [ $# -gt 0 ]; do PAIRS+=(--pair "$1"); shift; done
OUT=../out/latent_physical_SGM
DEPS=""
SETS=("paper=../diagnostics/sde_split_control/4187940/final_states_sde_unsplit.npy")
for R in "${RUNS[@]}"; do
  [ -f ${OUT}/${R}/DONE ] || { echo "run ${R} not finished"; exit 1; }
  if [ ! -f ${OUT}/${R}/samples/sde_base_n500.npy ]; then
    J=$(sbatch --parsable run_sample_latent_neuronic.sh --run ${R} --set base --n 500)
    DEPS="${DEPS}:${J}"; echo "sampling ${R}: job ${J}"
  fi
  SETS+=("${R}=${R}/samples/sde_base_n500.npy")
done
if [[ " ${RUNS[*]} " == *" cls0 "* && " ${RUNS[*]} " == *" cls1 "* ]]; then
  SETS+=("cls0+cls1=cls0/samples/sde_base_n500.npy+cls1/samples/sde_base_n500.npy")
fi
DEP_ARG=""; [ -n "${DEPS}" ] && DEP_ARG="--dependency=afterok${DEPS}"
J=$(sbatch --parsable ${DEP_ARG} --job-name=latent_eval_${LABEL} run_evaluate_neuronic.sh "${SETS[@]}" "${PAIRS[@]}")
echo "evaluation ${LABEL}: job ${J} (after${DEPS:-: nothing})"
