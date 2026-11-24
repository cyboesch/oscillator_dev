#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=fig1_quench
#SBATCH --output=quench_%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --exclude=neu306
# Quenches one final state for the conceptual figure (CPU). Reads the run and the code; writes only into
# Conceptual-fig/images/.   sbatch make_quench_frame.sh --traj 0
ROOT=/n/fs/cyrills-part/oscillator_dev
cd "$ROOT/ICML-writeup/Conceptual-fig" || exit 1
export MPLBACKEND=Agg
apptainer exec --bind "$ROOT:$ROOT" "$ROOT/container/container_image.sif" \
    python3 -u "$ROOT/ICML-writeup/Conceptual-fig/make_quench_frame.py" "$@"
echo "exit $?"
