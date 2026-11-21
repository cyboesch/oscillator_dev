#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=fig1_frames
#SBATCH --output=frames_%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --exclude=neu306
# Renders the image frames of the conceptual figure (CPU, seconds). Reads an
# existing sample run; writes only into Conceptual-fig/images/.
#   sbatch make_frames.sh              contact sheets (to choose a trajectory)
#   sbatch make_frames.sh --traj 7     frames of trajectory 7
ROOT=/n/fs/cyrills-part/oscillator_dev
cd "$ROOT/ICML-writeup/Conceptual-fig" || exit 1
export MPLBACKEND=Agg
apptainer exec --bind "$ROOT:$ROOT" "$ROOT/container/container_image.sif" \
    python3 -u "$ROOT/ICML-writeup/Conceptual-fig/make_frames.py" "$@"
echo "exit $?"
