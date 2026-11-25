#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=talk_strip
#SBATCH --output=strip_%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --exclude=neu306
# Renders img/strip_c64_sampler.png for the talk (CPU, seconds). Reads existing samples; writes only into img/.
ROOT=/n/fs/cyrills-part/oscillator_dev
cd "$ROOT/ICML-writeup/presentation-krotov" || exit 1
apptainer exec --bind "$ROOT:$ROOT" "$ROOT/container/container_image.sif" \
    python3 -u "$ROOT/ICML-writeup/presentation-krotov/make_sampler_strip.py"
echo "exit $?"
