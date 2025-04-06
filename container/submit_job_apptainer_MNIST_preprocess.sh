#!/usr/bin/env bash
#
# --- admin
#SBATCH --account=lips
#SBATCH --job-name="MNIST_generation"
#SBATCH --mail-user=cb7454@princeton.edu
#SBATCH --mail-type=end
#SBATCH --time=01:00:00
#
# --- resources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64gb
#SBATCH --output=project_space.out

set -x
apptainer exec --nv -e --pwd /project container_image.sif python3 projects/physical-diffusion-models/scripts/MNIST-preprocess.py
exit 0