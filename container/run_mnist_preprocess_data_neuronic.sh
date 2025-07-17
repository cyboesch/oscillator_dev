#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=MNIST_generation
#SBATCH --output=MNIST_generation_%j.out
#SBATCH --error=MNIST_generation_%j.err
#SBATCH --mail-user=cb7454@princeton.edu
#SBATCH --mail-type=end
#SBATCH --time=40:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1

echo "Starting job..."
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"

# === Apptainer Setup ===
IMAGE="container_image.sif"  # since the .sif file is in the same folder as this script
HOST_PROJECTS_DIR="$(realpath ../projects)"  # resolve full path
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/projects"

# === Debug ===
echo "Resolved host bind directory: ${HOST_PROJECTS_DIR}"
ls -l "${IMAGE}" || { echo "Error: Container image not found!"; exit 1; }
ls -l "${HOST_PROJECTS_DIR}/physical-diffusion-models/scripts/preprocess-mnist.py" || { echo "Script not found!"; exit 1; }

# === Payload ===
PAYLOAD="python3 -u /projects/physical-diffusion-models/scripts/preprocess-mnist.py"

# === Run ===
set -ux
apptainer exec ${RUNTIME_ARGS} ${IMAGE} ${PAYLOAD}
