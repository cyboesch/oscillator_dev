#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=MNIST_analytic_gradient
#SBATCH --output=MNIST_generation_%j.out
#SBATCH --error=MNIST_generation_%j.err
#SBATCH --mail-user=cb7454@princeton.edu
#SBATCH --mail-type=end
#SBATCH --time=80:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1

echo "Starting job..."
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"

# === Apptainer Setup ===
IMAGE="container_image.sif"  # since the .sif file is in the same folder as this script
HOST_PROJECTS_DIR="$(realpath ../physical-diffusion-models)"  # resolve full path
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"

# === Debug ===
echo "Resolved host bind directory: ${HOST_PROJECTS_DIR}"
ls -l "${IMAGE}" || { echo "Error: Container image not found!"; exit 1; }
ls -l "${HOST_PROJECTS_DIR}/scripts/duffing_network_learning_MNIST_sde_memory_efficient.py" || { echo "Script not found!"; exit 1; }

# === Payload ===
PAYLOAD="python3 -u /physical-diffusion-models/scripts/duffing_network_learning_MNIST_sde_memory_efficient.py"

# === Run ===
set -ux
apptainer exec ${RUNTIME_ARGS} ${IMAGE} ${PAYLOAD}
