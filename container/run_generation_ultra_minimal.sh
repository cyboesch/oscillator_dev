#!/bin/bash
#SBATCH --job-name=MNIST_generation_ultra_minimal
#SBATCH --output=MNIST_generation_%j.out
#SBATCH --error=MNIST_generation_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

echo "Starting ultra-minimal job..."
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"

# Set memory limits
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_MMAP_MAX_=65536

# JAX memory settings
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu

# Container settings
IMAGE="container_image.sif"
BIND_DIR="/n/fs/cyrills-part/oscillator_dev/physical-diffusion-models"
RUNTIME_ARGS="--nv --bind ${BIND_DIR}:/physical-diffusion-models"

# Check if image exists
if [ ! -f "$IMAGE" ]; then
    echo "Error: Container image $IMAGE not found"
    exit 1
fi

# Check bind directory
echo "Resolved host bind directory: $BIND_DIR"
ls -la "$IMAGE"
ls -la "${BIND_DIR}/scripts/duffing_network_learning_MNIST_sde_ultra_minimal.py"

# Set the payload
PAYLOAD="python3 -u /physical-diffusion-models/scripts/duffing_network_learning_MNIST_sde_ultra_minimal.py"

# Run the container
set -x
apptainer exec ${RUNTIME_ARGS} ${IMAGE} ${PAYLOAD}
