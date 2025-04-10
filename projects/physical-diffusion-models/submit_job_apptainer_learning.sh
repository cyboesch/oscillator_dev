#!/bin/bash
#
#SBATCH --account=lips
#SBATCH --job-name=MNIST_generation       # Job name
#SBATCH --output=MNIST_generation_%j.out   # Standard output file (%j = job ID)
#SBATCH --mail-user=cb7454@princeton.edu
#SBATCH --mail-type=end
#SBATCH --error=MNIST_generation_%j.err    # Standard error file
#SBATCH --time=10:00:00                    # Wall clock time limit (hh:mm:ss)
#SBATCH --partition=lips                   # Partition/queue
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                       # Request one GPU
#SBATCH --chdir=/n/fs/gen-osc/oscillator_dev/physical-diffusion-models
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Set the working directory to where you typically submit from.
# For this script, we assume this is the directory from which $(pwd)
# returns your "physical-diffusion-models" folder.

# Container execution settings
CONTAINER_RUNTIME="/usr/bin/apptainer exec"
# If container_image.sif is in the current folder, use this relative path.
# Otherwise, specify its full path.
IMAGE="container_image.sif"

# Compute the host path for your projects folder as: <current directory>/../projects
HOST_PROJECTS_DIR="$(pwd)/../projects"
RUNTIME_ARGS="--pwd /projects --nv -e --bind ${HOST_PROJECTS_DIR}:/projects"

# The payload command (this is the same as your interactive command)
PAYLOAD="python3 physical-diffusion-models/scripts/MNIST-preprocess.py"

# Debug: print the current working directory and list the projects folder,
# so you can verify the file structure in the output.
echo "Current working directory: $(pwd)"
echo "Listing projects directory: ${HOST_PROJECTS_DIR}"
ls -l "${HOST_PROJECTS_DIR}"

set -ux

# Execute the container with the payload
${CONTAINER_RUNTIME} ${RUNTIME_ARGS} ${IMAGE} ${PAYLOAD}
