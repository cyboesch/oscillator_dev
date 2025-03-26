#!/bin/bash
#SBATCH --job-name=duffing_mnist
#SBATCH --output=duffing_mnist_%j.out
#SBATCH --error=duffing_mnist_%j.err
#SBATCH --partition=lips
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Activate the virtual environment
source /n/fs/thermo-ai/oscillator_dev/jax-env-thermo/bin/activate

# Run your script
python duffing_network_learning_MNIST_ode_flow_sampling.py
