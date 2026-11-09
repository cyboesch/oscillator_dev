#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latex_theory
#SBATCH --output=compile_%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
cd /n/fs/cyrills-part/oscillator_dev/physical-diffusion-models/latent_physical_SGM_writeup
latexmk -pdf -interaction=nonstopmode -halt-on-error theory.tex
echo "exit $?"
