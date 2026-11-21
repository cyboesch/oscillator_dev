#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latex_fig1
#SBATCH --output=compile_fig_%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# Compiles the conceptual figure to figure_hidden.pdf (shell escape off).
cd /n/fs/cyrills-part/oscillator_dev/ICML-writeup/Conceptual-fig
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error figure_hidden.tex
echo "exit $?"
