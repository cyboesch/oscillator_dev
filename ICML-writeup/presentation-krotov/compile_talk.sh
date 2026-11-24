#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latex_talk
#SBATCH --output=compile_talk_%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# Compiles talk.tex (beamer) twice, shell escape off. Writes talk.pdf.
# Submit from anywhere:  sbatch /n/fs/cyrills-part/oscillator_dev/ICML-writeup/presentation-krotov/compile_talk.sh
cd /n/fs/cyrills-part/oscillator_dev/ICML-writeup/presentation-krotov
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error talk.tex > /dev/null
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error talk.tex | tail -n 3
echo "exit talk: $?"
