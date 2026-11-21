#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latex_icml
#SBATCH --output=compile_%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# Compiles main.tex twice (pdflatex + bibtex via latexmk, shell escape off):
#   main.pdf           anonymous ICML submission
#   main_preprint.pdf  arXiv version with the authors shown
# Submit from anywhere:  sbatch /n/fs/cyrills-part/oscillator_dev/ICML-writeup/compile.sh
cd /n/fs/cyrills-part/oscillator_dev/ICML-writeup
PDFLATEX='pdflatex -no-shell-escape %O %P'
latexmk -pdf -interaction=nonstopmode -halt-on-error -pdflatex="$PDFLATEX" main.tex
echo "exit main: $?"
latexmk -pdf -interaction=nonstopmode -halt-on-error -pdflatex="$PDFLATEX" \
        -jobname=main_preprint -usepretex='\def\forcepreprint{1}' main.tex
echo "exit main_preprint: $?"
