#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build pi_presentation.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build speaker_notes.tex
cp -- build/pi_presentation.pdf pi_presentation.pdf
cp -- build/speaker_notes.pdf speaker_notes.pdf
