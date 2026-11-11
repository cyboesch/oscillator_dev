PI presentation: Physical diffusion models with fast hidden units
11 November 2026

Open pi_presentation.pdf. The main talk has 19 slides, followed by six backup
slides. Speaker notes are in speaker_notes.pdf, with one note page per slide.

MOVIE
The 20-second MP4 is embedded as a PDF attachment on the movie slide and is also
provided separately at media/potts-state-probabilities.mp4. PDF viewers vary in
attachment/launch support. If clicking does not open it, play the MP4 directly
in your usual video player. The following slide provides a static overview.
Keep the media directory alongside the PDF for the companion-movie link.

BUILD
Run: bash build.sh
Requires pdflatex, latexmk, Beamer, lmodern, TikZ and attachfile2 (TeX Live).
The source and every figure/movie dependency are included in this folder.
Building the slides does not read checkpoints or run training/sampling.

DATA AND FIGURES
Existing scientific PDFs were copied from ../figures and cropped within LaTeX
for legible landscape slides. The originals were not edited. The overview is a
schematic; its image sequence is not research data. The current Potts run also
contains additional local Ising units, explicitly described in the slides.

Sample galleries use the first 100 unfiltered images of each already-saved
base-2000 sample set. The extra build_sample_galleries.py script and its JSON
provenance describe those read-only inputs. It is not required to rebuild the
presentation from the supplied gallery PDFs.

The primary diagnostic run is:
mnist28_d0123456789_K1536_ema_potts_quad_local_quartic_r1_lin_pooled

Parameter schedules are functions of generation time s=5-t, not optimizer
history. The force statistics concern all 2000 final samples. The movie is one
real reverse-SDE trajectory showing exact conditional probabilities; no
discrete hidden-state kinetics or switching rates were simulated. The ten-digit
examples use exact conditional-statistics training, not finite hidden reads.

The source theory is ../../latent_physical_SGM_writeup/theory.tex.
