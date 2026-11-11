# Figure style for npj Unconventional Computing

Checked against official guidance on **11 November 2026**. Use the journal's
own instructions first; general Nature artwork guides provide supplementary
style advice. This file defines practical defaults for figures in this paper.

**Default: Arial or Helvetica, 8 pt text, bold 8 pt panel letters, strokes at
least 1 pt at final size, white canvas, RGB, editable vector PDF.**

## Journal specifications

| Item | Setting |
| --- | --- |
| Typeface | Arial or Helvetica; consistent across figures. Use a compatible symbol font for Greek letters. |
| Text size | **8 pt at final display size** is the journal's stated optimum. |
| Panel letters | Lowercase, bold, same size as other text. Use `a)`, `b)`, `c)` consistently. |
| Lines | **At least 1 pt at final size**, including axes, ticks, arrows and borders. |
| Canvas and colours | White figure canvas; RGB; accessible colours; avoid rainbow scales and red/green-only distinctions. |
| Raster images | At least 300 dpi at final size. |
| Vector artwork | Prefer editable PDF, AI, EPS or SVG; retain editable text and elements. |
| Figure files | One complete file per figure, including all panels. |
| Captions | Brief title, then panel descriptions; legend body limited to 350 words. Define symbols, colours and error bars. |

These settings come from the journal's [submission guidelines](https://www.nature.com/npjunconvcomput/for-authors-and-referees/submission-guidelines),
particularly “General figure guidelines”, “Figure legends” and “Figures for
publication”. Initial submission permits flexible formatting; final artwork
must follow the publication instructions.

## Dimensions and scaling

The journal asks for artwork sized for its intended display and fitting an A4
page width; it does **not** specify an exact 180 or 183 mm figure width on that
page. Our working sizes are **180 mm full width** and **88 mm single column**,
following the [Nature-branded research journals artwork guide](https://www.nature.com/documents/NRJs-guide-to-preparing-final-artwork.pdf).
These are project defaults, not a journal-specific dimensional mandate. Choose
height to fit the content and leave room for the caption.

The [general Nature figure guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)
also lists 89/183 mm widths. Its 5–7 pt text guidance differs from this journal's
8 pt recommendation. Use the journal-specific font and line settings above.

Create figures at their intended size before placing labels. Scaling changes
both fonts and line widths:

```text
final font size = source font size × placed width / source width
final line width = source line width × placed width / source width
```

For example, reducing an 8 pt, 183 mm figure to 180 mm gives 7.87 pt text.
Re-layout at the intended width when practical. Avoid `bbox_inches="tight"`
when the exported page must retain an exact size: it changes the canvas bounds.

## Project conventions

- Keep MNIST digits **white on black**. The surrounding figure canvas stays
  white; data-image backgrounds retain their established intensity convention.
- Compare trajectory frames using the same grayscale limits and normalization.
  Record checkpoint, seed, trajectory index and displayed times with the data.
- Preserve the native 28 × 28 pixel grid using nearest-neighbour display or
  vector pixel rectangles. Enlarging those pixels does not add image detail.
- Use dark labels, short phrases and minimal equations inside panels. Put
  assumptions and methodological detail in the caption or Methods.
- Use colour for meaning: blue `#007FA8` for Ising/visible elements, violet
  `#7559B5` for Potts elements and orange `#D56A30` for learned control. Maintain
  labels, shapes or line patterns so colour is not the only distinction.
- A red `#C43B36` enclosure labelled `T` surrounds both complete network
  alternatives in the conceptual figure: visible and hidden units share
  temperature `T`. This boundary is an annotation, not an additional coupling.
- Use 1 pt axes, ticks and secondary edges; start with 1.2 pt data curves and
  arrows. Remove decorative frames or repeated labels when space is tight.

## Matplotlib starting settings

This is a reusable preset, not a requirement to change plotting libraries.
Select an installed preferred font explicitly; the check prevents silent font
substitution. This example requires Arial to be installed.

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.findfont("Arial", fallback_to_default=False)
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 8,
    "axes.linewidth": 1,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": 1,
    "ytick.minor.width": 1,
    "lines.linewidth": 1.2,
    "patch.linewidth": 1,
    "lines.markeredgewidth": 1,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": None,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
fig, ax = plt.subplots(figsize=(180 / 25.4, 100 / 25.4), layout="constrained")
# Add the plotted data here. Set annotation/arrow widths explicitly as needed.
fig.savefig("figure.pdf")
```

Explicit `fontsize` or `linewidth` arguments override these defaults. Inspect
those overrides, colour-bar ticks and inset labels as well. The PDF/SVG font
settings follow Nature's [editable-text export advice](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
Keep the vector PDF as the submission master. This project retains PDF figure
outputs and does not generate PNG previews.

For TikZ, use `\fontsize{8bp}{9.6bp}\selectfont` for ordinary labels, add
`\bfseries` for panel letters, and use `line width=1bp` or greater. A `bp` is
1/72 inch, matching PDF/Matplotlib points. The current TeX source uses
`TeX Gyre Heros`, a Helvetica-style substitute; check the actual embedded font
and keep the family consistent across figures.

## Current conceptual figure

[conceptual_draft.pdf](conceptual_draft.pdf) currently uses a **183 × 98 mm**
canvas, **7 pt** main labels, **8 pt** panel letters, and several strokes below
1 pt. It was designed to general Nature styling before the journal-specific
instructions were checked. A final npj formatting pass should increase the
ordinary labels to 8 pt, make panel-letter formatting consistent, and bring
all visible strokes to at least 1 pt at the placed size. This README does not
change the existing artwork.

The PDF embeds TeX Gyre Heros as Type 1 fonts. Keep the editable TikZ source;
the `pdftocairo` SVG conversion turns text into glyph paths, so that SVG is
not an editable-text master. Native Matplotlib SVG output with
`svg.fonttype="none"` retains text. Prefer embedded TrueType output when
preparing a new final master, in line with the linked Nature export advice.

Rebuild commands and data provenance are in
[plotting_scripts/README.txt](../plotting_scripts/README.txt).

## Before submission

Inspect the complete figure at its intended physical size in the manuscript:
labels should fit, legends should remain readable, and every panel should
have the same font treatment. Check the exported PDF at its intended display size.

```bash
pdfinfo conceptual_draft.pdf      # page dimensions; PDF points × 25.4 / 72 = mm
pdffonts conceptual_draft.pdf     # embedded fonts; check emb=yes, avoid Type 3
pdftotext conceptual_draft.pdf -  # text should remain extractable
```

Text extraction alone does not prove full editability; inspect the master in
a vector editor if needed. Retain the plotting source and underlying data,
and follow any figure instructions supplied by the journal with the decision.
