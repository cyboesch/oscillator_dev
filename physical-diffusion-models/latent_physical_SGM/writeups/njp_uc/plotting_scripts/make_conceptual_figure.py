#!/usr/bin/env python3
"""Render the editable TikZ conceptual schematic using installed CPU tools.

Usage:
    python3 plotting_scripts/make_conceptual_figure.py

Requires pdflatex; uses only Python's standard library. No model,
training data, checkpoint, JAX, or GPU is loaded. Temporary TeX files go to the
system temporary directory. The PDF goes to the manuscript's figures directory.
If pdftocairo is available, an SVG is exported to sibling figure_data/.
All graphical content is deliberately schematic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir.parent / "figures"
    )
    args = parser.parse_args()
    if shutil.which("pdflatex") is None:
        parser.error("Required rendering tool not found: pdflatex")

    output_dir = args.output_dir.resolve()
    data_output = (output_dir.parent / "figure_data").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = script_dir / "make_conceptual_figure.tex"
    with tempfile.TemporaryDirectory(prefix="conceptual_figure_") as temp:
        work = Path(temp)
        command = [
            "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
            "-output-directory", str(work), str(source),
        ]
        result = subprocess.run(command, cwd=script_dir, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"TeX rendering failed:\n{result.stdout}\n{result.stderr}")
        # Second pass resolves the absolute current-page anchor consistently.
        subprocess.run(command, cwd=script_dir, check=True, capture_output=True, text=True)
        pdf = output_dir / "conceptual_draft.pdf"
        shutil.copy2(work / "make_conceptual_figure.pdf", pdf)

    print(pdf)
    if shutil.which("pdftocairo"):
        data_output.mkdir(parents=True, exist_ok=True)
        svg = data_output / "conceptual_draft.svg"
        subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
        print(svg)


if __name__ == "__main__":
    main()
