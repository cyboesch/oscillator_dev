#!/usr/bin/env python3
"""Render the editable TikZ conceptual schematic using installed CPU tools.

Usage:
    python3 plotting_scripts/make_conceptual_figure.py

Requires pdflatex and pdftoppm; uses only Python's standard library. No model,
training data, checkpoint, JAX, or GPU is loaded. Temporary TeX files go to the
system temporary directory. Final files are conceptual_draft.pdf and .png in
the manuscript's figures directory. If pdftocairo is installed, also exports
conceptual_draft.svg. All graphical content is deliberately schematic.
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
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    for command in ("pdflatex", "pdftoppm"):
        if shutil.which(command) is None:
            parser.error(f"Required rendering tool not found: {command}")

    output_dir = args.output_dir.resolve()
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

    stem = output_dir / "conceptual_draft"
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", str(args.dpi), str(pdf), str(stem)],
        check=True,
    )
    outputs = [pdf, stem.with_suffix(".png")]
    if shutil.which("pdftocairo"):
        svg = stem.with_suffix(".svg")
        subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
        outputs.append(svg)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
