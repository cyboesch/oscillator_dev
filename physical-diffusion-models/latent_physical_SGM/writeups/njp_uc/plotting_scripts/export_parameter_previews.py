#!/usr/bin/env python3
"""Create faithful PNG previews from the completed parameter-figure PDFs.

Runs on the host after the container renderer; requires pdftoppm. The wrapper
passes its full argument list, so unrelated extraction flags are ignored.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "figures")
    parser.add_argument("--also-linear", action="store_true")
    parser.add_argument("--colour-scale", default="log")
    args, _ = parser.parse_known_args()
    if shutil.which("pdftoppm") is None:
        parser.error("pdftoppm is required for PDF-based previews")
    output = args.output_dir.resolve()
    stems = ["parameter_evolution_draft"]
    if args.also_linear and args.colour_scale != "linear":
        stems.append("parameter_evolution_linear")
    for stem in stems:
        provenance_name = "parameter_evolution_provenance.json" if stem.endswith("draft") else stem + "_provenance.json"
        metadata = json.loads((output.parent / "figure_data" / provenance_name).read_text())
        for page in metadata["figure_pages"]:
            number = str(page["page"])
            prefix = output / f"{stem}_page{number}"
            subprocess.run(["pdftoppm", "-f", number, "-l", number, "-r", "220",
                            "-png", "-singlefile", str(output / f"{stem}.pdf"), str(prefix)], check=True)
            print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
