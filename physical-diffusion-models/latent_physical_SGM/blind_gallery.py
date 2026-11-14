#!/usr/bin/env python3
"""Blind sheets for counting malformed digits by eye: which model made a sheet is hidden until the count is done.

FID and the classifier confidence hardly see shape errors (a broken 3, a scribble) once the patch units have removed
the grain (2026-11-13), so the shape quality of models is compared by a blind count.

For one problem and sample sets NAME=PATH (PATH as in make_gallery.py / evaluate_samples.py), plus the word 'real' for
random training images (a control: how many real digits look malformed by the same rule), writes into --out:
  sheet_01.png, ...   10 x 10 tiles each, all from ONE set, in a random order. The sheets of all sets are shuffled
                      together, so neither the file order nor the position on a sheet says which set it is.
  COUNTS.txt          blank form: per sheet, the number of malformed tiles (and, separately, stray dots)
  README.txt          what to count
  KEY.txt             sheet -> set name, sample file and the sample indices shown. Open only after counting.
Every model set shows the same --per-model trajectory indices (the same starts and noise; random, without
replacement, among the first n of the smallest set), each sheet in its own random order. Diverged samples (|x| > 15 or
not finite) are red tiles, as in make_gallery.py. The shuffle seed defaults to the Slurm job id, so it is not known in
advance.

Usage (CPU Slurm job): python blind_gallery.py --problem mnist28_d0123 --out DIR [--per-model 200] [--seed N]
                       real NAME=PATH ...
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402
import evaluate_samples as es  # noqa: E402
import make_gallery as mg  # noqa: E402

RULE = """What counts as malformed (one tick per tile):
  - a digit with a broken or open stroke that changes or destroys it (a 0 open like a C, a 3 or 2 with a gap)
  - two digits merged, or a hybrid you cannot assign (4/9, 2/3)
  - a scribble, squiggle or blob with no readable digit
  - a digit made of disconnected pieces
  - a red tile (diverged sample)
Do NOT count: thick, thin, slanted, small, or ugly-but-readable digits. Count a small isolated dot next to an
otherwise fine digit separately as a 'stray dot' (it is an artefact of the patch units, not a shape error).
Count the sheets in any order; do not open KEY.txt until all sheets are counted."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="'real' or NAME=PATH")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--out", required=True, help="output folder (created)")
    ap.add_argument("--per-model", type=int, default=200, help="samples per set, a multiple of 100 (one sheet = 100)")
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed (default: the Slurm job id, else the time)")
    args = ap.parse_args()
    m = args.per_model
    if m <= 0 or m % 100:
        raise ValueError("--per-model must be a positive multiple of 100")
    seed = args.seed if args.seed is not None else int(os.environ.get("SLURM_JOB_ID", "0") or 0) or int(time.time())
    rng = np.random.default_rng(seed)
    os.makedirs(args.out, exist_ok=True)

    prob = lm.make_problem(args.problem)
    data, mean, std, g = np.asarray(prob["data"]), prob["mean"], prob["std"], prob["grid"]
    px = np.clip(data * std + mean, 0.0, 1.0)
    scale = max(1, 112 // g)

    sets = []                                # (name, sample file or None, P, div)
    for spec in args.sets:
        if spec == "real":
            sets.append(("real training images (random)", None, None, None))
            continue
        name, path = spec.split("=", 1)
        P, div = mg.to_px(es.load_set(path), mean, std)
        sets.append((name, path, P, div))
    sizes = [P.shape[0] for _, _, P, _ in sets if P is not None]
    n_min = min(sizes) if sizes else px.shape[0]
    if m > n_min:
        raise ValueError(f"--per-model {m} > the smallest set ({n_min} samples)")
    traj = np.sort(rng.choice(n_min, size=m, replace=False))       # the same trajectories in every model set
    real = rng.choice(px.shape[0], size=m, replace=False)

    sheets = []                              # (name, path, tiles, diverged, indices shown)
    for name, path, P, div in sets:
        if P is None:
            Ps, ds, shown = px[real], np.zeros(m, bool), real
        else:
            Ps, ds, shown = P[traj], div[traj], traj
        perm = rng.permutation(m)
        for k in range(m // 100):
            sel = perm[100 * k:100 * (k + 1)]
            sheets.append((name, path, Ps[sel], ds[sel], shown[sel]))
    order = rng.permutation(len(sheets))

    key = [f"KEY: open only after counting. Seed {seed}, Slurm job {os.environ.get('SLURM_JOB_ID', 'manual')}, "
           f"problem {args.problem}, {m} samples per set ({m // 100} sheets each).", ""]
    counts = ["Counts of malformed tiles per sheet (rule in README.txt). Fill in, then open KEY.txt.", "",
              "sheet       malformed   stray dots"]
    for s, j in enumerate(order, 1):
        name, path, P, div, shown = sheets[j]
        fn = f"sheet_{s:02d}.png"
        mg.save(mg.canvas(P, div, g, 10, 10), os.path.join(args.out, fn), f"sheet {s:02d}", scale)
        what = "training-image indices" if path is None else "trajectory indices"
        key += [f"{fn}: {name}" + ("" if path is None else f"  ({path})"),
                f"    {what} in tile order (row by row): {' '.join(str(int(i)) for i in shown)}"]
        counts.append(f"sheet_{s:02d}      ___         ___")
    with open(os.path.join(args.out, "KEY.txt"), "w") as f:
        f.write("\n".join(key) + "\n")
    with open(os.path.join(args.out, "COUNTS.txt"), "w") as f:
        f.write("\n".join(counts) + "\n")
    names = sorted({n for n, *_ in sets})
    with open(os.path.join(args.out, "README.txt"), "w") as f:
        f.write(f"Blind shape count, problem {args.problem}, made {time.strftime('%Y-%m-%d %H:%M')} by blind_gallery.py "
                f"(Slurm job {os.environ.get('SLURM_JOB_ID', 'manual')}).\n\n"
                f"{len(sheets)} sheets of 100 tiles; {m // 100} sheets per set. The sets (in alphabetical order, NOT "
                f"the sheet order):\n" + "".join(f"  {n}\n" for n in names) +
                "Model sets show the same trajectories (same starts and noise), each sheet in its own random order.\n\n"
                + RULE + "\n")
    print(f"wrote {len(sheets)} sheets to {args.out} (seed {seed})", flush=True)


if __name__ == "__main__":
    main()
