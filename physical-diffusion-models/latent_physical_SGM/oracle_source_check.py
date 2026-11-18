"""Diagnostic (2026-11-18): what the end of the reverse process and the zero-temperature quench do to real images.

The oracle-start scan (sample_latent --oracle-start K) starts the reverse process from exactly noised training images
x0; its samples' .json records their indices. This script, for one run:
  (1) saves those x0 as <run>/samples/oracle-source_train-x0_n<N>.npy (normalised units, for inception_fid);
  (2) quenches x0 directly under the run's t = 0 free energy (sample_latent.do_quench, the settings of --quench),
      saving <run>/samples/oracle-source_train-x0_n<N>_q<S>.npy (never overwrites: existing files are reused);
  (3) compares, image by image with its source x0 (pixels x * std + mean clipped to [0, 1]), x0 quenched and every
      oracle set given (as sampled and quenched): RMS pixel change, correlation, per-image contrast (std) ratio,
      brightness shift, dark share (< 0.1), and the power of the image in three spatial-frequency bands relative
      to the source (low: |k| <= g/8, mid: g/8 < |k| <= g/4, high: |k| > g/4; image mean removed). Unpaired: the
      same band powers of the run's normal quenched samples.
Writes report.txt, examples.png (sources, the sets, and the change of the t = 0.2 quenched set x 4) and
spectra.png (radial power spectra) to --out, with a README.

  python oracle_source_check.py --run RUN --slices 95,85,70,55 --out DIR [--n 2000 --chunk 125 --quench 1]
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sample_latent as sl  # noqa: E402

lm = sl.lm


def band_power(P, g):
    """Mean power per image in the low / mid / high bands and the radial spectrum (integer |k| bins), image mean removed."""
    F = np.fft.fft2((P - P.mean(1, keepdims=True)).reshape(-1, g, g))
    pw = np.abs(F) ** 2
    k = np.fft.fftfreq(g) * g
    r = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    bands = [r <= g / 8, (r > g / 8) & (r <= g / 4), r > g / 4]
    per_img = np.stack([pw[:, b].sum(1) for b in bands], 1)          # (n, 3)
    rb = np.rint(r).astype(int)
    radial = np.array([pw[:, rb == i].sum(1).mean() for i in range(1, rb.max() + 1)])
    return per_img, radial


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--slices", default="95,85,70,55", help="oracle start slices whose sample sets to compare")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=125)
    ap.add_argument("--quench", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    theta_hist, W_hist, b_hist, run_dir, prov, problem = sl.load_params(args.run, with_harmonic=True, with_patch=True)
    taus = sl.slice_times(prov)
    local_p, quad_p = prov.pop("local", None), prov.pop("quad", None)
    harm_p, patch_p = prov.pop("harmonic", None), prov.pop("patch", None)
    sdir = os.path.join(run_dir, "samples")
    slices = [int(s) for s in args.slices.split(",") if s.strip()]
    times = {k: float(taus[k]) for k in slices}

    pick = None
    for k in slices:                                   # the oracle sets must share their source images
        with open(os.path.join(sdir, f"sde_base_n{args.n}_oracle{k}.json")) as f:
            idx = json.load(f)["oracle_start"]["x0_index"]
        if pick is None:
            pick = np.asarray(idx)
        elif not np.array_equal(pick, idx):
            raise SystemExit(f"oracle slice {k}: different source images than slice {slices[0]}")
    prob = lm.make_problem(problem, prov.get("radius"), prov.get("neighbourhood", "chebyshev"))
    data = np.asarray(prob["data"], dtype=np.float64)
    x0 = data[pick]
    g, mean, std = prob["grid"], prob["mean"], prob["std"]

    stem = os.path.join(sdir, f"oracle-source_train-x0_n{args.n}")
    if os.path.exists(stem + ".npy"):
        if not np.array_equal(np.load(stem + ".npy"), x0):
            raise SystemExit(f"{stem}.npy exists and differs from the oracle source images")
        out(f"source images: reusing {stem}.npy")
    else:
        np.save(stem + ".npy", x0)
        out(f"source images: saved {stem}.npy ({x0.shape[0]} training images, indices from the oracle .json)")
    qname = f"{stem}_q{args.quench:g}.npy"
    if os.path.exists(qname):
        out(f"quench of the sources: reusing {qname}")
    else:
        drift_ode, _ = sl.make_drift(theta_hist, W_hist, b_hist, "ode", problem, prov.get("hidden", "ising"),
                                     local_p, quad_p, prov.get("radius"), prov.get("neighbourhood", "chebyshev"),
                                     taus, harmonic=harm_p, patch=patch_p)
        sl.do_quench(x0, drift_ode, [args.quench], stem, SimpleNamespace(chunk=args.chunk), time.time())

    def px(x):
        return np.clip(x * std + mean, 0.0, 1.0)

    P0 = px(x0)
    sets = {f"training images quenched ({args.quench:g})": np.load(qname)}
    for k in sorted(slices, key=lambda k: times[k]):
        for suf, lab in (("", "as sampled"), (f"_q{args.quench:g}", f"quench {args.quench:g}")):
            sets[f"oracle start t = {times[k]:.2f}, {lab}"] = np.load(os.path.join(sdir, f"sde_base_n{args.n}_oracle{k}{suf}.npy"))
    normal = np.load(os.path.join(sdir, f"sde_base_n{args.n}_q{args.quench:g}.npy"))

    bp0, rad0 = band_power(P0, g)
    out()
    out(f"run {args.run}; {x0.shape[0]} source training images; pixels in [0, 1]")
    out(f"sources: contrast (mean per-image std) {P0.std(1).mean():.4f}; dark share {100 * (P0 < 0.1).mean():.2f}%; "
        f"band power low / mid / high {bp0.mean(0)[0]:.1f} / {bp0.mean(0)[1]:.2f} / {bp0.mean(0)[2]:.3f}")
    out()
    out(f"{'set (paired with its source image)':<44}{'n':>5}  {'RMS change':>10} {'corr':>6}  {'contrast':>9} "
        f"{'bright':>7}  {'dark':>6}   power vs source: low   mid   high   change power: low   mid   high")
    radials = {"training images (sources)": rad0}
    for lab, x in sets.items():
        ok = np.all(np.isfinite(x), 1)
        P = px(x[ok])
        S = P0[ok]
        d = P - S
        rms = np.sqrt((d ** 2).mean(1))
        Pc, Sc = P - P.mean(1, keepdims=True), S - S.mean(1, keepdims=True)
        corr = (Pc * Sc).sum(1) / np.sqrt((Pc ** 2).sum(1) * (Sc ** 2).sum(1))
        cr = P.std(1) / S.std(1)
        bp, rad = band_power(P, g)
        bpd, _ = band_power(d, g)
        bs = bp0[ok].mean(0)
        radials[lab] = rad
        out(f"{lab:<44}{int(ok.sum()):>5}  {np.median(rms):>10.4f} {np.median(corr):>6.3f}  {np.median(cr):>9.3f} "
            f"{d.mean():>+7.4f}  {100 * (P < 0.1).mean():>5.2f}%   "
            f"{bp.mean(0)[0] / bs[0]:>21.3f} {bp.mean(0)[1] / bs[1]:>5.3f} {bp.mean(0)[2] / bs[2]:>6.3f}   "
            f"{bpd.mean(0)[0] / bs[0]:>17.3f} {bpd.mean(0)[1] / bs[1]:>5.3f} {bpd.mean(0)[2] / bs[2]:>6.3f}")
    okn = np.all(np.isfinite(normal), 1)
    Pn = px(normal[okn])
    bpn, radn = band_power(Pn, g)
    radials[f"normal samples (noise start), quench {args.quench:g}"] = radn
    out()
    out(f"unpaired: normal samples (noise start, quench {args.quench:g}), n {int(okn.sum())}: contrast "
        f"{Pn.std(1).mean():.4f} (sources {P0.std(1).mean():.4f}); dark {100 * (Pn < 0.1).mean():.2f}%; band power vs "
        f"the sources low / mid / high {bpn.mean(0)[0] / bp0.mean(0)[0]:.3f} / {bpn.mean(0)[1] / bp0.mean(0)[1]:.3f} / "
        f"{bpn.mean(0)[2] / bp0.mean(0)[2]:.3f}")
    out()
    out("(RMS change and corr: median over images of the pixel RMS difference to the source and the correlation with "
        "it; contrast: median ratio of the image's pixel std to its source's; bright: mean pixel shift; dark: share of "
        "pixels < 0.1; power vs source: the set's mean power per band over the sources'; change power: the power of "
        "(image - source) per band over the sources' power, i.e. how much was changed at that scale)")
    out(f"total {time.time() - t_start:.0f} s")
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 12
    rows = [("training images (sources)", P0)] + [(lab, px(np.nan_to_num(x))) for lab, x in sets.items()
                                                  if "quench" in lab]
    t02 = min(times.values())
    key = f"oracle start t = {t02:.2f}, quench {args.quench:g}"
    diff = px(np.nan_to_num(sets[key])) - P0
    fig, axes = plt.subplots(len(rows) + 1, cols, figsize=(cols * 1.1, (len(rows) + 1) * 1.2))
    for r, (lab, P) in enumerate(rows):
        for c in range(cols):
            axes[r, c].imshow(P[c].reshape(g, g), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(lab, fontsize=7, loc="left")
    for c in range(cols):
        axes[-1, c].imshow(np.clip(0.5 + 4 * diff[c].reshape(g, g), 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[-1, c].axis("off")
    axes[-1, 0].set_title(f"change of '{key}' against its source, x 4 (grey = 0)", fontsize=7, loc="left")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "examples.png"), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lab, rad in radials.items():
        ax.semilogy(np.arange(1, rad.size + 1), rad, label=lab, lw=2 if "sources" in lab else 1)
    ax.set_xlabel("spatial frequency |k| (cycles per image)")
    ax.set_ylabel("mean power per image")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "spectra.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(args.out, "README.txt"), "w") as f:
        f.write(__doc__ + f"\nRun: {args.run}\nOracle slices: {slices} (t = {[round(times[k], 2) for k in slices]})\n"
                f"Job: {os.environ.get('SLURM_JOB_ID', 'manual')}\n")


if __name__ == "__main__":
    main()
