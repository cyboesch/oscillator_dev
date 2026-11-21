"""Image frames for the conceptual figure (Figure 1 of the ICML write-up).

Reads an existing reverse-SDE trajectory with snapshots (written by sample_latent.py --snapshots) and writes
  images/frame_s<s>.png          one generated face at reverse time s = tau - t (per-frame contrast stretch)
  images/frame_quench.png        the same trajectory after the quench, only if <stem>_q1.npy exists
  images/clamp_noised.png        a real training face noised to forward time T_CLAMP (the clamped state in panel b)
  images/sheet_final.png         contact sheet of the final states of the first trajectories (to choose --traj)
  images/sheet_time.png          contact sheet: the first trajectories (rows) at every snapshot time (columns)
Nothing is written outside this folder; the run folder is only read.

Usage (inside the container; see make_frames.sh):
  python3 make_frames.py                 # contact sheets only
  python3 make_frames.py --traj 7        # frames of trajectory 7
"""
import argparse
import os

import numpy as np

ROOT = "/n/fs/cyrills-part/oscillator_dev"
RUN = (ROOT + "/physical-diffusion-models/out/latent_physical_SGM/"
       "celeba32g_K64_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S32on10+patch8s2S32on55_seed1")
STEM = RUN + "/samples/sde_base_n1000"
TRAIN = ROOT + "/physical-diffusion-models/data/CelebA/celeba_gray32_train.npz"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
GRID = 32
TEMP = 0.02            # bath temperature of the model
TAU = 5.0
PIXEL_NOISE = 0.01     # sd of the pixel noise added to the data before normalisation
T_CLAMP = 1.0          # forward time of the clamped example in panel b


def save_png(path, img01, up=8):
    """img01: 2-D array in [0, 1]; nearest-neighbour upscaling so the pixels stay crisp."""
    a = np.clip(img01, 0.0, 1.0)
    a = np.kron(a, np.ones((up, up)))
    a8 = (255.0 * a + 0.5).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(a8).save(path)
    except ImportError:
        import matplotlib.image as mpimg
        mpimg.imsave(path, a8, cmap="gray", vmin=0, vmax=255)
    print("wrote", path, a8.shape)


def stretch(frame):
    """Per-frame contrast stretch between the 1st and 99th percentile (the 'autoscale' of the old figure)."""
    lo, hi = np.percentile(frame, [1.0, 99.0])
    return (frame - lo) / max(hi - lo, 1e-12)


def sheet(frames, ncol, pad=1):
    """frames: (n, GRID, GRID) in [0, 1] -> one tiled image with white separators."""
    n = frames.shape[0]
    nrow = int(np.ceil(n / ncol))
    H = nrow * GRID + (nrow + 1) * pad
    W = ncol * GRID + (ncol + 1) * pad
    out = np.ones((H, W))
    for k in range(n):
        r, c = divmod(k, ncol)
        y0 = pad + r * (GRID + pad)
        x0 = pad + c * (GRID + pad)
        out[y0:y0 + GRID, x0:x0 + GRID] = np.clip(frames[k], 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=int, default=None, help="trajectory index to render; omit for the contact sheets")
    ap.add_argument("--clamp-image", type=int, default=12345, help="index of the training face used for the clamp")
    ap.add_argument("--n-sheet", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- normalisation of the data: one global mean and sd (as in latent_model._normalise_like_paper)
    tr = np.load(TRAIN)
    print("train file keys:", {k: tr[k].shape for k in tr.files})
    key = max(tr.files, key=lambda k: tr[k].size)
    Xtr = tr[key].reshape(-1, GRID * GRID).astype(np.float64) / 255.0
    mean = Xtr.mean()
    sd = np.sqrt(Xtr.var() + PIXEL_NOISE ** 2)
    print(f"data mean {mean:.4f}, sd {sd:.4f}, n {Xtr.shape[0]}")

    # ---- snapshots
    snap = np.load(STEM + "_snapshots.npz")
    for k in snap.files:
        print("snapshot key:", k, snap[k].shape, snap[k].dtype)
    big = max(snap.files, key=lambda k: snap[k].size)
    S = np.asarray(snap[big], dtype=np.float64)
    assert S.ndim == 3, S.shape
    ax_pix = [i for i, d in enumerate(S.shape) if d == GRID * GRID][0]
    S = np.moveaxis(S, ax_pix, 2)
    if S.shape[0] < S.shape[1]:            # (n_times, n, pix) -> (n, n_times, pix)
        S = np.swapaxes(S, 0, 1)
    n, nt, _ = S.shape
    print("snapshots as (n, n_times, pixels):", S.shape)
    times = None
    for k in snap.files:
        if snap[k].ndim == 1 and snap[k].shape[0] == nt:
            times = np.asarray(snap[k], dtype=np.float64)
            print("times key:", k, times)
    sds = S.std(axis=(0, 2))
    print("sd of the state at each snapshot:", np.round(sds, 3))
    if times is None:
        times = np.linspace(0.0, TAU, nt)
        print("no time array in the file; assuming", times)
    # reverse time s = 0 is the noise state (smallest sd); make column 0 the noise state
    order = np.argsort(sds)
    if not np.all(np.diff(order) > 0):
        print("reordering snapshots by increasing sd (noise first)")
    S = S[:, order]
    s_rev = np.linspace(0.0, TAU, nt)
    print("columns are reverse times s =", s_rev)

    final = np.load(STEM + ".npy").reshape(n, -1)
    print("final samples:", final.shape, "max |final - last snapshot| =",
          float(np.abs(final - S[:, -1]).max()))

    def to_pix(x):
        return (x * sd + mean).reshape(GRID, GRID)

    if args.traj is None:
        m = min(args.n_sheet, n)
        fin = np.stack([to_pix(final[i]) for i in range(m)])
        save_png(os.path.join(OUT, "sheet_final.png"), sheet(fin, ncol=8), up=4)
        rows = [stretch(to_pix(S[i, j])) for i in range(m) for j in range(nt)]
        save_png(os.path.join(OUT, "sheet_time.png"), sheet(np.stack(rows), ncol=nt), up=3)
        return

    i = args.traj
    for j in range(nt):
        save_png(os.path.join(OUT, f"frame_s{s_rev[j]:.2f}.png"), stretch(to_pix(S[i, j])))
    qfile = STEM + "_q1.npy"
    if os.path.exists(qfile):
        q = np.load(qfile).reshape(n, -1)
        save_png(os.path.join(OUT, "frame_quench.png"), stretch(to_pix(q[i])))
    else:
        print("no quenched file", qfile, "- frame_quench.png not written")

    # ---- the clamped example of panel b: a real training face at forward time T_CLAMP
    rng = np.random.default_rng(0)
    x0 = (Xtr[args.clamp_image] - mean) / sd
    xt = np.exp(-T_CLAMP) * x0 + np.sqrt(TEMP * (1.0 - np.exp(-2.0 * T_CLAMP))) * rng.standard_normal(x0.shape)
    save_png(os.path.join(OUT, "clamp_noised.png"), stretch(to_pix(xt)))
    save_png(os.path.join(OUT, "clamp_clean.png"), np.clip(to_pix(x0), 0, 1))


if __name__ == "__main__":
    main()
