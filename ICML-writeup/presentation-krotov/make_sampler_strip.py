"""Image strip for the talk: what the quench and the corrector do (CelebA 64x64, exact training).

Rows: real test images; reverse SDE as sampled; + quench; + corrector and quench; + fixed-dwell corrector and quench.
Reads existing sample files (read-only) and writes img/strip_c64_sampler.png. CPU, seconds.
"""
import os
import numpy as np

ROOT = "/n/fs/cyrills-part/oscillator_dev"
RUN = (ROOT + "/physical-diffusion-models/out/latent_physical_SGM/"
       "celeba64g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch32s8S64on10+patch16s4S64on55_seed1/samples/")
DATA = ROOT + "/physical-diffusion-models/data/CelebA/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "strip_c64_sampler.png")
G, NCOL, UP, PAD = 64, 10, 2, 2
SETS = ["sde_base_n2000.npy", "sde_base_n2000_q1.npy", "sde_base_n2000_corr2x0.16_q1.npy",
        "sde_base_n2000_dwell0.3_q1.npy"]


def load_images(path, n):
    z = np.load(path)
    key = max(z.files, key=lambda k: z[k].size)
    return z[key][:n].reshape(n, G * G).astype(np.float64) / 255.0


def main():
    tr = load_images(DATA + "celeba_gray64_train.npz", 20000)
    mean, sd = tr.mean(), np.sqrt(tr.var() + 0.01 ** 2)
    print(f"mean {mean:.4f} sd {sd:.4f}")
    rows = [load_images(DATA + "celeba_gray64_test.npz", 1500)[::150][:NCOL]]  # spread over the file: neighbours are one person
    for s in SETS:
        x = np.load(RUN + s)[:NCOL]
        print(s, x.shape, "finite", bool(np.all(np.isfinite(x))))
        rows.append(np.clip(x * sd + mean, 0.0, 1.0))
    t = G * UP
    H = len(rows) * t + (len(rows) + 1) * PAD
    W = NCOL * t + (NCOL + 1) * PAD
    out = np.full((H, W), 0.27)
    for r, row in enumerate(rows):
        for c in range(NCOL):
            tile = np.kron(row[c].reshape(G, G), np.ones((UP, UP)))
            y0, x0 = PAD + r * (t + PAD), PAD + c * (t + PAD)
            out[y0:y0 + t, x0:x0 + t] = tile
    from PIL import Image
    Image.fromarray((255.0 * out + 0.5).astype(np.uint8)).save(OUT)
    print("wrote", OUT, out.shape)


if __name__ == "__main__":
    main()
