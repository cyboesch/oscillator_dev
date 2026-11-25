"""Sample grids for the talk (backup): 100 samples each of the digital corrector and of the fixed dwell 0.1 and 0.3
(CelebA 64x64, exact training, quench at the end). Reads existing sample files; writes img/grid_c64_*.png. CPU."""
import os
import numpy as np

ROOT = "/n/fs/cyrills-part/oscillator_dev"
RUN = (ROOT + "/physical-diffusion-models/out/latent_physical_SGM/"
       "celeba64g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch32s8S64on10+patch16s4S64on55_seed1/samples/")
DATA = ROOT + "/physical-diffusion-models/data/CelebA/"
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
G, N, NCOL, UP, PAD = 64, 100, 10, 2, 2
SETS = {"grid_c64_corrector.png": "sde_base_n2000_corr2x0.16_q1.npy",
        "grid_c64_dwell0.1.png": "sde_base_n2000_dwell0.1_q1.npy",
        "grid_c64_dwell0.3.png": "sde_base_n2000_dwell0.3_q1.npy"}


def main():
    z = np.load(DATA + "celeba_gray64_train.npz")
    key = max(z.files, key=lambda k: z[k].size)
    tr = z[key][:20000].reshape(20000, G * G).astype(np.float64) / 255.0
    mean, sd = tr.mean(), np.sqrt(tr.var() + 0.01 ** 2)
    from PIL import Image
    t = G * UP
    side = NCOL * t + (NCOL + 1) * PAD
    for name, f in SETS.items():
        x = np.clip(np.load(RUN + f)[:N] * sd + mean, 0.0, 1.0)
        out = np.full((side, side), 0.27)
        for k in range(N):
            r, c = divmod(k, NCOL)
            y0, x0 = PAD + r * (t + PAD), PAD + c * (t + PAD)
            out[y0:y0 + t, x0:x0 + t] = np.kron(x[k].reshape(G, G), np.ones((UP, UP)))
        Image.fromarray((255.0 * out + 0.5).astype(np.uint8)).save(os.path.join(IMG, name))
        print("wrote", name, out.shape, "finite", bool(np.all(np.isfinite(x))))


if __name__ == "__main__":
    main()
