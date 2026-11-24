"""The 'after quench' frame of the conceptual figure.

The trajectory shown in the figure comes from sde_base_n1000 (the only set with snapshots), for which no quenched
file exists. This script quenches ONE final state of that set under the run's t = 0 free energy, exactly as
sample_latent.do_quench does (dx/ds = -grad F for a duration of 1), and writes images/frame_quench.png.
CPU only. The run folder and the code are only read; nothing is written outside this folder.

Usage (inside the container; see make_quench_frame.sh):  python3 make_quench_frame.py --traj 0
"""
import argparse
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = "/n/fs/cyrills-part/oscillator_dev"
CODE = ROOT + "/physical-diffusion-models/latent_physical_SGM"
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import latent_model as lm  # noqa: E402
import sample_latent as sl  # noqa: E402
sys.path.insert(0, ROOT + "/ICML-writeup/Conceptual-fig")
import make_frames as mf  # noqa: E402

RUN_NAME = os.path.basename(mf.RUN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--duration", type=float, default=1.0)
    args = ap.parse_args()

    theta_hist, W_hist, b_hist, run_dir, prov, problem = sl.load_params(RUN_NAME, with_harmonic=True, with_patch=True)
    print("run", run_dir, "problem", problem, "K", W_hist.shape[1], flush=True)
    taus = sl.slice_times(prov)
    local_p, quad_p = prov.pop("local", None), prov.pop("quad", None)
    harm_p = prov.pop("harmonic", None)
    patch_p = prov.pop("patch", None)
    drift_ode, _ = sl.make_drift(theta_hist, W_hist, b_hist, "ode", problem, prov.get("hidden", "ising"),
                                 local_p, quad_p, prov.get("radius"), prov.get("neighbourhood", "chebyshev"), taus,
                                 harmonic=harm_p, patch=patch_p, cond=None)

    def qdrift(s, x, a):
        return drift_ode(lm.T_FORWARD, x, a) - x

    final = np.load(mf.STEM + ".npy")
    x = final[args.traj]
    sol = sl.solve_ODE(qdrift, jnp.asarray(x), 0.0, args.duration, 2, lm.DT0, rtol=lm.RTOL_ODE, atol=lm.ATOL_ODE)
    xq = np.asarray(sol.ys[-1])
    print("finite:", bool(np.all(np.isfinite(xq))), "displacement", float(np.linalg.norm(xq - x)),
          "sd before / after", float(x.std()), float(xq.std()), flush=True)

    tr = np.load(mf.TRAIN)
    key = max(tr.files, key=lambda k: tr[k].size)
    Xtr = tr[key].reshape(-1, mf.GRID * mf.GRID).astype(np.float64) / 255.0
    mean, sd = Xtr.mean(), np.sqrt(Xtr.var() + mf.PIXEL_NOISE ** 2)
    np.save(os.path.join(mf.OUT, f"quench_traj{args.traj}.npy"), xq)
    mf.save_png(os.path.join(mf.OUT, "frame_quench.png"), mf.stretch((xq * sd + mean).reshape(mf.GRID, mf.GRID)))


if __name__ == "__main__":
    main()
