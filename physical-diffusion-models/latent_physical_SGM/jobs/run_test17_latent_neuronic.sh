#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test17
#SBATCH --output=logs/latent_test17_%j.out
#SBATCH --error=logs/latent_test17_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306
# Seventeenth validation round, after adding --hidden-jackknife-order 2 (latent_model.make_loss_per_sample with
# jackknife = 2: Richardson extrapolation over S, S/2 and S/4 of the same draws, 8/3 L(S) - 2 L(halves) + 1/3
# L(quarters), which removes the O(1/S^2) bias that the split-half jackknife leaves). mnist14, 0/1:
#   0. (a) exact enumeration: the three combinations (plug-in, order 1, order 2) of a cubic of a Bernoulli sample mean,
#          over all 2^8 outcomes of S = 8 draws: order 2 must be unbiased to rounding, order 1 not;
#      (b) Monte Carlo through make_loss itself (Potts 3 + both, 4 configurations, S = 8, 400,000 draws): the mean
#          gradient of every parameter against the exact gradient, in standard errors; order 2 must be consistent
#          with zero bias (max |z| < 6 over all elements), and order 1 should show its O(1/S^2) bias;
#   1. the order-1 path is unchanged (Potts 128 + both, S = 64 jackknife, slices 0-1 at full steps, against the
#      production run);
#   2. gradient check on the trained Potts 128 + both (cd1_check.py): S = 16 and 64 with the jackknife of order 1 and
#      2, error of one estimate and of the mean of 64 (bias + noise / 8), slices 10, 20, 30 and t = 0;
#   3. smokes (Potts 8 + both, 3 slices at 2% of the steps): order 2 with S = 16 (per-element Adam and the visit-count
#      rule) and S = 8; the config must record hidden_jackknife_order 2;
#   4. invalid combinations must be refused (order 2 with S = 6; order 2 without --hidden-jackknife);
#   5. Potts 128 + both, S = 16, order 2, visit-count rule, slices 0-2 at full steps, next to the production run's
#      slices 0-2 with order 1.
# Submit from latent_physical_SGM/:   cd physical-diffusion-models/latent_physical_SGM && sbatch jobs/run_test17_latent_neuronic.sh

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
export GIT_COMMIT="$(git -C "${HOST_PROJECTS_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
D=/physical-diffusion-models/latent_physical_SGM
OUT=../out/latent_physical_SGM
J=${SLURM_JOB_ID}
run() { apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u "$@"; }
P128="--problem mnist14 --hidden potts --K 128 --quad --local --select ema"
P8="--problem mnist14 --hidden potts --K 8 --quad --local --select ema"

rc=0
echo "== 0. unit checks"
run - <<'EOF' || rc=$((rc | 1))
import itertools
import sys
sys.path.insert(0, "/physical-diffusion-models/latent_physical_SGM")
import numpy as np
import latent_model as lm
import jax
import jax.numpy as jnp
import jax.random as jr

ok = True
print(" (a) exact enumeration, f(m) = m^3 + 2 m^2 - m of the mean of S = 8 Bernoulli(0.3) draws")
mu, S = 0.3, 8
f = lambda m: m ** 3 + 2.0 * m ** 2 - m
def comb(draws, order):
    full = f(np.mean(draws))
    if order == 0:
        return full
    halves = 0.5 * (f(np.mean(lm._part(draws, 0))) + f(np.mean(lm._part(draws, 1))))
    if order == 1:
        return 2.0 * full - halves
    quarters = 0.25 * sum(f(np.mean(lm._part(draws, (4, q)))) for q in range(4))
    return (8.0 / 3.0) * full - 2.0 * halves + (1.0 / 3.0) * quarters
E = {o: 0.0 for o in (0, 1, 2)}
for bits in itertools.product([0.0, 1.0], repeat=S):
    d = np.array(bits)
    pr = mu ** d.sum() * (1.0 - mu) ** (S - d.sum())
    for o in E:
        E[o] += pr * comb(d, o)
bias = {o: E[o] - f(mu) for o in E}
k3 = mu * (1 - mu) * (1 - 2 * mu)
print(f"     bias plug-in {bias[0]:.3e}, order 1 {bias[1]:.3e} (expected -2 k3 / S^2 = {-2 * k3 / S ** 2:.3e}), "
      f"order 2 {bias[2]:.3e}")
good = abs(bias[2]) < 1e-13 and abs(bias[1] + 2 * k3 / S ** 2) < 1e-13 and abs(bias[0]) > abs(bias[1]) > 1e-4
ok &= good
print(f"     {'ok' if good else 'FAIL'}")

print(" (b) Monte Carlo through make_loss: Potts 3 + both on mnist14, 4 configurations, S = 8, 400,000 draws")
prob = lm.make_problem("mnist14")
n = prob["n"]
vis = lm.DenseVisible(prob["conn"], n)
theta0, _ = lm.paper_theta_layout(vis.P, n)
K, T, S = 3, lm.TEMP, 8
rng = np.random.default_rng(1)
X = jnp.asarray(np.asarray(prob["data"][:4]) + 0.3 * rng.normal(size=(4, n)))
params = {"theta": jnp.asarray(theta0), "W": jnp.asarray(0.001 * rng.normal(size=(K, n))), "b": jnp.zeros(K),
          "Q": jnp.asarray(0.0005 * rng.normal(size=(K, n))), "wl": jnp.asarray(0.01 * rng.normal(size=n)),
          "bl": jnp.zeros(n)}
A = X @ params["W"].T + (X * X) @ params["Q"].T
print("     Potts occupations of the 4 configurations:", np.round(np.asarray(jax.nn.softmax(A / T, axis=1)), 3).tolist())
g0 = jax.jit(jax.grad(lm.make_loss(vis, T, K, "potts", True)))(params, X)
N, chunk = 400_000, 2_000
for order, jk in ((0, False), (1, True), (2, 2)):
    gfun = jax.jit(jax.vmap(jax.grad(lm.make_loss(vis, T, K, "potts", True, samples=S, jackknife=jk)),
                            in_axes=(None, None, 0)))
    s1 = {k: jnp.zeros_like(v) for k, v in params.items()}
    s2 = {k: jnp.zeros_like(v) for k, v in params.items()}
    for c in range(N // chunk):
        G = gfun(params, X, jr.split(jr.PRNGKey(1000 + c), chunk))
        s1 = {k: s1[k] + jnp.sum(G[k], axis=0) for k in s1}
        s2 = {k: s2[k] + jnp.sum(G[k] * G[k], axis=0) for k in s2}
    zs = []
    line = []
    for k in ("W", "b", "Q", "wl", "bl", "theta"):
        m = np.asarray(s1[k]) / N
        se = np.sqrt(np.maximum(np.asarray(s2[k]) / N - m * m, 0.0) / N)
        se = np.maximum(se, 1e-9 * np.abs(np.asarray(g0[k])) + 1e-300)   # draw-independent elements: rounding only
        z = (m - np.asarray(g0[k])) / se
        zs.append(np.abs(z).ravel())
        line.append(f"{k} max|z| {np.max(np.abs(z)):.1f} rms {np.sqrt(np.mean(z * z)):.2f}")
    zmax = float(np.max(np.concatenate(zs)))
    print(f"     order {order}: " + "; ".join(line))
    if order == 2:
        good = zmax < 6.0
        ok &= good
        print(f"     order 2 consistent with zero bias (max |z| {zmax:.1f} < 6): {'ok' if good else 'FAIL'}")
    else:
        print(f"     order {order}: max |z| {zmax:.1f} (power check: its bias should be visible)")
print("unit checks:", "ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF

set -x
run ${D}/train_latent.py --name smoke17_m14_P128_S64jk_${J} ${P128} --hidden-samples 64 --hidden-jackknife --max-slices 2 || rc=$((rc | 2))
run ${D}/cd1_check.py --run mnist14_K128_ema_potts_quad_local --slices 10,20,30,99 --samples 16,16o2,64,64o2 --samples-jk --skip-cd1 --draws 64 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke17_m14_P8_S16jk2_${J} ${P8} --hidden-samples 16 --hidden-jackknife --hidden-jackknife-order 2 --max-slices 3 --steps-scale 0.02 || rc=$((rc | 8))
run ${D}/train_latent.py --name smoke17_m14_P8_S16jk2_occ_${J} ${P8} --hidden-norm occupancy --hidden-samples 16 --hidden-jackknife --hidden-jackknife-order 2 --max-slices 3 --steps-scale 0.02 || rc=$((rc | 8))
run ${D}/train_latent.py --name smoke17_m14_P8_S8jk2_occ_${J} ${P8} --hidden-norm occupancy --hidden-samples 8 --hidden-jackknife --hidden-jackknife-order 2 --max-slices 3 --steps-scale 0.02 || rc=$((rc | 8))
bad=0
run ${D}/train_latent.py --name smoke17_bad1_${J} ${P8} --hidden-samples 6 --hidden-jackknife --hidden-jackknife-order 2 --max-slices 1 && bad=1
run ${D}/train_latent.py --name smoke17_bad2_${J} ${P8} --hidden-samples 16 --hidden-jackknife-order 2 --max-slices 1 && bad=1
[ ${bad} -eq 0 ] || rc=$((rc | 16))
run ${D}/train_latent.py --name smoke17_m14_P128_S16jk2_occ_${J} ${P128} --hidden-norm occupancy --hidden-samples 16 --hidden-jackknife --hidden-jackknife-order 2 --max-slices 3 || rc=$((rc | 32))
set +x

echo "== 1. order-1 path: validation loss of slices 0-1, new vs the production run (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke17_m14_P128_S64jk_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_S64jk/train_log.jsonl | head -2)
echo "== 3. smokes (Potts 8 + both)"
for r in P8_S16jk2 P8_S16jk2_occ P8_S8jk2_occ; do
  echo "-- ${r}"
  grep -o -E '"(slice|val_loss|W_norm|eff_states)": [-0-9.e]+' ${OUT}/smoke17_m14_${r}_${J}/train_log.jsonl | paste - - - -
  grep -o -E '"(hidden_samples|hidden_jackknife_order|hidden_norm)": ("[a-z]*"|[0-9]+)' ${OUT}/smoke17_m14_${r}_${J}/config.json | tr '\n' ' '; echo
done
echo "== 4. invalid combinations refused: $([ ${bad} -eq 0 ] && echo yes || echo NO)"
echo "== 5. Potts 128 + both, S = 16 visit-count rule, slices 0-2: order 2 (new) vs order 1 (production)"
for r in smoke17_m14_P128_S16jk2_occ_${J} mnist14_K128_ema_potts_quad_local_S16jk_occ; do
  echo "-- ${r}"
  head -3 ${OUT}/${r}/train_log.jsonl | grep -o -E '"(slice|val_loss|eff_states|mean_max_p|W_norm|Q_norm)": [-0-9.e]+' | paste - - - - - -
done
echo "Finished at $(date) (exit code ${rc}: bits 1 unit checks, 2 regression run, 4 gradient check, 8 smokes, 16 invalid accepted, 32 Potts 128)"
exit ${rc}
