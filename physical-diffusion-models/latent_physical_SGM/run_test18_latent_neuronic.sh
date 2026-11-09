#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test18
#SBATCH --output=latent_test18_%j.out
#SBATCH --error=latent_test18_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306
# Eighteenth validation round, after adding --occ-same-draws (the visit-count rule counts each Potts state's visits
# among the S draws the gradient statistics already use, latent_model.loss_draw_occupation, instead of S fresh draws:
# S reads per configuration and step instead of 2S). mnist14, 0/1:
#   0. unit check: loss_draw_occupation reproduces the loss's own draws. A manual plug-in loss built from
#      sampled_state_mean / sampled_spin_mean with k_h, k_l = jr.split(key) must equal make_loss_per_sample(samples
#      = S)(params, X, key) exactly (this pins down the key the loss uses), and the batch mean of that
#      sampled_state_mean must equal loss_draw_occupation exactly;
#   1. the visit-count rule without the flag is unchanged (Potts 128 + both, S = 16 jackknife, slices 0-1 at full
#      steps, against the production run mnist14_K128_ema_potts_quad_local_S16jk_occ);
#   2. smoke with the flag (Potts 8 + both, S = 8 jackknife, 3 slices at 2% of the steps); config must record it;
#   3. invalid combinations refused (the flag without --hidden-norm occupancy; without --hidden-samples);
#   4. Potts 128 + both, S = 8 jackknife, visit-count rule with the flag, slices 0-2 at full steps, next to the
#      production run with fresh draws (mnist14_K128_ema_potts_quad_local_S8jk_occ).
# Submit from this folder:   cd physical-diffusion-models/latent_physical_SGM && sbatch run_test18_latent_neuronic.sh

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
echo "== 0. unit check"
run - <<'EOF' || rc=$((rc | 1))
import sys
sys.path.insert(0, "/physical-diffusion-models/latent_physical_SGM")
from functools import partial
import numpy as np
import latent_model as lm
import jax
import jax.numpy as jnp
import jax.random as jr

prob = lm.make_problem("mnist14")
n = prob["n"]
vis = lm.DenseVisible(prob["conn"], n)
theta0, _ = lm.paper_theta_layout(vis.P, n)
K, T, S, B = 8, lm.TEMP, 8, 16
rng = np.random.default_rng(3)
X = jnp.asarray(np.asarray(prob["data"][:B]) + 0.3 * rng.normal(size=(B, n)))
params = {"theta": jnp.asarray(theta0), "W": jnp.asarray(0.001 * rng.normal(size=(K, n))), "b": jnp.zeros(K),
          "Q": jnp.asarray(0.0005 * rng.normal(size=(K, n))), "wl": jnp.asarray(0.01 * rng.normal(size=n)),
          "bl": jnp.zeros(n)}
ok = True
for trial in range(3):
    key = jr.PRNGKey(77 + trial)
    L = lm.make_loss_per_sample(vis, T, K, "potts", True, samples=S)(params, X, key)
    k_h, k_l = jr.split(key)
    st_h = partial(lm.sampled_state_mean, T=T, key=k_h, S=S)
    st_l = partial(lm.sampled_spin_mean, T=T, key=k_l, S=S)
    gv, lapv = vis.grad_lap(params["theta"], X)
    gh, laph, A, ph = lm.hidden_terms_quad(params["W"], params["Q"], params["b"], X, T, "potts", stat=st_h)
    gl, lapl, _, _ = lm.local_terms(params["wl"], params["bl"], X, T, stat=st_l)
    g, lap = gv + gh + gl, lapv + laph + lapl
    L_manual = -lap / T + jnp.sum(g * g, axis=1) / (2.0 * T * T)
    occ = lm.loss_draw_occupation(params, X, T, key, S)
    d_loss = float(jnp.max(jnp.abs(L - L_manual)))
    d_occ = float(jnp.max(jnp.abs(jnp.mean(ph, axis=0) - occ)))
    good = d_loss <= 1e-10 * float(jnp.max(jnp.abs(L))) and d_occ == 0.0 and abs(float(jnp.sum(occ)) - 1.0) < 1e-12
    ok &= good
    print(f"  key {77 + trial}: |loss - manual loss| {d_loss:.1e}, |batch mean of the loss's p_hat - "
          f"loss_draw_occupation| {d_occ:.1e}, occupation {np.round(np.asarray(occ), 4).tolist()} "
          f"{'ok' if good else 'FAIL'}")
print("unit check:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF

set -x
run ${D}/train_latent.py --name smoke18_m14_P128_S16jk_occ_${J} ${P128} --hidden-norm occupancy --hidden-samples 16 --hidden-jackknife --max-slices 2 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke18_m14_P8_S8jk_occsame_${J} ${P8} --hidden-norm occupancy --hidden-samples 8 --hidden-jackknife --occ-same-draws --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
bad=0
run ${D}/train_latent.py --name smoke18_bad1_${J} ${P8} --hidden-samples 8 --hidden-jackknife --occ-same-draws --max-slices 1 && bad=1
run ${D}/train_latent.py --name smoke18_bad2_${J} ${P8} --hidden-norm occupancy --occ-same-draws --max-slices 1 && bad=1
[ ${bad} -eq 0 ] || rc=$((rc | 8))
run ${D}/train_latent.py --name smoke18_m14_P128_S8jk_occsame_${J} ${P128} --hidden-norm occupancy --hidden-samples 8 --hidden-jackknife --occ-same-draws --max-slices 3 || rc=$((rc | 16))
set +x

echo "== 1. visit-count rule without the flag: validation loss of slices 0-1, new vs the production run (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke18_m14_P128_S16jk_occ_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_S16jk_occ/train_log.jsonl | head -2)
echo "== 2. smoke with the flag (Potts 8 + both)"
grep -o -E '"(slice|val_loss|W_norm|eff_states)": [-0-9.e]+' ${OUT}/smoke18_m14_P8_S8jk_occsame_${J}/train_log.jsonl | paste - - - -
grep -o -E '"(hidden_samples|hidden_norm|occ_same_draws)": ("[a-z]*"|[0-9]+|true|false)' ${OUT}/smoke18_m14_P8_S8jk_occsame_${J}/config.json | tr '\n' ' '; echo
echo "== 3. invalid combinations refused: $([ ${bad} -eq 0 ] && echo yes || echo NO)"
echo "== 4. Potts 128 + both, S = 8 visit-count rule, slices 0-2: same draws (new) vs fresh draws (production)"
for r in smoke18_m14_P128_S8jk_occsame_${J} mnist14_K128_ema_potts_quad_local_S8jk_occ; do
  echo "-- ${r}"
  head -3 ${OUT}/${r}/train_log.jsonl | grep -o -E '"(slice|val_loss|eff_states|mean_max_p|W_norm|Q_norm)": [-0-9.e]+' | paste - - - - - -
done
echo "Finished at $(date) (exit code ${rc}: bits 1 unit check, 2 regression run, 4 smoke, 8 invalid accepted, 16 Potts 128)"
exit ${rc}
