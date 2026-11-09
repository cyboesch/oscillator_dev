#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=latent_test19
#SBATCH --output=latent_test19_%j.out
#SBATCH --error=latent_test19_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306
# Nineteenth validation round, after adding --pair-terms (which visible pair-coupling groups exist: lin c_lin
# (x_i - x_j)^2, om c_om x_i^2 x_j, duff c_duff (x_i - x_j)^4 / 4; the others held at exactly zero) and
# --neighbourhood von_neumann (pairs within Manhattan instead of Chebyshev distance --radius). mnist14, 0/1:
#   0. unit checks: connectivity against brute force (Chebyshev and Manhattan, radius 1-3; the default call equals
#      the paper helper's list); parse_pair_terms; dropped_pair_slices (disjoint, P entries each, the right groups);
#      sample_latent.make_drift rebuilds a von Neumann model (parameter count) and refuses a mismatched one;
#   1. default path unchanged: Potts 128 + both, power 4, slices 0-1 at full steps, validation loss equal to the
#      production run mnist14_K128_ema_potts_quad_local_quartic;
#   2. smokes (Potts 8 + both, power 4, radius 1, 3 slices at 2% of the steps): von Neumann with lin only, Chebyshev
#      with lin + om, and no pair terms; the config records the options, the log the pair count, and the saved
#      visible parameters have every dropped group exactly zero in every slice and the kept groups non-zero;
#   3. invalid options refused (an unknown pair term; von Neumann without --radius).
# Submit from this folder:   cd physical-diffusion-models/latent_physical_SGM && sbatch run_test19_latent_neuronic.sh

set -uo pipefail
echo "Starting job on $(hostname) in $(pwd) at $(date)"
IMAGE="../../container/container_image.sif"
HOST_PROJECTS_DIR="$(realpath ..)"
RUNTIME_ARGS="--nv --bind ${HOST_PROJECTS_DIR}:/physical-diffusion-models"
ls -l "${IMAGE}" || { echo "Error: container image not found (submit from physical-diffusion-models/latent_physical_SGM/)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
export JAX_PLATFORMS=cuda,cpu
D=/physical-diffusion-models/latent_physical_SGM
OUT=../out/latent_physical_SGM
J=${SLURM_JOB_ID}
run() { apptainer exec ${RUNTIME_ARGS} ${IMAGE} python3 -u "$@"; }
P128="--problem mnist14 --hidden potts --K 128 --quad --local --select ema --visible quartic"
P8="--problem mnist14 --hidden potts --K 8 --quad --local --select ema --visible quartic"

rc=0
echo "== 0. unit checks"
run - <<'EOF' || rc=$((rc | 1))
import sys
sys.path.insert(0, "/physical-diffusion-models/latent_physical_SGM")
import numpy as np
import latent_model as lm
import sample_latent as sl
from physical_diffusion_fns.network_fns import create_2d_square_lattice_connectivity
import jax.numpy as jnp

ok = True
def check(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(f"  {name}: {'ok' if good else 'FAIL'} {extra}")

for grid in (14, 28):
    for r in (1, 2, 3):
        idx = np.arange(grid * grid)
        rr, cc = np.divmod(idx, grid)
        I, Jj = np.triu_indices(grid * grid, 1)
        dr, dc = np.abs(rr[I] - rr[Jj]), np.abs(cc[I] - cc[Jj])
        cheb_bf = {(int(a), int(b)) for a, b in zip(I[np.maximum(dr, dc) <= r], Jj[np.maximum(dr, dc) <= r])}
        manh_bf = {(int(a), int(b)) for a, b in zip(I[dr + dc <= r], Jj[dr + dc <= r])}
        cheb = lm.connectivity(grid, r)
        vn = lm.connectivity(grid, r, "von_neumann")
        cheb_set = {(int(a), int(b)) for a, b in cheb}
        vn_set = {(int(a), int(b)) for a, b in vn}
        check(f"grid {grid} radius {r} chebyshev = brute force", cheb_set == cheb_bf, f"({len(cheb_set)} pairs)")
        check(f"grid {grid} radius {r} von_neumann = brute force", vn_set == manh_bf and vn_set <= cheb_set,
              f"({len(vn_set)} pairs)")
default_new = lm.connectivity()
default_old = np.asarray(create_2d_square_lattice_connectivity(grid_size=lm.GRID, n_neighbour_couplings=lm.N_NEIGHBOUR_COUPLINGS))
check("default connectivity unchanged", default_new.shape == default_old.shape and np.array_equal(default_new, default_old),
      f"({default_new.shape[0]} pairs)")
check("parse all", lm.parse_pair_terms("all") == ("duff", "lin", "om"))
check("parse lin,om", lm.parse_pair_terms(" om , lin ") == ("lin", "om"))
check("parse none", lm.parse_pair_terms("none") == ())
for bad in ("foo", "lin,foo", ""):
    try:
        lm.parse_pair_terms(bad); check(f"parse refuses {bad!r}", False)
    except ValueError:
        check(f"parse refuses {bad!r}", True)
n, P = 196, 364
for kept, dropped in ((("lin",), ("duff", "om")), (("lin", "om"), ("duff",)), ((), ("duff", "lin", "om")),
                      (("duff", "lin", "om"), ())):
    sl_ = lm.dropped_pair_slices(n, P, kept)
    want = tuple(lm.PAIR_TERMS[t](n, P) for t in dropped)
    cover = np.zeros(4 * n + 3 * P, int)
    for s in sl_:
        cover[s] += 1
    good = sl_ == want and cover.max(initial=0) <= 1 and all(s.stop - s.start == P for s in sl_)
    check(f"dropped slices for kept {kept}", good, f"{[(s.start, s.stop) for s in sl_]}")
check("group slices tile the pair block", lm.clin_slice(n, P).stop == lm.com_slice(n, P).start and
      lm.com_slice(n, P).stop == lm.cduff_slice(n, P).start and lm.clin_slice(n, P).start == 3 * n)
# the sampler rebuilds a von Neumann model from its config
nt = len(lm.forward_time_pts())
K = 2
conn_vn = lm.connectivity(14, 1, "von_neumann")
n_theta = 4 * n + 3 * conn_vn.shape[0]
rng = np.random.default_rng(0)
th = np.tile(np.asarray(lm.paper_theta_layout(conn_vn.shape[0], n)[0]), (nt, 1)) + 0.01 * rng.standard_normal((nt, n_theta))
Wh, bh = 0.01 * rng.standard_normal((nt, K, n)), np.zeros((nt, K))
drift, _ = sl.make_drift(th, Wh, bh, "sde", "mnist14", "potts", None, None, 1, "von_neumann")
x = jnp.asarray(rng.standard_normal(n))
d = np.asarray(drift(0.5, x, None))
check("make_drift von_neumann radius 1 builds and is finite", np.all(np.isfinite(d)), f"(|drift| {np.linalg.norm(d):.3g})")
try:
    sl.make_drift(th, Wh, bh, "sde", "mnist14", "potts", None, None, 1, "chebyshev")
    check("make_drift refuses a mismatched neighbourhood", False)
except ValueError:
    check("make_drift refuses a mismatched neighbourhood", True)
print("unit checks:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF

set -x
run ${D}/train_latent.py --name smoke19_m14_P128q_${J} ${P128} --max-slices 2 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke19_m14_P128s_${J} --problem mnist14 --hidden potts --K 128 --quad --local --select ema --max-slices 2 || rc=$((rc | 2))
run ${D}/train_latent.py --name smoke19_m14_P8q_vn1_lin_${J} ${P8} --radius 1 --neighbourhood von_neumann --pair-terms lin --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke19_m14_P8q_r1_linom_${J} ${P8} --radius 1 --pair-terms lin,om --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
run ${D}/train_latent.py --name smoke19_m14_P8q_r1_none_${J} ${P8} --radius 1 --pair-terms none --max-slices 3 --steps-scale 0.02 || rc=$((rc | 4))
bad=0
run ${D}/train_latent.py --name smoke19_bad1_${J} ${P8} --radius 1 --pair-terms lin,foo --max-slices 1 && bad=1
run ${D}/train_latent.py --name smoke19_bad2_${J} ${P8} --neighbourhood von_neumann --max-slices 1 && bad=1
[ ${bad} -eq 0 ] || rc=$((rc | 8))
set +x

echo "== 1. default path: validation loss of slices 0-1, new vs the production run (must agree)"
paste <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke19_m14_P128q_${J}/train_log.jsonl) \
      <(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_quartic/train_log.jsonl | head -2)
a=$(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke19_m14_P128q_${J}/train_log.jsonl | head -2 | tr '\n' ' ')
b=$(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local_quartic/train_log.jsonl | head -2 | tr '\n' ' ')
if [ "$a" = "$b" ] && [ -n "$a" ]; then echo "default path (power 4): identical"; else echo "default path (power 4): DIFFERENT"; rc=$((rc | 16)); fi
a=$(grep -o '"val_loss": [-0-9.e]*' ${OUT}/smoke19_m14_P128s_${J}/train_log.jsonl | head -2 | tr '\n' ' ')
b=$(grep -o '"val_loss": [-0-9.e]*' ${OUT}/mnist14_K128_ema_potts_quad_local/train_log.jsonl | head -2 | tr '\n' ' ')
echo "power 6: new $a / production $b"
if [ "$a" = "$b" ] && [ -n "$a" ]; then echo "default path (power 6): identical"; else echo "default path (power 6): DIFFERENT"; rc=$((rc | 16)); fi
echo "== 2. smokes: config, pair count, dropped groups exactly zero in every slice"
for r in smoke19_m14_P8q_vn1_lin_${J}:lin smoke19_m14_P8q_r1_linom_${J}:lin,om smoke19_m14_P8q_r1_none_${J}:none; do
  name=${r%%:*}; kept=${r##*:}
  grep -o -E '"(pair_terms|neighbourhood|radius)": ("[a-z_,]*"|[0-9]+|null)' ${OUT}/${name}/config.json | tr '\n' ' '; echo
  grep -o -E '[0-9]+ pairs' ${OUT}/${name}/train_log.txt | head -1
  run - "/physical-diffusion-models/out/latent_physical_SGM/${name}" "${kept}" <<'EOF' || rc=$((rc | 32))
import sys, glob, os
sys.path.insert(0, "/physical-diffusion-models/latent_physical_SGM")
import numpy as np
import latent_model as lm
run_dir, kept = sys.argv[1], sys.argv[2]
kept = lm.parse_pair_terms(kept)
files = sorted(glob.glob(os.path.join(run_dir, "slices", "slice_*.npz")))
ok = len(files) == 3
for f in files:
    z = np.load(f)
    th = z["theta"] if "theta" in z.files else z[z.files[0]]
    n = 196
    P = (th.shape[-1] - 4 * n) // 3
    for t in ("lin", "om", "duff"):
        v = np.asarray(th[..., lm.PAIR_TERMS[t](n, P)])
        if t in kept:
            good = np.any(v != 0.0)
        else:
            good = np.all(v == 0.0)
        ok &= bool(good)
        print(f"  {os.path.basename(f)} {t}: max |c| {np.max(np.abs(v)) if v.size else 0:.3g} ({'kept' if t in kept else 'dropped'}) {'ok' if good else 'FAIL'}")
print(os.path.basename(run_dir), "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF
done
echo "== 3. invalid options refused: $([ ${bad} -eq 0 ] && echo yes || echo NO)"
echo "Finished at $(date) (exit code ${rc}: bits 1 unit checks, 2 default run, 4 smoke, 8 invalid accepted, 16 default path differs, 32 dropped groups not zero)"
exit ${rc}
