#!/bin/bash
#SBATCH --account=seas
#SBATCH --job-name=tree_trainer_test
#SBATCH --output=logs/tree_trainer_test_%j.out
#SBATCH --error=logs/tree_trainer_test_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=neu306
# Test of train_tree.py on the 8x8 toy problem (2026-11-25), CPU. Submit from latent_physical_SGM_tree_2026-11-25/.
#  A. without --tree it gives the locked trainer's slices bit for bit (4 slices, two patch sets switched on at 1 and 2);
#  B. with --tree: J is zero before its set's switch-on and trained after, the validation loss is finite;
#  D. --init-from / --init-slices continue a tree run's slices with other step counts (J carried over);
#  C. with the last slice file removed, the same call resumes from the saved J and reproduces that slice.
# Everything is written below out/test_trainer_<job>/ of this folder.
set -uo pipefail
IMAGE="../../container/container_image.sif"
B="apptainer exec --bind $(realpath ..):/physical-diffusion-models ${IMAGE} python3 -u"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
D=/physical-diffusion-models/latent_physical_SGM_tree_2026-11-25
O=${D}/out/test_trainer_${SLURM_JOB_ID}
C="--problem gauss2_8 --K 3 --hidden potts --quad --local --visible quartic --radius 1 --pair-terms lin --proto-init pooled --seed 1 --select ema --patch 4:2:3 --patch2 2:2:4 --patch-init-scale 0.06 --patch-on-slice 1 --patch2-on-slice 2 --steps-scale 0.02 --visible-impl stencil --precision mixed --out-root ${O}"
rc=0
echo "== A1 locked trainer"; ${B} /physical-diffusion-models/latent_physical_SGM/train_latent.py --name locked ${C} --max-slices 4 2>&1 | grep -E "^slice|Error|error|Traceback" | cut -c1-200 || rc=1
echo "== A2 train_tree.py without --tree"; ${B} ${D}/train_tree.py --name copy ${C} --max-slices 4 2>&1 | grep -E "^slice|Error|error|Traceback" | cut -c1-200 || rc=1
echo "== B train_tree.py --tree"; ${B} ${D}/train_tree.py --name tree ${C} --tree --max-slices 4 2>&1 | grep -E "^slice|Error|error|Traceback|tree" | cut -c1-260 || rc=1
echo "== C resume: slice 3 removed, the same call again"
HO=out/test_trainer_${SLURM_JOB_ID}/tree
cp ${HO}/slices/slice_003.npz ${HO}/slice_003_first.npz && rm -f ${HO}/slices/slice_003.npz ${HO}/DONE || rc=1
${B} ${D}/train_tree.py --name tree ${C} --tree --max-slices 4 2>&1 | grep -E "^slice|resum|Error|error|Traceback" | cut -c1-260 || rc=1
echo "== D --init-from tree --init-slices 3 with 4 x the steps in slice 3"
${B} ${D}/train_tree.py --name tree_4x ${C/--steps-scale 0.02/--steps-scale 0.08} --tree --max-slices 4 --init-from tree --init-slices 3 2>&1 | grep -E "^slice|resum|init|Error|error|Traceback" | cut -c1-260 || rc=1
${B} -c "
import numpy as np, json, os
O = '${O}'
bad = 0
for k in range(4):
    a = np.load(f'{O}/locked/slices/slice_{k:03d}.npz'); b = np.load(f'{O}/copy/slices/slice_{k:03d}.npz')
    same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[f], b[f]) for f in a.files if f != 'record')
    print(f'A. slice {k}: train_tree without --tree = locked trainer, every array bit for bit: {\"PASS\" if same else \"FAIL\"} ({len(a.files)} members)'); bad += not same
for k in range(4):
    z = np.load(f'{O}/tree/slices/slice_{k:03d}.npz')
    J, J2 = z['J'], z['J2']
    rec = json.loads(str(z['record']))
    ok = np.isfinite(z['val_per_sample']).all() and J.shape == (9, 3, 3) and J2.shape == (16, 3, 4) and (np.abs(J).max() > 0) == (k >= 1) and (np.abs(J2).max() > 0) == (k >= 2)
    print(f'B/C. slice {k}: J {J.shape} rms/T {np.sqrt((J**2).mean())/0.02:.3f}, J2 {J2.shape} rms/T {np.sqrt((J2**2).mean())/0.02:.3f}, val {z[\"val_per_sample\"].mean():.3f}: {\"PASS\" if ok else \"FAIL\"}'); bad += not ok
f = np.load(f'{O}/tree/slice_003_first.npz'); r = np.load(f'{O}/tree/slices/slice_003.npz')
d = max(float(np.abs(f[m] - r[m]).max()) for m in ('theta', 'W', 'J', 'J2', 'Wp', 'Wp2'))
print(f'C. resume: slice 3 trained again from the saved slice 2 (with its J): max |difference| to the first time {d:.1e}: {\"PASS\" if d < 1e-9 else \"FAIL\"}'); bad += not d < 1e-9
a = np.load(f'{O}/locked/slices/slice_003.npz'); t = np.load(f'{O}/tree/slices/slice_003.npz')
print(f'   slice 3 validation loss: locked {a[\"val_per_sample\"].mean():.3f}, tree {t[\"val_per_sample\"].mean():.3f} (same validation samples: {np.array_equal(a[\"val_idx\"], t[\"val_idx\"])})')
a2, b2 = np.load(f'{O}/tree/slices/slice_002.npz'), np.load(f'{O}/tree_4x/slices/slice_002.npz'); b3 = np.load(f'{O}/tree_4x/slices/slice_003.npz'); r3 = json.loads(str(b3['record']))
okd = np.array_equal(a2['J'], b2['J']) and np.abs(b3['J']).max() > 0 and np.isfinite(b3['val_per_sample']).all() and r3.get('n_steps', r3.get('steps', 0)) != 232
print(f'D. --init-from with --tree: slices 0-2 are the source run\'s (J equal), slice 3 trained with {r3.get(\"n_steps\", r3.get(\"steps\", \"?\"))} steps (232 before), J rms/T {np.sqrt((b3[\"J\"]**2).mean())/0.02:.3f}, val {b3[\"val_per_sample\"].mean():.3f} against {r[\"val_per_sample\"].mean():.3f} with 1 x: {\"PASS\" if okd else \"FAIL\"}'); bad += not okd
c = json.load(open(f'{O}/tree/config.json')); print('   config:', {k: c[k] for k in ('tree', 'tree_lr_scale')}, 'copy has tree key:', 'tree' in json.load(open(f'{O}/copy/config.json')))
print('test_trainer:', 'ALL PASSED' if not bad else f'{bad} FAILED')
raise SystemExit(1 if bad else 0)
" || rc=1
echo "Finished at $(date) with exit code ${rc}"
exit ${rc}
