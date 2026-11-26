#!/usr/bin/env python3
"""Writes train_tree.py = the locked trainer (../latent_physical_SGM/train_latent.py) plus the option --tree
(2026-11-25). The locked file is read, not changed; every edit below must match exactly once, otherwise this script
stops (so a changed train_latent.py cannot be patched silently wrongly).

--tree: the global Potts unit and the Potts patch units are coupled by tables J (tree_model.py); all parameters are
trained together from the start. What the edits do:
 1. imports: latent_model from the locked folder, tree_model from this one; the code hashes in config.json;
 2. --out-root defaults to this folder's out/ (nothing is written to the locked out/ by default);
 3. options --tree and --tree-lr-scale, the guard (the locked recipe's kind of model, exact statistics) and the
    config keys tree, tree_lr_scale, tree_model_sha256;
 4. the training loss and the validation loss: tree_model.make_tree_loss / make_tree_loss_per_sample (validation in
    chunks of 512; 128 above 32x32);
 5. the tables J, J2, J3 ride on the trainer's mechanism for the label parameters of --cond (created as zeros when
    their patch set switches on, saved in every slice file, taken up on a resume, stacked into hidden_history.npz);
 6. their learning rate: the patch units' bias rate (per patch) x --tree-lr-scale.
--init-from / --init-slices work with --tree (the linked slices carry J; used to re-train late slices with more steps).
Without --tree the trainer runs the locked code path (checked by a short run against train_latent.py).

Usage: python3 make_train_tree.py   (run once; rerun after any change of the locked train_latent.py)
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "latent_physical_SGM", "train_latent.py")

EDITS = [
    # 1. imports
    ('''HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402  (enables float64)
''',
     '''HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(os.path.dirname(HERE), "latent_physical_SGM")     # the locked code: imported, not changed
sys.path.insert(0, MAIN)
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402  (enables float64)
import tree_model as tm  # noqa: E402  (--tree)
'''),
    ('''    code_sha = {f: lm.sha256_file(os.path.join(HERE, f)) for f in ("latent_model.py", "train_latent.py")}''',
     '''    code_sha = {"latent_model.py": lm.sha256_file(os.path.join(MAIN, "latent_model.py")),
                "train_latent.py": lm.sha256_file(os.path.abspath(__file__))}     # this file (train_tree.py)'''),
    # 2. output root
    ('''    ap.add_argument("--out-root", default=lm.OUT_ROOT)''',
     '''    ap.add_argument("--out-root", default=os.path.join(HERE, "out"))
    ap.add_argument("--tree", action="store_true",
                    help="couple the global Potts unit and the Potts patch units by tables J (tree_model.py), all "
                         "hidden units still summed exactly; needs the locked recipe's model (--hidden potts --quad "
                         "--local --patch ...) and exact statistics")
    ap.add_argument("--tree-lr-scale", type=float, default=1.0,
                    help="--tree: factor on J's learning rate (the patch units' bias rate)")'''),
    # 3. guard and config
    ('''    if args.cond:                                # --cond only: other configs as before
        config["cond"] = int(args.cond)''',
     '''    if args.tree:                                # --tree only: other configs as before
        if (args.hidden != "potts" or not args.quad or args.K < 2 or not args.local or not args.patch
                or args.hidden_samples or args.hidden_rule != "exact" or args.hidden_norm != "element"
                or args.hidden_floor > 0 or args.harmonic or args.hidden_fixed or args.step_impl != "default"
                or args.patch_kind != "potts" or args.split_dead or args.cond):
            raise ValueError("--tree needs --hidden potts --quad --local, K >= 2, Potts --patch units, exact statistics, "
                             "--hidden-norm element, --step-impl default and none of --hidden-floor / --harmonic / "
                             "--hidden-fixed / --split-dead / --cond")
        config["tree"] = True
        config["tree_lr_scale"] = float(args.tree_lr_scale)
        config["tree_model_sha256"] = lm.sha256_file(os.path.abspath(tm.__file__))
    if args.cond:                                # --cond only: other configs as before
        config["cond"] = int(args.cond)'''),
    # 4. losses
    ('''    loss_ps_k0 = jax.jit(lm.make_loss_per_sample(vis_dense, lm.TEMP, 0))
''',
     '''    loss_ps_k0 = jax.jit(lm.make_loss_per_sample(vis_dense, lm.TEMP, 0))
    if args.tree:                                # the tree in place of the global unit's and the patch units' terms
        loss_fn = tm.make_tree_loss(vis, lm.TEMP, args.local, compute_dtype=cdt)
        _tree_ps = jax.jit(tm.make_tree_loss_per_sample(vis_dense, lm.TEMP, args.local))

        _vc = 512 if n_vis <= 1024 else 128      # 64x64: smaller chunks (GPU memory)

        def loss_ps(p, X):                       # validation in chunks: the tree's (P, B, K, S) arrays in float64
            return jnp.concatenate([_tree_ps(p, X[i:i + _vc]) for i in range(0, X.shape[0], _vc)])
'''),
    # 5. J as extra per-set tables
    ('''    ckeys = tuple(g for g in lm.COND_KEYS if g in cond_shapes)
''',
     '''    ckeys = tuple(g for g in lm.COND_KEYS if g in cond_shapes)
    if args.tree:                                # the tables J: created (zeros) when their patch set switches on, saved,
        cond_shapes["J"] = (geom.P, K, patch[2])  # resumed and stacked into hidden_history.npz like the label parameters
        for x in xsets:
            cond_shapes[tm.J_KEYS[x["n"] - 1]] = (x["geom"].P, K, x["spec"][2])
        ckeys = ckeys + tuple(kj for kj in tm.J_KEYS if kj in cond_shapes)
'''),
    ('''            if g in lm.COND_PATCH_KEYS and lm.PATCH_SETS[lm.COND_PATCH_KEYS.index(g)][0] not in params:
                continue
            if prev is not None and g in prev.files:''',
     '''            if g in lm.COND_PATCH_KEYS and lm.PATCH_SETS[lm.COND_PATCH_KEYS.index(g)][0] not in params:
                continue
            if g in tm.J_KEYS and lm.PATCH_SETS[tm.J_KEYS.index(g)][0] not in params:   # --tree: with its patch set
                continue
            if prev is not None and g in prev.files:'''),
    # 6. learning rate of J
    ('''        lrs = {g: v * args.lr_scale for g, v in lrs.items()}
        return lrs, train_hidden''',
     '''        for (_, kb_t, _), kj in zip(lm.PATCH_SETS, tm.J_KEYS):   # --tree: J at its patch units' bias rate (per patch)
            if kj in params:
                lrs[kj] = jnp.asarray(lrs[kb_t])[:, :, None] * args.tree_lr_scale
        lrs = {g: v * args.lr_scale for g, v in lrs.items()}
        return lrs, train_hidden'''),
]


def main():
    src = open(SRC).read()
    out = src
    for old, new in EDITS:
        if out.count(old) != 1:
            raise SystemExit(f"edit does not match exactly once ({out.count(old)} matches):\n{old[:200]}")
        out = out.replace(old, new)
    head = (f"# GENERATED by make_train_tree.py from ../latent_physical_SGM/train_latent.py (sha256 "
            f"{hashlib.sha256(src.encode()).hexdigest()}): the locked trainer plus --tree. Do not edit; edit "
            f"make_train_tree.py and regenerate.\n")
    lines = out.split("\n", 1)
    out = lines[0] + "\n" + head + lines[1] if lines[0].startswith("#!") else head + out
    with open(os.path.join(HERE, "train_tree.py"), "w") as f:
        f.write(out)
    print(f"wrote train_tree.py ({len(EDITS)} edits, {out.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
