#!/usr/bin/env python3
"""The GPU k-means of the prototype reference (--ref-kmeans gpu, latent_model.kmeans_jax) against the numpy k-means
(latent_model.kmeans), and where the start-up time of the hidden-unit references goes. Two parts:

  cpu (a CPU job): numpy k-means at K = 784 and 3072 on the four-digit and the ten-digit data (k-means seed 0, as
      prototype_reference with --seed 1 and --proto-init pooled): wall time; centres and assignments saved; the
      seeding indices of kmeans at K = 784; the time of the rest of prototype_reference (pooled covariance, 100
      solves); the Ising split-tree reference (split_reference, ten digits, K = 784 and 3136): total time, time in
      split_tree and in its numpy 2-means calls
  gpu <cpu job id> (a GPU job): kmeans_jax for the same cases (and K = 6144, ten digits): wall time (first call
      with compilation, second call), determinism, k-means objective vs the numpy one, centres, assignments, seeding
      indices at K = 784; W*, b* and the hidden learning-rate scales (rms over states per slice) from the GPU
      centres vs the numpy ones

Usage (from latent_physical_SGM/):
  sbatch --time=00:45:00 --cpus-per-task=4 --mem=16G jobs/run_cpu_neuronic.sh kmeans_check.py cpu
  sbatch --time=00:15:00 jobs/run_gpu_neuronic.sh kmeans_check.py gpu <cpu job id>
Writes out/latent_physical_SGM/kmeans_check/<job>/report.txt and results.json (cpu: also <case>.npz).
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402  (enables float64)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

JOB = os.environ.get("SLURM_JOB_ID", "manual")
OUT = os.path.join(lm.OUT_ROOT, "kmeans_check", JOB)
os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(OUT, "report.txt")
CASES = [("mnist28_d0123", 784), ("mnist28_d0123456789", 784), ("mnist28_d0123", 3072),
         ("mnist28_d0123456789", 3072), ("mnist28_d0123456789", 6144)]      # 6144: GPU only


def log(msg=""):
    print(msg, flush=True)
    with open(REPORT, "a") as f:
        f.write(msg + "\n")


def objective(X, C, a):
    """k-means objective: sum over points of the squared distance to the assigned centre."""
    return float(np.sum((X - C[a]) ** 2))


def cpu_seeds(X, k, seed):
    """The seeding indices of lm.kmeans (the same draws in the same order)."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(X.shape[0]))]
    d2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
    for _ in range(1, k):
        idx.append(int(rng.choice(X.shape[0], p=d2 / d2.sum())))
        d2 = np.minimum(d2, np.sum((X - X[idx[-1]]) ** 2, axis=1))
    return np.array(idx)


def gpu_seeds(X, k, seed):
    """The seeding indices of lm.kmeans_jax."""
    rng = np.random.default_rng(seed)
    i0 = int(rng.integers(X.shape[0]))
    us = rng.random(k - 1)
    return np.asarray(lm._kmeanspp_device(jnp.asarray(X), jnp.asarray(i0), jnp.asarray(us)))


DATA = {}


def data_of(problem):
    if problem not in DATA:
        prob = lm.make_problem(problem, 1)
        DATA[problem] = (np.asarray(prob["data"], dtype=np.float64), prob["labels"])
    return DATA[problem]


def lr_scales(W, b):
    return np.sqrt(np.mean(W ** 2, axis=(1, 2))), np.maximum(np.sqrt(np.mean(b ** 2, axis=1)), lm.TEMP)


def part_cpu():
    out = {}
    taus = lm.forward_time_pts()
    for problem, K in CASES:
        if K > 3072:
            continue
        X, labels = data_of(problem)
        name = f"{problem}_K{K}"
        t = time.time()
        C, a = lm.kmeans(X, K, 0)
        r = {"n": int(X.shape[0]), "cpu_s": time.time() - t, "cpu_objective": objective(X, C, a),
             "cpu_empty_clusters": int(K - np.unique(a).size)}
        t = time.time()
        lm.prototype_reference(X, labels, taus, lm.TEMP, K, seed=0, pooled=True, kmeans_fn=lambda Xa, k, s: (C, a))
        r["reference_rest_s"] = time.time() - t
        extra = {}
        if K == 784:
            extra["seeds"] = cpu_seeds(X, K, 0)
        np.savez(os.path.join(OUT, f"{name}.npz"), C=C, assign=a, **extra)
        out[name] = r
        log(f"  {name}: " + ", ".join(f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    X, labels = data_of("mnist28_d0123456789")
    orig_km, orig_tree = lm.kmeans, lm.split_tree
    for K in (784, 3136):
        acc = {"km_s": 0.0, "km_calls": 0, "tree_s": 0.0}

        def km(Xa, k, seed, iters=50):
            t = time.time()
            res = orig_km(Xa, k, seed, iters)
            acc["km_s"] += time.time() - t
            acc["km_calls"] += 1
            return res

        def tree(*a, **kw):
            t = time.time()
            res = orig_tree(*a, **kw)
            acc["tree_s"] += time.time() - t
            return res

        lm.kmeans, lm.split_tree = km, tree
        try:
            t = time.time()
            lm.split_reference(X, labels, taus, lm.TEMP, K, seed=0, pooled=True, n_min=lm.tree_n_min(len(X), K))
            acc["total_s"] = time.time() - t
        finally:
            lm.kmeans, lm.split_tree = orig_km, orig_tree
        out[f"ising_K{K}"] = acc
        log(f"  ising split_reference ten digits K {K}: total {acc['total_s']:.1f} s, split_tree {acc['tree_s']:.1f} s, "
            f"{acc['km_calls']} numpy 2-means calls {acc['km_s']:.1f} s")
    return out


def part_gpu(cpu_job):
    out = {}
    taus = lm.forward_time_pts()
    cdir = os.path.join(lm.OUT_ROOT, "kmeans_check", cpu_job)
    for problem, K in CASES:
        X, labels = data_of(problem)
        name = f"{problem}_K{K}"
        t = time.time()
        Cg, ag = lm.kmeans_jax(X, K, 0)
        r = {"n": int(X.shape[0]), "gpu_first_s": time.time() - t}          # includes compilation
        t = time.time()
        Cg2, ag2 = lm.kmeans_jax(X, K, 0)
        r["gpu_s"] = time.time() - t
        r["gpu_deterministic"] = bool(np.array_equal(Cg, Cg2) and np.array_equal(ag, ag2))
        r["gpu_objective"] = objective(X, Cg, ag)
        r["gpu_empty_clusters"] = int(K - np.unique(ag).size)
        t = time.time()
        Wg, bg, _ = lm.prototype_reference(X, labels, taus, lm.TEMP, K, seed=0, pooled=True,
                                           kmeans_fn=lambda Xa, k, s: (Cg, ag))
        r["reference_rest_s"] = time.time() - t
        cpath = os.path.join(cdir, f"{name}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            Cc, ac = z["C"], z["assign"]
            r["cpu_objective"] = objective(X, Cc, ac)
            r["objective_rel_diff"] = (r["gpu_objective"] - r["cpu_objective"]) / r["cpu_objective"]
            r["centres_max_abs_diff"] = float(np.max(np.abs(Cg - Cc)))
            r["assignments_equal_share"] = float(np.mean(ag == ac))
            r["bitwise_equal"] = bool(np.array_equal(Cg, Cc) and np.array_equal(ag, ac))
            if "seeds" in z.files:
                sg = gpu_seeds(X, K, 0)
                r["seeding_indices_differing"] = int(np.sum(sg != z["seeds"]))
            Wc, bc, _ = lm.prototype_reference(X, labels, taus, lm.TEMP, K, seed=0, pooled=True,
                                               kmeans_fn=lambda Xa, k, s: (Cc, ac))
            r["W_ref_rel_diff_slices_0_50_99"] = [float(np.linalg.norm(Wg[k] - Wc[k]) / np.linalg.norm(Wc[k]))
                                                  for k in (0, 50, 99)]
            (sWc, sbc), (sWg, sbg) = lr_scales(Wc, bc), lr_scales(Wg, bg)
            r["lr_scale_W_max_rel_diff"] = float(np.max(np.abs(sWg / sWc - 1.0)))
            r["lr_scale_b_max_rel_diff"] = float(np.max(np.abs(sbg / sbc - 1.0)))
            del Wc, bc
        del Wg, bg
        out[name] = r
        log(f"  {name}: " + ", ".join(f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    log(f"kmeans_check job {JOB} ({mode}), {time.strftime('%Y-%m-%d %H:%M:%S')}; jax {jax.__version__}, "
        f"{jax.devices()}; OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS')}")
    res = part_cpu() if mode == "cpu" else part_gpu(sys.argv[2])
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"job": JOB, "mode": mode, "results": res}, f, indent=1, default=str)
    log(f"\nreport: {REPORT}")


if __name__ == "__main__":
    main()
