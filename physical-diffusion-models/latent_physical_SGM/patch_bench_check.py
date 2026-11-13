#!/usr/bin/env python3
"""E0 of the patch Potts build (train_latent.py --patch; spec out/latent_physical_SGM/patch_design/judge/final_spec.md):
the GPU cost, the reproducibility across processes and the --patch-on-slice identity, from finished run folders only
(reads only; a CPU job).

  1. step time per slice: steps, seconds and steps/s of every slice of every run (train_log.jsonl), and the ratio of
     steps/s to the --ref run's (the same job's plain run: the same GPU node) and to the --plain run (the plain seed-1
     study run, job 4222701) at the same slice. Slice 0 includes the compilation; the cost is read from slices >= 1.
  2. reproducibility (--pair A,B, repeatable; the first pair is the baseline, e.g. two plain runs of the same job):
     per slice, every array of the slice files bit for bit, else the largest relative difference per member; the
     records but for the timing fields (seconds, steps_per_s, job, code_sha256). Verdict for every later pair: bit for
     bit if the baseline is, otherwise no worse than 10 x the baseline's largest relative difference.
  3. switch-on identity (--switch ON,PLAIN, repeatable): ON is a --patch-on-slice K run. In slices k < K its members
     theta, W, b, Q, wl, bl (V, c) and val_per_sample equal PLAIN's (bit for bit if the baseline pair is, else to its
     largest relative difference), its patch arrays Wp, bp (Qp) are exactly zero, and its records equal PLAIN's but for
     the timing fields; at slice K the units are present (nonzero, finite) with patch_* record keys.
  4. every array of every slice file finite.
Usage (from latent_physical_SGM/):
  sbatch --time=00:10:00 jobs/run_cpu_neuronic.sh patch_bench_check.py RUN ... --ref RUN [--plain RUN] \
      [--pair A,B ...] [--switch ON,PLAIN ...]
Writes out/latent_physical_SGM/patch_bench_check/<job>/report.txt and results.json; exit code 1 if a check fails.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_model as lm  # noqa: E402

OUT = os.path.join(lm.OUT_ROOT, "patch_bench_check", os.environ.get("SLURM_JOB_ID", "manual"))
PLAIN = "mnist28_d0123_K784_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_seed1"     # job 4222701
VOL = {"seconds", "steps_per_s", "job", "code_sha256"}
LINES, FAILED = [], []


def out(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def check(name, ok, detail=""):
    out(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def records(run):
    p = os.path.join(lm.OUT_ROOT, run, "train_log.jsonl")
    recs = {}
    if os.path.exists(p):
        with open(p) as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    if "slice" in r:
                        recs[int(r["slice"])] = r
    return recs


def slices(run):
    d = os.path.join(lm.OUT_ROOT, run, "slices")
    ks = []
    while os.path.exists(os.path.join(d, f"slice_{len(ks):03d}.npz")):
        ks.append(len(ks))
    return ks


def load(run, k):
    with np.load(os.path.join(lm.OUT_ROOT, run, "slices", f"slice_{k:03d}.npz")) as z:
        return {m: z[m] for m in z.files}


def relmax(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def rec_diff(ra, rb):
    return sorted(f for f in set(ra) | set(rb) if f not in VOL and json.dumps(ra.get(f)) != json.dumps(rb.get(f)))


def compare_runs(a, b, ks, members=None):
    """Per slice: (bitwise, {member: rel diff} of the arrays that differ, record fields that differ)."""
    rows = {}
    for k in ks:
        za, zb = load(a, k), load(b, k)
        ms = sorted((set(za) & set(zb)) - {"record"}) if members is None else [m for m in members if m in za]
        diff = {m: relmax(za[m], zb[m]) for m in ms if m in zb and not same(za[m], zb[m])}
        missing = [m for m in ms if m not in zb]
        extra = sorted(set(za) ^ set(zb)) if members is None else []
        rd = rec_diff(json.loads(str(za["record"])), json.loads(str(zb["record"])))
        rows[k] = {"bitwise": not diff and not missing and not extra, "diff": diff, "missing": missing,
                   "extra": extra, "record_diff": rd}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None, help="the speed reference among the runs (default: the first)")
    ap.add_argument("--plain", default=PLAIN, help="the plain seed-1 study run (job 4222701) for the speed table")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--switch", action="append", default=[])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out(f"=== patch_bench_check {time.strftime('%Y-%m-%d %H:%M:%S')}; job {os.environ.get('SLURM_JOB_ID', 'manual')}; "
        f"runs {args.runs}")
    res = {"runs": {}, "pairs": {}, "switch": {}}
    ref = args.ref or args.runs[0]
    recs = {r: records(r) for r in set(args.runs) | {ref, args.plain}}
    cfgs = {}
    for r in args.runs:
        with open(os.path.join(lm.OUT_ROOT, r, "config.json")) as f:
            cfgs[r] = json.load(f)
    # ---- 1. step time per slice -------------------------------------------------------------------------------
    out(f"\n== 1. steps/s per slice (ratio to {ref}, and to the plain seed-1 study run {args.plain})")
    for r in args.runs:
        rr = recs[r]
        rows = {}
        for k in sorted(rr):
            sp = float(rr[k]["steps_per_s"])
            q_ref = sp / float(recs[ref][k]["steps_per_s"]) if k in recs[ref] else None
            q_pl = sp / float(recs[args.plain][k]["steps_per_s"]) if k in recs[args.plain] else None
            rows[k] = {"steps": rr[k]["steps"], "seconds": rr[k]["seconds"], "steps_per_s": sp,
                       "ratio_to_ref": q_ref, "ratio_to_plain_4222701": q_pl}
        res["runs"][r] = {"K": cfgs[r].get("K"), "patch": cfgs[r].get("patch", ""),
                          "patch_on_slice": cfgs[r].get("patch_on_slice", 0), "slices": rows}
        cells = [f"slice {k}: {v['steps']} steps in {v['seconds']:.1f} s = {v['steps_per_s']:.0f}/s"
                 + (f" (x{v['ratio_to_ref']:.3f} {ref}" if v["ratio_to_ref"] is not None else " (")
                 + (f", x{v['ratio_to_plain_4222701']:.3f} plain)" if v["ratio_to_plain_4222701"] is not None else ")")
                 for k, v in rows.items()]
        out(f"  {r} (K {cfgs[r].get('K')}, patch '{cfgs[r].get('patch', '')}', on slice "
            f"{cfgs[r].get('patch_on_slice', 0)}): " + "; ".join(cells))
        late = [v["ratio_to_ref"] for k, v in rows.items() if k >= 1 and v["ratio_to_ref"] is not None]
        if late:
            res["runs"][r]["cost_vs_ref_slices_ge1"] = float(1.0 / np.mean(late))
            out(f"    step time relative to {ref} (slices >= 1): x{1.0 / np.mean(late):.3f}")
    # ---- 4. finite --------------------------------------------------------------------------------------------
    bad = []
    for r in args.runs:
        for k in slices(r):
            z = load(r, k)
            bad += [f"{r}:{k}:{m}" for m, v in z.items() if m != "record" and np.issubdtype(np.asarray(v).dtype,
                                                                                             np.floating)
                    and not np.all(np.isfinite(v))]
    check("4. every array of every slice file is finite", not bad, f"not finite: {bad[:8]}" if bad else "")
    # ---- 2. reproducibility -----------------------------------------------------------------------------------
    out("\n== 2. reproducibility across processes")
    base_bitwise, base_rel = None, None
    for i, pr in enumerate(args.pair):
        a, b = pr.split(",")
        ks = sorted(set(slices(a)) & set(slices(b)))
        rows = compare_runs(a, b, ks)
        bitwise = all(v["bitwise"] for v in rows.values()) and bool(ks)
        worst = max([max(v["diff"].values(), default=0.0) for v in rows.values()], default=0.0)
        rdiff = sorted({f for v in rows.values() for f in v["record_diff"]})
        res["pairs"][pr] = {"slices": ks, "bitwise": bitwise, "worst_rel": worst, "record_fields_differing": rdiff,
                            "per_slice": rows}
        for k, v in rows.items():
            out(f"  {a} vs {b}, slice {k}: " + ("bit for bit" if v["bitwise"] else
                                               f"differing {', '.join(f'{m} {d:.2e}' for m, d in v['diff'].items())}"
                                               f"{' missing ' + str(v['missing']) if v['missing'] else ''}"
                                               f"{' members ' + str(v['extra']) if v['extra'] else ''}")
                + (f"; record fields differing {v['record_diff']}" if v["record_diff"] else ""))
        if i == 0:
            base_bitwise, base_rel = bitwise, worst
            check(f"2. baseline pair {pr}: {len(ks)} slices compared", bool(ks),
                  "bit for bit" if bitwise else f"largest relative difference {worst:.2e}")
        else:
            ok = bitwise if base_bitwise else (worst <= max(10.0 * base_rel, 1e-12))
            check(f"2. pair {pr} as reproducible as the baseline ({'bit for bit' if base_bitwise else f'{base_rel:.2e}'})",
                  ok and bool(ks), "bit for bit" if bitwise else f"largest relative difference {worst:.2e}")
    # ---- 3. switch-on identity --------------------------------------------------------------------------------
    out("\n== 3. --patch-on-slice identity")
    for sw in args.switch:
        on, pl = sw.split(",")
        K_on = int(cfgs.get(on, {}).get("patch_on_slice", 0)) if on in cfgs else 0
        if on not in cfgs:
            with open(os.path.join(lm.OUT_ROOT, on, "config.json")) as f:
                K_on = int(json.load(f).get("patch_on_slice", 0))
        ks = [k for k in sorted(set(slices(on)) & set(slices(pl))) if k < K_on]
        mem = ["theta", "W", "b", "Q", "wl", "bl", "V", "c", "val_per_sample"]
        rows = compare_runs(on, pl, ks, members=mem)
        zero_bad, rec_bad, worst = [], [], 0.0
        for k in ks:
            z = load(on, k)
            zero_bad += [f"{k}:{m}" for m in ("Wp", "bp", "Qp") if m in z and np.any(np.asarray(z[m]) != 0.0)]
            zero_bad += [f"{k}: no Wp"] if "Wp" not in z else []
            rec_bad += [f"{k}: {rows[k]['record_diff']}"] if rows[k]["record_diff"] else []
            worst = max(worst, max(rows[k]["diff"].values(), default=0.0))
        bitwise = all(rows[k]["bitwise"] for k in ks) and bool(ks)
        on_ok, detail_on = False, "slice K not present"
        if K_on in slices(on):
            z = load(on, K_on)
            r_k = json.loads(str(z["record"]))
            on_ok = ("Wp" in z and np.any(z["Wp"] != 0.0) and np.all(np.isfinite(z["Wp"])) and "patch_W_norm" in r_k)
            detail_on = f"|Wp| {np.linalg.norm(z['Wp']):.4g}, patch_eff median {r_k.get('patch_eff_states_median')}"
        ok_eq = bitwise if (base_bitwise or base_bitwise is None) else worst <= max(base_rel, 1e-12)
        res["switch"][sw] = {"K_on": K_on, "slices": ks, "bitwise": bitwise, "worst_rel": worst,
                             "nonzero_patch_arrays": zero_bad, "record_diff": rec_bad, "slice_K": detail_on}
        check(f"3. {on} (--patch-on-slice {K_on}) slices {ks} = {pl}'s (theta, W, b, Q, wl, bl, val_per_sample)",
              ok_eq and bool(ks), "bit for bit" if bitwise else f"largest relative difference {worst:.2e}")
        check(f"3. {on} slices {ks}: patch arrays exactly zero, records = {pl}'s but for timing", not zero_bad
              and not rec_bad, f"nonzero {zero_bad[:6]}; records {rec_bad[:4]}" if zero_bad or rec_bad else "")
        check(f"3. {on} slice {K_on}: patch units present, finite, with patch_* records", on_ok, detail_on)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    out(f"\n{'ALL OK' if not FAILED else 'FAILED: ' + '; '.join(FAILED)}")
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
