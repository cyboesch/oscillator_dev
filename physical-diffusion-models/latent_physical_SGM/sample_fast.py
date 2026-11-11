#!/usr/bin/env python3
"""Fast reverse-SDE sampler for runs of train_latent.py (opt-in: sample_latent.py --fast).

Same SDE, solver, Brownian paths and parameter interpolation as sample_latent.py:
    dx = [x - 2 grad F(x; params(5 - t))] dt + sqrt(2T) dW,     reverse time t from 0 to 5,
SRA1 with PIDController(rtol 1e-4, atol 1e-6), first step dt0 = 1e-8, one VirtualBrownianTree over [0, 5] per
trajectory (key from latent_model.trajectory_set, SpaceTimeLevyArea, tolerance --brownian-tol, default 1e-12), and
every parameter linearly interpolated in forward time tau = 5 - t between the 100 slice knots exactly as
physical_diffusion_fns.helper_fns.interpolate_parameters (jnp.interp) does it:
    P(tau) = P_lo + l (P_hi - P_lo),   l = (tau - tau_lo) / (tau_hi - tau_lo),   tau_lo <= tau <= tau_hi neighbours.
What changes is how the drift is evaluated and how the knots are handled:

1. Knot intervals. The 99 intervals between consecutive knots are integrated one after the other, all trajectories
   of a chunk together (one vmapped diffeqsolve per interval), so that every trajectory is between the same two
   knots lo, hi. The fields and the hidden force are linear in W, Q, b, so instead of interpolating the K x N
   matrices for every trajectory and drift evaluation (sample_latent.make_drift), they are mixed from the two knots
   with each trajectory's own l, using one stacked matrix per interval
       M = [[W_lo^T, dW^T], [Q_lo^T, dQ^T]]   (2N x 2K; without --quad [W_lo^T, dW^T]),  dW = W_hi - W_lo:
       [x, x^2] M = [A_lo, dA]            a = (A_lo + b_lo) + l (dA + db)      (= W(t) x + Q(t) x^2 + b(t))
       [m, l m] M^T = [m W(t), m Q(t)]     hidden force m W(t) + 2 x (m Q(t)),  m = softmax(a/T) (Potts), tanh(a/T)
   That is two GEMMs per drift evaluation for the whole chunk (B x 2N x 2K each). The visible gradient is linear in
   theta too: -grad E(x; s th(t) - th_lin) = -[g(x; s th_lo - th_lin) + l g(x; s dth)] (s = 2 for the SDE), with the
   analytic batched gradient of latent_model (StencilVisible for small-radius lattices, DenseVisible otherwise,
   only the pair groups and k_6 that are nonzero somewhere in the run's history). The per-pixel units (wl, bl) are
   nonlinear in their parameters and are interpolated per trajectory (2N numbers). Equal to make_drift up to rounding
   (sampler_check.py section 2).
2. Step sizes. The solve over [0, 5] with step_ts = knots (sample_latent) steps onto every knot, and the controller
   then proposes the next step from the clipped step (made_jump is False for step_ts). Each interval solve here ends
   on the knot the same way, and the next one starts from the final PIDController state of the previous one
   (SaveAt(controller_state=True); dt0 = the step it proposed), so the step sequence is the same as in the single
   solve; only rounding-level differences of the drift can change it. The first interval starts at dt0 = 1e-8 with
   the state PIDController.init gives. SRA1 has no solver state.
3. Diagonal noise: ControlTerm with a lineax.DiagonalLinearOperator (sqrt(2T) * dW elementwise), instead of the
   dense sqrt(2T) * eye(784) matrix and its three matrix-vector products per step (the same numbers: the dense
   product only adds exact zeros).
4. No progress meter (TqdmProgressMeter makes a host callback per step); max_steps per interval (default 100,000;
   diffeqsolve raises if a trajectory needs more).
5. --precision mixed: the two hidden-unit GEMMs in float32 at HIGHEST precision (as latent_model.in_compute_dtype;
   XLA may use TF32 for float32 products otherwise); M is cast once per interval. Time, state, Brownian path,
   visible force, softmax and everything else stay float64.
6. --brownian-tol: the VirtualBrownianTree tolerance (the depth of its bisection: 43 levels at 1e-12, 29 at 1e-8,
   23 at 1e-6, two descents per step). The path is the same random function of the key at every tolerance, resolved
   to a different depth.

Only the reverse SDE (--sampler sde) has this path; the PF-ODE, --quench and --snapshots stay on sample_latent.py's
code. Use through sample_latent.py --fast (writes <run>/samples/sde_<set>_n<n><suffix>.npy, refuses to overwrite),
or FastSampler directly (sampler_check.py).
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
import diffrax  # noqa: E402
import lineax  # noqa: E402

PRECISIONS = ("f64", "mixed")
VISIBLE_IMPLS = ("auto", "stencil", "dense")
STENCIL_MAX_OFFSETS = 24          # 'auto': stencil up to 24 lattice offsets (Chebyshev radius 3), dense above
MAX_STEPS_INTERVAL = 100_000
N_KNOTS = lm.N_TIME_STEPS
N_INTERVALS = N_KNOTS - 1


def knot_times():
    """(order, xp, s). xp: the forward knot times in increasing order, exactly the grid jnp.interp sees in
    interpolate_parameters (lm.forward_time_pts()[order], order = argsort). s: the reverse-time knots
    sort(T_FORWARD - forward_time_pts()) (s[0] = 0, s[-1] = 5; s[1:-1] are sample_latent's step_ts).
    Reverse-time interval j = [s[j], s[j+1]] (j = 0..98) lies between the forward knots xp[lo] and xp[lo + 1],
    lo = 98 - j: the bracket jnp.interp uses for every t inside it."""
    taus = lm.forward_time_pts()
    order = np.argsort(taus)
    return order, taus[order], np.sort(lm.T_FORWARD - taus)


def interval_lo(j):
    """Index (in xp order) of the lower forward knot of reverse-time interval j."""
    return N_KNOTS - 2 - j


def lattice_offsets(conn, grid):
    """Number of distinct lattice offsets (dy, dx) of a pair list on a grid x grid lattice."""
    conn = np.asarray(conn)
    ri, ci = np.divmod(conn[:, 0], grid)
    rj, cj = np.divmod(conn[:, 1], grid)
    return len(set(zip((rj - ri).tolist(), (cj - ci).tolist())))


class FastSampler:
    """The reverse SDE of a trained run, integrated knot interval by knot interval (module docstring).

    Arguments as sample_latent.make_drift (theta_hist (100, n_theta), W_hist (100, K, N), b_hist (100, K), all in
    the run's slice order, t = 5 first), plus:
      precision     'f64' (default) or 'mixed' (the two hidden-unit GEMMs in float32 at HIGHEST precision)
      visible_impl  'auto' (stencil up to STENCIL_MAX_OFFSETS lattice offsets, else dense), 'stencil', 'dense'
      brownian_tol  VirtualBrownianTree tolerance (default lm.BROWNIAN_TOL = 1e-12, sample_latent's)
      rtol, atol, dt0  PIDController tolerances and the first step (default the paper's SDE settings)
      max_steps     per interval and trajectory
    sampler 'ode' builds the PF-ODE drift (for checks; solve_chunk integrates the SDE only).
    """

    def __init__(self, theta_hist, W_hist, b_hist, problem="mnist28", hidden="ising", local=None, quad=None,
                 radius=None, neighbourhood="chebyshev", sampler="sde", precision="f64", visible_impl="auto",
                 brownian_tol=lm.BROWNIAN_TOL, rtol=lm.RTOL_SDE, atol=lm.ATOL_SDE, dt0=lm.DT0,
                 max_steps=MAX_STEPS_INTERVAL):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, not {precision!r}")
        if visible_impl not in VISIBLE_IMPLS:
            raise ValueError(f"visible_impl must be one of {VISIBLE_IMPLS}, not {visible_impl!r}")
        if hidden not in ("ising", "potts"):
            raise ValueError(f"hidden must be 'ising' or 'potts', not {hidden!r}")
        theta_hist = np.asarray(theta_hist, dtype=np.float64)
        n_theta = theta_hist.shape[1]
        K = int(W_hist.shape[1])
        grid, r0 = lm.problem_geometry(problem)
        radius = r0 if radius is None else int(radius)
        n = grid * grid
        conn = lm.connectivity(grid, radius, neighbourhood)
        P = conn.shape[0]
        if 4 * n + 3 * P != n_theta:
            raise ValueError(f"parameter count {n_theta} does not match problem {problem} (radius {radius})")
        if theta_hist.shape[0] != N_KNOTS or W_hist.shape[0] != N_KNOTS:
            raise ValueError(f"expected {N_KNOTS} knots")

        self.problem, self.hidden, self.K, self.n = problem, hidden, K, n
        self.quad, self.local = quad is not None, local is not None
        self.sampler = sampler
        self.scale = 2.0 if sampler == "sde" else 1.0
        self.precision = precision
        self.brownian_tol, self.rtol, self.atol, self.dt0 = float(brownian_tol), float(rtol), float(atol), float(dt0)
        self.max_steps = int(max_steps)
        self.order, self.xp, self.s = knot_times()

        # visible energy: only the pair groups (and k_6) that are nonzero somewhere in the history; the others are
        # exactly zero (--pair-terms, the quartic projection), so leaving them out is exact
        terms = tuple(t for t in sorted(lm.PAIR_TERMS) if np.any(theta_hist[:, lm.PAIR_TERMS[t](n, P)] != 0.0))
        sextic = bool(np.any(theta_hist[:, lm.k6_slice(n)] != 0.0))
        n_off = lattice_offsets(conn, grid)
        impl = visible_impl
        if impl == "auto":
            impl = "stencil" if n_off <= STENCIL_MAX_OFFSETS else "dense"
        self.vis = (lm.StencilVisible(conn, n, terms=terms, sextic=sextic) if impl == "stencil"
                    else lm.DenseVisible(conn, n, terms=terms, sextic=sextic))
        self.visible = {"impl": impl, "pair_terms": list(terms), "sextic": sextic, "offsets": n_off, "pairs": int(P)}

        # knot histories on the device, in the run's own slice order; interval_args picks the two knots. They are
        # passed to the jitted functions as arguments: a jitted function that closes over a device array embeds it
        # in the compiled program as a constant (gigabytes at K = 3072)
        hist = {"th": theta_hist, "th_lin": np.asarray(lm.theta_lin(n_theta, n))}
        if K:
            hist["W"], hist["b"] = W_hist, b_hist
            if self.quad:
                hist["Q"] = quad
        if self.local:
            hist["wl"], hist["bl"] = local[0], local[1]
        self._hist = {k: jnp.asarray(np.asarray(v, dtype=np.float64)) for k, v in hist.items()}
        self._sig = jnp.full((n,), jnp.sqrt(2.0 * lm.TEMP))       # sample_latent: jnp.sqrt(2.0 * lm.TEMP) * eye(n)

        self._args_jit = jax.jit(self._interval_args)
        self._solve_jit = jax.jit(jax.vmap(self._solve_one, in_axes=(0, 0, 0, None, None, None)))
        self._drift_batch = jax.jit(jax.vmap(self.drift, in_axes=(0, 0, None)))

    # ------------------------------------------------------------------------------------------------ settings --
    def settings(self):
        """The sampler settings, for the samples manifest."""
        return {"impl": "fast", "precision": self.precision, "visible": self.visible, "brownian_tol": self.brownian_tol,
                "rtol": self.rtol, "atol": self.atol, "dt0": self.dt0, "solver": "SRA1",
                "levy_area": "SpaceTimeLevyArea", "diffusion": "diagonal (lineax.DiagonalLinearOperator)",
                "interp": "linear (jnp.interp convention), hidden fields and forces mixed from the two knots",
                "step_ts": "knot intervals solved in sequence, PIDController state carried over",
                "max_steps_interval": self.max_steps, "progress_meter": "none"}

    # ---------------------------------------------------------------------------------------------- the drift --
    def interval_args(self, j):
        """Device arrays of reverse-time interval j (the two knots it lies between), the `args` of the drift."""
        lo = interval_lo(j)
        return self._args_jit(self._hist, jnp.int32(self.order[lo]), jnp.int32(self.order[lo + 1]),
                              jnp.asarray(self.xp[lo]), jnp.asarray(self.xp[lo + 1]))

    def _interval_args(self, h, k_lo, k_hi, xp_lo, xp_hi):
        f64 = jnp.float64
        a = {"xp_lo": xp_lo.astype(f64), "dxp": (xp_hi - xp_lo).astype(f64)}       # jnp.interp: delta / dx
        th_lo, th_hi = h["th"][k_lo], h["th"][k_hi]
        a["th_lo"] = self.scale * th_lo - h["th_lin"]                                  # s th(t) - th_lin, at l = 0
        a["dth"] = self.scale * (th_hi - th_lo)
        if self.K:
            W_lo, dW = h["W"][k_lo], h["W"][k_hi] - h["W"][k_lo]
            cols = [jnp.concatenate([W_lo.T, dW.T], axis=1)]                           # (N, 2K)
            if self.quad:
                Q_lo, dQ = h["Q"][k_lo], h["Q"][k_hi] - h["Q"][k_lo]
                cols.append(jnp.concatenate([Q_lo.T, dQ.T], axis=1))
            M = jnp.concatenate(cols, axis=0)                                          # (2N or N, 2K)
            a["M"] = M.astype(jnp.float32) if self.precision == "mixed" else M
            a["b_lo"], a["db"] = h["b"][k_lo], h["b"][k_hi] - h["b"][k_lo]
        if self.local:
            a["wl_lo"], a["dwl"] = h["wl"][k_lo], h["wl"][k_hi] - h["wl"][k_lo]
            a["bl_lo"], a["dbl"] = h["bl"][k_lo], h["bl"][k_hi] - h["bl"][k_lo]
        return a

    def _dot(self, v, M):
        """v @ M, in float32 at HIGHEST precision for --precision mixed (M is float32 then), else float64."""
        if self.precision == "mixed":
            return jnp.dot(v.astype(jnp.float32), M, precision=jax.lax.Precision.HIGHEST).astype(jnp.float64)
        return jnp.dot(v, M)

    def drift(self, t, x, a):
        """Drift at reverse time t for one state x (N,), inside the interval whose interval_args are `a` (t may be
        either end of it). x - 2 grad F for the SDE, x - grad F for the PF-ODE, as sample_latent.make_drift."""
        n, K = self.n, self.K
        tau = lm.T_FORWARD - t
        l = (tau - a["xp_lo"]) / a["dxp"]
        X = x[None]
        d = -(self.vis.grad_lap(a["th_lo"], X)[0][0] + l * self.vis.grad_lap(a["dth"], X)[0][0])
        if K > 0:
            xx = jnp.concatenate([x, x * x]) if self.quad else x
            A = self._dot(xx, a["M"])                                    # [A_lo, dA], (2K,)
            z = ((A[:K] + a["b_lo"]) + l * (A[K:] + a["db"])) / lm.TEMP
            m = jnp.tanh(z) if self.hidden == "ising" else jax.nn.softmax(z)
            G = self._dot(jnp.concatenate([m, l * m]), a["M"].T)          # [m W(t), m Q(t)] or m W(t)
            d = d + self.scale * ((G[:n] + 2.0 * x * G[n:]) if self.quad else G)
        if self.local:
            wl = a["wl_lo"] + l * a["dwl"]
            bl = a["bl_lo"] + l * a["dbl"]
            d = d + self.scale * jnp.tanh((wl * x + bl) / lm.TEMP) * wl
        return d

    def drift_batch(self, t, X, j):
        """Drift (B, N) at times t (B,) for states X (B, N), all in reverse-time interval j (for checks)."""
        return self._drift_batch(jnp.asarray(t, dtype=jnp.float64), jnp.asarray(X), self.interval_args(j))

    def interval_of(self, t):
        """The reverse-time interval containing t (a knot belongs to the interval it starts; t = 5 to the last)."""
        return int(np.clip(np.searchsorted(self.s, t, side="right") - 1, 0, N_INTERVALS - 1))

    # -------------------------------------------------------------------------------------------- the solver --
    def _diffusion(self, t, y, args):
        return lineax.DiagonalLinearOperator(self._sig)

    def _solve_one(self, y, key, cstate, t0, t1, args):
        """One trajectory over one knot interval [t0, t1] (vmapped over the chunk). The Brownian tree is the same
        object solve_SDE builds (over [0, T_FORWARD], same key and tolerance), so every interval reads the same path."""
        bm = diffrax.VirtualBrownianTree(0.0, lm.T_FORWARD, self.brownian_tol, (self.n,), key,
                                         diffrax.SpaceTimeLevyArea)
        terms = diffrax.MultiTerm(diffrax.ODETerm(self.drift), diffrax.ControlTerm(self._diffusion, bm))
        sol = diffrax.diffeqsolve(
            terms, diffrax.SRA1(), t0=t0, t1=t1, dt0=cstate[2], y0=y, args=args,
            saveat=diffrax.SaveAt(t1=True, controller_state=True),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            controller_state=cstate, max_steps=self.max_steps)
        return sol.ys[-1], sol.controller_state, sol.stats["num_steps"], sol.stats["num_accepted_steps"]

    def initial_controller_state(self, B):
        """PIDController.init's state for dt0 (no step_ts / jump_ts clipping at t = 0): (made_jump, at_dtmin, dt,
        inverse scaled error, previous one)."""
        return (jnp.zeros(B, dtype=bool), jnp.zeros(B, dtype=bool), jnp.full(B, self.dt0, dtype=jnp.float64),
                jnp.ones(B, dtype=jnp.float64), jnp.ones(B, dtype=jnp.float64))

    def solve_chunk(self, x0, keys, intervals=None):
        """Integrates the reverse SDE from t = 0 to 5 for the states x0 (B, N) with Brownian keys (B, 2).
        Returns (x at t = 5 (B, N) numpy, steps (B, 99), accepted steps (B, 99)) per knot interval.
        intervals = m < 99: only the first m intervals (to time a part; x is then the state at s[m])."""
        if self.sampler != "sde":
            raise ValueError("FastSampler.solve_chunk integrates the reverse SDE only")
        y = jnp.asarray(x0, dtype=jnp.float64)
        keys = jnp.asarray(keys)
        cstate = self.initial_controller_state(y.shape[0])
        steps, acc = [], []
        for j in range(N_INTERVALS if intervals is None else intervals):
            y, cstate, ns, na = self._solve_jit(y, keys, cstate, jnp.asarray(self.s[j]), jnp.asarray(self.s[j + 1]),
                                                self.interval_args(j))
            steps.append(ns)
            acc.append(na)
        y = np.asarray(y)
        return y, np.stack([np.asarray(s) for s in steps], 1), np.stack([np.asarray(s) for s in acc], 1)


def step_summary(steps, accepted, chunk):
    """Solver statistics of a sampling job: total steps per trajectory, rejected share, and the loop iterations a
    chunk needs (per interval the most steps any trajectory in the chunk takes, summed over intervals) relative to
    the mean total steps (the cost of synchronising the trajectories at the knots)."""
    tot = steps.sum(1)
    iters = [int(steps[c:c + chunk].max(0).sum()) for c in range(0, steps.shape[0], chunk)]
    return {"steps_mean": float(tot.mean()), "steps_median": float(np.median(tot)), "steps_max": int(tot.max()),
            "rejected_share": float(1.0 - accepted.sum() / max(steps.sum(), 1)),
            "loop_iterations_per_chunk": iters,
            "loop_iterations_over_mean_steps": float(np.mean(iters) / tot.mean()) if tot.mean() > 0 else None}


def sample(args, fs, problem, stem, prov, W_hist, t_start):
    """The --fast counterpart of sample_latent.sample: the same trajectory set, chunks (the last one padded with
    its last trajectory) and files (<stem>.npy and .json; <stem>.partial.npy while running), plus <stem>_steps.npz
    (solver steps per trajectory and knot interval). args: run, set, n, chunk, sampler (must be 'sde')."""
    grid, _ = lm.problem_geometry(problem)
    x0, keys = lm.trajectory_set(args.set, args.n, problem, grid * grid)
    if x0.shape[0] < args.n:
        raise SystemExit(f"set {args.set} has only {x0.shape[0]} trajectories for {problem}; --n {args.n} is too large")
    x_final = np.full_like(x0, np.nan)
    steps = np.zeros((args.n, N_INTERVALS), dtype=np.int64)
    accepted = np.zeros_like(steps)
    chunk_times = []
    for c0 in range(0, args.n, args.chunk):
        t_c = time.time()
        idx = np.arange(c0, min(c0 + args.chunk, args.n))
        pad = np.concatenate([idx, np.repeat(idx[-1:], args.chunk - idx.size)])
        xf, ns, na = fs.solve_chunk(x0[pad], keys[pad])
        x_final[idx] = xf[: idx.size]
        steps[idx], accepted[idx] = ns[: idx.size], na[: idx.size]
        np.save(stem + ".partial.npy", x_final[: idx[-1] + 1])
        chunk_times.append(time.time() - t_c)
        print(f"chunk {c0}-{idx[-1]} done in {chunk_times[-1]:.1f} s; steps per trajectory mean "
              f"{ns[: idx.size].sum(1).mean():.0f}, loop iterations {int(ns.max(0).sum())}", flush=True)
    np.save(stem + ".npy", x_final)
    np.savez(stem + "_steps.npz", steps=steps, accepted=accepted, s=fs.s)
    os.remove(stem + ".partial.npy")
    manifest = {
        "run": args.run, "problem": problem, "K": int(W_hist.shape[1]), "sampler": args.sampler, "set": args.set,
        "n": args.n, "chunk": args.chunk, **fs.settings(), **step_summary(steps, accepted, args.chunk),
        "chunk_seconds": chunk_times,
        "max_abs_final": float(np.nanmax(np.abs(x_final))), "n_nonfinite": int(np.sum(~np.isfinite(x_final).all(1))),
        "slurm_job": os.environ.get("SLURM_JOB_ID", "manual"), "jax": jax.__version__, "diffrax": diffrax.__version__,
        **prov,
        "sha256_sample_latent": lm.sha256_file(os.path.join(HERE, "sample_latent.py")),
        "sha256_sample_fast": lm.sha256_file(os.path.join(HERE, "sample_fast.py")),
    }
    with open(stem + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved {stem}.npy; total {time.time() - t_start:.0f} s", flush=True)
    return x_final
