# Fig. 1(c): a real denoising trajectory from the paper MNIST model

**Goal.** Replace the schematic digit sequence in Fig. 1(c) of the resubmission with frames of one real reverse-SDE trajectory of the paper model.

**Which trajectory.** Fig. 4(b) shows the first 20 of the 40 SDE samples from the run of 2026-07-21:
`final_plots/samples_non_smoothed_sde_memory_efficient_..._False_clipped_-0.4096_to_2.8727.png`.
Grid position k (row-major, starting at 1) is trajectory k−1.
- trajectory 0: the first "0" in Fig. 4(b) (top-left)
- trajectory 4: the cleaner "0" at position 5 of the top row

**Why a new run is needed.** The paper sampler integrated all 40 trajectories but kept only the final states.

**What the replay does** (`replay_reverse_sde_trajectories.py`):
- It uses the same checkpoint (`out/2026_07_08/.../params_history.npy`, SHA-256 `37cfe46e…7249`, checked before use).
- It uses the same RNG splits (master key 1) and the same solver settings: SRA1, rtol 1e-4, atol 1e-6, dt0 1e-8, Brownian tolerance 1e-12, linear interpolation of θ.
- It does not pass `step_ts`, matching the paper sampler.
- The only change is that it saves the state at 501 reverse times, s = 0, 0.01, …, 5.
  - Saving more points does not change the solver steps.
  - Saved states between accepted steps are diffrax's local linear interpolation.
- It compares the replayed final states with the saved paper samples. The GPU's scatter-add order is not deterministic, so expect small differences, not a bit-for-bit match.

**Outputs.** Written to `out/<job id>/`:
- `trajectories_all40_float32.npy`, with shape (40, 501, 784) in normalised units, and `reverse_times_s.npy`
- `trajectory_00_float64.npy` and `trajectory_04_float64.npy`
- `traj{00,04}_dense_every0.25_{fixed,autoscale}.png` and `traj{00,04}_five_frames_{...}.png`, plus single frames `frame_traj*_s*.png`
- `overview_fig4b_trajectories_0-19_{fixed,autoscale}.png`, for picking a different trajectory
- `summary.json`, holding the final-state check, solver steps and versions

**The two grey scales.**
- `fixed`: every frame uses the same grey scale, the data range [−0.41, 2.87].
- `autoscale`: each frame is clipped to that range and then stretched to its own min–max. This is how Fig. 4(b) was drawn.
- In the `fixed` version, early frames are nearly uniform grey, because the noise std is √0.02 ≈ 0.14.

**Submit.** `sbatch run_replay_neuronic.sh`. Logs go to `logs/`.

## Results

### 2026-11-16, Slurm job 4263277: replay done (NVIDIA L40, 1 min 41 s)

Job 4263210 failed first because the job file set `JAX_PLATFORMS=cuda`. diffrax's progress meter needs a CPU device, so the setting is now `cuda,cpu`.

**The replay reproduces the paper samples.**
- Initial states equal the saved ones exactly (max difference 0).
- Final states differ from the paper run of 2026-07-21 by at most 8.2e-4 over all 40 trajectories and 784 pixels, against a pixel range of about 3.3. For trajectory 0 the difference is 3.4e-5, for trajectory 4 it is 1.3e-4.
- Against the reproduction check of 2026-08-22 the maximum difference is 1.8e-4.
- The differences come from GPU summation order. The frames are therefore the paper's samples.
- The solver took 4,337–5,037 steps per trajectory (median 4,595).

**When the digit appears.** Read from the 0.25-step strips:
- trajectory 0: a faint ring at s ≈ 2.5, clearly a "0" by s ≈ 3.5, sharpening up to s = 5;
- trajectory 4: a ring visible at s ≈ 2.0–2.25, clear by s ≈ 3.0;
- s ≈ 0–1.75 is pure noise in both.

**Suggested Fig. 1(c) panels** (Noise / – / Emerging structure / – / Sample):
- trajectory 0: s = 0, 2.5, 3.25, 4.0, 5.0
- trajectory 4: s = 0, 2.25, 3.0, 4.0, 5.0

The single frames are `out/4263277/frame_traj{00,04}_s{…}_{fixed,autoscale}.png`. Use `autoscale` to match Fig. 4(b), and state the per-frame scaling in the caption.
