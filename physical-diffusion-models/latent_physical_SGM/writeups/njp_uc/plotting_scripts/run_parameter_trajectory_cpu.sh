#!/usr/bin/env bash
# Replay the visually selected digit-2 trajectory used in the parameter figure.
# Submit: sbatch --chdir=/path/to/oscillator_dev --output=/path/to/sampling.log \
#   plotting_scripts/run_parameter_trajectory_cpu.sh /path/to/oscillator_dev
#SBATCH --account=seas
#SBATCH --job-name=parameter_potts2
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --exclude=neu306
set -euo pipefail
figure_workspace_dir="${1:?Pass the absolute oscillator_dev workspace directory}"
figure_paper_rel="physical-diffusion-models/latent_physical_SGM/writeups/njp_uc"
figure_run_name="mnist28_d0123456789_K1536_ema_potts_quad_local_quartic_r1_lin_pooled"
export APPTAINERENV_JAX_PLATFORMS=cpu
export APPTAINERENV_JAX_ENABLE_X64=true
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_OMP_NUM_THREADS=2
export APPTAINERENV_OPENBLAS_NUM_THREADS=2
export APPTAINERENV_XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores apptainer exec \
    --bind "${figure_workspace_dir}:/workspace" --pwd /workspace \
    "${figure_workspace_dir}/container/container_image.sif" \
    python3 -u "/workspace/${figure_paper_rel}/plotting_scripts/sample_potts_video.py" \
    --run "/workspace/physical-diffusion-models/out/latent_physical_SGM/${figure_run_name}" \
    --base-size 2000 --index 21 --save-per-interval 5 \
    --output-dir "/workspace/${figure_paper_rel}/figure_data/parameter_potts2"
