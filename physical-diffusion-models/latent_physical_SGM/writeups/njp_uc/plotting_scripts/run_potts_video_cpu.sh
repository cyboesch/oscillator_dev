#!/usr/bin/env bash
# Submit with: sbatch --chdir=/path/to/workspace --output=/path/to/video-job.log \
#   plotting_scripts/run_potts_video_cpu.sh /path/to/workspace
# Uses a separate CPU allocation and writes only the manuscript video outputs.
#SBATCH --account=seas
#SBATCH --job-name=codex_potts_video
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --exclude=neu306

set -euo pipefail
video_workspace_dir="${1:?Pass the absolute oscillator_dev workspace directory}"
video_paper_rel="physical-diffusion-models/latent_physical_SGM/writeups/njp_uc"
video_paper_dir="${video_workspace_dir}/${video_paper_rel}"
video_data_dir="${video_paper_dir}/figure_data/potts_video"
video_output_dir="${video_paper_dir}/videos"
video_filename="${video_output_dir}/potts_state_probabilities.mp4"
video_partial="${video_output_dir}/potts_state_probabilities.partial.mp4"
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
if [[ -e "${video_filename}" ]]; then
    echo "Refusing to overwrite existing video: ${video_filename}" >&2
    exit 1
fi
mkdir -p "${video_data_dir}" "${video_output_dir}"
export APPTAINERENV_JAX_PLATFORMS=cpu
export APPTAINERENV_JAX_ENABLE_X64=true
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_OMP_NUM_THREADS=2
export APPTAINERENV_OPENBLAS_NUM_THREADS=2
export APPTAINERENV_XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
video_container_args=(--bind "${video_workspace_dir}:/workspace" --pwd /workspace
                      "${video_workspace_dir}/container/container_image.sif")

srun --cpu-bind=cores apptainer exec "${video_container_args[@]}" \
    python3 -u "/workspace/${video_paper_rel}/plotting_scripts/sample_potts_video.py" \
    --index 0 --output-dir "/workspace/${video_paper_rel}/figure_data/potts_video"

# RGBA frames travel directly to ffmpeg; no PNG frame files are saved.
srun --cpu-bind=cores apptainer exec "${video_container_args[@]}" \
    python3 -u "/workspace/${video_paper_rel}/plotting_scripts/render_potts_video.py" \
    --data "/workspace/${video_paper_rel}/figure_data/potts_video/trajectory_000.npz" \
    --raw-video --fps 20 \
    --summary "/workspace/${video_paper_rel}/figures/potts_video_summary.pdf" \
    | ffmpeg -hide_banner -loglevel warning -y \
        -f rawvideo -pixel_format rgba -video_size 1280x800 -framerate 20 -i pipe:0 \
        -an -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -threads 1 \
        -movflags +faststart "${video_partial}"

ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,avg_frame_rate \
    -show_entries format=duration,size -of json "${video_partial}" \
    > "${video_data_dir}/video_info.json"
mv -- "${video_partial}" "${video_filename}"
printf 'Video saved: %s\n' "${video_filename}"
