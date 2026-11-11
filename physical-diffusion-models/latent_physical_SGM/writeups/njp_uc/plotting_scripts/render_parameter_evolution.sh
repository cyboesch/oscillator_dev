#!/usr/bin/env bash
# Uses the existing project container, one CPU thread, no GPU binding or jobs.
set -euo pipefail
figure_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
figure_project_dir="$(cd -- "${figure_script_dir}/../../../.." && pwd)"
figure_workspace_dir="$(dirname -- "${figure_project_dir}")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_OMP_NUM_THREADS=1
cd "${figure_workspace_dir}"
apptainer exec \
    --bind "${figure_workspace_dir}:${figure_workspace_dir}" \
    --pwd "${figure_workspace_dir}" \
    "${figure_workspace_dir}/container/container_image.sif" \
    python3 "${figure_script_dir}/make_parameter_evolution.py" "$@"

# Rasterize the finished PDF on the host so previews match the submission file.
python3 "${figure_script_dir}/export_parameter_previews.py" "$@"
