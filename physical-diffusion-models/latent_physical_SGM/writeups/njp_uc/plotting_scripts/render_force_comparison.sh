#!/usr/bin/env bash
set -euo pipefail
force_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
force_project_dir="$(cd -- "${force_script_dir}/../../../.." && pwd)"
force_workspace_dir="$(dirname -- "${force_project_dir}")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_OMP_NUM_THREADS=1
exec apptainer exec \
    --bind "${force_workspace_dir}:${force_workspace_dir}" \
    --pwd "${force_workspace_dir}" \
    "${force_workspace_dir}/container/container_image.sif" \
    python3 "${force_script_dir}/make_force_comparison.py" "$@"
