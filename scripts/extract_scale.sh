#!/bin/bash
set -eo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runs_dir=${1:-"${repo_dir}/runs/selected_cells_seed0_2_epoch50000"}
output_dir=${2:-"${repo_dir}/flow-artifacts/raw-main-50k"}
flow_workers=${FLOW_WORKERS:-12}

mapfile -t run_dirs < <(find "${runs_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#run_dirs[@]} -ne 27 ]]; then
  echo "expected exactly 27 main-study run directories, found ${#run_dirs[@]}" >&2
  exit 1
fi

for run_dir in "${run_dirs[@]}"; do
  "${repo_dir}/.venv/bin/grokking-lab" extract-flows \
    --run "${run_dir}" \
    --output "${output_dir}" \
    --device cpu \
    --workers "${flow_workers}" \
    --resume \
    --compression-level 6 \
    --acknowledge-phase-gate
done
