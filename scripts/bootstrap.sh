#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -x .venv/bin/python ]]; then
  uv venv --system-site-packages --python /venv/main/bin/python .venv
fi
# The Vast PyTorch image already carries a GPU-matched torch/numpy/HF stack.
# --no-deps prevents uv from duplicating several GB of CUDA wheels locally.
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff huggingface-hub
.venv/bin/python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")'
