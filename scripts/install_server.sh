#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-$repo_dir/.venv}"

# These defaults are a known-compatible stack. They can be overridden without
# editing this script when a server needs another PyTorch wheel/index.
pytorch_index_url="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
torch_spec="${TORCH_SPEC:-torch==2.6.0}"
torchrl_spec="${TORCHRL_SPEC:-torchrl==0.7.2}"
tensordict_spec="${TENSORDICT_SPEC:-tensordict==0.7.2}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 1
fi

"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("MARS requires Python 3.10 or newer")
PY

"$python_bin" -m venv "$venv_dir"
venv_python="$venv_dir/bin/python"

"$venv_python" -m pip install --upgrade pip setuptools wheel
"$venv_python" -m pip install "$torch_spec" --index-url "$pytorch_index_url"
"$venv_python" -m pip install "$torchrl_spec" "$tensordict_spec"
"$venv_python" -m pip install -r "$repo_dir/requirements-server.txt"

# Install this checkout, while keeping the compatible PyTorch stack selected
# above instead of resolving the upstream pyproject CUDA/PyTorch constraints.
"$venv_python" -m pip install --editable "$repo_dir" --no-deps
"$venv_python" "$repo_dir/scripts/verify_server_install.py"

echo
echo "MARS environment is ready. Activate it with:"
echo "  source $venv_dir/bin/activate"
echo
echo "Datasets are not installed by this script. Copy or generate MARS/data separately."
echo "For W&B logging, run: wandb login"
