#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env_name="${CONDA_ENV_NAME:-mars}"
python_version="${PYTHON_VERSION:-3.12}"

# Driver 595.91.07 is backward-compatible with these CUDA 12.4 wheels. The
# server does not need a separately installed CUDA toolkit for PyTorch wheels.
pytorch_index_url="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
torch_spec="${TORCH_SPEC:-torch==2.6.0}"
torchrl_spec="${TORCHRL_SPEC:-torchrl==0.7.2}"
tensordict_spec="${TENSORDICT_SPEC:-tensordict==0.7.2}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$conda_env_name"; then
    echo "Using existing Conda environment: $conda_env_name"
else
    conda create --name "$conda_env_name" "python=$python_version" pip --yes
fi

conda run --name "$conda_env_name" python -m pip install --upgrade pip setuptools wheel
conda run --name "$conda_env_name" python -m pip install \
    "$torch_spec" --index-url "$pytorch_index_url"
conda run --name "$conda_env_name" python -m pip install \
    "$torchrl_spec" "$tensordict_spec"
conda run --name "$conda_env_name" python -m pip install \
    --requirement "$repo_dir/requirements-server.txt"

# Keep the explicitly selected PyTorch stack instead of resolving the upstream
# pyproject CUDA/PyTorch entries again.
conda run --name "$conda_env_name" python -m pip install \
    --editable "$repo_dir" --no-deps
conda run --name "$conda_env_name" python \
    "$repo_dir/scripts/verify_server_install.py"

echo
echo "MARS environment is ready. Activate it with:"
echo "  conda activate $conda_env_name"
echo
echo "Datasets are not installed by this script. Copy or generate MARS/data separately."
echo "For W&B logging, run: wandb login"
