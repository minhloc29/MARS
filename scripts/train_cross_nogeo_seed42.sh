#!/usr/bin/env bash
# (a) Cross-attention, geometry OFF (alpha=0, beta=0.01, seed 42)
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data_dir="${DATA_DIR:-${repo_dir}/data/slot_datasets_v2}"
output_dir="${OUTPUT_DIR:-${repo_dir}/output_cross_nogeo_seed42}"
device="${DEVICE:-0}"
logger="${LOGGER:-csv}"

cd "${repo_dir}"
exec "${python_bin}" train.py \
    --backbone pomo \
    --slot_module cross \
    --variant D \
    --teacher nn \
    --ins_method construction \
    --num_loc 100 \
    --dist uniform \
    --n_train 100000 \
    --n_val 1000 \
    --batch_size 64 \
    --epochs 100 \
    --embed_dim 64 \
    --num_slots 8 \
    --lr 1e-4 \
    --alpha_metric 0.0 \
    --beta_entropy 0.01 \
    --seed 42 \
    --device "${device}" \
    --data_dir "${data_dir}" \
    --output "${output_dir}" \
    --logger "${logger}"
