#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
train_data="${LEHD_TRAIN_DATA:-${repo_dir}/../MARS/data/lehd/data/CVRP/training dataset/vrp100_hgs_train_100w.txt}"
output_dir="${LEHD_OUTPUT:-${repo_dir}/output_lehd_slot_stable}"
logger="${LEHD_LOGGER:-csv}"
seed="${LEHD_SEED:-42}"
val_args=()

if [[ ! -f "${train_data}" ]]; then
    echo "LEHD training data not found: ${train_data}" >&2
    echo "Set LEHD_TRAIN_DATA to the vrp100_hgs_train_100w.txt path." >&2
    exit 1
fi

if [[ -n "${LEHD_VAL_DATA:-}" ]]; then
    if [[ ! -f "${LEHD_VAL_DATA}" ]]; then
        echo "LEHD validation data not found: ${LEHD_VAL_DATA}" >&2
        exit 1
    fi
    val_args=(--lehd_val_data_path "${LEHD_VAL_DATA}")
fi

cd "${repo_dir}"
exec "${python_bin}" train.py \
    --backbone lehd_slot \
    --lehd_data_path "${train_data}" \
    "${val_args[@]}" \
    --num_loc 100 \
    --num_slots 8 \
    --metric_variant D \
    --alpha_metric 0.1 \
    --beta_entropy 0.01 \
    --slot_iters 3 \
    --proj_dim 64 \
    --lambda_init 1.0 \
    --lr_dual 1e-3 \
    --ins_method construction \
    --embed_dim 64 \
    --lehd_decoder_layers 6 \
    --epochs 200 \
    --batch_size 64 \
    --lr 5e-5 \
    --n_train 50000 \
    --n_val 500 \
    --device 0 \
    --seed "${seed}" \
    --output "${output_dir}" \
    --logger "${logger}" \
    --symmetrize_target \
    --normalize_target
