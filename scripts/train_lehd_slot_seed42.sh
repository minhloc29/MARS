#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
local_train="${repo_dir}/data/lehd/data/CVRP/training dataset/vrp100_hgs_train_100w.txt"
sibling_train="${repo_dir}/../MARS/data/lehd/data/CVRP/training dataset/vrp100_hgs_train_100w.txt"
local_val="${repo_dir}/data/lehd/validation/vrp100_hgs_val_100001_101000.txt"
sibling_val="${repo_dir}/../MARS/data/lehd/validation/vrp100_hgs_val_100001_101000.txt"
if [[ -f "${local_train}" ]]; then
    default_train="${local_train}"
else
    default_train="${sibling_train}"
fi
train_data="${LEHD_TRAIN_DATA:-${default_train}}"
output_dir="${LEHD_OUTPUT:-${repo_dir}/output_lehd_slot_optim_fixed}"
logger="${LEHD_LOGGER:-csv}"
seed="${LEHD_SEED:-42}"
epochs="${LEHD_EPOCHS:-40}"
batch_size="${LEHD_BATCH_SIZE:-256}"
learning_rate="${LEHD_LR:-1e-4}"
n_train="${LEHD_N_TRAIN:-100000}"
n_val="${LEHD_N_VAL:-1000}"
val_args=()

if [[ ! -f "${train_data}" ]]; then
    echo "LEHD training data not found: ${train_data}" >&2
    echo "Set LEHD_TRAIN_DATA to the vrp100_hgs_train_100w.txt path." >&2
    exit 1
fi

val_data="${LEHD_VAL_DATA:-}"
if [[ -z "${val_data}" ]]; then
    if [[ -f "${local_val}" ]]; then
        val_data="${local_val}"
    elif [[ -f "${sibling_val}" ]]; then
        val_data="${sibling_val}"
    fi
fi
if [[ -n "${val_data}" ]]; then
    if [[ ! -f "${val_data}" ]]; then
        echo "LEHD validation data not found: ${val_data}" >&2
        exit 1
    fi
    val_args=(--lehd_val_data_path "${val_data}")
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
    --epochs "${epochs}" \
    --batch_size "${batch_size}" \
    --lr "${learning_rate}" \
    --n_train "${n_train}" \
    --n_val "${n_val}" \
    --device 0 \
    --seed "${seed}" \
    --output "${output_dir}" \
    --logger "${logger}" \
    --symmetrize_target \
    --normalize_target
