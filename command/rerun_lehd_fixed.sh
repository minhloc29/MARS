#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

exec /media/data3/users/longnd/anaconda3/envs/nco/bin/python train.py \
  --backbone lehd \
  --num_loc 100 \
  --logger wandb \
  --device 0 \
  --seed 42 \
  --batch_size 256 \
  --embed_dim 64 \
  --lehd_data_path "data/lehd/data/CVRP/training dataset/vrp100_hgs_train_100w.txt" \
  --lehd_val_data_path "data/lehd/data/CVRP/testing dataset/vrp100_test_lkh.txt" \
  --output output_lehd_fixed
