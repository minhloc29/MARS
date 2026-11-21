#!/usr/bin/env bash
# Reference-tour teacher ablation (nn_split / nn_2opt vs nn).
#
# Offline orchestration: generate each teacher's cached d_cons splits once, then
# train each teacher with the SAME budget as the default LEGO run (seed 0). No
# solver or tour construction runs at training time — d_cons is baked into the
# cache below.
#
# Adjust DATA_DIR / LOGGER / DEVICE for your machine. Requires the GPU resources
# for training; run each train.py line separately if you manage devices manually.
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data/slot_datasets_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
SEED=0
N=100
DIST=uniform
METHOD=construction

echo "== Generating d_cons caches for each teacher (offline, once) =="

# nn_split (hard) + nn_split (soft)
python -m rl4co.data.generate_slot_dataset \
  --num_locs "$N" --dist "$DIST" --method "$METHOD" --teacher nn_split \
  --n_train 100000 --n_val 1000 --n_test 1000 --out_dir "$DATA_DIR" --seed "$SEED"
python -m rl4co.data.generate_slot_dataset \
  --num_locs "$N" --dist "$DIST" --method "$METHOD" --teacher nn_split \
  --cross_route_mode soft --n_train 100000 --n_val 1000 --n_test 1000 \
  --out_dir "$DATA_DIR" --seed "$SEED"

# nn_2opt + nn_2opt matched-scale control
python -m rl4co.data.generate_slot_dataset \
  --num_locs "$N" --dist "$DIST" --method "$METHOD" --teacher nn_2opt \
  --n_train 100000 --n_val 1000 --n_test 1000 --out_dir "$DATA_DIR" --seed "$SEED"
python -m rl4co.data.generate_slot_dataset \
  --num_locs "$N" --dist "$DIST" --method "$METHOD" --teacher nn_2opt --match_scale \
  --n_train 100000 --n_val 1000 --n_test 1000 --out_dir "$DATA_DIR" --seed "$SEED"

# (nn is the default; only needed here if no legacy construction cache exists)
python -m rl4co.data.generate_slot_dataset \
  --num_locs "$N" --dist "$DIST" --method "$METHOD" --teacher nn \
  --n_train 100000 --n_val 1000 --n_test 1000 --out_dir "$DATA_DIR" --seed "$SEED"

echo
echo "== Training each teacher with seed $SEED (same budget as LEGO) =="
echo "(run each line individually to control GPU / devices)"

train () {
  local teacher=$1; shift
  python train.py --backbone pomo --variant D --teacher "$teacher" \
    --num_loc "$N" --dist "$DIST" --ins_method "$METHOD" \
    --num_slots 8 --seed "$SEED" --data_dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" --logger csv "$@"
}

train nn
train nn_split
train nn_split --cross_route_mode soft
train nn_2opt
train nn_2opt --match_scale

echo
echo "== Done. Results are saved to $OUTPUT_DIR/ablation_*_N${N}.json and "
echo "   evaluated via test.py (see README 'Evaluation'). Treat the default "
echo "   nn run as the current LEGO run for the Table-4 comparison. =="
