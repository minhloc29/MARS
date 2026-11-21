</br>
<div align="center">

# MeTRA Slot NCO

**Metric-Aware Slot Abstraction for Neural Combinatorial Optimization on the CVRP**

A research fork of [RL4CO](https://github.com/ai4co/rl4co) that augments classic
constructive routing backbones (POMO, AM) with **learned slot/region abstraction**,
coupled to an explicit **metric-preservation objective**. The core idea is to force the
latent geometry of slot embeddings to mirror a routing-aware target — most importantly
the **insertion-cost / route-conditioned dissimilarity** between regions — while training
the backbone with its usual RL signal (REINFORCE / POMO multi-start).

This README documents the method and how to **reproduce every result**: generate the
shared cached datasets, train each variant/backbone under a fair unified setting, and
evaluate on identical test sets.

</div>

---

## 1. Method at a glance

The pipeline adds three learnable pieces on top of a standard attention encoder:

1. **Slot attention** (`rl4co/models/nn/slot_attention.py`) — iteratively clusters the
   `N` node embeddings into `K` slot embeddings `z_k ∈ R^d` with a soft assignment matrix
   `A_ik ∈ R^{N×K}` (slots as rows = regions, columns = slots).
2. **Slot injection** (`rl4co/models/zoo/pomo_slot/policy.py`) — the slot context is
   added back into each customer embedding before decoding:
   `hidden[i] += Σ_k A_ik z_k`. Casting a slot-aware decoder: on every step the decoder
   sees *where* each customer sits in region space.
3. **Metric-preservation loss** (`rl4co/models/nn/metric_loss.py`) — a
   METRA-style lower-bound objective with dual ascent:
   ```
   L_metric = -E[‖φ(z_k) − φ(z_l)‖]
              + λ · E[ max(0, ‖φ(z_k) − φ(z_l)‖ − D_target(k,l))² ]
   ```
   `φ` is a shared projection head (`ProjectionHead`), `log λ` is a Lagrange multiplier
   trained by dual ascent in its own optimizer group, and `D_target(k,l) ∈ R^{K×K}` is a
   **slot-pairwise target dissimilarity** computed from the assignments and the data.

A **slot-entropy regulariser** keeps the soft assignment `A_ik` from collapsing to a
trivial clustering. The total auxiliary objective is added to the policy loss.

### The target `D_target` and ablation variants

The heart of the method is *what* `D_target` should be. That choice is exposed as
`--metric_variant` (a.k.a. `--variant`):

| Variant | Target `D_target(k,l)` | Supervision | Code path |
|---------|------------------------|-------------|-----------|
| **A** | Slot-centroid **reconstruction** of node coordinates (`Σ_ik A_ik c_k` → MSE vs `x_i`) | spatial, unsupervised | `model.py` recon branch |
| **B** | — (no metric/target). Plain **slot-only** baseline: slots + injection, RL only | none | `model.py` `metric_variant in ("none","B")` |
| **C** | **Centroid distance** `‖c_k − c_l‖₂` of slot coordinate centroids | spatial | `_euclidean_target` |
| **D** ⭐ | **Insertion / route-conditioned cost** aggregated as `Aᵀ d_ins A` (from cached sparse `d_ins`) | **routing-aware** | `_aggregate_d_ins_sparse` |
| **E** | **LEGO-Euclidean**: bilinear spatial distance `Aᵀ d_euc A` with `d_euc = ‖x_i−x_j‖₂` | spatial control for D | `_euclidean_aggregate` |

**Variant D (recommended)** is the routing-aware control: `d_ins(i,j)` is the cost of
inserting customer `j` into a partial tour/route built from `i`, aggregated over slot
assignments. Three target *algorithms* for `d_ins` are supported (`--ins_method`):
`savings` (Clarke–Wright savings), `construction` (sequential construction), and
`insertion` / **RCIC** (Route-Conditioned Insertion Cost — the recommended target).
These are precomputed and cached with each dataset (see §4).

> **E vs D.** Variants D and E use the *same* bilinear `Aᵀ d A` aggregation, `φ`, dual
> `λ`, spread and entropy terms — they differ **only** in the underlying node-level
> distance (`d_ins`, routing-aware, vs `d_euc`, plain spatial). E is therefore the clean
> control for “is the routing-aware target better than spatial proximity?”.

### Geometry modes (`--geom_mode`)

| Mode | Meaning |
|------|---------|
| `full` (default) | `L_geom = L_spread + λ·hinge` (spread term + violation penalty, dual ascent) |
| `spread_only` | `L_geom = L_spread` only — no hinge, no dual variable (unbounded; watch `metric_D_latent_mean`) |
| `shuffled` | Same as `full` but the cumulative `p_i` ordering is randomly permuted before `d_cons` — a permutation control |

`--disable_slots` strips the entire slot module and trains the stock backbone
(an equivalent param-count POMO/AM baseline). `--geometry_aware_no_slots` keeps the
Euclidean geometry objective on the *plain* encoder embeddings without any slots.

---

## 2. Repository layout

```
train.py                       # unified trainer: every backbone/variant under one CLI
test.py                        # evaluate a checkpoint on an NPZ test set (or CVRPLIB Set X)
scripts/
  gen_data.py                  # generate & cache a deterministic CVRP eval NPZ set
  eval_metric.py               # metric-geometry eval (Variant D): Spearman correlation
  check_geometry_learning.py   # diagnostics for learned slot geometry
  visualize_geometry_preservation.py / plot_geometry_figures.py / plot_metrics.py
rl4co/
  data/
    generate_slot_dataset.py   # build Slot NCO cached splits + sparse d_ins targets
    slot_dataset.py            # SlotDataset / dataloaders wrapping the cached .pt files
    insertion_cost.py          # sparse insertion/construction/savings cost computation
  models/
    nn/slot_attention.py       # iterative Slot Attention
    nn/metric_loss.py          # Euclidean / MetricPreservation / SlotEntropy losses
    zoo/pomo_slot/             # POMOSlot, AMSlot + SlotInjectingEncoder
  envs/routing/                # CVRP (and TSP, etc.) environments (RL4CO)
data/slot_datasets_v2/<method>/  # generated cached splits (train/val/test .pt)
output/<run>/                    # checkpoints + CSV logs per run
```

---

## 3. Environment setup

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# sanity check
python -c "import torch, lightning, tensordict; print('torch', torch.__version__)"
```

---

## 4. Generate the shared slot datasets

> ⚠️ **Each `--ins_method` target gets its own dataset.** A Variant-D checkpoint trained
> with `--ins_method insertion` must be trained *and* evaluated with `insertion`. Keep
> split sizes and seed identical across methods and backbones so comparisons are fair.

```bash
python -m rl4co.data.generate_slot_dataset \
  --num_locs 100 --dist uniform \
  --n_train 100000 --n_val 1000 --n_test 2000 \
  --method insertion --out_dir ./data/slot_datasets_v2

# optionally also generate the construction / savings targets
python -m rl4co.data.generate_slot_dataset \
  --num_locs 100 --dist uniform \
  --n_train 100000 --n_val 1000 --n_test 2000 \
  --method construction --out_dir ./data/slot_datasets_v2
python -m rl4co.data.generate_slot_dataset \
  --num_locs 100 --dist uniform \
  --n_train 100000 --n_val 1000 --n_test 2000 \
  --method savings --out_dir ./data/slot_datasets_v2
```

Outputs land in `data/slot_datasets_v2/<method>/{cvrpN_{dist}_{train,val,test}.pt}` in the
`sparse_v2` format (cached `d_ins_idx`/`d_ins_val`). Other flags:

```text
--dist       uniform | clustered | both
--k_neighbors  number of sparse neighbors (default 15)
--seed         reproducibility; splits use seed + split_index
--chunk_size   instances per generation chunk (default: adaptive to N)
```

`train.py --generate_missing_data` will call this generator automatically if a required
split is missing.

---

## 5. Train — unified setting

All commands use the **same fair setting**: `N=100` uniform, `seed=42` (train),
100k/1k/2k splits, `embed_dim=64`, `batch_size=256`, 100 epochs, one GPU `--device`.
Set `--logger csv` if W&B is not configured.

### Ours — Variant D slot model (recommended)

```bash
python train.py --backbone pomo --num_loc 100 --dist uniform \
  --data_dir ./data/slot_datasets_v2 --ins_method insertion \
  --metric_variant D --num_slots 8 --embed_dim 64 --proj_dim 32 \
  --batch_size 256 --n_train 100000 --n_val 1000 --epochs 100 \
  --seed 42 --device 0 --logger csv --output ./output
```

### Ours — across target algorithms

```bash
for METHOD in insertion construction savings; do
  python train.py --backbone pomo --num_loc 100 --dist uniform \
    --data_dir ./data/slot_datasets_v2 --ins_method "$METHOD" \
    --metric_variant D --num_slots 8 --embed_dim 64 --proj_dim 32 \
    --slot_iters 3 --alpha_metric 0.1 --beta_entropy 0.01 \
    --batch_size 256 --n_train 100000 --n_val 1000 --epochs 100 \
    --seed 42 --device 0 --logger csv --output ./output
done
```

### Slot-enabled backbones (AM, and full-method baselines)

```bash
# Attention Model with slots
python train.py --backbone am --num_loc 100 --dist uniform \
  --data_dir ./data/slot_datasets_v2 --ins_method insertion \
  --batch_size 256 --n_train 100000 --n_val 1000 --epochs 100 \
  --seed 42 --device 0 --logger csv --output ./output

# SIL (repair loop) — same unified data setting
python train.py --backbone sil --num_loc 100 --dist uniform \
  --data_dir ./data/slot_datasets_v2 --ins_method insertion \
  --batch_size 256 --n_train 100000 --n_val 1000 --epochs 100 \
  --seed 42 --device 0 --logger csv --output ./output

# INViT (tuned to keep the 500–600k param target)
python train.py --backbone invit --num_loc 100 --dist uniform \
  --data_dir ./data/slot_datasets_v2 --ins_method insertion \
  --embed_dim 64 --invit_num_heads 4 --invit_state_sizes 15 25 35 \
  --invit_action_size 8 --invit_backprop_chunk_size 16 \
  --batch_size 256 --n_train 100000 --n_val 1000 --epochs 100 \
  --seed 42 --device 0 --logger csv --output ./output
```

### No-slot and non-slot baselines

```bash
# POMO baseline: no slot module, no metric target
python train.py --backbone pomo --disable_slots --num_loc 100 --dist uniform \
  --embed_dim 64 --batch_size 256 --n_train 100000 --n_val 1000 --epochs 100 \
  --seed 42 --device 0 --logger csv --output ./output

# Other backbones in the same unified setting
for MODEL in am l2r icam; do
  python train.py --backbone "$MODEL" --num_loc 100 --dist uniform \
    --embed_dim 64 --batch_size 256 --n_train 100000 --n_val 1000 --epochs 100 \
    --seed 42 --device 0 --logger csv --output ./output
done
```

> **LEHD / TTPL** use their own data-loader format (`--lehd_data_path`,
> `--lehd_val_data_path`) and are **not** directly compatible with the slot `.pt` files.

### Key flags

```text
--backbone        pomo | am | l2r | icam | sil | invit | dgl | elg | radar | lehd | ttpl
--metric_variant  none | A | B | C | D | E      (default D)
--num_slots K     number of regions (default 8)
--slot_iters      slot-attention refinement iters (default 3)
--proj_dim        project φ(z_k) dim (default 64)
--alpha_metric    metric/recon loss weight     --beta_entropy   slot entropy weight
--ins_method      savings | construction | insertion   (must match the cached dataset)
--geom_mode       full | spread_only | shuffled
--lambda_init / --lr_dual / --lambda_max     dual-ascent λ schedule
--normalize_target / --symmetrize_target     target preprocessing (default on)
--disable_slots   train the stock backbone (no slots / aux)
--device 0 1      single GPU, or multiple for DDP
--logger          csv | wandb        --resume  <ckpt>
```

---

## 6. Evaluate every checkpoint on the same test set

### 6.1 Build one shared, deterministic NPZ test set

```bash
python scripts/gen_data.py --num_loc 100 --n_inst 2000 --seed 1234 \
  --loc_dist uniform --out ./data/eval_cvrp100_uniform.npz
```

Reuse this *exact* file for every checkpoint so model comparisons use identical
instances. `gen_data.py` also supports `--loc_dist gaussian / cluster / gaussian_mixture /
mixed / exponential / poisson`, distribution parameters, `--capacity`, and `--alpha`
(spatial stress shift around the depot).

### 6.2 Run the evaluation

```bash
CKPT=./output/<run>/checkpoints/<best-checkpoint>.ckpt

python test.py --model pomo --ckpt "$CKPT" --num_loc 100 --n_inst 2000 \
  --data_path ./data/eval_cvrp100_uniform.npz --batch_size 256 \
  --seed 1234 --device mps --out ./results/pomo_eval.json
```

Use the matching `--model` value (`pomo | am | l2r | icam | sil | invit | …`). Add
`--augment` only if it is part of the comparison — **do not compare augmented and
non-augmented runs as the same decoding budget**.

### 6.3 Generalization test (different size)

```bash
python scripts/gen_data.py --num_loc 500 --n_inst 2000 --seed 1234 \
  --loc_dist uniform --out ./data/eval_cvrp500_uniform.npz

python test.py --model pomo --ckpt "$CKPT" --num_loc 500 --n_inst 2000 \
  --data_path ./data/eval_cvrp500_uniform.npz --batch_size 256 \
  --seed 1234 --device mps --out ./results/pomo_N500_eval.json
```

`test.py` also evaluates on **CVRPLIB Set X** (`--dataset cvrplib_setX --data_dir
./data/cvrplib_setX`) for standard benchmark reporting.

---

## 7. Metric-geometry evaluation (Variant D)

Checks whether learned slot geometry follows the insertion-cost target (Spearman
correlation + one-sided metric violation). **Only** for slot-enabled Variant-D
POMO/AM checkpoints, with the matching `--method` dataset:

```bash
python scripts/eval_metric.py --model pomo --ckpt "$CKPT" \
  --num_loc 100 --dist uniform --method insertion \
  --data_dir ./data/slot_datasets_v2 --n_inst 2000 --batch_size 256 \
  --seed 1234 --out ./results/pomo_insertion_metric.json \
  --plot ./results/pomo_insertion_metric.png
```

This reports metric-geometry fidelity, **not** routing quality.

---

## 8. Parameter budget (fair-comparison check)

The **500k–600k parameter** target is an experimental constraint. Count parameters from a
saved checkpoint and adjust width/depth (embed_dim, slot count, or the backbone's own
knobs) to stay in range, then retrain from seed 42:

```bash
python -c 'import sys, torch; p=sys.argv[1]; \
  c=torch.load(p,map_location="cpu",weights_only=False); \
  s=c.get("state_dict",c); \
  n=sum(v.numel() for v in s.values() if torch.is_tensor(v)); \
  print(f"{n:,} parameters ({n/1e6:.3f}M)")' "$CKPT"
```

---

## 9. Tests / smoke checks

```bash
python -m py_compile train.py test.py scripts/gen_data.py scripts/eval_metric.py
pytest -q tests/test_policy.py tests/test_metric_loss.py tests/test_slot_attention.py
```

Useful output locations:

* checkpoints: `./output/<run>/checkpoints/`
* evaluation JSON: `./results/`
* CSV / W&B logs: `./output/` or `./logs/`

---

## 10. Credits

This repository is a **research fork of [RL4CO](https://github.com/ai4co/rl4co)**
(Berto *et al.*, KDD 2025), which provides the routing environments, RL algorithms, and
training infrastructure on top of PyTorch / TorchRL / PyTorch Lightning. New contributions
in this fork: iterative **Slot Attention**, **slot context injection**, the
**metric-preservation loss** (Variant D `d_ins` target with dual ascent), the ablation
variants A/B/C/E, the cached sparse-`d_ins` dataset pipeline (`generate_slot_dataset.py`),
and the unified `train.py` / `test.py` / `scripts/eval_metric.py` harness.

If you use or compare against this work, please cite RL4CO:

```bibtex
@inproceedings{berto2025rl4co,
    title={{RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark}},
    author={Federico Berto and Chuanbo Hua and Junyoung Park and Laurin Luttmann and Yining Ma and Fanchen Bu and Jiarui Wang and Haoran Ye and Minsu Kim and Sanghyeok Choi and Nayeli Gast Zepeda and Andr\'e Hottung and Jianan Zhou and Jieyi Bi and Yu Hu and Fei Liu and Hyeonah Kim and Jiwoo Son and Haeyeon Kim and Davide Angioni and Wouter Kool and Zhiguang Cao and Jie Zhang and Kijung Shin and Cathy Wu and Sungsoo Ahn and Guojie Song and Changhyun Kwon and Lin Xie and Jinkyoo Park},
    booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    year={2025},
    url={https://github.com/ai4co/rl4co}
}
```
