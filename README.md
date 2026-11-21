# LEGO: Metric-Aware Slot Abstraction for Routing

A fork of the [RL4CO](https://github.com/ai4co/rl4co) benchmark that extends POMO / AM
with **metric-aware slot attention** for combinatorial routing (CVRP and its open /
time-windowed variants). A set of *slot/region* embeddings is learned over the customer
graph, and the preserved inter-node **intrinsic metric** (a sparse insertion-cost
geometry) is injected back into the base encoder. The result is a constructive policy
that keeps the network's learned geometry aligned with the problem's true metric.

This repository also integrates several published baselines (SIL, DGL, ELG, RADAR, INViT,
LEHD, ICAM, L2R) so that the slot method and its competitors are trained and evaluated on
the **same cached instance splits**.


## Contents

- [Model](#model) — the slot-attention method and its ablation variants.
- [Installation](#installation) — from source, using `uv`.
- [Data](#data) — cached splits with sparse insertion costs (`d_ins`).
- [Training](#training) — run `train.py` on any integrated backbone.
- [Evaluation](#evaluation) — decode a checkpoint with `test.py`.
- [Baseline guides](#baseline-guides) — method-specific notes and fairness caveats.
- [Layout](#layout) — where things live.


## Model

The core contribution lives in `rl4co/models/zoo/pomo_slot/`:

- **`POMOSlot`** — `POMO` extended with metric-aware slot abstraction. Runs the full
  POMO policy-gradient loop, replacing the encoder with a `SlotInjectingEncoder` and
  stacking auxiliary losses (slot-entropy regularisation, metric preservation).
- **`AMSlot`** — the same slot machinery on the single-start Attention Model.
- **`SlotInjectingEncoder`** — runs a base `AttentionModelEncoder`, applies SlotAttention
  over the customer embeddings, and additively injects the resulting slot context back
  into the hidden states (the depot embedding is left unchanged). Exposes the slot
  embeddings and soft assignment matrix (`last_slots`, `last_A_ik`) as a side-channel for
  the auxiliary losses.
- **`SlotAttention`** (`rl4co/models/nn/slot_attention.py`) — the slot-attention operator.

`POMOSlot` implements four ablation variants selected by `--metric_variant`:

| Variant | Objective |
| --- | --- |
| `none` / `B` | Plain POMO — no metric-preservation loss. |
| `A` | Reconstruction loss: slot centroids reconstructed back onto node coordinates. |
| `C` | Metric preservation on the learned slot geometry. |
| `D` *(default)* | Metric preservation on the sparse insertion-cost geometry `d_ins`. |

Pass `--disable_slots` to run the base backbone as a true no-slot baseline.


## Installation

Install from source with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone <this-repo> && cd LEGO
uv sync --all-extras
source .venv/bin/activate
```

The project is a fork of RL4CO and reuses its namespace (`rl4co`); the extras
(`dev`, `graph`, `routing`, `docs`) are declared in `pyproject.toml`.


## Data

The slot method is trained on cached instance splits that store, alongside the routing
data, a **sparse insertion-cost target** `d_ins` (`d_ins_idx` / `d_ins_val`) used by the
Variant-D metric-preservation loss. Splits are generated once and shared by every
backbone so that comparisons train on identical instances.

Generate the shared splits for a problem / size / distribution:

```bash
python -m rl4co.data.generate_slot_dataset \
  --num_locs 100 --dist uniform --n_train 100000 --n_val 1000 --n_test 1000 \
  --out_dir ./data/slot_datasets_v2 --method insertion --problem cvrp --seed 42
```

`--method` selects the insertion-cost definition (`savings` | `construction` |
`insertion`). The training script validates that the cached data's method matches
`--ins_method` so you never train on the wrong cost.

> N=1000 preparation is computationally expensive and creates a multi-gigabyte cache.
> It is opt-in: `train.py` only generates missing splits when passed
> `--generate_missing_data`, and errors otherwise with an exact command to run.


## Training

Train the slot method (POMO start, Variant D) on CVRP with N=100 customers:

```bash
python train.py --backbone pomo --variant D --num_loc 100 --dist uniform \
  --num_slots 8 --ins_method insertion --device 0 --seed 42
```

Choose the Attention-model start with `--backbone am`, or a no-slot baseline with
`--backbone pomo --disable_slots`. Run any integrated baseline with its backbone key:
`sil`, `invit`, `dgl`, `elg`, `radar`, `icam`, `l2r`, `lehd`, `ttpl`. OVRP and VRPTW
reuse the CVRP embedding / objective and are selected with `--problem ovrp|vrptw`.

Key flags (see `python train.py --help` for the full list):

| Option | Default | Meaning |
| --- | --- | --- |
| `--backbone` | `pomo` | Model to train. |
| `--variant` / `--metric_variant` | `D` | Slot-metric ablation variant. |
| `--num_slots` | `8` | Number `K` of slot/region embeddings. |
| `--slot_iters` | `3` | SlotAttention refinement iterations. |
| `--alpha_metric` | `0.1` | Weight of the metric-preservation loss. |
| `--beta_entropy` | `0.01` | Weight of the slot-entropy regulariser. |
| `--ins_method` | `construction` | Insertion-cost definition (must match data). |
| `--num_loc` / `--dist` | `100` / `uniform` | Instance size and distribution. |
| `--problem` | `cvrp` | `cvrp` \| `ovrp` \| `vrptw`. |
| `--logger` | `csv` | `csv` (default) or `wandb`. |
| `--device` | `0` | GPU id(s); pass several for DDP. |

If the two required cached split files are missing, `train.py` prints the exact
data-generation command to run (or accepts `--generate_missing_data` to build them inline
and then train).


## Evaluation

Decode a checkpoint on a cached test split:

```bash
python test.py --model pomo --ckpt /path/to/pomo.ckpt \
  --num_loc 100 --data_path data/slot_datasets_v2/construction/cvrp100_uniform_test.pt \
  --n_inst 500 --batch_size 64 --out output/pomo_test.json
```

`test.py` also supports dataset-free evaluation on freshly seeded instances when
`--data_path` is omitted.

> Decoding budgets differ between methods (e.g. POMO uses multi-start rollouts with
> augmentation; SIL uses one greedy start). Report these budgets alongside results.


## Baseline guides

- [SIL](rl4co/models/zoo/sil/README.md)
- [DGL](rl4co/models/zoo/dgl/README.md)
- [ELG](rl4co/models/zoo/elg/README.md)
- [INViT](rl4co/models/zoo/invit/README.md)

These document the integration, the shared train/eval setting, and the fairness caveats
when comparing each method against the slot approach.


## Layout

```
train.py                 # unified training entry point for every backbone
test.py                  # checkpoint evaluation / decoding
rl4co/data/              # dataset generation, slot dataset + loaders, insertion costs
rl4co/envs/routing/      # environments (cvrp, ovrp, cvrptw, atsp, mtsp, ...)
rl4co/models/zoo/        # model zoo, incl. pomo_slot/ (the slot method)
rl4co/models/nn/         # shared layers: slot_attention, metric_loss, attention, ...
rl4co/models/rl/         # RL training loops: REINFORCE, PPO, A2C
scripts/                 # data generation, evaluation, plotting helpers
```


## Citation

If you use this work, cite the slot-attention method (LEGO/MARS slot NCO) — add the
preprint reference here once available — alongside the RL4CO benchmark it builds on:

```bibtex
@inproceedings{berto2025rl4co,
    title={{RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark}},
    author={Federico Berto and Chuanbo Hua and Junyoung Park and Laurin Luttmann and Yining Ma and Fanchen Bu and Jiarui Wang and Haoran Ye and Minsu Kim and Sanghyeok Choi and Nayeli Gast Zepeda and Andr\'e Hottung and Jianan Zhou and Jieyi Bi and Yu Hu and Fei Liu and Hyeonah Kim and Jiwoo Son and Haeyeon Kim and Davide Angioni and Wouter Kool and Zhiguang Cao and Jie Zhang and Kijung Shin and Sungsoo Ahn and Guojie Song and Changhyun Kwon and Lin Xie and Jinkyoo Park},
    booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    year={2025}
}
```
