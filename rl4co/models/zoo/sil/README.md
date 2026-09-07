# SIL baseline in MARS

This package adapts the local `SIL/CVRP/Train` implementation of Luo et al.,
*Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing
Problems* (ICLR 2025). `_network.py` contains the original encoder, 16 context
tokens, two-way cross-attention layers, direct/via-depot classifier, detached
customer embeddings, and probability smoothing. It has no runtime dependency
on the sibling `SIL/` directory or its compiled insertion extension.

From `MARS/`, change the original command's backbone to `sil`:

```bash
python train.py --backbone sil --variant D --num_loc 1000 --logger wandb \
  --num_slots 64 --ins_method insertion --device 0 --seed 42 \
  --batch_size 64 --embed_dim 64 --beta_entropy 0.01
```

The cached N=1000 splits must exist first. To generate the exact split sizes
used by the launcher and then train in one invocation, append:

```bash
--generate_missing_data
```

This prepares 50,000 training, 500 validation, and 500 test instances with seed
42. N=1000 insertion-target generation is computationally expensive and creates
a multi-gigabyte cache, so it is opt-in and only runs when files are missing.
The default data path is anchored to the `MARS/` repository, so the training
command works whether it is launched inside `MARS/` or as `python MARS/train.py`.

`variant`, `num_slots`, `beta_entropy`, and other slot/metric options are accepted
for command compatibility but do not change SIL. `ins_method` selects the same
cached data directory; SIL does not consume MARS's auxiliary insertion targets.
The launcher prints this distinction and gives SIL a separate run name.

## Shared experimental setting

Both backbones read these exact files by default:

```text
data/slot_datasets_v2/insertion/cvrp1000_uniform_train.pt
data/slot_datasets_v2/insertion/cvrp1000_uniform_val.pt
```

Coordinates, depots and **already normalized demands** are passed unchanged to
the same `CVRPEnv` with vehicle capacity 1. No SIL data generator or external
labels are used. In particular, SIL's original CVRP1000 capacity of 250 is not
substituted for the cached MARS demands. The existing MARS dataset generator
uses its fallback capacity of 2250 at N=1000; existing files remain authoritative
for both methods. This integration does not change that data convention.

The launcher previously accepted N=1000 without defining training defaults.
Both methods now use 200 epochs, batch size 64, Adam LR 5e-5, at most 50,000
training instances and 500 validation instances at that size. Other sizes keep
their previous defaults. The same flags override epochs, batch size, instance
limits, seed and distribution. Each data loader has an independent seeded
generator, so model initialization and reconstruction randomness do not alter
the initial data shuffle. Both use `val/reward`, patience 20, and gradient norm
clipping at 1.0. Dataset paths, actual split sizes and settings appear in the
results JSON; SIL also records a content hash for safe pseudo-label reuse.

The default `batch` update mode averages the teacher-forced node losses and
performs one optimizer update per batch. MARS and SIL therefore see the same
number of instances and optimizer updates per epoch. Their individual updates
still contain different work: SIL imitates a sampled subpath, while POMO uses
policy gradients over multi-start rollouts. POMO also has its own validation
starts and augmentations; SIL validation uses one greedy rollout. Report these
decoding budgets when comparing results.

## Self-improved training

1. Initialize feasible labels with angle-ordered cheapest insertion (the SIL
   helper's default order and exploration=1). The portable NumPy implementation
   preserves floating point demands instead of casting them to unsigned integers.
2. Sample a random subpath of 4 to `sil_max_subtour_length` customers, ending at a
   route boundary. Rotate/reverse routes, preserve the incoming capacity, fix the
   first customer, and imitate the remaining node/depot decisions. Comparable
   mode averages these decisions into one Adam update per batch. Strict upstream
   mode performs an update after each decoding step.
3. After each block of `sil_improve_every` epochs, reconstruct labels using a
   frozen copy of the best validation policy. Order routes by their centroid's
   angle around the depot, then rotate/reverse them. Default PRC reconstructs disjoint
   route-ending subpaths in parallel. Keep only feasible, shorter complete tours.
   Labels are updated lazily when a training instance is next visited, using the
   same frozen teacher for the entire round.

The portable reconstruction implementation retains centroid route ordering and
route rotations/reversals, but uses randomized route ends instead of upstream's
exact PRC sampling order. This is an adaptation to MARS's training setup,
not a bit-for-bit reproduction of the upstream experiments. MARS's constant
learning rate, clipping and monotonic epoch schedule replace upstream's LR decay,
clip norm 10 and epoch rewind. Student weights/optimizer continue between rounds;
the best validation weights are used as the repair teacher.

| Option | Default | Meaning |
| --- | --- | --- |
| `--sil_repair_budget` | 5 | Reconstruction passes per improvement round; 0 disables improvement |
| `--sil_improve_every` | 20 | Imitation epochs between improvement rounds |
| `--sil_max_subtour_length` | 64 | Maximum imitation/reconstruction length |
| `--sil_num_layers` | 6 | Cross-attention decoder depth |
| `--sil_update_mode` | `batch` | One update per batch; `node` restores upstream behavior |
| `--sil_no_prc` | off | Reconstruct only one subpath per instance per pass |

The fast defaults are intended for a controlled MARS comparison. To reproduce
the original expensive update schedule at N=1000, pass
`--sil_update_mode node --sil_max_subtour_length 1000 --sil_repair_budget 100`.

The first block uses insertion labels. An experiment shorter than 21 epochs at
the default interval will not exercise learned label improvement. For a small
functional check on an existing cached dataset, add:

```bash
--logger csv --epochs 2 --max_instances 4 --batch_size 2 --num_workers 0 \
--sil_improve_every 1 --sil_repair_budget 2 --sil_max_subtour_length 8
```

Checkpoints include model/optimizer state, pseudo-labels keyed by dataset row,
improvement rounds, and repair/best-validation weights. Resume with
`--resume /path/to/checkpoint.ckpt` and the same data and model options. A different
training data hash is rejected. Pseudo-labels use int16 storage and add space
proportional to dataset size times customer count. Evaluation does not need
training labels.

## Evaluation on the same cached test split

```bash
python test.py --model sil --ckpt /path/to/sil.ckpt --num_loc 1000 \
  --data_path data/slot_datasets_v2/insertion/cvrp1000_uniform_test.pt \
  --n_inst 500 --batch_size 64 --out output/sil_test.json
```

Use `--model pomo` and its checkpoint with the same `--data_path` for MARS.
`test.py` retains its existing CPU execution behavior. SIL supports one greedy
start (`--num_starts 1`); POMO's decoding budget should be reported separately.
Without `--data_path`, `test.py` retains its seeded generated-data behavior,
whose demand distribution can differ from the cached training data.
