# DGL baseline

The adapter keeps DGL's dynamic union of current-node/depot neighborhoods and
self-improvement learning. A persistent per-instance solution pool is initialized
with feasible nearest-neighbor routes, improved by multi-start policy rollouts,
and optimized by teacher-forced imitation. The pool and its dataset signature are
saved in Lightning checkpoints, so `--resume` is safe.
Routes are stored as compact `int16` tensors (valid for the supported N <= 1000)
to keep the full shared-data solution pool manageable.

```bash
python train.py --backbone dgl --variant D --num_loc 1000 \
  --logger wandb --num_slots 64 --ins_method insertion --device 0 \
  --seed 42 --batch_size 64 --embed_dim 64 --beta_entropy 0.01 \
  --dgl_embed_dim 128 --dgl_knn 100 --dgl_depot_knn 100 \
  --dgl_pomo_size 16
```

The matched default (`dgl_embed_dim=128`, three layers) has **612,737 trainable
parameters**, versus **610,561** for the shown MARS configuration (+0.36%). The
128-dimensional setting is also DGL's upstream embedding width. The shared
`--embed_dim 64` remains accepted so comparison command templates need not
change; `--dgl_embed_dim` controls DGL itself.

Slot, metric, entropy, and insertion-cost values do not alter DGL; `ins_method`
only selects the same cached MARS dataset folder. The integrated loop visits each
cached training instance once per epoch, unlike upstream DGL's fixed 64-instance
pool with 100 repeated updates, making epoch/data budgets directly comparable.
