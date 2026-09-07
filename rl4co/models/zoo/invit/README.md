# INViT baseline

This adapter ports the official **INViT: A Generalizable Routing Problem Solver
with Invariant Nested View Transformer** CVRP policy into the MARS training
entry point.

For a controlled comparison, it reads the same cached MARS train and validation
splits and uses MARS's epoch, batch-size, learning-rate, seed, logging, early
stopping, and Lightning checkpoint settings. Demands are already normalized by
the shared dataset, so the model uses capacity 1. INViT does not use MARS slots,
the insertion matrix, `variant`, `num_slots`, or `beta_entropy`; `ins_method`
selects the shared dataset folder only.

The model keeps INViT's nested state views (35, 50, 65), 15-action view,
stochastic REINFORCE student, greedy rollout baseline, and validation-driven
baseline replacement. The upstream `torch_cluster.knn` call is implemented with
batched PyTorch top-k selection to avoid an extra binary dependency.

For N=1000 training, sampled routes are replayed in 16-step gradient chunks.
This retains the same REINFORCE objective and one optimizer update per batch
without retaining all N=1000 decoder activations at once. Adjust this with
`--invit_backprop_chunk_size` if needed.

```bash
python train.py --backbone invit --variant D --num_loc 1000 \
  --logger wandb --num_slots 64 --ins_method insertion \
  --device 0 --seed 42 --batch_size 64 --embed_dim 64 \
  --beta_entropy 0.01
```

Resume by appending `--resume "path/to/checkpoint.ckpt"`.
