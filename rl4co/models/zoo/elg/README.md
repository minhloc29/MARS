# ELG baseline

This adapter retains ELG-POMO's global policy, local-neighbor attention,
distance penalty, POMO rollouts, and within-instance REINFORCE baseline. It reads
the same cached train/validation files as MARS. `variant`, slot, entropy, and
insertion-cost flags are accepted only for command compatibility.

```bash
python train.py --backbone elg --variant D --num_loc 1000 \
  --logger wandb --num_slots 64 --ins_method insertion --device 0 \
  --seed 42 --batch_size 64 --embed_dim 64 --beta_entropy 0.01 \
  --elg_embed_dim 88 --elg_pomo_size 50 --elg_mode joint
```

The matched default (`elg_embed_dim=88`, six layers) has **606,088 trainable
parameters**, versus **610,561** for the shown MARS configuration (-0.73%). It
keeps ELG's upstream depth while reducing its upstream width from 128. The shared
`--embed_dim 64` remains accepted for comparison-command compatibility;
`--elg_embed_dim` controls ELG itself.

The default width is 50, following the corrected comparison noted by the DGL
authors. Upstream ELG delays adding its local policy until step 200,000; use
`--elg_warmup_epochs` to reproduce a corresponding staged run. Zero trains the
full ensemble for the whole shared epoch budget. For N=1000, global attention is
expensive; `--elg_mode only_local` is scalable but is a different ELG ablation.
