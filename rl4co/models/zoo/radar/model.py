"""Lightning adapter for RADAR on cached MARS/CVRP instances.

Follows the same structure as the ELG and DGL adapters so that RADAR can
be launched with ``--backbone radar`` from ``train.py``.

Default configuration (embed_dim=64, encoder_layers=6, num_heads=8,
ff_dim=256) gives ~325 k parameters — matching the MARS AttentionModel
policy backbone trained with ``--embed_dim 64``.

Equivalent training command (comparable with MARS):
    python train.py --backbone radar --num_loc 100 \\
        --logger wandb --ins_method insertion \\
        --device 0 --seed 42 --batch_size 256 \\
        --radar_embed_dim 64 --radar_encoder_layers 6 \\
        --radar_num_heads 8 --radar_ff_dim 256 \\
        --radar_pomo_size 100 --generate_missing_data
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from ..baseline_cvrp import rollout
from .policy import RADARPolicy


class RADAR(pl.LightningModule):
    """Lightning module wrapping RADARPolicy for MARS experiments.

    Args:
        env: Unused (kept for API compatibility with other baselines).
        embed_dim: Node embedding dimension.  Default 64.
        encoder_layers: Number of Sinkhorn-MHA encoder layers.  Default 6.
        num_heads: Attention heads (must divide embed_dim).  Default 8.
        ff_dim: Feed-forward hidden dimension.  Default 256.
        ms_hidden_dim: Hidden size of per-head MLP in Mixed-Score MHA.  Default 16.
        svd_rank: SVD-lowrank rank k.  Default 10.
        logit_clipping: Tanh clipping coefficient C.  Default 10.0.
        sinkhorn_iters: Sinkhorn normalisation iterations.  Default 10.
        pomo_size: Number of parallel POMO threads during rollout.
            Set to num_loc for POMO-style multi-start (default 100).
        scale_norm: Optionally normalise REINFORCE advantage by its max abs
            value. Disabled by default to match the original RADAR objective.
        optimizer_kwargs: Passed to ``torch.optim.Adam``.
    """

    def __init__(
        self,
        env=None,
        embed_dim:      int   = 64,
        encoder_layers: int   = 6,
        num_heads:      int   = 8,
        ff_dim:         int   = 256,
        ms_hidden_dim:  int   = 16,
        svd_rank:       int   = 10,
        logit_clipping: float = 10.0,
        sinkhorn_iters: int   = 10,
        pomo_size:      int   = 100,
        scale_norm:     bool  = False,
        optimizer_kwargs: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["env"])
        self.env = env
        self.policy = RADARPolicy(
            embed_dim=embed_dim,
            encoder_layers=encoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            ms_hidden_dim=ms_hidden_dim,
            svd_rank=svd_rank,
            logit_clipping=logit_clipping,
            sinkhorn_iters=sinkhorn_iters,
        )

    # ── Optimiser ────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            weight_decay=1e-6,
            **(self.hparams.optimizer_kwargs or {"lr": 1e-4}),
        )

    # ── Training ─────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        out = rollout(
            self.policy, batch,
            width=self.hparams.pomo_size,
            decode_type="sampling",
            return_log_probs=True,
        )
        reward, log_prob = out["reward"], out["log_prob"]
        # POMO REINFORCE: advantage over in-batch mean
        advantage = reward - reward.mean(1, keepdim=True)
        objective = -(advantage.detach() * log_prob)
        if self.hparams.scale_norm:
            # Normalise by max |advantage| for stable gradients (ELG-style)
            objective = objective / (
                advantage.detach().abs().amax(1, keepdim=True).clamp_min(1e-6)
            )
        loss = objective.mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 batch_size=reward.size(0))
        self.log("train/reward", reward.max(1).values.mean(),
                 on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=reward.size(0))
        return loss

    # ── Validation ───────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        out = rollout(
            self.policy, batch,
            width=self.hparams.pomo_size,
            decode_type="greedy",
            return_log_probs=False,
        )
        reward = out["reward"].max(1).values
        self.log("val/reward", reward.mean(), on_epoch=True, prog_bar=True,
                 batch_size=reward.size(0))
        return reward

    # ── Test ─────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        out = rollout(
            self.policy, batch,
            width=self.hparams.pomo_size,
            decode_type="greedy",
            return_log_probs=False,
        )
        reward = out["reward"].max(1).values
        self.log("test/reward", reward.mean(), batch_size=reward.size(0))
        return out
