"""Lightning adapter for ELG-POMO using MARS datasets and experiment plumbing."""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from ..baseline_cvrp import rollout
from .policy import ELGPolicy


class ELG(pl.LightningModule):
    def __init__(
        self,
        env=None,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int | None = None,
        local_size: int = 40,
        local_dim: int = 32,
        local_heads: int = 4,
        pomo_size: int = 50,
        mode: str = "joint",
        warmup_epochs: int = 0,
        scale_norm: bool = True,
        distance_penalty: bool = True,
        logit_clipping: float = 50.0,
        xi: float = -1.0,
        optimizer_kwargs: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["env"])
        self.env = env
        self.policy = ELGPolicy(
            embed_dim, num_layers, num_heads, feedforward_dim, local_size,
            local_dim, local_heads, mode, distance_penalty, logit_clipping, xi,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), weight_decay=1e-6,
            **(self.hparams.optimizer_kwargs or {"lr": 1e-4}),
        )

    def on_train_epoch_start(self):
        if self.hparams.mode == "joint":
            self.policy.local_enabled = self.current_epoch >= self.hparams.warmup_epochs

    def training_step(self, batch, batch_idx):
        out = rollout(self.policy, batch, self.hparams.pomo_size, "sampling")
        reward, log_prob = out["reward"], out["log_prob"]
        advantage = reward - reward.mean(1, keepdim=True)
        objective = -(advantage.detach() * log_prob)
        if self.hparams.scale_norm:
            objective = objective / advantage.detach().abs().amax(1, keepdim=True).clamp_min(1e-6)
        loss = objective.mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=reward.size(0))
        self.log("train/reward", reward.max(1).values.mean(), on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=reward.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        out = rollout(self.policy, batch, self.hparams.pomo_size, "greedy", False)
        reward = out["reward"].max(1).values
        self.log("val/reward", reward.mean(), on_epoch=True, prog_bar=True, batch_size=reward.size(0))
        return reward

    def test_step(self, batch, batch_idx):
        out = rollout(self.policy, batch, self.hparams.pomo_size, "greedy", False)
        reward = out["reward"].max(1).values
        self.log("test/reward", reward.mean(), batch_size=reward.size(0))
        return out
