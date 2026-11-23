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
        problem: str = "cvrp",
        tw_travel_scale: float = 1.0,
        micro_batch_size: int | None = None,
        optimizer_kwargs: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["env"])
        self.env = env
        # Gradient accumulation inside one optimizer step. The POMO advantage is
        # computed within each instance and the loss is a batch mean, so summing
        # chunk losses weighted by chunk/batch gives the exact full-batch gradient
        # while only one chunk's rollout graph is alive at a time.
        if micro_batch_size is not None and micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        self.automatic_optimization = micro_batch_size is None
        self.policy = ELGPolicy(
            embed_dim, num_layers, num_heads, feedforward_dim, local_size,
            local_dim, local_heads, mode, distance_penalty, logit_clipping, xi,
            use_tw=problem == "vrptw",
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), weight_decay=1e-6,
            **(self.hparams.optimizer_kwargs or {"lr": 1e-4}),
        )

    def on_train_epoch_start(self):
        if self.hparams.mode == "joint":
            self.policy.local_enabled = self.current_epoch >= self.hparams.warmup_epochs

    def _chunk_objective(self, batch):
        problem = self.hparams.get("problem", "cvrp")
        out = rollout(self.policy, batch, self.hparams.pomo_size, "sampling", problem=problem,
                      tw_travel_scale=self.hparams.tw_travel_scale)
        reward, log_prob = out["reward"], out["log_prob"]
        advantage = reward - reward.mean(1, keepdim=True)
        objective = -(advantage.detach() * log_prob)
        if self.hparams.scale_norm:
            objective = objective / advantage.detach().abs().amax(1, keepdim=True).clamp_min(1e-6)
        return objective.mean(), reward

    def training_step(self, batch, batch_idx):
        micro = self.hparams.get("micro_batch_size")
        if micro is None:
            loss, reward = self._chunk_objective(batch)
        else:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            total = batch["locs"].size(0)
            loss_sum, rewards = 0.0, []
            for start in range(0, total, micro):
                chunk = {k: v[start:start + micro] for k, v in batch.items()
                         if torch.is_tensor(v) and v.size(0) == total}
                chunk_loss, chunk_reward = self._chunk_objective(chunk)
                weight = chunk_reward.size(0) / total
                self.manual_backward(chunk_loss * weight)
                loss_sum += chunk_loss.detach() * weight
                rewards.append(chunk_reward.detach())
            optimizer.step()
            loss, reward = loss_sum, torch.cat(rewards)
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=reward.size(0))
        self.log("train/reward", reward.max(1).values.mean(), on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=reward.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        problem = self.hparams.get("problem", "cvrp")
        out = rollout(self.policy, batch, self.hparams.pomo_size, "greedy", False, problem=problem,
                      tw_travel_scale=self.hparams.tw_travel_scale)
        reward = out["reward"].max(1).values
        self.log("val/reward", reward.mean(), on_epoch=True, prog_bar=True, batch_size=reward.size(0))
        if "tw_late" in out:
            # Deadline violations / deadlock fallbacks across all POMO threads;
            # both must stay 0 for a valid VRPTW result.
            self.log("val/tw_late", float(out["tw_late"]), on_epoch=True,
                     reduce_fx="sum", batch_size=reward.size(0))
            self.log("val/tw_fallbacks", float(out["tw_fallbacks"]), on_epoch=True,
                     reduce_fx="sum", batch_size=reward.size(0))
        return reward

    def test_step(self, batch, batch_idx):
        problem = self.hparams.get("problem", "cvrp")
        out = rollout(self.policy, batch, self.hparams.pomo_size, "greedy", False, problem=problem,
                      tw_travel_scale=self.hparams.tw_travel_scale)
        reward = out["reward"].max(1).values
        self.log("test/reward", reward.mean(), batch_size=reward.size(0))
        return out
