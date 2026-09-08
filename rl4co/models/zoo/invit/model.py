"""Lightning training adapter for INViT on MARS CVRP datasets."""

from __future__ import annotations

import copy

import lightning.pytorch as pl
import torch

from ._network import INViTPolicy


class INViT(pl.LightningModule):
    """Train INViT with its greedy rollout baseline on fixed MARS splits."""

    def __init__(
        self,
        env=None,
        embed_dim: int = 128,
        feedforward_dim: int | None = None,
        num_heads: int = 8,
        state_sizes: tuple[int, ...] = (35, 50, 65),
        action_size: int = 15,
        state_encoder_layers: int = 2,
        action_encoder_layers: int = 2,
        decoder_layers: int = 3,
        baseline_tolerance: float = 1e-3,
        scheduler_gamma: float = 0.99,
        backprop_chunk_size: int = 16,
        optimizer_kwargs: dict | None = None,
    ):
        super().__init__()
        if backprop_chunk_size < 1:
            raise ValueError("INViT backprop_chunk_size must be positive")
        self.save_hyperparameters(ignore=["env"])
        self.automatic_optimization = False
        self.env = env
        self.policy = INViTPolicy(
            embed_dim=embed_dim,
            feedforward_dim=feedforward_dim,
            num_heads=num_heads,
            state_sizes=tuple(state_sizes),
            action_size=action_size,
            state_encoder_layers=state_encoder_layers,
            action_encoder_layers=action_encoder_layers,
            decoder_layers=decoder_layers,
        )
        self.baseline_policy = copy.deepcopy(self.policy).requires_grad_(False)
        self.baseline_policy.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.policy.parameters(), **(self.hparams.optimizer_kwargs or {"lr": 2e-5})
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.scheduler_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.baseline_policy.eval()
        with torch.no_grad():
            baseline = self.baseline_policy(batch, phase="val", decode_type="greedy")
            student = self.policy(batch, phase="train", decode_type="sampling")
        student_cost = -student["reward"]
        baseline_cost = -baseline["reward"]
        advantage = student_cost - baseline_cost

        # Replaying fixed sampled actions produces the same score-function
        # gradient while bounding activations to a small number of decode steps.
        loss = torch.zeros((), device=self.device)
        chunk_loss = torch.zeros((), device=self.device)
        for step, log_probability in enumerate(
            self.policy.replay_log_probabilities(batch, student["actions"]), start=1
        ):
            contribution = (advantage * log_probability).mean()
            loss = loss + contribution.detach()
            chunk_loss = chunk_loss + contribution
            if step % self.hparams.backprop_chunk_size == 0:
                self.manual_backward(chunk_loss)
                chunk_loss = torch.zeros((), device=self.device)
        if chunk_loss.requires_grad:
            self.manual_backward(chunk_loss)
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer.step()

        batch_size = student_cost.size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(
            "train/reward",
            student["reward"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/baseline_reward",
            baseline["reward"].mean(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

    def on_validation_epoch_start(self):
        device = self.device
        self._student_reward_sum = torch.tensor(0.0, device=device)
        self._baseline_reward_sum = torch.tensor(0.0, device=device)
        self._validation_count = 0

    def validation_step(self, batch, batch_idx):
        self.baseline_policy.eval()
        student = self.policy(batch, phase="val", decode_type="greedy")
        baseline = self.baseline_policy(batch, phase="val", decode_type="greedy")
        batch_size = student["reward"].size(0)
        self._student_reward_sum += student["reward"].detach().sum()
        self._baseline_reward_sum += baseline["reward"].detach().sum()
        self._validation_count += batch_size
        self.log(
            "val/reward",
            student["reward"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/baseline_reward",
            baseline["reward"].mean(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return student["reward"]

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self._validation_count:
            return
        student_reward = self._student_reward_sum / self._validation_count
        baseline_reward = self._baseline_reward_sum / self._validation_count
        improved = bool(student_reward > baseline_reward + self.hparams.baseline_tolerance)
        if improved:
            self.baseline_policy.load_state_dict(self.policy.state_dict())
        self.log("val/baseline_updated", float(improved), prog_bar=False)

    def test_step(self, batch, batch_idx):
        output = self.policy(batch, phase="test", decode_type="greedy")
        self.log("test/reward", output["reward"].mean(), batch_size=output["reward"].size(0))
        return output
