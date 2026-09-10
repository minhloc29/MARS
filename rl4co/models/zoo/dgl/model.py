"""Lightning self-improvement adapter for DGL on cached MARS instances."""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from ..baseline_cvrp import CVRPState, rollout
from .policy import DGLPolicy


def _nearest_neighbor_labels(batch):
    """Build the feasible greedy pool used before DGL has learned improvements."""
    state = CVRPState.from_batch(batch)
    b = state.batch_size
    state.actions.append(torch.zeros(b, 1, dtype=torch.long, device=state.xy.device))
    while not bool(state.finished.all()):
        mask = state.mask()
        distance = state.distance.gather(
            1, state.current[:, 0, None, None].expand(b, 1, state.xy.size(1))
        ).squeeze(1)
        selected = distance.masked_fill(mask[:, 0], float("inf")).argmin(-1, keepdim=True)
        state.step(selected)
    return torch.stack(state.actions, -1)[:, 0], state.reward()[:, 0]


class DGL(pl.LightningModule):
    def __init__(
        self,
        env=None,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        feedforward_dim: int | None = None,
        knn: int = 100,
        depot_knn: int = 100,
        pomo_size: int = 16,
        improve_every: int = 1,
        logit_clipping: float = 10.0,
        optimizer_kwargs: dict | None = None,
    ):
        super().__init__()
        if improve_every < 1:
            raise ValueError("dgl_improve_every must be positive")
        self.save_hyperparameters(ignore=["env"])
        self.env = env
        self.policy = DGLPolicy(
            embed_dim, num_layers, num_heads, feedforward_dim, knn, depot_knn,
            logit_clipping,
        )
        self.solution_pool: dict[int, torch.Tensor] = {}
        self.solution_rewards: dict[int, float] = {}
        self.dataset_signature = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), weight_decay=1e-6,
            **(self.hparams.optimizer_kwargs or {"lr": 1e-4}),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97),
        }

    def _labels_for_batch(self, batch):
        ids = batch["instance_id"].tolist()
        missing_rows = [i for i, key in enumerate(ids) if key not in self.solution_pool]
        if missing_rows:
            subset = {k: v[missing_rows] for k, v in batch.items() if k != "instance_id"}
            actions, rewards = _nearest_neighbor_labels(subset)
            for row, action, reward in zip(missing_rows, actions, rewards):
                self.solution_pool[ids[row]] = action.detach().cpu().to(torch.int16)
                self.solution_rewards[ids[row]] = float(reward)
        labels = [self.solution_pool[key].to(self.device).long() for key in ids]
        max_len = max(map(len, labels))
        padded = torch.full((len(labels), max_len), -1, dtype=torch.long, device=self.device)
        for row, actions in enumerate(labels):
            padded[row, : len(actions)] = actions
        return padded

    def _imitation_loss(self, batch, labels):
        state = CVRPState.from_batch(batch)
        b = state.batch_size
        state.actions.append(torch.zeros(b, 1, dtype=torch.long, device=self.device))
        self.policy.prepare(state)
        terms = []
        # labels include the initial depot at column zero.
        for step in range(1, labels.size(1)):
            active = labels[:, step] >= 0
            if not bool(active.any()):
                break
            logits = self.policy.logits(state).masked_fill(state.mask(), float("-inf"))
            selected = torch.where(active, labels[:, step], torch.zeros_like(labels[:, step]))[:, None]
            logp = torch.log_softmax(logits[:, 0], -1).gather(-1, selected)
            terms.append(-logp[active].mean())
            state.step(selected)
        return torch.stack(terms).mean()

    @torch.no_grad()
    def _improve_pool(self, batch):
        result = rollout(self.policy, batch, self.hparams.pomo_size, "greedy", False)
        best_reward, best_idx = result["reward"].max(1)
        ids = batch["instance_id"].tolist()
        for row, key in enumerate(ids):
            reward = float(best_reward[row])
            if reward > self.solution_rewards.get(key, float("-inf")):
                actions = result["actions"][row, best_idx[row]].detach().cpu()
                # Remove repeated depot padding accumulated after this rollout finished.
                while len(actions) > 2 and actions[-1] == 0 and actions[-2] == 0:
                    actions = actions[:-1]
                self.solution_pool[key] = actions.to(torch.int16)
                self.solution_rewards[key] = reward

    def training_step(self, batch, batch_idx):
        labels = self._labels_for_batch(batch)
        loss = self._imitation_loss(batch, labels)
        if self.current_epoch % self.hparams.improve_every == 0:
            self._improve_pool(batch)
        rewards = torch.tensor(
            [self.solution_rewards[key] for key in batch["instance_id"].tolist()], device=self.device
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=len(rewards))
        self.log("train/reward", rewards.mean(), on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=len(rewards))
        return loss

    def validation_step(self, batch, batch_idx):
        result = rollout(self.policy, batch, self.hparams.pomo_size, "greedy", False)
        reward = result["reward"].max(1).values
        self.log("val/reward", reward.mean(), on_epoch=True, prog_bar=True, batch_size=len(reward))
        return reward

    def on_save_checkpoint(self, checkpoint):
        checkpoint["dgl_solution_pool"] = self.solution_pool
        checkpoint["dgl_solution_rewards"] = self.solution_rewards
        checkpoint["dgl_dataset_signature"] = self.dataset_signature

    def on_load_checkpoint(self, checkpoint):
        saved_signature = checkpoint.get("dgl_dataset_signature")
        if self.dataset_signature is not None and saved_signature not in (None, self.dataset_signature):
            raise RuntimeError("DGL checkpoint solution pool belongs to a different training dataset")
        self.solution_pool = checkpoint.get("dgl_solution_pool", {})
        self.solution_rewards = checkpoint.get("dgl_solution_rewards", {})

    def test_step(self, batch, batch_idx):
        result = rollout(self.policy, batch, self.hparams.pomo_size, "greedy", False)
        reward = result["reward"].max(1).values
        self.log("test/reward", reward.mean(), batch_size=len(reward))
        return result
