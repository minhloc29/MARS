from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ICAMEncoderLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))
        self.norm1 = nn.InstanceNorm1d(dim, affine=True)
        self.norm2 = nn.InstanceNorm1d(dim, affine=True)
        self.ff1 = nn.Linear(dim, hidden_dim)
        self.ff2 = nn.Linear(hidden_dim, dim)

    def forward(self, hidden: torch.Tensor, distance_bias: torch.Tensor) -> torch.Tensor:
        q = self.query(hidden)
        k = self.key(hidden)
        v = self.value(hidden)
        bias = torch.exp(self.alpha * distance_bias)
        weighted = (bias @ (torch.exp(k) * v)) / \
            (bias @ torch.exp(k)).clamp_min(1e-8)
        attended = torch.sigmoid(q) * weighted
        hidden = self.norm1(
            (hidden + attended).transpose(1, 2)).transpose(1, 2)
        feed_forward = self.ff2(F.relu(self.ff1(hidden)))
        return self.norm2((hidden + feed_forward).transpose(1, 2)).transpose(1, 2)


class _ICAMPolicy(nn.Module):
    def __init__(self, embedding_dim: int, encoder_layers: int, feed_forward_dim: int, logit_clipping: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.logit_clipping = logit_clipping
        self.depot_embedding = nn.Linear(2, embedding_dim)
        self.node_embedding = nn.Linear(3, embedding_dim)
        self.encoder = nn.ModuleList(
            [_ICAMEncoderLayer(embedding_dim, feed_forward_dim)
             for _ in range(encoder_layers)]
        )
        self.query_last = nn.Linear(
            embedding_dim + 1, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.alpha_distance = nn.Parameter(torch.ones(1))
        self.alpha_score = nn.Parameter(torch.ones(1))

    def encode(self, depot, nodes, demand, distance, log_scale):
        depot_hidden = self.depot_embedding(depot)
        node_hidden = self.node_embedding(
            torch.cat((nodes, demand[..., None]), dim=-1))
        hidden = torch.cat((depot_hidden, node_hidden), dim=1)
        for layer in self.encoder:
            hidden = layer(hidden, -log_scale * distance)
        return hidden

    def logits(self, hidden, current, load, current_distance, mask, log_scale):
        last = hidden.gather(
            1, current[..., None].expand(-1, -1, self.embedding_dim))
        query = self.query_last(torch.cat((last, load[..., None]), dim=-1))
        key = self.key(hidden)
        value = self.value(hidden)
        attention_bias = -log_scale * self.alpha_distance * current_distance + mask
        weights = torch.exp(attention_bias)
        context = (weights @ (torch.exp(key) * value)) / \
            (weights @ torch.exp(key)).clamp_min(1e-8)
        score = torch.matmul(torch.sigmoid(query) *
                             context, hidden.transpose(1, 2))
        score = score / (self.embedding_dim ** 0.5)
        score = score - log_scale * self.alpha_score * current_distance
        return self.logit_clipping * torch.tanh(score) + mask


class ICAMCVRP(pl.LightningModule):
    """Native ICAM CVRP POMO model for MeTRA's Lightning runner."""

    def __init__(
        self,
        env,
        embed_dim: int = 128,
        num_starts: int | None = None,
        encoder_layers: int = 6,
        feed_forward_dim: int = 512,
        logit_clipping: float = 10.0,
        optimizer_kwargs: dict[str, Any] | None = None,
        problem: str = "cvrp",
        **kwargs,
    ):
        super().__init__()
        if problem != "cvrp":
            raise NotImplementedError("ICAMCVRP currently supports CVRP only.")
        self.num_starts = num_starts
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.policy = _ICAMPolicy(
            embed_dim, encoder_layers, feed_forward_dim, logit_clipping)
        self.save_hyperparameters(logger=False, ignore=["env"])

    @staticmethod
    def _batch(batch):
        capacity = batch["capacity"].reshape(-1, 1).to(dtype=torch.float32)
        demand = batch["demand"].to(dtype=torch.float32)
        if demand.max() > 1:
            demand = demand / capacity
        depot = batch["depot"].to(dtype=torch.float32).unsqueeze(1)
        nodes = batch["locs"].to(dtype=torch.float32)
        return depot, nodes, demand

    def _rollout(self, batch, sampling: bool):
        depot, nodes, demand = self._batch(batch)
        batch_size, problem_size = nodes.shape[:2]
        pomo_size = self.num_starts or problem_size
        device = nodes.device
        all_xy = torch.cat((depot, nodes), dim=1)
        distance = torch.cdist(all_xy, all_xy, p=2)
        log_scale = distance.new_tensor(problem_size).log2()
        hidden = self.policy.encode(depot, nodes, demand, distance, log_scale)

        current = torch.zeros(batch_size, pomo_size,
                              dtype=torch.long, device=device)
        load = torch.ones(batch_size, pomo_size, device=device)
        visited = torch.zeros(batch_size, pomo_size,
                              problem_size + 1, device=device)
        selected = []
        log_probs = []
        finished = torch.zeros(batch_size, pomo_size,
                               dtype=torch.bool, device=device)

        for step in range(problem_size + 1):
            if step == 1 and pomo_size > 1:
                current = torch.arange(
                    1, pomo_size + 1, device=device).clamp_max(problem_size)[None].expand(batch_size, -1)
                probability = torch.ones(batch_size, pomo_size, device=device)
            elif step == 0:
                probability = torch.ones(batch_size, pomo_size, device=device)
            else:
                current_distance = distance.gather(
                    1, current[..., None].expand(-1, -1, problem_size + 1))
                mask = visited.clone()
                demand_all = torch.cat(
                    (torch.zeros(batch_size, 1, device=device), demand), dim=1)
                mask[demand_all[:, None, :] > load[..., None]] = float("-inf")
                mask[:, :, 0][current != 0] = 0
                logits = self.policy.logits(
                    hidden, current, load, current_distance, mask, log_scale)
                probs = torch.softmax(logits, dim=-1)
                if sampling:
                    next_node = probs.reshape(
                        batch_size * pomo_size, -1).multinomial(1).reshape(batch_size, pomo_size)
                    probability = probs.gather(-1,
                                               next_node[..., None]).squeeze(-1)
                else:
                    next_node = probs.argmax(dim=-1)
                    probability = torch.ones(
                        batch_size, pomo_size, device=device)
                current = next_node

            selected.append(current)
            log_probs.append(probability.clamp_min(1e-12).log())
            selected_demand = torch.cat(
                (torch.zeros(batch_size, 1, device=device), demand), dim=1).gather(1, current)
            load = load - selected_demand
            at_depot = current == 0
            load[at_depot] = 1
            visited.scatter_(-1, current[..., None], float("-inf"))
            visited[:, :, 0][~at_depot] = 0
            finished = finished | (visited == float("-inf")).all(dim=-1)
            if bool(finished.all()):
                break

        route = torch.stack(selected, dim=-1)
        route_xy = all_xy[:, None].expand(-1, pomo_size, -1, -1).gather(
            2, route[..., None].expand(-1, -1, -1, 2))
        reward = -((route_xy - route_xy.roll(-1, dims=2))
                   ** 2).sum(-1).sqrt().sum(-1)
        return reward, torch.stack(log_probs, dim=-1).sum(-1)

    def training_step(self, batch, batch_idx):
        reward, log_prob = self._rollout(batch, sampling=True)
        baseline = reward.mean(dim=1, keepdim=True).detach()
        loss = -((reward.detach() - baseline) * log_prob).mean()
        self.log("train/reward", reward.max(dim=1).values.mean(), prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            reward, _ = self._rollout(batch, sampling=False)
        # Match POMO's logged val metric: MEAN over all multi-start rollouts,
        # not the best start. (POMO logs out["reward"].mean() because its
        # val_metrics only contains "reward", never "max_reward".)
        score = reward.mean(dim=1).values.mean()
        self.log("val/reward", score, prog_bar=True)
        return -score

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)
