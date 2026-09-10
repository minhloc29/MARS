"""Small, dependency-free CVRP rollout utilities for integrated baselines.

The cached MARS data already stores normalized demands (vehicle capacity is one).
This module deliberately accepts ordinary dictionaries so baseline adapters do not
depend on RL4CO's internal TensorDict state representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def normalize_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the common depot-first representation used by DGL and ELG."""
    locs = batch["locs"].float()
    depot = batch["depot"].float()
    if depot.ndim == 2:
        depot = depot[:, None]
    demand = batch["demand"].float()
    capacity = batch.get("capacity")
    if capacity is not None:
        capacity = capacity.float().reshape(-1, 1)
        # Current generated data is normalized already (capacity == 1).  This
        # also makes the adapter safe for a cache containing raw demands.
        if bool((capacity > 1 + 1e-6).any()):
            demand = demand / capacity
    return {
        "xy": torch.cat((depot, locs), 1),
        "demand": torch.cat((torch.zeros_like(demand[:, :1]), demand), 1),
    }


@dataclass
class CVRPState:
    xy: torch.Tensor
    demand: torch.Tensor
    distance: torch.Tensor
    visited: torch.Tensor
    load: torch.Tensor
    current: torch.Tensor
    finished: torch.Tensor
    actions: list[torch.Tensor]

    @classmethod
    def from_batch(cls, batch: dict[str, torch.Tensor], width: int = 1):
        data = normalize_batch(batch)
        xy, demand = data["xy"], data["demand"]
        b, n1, _ = xy.shape
        return cls(
            xy=xy,
            demand=demand,
            distance=torch.cdist(xy, xy),
            visited=torch.zeros(b, width, n1, dtype=torch.bool, device=xy.device),
            load=torch.ones(b, width, device=xy.device),
            current=torch.zeros(b, width, dtype=torch.long, device=xy.device),
            finished=torch.zeros(b, width, dtype=torch.bool, device=xy.device),
            actions=[],
        )

    @property
    def batch_size(self):
        return self.xy.size(0)

    @property
    def width(self):
        return self.current.size(1)

    @property
    def num_customers(self):
        return self.xy.size(1) - 1

    def mask(self) -> torch.Tensor:
        """True means unavailable; depot remains available between routes."""
        mask = self.visited.clone()
        demand = self.demand[:, None].expand_as(mask)
        mask |= demand > self.load[:, :, None] + 1e-6
        all_visited = self.visited[:, :, 1:].all(-1)
        mask[:, :, 0] = (self.current == 0) & ~all_visited
        mask[:, :, 0] &= ~self.finished
        # Finished rows are pinned at depot.
        mask[self.finished] = True
        mask[:, :, 0][self.finished] = False
        return mask

    def step(self, selected: torch.Tensor):
        at_depot = selected == 0
        picked_demand = self.demand[:, None].expand(
            -1, self.width, -1
        ).gather(2, selected[..., None]).squeeze(-1)
        self.load = torch.where(at_depot, torch.ones_like(self.load), self.load - picked_demand)
        self.visited.scatter_(2, selected[..., None], True)
        self.visited[:, :, 0] = False
        self.current = selected
        self.actions.append(selected)
        self.finished = self.visited[:, :, 1:].all(-1) & at_depot

    def features(self):
        b, w = self.current.shape
        n1 = self.xy.size(1)
        dist = self.distance[:, None].expand(b, w, n1, n1)
        current = self.current[:, :, None, None].expand(b, w, 1, n1)
        cur_dist = dist.gather(2, current).squeeze(2)
        rel_xy = self.xy[:, None].expand(b, w, n1, 2) - self.xy.gather(
            1, self.current[:, :, None].expand(b, w, 2)
        )[:, :, None]
        theta = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])
        norm_demand = self.demand[:, None].expand(b, w, n1) / self.load.clamp_min(1e-7)[..., None]
        return cur_dist, theta, rel_xy, norm_demand

    def reward(self) -> torch.Tensor:
        actions = torch.stack(self.actions, -1)
        b, w, steps = actions.shape
        coords = self.xy[:, None].expand(b, w, -1, 2).gather(
            2, actions[..., None].expand(b, w, steps, 2)
        )
        # Rollout begins and ends at depot, so adjacent segments contain the
        # complete route cost and no cyclic shortcut is introduced.
        return -(coords[:, :, 1:] - coords[:, :, :-1]).norm(dim=-1).sum(-1)


def rollout(policy, batch, width: int, decode_type: str, return_log_probs: bool = True):
    """POMO-compatible rollout shared by ELG and DGL adapters."""
    state = CVRPState.from_batch(batch, width=min(width, batch["locs"].size(1)))
    b, w = state.current.shape
    depot = torch.zeros(b, w, dtype=torch.long, device=state.xy.device)
    state.actions.append(depot)
    policy.prepare(state)
    log_probs = []

    # Match POMO: distinct forced first customers, identical across a batch.
    if decode_type == "sampling":
        starts = torch.randperm(state.num_customers, device=state.xy.device)[:w] + 1
    else:
        starts = torch.arange(1, w + 1, device=state.xy.device)
    selected = starts[None].expand(b, w)
    state.step(selected)

    while not bool(state.finished.all()):
        logits = policy.logits(state)
        logits = logits.masked_fill(state.mask(), float("-inf"))
        if decode_type == "sampling":
            selected = torch.distributions.Categorical(logits=logits).sample()
        else:
            selected = logits.argmax(-1)
        if return_log_probs:
            log_probs.append(
                torch.log_softmax(logits, -1).gather(-1, selected[..., None]).squeeze(-1)
            )
        state.step(selected)

    reward = state.reward()
    log_prob = torch.stack(log_probs, -1).sum(-1) if log_probs else torch.zeros_like(reward)
    return {"reward": reward, "log_prob": log_prob, "actions": torch.stack(state.actions, -1)}
