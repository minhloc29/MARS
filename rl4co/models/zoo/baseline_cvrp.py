"""Small, dependency-free CVRP rollout utilities for integrated baselines.

The cached MARS data already stores normalized demands (vehicle capacity is one).
This module deliberately accepts ordinary dictionaries so baseline adapters do not
depend on RL4CO's internal TensorDict state representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def open_route_reward(coords: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Open (OVRP) route reward from ordered 3D coordinates and the encoded
    action IDs that produced them.

    ``coords`` is (B, W, T, 2) = the coordinate at each visit *after* the
    initial depot, i.e. ``coords[..., k, :]`` is the node reached by action
    ``actions[..., k]`` (0 = depot, 1..N = customers). The arc from the previous
    visit into ``coords[..., k]`` belongs to an open route iff its destination
    is a customer (``actions[..., k] != 0``); the final return-to-depot arc —
    and any arc whose destination is the depot — is excluded. This mirrors
    :meth:`rl4co.envs.routing.ovrp.env.OVRPEnv._get_reward`.

    Args:
        coords: (B, W, T, 2) coordinates of every visited node including the
            depot visits recorded in ``actions``.
        actions: (B, W, T) encoded actions (0 = depot, 1..N = customers).

    Returns:
        (B, W) negative open-route tour length, matching CVRP closed-tour sign
        convention so shared POMO-style advantages/right uses it unchanged.
    """
    arc = (coords[:, :, 1:] - coords[:, :, :-1]).norm(dim=-1)          # (B, W, T-1)
    dest_customer = (actions[:, :, 1:] != 0).to(arc.dtype)             # (B, W, T-1)
    return -(arc * dest_customer).sum(-1)


def normalize_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    locs = batch["locs"].float()
    depot = batch["depot"].float()
    if depot.ndim == 2:
        depot = depot[:, None]
    demand = batch["demand"].float()
    capacity = batch.get("capacity")
    if capacity is not None:
        capacity = capacity.float().reshape(-1, 1)
        # Only divide if demand itself still looks raw (i.e. values plausibly
        # exceed 1, meaning it hasn't been pre-normalized by capacity yet).
        # A capacity field alone is not sufficient evidence — some caches
        # store normalized demand alongside the original raw capacity value.
        if bool((demand.amax(dim=tuple(range(1, demand.dim()))) > 1 + 1e-6).any()):
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
    # ---- VRPTW only (None for CVRP/OVRP; behaviour then unchanged) ----
    # Follows CVRPTWEnv: service must *start* by tw_end; wait until tw_start;
    # time resets to 0 at the depot. travel time = distance * tw_travel_scale
    # (1.0 matches CVRPTWEnv, the env LEGO trains on).
    time_windows: torch.Tensor | None = None   # (B, N+1, 2), depot first
    durations: torch.Tensor | None = None      # (B, N+1)
    horizon: torch.Tensor | None = None        # (B,) depot window end
    time: torch.Tensor | None = None           # (B, W) current time
    tw_travel_scale: float = 1.0
    tw_late: torch.Tensor | int = 0            # deadline violations (must stay 0)
    tw_fallbacks: torch.Tensor | int = 0       # safety-valve hits (summed over mask() calls; must stay 0)

    @classmethod
    def from_batch(cls, batch: dict[str, torch.Tensor], width: int = 1,
                   problem: str = "cvrp", tw_travel_scale: float = 1.0):
        data = normalize_batch(batch)
        xy, demand = data["xy"], data["demand"]
        b, n1, _ = xy.shape
        tw_kwargs = {}
        if problem == "vrptw":
            if "time_windows" not in batch:
                raise KeyError("problem='vrptw' needs 'time_windows' in the batch; "
                               "regenerate the cache with --problem vrptw")
            tw = batch["time_windows"].float()
            dur = batch["durations"].float() if "durations" in batch else torch.zeros_like(tw[..., 0])
            if tw.size(1) != n1:
                raise ValueError("time_windows must include the depot as node 0")
            tw_kwargs = dict(
                time_windows=tw, durations=dur, horizon=tw[:, 0, 1],
                time=torch.zeros(b, width, device=xy.device),
                tw_travel_scale=float(tw_travel_scale),
            )
        return cls(
            xy=xy,
            demand=demand,
            distance=torch.cdist(xy, xy),
            visited=torch.zeros(b, width, n1, dtype=torch.bool, device=xy.device),
            load=torch.ones(b, width, device=xy.device),
            current=torch.zeros(b, width, dtype=torch.long, device=xy.device),
            finished=torch.zeros(b, width, dtype=torch.bool, device=xy.device),
            actions=[],
            **tw_kwargs,
        )

    @property
    def has_tw(self) -> bool:
        return self.time_windows is not None

    def _cur_dist(self) -> torch.Tensor:
        """(B, W, N+1) distance from each thread's current node."""
        b, w = self.current.shape
        n1 = self.xy.size(1)
        return self.distance.gather(1, self.current[..., None].expand(b, w, n1))

    def tw_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Horizon-normalized (static node windows (B,N+1,2), current time (B,W))."""
        h = self.horizon.clamp_min(1e-6)
        return self.time_windows / h[:, None, None], self.time / h[:, None]

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
        if self.has_tw:
            capacity_mask = mask.clone()
            arrival = self.time[..., None] + self._cur_dist() * self.tw_travel_scale
            late = arrival > self.time_windows[:, None, :, 1] + 1e-6
            late[:, :, 0] = False
            mask |= late
            # Safety valve: at the depot (time 0) every customer is reachable by
            # construction; if none is, relax time for that thread so it cannot
            # deadlock, and count it (must stay 0 for a valid VRPTW result).
            stuck = (self.current == 0) & ~all_visited & ~self.finished \
                & mask[:, :, 1:].all(-1)
            mask = torch.where(stuck[..., None], capacity_mask, mask)
            self.tw_fallbacks = self.tw_fallbacks + stuck.sum()
        mask[:, :, 0] = (self.current == 0) & ~all_visited
        mask[:, :, 0] &= ~self.finished
        # Finished rows are pinned at depot.
        mask[self.finished] = True
        mask[:, :, 0][self.finished] = False
        return mask

    def step(self, selected: torch.Tensor):
        at_depot = selected == 0
        if self.has_tw:
            travel = self._cur_dist().gather(2, selected[..., None]).squeeze(-1) \
                * self.tw_travel_scale
            arrival = self.time + travel
            tw_start = self.time_windows[..., 0].gather(1, selected)   # (B, W)
            tw_end = self.time_windows[..., 1].gather(1, selected)
            service = self.durations.gather(1, selected)
            self.tw_late = self.tw_late + (~at_depot & (arrival > tw_end + 1e-6)).sum()
            self.time = torch.where(
                at_depot, torch.zeros_like(arrival),
                torch.maximum(arrival, tw_start) + service,
            )
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

    def reward(self, problem: str = "cvrp") -> torch.Tensor:
        actions = torch.stack(self.actions, -1)
        b, w, steps = actions.shape
        coords = self.xy[:, None].expand(b, w, -1, 2).gather(
            2, actions[..., None].expand(b, w, steps, 2)
        )
        if problem == "ovrp":
            # Open routes: drop every arc whose destination is the depot, i.e.
            # the final return-to-depot leg of each route and any re-anchoring
            # depot visit. Same convention as OVRPEnv._get_reward.
            return open_route_reward(coords, actions)
        # Rollout begins and ends at depot, so adjacent segments contain the
        # complete route cost and no cyclic shortcut is introduced.
        return -(coords[:, :, 1:] - coords[:, :, :-1]).norm(dim=-1).sum(-1)


def rollout(policy, batch, width: int, decode_type: str, return_log_probs: bool = True,
            problem: str = "cvrp", tw_travel_scale: float = 1.0):
    """POMO-compatible rollout shared by ELG and DGL adapters.

    ``problem`` selects the reward convention: "ovrp" uses open-route reward
    (drop return-to-depot arcs), otherwise closed-tour CVRP reward.
    """
    state = CVRPState.from_batch(batch, width=min(width, batch["locs"].size(1)),
                                 problem=problem, tw_travel_scale=tw_travel_scale)
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

    reward = state.reward(problem=problem)
    log_prob = torch.stack(log_probs, -1).sum(-1) if log_probs else torch.zeros_like(reward)
    out = {"reward": reward, "log_prob": log_prob, "actions": torch.stack(state.actions, -1)}
    if state.has_tw:
        out["tw_late"] = int(state.tw_late)
        out["tw_fallbacks"] = int(state.tw_fallbacks)
    return out
