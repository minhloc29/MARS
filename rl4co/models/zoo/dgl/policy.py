"""Dynamic global-local CVRP policy used by the integrated DGL baseline."""

from __future__ import annotations

import torch

from torch import nn


class DGLPolicy(nn.Module):
    """Re-encode a bounded dynamic neighborhood at every construction step.

    Candidates are the union of nearest unvisited nodes to the current node and
    depot. This is the same global-local candidate construction used by upstream
    DGL, expressed with standard depot actions rather than its doubled action IDs.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        feedforward_dim: int | None = None,
        knn: int = 100,
        depot_knn: int = 100,
        logit_clipping: float = 10.0,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by dgl_num_heads")
        self.knn = knn
        self.depot_knn = depot_knn
        self.logit_clipping = logit_clipping
        self.input = nn.Linear(9, embed_dim)
        layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, feedforward_dim or 4 * embed_dim,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def prepare(self, state):
        # Distance is computed once by CVRPState and reused at every DGL step.
        return None

    def logits(self, state):
        mask = state.mask()
        b, w, n1 = mask.shape
        cur_dist, _, relative_xy, norm_demand = state.features()
        depot_dist = state.distance[:, 0][:, None].expand(b, w, n1)
        unavailable = mask.clone()
        unavailable[:, :, 0] = True

        def nearest(distance, k):
            k = min(k, state.num_customers)
            ranked = distance[:, :, 1:].masked_fill(unavailable[:, :, 1:], float("inf"))
            return ranked.topk(k, -1, largest=False).indices + 1

        local_idx = nearest(cur_dist, self.knn)
        depot_idx = nearest(depot_dist, self.depot_knn)
        depot = torch.zeros(b, w, 1, dtype=torch.long, device=mask.device)
        candidates = torch.cat((depot, local_idx, depot_idx), -1)
        valid = ~mask.gather(-1, candidates)

        def gather(values):
            return values.gather(-1, candidates)

        cand_cur_dist = gather(cur_dist).masked_fill(~valid, 0)
        cand_depot_dist = gather(depot_dist).masked_fill(~valid, 0)
        scale = torch.maximum(
            cand_cur_dist.amax(-1, keepdim=True), cand_depot_dist.amax(-1, keepdim=True)
        ).clamp_min(1e-6)
        x = relative_xy[..., 0].gather(-1, candidates) / scale
        y = relative_xy[..., 1].gather(-1, candidates) / scale
        demand = gather(norm_demand)
        unvisited_dist = cur_dist.masked_fill(unavailable, 0)
        count = (~unavailable).sum(-1, keepdim=True).clamp_min(1)
        mean = unvisited_dist.sum(-1, keepdim=True) / count
        variance = ((unvisited_dist - mean) ** 2).masked_fill(unavailable, 0).sum(-1, keepdim=True) / count
        mean = mean.expand_as(cand_cur_dist) / scale
        std = variance.sqrt().expand_as(cand_cur_dist) / scale
        is_depot_neighborhood = torch.cat(
            (torch.zeros_like(depot), torch.zeros_like(local_idx), torch.ones_like(depot_idx)), -1
        ).float()
        features = torch.stack((
            x, y, demand, cand_cur_dist / scale, cand_depot_dist / scale,
            state.load[..., None].expand_as(cand_cur_dist), mean, std,
            is_depot_neighborhood,
        ), -1)
        flat_features = features.reshape(b * w, candidates.size(-1), 9)
        flat_valid = valid.reshape(b * w, candidates.size(-1))
        encoded = self.encoder(self.input(flat_features), src_key_padding_mask=~flat_valid)
        local_score = self.score(encoded).squeeze(-1).reshape_as(candidates)
        local_score = local_score.masked_fill(~valid, float("-inf"))

        # A node can occur in both neighborhoods. scatter_reduce preserves its
        # better score and leaves non-candidates unavailable.
        output = torch.full((b, w, n1), float("-inf"), device=mask.device)
        output.scatter_reduce_(-1, candidates, local_score, reduce="amax", include_self=True)
        # Numerical clipping follows the bounded-logit convention in DGL/POMO.
        finite = torch.isfinite(output)
        clipped = self.logit_clipping * torch.tanh(output.masked_fill(~finite, 0))
        return torch.where(finite, clipped, output)
