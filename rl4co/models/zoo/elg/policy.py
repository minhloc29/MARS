"""ELG-POMO policy adapted to the shared MARS CVRP tensor format.

The implementation retains ELG's three defining pieces: a POMO global policy,
distance-penalized logits, and an additive attention policy over local neighbors.
"""

from __future__ import annotations

import math

import torch

from torch import nn


class _EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.norm1 = nn.InstanceNorm1d(dim, affine=True)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.norm2 = nn.InstanceNorm1d(dim, affine=True)

    @staticmethod
    def _norm(norm, x):
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        attended, _ = self.attn(x, x, x, need_weights=False)
        x = self._norm(self.norm1, x + attended)
        return self._norm(self.norm2, x + self.ff(x))


class _LocalPolicy(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, local_size: int):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("elg_local_dim must be divisible by elg_local_heads")
        self.hidden_dim = hidden_dim
        self.local_size = local_size
        self.input = nn.Linear(3, hidden_dim)
        self.query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.uniform_(self.query, -1, 1)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True, bias=False)

    def forward(self, state, mask):
        cur_dist, theta, _, norm_demand = state.features()
        b, w, n1 = cur_dist.shape
        feasible_customer = ~mask[:, :, 1:]
        # Masked nodes sort behind every finite distance.
        ranked_dist = cur_dist[:, :, 1:].masked_fill(~feasible_customer, float("inf"))
        k = min(self.local_size, state.num_customers)
        _, idx = ranked_dist.topk(k, -1, largest=False)
        idx = idx + 1
        depot = torch.zeros(b, w, 1, dtype=torch.long, device=idx.device)
        idx = torch.cat((depot, idx), -1)
        valid = ~mask.gather(-1, idx)

        distance = cur_dist.gather(-1, idx)
        angle = theta.gather(-1, idx)
        demand = norm_demand.gather(-1, idx)
        finite_distance = distance.masked_fill(~valid, 0)
        scale = finite_distance.amax(-1, keepdim=True).clamp_min(1e-6)
        local_input = torch.stack((finite_distance / scale, angle / math.pi, demand), -1)
        local_input = local_input.reshape(b * w, k + 1, 3)
        valid_flat = valid.reshape(b * w, k + 1)
        encoded = self.input(local_input)
        query = self.query[None, None].expand(b * w, 1, -1)
        context, _ = self.attn(
            query, encoded, encoded, key_padding_mask=~valid_flat, need_weights=False
        )
        scores = (context * encoded).sum(-1) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~valid_flat, 0).reshape(b, w, k + 1)
        output = torch.zeros(b, w, n1, device=idx.device, dtype=scores.dtype)
        return output.scatter(-1, idx, scores)


class ELGPolicy(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int | None = None,
        local_size: int = 40,
        local_dim: int = 32,
        local_heads: int = 4,
        mode: str = "joint",
        distance_penalty: bool = True,
        logit_clipping: float = 50.0,
        xi: float = -1.0,
    ):
        super().__init__()
        if mode not in {"joint", "only_global", "only_local"}:
            raise ValueError("elg_mode must be joint, only_global, or only_local")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by elg_num_heads")
        self.mode = mode
        self.local_size = local_size
        self.distance_penalty = distance_penalty
        self.logit_clipping = logit_clipping
        self.xi = xi
        self.local_enabled = mode != "only_global"
        self.depot_embed = nn.Linear(2, embed_dim)
        self.customer_embed = nn.Linear(3, embed_dim)
        self.encoder = nn.ModuleList(
            _EncoderLayer(embed_dim, num_heads, feedforward_dim or 4 * embed_dim)
            for _ in range(num_layers)
        )
        self.query = nn.Linear(embed_dim + 1, embed_dim, bias=False)
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_out = nn.Linear(embed_dim, embed_dim)
        self.local = _LocalPolicy(local_dim, local_heads, local_size)
        self.encoded = None

    def prepare(self, state):
        if self.mode == "only_local":
            self.encoded = None
            return
        depot = self.depot_embed(state.xy[:, :1])
        customers = self.customer_embed(torch.cat((state.xy[:, 1:], state.demand[:, 1:, None]), -1))
        encoded = torch.cat((depot, customers), 1)
        for layer in self.encoder:
            encoded = layer(encoded)
        self.encoded = encoded

    def logits(self, state):
        mask = state.mask()
        b, w = state.current.shape
        n1 = state.xy.size(1)
        if self.mode == "only_local":
            score = torch.zeros(b, w, n1, device=state.xy.device)
        else:
            current = self.encoded.gather(
                1, state.current[..., None].expand(b, w, self.encoded.size(-1))
            )
            query = self.query(torch.cat((current, state.load[..., None]), -1))
            dim = query.size(-1)
            head_dim = dim // self.num_heads
            q = self.q_proj(query).view(b, w, self.num_heads, head_dim).transpose(1, 2)
            k = self.k_proj(self.encoded).view(b, n1, self.num_heads, head_dim).transpose(1, 2)
            v = self.v_proj(self.encoded).view(b, n1, self.num_heads, head_dim).transpose(1, 2)
            weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            weights = weights.masked_fill(mask[:, None], float("-inf")).softmax(-1)
            context = torch.matmul(weights, v).transpose(1, 2).reshape(b, w, dim)
            context = self.attn_out(context)
            score = torch.matmul(context, self.encoded.transpose(1, 2)) / math.sqrt(self.encoded.size(-1))

        cur_dist = state.features()[0]
        if self.distance_penalty:
            ranked = cur_dist[:, :, 1:].masked_fill(mask[:, :, 1:], float("inf"))
            k = min(self.local_size, state.num_customers)
            dist, idx = ranked.topk(k, -1, largest=False)
            valid = torch.isfinite(dist)
            dist = dist.masked_fill(~valid, 0)
            dist = dist / dist.amax(-1, keepdim=True).clamp_min(1e-6)
            penalty = torch.full_like(score, self.xi)
            penalty.scatter_(-1, idx + 1, -dist)
            penalty[:, :, 0] = 0
            score = score + penalty
        if self.local_enabled:
            score = score + self.local(state, mask)
        return self.logit_clipping * torch.tanh(score)
