"""Invariant Nested View Transformer network for CVRP.

This is a dependency-free adaptation of the official INViT implementation.
The original ``torch_cluster.knn`` candidate lookup is replaced by an
equivalent batched PyTorch top-k lookup so INViT runs in the MARS environment.
"""

from __future__ import annotations

import torch

from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


def _scale_local_view(x: torch.Tensor, reference_length: int) -> torch.Tensor:
    """Translate and isotropically scale a nested view, as upstream INViT."""
    # what is x: x is a tensor of shape (batch_size, num_points, num_features) representing a local view of points in a nested structure. The function scales and translates this view based on the reference points provided in the first `reference_length` points of the tensor. It normalizes the coordinates to fit within a unit cube by subtracting the minimum value and dividing by the extent (max - min) of the reference points. This ensures that the local view is invariant to translation and scaling, which is important for the INViT model to generalize across different instances of the problem.
    reference = x[:, :reference_length]
    minimum = reference.amin(dim=1, keepdim=True)
    extent = (reference.amax(dim=1) - reference.amin(dim=1)).amax(dim=-1)
    extent = extent.clamp_min(torch.finfo(x.dtype).eps)
    return (x - minimum) / extent[:, None, None]


def _nearest_candidates(
    nodes: torch.Tensor,
    current: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return k nearest unmasked customer indices plus a padding mask."""
    batch, n, _ = nodes.shape
    count = min(k, n)
    distance = (nodes - current).square().sum(-1).masked_fill(mask, float("inf"))
    values, indices = distance.topk(count, dim=-1, largest=False, sorted=True)
    invalid = values.isinf()
    if count < k:
        pad = k - count
        indices = torch.cat(
            (indices, torch.zeros(batch, pad, dtype=torch.long, device=nodes.device)), dim=1
        )
        invalid = torch.cat(
            (invalid, torch.ones(batch, pad, dtype=torch.bool, device=nodes.device)), dim=1
        )
    return indices, invalid


class MultiHeadAttention(nn.Module):
    """INViT encoder attention, including its internal residual connection."""

    def __init__(self, num_heads: int, embed_dim: int):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("INViT embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, query, key, value, mask=None):
        batch, q_len, embed_dim = query.shape
        k_len = key.size(1)
        residual = query
        q = self.w_q(query).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q / self.head_dim**0.5, k.transpose(2, 3))
        if mask is not None:
            scores = scores.masked_fill(mask[:, None].bool(), -1e9)
        attention = scores.softmax(dim=-1)
        output = (
            torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch, q_len, embed_dim)
        )
        return self.norm(self.out(output) + residual), attention


class TransformerEncoder(nn.Module):
    def __init__(self, layers: int, embed_dim: int, num_heads: int, feedforward_dim: int):
        super().__init__()
        self.attention = nn.ModuleList(
            MultiHeadAttention(num_heads, embed_dim) for _ in range(layers)
        )
        self.linear1 = nn.ModuleList(nn.Linear(embed_dim, feedforward_dim) for _ in range(layers))
        self.linear2 = nn.ModuleList(nn.Linear(feedforward_dim, embed_dim) for _ in range(layers))
        self.norm1 = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(layers))
        self.norm2 = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(layers))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        for attention, linear1, linear2, norm1, norm2 in zip(
            self.attention, self.linear1, self.linear2, self.norm1, self.norm2
        ):
            residual = hidden
            hidden, _ = attention(hidden, hidden, hidden, mask)
            hidden = norm1(residual + hidden)
            residual = hidden
            hidden = norm2(residual + linear2(F.relu(linear1(hidden))))
        return hidden


class NestedViewEncoder(nn.Module):
    def __init__(self, embed_dim, feedforward_dim, layers, num_heads):
        super().__init__()
        self.customer_embedding = nn.Linear(3, embed_dim)
        self.current_embedding = nn.Linear(3, embed_dim)
        self.depot_embedding = nn.Linear(2, embed_dim)
        self.encoder = TransformerEncoder(layers, embed_dim, num_heads, feedforward_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        indices: torch.Tensor,
        current: torch.Tensor,
        depot: torch.Tensor,
        demands: torch.Tensor,
        remaining_capacity: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, k = indices.shape
        gather_idx = indices[..., None].expand(-1, -1, 2)
        group = nodes.gather(1, gather_idx)
        group_demands = demands.gather(1, indices)
        coordinates = torch.cat((group, current, depot[:, None]), dim=1)
        coordinates = _scale_local_view(coordinates, k + 1)
        customer = torch.cat((coordinates[:, :k], group_demands[..., None]), dim=-1)
        current_features = torch.cat(
            (coordinates[:, k : k + 1], remaining_capacity[:, None, None]), dim=-1
        )
        hidden = torch.cat(
            (
                self.customer_embedding(customer),
                self.current_embedding(current_features),
                self.depot_embedding(coordinates[:, k + 1 : k + 2]),
            ),
            dim=1,
        )
        encoder_mask = torch.cat(
            (mask, torch.zeros(batch, 2, dtype=torch.bool, device=nodes.device)), dim=1
        )
        return self.encoder(hidden, encoder_mask[:, None])


def _decoder_attention(query, key, value, num_heads, mask=None, clip=None):
    batch, length, embed_dim = key.shape
    head_dim = embed_dim // num_heads
    q = query.view(batch, 1, num_heads, head_dim).transpose(1, 2)
    k = key.view(batch, length, num_heads, head_dim).transpose(1, 2)
    v = value.view(batch, length, num_heads, head_dim).transpose(1, 2)
    logits = torch.matmul(q, k.transpose(2, 3)) / head_dim**0.5
    if clip is not None:
        logits = clip * torch.tanh(logits)
    if mask is not None:
        logits = logits.masked_fill(mask[:, None, None], -1e9)
    weights = logits.softmax(-1)
    output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, 1, embed_dim)
    return output, weights.mean(1)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, hidden, key, value, mask):
        hidden = hidden.view(hidden.size(0), 1, -1)
        attended, _ = _decoder_attention(self.q(hidden), key, value, self.num_heads, mask)
        hidden = self.norm1((hidden + self.out(attended)).squeeze(1))[:, None]
        return self.norm2((hidden + self.linear2(F.relu(self.linear1(hidden)))).squeeze(1))


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = layers
        self.intermediate = nn.ModuleList(
            DecoderLayer(embed_dim, num_heads) for _ in range(layers - 1)
        )
        self.final_query = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden, keys, values, mask):
        for layer_index, layer in enumerate(self.intermediate):
            start = layer_index * self.embed_dim
            hidden = layer(
                hidden,
                keys[:, :, start : start + self.embed_dim],
                values[:, :, start : start + self.embed_dim],
                mask,
            )
        start = (self.layers - 1) * self.embed_dim
        hidden = hidden.view(hidden.size(0), 1, self.embed_dim)
        _, weights = _decoder_attention(
            self.final_query(hidden),
            keys[:, :, start : start + self.embed_dim],
            values[:, :, start : start + self.embed_dim],
            1,
            mask,
            clip=10,
        )
        return weights.squeeze(1)


class INViTPolicy(nn.Module):
    """Construct CVRP routes from invariant local and nested state views."""

    def __init__(
        self,
        embed_dim: int = 128,
        feedforward_dim: int | None = None,
        num_heads: int = 8,
        state_sizes: tuple[int, ...] = (35, 50, 65),
        action_size: int = 15,
        state_encoder_layers: int = 2,
        action_encoder_layers: int = 2,
        decoder_layers: int = 3,
    ):
        super().__init__()
        if not state_sizes or any(size < action_size for size in state_sizes):
            raise ValueError("Every INViT state size must be at least the action size")
        if tuple(state_sizes) != tuple(sorted(state_sizes)):
            raise ValueError("INViT state sizes must be sorted")
        if action_size < 1 or decoder_layers < 1:
            raise ValueError("INViT action_size and decoder_layers must be positive")
        feedforward_dim = feedforward_dim or 4 * embed_dim
        self.action_size = action_size
        self.state_sizes = tuple(state_sizes)
        self.action_encoder = NestedViewEncoder(
            embed_dim, feedforward_dim, action_encoder_layers, num_heads
        )
        self.state_encoders = nn.ModuleList(
            NestedViewEncoder(embed_dim, feedforward_dim, state_encoder_layers, num_heads)
            for _ in state_sizes
        )
        combined_dim = (len(state_sizes) + 1) * embed_dim
        self.decoder = TransformerDecoder(embed_dim, num_heads, decoder_layers)
        self.key_projection = nn.Linear(combined_dim, decoder_layers * embed_dim)
        self.value_projection = nn.Linear(combined_dim, decoder_layers * embed_dim)
        self.query_projection = nn.Linear(2 * combined_dim, embed_dim)

    @staticmethod
    def _inputs(td) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        demand = td["demand"].float()
        locs = td["locs"].float()
        if locs.size(1) == demand.size(1) + 1:
            return locs[:, 1:], locs[:, 0], demand
        return locs, td["depot"].float(), demand

    def _step_probabilities(
        self,
        nodes,
        depot,
        demands,
        current,
        current_index,
        used_capacity,
    ):
        batch = nodes.size(0)
        unfinished = demands.gt(1e-6).any(dim=1)
        remaining_capacity = 1.0 - used_capacity
        feasible = demands.gt(1e-6) & demands.le(remaining_capacity[:, None] + 1e-6)
        action_indices, action_mask = _nearest_candidates(
            nodes, current, ~feasible, self.action_size
        )

        state_mask = (~feasible).scatter(
            1, action_indices, torch.ones_like(action_indices, dtype=torch.bool)
        )
        extra_count = max(self.state_sizes) - self.action_size
        extra_indices, extra_mask = _nearest_candidates(nodes, current, state_mask, extra_count)
        state_indices = torch.cat((action_indices, extra_indices), dim=1)
        state_padding = torch.cat((action_mask, extra_mask), dim=1)

        action_hidden = self.action_encoder(
            nodes,
            action_indices,
            current,
            depot,
            demands,
            remaining_capacity,
            action_mask,
        )
        query_parts = [
            action_hidden[:, self.action_size : self.action_size + 1],
            action_hidden[:, self.action_size + 1 : self.action_size + 2],
        ]
        other_parts = [
            torch.cat(
                (
                    action_hidden[:, : self.action_size],
                    action_hidden[:, self.action_size + 1 : self.action_size + 2],
                ),
                dim=1,
            )
        ]
        for size, encoder in zip(self.state_sizes, self.state_encoders):
            hidden = encoder(
                nodes,
                state_indices[:, :size],
                current,
                depot,
                demands,
                remaining_capacity,
                state_padding[:, :size],
            )
            query_parts.extend((hidden[:, size : size + 1], hidden[:, size + 1 : size + 2]))
            other_parts.append(
                torch.cat(
                    (hidden[:, : self.action_size], hidden[:, size + 1 : size + 2]), dim=1
                )
            )

        query = self.query_projection(torch.cat(query_parts, dim=-1))
        other = torch.cat(other_parts, dim=-1)
        decoder_mask = torch.cat(
            (action_mask, torch.zeros(batch, 1, dtype=torch.bool, device=nodes.device)), dim=1
        )
        decoder_mask[:, -1] = current_index.eq(nodes.size(1)) & unfinished
        probabilities = self.decoder(
            query,
            self.key_projection(other),
            self.value_projection(other),
            decoder_mask,
        )
        return probabilities, action_indices, unfinished

    @staticmethod
    def _advance(nodes, depot, demands, used_capacity, next_index):
        batch, n, _ = nodes.shape
        batch_index = torch.arange(batch, device=nodes.device)
        customer = next_index.lt(n)
        safe_index = next_index.clamp_max(n - 1)
        selected_demand = demands[batch_index, safe_index]
        demands[batch_index[customer], safe_index[customer]] = 0
        used_capacity = torch.where(customer, used_capacity + selected_demand, 0.0)
        current = torch.where(
            customer[:, None, None],
            nodes[batch_index, safe_index][:, None],
            depot[:, None],
        )
        return demands, current, next_index, used_capacity

    def replay_log_probabilities(self, td, actions):
        """Yield teacher-forced log-probabilities without retaining a full tour graph."""
        nodes, depot, demands = self._inputs(td)
        demands = demands.clone()
        batch, n, _ = nodes.shape
        current = depot[:, None]
        current_index = torch.full((batch,), n, dtype=torch.long, device=nodes.device)
        used_capacity = torch.zeros(batch, dtype=demands.dtype, device=nodes.device)
        batch_index = torch.arange(batch, device=nodes.device)

        for target in actions.unbind(1):
            probabilities, action_indices, unfinished = self._step_probabilities(
                nodes, depot, demands, current, current_index, used_capacity
            )
            target_customer = target.gt(0)
            matches = action_indices.eq((target - 1).clamp_min(0)[:, None])
            if not matches[target_customer & unfinished].any(dim=1).all():
                raise RuntimeError("Teacher-forced INViT action is absent from its local view")
            customer_choice = matches.to(torch.int64).argmax(1)
            choice = torch.where(
                target_customer, customer_choice, torch.full_like(customer_choice, self.action_size)
            )
            chosen_probability = probabilities[batch_index, choice].clamp_min(1e-12)
            yield torch.where(unfinished, chosen_probability.log(), 0.0)
            next_index = torch.where(target_customer, target - 1, torch.full_like(target, n))
            demands, current, current_index, used_capacity = self._advance(
                nodes, depot, demands, used_capacity, next_index
            )

    def forward(self, td, env=None, phase="train", decode_type=None, **unused):
        nodes, depot, demands = self._inputs(td)
        demands = demands.clone()
        batch, n, _ = nodes.shape
        greedy = decode_type == "greedy" or (decode_type is None and phase != "train")
        current = depot[:, None]
        current_index = torch.full((batch,), n, dtype=torch.long, device=nodes.device)
        used_capacity = torch.zeros(batch, dtype=demands.dtype, device=nodes.device)
        batch_index = torch.arange(batch, device=nodes.device)
        actions, log_probabilities = [], []

        for _ in range(2 * n + 1):
            unfinished = demands.gt(1e-6).any(dim=1)
            if not unfinished.any():
                break
            probabilities, action_indices, unfinished = self._step_probabilities(
                nodes, depot, demands, current, current_index, used_capacity
            )
            choice = probabilities.argmax(1) if greedy else Categorical(probabilities).sample()
            chosen_probability = probabilities[batch_index, choice].clamp_min(1e-12)
            next_index = torch.where(
                choice.eq(self.action_size),
                torch.full_like(choice, n),
                action_indices[batch_index, choice.clamp_max(self.action_size - 1)],
            )
            actions.append(torch.where(next_index.eq(n), 0, next_index + 1))
            log_probabilities.append(torch.where(unfinished, chosen_probability.log(), 0.0))

            demands, current, current_index, used_capacity = self._advance(
                nodes, depot, demands, used_capacity, next_index
            )
        else:
            raise RuntimeError("INViT decoding exceeded the CVRP 2N+1 step safety bound")

        action_tensor = torch.stack(actions, dim=1)
        log_likelihood = torch.stack(log_probabilities, dim=1).sum(1)
        reward = -route_length(nodes, depot, action_tensor)
        return {
            "reward": reward,
            "actions": action_tensor,
            "log_likelihood": log_likelihood,
        }


def route_length(nodes: torch.Tensor, depot: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Compute closed CVRP route length for RL4CO-style 0=depot actions."""
    coordinates = torch.cat((depot[:, None], nodes), dim=1)
    selected = coordinates.gather(1, actions[..., None].expand(-1, -1, 2))
    path = torch.cat((depot[:, None], selected, depot[:, None]), dim=1)
    return (path[:, 1:] - path[:, :-1]).norm(dim=-1).sum(1)
