"""LEHD CVRP neural network layers.

Self-contained copy of the LEHD (Light Encoder Heavy Decoder) architecture for CVRP,
ported from NCO_code/single_objective/LEHD/CVRP/VRPModel.py by Fu Luo et al.
(NeurIPS 2023: "Neural Combinatorial Optimization with Heavy Decoder: Toward
Large Scale Generalization").

Design:
  - Encoder: 1-layer Transformer on (x, y, demand/capacity) features.
  - Decoder: multi-layer cross-attention between "context nodes"
    (first depot + last visited node) and "candidate nodes" (remaining customers),
    outputs a probability over {visit customer directly, visit via depot}.

Differences from the original:
  - Pure PyTorch nn.Module, no global state or cuda hard-coding.
  - All tensor factories use the model's device (passed through forward).
  - embed_dim / head_num / decoder_layer_num are constructor arguments.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    B, n, _ = qkv.shape
    return qkv.reshape(B, n, head_num, -1).transpose(1, 2)


def _multi_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    key_dim = q.size(3)
    score = torch.matmul(q, k.transpose(2, 3)) / (key_dim ** 0.5)
    w = F.softmax(score, dim=3)
    out = torch.matmul(w, v)
    B, H, n, d = out.shape
    return out.transpose(1, 2).reshape(B, n, H * d)


class _FFN(nn.Module):
    def __init__(self, embed_dim: int, ff_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_hidden)
        self.w2 = nn.Linear(ff_hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class _EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, head_num: int, qkv_dim: int, ff_hidden: int):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(head_num * qkv_dim, embed_dim)
        self.ffn = _FFN(embed_dim, ff_hidden)
        self.head_num = head_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = _reshape_by_heads(self.Wq(x), self.head_num)
        k = _reshape_by_heads(self.Wk(x), self.head_num)
        v = _reshape_by_heads(self.Wv(x), self.head_num)
        out = self.combine(_multi_head_attention(q, k, v))
        x = x + out
        return x + self.ffn(x)


class LEHDEncoder(nn.Module):
    """Light 1-layer transformer encoder over (x, y, normalized_demand)."""

    def __init__(self, embed_dim: int = 128, head_num: int = 8,
                 qkv_dim: int = 16, ff_hidden: int = 512):
        super().__init__()
        self.embed = nn.Linear(3, embed_dim)
        self.layer = _EncoderLayer(embed_dim, head_num, qkv_dim, ff_hidden)

    def forward(self, data: torch.Tensor, capacity: float) -> torch.Tensor:
        """
        Args:
            data: (B, V+1, 4)  — col0/1 = x/y, col2 = integer demand (0 for depot)
            capacity: scalar float (same capacity for all instances in batch)
        Returns:
            (B, V+1, embed_dim)
        """
        feat = data[:, :, :3].clone()
        feat[:, :, 2] = feat[:, :, 2] / capacity
        return self.layer(self.embed(feat))


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class _DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, head_num: int, qkv_dim: int, ff_hidden: int):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(head_num * qkv_dim, embed_dim)
        self.ffn = _FFN(embed_dim, ff_hidden)
        self.head_num = head_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = _reshape_by_heads(self.Wq(x), self.head_num)
        k = _reshape_by_heads(self.Wk(x), self.head_num)
        v = _reshape_by_heads(self.Wv(x), self.head_num)
        x = x + self.combine(_multi_head_attention(q, k, v))
        return x + self.ffn(x)


class LEHDDecoder(nn.Module):
    """Heavy multi-layer decoder producing P(node, via_depot | context).

    Output is a probability vector of size 2*N_customers (first half = direct,
    second half = via depot), with already-visited nodes masked to 0.
    """

    def __init__(self, embed_dim: int = 128, decoder_layer_num: int = 6,
                 head_num: int = 8, qkv_dim: int = 16, ff_hidden: int = 512):
        super().__init__()
        # Context embeddings for first-node (depot) and last-node
        self.emb_first = nn.Linear(embed_dim + 1, embed_dim)
        self.emb_last = nn.Linear(embed_dim + 1, embed_dim)
        self.layers = nn.ModuleList([
            _DecoderLayer(embed_dim, head_num, qkv_dim, ff_hidden)
            for _ in range(decoder_layer_num)
        ])
        self.linear_out = nn.Linear(embed_dim, 2)

    # ------------------------------------------------------------------
    # Internal helpers (matching original logic exactly)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_remaining(data: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """Return encoded embeddings for *unselected* customer indices."""
        B, prob_size, D = data.shape
        remaining_len = prob_size - selected.shape[1]
        mask = torch.ones(B, prob_size, dtype=torch.bool, device=data.device)
        idx = selected.long().clamp(0, prob_size - 1)
        mask.scatter_(1, idx, False)
        return data[mask].view(B, remaining_len, D)

    @staticmethod
    def _get_encoding(data: torch.Tensor, node_idx: torch.Tensor) -> torch.Tensor:
        B, P, D = data.shape
        idx = node_idx[:, :, None].expand(B, node_idx.size(1), D)
        return data.gather(1, idx)

    # ------------------------------------------------------------------

    def forward(
        self,
        encoded: torch.Tensor,            # (B, V+1, D)  encoder output
        selected_node_list: torch.Tensor,  # (B, t)       1-indexed visited customers
        capacity: float,
        remaining_capacity: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Returns:
            probs: (B, 2*V)   probability mass; already-visited positions = 0.
        """
        # Customer embeddings only (strip depot at index 0)
        cust_enc = encoded[:, 1:, :]          # (B, V, D)
        # Convert 1-indexed → 0-indexed
        sel = (selected_node_list.long() - 1).clamp(min=0)  # (B, t)

        B, V, D = cust_enc.shape
        remaining_len = V - sel.shape[1]

        left = self._get_remaining(cust_enc, sel)  # (B, V-t, D)

        # First node = depot embedding; last node = most recently visited customer
        first_enc = encoded[:, [0], :]  # (B, 1, D)
        if sel.shape[1] == 0:
            last_enc = first_enc
        else:
            last_enc = self._get_encoding(cust_enc, sel[:, [-1]])  # (B, 1, D)

        rem_cap = remaining_capacity.float().reshape(B, 1, 1) / capacity
        first_cat = torch.cat([first_enc, rem_cap], dim=2)
        last_cat = torch.cat([last_enc, rem_cap], dim=2)

        emb_first = self.emb_first(first_cat)   # (B, 1, D)
        emb_last = self.emb_last(last_cat)       # (B, 1, D)

        # Sequence: [depot_context, ...remaining_customers..., last_node_context]
        seq = torch.cat([emb_first, left, emb_last], dim=1)  # (B, V-t+2, D)

        for layer in self.layers:
            seq = layer(seq)

        out = self.linear_out(seq)               # (B, V-t+2, 2)
        # Mask the first (depot-context) and last (last-node-context) positions
        out[:, [0, -1], :] = float("-inf")
        # Flatten to (B, 2*(V-t+2)) — col0 = direct, col1 = via-depot
        out = torch.cat([out[:, :, 0], out[:, :, 1]], dim=1)
        probs = F.softmax(out, dim=-1)

        # Extract customer probabilities (skip context slots 0 and -1)
        cust_direct = probs[:, 1: remaining_len + 1]
        cust_via = probs[:, remaining_len + 2 + 1: 2 * remaining_len + 2 + 1]
        probs_cust = torch.cat([cust_direct, cust_via], dim=1)  # (B, 2*(V-t))

        # Smooth tiny probs
        mask_small = probs_cust <= 1e-5
        probs_cust = probs_cust.clone()
        probs_cust[mask_small] = probs_cust[mask_small] + 1e-7

        # Build full 2*V output with visited positions zeroed
        full = torch.zeros(B, 2 * V, device=encoded.device)
        mask_full = torch.ones(B, 2 * V, dtype=torch.bool, device=encoded.device)
        sel_idx = sel.long()
        mask_full.scatter_(1, sel_idx, False)
        mask_full.scatter_(1, sel_idx + V, False)
        full[mask_full] = probs_cust.reshape(-1)

        return full


# ---------------------------------------------------------------------------
# Top-level VRPModel (thin wrapper used by LEHDModel)
# ---------------------------------------------------------------------------

class LEHDVRPModel(nn.Module):
    """Encoder + Decoder combined, matching the original VRPModel.forward() interface."""

    def __init__(self, embed_dim: int = 128, decoder_layer_num: int = 6,
                 head_num: int = 8, qkv_dim: int = 16, ff_hidden: int = 512):
        super().__init__()
        self.encoder = LEHDEncoder(embed_dim, head_num, qkv_dim, ff_hidden)
        self.decoder = LEHDDecoder(embed_dim, decoder_layer_num, head_num, qkv_dim, ff_hidden)
        self._encoded: torch.Tensor | None = None

    def encode(self, problems: torch.Tensor, capacity: float) -> torch.Tensor:
        self._encoded = self.encoder(problems, capacity)
        return self._encoded

    def decode_step(
        self,
        problems: torch.Tensor,
        selected_node_list: torch.Tensor,
        capacity: float,
        remaining_capacity: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        """Run decoder; re-encode at every step in training mode to support per-step backprop."""
        if self.training:
            encoded = self.encode(problems, capacity)
            return self.decoder(encoded, selected_node_list, capacity, remaining_capacity)
        else:
            if current_step <= 1:
                self.encode(problems, capacity)
            assert self._encoded is not None
            return self.decoder(self._encoded, selected_node_list, capacity, remaining_capacity)
