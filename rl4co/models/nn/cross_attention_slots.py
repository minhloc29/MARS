"""
Cross-attention slot aggregation module for Metric-Aware NCO.

Ablation replacing Slot Attention with standard cross-attention where
queries are learnable vectors and attention is normalised over CUSTOMERS
(keys), rather than slots competing over customers.

Modes:
  - cross (iterative=False, T=1): single-pass cross-attention + residual MLP, NO GRU.
  - cross_iter (iterative=True, T=iters): recurrent cross-attention with GRU + MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class CrossAttentionSlots(nn.Module):
    """
    Cross-attention module mapping graph node embeddings to slot/region embeddings.

    Main differences from SlotAttention:
      1. Query vectors Z0 are K learnable parameters (deterministic init, no sampling noise).
      2. Attention logits M = K Q^T / sqrt(d_a) of shape (B, N, K).
      3. Attention weights W = softmax over CUSTOMERS (dim=1) of M, so sum_{i=1}^N W_{bik} = 1.
         Aggregation: Z = W^T V of shape (B, K, D).
      4. Soft assignment A returned to downstream losses is row-normalized over slots:
         A_ik = W_ik / (sum_l W_il + eps), so rows sum to 1 (same interface and property as Slot Attention).
      5. In single-pass mode ('cross', iterative=False), uses residual MLP post-update without GRU.
         In iterative mode ('cross_iter', iterative=True), uses recurrent GRU + MLP updates.

    Args:
        num_slots (int): K - number of routing region embeddings.
        dim (int): Feature dimension of input node embeddings h_i.
        iters (int): Number of iterative refinement steps (default 1 for cross, 3 for cross_iter).
        eps (float): Small epsilon for numerical stability.
        hidden_dim (int): Hidden dimension of the feed-forward MLP.
        iterative (bool): If True, use GRU recurrent update across iterations (cross_iter).
            If False, single pass without GRU (cross).
        slot_dim (int | None): Alias for dim.
        in_dim (int | None): Input dimension if different from dim.
        num_iterations (int | None): Alias for iters.
    """

    def __init__(
        self,
        num_slots: int,
        dim: int | None = None,
        iters: int = 1,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        iterative: bool = False,
        slot_dim: int | None = None,
        in_dim: int | None = None,
        num_iterations: int | None = None,
    ) -> None:
        super().__init__()

        # Resolve aliases
        if dim is None and slot_dim is not None:
            dim = slot_dim
        elif dim is None:
            raise ValueError("Must provide 'dim' or 'slot_dim'")
        if num_iterations is not None:
            iters = num_iterations
        in_dim = in_dim if in_dim is not None else dim

        self.num_slots = num_slots
        self.dim = dim
        self.in_dim = in_dim
        self.iters = iters if iterative else 1
        self.eps = eps
        self.scale = dim ** -0.5
        self.iterative = iterative

        # K learnable slot query vectors (deterministic init)
        self.slots_init = nn.Parameter(torch.empty(num_slots, dim))
        init.xavier_uniform_(self.slots_init)

        # Optional input projection if in_dim != dim
        self.input_proj = (
            nn.Linear(in_dim, dim, bias=False) if in_dim != dim else nn.Identity()
        )

        # Projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Recurrent GRU (only used if iterative=True)
        if self.iterative:
            self.gru = nn.GRUCell(dim, dim)
        else:
            self.gru = None

        # Feed-forward MLP
        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        # Layer normalisations
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(
        self,
        inputs: torch.Tensor,
        num_slots: int | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs:    (B, N, dim) - node embeddings from backbone encoder.
            num_slots: Optional override for K.
            mask:      (B, N) optional boolean mask (True for valid nodes).

        Returns:
            slots: (B, K, dim) - slot embeddings Z
            A_ik:  (B, N, K)   - row-normalized soft assignments (rows sum to 1)
        """
        B, N, _ = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots

        # Initialize slot representations: (B, K, dim)
        if K == self.num_slots:
            slots = self.slots_init.unsqueeze(0).expand(B, K, -1)
        else:
            slots = self.slots_init[:K].unsqueeze(0).expand(B, K, -1)

        # Input projections
        inputs = self.input_proj(inputs)
        normed_inputs = self.norm_input(inputs)
        k = self.to_k(normed_inputs)  # (B, N, dim)
        v = self.to_v(normed_inputs)  # (B, N, dim)

        W = None
        A_ik = None

        iters = self.iters if self.iterative else 1
        for _ in range(iters):
            slots_prev = slots

            q = self.to_q(self.norm_slots(slots))  # (B, K, dim)

            # Logits M: (B, N, K)
            # M[b, i, k] = k_i · q_k / sqrt(dim)
            M = torch.einsum("bnd,bkd->bnk", k, q) * self.scale

            if mask is not None:
                # Mask out invalid nodes across dim 1
                mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
                M = M.masked_fill(~mask_expanded, -1e9)

            # Standard cross-attention: softmax over CUSTOMERS (dim=1)
            W = M.softmax(dim=1)  # (B, N, K), sum along dim=1 is 1

            # Aggregate values: Z = W^T V -> (B, K, dim)
            updates = torch.einsum("bnk,bnd->bkd", W, v)

            if self.iterative and self.gru is not None:
                slots = self.gru(
                    updates.reshape(B * K, self.dim),
                    slots_prev.reshape(B * K, self.dim),
                ).reshape(B, K, self.dim)
                slots = slots + self.mlp(self.norm_pre_ff(slots))
            else:
                # Single pass: residual MLP on Z
                slots = updates + self.mlp(self.norm_pre_ff(updates))

        # Row-normalize W over SLOTS (dim=2) so rows sum to 1
        # A_ik[b, i, k] = W_ik / (sum_l W_il + eps)
        A_ik = W / (W.sum(dim=-1, keepdim=True) + self.eps)

        return slots, A_ik
