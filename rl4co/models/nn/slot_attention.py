"""
M2 — SlotAttention module for Metric-Aware NCO.

Adapted from the reference implementation in
    references/slot-attention/slot_attention/slot_attention.py
(Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020)

Key differences from the vision version:
  - Returns BOTH slot embeddings z_k AND soft assignment matrix A_ik,
    which is required for aggregating D_ins(k, l) in the metric loss.
  - Deterministic slot initialisation option (useful for val/test).
  - Gradient-friendly: attn is normalised across slots (softmax dim=1)
    then across nodes (l1-norm dim=2) — matching the original paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _l1norm(t: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(t + eps, p=1, dim=dim)


# ────────────────────────────────────────────────────────────────────────────────
# SlotAttention
# ────────────────────────────────────────────────────────────────────────────────

class SlotAttention(nn.Module):
    """
    Iterative Slot Attention applied to graph node embeddings for routing.

    Args:
        num_slots (int): K — number of routing region embeddings.
        dim (int): Feature dimension of input node embeddings h_i.
        iters (int): Number of iterative refinement steps (default 3).
        eps (float): Small value for numerical stability in L1-norm.
        hidden_dim (int): Hidden dim of the feed-forward MLP after each iter.
        deterministic_init (bool): If True, use learned mean only (no noise)
            for slot initialisation. Useful for reproducible val/test passes.

    Forward:
        inputs: (B, N, dim)  — node embeddings from the backbone encoder.
        Returns:
            slots:   (B, K, dim)   — region embeddings z_k
            attn:    (B, N, K)     — soft assignment matrix A_ik
                                      (rows = nodes, cols = slots)
    """

    def __init__(
        self,
        num_slots: int,
        dim: int | None = None,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        deterministic_init: bool = False,
        # Backward-compatible aliases
        slot_dim: int | None = None,
        in_dim: int | None = None,
        num_iterations: int | None = None,
    ) -> None:
        super().__init__()

        # Resolve dimension aliases
        if dim is None and slot_dim is not None:
            dim = slot_dim
        elif dim is None:
            raise ValueError("Must provide 'dim' or 'slot_dim'")
        # Resolve iters alias
        if num_iterations is not None:
            iters = num_iterations
        # Input dimension (may differ from slot dim)
        in_dim = in_dim if in_dim is not None else dim

        self.num_slots = num_slots
        self.dim = dim
        self.in_dim = in_dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.deterministic_init = deterministic_init

        # Learnable slot initialisers (shared across batch)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        # Optional input projection if in_dim != slot dim
        self.input_proj = nn.Linear(in_dim, dim, bias=False) if in_dim != dim else nn.Identity()

        # Cross-attention projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Recurrent update (GRU)
        self.gru = nn.GRUCell(dim, dim)

        # Feed-forward after each iteration
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

    # ────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        inputs: torch.Tensor,
        num_slots: int | None = None,
        init_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs:    (B, N, dim)
            num_slots: Override self.num_slots at runtime (optional).
            init_noise: Optional standard-normal noise used to initialise the
                slots. It may have shape ``(1, K, dim)`` or ``(B, K, dim)``.
                Supplying it lets sequential decoders reuse the same distinct
                slot identities across repeated encoder calls.

        Returns:
            slots:   (B, K, dim)
            attn:    (B, N, K)   soft assignments (node-to-slot)
        """
        B, N, D = inputs.shape
        device, dtype = inputs.device, inputs.dtype
        K = num_slots if num_slots is not None else self.num_slots

        # ── Slot initialisation ──────────────────────────────────────────
        mu = self.slots_mu.expand(B, K, -1)
        if init_noise is not None:
            if init_noise.ndim != 3 or init_noise.shape[1:] != (K, self.dim):
                raise ValueError(
                    "init_noise must have shape (1 or B, K, dim); "
                    f"got {tuple(init_noise.shape)} for B={B}, K={K}, dim={self.dim}"
                )
            if init_noise.shape[0] not in (1, B):
                raise ValueError(
                    f"init_noise batch dimension must be 1 or {B}, got {init_noise.shape[0]}"
                )
            noise = init_noise.to(device=device, dtype=dtype).expand(B, -1, -1)
            sigma = self.slots_logsigma.exp().expand(B, K, -1)
            slots = mu + sigma * noise
        elif self.deterministic_init:
            slots = mu
        else:
            sigma = self.slots_logsigma.exp().expand(B, K, -1)
            slots = mu + sigma * torch.randn(B, K, self.dim, device=device, dtype=dtype)

        # ── Project inputs to slot dim if needed ────────────────────────
        inputs = self.input_proj(inputs)                 # (B, N, dim)

        # ── Pre-compute keys & values (inputs don't change across iters) ──
        normed_inputs = self.norm_input(inputs)          # (B, N, dim)
        k = self.to_k(normed_inputs)                     # (B, N, dim)
        v = self.to_v(normed_inputs)                     # (B, N, dim)

        attn = None  # will be overwritten each iteration

        # ── Iterative refinement ─────────────────────────────────────────
        for _ in range(self.iters):
            slots_prev = slots

            q = self.to_q(self.norm_slots(slots))        # (B, K, dim)

            # Attention logits: (B, K, N)
            dots = torch.einsum("bkd,bnd->bkn", q, k) * self.scale

            # Softmax across SLOTS (dim=1) → each node competes across slots
            attn = dots.softmax(dim=1)                   # (B, K, N)

            # L1-normalise across NODES (dim=2) → weighted mean aggregation
            attn_norm = _l1norm(attn, dim=2, eps=self.eps)  # (B, K, N)

            # Aggregate values: (B, K, dim)
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v)

            # GRU update (operates on flattened (B*K, dim))
            slots = self.gru(
                updates.reshape(B * K, self.dim),
                slots_prev.reshape(B * K, self.dim),
            ).reshape(B, K, self.dim)

            slots = slots + self.mlp(self.norm_pre_ff(slots))

        # attn: (B, K, N) → transpose to (B, N, K) for downstream use as A_ik
        # A_ik[b, i, k] = probability that node i is assigned to slot k
        A_ik = attn.transpose(1, 2).contiguous()        # (B, N, K)

        return slots, A_ik
