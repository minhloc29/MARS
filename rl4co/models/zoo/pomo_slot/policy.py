from __future__ import annotations

import torch
import torch.nn as nn


class SlotInjectingEncoder(nn.Module):
    """Encoder wrapper: runs base encoder, then SlotAttention on customer
    embeddings and additively injects slot context back:

        hidden[:, 1:, :] += A_ik @ slots   (slot_ctx for each customer node)
        hidden[:, 0,  :] unchanged          (depot embedding unchanged)

    For no-depot problems (TSP), there is no separate depot row — node 0 is a
    real customer that must be routed, so Slot Attention runs on ALL nodes and
    the slot context is injected into every node:

        hidden += A_ik @ slots              (slot_ctx for all N nodes)

    Side-channel (after each forward, batch size B — augmentation is off
    during training so slots/d_ins always share B):
        last_slots: (B, K, d)  — slot embeddings
        last_A_ik:  (B, N, K)  — soft assignment matrix

    Args:
        base_encoder (nn.Module): Base (e.g. AttentionModel) encoder.
        slot_attn (nn.Module): SlotAttention module.
        has_depot (bool): True for CVRP (node 0 = depot, excluded from slots).
            False for TSP (all N nodes are customers).
    """

    def __init__(
        self, base_encoder: nn.Module, slot_attn: nn.Module, has_depot: bool = True
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.slot_attn = slot_attn
        self.has_depot = has_depot

        # Side-channel: populated after every forward, read by POMOSlot.shared_step
        self.last_slots: torch.Tensor | None = None
        self.last_A_ik: torch.Tensor | None = None

    def forward(self, td) -> tuple[torch.Tensor, torch.Tensor]:
        # Run base encoder
        hidden, init_embeds = self.base_encoder(td)  # (B, N+1, d) w/ depot, (B, N, d) without

        if self.has_depot:
            # Slot Attention on customer nodes only (skip depot at index 0)
            node_embs = hidden[:, 1:, :]                  # (B, N, d) — exclude depot
            slots, A_ik = self.slot_attn(node_embs)       # (B, K, d), (B, N, K)

            # Per-node slot context, injected additively; depot unchanged
            slot_ctx = torch.bmm(A_ik, slots)             # (B, N, d)
            pad_depot = torch.zeros_like(hidden[:, :1, :])  # (B, 1, d)
            hidden = hidden + torch.cat([pad_depot, slot_ctx], dim=1)  # (B, N+1, d)
        else:
            # No depot (TSP): all N nodes are customers — slot on every node.
            node_embs = hidden                            # (B, N, d)
            slots, A_ik = self.slot_attn(node_embs)       # (B, K, d), (B, N, K)
            slot_ctx = torch.bmm(A_ik, slots)             # (B, N, d)
            hidden = hidden + slot_ctx                    # (B, N, d)

        # Expose via side-channel for aux loss
        self.last_slots = slots   # (B, K, d)
        self.last_A_ik  = A_ik   # (B, N, K)

        return hidden, init_embeds
