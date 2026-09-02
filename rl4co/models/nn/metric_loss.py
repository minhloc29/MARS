"""
M3 — MetricPreservationLoss (METRA-style) for Metric-Aware Slot NCO.

Implements the core novelty from the proposal:

    L_metric = -E[||phi(z_k) - phi(z_l)||]
              + lambda * E[max(0, ||phi(z_k) - phi(z_l)|| - D_target(k, l))^2]

where:
  - phi  : a small MLP projection head (projects slot embeddings to metric space)
  - D_target(k, l) : target inter-region distance (depends on chosen variant)
  - lambda : Lagrange multiplier updated via dual ascent (separate param group)

Target D_target per ablation variant:
  Variant A: reconstruction loss (handled separately in model.py)
  Variant B: no metric loss (alpha = 0), handled in model.py
  Variant C: Euclidean distance between slot-centroid coordinates
  Variant D: insertion-cost distance (proposed) — aggregated sparse d_ins over A_ik
  # Variant E: future-regret distance — RESERVED, not implemented yet

Sparse d_ins format (B3):
  d_ins is stored and passed as two tensors:
    d_ins_idx: (B, N, k) int16  — k-NN neighbor indices (cast to long before use)
    d_ins_val: (B, N, k) float32 — insertion costs to those neighbors

  The aggregation _aggregate_d_ins_sparse computes D_ins(k,l) without
  materialising the dense (B, N, N) matrix.

Reference for dual ascent mechanism:
    METRA, Park et al. (ICLR 2024) — iod/metra.py dual_reg logic
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────────
# Projection head  phi(z_k)
# ────────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Small MLP mapping slot embeddings to the metric space.

    Args:
        input_dim (int): Dimension of slot embeddings (same as encoder embed_dim).
        proj_dim (int): Output projection dimension (default 64).
        hidden_dim (int): Hidden layer size (default 256).
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, K, d) slot embeddings
        Returns:
            (B, K, proj_dim) projected embeddings
        """
        return self.net(z)


# ────────────────────────────────────────────────────────────────────────────────
# Target distance aggregators
# ────────────────────────────────────────────────────────────────────────────────

# def _aggregate_d_ins_sparse(
#     d_ins_idx: torch.Tensor,   # (B, N, k) int16 on disk, must be cast to long
#     d_ins_val: torch.Tensor,   # (B, N, k) float32
#     A_ik: torch.Tensor,        # (B, N, K) float32
# ) -> torch.Tensor:
#     """
#     Aggregate sparse node-level insertion cost to region-level D_ins(slot_k, slot_l).

#     Computes:
#         D_ins(k, l) = sum_{i} sum_{j in kNN(i)} A_ik[i] * A_jl[j] * d_ins(i, j)

#     This is equivalent to A^T @ d_ins_dense @ A but without materialising
#     the dense (B, N, N) matrix.

#     Args:
#         d_ins_idx: (B, N, k) int16 — k-NN neighbor indices. Cast to long before gather.
#         d_ins_val: (B, N, k) float32 — insertion costs to k neighbors.
#         A_ik:      (B, N, K) float32 — soft slot assignment matrix.

#     Returns:
#         D_ins: (B, K, K) float32 — aggregated region-level insertion cost.
#     """
#     B, N, k = d_ins_val.shape
#     K = A_ik.shape[2]

#     # Cast index to long (int16 stored on disk, int64 required by gather)
#     idx = d_ins_idx.long()  # (B, N, k)

#     # Gather slot assignments for each neighbor j: A_jl shape (B, N, k, K)
#     # idx: (B, N, k) → expand to (B, N, k, K) for gather along dim=1
#     idx_expanded = idx.unsqueeze(-1).expand(B, N, k, K)   # (B, N, k, K)
#     A_neighbors = torch.gather(
#         A_ik.unsqueeze(2).expand(B, N, k, K),  # (B, N, k, K) — broadcast A_ik
#         dim=1,
#         index=idx_expanded,
#     )  # (B, N, k, K) — A_jl for each neighbor j of each node i

#     # Weighted contribution: d_ins(i,j) * A_jl  → (B, N, k, K)
#     w = d_ins_val.unsqueeze(-1) * A_neighbors  # (B, N, k, K)

#     # Sum over k neighbors → (B, N, K): total weighted slot-assignment per node i
#     node_weighted = w.sum(dim=2)  # (B, N, K)

#     # Final: A^T @ node_weighted → (B, K, K)
#     D_ins = torch.bmm(A_ik.transpose(1, 2), node_weighted)  # (B, K, K)
#     return D_ins

#NEW: normalization, let's see
def _aggregate_d_ins_sparse(d_ins_idx, d_ins_val, A_ik, normalize=True, symmetrize=True, eps=1e-8):
    B, N, k = d_ins_val.shape
    K = A_ik.shape[2]
    idx = d_ins_idx.long()
    idx_expanded = idx.unsqueeze(-1).expand(B, N, k, K)
    A_neighbors = torch.gather(
        A_ik.unsqueeze(2).expand(B, N, k, K), dim=1, index=idx_expanded,
    )
    w = d_ins_val.unsqueeze(-1) * A_neighbors
    node_weighted = w.sum(dim=2)
    D_ins = torch.bmm(A_ik.transpose(1, 2), node_weighted)  # (B, K, K) -- raw, asymmetric

    if symmetrize:
        D_ins = 0.5 * (D_ins + D_ins.transpose(-1, -2))      # now a well-defined dissimilarity

    if normalize:
        mass = A_ik.sum(dim=1)                                # (B, K)
        D_ins = D_ins / (mass.unsqueeze(-1) * mass.unsqueeze(-2) + eps)

    return D_ins


def _euclidean_target(
    locs: torch.Tensor,
    A_ik: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Euclidean centroid distance as target for Variant C.

    Centroid of slot k: c_k = sum_i A_ik * x_i  (weighted mean of node coords)
    Target: D_euclid(k, l) = ||c_k - c_l||_2

    Args:
        locs:  (B, N, 2)
        A_ik:  (B, N, K)

    Returns:
        D_euclid: (B, K, K)
    """
    # Normalise assignments to sum-to-1 across nodes
    A_norm = A_ik / (A_ik.sum(dim=1, keepdim=True) + 1e-8)  # (B, N, K)
    # Centroid: (B, K, 2)
    centroids = torch.einsum("bnk,bnc->bkc", A_norm, locs)
    # Pairwise centroid distances: (B, K, K)
    diff = centroids.unsqueeze(2) - centroids.unsqueeze(1)  # (B, K, K, 2)
    return torch.norm(diff, p=2, dim=-1)


# ────────────────────────────────────────────────────────────────────────────────
# MetricPreservationLoss
# ────────────────────────────────────────────────────────────────────────────────

class MetricPreservationLoss(nn.Module):
    """
    METRA-style lower-bound metric preservation loss with dual ascent.

    L_metric = -E[||phi(z_k) - phi(z_l)||]
              + lambda * E[max(0, ||phi(z_k) - phi(z_l)|| - D_target(k,l))^2]

    log_lambda is a learnable parameter placed in its own optimizer param group
    with lr=lr_dual (configured in POMOSlot.configure_optimizers) so that it
    receives dual-ascent updates independent of the main model lr/scheduler.

    Args:
        proj_head (ProjectionHead): Shared phi projection head.
        variant (str): One of 'C' (Euclidean), 'D' (insertion cost).
        lambda_init (float): Initial value for the Lagrange multiplier.
        lr_dual (float): Stored for reference; actual lr is set via param group
            in POMOSlot.configure_optimizers.
        sample_pairs (int | None): If set, subsample this many (k, l) pairs
            per batch to reduce memory for large K. None = all K^2 pairs.
    """

    # E excluded: future-regret target not yet implemented
    SUPPORTED_VARIANTS = {"C", "D"}

    def __init__(
        self,
        proj_head: ProjectionHead,
        variant: str = "D",
        lambda_init: float = 1.0,
        lr_dual: float = 1e-3,
        sample_pairs: int | None = None,
        normalize_target: bool = True,
        symmetrize_target: bool = True,
    ) -> None:
        super().__init__()

        assert variant in self.SUPPORTED_VARIANTS, (
            f"variant must be one of {self.SUPPORTED_VARIANTS}, got '{variant}'. "
            f"Note: Variant E (future-regret) is reserved but not yet implemented."
        )
        self.proj_head = proj_head
        self.variant = variant
        self.lr_dual = lr_dual   # reference only; actual lr set via param group
        self.sample_pairs = sample_pairs
        self.normalize_target = normalize_target
        self.symmetrize_target = symmetrize_target

        # Lagrange multiplier — stored as log for positivity constraint
        self.log_lambda = nn.Parameter(
            torch.tensor(lambda_init).log(), requires_grad=True
        )

    def extra_repr(self) -> str:
        """Visibility in model summaries / logs — surfaces the target-aggregation
        config so a checkpoint trained one way and evaluated another is obvious
        immediately rather than after several eval runs."""
        return (
            f"variant={self.variant}, normalize_target={self.normalize_target}, "
            f"symmetrize_target={self.symmetrize_target}"
        )

    @property
    def lmbda(self) -> torch.Tensor:
        """Always positive lambda via exp(log_lambda)."""
        return self.log_lambda.exp()

    def _get_target(
        self,
        A_ik: torch.Tensor,
        locs: torch.Tensor | None,
        d_ins_idx: torch.Tensor | None,
        d_ins_val: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dispatch to appropriate target computation based on variant."""
        if self.variant == "C":
            assert locs is not None, "locs required for Variant C"
            return _euclidean_target(locs, A_ik)          # (B, K, K)
        elif self.variant == "D":
            assert d_ins_idx is not None and d_ins_val is not None, \
                "d_ins_idx and d_ins_val required for Variant D"
            return _aggregate_d_ins_sparse(
                d_ins_idx,
                d_ins_val,
                A_ik,
                normalize=self.normalize_target,
                symmetrize=self.symmetrize_target,
            )  # (B, K, K)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def forward(
        self,
        slots: torch.Tensor,
        A_ik: torch.Tensor,
        locs: torch.Tensor | None = None,
        d_ins_idx: torch.Tensor | None = None,
        d_ins_val: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the metric preservation loss.

        Args:
            slots:     (B, K, d)   — slot embeddings from SlotAttention
            A_ik:      (B, N, K)   — soft assignment matrix
            locs:      (B, N, 2)   — node coordinates (needed for Variant C)
            d_ins_idx: (B, N, k) int16 — k-NN indices (needed for Variant D)
            d_ins_val: (B, N, k) float32 — k-NN insertion costs (needed for Variant D)

        Returns:
            loss (scalar), info_dict (for logging)
        """
        B, K, _ = slots.shape

        # Project slots to metric space
        z_proj = self.proj_head(slots)                   # (B, K, proj_dim)

        # Compute pairwise latent distances: (B, K, K)
        diff = z_proj.unsqueeze(2) - z_proj.unsqueeze(1)
        latent_dist = torch.norm(diff, p=2, dim=-1)      # (B, K, K)

        # Compute target distances (detached — target should not backprop)
        D_target = self._get_target(A_ik, locs, d_ins_idx, d_ins_val).detach()

        # Off-diagonal mask (exclude self-pairs k==l)
        off_diag = ~torch.eye(K, dtype=torch.bool, device=slots.device)
        off_diag = off_diag.unsqueeze(0).expand(B, -1, -1)

        latent_off = latent_dist[off_diag]               # (B * K*(K-1),)
        target_off = D_target[off_diag]

        # Optional pair subsampling for large K
        if self.sample_pairs is not None and latent_off.numel() > self.sample_pairs:
            idx = torch.randperm(latent_off.numel(), device=slots.device)[: self.sample_pairs]
            latent_off = latent_off[idx]
            target_off = target_off[idx]

        # ── Spread term: maximise latent distances (negative mean) ────────
        spread_loss = -latent_off.mean()

        # ── Constraint violation: ||phi(z_k)-phi(z_l)|| must NOT exceed D_target
        violation = torch.clamp(latent_off - target_off, min=0.0)
        penalty = (violation ** 2).mean()

        # ── Combined loss (lmbda detached for policy gradient step) ───────
        loss = spread_loss + self.lmbda.detach() * penalty

        # ── Dual loss: gradient ascent on lambda (via negative penalty) ───
        # log_lambda is in its own param group with lr=lr_dual
        # Maximise lambda * penalty <=> minimise -lambda * penalty
        dual_loss = -self.lmbda * penalty.detach()

        info = {
            "metric_spread":         spread_loss.detach(),
            "metric_penalty":        penalty.detach(),
            "metric_lambda":         self.lmbda.detach(),
            "metric_violation_mean": violation.mean().detach(),
        }

        return loss + dual_loss, info


# ────────────────────────────────────────────────────────────────────────────────
# Slot Entropy Regulariser
# ────────────────────────────────────────────────────────────────────────────────

class SlotEntropyLoss(nn.Module):
    """
    Entropy regulariser on slot assignments to prevent slot collapse.

    Maximises the entropy of the marginal assignment distribution
    p(k) = mean_i A_ik — encourages all slots to be used equally.

    L_entropy = -H[p(k)] = sum_k p(k) * log(p(k))   (to be minimised)
    """

    def forward(self, A_ik: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_ik: (B, N, K) soft assignment matrix (softmax-normalised across K)
        Returns:
            scalar entropy loss (minimise to maximise slot entropy)
        """
        p_k = A_ik.mean(dim=1)           # (B, K) marginal distribution
        p_k = p_k.clamp(min=1e-8)
        return (p_k * p_k.log()).sum(dim=-1).mean()
