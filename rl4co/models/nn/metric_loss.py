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


class EuclideanGeometryLoss(nn.Module):
    """Match projected node-embedding distances to spatial distances."""

    def __init__(self, input_dim: int, proj_dim: int = 64,
                 sample_pairs: int = 1024) -> None:
        super().__init__()
        self.proj_head = ProjectionHead(input_dim, proj_dim)
        self.sample_pairs = sample_pairs

    def forward(self, node_embeddings: torch.Tensor,
                locs: torch.Tensor,
                d_ins_idx: torch.Tensor | None = None,
                d_ins_val: torch.Tensor | None = None) -> torch.Tensor:
        _, num_nodes, _ = node_embeddings.shape
        if d_ins_idx is not None and d_ins_val is not None:
            pair_count = min(self.sample_pairs,
                             d_ins_idx.shape[1] * d_ins_idx.shape[2])
            first = torch.randint(num_nodes, (pair_count,), device=locs.device)
            edge = torch.randint(
                d_ins_idx.shape[2], (pair_count,), device=locs.device)
            second = d_ins_idx[:, first, edge].long()
            target_dist = d_ins_val[:, first, edge].to(locs.dtype)
            projected_second = None
        else:
            pair_count = min(self.sample_pairs, num_nodes * num_nodes)
            first = torch.randint(num_nodes, (pair_count,), device=locs.device)
            second = torch.randint(
                num_nodes, (pair_count,), device=locs.device)
            target_dist = (locs[:, first] - locs[:, second]).norm(dim=-1)
            projected_second = second

        projected = self.proj_head(node_embeddings)
        if projected_second is not None:
            second_embeddings = projected[:, projected_second]
        else:
            second_embeddings = projected.gather(
                1, second.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
            )
        latent_dist = (projected[:, first] - second_embeddings).norm(dim=-1)
        latent_dist = latent_dist / latent_dist.detach().amax(dim=1,
                                                              keepdim=True).clamp_min(1e-6)
        target_dist = target_dist / \
            target_dist.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return F.mse_loss(latent_dist, target_dist)


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

# NEW: normalization, let's see
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
    # (B, K, K) -- raw, asymmetric
    D_ins = torch.bmm(A_ik.transpose(1, 2), node_weighted)

    if symmetrize:
        # now a well-defined dissimilarity
        D_ins = 0.5 * (D_ins + D_ins.transpose(-1, -2))

    if normalize:
        mass = A_ik.sum(dim=1)                                # (B, K)
        D_ins = D_ins / (mass.unsqueeze(-1) * mass.unsqueeze(-2) + eps)

    return D_ins


def _euclidean_aggregate(
    locs: torch.Tensor,
    A_ik: torch.Tensor,
    normalize: bool = True,
    symmetrize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Bilinear aggregation of spatial Euclidean distance (LEGO-Euclidean variant).

        D_euc(k, l) = sum_i sum_j A_ik * A_jl * ||x_i - x_j||_2   =  A^T d_euc A

    This is the direct analog of Variant D's construction-aggregation, but with
    the ordinary spatial distance d_euc(i,j) = ||x_i - x_j||_2 in place of the
    route-induced d_cons. Everything else (K, A, phi, spread, lambda, entropy) is
    unchanged — this is the pure routing-aware-vs-spatial control: whether the
    *routing-aware* target is more useful than plain spatial proximity.

    Args:
        locs: (B, N, 2) customer coordinates.
        A_ik: (B, N, K) soft slot assignment matrix.
        normalize (bool): Divide by the slot mass product (c_k * c_l) — matches
            how Variant D's aggregate is normalised so the two differ only in
            the underlying node-level distance d_cons vs d_euc.
        symmetrize (bool): Average with its transpose (d_euc already symmetric,
            so this is a no-op kept for parity with Variant D's pipeline).

    Returns:
        D_euc: (B, K, K) bilinearly-aggregated spatial distance.
    """
    B, N, K = A_ik.shape
    # Pairwise spatial distance: d_euc(i,j) = ||x_i - x_j||_2 -> (B, N, N)
    diff = locs.unsqueeze(2) - locs.unsqueeze(1)   # (B, N, N, 2)
    d_euc = torch.norm(diff, p=2, dim=-1)          # (B, N, N)
    # Bilinear aggregation: A^T d A  -> (B, K, K)
    D_euc = torch.bmm(A_ik.transpose(1, 2), torch.bmm(d_euc, A_ik))

    if symmetrize:
        D_euc = 0.5 * (D_euc + D_euc.transpose(-1, -2))

    if normalize:
        mass = A_ik.sum(dim=1)                     # (B, K)
        D_euc = D_euc / (mass.unsqueeze(-1) * mass.unsqueeze(-2) + eps)

    return D_euc


def _construction_aggregate(
    d_cons: torch.Tensor,
    A_ik: torch.Tensor,
    normalize: bool = True,
    symmetrize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Bilinear aggregation of a dense node-level construction cost d_cons.

        D_target(k, l) = sum_i sum_j A_ik * A_jl * d_cons(i,j)  =  A^T d_cons A

    Mirrors :func:`_euclidean_aggregate` so the shuffled-ablation target and the
    Variant-E spatial target share the same aggregation/normalisation — the only
    thing that differs between geom modes is the node-level d_cons fed in.

    Args:
        d_cons: (B, N, N) dense node-level construction/arc distance.
        A_ik:   (B, N, K) soft slot assignment matrix.
        normalize (bool): Divide by the slot mass product (c_k * c_l).
        symmetrize (bool): Average with its transpose (d_cons is symmetric, kept
            for parity with the other aggregators).

    Returns:
        D_target: (B, K, K) bilinearly-aggregated target distance.
    """
    B, N, K = A_ik.shape
    D = torch.bmm(A_ik.transpose(1, 2), torch.bmm(d_cons, A_ik))  # (B, K, K)

    if symmetrize:
        D = 0.5 * (D + D.transpose(-1, -2))

    if normalize:
        mass = A_ik.sum(dim=1)                     # (B, K)
        D = D / (mass.unsqueeze(-1) * mass.unsqueeze(-2) + eps)

    return D


def _construction_d_cons(
    locs: torch.Tensor,
    depot: torch.Tensor,
    shuffled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return node-level cycle distances and route positions.

    The optional permutation changes only the customer-to-position mapping;
    the position multiset and cycle length remain unchanged.
    """
    from rl4co.data.insertion_cost import _build_nn_route_order

    batch_size, num_nodes, _ = locs.shape
    _, prefpos, total = _build_nn_route_order(locs, depot)
    positions = prefpos
    if shuffled:
        permutation = torch.stack([
            torch.randperm(num_nodes, device=locs.device)
            for _ in range(batch_size)
        ], dim=0)
        positions = torch.gather(prefpos, 1, permutation)

    difference = (positions.unsqueeze(2) - positions.unsqueeze(1)).abs()
    d_cons = torch.minimum(
        difference, total[:, None, None] - difference)
    d_cons = torch.clamp(d_cons, min=0.0)
    diagonal = torch.eye(
        num_nodes, dtype=torch.bool, device=locs.device).unsqueeze(0)
    d_cons = d_cons.masked_fill(diagonal, 0.0)
    return d_cons, prefpos, positions


def _construction_target(
    locs: torch.Tensor,
    depot: torch.Tensor,
    A_ik: torch.Tensor,
    normalize: bool = True,
    symmetrize: bool = True,
    shuffled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate NN-tour cycle distances and return positions for diagnostics."""
    d_cons, prefpos, positions = _construction_d_cons(
        locs, depot, shuffled=shuffled)
    target = _construction_aggregate(
        d_cons, A_ik, normalize=normalize, symmetrize=symmetrize)
    return target, prefpos, positions


def _shuffled_construction_target(
    locs: torch.Tensor,
    depot: torch.Tensor,
    A_ik: torch.Tensor,
    normalize: bool = True,
    symmetrize: bool = True,
) -> torch.Tensor:
    """
    Shuffled-slots ablation target for ``geom_mode="shuffled"``.

    Builds the deterministic nearest-neighbour tour and cumulative along-tour
    positions ``p_i`` exactly as ``full`` mode, then randomly permutes ``p``
    *across customers within each instance* (fresh permutation every call, i.e.
    every instance and every training step). The node-level construction arc
    cost is then computed from the shuffled positions and aggregated with A into
    D_target exactly as in ``full`` mode. This breaks the coupling between the
    route geometry and the spatial location of each slot, isolating whether the
    policy benefits from *any* route-aware target vs the specific NN-tour one.

    Args:
        locs:  (B, N, 2) customer coordinates (matches ``A_ik`` rows).
        depot: (B, 1, 2) depot coordinates used to build the NN tour.
        A_ik:  (B, N, K) soft slot assignment over the N customer nodes.

    Returns:
        D_target: (B, K, K) aggregated target from the shuffled p values.
    """
    target, _, _ = _construction_target(
        locs, depot, A_ik, normalize=normalize,
        symmetrize=symmetrize, shuffled=True)
    return target


def _spearman(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation between two 1-D tensors.

    Equivalent to Pearson correlation on rank transforms. Returns a scalar.
    """
    pred = pred.detach().float()
    target = target.detach().float()
    pred_rank = pred.argsort().argsort().float()
    target_rank = target.argsort().argsort().float()
    pred_rank = pred_rank - pred_rank.mean()
    target_rank = target_rank - target_rank.mean()
    denom = pred_rank.norm() * target_rank.norm()
    if denom < 1e-12:
        return torch.zeros((), device=pred.device)
    return (pred_rank * target_rank).sum() / denom


def _geometry_diagnostics(
    latent_dist: torch.Tensor,
    target_dist: torch.Tensor,
    A_ik: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-instance geometry statistics over off-diagonal pairs."""
    batch_size, num_slots, _ = latent_dist.shape
    off_diag = ~torch.eye(num_slots, dtype=torch.bool,
                          device=latent_dist.device)
    latent_off = latent_dist[:, off_diag].view(batch_size, -1)
    target_off = target_dist[:, off_diag].view(batch_size, -1)
    active = latent_off > target_off
    spearman = torch.stack([
        _spearman(latent_off[index], target_off[index])
        for index in range(batch_size)
    ])
    stats = {
        "metric_D_latent_mean": latent_off.mean(),
        "metric_D_target_mean": target_off.mean(),
        "metric_D_latent_median": latent_off.median(dim=1).values.mean(),
        "metric_D_target_median": target_off.median(dim=1).values.mean(),
        "metric_ratio": (latent_off / target_off.clamp_min(1e-8)).mean(),
        "metric_hinge_active_frac": active.float().mean(),
        "metric_spearman": spearman.mean(),
    }
    if A_ik is not None:
        marginal = A_ik.mean(dim=1).clamp_min(1e-8)
        entropy = -(marginal * marginal.log()).sum(dim=-1)
        stats["slot_entropy_normalized"] = (
            entropy /
            torch.log(torch.tensor(float(num_slots), device=entropy.device))
        ).mean()
        stats["slot_max_assignment_mean"] = A_ik.max(dim=-1).values.mean()
    return stats


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
        variant (str): One of 'C' (Euclidean centroid), 'D' (insertion cost),
            'E' (bilinear spatial Euclidean).
            'E' is LEGO-Euclidean: same bilinear A^T d A aggregation as 'D' but
            with the ordinary spatial distance d_euc(i,j)=||x_i-x_j||_2 instead
            of the route-induced d_cons — the routing-aware-vs-spatial control.
        lambda_init (float): Initial value for the Lagrange multiplier.
        lr_dual (float): Stored for reference; actual lr is set via param group
            in POMOSlot.configure_optimizers.
        sample_pairs (int | None): If set, subsample this many (k, l) pairs
            per batch to reduce memory for large K. None = all K^2 pairs.
        geom_mode (str): One of:
            - "full":       L_geom = L_spread + lambda * penalty (current behavior).
            - "spread_only": L_geom = L_spread only. No hinge/violation term, no
              dual variable lambda / optimizer param group; D_target and the NN
              tour are skipped. L_spread is unbounded, so log mean D_latent each
              epoch and optionally clip distances (`spread_cap`).
            - "shuffled":   Same NN tour + cumulative p_i as "full", but p is
              randomly permuted across customers per instance/step before the
              short-arc d_cons is computed. Target aggregated with A exactly as
              "full"; hinge + dual kept as is.
        spread_cap (float | None): If set, clip the latent distances used by
            L_spread to `max=spread_cap` (bounds the unbounded spread term).
            Off by default.
    """

    # E excluded: future-regret target not yet implemented. E is LEGO-Euclidean.
    SUPPORTED_VARIANTS = {"C", "D", "E"}
    SUPPORTED_GEOM_MODES = {"full", "spread_only", "shuffled"}

    def __init__(
        self,
        proj_head: ProjectionHead,
        variant: str = "D",
        lambda_init: float = 1.0,
        lr_dual: float = 1e-3,
        sample_pairs: int | None = None,
        normalize_target: bool = True,
        symmetrize_target: bool = True,
        geom_mode: str = "full",
        spread_cap: float | None = None,
    ) -> None:
        super().__init__()

        assert variant in self.SUPPORTED_VARIANTS, (
            f"variant must be one of {self.SUPPORTED_VARIANTS}, got '{variant}'. "
            f"Note: Variant E (future-regret) is reserved but not yet implemented."
        )
        assert geom_mode in self.SUPPORTED_GEOM_MODES, (
            f"geom_mode must be one of {self.SUPPORTED_GEOM_MODES}, got '{geom_mode}'"
        )
        self.proj_head = proj_head
        self.variant = variant
        self.lr_dual = lr_dual   # reference only; actual lr set via param group
        self.sample_pairs = sample_pairs
        self.normalize_target = normalize_target
        self.symmetrize_target = symmetrize_target
        self.geom_mode = geom_mode
        self.spread_cap = spread_cap

        # Lagrange multiplier — stored as log for positivity constraint.
        # Not created in spread_only mode (L_geom = L_spread only; no hinge, no
        # dual variable, no dual optimizer param group).
        if geom_mode != "spread_only":
            self.log_lambda = nn.Parameter(
                torch.tensor(lambda_init).log(), requires_grad=True
            )

    def extra_repr(self) -> str:
        """Visibility in model summaries / logs — surfaces the target-aggregation
        config so a checkpoint trained one way and evaluated another is obvious
        immediately rather than after several eval runs."""
        return (
            f"variant={self.variant}, geom_mode={self.geom_mode}, "
            f"spread_cap={self.spread_cap}, "
            f"normalize_target={self.normalize_target}, "
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
        elif self.variant == "E":
            assert locs is not None, "locs required for Variant E"
            return _euclidean_aggregate(
                locs,
                A_ik,
                normalize=self.normalize_target,
                symmetrize=self.symmetrize_target,
            )  # (B, K, K)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def target_for_logging(
        self,
        A_ik: torch.Tensor,
        locs: torch.Tensor,
        depot: torch.Tensor,
        d_ins_idx: torch.Tensor | None = None,
        d_ins_val: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a detached comparable target for train/validation logging."""
        if self.geom_mode == "spread_only":
            target, _, _ = _construction_target(
                locs, depot, A_ik,
                normalize=self.normalize_target,
                symmetrize=self.symmetrize_target,
            )
            return target.detach()
        if self.geom_mode == "shuffled":
            return _shuffled_construction_target(
                locs, depot, A_ik,
                normalize=self.normalize_target,
                symmetrize=self.symmetrize_target,
            ).detach()
        return self._get_target(
            A_ik, locs, d_ins_idx, d_ins_val).detach()

    @torch.no_grad()
    def diagnostics(
        self,
        slots: torch.Tensor,
        A_ik: torch.Tensor,
        locs: torch.Tensor,
        depot: torch.Tensor,
        d_ins_idx: torch.Tensor | None = None,
        d_ins_val: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        projected = self.proj_head(slots)
        latent_dist = torch.cdist(projected, projected, p=2)
        target_dist = self.target_for_logging(
            A_ik, locs, depot, d_ins_idx, d_ins_val)
        stats = _geometry_diagnostics(latent_dist, target_dist, A_ik)
        stats["metric_lambda"] = (
            self.lmbda.detach()
            if self.geom_mode != "spread_only"
            else torch.zeros((), device=slots.device)
        )
        stats["metric_L_viol"] = torch.clamp(
            latent_dist - target_dist, min=0.0
        ).pow(2).mean()
        return stats

    def forward(
        self,
        slots: torch.Tensor,
        A_ik: torch.Tensor,
        locs: torch.Tensor | None = None,
        d_ins_idx: torch.Tensor | None = None,
        d_ins_val: torch.Tensor | None = None,
        depot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the metric preservation loss (geometry branch, training only).

        Args:
            slots:     (B, K, d)   — slot embeddings from SlotAttention
            A_ik:      (B, N, K)   — soft assignment matrix
            locs:      (B, N, 2)   — customer coordinates (needed for Variant C,
                                     E, and shuffled geom_mode).
            d_ins_idx: (B, N, k) int16 — k-NN indices (needed for Variant D full).
            d_ins_val: (B, N, k) float32 — k-NN insertion costs (needed for
                                     Variant D full).
            depot:     (B, 1, 2) (or (B, 2)) — depot coordinates (needed for
                                     geom_mode="shuffled" to build the NN tour).

        Returns:
            loss (scalar), info_dict (for logging)
        """
        B, K, _ = slots.shape

        # Project slots to metric space
        z_proj = self.proj_head(slots)                   # (B, K, proj_dim)

        # Compute pairwise latent distances: (B, K, K)
        diff = z_proj.unsqueeze(2) - z_proj.unsqueeze(1)
        latent_dist = torch.norm(diff, p=2, dim=-1)      # (B, K, K)

        # Off-diagonal mask (exclude self-pairs k==l)
        off_diag = ~torch.eye(K, dtype=torch.bool, device=slots.device)
        off_diag = off_diag.unsqueeze(0).expand(B, -1, -1)

        latent_off = latent_dist[off_diag]               # (B * K*(K-1),)

        # Target distance (full / shuffled only). Detached — should not backprop.
        D_target = None
        if self.geom_mode == "spread_only":
            pass
        elif self.geom_mode == "shuffled":
            assert locs is not None and depot is not None, \
                "locs and depot required for geom_mode='shuffled'"
            D_target = _shuffled_construction_target(
                locs, depot, A_ik,
                normalize=self.normalize_target,
                symmetrize=self.symmetrize_target,
            )                                              # (B, K, K)
        else:  # "full"
            D_target = self._get_target(
                A_ik, locs, d_ins_idx, d_ins_val).detach()
        target_off = D_target[off_diag] if D_target is not None else None

        # Optional pair subsampling for large K — subsample latent and target
        # with the SAME index so the (latent, target) pairing stays intact.
        if self.sample_pairs is not None and latent_off.numel() > self.sample_pairs:
            idx = torch.randperm(latent_off.numel(), device=slots.device)[
                : self.sample_pairs]
            latent_off = latent_off[idx]
            if target_off is not None:
                target_off = target_off[idx]

        # Optional spread cap: clip latent distances so the unbounded spread
        # term cannot run away. Off by default.
        spread_off = latent_off
        if self.spread_cap is not None:
            spread_off = torch.clamp(spread_off, max=self.spread_cap)

        # ── Spread term: maximise latent distances (negative mean) ────────
        spread_loss = -spread_off.mean()
        info = {
            "metric_spread": spread_loss.detach(),
            # Log mean latent distance (unbounded in spread_only) for monitoring.
            "metric_D_latent_mean": latent_off.mean().detach(),
        }

        # ── spread_only: L_geom = L_spread only (no hinge, dual, or target) ──
        if self.geom_mode == "spread_only":
            info.update({
                "metric_lambda": torch.zeros((), device=slots.device),
                "metric_L_viol": torch.zeros((), device=slots.device),
                "metric_hinge_active_frac": torch.zeros((), device=slots.device),
            })
            return spread_loss, info

        # ── Constraint violation: ||phi(z_k)-phi(z_l)|| must NOT exceed D_target
        violation = torch.clamp(latent_off - target_off, min=0.0)
        penalty = (violation ** 2).mean()
        hinge_active_frac = (violation > 0).float().mean()

        # ── Combined loss (lmbda detached for policy gradient step) ───────
        loss = spread_loss + self.lmbda.detach() * penalty

        # ── Dual loss: gradient ascent on lambda (via negative penalty) ───
        # log_lambda is in its own param group with lr=lr_dual
        # Maximise lambda * penalty <=> minimise -lambda * penalty
        dual_loss = -self.lmbda * penalty.detach()

        info.update({
            "metric_penalty":        penalty.detach(),
            "metric_L_viol":         penalty.detach(),
            "metric_lambda":         self.lmbda.detach(),
            "metric_violation_mean": violation.mean().detach(),
            "metric_hinge_active_frac": hinge_active_frac.detach(),
        })
        info.update(_geometry_diagnostics(
            latent_dist, D_target, A_ik))

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
