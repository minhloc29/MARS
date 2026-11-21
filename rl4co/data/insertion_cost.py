"""
Insertion-cost utilities. Sparse (B, N, k) format for dataset storage.

Methods:
  savings      d_ins(i,j) = dist(D,i) + dist(i,j) - dist(D,j)  [CW proxy]
  construction along-tour arc distance on a greedy NN tour
  insertion    RCIC: D_RCIC(i,j) = 0.5*[Δ(i|j)+Δ(j|i)] — recommended for Variant D
               (see _compute_route_adjacent_insertion_cost)

Sparsify AFTER symmetrization so d_ins(i,j)==d_ins(j,i) before top-k selection.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def compute_pairwise_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
   
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    orig_dtype = coords.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        coords_f32 = coords.to(torch.float32)
        dist_matrix = torch.cdist(coords_f32, coords_f32, p=2.0).to(orig_dtype)
    else:
        dist_matrix = torch.cdist(coords, coords, p=2.0)

    if squeeze_batch:
        dist_matrix = dist_matrix.squeeze(0)

    return dist_matrix


def _compute_dense_d_ins(
    locs: torch.Tensor,
    depot_loc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    B, N, _ = locs.shape
    device = locs.device

    if depot_loc is None:
        depot_loc = torch.full((B, 1, 2), 0.5, device=device, dtype=locs.dtype)

    dist_customers = compute_pairwise_distance_matrix(locs)

    # Distance from depot to all customers: (B, N)
    dist_depot = torch.norm(locs - depot_loc, p=2, dim=-1)

    # Marginal insertion cost: d_ins(i, j) = dist(D,i) + dist(i,j) - dist(D,j)
    d_ins = dist_depot.unsqueeze(2) + dist_customers - dist_depot.unsqueeze(1)
    d_ins = torch.clamp(d_ins, min=0.0)

    # Symmetrize BEFORE sparsifying (avoids asymmetric kNN issues)
    d_ins = (d_ins + d_ins.transpose(-1, -2)) / 2.0

    # Zero out self-pairs
    eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    d_ins = d_ins.masked_fill(eye_mask, 0.0)

    return d_ins


def compute_marginal_insertion_cost(
    locs: torch.Tensor,
    k_neighbors: Optional[int] = 15,
    depot_loc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute dense (B, N, N) insertion cost, sparsified with inf for non-kNN pairs.

    This is kept for backward compatibility and internal use.
    For dataset storage, prefer compute_sparse_insertion_cost().

    Args:
        locs:        (B, N, 2) or (N, 2) customer coordinates.
        k_neighbors: k for kNN sparsification. None = dense (no inf mask).
        depot_loc:   Optional depot coords.

    Returns:
        d_ins: (B, N, N) or (N, N) — inf for non-kNN pairs.
    """
    if locs.dim() == 2:
        locs = locs.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    B, N, _ = locs.shape
    device = locs.device

    if depot_loc is None:
        depot_loc_3d = torch.full((B, 1, 2), 0.5, device=device, dtype=locs.dtype)
    elif depot_loc.dim() == 1:
        depot_loc_3d = depot_loc.unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
    elif depot_loc.dim() == 2:
        depot_loc_3d = depot_loc.unsqueeze(1)
    else:
        depot_loc_3d = depot_loc

    d_ins = _compute_dense_d_ins(locs, depot_loc_3d)

    if k_neighbors is not None and k_neighbors < N:
        dist_customers = compute_pairwise_distance_matrix(locs)
        _, knn_indices = torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)
        knn_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
        knn_mask.scatter_(2, knn_indices, torch.ones_like(knn_indices, dtype=torch.bool))
        d_ins = d_ins.masked_fill(~knn_mask, float("inf"))

    if squeeze_batch:
        d_ins = d_ins.squeeze(0)

    return d_ins


def compute_sparse_insertion_cost(
    locs: torch.Tensor,
    k_neighbors: int = 15,
    depot_loc: Optional[torch.Tensor] = None,
    method: str = "construction",
    teacher: str = "nn",
    seed: int = 0,
    match_scale: bool = False,
    demand: Optional[torch.Tensor] = None,
    capacity: Optional[torch.Tensor] = None,
    cross_route_mode: str = "hard",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sparse (B, N, k) insertion cost for efficient storage and aggregation.

    Symmetrizes BEFORE selecting top-k to ensure d_ins(i,j) == d_ins(j,i)
    and consistent neighbor selection.

    Storage format:
        idx: (B, N, k) int16  — k-NN neighbor indices (cast to long before use)
        val: (B, N, k) float32 — insertion costs to those k neighbors

    Args:
        locs:        (B, N, 2) customer coordinates. Must be 3D.
        k_neighbors: Number of nearest neighbors per node.
        depot_loc:   (B, 1, 2) depot coordinates. Defaults to (0.5, 0.5).
        method:      Insertion-cost definition to use.
                     - "savings": Clarke-Wright-style marginal proxy
                       d_ins(i,j) = dist(D,i) + dist(i,j) - dist(D,j).
                     - "construction": along-tour arc distance from a
                       reference tour whose order is given by ``teacher``
                       (see compute_construction_aware_insertion_cost).
                     - "insertion": Route-Conditioned Insertion Cost (RCIC) —
                       a genuine marginal cost of inserting i into a route edge
                       adjacent to j, symmetrized (see
                       _compute_route_adjacent_insertion_cost). Recommended
                       target for Variant D.
        teacher:    Reference-tour teacher used when ``method=="construction"``:
                    "nn" | "random" | "sweep" | "nn_2opt" | "nn_split". Ignored
                    for other methods.
        seed:       Seed for random teachers (e.g. "random").
        match_scale: Rescale the random/nn_2opt teacher's d_cons to match the nn
                    teacher's batch mean (control for scale; off by default).
        demand:     (B, N) normalized demands — required only for "nn_split".
        capacity:   (B,) vehicle capacity (normalized) — required for "nn_split".
        cross_route_mode: "hard" | "soft" — nn_split cross-route handling.

    Returns:
        (idx, val):
            idx: (B, N, k) int16
            val: (B, N, k) float32
    """
    B, N, _ = locs.shape
    device = locs.device

    if depot_loc is None:
        depot_loc = torch.full((B, 1, 2), 0.5, device=device, dtype=locs.dtype)
    elif depot_loc.dim() == 2:
        depot_loc = depot_loc.unsqueeze(1)

    # Compute dense symmetric d_ins: (B, N, N)
    if method == "insertion":
        d_ins = _compute_route_adjacent_insertion_cost(locs, depot_loc)
    elif method == "construction":
        d_ins = _compute_dense_construction_d_ins(
            locs, depot_loc, teacher=teacher, seed=seed, match_scale=match_scale,
            demand=demand, capacity=capacity, cross_route_mode=cross_route_mode)
    else:
        d_ins = _compute_dense_d_ins(locs, depot_loc)

    dist_customers = compute_pairwise_distance_matrix(locs)  # (B, N, N)

    dist_customers = dist_customers + torch.eye(N, device=device).unsqueeze(0) * 1e9

    k_actual = min(k_neighbors, N - 1)
    _, knn_idx = torch.topk(dist_customers, k=k_actual, dim=-1, largest=False)

    knn_val = torch.gather(d_ins, dim=2, index=knn_idx)  # (B, N, k_actual)

    assert N <= 32767, f"N={N} exceeds int16 range; use int32 for larger instances"
    knn_idx_i16 = knn_idx.to(torch.int16)

    return knn_idx_i16, knn_val


# Construction-aware insertion cost (method="construction"): d_ins(i,j) = shortest
# along-tour arc distance between i and j on a greedy NN tour (depot -> ... -> depot).
# Gives routing-structural separation (not raw coordinate proximity): customers far
# apart in a good route have very different marginal consequences even if spatially
# close. Capacity is ignored (single Hamiltonian NN cycle), consistent with the CW proxy.

def _build_nn_customer_order(
    locs: torch.Tensor,          # (B, N, 2) customers
    depot_loc: torch.Tensor,     # (B, 1, 2) depot
) -> torch.Tensor:
    """
    Build a deterministic nearest-neighbour tour from the depot.

    At each step move to the nearest unvisited customer. Returns the occupancy
    order alone (the RCIC method only needs the visit sequence, not the tour
    geometry). Shareable with the construction method.

    Returns:
        order: (B, N) — customer indices (0..N-1) in visit order.
    """
    B, N, _ = locs.shape
    device = locs.device

    nodes = torch.cat([depot_loc, locs], dim=1)  # (B, N+1, 2); index 0 = depot
    dist = torch.cdist(nodes, nodes)             # (B, N+1, N+1)

    visited = torch.zeros(B, N, dtype=torch.bool, device=device)
    order = torch.empty(B, N, dtype=torch.long, device=device)
    current = torch.zeros(B, dtype=torch.long, device=device)   # node index; 0 = depot
    batch_idx = torch.arange(B, device=device)

    for step in range(N):
        # Distances from the current node to each customer (nodes 1..N)
        candidate_dist = dist[batch_idx, current, 1:]           # (B, N)
        candidate_dist = candidate_dist.masked_fill(visited, float("inf"))
        next_customer = candidate_dist.argmin(dim=1)            # (B,) in 0..N-1
        order[:, step] = next_customer
        visited[batch_idx, next_customer] = True
        current = next_customer + 1                             # +1: node index 0 is depot

    return order


def _compute_dense_construction_d_ins(
    locs: torch.Tensor,          # (B, N, 2)
    depot_loc: torch.Tensor,     # (B, 1, 2)
    teacher: str = "nn",
    seed: int = 0,
    match_scale: bool = False,
    demand: Optional[torch.Tensor] = None,
    capacity: Optional[torch.Tensor] = None,
    cross_route_mode: str = "hard",
) -> torch.Tensor:
    """Reference-tour distance ``d_cons`` on the given teacher's customer order.

    Dispatches to the teacher-specific order producer:
      - "nn":       nearest-neighbour order (== historical construction target).
      - "random":   fixed-seed random permutation.
      - "sweep":    polar sweep around the depot.
      - "nn_2opt":  2-opt-improved NN tour (capacity-agnostic).
      - "nn_split": capacity-aware Prins split of the NN tour (needs ``demand``).

    Only the order — and hence the tour geometry and total length ``L`` — differs
    between teachers. ``nn`` exactly reproduces the historical construction target.

    Args:
        locs:      (B, N, 2) customer coordinates.
        depot_loc: (B, 1, 2) depot coordinates.
        teacher:   reference-tour teacher.
        seed:      seed for random teachers ("random").
        match_scale: When True (and teacher uses a longer tour, e.g. "random"),
            rescale ``d_cons`` so its batch mean equals the ``nn`` teacher's batch
            mean. Off by default.
        demand:    (B, N) normalized demands — required for "nn_split".
        capacity:  (B,) vehicle capacity (normalized) — required for "nn_split".
        cross_route_mode: "hard" | "soft" — nn_split cross-route handling.

    Returns:
        (B, N, N) symmetric ``d_cons`` matrix, zero on the diagonal.
    """
    if teacher == "nn_split":
        if demand is None or capacity is None:
            raise ValueError("nn_split requires demand and capacity.")
        order = _build_nn_customer_order(locs, depot_loc)
        d, _ = _d_cons_nn_split(
            order, locs, depot_loc, demand, capacity, cross_route_mode=cross_route_mode)
        return d

    if teacher == "nn_2opt":
        order = _build_nn_customer_order(locs, depot_loc)
        d, _ = _d_cons_nn_2opt(order, locs, depot_loc)
        if match_scale:
            _, nn_pref, nn_total = _d_cons_from_order(_build_nn_customer_order(locs, depot_loc), locs, depot_loc)
            d_nn = d_cons_from_positions(nn_pref, nn_total)
            scale = d_nn.sum() / d.sum() if d.sum() > 0 else 1.0
            d = d * scale
        return d

    _, _, prefpos, total = get_customer_order(
        locs, depot_loc, teacher, seed=seed)
    d = d_cons_from_positions(prefpos, total)

    if match_scale and teacher == "random":
        _, _, nn_pref, nn_total = get_customer_order(locs, depot_loc, "nn")
        d_nn = d_cons_from_positions(nn_pref, nn_total)
        # Scale so the random tour's mean d_cons equals the nn tour's mean.
        scale = d_nn.sum() / d.sum() if d.sum() > 0 else 1.0
        d = d * scale

    return d


def _d_cons_from_order(
    order: torch.Tensor,         # (B, N) customer indices in visit order
    locs: torch.Tensor,          # (B, N, 2) customer coordinates
    depot_loc: torch.Tensor,     # (B, 1, 2) depot
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Positions ``p_i``, total length ``L`` and ordered coords for a given order.

    Rebuilds the reference-tour geometry the shared :func:`d_cons` needs from an
    arbitrary customer order — the only teacher-dependent input. The positions
    are the cumulative Euclidean travel distance just after each customer along
    the closed route ``depot -> order[0..N-1] -> depot``.

    Args:
        order:    (B, N) customer indices (0..N-1) in visit order.
        locs:     (B, N, 2) customer coordinates.
        depot_loc:(B, 1, 2) depot coordinates.

    Returns:
        (cust_ord, prefpos, total):
            cust_ord: (B, N, 2) customer coordinates in visit order.
            prefpos:  (B, N) cumulative along-tour distance just after each customer.
            total:    (B,) total closed tour length (depot -> ... -> depot).
    """
    B, N, _ = locs.shape
    device = locs.device

    cust_ord = torch.gather(locs, 1, order.unsqueeze(-1).expand(B, N, 2))
    node_seq = torch.cat([depot_loc, cust_ord, depot_loc], dim=1)   # (B, N+2, 2)
    edges = torch.norm(node_seq[:, 1:] - node_seq[:, :-1], dim=2)   # (B, N+1)
    total = edges.sum(dim=1)                                        # (B,)
    pref = torch.cat([torch.zeros(B, 1, device=device), edges.cumsum(dim=1)], dim=1)
    prefpos = pref[:, 1 : 1 + N]                                    # (B, N)
    return cust_ord, prefpos, total


def d_cons_from_positions(prefpos: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    """Shared reference-tour distance ``d_cons`` from along-tour positions.

    Given a customer order's cumulative positions ``p_i`` (Euclidean distance
    along the closed tour just after each customer) and total tour length ``L``:

        d_cons(i, j) = min( |p_i - p_j|,  L - |p_i - p_j| )

    i.e. the shorter of the two arcs between customers ``i`` and ``j`` on the
    reference tour. This is the exact formula used for the ``construction``
    (nearest-neighbour) target and is shared verbatim by every reference-tour
    teacher — only the *order* producer differs.

    Args:
        prefpos: (B, N) cumulative along-tour distance just after each customer.
        total:   (B,) total closed tour length.

    Returns:
        (B, N, N) symmetric ``d_cons`` matrix, zero on the diagonal.
    """
    B, N = prefpos.shape
    device = prefpos.device

    d_ptp = (prefpos.unsqueeze(2) - prefpos.unsqueeze(1)).abs()          # (B, N, N)
    d_arc = torch.minimum(d_ptp, total.unsqueeze(-1).unsqueeze(-1) - d_ptp)
    d_arc = torch.clamp(d_arc, min=0.0)

    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    d_arc = d_arc.masked_fill(eye, 0.0)
    return d_arc


def _build_random_order(
    locs: torch.Tensor,          # (B, N, 2) customers
    seed: int = 0,
) -> torch.Tensor:
    """Teacher a): random permutation of customers, fixed seed per instance.

    Each batch row gets an independent, deterministic-per-row permutation drawn
    from a generator seeded by ``seed`` and the batch (row) index. Deterministic
    across epochs and processes.

    Returns:
        order: (B, N) independent random permutation per row.
    """
    B, N, _ = locs.shape
    device = locs.device
    g = torch.Generator(device=device)
    rows = []
    for b in range(B):
        g.manual_seed((seed * 2654435761 + b * 40503) & 0xFFFFFFFF)
        rows.append(torch.randperm(N, generator=g, device=device))
    return torch.stack(rows, dim=0)


def _build_sweep_order(
    locs: torch.Tensor,          # (B, N, 2) customers
    depot_loc: torch.Tensor,     # (B, 1, 2) depot
) -> torch.Tensor:
    """Teacher b): polar sweep around the depot.

    Orders customers by polar angle ``atan2(y, x)`` about the depot, ties broken
    by distance to the depot. Produces a spatially coherent angular sweep tour —
    a cheap proxy for a route that fans out around the depot.

    Returns:
        order: (B, N) customer indices, sorted by angle then depot distance.
    """
    B, N, _ = locs.shape
    device = locs.device
    depot = depot_loc[:, 0:1, :]                           # (B, 1, 2)
    rel = locs - depot                                     # (B, N, 2)
    theta = torch.atan2(rel[..., 1], rel[..., 0])          # (B, N) in [-pi, pi]
    # Primary key: angle shifted to [0, 2pi) so the sweep wraps continuously.
    angle = torch.remainder(theta, 2 * math.pi)            # (B, N) in [0, 2pi)
    dist = torch.norm(rel, dim=-1)                         # (B, N)

    # Composite key for a lexicographic (angle, distance) sort. Angles span up to
    # ~6.28; multiply so the integer part encodes the angle bucket and the tiny
    # fractional part breaks ties within a bucket. Distances are bounded well below
    # 1.0 for unit-square instances, so they only ever break near-equal angles.
    ANG_SCALE = 10000                  # distinct angles map to distinct integers
    key = (angle * ANG_SCALE).round() + torch.clamp(dist, 0.0, 0.99999)
    return torch.argsort(key, dim=-1, stable=True)


def get_customer_order(
    locs: torch.Tensor,          # (B, N, 2)
    depot_loc: torch.Tensor,     # (B, 1, 2)
    teacher: str = "nn",
    **kw,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch a reference-tour order producer by teacher name.

    Args:
        locs:     (B, N, 2) customer coordinates.
        depot_loc:(B, 1, 2) depot coordinates.
        teacher:  one of "nn" | "random" | "sweep" | "hgs".
        **kw:     passed through (e.g. ``seed`` for the random teacher).

    Returns:
        (name, cust_ord, prefpos, total):
            name:      the canonical teacher name (resolves aliases, e.g. "nn").
            cust_ord:  (B, N, 2) customer coords in visit order.
            prefpos:   (B, N) cumulative along-tour distance just after each customer.
            total:     (B,) total closed tour length.
    """
    B, N, _ = locs.shape
    device = locs.device

    if teacher == "nn":
        order = _build_nn_customer_order(locs, depot_loc)
    elif teacher == "sweep":
        order = _build_sweep_order(locs, depot_loc)
    elif teacher == "random":
        order = _build_random_order(locs, seed=kw.get("seed", 0))
    elif teacher == "hgs":
        raise NotImplementedError(
            "The 'hgs' teacher requires a VRP solver (pyvrp or HGS-CVRP) and is "
            "not enabled in this build. See README section on reference teachers."
        )
    else:
        raise ValueError(f"Unknown teacher: {teacher!r}. Expected one of nn|random|sweep|hgs.")

    cust_ord, prefpos, total = _d_cons_from_order(order, locs, depot_loc)
    return teacher, cust_ord, prefpos, total


def compute_construction_aware_insertion_cost(
    locs: torch.Tensor,
    k_neighbors: int = 15,
    depot_loc: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse (B, N, k) construction-aware insertion cost, same storage format as
    compute_sparse_insertion_cost(method="construction").

    Args:
        locs:        (B, N, 2) customer coordinates. Must be 3D.
        k_neighbors: Number of nearest neighbors per node.
        depot_loc:   (B, 1, 2) depot coordinates. Defaults to (0.5, 0.5).

    Returns:
        (idx, val):
            idx: (B, N, k) int16 — k-NN neighbor indices (Euclidean kNN).
            val: (B, N, k) float32 — along-tour arc distances to those neighbors.
    """
    B, N, _ = locs.shape
    device = locs.device

    if depot_loc is None:
        depot_loc = torch.full((B, 1, 2), 0.5, device=device, dtype=locs.dtype)
    elif depot_loc.dim() == 2:
        depot_loc = depot_loc.unsqueeze(1)

    d_ins = _compute_dense_construction_d_ins(locs, depot_loc)

    # Euclidean kNN for neighbor selection (consistent with training convention)
    dist_customers = compute_pairwise_distance_matrix(locs)
    dist_customers = dist_customers + torch.eye(N, device=device).unsqueeze(0) * 1e9
    k_actual = min(k_neighbors, N - 1)
    _, knn_idx = torch.topk(dist_customers, k=k_actual, dim=-1, largest=False)

    knn_val = torch.gather(d_ins, dim=2, index=knn_idx)  # (B, N, k_actual)

    assert N <= 32767, f"N={N} exceeds int16 range; use int32 for larger instances"
    return knn_idx.to(torch.int16), knn_val


# Route-Conditioned Insertion Cost (RCIC), method="insertion" — recommended for
# Variant D. Genuine marginal insertion cost conditioned on a reference route:
#
#   D_RCIC(i, j) = 0.5 * [ Δ(i | j) + Δ(j | i) ]
#   Δ(i | j) = min over edges (u,v) adjacent to j of  d(u,i) + d(i,v) - d(u,v)
#
# i.e. cheapest extra route length of inserting i into an edge around j — the
# routing structure we want slots to encode, not raw coordinate distance. Symmetric
# (B, N, N) so MetricPreservationLoss consumes it directly. Reference route is a
# deterministic NN tour (capacity ignored, as with the other proxies).

def _compute_route_adjacent_insertion_cost(
    locs: torch.Tensor,          # (B, N, 2) customers
    depot_loc: torch.Tensor,     # (B, 1, 2) depot
) -> torch.Tensor:
    """
    Compute a symmetric, route-conditioned insertion-cost matrix (RCIC).

    Builds a deterministic nearest-neighbour reference route, then for every
    pair (i, j) evaluates the cheapest genuine marginal cost of inserting i
    into an edge adjacent to j (and vice versa), symmetrized:

        D_RCIC(i, j) = 0.5 * [ Δ(i|j) + Δ(j|i) ]

    Args:
        locs:      (B, N, 2) customer coordinates (batch must already be 3D).
        depot_loc: (B, 1, 2) depot coordinates.

    Returns:
        D_ins: (B, N, N) float32 — symmetric route-conditioned insertion cost.
    """
    B, N, _ = locs.shape
    device = locs.device

    if N < 3:
        # Not enough customers to form meaningful route-local insertion
        # relationships — fall back to plain pairwise distance.
        return torch.cdist(locs, locs)

    # 1. Reference route: customer indices in nearest-neighbour visit order
    order = _build_nn_customer_order(locs, depot_loc)   # (B, N)

    # 2. Route order -> predecessor / successor (customers at each position)
    #    First customer's predecessor and last customer's successor is the depot.
    pred = torch.empty_like(order)
    succ = torch.empty_like(order)
    pred[:, 0] = -1
    succ[:, -1] = -1
    if N > 1:
        pred[:, 1:] = order[:, :-1]
        succ[:, :-1] = order[:, 1:]
    depot_idx = torch.full((B,), N, dtype=torch.long, device=device)
    pred[:, 0] = depot_idx
    succ[:, -1] = depot_idx

    # 3. Route nodes: customers (0..N-1) then depot (N)
    route_nodes = torch.cat([locs, depot_loc], dim=1)   # (B, N+1, 2)
    D = torch.cdist(route_nodes, route_nodes)           # (B, N+1, N+1)

    # 4. directed[i, j] = cheapest insertion cost of i into an edge adjacent to j
    directed = torch.full((B, N, N), float("inf"), dtype=locs.dtype, device=device)
    batch_idx = torch.arange(B, device=device)

    for j_pos in range(N):
        j = order[:, j_pos]     # reference customer

        # We want, for every batch b and candidate i in [0, N):
        #   cost_before[i] = D[b, u[b], i] + D[b, i, j[b]] - D[b, u[b], j[b]]
        # All three dims are indexed with aligned advanced indices so each
        # term stays (B, N) — a trailing advanced index on a slice would push
        # the advanced dim to the front and give (B, B, N).
        i_range = torch.arange(N, device=device)

        # Edge 1: pred(j) -> j.
        u = pred[:, j_pos]
        v = j
        cost_before = (
            D[batch_idx, u, :N]
            + D[batch_idx[:, None], i_range[None, :], v[:, None]]
            - D[batch_idx, u, v].unsqueeze(-1)
        )

        # Edge 2: j -> succ(j).
        u2 = j
        v2 = succ[:, j_pos]
        cost_after = (
            D[batch_idx, u2, :N]
            + D[batch_idx[:, None], i_range[None, :], v2[:, None]]
            - D[batch_idx, u2, v2].unsqueeze(-1)
        )

        # Do NOT allow inserting i into an edge that already contains i —
        # otherwise d(j,i) + d(i,i) - d(j,i) = 0 creates an artificial zero target.
        i_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        invalid_before = (i_indices == u.unsqueeze(-1)) | (i_indices == v.unsqueeze(-1))
        invalid_after = (i_indices == u2.unsqueeze(-1)) | (i_indices == v2.unsqueeze(-1))
        cost_before = cost_before.masked_fill(invalid_before, float("inf"))
        cost_after = cost_after.masked_fill(invalid_after, float("inf"))

        # Cheapest insertion around j.
        best = torch.minimum(cost_before, cost_after)
        directed[batch_idx, :, j] = best

    # 5. Symmetrize: directed[i,j] = Δ(i|j), directed[j,i] = Δ(j|i)
    D_ins = 0.5 * (directed + directed.transpose(-1, -2))

    # Numerical cleanup.
    D_ins = torch.nan_to_num(D_ins, nan=0.0, posinf=0.0, neginf=0.0)
    D_ins = torch.clamp(D_ins, min=0.0)

    # Self distance = 0.
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    D_ins = D_ins.masked_fill(eye, 0.0)

    return D_ins


def compute_route_conditioned_insertion_cost(
    locs: torch.Tensor,
    k_neighbors: int = 15,
    depot_loc: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse (B, N, k) Route-Conditioned Insertion Cost (RCIC), same storage
    format as compute_sparse_insertion_cost(method="insertion").

    Args:
        locs:        (B, N, 2) customer coordinates. Must be 3D.
        k_neighbors: Number of nearest neighbors per node.
        depot_loc:   (B, 1, 2) depot coordinates. Defaults to (0.5, 0.5).

    Returns:
        (idx, val):
            idx: (B, N, k) int16 — k-NN neighbor indices (Euclidean kNN).
            val: (B, N, k) float32 — RCIC values to those neighbors.
    """
    B, N, _ = locs.shape
    device = locs.device

    if depot_loc is None:
        depot_loc = torch.full((B, 1, 2), 0.5, device=device, dtype=locs.dtype)
    elif depot_loc.dim() == 2:
        depot_loc = depot_loc.unsqueeze(1)

    d_ins = _compute_route_adjacent_insertion_cost(locs, depot_loc)

    # Euclidean kNN for neighbor selection (consistent with training convention)
    dist_customers = compute_pairwise_distance_matrix(locs)
    dist_customers = dist_customers + torch.eye(N, device=device).unsqueeze(0) * 1e9
    k_actual = min(k_neighbors, N - 1)
    _, knn_idx = torch.topk(dist_customers, k=k_actual, dim=-1, largest=False)

    knn_val = torch.gather(d_ins, dim=2, index=knn_idx)  # (B, N, k_actual)

    assert N <= 32767, f"N={N} exceeds int16 range; use int32 for larger instances"
    return knn_idx.to(torch.int16), knn_val


# ────────────────────────────────────────────────────────────────────────────────
# nn_split and nn_2opt teachers
# ────────────────────────────────────────────────────────────────────────────────

def _tour_length(
    order: torch.Tensor,         # (B, N) customer indices
    locs: torch.Tensor,          # (B, N, 2)
    depot_loc: torch.Tensor,     # (B, 1, 2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordered customer coords and closed-tour lengths for a given order.

    Returns:
        (cust_ord, length):
            cust_ord: (B, N, 2) customer coordinates in visit order.
            length:   (B,) total closed tour length (depot -> ... -> depot).
    """
    B, N, _ = locs.shape
    cust_ord = torch.gather(locs, 1, order.unsqueeze(-1).expand(B, N, 2))
    node_seq = torch.cat([depot_loc, cust_ord, depot_loc], dim=1)   # (B, N+2, 2)
    edges = torch.norm(node_seq[:, 1:] - node_seq[:, :-1], dim=2)   # (B, N+1)
    return cust_ord, edges.sum(dim=1)


def _two_opt(
    order: torch.Tensor,         # (B, N) customer indices in visit order
    locs: torch.Tensor,          # (B, N, 2) customer coordinates
    depot_loc: torch.Tensor,     # (B, 1, 2) depot
    max_passes: int = 100,
) -> tuple[torch.Tensor, float]:
    """2-opt improvement of a depot-anchored tour (capacity-agnostic).

    Reverses the customer sub-tour between two interior edges whenever doing so
    shortens the tour. The depot stays fixed as the first and last node: only the
    segment *between* the two swapped edges (both strictly interior) is reversed.
    Vectorized over the batch; sweeps until no instance improves. Deterministic
    (always applies the first improving swap in canonical scan order).

    Args:
        order:      (B, N) customer indices in visit order.
        locs:       (B, N, 2) customer coordinates.
        depot_loc:  (B, 1, 2) depot coordinates.
        max_passes: Upper bound on improvement sweeps.

    Returns:
        (improved_order, mean_improvement):
            improved_order:    (B, N) capacity-agnostic 2-opt-optimised order.
            mean_improvement:  mean relative tour-length reduction over the batch.
    """
    B, N, _ = locs.shape
    device = locs.device
    batch_idx = torch.arange(B, device=device)

    cur = order.clone()                                    # (B, N)
    _, best_len = _tour_length(cur, locs, depot_loc)       # (B,)
    _, len0 = _tour_length(order, locs, depot_loc)

    nodes = torch.cat([depot_loc, locs], dim=1)            # (B, N+1, 2); 0=depot
    # dist includes the depot; map customer index c -> node c+1.
    D = torch.cdist(nodes, nodes)

    for _ in range(max_passes):
        improved = torch.zeros(B, dtype=torch.bool)
        # Scan candidate non-adjacent edge pairs (i, j) with 0 < i < j < N-1.
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                # Current edges: (prev[i], cur[i]) and (cur[j], next[j]).
                # Doing the 2-opt swap on positions i..j reverses that segment:
                #   removed edges: (cur[i-1], cur[i]) and (cur[j], cur[j+1])
                #   added edges:   (cur[i-1], cur[j]) and (cur[i], cur[j+1])
                # Improvement when: d(i-1,j) + d(i,j+1) < d(i-1,i) + d(j,j+1).
                a = cur[:, i - 1] + 1                       # node index of customer before i
                b = cur[:, i] + 1                           # customer at i
                c = cur[:, j] + 1                           # customer at j
                e = cur[:, j + 1] + 1 if j + 1 < N else torch.zeros(
                    B, dtype=torch.long, device=device)     # customer after j
                if j + 1 < N:
                    e = cur[:, j + 1] + 1
                else:
                    e = torch.zeros(B, dtype=torch.long, device=device)  # depot (index 0)

                old = D[batch_idx, a, b] + D[batch_idx, c, e]
                new = D[batch_idx, a, c] + D[batch_idx, b, e]
                gains = old - new                           # (B,)
                upd = (gains > 1e-9) & ~improved
                if upd.any():
                    # Reverse the order segment [i..j] in each improved row.
                    seg = cur[upd, i:j+1]
                    flipped = cur.clone()
                    flipped[upd, i:j+1] = seg.flip(1)
                    cur = flipped
                    improved = improved | upd
                    _, new_len = _tour_length(cur, locs, depot_loc)
                    best_len[upd] = new_len[upd]
        if not improved.any():
            break

    mean_improvement = ((best_len - len0) / len0.clamp_min(1e-12)).mean().item()
    return cur, mean_improvement


def _prins_split(
    order: torch.Tensor,         # (B, N) customer indices in visit order
    demand: torch.Tensor,        # (B, N) normalized demands in [0, cap]
    capacity: torch.Tensor,      # (B,) vehicle capacity (normalized, e.g. 1)
    locs: torch.Tensor,          # (B, N, 2) customer coordinates
    depot_loc: torch.Tensor,     # (B, 1, 2) depot
) -> tuple[torch.Tensor, float]:
    """Prins split of an order into capacity-feasible routes (min total length).

    Given a giant tour order, finds the route segmentation (a list of contiguous
    sub-orders, each respecting the vehicle capacity) that minimises the total
    route length (sum over routes of depot->stops->depot). Implemented with a
    dynamic program over the order positions; vectorized over the batch.

    Args:
        order:    (B, N) customer indices in visit order.
        demand:   (B, N) normalized demands.
        capacity: (B,) vehicle capacity.
        locs:     (B, N, 2) customer coordinates.
        depot_loc:(B, 1, 2) depot coordinates.

    Returns:
        (route_ids, mean_num_routes):
            route_ids:       (B, N) int route id r(i) for each customer.
            mean_num_routes: mean number of routes over the batch.
    """
    B, N, _ = locs.shape
    device = locs.device
    if N == 0:
        return torch.empty(B, 0, dtype=torch.long, device=device), 0.0

    batch_idx = torch.arange(B, device=device)

    nodes = torch.cat([depot_loc, locs], dim=1)     # (B, N+1, 2); 0=depot
    D = torch.cdist(nodes, nodes)                   # (B, N+1, N+1)

    # Consecutive customer distances along the order: seg[t] = dist(order[t], order[t+1]).
    o = order                                       # (B, N) these are customer ids
    # Distance along the sequence between consecutive ordered customers (B, N-1).
    cn = torch.norm(locs.gather(1, o[..., 1:].unsqueeze(-1).expand(B, N - 1, 2)) -
                    locs.gather(1, o[..., :-1].unsqueeze(-1).expand(B, N - 1, 2)), dim=-1)
    # cpref[b, j] = sum of seg[0..j-1] (the first j edges). Length N (indices 0..N-1).
    cpref = torch.zeros(B, N, device=device)
    if N > 1:
        cpref[:, 1:] = torch.cumsum(cn, dim=1)

    dem_seq = torch.gather(demand, 1, order)        # (B, N) demands along the order

    # Prefix sums of demands: load_pref[j] = sum dem_seq[0..j-1], length N+1.
    load_pref = torch.zeros(B, N + 1, device=device)
    load_pref[:, 1:] = torch.cumsum(dem_seq, dim=1)

    # Route cost for a window order[k..i-1] (customers positions k..i-1):
    #   internal edges seg[k..i-2] = cpref[i-1] - cpref[k]   (needs i-1 <= N-1)
    #   cost(k, i) = D(depot, o[k]) + (cpref[i-1] - cpref[k]) + D(o[i-1], depot)
    depot_node = torch.zeros(B, dtype=torch.long, device=device)  # depot is node 0
    co = o + 1                                      # node index (customer id + depot shift)

    # DP. dp[i] = min route length serving order[0..i-1]; dp[0] = 0.
    INF = float("inf")
    dp = torch.full((B, N + 1), INF, device=device)
    dp[:, 0] = 0.0
    bp = torch.zeros((B, N + 1), dtype=torch.long, device=device)  # best split point k

    for i in range(1, N + 1):
        best = torch.full((B,), INF, device=device)
        best_k = torch.zeros(B, dtype=torch.long, device=device)
        # Iterate k for the window order[k..i-1]; O(N^2) total over i.
        for k in range(i - 1, -1, -1):
            win_load = load_pref[:, i] - load_pref[:, k]   # sum dem[k..i-1]
            feasible = (win_load <= capacity + 1e-9) & (dp[:, k] < INF)
            # route cost: depot -> o[k] ... o[i-1] -> depot
            internal = cpref[:, i - 1] - cpref[:, k]       # edges seg[k..i-2]
            c = D[batch_idx, depot_node, co[:, k]] + internal + D[batch_idx, co[:, i - 1], depot_node]
            cand = dp[:, k] + c
            upd = feasible & (cand < best)
            best = torch.where(upd, cand, best)
            best_k = torch.where(upd, torch.full_like(best_k, k, dtype=torch.long), best_k)
        dp[:, i] = best
        bp[:, i] = best_k

    # Backtrack route ids from bp.
    route_ids = torch.zeros(B, N, dtype=torch.long, device=device)
    # Trace from i=N back. While reconstructing, customer order[..] belongs to a route.
    i = torch.full((B,), N, dtype=torch.long, device=device)
    # We'll assign a route id per (k..i-1) segment from the end.
    n_routes = torch.zeros(B, dtype=torch.long, device=device)
    # Simpler iterative traceback per batch row (N small, B vectorized over the k-loop).
    # Because bp depends on i and k per batch, we do a python loop over segments.
    # This is acceptable: number of routes is small relative to N.
    for _ in range(N):
        done = i <= 0
        if done.all():
            break
        k = torch.gather(bp, 1, i.unsqueeze(1)).squeeze(1)   # split point for prefix i
        # Customers order[k..i-1] get route id n_routes (assign increasing).
        for b in range(B):
            if i[b] > 0:
                route_ids[b, k[b]:i[b]] = n_routes[b]
                n_routes[b] += 1
        i = k

    mean_routes = n_routes.float().mean().item()
    return route_ids, mean_routes


def _d_cons_nn_2opt(
    order: torch.Tensor,         # (B, N) NN customer order (before 2-opt)
    locs: torch.Tensor,          # (B, N, 2)
    depot_loc: torch.Tensor,     # (B, 1, 2)
    max_2opt_passes: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """d_cons for the nn_2opt teacher: 2-opt-improved NN tour.

    Improves the nearest-neighbour order with capacity-agnostic 2-opt (depot
    fixed as endpoints) and returns the along-tour ``d_cons`` on the improved
    order plus a stats dict.

    Returns:
        (d_cons, stats):
            d_cons: (B, N, N) symmetric d_cons on the improved order.
            stats:  mean relative 2-opt improvement (log this).
    """
    opt, mean_impr = _two_opt(order, locs, depot_loc, max_passes=max_2opt_passes)
    _, prefpos, total = _d_cons_from_order(opt, locs, depot_loc)
    return d_cons_from_positions(prefpos, total), mean_impr


def _d_cons_nn_split(
    order: torch.Tensor,         # (B, N) NN customer order
    locs: torch.Tensor,          # (B, N, 2)
    depot_loc: torch.Tensor,     # (B, 1, 2)
    demand: torch.Tensor,        # (B, N) normalized demands
    capacity: torch.Tensor,      # (B,) vehicle capacity
    cross_route_mode: str = "hard",
) -> tuple[torch.Tensor, dict]:
    """d_cons for the nn_split teacher (capacity-aware Prins split).

    Splits the NN order into capacity-feasible routes (Prins split), giving a
    route id ``r(i)`` per customer. Along a single route, ``d_cons`` is the usual
    arc distance; customers on different routes are pinned to ``L/2`` (hard) or
    pulled by ``arc + L/4`` clipped at ``L/2`` (soft).

    Args:
        order:          (B, N) NN customer order.
        locs:           (B, N, 2) customer coordinates.
        depot_loc:      (B, 1, 2) depot coordinates.
        demand:         (B, N) normalized demands.
        capacity:       (B,) vehicle capacity (normalized, e.g. 1).
        cross_route_mode: "hard" (L/2) or "soft" (arc + L/4, clipped at L/2).

    Returns:
        (d_cons, stats):
            d_cons: (B, N, N) symmetric d_cons, zero diagonal, 0 <= d <= L/2.
            stats:  dict with mean cross-route fraction and mean route count.
    """
    B, N, _ = locs.shape
    device = locs.device

    _, prefpos, total = _d_cons_from_order(order, locs, depot_loc)
    L = total.unsqueeze(-1).unsqueeze(-1)                # (B, 1, 1)
    d_arc = d_cons_from_positions(prefpos, total)        # (B, N, N)

    route_ids, mean_routes = _prins_split(order, demand, capacity, locs, depot_loc)
    r = route_ids                                          # (B, N) customer ids in ORIGINAL order
    # Build route-id matrix over customer pairs (same customer indexing as locs/d_arc).
    same = r.unsqueeze(2).eq(r.unsqueeze(1))               # (B, N, N) r(i)==r(j)
    cross = ~same
    cross = cross & ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)

    if cross_route_mode == "hard":
        d = torch.where(same, d_arc, L / 2)
    elif cross_route_mode == "soft":
        gamma = L / 4
        soft = d_arc + gamma
        soft = torch.minimum(soft, L / 2)
        d = torch.where(same, d_arc, soft)
    else:
        raise ValueError(f"cross_route_mode must be 'hard'|'soft', got {cross_route_mode!r}")

    # Ensure exact zero diagonal and symmetry.
    d = (d + d.transpose(-1, -2)) / 2.0
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    d = d.masked_fill(eye, 0.0)

    n_pairs = B * N * N
    stats = {
        "mean_cross_route_fraction": cross.sum().item() / n_pairs,
        "mean_num_routes": mean_routes,
    }
    return d, stats
