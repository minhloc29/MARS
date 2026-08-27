"""
M1 — Offline dataset generator for Metric-Aware Slot NCO.

Generates CVRP instance pools with two spatial distributions:
  - Uniform: nodes sampled uniform in [0, 1]^2
  - Clustered: nodes sampled from a Gaussian mixture (3-7 clusters)

For each instance, computes and caches the sparse k-NN d_ins matrices.

Output format (per split): a .pt file containing a dict:
    {
        "locs":      (N_instances, N, 2)    float32
        "depot":     (N_instances, 2)       float32   (shape B,2 — not B,1,2)
        "demand":    (N_instances, N)       float32   normalized to [0,1] by capacity
        "capacity":  (N_instances, 1)       float32   always 1.0 (demand already normalized)
        "d_ins_idx": (N_instances, N, k)    int16     k-NN neighbor indices
        "d_ins_val": (N_instances, N, k)    float32   insertion costs to k neighbors
        "format_version": "sparse_v2"               metadata string
    }

NOTE: demand convention
    demand is normalized by vehicle_capacity (Kool 2019 convention), so the
    effective vehicle capacity seeno by the model is always 1.0. This matches
    CVRPGenerator's convention in rl4co where capacity=1.0 means demands
    are pre-normalized. Cross-checked against CVRPGenerator:
        capacity = max_demand * N / 4  (Kool 2019)
        demand_norm = demand_raw / capacity  ->  effective capacity = 1.0

Usage:
    python -m rl4co.data.generate_slot_dataset \\
        --num_locs 100 --dist uniform --n_train 100000 --n_val 1000 \\
        --out_dir ./data/slot_datasets_v2 --k_neighbors 15 --seed 42

The insertion-cost target is selected via --method (savings / construction /
insertion). Datasets are written to a method-scoped subdirectory of --out_dir
(e.g. .../insertion/cvrp100_uniform_train.pt) so different d_ins targets never
collide; pass the matching directory as train.py's --data_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


# CVRP standard capacities (Kool et al. 2019)
CAPACITIES = {50: 40.0, 100: 50.0, 200: 70.0, 500: 100.0}


def _gen_uniform(batch: int, n: int) -> torch.Tensor:
    """Sample node locations uniformly in [0, 1]^2. Returns (batch, n, 2)."""
    return torch.rand(batch, n, 2)


def _gen_clustered(
    batch: int,
    n: int,
    n_clusters_range: tuple[int, int] = (3, 7),
    cluster_std: float = 0.07,
) -> torch.Tensor:
    """
    Sample node locations from a Gaussian mixture in [0, 1]^2.
    Vectorized — no Python loop over instances.

    Each instance draws its own n_clusters in [lo, hi]. Nodes are assigned
    to clusters using per-instance random indices clamped to valid range,
    so instances with fewer clusters never sample from 'phantom' cluster centers.

    Returns (batch, n, 2).
    """
    lo, hi = n_clusters_range
    max_k = hi

    # Per-instance cluster counts
    n_clust = torch.randint(lo, hi + 1, (batch,))

    # Sample cluster centers for all instances
    centers = torch.rand(batch, max_k, 2)

    # Assign nodes: random float in [0, n_clust[b]) per node, clamped to valid index
    rand_float = torch.rand(batch, n) * n_clust.float().unsqueeze(1)
    assign_idx = rand_float.long().clamp(max=(n_clust - 1).unsqueeze(1))

    # Gather cluster centers for each node
    assign_expanded = assign_idx.unsqueeze(-1).expand(batch, n, 2)
    node_centers = torch.gather(
        centers.unsqueeze(2).expand(batch, max_k, n, 2).permute(0, 2, 1, 3),
        dim=2,
        index=assign_idx.unsqueeze(-1).unsqueeze(-1).expand(batch, n, 1, 2),
    ).squeeze(2)

    # Add Gaussian noise and clamp to [0, 1]^2
    locs = (node_centers + cluster_std * torch.randn(batch, n, 2)).clamp(0.0, 1.0)
    return locs


def _gen_demands(batch: int, n: int, max_demand: int = 9) -> torch.Tensor:
    """
    Sample integer demands in [1, max_demand] and normalise by vehicle capacity.
    Uses Kool 2019 convention: capacity = max(max_demand, round(max_demand * n / 4)).

    Returns (batch, n) float32 in (0, 1].
    """
    vehicle_capacity = CAPACITIES.get(n, max(max_demand, round(max_demand * n / 4)))
    demands = torch.randint(1, max_demand + 1, (batch, n)).float()
    return demands / float(vehicle_capacity)


def _compute_sparse_d_ins(
    locs: torch.Tensor,
    depot: torch.Tensor,
    k_neighbors: int,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sparse (B, N, k) insertion cost. Returns (idx int16, val float32).
    depot: (B, 2) or (B, 1, 2) — both accepted.
    method: "savings" | "construction" | "insertion" — d_ins target definition.
    """
    from rl4co.data.insertion_cost import compute_sparse_insertion_cost

    if depot.dim() == 2:
        depot_3d = depot.unsqueeze(1)  # (B, 1, 2)
    else:
        depot_3d = depot

    return compute_sparse_insertion_cost(
        locs, k_neighbors=k_neighbors, depot_loc=depot_3d, method=method
    )


def generate_split(
    n_instances: int,
    n: int,
    dist: str,
    k_neighbors: int,
    method: str,
    chunk_size: int = 512,
) -> dict[str, torch.Tensor | str]:
    """
    Generate a full split (train/val/test) in memory-friendly chunks.

    Args:
        n_instances: Total number of instances.
        n:           Number of customer nodes per instance.
        dist:        "uniform" or "clustered".
        k_neighbors: k for kNN sparsification of d_ins.
        method:      "savings" | "construction" | "insertion" — d_ins target.
        chunk_size:  Instances per processing chunk (avoid OOM for large N).

    Returns:
        dict with keys: locs, depot, demand, capacity, d_ins_idx, d_ins_val,
                        format_version.
    """
    gen_fn = _gen_uniform if dist == "uniform" else _gen_clustered

    all_locs, all_depots, all_demands = [], [], []
    all_dins_idx, all_dins_val = [], []

    for start in range(0, n_instances, chunk_size):
        end = min(start + chunk_size, n_instances)
        bs = end - start

        locs   = gen_fn(bs, n)              # (bs, N, 2)
        depot  = torch.rand(bs, 2)          # (bs, 2) — CVRPEnv expects (B, 2) not (B,1,2)
        demand = _gen_demands(bs, n)        # (bs, N) already normalized

        d_idx, d_val = _compute_sparse_d_ins(locs, depot, k_neighbors, method)  # (bs,N,k) each

        all_locs.append(locs)
        all_depots.append(depot)
        all_demands.append(demand)
        all_dins_idx.append(d_idx)
        all_dins_val.append(d_val)

        if (start // chunk_size) % 10 == 0:
            print(
                f"  [{dist}] N={n} -- generated {end}/{n_instances} instances...",
                flush=True,
            )

    n_total = sum(t.shape[0] for t in all_locs)
    return {
        "locs":           torch.cat(all_locs,      dim=0),        # (B, N, 2)
        "depot":          torch.cat(all_depots,    dim=0),        # (B, 2)
        "demand":         torch.cat(all_demands,   dim=0),        # (B, N)
        "capacity":       torch.full((n_total, 1), 1.0),          # (B, 1) always 1.0
        "d_ins_idx":      torch.cat(all_dins_idx,  dim=0),        # (B, N, k) int16
        "d_ins_val":      torch.cat(all_dins_val,  dim=0),        # (B, N, k) float32
        "format_version": "sparse_v2",                            # metadata
    }


def generate_and_save(
    out_dir: str | Path,
    n: int,
    dist: str,
    n_train: int,
    n_val: int,
    n_test: int,
    k_neighbors: int,
    method: str,
    chunk_size: int = 512,
) -> None:
    """
    Generate and save train/val/test splits to out_dir.

    Datasets are written to a method-scoped subdirectory so different d_ins
    targets (savings/construction/insertion) never collide:
        {out_dir}/{method}/cvrp{n}_{dist}_train.pt
        {out_dir}/{method}/cvrp{n}_{dist}_val.pt
        {out_dir}/{method}/cvrp{n}_{dist}_test.pt
    """
    out_dir = Path(out_dir) / method
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, n_instances in [("train", n_train), ("val", n_val), ("test", n_test)]:
        if n_instances == 0:
            continue
        fpath = out_dir / f"cvrp{n}_{dist}_{split}.pt"
        if fpath.exists():
            print(f"  Skipping {fpath} (already exists).")
            continue
        print(f"\nGenerating {split} split: {n_instances} instances, N={n}, dist={dist}")
        data = generate_split(n_instances, n, dist, k_neighbors, method, chunk_size)
        torch.save(data, fpath)
        d_idx_shape = data["d_ins_idx"].shape
        print(f"  Saved -> {fpath}  (d_ins_idx shape: {d_idx_shape}, format: sparse_v2)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Slot NCO datasets with cached sparse d_ins (sparse_v2 format)"
    )
    parser.add_argument("--num_locs",    type=int,   default=100,
                        help="Number of customer nodes N")
    parser.add_argument("--dist",        type=str,   default="uniform",
                        choices=["uniform", "clustered", "both"])
    parser.add_argument("--n_train",     type=int,   default=100_000)
    parser.add_argument("--n_val",       type=int,   default=1_000)
    parser.add_argument("--n_test",      type=int,   default=1_000)
    parser.add_argument("--out_dir",     type=str,   default="./data/slot_datasets_v2")
    parser.add_argument("--method",      type=str,   default="construction",
                        choices=["savings", "construction", "insertion"],
                        help="d_ins target definition. 'insertion' = Route-Conditioned "
                             "Insertion Cost (RCIC), the recommended Variant D target. "
                             "Default 'construction' matches current cached datasets.")
    parser.add_argument("--k_neighbors", type=int,   default=15)
    parser.add_argument("--chunk_size",  type=int,   default=512)
    parser.add_argument("--seed",        type=int,   default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    dists = ["uniform", "clustered"] if args.dist == "both" else [args.dist]
    for d in dists:
        generate_and_save(
            out_dir=args.out_dir,
            n=args.num_locs,
            dist=d,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            k_neighbors=args.k_neighbors,
            method=args.method,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
