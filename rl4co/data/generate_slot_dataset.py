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
    """Sample node locations from a Gaussian mixture in [0, 1]^2. Vectorized.

    Each instance draws its own n_clusters; nodes are assigned via clamped
    indices so fewer-cluster instances never sample phantom centers.

    Returns (batch, n, 2).
    """
    lo, hi = n_clusters_range
    max_k = hi

    n_clust = torch.randint(lo, hi + 1, (batch,))           # per-instance count
    centers = torch.rand(batch, max_k, 2)                   # (batch, max_k, 2)

    # Assign nodes: random float in [0, n_clust[b]) clamped to valid index
    rand_float = torch.rand(batch, n) * n_clust.float().unsqueeze(1)
    assign_idx = rand_float.long().clamp(max=(n_clust - 1).unsqueeze(1))

    # Gather cluster center for each node, add Gaussian noise, clamp to [0,1]^2
    node_centers = torch.gather(
        centers.unsqueeze(2).expand(batch, max_k, n, 2).permute(0, 2, 1, 3),
        dim=2,
        index=assign_idx.unsqueeze(-1).unsqueeze(-1).expand(batch, n, 1, 2),
    ).squeeze(2)
    locs = (node_centers + cluster_std * torch.randn(batch, n, 2)).clamp(0.0, 1.0)
    return locs


def _gen_demands(batch: int, n: int, max_demand: int = 9) -> torch.Tensor:
    """Integer demands in [1, max_demand], normalized by vehicle capacity
    (Kool 2019: capacity = max(max_demand, round(max_demand * n / 4))).

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
    """Sparse (B, N, k) insertion cost -> (idx int16, val float32).
    depot: (B, 2) or (B, 1, 2). method: savings | construction | insertion.
    """
    from rl4co.data.insertion_cost import compute_sparse_insertion_cost

    depot_3d = depot.unsqueeze(1) if depot.dim() == 2 else depot  # (B, 1, 2)
    return compute_sparse_insertion_cost(
        locs, k_neighbors=k_neighbors, depot_loc=depot_3d, method=method
    )


def _gen_time_windows(
    locs: torch.Tensor,
    depot: torch.Tensor,
    max_time: float = 480,
    scale: float = 150.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Solomon-style time windows + dummy service durations for
    CVRP-coordinate instances (locs/depot in [0,1]).

    Mirrors :class:`CVRPTWGenerator`: travel time is scaled by ``scale`` (150)
    so a [0,1] distance unit maps to the classic Solomon 150~ coordinate frame
    with ``max_time=480``. Produces ``time_windows`` (B, N+1, 2) of "service
    must *start* within [start, end]" bounds, and zero service durations.

    Returns:
        (time_windows, durations), both float32, in raw time units.
    """
    B = locs.shape[0]
    N = locs.shape[1]
    # distance from depot to each customer, scaled to Solomon frame
    dist = torch.cdist(depot.unsqueeze(1), locs, p=2).squeeze(1) * scale   # (B, N)
    dist = torch.cat((torch.zeros(B, 1), dist), dim=1)                     # (B, N+1)
    durations = torch.zeros(B, N + 1, dtype=torch.float32)

    upper_bound = max_time - dist - durations                              # (B, N+1)
    ts_1 = torch.rand(B, N + 1)
    ts_2 = torch.rand(B, N + 1)
    min_ts = (dist + (upper_bound - dist) * ts_1).int()
    max_ts = (dist + (upper_bound - dist) * ts_2).int()
    min_times = torch.min(min_ts, max_ts).float()
    max_times = torch.max(min_ts, max_ts).float()

    # depot window [0, max_time]
    min_times[:, 0] = 0.0
    max_times[:, 0] = max_time

    # guard min < max (as in CVRPTWGenerator)
    mask = min_times == max_times
    if torch.any(mask):
        min_tmp = min_times.clone()
        min_tmp[mask] = torch.max(dist[mask].int(), min_tmp[mask] - 1).float()
        min_times = min_tmp
        mask = min_times == max_times
        if torch.any(mask):
            max_tmp = max_times.clone()
            max_tmp[mask] = torch.min(
                torch.floor(upper_bound[mask]).int(),
                torch.max(
                    torch.ceil(min_tmp[mask] + durations[mask]).int(),
                    max_tmp[mask] + 1,
                ),
            ).float()
            max_times = max_tmp

    time_windows = torch.stack((min_times, max_times), dim=-1)             # (B, N+1, 2)
    durations[:, 0] = 0.0
    return time_windows, durations


def generate_split(
    n_instances: int,
    n: int,
    dist: str,
    k_neighbors: int,
    method: str,
    chunk_size: int = 512,
    problem: str = "cvrp",
) -> dict[str, torch.Tensor | str]:
    """Generate one full split in memory-friendly chunks.

    Returns dict: locs, depot, demand, capacity, d_ins_idx, d_ins_val,
                  format_version, method. For problem="vrptw" the returned dict
                  additionally carries durations and time_windows.
    """
    gen_fn = _gen_uniform if dist == "uniform" else _gen_clustered

    all_locs, all_depots, all_demands = [], [], []
    all_dins_idx, all_dins_val = [], []
    all_durs, all_tws = [], []

    for start in range(0, n_instances, chunk_size):
        end = min(start + chunk_size, n_instances)
        bs = end - start

        locs   = gen_fn(bs, n)              # (bs, N, 2)
        depot  = torch.rand(bs, 2)          # (bs, 2) — CVRPEnv expects (B, 2)
        demand = _gen_demands(bs, n)        # (bs, N) already normalized

        d_idx, d_val = _compute_sparse_d_ins(locs, depot, k_neighbors, method)  # (bs,N,k) each

        all_locs.append(locs)
        all_depots.append(depot)
        all_demands.append(demand)
        all_dins_idx.append(d_idx)
        all_dins_val.append(d_val)

        if problem == "vrptw":
            tw, durs = _gen_time_windows(locs, depot)          # (bs,N+1,2), (bs,N+1)
            all_durs.append(durs)
            all_tws.append(tw)

        if (start // chunk_size) % 10 == 0:
            print(
                f"  [{problem}:{dist}] N={n} -- generated {end}/{n_instances} instances...",
                flush=True,
            )

    n_total = sum(t.shape[0] for t in all_locs)
    out: dict[str, torch.Tensor | str] = {
        "locs":           torch.cat(all_locs,      dim=0),        # (B, N, 2)
        "depot":          torch.cat(all_depots,    dim=0),        # (B, 2)
        "demand":         torch.cat(all_demands,   dim=0),        # (B, N)
        "capacity":       torch.full((n_total, 1), 1.0),          # (B, 1) always 1.0
        "d_ins_idx":      torch.cat(all_dins_idx,  dim=0),        # (B, N, k) int16
        "d_ins_val":      torch.cat(all_dins_val,  dim=0),        # (B, N, k) float32
        "format_version": "sparse_v2",                            # metadata
        "method":         method,                                 # d_ins cost method tag
        "problem":        problem,                                # cvrp | ovrp | vrptw
    }
    if problem == "vrptw":
        out["durations"] = torch.cat(all_durs, dim=0)          # (B, N+1)
        out["time_windows"] = torch.cat(all_tws, dim=0)        # (B, N+1, 2)
    return out


def generate_and_save(
    out_dir: str | Path,
    n: int,
    dist: str,
    n_train: int,
    n_val: int,
    n_test: int,
    k_neighbors: int,
    method: str,
    chunk_size: int | None = None,
    seed: int | None = None,
    problem: str = "cvrp",
) -> None:
    """Generate and save train/val/test splits to {out_dir}/{method}/ (so
    different d_ins targets never collide). Problem determines the filename
    prefix (cvrpN/ovrpN/vrptwN) and whether time-window fields are attached."""
    if chunk_size is None:
        # Dense pairwise work scales as B*N*N. Keep each dense tensor near
        # 64 MB so the several live intermediates remain practical at N=1000.
        chunk_size = max(1, min(512, 16_000_000 // (n * n)))

    out_dir = Path(out_dir) / method
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{problem}{n}"
    splits = [("train", n_train), ("val", n_val), ("test", n_test)]
    for split_idx, (split, n_instances) in enumerate(splits):
        if n_instances == 0:
            continue
        fpath = out_dir / f"{prefix}_{dist}_{split}.pt"
        if fpath.exists():
            print(f"  Skipping {fpath} (already exists).")
            continue
        # Split-local seeds make recovery deterministic if one output already
        # exists after an interrupted generation job.
        if seed is not None:
            torch.manual_seed(seed + split_idx)
        print(
            f"\nGenerating {split} split: {n_instances} instances, "
            f"problem={problem}, N={n}, dist={dist}, chunk_size={chunk_size}"
        )
        data = generate_split(
            n_instances, n, dist, k_neighbors, method, chunk_size, problem=problem)
        temporary = fpath.with_suffix(fpath.suffix + ".partial")
        torch.save(data, temporary)
        temporary.replace(fpath)
        print(f"  Saved -> {fpath}  (d_ins_idx shape: {data['d_ins_idx'].shape})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Slot NCO datasets with cached sparse d_ins")
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
                             "Insertion Cost (RCIC), the recommended Variant D target.")
    parser.add_argument("--problem",     type=str,   default="cvrp",
                        choices=["cvrp", "ovrp", "vrptw"],
                        help="Problem instances to generate. ovrp = same CVRP data "
                             "under an different filename prefix (open-route reward is "
                             "a decode-time choice). vrptw additionally attaches "
                             "durations + time_windows to each instance.")
    parser.add_argument("--k_neighbors", type=int,   default=15)
    parser.add_argument("--chunk_size",  type=int,   default=None,
                        help="Instances per generation chunk (default: adaptive to num_locs)")
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
            seed=args.seed,
            problem=args.problem,
        )


if __name__ == "__main__":
    main()
