from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

# Capacity table following Kool et al. (2019) convention
CAPACITIES = {
    50: 40.0,
    100: 50.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}


def gen_uniform(batch: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform customer locations in [0, 1]^2."""
    return rng.uniform(0.0, 1.0, size=(batch, n, 2)).astype(np.float32)


def gen_clustered(
    batch: int,
    n: int,
    rng: np.random.Generator,
    n_clusters_range: tuple[int, int] = (3, 7),
    cluster_std: float = 0.07,
) -> np.ndarray:
    """Gaussian mixture clustered locations in [0, 1]^2.

    Vectorized implementation matching RL4CO's _gen_clustered:
    each instance draws 3..7 cluster centers, assigns nodes, adds Gaussian noise,
    and clamps to [0, 1]^2.
    """
    lo, hi = n_clusters_range
    n_clust = rng.integers(lo, hi + 1, size=(batch,))
    centers = rng.uniform(0.0, 1.0, size=(batch, hi, 2)).astype(np.float32)

    rand_float = rng.uniform(0.0, 1.0, size=(batch, n)) * n_clust[:, None]
    assign_idx = np.clip(rand_float.astype(int), 0, n_clust[:, None] - 1)

    batch_idx = np.arange(batch)[:, None]
    node_centers = centers[batch_idx, assign_idx]

    noise = rng.standard_normal(size=(batch, n, 2)).astype(np.float32)
    locs = np.clip(node_centers + cluster_std * noise, 0.0, 1.0)
    return locs.astype(np.float32)


def gen_cvrp_split(
    batch: int,
    n: int,
    dist: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate a full CVRP split with locs, depot, demand, capacity."""
    rng = np.random.default_rng(seed)

    if dist in ("gaussian", "clustered", "cluster"):
        locs = gen_clustered(batch, n, rng)
    else:
        locs = gen_uniform(batch, n, rng)

    depot = rng.uniform(0.0, 1.0, size=(batch, 2)).astype(np.float32)

    capacity = CAPACITIES.get(n, max(9.0, round(9.0 * n / 4.0)))
    raw_demand = rng.integers(1, 10, size=(batch, n)).astype(np.float32)
    demand = (raw_demand / capacity).astype(np.float32)
    capacity_arr = np.ones((batch, 1), dtype=np.float32)

    return {
        "locs": locs,
        "depot": depot,
        "demand": demand,
        "capacity": capacity_arr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CVRP test datasets (Numpy only, no heavy dependencies required)"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000],
        help="Target number of customers N (e.g. 50 100 200 500 1000)",
    )
    parser.add_argument(
        "--dists",
        type=str,
        nargs="+",
        default=["uniform", "gaussian"],
        help="Distributions to generate ('uniform', 'gaussian')",
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=1000,
        help="Number of instances per split (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/test",
        help="Output directory to save .npz files",
    )

    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("Generating CVRP Test Datasets (Standalone Numpy)")
    print(f"Sizes: {args.sizes}")
    print(f"Distributions: {args.dists}")
    print(f"Instances per split: {args.n_instances}")
    print(f"Seed: {args.seed}")
    print(f"Output folder: {out_path.resolve()}")
    print("==================================================")

    for n in args.sizes:
        for dist in args.dists:
            save_file = out_path / f"cvrp_{n}_{dist}_seed{args.seed}.npz"
            print(f"\nGenerating N={n} {dist} ({args.n_instances} instances)...", end=" ", flush=True)

            data = gen_cvrp_split(
                batch=args.n_instances,
                n=n,
                dist=dist,
                seed=args.seed + n,
            )

            np.savez_compressed(
                save_file,
                locs=data["locs"],
                depot=data["depot"],
                demand=data["demand"],
                capacity=data["capacity"],
            )

            file_size_mb = save_file.stat().st_size / (1024 * 1024)
            print(f"Done! -> {save_file} ({file_size_mb:.2f} MB)")

    print("\n[OK] All test datasets generated successfully!")
    print(f"You can now run benchmarks using:")
    print(f"  python run_classical_baselines.py --solver hgs --sizes {' '.join(map(str, args.sizes))} --dists {' '.join(args.dists)}")


if __name__ == "__main__":
    main()
