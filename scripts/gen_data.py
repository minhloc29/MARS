from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Capacity table following Kool et al. 2019
CAPACITIES = {
    50: 40.0,
    100: 50.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}


def _seed_everything(seed: int) -> None:
    """Set random seed cleanly without requiring lightning."""
    try:
        import lightning.pytorch as pl
        pl.seed_everything(seed, workers=True)
    except Exception:
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:
            pass


def _gen_numpy_dataset(
    num_loc: int,
    n_inst: int,
    loc_dist: str,
    seed: int,
    loc_mean: float = 0.5,
    loc_std: float = 0.2,
) -> dict[str, np.ndarray]:
    """Fallback generator using pure NumPy (zero heavy dependencies)."""
    rng = np.random.default_rng(seed)

    if loc_dist in ("gaussian", "normal"):
        # Normal distribution centered at loc_mean with loc_std
        locs = rng.normal(loc_mean, loc_std, size=(n_inst, num_loc, 2))
        locs = np.clip(locs, 0.0, 1.0).astype(np.float32)
    elif loc_dist in ("cluster", "gaussian_mixture", "clustered"):
        # Gaussian mixture with 3..7 clusters
        n_clust = rng.integers(3, 8, size=(n_inst,))
        centers = rng.uniform(0.0, 1.0, size=(n_inst, 7, 2)).astype(np.float32)
        rand_f = rng.uniform(0.0, 1.0, size=(n_inst, num_loc)) * n_clust[:, None]
        assign_idx = np.clip(rand_f.astype(int), 0, n_clust[:, None] - 1)
        b_idx = np.arange(n_inst)[:, None]
        node_centers = centers[b_idx, assign_idx]
        noise = rng.standard_normal(size=(n_inst, num_loc, 2)).astype(np.float32)
        locs = np.clip(node_centers + 0.07 * noise, 0.0, 1.0).astype(np.float32)
    else:
        # Uniform [0, 1]^2
        locs = rng.uniform(0.0, 1.0, size=(n_inst, num_loc, 2)).astype(np.float32)

    depot = rng.uniform(0.0, 1.0, size=(n_inst, 2)).astype(np.float32)
    capacity_val = CAPACITIES.get(num_loc, max(9.0, round(9.0 * num_loc / 4.0)))
    raw_demand = rng.integers(1, 10, size=(n_inst, num_loc)).astype(np.float32)
    demand = (raw_demand / capacity_val).astype(np.float32)
    capacity = np.ones((n_inst, 1), dtype=np.float32)

    return {
        "locs": locs,
        "depot": depot,
        "demand": demand,
        "capacity": capacity,
    }


def _gen_single_split(
    num_loc: int,
    n_inst: int,
    loc_dist: str,
    seed: int,
    out_path: Path,
    dist_kwargs: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try RL4CO CVRPEnv if available, otherwise fallback to NumPy
    use_rl4co = False
    try:
        from rl4co.envs import CVRPEnv
        from rl4co.data.utils import save_tensordict_to_npz

        generator_params = dict(num_loc=num_loc, loc_distribution=loc_dist)
        generator_params.update(dist_kwargs)
        env = CVRPEnv(generator_params=generator_params)
        td = env.generator([n_inst])
        save_tensordict_to_npz(td, out_path)
        use_rl4co = True
    except Exception:
        use_rl4co = False

    if not use_rl4co:
        data = _gen_numpy_dataset(
            num_loc=num_loc,
            n_inst=n_inst,
            loc_dist=loc_dist,
            seed=seed,
            loc_mean=dist_kwargs.get("loc_mean", 0.5),
            loc_std=dist_kwargs.get("loc_std", 0.2),
        )
        np.savez_compressed(out_path, **data)

    print(
        f"[OK] generated {n_inst} CVRP instances at num_loc={num_loc} "
        f"(dist={loc_dist}, seed={seed}) -> {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate & cache CVRP eval datasets")
    parser.add_argument(
        "--num_loc",
        type=int,
        nargs="*",
        default=None,
        help="Number of CUSTOMERS N (e.g. 50 or multiple: 50 100 200 500 1000).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Alias for --num_loc with multiple sizes.",
    )
    parser.add_argument(
        "--n_inst",
        type=int,
        default=1000,
        help="Number of instances per split (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for generated instances (default: 1234).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Explicit output .npz path (for single size run).",
    )
    parser.add_argument(
        "--loc_dist",
        type=str,
        nargs="+",
        default=["uniform"],
        choices=[
            "uniform",
            "gaussian",
            "cluster",
            "gaussian_mixture",
            "mixed",
            "mix_distribution",
            "exponential",
            "poisson",
        ],
        help="Customer location distribution(s). Default: uniform.",
    )

    # Distribution-specific arguments
    dist_args = parser.add_argument_group("distribution arguments")
    dist_args.add_argument("--num_modes", type=int, default=None)
    dist_args.add_argument("--cdist", type=float, default=None)
    dist_args.add_argument("--n_cluster", type=int, default=None)
    dist_args.add_argument("--n_cluster_mix", type=int, default=None)
    dist_args.add_argument("--loc_mean", type=float, default=0.5)
    dist_args.add_argument("--loc_std", type=float, default=0.2)
    dist_args.add_argument("--loc_rate", type=float, default=None)

    args = parser.parse_args()

    _seed_everything(args.seed)

    # Resolve target sizes
    sizes = args.sizes or args.num_loc or [50, 100, 200, 500, 1000]
    if isinstance(sizes, int):
        sizes = [sizes]

    dists = args.loc_dist if isinstance(args.loc_dist, list) else [args.loc_dist]

    dist_kwargs = {}
    for key in (
        "num_modes",
        "cdist",
        "n_cluster",
        "n_cluster_mix",
        "loc_mean",
        "loc_std",
        "loc_rate",
    ):
        v = getattr(args, key, None)
        if v is not None:
            dist_kwargs[key] = v

    # If single run with explicit --out
    if len(sizes) == 1 and len(dists) == 1 and args.out:
        _gen_single_split(
            num_loc=sizes[0],
            n_inst=args.n_inst,
            loc_dist=dists[0],
            seed=args.seed,
            out_path=Path(args.out),
            dist_kwargs=dist_kwargs,
        )
        return

    # Batch run over all requested sizes and distributions
    for n in sizes:
        for d in dists:
            out_file = Path(f"data/test/cvrp_{n}_{d}_seed{args.seed}.npz")
            _gen_single_split(
                num_loc=n,
                n_inst=args.n_inst,
                loc_dist=d,
                seed=args.seed + n,
                out_path=out_file,
                dist_kwargs=dist_kwargs,
            )


if __name__ == "__main__":
    main()
