from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
from rl4co.envs import CVRPEnv
from rl4co.data.utils import save_tensordict_to_npz


def main() -> None:
    # Supported --loc_dist values (see rl4co/envs/common/utils.py get_sampler()):
    #   uniform             -> Uniform(min_loc, max_loc)                       (default)
    #   normal / gaussian   -> Normal(loc_mean, loc_std)
    #   exponential         -> Exponential(rate)
    #   poisson             -> Poisson(rate)      (use with demand, not locs)
    #   gaussian_mixture    -> Gaussian_Mixture(num_modes, cdist)
    #   cluster             -> Cluster(n_cluster)
    #   mixed               -> Mixed(n_cluster_mix)
    #   mix_distribution    -> Mix_Distribution(n_cluster, n_cluster_mix)
    parser = argparse.ArgumentParser(description="Generate & cache a CVRP eval dataset")
    parser.add_argument("--num_loc", type=int, required=True,
                        help="Number of CUSTOMERS N (must match --num_loc in test.py).")
    parser.add_argument("--n_inst", type=int, default=1000, help="Number of instances")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for the generated instances. Use the SAME seed as "
                             "test.py so the eval set is reproducible / comparable.")
    parser.add_argument("--out", type=str, help="Output .npz path to save.")
    parser.add_argument("--loc_dist", type=str, default="uniform",
                        choices=["uniform", "gaussian", "cluster", "gaussian_mixture",
                                 "mixed", "mix_distribution", "exponential", "poisson"],
                        help="Customer location distribution. Default: uniform.")

    # Distribution-specific arguments (only the ones matching --loc_dist are consumed).
    dist_args = parser.add_argument_group("distribution arguments")
    dist_args.add_argument("--num_modes", type=int, default=None,
                           help="gaussian_mixture: number of modes.")
    dist_args.add_argument("--cdist", type=float, default=None,
                           help="gaussian_mixture: center distance between modes.")
    dist_args.add_argument("--n_cluster", type=int, default=None,
                           help="cluster: number of clusters.")
    dist_args.add_argument("--n_cluster_mix", type=int, default=None,
                           help="mixed/mix_distribution: number of mixed-cluster points.")
    dist_args.add_argument("--loc_mean", type=float, default=0.5,
                           help="gaussian: mean of the location Normal distribution.")
    dist_args.add_argument("--loc_std", type=float, default=0.2,
                           help="gaussian: std of the location Normal distribution.")
    dist_args.add_argument("--loc_rate", type=float, default=None,
                           help="exponential/poisson: rate parameter.")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    # Build generator_params: base + the chosen distribution. Only pass the args that
    # the distribution actually needs, so the unused ones don't leak into the sampler.
    generator_params = dict(num_loc=args.num_loc, loc_distribution=args.loc_dist)
    for key in ("num_modes", "cdist", "n_cluster", "n_cluster_mix",
                "loc_mean", "loc_std", "loc_rate"):
        value = getattr(args, key, None)
        if value is not None:
            generator_params[key] = value

    # Mirrors test.py: CVRPEnv with the same generator_params at the same target size.
    # NOTE: use env.generator() directly (not env.dataset()) so we get the raw
    # TensorDict, which is what save_tensordict_to_npz() expects (env.dataset()
    # returns a TensorDictDataset wrapper object).
    env = CVRPEnv(generator_params=generator_params)
    td = env.generator([args.n_inst])

    if args.out is None:
            if args.loc_dist == "uniform":
                out = Path(f"data/test/cvrp_{args.num_loc}_uniform_seed{args.seed}.npz")
            elif args.loc_dist == "gaussian":
                out = Path(f"data/test/cvrp_{args.num_loc}_gaussian_mean{args.loc_mean}_std{args.loc_std}_seed{args.seed}.npz")
            elif args.loc_dist == "cluster":
                out = Path(f"data/test/cvrp_{args.num_loc}_cluster_n{args.n_cluster}_seed{args.seed}.npz")
    else:
        out = Path(args.out)
    
    out.parent.mkdir(parents=True, exist_ok=True)
    save_tensordict_to_npz(td, out)
    print(f"[OK] generated {len(td)} CVRP instances at num_loc={args.num_loc} "
          f"(seed={args.seed}) -> {out}")


if __name__ == "__main__":
    main()
