"""Generate the eval CVRP dataset once and cache it to an NPZ file.

Run once per (num_loc, n_inst, seed) combo, then test.py can consume the cached file:

    python gen_data.py --num_loc 100 --n_inst 1024 --seed 1234 \
        --out data/eval_cvrp100_seed1234.npz
    python test.py --ckpt PATH --model am --num_loc 100 --n_inst 1024 --seed 1234 \
        --data data/eval_cvrp100_seed1234.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
from rl4co.envs import CVRPEnv
from rl4co.data.utils import save_tensordict_to_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate & cache a CVRP eval dataset")
    parser.add_argument("--num_loc", type=int, required=True,
                        help="Number of CUSTOMERS N (must match --num_loc in test.py).")
    parser.add_argument("--n_inst", type=int, default=1024, help="Number of instances")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for the generated instances. Use the SAME seed as "
                             "test.py so the eval set is reproducible / comparable.")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path to save.")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    # Mirrors test.py: CVRPEnv with the same generator_params at the same target size.
    # NOTE: use env.generator() directly (not env.dataset()) so we get the raw
    # TensorDict, which is what save_tensordict_to_npz() expects (env.dataset()
    # returns a TensorDictDataset wrapper object).
    env = CVRPEnv(generator_params=dict(num_loc=args.num_loc))
    td = env.generator([args.n_inst])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_tensordict_to_npz(td, out)
    print(f"[OK] generated {len(td)} CVRP instances at num_loc={args.num_loc} "
          f"(seed={args.seed}) -> {out}")


if __name__ == "__main__":
    main()
