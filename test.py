from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import lightning.pytorch as pl

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot, AMSlot
from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline


def _allow_safe_globals() -> None:
    """Allowlist our custom classes for torch.load(weights_only=True)."""
    try:
        torch.serialization.add_safe_globals([SingleSharedBaseline])
    except Exception:
        # torch <2.6 has no safe-globals; weights_only is a no-op there.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a slot-model checkpoint at a target size")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a trained .ckpt")
    parser.add_argument("--model", type=str, default="am", choices=["am", "pomo"],
                        help="Model class: 'am' (AMSlot) or 'pomo' (POMOSlot). Must match the checkpoint.")
    parser.add_argument("--num_loc", type=int, required=True,
                        help="Target number of CUSTOMERS N (e.g. 50/100/200/500/1000).")
    parser.add_argument("--n_inst", type=int, default=1024, help="Number of eval instances")
    parser.add_argument("--batch_size", type=int, default=256, help="Eval batch size")
    parser.add_argument("--num_starts", type=int, default=None,
                        help="Multi-start greedy for POMO checkpoints (1 = single-start/AM). Default: policy default.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for the generated eval instances, so every run uses the "
                             "SAME test set (fair cross-checkpoint comparison).")
    parser.add_argument("--out", type=str, default=None, help="JSON path to save results (default: results/eval_<run>.json)")
    args = parser.parse_args()

    num_loc = args.num_loc
    _allow_safe_globals()

    # Seed BEFORE generating the eval set so every run evaluates on the same data.
    pl.seed_everything(args.seed, workers=True)

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    env = CVRPEnv(generator_kwargs=dict(num_loc=num_loc))
    model_cls = POMOSlot if args.model == "pomo" else AMSlot
    model = model_cls.load_from_checkpoint(args.ckpt, env=env, map_location="cpu")

    # Eval dataloader — fresh instances at target size
    ds = env.dataset(batch_size=[args.n_inst])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()

    # Greedy decode via the policy
    rewards = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            td = env.reset(batch)
            # CVRP env prepends the depot, so locs == num_loc + 1 rows.
            n_customers = int(td["locs"].shape[-2]) - 1
            if n_customers != num_loc:
                raise RuntimeError(
                    f"num_loc mismatch: requested --num_loc {num_loc} but the "
                    f"CVRPEnv generated {n_customers} customers (locs {tuple(td['locs'].shape)}). "
                    f"Refusing to evaluate on the wrong size."
                )
            out = model.policy(td, env, phase="test", num_starts=args.num_starts)
            rewards.append(out["reward"].cpu())

    reward = torch.cat(rewards)
    tour_len = -reward  # CVRP: reward = -tour_length
    result = {
        "backbone": args.model,
        "num_loc": num_loc,
        "n_inst": len(reward),
        "seed": args.seed,
        "mean_reward": float(reward.mean()),
        "mean_tour_length": float(tour_len.mean()),
        "std_tour_length": float(tour_len.std()),
        "ckpt": str(args.ckpt),
    }

    print(f"[OK] evaluated at num_loc={num_loc} customers  n={len(reward)}  "
          f"mean tour = {result['mean_tour_length']:.4f} "
          f"(mean reward = {result['mean_reward']:.4f})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
