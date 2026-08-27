from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import lightning.pytorch as pl
from tensordict import TensorDict

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot, AMSlot


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a slot-model checkpoint at a target size")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a trained .ckpt")
    parser.add_argument("--model", type=str, default="am", choices=["am", "pomo"],
                        help="Model class to load: 'am' (AMSlot) or 'pomo' (POMOSlot). Must match the checkpoint.")
    parser.add_argument("--num_loc", type=int, default=None, help="Target size N (default: from checkpoint)")
    parser.add_argument("--n_inst", type=int, default=1024, help="Number of eval instances")
    parser.add_argument("--batch_size", type=int, default=256, help="Eval batch size")
    parser.add_argument("--num_starts", type=int, default=None,
                        help="Multi-start greedy for POMO checkpoints (1 = single-start/AM). Default: use checkpoint's.")
    parser.add_argument("--out", type=str, default=None, help="JSON path to save results (default: results/eval_<run>.json)")
    parser.add_argument("--devices", type=int, default=1, help="GPUs (1 default)")
    args = parser.parse_args()

    # ── Reconstruct the model from checkpoint hyperparameters ────────────
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    backbone = args.model  # model class comes from --model, not hparams
    num_loc = args.num_loc or hparams.get("num_loc", 100)

    # Build env at the target size and the matching model class.
    env = CVRPEnv(generator_kwargs=dict(num_loc=num_loc))
    model_cls = POMOSlot if backbone == "pomo" else AMSlot
    model = model_cls.load_from_checkpoint(args.ckpt, env=env, map_location="cpu", weights_only = False)
    
    print("=" * 60)
    print("Requested num_loc:", num_loc)

    env = CVRPEnv(generator_kwargs=dict(num_loc=num_loc))

    print("ENV:", env)
    print("GENERATOR:", env.generator)

    if hasattr(env.generator, "__dict__"):
        print("GENERATOR DICT:", env.generator.__dict__)

    print("=" * 60)

    # ── Eval dataloader (fresh instances at target size) ─────────────────
    ds = env.dataset(batch_size=[args.n_inst])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        logger=False,
        enable_progress_bar=True,
    )

    # ── Greedy decode via the policy ─────────────────────────────────────
    rewards = []
    with torch.no_grad():
        for batch in loader:
            device = next(model.parameters()).device    
            batch = TensorDict(
                batch,
                batch_size=[batch["locs"].shape[0]],
                device=device,
            )
            td = env.reset(batch)
            # num_starts: default to policy default (1 for AM / policy default for POMO).
            ns = args.num_starts
            out = model.policy(td, env, phase="test", num_starts=ns)
            
            r = out["reward"]

            print(
                f"N={num_loc} | "
                f"locs={tuple(td['locs'].shape)} | "
                f"coord_range=({td['locs'].min().item():.3f}, "
                f"{td['locs'].max().item():.3f}) | "
                f"reward_mean={r.mean().item():.4f} | "
                f"reward_min={r.min().item():.4f} | "
                f"reward_max={r.max().item():.4f}"
            )

            rewards.append(out["reward"].cpu())

    reward = torch.cat(rewards)
    tour_len = -reward  # CVRP: reward = -tour_length
    result = {
        "backbone": backbone,
        "num_loc": num_loc,
        "n_inst": len(reward),
        "mean_reward": float(reward.mean()),
        "mean_tour_length": float(tour_len.mean()),
        "std_tour_length": float(tour_len.std()),
        "ckpt": str(args.ckpt),
    }

    print(f"num_loc={num_loc}  n={len(reward)}  mean tour = {result['mean_tour_length']:.4f} "
          f"(mean reward = {result['mean_reward']:.4f})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
