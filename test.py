from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import lightning.pytorch as pl
from tensordict import TensorDict
from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot, AMSlot
from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline
from rl4co.models.zoo.sil import SIL
from rl4co.models.zoo.l2r import L2RModel
from rl4co.models.zoo.icam import ICAMCVRP
from rl4co.models.zoo.invit import INViT
from rl4co.data.slot_dataset import SlotDataset
from rl4co.data.transforms import StateAugmentation
from rl4co.data.utils import load_npz_to_tensordict


def _allow_safe_globals() -> None:

    def _pick_device(value: str | None) -> torch.device:
        if value:
            return torch.device(value)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    """Allowlist our custom classes for torch.load(weights_only=True)."""
    try:
        torch.serialization.add_safe_globals([SingleSharedBaseline])
    except Exception:
        # torch <2.6 has no safe-globals; weights_only is a no-op there.
        pass


def _pick_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a routing checkpoint at a target size")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a trained .ckpt")
    parser.add_argument("--model", type=str, default="am", choices=["am", "pomo", "l2r", "icam", "sil", "invit"],
                        help="Model class. Must match the checkpoint.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Cached MARS .pt test split shared across methods; recommended for comparisons")
    parser.add_argument("--num_loc", type=int, required=True,
                        help="Target number of CUSTOMERS N (e.g. 50/100/200/500/1000).")
    parser.add_argument("--n_inst", type=int, default=1024,
                        help="Number of eval instances")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="Eval batch size")
    parser.add_argument("--num_starts", type=int, default=None,
                        help="Multi-start greedy for POMO checkpoints (1 = single-start/AM). Default: policy default.")
    parser.add_argument("--augment", action="store_true",
                        help="Evaluate eight geometric augmentations")
    parser.add_argument("--device", type=str, default=None,
                        help="Device such as cuda, mps, or cpu")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for the generated eval instances, so every run uses the "
                             "SAME test set (fair cross-checkpoint comparison).")
    parser.add_argument("--out", type=str, default=None,
                        help="JSON path to save results (default: results/eval_<run>.json)")
    args = parser.parse_args()

    num_loc = args.num_loc
    _allow_safe_globals()

    # Seed BEFORE generating the eval set so every run evaluates on the same data.
    pl.seed_everything(args.seed, workers=True)

    # Load model
    device = _pick_device(args.device)
    env = CVRPEnv(generator_params=dict(num_loc=num_loc))
    model_cls = {"pomo": POMOSlot, "am": AMSlot, "l2r": L2RModel,
                 "icam": ICAMCVRP, "sil": SIL, "invit": INViT}[args.model]
    model_kwargs = {"env": env, "map_location": device, "weights_only": False}
    model = model_cls.load_from_checkpoint(args.ckpt, **model_kwargs)
    model.to(device)
    if args.model == "sil":
        # Inference only needs the learned policy, not the training label cache.
        model.labels.clear()
        model.best_policy_state = model.repair_policy_state = None

    # Use the shared cached split when supplied, otherwise seeded fresh instances.
    if args.data:
        ds = load_npz_to_tensordict(args.data)
    else:
        ds = (SlotDataset(args.data_path, variant="none", max_instances=args.n_inst)
              if args.data_path else env.dataset(batch_size=[args.n_inst]))

    collate_fn = getattr(ds, "collate_fn", None)
    if collate_fn is None and not args.data_path:
        collate_fn = torch.stack  # TensorDict supports stacking a list of TensorDicts

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    model.eval()

    # Greedy decode via the policy
    rewards = []
    actual_num_starts = 1
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                batch = TensorDict(batch, batch_size=[batch["demand"].size(0)])
            batch = batch.to(device)
            if args.model == "icam":
                out_reward, _ = model._rollout(batch, sampling=False)
                rewards.append(out_reward.mean(dim=1).cpu())
                continue
            td = env.reset(batch)
            # CVRP env prepends the depot, so locs == num_loc + 1 rows.
            n_customers = int(td["locs"].shape[-2]) - 1
            if n_customers != num_loc:
                raise RuntimeError(
                    f"num_loc mismatch: requested --num_loc {num_loc} but the "
                    f"CVRPEnv generated {n_customers} customers (locs {tuple(td['locs'].shape)}). "
                    f"Refusing to evaluate on the wrong size."
                )
            if args.augment:
                if args.model not in {"am", "pomo"}:
                    raise ValueError(
                        "--augment is supported only for AM and POMO")
                td = StateAugmentation(num_augment=8, augment_fn="dihedral8",
                                       first_aug_identity=True)(td)
            if args.model == "l2r":
                out = model._rollout(batch, sampling=False)
                reward_batch = out[0]
            else:
                out = model.policy(td, env, phase="test",
                                   num_starts=args.num_starts)
                reward_batch = out["reward"]
            # Multi-start rollouts are stacked start-major by RL4CO. Report
            # one best-start score per instance, not B * starts pseudo-instances.
            batch_size = batch.batch_size[0]
            if args.augment:
                reward_batch = reward_batch.reshape(
                    8, batch_size, -1).max(dim=-1).values
            else:
                actual_num_starts = reward_batch.numel() // batch_size
                reward_batch = reward_batch.reshape(
                    batch_size, actual_num_starts).max(dim=1).values
            rewards.append(reward_batch.cpu())

    reward = torch.cat(rewards)
    tour_len = -reward  # CVRP: reward = -tour_length
    result = {
        "backbone": args.model,
        "num_loc": num_loc,
        "n_inst": len(reward),
        "seed": args.seed,
        "mean_reward": float(reward.mean()),
        "mean_tour_length": float(tour_len.mean()),
        "std_tour_length": float(tour_len.std()) if len(tour_len) > 1 else 0.0,
        "ckpt": str(args.ckpt),
        "data_path": str(Path(args.data_path).resolve()) if args.data_path else None,
        "num_starts": actual_num_starts,
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
