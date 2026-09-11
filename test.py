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


def _load_npz(data_path: str, num_loc: int) -> TensorDict:
   
    td = load_npz_to_tensordict(data_path)

    # _reset prepends the depot row, so locs here must be the N customers only.
    n_rows = td["locs"].shape[-2]
    if n_rows == num_loc + 1:
        if "depot" not in td.keys():
            raise RuntimeError(
                f"{data_path}: locs has {n_rows} rows (= num_loc+1, depot already "
                "included) but no separate 'depot' key — the env always prepends the "
                "depot, so locs must be the {num_loc} customers only."
            )
        td.set("locs", td["locs"][:, 1:])  # strip the depot row the env will re-add
    elif n_rows != num_loc:
        raise RuntimeError(
            f"{data_path}: num_loc mismatch — requested --num_loc {num_loc} but locs has "
            f"{n_rows} rows. Expected {num_loc} customers with a separate 'depot' key."
        )
    return td


def _batch_iter(ds, batch_size: int):
    """Yield fixed-size batches from either a batched TensorDict (npz) or a
    list of TensorDicts (generated), without DataLoader collation."""
    n = ds.shape[0] if isinstance(ds, TensorDict) else len(ds)
    is_td = isinstance(ds, TensorDict)
    for i in range(0, n, batch_size):
        chunk = ds[i:i + batch_size]
        if not is_td:
            chunk = torch.stack(chunk)
        yield chunk


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
    parser.add_argument("--n_inst", type=int, default=1000,
                        help="Number of eval instances")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="Eval batch size")
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

    # Use the shared cached npz split when supplied, otherwise seeded fresh instances.
    if args.data_path:
        ds = _load_npz(args.data_path, num_loc)
    else:
        ds = env.dataset(batch_size=[args.n_inst])

    model.eval()

    # Greedy decode via the policy
    rewards = []
    actual_num_starts = 1
    with torch.no_grad():
        for batch in _batch_iter(ds, args.batch_size):
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

    elif args.out is None:
        out_path = Path("results") / f"eval_{args.model}_{num_loc}_{args.seed}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
