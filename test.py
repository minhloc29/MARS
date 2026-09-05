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
from rl4co.data.utils import load_npz_to_tensordict


def _pick_device(device: str | None) -> torch.device:
    """Resolve a --device value to a torch.device, auto-picking CUDA/MPS/CPU.

    Recognizes 'cuda', 'cuda:0', 'cuda:1', 'cpu', 'mps', etc. Bare 'cuda' -> 'cuda:0'.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _describe_device(device: torch.device) -> str:
    """Human-readable description of the selected device (includes GPU name)."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device.index or 0)
        return f"{device} ({name})"
    return str(device)


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
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: 'cuda', 'cpu', 'mps', or 'cuda:0'. "
                             "Default: auto-pick cuda -> mps -> cpu.")
    parser.add_argument("--data", type=str, default=None,
                        help="Optional pre-generated .npz eval set (from gen_data.py). "
                             "When given, loads this instead of generating fresh instances "
                             "(default: generate as before).")
    parser.add_argument("--out", type=str, default=None, help="JSON path to save results (default: results/eval_<run>.json)")
    args = parser.parse_args()

    num_loc = args.num_loc
    _allow_safe_globals()

    # Seed BEFORE generating the eval set so every run evaluates on the same data.
    pl.seed_everything(args.seed, workers=True)

    device = _pick_device(args.device)
    print(f"[device] {_describe_device(device)}")

    # Load model directly onto the compute device (CUDA, MPS, or CPU).
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    env = CVRPEnv(generator_params=dict(num_loc=num_loc))
    model_cls = POMOSlot if args.model == "pomo" else AMSlot
    model = model_cls.load_from_checkpoint(args.ckpt, env=env, map_location=device)

    # Eval dataset — fresh instances at target size, or load a pre-generated NPZ if given.
    if args.data:
        ds = load_npz_to_tensordict(args.data)
        if len(ds) != args.n_inst:
            print(f"[warn] {args.data} has {len(ds)} instances, --n_inst={args.n_inst} requested. "
                  f"Using {len(ds)}.")
        collate_fn = torch.stack
    else:
        ds = env.dataset(batch_size=[args.n_inst])
        collate_fn = getattr(ds, "collate_fn", None)
        if collate_fn is None:
            collate_fn = torch.stack  # TensorDict supports stacking a list of TensorDicts

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    model.eval()

    # Greedy decode via the policy
    rewards = []
    n_batches = len(loader)
    print(f"[progress] decoding {n_batches} batches of batch_size={args.batch_size}")
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
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
            # Running mean so you can see progress during the run.
            running_mean = float(torch.cat(rewards).mean())
            print(f"[progress] batch {i}/{n_batches}  done  "
                  f"running_mean_reward={running_mean:.4f} "
                  f"(mean_tour={-running_mean:.4f})", flush=True)

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
