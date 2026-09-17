from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import torch

from rl4co.data.transforms import StateAugmentation
from rl4co.envs import CVRPEnv
from rl4co.models.zoo.elg import ELG
from rl4co.models.zoo.icam import ICAMCVRP
from rl4co.models.zoo.invit import INViT
from rl4co.models.zoo.l2r import L2RModel
from rl4co.models.zoo.pomo import POMO
from rl4co.models.zoo.pomo_slot import AMSlot, POMOSlot
from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline
from rl4co.models.zoo.sil import SIL
from rl4co.utils import eval_utils

MODEL_CLASSES = {
    "am": AMSlot,
    "pomo": POMOSlot,
    "pomo_base": POMO,
    "sil": SIL,
    "icam": ICAMCVRP,
    "l2r": L2RModel,
    "elg": ELG,
    "invit": INViT,
}

AUGMENT = StateAugmentation(
    num_augment=8, augment_fn="dihedral8", first_aug_identity=True
)


def _allow_safe_globals() -> None:
    try:
        torch.serialization.add_safe_globals([SingleSharedBaseline])
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a routing checkpoint at a target size"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a trained checkpoint")
    parser.add_argument("--model", type=str, default="am",
                        choices=sorted(eval_utils.REGISTRY),
                        help="Model class. Must match the checkpoint.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Cached dataset split shared across methods.")
    parser.add_argument("--num_loc", type=int, required=True,
                        help="Target number of customers N.")
    parser.add_argument("--n_inst", type=int, default=1000,
                        help="Number of eval instances")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Eval batch size")
    parser.add_argument("--num_starts", type=int, default=50,
                        help="Multi-start greedy for POMO/AM checkpoints.")
    parser.add_argument("--decode", type=str, default="greedy",
                        choices=["greedy", "sampling", "multistart_greedy",
                                 "multistart_sampling"],
                        help="Decoding strategy wired to every model.")
    parser.add_argument("--augment", action="store_true",
                        help="Evaluate eight geometric augmentations (AM/POMO).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device such as cuda, mps, or cpu")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed so every run uses the same test set.")
    parser.add_argument("--out", type=str, default=None,
                        help="JSON path to save results.")
    args = parser.parse_args()

    _allow_safe_globals()
    pl.seed_everything(args.seed, workers=True)

    args.device = eval_utils.pick_device(args.device)
    env = CVRPEnv(generator_params=dict(num_loc=args.num_loc))
    model_cls = MODEL_CLASSES[args.model]
    model = model_cls.load_from_checkpoint(
        args.ckpt, env=env, map_location=args.device)
    model.to(args.device)
    if args.model == "sil":
        model.labels.clear()
        model.best_policy_state = model.repair_policy_state = None

    if args.data_path:
        ds = eval_utils.load_npz(args.data_path, args.num_loc)
    else:
        ds = env.dataset(batch_size=[args.n_inst])

    result = eval_utils.evaluate(model, args, env, ds)
    result["backbone"] = args.model
    result["num_loc"] = args.num_loc
    result["seed"] = args.seed
    result["ckpt"] = str(args.ckpt)
    result["data_path"] = (
        str(Path(args.data_path).resolve()) if args.data_path else None)

    print(f"[OK] evaluated at num_loc={args.num_loc} customers  "
          f"n={result['n_inst']}  "
          f"mean tour = {result['mean_tour_length']:.4f} "
          f"(mean reward = {result['mean_reward']:.4f})")

    out_path = args.out or Path("results") / \
        f"eval_{args.model}_{args.num_loc}_{args.seed}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
