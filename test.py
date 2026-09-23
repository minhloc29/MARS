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
from rl4co.models.zoo.radar import RADAR
from rl4co.models.zoo.sil import SIL
from rl4co.utils import eval_utils
from rl4co.utils.cvrplib import evaluate_cvrplib

MODEL_CLASSES = {
    "am": AMSlot,
    "pomo": POMOSlot,
    "pomo_base": POMO,
    "sil": SIL,
    "icam": ICAMCVRP,
    "l2r": L2RModel,
    "elg": ELG,
    "invit": INViT,
    "radar": RADAR,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a routing checkpoint at a target size"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a trained checkpoint")
    parser.add_argument("--model", type=str, default="am",
                        choices=sorted(MODEL_CLASSES),
                        help="Model class. Must match the checkpoint.")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "cvrplib"])
    parser.add_argument("--data_path", type=str, default=None,
                        help="Cached dataset split shared across methods.")
    parser.add_argument("--data_dir", type=str, default="./data/cvrplib_setX",
                        help="Directory holding setX.pt (only for "
                             "--dataset cvrplib).")
    parser.add_argument("--num_loc", type=int, default=None,
                        help="Target number of customers N (required for "
                             "--dataset synthetic).")
    parser.add_argument("--n_inst", type=int, default=1000,
                        help="Number of eval instances")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Eval batch size")
    parser.add_argument("--num_starts", type=int, default=50,
                        help="Multi-start greedy for POMO/AM checkpoints.")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma list of n to evaluate on CVRPLIB Set X "
                             "(default: all). e.g. 101,110")
    parser.add_argument("--decode", type=str, default="greedy",
                        choices=["greedy", "sampling", "multistart_greedy",
                                 "multistart_sampling", "beam_search"],
                        help="Decoding strategy wired to every model.")
    parser.add_argument("--plot", type=int, default=4)
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Output directory for --plot figures "
                             "(default: results/plots).")
    parser.add_argument("--augment", action="store_true",
                        help="Evaluate eight geometric augmentations (AM/POMO).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device such as cuda, mps, or cpu")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed so every run uses the same test set.")
    parser.add_argument("--out", type=str, default=None,
                        help="JSON path to save results.")
    args = parser.parse_args()

    # ---- dataset / model compatibility ----
    if args.dataset == "synthetic" and args.num_loc is None:
        parser.error("--dataset synthetic requires --num_loc.")

    eval_utils._allow_safe_globals()

    pl.seed_everything(args.seed, workers=True)

    args.device = eval_utils.pick_device(args.device)

    # ---- CVRPLIB Set X: shared per-instance harness (rl4co/utils/cvrplib.py)
    if args.dataset == "cvrplib":
        result = evaluate_cvrplib(
            args.model, args.ckpt, data_dir=args.data_dir,
            sizes=args.sizes, num_starts=args.num_starts,
            device=str(args.device))
        result["backbone"] = args.model
        result["seed"] = args.seed
        result["ckpt"] = str(args.ckpt)
        result["sizes"] = args.sizes
        result["data_dir"] = str(Path(args.data_dir).resolve())

        print(f"[OK] Set X eval: n_inst={result['n_inst']}  "
              f"mean gap = {result['mean_gap']*100:.2f}% "
              f"(std {result['std_gap']*100:.2f}%)  "
              f"feasible {result['feasible']}/{result['n_inst']}  "
              f"mean cost = {result['mean_cost']:.2f}")

        out_path = args.out or Path("results") / \
            f"setX_{args.model}_{args.seed}.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
        print(f"Saved to {out_path}")
        return

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

    if args.plot:
        result, plot_instances = eval_utils.evaluate(
            model, args, env, ds, return_instances=args.plot)
    else:
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
          f"(mean reward = {result['mean_reward']:.4f})  "
          f"inference = {result['elapsed_seconds']:.2f}s "
          f"({result['throughput_per_sec']:.0f} inst/s)")

    if args.plot:
        from rl4co.utils.cvrp_plot import render_cvrp_solution

        plot_dir = Path(args.plot_dir) if args.plot_dir \
            else Path("results") / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        if not plot_instances:
            print(f"[warn] --plot requested but no per-instance actions were "
                  f"collected (backbone {args.model} or --augment) — skipping.")
        rendered = 0
        for i, inst in enumerate(plot_instances):
            if rendered >= args.plot:
                break
            render_cvrp_solution(
                inst["td"], inst["actions"],
                plot_dir / f"{args.model}_inst{i}")
            rendered += 1
        if rendered < args.plot:
            print(f"[warn] --plot {args.plot} but only {rendered} instance(s) "
                  f"had actions to render.")
        print(f"[OK] rendered {rendered} route plot(s) -> {plot_dir}")

    out_path = args.out or Path("results") / \
        f"eval_{args.model}_{args.num_loc}_{args.seed}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
