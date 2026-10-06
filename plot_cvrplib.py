"""Decode selected CVRPLIB Set X instances and render route figures.

Examples:
    python plot_cvrplib.py --ckpt path/to/model.ckpt --model pomo \
        --instances 0,3 --plot_dir figures/cvrplib
    python plot_cvrplib.py --ckpt path/to/model.ckpt --model am \
        --instances X-n101-k25,X-n200-k36
    python plot_cvrplib.py --ckpt path/to/model.ckpt --model pomo \
        --instances all --sizes 101,200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl4co.utils.cvrplib import evaluate_cvrplib


MODEL_NAMES = ("am", "pomo", "pomo_base", "sil", "icam", "l2r",
               "elg", "invit", "radar")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot selected CVRPLIB Set X solutions")
    parser.add_argument("--ckpt", required=True,
                        help="Path to the trained checkpoint")
    parser.add_argument("--model", default="pomo", choices=MODEL_NAMES,
                        help="Model class used by the checkpoint")
    parser.add_argument("--data_dir", default="./data/cvrplib_setX",
                        help="Directory containing setX.pt")
    parser.add_argument("--instances", required=True,
                        help="Comma-separated names or zero-based positions "
                             "in the sorted list; use 'all' for every instance")
    parser.add_argument("--sizes", default=None,
                        help="Optional comma-separated node counts, e.g. 101,200")
    parser.add_argument("--num_starts", type=int, default=100,
                        help="POMO multi-start count")
    parser.add_argument("--device", default="cpu",
                        help="Inference device, such as cpu, cuda, or mps")
    parser.add_argument("--plot_dir", type=Path, default=Path("results/plots"),
                        help="Directory for PDF and PNG route figures")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution")
    args = parser.parse_args()

    selectors = [item.strip() for item in args.instances.split(",")
                 if item.strip()]
    if not selectors:
        parser.error(
            "--instances must contain at least one name, index, or 'all'")

    evaluate_cvrplib(
        args.model,
        args.ckpt,
        data_dir=args.data_dir,
        sizes=args.sizes,
        num_starts=args.num_starts,
        device=args.device,
        plot_instances=selectors,
        plot_dir=args.plot_dir,
        plot_dpi=args.dpi,
        only_instances=selectors,
    )
    print(f"Saved selected route plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
