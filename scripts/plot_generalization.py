"""Plot illustrative CVRP shifts along cardinality and spatial-distribution axes.

The figure contains four panels:
  (a) uniform CVRP-100, (b) uniform CVRP-500,
  (c) uniform CVRP-100, (d) clustered CVRP-100.

The routes are feasible greedy reference routes intended for illustration, not
for reporting solution quality.
"""

from __future__ import annotations
from tensordict import TensorDict
import torch
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

import argparse
from pathlib import Path
from rl4co.models.zoo.elg import ELG
from rl4co.models.zoo.icam import ICAMCVRP
from rl4co.models.zoo.invit import INViT
from rl4co.models.zoo.l2r import L2RModel
from rl4co.models.zoo.pomo import POMO
from rl4co.models.zoo.pomo_slot import AMSlot, POMOSlot
from rl4co.models.zoo.radar import RADAR
from rl4co.models.zoo.sil import SIL
from rl4co.utils import eval_utils
import matplotlib
matplotlib.use("Agg")

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


DEPOT_COLOR = "#B2182B"
CUSTOMER_EDGE = "#263746"

ROUTE_PALETTE = (
    "#005A8D",  # blue
    "#B43C00",  # vermillion
    "#007A59",  # green
    "#A33B79",  # magenta
    "#542C83",  # purple
    "#704032",  # brown
    "#075E6B",  # teal
    "#B36B00",  # orange
    "#333333",  # dark gray
    "#214579",  # deep blue
)


def make_points(n: int, distribution: str, rng: np.random.Generator):
    if distribution == "uniform":
        points = rng.uniform(0.04, 0.96, size=(n, 2))
    else:
        centers = np.array([[0.20, 0.22], [0.78, 0.22],
                            [0.22, 0.78], [0.78, 0.78]])
        cluster = rng.integers(0, len(centers), size=n)
        points = centers[cluster] + rng.normal(0.0, 0.045, size=(n, 2))
        points = np.clip(points, 0.04, 0.96)
    depot = np.array([0.50, 0.50])
    demand = rng.integers(1, 10, size=n)
    return depot, points, demand


def greedy_routes(depot: np.ndarray, points: np.ndarray, demand: np.ndarray):
    """Build feasible depot-to-depot routes with a nearest-feasible heuristic."""
    capacity = 50 if len(points) <= 100 else 100
    remaining = set(range(len(points)))
    routes = []

    while remaining:
        route = [depot]
        load = 0
        current = depot
        while remaining:
            feasible = [i for i in remaining if load +
                        int(demand[i]) <= capacity]
            if not feasible:
                break
            next_customer = min(
                feasible,
                key=lambda i: float(np.linalg.norm(points[i] - current)),
            )
            route.append(points[next_customer])
            load += int(demand[next_customer])
            current = points[next_customer]
            remaining.remove(next_customer)
        route.append(depot)
        routes.append(np.asarray(route))
    return routes


def route_color(index: int, count: int) -> tuple[float, float, float]:
    return ROUTE_PALETTE[index % len(ROUTE_PALETTE)]


def draw_decoded_routes(ax, td, actions):
    from rl4co.utils.cvrp_plot import best_tour

    locs = td["locs"][0].numpy()
    tour = best_tour(td, actions).tolist()
    route_count = max(1, sum(node == 0 for node in tour) - 1)
    segment = [tour[0]]
    route_index = 0
    for node in tour[1:]:
        segment.append(node)
        if node == 0:
            route = locs[segment]
            ax.plot(route[:, 0], route[:, 1],
                    color=route_color(route_index, route_count),
                    linewidth=0.8 if len(locs) > 101 else 1.35,
                    alpha=0.88, zorder=1)
            route_index += 1
            segment = [0]
    return locs


def draw_panel(ax, title: str, n: int, distribution: str,
               rng: np.random.Generator, solution: dict | None = None):
    if solution is None:
        depot, points, demand = make_points(n, distribution, rng)
        routes = greedy_routes(depot, points, demand)
        locs = np.concatenate((depot[None], points), axis=0)
        for route_index, route in enumerate(routes):
            ax.plot(route[:, 0], route[:, 1],
                    color=route_color(route_index, len(routes)),
                    linewidth=0.8 if n > 100 else 1.35, alpha=0.88, zorder=1)
    else:
        locs = draw_decoded_routes(ax, solution["td"], solution["actions"])

    ax.scatter(locs[1:, 0], locs[1:, 1], s=20 if n > 100 else 38,
               facecolors="white", edgecolors=CUSTOMER_EDGE,
               linewidths=0.7 if n > 100 else 1.0, zorder=2)
    ax.scatter(*locs[0], marker="*", s=190, color=DEPOT_COLOR,
               edgecolors="white", linewidths=1.1, zorder=3)

    ax.text(0.0, -0.05, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=13, fontweight="semibold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_batch(n: int, distribution: str, seed: int) -> TensorDict:
    rng = np.random.default_rng(seed)
    depot, points, demand = make_points(n, distribution, rng)
    capacity = 50 if n <= 100 else 100
    return TensorDict({
        "locs": torch.tensor(points, dtype=torch.float32).unsqueeze(0),
        "depot": torch.tensor(depot, dtype=torch.float32).unsqueeze(0),
        "demand": (torch.tensor(demand, dtype=torch.float32) / capacity).unsqueeze(0),
        "capacity": torch.ones(1),
    }, batch_size=[1])


def decode_panel(ckpt: str, model_name: str, n: int, distribution: str,
                 seed: int, device: str, num_starts: int) -> dict:
    from types import SimpleNamespace

    from rl4co.envs import CVRPEnv
    from rl4co.utils import eval_utils

    env = CVRPEnv(generator_params=dict(num_loc=n))
    model = MODEL_CLASSES[model_name].load_from_checkpoint(
        ckpt, env=env, map_location=device)
    model.to(device).eval()
    args = SimpleNamespace(
        model=model_name, num_loc=n, decode="greedy", num_starts=num_starts,
        augment=False, batch_size=1, device=torch.device(device),
    )
    _, instances = eval_utils.evaluate(
        model, args, env, make_batch(n, distribution, seed), return_instances=1)
    if not instances or instances[0].get("actions") is None:
        raise RuntimeError(f"{model_name} did not return actions for plotting")
    return instances[0]


def make_figure(seed: int = 1234, solutions: list[dict] | None = None):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 4, figsize=(
        15.2, 4.2), constrained_layout=False)
    fig.patch.set_facecolor("white")

    panels = [
        ("(a) Cardinality: N=100", 100, "uniform"),
        ("(b) Cardinality: N=500", 500, "uniform"),
        ("(c) Distribution: uniform", 100, "uniform"),
        ("(d) Distribution: clustered", 100, "clustered"),
    ]
    for index, (title, n, distribution) in enumerate(panels):
        solution = solutions[index] if solutions is not None else None
        draw_panel(axes[index], title, n, distribution, rng, solution)

    fig.text(0.25, 0.09, "Cardinality shift", ha="center", va="top",
             fontsize=15, fontweight="bold")
    fig.text(0.75, 0.09, "Spatial-distribution shift", ha="center", va="top",
             fontsize=15, fontweight="bold")
 
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.22,
                        wspace=0.14)
    return fig

#python scripts/plot_generalization.py --ckpt output\pomo_slot_D_K8_N100_uniform_construction_seed0\checkpoints\best-epoch=050-val\reward=-16.8241.ckpt --model pomo

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint decoded through test.py's evaluation path.")
    parser.add_argument("--model", type=str, default="pomo",
                        choices=["am", "pomo", "pomo_base", "elg", "radar"])
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num_starts", type=int, default=500)
    parser.add_argument("--output", type=Path,
                        default=Path("figures/cvrp_generalization_axes"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    solutions = None
    if args.ckpt:
        panel_specs = [(100, "uniform"), (500, "uniform"),
                       (100, "uniform"), (100, "clustered")]
        solutions = [
            decode_panel(args.ckpt, args.model, n, distribution,
                         args.seed + index, args.device, args.num_starts)
            for index, (n, distribution) in enumerate(panel_specs)
        ]
    figure = make_figure(args.seed, solutions)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight",
                   facecolor="white")
    figure.savefig(args.output.with_suffix(".png"), dpi=args.dpi,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {args.output.with_suffix('.pdf')}")
    print(f"Saved {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
