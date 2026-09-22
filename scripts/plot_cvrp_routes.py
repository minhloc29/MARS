#!/usr/bin/env python
"""Draw complete CVRP vehicle routes from MARS_plot's LEHD path or route JSON.

LEHD mode plots one decoded route plan beside its HGS reference. JSON mode
accepts outputs from other backbones; see scripts/README_cvrp_figures.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mars-plot-mpl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#172433"
MUTED = "#657384"
GRID = "#E9EEF1"
COLORS = (
    "#DE6848", "#287C88", "#BE8A3C", "#775FA5", "#4F9467",
    "#BC5B83", "#5477B5", "#9A7654", "#729248", "#B66A59",
    "#557B82", "#A374A1", "#977C39", "#6386A8", "#BD6F50",
)


@dataclass(frozen=True)
class Plan:
    label: str
    routes: list[list[int]]  # customer indices 1..N, depot implicit


def routes_from_actions(actions) -> list[list[int]]:
    """Convert RL4CO/MARS actions, where node 0 is a depot return."""
    routes, current = [], []
    for node in np.asarray(actions, dtype=int).reshape(-1).tolist():
        if node == 0:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(node)
    if current:
        routes.append(current)
    return routes


def routes_from_lehd(node_flag) -> list[list[int]]:
    """Convert LEHD [customer, via-depot flag] solution to vehicle routes."""
    solution = np.asarray(node_flag, dtype=int)
    if (solution.ndim != 2 or solution.shape[1] != 2 or not len(solution)
            or solution[0, 1] != 1 or not np.isin(solution[:, 1], [0, 1]).all()):
        raise ValueError("LEHD solution must be Nx2, with first depot flag 1")
    routes, current = [], []
    for node, flag in solution:
        if flag and current:
            routes.append(current)
            current = []
        current.append(int(node))
    routes.append(current)
    return routes


def validate_plan(xy, demands, capacity, routes) -> tuple[float, np.ndarray]:
    """Check coverage and capacity; calculate true depot-to-depot distance."""
    xy = np.asarray(xy, dtype=float)
    demands = np.asarray(demands, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError("coordinates must be (N+1, 2), depot first")
    if (demands.shape != (len(xy),) or not np.isfinite(xy).all()
            or not np.isfinite(demands).all() or (demands < 0).any()
            or abs(demands[0]) > 1e-7 or not np.isfinite(capacity) or capacity <= 0):
        raise ValueError("invalid demands, coordinates, or capacity")
    visited = [int(node) for route in routes for node in route]
    if any(not route for route in routes) or sorted(visited) != list(range(1, len(xy))):
        raise ValueError("routes must visit each customer exactly once")
    loads = np.array([demands[np.asarray(route, dtype=int)].sum() for route in routes])
    if np.any(loads > capacity + 1e-5):
        raise ValueError("route exceeds vehicle capacity")
    length = sum(np.linalg.norm(np.diff(xy[[0, *route, 0]], axis=0), axis=1).sum()
                 for route in routes)
    return float(length), loads


def _map(ax, xy, demands, capacity, routes):
    ax.set_facecolor("#FAFBFC")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    span = max(float(np.ptp(xy, axis=0).max()), 1e-6)
    pad = span * 0.065
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    for i, route in enumerate(routes):
        color = COLORS[i % len(COLORS)]
        path = xy[[0, *route, 0]]
        ax.plot(path[:, 0], path[:, 1], color="white", lw=4.1, alpha=0.88,
                solid_capstyle="round", zorder=1)
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.65, alpha=0.92,
                solid_capstyle="round", zorder=2)
        points = xy[route]
        sizes = 22 + 28 * np.sqrt(np.clip(demands[route] / capacity, 0, 1))
        ax.scatter(points[:, 0], points[:, 1], s=sizes, color=color,
                   edgecolors="white", linewidths=0.75, zorder=3)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=280, marker="D", color="white", zorder=5)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=170, marker="D", color=INK,
               edgecolors="white", linewidths=1.25, zorder=6)
    ax.annotate("DEPOT", xy[0], xytext=(0, -22), textcoords="offset points",
                ha="center", va="top", fontsize=8.5, color=INK, weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.6),
                zorder=7)


def _loads(ax, loads, capacity):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0, 0.94, "VEHICLE LOADS", transform=ax.transAxes, color=MUTED,
            fontsize=8.2, weight="bold", va="top")
    columns = 2 if len(loads) <= 20 else 3
    rows = int(np.ceil(len(loads) / columns))
    for i, load in enumerate(loads):
        col, row = divmod(i, rows)
        x = col / columns + 0.025
        y = 0.79 - row * 0.7 / rows
        width = 0.35 if columns == 2 else 0.2
        ax.text(x, y, f"{i + 1:02d}", transform=ax.transAxes, color=INK,
                fontsize=8, va="center", fontfamily="DejaVu Sans Mono")
        start = x + 0.048
        ax.plot([start, start + width], [y, y], color=GRID, lw=5,
                solid_capstyle="round", transform=ax.transAxes)
        ax.plot([start, start + width * load / capacity], [y, y],
                color=COLORS[i % len(COLORS)], lw=5, solid_capstyle="round",
                transform=ax.transAxes)
        ax.text(start + width + 0.012, y, f"{load:g}/{capacity:g}",
                transform=ax.transAxes, fontsize=7.7, color=MUTED, va="center")


def plot_instance(xy, demands, capacity, plans: list[Plan], output: Path,
                  title="CVRP route plans", note="") -> list[float]:
    """Write a PNG/PDF figure of one or two plans for the same instance."""
    if not 1 <= len(plans) <= 2:
        raise ValueError("plot accepts one or two plans")
    xy = np.asarray(xy, dtype=float)
    demands = np.asarray(demands, dtype=float)
    metrics = [validate_plan(xy, demands, capacity, plan.routes) for plan in plans]
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2 * len(plans), 9), facecolor="white")
    outer = fig.add_gridspec(1, len(plans), left=0.045, right=0.955, top=0.82,
                             bottom=0.07, wspace=0.105)
    for i, (plan, (distance, loads)) in enumerate(zip(plans, metrics)):
        pane = outer[0, i].subgridspec(3, 1, height_ratios=[0.68, 5.8, 1.25],
                                       hspace=0.075)
        header = fig.add_subplot(pane[0])
        header.axis("off")
        header.text(0, 0.84, f"{i + 1:02d}  /  {plan.label.upper()}",
                    color=INK, fontsize=12.5, weight="bold", va="top")
        subtitle = f"{len(plan.routes)} vehicles   ·   {distance:,.2f} distance"
        if len(plans) == 2 and i == 0 and metrics[1][0] > 0:
            subtitle += f"   ·   {(distance / metrics[1][0] - 1) * 100:+.1f}% vs reference"
        header.text(0, 0.2, subtitle, color=MUTED, fontsize=9.5, va="center")
        _map(fig.add_subplot(pane[1]), xy, demands, capacity, plan.routes)
        _loads(fig.add_subplot(pane[2]), loads, capacity)
    fig.text(0.045, 0.955, title, color=INK, fontsize=23, weight="bold", va="top")
    fig.text(0.045, 0.902,
             f"{len(xy) - 1} customers   /   capacity {capacity:g}   /   "
             "color = vehicle route   /   circle size = demand",
             color=MUTED, fontsize=10.3, va="top")
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.855, 0.855],
                              transform=fig.transFigure, color=GRID, lw=1.4))
    if note:
        fig.text(0.955, 0.032, note, color=MUTED, fontsize=8, ha="right")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=230, facecolor="white")
    plt.close(fig)
    return [distance for distance, _ in metrics]


def load_json(path: Path):
    data = json.loads(path.read_text())
    plans = []
    for solution in data["solutions"]:
        if ("routes" in solution) == ("actions" in solution):
            raise ValueError("each solution needs exactly one of routes or actions")
        routes = solution["routes"] if "routes" in solution else routes_from_actions(solution["actions"])
        plans.append(Plan(str(solution["label"]), routes))
    return (np.asarray(data["coordinates"], dtype=float),
            np.asarray(data["demands"], dtype=float), float(data["capacity"]),
            plans, str(data.get("title", "CVRP route plans")))


def effective_lehd_flag(flag, remaining, demand):
    """Record the depot return actually required by the LEHD env transition.

    The newer env repairs the teacher flag on insufficient capacity but leaves
    ``selected_student_flag`` unchanged. Plot the physically executed route,
    not that stale student flag, and still validate every vehicle load.
    """
    import torch

    return torch.where(remaining < demand, torch.ones_like(flag), flag)


def decode_lehd(checkpoint: Path, env, device: str):
    """Match the newer MARS_plot LEHD validation/test first-node convention."""
    import torch
    from rl4co.models.zoo.lehd import LEHDModel

    model = LEHDModel.load_from_checkpoint(str(checkpoint), map_location=device)
    model = model.to(device).eval()
    env.problems = env.problems.to(device)
    env.solution = env.solution.to(device)
    env.reset("test")
    state, _, _, done = env.pre_step()
    capacity = float(env.raw_data_capacity[0].item())
    step = 0
    with torch.inference_mode():
        while not done:
            if step == 0:
                node = env.solution[:, 0, 0].long()
                flag = env.solution[:, 0, 1].long()
            else:
                probs = model.model.decode_step(state.problems, env.selected_node_list,
                                                capacity, state.problems[:, 0, 3], step)
                n = state.problems.shape[1] - 1
                choice = probs.argmax(dim=1)
                flag = (choice >= n).long()
                node = torch.where(choice >= n, choice - n + 1, choice + 1).long()
            demand = state.problems.gather(
                1, node[:, None, None].expand(-1, 1, 4))[:, 0, 2]
            flag = effective_lehd_flag(flag, state.problems[:, 0, 3], demand)
            state, _, _, done = env.step(node, node, flag, flag)
            step += 1
    return torch.stack([env.selected_student_list[0], env.selected_student_flag[0]],
                       dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--lehd-data", type=Path, help="LEHD .txt instance file")
    source.add_argument("--input-json", type=Path, help="coordinates/demands/routes JSON")
    parser.add_argument("--checkpoint", type=Path, help="LEHD checkpoint")
    parser.add_argument("--index", type=int, default=0, help="zero-based LEHD instance row")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True, help="PNG or PDF path")
    args = parser.parse_args()
    if args.index < 0:
        parser.error("--index must be nonnegative")
    if args.input_json:
        if args.checkpoint:
            parser.error("--checkpoint applies only to --lehd-data")
        xy, demand, capacity, plans, title = load_json(args.input_json)
        note = "MARS_plot · decoded CVRP routes"
    else:
        from rl4co.models.zoo.lehd._env import LEHDVRPEnv

        env = LEHDVRPEnv(str(args.lehd_data), mode="test", sub_path=False)
        env.load_raw_data(args.index + 1)
        env.load_problems(args.index, 1)
        xy = env.problems[0, :, :2].cpu().numpy().copy()
        demand = env.problems[0, :, 2].cpu().numpy().copy()
        capacity = float(env.raw_data_capacity[args.index].item())
        plans = []
        if args.checkpoint:
            plans.append(Plan("LEHD · greedy", routes_from_lehd(
                decode_lehd(args.checkpoint, env, args.device))))
        plans.append(Plan("HGS · reference", routes_from_lehd(env.solution[0].cpu().numpy())))
        title = "A CVRP instance, two route plans" if args.checkpoint else "CVRP reference routes"
        note = f"MARS_plot · LEHD data · instance {args.index} · Euclidean distance"
    distances = plot_instance(xy, demand, capacity, plans, args.output, title, note)
    for plan, distance in zip(plans, distances):
        print(f"{plan.label}: {distance:.4f} distance, {len(plan.routes)} vehicles")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
