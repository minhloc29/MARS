"""Publication-quality CVRP route renderer.

Draws a single decoded CVRP solution as a clean, conference-paper-style figure
(depot + customer markers, muted per-route lines) to both vector PDF and
high-DPI PNG. matplotlib is imported lazily so modules that only *import* this
file do not require it — it is only needed when a plot is actually produced.
"""

from __future__ import annotations

from pathlib import Path

import torch

from rl4co.utils.ops import get_tour_length


def best_tour(td_reset, actions):
    """Return the depot-inclusive index sequence for the best-cost start.

    ``actions`` may carry multiple starts (POMO/ELG): shape ``(steps,)`` for a
    single start, or ``(n_start, steps)``. We pick the start whose tour is
    shortest, so the drawn route always matches the reported (min) cost.

    Post-reset ``td_reset["locs"]`` has the depot at index 0, so action values
    already index coordinates directly (0 = depot, 1..N = customers).

    Returns a long tensor ``[0, a_0, a_1, ..., 0]`` (starting and ending at the
    depot).
    """
    locs = td_reset["locs"]                     # (1, N_full, 2) depot-first
    acts = actions.detach()
    if acts.ndim == 1:
        acts = acts.unsqueeze(0)
    starts = acts.shape[0]
    best = None
    best_len = float("inf")
    for s in range(starts):
        seq = acts[s]
        ordered = torch.cat([locs[:, :1, :], locs[:, seq, :]], dim=1)
        length = float(get_tour_length(ordered)[0].item())
        if length < best_len:
            best_len = length
            best = seq
    # Wrap with the depot at both ends (get_tour_length connects the last node
    # back to the first, so the leading depot is implicit; we duplicate it so
    # the polyline visibly returns to the depot).
    tour = torch.cat([torch.zeros(1, dtype=best.dtype), best])   # [0, a_0, ...]
    return torch.cat([tour, torch.zeros(1, dtype=best.dtype)])   # [..., 0]


def render_cvrp_solution(td_reset, actions, out_stem, dpi=300):
    """Draw one CVRP solution and save ``<out_stem>.pdf`` and ``<out_stem>.png``.

    Args:
        td_reset: post-reset TensorDict (batch 1) with ``locs`` = [depot, *customers].
        actions: decoded action sequence (may be multistart; best start is drawn).
        out_stem: output path without extension (e.g. ``results/plots/am_inst0``).
        dpi: raster resolution for the PNG.

    Returns the matplotlib Figure (caller may close it).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tour = best_tour(td_reset, actions).tolist()
    locs = td_reset["locs"][0].cpu().numpy()      # (N_full, 2), depot first
    tour_xy = locs[tour]                          # (len(tour), 2)

    fig, ax = plt.subplots(figsize=(5, 5))
    # Muted, publication-friendly palette.
    depot_color = "#222222"
    cust_color = "#4C72B0"
    route_colors = ["#8C8C8C", "#C44E52", "#55A868", "#CCB974", "#8172B2",
                    "#64B5CD", "#DD8452"]

    # ---- routes: split the tour at depot returns, each as a thin polyline ----
    seg = []
    color_idx = 0
    for a in tour:
        if a == 0:
            if len(seg) > 1:
                pts = locs[seg]
                ax.plot(pts[:, 0], pts[:, 1],
                        color=route_colors[color_idx % len(route_colors)],
                        lw=1.4, alpha=0.85, zorder=2)
                color_idx += 1
            seg = [a]
        else:
            seg.append(a)
    if len(seg) > 1:
        pts = locs[seg]
        ax.plot(pts[:, 0], pts[:, 1],
                color=route_colors[color_idx % len(route_colors)],
                lw=1.4, alpha=0.85, zorder=2)

    # ---- nodes ----
    # Customers first (under the depot marker).
    ax.scatter(locs[1:, 0], locs[1:, 1], s=18, color=cust_color,
               edgecolors="white", linewidths=0.4, zorder=3)
    # Depot as a distinct filled square.
    ax.scatter(locs[0, 0], locs[0, 1], s=64, marker="s", color=depot_color,
               zorder=4)

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.margins(0.06)
    fig.set_facecolor("white")

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight",
                facecolor="white", transparent=False)
    fig.savefig(str(out) + ".png", dpi=dpi, bbox_inches="tight",
                facecolor="white", transparent=False)
    return fig
