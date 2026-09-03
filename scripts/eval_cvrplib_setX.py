#!/usr/bin/env python
"""Evaluate a trained CVRP slot checkpoint on the CVRPLIB Set X benchmark.

Set X (Uchoa et al., 2017) contains 100 instances with n from ~100 to ~1000.
This harness loads the cached copy produced by ``download_cvrplib_setX.py``
(``data/cvrplib_setX/setX.pt``), converts each instance into the RL4CO CVRP
env's tensor format, and decodes a solution with a POMO or AM slot
checkpoint, one instance at a time (N varies per instance).

Reported per instance / aggregate:
  * decoded cost (route length) and gap vs the official best-known: (c - bks)/bks
  * feasibility (RL4CO's CVRP decoder capacity-masks, so routes are feasible by
    construction; we still verify loads over every route)
  * mean/std gap, fraction of instances solved, and (optionally) fraction of
    best-known gaps for a size subset.

Limitations (inherent to "evaluate an N=100-trained model on all sizes"):
  * The checkpoint is trained on synthetic uniform [0,1]^2 CVRP at N=100; the
    attention machinery is size-agnostic so bigger instances decode, but
    quality degrades as n grows. Compare the n~100 subset for a fair read.
  * Coordinates are normalized per instance (divide by max coord) to map the
    integer CVRPLIB coords into [0,1]. Demands/capacity are passed through in
    raw integer units (the env's capacity mask is unit-agnostic, so feasible
    routes are still guaranteed).

Run (proxy must be unset for any network; none needed — data is cached):
    python scripts/eval_cvrplib_setX.py --ckpt <best.ckpt> --model pomo|am
        [--data_dir ./data/cvrplib_setX] [--starts N] [--sizes 101,106,...]
        [--out results/setX_eval.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tensordict import TensorDict

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot, AMSlot
from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline


def _allow_safe_globals() -> None:
    try:
        torch.serialization.add_safe_globals([SingleSharedBaseline])
    except Exception:
        pass


def load_checkpoint(ckpt: str, model: str):
    """Reconstruct the CVRP slot model from a checkpoint."""
    env = CVRPEnv(generator_kwargs=dict(num_loc=100))
    model_cls = POMOSlot if model.lower() == "pomo" else AMSlot
    net = model_cls.load_from_checkpoint(ckpt, env=env, map_location="cpu")
    net.eval()
    return net


def build_td(coords, demand, capacity, device):
    """Pack one instance into a pre-reset TensorDict for the CVRP env.

    RL4CO's CVRP env expects ``locs`` = customers only, ``depot`` separate,
    and ``demand`` = customers only (it prepends the depot in ``_reset`` and
    sizes the ``visited`` mask as N_cust+1). The env fixes vehicle_capacity
    via its generator (default 1.0), so we normalize demand by the instance
    capacity -> demands in (0,1], capacity 1.0 — exactly the distribution the
    synthetic CVRP data was trained with.
    """
    coords = coords.to(device)                      # (N, 2)  — node 0 is the depot
    demand = demand.to(device)                      # (N,)    — node 0 demand is 0
    depot = coords[:1]                              # (1, 2)
    customers = coords[1:]                          # (N-1, 2)
    dem_cust = demand[1:] / capacity                # (N-1,) normalized, <= 1
    dem_cust = torch.clamp(dem_cust, min=1e-8)      # avoid 0-demand (mask logic edge)
    N = customers.shape[0]
    td = TensorDict(
        {
            "locs": customers.unsqueeze(0),         # (1, N, 2)
            "depot": depot,                         # (1, 2)
            "demand": dem_cust.unsqueeze(0),        # (1, N)
        },
        batch_size=[1],
    )
    return td


def normalize(coords):
    """Map integer CVRPLIB coords into [0,1] (per-instance).

    Returns ``(coords_norm, scale)`` where ``scale`` is the divisor used; the
    model's decoded tour length is in [0,1]^2 units and must be multiplied by
    ``scale`` to compare against the raw-coordinate best-known cost.
    """
    m = float(coords.max())
    if m <= 0:
        return coords, 1.0
    return coords / m, m


def costs_from_actions(env, td_r, actions):
    """Return per-start cost (negative reward) for the decoded actions.

    ``td_r`` is the *reset* td whose ``locs`` is the full node set (depot at
    index 0, then customers). POMO actions are ``(1, n_start, N)``; the tour
    for a start is ``[depot, locs[a_0], locs[a_1], ...]`` with the loop closed
    by ``get_tour_length`` (matches ``env._get_reward``). AM actions are
    ``(1, N)``.
    """
    from rl4co.utils.ops import get_tour_length

    locs = td_r["locs"]  # (1, N_full, 2)  — full node set, depot at idx 0
    costs = []
    starts = actions.shape[1] if actions.ndim == 3 else 1
    for s in range(starts):
        acts = actions[0, s] if actions.ndim == 3 else actions[0]  # (N,)
        ordered = torch.cat([locs[:, :1, :], locs[:, acts, :]], dim=1)  # depot + tour
        costs.append(get_tour_length(ordered))       # (1,)
    return torch.stack(costs).squeeze(-1) if starts > 1 else costs[0].squeeze(-1)


def decode(net, env, td, starts, device):
    """Decode one instance; returns (best_cost, actions) over all starts."""
    with torch.no_grad():
        if isinstance(net, POMOSlot):
            out = net.policy(td, env, phase="test", num_starts=starts)
        else:
            out = net.policy(td, env, phase="test", num_starts=1)
    actions = out["actions"]  # (1, n_start, N) POMO | (1, N) AM
    cost = costs_from_actions(env, td, actions)  # (n_start,) POMO | () AM
    cost = cost.reshape(-1)
    best = cost.min() if cost.numel() > 1 else cost[0]
    return float(best.item()), actions


def verify_feasible(env, td_r, actions):
    """Verify no route exceeds capacity.

    RL4CO's CVRP decoder capacity-masks during decoding, so returned routes are
    feasible by construction. This re-checks independently by segmenting the
    decoded best route at depot returns and summing each route's (normalized)
    demand against capacity 1.0.
    """
    td_r = td_r  # reset td: locs full (1,N+1,2); demand customers-only (1,N)
    demand = td_r["demand"][0]                       # (N,) customers-only, normalized
    from rl4co.utils.ops import get_tour_length
    locs = td_r["locs"]
    # best route = the start (or single) with minimum cost
    if actions.ndim == 3:
        per = []
        for s in range(actions.shape[1]):
            acts = actions[0, s]
            ordered = torch.cat([locs[:, :1, :], locs[:, acts, :]], dim=1)
            per.append(get_tour_length(ordered))
        per = torch.stack(per)
        acts = actions[0, int(per.argmin())]
    else:
        acts = actions[0]
    # Segment actions at depot returns.
    route = []
    ok = True
    for a in acts.tolist():
        if a == 0:
            if route:
                if sum(demand[i - 1].item() for i in route) > 1.0 + 1e-6:
                    ok = False
                route = []
        else:
            route.append(int(a))
    if route and sum(demand[i - 1].item() for i in route) > 1.0 + 1e-6:
        ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate CVRP slot checkpoint on CVRPLIB Set X")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str, default="pomo", choices=["pomo", "am"])
    ap.add_argument("--data_dir", type=str, default="./data/cvrplib_setX")
    ap.add_argument("--starts", type=int, default=None,
                    help="POMO multi-start count. Default: min(num_loc, 100).")
    ap.add_argument("--sizes", type=str, default=None,
                    help="Comma list of n to evaluate (default: all). e.g. 101,110")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    _allow_safe_globals()
    torch.manual_seed(0)

    data = torch.load(Path(args.data_dir) / "setX.pt", map_location="cpu", weights_only=False)
    sizes = {int(v["node_coord"].shape[0]): True for v in data.values()}
    include = None
    if args.sizes:
        include = {int(s) for s in args.sizes.split(",") if s.strip()}
        for s in include:
            if s not in sizes:
                raise RuntimeError(f"Size n={s} not present in Set X. Available: {sorted(sizes)}")

    net = load_checkpoint(args.ckpt, args.model)
    net = net.to(args.device)
    env = net.env
    starts = args.starts or 100

    results = []
    agg = {"count": 0, "gap_sum": 0.0, "gap_sq": 0.0, "feasible": 0, "cost_sum": 0.0}
    print(f"\n=== CVRPLIB Set X eval | {args.model} | starts_pomo={starts} ===\n")
    print(f"{'instance':<14}{'n':>5}{'k':>5}{'capacity':>9}{'cost':>12}{'bks':>12}"
          f"{'gap%':>9}{'feas':>5}")

    for name in sorted(data, key=lambda k: data[k]["node_coord"].shape[0]):
        rec = data[name]
        n = rec["node_coord"].shape[0]
        if include and n not in include:
            continue
        bks = rec["best_cost"]
        if bks is None:
            bks = float("nan")

        coords, scale = normalize(torch.tensor(rec["node_coord"], dtype=torch.float32))
        demand = torch.tensor(rec["demand"], dtype=torch.float32)
        capacity = rec["capacity"]

        td = build_td(coords, demand, capacity, args.device)
        td_reset = env.reset(td)
        cost, actions = decode(net, env, td_reset, starts, args.device)
        feasible = verify_feasible(env, td_reset, actions)

        # Decoded cost is in [0,1]^2 units; rescale to the raw-coordinate frame
        # that the CVRPLIB best-known cost lives in.
        cost = cost * scale
        gap = (cost - bks) / bks if bks == bks else float("nan")
        k_approx = int(round(float(demand.sum()) / capacity)) if capacity else 0

        results.append({
            "name": name, "n": n, "capacity": capacity, "cost": round(cost, 2),
            "best": round(bks, 2) if bks == bks else None,
            "gap": round(gap * 100, 3) if gap == gap else None,
            "feasible": feasible,
        })

        agg["count"] += 1
        if feasible and (gap == gap):
            agg["gap_sum"] += gap
            agg["gap_sq"] += gap * gap
            agg["feasible"] += 1
        agg["cost_sum"] += cost

        print(f"{name:<14}{n:>5}{k_approx:>5}{capacity:>9.0f}{cost:>12.2f}"
              f"{bks:>12.2f}{(gap*100 if gap==gap else float('nan')):>9.2f}"
              f"{'Y' if feasible else 'N':>5}")

    cnt = agg["count"]
    if cnt:
        mean_gap = agg["gap_sum"] / max(1, agg["feasible"])
        std_gap = (agg["gap_sq"] / max(1, agg["feasible"]) - mean_gap**2) ** 0.5
        print(f"\n--- aggregate ({cnt} instances) ---")
        print(f"  feasible: {agg['feasible']}/{cnt}")
        print(f"  mean gap (vs bks): {mean_gap*100:.2f}%  (std {std_gap*100:.2f}%)")
        print(f"  mean cost: {agg['cost_sum']/cnt:.2f}")
    else:
        mean_gap = std_gap = 0.0

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "ckpt": args.ckpt, "model": args.model, "starts": starts,
            "n": cnt, "mean_gap": mean_gap, "std_gap": std_gap,
            "feasible": agg["feasible"],
            "mean_cost": agg["cost_sum"] / max(1, cnt),
            "instances": results,
        }, indent=2))
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
