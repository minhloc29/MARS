from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict

from rl4co.envs import CVRPEnv
from rl4co.utils.eval_utils import _allow_safe_globals
from rl4co.utils.ops import get_tour_length

ACTION_BACKBONES = {"pomo", "am", "pomo_base", "elg", "dgl", "radar"}


def _load_checkpoint_model(ckpt: str, model: str):
    """Reconstruct the CVRP model from a checkpoint (Set X path)."""
    from rl4co.models.zoo.pomo_slot import AMSlot, POMOSlot
    from rl4co.models.zoo.pomo import POMO
    from rl4co.models.zoo.sil import SIL
    from rl4co.models.zoo.icam import ICAMCVRP
    from rl4co.models.zoo.l2r import L2RModel
    from rl4co.models.zoo.elg import ELG
    from rl4co.models.zoo.invit import INViT
    from rl4co.models.zoo.radar import RADAR
    from rl4co.models.zoo.dgl import DGL

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
        "dgl": DGL,
    }
    env = CVRPEnv(generator_kwargs=dict(num_loc=100))
    net = MODEL_CLASSES[model].load_from_checkpoint(ckpt, env=env,
                                                    map_location="cpu")
    net.eval()
    if model == "sil":
        net.labels.clear()
        net.best_policy_state = net.repair_policy_state = None
    return net


def costs_from_actions(env, td_r, actions):
    """Return per-start cost (negative reward) for the decoded actions.

    ``td_r`` is the *reset* td whose ``locs`` is the full node set (depot at
    index 0, then customers). POMO actions are ``(1, n_start, N)``; the tour
    for a start is ``[depot, locs[a_0], locs[a_1], ...]`` with the loop closed
    by ``get_tour_length`` (matches ``env._get_reward``). AM actions are
    ``(1, N)``.
    """
    locs = td_r["locs"]  # (1, N_full, 2)  -- full node set, depot at idx 0
    costs = []
    starts = actions.shape[1] if actions.ndim == 3 else 1
    for s in range(starts):
        acts = actions[0, s] if actions.ndim == 3 else actions[0]  # (N,)
        ordered = torch.cat([locs[:, :1, :], locs[:, acts, :]], dim=1)  # depot + tour
        costs.append(get_tour_length(ordered))       # (1,)
    return torch.stack(costs).squeeze(-1) if starts > 1 else costs[0].squeeze(-1)


def decode_cvrplib(net, env, td, td_reset, model, starts, device):
    """Decode one instance; returns ``(best_cost, actions_or_None)``.

    ``best_cost`` is in normalized [0,1]^2 units. ``actions`` is the decoded
    action sequence (None for backbones that only surface a scalar reward).
    """
    with torch.no_grad():
        if model in ("pomo", "pomo_base", "am"):
            num_starts = starts if model in ("pomo", "pomo_base") else 1
            out = net.policy(td_reset, env, phase="test", num_starts=num_starts)
            actions = out["actions"]
            cost = costs_from_actions(env, td_reset, actions).reshape(-1)
            best = cost.min() if cost.numel() > 1 else cost[0]
            return float(best.item()), actions
        elif model == "sil":
            out = net.policy(td_reset, env, phase="test")
            return float(-out["reward"].reshape(-1).max().item()), None
        elif model in ("icam", "l2r"):
            r, _ = net._rollout(td, sampling=False)
            return float(-r.reshape(-1).max().item()), None
        elif model == "invit":
            out = net.policy(td, phase="test", decode_type="greedy")
            return float(-out["reward"].reshape(-1).max().item()), None
        else:  # elg / dgl / radar
            from rl4co.models.zoo.baseline_cvrp import rollout

            width = net.hparams.pomo_size
            out = rollout(net.policy, td, min(width, td["locs"].size(1)),
                          "greedy", False)
            actions = out["actions"]
            cost = costs_from_actions(env, td_reset, actions).reshape(-1)
            best = cost.min() if cost.numel() > 1 else cost[0]
            return float(best.item()), actions


def build_td(coords, demand, capacity, device):
    """Pack one instance into a pre-reset TensorDict for the CVRP env.

    RL4CO's CVRP env expects ``locs`` = customers only, ``depot`` separate, and
    ``demand`` = customers only (it prepends the depot in ``_reset``). The env
    fixes vehicle_capacity via its generator (default 1.0), so we normalize
    demand by the instance capacity -> demands in (0,1], capacity 1.0 — exactly
    the distribution the synthetic CVRP data was trained with. A ``capacity``
    key is included for backbones that read it directly (icam/l2r).
    """
    coords = coords.to(device)                      # (N, 2)  -- node 0 is the depot
    demand = demand.to(device)                      # (N,)    -- node 0 demand is 0
    depot = coords[:1]                              # (1, 2)
    customers = coords[1:]                          # (N-1, 2)
    dem_cust = demand[1:] / capacity                # (N-1,) normalized, <= 1
    dem_cust = torch.clamp(dem_cust, min=1e-8)      # avoid 0-demand (mask logic edge)
    td = TensorDict(
        {
            "locs": customers.unsqueeze(0),         # (1, N, 2)
            "depot": depot,                         # (1, 2)
            "demand": dem_cust.unsqueeze(0),        # (1, N)
            "capacity": torch.ones(1, device=device),  # normalized capacity
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


def verify_feasible(env, td_r, actions):
    """Verify no route exceeds capacity.

    RL4CO's CVRP decoder capacity-masks during decoding, so returned routes are
    feasible by construction. This re-checks independently by segmenting the
    decoded best route at depot returns and summing each route's (normalized)
    demand against capacity 1.0.
    """
    demand = td_r["demand"][0]                       # (N,) customers-only, normalized
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


def evaluate_cvrplib(model, ckpt, data_dir="./data/cvrplib_setX", sizes=None,
                     num_starts=None, device="cpu") -> dict:
    """Evaluate a checkpoint on the CVRPLIB Set X benchmark.

    Args:
        model: model name (pomo/am/pomo_base/sil/icam/l2r/elg/dgl/invit/radar).
        ckpt: path to a trained checkpoint.
        data_dir: directory holding ``setX.pt``.
        sizes: optional comma string (or iterable) of n to evaluate.
        num_starts: POMO multi-start count (default 100). Ignored by
            elg/dgl/radar (baked pomo_size) and sil/icam/l2r/invit.
        device: torch device string.

    Returns an aggregate dict (n_inst/num_starts/mean_gap/std_gap/feasible/
    mean_cost/instances) and prints a per-instance table.
    """
    if isinstance(sizes, str):
        sizes = [int(s) for s in sizes.split(",") if s.strip()] or None

    data = torch.load(Path(data_dir) / "setX.pt", map_location="cpu",
                      weights_only=False)
    available = {int(v["node_coord"].shape[0]): True for v in data.values()}
    if sizes:
        for s in sizes:
            if s not in available:
                raise RuntimeError(
                    f"Size n={s} not present in Set X. Available: {sorted(available)}")

    _allow_safe_globals()

    net = _load_checkpoint_model(ckpt, model).to(device)
    env = net.env
    starts = num_starts or 100

    hparams = getattr(net, "hparams", None)
    tr_num_loc = getattr(hparams, "num_loc", None) if hparams else None
    tr_pomo = getattr(hparams, "pomo_size", None) if hparams else None
    print(f"[info] checkpoint trained num_loc={tr_num_loc}  pomo_size={tr_pomo}"
          f"  | env generator num_loc="
          f"{getattr(env.generator, 'num_loc', None)}  vehicle_capacity="
          f"{getattr(env.generator, 'vehicle_capacity', None)}")

    results = []
    agg = {"count": 0, "gap_sum": 0.0, "gap_sq": 0.0, "feasible": 0,
           "cost_sum": 0.0}
    print(f"\n=== CVRPLIB Set X eval | {model} | starts_pomo={starts} ===\n")
    print(f"{'instance':<14}{'n':>5}{'k':>5}{'capacity':>9}{'cost':>12}{'bks':>12}"
          f"{'gap%':>9}{'feas':>5}")

    for name in sorted(data, key=lambda k: data[k]["node_coord"].shape[0]):
        rec = data[name]
        n = rec["node_coord"].shape[0]
        if sizes and n not in sizes:
            continue
        bks = rec["best_cost"]
        if bks is None:
            bks = float("nan")

        coords, scale = normalize(
            torch.tensor(rec["node_coord"], dtype=torch.float32))
        demand = torch.tensor(rec["demand"], dtype=torch.float32)
        capacity = rec["capacity"]

        # env.reset() mutates td in place (prepends the depot to locs). Reset a
        # clone so the raw td stays customers-only for the backbones that decode
        # from it directly (icam/l2r/invit/elg/dgl/radar).
        td = build_td(coords, demand, capacity, device)
        td_reset = env.reset(td.clone())
        try:
            cost, actions = decode_cvrplib(
                net, env, td, td_reset, model, starts, device)
        except Exception as e:
            # A decode that violates capacity (or any other error) for a single
            # instance must not abort the whole Set-X run. Flag it infeasible and
            # continue so we can see which instances/sizes are affected.
            print(f"  ! {name}: decode failed -> {type(e).__name__}: {e}")
            results.append({
                "name": name, "n": n, "capacity": capacity,
                "cost": None, "best": round(bks, 2) if bks == bks else None,
                "gap": None, "feasible": False,
            })
            agg["count"] += 1
            continue
        feasible = (verify_feasible(env, td_reset, actions)
                    if actions is not None else True)

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

    return {
        "n_inst": cnt,
        "num_starts": starts,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "feasible": agg["feasible"],
        "mean_cost": agg["cost_sum"] / max(1, cnt),
        "instances": results,
    }
