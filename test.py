from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
from tensordict import TensorDict

from rl4co.data.transforms import StateAugmentation
from rl4co.envs import CVRPEnv
from rl4co.models.zoo.dgl import DGL
from rl4co.models.zoo.elg import ELG
from rl4co.models.zoo.icam import ICAMCVRP
from rl4co.models.zoo.invit import INViT
from rl4co.models.zoo.l2r import L2RModel
from rl4co.models.zoo.lehd import LEHDModel, TTRLModel
from rl4co.models.zoo.lehd._env import LEHDVRPEnv
from rl4co.models.zoo.lehd.model import CAPACITY_MAP
from rl4co.models.zoo.pomo import POMO
from rl4co.models.zoo.pomo_slot import AMSlot, POMOSlot
from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline
from rl4co.models.zoo.radar import RADAR
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
    "radar": RADAR,
    "dgl": DGL,
    "lehd": LEHDModel,
    "ttpl": TTRLModel,
}

AUGMENT = StateAugmentation(
    num_augment=8, augment_fn="dihedral8", first_aug_identity=True
)


def _allow_safe_globals() -> None:
    try:
        torch.serialization.add_safe_globals(
            [SingleSharedBaseline])
    except Exception:
        pass


def _make_lehd_synthetic(n_inst, num_loc, dist, capacity, seed, device):
    """Generate in-memory LEHD ``problems`` tensors for synthetic CVRP instances.

    LEHD's ``problems`` tensor is (B, V+1, 4) with cols [x, y, demand,
    remaining_capacity] and index 0 = depot. We build it directly from a
    distribution so LEHD can be evaluated on uniform/gaussian CVRP
    (100/200/500/1000) without needing a LEHD-format ``.txt`` / HGS solution.

    Demands are integer 1..9 (standard NCO demand), coords in [0,1]^2.

    Returns:
        problems: (B, V+1, 4) float32 on ``device``
        capacity: int
    """
    g = torch.Generator(device=device).manual_seed(seed)
    if dist == "uniform":
        nodes = torch.rand(n_inst, num_loc + 1, 2, generator=g, device=device)
    else:  # gaussian — rotated Normal(0.5, 0.2), folded back into [0,1)
        nodes = 0.5 + 0.2 * torch.randn(n_inst, num_loc + 1, 2,
                                        generator=g, device=device)
        theta = 2 * torch.pi * torch.rand(n_inst, generator=g, device=device)
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.zeros(n_inst, 2, 2, device=device)
        R[:, 0, 0], R[:, 0, 1] = c, -s
        R[:, 1, 0], R[:, 1, 1] = s, c
        nodes = torch.einsum("bij,bjk->bik", nodes, R)
        nodes = nodes % 1.0  # fold back to [0, 1)

    demand = torch.randint(1, 10, (n_inst, num_loc + 1),
                           generator=g, device=device).float()
    demand[:, 0] = 0.0  # depot demand = 0

    problems = torch.cat([nodes, demand[:, :, None]], dim=2)   # (B, V+1, 3)
    cap = torch.full((n_inst, num_loc + 1, 1), float(capacity), device=device)
    problems = torch.cat([problems, cap], dim=2)                # (B, V+1, 4)
    return problems, capacity


def _prepare_lehd_env(model, args, device):
    """Return a ready-to-decode (env, capacity).

    Data source is chosen automatically:
      * ``--data_path *.txt``  -> LEHD-format file (with HGS node_flag).
      * ``--data_path *.npz``  -> rl4co TensorDict (from scripts/gen_data.py):
        we reuse its coordinates and draw LEHD-style integer demand 1..9 with
        the checkpoint's capacity (the model normalizes demand/capacity itself).
      * ``--dist <dist>``      -> synthetic uniform/gaussian instances.
    """
    env = LEHDVRPEnv(data_path=args.data_path or "", mode="test", sub_path=False)

    if args.data_path and args.data_path.lower().endswith(".npz"):
        return _prepare_lehd_env_from_npz(env, args, device)

    if args.data_path:
        env.load_raw_data(args.n_inst)
        env.load_problems(0, args.n_inst)
        return env, float(env.raw_data_capacity[0].item())

    # Synthetic path: build problems in-memory + a placeholder solution. The
    # solution only anchors the tour start; the reported length is start-invariant.
    num_loc = args.num_loc
    capacity = CAPACITY_MAP.get(num_loc, 50)
    problems, capacity = _make_lehd_synthetic(
        args.n_inst, num_loc, args.dist, capacity, args.seed, device)
    env.problems = problems
    env.problem_size = num_loc
    env.batch_size = args.n_inst
    env.raw_data_capacity = torch.full(
        (args.n_inst,), float(capacity), device=device)
    env.solution = torch.ones(args.n_inst, num_loc, 2, dtype=torch.long, device=device)
    env.solution[:, :, 1] = 0
    return env, float(capacity)


def _prepare_lehd_env_from_npz(env, args, device):
    """Build LEHD env from an rl4co .npz (scripts/gen_data.py output)."""
    from rl4co.utils import eval_utils

    td = eval_utils.load_npz(args.data_path, args.num_loc)  # -> TensorDict
    B = td["locs"].shape[0]
    num_loc = td["locs"].shape[-2]
    capacity = CAPACITY_MAP.get(num_loc, 50)

    # problems = (B, V+1, 4) [x, y, demand, remaining_capacity], index 0 = depot.
    nodes = torch.cat([td["depot"].unsqueeze(1), td["locs"]], dim=1)   # (B, V+1, 2)
    # LEHD uses integer demand 1..9 with its own capacity (it normalizes internally).
    g = torch.Generator(device=device).manual_seed(args.seed)
    demand = torch.randint(1, 10, (B, num_loc), generator=g, device=device).float()
    demand = torch.cat([torch.zeros(B, 1, device=device), demand], dim=1)  # depot = 0
    problems = torch.cat([nodes.to(device), demand[:, :, None]], dim=2)
    cap = torch.full((B, num_loc + 1, 1), float(capacity), device=device)
    problems = torch.cat([problems, cap], dim=2)

    env.problems = problems.to(device)
    env.problem_size = num_loc
    env.batch_size = B
    env.raw_data_capacity = torch.full((B,), float(capacity), device=device)
    env.solution = torch.ones(B, num_loc, 2, dtype=torch.long, device=device)
    env.solution[:, :, 1] = 0
    return env, float(capacity)


def _lehd_student_tour_length(problems, node_list, flag):
   
    B, V = node_list.shape
    coords = problems[:, :, :2]                       # (B, V+1, 2)
    depot = coords[:, [0], :]                          # (B, 1, 2)  depot at index 0
    cust = coords.gather(
        1, node_list.clamp(1, V).unsqueeze(2).expand(B, V, 2))   # (B, V, 2)

    prev = depot                                       # start at the depot (B, 1, 2)
    total = cust.new_zeros(B)
    for i in range(V):
        c = cust[:, [i], :]                            # (B, 1, 2) current customer
        # Direct leg: travel straight from the previous stop to this customer.
        d_to_c = (c - prev).squeeze(1).norm(dim=1)     # (B,)
        # Via-depot leg: return to the depot, then drive out to the customer.
        d_via = ((depot - prev).squeeze(1).norm(dim=1)
                 + (c - depot).squeeze(1).norm(dim=1))  # (B,)
        flag_i = flag[:, i].bool()                     # (B,) per-instance
        total = total + torch.where(flag_i, d_via, d_to_c)
        prev = c
    total = total + (depot - prev).squeeze(1).norm(dim=1)  # final return to depot
    return total


def _evaluate_lehd(model, args, device) -> dict:
    """Greedy decoding for LEHD / TTPL checkpoints (their own env + dataloader).

    LEHD drives its own ``LEHDVRPEnv`` and decodes one action per step via
    ``model.model.decode_step``. It consumes either a LEHD-format ``.txt`` test
    file (``--data_path``) or freshly generated uniform/gaussian instances
    (``--dist``). We mirror the model's own ``validation_step`` loop so numbers
    are directly comparable.
    """
    model.eval()
    dev = torch.device(device)

    env, capacity = _prepare_lehd_env(model, args, dev)
    in_memory = (args.data_path is None) or args.data_path.lower().endswith(".npz")
    total = env.problems.shape[0] if in_memory else args.n_inst
    # Keep the full in-memory tensors; slice from these each batch (env.problems
    # is mutated/reassigned as we decode, so never slice from it directly).
    full_problems = env.problems
    full_solution = env.solution

    rewards = []
    batch_size = args.batch_size
    t0 = time.perf_counter()
    with torch.no_grad():
        episode = 0
        while episode < total:
            bs = min(batch_size, total - episode)
            if in_memory:
                env.problems = full_problems[episode: episode + bs]
                env.solution = full_solution[episode: episode + bs]
            else:
                env.load_problems(episode, bs)
                env.problems = env.problems.to(dev)
                env.solution = env.solution.to(dev)

            env.reset("test")
            state, _, _, done = env.pre_step()

            step = 0
            while not done:
                if step == 0:
                    # Anchor the tour start on the solution's first node (as the
                    # original validation does); tour length is start-invariant.
                    selected = env.solution[:, 0, 0].long()
                    sel_flag = env.solution[:, 0, 1].long()
                    step += 1
                    state, _, _, done = env.step(
                        selected, selected, sel_flag, sel_flag)
                    continue

                remaining_cap = state.problems[:, 0, 3]
                probs = model.model.decode_step(
                    state.problems, env.selected_node_list,
                    capacity, remaining_cap, step)
                V = state.problems.shape[1] - 1
                flat = probs.argmax(dim=1)
                is_via = flat >= V
                sel_node = torch.where(
                    is_via, flat - V + 1, flat + 1).long()
                sel_flag = is_via.long()
                step += 1
                state, _, r_stud, done = env.step(
                    sel_node, sel_node, sel_flag, sel_flag)

            # True route length from the full coordinates (LEHD's own reward
            # degenerates to 0 for flag-free routes). env.problems here is the
            # full batch slice with the depot at index 0 and intact coords.
            length = _lehd_student_tour_length(
                env.problems,
                env.selected_node_list[:, :step],
                env.selected_student_flag[:, :step],
            )
            rewards.append(length.detach().cpu())
            episode += bs
    elapsed = time.perf_counter() - t0

    tour_len = torch.cat(rewards)   # positive distances (higher = longer/better)
    return {
        "n_inst": int(tour_len.numel()),
        "mean_reward": float(-tour_len.mean()),
        "mean_tour_length": float(tour_len.mean()),
        "std_tour_length": float(tour_len.std()) if tour_len.numel() > 1 else 0.0,
        "num_starts": 1,
        "capacity": float(capacity),
        "dist": args.dist or ("file" if args.data_path else None),
        "elapsed_seconds": elapsed,
        "throughput_per_sec": float(r.numel() / elapsed) if elapsed > 0 else 0.0,
    }


ACTION_BACKBONES = {"pomo", "am", "pomo_base", "elg", "dgl", "radar"}
ROLLOUT_BACKBONES = {"elg", "dgl", "radar"}
RESET_TD_BACKBONES = {"pomo", "am", "pomo_base"}


def _load_checkpoint_cvrplib(ckpt: str, model: str):
    """Reconstruct the CVRP model from a checkpoint (Set X path)."""
    env = CVRPEnv(generator_kwargs=dict(num_loc=100))
    cls = MODEL_CLASSES[model]
    net = cls.load_from_checkpoint(ckpt, env=env, map_location="cpu")
    net.eval()
    if model == "sil":
        net.labels.clear()
        net.best_policy_state = net.repair_policy_state = None
    return net


def build_td(coords, demand, capacity, device):
    """Pack one instance into a pre-reset TensorDict for the CVRP env.

    RL4CO's CVRP env expects ``locs`` = customers only, ``depot`` separate,
    and ``demand`` = customers only (it prepends the depot in ``_reset`` and
    sizes the ``visited`` mask as N_cust+1). The env fixes vehicle_capacity
    via its generator (default 1.0), so we normalize demand by the instance
    capacity -> demands in (0,1], capacity 1.0 -- exactly the distribution the
    synthetic CVRP data was trained with. A ``capacity`` key is included for
    backbones that read it directly (icam/l2r); the env ignores it on reset.
    """
    coords = coords.to(device)                      # (N, 2)  -- node 0 is the depot
    demand = demand.to(device)                      # (N,)    -- node 0 demand is 0
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


def costs_from_actions(env, td_r, actions):
    """Return per-start cost (negative reward) for the decoded actions.

    ``td_r`` is the *reset* td whose ``locs`` is the full node set (depot at
    index 0, then customers). POMO actions are ``(1, n_start, N)``; the tour
    for a start is ``[depot, locs[a_0], locs[a_1], ...]`` with the loop closed
    by ``get_tour_length`` (matches ``env._get_reward``). AM actions are
    ``(1, N)``.
    """
    from rl4co.utils.ops import get_tour_length

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
        if model in RESET_TD_BACKBONES:
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


def _evaluate_cvrplib(model, args, device) -> dict:
    """Evaluate a checkpoint on the CVRPLIB Set X benchmark.

    Loads the cached ``setX.pt`` (from ``args.data_dir``), converts each instance
    into the CVRP env tensor format, decodes one instance at a time (N varies per
    instance), and reports cost + gap vs the official best-known.
    """
    data = torch.load(Path(args.data_dir) / "setX.pt", map_location="cpu",
                      weights_only=False)
    sizes = {int(v["node_coord"].shape[0]): True for v in data.values()}
    include = None
    if args.sizes:
        include = {int(s) for s in args.sizes.split(",") if s.strip()}
        for s in include:
            if s not in sizes:
                raise RuntimeError(
                    f"Size n={s} not present in Set X. Available: {sorted(sizes)}")

    net = _load_checkpoint_cvrplib(args.ckpt, model).to(device)
    env = net.env
    starts = args.num_starts or 100

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
        if include and n not in include:
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
                        choices=["synthetic", "cvrplib"],
                        help="Evaluate on freshly generated synthetic instances "
                             "(or a cached --data_path) or on the CVRPLIB "
                             "Set X benchmark (--data_dir).")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Cached dataset split shared across methods, or the "
                             "LEHD-format .txt test file for --model lehd/ttpl.")
    parser.add_argument("--data_dir", type=str, default="./data/cvrplib_setX",
                        help="Directory holding setX.pt (only for "
                             "--dataset cvrplib).")
    parser.add_argument("--dist", type=str, default=None,
                        choices=["uniform", "gaussian"],
                        help="For --model lehd/ttpl only: generate synthetic "
                             "instances of this distribution instead of reading "
                             "a .txt data_path. Ignores data_path.")
    parser.add_argument("--num_loc", type=int, default=None,
                        help="Target number of customers N (required for "
                             "--dataset synthetic).")
    parser.add_argument("--n_inst", type=int, default=1000,
                        help="Number of eval instances")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Eval batch size")
    parser.add_argument("--num_starts", type=int, default=50,
                        help="Multi-start greedy for POMO/AM checkpoints. For "
                             "--dataset cvrplib this is the POMO multi-start "
                             "count (default 100). Not used by elg/dgl/radar "
                             "(baked pomo_size) nor sil/icam/l2r/invit.")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma list of n to evaluate on CVRPLIB Set X "
                             "(default: all). e.g. 101,110")
    parser.add_argument("--decode", type=str, default="greedy",
                        choices=["greedy", "sampling", "multistart_greedy",
                                 "multistart_sampling", "beam_search"],
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

    # ---- dataset / model compatibility ----
    if args.dataset == "cvrplib":
        if args.model in ("lehd", "ttpl"):
            parser.error(
                "--model lehd/ttpl only works on --dataset synthetic "
                "(they use their own LEHD-format .txt data, not Set X).")
    else:  # synthetic
        if args.num_loc is None:
            parser.error("--dataset synthetic requires --num_loc.")
        if args.model == "dgl":
            parser.error(
                "--model dgl only works on --dataset cvrplib "
                "(the synthetic path has no dgl decoder).")

    _allow_safe_globals()
    pl.seed_everything(args.seed, workers=True)

    args.device = eval_utils.pick_device(args.device)

    # ---- CVRPLIB Set X: dedicated per-instance harness (no rl4co synthetic env)
    if args.dataset == "cvrplib":
        result = _evaluate_cvrplib(args.model, args, args.device)
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

    # ---- LEHD / TTPL: dedicated env + dataloader (no rl4co CVRP env) ----
    if args.model in ("lehd", "ttpl"):
        if not args.data_path and not args.dist:
            parser.error(
                "--model lehd/ttpl needs --data_path (LEHD-format .txt) "
                "or --dist uniform|gaussian (synthetic).")
        model_cls = MODEL_CLASSES[args.model]
        model = model_cls.load_from_checkpoint(
            args.ckpt, map_location=args.device)
        model.to(args.device)
        # Use the checkpoint's trained size if --num_loc was not explicitly set
        # to match it; LEHD capacity is keyed off num_loc.
        hp_num_loc = getattr(model.hparams, "num_loc", None)
        if hp_num_loc is not None:
            args.num_loc = int(hp_num_loc)
        result = _evaluate_lehd(model, args, args.device)
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

        out_path = args.out or Path("results") / \
            f"eval_{args.model}_{args.num_loc}_{args.seed}.json"
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

    out_path = args.out or Path("results") / \
        f"eval_{args.model}_{args.num_loc}_{args.seed}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
