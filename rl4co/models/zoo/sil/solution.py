"""CVRP node/route-start labels and reconstruction for SIL.

Labels have shape (batch, customers, 2): one-based customer ID and a flag
indicating a depot visit BEFORE that customer. Demands stay normalized.
"""

import numpy as np
import torch


def to_actions(solution):
    """RL4CO actions, padded with harmless depot visits to twice the node count."""
    nodes, flags = solution.unbind(-1)
    actions = nodes.new_zeros(nodes.size(0), 2 * nodes.size(1))
    positions = (flags + 1).cumsum(-1) - 1
    return actions.scatter(1, positions, nodes)


def route_length(locs, solution):
    nodes, flags = solution.unbind(-1)
    ordered = locs.gather(1, nodes[..., None].expand(-1, -1, 2))
    previous = torch.cat((locs[:, :1], ordered[:, :-1]), 1)
    depot = locs[:, :1]
    direct = (ordered - previous).norm(dim=-1)
    via = (ordered - depot).norm(dim=-1) + (previous - depot).norm(dim=-1)
    return torch.where(flags.bool(), via, direct).sum(-1) + (ordered[:, -1] - depot[:, 0]).norm(
        dim=-1
    )


def feasible(demand, solution, tolerance=1e-5):
    """Check permutation, route starts and capacity without changing a label."""
    nodes, flags = solution.unbind(-1)
    expected = torch.arange(1, demand.size(1) + 1, device=demand.device)
    valid = (nodes.sort(-1).values == expected).all(-1)
    valid &= ((flags == 0) | (flags == 1)).all(-1) & (flags[:, 0] == 1)
    ordered = demand.gather(1, nodes - 1)
    # Segment sums through cumulative demand and the most recent route boundary.
    cumulative = ordered.cumsum(-1)
    before = cumulative - ordered
    starts = torch.where(flags.bool(), before, torch.zeros_like(before))
    used = cumulative - starts.cummax(-1).values
    return valid & (used <= 1.0 + tolerance).all(-1)


def insertion_labels(locs, demand):
    """SIL's angle-ordered cheapest insertion, using floating point demands.

    This replaces the bundled integer-only C++ extension. The route opening
    threshold is the depot round trip (upstream exploration=1).
    """
    results = []
    for xy, demands in zip(locs.detach().cpu().numpy(), demand.detach().cpu().numpy()):
        delta = xy[1:] - xy[0]
        order = np.argsort(np.arctan2(delta[:, 1], delta[:, 0])) + 1
        routes, loads = [], []
        for node in order:
            distance = np.linalg.norm(xy - xy[node], axis=-1)
            best_cost, best_route, best_position = 2 * distance[0], None, None
            for r, route in enumerate(routes):
                if loads[r] + float(demands[node - 1]) > 1.0 + 1e-6:
                    continue
                path = np.asarray([0, *route, 0])
                costs = (
                    distance[path[:-1]]
                    + distance[path[1:]]
                    - np.linalg.norm(xy[path[:-1]] - xy[path[1:]], axis=-1)
                )
                pos = int(costs.argmin())
                if costs[pos] < best_cost:
                    best_cost, best_route, best_position = costs[pos], r, pos
            if best_route is None:
                routes.append([int(node)])
                loads.append(float(demands[node - 1]))
            else:
                routes[best_route].insert(best_position, int(node))
                loads[best_route] += float(demands[node - 1])
        # Upstream returns its route vector in reverse order.
        results.append(
            [(node, int(i == 0)) for route in reversed(routes) for i, node in enumerate(route)]
        )
    return torch.tensor(results, dtype=torch.long, device=locs.device)


def order_routes_by_centroid(locs, solution):
    """SIL's geometric route ordering before reconstruction, without padding."""
    rows = []
    for xy, row in zip(locs, solution):
        starts = row[:, 1].nonzero().flatten().tolist()
        ends = starts[1:] + [len(row)]
        routes = [row[a:b] for a, b in zip(starts, ends)]
        centers = torch.stack(
            [(xy[route[:, 0]].sum(0) + xy[0]) / (len(route) + 1) for route in routes]
        )
        vectors = centers - xy[0]
        angles = torch.atan2(vectors[:, 1], vectors[:, 0])
        order = angles.argsort().tolist()
        rows.append(torch.cat([routes[i] for i in order]))
    return torch.stack(rows)


def augment_routes(solution):
    """Random route rotation and independent route reversal, as in SIL."""
    rows = []
    for row in solution:
        starts = row[:, 1].nonzero().flatten().tolist()
        ends = starts[1:] + [len(row)]
        routes = [row[a:b, 0].clone() for a, b in zip(starts, ends)]
        shift = int(torch.randint(len(routes), (1,)).item())
        routes = routes[shift:] + routes[:shift]
        if torch.rand(()).item() < 0.5:
            routes.reverse()
        pieces = []
        for route in routes:
            if torch.rand(()).item() < 0.5:
                route = route.flip(0)
            flags = torch.zeros_like(route)
            flags[0] = 1
            pieces.append(torch.stack((route, flags), -1))
        rows.append(torch.cat(pieces))
    return torch.stack(rows)


def sample_subpaths(locs, demand, solution, length, parallel=False):
    """Select route-ending subpaths, carry prefix load, and relabel local nodes.

    With parallel=True choose non-overlapping subpaths per instance (PRC).
    Training selects one uniformly sampled route end per instance. Every path
    keeps its first customer fixed, so its incoming edge remains unchanged.
    """
    batch, n, _ = solution.shape
    owners, positions = [], []
    for b in range(batch):
        ends = ((solution[b, :, 1].nonzero().flatten() - 1) % n).tolist()
        ends = [ends[i] for i in torch.randperm(len(ends)).tolist()]
        occupied = set()
        max_paths = min(n // length, len(ends)) if parallel else 1
        if parallel and n < 200:
            max_paths = min(max_paths, 2)
        for end in ends:
            pos = [(end - length + 1 + j) % n for j in range(length)]
            if occupied.isdisjoint(pos):
                owners.append(b)
                positions.append(pos)
                occupied.update(pos)
                if sum(owner == b for owner in owners) >= max_paths:
                    break
    owner = torch.tensor(owners, device=locs.device)
    position = torch.tensor(positions, device=locs.device)
    sub = solution[owner[:, None], position].clone()
    # Remaining capacity just before the first selected customer.
    ordered_demand = demand.gather(1, solution[:, :, 0] - 1)
    cumulative = ordered_demand.cumsum(-1)
    before = cumulative - ordered_demand
    starts = torch.where(solution[:, :, 1].bool(), before, torch.zeros_like(before))
    prefix_load = before - starts.cummax(-1).values
    remaining = 1.0 - prefix_load[owner, position[:, 0]]
    mapping, rank = sub[:, :, 0].sort(-1)
    sub[:, :, 0] = rank.argsort(-1) + 1
    sub_locs = torch.cat((locs[owner, :1], locs[owner[:, None], mapping]), 1)
    sub_demand = demand[owner[:, None], mapping - 1]
    return sub_locs, sub_demand, sub, remaining, owner, position, mapping
