from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Sequence

import numpy as np

COORD_SCALE = 100_000

# Capacity table following Kool et al. (2019) convention
CAPACITIES = {
    50: 40.0,
    100: 50.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}


@dataclass
class CVRPInstance:
    depot_float: np.ndarray        # (2,) float32 in [0, 1]^2
    locs_float: np.ndarray         # (N, 2) float32 in [0, 1]^2
    coords_float: np.ndarray       # (N+1, 2) float32: row 0 is depot, rows 1..N are clients
    coords_int: np.ndarray         # (N+1, 2) int64: scaled by COORD_SCALE
    demand_int: np.ndarray         # (N,) int64: un-normalized client demands (clients 1..N)
    capacity_int: int              # un-normalized vehicle capacity
    num_loc: int                   # N: number of customers


def load_instance_batch(path: str | Path) -> dict[str, np.ndarray]:
    """Load a CVRP dataset split from either .pt (torch) or .npz (numpy).

    Returns a dict with numpy arrays for keys: locs, depot, demand, capacity.
    Torch import is lazy so .npz files can be read without PyTorch if desired.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pt":
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)

        def _to_numpy(val):
            if hasattr(val, "detach"):
                return val.detach().cpu().numpy()
            if hasattr(val, "numpy"):
                return val.numpy()
            return np.asarray(val)

        return {
            "locs": _to_numpy(data["locs"]),
            "depot": _to_numpy(data["depot"]),
            "demand": _to_numpy(data["demand"]),
            "capacity": _to_numpy(data["capacity"]) if "capacity" in data else None,
        }
    elif suffix == ".npz":
        data = np.load(path)
        return {
            "locs": np.asarray(data["locs"]),
            "depot": np.asarray(data["depot"]),
            "demand": np.asarray(data["demand"]),
            "capacity": np.asarray(data["capacity"]) if "capacity" in data else None,
        }
    else:
        raise ValueError(f"Unsupported dataset extension: {suffix} (expected .pt or .npz)")


def build_instance(
    depot_xy: np.ndarray,
    locs_xy: np.ndarray,
    demand_norm: np.ndarray,
    num_loc: int,
    capacity_override: float | int | None = None,
) -> CVRPInstance:
    """Construct a CVRPInstance with both float and integer coordinates/demands."""
    depot_float = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    locs_float = np.asarray(locs_xy, dtype=np.float32).reshape(num_loc, 2)
    demand_float = np.asarray(demand_norm, dtype=np.float32).reshape(num_loc)

    if capacity_override is not None:
        capacity_val = float(capacity_override)
    else:
        capacity_val = CAPACITIES.get(num_loc, max(9.0, round(9.0 * num_loc / 4.0)))

    capacity_int = int(round(capacity_val))
    # Un-normalize demand: round(demand_norm * capacity), clip to >= 1
    demand_int = np.clip(np.round(demand_float * capacity_val), 1, None).astype(np.int64)

    coords_float = np.vstack([depot_float[None, :], locs_float])
    coords_int = np.round(coords_float * COORD_SCALE).astype(np.int64)

    return CVRPInstance(
        depot_float=depot_float,
        locs_float=locs_float,
        coords_float=coords_float,
        coords_int=coords_int,
        demand_int=demand_int,
        capacity_int=capacity_int,
        num_loc=num_loc,
    )


def iter_instances(
    batch: dict[str, np.ndarray],
    num_loc: int,
    n_instances: int | None = None,
) -> Generator[CVRPInstance, None, None]:
    """Yield CVRPInstances from a loaded dataset batch."""
    total = len(batch["locs"])
    limit = min(total, n_instances) if n_instances is not None else total
    for i in range(limit):
        yield build_instance(
            depot_xy=batch["depot"][i],
            locs_xy=batch["locs"][i],
            demand_norm=batch["demand"][i],
            num_loc=num_loc,
        )


def routes_length(routes: Sequence[Sequence[int]], coords_float: np.ndarray) -> float:
    """Compute exact float Euclidean tour length from routes and original coordinates.

    routes: list of routes, where each route is a list of 1-based client IDs (1..num_loc).
    coords_float: (N+1, 2) array where row 0 is depot and rows 1..N are clients.
    """
    total_length = 0.0
    for route in routes:
        if not route:
            continue
        # Depot -> first client
        curr = 0
        for nxt in route:
            dx = coords_float[nxt, 0] - coords_float[curr, 0]
            dy = coords_float[nxt, 1] - coords_float[curr, 1]
            total_length += math.hypot(dx, dy)
            curr = nxt
        # Last client -> depot
        dx = coords_float[0, 0] - coords_float[curr, 0]
        dy = coords_float[0, 1] - coords_float[curr, 1]
        total_length += math.hypot(dx, dy)

    return float(total_length)


def validate_routes(
    routes: Sequence[Sequence[int]],
    num_loc: int,
    demand_int: np.ndarray | None = None,
    capacity_int: int | None = None,
) -> None:
    """Assert solution covers all clients exactly once without duplicates,
    and each route satisfies vehicle capacity."""
    all_clients = [c for r in routes for c in r]
    expected = list(range(1, num_loc + 1))

    if sorted(all_clients) != expected:
        missing = set(expected) - set(all_clients)
        duplicates = len(all_clients) - len(set(all_clients))
        raise AssertionError(
            f"Coverage mismatch: expected clients 1..{num_loc} (total {num_loc}), "
            f"got {len(all_clients)} visits with {duplicates} duplicates and {len(missing)} missing."
        )

    if demand_int is not None and capacity_int is not None:
        for idx, route in enumerate(routes):
            route_demand = sum(int(demand_int[c - 1]) for c in route)
            if route_demand > capacity_int:
                raise AssertionError(
                    f"Capacity violated on route {idx}: demand sum {route_demand} "
                    f"> capacity {capacity_int}."
                )


def to_vrplib_string(instance: CVRPInstance, name: str = "cvrp_instance") -> str:
    """Generate VRPLIB CVRP format string for LKH-3."""
    lines = [
        f"NAME : {name}",
        "TYPE : CVRP",
        f"DIMENSION : {instance.num_loc + 1}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        f"CAPACITY : {instance.capacity_int}",
        "NODE_COORD_SECTION",
    ]

    # Node 1: Depot
    lines.append(f"1 {instance.coords_int[0, 0]} {instance.coords_int[0, 1]}")
    # Nodes 2..N+1: Clients (1-based index in coords_int)
    for i in range(1, instance.num_loc + 1):
        lines.append(f"{i + 1} {instance.coords_int[i, 0]} {instance.coords_int[i, 1]}")

    lines.append("DEMAND_SECTION")
    lines.append("1 0")
    for i in range(1, instance.num_loc + 1):
        lines.append(f"{i + 1} {instance.demand_int[i - 1]}")

    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")

    return "\n".join(lines) + "\n"


def parse_lkh_tour_file(tour_path: str | Path, num_loc: int) -> list[list[int]]:
    """Parse LKH-3 .tour output file into routes with 1-based client indices (1..num_loc).

    In LKH-3 CVRP tours:
    - Node 1 is the primary depot.
    - Nodes 2..N+1 correspond to clients 1..N.
    - Nodes > N+1 are duplicate copies of the depot created for multiple vehicles.
    The output is a cyclic TSP tour, so we rotate it to start at a depot node
    and split into routes at every depot visit.
    """
    tour_path = Path(tour_path)
    if not tour_path.exists():
        raise FileNotFoundError(f"LKH tour file not found: {tour_path}")

    with open(tour_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    in_tour = False
    flat_nodes = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("TOUR_SECTION"):
            in_tour = True
            continue
        if in_tour:
            if line == "-1" or line.startswith("EOF"):
                break
            try:
                node_id = int(line)
                flat_nodes.append(node_id)
            except ValueError:
                continue

    if not flat_nodes:
        return []

    def is_depot(node: int) -> bool:
        return node == 1 or node > num_loc + 1

    # Rotate cyclic tour so it starts at a depot node
    first_depot_idx = None
    for idx, node in enumerate(flat_nodes):
        if is_depot(node):
            first_depot_idx = idx
            break

    if first_depot_idx is not None:
        flat_nodes = flat_nodes[first_depot_idx:] + flat_nodes[:first_depot_idx]

    routes: list[list[int]] = []
    current_route: list[int] = []

    for node in flat_nodes:
        if is_depot(node):
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            client_id = node - 1
            if 1 <= client_id <= num_loc:
                current_route.append(client_id)
            else:
                raise ValueError(
                    f"Parsed invalid client node {node} (client_id={client_id}) for N={num_loc}"
                )

    if current_route:
        routes.append(current_route)

    return routes


def routes_from_pyvrp_result(result) -> list[list[int]]:
    """Extract 1-based client routes from a PyVRP solve result.

    Applies the vital act.idx + 1 offset fix for client ScheduledActivity visits.
    """
    routes: list[list[int]] = []
    for r in result.best.routes():
        # r yields ScheduledActivity objects; filter client visits only
        client_route = [act.idx + 1 for act in r if act.is_client()]
        if client_route:
            routes.append(client_route)

    return routes
