from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyvrp
from pyvrp import Client, Depot, Location, ProblemData, VehicleType

from baselines.utils import (
    CVRPInstance,
    routes_from_pyvrp_result,
    routes_length,
    validate_routes,
)


def solve_hgs(
    instance: CVRPInstance,
    time_limit: float = 10.0,
    max_iterations: int | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    """Solve a CVRPInstance using PyVRP (HGS algorithm).

    Args:
        instance: CVRPInstance with coords_int, coords_float, demand_int, capacity_int.
        time_limit: Maximum solve time in seconds.
        max_iterations: Optional max iterations stop criterion (overrides time_limit).
        seed: Random seed for PyVRP reproducibility.

    Returns:
        dict with:
            routes: list[list[int]] (1-based client indices)
            cost: float (recomputed Euclidean tour length on float coordinates)
            elapsed_seconds: float runtime
            feasible: bool
    """
    n = instance.num_loc
    # Vectorized pairwise integer distance matrix on coords_int
    diff = instance.coords_int[:, None, :] - instance.coords_int[None, :, :]
    dist_matrix = np.round(np.hypot(diff[..., 0], diff[..., 1])).astype(int)

    # Low-level ProblemData construction for maximum efficiency (no Model.add_edge loop)
    locations = [
        Location(x=float(instance.coords_int[i, 0]), y=float(instance.coords_int[i, 1]))
        for i in range(n + 1)
    ]
    depots = [Depot(location=0)]
    clients = [
        Client(location=i + 1, delivery=[int(instance.demand_int[i])])
        for i in range(n)
    ]
    # CVRP fleet upper bound is num_loc vehicles
    vehicle_types = [
        VehicleType(num_available=n, capacity=[int(instance.capacity_int)])
    ]

    problem_data = ProblemData(
        locations=locations,
        clients=clients,
        depots=depots,
        vehicle_types=vehicle_types,
        distance_matrices=[dist_matrix],
        duration_matrices=[dist_matrix],
    )

    if max_iterations is not None:
        stop = pyvrp.stop.MaxIterations(max_iterations)
    else:
        stop = pyvrp.stop.MaxRuntime(time_limit)

    start_time = time.perf_counter()
    result = pyvrp.solve(problem_data, stop=stop, seed=seed)
    elapsed = time.perf_counter() - start_time

    routes = routes_from_pyvrp_result(result)

    # Validate client coverage and capacity constraint
    validate_routes(
        routes,
        num_loc=n,
        demand_int=instance.demand_int,
        capacity_int=instance.capacity_int,
    )

    # Recompute exact float Euclidean tour length
    cost = routes_length(routes, instance.coords_float)

    return {
        "routes": routes,
        "cost": cost,
        "elapsed_seconds": elapsed,
        "feasible": result.is_feasible(),
    }
