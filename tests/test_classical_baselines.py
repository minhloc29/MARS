from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from baselines.hgs import solve_hgs
from baselines.lkh import check_lkh_available, solve_lkh
from baselines.utils import (
    CAPACITIES,
    COORD_SCALE,
    CVRPInstance,
    build_instance,
    parse_lkh_tour_file,
    routes_length,
    to_vrplib_string,
    validate_routes,
)


def _create_synthetic_instance(num_loc: int = 50, seed: int = 42) -> CVRPInstance:
    rng = np.random.default_rng(seed)
    depot_xy = rng.uniform(0.0, 1.0, size=(2,)).astype(np.float32)
    locs_xy = rng.uniform(0.0, 1.0, size=(num_loc, 2)).astype(np.float32)

    capacity = CAPACITIES.get(num_loc, 40.0)
    # Integer demands between 1 and 9 normalized by capacity
    raw_demand = rng.integers(1, 10, size=(num_loc,)).astype(np.float32)
    demand_norm = raw_demand / capacity

    return build_instance(depot_xy, locs_xy, demand_norm, num_loc=num_loc)


def test_build_instance_and_capacity_table():
    for num_loc, expected_cap in CAPACITIES.items():
        inst = _create_synthetic_instance(num_loc=num_loc)
        assert inst.capacity_int == int(expected_cap)
        assert inst.num_loc == num_loc
        assert inst.coords_float.shape == (num_loc + 1, 2)
        assert inst.coords_int.shape == (num_loc + 1, 2)
        assert len(inst.demand_int) == num_loc
        assert (inst.demand_int >= 1).all()
        # Ensure scaling roundtrip
        np.testing.assert_allclose(
            inst.coords_int / COORD_SCALE,
            inst.coords_float,
            atol=1e-4,
        )


def test_vrplib_writer_and_tour_parser(tmp_path: Path):
    inst = _create_synthetic_instance(num_loc=5)
    vrp_text = to_vrplib_string(inst, name="test_5")

    assert "NAME : test_5" in vrp_text
    assert "TYPE : CVRP" in vrp_text
    assert "DIMENSION : 6" in vrp_text
    assert "NODE_COORD_SECTION" in vrp_text
    assert "DEMAND_SECTION" in vrp_text
    assert "DEPOT_SECTION" in vrp_text
    assert "EOF" in vrp_text

    # Test tour parser with mock LKH output
    # TSPLIB nodes: 1 is depot, 2..6 are clients 1..5
    mock_tour_content = """NAME : test_5.tour
TYPE : TOUR
DIMENSION : 6
TOUR_SECTION
1
2
3
1
4
5
6
1
-1
EOF
"""
    tour_file = tmp_path / "mock.tour"
    tour_file.write_text(mock_tour_content, encoding="utf-8")

    routes = parse_lkh_tour_file(tour_file, num_loc=5)
    assert routes == [[1, 2], [3, 4, 5]]

    # Validation should pass
    validate_routes(routes, num_loc=5)


def test_validate_routes_catches_coverage_and_capacity_violations():
    # Missing client
    with pytest.raises(AssertionError, match="Coverage mismatch"):
        validate_routes([[1, 2], [3]], num_loc=4)

    # Duplicate client
    with pytest.raises(AssertionError, match="Coverage mismatch"):
        validate_routes([[1, 2], [2, 3, 4]], num_loc=4)

    # Capacity violation
    demand = np.array([10, 20, 15, 5])
    with pytest.raises(AssertionError, match="Capacity violated"):
        validate_routes([[1, 2], [3, 4]], num_loc=4, demand_int=demand, capacity_int=25)


def test_hgs_solve_and_idx_offset():
    inst = _create_synthetic_instance(num_loc=50, seed=123)
    res = solve_hgs(inst, time_limit=3.0, seed=42)

    assert res["feasible"] is True
    assert res["cost"] > 0.0
    assert len(res["routes"]) > 0

    # Ensure all clients 1..50 are visited exactly once
    validate_routes(
        res["routes"],
        num_loc=50,
        demand_int=inst.demand_int,
        capacity_int=inst.capacity_int,
    )

    # Sanity check cost is in reasonable range for CVRP-50 (e.g. 5.0 to 20.0)
    assert 4.0 < res["cost"] < 25.0


def test_lkh_solve_native():
    exe = check_lkh_available()
    assert exe.exists()

    inst = _create_synthetic_instance(num_loc=20, seed=777)
    res = solve_lkh(
        instance=inst,
        lkh_executable=exe,
        time_limit=5.0,
        runs=1,
        max_trials=1000,
        seed=42,
    )

    assert res["feasible"] is True
    assert res["cost"] > 0.0

    validate_routes(
        res["routes"],
        num_loc=20,
        demand_int=inst.demand_int,
        capacity_int=inst.capacity_int,
    )

    # Float cost check
    float_len = routes_length(res["routes"], inst.coords_float)
    assert abs(res["cost"] - float_len) < 1e-6
    assert 2.0 < res["cost"] < 15.0
