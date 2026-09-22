"""Regression checks for route indexing, capacity, and figure export."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_cvrp_routes.py"
SPEC = importlib.util.spec_from_file_location("plot_cvrp_routes", SCRIPT)
PLOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT
SPEC.loader.exec_module(PLOT)


def _instance():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    demands = np.array([0., 2., 3., 4.])
    return xy, demands, 5.


def test_lehd_flags_and_rl4co_actions_split_at_depot():
    assert PLOT.routes_from_lehd([[1, 1], [2, 0], [3, 1]]) == [[1, 2], [3]]
    assert PLOT.routes_from_actions([1, 2, 0, 0, 3, 0]) == [[1, 2], [3]]
    with pytest.raises(ValueError, match="first depot flag"):
        PLOT.routes_from_lehd([[1, 0], [2, 1], [3, 0]])


def test_validate_coverage_capacity_and_distance():
    xy, demands, capacity = _instance()
    distance, loads = PLOT.validate_plan(xy, demands, capacity, [[1, 2], [3]])
    assert distance == pytest.approx(4 + 2**0.5)
    np.testing.assert_array_equal(loads, [5, 4])
    with pytest.raises(ValueError, match="exactly once"):
        PLOT.validate_plan(xy, demands, capacity, [[1, 2], [2]])
    with pytest.raises(ValueError, match="capacity"):
        PLOT.validate_plan(xy, demands, capacity, [[1, 2, 3]])


def test_effective_flag_records_forced_depot_return():
    flags = torch.tensor([0, 0, 1])
    remaining = torch.tensor([2., 5., 1.])
    demand = torch.tensor([3., 5., 2.])
    assert PLOT.effective_lehd_flag(flags, remaining, demand).tolist() == [1, 0, 1]


def test_json_input_and_png_pdf_output(tmp_path):
    xy, demands, capacity = _instance()
    source = tmp_path / "instance.json"
    source.write_text(json.dumps({
        "title": "CVRP check", "coordinates": xy.tolist(),
        "demands": demands.tolist(), "capacity": capacity,
        "solutions": [{"label": "MARS", "actions": [1, 2, 0, 3]},
                      {"label": "Reference", "routes": [[3], [1, 2]]}],
    }))
    coords, dem, cap, plans, title = PLOT.load_json(source)
    assert title == "CVRP check"
    assert plans[0].routes == [[1, 2], [3]]
    png, pdf = tmp_path / "routes.png", tmp_path / "routes.pdf"
    assert len(PLOT.plot_instance(coords, dem, cap, plans, png, title)) == 2
    PLOT.plot_instance(coords, dem, cap, plans, pdf, title)
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
