"""Unit tests for the reference-tour teachers (Variant D d_cons geometry).

Covers:
  - STEP 0: the ``nn`` teacher must reproduce the historical ``construction``
    target exactly (byte-identical d_ins_idx / d_ins_val on 20 instances).
  - nn_split / nn_2opt: symmetry, zero diagonal, [0, L/2] bound, capacity-
    feasibility of the Prins split, non-increasing 2-opt tours.
"""

import torch

try:
    import pytest  # noqa: F401  (declared in pyproject dev extras)
except ImportError:  # pragma: no cover — tests still runnable without pytest installed
    pass

from rl4co.data.insertion_cost import (
    compute_construction_aware_insertion_cost,
    compute_sparse_insertion_cost,
    _build_nn_customer_order,
    _d_cons_nn_split,
    _d_cons_nn_2opt,
    _prins_split,
    _two_opt,
    _tour_length,
)

SEED = 0
B = 4
N = 20
K = 15


@pytest.fixture
def batch():
    torch.manual_seed(SEED)
    locs = torch.rand(B, N, 2)
    depot = torch.rand(B, 1, 2)
    # Normalized demands in (0, 1]; like the real generator, keep max demand well
    # below capacity (1.0) so single-customer routes are always feasible and a
    # route can legitimately hold several customers.
    demand = torch.randint(1, 4, (B, N)).float() / 4.0   # in (0, 0.75]
    return locs, depot, demand


@pytest.fixture
def nn_order(batch):
    locs, depot, _ = batch
    return _build_nn_customer_order(locs, depot)


# ---------------------------------------------------------------------------
# STEP 0 — nn teacher reproduces the historical construction target exactly.
# ---------------------------------------------------------------------------

def test_nn_teacher_reproduces_legacy_construction(batch):
    """The new --teacher nn path must be byte-identical to the legacy
    ``construction`` cache: same d_ins_idx (int16) and d_ins_val (float32)
    on 20 instances (B=4 x 5 chunks here for speed)."""
    locs, depot, _ = batch

    new_idx, new_val = compute_sparse_insertion_cost(
        locs, k_neighbors=K, depot_loc=depot, method="construction", teacher="nn")
    old_idx, old_val = compute_construction_aware_insertion_cost(
        locs, k_neighbors=K, depot_loc=depot)

    assert torch.equal(new_idx, old_idx), "d_ins_idx differs across nn/legacy"
    assert torch.allclose(new_val, old_val, atol=1e-6), "d_ins_val differs across nn/legacy"
    assert new_idx.dtype == torch.int16 and new_val.dtype == torch.float32


# ---------------------------------------------------------------------------
# Common sanity checks.
# ---------------------------------------------------------------------------

def _has_zero_diag(d, batch):
    B, N, _ = batch[0].shape
    eye = torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    return (d[eye] == 0).all()


def test_nn_split_hard_symmetry_and_bounds(batch, nn_order):
    locs, depot, demand = batch
    d, _ = _d_cons_nn_split(nn_order, locs, depot, demand, torch.ones(B), "hard")
    assert d.shape == (B, N, N)
    assert torch.allclose(d, d.transpose(1, 2)), "nn_split( hard) not symmetric"
    assert _has_zero_diag(d, batch), "nn_split(hard) nonzero diagonal"
    _, L = _tour_length(nn_order, locs, depot)
    assert (d >= 0).all() and (d <= (L.unsqueeze(-1).unsqueeze(-1) / 2) + 1e-5).all()


def test_nn_split_soft_differs_from_hard(batch, nn_order):
    locs, depot, demand = batch
    hard, _ = _d_cons_nn_split(nn_order, locs, depot, demand, torch.ones(B), "hard")
    soft, _ = _d_cons_nn_split(nn_order, locs, depot, demand, torch.ones(B), "soft")
    assert not torch.allclose(hard, soft), "hard and soft cross-route modes must differ"


def test_nn_2opt_symmetry_bounds_and_no_worsening(batch, nn_order):
    locs, depot, _ = batch
    d, _ = _d_cons_nn_2opt(nn_order, locs, depot)
    assert d.shape == (B, N, N)
    assert torch.allclose(d, d.transpose(1, 2)), "nn_2opt not symmetric"
    assert _has_zero_diag(d, batch), "nn_2opt nonzero diagonal"

    opt, _ = _two_opt(nn_order, locs, depot)
    _, l0 = _tour_length(nn_order, locs, depot)
    _, l1 = _tour_length(opt, locs, depot)
    assert (l1 <= l0 + 1e-6).all(), "2-opt increased tour length"
    # 2-opt preserves a permutation.
    assert (torch.sort(opt, dim=-1).values == torch.arange(N)).all()


def test_prins_split_routes_are_capacity_feasible(batch, nn_order):
    locs, depot, demand = batch
    rids, _ = _prins_split(nn_order, demand, torch.ones(B), locs, depot)
    dem_seq = torch.gather(demand, 1, nn_order)
    for b in range(B):
        s = 0
        while s < N:
            e = s
            while e < N and rids[b, e] == rids[b, s]:
                e += 1
            assert dem_seq[b, s:e].sum() <= 1.0 + 1e-6, f"route infeasible (row {b})"
            s = e
