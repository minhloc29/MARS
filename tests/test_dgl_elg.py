import torch

from rl4co.models.zoo.baseline_cvrp import CVRPState, rollout
from rl4co.models.zoo.dgl import DGL
from rl4co.models.zoo.elg import ELG


def _batch(batch_size=2, num_loc=8):
    generator = torch.Generator().manual_seed(7)
    return {
        "locs": torch.rand(batch_size, num_loc, 2, generator=generator),
        "depot": torch.rand(batch_size, 2, generator=generator),
        "demand": torch.randint(1, 4, (batch_size, num_loc), generator=generator).float() / 10,
        "capacity": torch.ones(batch_size, 1),
        "instance_id": torch.arange(batch_size),
    }


def _assert_feasible(batch, actions):
    demands = torch.cat((torch.zeros(len(batch["demand"]), 1), batch["demand"]), 1)
    for row in range(actions.size(0)):
        route = actions[row].tolist()
        customers = [node for node in route if node]
        assert sorted(set(customers)) == list(range(1, batch["locs"].size(1) + 1))
        load = 1.0
        for node in route:
            if node == 0:
                load = 1.0
            else:
                load -= float(demands[row, node])
                assert load >= -1e-5


def test_elg_rollout_is_finite_feasible_and_differentiable():
    batch = _batch()
    model = ELG(
        embed_dim=32, num_layers=1, num_heads=4, local_dim=16,
        local_heads=4, local_size=4, pomo_size=4,
    )
    result = rollout(model.policy, batch, 4, "sampling")
    assert result["reward"].shape == (2, 4)
    assert torch.isfinite(result["reward"]).all()
    advantage = result["reward"] - result["reward"].mean(1, keepdim=True)
    loss = -(advantage.detach() * result["log_prob"]).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    _assert_feasible(batch, result["actions"][:, 0])


def test_dgl_solution_pool_and_checkpoint_roundtrip():
    batch = _batch()
    model = DGL(
        embed_dim=32, num_layers=1, num_heads=4, knn=4,
        depot_knn=4, pomo_size=4,
    )
    labels = model._labels_for_batch(batch)
    loss = model._imitation_loss(batch, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert len(model.solution_pool) == len(batch["locs"])
    _assert_feasible(batch, labels)

    model.dataset_signature = "same-data"
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    restored = DGL(embed_dim=32, num_layers=1, num_heads=4, knn=4, depot_knn=4)
    restored.dataset_signature = "same-data"
    restored.on_load_checkpoint(checkpoint)
    assert restored.solution_rewards == model.solution_rewards


def test_state_uses_normalized_capacity_data():
    batch = _batch()
    state = CVRPState.from_batch(batch)
    assert state.xy.shape == (2, 9, 2)
    assert state.demand[:, 0].eq(0).all()


def test_matched_defaults_are_within_one_percent_of_mars():
    mars_parameters = 610_561
    dgl = DGL(embed_dim=128, num_layers=3, num_heads=8)
    elg = ELG(embed_dim=88, num_layers=6, num_heads=8)
    for model in (dgl, elg):
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        assert abs(count - mars_parameters) / mars_parameters < 0.01
