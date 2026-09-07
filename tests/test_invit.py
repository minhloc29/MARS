from __future__ import annotations

import torch

from rl4co.models.zoo.invit import INViT, INViTPolicy, route_length


def _batch(batch_size=2, n=6):
    generator = torch.Generator().manual_seed(7)
    return {
        "locs": torch.rand(batch_size, n, 2, generator=generator),
        "depot": torch.rand(batch_size, 2, generator=generator),
        "demand": torch.tensor([[0.4, 0.4, 0.4, 0.4, 0.2, 0.2], [0.6, 0.2, 0.6, 0.2, 0.2, 0.2]])[
            :batch_size
        ],
    }


def _small_policy():
    return INViTPolicy(
        embed_dim=8,
        feedforward_dim=16,
        num_heads=2,
        # The largest view exceeds N, as the upstream defaults do for CVRP-50.
        state_sizes=(3, 8),
        action_size=2,
        state_encoder_layers=1,
        action_encoder_layers=1,
        decoder_layers=3,
    )


def test_route_length_with_depot_separators():
    nodes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    depot = torch.zeros(1, 2)
    actions = torch.tensor([[1, 0, 2]])
    assert torch.allclose(route_length(nodes, depot, actions), torch.tensor([4.0]))


def test_policy_outputs_feasible_routes_and_gradients():
    torch.manual_seed(11)
    batch = _batch()
    policy = _small_policy()
    output = policy(batch, decode_type="sampling")

    assert output["reward"].shape == (2,)
    assert output["log_likelihood"].shape == (2,)
    assert torch.isfinite(output["reward"]).all()
    assert torch.isfinite(output["log_likelihood"]).all()
    replayed = torch.stack(
        list(policy.replay_log_probabilities(batch, output["actions"])), dim=1
    ).sum(1)
    assert torch.allclose(replayed, output["log_likelihood"], atol=1e-6)

    for actions, demand in zip(output["actions"], batch["demand"]):
        customers = actions[actions > 0]
        assert torch.equal(customers.sort().values, torch.arange(1, 7))
        load = 0.0
        for action in actions.tolist():
            if action == 0:
                assert load <= 1.0 + 1e-6
                load = 0.0
            else:
                load += demand[action - 1].item()
        assert load <= 1.0 + 1e-6

    replayed.sum().backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_model_checkpoint_restores_student_and_rollout_baseline(tmp_path):
    torch.manual_seed(13)
    model = INViT(
        embed_dim=8,
        feedforward_dim=16,
        num_heads=2,
        state_sizes=(3, 5),
        action_size=2,
        state_encoder_layers=1,
        action_encoder_layers=1,
        decoder_layers=2,
    )
    with torch.no_grad():
        next(model.policy.parameters()).add_(1.0)
        next(model.baseline_policy.parameters()).sub_(1.0)
    checkpoint = tmp_path / "invit.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": "2.5.0",
        },
        checkpoint,
    )
    restored = INViT.load_from_checkpoint(checkpoint, weights_only=False)
    for expected, actual in zip(model.policy.parameters(), restored.policy.parameters()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(
        model.baseline_policy.parameters(), restored.baseline_policy.parameters()
    ):
        assert torch.equal(expected, actual)
