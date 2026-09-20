import torch

from rl4co.models.zoo.radar import RADAR, RADARPolicy
from rl4co.models.zoo.radar.policy import _RADARDecoder


def _small_batch(batch_size=2, num_customers=5):
    return {
        "locs": torch.rand(batch_size, num_customers, 2),
        "depot": torch.rand(batch_size, 2),
        "demand": torch.full((batch_size, num_customers), 0.1),
        "capacity": torch.ones(batch_size, 1),
    }


def test_radar_uses_original_unnormalized_advantage_by_default():
    model = RADAR(
        embed_dim=16,
        encoder_layers=1,
        num_heads=4,
        ff_dim=32,
        svd_rank=3,
        pomo_size=5,
    )
    assert model.hparams.scale_norm is False


def test_decoder_excludes_unavailable_nodes_from_attention_context():
    torch.manual_seed(0)
    decoder = _RADARDecoder(dim=8, head_num=2, qkv_dim=4, logit_clipping=10.0)
    encoded = torch.randn(1, 4, 8, requires_grad=True)
    decoder.set_kv(encoded)
    mask = torch.tensor([[[False, False, True, False]]])

    logits = decoder(torch.randn(1, 1, 8), torch.ones(1, 1), mask)
    logits[..., [0, 1, 3]].sum().backward()

    torch.testing.assert_close(encoded.grad[:, 2], torch.zeros_like(encoded.grad[:, 2]))


def test_radar_training_step_is_finite_and_backpropagates():
    torch.manual_seed(0)
    model = RADAR(
        embed_dim=16,
        encoder_layers=1,
        num_heads=4,
        ff_dim=32,
        svd_rank=3,
        pomo_size=5,
    )
    loss = model.training_step(_small_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.policy.parameters())


def test_policy_rollout_shapes():
    from rl4co.models.zoo.baseline_cvrp import rollout

    torch.manual_seed(0)
    policy = RADARPolicy(
        embed_dim=16,
        encoder_layers=1,
        num_heads=4,
        ff_dim=32,
        svd_rank=3,
    )
    out = rollout(policy, _small_batch(), width=5, decode_type="greedy")
    assert out["reward"].shape == (2, 5)
    assert out["log_prob"].shape == (2, 5)
    assert torch.isfinite(out["reward"]).all()
    assert torch.isfinite(out["log_prob"]).all()
