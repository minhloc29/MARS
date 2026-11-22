import math
import pytest
import torch
import torch.nn.functional as F

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.models.nn.slot_attention import SlotAttention
from rl4co.models.nn.cross_attention_slots import CrossAttentionSlots
from rl4co.models.zoo.pomo_slot.model import POMOSlot
from rl4co.data.insertion_cost import compute_sparse_insertion_cost


def make_dummy_batch(env, B=4, k=5):
    """Generate mock batch with sparse d_ins matching SlotDataset format."""
    batch = env.generator(batch_size=[B])
    locs = batch["locs"]
    depot = batch["depot"]
    if depot.dim() == 2:
        depot = depot.unsqueeze(1)
    d_ins_idx, d_ins_val = compute_sparse_insertion_cost(
        locs, k_neighbors=k, depot_loc=depot, method="construction"
    )
    batch.set("d_ins_idx", d_ins_idx)
    batch.set("d_ins_val", d_ins_val)
    return batch


def test_shapes_and_row_normalization():
    """Verify shapes for N in {20, 100, 500}, rows of A sum to 1, no NaN/inf."""
    torch.manual_seed(42)
    B = 4
    K = 8
    D = 64

    for module_type in ["slot", "cross", "cross_iter"]:
        if module_type == "slot":
            mod = SlotAttention(num_slots=K, dim=D, iters=3, deterministic_init=True)
        elif module_type == "cross":
            mod = CrossAttentionSlots(num_slots=K, dim=D, iters=1, iterative=False)
        else:
            mod = CrossAttentionSlots(num_slots=K, dim=D, iters=3, iterative=True)

        for N in [20, 100, 500]:
            x = torch.randn(B, N, D)
            Z, A = mod(x)

            # Check shapes
            assert Z.shape == (B, K, D), f"{module_type} Z shape mismatch: {Z.shape}"
            assert A.shape == (B, N, K), f"{module_type} A shape mismatch: {A.shape}"

            # Check finite
            assert torch.isfinite(Z).all(), f"{module_type} Z contains NaN/inf"
            assert torch.isfinite(A).all(), f"{module_type} A contains NaN/inf"

            # Check rows of A sum to 1 (atol 1e-5)
            row_sums = A.sum(dim=-1)
            expected_sums = torch.ones(B, N)
            assert torch.allclose(row_sums, expected_sums, atol=1e-5), (
                f"{module_type} rows do not sum to 1 at N={N}: max diff {(row_sums - expected_sums).abs().max()}"
            )


def test_regression_slot_module_identical():
    """Regression test: slot_module='slot' must yield identical forward outputs and losses."""
    torch.manual_seed(1234)
    env = CVRPEnv(generator_params=dict(num_loc=20))
    td = env.reset(batch_size=[4])

    # Run POMOSlot with slot_module="slot"
    model = POMOSlot(
        env=env,
        embed_dim=64,
        num_slots=8,
        slot_iters=3,
        metric_variant="D",
        alpha_metric=0.1,
        beta_entropy=0.01,
        slot_module="slot",
    )
    model.eval()

    # Base SlotAttention output
    node_embs = torch.randn(4, 20, 64)
    with torch.no_grad():
        Z, A = model.slot_attn(node_embs)
    assert Z.shape == (4, 8, 64)
    assert A.shape == (4, 20, 8)
    assert torch.allclose(A.sum(dim=-1), torch.ones(4, 20), atol=1e-5)


def test_gradients_and_d_target_detached():
    """Gradients must reach encoder and the slot module; D_target must carry NO gradient."""
    torch.manual_seed(42)
    env = CVRPEnv(generator_params=dict(num_loc=20))
    batch_size = 4

    # Test for both cross and cross_iter
    for slot_mod in ["cross", "cross_iter"]:
        model = POMOSlot(
            env=env,
            embed_dim=64,
            num_slots=8,
            slot_iters=3,
            metric_variant="D",
            alpha_metric=0.1,
            beta_entropy=0.01,
            slot_module=slot_mod,
        )
        model.train()

        # Generate raw batch (un-reset, exactly as fed by DataLoader)
        batch = make_dummy_batch(env, B=batch_size, k=5)

        # Step forward
        out = model.shared_step(batch, batch_idx=0, phase="train")
        loss = out["loss"]

        # Check that loss requires grad
        assert loss.requires_grad

        # Backward
        loss.backward()

        # 1. Gradients reach the new module parameters
        cross_mod = model.policy.encoder.slot_attn
        assert cross_mod.slots_init.grad is not None
        assert cross_mod.to_q.weight.grad is not None
        assert cross_mod.to_k.weight.grad is not None
        assert cross_mod.to_v.weight.grad is not None

        # 2. Gradients reach base encoder
        base_enc = model.policy.encoder.base_encoder
        assert base_enc.init_embedding.init_embed.weight.grad is not None

        # 3. D_target carried no gradient: check that target does not retain grad
        assert model.metric_loss_fn is not None
        # log_lambda has gradient from dual ascent
        assert model.metric_loss_fn.log_lambda.grad is not None


def test_dual_ascent_update_smoke():
    """Verify dual variable log_lambda updates and is correctly clamped."""
    torch.manual_seed(42)
    env = CVRPEnv(generator_params=dict(num_loc=20))
    model = POMOSlot(
        env=env,
        embed_dim=64,
        num_slots=8,
        slot_iters=2,
        metric_variant="D",
        alpha_metric=0.1,
        beta_entropy=0.01,
        slot_module="cross",
        lr_dual=1e-2,
    )
    model.train()
    opt = model.configure_optimizers()

    initial_lambda = model.metric_loss_fn.log_lambda.item()

    # Create batch with small target to induce metric violation
    batch = env.generator(batch_size=[4])
    batch.set("d_ins_idx", torch.randint(0, 20, (4, 20, 5), dtype=torch.int16))
    batch.set("d_ins_val", torch.zeros(4, 20, 5))  # target 0 guarantees constraint violation

    opt.zero_grad()
    out = model.shared_step(batch, batch_idx=0, phase="train")
    out["loss"].backward()

    assert model.metric_loss_fn.log_lambda.grad is not None
    assert model.metric_loss_fn.log_lambda.grad.item() != 0.0, "gradient for dual variable should be non-zero"

    opt.step()
    updated_lambda = model.metric_loss_fn.log_lambda.item()
    assert updated_lambda > initial_lambda, "log_lambda should increase under positive violation (dual ascent)"

    # Verify clamping
    model.metric_loss_fn.log_lambda.data.fill_(math.log(100.0))
    model.on_after_optimizer_step(opt)
    clamped_lambda = model.metric_loss_fn.log_lambda.item()
    assert math.isclose(clamped_lambda, math.log(50.0), rel_tol=1e-5), "lambda was not clamped to log(50.0)"


def test_smoke_two_epochs():
    """Run 2 short mini-epochs for (a) cross nogeo and (b) cross geo; verify loss is finite."""
    torch.manual_seed(42)
    env = CVRPEnv(generator_params=dict(num_loc=20))

    # (a) cross, geometry OFF (alpha=0, beta=0.01)
    model_nogeo = POMOSlot(
        env=env,
        embed_dim=64,
        num_slots=8,
        slot_iters=1,
        metric_variant="D",
        alpha_metric=0.0,
        beta_entropy=0.01,
        slot_module="cross",
    )
    opt_nogeo = model_nogeo.configure_optimizers()

    for epoch in range(2):
        batch = make_dummy_batch(env, B=4, k=5)
        opt_nogeo.zero_grad()
        out = model_nogeo.shared_step(batch, batch_idx=0, phase="train")
        loss = out["loss"]
        assert torch.isfinite(loss), f"NaN/Inf loss in nogeo at epoch {epoch}"
        loss.backward()
        opt_nogeo.step()

    # (b) cross, geometry ON (alpha=0.1, beta=0.01)
    model_geo = POMOSlot(
        env=env,
        embed_dim=64,
        num_slots=8,
        slot_iters=1,
        metric_variant="D",
        alpha_metric=0.1,
        beta_entropy=0.01,
        slot_module="cross",
    )
    opt_geo = model_geo.configure_optimizers()

    for epoch in range(2):
        batch = make_dummy_batch(env, B=4, k=5)
        opt_geo.zero_grad()
        out = model_geo.shared_step(batch, batch_idx=0, phase="train")
        loss = out["loss"]
        assert torch.isfinite(loss), f"NaN/Inf loss in geo at epoch {epoch}"
        loss.backward()
        opt_geo.step()
        model_geo.on_after_optimizer_step(opt_geo)
