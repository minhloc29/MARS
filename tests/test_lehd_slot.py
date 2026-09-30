from pathlib import Path

import lightning.pytorch as pl
import torch

from rl4co.models.zoo.lehd.model import make_lehd_dataloaders
from rl4co.models.zoo.lehd.model_slot import LEHDSlotModel


def _write_tiny_lehd_dataset(path: Path) -> None:
    # Four customers, one depot, one feasible route.  Two lines allow the
    # model's default non-overlapping train/validation split.
    line = (
        "depot,0,0,customer,1,0,1,1,0,1,0.5,0.5,capacity,50,"
        "demand,0,1,1,1,1,cost,4,node_flag,1,2,3,4,1,0,0,0\n"
    )
    path.write_text(line * 2, encoding="utf-8")


def _make_model(data_path: Path) -> LEHDSlotModel:
    return LEHDSlotModel(
        data_path=str(data_path),
        num_loc=4,
        embed_dim=8,
        decoder_layer_num=1,
        head_num=1,
        qkv_dim=8,
        ff_hidden=16,
        n_train_episodes=1,
        n_val_episodes=1,
        optimizer_kwargs={"lr": 5e-5},
        num_slots=2,
        metric_variant="D",
        slot_iters=1,
        proj_dim=4,
        k_neighbors=2,
        normalize_target=False,
        symmetrize_target=False,
    )


def test_slot_encoding_is_stable_and_noncollapsed(tmp_path: Path) -> None:
    data_path = tmp_path / "tiny_lehd.txt"
    _write_tiny_lehd_dataset(data_path)
    model = _make_model(data_path)
    problems = torch.rand(2, 5, 4)

    model.train()
    first = model.model.encoder(problems, 50.0)
    second = model.model.encoder(problems, 50.0)

    torch.testing.assert_close(first, second)
    assignments = model.model.encoder.last_A_ik
    assert assignments.std(dim=-1).max() > 1e-6

    model.eval()
    third = model.model.encoder(problems, 50.0)
    torch.testing.assert_close(first, third)


def test_slot_residual_starts_near_lehd(tmp_path: Path) -> None:
    data_path = tmp_path / "tiny_lehd.txt"
    _write_tiny_lehd_dataset(data_path)
    model = _make_model(data_path)
    problems = torch.rand(2, 5, 4)
    encoder = model.model.encoder

    base = encoder.base_encoder(problems, 50.0)
    slotted = encoder(problems, 50.0)

    assert encoder.slot_gate.item() < 0.02
    assert (slotted - base).square().mean().sqrt() < base.square().mean().sqrt()


def test_policy_optimizer_steps_once_per_decoded_token(tmp_path: Path) -> None:
    data_path = tmp_path / "tiny_lehd.txt"
    _write_tiny_lehd_dataset(data_path)
    model = _make_model(data_path)
    train_loader, val_loader = make_lehd_dataloaders(
        n_train=1, n_val=1, batch_size=1, seed=42
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)

    optimizer = trainer.optimizers[0]
    decoder_param = next(model.model.decoder.parameters())
    encoder_param = next(model.model.encoder.base_encoder.parameters())
    slot_param = next(model.model.encoder.slot_attn.parameters())
    gate_param = model.model.encoder.slot_gate_logit
    dual_param = model.metric_loss_fn.log_lambda

    # Four customers produce three teacher-forced CE updates.  The encoder
    # receives those three plus the one per-batch slot auxiliary update.
    # Lambda uses an explicit projected-ascent update and is not in Adam.
    assert int(optimizer.state[decoder_param]["step"].item()) == 3
    assert int(optimizer.state[encoder_param]["step"].item()) == 4
    assert int(optimizer.state[slot_param]["step"].item()) == 4
    assert int(optimizer.state[gate_param]["step"].item()) == 3
    assert dual_param not in optimizer.state
    assert trainer.global_step == 4
    assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == 4.5e-5


def test_dual_update_is_violation_scaled_and_projected(tmp_path: Path) -> None:
    data_path = tmp_path / "tiny_lehd.txt"
    _write_tiny_lehd_dataset(data_path)
    model = _make_model(data_path)

    before = model.metric_loss_fn.lmbda.detach().clone()
    model._update_dual(torch.tensor(2.0))
    expected = before + model.hparams.lr_dual * 2.0
    torch.testing.assert_close(model.metric_loss_fn.lmbda, expected)

    model._update_dual(torch.tensor(1e9))
    torch.testing.assert_close(
        model.metric_loss_fn.lmbda,
        torch.tensor(model.lambda_max),
    )
