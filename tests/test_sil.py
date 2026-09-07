"""SIL data parity, feasibility, upstream architecture and training lifecycle."""

import importlib.util

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.sil import SIL, SILPolicy
from rl4co.models.zoo.sil.solution import (
    augment_routes,
    feasible,
    insertion_labels,
    order_routes_by_centroid,
    route_length,
    sample_subpaths,
    to_actions,
)
from train import TRAIN_DEFAULTS, SlotDataset, make_dataloader


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(previous)


def cached_data(path, batch=4, n=10):
    data = dict(
        locs=torch.rand(batch, n, 2),
        depot=torch.rand(batch, 2),
        demand=torch.randint(1, 10, (batch, n)).float() / 20,
        capacity=torch.ones(batch, 1),
        format_version="sparse_v2",
        method="insertion",
        d_ins_idx=torch.zeros(batch, n, 2, dtype=torch.int16),
        d_ins_val=torch.ones(batch, n, 2),
    )
    torch.save(data, path)
    return data


def test_shared_data_and_rng_independent_shuffle(tmp_path):
    path = tmp_path / "data.pt"
    raw = cached_data(path)
    mars = SlotDataset(path, variant="D", max_instances=3)
    sil = SlotDataset(path, variant="none", max_instances=3, include_instance_id=True)
    for key in ("locs", "depot", "demand", "capacity"):
        assert torch.equal(mars[1][key], sil[1][key])
        assert torch.equal(sil[1][key], raw[key][1])
    assert "d_ins_val" not in sil[1] and sil[1]["instance_id"] == 1
    first = make_dataloader(path, "D", 2, True, include_instance_id=True, num_workers=0)
    torch.rand(1000)  # model-specific RNG use must not change data order
    second = make_dataloader(path, "none", 2, True, include_instance_id=True, num_workers=0)
    assert torch.equal(
        torch.cat([b["instance_id"] for b in first]), torch.cat([b["instance_id"] for b in second])
    )
    assert mars.signature() == sil.signature()
    sil.demand[0, 0] += 0.01
    assert mars.signature() != sil.signature()
    assert TRAIN_DEFAULTS[1000]["batch"] == 64


@pytest.mark.parametrize("batch", [1, 3])
def test_feasible_labels_decode_and_mars_reward(batch):
    env = CVRPEnv(generator_params={"num_loc": 10})
    td = env.reset(env.generator([batch]))
    initial = insertion_labels(td["locs"], td["demand"])
    policy = SILPolicy(embed_dim=16, num_layers=1)
    with torch.no_grad():
        decoded = policy(td, env)["solution"]
    for labels in (
        initial,
        augment_routes(initial),
        order_routes_by_centroid(td["locs"], initial),
        decoded,
    ):
        assert feasible(td["demand"], labels).all()
        actions = to_actions(labels)
        env.check_solution_validity(td, actions)
        torch.testing.assert_close(route_length(td["locs"], labels), -env.get_reward(td, actions))
    with pytest.raises(ValueError, match="single greedy"):
        policy(td, env, num_starts=2)


@pytest.mark.parametrize("parallel", [False, True])
def test_subpaths_preserve_capacity_and_disjointness(parallel):
    env = CVRPEnv(generator_params={"num_loc": 20})
    td = env.reset(env.generator([3]))
    labels = augment_routes(insertion_labels(td["locs"], td["demand"]))
    xy, demand, sub, remaining, owner, pos, mapping = sample_subpaths(
        td["locs"], td["demand"], labels, 4, parallel
    )
    torch.testing.assert_close(mapping.gather(1, sub[:, :, 0] - 1), labels[owner[:, None], pos, 0])
    for b in range(3):
        positions = pos[owner == b].flatten()
        assert len(positions.unique()) == len(positions)
    for step in range(4):
        node, flag = sub[:, step].unbind(-1)
        expected_flag = flag.clone()
        remaining, actual_flag = SILPolicy.advance(demand, remaining, node, flag)
        assert torch.equal(actual_flag, expected_flag)
        assert (remaining >= -1e-5).all()


def test_upstream_decoder_numerical_parity():
    source = Path(__file__).resolve().parents[2] / "SIL/CVRP/Train/VRPModel.py"
    if not source.exists():
        pytest.skip("Optional upstream checkout unavailable")
    spec = importlib.util.spec_from_file_location("upstream_sil", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    policy = SILPolicy(embed_dim=16, num_layers=1)
    reference = upstream.VRPModel(
        mode="test",
        embedding_dim=16,
        decoder_layer_num=1,
        head_num=8,
        qkv_dim=2,
        ff_hidden_dim=512,
        k_nearest_num=1000,
    )
    reference.load_state_dict(policy.state_dict())
    xy = torch.rand(2, 11, 2)
    demand = torch.rand(2, 10) / 10
    problems = policy.problems(xy, demand, torch.ones(2))
    selected = torch.tensor([[1, 3], [4, 5]])
    expected = reference.decoder(
        reference.encoder(problems, 1.0), problems, selected, 2, 1.0, problems[:, 0, 3]
    )
    actual = policy.probabilities(problems, selected)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    loss = -actual[:, 1].log().mean()
    loss.backward()
    assert policy.decoder.Linear_final.weight.grad.abs().sum() > 0


def test_reconstruction_accepts_shorter_and_rejects_infeasible(monkeypatch):
    import rl4co.models.zoo.sil.model as model_module

    env = CVRPEnv(generator_params={"num_loc": 4})
    model = SIL(env, embed_dim=16, num_layers=1, repair_budget=1)
    xy = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]]])
    demand = torch.full((1, 4), 0.2)
    label = torch.tensor([[[1, 1], [3, 0], [2, 0], [4, 0]]])
    candidate = torch.tensor([[[1, 1], [2, 0], [3, 0], [4, 0]]])
    monkeypatch.setattr(model_module, "augment_routes", lambda x: x)
    monkeypatch.setattr(model_module, "order_routes_by_centroid", lambda xy, x: x)
    monkeypatch.setattr(
        model_module,
        "sample_subpaths",
        lambda *args: (
            xy,
            demand,
            label,
            torch.ones(1),
            torch.tensor([0]),
            torch.arange(4)[None],
            torch.arange(1, 5)[None],
        ),
    )
    monkeypatch.setattr(model.policy, "decode", lambda *args: candidate.clone())
    improved = model.improve_labels(xy, demand, label)
    assert torch.equal(improved, candidate)
    assert (route_length(xy, improved) < route_length(xy, label)).all()
    # The same geometrically shorter proposal is illegal at higher demand.
    demand.fill_(0.4)
    label[:, :, 1] = torch.tensor([1, 0, 1, 0])
    assert torch.equal(model.improve_labels(xy, demand, label), label)


def test_train_improve_checkpoint_resume(tmp_path):
    path = tmp_path / "data.pt"
    cached_data(path, batch=2, n=6)
    train_loader = make_dataloader(path, "none", 2, True, include_instance_id=True, num_workers=0)
    val_loader = make_dataloader(path, "none", 2, False, num_workers=0)
    env = CVRPEnv(generator_params={"num_loc": 6})
    model = SIL(
        env, embed_dim=16, num_layers=1, improve_every=1, repair_budget=1, max_subtour_length=4
    )
    model.dataset_signature = train_loader.dataset.signature()
    initial = model.policy.decoder.Linear_final.weight.detach().clone()
    kwargs = dict(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer = pl.Trainer(max_epochs=2, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    # One batch per epoch and comparable mode means one optimizer update per epoch.
    assert trainer.global_step == 2
    assert not torch.equal(model.policy.decoder.Linear_final.weight, initial)
    assert len(model.labels) == 2 and set(model.label_rounds.values()) == {1}
    assert {label.dtype for label in model.labels.values()} == {torch.int16}
    assert model._repair_policy is not None
    checkpoint = tmp_path / "sil.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = SIL.load_from_checkpoint(checkpoint, env=env, weights_only=False)
    assert restored.dataset_signature == model.dataset_signature
    for key in model.labels:
        assert torch.equal(restored.labels[key], model.labels[key])
    model.eval()
    restored.eval()
    td = model._reset(next(iter(val_loader)))
    with torch.no_grad():
        torch.testing.assert_close(
            model.policy(td, env)["reward"], restored.policy(td, env)["reward"]
        )
    new_trainer = pl.Trainer(max_epochs=3, **kwargs)
    new_trainer.fit(restored, train_loader, val_loader, ckpt_path=checkpoint)
    assert set(restored.label_rounds.values()) == {2}
    assert new_trainer.global_step > trainer.global_step
    restored.dataset_signature = "different cached instances"
    with pytest.raises(ValueError, match="different training dataset"):
        restored.on_load_checkpoint(torch.load(checkpoint, weights_only=False))

    # Checkpoints written before comparable mode had no update_mode and stored
    # labels as int64. They should resume with the new default and compact data.
    legacy = torch.load(checkpoint, weights_only=False)
    legacy["hyper_parameters"].pop("update_mode")
    legacy["sil_labels"] = {key: value.long() for key, value in legacy["sil_labels"].items()}
    legacy_path = tmp_path / "legacy.ckpt"
    torch.save(legacy, legacy_path)
    legacy_restored = SIL.load_from_checkpoint(legacy_path, env=env, weights_only=False)
    assert legacy_restored.hparams.update_mode == "batch"
    assert {label.dtype for label in legacy_restored.labels.values()} == {torch.int16}


def test_node_update_mode_retains_upstream_step_schedule(tmp_path):
    path = tmp_path / "data.pt"
    cached_data(path, batch=2, n=6)
    train_loader = make_dataloader(path, "none", 2, True, include_instance_id=True, num_workers=0)
    env = CVRPEnv(generator_params={"num_loc": 6})
    model = SIL(
        env,
        embed_dim=16,
        num_layers=1,
        update_mode="node",
        max_subtour_length=4,
        repair_budget=0,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader)
    assert trainer.global_step == 3  # fixed length 4: update steps 1, 2 and 3
