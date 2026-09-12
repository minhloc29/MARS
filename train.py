from __future__ import annotations

import argparse
import json
import time

from pathlib import Path

import lightning.pytorch as pl
import torch

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

try:
    from lightning.pytorch.loggers import WandbLogger
    HAVE_WANDB = True
except Exception:
    HAVE_WANDB = False

from rl4co.data.slot_dataset import make_dataloader

try:
    from rl4co.envs import CVRPEnv
    from rl4co.models.zoo.dgl import DGL
    from rl4co.models.zoo.elg import ELG
    from rl4co.models.zoo.icam import ICAMCVRP
    from rl4co.models.zoo.invit import INViT
    from rl4co.models.zoo.l2r import L2RModel
    from rl4co.models.zoo.lehd import LEHDModel, TTRLModel
    from rl4co.models.zoo.lehd.model import make_lehd_dataloaders
    from rl4co.models.zoo.pomo_slot import AMSlot, POMOSlot
    from rl4co.models.zoo.sil import SIL
    FULL_RL4CO = True
except Exception as e:
    print(f"[WARN] Full rl4co import failed: {e}")
    FULL_RL4CO = False


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_DIR = REPO_ROOT / "data" / "slot_datasets_v2"
MARS_DATA_DIR = REPO_ROOT.parent / "MARS" / "data" / "slot_datasets_v2"
DEFAULT_DATA_DIR = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else MARS_DATA_DIR


MODEL_CLASSES = {
    "pomo": POMOSlot,
    "am": AMSlot,
    "l2r": L2RModel,
    "icam": ICAMCVRP,
    "invit": INViT,
    "dgl": DGL,
    "elg": ELG,
    "sil": SIL,
    "lehd": LEHDModel,
    "ttpl": TTRLModel,
}


BASELINE_BACKBONES = {"sil", "invit", "dgl", "elg"}


def _make_trainer(run_name: str, log_path: Path, epochs: int,
                  device: int | list[int], logger: str,
                  gradient_clip_val: float | None):

    checkpoint_cb = ModelCheckpoint(
        dirpath=log_path / "checkpoints",
        monitor="val/reward",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:03d}-{val/reward:.4f}",
    )
    early_stop_cb = EarlyStopping(
        monitor="val/reward",
        patience=20,
        mode="max",
    )

    if logger == "wandb":
        if not HAVE_WANDB:
            raise RuntimeError(
                "wandb requested but WandbLogger is not installed. "
                "Run `pip install wandb lightning` and login with `wandb login`."
            )
        logger_obj = WandbLogger(
            project="MeTRA_Slot_NCO",
            name=run_name,
            log_model="all",
        )
    else:
        logger_obj = CSVLogger(save_dir=str(log_path), name="metrics")

    use_cuda = torch.cuda.is_available()
    devices = device if isinstance(device, list) else [device]
    # One device: single-GPU run. Two or more: parallel DDP (data-parallel).
    strategy = "ddp" if len(devices) > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_cuda else "cpu",
        devices=devices if use_cuda else 1,
        strategy=strategy,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger_obj,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    return trainer, checkpoint_cb, early_stop_cb


def train(
    num_loc: int = 100,
    dist: str = "uniform",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output: str = "./output",
    seed: int = 42,
    device: int | list[int] = 0,
    n_train: int = 100_000,
    n_val: int = 1_000,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    embed_dim: int = 128,
    num_slots: int = 8,
    proj_dim: int = 64,
    slot_iters: int = 3,
    lambda_init: float = 1.0,
    lr_dual: float = 1e-3,
    metric_variant: str = "D",
    alpha_metric: float = 0.1,
    beta_entropy: float = 0.01,
    normalize_target: bool = True,
    symmetrize_target: bool = True,
    max_instances: int | None = None,
    backbone: str = "pomo",
    baseline: str | None = None,
    disable_slots: bool = False,
    ins_method: str = "construction",
    lower_neighbors_num: int = 50,
    reduction_percentage: float = 0.1,
    logger: str = "csv",
    resume: str | None = None,
    num_workers: int = 4,
    sil_repair_budget: int = 5,
    sil_improve_every: int = 20,
    sil_max_subtour_length: int = 64,
    sil_num_layers: int = 6,
    sil_parallel_reconstruction: bool = True,
    sil_update_mode: str = "batch",
    generate_missing_data: bool = False,
    generation_chunk_size: int | None = None,
    # ---- LEHD / TTPL baseline arguments ----
    lehd_data_path: str | None = None,
    lehd_val_data_path: str | None = None,
    lehd_decoder_layers: int = 6,
    invit_action_size: int = 15,
    invit_state_sizes: tuple[int, ...] = (35, 50, 65),
    invit_num_heads: int = 8,
    invit_state_encoder_layers: int = 2,
    invit_action_encoder_layers: int = 2,
    invit_decoder_layers: int = 3,
    invit_feedforward_dim: int | None = None,
    invit_baseline_tolerance: float = 1e-3,
    invit_scheduler_gamma: float = 0.99,
    invit_backprop_chunk_size: int = 16,
    # ---- DGL baseline arguments ----
    dgl_knn: int = 100,
    dgl_depot_knn: int = 100,
    dgl_pomo_size: int = 16,
    dgl_num_layers: int = 3,
    dgl_num_heads: int = 8,
    dgl_feedforward_dim: int | None = None,
    dgl_improve_every: int = 1,
    # ---- ELG baseline arguments ----
    elg_pomo_size: int = 50,
    elg_num_layers: int = 6,
    elg_num_heads: int = 8,
    elg_feedforward_dim: int | None = None,
    elg_local_size: int = 40,
    elg_local_dim: int = 32,
    elg_local_heads: int = 4,
    elg_mode: str = "joint",
    elg_warmup_epochs: int = 0,
    elg_scale_norm: bool = True,
):
    assert FULL_RL4CO, (
        "Full rl4co import failed. Ensure torchrl DLL is installed correctly "
        "or run on a compatible machine."
    )

    # ----------------------------------------------------------------
    # LEHD / TTPL branch — separate training loop using LEHD's own data
    # ----------------------------------------------------------------
    if backbone in ("lehd", "ttpl"):
        return _train_lehd(
            backbone=backbone,
            num_loc=num_loc,
            output=output,
            seed=seed,
            device=device,
            embed_dim=embed_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            n_train=n_train,
            n_val=n_val,
            max_instances=max_instances,
            logger=logger,
            resume=resume,
            lehd_data_path=lehd_data_path,
            lehd_val_data_path=lehd_val_data_path,
            lehd_decoder_layers=lehd_decoder_layers,
        )

    pl.seed_everything(seed)
    n_train_eff = min(
        n_train, max_instances) if max_instances is not None else n_train
    n_val_eff = min(n_val, max(1, max_instances // 10)
                    ) if max_instances is not None else n_val
    v_cfg = dict(metric_variant=metric_variant, alpha_metric=alpha_metric,
                 beta_entropy=beta_entropy)

    data_dir = Path(data_dir)
    train_path = data_dir / ins_method / f"cvrp{num_loc}_{dist}_train.pt"
    val_path = data_dir / ins_method / f"cvrp{num_loc}_{dist}_val.pt"

    missing_paths = [path for path in (
        train_path, val_path) if not path.exists()]
    if missing_paths and generate_missing_data:
        from rl4co.data.generate_slot_dataset import generate_and_save

        print(
            f"Missing {len(missing_paths)} required split(s); generating the shared "
            f"N={num_loc} {dist} dataset in {data_dir}."
        )
        generate_and_save(
            out_dir=data_dir,
            n=num_loc,
            dist=dist,
            n_train=n_train_eff,
            n_val=n_val_eff,
            n_test=n_val_eff,
            k_neighbors=15,
            method=ins_method,
            chunk_size=generation_chunk_size,
            seed=seed,
        )
        # Dataset generation consumes random numbers. Restore the experiment
        # seed so a first run and a run reusing the cache initialize identically.
        pl.seed_everything(seed)
        missing_paths = [path for path in (
            train_path, val_path) if not path.exists()]

    if missing_paths:
        preparation = (
            f"python -m rl4co.data.generate_slot_dataset --num_locs {num_loc} "
            f"--dist {dist} --n_train {n_train_eff} --n_val {n_val_eff} "
            f"--n_test {n_val_eff} --out_dir {data_dir} --method {ins_method} "
            f"--seed {seed}"
        )
        raise FileNotFoundError(
            "Missing required cached dataset split(s):\n  "
            + "\n  ".join(str(path) for path in missing_paths)
            + "\n\nPrepare the shared MARS/SIL data once with:\n  "
            + preparation
            + "\n\nOr append --generate_missing_data to the training command. "
              "N=1000 preparation is computationally expensive."
        )

    # Data
    data_variant = "none" if backbone in BASELINE_BACKBONES else metric_variant
    loader_kwargs = dict(seed=seed, num_workers=num_workers)
    train_loader = make_dataloader(train_path, batch_size, shuffle=True,
                                   variant=data_variant, max_instances=n_train_eff,
                                   include_instance_id=backbone in {"sil", "dgl"}, **loader_kwargs)
    val_loader = make_dataloader(val_path, batch_size, shuffle=False,
                                 variant=data_variant, max_instances=n_val_eff, **loader_kwargs)
    for loader in (train_loader, val_loader):
        if len(loader.dataset) == 0 or loader.dataset.locs.shape[1] != num_loc:
            raise ValueError(
                f"Expected a nonempty dataset with {num_loc} customers")

    # Validate d_ins cost method (Variant D consumes d_ins). The data was baked
    # with a specific method; refuse a mismatch so we never train on the wrong cost.
    if metric_variant == "D" and backbone not in BASELINE_BACKBONES:
        data_method = train_loader.dataset.method
        if data_method is not None and data_method != ins_method:
            raise RuntimeError(
                f"ins_method mismatch: --ins_method={ins_method!r} but cached dataset "
                f"{train_path} was generated with method={data_method!r} (see the "
                f"'method' tag in the .pt, and the {data_dir.name}/ subfolder). "
                f"Regenerate with --method {ins_method} or pass --ins_method {data_method}."
            )
        elif data_method is None:
            print(f"[WARN] {train_path} has no 'method' tag (legacy dataset) — "
                  f"cannot verify it matches --ins_method={ins_method!r}. "
                  f"Regenerate datasets with the current generator to stamp the method.")
        else:
            print(f"Dataset method '{data_method}' matches --ins_method. OK.")

    # Environment
    env = CVRPEnv(generator_params=dict(num_loc=num_loc))

    # Model
    model_cls = MODEL_CLASSES[backbone]
    model_kwargs = dict(
        env=env,
        embed_dim=embed_dim,
        num_slots=num_slots,
        **v_cfg,
        proj_dim=proj_dim,
        slot_iters=slot_iters,
        lambda_init=lambda_init,
        lr_dual=lr_dual,
        normalize_target=normalize_target,
        symmetrize_target=symmetrize_target,
        ins_method=ins_method,
        optimizer_kwargs={"lr": lr},
    )

    if backbone == "l2r":
        model_kwargs["lower_neighbors_num"] = lower_neighbors_num
        model_kwargs["reduction_percentage"] = reduction_percentage
    elif backbone == "icam":
        model_kwargs = dict(
            env=env,
            embed_dim=embed_dim,
            num_starts=num_loc,
            problem="cvrp",
            optimizer_kwargs={"lr": lr},
        )
    elif backbone == "sil":
        model_kwargs = dict(
            env=env, embed_dim=embed_dim, num_layers=sil_num_layers,
            repair_budget=sil_repair_budget, improve_every=sil_improve_every,
            max_subtour_length=sil_max_subtour_length,
            parallel_reconstruction=sil_parallel_reconstruction,
            update_mode=sil_update_mode,
            optimizer_kwargs={"lr": lr},
        )
    elif backbone == "invit":
        model_kwargs = dict(
            env=env,
            embed_dim=embed_dim,
            feedforward_dim=invit_feedforward_dim,
            num_heads=invit_num_heads,
            state_sizes=tuple(invit_state_sizes),
            action_size=invit_action_size,
            state_encoder_layers=invit_state_encoder_layers,
            action_encoder_layers=invit_action_encoder_layers,
            decoder_layers=invit_decoder_layers,
            baseline_tolerance=invit_baseline_tolerance,
            scheduler_gamma=invit_scheduler_gamma,
            backprop_chunk_size=invit_backprop_chunk_size,
            optimizer_kwargs={"lr": lr},
        )
    elif backbone == "dgl":
        model_kwargs = dict(
            env=env, embed_dim=embed_dim, num_layers=dgl_num_layers,
            num_heads=dgl_num_heads, feedforward_dim=dgl_feedforward_dim,
            knn=dgl_knn, depot_knn=dgl_depot_knn, pomo_size=dgl_pomo_size,
            improve_every=dgl_improve_every, optimizer_kwargs={"lr": lr},
        )
    elif backbone == "elg":
        model_kwargs = dict(
            env=env, embed_dim=embed_dim, num_layers=elg_num_layers,
            num_heads=elg_num_heads, feedforward_dim=elg_feedforward_dim,
            local_size=elg_local_size, local_dim=elg_local_dim,
            local_heads=elg_local_heads, pomo_size=elg_pomo_size,
            mode=elg_mode, warmup_epochs=elg_warmup_epochs,
            scale_norm=elg_scale_norm, optimizer_kwargs={"lr": lr},
        )
    elif backbone == "am":
        model_kwargs["baseline"] = baseline if baseline is not None else "shared"
    # disable_slots: run backbone as a true no-slot baseline (no slot/aux).
    if disable_slots and backbone not in BASELINE_BACKBONES:
        model_kwargs["disable_slots"] = True
    model = model_cls(**model_kwargs)
    if backbone == "sil":
        model.dataset_signature = train_loader.dataset.signature()
    elif backbone == "dgl":
        model.dataset_signature = train_loader.dataset.signature()

    # run_name uniquely IDs the run (backbone, variant, K, N, dist, seed, ins_method,
    # and — for Variant D — the normalize/symmetrize target-aggregation flags).
    norm_tag = ""
    if metric_variant == "D":
        norm_tag = f"_n{int(normalize_target)}s{int(symmetrize_target)}"

    base_suffix = f"_bl{baseline}" if backbone == "am" and baseline else ""
    if backbone == "sil":
        run_name = (f"sil_N{num_loc}_{dist}_{ins_method}_seed{seed}_d{embed_dim}"
                    f"_l{sil_num_layers}_r{sil_repair_budget}_i{sil_improve_every}"
                    f"_s{sil_max_subtour_length}_u{sil_update_mode}"
                    f"_prc{int(sil_parallel_reconstruction)}")
    elif backbone == "invit":
        state_tag = "-".join(map(str, invit_state_sizes))
        run_name = (f"invit_N{num_loc}_{dist}_{ins_method}_seed{seed}_d{embed_dim}"
                    f"_a{invit_action_size}_s{state_tag}_h{invit_num_heads}"
                    f"_se{invit_state_encoder_layers}_ae{invit_action_encoder_layers}"
                    f"_de{invit_decoder_layers}_c{invit_backprop_chunk_size}")
    elif backbone == "dgl":
        run_name = (f"dgl_N{num_loc}_{dist}_{ins_method}_seed{seed}_d{embed_dim}"
                    f"_k{dgl_knn}-{dgl_depot_knn}_p{dgl_pomo_size}"
                    f"_l{dgl_num_layers}_i{dgl_improve_every}")
    elif backbone == "elg":
        run_name = (f"elg_N{num_loc}_{dist}_{ins_method}_seed{seed}_d{embed_dim}"
                    f"_p{elg_pomo_size}_l{elg_num_layers}_local{elg_local_size}"
                    f"_{elg_mode}_w{elg_warmup_epochs}")
    elif disable_slots:
        run_name = f"{backbone}_noslot_N{num_loc}_{dist}_seed{seed}{base_suffix}"
    else:
        run_name = (f"{backbone}_slot_{metric_variant}_K{num_slots}_N{num_loc}_{dist}_"
                    f"{ins_method}{norm_tag}_seed{seed}{base_suffix}")
    log_path = Path(output) / run_name

    trainer, checkpoint_cb, _ = _make_trainer(
        run_name, log_path, epochs, device, logger,
        gradient_clip_val=None if backbone in BASELINE_BACKBONES else 1.0,
    )

    print(f"\n{'='*60}")
    print(f"Training {model_cls.__name__} | N={num_loc} | {dist}")
    print(
        f"  Epochs: {epochs}  Batch: {batch_size}  LR: {lr}")
    if backbone == "sil":
        print(f"  SIL: repair_budget={sil_repair_budget}, improve_every={sil_improve_every}, "
              f"max_subtour_length={sil_max_subtour_length}, update_mode={sil_update_mode}, "
              f"PRC={sil_parallel_reconstruction}")
        print("  Slot/metric/entropy flags do not apply to SIL; ins_method selects the shared data folder.")
    elif backbone == "dgl":
        print(f"  DGL: knn={dgl_knn}, depot_knn={dgl_depot_knn}, pomo={dgl_pomo_size}, "
              f"layers={dgl_num_layers}, improve_every={dgl_improve_every}")
        print("  Slot/metric/entropy flags do not apply to DGL; ins_method selects the shared data folder.")
    elif backbone == "elg":
        print(f"  ELG: pomo={elg_pomo_size}, layers={elg_num_layers}, "
              f"local_size={elg_local_size}, mode={elg_mode}, warmup={elg_warmup_epochs}")
        print("  Slot/metric/entropy flags do not apply to ELG; ins_method selects the shared data folder.")
    else:
        print(
            f"  Slots: K={num_slots}  proj_dim={proj_dim}  iters={slot_iters}")
    print(f"  ins_method: {ins_method}")
    print(f"  Output: {log_path}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)
    elapsed = time.time() - t0

    best_reward = checkpoint_cb.best_model_score.item(
    ) if checkpoint_cb.best_model_score else None
    result = {
        "backbone": backbone,
        "metric_variant": None if backbone in BASELINE_BACKBONES else metric_variant,
        "num_slots": None if backbone in BASELINE_BACKBONES else num_slots,
        "num_loc": num_loc,
        "dist": dist,
        "seed": seed,
        "ins_method": ins_method,
        "normalize_target": normalize_target,
        "symmetrize_target": symmetrize_target,
        "best_val_reward": best_reward,
        "elapsed_min": round(elapsed / 60, 1),
        "checkpoint": str(checkpoint_cb.best_model_path),
        "embed_dim": embed_dim,
        "batch_size": batch_size,
        "lr": lr,
        "train_path": str(train_path.resolve()),
        "val_path": str(val_path.resolve()),
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
    }
    if backbone == "sil":
        result.update(normalize_target=None, symmetrize_target=None,
                      sil_repair_budget=sil_repair_budget, sil_improve_every=sil_improve_every,
                      sil_max_subtour_length=sil_max_subtour_length, sil_num_layers=sil_num_layers,
                      sil_update_mode=sil_update_mode,
                      sil_parallel_reconstruction=sil_parallel_reconstruction,
                      dataset_signature=model.dataset_signature)
    elif backbone == "invit":
        result.update(
            invit_action_size=invit_action_size,
            invit_state_sizes=list(invit_state_sizes),
            invit_num_heads=invit_num_heads,
            invit_state_encoder_layers=invit_state_encoder_layers,
            invit_action_encoder_layers=invit_action_encoder_layers,
            invit_decoder_layers=invit_decoder_layers,
            invit_feedforward_dim=invit_feedforward_dim or 4 * embed_dim,
            invit_baseline_tolerance=invit_baseline_tolerance,
            invit_scheduler_gamma=invit_scheduler_gamma,
            invit_backprop_chunk_size=invit_backprop_chunk_size,
        )
    elif backbone == "dgl":
        result.update(
            normalize_target=None, symmetrize_target=None,
            dgl_knn=dgl_knn, dgl_depot_knn=dgl_depot_knn,
            dgl_pomo_size=dgl_pomo_size, dgl_num_layers=dgl_num_layers,
            dgl_num_heads=dgl_num_heads, dgl_feedforward_dim=dgl_feedforward_dim or 4 * embed_dim,
            dgl_improve_every=dgl_improve_every,
            dataset_signature=model.dataset_signature,
        )
    elif backbone == "elg":
        result.update(
            normalize_target=None, symmetrize_target=None,
            elg_pomo_size=elg_pomo_size, elg_num_layers=elg_num_layers,
            elg_num_heads=elg_num_heads, elg_feedforward_dim=elg_feedforward_dim or 4 * embed_dim,
            elg_local_size=elg_local_size, elg_local_dim=elg_local_dim,
            elg_local_heads=elg_local_heads, elg_mode=elg_mode,
            elg_warmup_epochs=elg_warmup_epochs, elg_scale_norm=elg_scale_norm,
        )

    # Dedup identical configs: rerun replaces, never appends a duplicate row.
    result_dir = Path(output)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"ablation_N{num_loc}.json"
    results = json.loads(result_file.read_text()
                         ) if result_file.exists() else []
    dedup_key = {k: result[k] for k in (
        "backbone", "metric_variant", "num_slots", "num_loc", "dist", "seed",
        "ins_method", "normalize_target", "symmetrize_target",
    )}
    if backbone == "sil":
        dedup_key.update(
            {k: v for k, v in result.items() if k.startswith("sil_")})
        dedup_key.update({k: result[k] for k in (
            "embed_dim", "batch_size", "dataset_signature")})
    elif backbone == "invit":
        dedup_key.update({k: v for k, v in result.items()
                         if k.startswith("invit_")})
    elif backbone == "dgl":
        dedup_key.update({k: v for k, v in result.items() if k.startswith("dgl_")})
        dedup_key.update({"dataset_signature": result["dataset_signature"]})
    elif backbone == "elg":
        dedup_key.update({k: v for k, v in result.items() if k.startswith("elg_")})
    results = [r for r in results if not all(
        r.get(k) == v for k, v in dedup_key.items())]
    results.append(result)
    result_file.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Best val reward: {best_reward} | {elapsed/60:.1f} min")
    return result


def _train_lehd(
    backbone: str,
    num_loc: int,
    output: str,
    seed: int,
    device: int | list[int],
    embed_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    n_train: int,
    n_val: int,
    max_instances: int | None,
    logger: str,
    resume: str | None,
    lehd_data_path: str | None,
    lehd_val_data_path: str | None,
    lehd_decoder_layers: int,
) -> dict:
    """Train LEHD or TTPL backbone under the same Lightning Trainer as MARS."""
    import json

    if lehd_data_path is None:
        raise ValueError(
            "--backbone lehd/ttpl requires --lehd_data_path pointing to a "
            "LEHD-format .txt training file "
            "(e.g. vrp1000_hgs_train_100w.txt from the LEHD Google Drive)."
        )

    pl.seed_everything(seed)

    if max_instances is not None:
        n_train = min(n_train, max_instances)
        n_val = min(n_val, max(1, max_instances // 10))

    model_cls = MODEL_CLASSES[backbone]  # LEHDModel or TTRLModel
    model = model_cls(
        data_path=lehd_data_path,
        val_data_path=lehd_val_data_path,
        num_loc=num_loc,
        embed_dim=embed_dim,
        decoder_layer_num=lehd_decoder_layers,
        n_train_episodes=n_train,
        n_val_episodes=n_val,
        optimizer_kwargs={"lr": lr},
    )

    train_loader, val_loader = make_lehd_dataloaders(
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        seed=seed,
        num_workers=0,  # data lives inside the model; no worker overhead needed
    )

    run_name = (
        f"{backbone}_N{num_loc}_seed{seed}_d{embed_dim}"
        f"_dec{lehd_decoder_layers}"
    )
    log_path = Path(output) / run_name

    trainer, checkpoint_cb, _ = _make_trainer(
        run_name, log_path, epochs, device, logger, gradient_clip_val=None)

    print(f"\n{'='*60}")
    print(f"Training {model_cls.__name__} | N={num_loc}")
    print(f"  Epochs: {epochs}  Batch: {batch_size}  LR: {lr}")
    print(f"  embed_dim: {embed_dim}  decoder_layers: {lehd_decoder_layers}")
    print(f"  n_train: {n_train}  n_val: {n_val}")
    print(f"  data: {lehd_data_path}")
    print(f"  Output: {log_path}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)
    elapsed = time.time() - t0

    best_reward = checkpoint_cb.best_model_score.item(
    ) if checkpoint_cb.best_model_score else None
    result = {
        "backbone": backbone,
        "num_loc": num_loc,
        "seed": seed,
        "embed_dim": embed_dim,
        "decoder_layers": lehd_decoder_layers,
        "best_val_reward": best_reward,
        "elapsed_min": round(elapsed / 60, 1),
        "checkpoint": str(checkpoint_cb.best_model_path),
        "batch_size": batch_size,
        "lr": lr,
        "n_train": n_train,
        "n_val": n_val,
        "lehd_data_path": lehd_data_path,
        "lehd_val_data_path": lehd_val_data_path,
    }

    result_dir = Path(output)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"ablation_N{num_loc}.json"
    results = json.loads(result_file.read_text()
                         ) if result_file.exists() else []
    dedup_key = {k: result[k]
                 for k in ("backbone", "num_loc", "seed", "embed_dim")}
    results = [r for r in results if not all(
        r.get(k) == v for k, v in dedup_key.items())]
    results.append(result)
    result_file.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Best val reward: {best_reward} | {elapsed/60:.1f} min")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train routing models")
    parser.add_argument("--num_loc", type=int, default=100,
                        choices=[50, 100, 200, 500, 1000])
    parser.add_argument("--dist", default="uniform",
                        choices=["uniform", "clustered"])
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, nargs="+", default=[0],
                        help="GPU id(s) to use. Pass multiple, e.g. `--device 0 1`, "
                             "for parallel DDP training; single value uses one GPU.")
    parser.add_argument("--n_train", type=int, default=100_000)
    parser.add_argument("--n_val", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--backbone", default="pomo",
                        choices=["pomo", "am", "l2r", "icam", "sil", "invit", "dgl", "elg", "lehd", "ttpl"])
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--slot_iters", type=int, default=3)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--lr_dual", type=float, default=1e-4)
    parser.add_argument("--metric_variant", "--variant", default="D",
                        choices=["none", "A", "B", "C", "D"])
    parser.add_argument("--alpha_metric", type=float, default=0.1)
    parser.add_argument("--beta_entropy", type=float, default=0.01)
    parser.add_argument("--ins_method", default="construction",
                        choices=["savings", "construction", "insertion"])
    parser.add_argument("--lower_neighbors_num", type=int, default=50)
    parser.add_argument("--reduction_percentage", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--generate_missing_data", action="store_true")
    parser.add_argument("--generation_chunk_size", type=int, default=None)
    parser.add_argument("--sil_repair_budget", type=int, default=5)
    parser.add_argument("--sil_improve_every", type=int, default=20)
    parser.add_argument("--sil_max_subtour_length", type=int, default=64)
    parser.add_argument("--sil_num_layers", type=int, default=6)
    parser.add_argument("--sil_update_mode",
                        choices=["batch", "node"], default="batch")
    parser.add_argument(
        "--sil_no_prc", dest="sil_parallel_reconstruction", action="store_false")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--disable_slots", action="store_true")
    parser.add_argument("--normalize_target",
                        action="store_true", default=True)
    parser.add_argument("--no_normalize_target",
                        dest="normalize_target", action="store_false")
    parser.add_argument("--symmetrize_target",
                        action="store_true", default=True)
    parser.add_argument("--no_symmetrize_target",
                        dest="symmetrize_target", action="store_false")
    parser.add_argument("--logger", default="csv", choices=["csv", "wandb"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lehd_data_path", default=None)
    parser.add_argument("--lehd_val_data_path", default=None)
    parser.add_argument("--lehd_decoder_layers", type=int, default=6)
    parser.add_argument("--invit_action_size", type=int, default=15)
    parser.add_argument("--invit_state_sizes", type=int,
                        nargs="+", default=[35, 50, 65])
    parser.add_argument("--invit_num_heads", type=int, default=8)
    parser.add_argument("--invit_state_encoder_layers", type=int, default=2)
    parser.add_argument("--invit_action_encoder_layers", type=int, default=2)
    parser.add_argument("--invit_decoder_layers", type=int, default=3)
    parser.add_argument("--invit_feedforward_dim", type=int, default=None)
    parser.add_argument("--invit_baseline_tolerance", type=float, default=1e-3)
    parser.add_argument("--invit_scheduler_gamma", type=float, default=0.99)
    parser.add_argument("--invit_backprop_chunk_size", type=int, default=16)
    parser.add_argument("--dgl_knn", type=int, default=100)
    parser.add_argument("--dgl_depot_knn", type=int, default=100)
    parser.add_argument("--dgl_pomo_size", type=int, default=16)
    parser.add_argument("--dgl_num_layers", type=int, default=3)
    parser.add_argument("--dgl_num_heads", type=int, default=8)
    parser.add_argument("--dgl_feedforward_dim", type=int, default=None)
    parser.add_argument("--dgl_improve_every", type=int, default=1)
    parser.add_argument("--elg_pomo_size", type=int, default=50)
    parser.add_argument("--elg_num_layers", type=int, default=6)
    parser.add_argument("--elg_num_heads", type=int, default=8)
    parser.add_argument("--elg_feedforward_dim", type=int, default=None)
    parser.add_argument("--elg_local_size", type=int, default=40)
    parser.add_argument("--elg_local_dim", type=int, default=32)
    parser.add_argument("--elg_local_heads", type=int, default=4)
    parser.add_argument("--elg_mode", choices=["joint", "only_global", "only_local"], default="joint")
    parser.add_argument("--elg_warmup_epochs", type=int, default=0)
    parser.add_argument("--elg_no_scale_norm", dest="elg_scale_norm", action="store_false")
    args = parser.parse_args()

    train(
        num_loc=args.num_loc, dist=args.dist, data_dir=args.data_dir,
        output=args.output, seed=args.seed, device=args.device,
        n_train=args.n_train, n_val=args.n_val, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, max_instances=args.max_instances,
        backbone=args.backbone, embed_dim=args.embed_dim, num_slots=args.num_slots,
        proj_dim=args.proj_dim, slot_iters=args.slot_iters,
        lambda_init=args.lambda_init, lr_dual=args.lr_dual,
        metric_variant=args.metric_variant, alpha_metric=args.alpha_metric,
        beta_entropy=args.beta_entropy, normalize_target=args.normalize_target,
        symmetrize_target=args.symmetrize_target, baseline=args.baseline,
        disable_slots=args.disable_slots, ins_method=args.ins_method,
        lower_neighbors_num=args.lower_neighbors_num,
        reduction_percentage=args.reduction_percentage, logger=args.logger,
        resume=args.resume, num_workers=args.num_workers,
        sil_repair_budget=args.sil_repair_budget,
        sil_improve_every=args.sil_improve_every,
        sil_max_subtour_length=args.sil_max_subtour_length,
        sil_num_layers=args.sil_num_layers,
        sil_update_mode=args.sil_update_mode,
        sil_parallel_reconstruction=args.sil_parallel_reconstruction,
        generate_missing_data=args.generate_missing_data,
        generation_chunk_size=args.generation_chunk_size,
        lehd_data_path=args.lehd_data_path,
        lehd_val_data_path=args.lehd_val_data_path,
        lehd_decoder_layers=args.lehd_decoder_layers,
        invit_action_size=args.invit_action_size,
        invit_state_sizes=tuple(args.invit_state_sizes),
        invit_num_heads=args.invit_num_heads,
        invit_state_encoder_layers=args.invit_state_encoder_layers,
        invit_action_encoder_layers=args.invit_action_encoder_layers,
        invit_decoder_layers=args.invit_decoder_layers,
        invit_feedforward_dim=args.invit_feedforward_dim,
        invit_baseline_tolerance=args.invit_baseline_tolerance,
        invit_scheduler_gamma=args.invit_scheduler_gamma,
        invit_backprop_chunk_size=args.invit_backprop_chunk_size,
        dgl_knn=args.dgl_knn,
        dgl_depot_knn=args.dgl_depot_knn,
        dgl_pomo_size=args.dgl_pomo_size,
        dgl_num_layers=args.dgl_num_layers,
        dgl_num_heads=args.dgl_num_heads,
        dgl_feedforward_dim=args.dgl_feedforward_dim,
        dgl_improve_every=args.dgl_improve_every,
        elg_pomo_size=args.elg_pomo_size,
        elg_num_layers=args.elg_num_layers,
        elg_num_heads=args.elg_num_heads,
        elg_feedforward_dim=args.elg_feedforward_dim,
        elg_local_size=args.elg_local_size,
        elg_local_dim=args.elg_local_dim,
        elg_local_heads=args.elg_local_heads,
        elg_mode=args.elg_mode,
        elg_warmup_epochs=args.elg_warmup_epochs,
        elg_scale_norm=args.elg_scale_norm,
    )


if __name__ == "__main__":
    main()
