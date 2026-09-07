from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

try:
    from lightning.pytorch.loggers import WandbLogger
    HAVE_WANDB = True
except Exception:
    HAVE_WANDB = False

# Lazy import to avoid torchrl DLL on some setups
try:
    from rl4co.envs import CVRPEnv
    from rl4co.models.zoo.pomo_slot import POMOSlot, AMSlot
    from rl4co.models.zoo.l2r import L2RModel
    from rl4co.models.zoo.icam import ICAMCVRP
    from rl4co.data.slot_dataset import SlotDataset, make_dataloader
    FULL_RL4CO = True
except Exception as e:
    print(f"[WARN] Full rl4co import failed: {e}")
    FULL_RL4CO = False


MODEL_CLASSES = {
    "pomo": POMOSlot,
    "am": AMSlot,
    "l2r": L2RModel,
    "icam": ICAMCVRP,
}


def train(
    # --- data / problem ---
    num_loc: int = 100,
    dist: str = "uniform",
    data_dir: str = "./data/slot_datasets_v2",
    n_train: int = 100_000,
    n_val: int = 1_000,
    max_instances: int | None = None,
    # --- training ---
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    seed: int = 42,
    device: int = 0,
    resume: str | None = None,
    logger: str = "csv",
    # --- model / slots ---
    backbone: str = "pomo",
    embed_dim: int = 128,
    num_slots: int = 8,
    proj_dim: int = 64,
    slot_iters: int = 3,
    alpha_metric: float = 0.1,
    beta_entropy: float = 0.01,
    lambda_init: float = 1.0,
    lr_dual: float = 1e-3,
    normalize_target: bool = True,
    symmetrize_target: bool = True,
    disable_slots: bool = False,
    ins_method: str = "construction",
    baseline: str | None = None,
    # --- L2R only ---
    lower_neighbors_num: int = 50,
    reduction_percentage: float = 0.1,
    # --- output ---
    output: str = "./output",
):
    assert FULL_RL4CO, (
        "Full rl4co import failed. Ensure torchrl DLL is installed correctly "
        "or run on a compatible machine."
    )

    pl.seed_everything(seed)

    # Cap dataset size for quick smoke tests: n_train/n_val are clipped to the
    # requested max_instances budget (val is a small fraction of it).
    n_train_eff = n_train
    n_val_eff = n_val
    if max_instances is not None:
        n_train_eff = min(n_train, max_instances)
        n_val_eff = min(n_val, max(1, max_instances // 10))

    data_dir = Path(data_dir)
    train_path = data_dir / ins_method / f"cvrp{num_loc}_{dist}_train.pt"
    val_path = data_dir / ins_method / f"cvrp{num_loc}_{dist}_val.pt"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {train_path}\n"
            f"Run: python -m rl4co.data.generate_slot_dataset "
            f"--num_locs {num_loc} --dist {dist} --out_dir {data_dir}"
        )

    # Data
    train_loader = make_dataloader(
        train_path, batch_size, shuffle=True, max_instances=n_train_eff)
    val_loader = make_dataloader(
        val_path,   batch_size, shuffle=False, max_instances=n_val_eff)

    # Validate d_ins cost method. The metric-preservation loss consumes d_ins,
    # and the data was baked with a specific method; refuse a mismatch so we
    # never train on the wrong cost.
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
    env = CVRPEnv(generator_kwargs=dict(num_loc=num_loc))

    # Model: shared slot-metric hyperparameters for pomo/am backbones.
    model_cls = MODEL_CLASSES[backbone]
    model_kwargs = dict(
        env=env,
        embed_dim=embed_dim,
        num_slots=num_slots,
        alpha_metric=alpha_metric,
        beta_entropy=beta_entropy,
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
    elif backbone == "am":
        model_kwargs["baseline"] = baseline if baseline is not None else "shared"
    # disable_slots: run backbone as a true no-slot baseline (no slot/aux).
    if disable_slots:
        model_kwargs["disable_slots"] = True
    model = model_cls(**model_kwargs)

    # run_name uniquely IDs the run (backbone, K, N, dist, seed, ins_method and
    # the normalize/symmetrize target-aggregation flags).
    norm_tag = f"_n{int(normalize_target)}s{int(symmetrize_target)}"

    base_suffix = f"_bl{baseline}" if backbone == "am" and baseline else ""
    if disable_slots:
        run_name = f"{backbone}_noslot_N{num_loc}_{dist}_seed{seed}{base_suffix}"
    else:
        run_name = (f"{backbone}_slot_K{num_slots}_N{num_loc}_{dist}_"
                    f"{ins_method}{norm_tag}_seed{seed}{base_suffix}")
    log_path = Path(output) / run_name

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

    # Trainer (single GPU; --device picks the index)
    use_cuda = torch.cuda.is_available()
    trainer_kwargs = dict(
        max_epochs=epochs,
        accelerator="gpu" if use_cuda else "cpu",
        devices=[device] if use_cuda else 1,
        strategy="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger_obj,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer = pl.Trainer(**trainer_kwargs)

    print(f"\n{'='*60}")
    print(f"Training {backbone} — metric-aware slots | N={num_loc} | {dist}")
    print(f"  Epochs: {epochs}  Batch: {batch_size}  LR: {lr}  "
          f"n_train: {n_train_eff}  n_val: {n_val_eff}")
    print(f"  Slots: K={num_slots}  proj_dim={proj_dim}  iters={slot_iters}")
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
        "num_slots": num_slots,
        "num_loc": num_loc,
        "dist": dist,
        "seed": seed,
        "ins_method": ins_method,
        "normalize_target": normalize_target,
        "symmetrize_target": symmetrize_target,
        "best_val_reward": best_reward,
        "elapsed_min": round(elapsed / 60, 1),
        "checkpoint": str(checkpoint_cb.best_model_path),
    }

    # Dedup identical configs: rerun replaces, never appends a duplicate row.
    result_dir = Path(output)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"ablation_N{num_loc}.json"
    results = json.loads(result_file.read_text()
                         ) if result_file.exists() else []
    dedup_key = {k: result[k] for k in (
        "backbone", "num_slots", "num_loc", "dist", "seed",
        "ins_method", "normalize_target", "symmetrize_target",
    )}
    results = [r for r in results if not all(
        r.get(k) == v for k, v in dedup_key.items())]
    results.append(result)
    result_file.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Best val reward: {best_reward:.4f} | {elapsed/60:.1f} min")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train POMOSlot/AMSlot -- Metric-Aware NCO")
    # --- data / problem ---
    parser.add_argument("--num_loc",      type=int,
                        default=100,      choices=[50, 100, 200, 500, 1000])
    parser.add_argument("--dist",         type=str,
                        default="uniform", choices=["uniform", "clustered"])
    parser.add_argument("--data_dir",     type=str,
                        default="./data/slot_datasets_v2")
    parser.add_argument("--n_train",      type=int,   default=100_000)
    parser.add_argument("--n_val",        type=int,   default=1_000)
    parser.add_argument("--max_instances", type=int,  default=None,
                        help="Cap dataset size for quick smoke tests")
    # --- training ---
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="Main model learning rate (REINFORCE optimizer).")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       type=int,   default=0,
                        help="GPU index (0 or 1) to use; single GPU only.")
    parser.add_argument("--logger",       type=str,   default="csv",
                        choices=["csv", "wandb"],
                        help="Logger: 'csv' (default, lightweight) or 'wandb' (requires wandb login).")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to a .ckpt to resume training from its last epoch (Lightning checkpoint).")
    # --- model / slots ---
    parser.add_argument("--backbone",     type=str,   default="pomo",
                        choices=["pomo", "am", "l2r", "icam"],
                        help="Backbone: 'pomo', 'am', 'l2r', or native ICAM CVRP")
    parser.add_argument("--embed_dim",    type=int,   default=128)
    parser.add_argument("--num_slots",    type=int,   default=8,
                        help="K — number of slot/region embeddings.")
    parser.add_argument("--proj_dim",     type=int,   default=64)
    parser.add_argument("--slot_iters",   type=int,   default=3)
    parser.add_argument("--alpha_metric", type=float, default=0.1,
                        help="Weight for the metric preservation / reconstruction loss.")
    parser.add_argument("--beta_entropy", type=float, default=0.01,
                        help="Slot-entropy regulariser weight. Set 0.0 to keep slots + "
                             "metric loss but drop the entropy regulariser.")
    parser.add_argument("--lambda_init",  type=float, default=1.0)
    parser.add_argument("--lr_dual",      type=float, default=1e-4,
                        help="Learning rate for dual ascent on the metric loss lambda.")
    parser.add_argument("--ins_method",   type=str,   default="construction",
                        choices=["savings", "construction", "insertion"],
                        help="d_ins insertion-cost method. Must match the cached dataset's "
                             "'method' tag (the generator stamps it into the .pt).")
    parser.add_argument("--normalize_target", dest="normalize_target",
                        action="store_true", default=True,
                        help="Normalize D_ins aggregation by realized sparse edge mass (default: True).")
    parser.add_argument("--no_normalize_target", dest="normalize_target",
                        action="store_false",
                        help="Use RAW (unnormalized) D_ins aggregation -- for the ablation baseline.")
    parser.add_argument("--symmetrize_target", dest="symmetrize_target",
                        action="store_true", default=True,
                        help="Symmetrize D_ins aggregation (default: True).")
    parser.add_argument("--no_symmetrize_target", dest="symmetrize_target",
                        action="store_false",
                        help="Keep D_ins aggregation asymmetric -- for the ablation baseline.")
    parser.add_argument("--disable_slots", action="store_true",
                        help="Run the backbone as a true no-slot baseline (skips SlotAttention + aux losses).")
    parser.add_argument("--baseline",     type=str,   default=None,
                        help="REINFORCE baseline for the AM backbone (e.g. rollout, shared). Ignored for pomo.")
    # --- L2R only ---
    parser.add_argument("--lower_neighbors_num", type=int, default=50,
                        help="L2R lower-model candidate count")
    parser.add_argument("--reduction_percentage", type=float, default=0.1,
                        help="L2R static farthest-edge reduction fraction")
    # --- output ---
    parser.add_argument("--output",       type=str,   default="./output",
                        help="Root dir for logs + results/ablation_N{num_loc}.json")
    args = parser.parse_args()

    train(
        num_loc=args.num_loc,
        dist=args.dist,
        data_dir=args.data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        max_instances=args.max_instances,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        logger=args.logger,
        resume=args.resume,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        num_slots=args.num_slots,
        proj_dim=args.proj_dim,
        slot_iters=args.slot_iters,
        alpha_metric=args.alpha_metric,
        beta_entropy=args.beta_entropy,
        lambda_init=args.lambda_init,
        lr_dual=args.lr_dual,
        normalize_target=args.normalize_target,
        symmetrize_target=args.symmetrize_target,
        disable_slots=args.disable_slots,
        ins_method=args.ins_method,
        baseline=args.baseline,
        lower_neighbors_num=args.lower_neighbors_num,
        reduction_percentage=args.reduction_percentage,
        output=args.output,
    )


if __name__ == "__main__":
    main()
