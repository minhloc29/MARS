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
    FULL_RL4CO = True
except Exception as e:
    print(f"[WARN] Full rl4co import failed: {e}")
    FULL_RL4CO = False


MODEL_CLASSES = {
    "pomo": POMOSlot,
    "am": AMSlot,
    "l2r": L2RModel,
}



class SlotDataset(torch.utils.data.Dataset):
    """Wraps cached .pt files from generate_slot_dataset.py (sparse_v2)."""
    def __init__(self, filepath: str | Path, variant: str = "D", max_instances: int | None = None):
        data = torch.load(filepath, map_location="cpu", weights_only=False)

        # Format version sanity check (reject old dense d_ins)
        fmt = data.get("format_version", None)
        if fmt is None:
            if "d_ins" in data:
                raise RuntimeError(
                    f"Old dense d_ins format detected in {filepath}.\n"
                    "Please regenerate datasets using the updated generate_slot_dataset.py "
                    "which produces sparse_v2 format (d_ins_idx + d_ins_val).\n"
                    "Command: python -m rl4co.data.generate_slot_dataset --num_locs N --dist DIST ..."
                )
        elif fmt != "sparse_v2":
            raise RuntimeError(f"Unknown dataset format_version: '{fmt}' in {filepath}")

        self.locs     = data["locs"]     # (N_inst, N, 2)
        self.depot    = data["depot"]    # (N_inst, 2)
        self.demand   = data["demand"]   # (N_inst, N)
        self.capacity = data.get("capacity", None)
        # d_ins cost-method tag stamped by the generator; None for legacy datasets.
        self.method: str | None = data.get("method", None)

        # Sparse d_ins only needed for Variant D
        needs_dins = variant == "D"
        self.d_ins_idx = data.get("d_ins_idx", None) if needs_dins else None  # (N_inst,N,k) int16
        self.d_ins_val = data.get("d_ins_val", None) if needs_dins else None  # (N_inst,N,k) float32
        self.variant = variant

        if max_instances is not None:
            self.locs     = self.locs[:max_instances]
            self.depot    = self.depot[:max_instances]
            self.demand   = self.demand[:max_instances]
            if self.capacity  is not None: self.capacity  = self.capacity[:max_instances]
            if self.d_ins_idx is not None: self.d_ins_idx = self.d_ins_idx[:max_instances]
            if self.d_ins_val is not None: self.d_ins_val = self.d_ins_val[:max_instances]

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        item = {
            "locs":   self.locs[idx],    # (N, 2)
            "depot":  self.depot[idx],   # (2,)
            "demand": self.demand[idx],  # (N,)
        }
        if self.capacity is not None:
            item["capacity"] = self.capacity[idx]   # (1,)
        if self.d_ins_idx is not None:
            item["d_ins_idx"] = self.d_ins_idx[idx]  # (N, k) int16
        if self.d_ins_val is not None:
            item["d_ins_val"] = self.d_ins_val[idx]  # (N, k) float32
        return item


def _collate_fn(batch: list[dict]) -> dict:
    """Collate dicts -> batched dict; shared_step converts to TensorDict internally."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def make_dataloader(filepath: str, variant: str, batch_size: int, shuffle: bool, max_instances: int | None = None):
    ds = SlotDataset(filepath, variant=variant, max_instances=max_instances)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


VARIANT_DEFAULTS = {
    "A": dict(metric_variant="A", alpha_metric=0.1,  beta_entropy=0.01),
    "B": dict(metric_variant="B", alpha_metric=0.0,  beta_entropy=0.00),
    "C": dict(metric_variant="C", alpha_metric=0.1,  beta_entropy=0.01),
    "D": dict(metric_variant="D", alpha_metric=0.1,  beta_entropy=0.01),
    # "E": future-regret target -- reserved, not implemented
}

TRAIN_DEFAULTS = {
    50:  dict(epochs=100, batch=512, lr=1e-4, n_train=100_000, n_val=1_000),
    100: dict(epochs=100, batch=256, lr=1e-4, n_train=100_000, n_val=1_000),
    200: dict(epochs=200, batch=128, lr=5e-5, n_train=100_000, n_val=1_000),
    500: dict(epochs=200, batch=32,  lr=5e-5, n_train=50_000,  n_val=500),
}


def train(
    variant: str = "D",
    num_loc: int = 100,
    dist: str = "uniform",
    data_dir: str = "./data/slot_datasets",
    output: str = "./output",
    seed: int = 42,
    device: int = 0,
    embed_dim: int = 128,
    num_slots: int = 8,
    proj_dim: int = 64,
    slot_iters: int = 3,
    lambda_init: float = 1.0,
    lr_dual: float = 1e-3,
    beta_entropy: float | None = None,
    normalize_target: bool = True,
    symmetrize_target: bool = True,
    epochs: int | None = None,
    batch_size: int | None = None,
    max_instances: int | None = None,
    backbone: str = "pomo",
    baseline: str | None = None,
    disable_slots: bool = False,
    ins_method: str = "construction",
    logger: str = "csv",
    resume: str | None = None,
    lower_neighbors_num: int = 50,
    reduction_percentage: float = 0.1,
):
    assert FULL_RL4CO, (
        "Full rl4co import failed. Ensure torchrl DLL is installed correctly "
        "or run on a compatible machine."
    )

    pl.seed_everything(seed)
    t_cfg = TRAIN_DEFAULTS[num_loc].copy()

    if epochs is not None:
        t_cfg["epochs"] = epochs
    if batch_size is not None:
        t_cfg["batch"] = batch_size
    if max_instances is not None:
        t_cfg["n_train"] = min(t_cfg["n_train"], max_instances)
        t_cfg["n_val"]   = min(t_cfg["n_val"],   max(1, max_instances // 10))
    v_cfg = VARIANT_DEFAULTS[variant]
    if beta_entropy is not None:
        v_cfg["beta_entropy"] = beta_entropy

    data_dir = Path(data_dir)
    train_path = data_dir / ins_method / f"cvrp{num_loc}_{dist}_train.pt"
    val_path   = data_dir / ins_method / f"cvrp{num_loc}_{dist}_val.pt"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {train_path}\n"
            f"Run: python -m rl4co.data.generate_slot_dataset "
            f"--num_locs {num_loc} --dist {dist} --out_dir {data_dir}"
        )

    # Data
    train_loader = make_dataloader(train_path, variant, t_cfg["batch"], shuffle=True, max_instances=t_cfg["n_train"])
    val_loader   = make_dataloader(val_path,   variant, t_cfg["batch"], shuffle=False, max_instances=t_cfg["n_val"])

    # Validate d_ins cost method (Variant D consumes d_ins). The data was baked
    # with a specific method; refuse a mismatch so we never train on the wrong cost.
    if variant == "D":
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
        optimizer_kwargs={"lr": t_cfg["lr"]},
    )

    if backbone == "l2r":
        model_kwargs["lower_neighbors_num"] = lower_neighbors_num
        model_kwargs["reduction_percentage"] = lower_neighbors_num
        
    if backbone == "am":
        model_kwargs["baseline"] = baseline if baseline is not None else "shared"
    # disable_slots: run backbone as a true no-slot baseline (no slot/aux).
    if disable_slots:
        model_kwargs["disable_slots"] = True
    model = model_cls(**model_kwargs)

    # run_name uniquely IDs the run (backbone, variant, K, N, dist, seed, ins_method,
    # and — for Variant D — the normalize/symmetrize target-aggregation flags).
    norm_tag = ""
    if variant == "D":
        norm_tag = f"_n{int(normalize_target)}s{int(symmetrize_target)}"

    base_suffix = f"_bl{baseline}" if backbone == "am" and baseline else ""
    if disable_slots:
        run_name = f"{backbone}_noslot_N{num_loc}_{dist}_seed{seed}{base_suffix}"
    else:
        run_name = (f"{backbone}_slot_{variant}_K{num_slots}_N{num_loc}_{dist}_"
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
        max_epochs=t_cfg["epochs"],
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
    print(f"Training {backbone} — Variant {variant} | N={num_loc} | {dist}")
    print(f"  Epochs: {t_cfg['epochs']}  Batch: {t_cfg['batch']}  LR: {t_cfg['lr']}")
    print(f"  Slots: K={num_slots}  proj_dim={proj_dim}  iters={slot_iters}")
    print(f"  ins_method: {ins_method}")
    print(f"  Output: {log_path}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)
    elapsed = time.time() - t0

    best_reward = checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score else None
    result = {
        "backbone": backbone,
        "variant": variant,
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
    results = json.loads(result_file.read_text()) if result_file.exists() else []
    dedup_key = {k: result[k] for k in (
        "backbone", "variant", "num_slots", "num_loc", "dist", "seed",
        "ins_method", "normalize_target", "symmetrize_target",
    )}
    results = [r for r in results if not all(r.get(k) == v for k, v in dedup_key.items())]
    results.append(result)
    result_file.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Best val reward: {best_reward:.4f} | {elapsed/60:.1f} min")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train POMOSlot -- Metric-Aware NCO")
    parser.add_argument("--variant",       type=str,   default="D",       choices=list("ABCD"),
                        help="Ablation variant. E is reserved (not implemented).")
    parser.add_argument("--num_loc",       type=int,   default=100,       choices=[50, 100, 200, 500, 1000])
    parser.add_argument("--dist",          type=str,   default="uniform", choices=["uniform", "clustered"])
    parser.add_argument("--data_dir",      type=str,   default="./data/slot_datasets_v2")
    parser.add_argument("--output",        type=str,   default="./output",
                        help="Root dir for logs + results/ablation_N{num_loc}.json")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        type=int,   default=0,
                        help="GPU index (0 or 1) to use; single GPU only.")
    parser.add_argument("--embed_dim",     type=int,   default=128)
    parser.add_argument("--num_slots",     type=int,   default=8)
    parser.add_argument("--proj_dim",      type=int,   default=64)
    parser.add_argument("--slot_iters",    type=int,   default=3)
    parser.add_argument("--lambda_init",   type=float, default=1.0)
    parser.add_argument("--lr_dual",       type=float, default=1e-4)
    parser.add_argument("--beta_entropy",  type=float, default=0.01,
                        help="Override the per-variant slot-entropy weight. Set 0.0 to "
                             "keep slots + metric loss but drop the entropy regulariser.")
    parser.add_argument("--ins_method",    type=str,   default="construction",
                        choices=["savings", "construction", "insertion"],
                        help="d_ins insertion-cost method. Must match the cached dataset's "
                             "'method' tag (the generator stamps it into the .pt).")
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--max_instances", type=int,   default=None,
                        help="Cap dataset size for quick smoke tests")
    parser.add_argument("--backbone",      type=str,   default="pomo", choices=["pomo", "am", "l2r"],
                        help="Backbone: 'pomo' (multi-start, shared baseline) or 'am' (single-start, rollout baseline)")
    parser.add_argument("--baseline",      type=str,   default=None,
                        help="REINFORCE baseline for the AM backbone (e.g. rollout, shared). Ignored for pomo.")
    parser.add_argument("--disable_slots", action="store_true",
                        help="Run the backbone as a true no-slot baseline (skips SlotAttention + aux losses).")
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
    parser.add_argument("--logger",        type=str,   default="csv", choices=["csv", "wandb"],
                        help="Logger: 'csv' (default, lightweight) or 'wandb' (requires wandb login).")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to a .ckpt to resume training from its last epoch (Lightning checkpoint).")
    parser.add_argument("--lower_neighbors_num", type=int, default=50,
                        help="L2R lower-model candidate count")
    parser.add_argument("--reduction_percentage", type=float, default=0.1,
                        help="L2R static farthest-edge reduction fraction")
    args = parser.parse_args()

    train(
        variant=args.variant,
        num_loc=args.num_loc,
        dist=args.dist,
        data_dir=args.data_dir,
        output=args.output,
        seed=args.seed,
        device=args.device,
        embed_dim=args.embed_dim,
        num_slots=args.num_slots,
        proj_dim=args.proj_dim,
        slot_iters=args.slot_iters,
        lambda_init=args.lambda_init,
        lr_dual=args.lr_dual,
        beta_entropy=args.beta_entropy,
        normalize_target=args.normalize_target,
        symmetrize_target=args.symmetrize_target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_instances=args.max_instances,
        backbone=args.backbone,
        baseline=args.baseline,
        disable_slots=args.disable_slots,
        ins_method=args.ins_method,
        logger=args.logger,
        resume=args.resume,
        lower_neighbors_num=args.lower_neighbors_num,
        reduction_percentage=args.reduction_percentage,
    )


if __name__ == "__main__":
    main()


