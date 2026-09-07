from __future__ import annotations

import argparse
import hashlib
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
    from rl4co.models.zoo.sil import SIL
    from rl4co.models.zoo.l2r import L2RModel
    from rl4co.models.zoo.icam import ICAMCVRP
    from rl4co.models.zoo.lehd import LEHDModel, TTRLModel
    from rl4co.models.zoo.lehd.model import make_lehd_dataloaders
    FULL_RL4CO = True
except Exception as e:
    print(f"[WARN] Full rl4co import failed: {e}")
    FULL_RL4CO = False


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "slot_datasets_v2"


MODEL_CLASSES = {
    "pomo": POMOSlot,
    "am": AMSlot,
    "l2r": L2RModel,
    "icam": ICAMCVRP,
    "sil": SIL,
    "lehd": LEHDModel,
    "ttpl": TTRLModel,
}


class SlotDataset(torch.utils.data.Dataset):
    """Wraps cached .pt files from generate_slot_dataset.py (sparse_v2)."""

    def __init__(self, filepath: str | Path, variant: str = "D", max_instances: int | None = None,
                 include_instance_id: bool = False):
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
            raise RuntimeError(
                f"Unknown dataset format_version: '{fmt}' in {filepath}")

        self.locs = data["locs"]     # (N_inst, N, 2)
        self.depot = data["depot"]    # (N_inst, 2)
        self.demand = data["demand"]   # (N_inst, N)
        self.capacity = data.get("capacity", None)
        # d_ins cost-method tag stamped by the generator; None for legacy datasets.
        self.method: str | None = data.get("method", None)

        # Sparse d_ins only needed for Variant D
        needs_dins = variant == "D"
        self.d_ins_idx = data.get(
            "d_ins_idx", None) if needs_dins else None  # (N_inst,N,k) int16
        self.d_ins_val = data.get(
            "d_ins_val", None) if needs_dins else None  # (N_inst,N,k) float32
        self.variant = variant
        self.include_instance_id = include_instance_id

        if max_instances is not None:
            self.locs = self.locs[:max_instances]
            self.depot = self.depot[:max_instances]
            self.demand = self.demand[:max_instances]
            if self.capacity is not None:
                self.capacity = self.capacity[:max_instances]
            if self.d_ins_idx is not None:
                self.d_ins_idx = self.d_ins_idx[:max_instances]
            if self.d_ins_val is not None:
                self.d_ins_val = self.d_ins_val[:max_instances]

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
        if self.include_instance_id:
            item["instance_id"] = torch.tensor(idx, dtype=torch.long)
        return item

    def signature(self):
        """Identify the actual cached CVRP inputs before reusing SIL labels."""
        digest = hashlib.sha256()
        for tensor in (self.locs, self.depot, self.demand):
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(memoryview(tensor.contiguous().numpy()).cast("B"))
        return digest.hexdigest()


def _collate_fn(batch: list[dict]) -> dict:
    """Collate dicts -> batched dict; shared_step converts to TensorDict internally."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def make_dataloader(filepath: str, variant: str, batch_size: int, shuffle: bool, max_instances: int | None = None,
                    include_instance_id: bool = False, seed: int = 42, num_workers: int = 4):
    ds = SlotDataset(filepath, variant=variant, max_instances=max_instances,
                     include_instance_id=include_instance_id)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
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
    1000: dict(epochs=200, batch=64, lr=5e-5, n_train=50_000, n_val=500),
}


def train(
    variant: str = "D",
    num_loc: int = 100,
    dist: str = "uniform",
    data_dir: str | Path = DEFAULT_DATA_DIR,
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
            max_instances=max_instances,
            logger=logger,
            resume=resume,
            lehd_data_path=lehd_data_path,
            lehd_val_data_path=lehd_val_data_path,
            lehd_decoder_layers=lehd_decoder_layers,
        )

    pl.seed_everything(seed)
    t_cfg = TRAIN_DEFAULTS[num_loc].copy()

    if epochs is not None:
        t_cfg["epochs"] = epochs
    if batch_size is not None:
        t_cfg["batch"] = batch_size
    if max_instances is not None:
        t_cfg["n_train"] = min(t_cfg["n_train"], max_instances)
        t_cfg["n_val"] = min(t_cfg["n_val"],   max(1, max_instances // 10))
    v_cfg = VARIANT_DEFAULTS[variant].copy()
    if beta_entropy is not None:
        v_cfg["beta_entropy"] = beta_entropy

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
            n_train=t_cfg["n_train"],
            n_val=t_cfg["n_val"],
            n_test=t_cfg["n_val"],
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
            f"--dist {dist} --n_train {t_cfg['n_train']} --n_val {t_cfg['n_val']} "
            f"--n_test {t_cfg['n_val']} --out_dir {data_dir} --method {ins_method} "
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
    data_variant = "none" if backbone == "sil" else variant
    loader_kwargs = dict(seed=seed, num_workers=num_workers)
    train_loader = make_dataloader(train_path, data_variant, t_cfg["batch"], shuffle=True,
                                   max_instances=t_cfg["n_train"], include_instance_id=backbone == "sil", **loader_kwargs)
    val_loader = make_dataloader(val_path, data_variant, t_cfg["batch"], shuffle=False,
                                 max_instances=t_cfg["n_val"], **loader_kwargs)
    for loader in (train_loader, val_loader):
        if len(loader.dataset) == 0 or loader.dataset.locs.shape[1] != num_loc:
            raise ValueError(
                f"Expected a nonempty dataset with {num_loc} customers")

    # Validate d_ins cost method (Variant D consumes d_ins). The data was baked
    # with a specific method; refuse a mismatch so we never train on the wrong cost.
    if variant == "D" and backbone != "sil":
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
        optimizer_kwargs={"lr": t_cfg["lr"]},
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
            optimizer_kwargs={"lr": t_cfg["lr"]},
        )
    elif backbone == "sil":
        model_kwargs = dict(
            env=env, embed_dim=embed_dim, num_layers=sil_num_layers,
            repair_budget=sil_repair_budget, improve_every=sil_improve_every,
            max_subtour_length=sil_max_subtour_length,
            parallel_reconstruction=sil_parallel_reconstruction,
            update_mode=sil_update_mode,
            optimizer_kwargs={"lr": t_cfg["lr"]},
        )
    elif backbone == "am":
        model_kwargs["baseline"] = baseline if baseline is not None else "shared"
    # disable_slots: run backbone as a true no-slot baseline (no slot/aux).
    if disable_slots and backbone != "sil":
        model_kwargs["disable_slots"] = True
    model = model_cls(**model_kwargs)
    if backbone == "sil":
        model.dataset_signature = train_loader.dataset.signature()

    # run_name uniquely IDs the run (backbone, variant, K, N, dist, seed, ins_method,
    # and — for Variant D — the normalize/symmetrize target-aggregation flags).
    norm_tag = ""
    if variant == "D":
        norm_tag = f"_n{int(normalize_target)}s{int(symmetrize_target)}"

    base_suffix = f"_bl{baseline}" if backbone == "am" and baseline else ""
    if backbone == "sil":
        run_name = (f"sil_N{num_loc}_{dist}_{ins_method}_seed{seed}_d{embed_dim}"
                    f"_l{sil_num_layers}_r{sil_repair_budget}_i{sil_improve_every}"
                    f"_s{sil_max_subtour_length}_u{sil_update_mode}"
                    f"_prc{int(sil_parallel_reconstruction)}")
    elif disable_slots:
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
        gradient_clip_val=None if backbone == "sil" else 1.0,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer = pl.Trainer(**trainer_kwargs)

    print(f"\n{'='*60}")
    print(f"Training {model_cls.__name__} | N={num_loc} | {dist}")
    print(
        f"  Epochs: {t_cfg['epochs']}  Batch: {t_cfg['batch']}  LR: {t_cfg['lr']}")
    if backbone == "sil":
        print(f"  SIL: repair_budget={sil_repair_budget}, improve_every={sil_improve_every}, "
              f"max_subtour_length={sil_max_subtour_length}, update_mode={sil_update_mode}, "
              f"PRC={sil_parallel_reconstruction}")
        print("  Slot/metric/entropy flags do not apply to SIL; ins_method selects the shared data folder.")
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
        "variant": None if backbone == "sil" else variant,
        "num_slots": None if backbone == "sil" else num_slots,
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
        "batch_size": t_cfg["batch"],
        "lr": t_cfg["lr"],
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

    # Dedup identical configs: rerun replaces, never appends a duplicate row.
    result_dir = Path(output)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"ablation_N{num_loc}.json"
    results = json.loads(result_file.read_text()
                         ) if result_file.exists() else []
    dedup_key = {k: result[k] for k in (
        "backbone", "variant", "num_slots", "num_loc", "dist", "seed",
        "ins_method", "normalize_target", "symmetrize_target",
    )}
    if backbone == "sil":
        dedup_key.update(
            {k: v for k, v in result.items() if k.startswith("sil_")})
        dedup_key.update({k: result[k] for k in (
            "embed_dim", "batch_size", "dataset_signature")})
    results = [r for r in results if not all(
        r.get(k) == v for k, v in dedup_key.items())]
    results.append(result)
    result_file.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Best val reward: {best_reward} | {elapsed/60:.1f} min")
    return result


# ---------------------------------------------------------------------------
# LEHD / TTPL standalone training branch
# ---------------------------------------------------------------------------

def _train_lehd(
    backbone: str,
    num_loc: int,
    output: str,
    seed: int,
    device: int,
    embed_dim: int,
    epochs: int | None,
    batch_size: int | None,
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

    # Use MARS's TRAIN_DEFAULTS for the same epochs / batch / lr as MARS
    t_cfg = TRAIN_DEFAULTS[num_loc].copy()
    if epochs is not None:
        t_cfg["epochs"] = epochs
    if batch_size is not None:
        t_cfg["batch"] = batch_size

    n_train = t_cfg["n_train"]
    n_val = t_cfg["n_val"]
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
        optimizer_kwargs={"lr": t_cfg["lr"]},
    )

    train_loader, val_loader = make_lehd_dataloaders(
        n_train=n_train,
        n_val=n_val,
        batch_size=t_cfg["batch"],
        seed=seed,
        num_workers=0,  # data lives inside the model; no worker overhead needed
    )

    run_name = (
        f"{backbone}_N{num_loc}_seed{seed}_d{embed_dim}"
        f"_dec{lehd_decoder_layers}"
    )
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

    use_cuda = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=t_cfg["epochs"],
        accelerator="gpu" if use_cuda else "cpu",
        devices=[device] if use_cuda else 1,
        strategy="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger_obj,
        # LEHD clips manually per step (not used here)
        gradient_clip_val=None,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    print(f"\n{'='*60}")
    print(f"Training {model_cls.__name__} | N={num_loc}")
    print(
        f"  Epochs: {t_cfg['epochs']}  Batch: {t_cfg['batch']}  LR: {t_cfg['lr']}")
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
        "batch_size": t_cfg["batch"],
        "lr": t_cfg["lr"],
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
    parser = argparse.ArgumentParser(
        description="Train MARS or SIL on shared cached CVRP data")
    parser.add_argument("--variant",       type=str,   default="D",       choices=list("ABCD"),
                        help="Ablation variant. E is reserved (not implemented).")
    parser.add_argument("--num_loc",       type=int,
                        default=100,       choices=[50, 100, 200, 500, 1000])
    parser.add_argument("--dist",          type=str,
                        default="uniform", choices=["uniform", "clustered"])
    parser.add_argument("--data_dir",      type=str,   default=str(DEFAULT_DATA_DIR),
                        help="Shared cached dataset root (default is anchored to the MARS repository)")
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
    parser.add_argument("--lower_neighbors_num", type=int, default=50)
    parser.add_argument("--reduction_percentage", type=float, default=0.1)
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--max_instances", type=int,   default=None,
                        help="Cap dataset size for quick smoke tests")
    parser.add_argument("--backbone",      type=str,   default="pomo",
                        choices=["pomo", "am", "l2r",
                                 "icam", "sil", "lehd", "ttpl"],
                        help="pomo/am slot models, SIL self-improved baseline, "
                             "LEHD (NeurIPS23) or TTPL (NeurIPS25) imitation baseline")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--generate_missing_data", action="store_true",
                        help="Generate missing shared train/val/test splits before training")
    parser.add_argument("--generation_chunk_size", type=int, default=None,
                        help="Dataset generation chunk size (default adapts to num_loc)")
    parser.add_argument("--sil_repair_budget", type=int, default=5,
                        help="Reconstruction passes per SIL improvement round (0 disables improvement)")
    parser.add_argument("--sil_improve_every", type=int, default=20,
                        help="Epochs of imitation between SIL label improvement rounds")
    parser.add_argument("--sil_max_subtour_length", type=int, default=64,
                        help="Maximum sampled SIL subpath length; 64 is the comparable fast default")
    parser.add_argument("--sil_num_layers", type=int, default=6)
    parser.add_argument("--sil_update_mode", choices=["batch", "node"], default="batch",
                        help="batch: one optimizer update per batch (comparable); node: upstream per-node updates")
    parser.add_argument("--sil_no_prc", dest="sil_parallel_reconstruction", action="store_false",
                        help="Reconstruct one subpath per instance instead of parallel disjoint subpaths")
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
    # ---- LEHD / TTPL specific ----
    parser.add_argument("--lehd_data_path", type=str, default=None,
                        help="Path to LEHD-format .txt training file (required for --backbone lehd/ttpl). "
                             "Download from: https://drive.google.com/drive/folders/1LptBUGVxQlCZeWVxmCzUOf9WPlsqOROR")
    parser.add_argument("--lehd_val_data_path", type=str, default=None,
                        help="Path to LEHD-format .txt validation file (optional; defaults to training file).")
    parser.add_argument("--lehd_decoder_layers", type=int, default=6,
                        help="Number of heavy-decoder Transformer layers for LEHD/TTPL (default: 6).")
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
        lower_neighbors_num=args.lower_neighbors_num,
        reduction_percentage=args.reduction_percentage,
        logger=args.logger,
        resume=args.resume,
        num_workers=args.num_workers,
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
    )


if __name__ == "__main__":
    main()
