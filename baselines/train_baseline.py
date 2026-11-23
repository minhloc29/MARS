"""Standalone CLI for routing baselines: sil, invit, dgl, elg, radar, lehd, ttpl.

The LEGO slot backbones (pomo/am/l2r/icam) live in ``train.py``. Their competing
baselines keep their own training loop here so ``train.py`` stays about LEGO.

Two groups:
  * slot-dataset baselines (sil/invit/dgl/elg/radar): reuse the shared MARS/SIL
    slot-format data via :func:`make_dataloader`, but consume the raw metric
    (``variant="none"``) and no slots/aux.
  * LEHD/TTPL (lehd/ttpl): train on an external LEHD-format ``.txt`` dataset via
    the model's own loader. Require ``--lehd_data_path``.

Invoke directly, or through ``train.py`` which dispatches --backbone here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

from pathlib import Path

import lightning.pytorch as pl

from rl4co.data.slot_dataset import make_dataloader

try:
    from rl4co.models.zoo.dgl import DGL
    from rl4co.models.zoo.elg import ELG
    from rl4co.models.zoo.invit import INViT
    from rl4co.models.zoo.lehd import LEHDModel, TTRLModel
    from rl4co.models.zoo.lehd.model import make_lehd_dataloaders
    from rl4co.models.zoo.radar import RADAR
    from rl4co.models.zoo.sil import SIL
    BASELINES_OK = True
except Exception as e:  # models may fail to import on minimal installs
    raise RuntimeError(
        "Baseline backbones (sil/invit/dgl/elg/radar/lehd/ttpl) could not be "
        "imported. LEGO backbones (pomo/am/l2r/icam) train via train.py and do "
        "not need these model packages."
    ) from e


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = REPO_ROOT / "data" / "slot_datasets_v2"
MARS_DATA_DIR = REPO_ROOT.parent / "MARS" / "data" / "slot_datasets_v2"
DEFAULT_DATA_DIR = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else MARS_DATA_DIR

CONFIG_DIR = REPO_ROOT / "configs" / "baselines"

# sil/invit/dgl/elg/radar train on the shared slot-format data (variant none).
SLOT_DATASET_BACKBONES = {"sil", "invit", "dgl", "elg", "radar"}
# lehd/ttpl train on their own external LEHD-format data.
EXTERNAL_DATA_BACKBONES = {"lehd", "ttpl"}
SLOT_BACKBONES = (
    "pomo", "am", "l2r", "icam",
)  # LEGO backbones; handled by train.py, not here.
ALL_BACKBONES = SLOT_DATASET_BACKBONES | EXTERNAL_DATA_BACKBONES | set(SLOT_BACKBONES)

MODEL_CLASSES = {
    "invit": INViT,
    "dgl": DGL,
    "elg": ELG,
    "radar": RADAR,
    "sil": SIL,
    "lehd": LEHDModel,
    "ttpl": TTRLModel,
}

CONFIG_PREFIX = {"sil": "sil_", "invit": "invit_", "dgl": "dgl_",
                 "elg": "elg_", "radar": "radar_"}


def config_path(backbone: str, config: str | None = None) -> Path:
    """Resolve a baseline's YAML: --config override, else configs/baselines/<backbone>.yaml.

    lehd/ttpl share ``lehd.yaml``. Raises FileNotFoundError if missing.
    """
    name = "lehd" if backbone in EXTERNAL_DATA_BACKBONES else backbone
    path = Path(config) if config else CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline config for '{backbone}' not found: {path}\n"
            f"Place a YAML at {CONFIG_DIR / (name + '.yaml')} or pass --config <path>."
        )
    return path


def load_baseline_config(backbone: str, config: str | None = None) -> dict:
    """Load a baseline's hyperparameters from YAML."""
    import yaml
    path = config_path(backbone, config)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    print(f"Loaded {backbone} baseline config from {path}")
    return cfg


def make_trainer(run_name: str, log_path: Path, epochs: int,
                 device: int | list[int], logger: str):
    """Build a Lightning Trainer for a baseline run (no gradient clip)."""
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

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
        try:
            from lightning.pytorch.loggers import WandbLogger
        except Exception:
            raise RuntimeError(
                "wandb requested but WandbLogger is not installed. "
                "Run `pip install wandb lightning` and login with `wandb login`.")
        logger_obj = WandbLogger(project="MeTRA_Slot_NCO", name=run_name,
                                 log_model="all")
    else:
        logger_obj = CSVLogger(save_dir=str(log_path), name="metrics")

    import torch
    use_cuda = torch.cuda.is_available()
    devices = device if isinstance(device, list) else [device]
    strategy = "ddp" if len(devices) > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_cuda else "cpu",
        devices=devices if use_cuda else 1,
        strategy=strategy,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger_obj,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    return trainer, checkpoint_cb, early_stop_cb


def _record(result_dir: Path, num_loc: int, result: dict, dedup_key: dict,
            csv_row: dict | None):
    """Append a deduplicated result row to ablation_N{num_loc}.json."""
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"ablation_N{num_loc}.json"
    results = json.loads(result_file.read_text()) if result_file.exists() else []
    results = [r for r in results if not all(
        r.get(k) == v for k, v in dedup_key.items())]
    results.append(result)
    result_file.write_text(json.dumps(results, indent=2))

    if csv_row is not None:
        import csv as _csv
        csv_path = result_dir / "teacher_ablation.csv"
        cols = ["teacher", "cross_route_mode", "match_scale", "num_loc", "dist",
                "problem", "seed", "best_val_reward", "elapsed_min",
                "checkpoint", "train_path", "val_path"]
        exists = csv_path.exists()
        try:
            with open(csv_path, "a", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                if not exists:
                    w.writeheader()
                w.writerow(csv_row)
        except Exception as _e:
            print(f"[WARN] could not write teacher_ablation.csv: {_e}")


def train_slot_dataset_baseline(backbone: str, num_loc: int, dist: str,
                                problem: str, data_dir: Path, output: Path,
                                seed: int, device, ins_method: str,
                                teacher: str, match_scale: bool,
                                cross_route_mode: str, embed_dim: int,
                                batch_size: int, lr: float, epochs: int,
                                n_train: int, n_val: int,
                                max_instances: int | None, logger: str,
                                resume: str | None, num_workers: int,
                                generate_missing_data: bool,
                                generation_chunk_size: int | None,
                                config: dict) -> dict:
    """Train a sil/invit/dgl/elg/radar baseline on the shared slot-format data."""
    from rl4co.data.generate_slot_dataset import teacher_subdir
    from rl4co.envs import CVRPEnv, CVRPTWEnv, OVRPEnv

    n_train_eff = min(n_train, max_instances) if max_instances is not None else n_train
    n_val_eff = min(n_val, max(1, max_instances // 10)
                    ) if max_instances is not None else n_val

    method_dir = data_dir / teacher_subdir(ins_method, teacher, match_scale, cross_route_mode)
    prefix = f"{problem}{num_loc}"
    train_path = method_dir / f"{prefix}_{dist}_train.pt"
    val_path = method_dir / f"{prefix}_{dist}_val.pt"

    missing_paths = [p for p in (train_path, val_path) if not p.exists()]
    if missing_paths and generate_missing_data:
        from rl4co.data.generate_slot_dataset import generate_and_save
        print(
            f"Missing {len(missing_paths)} required split(s); generating the shared "
            f"N={num_loc} {dist} {problem} dataset (teacher={teacher}) in {data_dir}."
        )
        generate_and_save(
            out_dir=data_dir, n=num_loc, dist=dist, n_train=n_train_eff,
            n_val=n_val_eff, n_test=n_val_eff, k_neighbors=15, method=ins_method,
            chunk_size=generation_chunk_size, seed=seed, problem=problem,
            teacher=teacher, match_scale=match_scale,
            cross_route_mode=cross_route_mode,
        )
        pl.seed_everything(seed)
        missing_paths = [p for p in (train_path, val_path) if not p.exists()]

    if missing_paths:
        preparation = (
            f"python -m rl4co.data.generate_slot_dataset --num_locs {num_loc} "
            f"--dist {dist} --n_train {n_train_eff} --n_val {n_val_eff} "
            f"--n_test {n_val_eff} --out_dir {data_dir} --method {ins_method} "
            f"--teacher {teacher} --problem {problem} --seed {seed}"
        )
        raise FileNotFoundError(
            "Missing required cached dataset split(s):\n  "
            + "\n  ".join(str(p) for p in missing_paths)
            + "\n\nPrepare the shared MARS/SIL data once with:\n  " + preparation
            + "\n\nOr append --generate_missing_data to the training command. "
              "N=1000 preparation is computationally expensive."
        )

    # Baselines consume the raw metric (variant="none"); sil/dgl need instance ids.
    loader_kwargs = dict(seed=seed, num_workers=num_workers)
    train_loader = make_dataloader(train_path, batch_size, shuffle=True,
                                   variant="none", max_instances=n_train_eff,
                                   include_instance_id=backbone in {"sil", "dgl"},
                                   **loader_kwargs)
    val_loader = make_dataloader(val_path, batch_size, shuffle=False,
                                 variant="none", max_instances=n_val_eff,
                                 **loader_kwargs)
    for loader in (train_loader, val_loader):
        if len(loader.dataset) == 0 or loader.dataset.locs.shape[1] != num_loc:
            raise ValueError(
                f"Expected a nonempty dataset with {num_loc} customers")

    env_cls = {"cvrp": CVRPEnv, "ovrp": OVRPEnv, "vrptw": CVRPTWEnv}[problem]
    env = env_cls(generator_params=dict(num_loc=num_loc))

    model_cls = MODEL_CLASSES[backbone]
    if backbone == "sil":
        model_kwargs = dict(env=env, embed_dim=embed_dim,
                            optimizer_kwargs={"lr": lr}, **config)
    elif backbone == "invit":
        cfg = dict(config)
        cfg["state_sizes"] = tuple(cfg["state_sizes"])
        model_kwargs = dict(env=env, embed_dim=embed_dim, problem=problem,
                            optimizer_kwargs={"lr": lr}, **cfg)
    elif backbone in ("dgl", "elg"):
        model_kwargs = dict(env=env, embed_dim=embed_dim, problem=problem,
                            optimizer_kwargs={"lr": lr}, **config)
    elif backbone == "radar":
        model_kwargs = dict(env=env, problem=problem,
                            optimizer_kwargs={"lr": lr}, **config)
    model = model_cls(**model_kwargs)
    if backbone in ("sil", "dgl"):
        model.dataset_signature = train_loader.dataset.signature()

    # Baseline runs carry a compact config digest so distinct hyperparameters
    # (from YAML or --config) don't collide in output dirs.
    cfg_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:6]
    run_name = (f"{backbone}_N{num_loc}_{dist}_{ins_method}_seed{seed}"
                f"_d{embed_dim}_c{cfg_digest}")
    if problem != "cvrp":
        run_name = f"{run_name}_prob{problem}"
    log_path = output / run_name

    trainer, checkpoint_cb, _ = make_trainer(run_name, log_path, epochs, device, logger)

    print(f"\n{'='*60}")
    print(f"Training {model_cls.__name__} baseline | N={num_loc} | {dist}")
    print(f"  Epochs: {epochs}  Batch: {batch_size}  LR: {lr}")
    print(f"  {backbone.upper()} config: {json.dumps(config, sort_keys=True)}")
    print("  Slot/metric/entropy flags do not apply to baselines; ins_method selects the shared data folder.")
    print(f"  ins_method: {ins_method}")
    print(f"  Output: {log_path}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)
    elapsed = time.time() - t0

    best_reward = (checkpoint_cb.best_model_score.item()
                   if checkpoint_cb.best_model_score else None)
    result = {
        "backbone": backbone,
        "metric_variant": None,
        "num_slots": None,
        "num_loc": num_loc,
        "dist": dist,
        "problem": problem,
        "seed": seed,
        "ins_method": ins_method,
        "teacher": teacher,
        "cross_route_mode": cross_route_mode if teacher == "nn_split" else None,
        "match_scale": match_scale,
        "normalize_target": None,
        "symmetrize_target": None,
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
    prefix = CONFIG_PREFIX[backbone]
    cfg_view = dict(config)
    if backbone == "invit" and "state_sizes" in cfg_view:
        cfg_view["state_sizes"] = list(cfg_view["state_sizes"])
    if backbone in ("dgl", "sil"):
        cfg_view["dataset_signature"] = model.dataset_signature
    result.update(**{f"{prefix}{k}": v for k, v in cfg_view.items()})

    dedup_key = {k: result[k] for k in (
        "backbone", "metric_variant", "num_slots", "num_loc", "dist", "seed",
        "ins_method", "teacher", "cross_route_mode", "match_scale",
        "normalize_target", "symmetrize_target",
    )}
    if problem != "cvrp":
        dedup_key["problem"] = problem
    dedup_key.update(
        {k: v for k, v in result.items() if k.startswith(prefix)})
    if backbone in ("dgl", "sil"):
        dedup_key["dataset_signature"] = result["dataset_signature"]
    dedup_key.update({"embed_dim": embed_dim, "batch_size": batch_size, "lr": lr})

    csv_row = {k: result[k] for k in (
        "teacher", "cross_route_mode", "match_scale", "num_loc", "dist",
        "problem", "seed", "best_val_reward", "elapsed_min",
        "checkpoint", "train_path", "val_path")}
    _record(output, num_loc, result, dedup_key, csv_row)

    print(f"\nDone. Best val reward: {best_reward} | {elapsed/60:.1f} min")
    return result


def train_lehd(backbone: str, num_loc: int, output: Path, seed: int, device,
               embed_dim: int, epochs: int, batch_size: int, lr: float,
               n_train: int, n_val: int, max_instances: int | None,
               logger: str, resume: str | None, lehd_data_path: str | None,
               lehd_val_data_path: str | None, config: dict) -> dict:
    """Train LEHD or TTPL on its own external LEHD-format .txt data."""
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

    model_cls = MODEL_CLASSES[backbone]
    model = model_cls(
        data_path=lehd_data_path,
        val_data_path=lehd_val_data_path,
        num_loc=num_loc,
        embed_dim=embed_dim,
        decoder_layer_num=config["decoder_layers"],
        n_train_episodes=n_train,
        n_val_episodes=n_val,
        optimizer_kwargs={"lr": lr},
    )

    train_loader, val_loader = make_lehd_dataloaders(
        n_train=n_train, n_val=n_val, batch_size=batch_size, seed=seed,
        num_workers=0,  # data lives inside the model; no worker overhead needed
    )

    run_name = (
        f"{backbone}_N{num_loc}_seed{seed}_d{embed_dim}"
        f"_dec{config['decoder_layers']}"
    )
    log_path = output / run_name

    trainer, checkpoint_cb, _ = make_trainer(run_name, log_path, epochs, device, logger)

    print(f"\n{'='*60}")
    print(f"Training {model_cls.__name__} baseline | N={num_loc}")
    print(f"  Epochs: {epochs}  Batch: {batch_size}  LR: {lr}")
    print(f"  embed_dim: {embed_dim}  decoder_layers: {config['decoder_layers']}")
    print(f"  n_train: {n_train}  n_val: {n_val}")
    print(f"  data: {lehd_data_path}")
    print(f"  Output: {log_path}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)
    elapsed = time.time() - t0

    best_reward = (checkpoint_cb.best_model_score.item()
                   if checkpoint_cb.best_model_score else None)
    result = {
        "backbone": backbone,
        "num_loc": num_loc,
        "seed": seed,
        "embed_dim": embed_dim,
        "decoder_layers": config["decoder_layers"],
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
    dedup_key = {k: result[k]
                 for k in ("backbone", "num_loc", "seed", "embed_dim")}
    _record(output, num_loc, result, dedup_key, csv_row=None)

    print(f"\nDone. Best val reward: {best_reward} | {elapsed/60:.1f} min")
    return result


def run(backbone: str, config: dict | None = None, **kwargs):
    """Entry point used by train.py's thin dispatch.

    ``kwargs`` carries the shared train() keyword arguments; unknown keys are
    tolerated. Returns the result dict.
    """
    if backbone in SLOT_BACKBONES:
        raise ValueError(
            f"--backbone {backbone} is a LEGO slot backbone; train it with "
            f"`python train.py --backbone {backbone}` instead of this baseline CLI.")
    cfg = config if config is not None else {}
    if backbone in EXTERNAL_DATA_BACKBONES:
        return train_lehd(
            backbone=backbone,
            num_loc=kwargs["num_loc"],
            output=Path(kwargs["output"]),
            seed=kwargs["seed"],
            device=kwargs["device"],
            embed_dim=kwargs["embed_dim"],
            epochs=kwargs["epochs"],
            batch_size=kwargs["batch_size"],
            lr=kwargs["lr"],
            n_train=kwargs["n_train"],
            n_val=kwargs["n_val"],
            max_instances=kwargs["max_instances"],
            logger=kwargs["logger"],
            resume=kwargs["resume"],
            lehd_data_path=kwargs["lehd_data_path"],
            lehd_val_data_path=kwargs["lehd_val_data_path"],
            config=cfg,
        )
    return train_slot_dataset_baseline(
        backbone=backbone,
        num_loc=kwargs["num_loc"],
        dist=kwargs["dist"],
        problem=kwargs["problem"],
        data_dir=Path(kwargs["data_dir"]),
        output=Path(kwargs["output"]),
        seed=kwargs["seed"],
        device=kwargs["device"],
        ins_method=kwargs["ins_method"],
        teacher=kwargs["teacher"],
        match_scale=kwargs["match_scale"],
        cross_route_mode=kwargs["cross_route_mode"],
        embed_dim=kwargs["embed_dim"],
        batch_size=kwargs["batch_size"],
        lr=kwargs["lr"],
        epochs=kwargs["epochs"],
        n_train=kwargs["n_train"],
        n_val=kwargs["n_val"],
        max_instances=kwargs["max_instances"],
        logger=kwargs["logger"],
        resume=kwargs["resume"],
        num_workers=kwargs["num_workers"],
        generate_missing_data=kwargs["generate_missing_data"],
        generation_chunk_size=kwargs["generation_chunk_size"],
        config=cfg,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train routing baselines")
    p.add_argument("--num_loc", type=int, default=100,
                   choices=[50, 100, 200, 500, 1000])
    p.add_argument("--dist", default="uniform", choices=["uniform", "clustered"])
    p.add_argument("--problem", default="cvrp", choices=["cvrp", "ovrp", "vrptw"])
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output", default="./output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, nargs="+", default=[0])
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_val", type=int, default=1_000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--backbone", default="sil",
                   choices=sorted(ALL_BACKBONES))
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--ins_method", default="construction",
                   choices=["savings", "construction", "insertion"])
    p.add_argument("--teacher", default="nn",
                   choices=["nn", "random", "sweep", "nn_2opt", "nn_split", "hgs"])
    p.add_argument("--cross_route_mode", default="hard", choices=["hard", "soft"])
    p.add_argument("--match_scale", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--generate_missing_data", action="store_true")
    p.add_argument("--generation_chunk_size", type=int, default=None)
    p.add_argument("--logger", default="csv", choices=["csv", "wandb"])
    p.add_argument("--resume", default=None)
    p.add_argument("--lehd_data_path", default=None)
    p.add_argument("--lehd_val_data_path", default=None)
    p.add_argument(
        "--config", default=None,
        help="Path to a baseline hyperparameter YAML (defaults to "
             "configs/baselines/<backbone>.yaml).")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    config = load_baseline_config(args.backbone, args.config)
    run(config=config, **vars(args))


if __name__ == "__main__":
    main()
