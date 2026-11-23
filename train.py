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
    from rl4co.envs import CVRPTWEnv, CVRPEnv, OVRPEnv
    from rl4co.models.zoo.icam import ICAMCVRP
    from rl4co.models.zoo.l2r import L2RModel
    from rl4co.models.zoo.pomo_slot import AMSlot, POMOSlot
    FULL_RL4CO = True
except Exception as e:
    raise RuntimeError(
        "Failed to import the routing environments or model backbones. "
        "Check that this clone contains all modules referenced by its package initializers."
    ) from e


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_DIR = REPO_ROOT / "data" / "slot_datasets_v2"
MARS_DATA_DIR = REPO_ROOT.parent / "MARS" / "data" / "slot_datasets_v2"
DEFAULT_DATA_DIR = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else MARS_DATA_DIR


MODEL_CLASSES = {
    "pomo": POMOSlot,
    "am": AMSlot,
    "l2r": L2RModel,
    "icam": ICAMCVRP,
}

# LEGO slot backbones trained directly in this module.
LEGO_BACKBONES = set(MODEL_CLASSES)
# Routing baselines living in baselines/train_baseline.py. main() dispatches them
# there, so a baseline stays CLI-runnable via train.py too.
BASELINE_BACKBONES = {"sil", "invit", "dgl", "elg", "radar", "lehd", "ttpl"}


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
    problem: str = "cvrp",
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
    teacher: str = "nn",
    match_scale: bool = False,
    cross_route_mode: str = "hard",
    lower_neighbors_num: int = 50,
    reduction_percentage: float = 0.1,
    logger: str = "csv",
    resume: str | None = None,
    num_workers: int = 4,
    generate_missing_data: bool = False,
    generation_chunk_size: int | None = None,
):
    assert FULL_RL4CO, (
        "Full rl4co import failed. Ensure torchrl DLL is installed correctly "
        "or run on a compatible machine."
    )
    if backbone not in LEGO_BACKBONES:
        raise ValueError(
            f"--backbone {backbone!r} is a routing baseline, not a LEGO slot "
            f"backbone. Run it with `python -m baselines.train_baseline "
            f"--backbone {backbone}` (or `python train.py --backbone {backbone}`, "
            f"which dispatches to it)."
        )

    pl.seed_everything(seed)
    n_train_eff = min(
        n_train, max_instances) if max_instances is not None else n_train
    n_val_eff = min(n_val, max(1, max_instances // 10)
                    ) if max_instances is not None else n_val
    v_cfg = dict(metric_variant=metric_variant, alpha_metric=alpha_metric,
                 beta_entropy=beta_entropy)

    data_dir = Path(data_dir)

    # Teacher-aware cache location, shared with the generator via the centralised
    # helper so training and data-generation always agree on the layout.
    from rl4co.data.generate_slot_dataset import teacher_subdir
    method_dir = data_dir / teacher_subdir(ins_method, teacher, match_scale, cross_route_mode)
    prefix = f"{problem}{num_loc}"
    train_path = method_dir / f"{prefix}_{dist}_train.pt"
    val_path = method_dir / f"{prefix}_{dist}_val.pt"

    missing_paths = [path for path in (
        train_path, val_path) if not path.exists()]
    if missing_paths and generate_missing_data:
        from rl4co.data.generate_slot_dataset import generate_and_save

        print(
            f"Missing {len(missing_paths)} required split(s); generating the shared "
            f"N={num_loc} {dist} {problem} dataset (teacher={teacher}) in {data_dir}."
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
            problem=problem,
            teacher=teacher,
            match_scale=match_scale,
            cross_route_mode=cross_route_mode,
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
            f"--teacher {teacher} --problem {problem} --seed {seed}"
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
    data_variant = metric_variant
    loader_kwargs = dict(seed=seed, num_workers=num_workers)
    train_loader = make_dataloader(train_path, batch_size, shuffle=True,
                                   variant=data_variant, max_instances=n_train_eff, **loader_kwargs)
    val_loader = make_dataloader(val_path, batch_size, shuffle=False,
                                 variant=data_variant, max_instances=n_val_eff, **loader_kwargs)
    for loader in (train_loader, val_loader):
        if len(loader.dataset) == 0 or loader.dataset.locs.shape[1] != num_loc:
            raise ValueError(
                f"Expected a nonempty dataset with {num_loc} customers")

    # Validate d_ins cost method (Variant D consumes d_ins). The data was baked
    # with a specific method; refuse a mismatch so we never train on the wrong cost.
    if metric_variant == "D":
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

    # Validate reference-tour teacher tag (Variant D consumes d_cons from a specific
    # teacher's order). A teacher mismatch silently trains on the wrong geometry, so
    # refuse it explicitly except for legacy data stamped before the teacher field.
    if metric_variant == "D":
        data_teacher = getattr(train_loader.dataset, "teacher", None)
        if data_teacher is not None and data_teacher != teacher:
            raise RuntimeError(
                f"teacher mismatch: --teacher={teacher!r} but cached dataset "
                f"{train_path} was generated with teacher={data_teacher!r} (see the "
                f"'teacher' tag in the .pt). Regenerate with the requested teacher "
                f"or pass --teacher {data_teacher}."
            )
        elif data_teacher is None:
            print(f"[WARN] {train_path} has no 'teacher' tag (legacy dataset) — "
                  f"assuming {teacher!r} (pre-teacher caches are always nn).")
        else:
            print(f"Dataset teacher '{data_teacher}' matches --teacher. OK.")

    # Environment. All three problems reuse the CVRP node features: ovrp drops
    # the return-to-depot arcs (open routes), vrptw attaches Solomon time windows
    # so the CVRPTWEnv simulates time-window feasibility during rollout. The
    # geometry teacher / init-context embeddings stay on the CVRP node features.
    env_cls = {  # constraint-agnostic by design: same model body, env differs only
        # in reward / time-window simulation.
        "cvrp": CVRPEnv,
        "ovrp": OVRPEnv,
        "vrptw": CVRPTWEnv,
    }[problem]
    env = env_cls(generator_params=dict(num_loc=num_loc))

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
        # L2R's reward is closed-tour (cvrp) or open-route (ovrp); the env and
        # its capacity/reduction logic are identical either way. Needs to be
        # threaded explicitly because the shared model_kwargs block omits it.
        model_kwargs["problem"] = problem
    elif backbone == "icam":
        model_kwargs = dict(
            env=env,
            embed_dim=embed_dim,
            num_starts=num_loc,
            problem=problem,
            optimizer_kwargs={"lr": lr},
        )
    elif backbone == "am":
        model_kwargs["baseline"] = baseline if baseline is not None else "shared"
    # disable_slots: run backbone as a true no-slot baseline (no slot/aux).
    if disable_slots:
        model_kwargs["disable_slots"] = True
    model = model_cls(**model_kwargs)

    # run_name uniquely IDs the run (backbone, variant, K, N, dist, seed, ins_method,
    # and — for Variant D — the normalize/symmetrize target-aggregation flags).
    norm_tag = ""
    if metric_variant == "D":
        norm_tag = f"_n{int(normalize_target)}s{int(symmetrize_target)}"

    base_suffix = f"_bl{baseline}" if backbone == "am" and baseline else ""
    if disable_slots:
        run_name = f"{backbone}_noslot_N{num_loc}_{dist}_seed{seed}{base_suffix}"
    else:
        run_name = (f"{backbone}_slot_{metric_variant}_K{num_slots}_N{num_loc}_{dist}_"
                    f"{ins_method}{norm_tag}_seed{seed}{base_suffix}")
    if problem != "cvrp":
        run_name = f"{run_name}_prob{problem}"
    log_path = Path(output) / run_name

    trainer, checkpoint_cb, _ = _make_trainer(
        run_name, log_path, epochs, device, logger, gradient_clip_val=1.0,
    )

    print(f"\n{'='*60}")
    print(f"Training {model_cls.__name__} | N={num_loc} | {dist}")
    print(
        f"  Epochs: {epochs}  Batch: {batch_size}  LR: {lr}")
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
        "metric_variant": metric_variant,
        "num_slots": num_slots,
        "num_loc": num_loc,
        "dist": dist,
        "problem": problem,
        "seed": seed,
        "ins_method": ins_method,
        "teacher": teacher,
        "cross_route_mode": cross_route_mode if teacher == "nn_split" else None,
        "match_scale": match_scale,
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

    # Dedup identical configs: rerun replaces, never appends a duplicate row.
    result_dir = Path(output)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"ablation_N{num_loc}.json"
    results = json.loads(result_file.read_text()
                         ) if result_file.exists() else []
    dedup_key = {k: result[k] for k in (
        "backbone", "metric_variant", "num_slots", "num_loc", "dist", "seed",
        "ins_method", "teacher", "cross_route_mode", "match_scale",
        "normalize_target", "symmetrize_target",
    )}
    if problem != "cvrp":
        dedup_key["problem"] = problem
    results = [r for r in results if not all(
        r.get(k) == v for k, v in dedup_key.items())]
    results.append(result)
    result_file.write_text(json.dumps(results, indent=2))

    # Also append a flat row to a teacher x setting CSV for the ablation table.
    try:
        csv_path = result_dir / "teacher_ablation.csv"
        cols = ["teacher", "cross_route_mode", "match_scale", "num_loc", "dist",
                "problem", "seed", "best_val_reward", "elapsed_min",
                "checkpoint", "train_path", "val_path"]
        import csv as _csv
        exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(result)
    except Exception as _e:  # never crash training on logging failure
        print(f"[WARN] could not write teacher_ablation.csv: {_e}")

    print(f"\nDone. Best val reward: {best_reward} | {elapsed/60:.1f} min")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train routing models")
    parser.add_argument("--num_loc", type=int, default=100,
                        choices=[50, 100, 200, 500, 1000])
    parser.add_argument("--dist", default="uniform",
                        choices=["uniform", "clustered"])
    parser.add_argument("--problem", default="cvrp",
                        choices=["cvrp", "ovrp", "vrptw"],
                        help="Problem to train/eval. ovrp and vrptw reuse the "
                             "CVRP embedding/objective (constraint-agnostic); "
                             "--problem selects the cached dataset prefix and the "
                             "environment. vrptw attaches Solomon time windows to "
                             "the data for feasibility reporting only.")
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
                        choices=sorted(LEGO_BACKBONES | BASELINE_BACKBONES),
                        help="LEGO slot backbones: " + ", ".join(sorted(LEGO_BACKBONES)) +
                             ". Routing baselines (" +
                             ", ".join(sorted(BASELINE_BACKBONES)) +
                             ") train via the baselines CLI (dispatched here).")
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
    parser.add_argument("--teacher", default="nn",
                        choices=["nn", "random", "sweep", "nn_2opt", "nn_split", "hgs"],
                        help="Reference-tour teacher for the d_cons geometry of "
                             "Variant D: nn (default) | random | sweep | nn_2opt | "
                             "nn_split | hgs (not enabled in this build). Non-nn "
                             "teachers read/write cached splits under teacher=<T>/.")
    parser.add_argument("--cross_route_mode", default="hard",
                        choices=["hard", "soft"],
                        help="nn_split cross-route handling: 'hard' pins cross-route "
                             "pairs to L/2; 'soft' adds L/4 (clipped at L/2). "
                             "Ignored unless --teacher nn_split.")
    parser.add_argument("--match_scale", action="store_true",
                        help="Rescale the random teacher's d_cons so its mean equals "
                             "the nn teacher's mean (control enabling a matched-scale "
                             "comparison; off by default).")
    parser.add_argument("--lower_neighbors_num", type=int, default=50)
    parser.add_argument("--reduction_percentage", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--generate_missing_data", action="store_true")
    parser.add_argument("--generation_chunk_size", type=int, default=None)
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
    # LEHD/TTPL baselines need their external data path (ignored for other backbones).
    parser.add_argument("--lehd_data_path", default=None)
    parser.add_argument("--lehd_val_data_path", default=None)
    args = parser.parse_args()

    if args.backbone in BASELINE_BACKBONES:
        # Routing baselines train in baselines/train_baseline.py; dispatch there so
        # they remain CLI-runnable through train.py without living in this file.
        from baselines.train_baseline import run as run_baseline
        from baselines.train_baseline import load_baseline_config
        config = load_baseline_config(args.backbone, None)
        args_dict = vars(args)
        args_dict.pop("config", None)  # --config is a path; run() wants the loaded dict.
        run_baseline(config=config, **args_dict)
        return

    train(
        num_loc=args.num_loc, dist=args.dist, problem=args.problem,
        data_dir=args.data_dir,
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
        teacher=args.teacher, match_scale=args.match_scale,
        cross_route_mode=args.cross_route_mode,
        lower_neighbors_num=args.lower_neighbors_num,
        reduction_percentage=args.reduction_percentage, logger=args.logger,
        resume=args.resume, num_workers=args.num_workers,
        generate_missing_data=args.generate_missing_data,
        generation_chunk_size=args.generation_chunk_size,
    )


if __name__ == "__main__":
    main()
