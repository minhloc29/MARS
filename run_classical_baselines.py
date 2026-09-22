from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from baselines.hgs import solve_hgs
from baselines.lkh import check_lkh_available, solve_lkh
from baselines.utils import CVRPInstance, iter_instances, load_instance_batch


def _solve_worker(args: tuple[int, CVRPInstance, str, dict[str, Any]]) -> dict[str, Any]:
    """Worker function for ProcessPoolExecutor."""
    idx, instance, solver, kwargs = args
    try:
        if solver == "hgs":
            res = solve_hgs(
                instance=instance,
                time_limit=kwargs.get("time_limit", 10.0),
                max_iterations=kwargs.get("max_iterations"),
                seed=kwargs.get("seed", 1) + idx,
            )
        elif solver == "lkh":
            # Each worker process gets its own subfolder inside work_dir based on pid
            worker_work_dir = Path(kwargs["work_dir"]) / f"proc_{os.getpid()}"
            res = solve_lkh(
                instance=instance,
                lkh_executable=kwargs.get("lkh_executable"),
                work_dir=worker_work_dir,
                time_limit=kwargs.get("time_limit", 30.0),
                runs=kwargs.get("runs", 1),
                max_trials=kwargs.get("max_trials", 10000),
                seed=kwargs.get("seed", 1) + idx,
                keep_files=kwargs.get("keep_files", False),
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")

        res["instance_idx"] = idx
        return res
    except Exception as exc:
        print(f"\n[WARN] Solver {solver.upper()} failed on instance {idx}: {exc}")
        return {
            "instance_idx": idx,
            "routes": [],
            "cost": float("nan"),
            "elapsed_seconds": 0.0,
            "feasible": False,
            "error": str(exc),
        }


def parse_time_limits(time_limits_str: list[str] | None, default_tl: float) -> dict[int, float]:
    """Parse '50=5 100=10' format into {50: 5.0, 100: 10.0}."""
    result: dict[int, float] = {}
    if not time_limits_str:
        return result
    for item in time_limits_str:
        if "=" in item:
            k, v = item.split("=", 1)
            result[int(k.strip())] = float(v.strip())
        else:
            default_tl = float(item.strip())
    return result


def find_dataset_file(
    data_root: Path,
    num_loc: int,
    dist: str,
    data_format: str,
    ins_method: str = "insertion",
) -> Path:
    """Find dataset file corresponding to (num_loc, dist), checking aliases."""
    dist_aliases = [dist]
    if dist == "gaussian":
        dist_aliases.append("clustered")
    elif dist == "clustered":
        dist_aliases.append("gaussian")

    candidates: list[Path] = []

    # 1. Check .pt format under data/slot_datasets_v2/{method}/ or data_root
    if data_format in ("pt", "auto"):
        for d in dist_aliases:
            # Check slot_datasets_v2 path
            candidates.extend(data_root.glob(f"{ins_method}/cvrp{num_loc}_{d}_test.pt"))
            candidates.extend(data_root.glob(f"**/cvrp{num_loc}_{d}_test.pt"))
            candidates.extend(data_root.glob(f"**/cvrp{num_loc}_{d}_*.pt"))

    # 2. Check .npz format under data/test/ or data_root
    if data_format in ("npz", "auto"):
        for d in dist_aliases:
            candidates.extend(data_root.glob(f"**/cvrp_{num_loc}_{d}_*.npz"))
            candidates.extend(data_root.glob(f"**/cvrp_{num_loc}_{d}.npz"))
            candidates.extend(data_root.glob(f"cvrp_{num_loc}_{d}_*.npz"))

    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        f"Could not find dataset file for N={num_loc}, dist={dist} "
        f"(format={data_format}) under {data_root.resolve()}.\n"
        f"Searched patterns for: {dist_aliases}"
    )


def run_benchmark(
    solver: str,
    sizes: list[int],
    dists: list[str],
    data_root: str | Path = "./data",
    data_format: str = "auto",
    ins_method: str = "insertion",
    n_instances: int | None = None,
    n_workers: int = 4,
    output_dir: str | Path = "./results",
    seed: int = 1234,
    time_limits_map: dict[int, float] | None = None,
    default_time_limit: float = 10.0,
    save_routes: bool = False,
    resume: bool = False,
    lkh_executable: str | Path | None = None,
    lkh_runs: int = 1,
    lkh_max_trials: int = 10000,
) -> None:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if solver == "lkh":
        exe_path = check_lkh_available(lkh_executable)
        print(f"[LKH] Using executable: {exe_path}")

    time_limits = time_limits_map or {}

    for size in sizes:
        for dist in dists:
            out_file = output_dir / f"eval_{solver}_{size}_{dist}.json"
            if resume and out_file.exists():
                print(f"Skipping N={size} dist={dist} (result already exists at {out_file})")
                continue

            try:
                data_path = find_dataset_file(data_root, size, dist, data_format, ins_method)
            except FileNotFoundError as err:
                print(f"[WARN] {err} — skipping this combination.")
                continue

            print(f"\n========================================================")
            print(f"Running {solver.upper()} on N={size} {dist} from {data_path}")
            print(f"========================================================")

            batch = load_instance_batch(data_path)
            instances = list(iter_instances(batch, num_loc=size, n_instances=n_instances))
            n_inst = len(instances)
            print(f"Loaded {n_inst} instances.")

            time_limit = time_limits.get(size, default_time_limit)
            print(f"Per-instance time budget: {time_limit}s | Workers: {n_workers}")

            base_work_dir = Path("scratch/lkh_work") / f"{size}_{dist}"

            worker_kwargs: dict[str, Any] = {
                "time_limit": time_limit,
                "seed": seed,
            }
            if solver == "lkh":
                worker_kwargs["lkh_executable"] = str(lkh_executable) if lkh_executable else None
                worker_kwargs["work_dir"] = str(base_work_dir)
                worker_kwargs["runs"] = lkh_runs
                worker_kwargs["max_trials"] = lkh_max_trials

            tasks = [
                (i, inst, solver, worker_kwargs)
                for i, inst in enumerate(instances)
            ]

            start_all = time.perf_counter()
            results: list[dict[str, Any]] = []

            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    for res in tqdm(
                        executor.map(_solve_worker, tasks),
                        total=n_inst,
                        desc=f"{solver.upper()} N={size} {dist}",
                    ):
                        results.append(res)
            else:
                for t in tqdm(tasks, desc=f"{solver.upper()} N={size} {dist}"):
                    results.append(_solve_worker(t))

            total_elapsed = time.perf_counter() - start_all

            # Sort results by instance_idx
            results.sort(key=lambda r: r["instance_idx"])

            feasible_results = [
                r for r in results if r.get("feasible", False) and not np.isnan(r.get("cost", float("nan")))
            ]
            costs = [r["cost"] for r in feasible_results]
            mean_cost = float(np.mean(costs)) if costs else float("nan")
            std_cost = float(np.std(costs)) if costs else float("nan")
            time_per_instance = float(total_elapsed / n_inst) if n_inst > 0 else 0.0

            output_data: dict[str, Any] = {
                "solver": solver,
                "num_loc": size,
                "dist": dist,
                "n_inst": n_inst,
                "n_feasible": len(feasible_results),
                "n_failed": n_inst - len(feasible_results),
                "mean_tour_length": mean_cost,
                "std_tour_length": std_cost,
                "elapsed_seconds": total_elapsed,
                "time_per_instance": time_per_instance,
                "time_limit_per_instance": time_limit,
                "data_path": str(data_path.resolve()),
            }

            if save_routes:
                output_data["routes"] = [r["routes"] for r in results]

            out_file.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
            status_str = f"({len(feasible_results)}/{n_inst} feasible)" if len(feasible_results) < n_inst else ""
            print(
                f"[OK] Completed N={size} {dist} {status_str}: "
                f"mean_tour = {mean_cost:.4f} (std = {std_cost:.4f}) | "
                f"total time = {total_elapsed:.2f}s ({time_per_instance:.3f}s/inst) -> Saved to {out_file}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run classical baseline solvers (HGS / LKH-3) on CVRP benchmark instances"
    )
    parser.add_argument(
        "--solver",
        type=str,
        required=True,
        choices=["hgs", "lkh"],
        help="Classical solver to evaluate",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000],
        help="Target number of customers N",
    )
    parser.add_argument(
        "--dists",
        type=str,
        nargs="+",
        default=["uniform", "gaussian"],
        help="Instance distributions",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root folder containing dataset splits",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "pt", "npz"],
        help="Dataset format to prioritize",
    )
    parser.add_argument(
        "--ins_method",
        type=str,
        default="insertion",
        help="Method subfolder under data/slot_datasets_v2",
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=None,
        help="Maximum instances to evaluate per split",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Folder to save result JSONs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed",
    )
    parser.add_argument(
        "--time_limits",
        type=str,
        nargs="*",
        default=None,
        help="Per-size time limits in seconds (e.g. 50=5 100=10 200=20 500=60 1000=120)",
    )
    parser.add_argument(
        "--default_time_limit",
        type=float,
        default=10.0,
        help="Default solve time limit in seconds per instance",
    )
    parser.add_argument(
        "--save_routes",
        action="store_true",
        help="Whether to save the route arrays in output JSONs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose output JSON already exists",
    )
    parser.add_argument(
        "--lkh_executable",
        type=str,
        default=None,
        help="Explicit path to LKH binary (default: baselines/lkh/LKH.exe)",
    )
    parser.add_argument(
        "--lkh_runs",
        type=int,
        default=1,
        help="Number of LKH runs per instance",
    )
    parser.add_argument(
        "--lkh_max_trials",
        type=int,
        default=10000,
        help="Maximum trials for LKH",
    )

    args = parser.parse_args()

    time_limits_map = parse_time_limits(args.time_limits, args.default_time_limit)

    run_benchmark(
        solver=args.solver,
        sizes=args.sizes,
        dists=args.dists,
        data_root=args.data_root,
        data_format=args.data_format,
        ins_method=args.ins_method,
        n_instances=args.n_instances,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        seed=args.seed,
        time_limits_map=time_limits_map,
        default_time_limit=args.default_time_limit,
        save_routes=args.save_routes,
        resume=args.resume,
        lkh_executable=args.lkh_executable,
        lkh_runs=args.lkh_runs,
        lkh_max_trials=args.lkh_max_trials,
    )


if __name__ == "__main__":
    main()
