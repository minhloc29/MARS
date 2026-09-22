from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from baselines.utils import (
    CVRPInstance,
    parse_lkh_tour_file,
    routes_length,
    to_vrplib_string,
    validate_routes,
)


def check_lkh_available(lkh_executable: str | Path | None = None) -> Path:
    """Verify that LKH executable exists and is runnable."""
    import sys

    is_windows = sys.platform == "win32"

    if lkh_executable:
        p = Path(lkh_executable).resolve()
        if p.exists() and p.is_file():
            if not is_windows:
                try:
                    os.chmod(p, p.stat().st_mode | 0o111)
                except OSError:
                    pass
            return p
        which = shutil.which(str(lkh_executable))
        if which:
            return Path(which).resolve()
        raise FileNotFoundError(
            f"Specified LKH executable not found: {lkh_executable}"
        )

    default_dir = Path(__file__).resolve().parent

    # On Windows prefer .exe; on Linux/macOS search for native ELF binary (never .exe)
    if is_windows:
        candidates = ("LKH.exe", "LKH-3.exe", "LKH", "lkh3", "lkh")
    else:
        candidates = ("LKH", "LKH-3", "lkh3", "lkh")

    for name in candidates:
        cand = default_dir / name
        if cand.exists() and cand.is_file():
            if not is_windows:
                try:
                    os.chmod(cand, cand.stat().st_mode | 0o111)
                except OSError:
                    pass
            return cand

    # Check system PATH
    for name in candidates:
        which = shutil.which(name)
        if which:
            return Path(which).resolve()

    if not is_windows:
        raise FileNotFoundError(
            "LKH native binary not found for Linux/macOS. "
            "Please compile LKH on your server by running:\n"
            "  cd baselines/lkh && "
            "curl -O http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.14.tgz && "
            "tar -xzf LKH-3.0.14.tgz && cd LKH-3.0.14 && make && cp LKH .. && cd .. && rm -rf LKH-3.0.14*\n"
            "or pass its path explicitly via --lkh_executable."
        )

    raise FileNotFoundError(
        "LKH executable not found. Please place 'LKH.exe' in 'baselines/lkh/' "
        "or pass its path explicitly via --lkh_executable."
    )


def solve_lkh(
    instance: CVRPInstance,
    lkh_executable: str | Path | None = None,
    work_dir: str | Path | None = None,
    time_limit: float | None = 30.0,
    runs: int = 1,
    max_trials: int = 10000,
    seed: int = 1,
    keep_files: bool = False,
) -> dict[str, Any]:
    """Solve a CVRPInstance using LKH-3.

    Args:
        instance: CVRPInstance with coordinates and demand.
        lkh_executable: Path to LKH binary. If None, resolves default.
        work_dir: Directory to place temporary .vrp, .par, .tour files.
                  Must be unique per parallel process to avoid collision.
        time_limit: Maximum solve time in seconds.
        runs: Number of independent LKH runs (default: 1).
        max_trials: Maximum trials per run.
        seed: Random seed for LKH.
        keep_files: If True, do not delete temporary files after solving.

    Returns:
        dict with routes, cost, elapsed_seconds, feasible.
    """
    exe_path = check_lkh_available(lkh_executable)

    if work_dir is None:
        work_dir = Path("scratch/lkh_work")
    work_path = Path(work_dir).resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex[:12]
    problem_name = f"cvrp_n{instance.num_loc}_{uid}"
    problem_file = work_path / f"{problem_name}.vrp"
    tour_file = work_path / f"{problem_name}.tour"
    par_file = work_path / f"{problem_name}.par"

    # 1. Write VRPLIB problem file
    vrp_content = to_vrplib_string(instance, name=problem_name)
    problem_file.write_text(vrp_content, encoding="utf-8")

    # 2. Write parameter file
    par_lines = [
        f"PROBLEM_FILE = {problem_file}",
        f"OUTPUT_TOUR_FILE = {tour_file}",
        f"RUNS = {runs}",
        f"MAX_TRIALS = {max_trials}",
        f"SEED = {seed}",
        "TRACE_LEVEL = 1",
    ]
    if time_limit is not None and time_limit > 0:
        par_lines.append(f"TIME_LIMIT = {int(round(time_limit))}")

    par_file.write_text("\n".join(par_lines) + "\n", encoding="utf-8")

    # 3. Execute LKH binary
    start_time = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(exe_path), str(par_file)],
            cwd=str(work_path),
            capture_output=True,
            text=True,
            timeout=(time_limit * 2 + 10) if time_limit else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"LKH process timed out after {exc.timeout} seconds."
        ) from exc
    elapsed = time.perf_counter() - start_time

    if proc.returncode != 0 or not tour_file.exists():
        raise RuntimeError(
            f"LKH failed with returncode {proc.returncode} (tour exists: {tour_file.exists()})\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    # 4. Parse output tour and validate
    routes = parse_lkh_tour_file(tour_file, instance.num_loc)

    validate_routes(
        routes,
        num_loc=instance.num_loc,
        demand_int=instance.demand_int,
        capacity_int=instance.capacity_int,
    )

    # 5. Recompute float Euclidean cost
    cost = routes_length(routes, instance.coords_float)

    # 6. Cleanup temporary files
    if not keep_files:
        for f in (par_file, problem_file, tour_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

    return {
        "routes": routes,
        "cost": cost,
        "elapsed_seconds": elapsed,
        "feasible": True,
    }
