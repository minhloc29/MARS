#!/usr/bin/env python
"""Download the CVRPLIB Set X benchmark (Uchoa et al., 2017) and cache it locally.

The official CVRPLIB site no longer serves files from the old
``vrp.galgos.inf.puc-rio.br/media/...`` paths. It now lives at
``https://galgos.inf.puc-rio.br/cvrplib/`` and exposes three download
endpoints (any of which returns the raw content):

    * whole instance set  : /cvrplib/en/download/instance-set/<set_id>   (a 7z archive)
    * single instance     : /cvrplib/en/download/instance/<inst_id>      (plain-text .vrp)
    * best-known solution : /cvrplib/en/download/bks/<inst_id>           (plain-text .sol)

For Set X (Uchoa et al. 2017) the set id is 17; the per-instance ids and the
best-known-solution ids are identical to the enumeration ids used by the
site's table (each instance row links ``download/instance/<id>`` and
``download/bks/<id>`` with the same <id>).

The archive is 7-zip compressed, so extraction needs ``py7zr`` (pure-Python,
installed in this project's venv). After extraction every ``.vrp``/``.sol`` is
parsed with ``vrplib`` and re-cached as one ``.pt`` tensor dump so downstream
evaluation never needs the network or the raw server again.

Run (proxy must be unset or the server is unreachable):
    unset HTTPS_PROXY http_proxy HTTP_PROXY https_proxy ALL_PROXY all_proxy
    python scripts/download_cvrplib_setX.py --out_dir ./data/cvrplib_setX
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import requests

try:
    import py7zr
    HAVE_PY7ZR = True
except Exception:
    HAVE_PY7ZR = False

try:
    import vrplib
    HAVE_VRPLIB = True
except Exception:
    HAVE_VRPLIB = False

BASE = "https://galgos.inf.puc-rio.br/cvrplib"
SET_X_ID = 17          # CVRP Set X (Uchoa et al., 2017)
BULK_URL = f"{BASE}/en/download/instance-set/{SET_X_ID}"
INSTANCE_URL = f"{BASE}/en/download/instance/{{id}}"
BKS_URL = f"{BASE}/en/download/bks/{{id}}"


def fetch(url: str, out: Path, timeout: int = 90, retries: int = 5) -> bool:
    """Download ``url`` into ``out`` with backoff; True on success.

    The CVRPLIB server is intermittently flaky (redirect loops, dropped
    connections, timeouts), so we retry a handful of times.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    import time

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code != 200:
                print(f"    [retry {attempt}/{retries}] {url} -> HTTP {r.status_code}")
                time.sleep(2 * attempt)
                continue
            out.write_bytes(r.content)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"    [retry {attempt}/{retries}] {url} -> {e!r}")
            time.sleep(2 * attempt)
    return False


def extract_7z(archive: Path, dest: Path) -> None:
    if not HAVE_PY7ZR:
        raise RuntimeError(
            "Need py7zr to extract the set archive. "
            "Install with: uv pip install --python .venv/bin/python py7zr"
        )
    with py7zr.SevenZipFile(str(archive), mode="r") as z:
        z.extractall(path=str(dest))


def parse_set_x(raw_dir: Path) -> Dict[str, dict]:
    """Parse every .vrp and .sol in ``raw_dir`` with vrplib -> {name: {coords, demand,
    capacity, depot, best_cost}}."""
    if not HAVE_VRPLIB:
        raise RuntimeError("Need vrplib to parse instances. Install: uv pip install --python .venv/bin/python vrplib")

    out: Dict[str, dict] = {}
    vrps = sorted(raw_dir.rglob("*.vrp"))
    sols = {p.stem: p for p in raw_dir.rglob("*.sol")}
    for v in vrps:
        inst = vrplib.read_instance(v)
        name = inst["name"].strip()
        depot = inst["depot"]
        # CVRPLIB depots are returned as an array (e.g. [0]); unwrap to a scalar.
        depot = int(depot[0] if hasattr(depot, "__len__") else depot)
        rec = {
            "name": name,
            "node_coord": inst["node_coord"],            # (n, 2)
            "demand": inst["demand"],                    # (n,)
            "capacity": float(inst["capacity"]),
            "depot": depot,
            "best_cost": None,
            "best_routes": None,
        }
        s = sols.get(name) or sols.get(v.stem)
        if s is not None:
            try:
                sol = vrplib.read_solution(s)
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] failed to parse solution {s.name}: {e!r}")
                sol = None
            if sol is not None:
                rec["best_cost"] = float(sol.get("cost"))
                rec["best_routes"] = sol.get("routes")
        out[name] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + cache CVRPLIB Set X")
    ap.add_argument("--out_dir", type=str, default="./data/cvrplib_setX")
    ap.add_argument("--bulk", action="store_true", default=True,
                    help="Download the whole Set X 7z archive (default).")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Skip network; only parse already-downloaded .vrp/.sol.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw = out_dir / "raw"
    archive = out_dir / f"setX_id{SET_X_ID}.7z"

    fetched = False
    if args.bulk and not archive.exists() and not args.no_fetch:
        print(f"[1/3] Downloading Set X archive -> {archive}")
        fetched = fetch(BULK_URL, archive)
        if not fetched:
            raise SystemExit("Failed to download Set X archive from the official server.")
    if archive.exists():
        print(f"[2/3] Extracting {archive}")
        extract_7z(archive, raw)
        fetched = True

    if args.no_fetch or (fetched and (raw / "X").exists()):
        pass

    print("[3/3] Parsing with vrplib...")
    data = parse_set_x(raw)
    if not data:
        raise RuntimeError("No .vrp files parsed — check the download/extract step.")

    print(f"Parsed {len(data)} CVRPLIB Set X instances (n ranges "
          f"{min(v['node_coord'].shape[0] for v in data.values())}.."
          f"{max(v['node_coord'].shape[0] for v in data.values())}).")

    # Re-cache as a single .pt (network-free for downstream eval).
    cache = out_dir / "setX.pt"
    import torch
    torch.save(data, cache)
    print(f"Saved cache -> {cache}")

    # Quick summary table.
    sizes = {}
    for name, v in data.items():
        n = v["node_coord"].shape[0]
        sizes.setdefault(n, []).append(name)
    print("Instances per size:")
    for n in sorted(sizes):
        print(f"  n={n:4d}: {len(sizes[n])} instances  e.g. {sizes[n][0]}, {sizes[n][-1]}")

    # Show a couple of entries to demonstrate the parse succeeded.
    sample = next(iter(data.values()))
    print("\nSample parsed record keys:", list(sample.keys()))
    print("  node_coord", sample["node_coord"].shape,
          "demand", sample["demand"].shape,
          "capacity", sample["capacity"],
          "depot", sample["depot"],
          "best_cost", sample["best_cost"])


if __name__ == "__main__":
    main()
