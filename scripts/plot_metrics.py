"""
Plot & compare training curves from multiple Lightning CSVLogger metrics.csv.

Reads several metrics.csv files (one per run) and overlays them — one line per
run — split across two panels: (left) val/reward, (right) train loss components.
This is the tool for the D-vs-B / K-sweep comparison: drop one CSV per run and
read which val/reward curve wins.

Usage:
    # single run
    python scripts/plot_metrics.py logs/am_slot_D_K8_.../metrics/metrics.csv

    # compare many runs (pass each run's dir or its metrics.csv)
    python scripts/plot_metrics.py \
        logs/am_slot_D_K8_N100_uniform_seed42 \
        logs/am_slot_B_K8_N100_uniform_seed42 \
        --out compare_D_vs_B.png

Metrics of interest:
    val/reward          - greedy validation reward (higher = better tour)
    train/reward        - training reward
    train/loss          - total loss (policy + aux)
    train/policy_loss   - REINFORCE loss component
    train/aux_loss      - metric/entropy aux loss component
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Panels on the right: which loss components to plot per run
LOSS_COLS = ["train/loss", "train/policy_loss", "train/aux_loss"]


def resolve_csv(path: Path) -> Path:
    """Accept a metrics.csv, a metrics/ dir, or a run dir; return the csv path."""
    p = Path(path)
    if p.is_dir():
        cand = p / "metrics.csv"
        if not cand.exists():
            cand = p / "metrics" / "metrics.csv"
        p = cand
    return p


def run_label(csv_path: Path) -> str:
    """Derive a readable run name from the path, e.g. am_slot_D_K8_..."""
    p = csv_path.parent
    if p.name == "metrics":
        p = p.parent
    return p.name


def _epoch_series(df: pd.DataFrame, key: str) -> pd.Series:
    """Per-epoch mean of a metric, preferring the _step column for density."""
    src = df[key] if key in df.columns else df.get(f"{key}_step")
    if src is None:
        return pd.Series(dtype=float)
    if "epoch" in df.columns:
        return pd.DataFrame({"epoch": df["epoch"], "v": src}).groupby("epoch")["v"].mean().dropna()
    return src.dropna()


def load_run(csv_path: Path):
    df = pd.read_csv(csv_path)
    val = _epoch_series(df, "val/reward")
    losses = {c: _epoch_series(df, c) for c in LOSS_COLS if c in df.columns or f"{c}_step" in df.columns}
    return val, losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot & compare training curves from Lightning CSVs")
    parser.add_argument("csvs", type=str, nargs="+", help="One or more metrics.csv / run dirs to compare")
    parser.add_argument("--out", type=str, default=None, help="Output image path (default: show)")
    parser.add_argument("--max", type=int, default=None, help="Cap epochs shown (optional)")
    parser.add_argument("--log_wandb", type=str, default=None,
                        help="If set, log the figure to a wandb run (pass the run_id / project:run or 'new'). Requires wandb login.")
    args = parser.parse_args()

    # ── Load all runs ────────────────────────────────────────────────────
    runs = []
    for raw in args.csvs:
        csv_path = resolve_csv(Path(raw))
        if not csv_path.exists():
            print(f"[skip] no metrics.csv at {csv_path}")
            continue
        label = run_label(csv_path)
        val, losses = load_run(csv_path)
        if args.max is not None:
            val = val[val.index <= args.max]
            losses = {k: s[s.index <= args.max] for k, s in losses.items()}
        runs.append((label, val, losses))
        print(f"[load] {label}: {len(val)} val points, losses={list(losses)}")

    if not runs:
        print("Nothing to plot.")
        return

    # ── Figure: left = val/reward, right = loss components ───────────────
    fig, (ax_val, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("tab10")

    # Left: val/reward, one line per run
    ax_val.set_title("val/reward (higher = better)")
    ax_val.set_xlabel("epoch")
    for i, (label, val, _) in enumerate(runs):
        c = cmap(i % 10)
        ax_val.plot(val.index, val.values, marker="o", ms=2.5, lw=1.4, color=c, label=label)
        if len(val):
            ax_val.axhline(val.max(), ls="--", lw=0.8, color=c, alpha=0.6)
    ax_val.legend(fontsize=7, loc="best")
    ax_val.grid(alpha=0.3)

    # Right: loss components, one line per run+component
    ax_loss.set_title("train loss")
    ax_loss.set_xlabel("epoch")
    for i, (label, _, losses) in enumerate(runs):
        c = cmap(i % 10)
        for comp, s in losses.items():
            if len(s):
                short = comp.split("/")[-1]
                ax_loss.plot(s.index, s.values, ls="--", lw=1.2, color=c, alpha=0.85,
                             label=f"{label} · {short}")
    ax_loss.legend(fontsize=6, loc="best")
    ax_loss.grid(alpha=0.3)

    fig.suptitle("Run comparison", fontsize=13)
    fig.tight_layout()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"Saved comparison to {args.out}")
    else:
        plt.show()

    # ── Optional: push the figure to a wandb run ────────────────────────
    if args.log_wandb:
        import wandb
        wandb.init(project="MeTRA_Slot_NCO", id=args.log_wandb if args.log_wandb != "new" else None,
                   resume="allow")
        wandb.log({"run_comparison": fig})
        wandb.finish()
        print("Logged comparison figure to wandb.")


if __name__ == "__main__":
    main()
