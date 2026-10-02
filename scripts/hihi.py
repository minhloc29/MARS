import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from tensordict import TensorDict

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot
from rl4co.models.nn.metric_loss import _aggregate_d_ins_sparse
from rl4co.data.slot_dataset import SlotDataset
from matplotlib.lines import Line2D


# ============================================================
# Shared publication style
# ============================================================

PALETTE = {
    "without": "#C44E52",
    "with": "#4C72B0",
    "diag": "#333333",
    "edge": "#1a1a1a",
}

SLOT_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#8172B2",
    "#C44E52", "#64B5CD", "#CCB974", "#937860",
    "#8C8C8C", "#B07AA1",
]


def set_pub_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# Geometry utilities
# ============================================================

def off_diagonal(D):
    """Extract upper-triangular off-diagonal entries.
    D: (..., K, K) -> (..., K*(K-1)/2)
    """
    K = D.shape[-1]
    mask = torch.triu(
        torch.ones(K, K, dtype=torch.bool, device=D.device),
        diagonal=1,
    )
    return D[..., mask]


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pooled Spearman rank correlation between two flat 1D tensors."""
    xr = torch.argsort(torch.argsort(x)).float()
    yr = torch.argsort(torch.argsort(y)).float()

    xr = xr - xr.mean()
    yr = yr - yr.mean()

    denom = xr.norm() * yr.norm()
    return float((xr * yr).sum() / (denom + 1e-8))


def load_model_and_slots(ckpt, env, locs, depot, demand):
    """Load a checkpoint and extract slots (B,K,D) and A_ik (B,N,K)."""

    print("\n" + "=" * 70)
    print("Loading checkpoint")
    print("=" * 70)
    print(ckpt)

    model = POMOSlot.load_from_checkpoint(
        ckpt,
        env=env,
        map_location="cpu",
    )
    model.eval()

    encoder = model.policy.encoder

    td = TensorDict(
        {
            "locs": locs,
            "depot": depot,
            "demand": demand,
            "capacity": torch.ones(locs.shape[0], 1),
        },
        batch_size=[locs.shape[0]],
    )

    with torch.no_grad():
        td = env.reset(td)
        encoder(td)
        slots = encoder.last_slots
        A_ik = encoder.last_A_ik

    print(f"slots shape : {tuple(slots.shape)}")
    print(f"A_ik shape  : {tuple(A_ik.shape)}")

    return model, slots.cpu(), A_ik.cpu()


def find_projection_head(model):
    """Find ProjectionHead in a checkpoint. Returns None if absent."""

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "ProjectionHead"
    ]

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print("\nWARNING: multiple ProjectionHeads found:")
        for name, _ in candidates:
            print(f"  {name}")

    name, proj_head = candidates[0]
    print(f"\nFound projection head at: model.{name}")
    proj_head.eval()

    return proj_head


def compute_latent_geometry(slots, proj_head=None):
    """
    Compute pairwise Euclidean distances in each model's own native space.

    For the B/no-geometry checkpoint, which has no projection head,
    distances are computed directly on raw slot representations.

    For the D/LEGO checkpoint, distances are computed after its learned
    projection head, matching the metric-loss space.
    """

    with torch.no_grad():
        if proj_head is not None:
            slots_dev = slots.to(next(proj_head.parameters()).device)
            z = proj_head(slots_dev)
        else:
            z = slots

        D_latent = torch.cdist(z, z, p=2)

    return z.cpu(), D_latent.cpu()


def compute_target_geometry(
    d_idx,
    d_val,
    A_ik,
    normalize,
    symmetrize,
):
    """Compute construction-induced slot geometry from a model's own A_ik."""

    with torch.no_grad():
        result = _aggregate_d_ins_sparse(
            d_idx,
            d_val,
            A_ik,
            normalize=normalize,
            symmetrize=symmetrize,
        )

    D_target = result[0] if isinstance(result, tuple) else result
    return D_target.cpu()


def violation_statistics(D_target, D_latent):
    """Compute upper-bound violation, ratio, and rank-correlation statistics."""

    target = off_diagonal(D_target)
    latent = off_diagonal(D_latent)

    violation = latent > target
    violation_rate = violation.float().mean().item() * 100

    excess = torch.clamp(latent - target, min=0.0)
    mean_excess = excess.mean().item()
    max_excess = excess.max().item()

    ratio = latent / target.clamp_min(1e-6)
    ratio_np = ratio.numpy()

    percentiles = {
        "median": float(np.median(ratio_np)),
        "mean": float(ratio_np.mean()),
        "p90": float(np.percentile(ratio_np, 90)),
        "p95": float(np.percentile(ratio_np, 95)),
        "p99": float(np.percentile(ratio_np, 99)),
    }

    rho = spearman(latent, target)

    return {
        "target": target,
        "latent": latent,
        "violation_rate": violation_rate,
        "mean_excess": mean_excess,
        "max_excess": max_excess,
        "ratio_percentiles": percentiles,
        "spearman_rho": rho,
    }


def print_stats_block(label, stats):
    print(f"\n{label}:")
    print(f"  violation rate = {stats['violation_rate']:.2f}%")
    print(f"  mean excess    = {stats['mean_excess']:.6f}")
    print(f"  max excess     = {stats['max_excess']:.6f}")
    print(f"  Spearman rho   = {stats['spearman_rho']:.4f}")

    p = stats["ratio_percentiles"]
    print(
        f"  D_latent/D_target ratio: "
        f"median={p['median']:.3f}  "
        f"mean={p['mean']:.3f}  "
        f"p90={p['p90']:.3f}  "
        f"p95={p['p95']:.3f}  "
        f"p99={p['p99']:.3f}"
    )

def make_combined_figure(
    x_before,
    y_before,
    x_after,
    y_after,
    out_stem: str,
    x_max: float = 150,
    max_points: int = 20_000,
):
    """Two scatter panels (without / with) in one image, shared legend below."""

    set_pub_style()
    rng = np.random.default_rng(0)

    def _prep(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) > max_points:
            idx = rng.choice(len(x), size=max_points, replace=False)
            x, y = x[idx], y[idx]
        return x, y

    xb, yb = _prep(x_before, y_before)
    xa, ya = _prep(x_after, y_after)

    y_lim = float(max(yb.max(), ya.max())) * 1.05
    diag_end = max(x_max, y_lim)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.8, 3.1),
        sharex=True,
        sharey=False,   # was True; this was hiding the right panel's y numbers
    )

    panels = [
        (axes[0], xb, yb, PALETTE["without"], "Without regularization"),
        (axes[1], xa, ya, PALETTE["with"], "With regularization"),
    ]

    for ax, x, y, color, title in panels:
        ax.scatter(
            x, y,
            s=5,
            alpha=0.35,
            color=color,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        ax.plot(
            [0, diag_end],
            [0, diag_end],
            color=PALETTE["diag"],
            lw=1.1,
            ls="--",
            zorder=1,
        )

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_lim)
        ax.set_xlabel("Target distance")
        ax.set_title(title)

        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        ax.grid(True, ls=":", lw=0.5, alpha=0.3, color="#cccccc")
        ax.tick_params(length=3)

    axes[0].set_ylabel("Latent Distance")

    # One shared legend, each entry only once
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor=PALETTE["without"], markeredgecolor="none",
               alpha=0.8, label="Without regularization"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor=PALETTE["with"], markeredgecolor="none",
               alpha=0.8, label="With regularization"),
        Line2D([0], [0], color=PALETTE["diag"], lw=1.1, ls="--",
               label=r"$D_{\mathrm{latent}}=D_{\mathrm{target}}$"),
    ]

    fig.tight_layout(pad=0.4, w_pad=1.2)

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        borderpad=0.5,
        handletextpad=0.5,
        columnspacing=1.8,
    )
    leg.get_frame().set_linewidth(0.6)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = out.with_suffix(".pdf")
    png_path = out.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Success] Saved:\n  - {pdf_path.resolve()}\n  - {png_path.resolve()}")
# ============================================================
# Figure 1: D_target vs D_latent scatter
# ============================================================

def make_scatter_figure(
    x,
    y,
    color,
    label,
    out_stem: str,
    lim: float,
    x_max: float = 150,
    max_points: int = 20_000,
):
    """Plot D_target (x) vs D_latent (y) for a single model."""

    set_pub_style()
    rng = np.random.default_rng(0)

    # Drop non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # Subsample for readability
    if len(x) > max_points:
        idx = rng.choice(len(x), size=max_points, replace=False)
        x, y = x[idx], y[idx]

    fig, ax = plt.subplots(figsize=(3.4, 3.1))

    ax.scatter(
        x,
        y,
        s=5,
        alpha=0.35,
        color=color,
        edgecolors="none",
        rasterized=True,
        zorder=2,
        label=label,
    )

    ax.plot(
        [0, max(lim, x_max)],
        [0, max(lim, x_max)],
        color=PALETTE["diag"],
        lw=1.1,
        ls="--",
        zorder=1,
        label=r"$D_{\mathrm{latent}}=D_{\mathrm{target}}$",
    )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.set_xlabel("Target distance")
    ax.set_ylabel("Latent Distance")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.grid(True, ls=":", lw=0.5, alpha=0.3, color="#cccccc")
    ax.tick_params(length=3)

    leg = ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        borderpad=0.5,
        handletextpad=0.5,
        markerscale=2.2,
    )
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout(pad=0.4)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = out.with_suffix(".pdf")
    png_path = out.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Success] Saved:\n  - {pdf_path.resolve()}\n  - {png_path.resolve()}")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_before",
        required=True,
        help="Slot-only / no-geometry checkpoint.",
    )

    parser.add_argument(
        "--ckpt_after",
        required=True,
        help="LEGO / variant-D checkpoint containing ProjectionHead.",
    )

    parser.add_argument(
        "--num_loc",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--dist",
        default="uniform",
    )

    parser.add_argument(
        "--method",
        default="construction",
    )

    parser.add_argument(
        "--data_dir",
        default="./data/slot_datasets_v2",
    )

    parser.add_argument(
        "--n_inst",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--instance_index",
        type=int,
        default=5,
        help="Instance used for the pairwise heatmaps.",
    )

    parser.add_argument(
        "--normalize_target",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--symmetrize_target",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--out",
        default="figures/lego_geometry_analysis",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Environment / dataset
    # --------------------------------------------------------

    env = CVRPEnv(
        generator_kwargs=dict(
            num_loc=args.num_loc,
        )
    )

    data_path = (
        Path(args.data_dir)
        / args.method
        / f"cvrp{args.num_loc}_{args.dist}_test.pt"
    )

    print("\n" + "=" * 70)
    print("Loading dataset")
    print("=" * 70)
    print(data_path)

    data = SlotDataset(
        data_path,
        variant="D",
    )

    n = min(
        args.n_inst,
        len(data),
    )

    if args.instance_index >= n:
        raise ValueError(
            f"instance_index={args.instance_index} "
            f"but only {n} instances are loaded."
        )

    print(f"Using {n} test instances.")

    locs = data.locs[:n]
    depot = data.depot[:n]
    demand = data.demand[:n]

    d_idx = data.d_ins_idx[:n]
    d_val = data.d_ins_val[:n]

    # --------------------------------------------------------
    # BEFORE / no-geometry model
    # --------------------------------------------------------

    before_model, slots_before, A_before = load_model_and_slots(
        args.ckpt_before,
        env,
        locs,
        depot,
        demand,
    )

    proj_before = find_projection_head(before_model)

    if proj_before is not None:
        print(
            "\nNOTE: --ckpt_before HAS a projection head; "
            "it will be used for its own geometry."
        )

    # --------------------------------------------------------
    # AFTER / LEGO model
    # --------------------------------------------------------

    after_model, slots_after, A_after = load_model_and_slots(
        args.ckpt_after,
        env,
        locs,
        depot,
        demand,
    )

    proj_after = find_projection_head(after_model)

    if proj_after is None:
        raise RuntimeError(
            "No ProjectionHead found in --ckpt_after. "
            "The AFTER checkpoint must contain the learned "
            "projection head used by the metric loss."
        )

    # --------------------------------------------------------
    # Latent geometry
    # --------------------------------------------------------

    print(
        "\nComputing latent geometry "
        "(each model in its own native space)..."
    )

    z_before, D_latent_before = compute_latent_geometry(
        slots_before,
        proj_before,
    )

    z_after, D_latent_after = compute_latent_geometry(
        slots_after,
        proj_after,
    )

    print(
        f"z_before shape: {tuple(z_before.shape)} "
        f"(proj_head={'yes' if proj_before else 'no'})"
    )
    print(
        f"z_after  shape: {tuple(z_after.shape)} "
        f"(proj_head=yes)"
    )

    print(
        f"D_latent_before shape: "
        f"{tuple(D_latent_before.shape)}"
    )
    print(
        f"D_latent_after shape: "
        f"{tuple(D_latent_after.shape)}"
    )

    # --------------------------------------------------------
    # Target geometry
    #
    # D_target depends on A_ik, so compute each model's target
    # using its OWN assignment matrix.
    # --------------------------------------------------------

    print(
        "\nComputing target geometry "
        "(per-model A_ik)..."
    )

    D_target_before = compute_target_geometry(
        d_idx,
        d_val,
        A_before,
        args.normalize_target,
        args.symmetrize_target,
    )

    D_target_after = compute_target_geometry(
        d_idx,
        d_val,
        A_after,
        args.normalize_target,
        args.symmetrize_target,
    )

    print(
        f"D_target_before shape: "
        f"{tuple(D_target_before.shape)}"
    )
    print(
        f"D_target_after shape: "
        f"{tuple(D_target_after.shape)}"
    )

    # Diagnostic only: the intended construction-induced pairwise distance
    # is symmetric. If the implementation produces a strongly asymmetric
    # target, keep the warning visible rather than silently correcting it.
    asym_b = (
        (D_target_before - D_target_before.transpose(-1, -2))
        .abs()
        .max()
        .item()
    )
    asym_a = (
        (D_target_after - D_target_after.transpose(-1, -2))
        .abs()
        .max()
        .item()
    )

    print(
        f"Target asymmetry max |D-D^T|: "
        f"B={asym_b:.6f}, D={asym_a:.6f}"
    )

    if max(asym_b, asym_a) > 1e-5:
        print(
            "WARNING: target geometry is asymmetric. "
            "The ratio heatmap displays only the upper triangle. "
            "If the intended construction distance is symmetric, "
            "verify the target aggregation or run with "
            "--symmetrize_target before using this figure."
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats_before = violation_statistics(
        D_target_before,
        D_latent_before,
    )

    stats_after = violation_statistics(
        D_target_after,
        D_latent_after,
    )

    print("\n" + "=" * 70)
    print("VIOLATION / CORRELATION STATISTICS")
    print("=" * 70)

    print_stats_block(
        "Without geometry regularization (B)",
        stats_before,
    )

    print_stats_block(
        "With geometry regularization (D)",
        stats_after,
    )

    rho_gap = (
        stats_after["spearman_rho"]
        - stats_before["spearman_rho"]
    )

    print(
        f"\nSpearman rho gap (D - B): "
        f"{rho_gap:+.4f}"
    )

    if rho_gap > 0.05:
        print(
            "  -> D's latent geometry rank-correlates with "
            "the construction target meaningfully more than B's."
        )
    else:
        print(
            "  -> D's correlation is not meaningfully higher "
            "than B's. Interpret expansion carefully."
        )

    # --------------------------------------------------------
    # Representative pair
    # Largest increase in D_latent / D_target ratio, B -> D.
    # --------------------------------------------------------

        # --------------------------------------------------------
    # Render two separate scatter plots (shared axis limits)
    # --------------------------------------------------------

    make_combined_figure(
        x_before=stats_before["target"].numpy(),
        y_before=stats_before["latent"].numpy(),
        x_after=stats_after["target"].numpy(),
        y_after=stats_after["latent"].numpy(),
        out_stem=f"{args.out}_scatter_combined",
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Combined scatter: {args.out}_scatter_combined.pdf / .png")

if __name__ == "__main__":
    main()
