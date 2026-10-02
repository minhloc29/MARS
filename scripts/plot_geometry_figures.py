"""
Publication-quality rendering of the slot-geometry analysis as TWO separate
figures:

  1. lego_geometry_scatter.{pdf,png}   -- the D_latent vs D_target scatter
  2. lego_geometry_pca.{pdf,png}       -- the two PCA panels (without / with)

Import `make_scatter_figure(...)` and `make_pca_figure(...)` from
`analyze_slot_geometry.py` after computing the real stats, OR run this file
standalone against the same stats dict structure produced there.

Design choices for a conference (two-column) figure:
  - A small, fixed, colorblind-safe palette (no default matplotlib tab10).
  - Larger marker face alpha with a thin dark edge so overlapping points in
    the scatter stay legible when the PDF is shrunk to ~0.48\\linewidth.
  - Legends carry ONLY the two series + rho (no redundant stats box baked
    into the plot area -- those numbers belong in the caption/prose, where
    a reviewer can read them without the figure fighting for space).
  - Consistent font sizes tuned for a ~3.3in half-column width at full
    resolution (so text doesn't shrink to illegibility after LaTeX scales it).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


# ----------------------------------------------------------------------
# Shared style
# ----------------------------------------------------------------------

PALETTE = {
    "without": "#C44E52",   # desaturated red
    "with":    "#4C72B0",   # desaturated blue
    "diag":    "#333333",
    "edge":    "#1a1a1a",
}

SLOT_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#8172B2",
    "#C44E52", "#64B5CD", "#CCB974", "#937860",
    "#8C8C8C", "#B07AA1",
]


def _resolve_label_offsets(pts: np.ndarray, data_radius: float):
    """
    Deterministically choose a label offset direction per point so that
    labels on nearby points don't render on top of each other.

    For each point, scores 4 candidate offset directions (NE, NW, SE, SW) by
    the distance to every OTHER point's position plus that candidate offset,
    and picks whichever direction keeps this label farthest from its
    neighbours' points and labels. This replaces a fixed "always offset
    up-right" placement, which silently collides when two slots end up close
    together in the PCA projection (as happens whenever the regularizer
    pulls/pushes slots into a tight sub-cluster).

    Returns a list of (dx, dy) in points (for `textcoords="offset points"`),
    one per input point, same order as `pts`.
    """
    n = len(pts)
    if n <= 1:
        return [(4, 4)] * n

    # Candidate directions in (dx, dy) point-offsets; diagonals so the label
    # never sits directly over the connecting dashed arrow between two points.
    candidates = [(6, 6), (-6, 6), (6, -6), (-6, -6)]

    # Convert a small fixed point-offset into data units using data_radius as
    # a rough data-units-per-point-offset proxy; this only needs to be good
    # enough to rank candidates relative to each other, not exact.
    data_offset = data_radius * 0.06

    offsets = []
    for i in range(n):
        best_cand, best_score = candidates[0], -np.inf
        for dx, dy in candidates:
            label_pos = pts[i] + np.array([dx, dy]) / 6.0 * data_offset
            # Distance from this candidate label position to every OTHER
            # point (not just point i) -- a label should avoid sitting near
            # any slot marker, not only its own.
            dists = np.linalg.norm(pts - label_pos, axis=1)
            dists[i] = np.inf  # ignore self
            score = dists.min()
            if score > best_score:
                best_score, best_cand = score, (dx, dy)
        offsets.append(best_cand)
    return offsets


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
        "pdf.fonttype": 42,     # editable text in PDF (avoid Type 3 fonts)
        "ps.fonttype": 42,
    })


# ----------------------------------------------------------------------
# Figure 1: scatter
# ----------------------------------------------------------------------

def make_scatter_figure(
    x_before, y_before, rho_before,
    x_after, y_after, rho_after,
    out_stem: str,
    max_points: int = 20_000,
):
    """
    x_*, y_* : 1D numpy arrays of D_target / D_latent (pooled, off-diagonal).
    rho_*    : Spearman rho (float) for each series, shown in the legend.
    """
    set_pub_style()

    rng = np.random.default_rng(0)

    def _subsample(x, y):
        if len(x) > max_points:
            idx = rng.choice(len(x), size=max_points, replace=False)
            return x[idx], y[idx]
        return x, y

    xb, yb = _subsample(x_before, y_before)
    xa, ya = _subsample(x_after, y_after)

    lim = float(max(x_before.max(), y_before.max(), x_after.max(), y_after.max()))
    lim = lim * 1.03

    fig, ax = plt.subplots(figsize=(3.4, 3.1))

    # Plot "without" first (background), "with" on top, since "with" is the
    # result the reader should focus on; z-order follows reading priority.
    ax.scatter(xb, yb, s=5, alpha=0.35, color=PALETTE["without"],
              edgecolors="none", rasterized=True, zorder=2,
              label=rf"Without ($\rho={rho_before:.2f}$)")
    ax.scatter(xa, ya, s=5, alpha=0.35, color=PALETTE["with"],
              edgecolors="none", rasterized=True, zorder=3,
              label=rf"With ($\rho={rho_after:.2f}$)")
    ax.plot([0, lim], [0, lim], color=PALETTE["diag"], lw=1.1, ls="--",
           zorder=1, label=r"$D_{\mathrm{latent}}=D_{\mathrm{target}}$")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Target distance $D_{\mathrm{target}}(k,l)$")
    ax.set_ylabel(r"Latent distance $D_{\mathrm{latent}}(k,l)$")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, ls=":", lw=0.5, alpha=0.3, color="#cccccc")
    ax.tick_params(length=3)

    # Minimal legend: 3 entries, placed where the point cloud is sparsest
    # (upper-left, since both clouds hug the lower-right / bottom in this
    # kind of data). Larger marker handles for legibility at print size.
    leg = ax.legend(loc="upper left", frameon=True, framealpha=0.92,
                    edgecolor="#cccccc", borderpad=0.5, handletextpad=0.5,
                    markerscale=2.2)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout(pad=0.4)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.pdf')} and .png")


# ----------------------------------------------------------------------
# Figure 2: PCA panels (without / with), side by side
# ----------------------------------------------------------------------

def make_pca_figure(
    pts_before, Dt_before, Dl_before,
    pts_after, Dt_after, Dl_after,
    k_sel, l_sel,
    out_stem: str,
):
    """
    pts_*  : (K, 2) PCA-projected slot coordinates for the representative
             instance (fit separately per model -- see analyze script note).
    Dt_*, Dl_* : (K, K) target / latent distance matrices for that instance.
    k_sel, l_sel : indices of the representative pair to annotate.
    """
    set_pub_style()
    K = pts_before.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))

    titles = ["Without geometry regularization", "With geometry regularization"]
    panels = [
        (axes[0], pts_before, Dt_before, Dl_before),
        (axes[1], pts_after, Dt_after, Dl_after),
    ]

    for ax, pts, Dt, Dl in panels:
        data_radius = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max())
        data_radius = max(data_radius, 1e-6)
        label_offsets = _resolve_label_offsets(pts, data_radius)

        for i in range(K):
            color = SLOT_PALETTE[i % len(SLOT_PALETTE)]
            ax.scatter(pts[i, 0], pts[i, 1], s=95, color=color,
                      edgecolor=PALETTE["edge"], linewidth=0.6, zorder=3)
            dx, dy = label_offsets[i]
            ha = "left" if dx > 0 else "right"
            va = "bottom" if dy > 0 else "top"
            ax.annotate(f"S{i+1}", (pts[i, 0], pts[i, 1]),
                       xytext=(dx, dy), textcoords="offset points",
                       fontsize=7.5, ha=ha, va=va, zorder=4,
                       bbox=dict(boxstyle="round,pad=0.08", fc="white",
                                ec="none", alpha=0.75))

        p1, p2 = pts[k_sel], pts[l_sel]
        dl, dt = Dl[k_sel, l_sel], Dt[k_sel, l_sel]
        violation = dl > dt
        arrow_color = PALETTE["without"] if violation else "#2E8B57"

        ax.annotate("", xy=p2, xytext=p1,
                   arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                   lw=1.6, linestyle="--", shrinkA=8, shrinkB=8))

        # Pad axes a bit so the annotation box has somewhere to sit without
        # overlapping the point cloud (the earlier version's #1 complaint).
        xr = pts[:, 0].max() - pts[:, 0].min()
        yr = pts[:, 1].max() - pts[:, 1].min()
        ax.set_xlim(pts[:, 0].min() - 0.12 * xr, pts[:, 0].max() + 0.12 * xr)
        ax.set_ylim(pts[:, 1].min() - 0.12 * yr, pts[:, 1].max() + 0.28 * yr)

        ratio = dl / max(dt, 1e-6)
        status = "violation" if violation else "satisfied"
        ax.text(
            0.5, 0.99,
            f"$D_{{\\mathrm{{latent}}}}$={dl:.1f}  $D_{{\\mathrm{{target}}}}$={dt:.1f}"
            f"  (ratio {ratio:.2f}, {status})",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.8,
            color=arrow_color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )

        ax.set_xlabel("PC 1")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, ls=":", lw=0.5, alpha=0.3, color="#cccccc")
        ax.tick_params(length=3)

    axes[0].set_ylabel("PC 2")
    axes[0].set_title(titles[0], fontsize=9.5)
    axes[1].set_title(titles[1], fontsize=9.5)

    fig.tight_layout(pad=0.5, w_pad=1.6)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.pdf')} and .png")


# ----------------------------------------------------------------------
# Standalone smoke test with synthetic data mimicking the reported stats
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Mimic: without -> rho=-0.30, median ratio 0.03 ; with -> rho=0.59, median ratio 0.75
    n = 50_000
    target = rng.uniform(20, 190, size=n)

    noise_b = rng.normal(0, 3, size=n)
    latent_before = np.clip(0.03 * target - 0.15 * (target - target.mean()) / target.std() * 5 + noise_b, 0, None)

    noise_a = rng.normal(0, 8, size=n)
    latent_after = np.clip(0.75 * target + 0.3 * (target - target.mean()) / target.std() * 15 + noise_a, 0, None)

    def spearman(x, y):
        xr = x.argsort().argsort().astype(float)
        yr = y.argsort().argsort().astype(float)
        xr -= xr.mean(); yr -= yr.mean()
        return float((xr * yr).sum() / (np.linalg.norm(xr) * np.linalg.norm(yr) + 1e-8))

    rho_b = spearman(target, latent_before)
    rho_a = spearman(target, latent_after)
    print(f"synthetic rho_before={rho_b:.3f}  rho_after={rho_a:.3f}")

    make_scatter_figure(
        target, latent_before, rho_b,
        target, latent_after, rho_a,
        out_stem="/home/claude/out/lego_geometry_scatter",
    )

    # Fake an 8-slot PCA layout for the two panels.
    K = 8
    pts_before = rng.normal(0, 1.5, size=(K, 2))
    pts_after = rng.normal(0, 18, size=(K, 2))
    Dt = np.abs(rng.normal(45, 15, size=(K, K)))
    Dt = (Dt + Dt.T) / 2
    np.fill_diagonal(Dt, 0)
    Dl_before = np.abs(np.subtract.outer(pts_before[:, 0], pts_before[:, 0])) + \
               np.abs(np.subtract.outer(pts_before[:, 1], pts_before[:, 1]))
    Dl_after = np.abs(np.subtract.outer(pts_after[:, 0], pts_after[:, 0])) + \
              np.abs(np.subtract.outer(pts_after[:, 1], pts_after[:, 1]))

    k_sel, l_sel = 2, 5
    make_pca_figure(
        pts_before, Dt, Dl_before,
        pts_after, Dt, Dl_after,
        k_sel, l_sel,
        out_stem="/home/claude/out/lego_geometry_pca",
    )