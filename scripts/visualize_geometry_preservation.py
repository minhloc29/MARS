from __future__ import annotations

"""
visualize_geometry_preservation.py — publication figure for the LEGO-CVRP paper.

Answers the scientific question:

    "Does the learned slot representation preserve the construction-induced
     routing geometry?"

One horizontal 3-panel figure (white background, ~3:1, vector PDF):

    (a) Construction-induced slot target   D_target = Agg(A, D_cons) (K x K)
                                          (exact operation: the sparse kNN
                                           aggregator of the metric objective,
                                           NOT a dense A^T D_cons A bilinear
                                           form — see D_target bullet below)
    (b) Learned slot geometry              D_latent(k,l) = ||phi(z_k)-phi(z_l)||_2
    (c) Target vs learned distances        scatter over upper-triangular k<l pairs

Everything here reuses the ACTUAL repo computation, never synthetic values:

  * D_cons   -- rl4co.data.insertion_cost._compute_dense_construction_d_ins
                (deterministic greedy NN tour -> cumulative along-route position
                 -> shortest arc distance min(|p_i-p_j|, L-|p_i-p_j|)),
                sparsified to per-node kNN exactly as the cached dataset stores it.
  * A_ik     -- rl4co.models.nn.slot_attention.SlotAttention soft assignment
                (softmax over slots, L1-norm over nodes).
  * D_target -- rl4co.models.nn.metric_loss._aggregate_d_ins_sparse
                (A^T (D_cons . A_neighbors)); the SAME sparse per-node kNN
                aggregator the training loss consumes. Written compactly as
                D_target = Agg(A, D_cons) — deliberately NOT the dense
                A^T D_cons A bilinear form, because the implementation only
                aggregates over each node's kNN neighbor mask (D_cons here is
                already sparsified to that mask), so a naive dense product
                would mis-state the operation. Aggregation mode is controllable
                via --normalize_target / --symmetrize_target (default: raw, i.e.
                neither, matching the guide's --no_normalize_target
                --no_symmetrize_target "OURS" LEHD-slot run).
  * D_latent -- rl4co.models.nn.metric_loss.ProjectionHead phi(z_k) on slot
                embeddings, pairwise L2 norm (the SAME projection head used by
                the MetricPreservationLoss geometry objective).

Statistics are computed over ALL upper-triangular slot pairs (k<l) across the
selected eval split. The heatmaps show one representative instance (deterministic
--instance_index); the scatter pools the whole eval split (deterministic
sub-sample when large to keep the figure readable).

Normalization convention (panel c): the construction target (cumulative
along-route arc length, O(1..150)) and the learned latent (an L2 embedding
norm, O(0.01..5)) live on incomparable absolute scales, so panel (c) min-max
normalizes BOTH to [0,1] and labels them "Normalized"; this makes the y=x
diagonal directly readable. Spearman rho is rank-based and scale-invariant;
the reported MSE is computed in that same [0,1] normalized space and labeled
"MSE (normalized)". This figure therefore claims only a monotonic (rank)
relationship, which is the honest strength of a partial-routing learned
geometry; whether the representation improves cross-size generalization is a
separate experimental question.

Run (repo root; requires a trained Variant-D construction checkpoint):

    export PYTHONPATH="$PWD:$PYTHONPATH"
    source .venv/bin/activate
    python scripts/visualize_geometry_preservation.py \
        --ckpt <path/to/variantD_construction.ckpt> \
        --model pomo --method construction \
        --data_dir ./data/slot_datasets_v2 \
        --out figures/lego_geometry_preservation
"""

import argparse
from pathlib import Path

import torch
from tensordict import TensorDict

# ── Geometry/data leaves (imported first; do not depend on the heavy model chain) ──
from rl4co.data.slot_dataset import SlotDataset
from rl4co.models.nn.metric_loss import _aggregate_d_ins_sparse

# ── Model loading (heavy; needs the full rl4co.models chain) ──────────────────────
from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot, AMSlot
from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline


# ────────────────────────────────────────────────────────────────────────────────
# Statistics helpers (mirror scripts/eval_metric.py so numbers are comparable)
# ────────────────────────────────────────────────────────────────────────────────

def off_diagonal(D: torch.Tensor) -> torch.Tensor:
    """Flatten strictly upper-triangular (k<l) entries -> (B, M). D: (B, K, K)."""
    B, K, _ = D.shape
    triu = torch.triu(torch.ones(K, K, dtype=torch.bool, device=D.device), diagonal=1)
    return D[:, triu]  # (B, K*(K-1)/2)


def spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation between matching flat vectors x, y."""
    xr = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()
    yr = torch.argsort(torch.argsort(y, dim=-1), dim=-1).float()
    xr = xr - xr.mean(dim=-1, keepdim=True)
    yr = yr - yr.mean(dim=-1, keepdim=True)
    denom = xr.norm(dim=-1) * yr.norm(dim=-1)
    return (xr * yr).sum(dim=-1) / (denom + 1e-8)


def pooled_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared error over the pooled (flattened) inputs."""
    return float(torch.mean((a - b) ** 2))


# ────────────────────────────────────────────────────────────────────────────────
# Figure rendering
# ────────────────────────────────────────────────────────────────────────────────

def _heatmap_axes(ax, D: torch.Tensor, cbar_ax, vmax: float,
                  cmap, label_panel: str, formula: str):
    """Render one K x K distance heatmap on `ax` with a colorbar on `cbar_ax`.

    Both panels (a) and (b) share the same colormap and the same [0, 1] color
    axis (each matrix max-normalized by its own max). The target (cumulative
    along-route arc length, in route-length units) and the latent (a learned
    embedding norm) live on incomparable absolute scales, so forcing one shared
    raw color scale would flatten one panel to a single color. Max-normalizing
    each makes the STRUCTURE (which slot pairs are near/far) comparable, and the
    scale-invariant rank/order agreement is reported quantitatively in panel (c).
    """
    Dn = D / (vmax + 1e-12)
    im = ax.imshow(Dn, cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest", rasterized=False)
    ax.set_xticks(range(D.shape[0]))
    ax.set_yticks(range(D.shape[1]))
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Slot index")
    ax.set_title(f"({label_panel})\n{formula}", fontsize=9)
    cb = ax.figure.colorbar(im, cax=cbar_ax)
    cb.set_label("distance (max-normalized)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    # Keep the diagonal visible.
    ax.plot([-0.5, D.shape[0] - 0.5], [-0.5, D.shape[1] - 0.5],
            color="#9a9a9a", lw=0.6, ls=":", alpha=0.9)
    return im


def render_figure(
    D_target_inst: torch.Tensor,   # (K, K) representative instance target
    D_latent_inst: torch.Tensor,   # (K, K) representative instance latent
    lat_flat: torch.Tensor,        # (M_pooled,) pooled latent upper-tri entries
    tgt_flat: torch.Tensor,        # (M_pooled,) pooled target upper-tri entries
    rho: float,
    mse: float,
    meta: dict,
    out_stem: str,
    dpi: int,
    max_points: int = 25_000,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "mathtext.fontset": "cm",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    K = D_target_inst.shape[0]
    # Shared blue-white-red sequential presentation of non-negative distances.
    cmap = plt.get_cmap("RdBu")
    vmax_t = float(D_target_inst.max())
    vmax_l = float(D_latent_inst.max())

    fig = plt.figure(figsize=(14.0, 4.4))
    # Panel (c) holds the actual quantitative evidence, so give it a touch
    # more width (1:1:1.2).
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.55,
                          left=0.07, right=0.97, top=0.86, bottom=0.18)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # Colorbar axes tucked to the right of each heatmap.
    cax_a = ax_a.inset_axes([1.06, 0.0, 0.06, 1.0])
    cax_b = ax_b.inset_axes([1.06, 0.0, 0.06, 1.0])

    # Panel (a) — construction-induced slot target.
    # Naming it D_target (= the aggregator output the loss consumes), and
    # writing it as Agg(A, D_cons) rather than a dense A^T D_cons A: the
    # implementation aggregates over each node's sparse kNN mask only.
    _heatmap_axes(
        ax_a, D_target_inst, cax_a, vmax_t, cmap, "a",
        r"$D_{\mathrm{target}} = \mathrm{Agg}(A_{\mathrm{kNN}}, D_{\mathrm{cons}})$",
    )

    # Panel (b) — learned slot geometry.
    _heatmap_axes(
        ax_b, D_latent_inst, cax_b, vmax_l, cmap, "b",
        r"$D_{\mathrm{latent}}(k,l) = \|\phi(z_k)-\phi(z_l)\|_2$",
    )

    # No "geometry preservation" arrow between (a) and (b): with only a
    # moderate Spearman rho the scatter in (c) demonstrates the monotonic
    # relationship, and a large visual assertion would overstate it.

    # Panel (c) — target vs learned distances (upper-triangular k<l only).
    #
    # The two dissimilarities live on incomparable absolute scales: the
    # construction target is cumulative along-route arc length (O(1..150) for
    # N=100), while the latent is an L2 norm in a normalized embedding space
    # (O(0.01..5)). A raw y=x overlay would flatten every point onto the x-axis
    # and make "learned ~ target" untestable by eye. We therefore min-max
    # normalize BOTH sets to [0,1] (see the normalization-convention note in
    # the module docstring); the labels read "Normalized". Spearman rho is
    # rank-based and unaffected; the reported MSE lives in this same [0,1]
    # space and is labeled "MSE (normalized)".
    lat_raw = lat_flat.detach().cpu().numpy()
    tgt_raw = tgt_flat.detach().cpu().numpy()
    n_all = len(lat_raw)

    # Min-max normalization to [0, 1] (deterministic across runs).
    t_lo, t_hi = float(tgt_raw.min()), float(tgt_raw.max())
    l_lo, l_hi = float(lat_raw.min()), float(lat_raw.max())
    tgt_s = (tgt_raw - t_lo) / (t_hi - t_lo + 1e-12)
    lat_s = (lat_raw - l_lo) / (l_hi - l_lo + 1e-12)

    lat_n, tgt_n = lat_s, tgt_s
    if n_all > max_points:  # deterministic sub-sample to keep the panel readable
        rng = np.random.default_rng(0)
        idx = rng.choice(n_all, size=max_points, replace=False)
        lat_n, tgt_n = lat_s[idx], tgt_s[idx]

    hue = "#2a78d6"
    ink = "#0b0b0b"
    line = "#8a8a86"

    # A 2D hexbin density field carries the pooled point-cloud so the reader
    # sees the positive relationship rather than a wall of individual points;
    # a light low-alpha scatter overlay keeps individual pairs visible. Both
    # share the same [0,1] unit square and the same blue family as (a)/(b).
    hb = ax_c.hexbin(tgt_n, lat_n, gridsize=70, cmap="Blues", mincnt=1,
                     linewidths=0.0, alpha=0.85, extent=(0.0, 1.0, 0.0, 1.0))
    ax_c.scatter(tgt_n, lat_n, s=3.5, alpha=0.18, color=hue,
                 edgecolors="none", rasterized=True, linewidths=0)
    ax_c.plot([0.0, 1.0], [0.0, 1.0], color=line, lw=1.2, ls="--", zorder=1,
              label="y = x (perfect agreement)")
    ax_c.set_xlabel("Normalized target distance  $D_{\\mathrm{target}}(k,l)$")
    ax_c.set_ylabel("Normalized latent distance  $D_{\\mathrm{latent}}(k,l)$")
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_title("(c) Target vs learned distances")
    # Statistics — real values computed over all k<l slot pairs in the eval split.
    box = (
        f"Spearman $\\rho$ = {rho:.3f}\n"
        f"MSE (normalized) = {mse:.5f}\n"
        f"$n$ = {n_all:,} slot pairs"
    )
    ax_c.text(0.03, 0.82, box, transform=ax_c.transAxes, va="top", fontsize=8,
              color=ink, bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb",
                                   ec="#c8c8c3", lw=0.8))
    ax_c.legend(loc="lower right", frameon=False, fontsize=7)
    ax_c.grid(True, ls=":", lw=0.5, alpha=0.35, color="#d6d6d2")
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)
    ax_c.tick_params(labelsize=7)

    # Density colorbar for the hexbin field (what the panel actually encodes).
    cax_c = ax_c.inset_axes([1.06, 0.0, 0.05, 1.0])
    fig.colorbar(hb, cax=cax_c)
    cax_c.set_ylabel("point density", fontsize=7)
    cax_c.tick_params(labelsize=6)

    # Reproducibility footer.
    foot = (
        f"checkpoint: {meta['ckpt']}  |  model: {meta['model']}  |  "
        f"dataset: {meta['dist']}, N={meta['num_loc']}, method={meta['method']}  |  "
        f"split: {meta['split']}  |  instance index: {meta['instance_index']}  |  "
        f"K={meta['K']}  |  agg: normalize={meta['normalize_target']}, "
        f"symmetrize={meta['symmetrize_target']}  |  eval n_inst: {meta['n_inst']}"
    )
    fig.text(0.5, 0.015, foot, ha="center", va="bottom", fontsize=6,
             color="#555555")

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    print(f"Saved PDF -> {out.with_suffix('.pdf')}")
    print(f"Saved PNG -> {out.with_suffix('.png')} (dpi={dpi})")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def _allow_safe_globals() -> None:
    try:
        torch.serialization.add_safe_globals([SingleSharedBaseline])
    except Exception:
        pass


def _load_test_split(path: Path, num_loc: int, dist: str, method: str):
    """Load the cached test split exactly as training's SlotDataset would."""
    fpath = path / method / f"cvrp{num_loc}_{dist}_test.pt"
    if not fpath.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {fpath}\n"
            "Generate it first, e.g.:\n"
            f"  python -m rl4co.data.generate_slot_dataset --num_locs {num_loc} "
            f"--dist {dist} --method {method} --out_dir {path} --n_test 2000"
        )
    return SlotDataset(fpath, variant="D", max_instances=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication figure: does the learned slot geometry preserve "
                    "the construction-induced routing geometry?"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Trained Variant-D (construction) .ckpt")
    parser.add_argument("--model", type=str, default="pomo", choices=["pomo", "am"])
    parser.add_argument("--num_loc", type=int, default=100)
    parser.add_argument("--dist", type=str, default="uniform",
                        choices=["uniform", "clustered"])
    parser.add_argument("--method", type=str, default="construction",
                        choices=["insertion", "construction", "savings"],
                        help="d_ins cost method; must match the checkpoint's ins_method.")
    parser.add_argument("--data_dir", type=str, default="./data/slot_datasets_v2")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n_inst", type=int, default=2000,
                        help="Max instances to include in the pooled scatter.")
    parser.add_argument("--instance_index", type=int, default=0,
                        help="Deterministic instance shown in panels (a)/(b).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--normalize_target", action="store_true", default=False,
                        help="Aggregate target with slot-mass normalization "
                             "(default False = raw A^T D_cons A, per guide's "
                             "--no_normalize_target).")
    parser.add_argument("--symmetrize_target", action="store_true", default=False,
                        help="Symmetrize the aggregated target (0.5*(D+D^T)) "
                             "(default False = raw, per guide's "
                             "--no_symmetrize_target).")
    parser.add_argument("--out", type=str,
                        default="figures/lego_geometry_preservation")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    _allow_safe_globals()
    torch.manual_seed(args.seed)

    # ── Model ────────────────────────────────────────────────────────────────
    env = CVRPEnv(generator_kwargs=dict(num_loc=args.num_loc))
    model_cls = POMOSlot if args.model == "pomo" else AMSlot
    model = model_cls.load_from_checkpoint(args.ckpt, env=env, map_location="cpu")
    model.eval()

    # Sanity: a Variant-D slot model matching the requested target method.
    if model.metric_variant != "D":
        raise RuntimeError(
            f"Checkpoint metric_variant is {model.metric_variant!r}; expected 'D'.")
    if getattr(model, "disable_slots", False):
        raise RuntimeError("Checkpoint has disable_slots=True (no slot module).")
    if model.ins_method != args.method:
        raise RuntimeError(
            f"Checkpoint ins_method={model.ins_method!r} != --method {args.method!r}.")

    encoder = model.policy.encoder          # SlotInjectingEncoder (side channel)
    proj_head = model.metric_loss_fn.proj_head
    K = int(model.num_slots)

    # ── Data (cached split with the same d_ins that supervised training) ──────
    data = _load_test_split(Path(args.data_dir), args.num_loc, args.dist, args.method)
    n_avail = len(data)
    n_inst = min(args.n_inst, n_avail)
    locs    = data.locs[:n_inst]       # (B, N, 2) customers (depot separate)
    depot   = data.depot[:n_inst]      # (B, 2)
    demand  = data.demand[:n_inst]     # (B, N)
    d_idx   = data.d_ins_idx[:n_inst]  # (B, N, k) int16
    d_val   = data.d_ins_val[:n_inst]  # (B, N, k) float32
    N = locs.shape[1]

    print(f"\n=== Geometry-preservation figure ({args.model} / variant D / {args.method}) ===")
    print(f"  checkpoint : {args.ckpt}")
    print(f"  dataset    : {args.dist}, N={args.num_loc}, method={args.method}, split={args.split}")
    print(f"  instance   : {args.instance_index} (panels a/b), {n_inst} pooled (panel c)")
    print(f"  K          : {K}")
    print(f"  target agg : normalize={args.normalize_target}, symmetrize={args.symmetrize_target}")

    dev = next(model.parameters()).device
    lat_all, tgt_all = [], []
    D_target_inst = D_latent_inst = None

    with torch.no_grad():
        for i in range(0, n_inst, args.batch_size):
            sl = slice(i, min(i + args.batch_size, n_inst))
            B = locs[sl].shape[0]

            td_in = TensorDict(
                {
                    "locs": locs[sl].to(dev),
                    "depot": depot[sl].to(dev),
                    "demand": demand[sl].to(dev),
                    "capacity": torch.ones(B, 1, device=dev),
                },
                batch_size=[B],
            )
            td = env.reset(td_in)

            # Run encoder -> populates last_slots / last_A_ik side channel.
            encoder(td)
            slots = encoder.last_slots      # (B, K, d)
            A_ik  = encoder.last_A_ik       # (B, N, K)
            if slots is None or A_ik is None:
                raise RuntimeError("Encoder did not populate the slot side-channel.")

            # Learned geometry in phi space (same projection head as the loss).
            z_proj = proj_head(slots)                        # (B, K, proj_dim)
            diff = z_proj.unsqueeze(2) - z_proj.unsqueeze(1)
            D_latent = torch.norm(diff, p=2, dim=-1)          # (B, K, K)

            # Construction target aggregated into slot space (same aggregator as
            # MetricPreservationLoss._get_target).
            D_target = _aggregate_d_ins_sparse(
                d_idx[sl].to(dev), d_val[sl].to(dev), A_ik,
                normalize=args.normalize_target,
                symmetrize=args.symmetrize_target,
            )                                                 # (B, K, K)

            # Upper-triangular (k<l) entries only -> no duplicate symmetric pairs.
            lat = off_diagonal(D_latent)                      # (B, M)
            tgt = off_diagonal(D_target)                      # (B, M)
            lat_all.append(lat)
            tgt_all.append(tgt)

            # Keep the representative instance for the heatmaps.
            if args.instance_index in range(i, i + B):
                j = args.instance_index - i
                D_target_inst = D_target[j].cpu()
                D_latent_inst = D_latent[j].cpu()

    lat_all = torch.cat(lat_all)
    tgt_all = torch.cat(tgt_all)
    lat_flat = lat_all.flatten()
    tgt_flat = tgt_all.flatten()

    if D_target_inst is None or D_latent_inst is None:
        raise RuntimeError(
            f"--instance_index {args.instance_index} outside the {n_inst}-instance eval split.")

    # ── Real statistics over all k<l slot pairs in the eval split ────────────
    rho = float(spearman(lat_flat, tgt_flat))
    # Normalized MSE in the [0,1] min-max space that panel (c) uses — the
    # natural unit for the y=x panel. (The raw-scale MSE is printed below for
    # reference but is NOT what the figure reports, since raw target and raw
    # latent live on incomparable scales.)
    mse_raw = pooled_mse(lat_flat, tgt_flat)
    def _minmax(v):
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / (hi - lo + 1e-12)
    mse_norm = pooled_mse(_minmax(lat_flat), _minmax(tgt_flat))

    # ── Symmetry / shape validation (see the task checklist) ─────────────────
    def _max_sym_err(M):
        return float((M - M.T).abs().max())

    lat_sym = _max_sym_err(D_latent_inst)
    tgt_sym = _max_sym_err(D_target_inst)
    print(f"  D_target shape : {tuple(D_target_inst.shape)}  sym-err={tgt_sym:.3e}  "
          f"diag-max={float(torch.diag(D_target_inst).abs().max()):.3e}")
    print(f"  D_latent shape : {tuple(D_latent_inst.shape)}  sym-err={lat_sym:.3e}  "
          f"diag-max={float(torch.diag(D_latent_inst).abs().max()):.3e}")
    print(f"  # scatter points (k<l) : {lat_flat.numel():,}"
          f"  ({lat_all.shape[1]} per instance x {lat_all.shape[0]} instances)")
    print(f"  Spearman rho  : {rho:.4f}")
    print(f"  MSE (raw scale)  : {mse_raw:.5f}   [reference only]")
    print(f"  MSE (normalized) : {mse_norm:.5f}   [reported in figure]")

    # ── Render ───────────────────────────────────────────────────────────────
    meta = {
        "ckpt": args.ckpt,
        "model": args.model,
        "num_loc": args.num_loc,
        "dist": args.dist,
        "method": args.method,
        "split": args.split,
        "instance_index": args.instance_index,
        "K": K,
        "n_inst": int(lat_all.shape[0]),
        "normalize_target": args.normalize_target,
        "symmetrize_target": args.symmetrize_target,
    }
    render_figure(
        D_target_inst, D_latent_inst, lat_flat, tgt_flat, rho, mse_norm,
        meta, args.out, args.dpi,
    )


if __name__ == "__main__":
    main()
