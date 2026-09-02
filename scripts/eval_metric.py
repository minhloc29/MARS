from __future__ import annotations

"""
eval_metric.py — Metric-geometry evaluation for the Slots-NCO (insertion) arm.

Tests the core claim: "slot latent geometry preserves the insertion-cost
geometry that supervised it." For a trained Variant-D (insertion) checkpoint:

    D_latent(k, l) = ||phi(z_k) - phi(z_l)||           (phi-space pairwise dist)
    D_slot  (k, l) = [A^T D_ins A]_{kl}                (aggregated target, sparse)

Metrics reported over the cached test split:
  * Spearman correlation rho = corr(D_latent, D_slot) on off-diagonal entries
    (higher = order preserved; matches the training claim).
  * One-sided metric violation V = E[(D_latent - D_slot)_+] (lower = better),
    consistent with the training penalty (latent must not exceed target).

Uses the SAME cached dataset + sparse d_ins the model was trained on, so the
target is identical to what supervised the slots. Latent is in phi(z) space
(matches the training loss), via the checkpoint's own projection head.
"""

import argparse
import json
from pathlib import Path

import torch
from tensordict import TensorDict

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot, AMSlot
from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline
from rl4co.models.nn.metric_loss import _aggregate_d_ins_sparse


def _allow_safe_globals() -> None:
    try:
        torch.serialization.add_safe_globals([SingleSharedBaseline])
    except Exception:
        pass


def _load_test_split(path: Path, num_loc: int, dist: str, method: str):
    """Load cached test split exactly as training's SlotDataset would."""
    fpath = path / method / f"cvrp{num_loc}_{dist}_test.pt"
    if not fpath.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {fpath}\n"
            "Generate it first, e.g.:\n"
            f"  python -m rl4co.data.generate_slot_dataset --num_locs {num_loc} "
            f"--dist {dist} --method {method} --out_dir {path} --n_test 1000"
        )
    data = torch.load(fpath, map_location="cpu", weights_only=False)
    fmt = data.get("format_version")
    if fmt != "sparse_v2":
        raise RuntimeError(f"Expected sparse_v2 dataset, got format_version={fmt!r}")
    data_method = data.get("method")
    if data_method is not None and data_method != method:
        raise RuntimeError(
            f"Dataset method tag {data_method!r} != requested {method!r}"
        )
    if data_method is None:
        print(
            f"[WARN] {fpath} has no 'method' tag (legacy dataset) — assuming it "
            f"matches --method {method!r}. Regenerate with the current generator "
            "to stamp the method tag."
        )
    return data


def off_diagonal(D: torch.Tensor) -> torch.Tensor:
    """Flatten strictly upper-triangular (off-diagonal) entries -> (B, M).
    D: (B, K, K)."""
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


def make_scatter(
    lat: torch.Tensor,     # (M,) pooled latent distances
    tgt: torch.Tensor,     # (M,) pooled target distances
    rho: float,
    violation: float,
    meta: dict,
    out_path: str,
    max_points: int = 25_000,
) -> None:
    """Scatter of latent vs target slot-pair distance with Spearman rho annotated.

    x = D_slot (insertion target), y = D_latent (phi-space latent). One series ->
    single hue; identity line y=x marks perfect agreement. Sub-samples dense clouds
    for tidy overplotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lat_n = lat.detach().cpu().numpy()
    tgt_n = tgt.detach().cpu().numpy()

    # Deterministic sub-sample to a manageable cloud size.
    if len(lat_n) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(lat_n), size=max_points, replace=False)
        lat_n, tgt_n = lat_n[idx], tgt_n[idx]

    hue = "#2a78d6"        # validated categorical blue (single series)
    ink = "#0b0b0b"
    line = "#8a8a86"

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(tgt_n, lat_n, s=9, alpha=0.35, color=hue, edgecolors="none", rasterized=True)

    # Identity diagonal: perfect agreement (D_latent == D_slot).
    lo = min(tgt_n.min(), lat_n.min())
    hi = max(tgt_n.max(), lat_n.max())
    ax.plot([lo, hi], [lo, hi], color=line, lw=1.4, ls="--", zorder=1,
            label="perfect agreement (y=x)")

    ax.set_xlabel("$D_{slot}$  (insertion target dist.)", color=ink)
    ax.set_ylabel("$D_{latent}$  ($\\phi(z_k)$-space dist.)", color=ink)
    ax.set_title(
        f"{meta['model']} · variant D · K={meta['K']} · N={meta['num_loc']}\n"
        f"slot geometry vs insertion cost — $n$={meta['n_inst']} inst",
        color=ink,
        fontsize=10,
    )
    # Speak the result directly (never color-alone).
    box = (
        f"Spearman $\\rho$ = {rho:.3f} ± {meta['spearman_std']:.3f}\n"
        f"one-sided violation $V$ = {violation:.4f}"
    )
    ax.text(0.03, 0.97, box, transform=ax.transAxes, va="top",
            fontsize=9, color=ink,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fcfcfb", ec="#c8c8c3", lw=1))
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(True, ls=":", lw=0.6, alpha=0.35, color="#d6d6d2")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c8c8c3")
    ax.tick_params(colors="#52514e")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Scatter saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate slot-metric geometry (insertion arm)")
    parser.add_argument("--ckpt", type=str, required=True, help="Trained Variant-D (insertion) .ckpt")
    parser.add_argument("--model", type=str, default="am", choices=["am", "pomo"],
                        help="Model class. Must match checkpoint.")
    parser.add_argument("--num_loc", type=int, default=100, help="N customers")
    parser.add_argument("--dist", type=str, default="uniform", choices=["uniform", "clustered"])
    parser.add_argument("--method", type=str, default="insertion",
                        choices=["insertion", "construction", "savings"],
                        help="d_ins cost method. Must match the checkpoint's ins_method.")
    parser.add_argument("--data_dir", type=str, default="./data/slot_datasets_v2")
    parser.add_argument("--n_inst", type=int, default=1024, help="Max instances to evaluate")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default=None, help="JSON output path")
    parser.add_argument("--plot", type=str, default=None,
                        help="Optional PNG path to also write the D_latent vs D_slot scatter with Spearman rho.")
    args = parser.parse_args()

    _allow_safe_globals()
    torch.manual_seed(args.seed)

    # ── Model ────────────────────────────────────────────────────────────────
    env = CVRPEnv(generator_kwargs=dict(num_loc=args.num_loc))
    model_cls = POMOSlot if args.model == "pomo" else AMSlot
    model = model_cls.load_from_checkpoint(args.ckpt, env=env, map_location="cpu")
    model.eval()
    dev = next(model.parameters()).device

    # Sanity: checkpoint must be an insertion (Variant D) slot model.
    if model.metric_variant != "D":
        raise RuntimeError(
            f"Checkpoint metric_variant is {model.metric_variant!r}; "
            "expected 'D' (insertion). This harness only evaluates the insertion arm."
        )
    if getattr(model, "disable_slots", False):
        raise RuntimeError("Checkpoint has disable_slots=True (no slot module) — nothing to evaluate.")
    if model.ins_method != args.method:
        raise RuntimeError(
            f"Checkpoint ins_method={model.ins_method!r} != --method {args.method!r}. "
            "The target must match what supervised the slots."
        )
    encoder = model.policy.encoder
    proj_head = model.metric_loss_fn.proj_head

    # ── Data (cached test split, same d_ins as training) ──────────────────────
    data = _load_test_split(Path(args.data_dir), args.num_loc, args.dist, args.method)
    locs    = data["locs"][: args.n_inst]    # (B, N, 2)
    depot   = data["depot"][: args.n_inst]   # (B, 2)
    demand  = data["demand"][: args.n_inst]  # (B, N)
    d_idx   = data["d_ins_idx"][: args.n_inst]  # (B, N, k) int16
    d_val   = data["d_ins_val"][: args.n_inst]  # (B, N, k) float32

    n_inst = locs.shape[0]
    N = locs.shape[1]

    corr_all, viol_all = [], []
    lat_all, tgt_all = [], []   # pooled off-diagonal entries for global scale
    with torch.no_grad():
        for i in range(0, n_inst, args.batch_size):
            sl = slice(i, min(i + args.batch_size, n_inst))
            B = locs[sl].shape[0]

            # Build pre-reset TensorDict (same keys as trainer collate).
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

            # Run encoder -> populates last_slots / last_A_ik side-channel.
            encoder(td)
            slots = encoder.last_slots      # (B, K, d)
            A_ik  = encoder.last_A_ik       # (B, N, K)
            if slots is None or A_ik is None:
                raise RuntimeError("Encoder did not populate slot side-channel.")

            # Latent geometry in phi-space (matches training loss).
            z_proj = proj_head(slots)                    # (B, K, proj_dim)
            diff = z_proj.unsqueeze(2) - z_proj.unsqueeze(1)
            D_latent = torch.norm(diff, p=2, dim=-1)     # (B, K, K)

            # Target geometry: A^T D_ins A (sparse), same as training aggregator.
            D_slot = _aggregate_d_ins_sparse(
                d_idx[sl].to(dev), d_val[sl].to(dev), A_ik, normalize=False, symmetrize=True
            )                                            # (B, K, K)

            # Off-diagonal pairs; ordered per-entry so correlation is meaningful.
            lat = off_diagonal(D_latent)                 # (B, M)
            tgt = off_diagonal(D_slot)                   # (B, M)
            lat_all.append(lat)
            tgt_all.append(tgt)

            # Per-instance Spearman, then aggregate.
            rho = spearman(lat, tgt)                     # (B,)
            corr_all.append(rho)

            # One-sided violation E[(D_latent - D_slot)_+] over off-diagonal.
            viol = torch.clamp(lat - tgt, min=0.0).mean(dim=-1)  # (B,)
            viol_all.append(viol)

    corr = torch.cat(corr_all)
    viol = torch.cat(viol_all)

    mean_corr = float(corr.mean())
    std_corr = float(corr.std())
    mean_viol = float(viol.mean())
    std_viol = float(viol.std())

    # Pooled latent/target scale over ALL instances (for interpretability:
    # latent and target live on very different absolute scales, hence Spearman).
    lat_flat = torch.cat(lat_all).flatten()
    tgt_flat = torch.cat(tgt_all).flatten()
    avg_lat = float(lat_flat.mean())
    avg_tgt = float(tgt_flat.mean())

    result = {
        "ckpt": args.ckpt,
        "model": args.model,
        "num_loc": args.num_loc,
        "dist": args.dist,
        "method": args.method,
        "K": slots.shape[1],
        "n_inst": corr.numel(),
        "pair_count_per_inst": lat.shape[1],
        "spearman_mean": mean_corr,
        "spearman_std": std_corr,
        "violation_one_sided_mean": mean_viol,
        "violation_one_sided_std": std_viol,
        "mean_latent_dist": avg_lat,
        "mean_target_dist": avg_tgt,
    }
    print(f"\n=== Metric-geometry eval (insertion, {args.method}) ===")
    print(f"  checkpoint variant D | K={result['K']} | N={result['num_loc']} | {result['n_inst']} inst")
    print(f"  Spearman rho (phi-space): {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"  One-sided violation  V:    {mean_viol:.6f} ± {std_viol:.6f}")
    print(f"  Scale: E[D_latent]={avg_lat:.4f}  E[D_slot]={avg_tgt:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Saved -> {out_path}")

    if args.plot:
        make_scatter(
            lat=torch.cat(lat_all).flatten(),
            tgt=torch.cat(tgt_all).flatten(),
            rho=mean_corr,
            violation=mean_viol,
            meta=result,
            out_path=args.plot,
        )


if __name__ == "__main__":
    main()
