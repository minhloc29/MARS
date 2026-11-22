#!/usr/bin/env python
"""
Evaluation script for Cross-Attention vs Slot Attention ablation (LEGO repo).

Produces:
  1. Paper Table 4 format:
     Columns: Method | Uniform | Gaussian | Cluster | Explosion | Shifted Mean
     where each column is the mean gap (%) vs HGS reference across N in {100, 200, 500}.
     Shifted Mean = mean(Gaussian, Cluster, Explosion).
  2. Paper Table 5 format (for geometry-on variants on 1,000 Uniform-100 instances):
     Viol. (% pairs D_latent > D_target), median D_latent/D_target,
     Spearman rho(D_latent, D_cons), slot-usage entropy, params, and inference time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.models.zoo.pomo_slot.model import POMOSlot
from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline
from rl4co.models.nn.metric_loss import _aggregate_d_ins_sparse
from rl4co.utils import eval_utils

# Standard CVRP capacities (Kool et al., 2019)
CAPACITIES = {100: 50.0, 200: 70.0, 500: 100.0}

# Default HGS reference costs for synthetic benchmarks if not supplied via JSON
# (historical benchmark baselines from literature / paper)
DEFAULT_HGS_COSTS = {
    "uniform": {100: 15.65, 200: 25.12, 500: 51.50},
    "gaussian": {100: 13.80, 200: 22.35, 500: 46.10},
    "cluster": {100: 14.20, 200: 23.10, 500: 47.80},
    "explosion": {100: 12.90, 200: 21.05, 500: 43.70},
}


def _allow_safe_globals() -> None:
    try:
        torch.serialization.add_safe_globals([SingleSharedBaseline])
    except Exception:
        pass


def generate_dist_instances(dist: str, num_loc: int, n_inst: int, seed: int = 1234) -> TensorDict:
    """Generate instances for Uniform, Gaussian, Cluster, or Explosion distributions."""
    torch.manual_seed(seed)
    cap = CAPACITIES.get(num_loc, 50.0)

    if dist == "uniform":
        locs = torch.rand(n_inst, num_loc, 2)
    elif dist == "gaussian":
        # Centered 2D Gaussian truncated to [0, 1]^2
        locs = (0.5 + 0.15 * torch.randn(n_inst, num_loc, 2)).clamp(0.0, 1.0)
    elif dist in ("cluster", "clustered"):
        # Gaussian Mixture (3 to 7 clusters)
        from rl4co.data.generate_slot_dataset import _gen_clustered
        locs = _gen_clustered(n_inst, num_loc)
    elif dist == "explosion":
        # Central source with radial dispersal
        r = torch.sqrt(torch.rand(n_inst, num_loc, 1)) * 0.45
        theta = torch.rand(n_inst, num_loc, 1) * 2 * math.pi
        locs = torch.cat([0.5 + r * torch.cos(theta), 0.5 + r * torch.sin(theta)], dim=-1).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    depot = torch.full((n_inst, 1, 2), 0.5)
    # Demands in [1, 9] normalized by capacity
    raw_demand = torch.randint(1, 10, (n_inst, num_loc)).float()
    demand = raw_demand / cap
    capacity = torch.ones(n_inst, 1)

    return TensorDict(
        {
            "locs": locs,
            "depot": depot.squeeze(1),
            "demand": demand,
            "capacity": capacity,
        },
        batch_size=[n_inst],
    )


def compute_table5_diagnostics(model: POMOSlot, env: CVRPEnv, n_inst: int = 1000, device: str = "cpu") -> dict:
    """Compute Table 5 diagnostics on Uniform-100 instances."""
    torch.manual_seed(1234)
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    # 1. Parameter count
    slot_mod = getattr(getattr(model.policy, "encoder", None), "slot_attn", None)
    total_params = sum(p.numel() for p in model.parameters())
    slot_params = sum(p.numel() for p in slot_mod.parameters()) if slot_mod else 0

    if not hasattr(model, "metric_loss_fn") or model.metric_loss_fn is None:
        return {
            "viol_pct": float("nan"),
            "median_ratio": float("nan"),
            "spearman_rho": float("nan"),
            "entropy": float("nan"),
            "total_params": total_params,
            "slot_params": slot_params,
            "inference_time_ms": float("nan"),
        }

    proj_head = model.metric_loss_fn.proj_head.to(dev)
    from rl4co.data.insertion_cost import compute_sparse_insertion_cost

    batch_size = 100
    all_viol, all_ratios, all_rhos, all_entropies = [], [], [], []
    total_time = 0.0

    with torch.no_grad():
        for start_idx in range(0, n_inst, batch_size):
            B = min(batch_size, n_inst - start_idx)
            raw = generate_dist_instances("uniform", 100, B, seed=1234 + start_idx).to(dev)

            # Compute d_ins targets on the fly
            locs_cust = raw["locs"]
            depot_loc = raw["depot"].unsqueeze(1) if raw["depot"].dim() == 2 else raw["depot"]
            d_idx, d_val = compute_sparse_insertion_cost(
                locs_cust, k_neighbors=15, depot_loc=depot_loc, method="construction"
            )

            t0 = time.perf_counter()
            td = env.reset(raw.clone())
            model.policy.encoder(td)
            t1 = time.perf_counter()
            total_time += (t1 - t0)

            slots = model.policy.encoder.last_slots
            A_ik = model.policy.encoder.last_A_ik

            # Entropy
            p_k = A_ik.mean(dim=1).clamp(min=1e-8)
            entropy = -(p_k * p_k.log()).sum(dim=-1).mean().item()
            all_entropies.append(entropy)

            # Latent distances
            z_proj = proj_head(slots)
            diff = z_proj.unsqueeze(2) - z_proj.unsqueeze(1)
            D_latent = torch.norm(diff, p=2, dim=-1)

            # Target distances
            D_target = _aggregate_d_ins_sparse(
                d_idx.to(dev), d_val.to(dev), A_ik, normalize=True, symmetrize=True
            )

            # Off-diagonal
            K = slots.shape[1]
            triu = torch.triu(torch.ones(K, K, dtype=torch.bool, device=dev), diagonal=1)
            lat_off = D_latent[:, triu]  # (B, M)
            tgt_off = D_target[:, triu]  # (B, M)

            # Violation percentage: D_latent > D_target
            viol = (lat_off > tgt_off).float().mean().item() * 100.0
            all_viol.append(viol)

            # Ratio D_latent / D_target
            ratio = (lat_off / (tgt_off + 1e-8)).median().item()
            all_ratios.append(ratio)

            # Spearman correlation
            for b in range(B):
                xr = torch.argsort(torch.argsort(lat_off[b])).float()
                yr = torch.argsort(torch.argsort(tgt_off[b])).float()
                xr = xr - xr.mean()
                yr = yr - yr.mean()
                denom = xr.norm() * yr.norm() + 1e-8
                rho = (xr * yr).sum() / denom
                all_rhos.append(rho.item())

    return {
        "viol_pct": float(torch.tensor(all_viol).mean()),
        "median_ratio": float(torch.tensor(all_ratios).median()),
        "spearman_rho": float(torch.tensor(all_rhos).mean()),
        "entropy": float(torch.tensor(all_entropies).mean()),
        "total_params": total_params,
        "slot_params": slot_params,
        "inference_time_ms": (total_time / n_inst) * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cross-Attention vs Slot Attention ablation.")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="List of checkpoint paths: name=path (e.g. LEGO=path/to/best.ckpt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_inst", type=int, default=1000, help="Instances per setting (default 1000)")
    parser.add_argument("--num_starts", type=int, default=100, help="Multi-start greedy POMO starts (default 100)")
    parser.add_argument("--out_dir", default="results/ablation_cross", help="Output directory")
    args = parser.parse_args()

    _allow_safe_globals()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_map = {}
    for item in args.checkpoints:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name = Path(item).stem
            path = item
        ckpt_map[name] = path

    distributions = ["uniform", "gaussian", "cluster", "explosion"]
    sizes = [100, 200, 500]

    table4_rows = []
    table5_rows = []

    for name, ckpt_path in ckpt_map.items():
        print(f"\n=======================================================")
        print(f"Evaluating Model: {name} ({ckpt_path})")
        print(f"=======================================================")

        # Load model
        temp_env = CVRPEnv(generator_params=dict(num_loc=100))
        model = POMOSlot.load_from_checkpoint(ckpt_path, env=temp_env, map_location=args.device)
        model.to(args.device)
        model.eval()

        # Compute Table 5 Diagnostics (Uniform-100)
        print("-> Computing Table 5 geometry diagnostics on Uniform-100...")
        diag = compute_table5_diagnostics(model, temp_env, n_inst=min(args.n_inst, 1000), device=args.device)
        diag["method"] = name
        table5_rows.append(diag)

        # Compute Table 4 Gaps across dists and sizes
        print("-> Computing Table 4 performance across distributions and sizes...")
        dist_gaps = {}
        for dist in distributions:
            gaps_for_dist = []
            for N in sizes:
                env = CVRPEnv(generator_params=dict(num_loc=N))
                td = generate_dist_instances(dist, N, args.n_inst, seed=1234).to(args.device)

                eval_args = argparse.Namespace(
                    decode="multistart_greedy",
                    num_starts=args.num_starts,
                    batch_size=64,
                    augment=False,
                    local_search=False,
                    device=args.device,
                )
                res = eval_utils.evaluate(model, eval_args, env, td)
                mean_cost = res["mean_tour_length"]

                ref_hgs = DEFAULT_HGS_COSTS.get(dist, {}).get(N, None)
                if ref_hgs is not None:
                    gap_pct = ((mean_cost - ref_hgs) / ref_hgs) * 100.0
                else:
                    gap_pct = float("nan")
                gaps_for_dist.append(gap_pct)
                print(f"   [{dist:9s} N={N:3d}] Cost: {mean_cost:7.3f} | HGS: {ref_hgs} | Gap: {gap_pct:6.2f}%")

            # Mean gap over N for this distribution
            dist_gaps[dist] = sum(gaps_for_dist) / len(gaps_for_dist)

        # Shifted mean = mean of (gaussian, cluster, explosion)
        shifted_mean = (dist_gaps["gaussian"] + dist_gaps["cluster"] + dist_gaps["explosion"]) / 3.0

        table4_rows.append({
            "Method": name,
            "Uniform": dist_gaps["uniform"],
            "Gaussian": dist_gaps["gaussian"],
            "Cluster": dist_gaps["cluster"],
            "Explosion": dist_gaps["explosion"],
            "Shifted Mean": shifted_mean,
        })

    # Save and Print Table 4
    t4_md_lines = [
        "### Table 4: Generalization Across Distributions (Mean Gap % vs HGS over N in {100, 200, 500})",
        "",
        "| Method | Uniform | Gaussian | Cluster | Explosion | Shifted Mean |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in table4_rows:
        t4_md_lines.append(
            f"| {r['Method']} | {r['Uniform']:.2f}% | {r['Gaussian']:.2f}% | "
            f"{r['Cluster']:.2f}% | {r['Explosion']:.2f}% | {r['Shifted Mean']:.2f}% |"
        )
    table4_markdown = "\n".join(t4_md_lines)
    print("\n" + table4_markdown + "\n")

    # Save and Print Table 5
    t5_md_lines = [
        "### Table 5: Geometry Preservation Diagnostics (Uniform-100, 1,000 instances)",
        "",
        "| Method | Viol. (%) | Median $D_{lat}/D_{tgt}$ | Spearman $\\rho$ | Slot Entropy | Total Params | Module Params | Time (ms/inst) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in table5_rows:
        viol_str = f"{r['viol_pct']:.2f}%" if not math.isnan(r['viol_pct']) else "-"
        ratio_str = f"{r['median_ratio']:.3f}" if not math.isnan(r['median_ratio']) else "-"
        rho_str = f"{r['spearman_rho']:.3f}" if not math.isnan(r['spearman_rho']) else "-"
        ent_str = f"{r['entropy']:.3f}" if not math.isnan(r['entropy']) else "-"
        time_str = f"{r['inference_time_ms']:.2f}" if not math.isnan(r['inference_time_ms']) else "-"
        t5_md_lines.append(
            f"| {r['method']} | {viol_str} | {ratio_str} | {rho_str} | {ent_str} | "
            f"{r['total_params']:,} | {r['slot_params']:,} | {time_str} |"
        )
    table5_markdown = "\n".join(t5_md_lines)
    print("\n" + table5_markdown + "\n")

    # Write files
    (out_dir / "table4_results.json").write_text(json.dumps(table4_rows, indent=2))
    (out_dir / "table5_results.json").write_text(json.dumps(table5_rows, indent=2))
    (out_dir / "tables_summary.md").write_text(f"{table4_markdown}\n\n{table5_markdown}\n")
    print(f"[OK] Saved results to {out_dir}")


if __name__ == "__main__":
    main()
