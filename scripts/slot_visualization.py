from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def extract(model, locs) -> dict:

    import tensordict  # noqa: F401  (required on the training machine)

    from tensordict import TensorDict

    env = model.env
    device = next(model.parameters()).device

    # Build a small batch tensor dict with the customer locs (B, N, 2).
    B = 1
    N = locs.shape[0]
    td = TensorDict({"locs": locs.unsqueeze(0)}, batch_size=[B]).to(device)
    td = env.reset(td)

    with torch.no_grad():
        model.policy(td, env, phase="train", num_starts=None)

    slots = model.policy.encoder.last_slots   # (B, K, d)
    A_ik = model.policy.encoder.last_A_ik     # (B, N, K)
    if slots is None or A_ik is None:
        raise RuntimeError("Encoder side-channel was None — is the policy a SlotInjectingEncoder?")

    slots = slots[0]        # (K, d)
    A = A_ik[0]             # (N_cust, K)
    return slots, A


def diagnostics(slots: torch.Tensor, A: torch.Tensor) -> dict:
    import torch.nn.functional as F

    K = slots.shape[0]
    D = torch.cdist(slots, slots)                       # (K, K) symmetric
    eye = torch.eye(K, dtype=torch.bool, device=D.device)
    off = D[~eye]

    usage = A.mean(dim=0)                               # (K,) marginal, sums to 1
    eps = 1e-12
    entropy = -(usage * torch.log(usage + eps)).clamp_min(0).sum()   # H[p(k)]
    perplexity = entropy.exp().item()                   # effective # of balanced slots
    logK = torch.log(torch.tensor(K, dtype=torch.float)).item()
    norm_entropy = entropy.item() / logK if logK > 0 else 0.0   # 1.0 = perfectly flat

    # Worst-pair representation-collapse metric (invariant to scale).
    n_max = off.max().item()
    norm_min = (off.min() / (n_max + eps)).item()       # ~0 -> at least one slot pair collapses
    norm_mean = (off.mean() / (n_max + eps)).item()

    return {
        "D": D.tolist(),
        "usage": usage.tolist(),
        "diag": {
            "min_offdiag": off.min().item(),
            "max_offdiag": off.max().item(),
            "mean_offdiag": off.mean().item(),
            "norm_min": norm_min,
            "norm_mean": norm_mean,
            "entropy": entropy.item(),
            "logK": logK,
            "norm_entropy": norm_entropy,
            "perplexity": perplexity,
            "eff_slots": perplexity,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to a POMOSlot checkpoint (.ckpt)")
    ap.add_argument("--num_locs", type=int, default=50)
    ap.add_argument("--num_slots", type=int, default=8)
    ap.add_argument("--out", default="slot_viz_data.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from rl4co.data.generate_slot_dataset import _gen_uniform  # customer locs generator
    from rl4co.envs import get_env
    from rl4co.models.zoo.pomo_slot import POMOSlot

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    env = get_env("tsp")
    model = POMOSlot(env, num_slots=args.num_slots, metric_variant="D")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # Customer locations, no depot (matches A_ik indexing: N_cust = num_locs - 1).
    locs = _gen_uniform(1, args.num_locs)[0]           # (num_locs, 2)
    locs_customers = locs[1:]                           # drop node 0 (pseudo-depot)

    with torch.no_grad():
        slots, A = extract(model, locs_customers)

    diag = diagnostics(slots, A)
    data = {
        "locs": locs_customers.tolist(),
        "slots": slots.tolist(),
        "A": A.tolist(),
        **diag,
        "log": {
            "num_locs": args.num_locs,
            "num_slots": args.num_slots,
            "ckpt": args.ckpt,
        },
    }
    Path(args.out).write_text(json.dumps(data))
    print(f"Wrote {args.out}  (K={args.num_slots}, N_cust={locs_customers.shape[0]})")
    print(f"  norm_min_offdiag={diag['diag']['norm_min']:.4f}  "
          f"perplexity={diag['diag']['perplexity']:.2f}  "
          f"norm_entropy={diag['diag']['norm_entropy']:.3f}")


if __name__ == "__main__":
    main()
