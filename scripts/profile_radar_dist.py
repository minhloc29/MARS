"""Profile RADAR's per-stage wall-clock and decode-step count across instance
distributions / spatial transforms.

Self-contained: times both stages directly with perf_counter (no in-policy prof
counters required), counts decode iterations per batch, and supports the
``alpha`` spatial transform from gen_data.py:

    x' = d + alpha*(x - d)      alpha < 1  implodes,  1 == no-op,  > 1 explodes
                                (clipped to [0,1]^2)

Measures per (dist, size, alpha):
  * svd_sec   : time to ``prepare`` (cdist + svd_lowrank + encoder)
  * decode_sec: time of the whole autoregressive decode loop
  * max_steps : decoder iterations until every POMO row finished (per batch max)
  * step_max / step_mean: per-instance finish horizon scaled to a batch
  * total_sec : svd_sec + decode_sec

Usage:
  python scripts/profile_radar_dist.py \
      --dists gaussian,uniform --sizes 100,500 \
      --alpha 1.0,3.0 --n_inst 64 --batch_size 16 --pomo 16 \
      --svd_rank 10 --sinkhorn_iters 10 --device cpu

Use --ckpt for trained weights; the untrained model reproduces per-step *cost*
but not the true behavioral step counts.
"""
from __future__ import annotations

import argparse
import time

import torch

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.baseline_cvrp import CVRPState
from rl4co.models.zoo.radar import RADAR


def make_batch(env, n_inst, num_loc, device, alpha=1.0):
    td = env.generator([n_inst])
    locs = td["locs"].to(device)
    if alpha != 1.0:
        d = locs[:, :1, :]                                   # depot row, kept fixed
        locs = (d + alpha * (locs - d)).clamp(0.0, 1.0)
    return {
        "locs": locs,
        "depot": td["depot"].to(device),
        "demand": td["demand"].to(device).float(),
        "capacity": td["capacity"].to(device).float()
                    if "capacity" in td else torch.ones(n_inst, 1, device=device),
    }


def measured_rollout(policy, batch, width):
    """Instrumented copy of baseline_cvrp.rollout; times stages + counts steps."""
    import torch as _t
    state = CVRPState.from_batch(batch, width=min(width, batch["locs"].size(1)))
    b, w = state.current.shape
    depot = _t.zeros(b, w, dtype=_t.long, device=state.xy.device)
    state.actions.append(depot)

    t0 = time.perf_counter()
    policy.prepare(state)
    svd_sec = time.perf_counter() - t0

    starts = _t.arange(1, w + 1, device=state.xy.device)
    selected = starts[None].expand(b, w)
    state.step(selected)

    steps = 0
    t1 = time.perf_counter()
    while not bool(state.finished.all()):
        logits = policy.logits(state)
        logits = logits.masked_fill(state.mask(), float("-inf"))
        selected = logits.argmax(-1)
        state.step(selected)
        steps += 1
        if steps > 200_000:
            raise RuntimeError("decode loop exceeded cap — likely a stalled instance")
    decode_sec = time.perf_counter() - t1

    per_inst = steps / w
    return (per_inst, steps, svd_sec, decode_sec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dists", default="uniform,gaussian")
    p.add_argument("--sizes", default="100,500")
    p.add_argument("--alpha", default="1.0,3.0",
                   help="comma list; >1 explodes (push away from depot)")
    p.add_argument("--n_inst", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pomo", type=int, default=16)
    p.add_argument("--svd_rank", type=int, default=10)
    p.add_argument("--sinkhorn_iters", type=int, default=10)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    alphas = [float(a) for a in args.alpha.split(",")]

    widths = {"38": "'%-10s'" % "", "default": ""}  # noqa
    print(f"{'dist':<12}{'n':>5}{'alpha':>6}{'batch':>6}{'svd_s':>9}"
          f"{'decode_s':>10}{'step_max':>10}{'total_s':>10}")
    for dist in args.dists.split(","):
        for n in (int(s) for s in args.sizes.split(",")):
            env = CVRPEnv(generator_params=dict(
                num_loc=n, loc_distribution=dist, loc_mean=0.5, loc_std=0.2,
                loc_input_rate=3.0))
            if dist in ("exponential", "poisson"):
                env = CVRPEnv(generator_params=dict(
                    num_loc=n, loc_distribution=dist, loc_rate=3.0))
            for alpha in alphas:
                model = RADAR(env=env, embed_dim=64, svd_rank=args.svd_rank,
                              sinkhorn_iters=args.sinkhorn_iters, pomo_size=args.pomo)
                if args.ckpt:
                    model = RADAR.load_from_checkpoint(args.ckpt, env=env)
                model = model.to(device).eval()
                batch = make_batch(env, args.n_inst, n, device, alpha=alpha)

                tot_svd = tot_decode = 0.0
                step_max = 0
                with torch.no_grad():
                    for i in range(0, args.n_inst, args.batch_size):
                        sub = {k: v[i:i + args.batch_size] for k, v in batch.items()}
                        pmin, st, svd_s, dec_s = measured_rollout(model.policy, sub, args.pomo)
                        tot_svd += svd_s; tot_decode += dec_s
                        step_max = max(step_max, st)
                print(f"{dist:<12}{n:>5}{alpha:>6}{args.n_inst:>6}"
                      f"{tot_svd:>9.3f}{tot_decode:>10.3f}{step_max:>10}{tot_svd + tot_decode:>10.3f}")

    print("\nInterpretation:")
    print("  svd_s flat across (dist, alpha)      -> SVD/encoder is NOT the culprit.")
    print("  decode_s / step_max rise w/ gaussian@alpha=1 vs exploded(alpha>1)")
    print("  -> decode-step count (loop stalls) is the driver of the inference gap.")


if __name__ == "__main__":
    main()
