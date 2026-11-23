"""Profile RADAR's per-stage wall-clock and decoder-step count across instance
distributions, to test whether a 200x Gaussian-vs-Explosion slowdown comes from
(a) the SVD encoding stage or (b) the autoregressive decode loop (and its step
count under Sinkhorn attention).

Measures five quantities per distribution / size / batch:
  * svd_encode_sec  : wall-clock of ``prepare`` (cdist + svd_lowrank + encoder)
  * decode_sec      : wall-clock of the whole decode loop
  * max_steps       : decoder iterations until every POMO row finished
  * mean_steps      : mean per-instance number of decode steps (guessed by
                      tracking the finish horizon; batch advances to the max)
  * total_sec       : svd_encode_sec + decode_sec

Usage:
  python scripts/profile_radar_dist.py \
      --dists uniform,gaussian,exponential \
      --sizes 500 --n_inst 8 --batch_size 8 --pomo 16 \
      --svd_rank 10 --sinkhorn_iters 10 --device cpu

Not tied to a trained checkpoint: any model (incl. untrained) reproduces the
per-step *cost*; only the absolute step-count quality of a trained model shifts
values, so run with --ckpt if you have one for the behavioral arm.
"""

from __future__ import annotations

import argparse
import time

import torch

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.baseline_cvrp import CVRPState
from rl4co.models.zoo.radar import RADAR


def make_env(num_loc, loc_dist, device):
    """Build a CVRPEnv whose generator samples the requested loc distribution."""
    gp = {"num_loc": num_loc, "loc_distribution": loc_dist,
          "loc_mean": 0.5, "loc_std": 0.2}
    if loc_dist in ("exponential", "poisson"):
        gp["loc_rate"] = 3.0          # mean 1/3 within [0,1]
    if loc_dist == "gaussian_mixture":
        gp["num_modes"] = 8
        gp["cdist"] = 1.0
    return CVRPEnv(generator_params=gp)


def make_batch(env, n_inst, num_loc, device):
    """Sample a batch via the env's generator (same TensorDict layout as eval)."""
    td = env.generator([n_inst])
    return {
        "locs": td["locs"].to(device),
        "depot": td["depot"].to(device),
        "demand": td["demand"].to(device).float(),
        "capacity": td["capacity"].to(device).float() if "capacity" in td
                    else torch.ones(n_inst, 1, device=device),
    }


def measured_rollout(policy, batch, width):
    """Instrumented copy of baseline_cvrp.rollout; counts steps per instance
    via the policy's prof counters. Returns (reward, max_steps, mean_steps,
    svd_sec, decode_sec)."""
    import torch as _t

    assert policy.prof_enabled, "policy.prof_enabled must be True"
    state = CVRPState.from_batch(batch, width=min(width, batch["locs"].size(1)))
    b, w = state.current.shape
    depot = _t.zeros(b, w, dtype=_t.long, device=state.xy.device)
    state.actions.append(depot)

    policy.prof_steps = 0
    policy.prof_svd_sec = 0.0
    t0 = time.perf_counter()
    policy.prepare(state)
    svd_sec = policy.prof_svd_sec

    starts = _t.arange(1, w + 1, device=state.xy.device)
    selected = starts[None].expand(b, w)
    state.step(selected)

    t1 = time.perf_counter()
    while not bool(state.finished.all()):
        logits = policy.logits(state)
        logits = logits.masked_fill(state.mask(), float("-inf"))
        selected = logits.argmax(-1)
        state.step(selected)
    decode_sec = time.perf_counter() - t1

    # Recompute the step horizon per instance from the recorded actions.
    reward = state.reward()
    actions = _t.stack(state.actions, -1)          # (B, W, T) with leading depot
    # Each row finishes at the depot after its last customer; the shared rollout
    # ends when the slowest row finishes. Per-row decode steps = index of the
    # last customer row.
    T = actions.shape[-1]
    acts = actions[:, :, 1:]                        # drop forced leading depot
    nz = acts != 0
    idx = _t.arange(T - 1, device=acts.device)
    last_nonzero = (nz * idx).max(-1).values        # (B, W)
    per_row = last_nonzero + 1                      # steps until last customer
    per_inst = per_row.max(-1).values.float()       # (B,)
    max_steps = int(per_inst.max())
    mean_steps = float(per_inst.mean())
    return -reward.max(-1).values, max_steps, mean_steps, svd_sec, decode_sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dists", default="uniform,gaussian,exponential")
    p.add_argument("--sizes", default="100,200,500")
    p.add_argument("--n_inst", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--pomo", type=int, default=16)
    p.add_argument("--svd_rank", type=int, default=10)
    p.add_argument("--sinkhorn_iters", type=int, default=10)
    p.add_argument("--ckpt", default=None, help="Optional trained RADAR ckpt.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'dist':<12}{'n':>6}{'batch':>6}{'svd_s':>10}{'decode_s':>10}"
          f"{'step_max':>10}{'step_mean':>10}{'total_s':>10}")
    for dist in args.dists.split(","):
        for n in (int(s) for s in args.sizes.split(",")):
            env = make_env(n, dist, device)
            model = RADAR(env=env, embed_dim=64, svd_rank=args.svd_rank,
                          sinkhorn_iters=args.sinkhorn_iters, pomo_size=args.pomo)
            if args.ckpt:
                model = RADAR.load_from_checkpoint(args.ckpt, env=env)
            model = model.to(device).eval()
            model.policy.prof_enabled = True
            batch = make_batch(env, args.n_inst, n, device)

            # accumulate over batches
            tot_svd = tot_decode = 0.0
            steps_max = []
            steps_mean = []
            n_eff = 0
            with torch.no_grad():
                for i in range(0, args.n_inst, args.batch_size):
                    sub = {k: v[i:i + args.batch_size] for k, v in batch.items()}
                    rew, mx, mn, svd_s, dec_s = measured_rollout(model.policy, sub, args.pomo)
                    tot_svd += svd_s; tot_decode += dec_s
                    steps_max.append(mx); steps_mean.append(mn)
                    n_eff += rew.numel()
            smx = float(max(steps_max)); smn = sum(steps_mean) / len(steps_mean)
            print(f"{dist:<12}{n:>6}{n_eff:>6}{tot_svd:>10.3f}{tot_decode:>10.3f}"
                  f"{smx:>10.1f}{smn:>10.1f}{tot_svd + tot_decode:>10.3f}")

    print("\nInterpretation:")
    print("  svd_encode_sec flat across dists       -> SVD is NOT the culprit.")
    print("  decode_sec / step counts rise w/ gaussian -> Sinkhorn decode behavior is.")


if __name__ == "__main__":
    main()
