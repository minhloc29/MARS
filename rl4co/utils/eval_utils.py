from __future__ import annotations

import torch
from tensordict import TensorDict

from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.decoding import get_decoding_strategy


def pick_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_npz(data_path: str, num_loc: int) -> TensorDict:
    td = load_npz_to_tensordict(data_path)
    n_rows = td["locs"].shape[-2]
    if n_rows == num_loc + 1:
        if "depot" not in td.keys():
            raise RuntimeError(f"{data_path}: locs has {n_rows} rows but no 'depot' key")
        td.set("locs", td["locs"][:, 1:])
    elif n_rows != num_loc:
        raise RuntimeError(
            f"{data_path}: num_loc mismatch — locs has {n_rows} rows, "
            f"expected {num_loc} customers"
        )
    return td


def batch_iter(ds, batch_size):
    n = ds.shape[0] if isinstance(ds, TensorDict) else len(ds)
    is_td = isinstance(ds, TensorDict)
    for i in range(0, n, batch_size):
        chunk = ds[i:i + batch_size]
        if not is_td:
            chunk = torch.stack(chunk)
        yield chunk


def check_num_loc(td, expected):
    if expected is None:
        return  # per-instance eval (e.g. CVRPLIB Set X) has no fixed size
    n_customers = int(td["locs"].shape[-2]) - 1
    if n_customers != expected:
        raise RuntimeError(
            f"num_loc mismatch: requested {expected} customers, got {n_customers}"
        )
        
        
def _allow_safe_globals() -> None:
    try:
        torch.serialization.add_safe_globals(
            [SingleSharedBaseline])
    except Exception:
        pass

def is_sampling(decode_type: str) -> bool:
    return decode_type in ("sampling", "multistart_sampling")


def greedy_name(decode_type: str) -> str:
    return "sampling" if is_sampling(decode_type) else "greedy"


def decode_reset(model, batch, args, env, num_starts):
    td = env.reset(batch)
    check_num_loc(td, args.num_loc)
    B = batch.batch_size[0]
    strategy = get_decoding_strategy(args.decode)
    kw = {"decode_type": strategy.name}
    if num_starts is not None:
        kw["num_starts"] = num_starts
    out = model.policy(td, env, phase="test", **kw)
    r = out["reward"]
    actions = out.get("actions", None)
    if args.augment:
        return r.reshape(8, B, -1).max(dim=-1).values, 1, actions
    ns = r.numel() // B
    return r.reshape(B, ns).max(dim=1).values, ns, actions


def decode_am(model, batch, args, env):
    return decode_reset(model, batch, args, env, args.num_starts or 1)


def decode_pomo(model, batch, args, env):
    return decode_reset(model, batch, args, env, args.num_starts)


def decode_sil(model, batch, args, env):
    td = env.reset(batch)
    check_num_loc(td, args.num_loc)
    out = model.policy(td, env, phase="test")
    return out["reward"].reshape(-1), 1, out.get("actions", None)


def decode_icam(model, batch, args, env):
    r, _ = model._rollout(batch, sampling=is_sampling(args.decode))
    # scalar-reward backbone: mark feasible by construction (no actions needed)
    return r.max(dim=1).values, 1, None


def decode_l2r(model, batch, args, env):
    r, _ = model._rollout(batch, sampling=is_sampling(args.decode))
    return r, 1, None


def decode_elg(model, batch, args, env):
    from rl4co.models.zoo.baseline_cvrp import rollout as elg_rollout

    width = model.hparams.pomo_size
    out = elg_rollout(model.policy, batch, width, greedy_name(args.decode), False)
    return out["reward"].max(dim=1).values, out["reward"].shape[1], out.get("actions", None)


def decode_invit(model, batch, args, env):
    out = model.policy(batch, phase="test", decode_type=greedy_name(args.decode))
    return out["reward"].reshape(-1), 1, out.get("actions", None)


def decode_radar(model, batch, args, env):
    from rl4co.models.zoo.baseline_cvrp import rollout as radar_rollout

    width = model.hparams.pomo_size
    out = radar_rollout(
        model.policy, batch, width, greedy_name(args.decode), False)
    return out["reward"].max(dim=1).values, out["reward"].shape[1], out.get("actions", None)


REGISTRY = {
    "am": decode_am,
    "pomo": decode_pomo,
    "pomo_base": decode_pomo,
    "sil": decode_sil,
    "icam": decode_icam,
    "l2r": decode_l2r,
    "elg": decode_elg,
    "invit": decode_invit,
    "radar": decode_radar,
}


def evaluate(model, args, env, ds):
    model.eval()
    dec = REGISTRY[args.model]
    rewards = []
    num_starts = 1
    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in batch_iter(ds, args.batch_size):
            if isinstance(batch, dict):
                batch = TensorDict(batch, batch_size=[batch["demand"].size(0)])
            batch = batch.to(args.device)
            reward_batch, num_starts, _ = dec(model, batch, args, env)
            rewards.append(reward_batch.cpu())
    elapsed = time.perf_counter() - t0
    reward = torch.cat(rewards)
    tour_len = -reward
    return {
        "n_inst": len(reward),
        "mean_reward": float(reward.mean()),
        "mean_tour_length": float(tour_len.mean()),
        "std_tour_length": float(tour_len.std()) if len(tour_len) > 1 else 0.0,
        "num_starts": num_starts,
        "elapsed_seconds": elapsed,
        "throughput_per_sec": len(reward) / elapsed if elapsed > 0 else 0.0,
    }
