from __future__ import annotations

import torch
from tensordict import TensorDict

from rl4co.models.zoo.pomo_slot.model_am import SingleSharedBaseline
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.decoding import get_decoding_strategy
from rl4co.utils.ops import batchify, gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


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


def apply_local_search(
    env, td_reset, actions, num_starts
):
    """Run env.local_search (HGS SWAP*) over a decoded batch and return improved
    tour lengths per instance, reduced the same way the reward is.

    ``actions`` comes back from the autogressive decoder interleaved by start
    (the ``batchify`` layout: [inst0_s0, inst1_s0, ..., inst0_s1, ...]), with
    num_starts copies of each base instance. env.local_search expects a flat
    (B, seq) batch padded with depot(0) at the head/tail and returns the same.

    To keep the comparison with the neural-only number exact, we run the
    improvement on the full (batchified) batch and reduce via the same
    reshape(B, num_starts).max(dim=1) used for the reward.

    Args:
        env: CVRPEnv (must have a working local_search implementation)
        td_reset: TensorDict post env.reset(batch), depot-first locs, matching
            the batch the decoder ran on ([B, ...]).
        actions: torch.Tensor actions from the decoder, [B*num_starts, seq],
            interleaved by start.
        num_starts: number of starts per instance (reward width).
    Returns:
        torch.Tensor [batch_size,] improved tour length per instance (or None
        if local search is unavailable).
    """
    try:
        ls = getattr(env, "local_search", None)
        if ls is None:
            return None
        B = td_reset.batch_size[0]
        # Expand the td to the interleaved [B*ns, ...] layout the decoder used,
        # so its locs/demand align with the action rows.
        td_ls = batchify(td_reset, num_starts)
        improved = ls(td_ls, actions)
        if improved is None:
            return None
        improved = improved.view(B, num_starts, -1)
        locs = td_reset["locs"]  # [B, N+1, 2], depot-first
        ordered = torch.cat(
            [
                locs[:, :1][:, None].expand(B, num_starts, 1, 2),
                gather_by_index(locs, improved, dim=1),
            ],
            dim=1,
        )
        lens = get_tour_length(ordered)  # [B, num_starts]
        return lens.max(dim=1).values
    except Exception as e:
        log.warning(f"local_search unavailable, skipping: {e}")
        return None


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


def evaluate(model, args, env, ds, return_instances: int | None = None):
    model.eval()
    dec = REGISTRY[args.model]
    rewards = []
    local_search_lens = []
    num_starts = 1
    # When plotting is requested, keep the first ``return_instances`` decoded
    # solutions (post-reset td with depot + actions) so routes can be drawn.
    plot_instances = [] if return_instances else None
    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in batch_iter(ds, args.batch_size):
            if isinstance(batch, dict):
                batch = TensorDict(batch, batch_size=[batch["demand"].size(0)])
            batch = batch.to(args.device)
            reward_batch, num_starts, actions = dec(model, batch, args, env)
            rewards.append(reward_batch.cpu())
            if getattr(args, "local_search", False) and actions is not None:
                # Reset the (depot-first) td for the full batch, then improve and
                # reduce per-instance identically to the reward path.
                td_reset = env.reset(batch)
                ls_len = apply_local_search(env, td_reset, actions, num_starts)
                if ls_len is not None:
                    local_search_lens.append(ls_len.cpu())
            if plot_instances is not None and actions is not None:
                B = batch.batch_size[0]
                # Reconstruct the reset td (depot-first locs) for the batch so
                # per-start tour lengths can be recomputed for the best route.
                td_reset = env.reset(batch)
                for i in range(B):
                    if len(plot_instances) >= return_instances:
                        break
                    plot_instances.append({
                        "td": td_reset[i:i + 1].cpu(),
                        "actions": actions[i].cpu(),
                    })
                if len(plot_instances) >= return_instances:
                    break
    elapsed = time.perf_counter() - t0
    reward = torch.cat(rewards)
    tour_len = -reward
    result = {
        "n_inst": len(reward),
        "mean_reward": float(reward.mean()),
        "mean_tour_length": float(tour_len.mean()),
        "std_tour_length": float(tour_len.std()) if len(tour_len) > 1 else 0.0,
        "num_starts": num_starts,
        "elapsed_seconds": elapsed,
        "throughput_per_sec": len(reward) / elapsed if elapsed > 0 else 0.0,
    }
    if local_search_lens:
        ls_all = torch.cat(local_search_lens)
        result["mean_tour_length_local_search"] = float(ls_all.mean())
        result["std_tour_length_local_search"] = (
            float(ls_all.std()) if len(ls_all) > 1 else 0.0
        )
    if return_instances is not None:
        return result, plot_instances
    return result
