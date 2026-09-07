"""RL4CO policy adapter for the upstream SIL CVRP decoder."""

import torch

from torch import nn

from ._network import CVRP_Decoder, CVRP_Encoder
from .solution import to_actions


class SILPolicy(nn.Module):
    def __init__(self, embed_dim=128, num_layers=6, num_heads=8, feedforward_hidden=512):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("SIL embed_dim must be divisible by num_heads")
        params = dict(
            embedding_dim=embed_dim,
            decoder_layer_num=num_layers,
            head_num=num_heads,
            qkv_dim=embed_dim // num_heads,
            ff_hidden_dim=feedforward_hidden,
            k_nearest_num=float("inf"),
        )
        self.encoder = CVRP_Encoder(**params)
        self.decoder = CVRP_Decoder(**params)

    @staticmethod
    def problems(locs, demand, remaining):
        depot_demand = demand.new_zeros(demand.size(0), 1)
        demands = torch.cat((depot_demand, demand), 1)
        return torch.cat(
            (locs, demands[..., None], remaining[:, None, None].expand(-1, locs.size(1), 1)), -1
        )

    def probabilities(self, problems, selected, encoded=None):
        # Upstream used a global CUDA default. Scope factory placement to this
        # call instead so CPU and any selected GPU work without global changes.
        with torch.device(problems.device):
            if encoded is None:
                encoded = self.encoder(problems, 1.0)
            return self.decoder(
                encoded, problems, selected, selected.size(1), 1.0, problems[:, 0, 3]
            )

    @staticmethod
    def advance(demand, remaining, node, flag):
        selected_demand = demand.gather(1, (node - 1)[:, None]).squeeze(1)
        # SIL predicts direct/via-depot jointly and forces a refill when needed.
        flag = flag.bool() | (selected_demand > remaining + 1e-5)
        remaining = torch.where(flag, torch.ones_like(remaining), remaining) - selected_demand
        return remaining, flag.long()

    def decode(self, locs, demand, first=None, remaining=None):
        batch, n = demand.shape
        if remaining is None:
            remaining = demand.new_ones(batch)
        selected = torch.empty(batch, 0, dtype=torch.long, device=locs.device)
        flags = []
        problems = self.problems(locs, demand, remaining)
        encoded = self.encoder(problems, 1.0)
        for step in range(n):
            if step == 0 and first is not None:
                node, flag = first.unbind(-1)
            else:
                probs = self.probabilities(problems, selected, encoded)
                action = probs.argmax(-1)
                node, flag = action % n + 1, (action >= n).long()
                if step == 0:
                    flag = torch.ones_like(flag)
            remaining, flag = self.advance(demand, remaining, node, flag)
            selected = torch.cat((selected, node[:, None]), 1)
            flags.append(flag)
            problems = self.problems(locs, demand, remaining)
        return torch.stack((selected, torch.stack(flags, 1)), -1)

    def forward(self, td, env, phase="test", num_starts=None, **kwargs):
        if num_starts not in (None, 1):
            raise ValueError("SIL uses a single greedy rollout; num_starts must be 1")
        solution = self.decode(td["locs"], td["demand"])
        actions = to_actions(solution)
        reward = env.get_reward(td, actions)
        return {"reward": reward, "actions": actions, "solution": solution}
