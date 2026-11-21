import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.cvrp.render import render
from rl4co.utils.ops import gather_by_index

from .generator import OVRPGenerator


class OVRPEnv(CVRPEnv):
    """Open Vehicle Routing Problem (OVRP) environment.

    The Open VRP relaxes CVRP's "return to the depot" requirement: each route
    still *starts* at the depot (to serve a bounded capacity of customers) but
    no longer must travel back to the depot — the vehicle stops at the last
    customer it serves on that route. The cost objective therefore excludes the
    final return-to-depot arc of every route.

    Implementation note: the decision process (action masking, capacity reset at
    the depot, done condition) is inherited unchanged from CVRP. Only the reward
    differs — OVRPEnv drops the arcs whose destination is the depot, so the cost
    is the total *open* route length rather than the closed tour length.

    This is exactly the "depot-anchored NN tour works as is" transfer: the
    LEHD/LEGO geometry teacher (insertion cost / NN-tour over depots) is
    unchanged, and a CVRP-trained checkpoint loads into this environment because
    the init embedding stays on the CVRP ``(x, y, demand)`` node features
    (``env.name == "ovrp"`` maps to :class:`VRPInitEmbedding`).

    Constraints:
        - each route starts at the depot and is capacity-bounded.
        - each customer must be visited exactly once.
        - the vehicle does NOT need to return to the depot at the end.

    Reward:
        - (minus) the total length of the open (depot-to-last-customer) routes.
    """

    name = "ovrp"

    def __init__(
        self,
        generator: OVRPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = OVRPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _get_reward(self, td: TensorDict, actions: torch.Tensor):
        """Open-route reward: sum consecutive edges, dropping every arc whose
        destination is the depot (the final return-to-depot of each route).

        ``actions`` is the (B, T) action sequence (0 = depot, 1..N = customers).
        ``locs_ordered`` = [depot, node(actions[0]), ..., node(actions[T-1])] so
        arc ``i`` connects ordered[i] -> ordered[i+1], i.e. ends at
        ``node(actions[i])``. Arc ``i`` is part of an open route iff its
        destination is a customer (``actions[i] != 0``); return-to-depot arcs
        (``actions[i] == 0``) are excluded from the cost.
        """
        locs_ordered = torch.cat(
            [
                td["locs"][..., 0:1, :],          # depot
                gather_by_index(td["locs"], actions),  # ordered locations
            ],
            dim=1,                                  # (B, T+1, 2)
        )
        open_arcs = (locs_ordered[:, :-1]
                     - locs_ordered[:, 1:]).norm(dim=-1)   # (B, T)
        dest_is_customer = (actions != 0).to(open_arcs.dtype) if actions.ndim == 2 \
            else (actions != 0).unsqueeze(-1).to(open_arcs.dtype)
        # (B, T) — multiply per arc; broadcast when actions is (B, T)
        if actions.ndim == 1:
            dest_is_customer = dest_is_customer.expand_as(open_arcs)
        total_open = (open_arcs * dest_is_customer).sum(-1)
        return -total_open

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
