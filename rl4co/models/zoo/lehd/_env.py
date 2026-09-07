"""LEHD CVRP Environment (subpath-training variant).

A self-contained, device-agnostic re-implementation of the original
NCO_code/single_objective/LEHD/CVRP/VRPEnv.py.

Key responsibilities:
  1. Load the LEHD-format training dataset (.txt with depot/customer/demand/
     capacity/cost/node_flag columns).
  2. Per-batch, sample a random *subpath* from the HGS-optimal solution so
     the model learns to reconstruct partial routes (the LEHD curriculum).
  3. Expose load_raw_data / shuffle_data / load_problems / reset / pre_step /
     step exactly as the original, so LEHDModel can drive training in a
     Lightning training_step without coupling to VRPTrainer.

Coordinate / tensor conventions (matching original):
  problems  : (B, V+1, 4)  — index 0 = depot; cols = [x, y, demand, capacity]
  solution  : (B, V,   2)  — each row = [customer_idx_1based, flag_via_depot]
  capacity  : scalar int (same for all instances in one dataset)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class StepState:
    problems: torch.Tensor   # (B, V+1, 4)


class LEHDVRPEnv:
    """Manages one-epoch data flow for LEHD CVRP training.

    Usage (mirrors original VRPTrainer loop)::

        env = LEHDVRPEnv(data_path=..., mode='train', sub_path=True)
        env.load_raw_data(n_episodes)
        for epoch in range(epochs):
            env.shuffle_data()
            episode = 0
            while episode < n_episodes:
                env.load_problems(episode, batch_size)
                env.reset('train')
                state, _, _, done = env.pre_step()
                step = 0
                while not done:
                    ...  # model predicts; call env.step(...)
                    step += 1
                episode += batch_size
    """

    def __init__(self, data_path: str, mode: str = "train", sub_path: bool = True):
        self.data_path = data_path
        self.mode = mode
        self.sub_path = sub_path

        # Raw tensors populated by load_raw_data
        self.raw_data_nodes: Optional[torch.Tensor] = None     # (N, V+1, 2)
        self.raw_data_demand: Optional[torch.Tensor] = None    # (N, V+1)
        self.raw_data_capacity: Optional[torch.Tensor] = None  # (N,)
        self.raw_data_cost: Optional[torch.Tensor] = None      # (N,)
        self.raw_data_node_flag: Optional[torch.Tensor] = None # (N, V, 2)

        # Per-batch tensors populated by load_problems
        self.problems: Optional[torch.Tensor] = None   # (B, V+1, 4)
        self.solution: Optional[torch.Tensor] = None   # (B, V, 2)
        self.batch_size: Optional[int] = None
        self.problem_size: Optional[int] = None

        # Step state
        self.selected_count = 0
        self.selected_node_list: Optional[torch.Tensor] = None
        self.selected_teacher_flag: Optional[torch.Tensor] = None
        self.selected_student_list: Optional[torch.Tensor] = None
        self.selected_student_flag: Optional[torch.Tensor] = None
        self.step_state: Optional[StepState] = None
        self.satisfy_demand: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_raw_data(self, n_episodes: int = 1_000_000) -> None:
        """Parse LEHD-format .txt dataset.  Loads up to n_episodes lines."""
        nodes_list, demand_list, cap_list, cost_list, flag_list = [], [], [], [], []

        def _two_col(node_flag):
            V = len(node_flag) // 2
            return [[node_flag[i], node_flag[V + i]] for i in range(V)]

        with open(self.data_path, "r") as f:
            lines = f.readlines()

        # For training LEHD reads 2×0.5M in two chunks; we simplify to one pass.
        for line in lines[:n_episodes]:
            tok = line.split(",")
            di = tok.index("depot")
            ci = tok.index("customer")
            qi = tok.index("capacity")
            ri = tok.index("demand")
            oi = tok.index("cost")
            fi = tok.index("node_flag")

            depot = [[float(tok[di + 1]), float(tok[di + 2])]]
            customers = [
                [float(tok[k]), float(tok[k + 1])]
                for k in range(ci + 1, qi, 2)
            ]
            loc = depot + customers

            cap = int(float(tok[qi + 1]))
            if int(tok[ri + 1]) == 0:
                demand = [int(tok[k]) for k in range(ri + 1, oi)]
            else:
                demand = [0] + [int(tok[k]) for k in range(ri + 1, oi)]
            cost = float(tok[oi + 1])
            node_flag = [int(tok[k]) for k in range(fi + 1, len(tok))]
            node_flag = _two_col(node_flag)

            nodes_list.append(loc)
            demand_list.append(demand)
            cap_list.append(cap)
            cost_list.append(cost)
            flag_list.append(node_flag)

        self.raw_data_nodes = torch.tensor(nodes_list, dtype=torch.float32)
        self.raw_data_demand = torch.tensor(demand_list, dtype=torch.float32)
        self.raw_data_capacity = torch.tensor(cap_list, dtype=torch.float32)
        self.raw_data_cost = torch.tensor(cost_list, dtype=torch.float32)
        self.raw_data_node_flag = torch.tensor(flag_list, dtype=torch.long)
        print(f"[LEHDVRPEnv] Loaded {len(self.raw_data_nodes)} instances from {self.data_path}")

    def shuffle_data(self) -> None:
        idx = torch.randperm(len(self.raw_data_nodes))
        self.raw_data_nodes = self.raw_data_nodes[idx]
        self.raw_data_demand = self.raw_data_demand[idx]
        self.raw_data_capacity = self.raw_data_capacity[idx]
        self.raw_data_cost = self.raw_data_cost[idx]
        self.raw_data_node_flag = self.raw_data_node_flag[idx]

    # ------------------------------------------------------------------
    # Per-batch setup
    # ------------------------------------------------------------------

    def load_problems(self, episode: int, batch_size: int) -> None:
        self.batch_size = batch_size
        nodes = self.raw_data_nodes[episode: episode + batch_size]       # (B, V+1, 2)
        demand = self.raw_data_demand[episode: episode + batch_size]     # (B, V+1)
        cap = self.raw_data_capacity[episode: episode + batch_size]      # (B,)
        solution = self.raw_data_node_flag[episode: episode + batch_size].float()  # (B, V, 2)

        # capacity replicated across V+1 positions → shape (B, V+1)
        cap_exp = cap[:, None].expand(-1, solution.shape[1] + 1)  # (B, V+1)

        # problems: (B, V+1, 4)  [x, y, demand, remaining_capacity]
        problems = torch.cat(
            [nodes, demand[:, :, None], cap_exp[:, :, None]], dim=2
        )

        if self.sub_path:
            problems, solution = self._sample_subpath(problems, solution.long())

        self.problems = problems
        self.solution = solution.long()
        self.problem_size = self.problems.shape[1] - 1

    # ------------------------------------------------------------------
    # Subpath sampling (ported from original sampling_subpaths)
    # ------------------------------------------------------------------

    def _sample_subpath(
        self, problems: torch.Tensor, solution: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random sub-route from the HGS solution for curriculum training."""
        B, V_plus1, _ = problems.shape
        V = solution.shape[1]
        device = problems.device

        length_sub = int(torch.randint(4, V + 1, (1,)).item())

        # Random shift/flip augmentation
        solution = self._random_inverse(solution)
        solution = self._random_shift_inverse(solution)

        # Find tour end-points (positions where flag=1, i.e., next visit is from depot)
        visit_depot_num = solution[:, :, 1].sum(dim=1)   # (B,) number of sub-tours
        start_from_depot = solution[:, :, 1].nonzero()   # rows: [batch_idx, position]

        end_with_depot = start_from_depot.clone()
        end_with_depot[:, 1] -= 1
        end_with_depot[end_with_depot[:, 1] < 0, 1] = V - 1

        # Pick one random sub-tour endpoint per instance
        p = torch.rand(B, device=device)
        select_idx = (p * visit_depot_num.float()).long()

        temp_tri = torch.from_numpy(
            np.triu(np.ones((B, B)), k=1)
        ).float().to(device)
        cum_offset = (visit_depot_num.float().unsqueeze(0) @ temp_tri).long().squeeze(0)
        pick_idx = select_idx + cum_offset
        # Safety clamp
        pick_idx = pick_idx.clamp(0, end_with_depot.shape[0] - 1)
        end_nodes = end_with_depot[pick_idx, 1]  # (B,) 0-indexed position in solution

        double_sol = torch.cat([solution, solution], dim=1)  # (B, 2V, 2)
        end_nodes = end_nodes + V  # shift into doubled range

        indices = (
            torch.arange(length_sub, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        offset = end_nodes - length_sub + 1
        full_idx = indices + offset.unsqueeze(1)  # (B, L)

        batch_idx = torch.arange(B, device=device)[:, None, None].expand(B, length_sub, 2)
        col_idx = full_idx.unsqueeze(2).expand(B, length_sub, 2)
        sub_solution = double_sol[batch_idx, col_idx, torch.arange(2, device=device)]  # (B, L, 2)

        # ---- Adjust remaining capacity for the subpath start ----
        start_pos = full_idx[:, 0]  # (B,) position in doubled solution
        x1 = (
            torch.arange(2 * V, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        ) <= start_pos.unsqueeze(1)

        before_depot_mask = double_sol[:, :, 1].float() * x1.float()
        visit_depot_before = before_depot_mask.sum(dim=1).long()  # (B,)
        # cumulative offset for indexing into start_from_depot
        cum2 = (visit_depot_before.float().unsqueeze(0) @ temp_tri).long().squeeze(0)
        pick2_idx = (visit_depot_before - 1 + cum2).clamp(0, start_from_depot.shape[0] - 1)
        last_depot_pos = start_from_depot[pick2_idx, 1]  # (B,)

        x2 = torch.arange(2 * V, device=device).unsqueeze(0) < start_pos.unsqueeze(1)
        x3 = torch.arange(2 * V, device=device).unsqueeze(0) >= last_depot_pos.unsqueeze(1)
        between = (x2 & x3).float()

        # demand of visited nodes before subpath start (using doubled solution node indices)
        demand = problems[:, :, 2]  # (B, V+1) original demand
        double_node_idx = double_sol[:, :, 0].long()  # (B, 2V) node indices (1-based)
        # gather demand from original problems (node 0 = depot, demand=0; nodes 1..V = customers)
        gathered_demand = demand[
            torch.arange(B, device=device)[:, None].expand(B, 2 * V),
            double_node_idx.clamp(0, V)
        ]
        satisfy_demand = (gathered_demand * between).sum(dim=1)  # (B,)
        problems[:, :, 3] -= satisfy_demand.unsqueeze(1)

        # ---- Re-index sub_solution to local 1..L ordering ----
        node_col = sub_solution[:, :, 0]                         # (B, L)
        _, rank = torch.sort(node_col, dim=1)
        _, rerank = torch.sort(rank, dim=1)
        sub_solution = sub_solution.clone()
        sub_solution[:, :, 0] = rerank + 1  # 1-indexed

        # ---- Select corresponding problem coordinates ----
        sorted_original, _ = node_col.sort(dim=1)  # (B, L) ascending original indices
        # replicated 4 times for the 4 feature columns
        idx2 = sorted_original.repeat(1, 4).long()  # (B, 4L)
        idx1 = torch.arange(B, device=device)[:, None].expand(B, 4 * length_sub)
        idx3 = torch.arange(4, device=device)[None, :].expand(B, 4).repeat(1, length_sub)
        new_data = problems[idx1, idx2, idx3].view(B, length_sub, 4)
        new_data = torch.cat([problems[:, [0], :], new_data], dim=1)  # prepend depot

        return new_data, sub_solution

    # ------------------------------------------------------------------
    # Augmentation helpers (ported from original)
    # ------------------------------------------------------------------

    def _random_inverse(self, solution: torch.Tensor) -> torch.Tensor:
        """Randomly reverse each instance's tour (50% chance)."""
        if torch.rand(1).item() >= 0.5:
            solution = torch.flip(solution, dims=[1])
            idx = torch.arange(solution.shape[1]).roll(1)
            solution[:, :, 1] = solution[:, idx, 1]
        return solution

    def _random_shift_inverse(self, solution: torch.Tensor) -> torch.Tensor:
        """Randomly rotate the starting sub-tour."""
        B, V, _ = solution.shape
        device = solution.device
        visit_depot_num = solution[:, :, 1].sum(dim=1)  # (B,)
        if visit_depot_num.min().item() < 1:
            return solution
        min_len = int(visit_depot_num.min().item())
        first_idx = int(torch.randint(0, min_len, (1,)).item())

        start_from_depot = solution[:, :, 1].nonzero()
        end_with_depot = start_from_depot.clone()
        end_with_depot[:, 1] -= 1
        end_with_depot[end_with_depot[:, 1] < 0, 1] = V - 1
        end_with_depot[:, 1] = end_with_depot[:, 1].roll(-1)

        temp_tri = torch.from_numpy(np.triu(np.ones((B, B)), k=1)).float().to(device)
        cum = (visit_depot_num.float().unsqueeze(0) @ temp_tri).long().squeeze(0)
        pick_idx = (cum + first_idx).clamp(0, end_with_depot.shape[0] - 1)
        first_end = end_with_depot[pick_idx, 1]  # (B,)

        double_sol = torch.cat([solution, solution], dim=1)
        end_nodes = first_end + V
        indices = torch.arange(V, device=device).unsqueeze(0).expand(B, -1)
        full_idx = indices + (end_nodes - V + 1).unsqueeze(1)
        batch_idx = torch.arange(B, device=device)[:, None, None].expand(B, V, 2)
        col_idx = full_idx.unsqueeze(2).expand(B, V, 2)
        return double_sol[batch_idx, col_idx, torch.arange(2, device=device)]

    # ------------------------------------------------------------------
    # RL step interface
    # ------------------------------------------------------------------

    def reset(self, mode: str = "train") -> Tuple[StepState, None, None]:
        assert self.problems is not None
        self.selected_count = 0
        B = self.problems.shape[0]
        dev = self.problems.device
        self.selected_node_list = torch.zeros(B, 0, dtype=torch.long, device=dev)
        self.selected_teacher_flag = torch.zeros(B, 0, dtype=torch.long, device=dev)
        self.selected_student_list = torch.zeros(B, 0, dtype=torch.long, device=dev)
        self.selected_student_flag = torch.zeros(B, 0, dtype=torch.long, device=dev)
        self.step_state = StepState(problems=self.problems)
        return self.step_state, None, None

    def pre_step(self) -> Tuple[StepState, None, None, bool]:
        return self.step_state, None, None, False

    def step(
        self,
        selected: torch.Tensor,
        selected_student: torch.Tensor,
        selected_flag_teacher: torch.Tensor,
        selected_flag_student: torch.Tensor,
    ) -> Tuple[StepState, Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """Advance the environment by one decoding step.

        Args:
            selected / selected_flag_teacher: teacher node + flag (1-indexed; flag=1 → via depot)
            selected_student / selected_flag_student: model's greedy prediction (unused during training)
        """
        self.selected_count += 1
        B = self.problems.shape[0]
        dev = self.problems.device
        capacity = float(self.raw_data_capacity[0].item())

        # ---- Update remaining capacity in problems[:, :, 3] ----
        is_depot = selected_flag_teacher == 1
        self.problems[is_depot, :, 3] = capacity

        gather_idx = selected[:, None, None].expand(B, 1, 4)
        current = self.problems.gather(1, gather_idx).squeeze(1)  # (B, 4)
        demands = current[:, 2]

        insufficient = self.problems[:, 0, 3] < demands
        selected_flag_teacher = selected_flag_teacher.clone()
        selected_flag_teacher[insufficient] = 1
        self.problems[insufficient, :, 3] = capacity

        self.problems[:, :, 3] -= demands[:, None]

        # ---- Append to trajectory lists ----
        self.selected_node_list = torch.cat(
            [self.selected_node_list, selected[:, None]], dim=1
        )
        self.selected_teacher_flag = torch.cat(
            [self.selected_teacher_flag, selected_flag_teacher[:, None]], dim=1
        )
        self.selected_student_list = torch.cat(
            [self.selected_student_list, selected_student[:, None]], dim=1
        )
        self.selected_student_flag = torch.cat(
            [self.selected_student_flag, selected_flag_student[:, None]], dim=1
        )

        done = self.selected_count == (self.problems.shape[1] - 1)
        reward = self._get_travel_distance() if done else (None, None)
        r_teach, r_stud = reward if isinstance(reward, tuple) else (None, None)
        return self.step_state, r_teach, r_stud, done

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _cal_length(
        self, xy: torch.Tensor, order_node: torch.Tensor, order_flag: torch.Tensor
    ) -> torch.Tensor:
        """Tour length for a batch of solutions."""
        V = order_node.shape[1]
        flag = order_flag.clone()
        flag_next = flag.clone()
        flag_next[:, 0] = 0

        # For each step: distance to next node if continuing, or depot+customer if flag=1
        node_idx = order_node
        roll_idx = order_node.roll(1, dims=1)

        def _gather(locs, idx):
            return locs.gather(1, idx.unsqueeze(2).expand(-1, V, 2))

        depot_loc = xy[:, [0], :].expand(-1, V, -1)
        # Positions in problems: node i is at index i in xy (0=depot, 1..V=customers)
        order_loc = _gather(xy, node_idx)
        roll_loc = _gather(xy, roll_idx)
        flag_loc = flag.unsqueeze(2).expand(-1, V, 2)
        # When flag=1, previous leg ends at depot rather than previous node
        leg_to_depot = torch.where(
            flag.bool().unsqueeze(2).expand(-1, V, 2),
            depot_loc, order_loc
        )
        flag_loc2 = flag_next.unsqueeze(2).expand(-1, V, 2)
        leg_from_depot = torch.where(
            flag_next.bool().unsqueeze(2).expand(-1, V, 2),
            depot_loc, roll_loc
        )
        lengths = (
            (order_loc - leg_to_depot).pow(2).sum(2).sqrt()
            + (roll_loc - leg_from_depot).pow(2).sum(2).sqrt()
        ).sum(1)
        return lengths

    def _get_travel_distance(self) -> Tuple[torch.Tensor, torch.Tensor]:
        xy = self.problems[:, :, :2]
        teach_len = self._cal_length(xy, self.solution[:, :, 0], self.solution[:, :, 1])
        stud_len = self._cal_length(xy, self.selected_student_list, self.selected_student_flag)
        return -teach_len, -stud_len
