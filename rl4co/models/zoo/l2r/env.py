from dataclasses import dataclass
import math

import torch


@dataclass
class ResetState:
    depot_xy: torch.Tensor | None = None
    node_xy: torch.Tensor | None = None
    node_demand: torch.Tensor | None = None
    log_scale: float | None = None


@dataclass
class StepState:
    batch_size: int | None = None
    problem_size: int | None = None
    current_node: torch.Tensor | None = None
    selected_count: int = 0
    load: torch.Tensor | None = None
    ninf_mask: torch.Tensor | None = None
    finished: torch.Tensor | None = None
    upper_cur_dist: torch.Tensor | None = None
    upper_cur_ninf_mask: torch.Tensor | None = None
    upper_unvisited_index: torch.Tensor | None = None
    lower_xy: torch.Tensor | None = None
    lower_demand: torch.Tensor | None = None
    lower_neighbors_index: torch.Tensor | None = None
    lower_pairwise_dist: torch.Tensor | None = None
    lower_cur_ninf_mask: torch.Tensor | None = None
    neighbors_num_list: torch.Tensor | None = None


class CandidateReductionCVRPEnv:
    def __init__(self, *, device, problem_size, lower_neighbors_num, reduction_percentage):
        self.device = device
        self.problem_size = problem_size
        self.lower_neighbors_num = lower_neighbors_num
        self.reduction_percentage = reduction_percentage
        self.round_error_epsilon = 1e-5
        self.reset_state = ResetState()
        self.step_state = StepState()

    def load_problems(self, dataset):
        self.batch_size = dataset["node_xy"].shape[0]
        self.depot_node_xy = torch.cat(
            (dataset["depot_xy"], dataset["node_xy"]), dim=1)
        depot_demand = torch.zeros(self.batch_size, 1, device=self.device)
        self.depot_node_demand = torch.cat(
            (depot_demand, dataset["node_demand"]), dim=1)
        self.reset_state = ResetState(
            depot_xy=dataset["depot_xy"],
            node_xy=dataset["node_xy"],
            node_demand=dataset["node_demand"],
            log_scale=math.log2(self.problem_size),
        )

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(
            self.batch_size, 0, dtype=torch.long, device=self.device)
        self.load = torch.ones(self.batch_size, device=self.device)
        self.visited_ninf_flag = torch.zeros(
            self.batch_size, self.problem_size + 1, device=self.device)
        self.ninf_mask = self.visited_ninf_flag.clone()
        self.finished = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device)
        self.first_xy = self.reset_state.depot_xy
        self.cur_xy = None
        self.cur_dist = None
        self.cur_dist_clone = None
        self.cur_sorted_idx = None
        self.nearest_valid_nodes = None
        self.nearest_valid_distance = None
        return self.reset_state, None, False

    def pre_step(self):
        self._update_step_state()
        return self.step_state, None, False

    def _update_step_state(self):
        self.step_state.batch_size = self.batch_size
        self.step_state.problem_size = self.problem_size
        self.step_state.current_node = self.current_node
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected[:, None]), dim=1)
        self.cur_xy = self.depot_node_xy.gather(
            1, selected[:, None, None].expand(-1, 1, 2))
        self.cur_dist = torch.cdist(
            self.cur_xy, self.depot_node_xy, p=2, compute_mode="donot_use_mm_for_euclid_dist"
        ).squeeze(1)
        self.cur_dist_clone = self.cur_dist.clone()

        reduction_count = int(self.reduction_percentage * self.problem_size)
        if reduction_count > 0:
            farthest = self.cur_dist[:, 1:].argsort(
                dim=-1, descending=True)[:, :reduction_count] + 1
            self.cur_dist.scatter_(1, farthest, float("inf"))

        selected_demand = self.depot_node_demand.gather(
            1, selected[:, None]).squeeze(1)
        self.load -= selected_demand
        at_depot = selected == 0
        if torch.any(self.load < -self.round_error_epsilon):
            raise RuntimeError("CVRP capacity state became negative")
        self.load[at_depot] = 1

        current_mask = self.ninf_mask.gather(1, selected[:, None]).squeeze(1)
        if not torch.all(current_mask == 0):
            raise RuntimeError("CVRP selected a masked node")
        self.visited_ninf_flag.scatter_(1, selected[:, None], float("-inf"))
        self.visited_ninf_flag[:, 0][~at_depot] = 0
        self.ninf_mask = self.visited_ninf_flag.clone()
        too_large = self.load[:, None] + \
            self.round_error_epsilon < self.depot_node_demand
        self.ninf_mask[too_large] = float("-inf")
        newly_finished = (self.visited_ninf_flag == float("-inf")).all(dim=-1)
        self.finished = self.finished | newly_finished
        self.ninf_mask[:, 0][self.finished] = 0
        self.cur_dist[self.ninf_mask < 0] = float("inf")
        self.cur_dist_clone[self.ninf_mask < 0] = float("inf")
        self.cur_sorted_idx = self.cur_dist.argsort(dim=-1)
        nearest_dist, nearest_idx = self.cur_dist_clone.topk(
            1, dim=-1, largest=False)
        self.nearest_valid_nodes = nearest_idx.squeeze(1)
        self.nearest_valid_distance = nearest_dist.squeeze(1)
        self._update_step_state()
        reward = -self._travel_distance() if bool(self.finished.all()) else None
        return self.step_state, reward, bool(self.finished.all())

    def _travel_distance(self):
        index = self.selected_node_list.unsqueeze(2).expand(-1, -1, 2)
        ordered = self.depot_node_xy.gather(1, index)
        return ((ordered - ordered.roll(-1, dims=1)) ** 2).sum(2).sqrt().sum(1)

    def get_upper_input(self):
        if self.current_node is None:
            return self.step_state
        self.cur_valid_num = (self.cur_dist < 2).sum(dim=-1)
        all_masked = self.cur_valid_num == 0
        self.cur_valid_num[all_masked] = 1
        self.cur_sorted_idx[:,
                            0][all_masked] = self.nearest_valid_nodes[all_masked]
        max_valid = int(self.cur_valid_num.max())
        valid_index = self.cur_sorted_idx[:, :max_valid]
        positions = torch.arange(max_valid, device=self.device)[None, :]
        cur_dist = self.cur_dist.gather(1, valid_index)
        cur_dist[:, 0][all_masked] = self.nearest_valid_distance[all_masked]
        cur_dist[positions >= self.cur_valid_num[:, None]] = 2
        mask = self.ninf_mask.gather(1, valid_index)
        mask[positions >= self.cur_valid_num[:, None]] = float("-inf")
        self.step_state.upper_cur_dist = cur_dist[:, None]
        self.step_state.upper_unvisited_index = valid_index
        self.step_state.upper_cur_ninf_mask = mask[:, None]
        return self.step_state

    def update_cur_scores(self, upper_scores):
        self.nodes_score_whole = upper_scores + self.ninf_mask
        self.cur_sorted_idx = self.nodes_score_whole.argsort(
            dim=-1, descending=True)
        all_masked = torch.isinf(self.cur_dist).all(dim=-1)
        self.cur_sorted_idx[:,
                            0][all_masked] = self.nearest_valid_nodes[all_masked]

    def get_lower_transformed_neighbors(self):
        if self.current_node is None:
            return self.step_state
        neighbors_num = torch.minimum(
            self.cur_valid_num,
            torch.full_like(self.cur_valid_num, self.lower_neighbors_num),
        )
        max_neighbors = int(neighbors_num.max())
        indices = self.cur_sorted_idx[:, :max_neighbors].clone()
        positions = torch.arange(max_neighbors, device=self.device)[None, :]
        padding = positions >= neighbors_num[:, None]
        indices[padding] = self.current_node[:,
                                             None].expand(-1, max_neighbors)[padding]
        neighbors_xy = self.depot_node_xy.gather(
            1, indices[..., None].expand(-1, -1, 2))
        xy = torch.cat((self.first_xy, self.cur_xy, neighbors_xy), dim=1)
        xy = self._transform(xy)
        demand = self.depot_node_demand.gather(1, indices)
        demand[padding] = 0
        neighbor_mask = self.ninf_mask.gather(1, indices)
        mask = torch.cat((torch.zeros(self.batch_size, 2,
                         device=self.device), neighbor_mask), dim=1)[:, None]
        pairwise = torch.cdist(
            xy, xy, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        self.step_state.neighbors_num_list = neighbors_num
        self.step_state.lower_neighbors_index = indices
        self.step_state.lower_xy = xy
        self.step_state.lower_demand = demand[..., None]
        self.step_state.lower_cur_ninf_mask = mask
        self.step_state.lower_pairwise_dist = pairwise
        return self.step_state

    @staticmethod
    def _transform(xy):
        tail = xy[:, 1:]
        minimum = tail.amin(dim=1, keepdim=True)
        maximum = tail.amax(dim=1, keepdim=True)
        ratio = (maximum - minimum).amax(dim=-1, keepdim=True).clamp_min(1e-12)
        return ((xy - minimum) / ratio).clamp(0, 1)
