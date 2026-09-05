from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def compatibility(model_params, query, nodes, bias, mask):
    score = torch.matmul(query, nodes.transpose(1, 2))
    score = score / model_params["sqrt_embedding_dim"] + bias
    score = model_params["logit_clipping"] * torch.tanh(score) + mask
    return F.softmax(score, dim=-1).squeeze(1)


def select_next_node(probs: Tensor, strategy: str = "sampling") -> Tuple[Tensor, Tensor | None]:
    if torch.isnan(probs).any():
        raise RuntimeError("L2R produced NaN action probabilities")
    if strategy == "sampling":
        selected = probs.multinomial(1).squeeze(1) # samples an index according to the probabilities in probs.
        probability = probs.gather(1, selected[:, None]).squeeze(1)
        return selected, probability
    if strategy == "greedy":
        return probs.argmax(dim=-1), None
    raise NotImplementedError(f"Unsupported decoding strategy: {strategy}")


def get_encoding(encoded_nodes, indices):
    embedding_dim = encoded_nodes.size(2)
    expanded = indices[:, :, None].expand(-1, -1, embedding_dim)
    return encoded_nodes.gather(1, expanded)


def adaptation_attention_free_module(query, key, value, adaptation_bias, ninf_mask=None):
    if ninf_mask is not None:
        adaptation_bias = adaptation_bias + ninf_mask
    query_gate = torch.sigmoid(query)
    max_key = key.amax(dim=-2, keepdim=True)
    exp_key = torch.exp(key - max_key)
    weights = torch.exp(adaptation_bias)
    denominator = (weights @ exp_key).clamp_min(1e-12)
    numerator = weights @ (exp_key * value)
    return query_gate * (numerator / denominator)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.first = nn.Linear(embedding_dim, hidden_dim)
        self.second = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, inputs):
        return self.second(F.relu(self.first(inputs)))


class DecoderLayer(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.feed_forward = FeedForward(
            embedding_dim, model_params["ff_hidden_dim"])
        self.alpha_attn = nn.Parameter(torch.ones(1))
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, data, pairwise_bias, mask):
        attention = adaptation_attention_free_module(
            self.query(data), self.key(data), self.value(data),
            self.alpha_attn * pairwise_bias, mask,
        )
        hidden = self.norm1(data + attention)
        return self.norm2(hidden + self.feed_forward(hidden))


class UpperModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.params = model_params
        dim = model_params["embedding_dim"]
        self.embedding = nn.Linear(3, dim)
        self.query_last = nn.Linear(dim + 1, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.alpha_attn = nn.Parameter(torch.ones(1))
        self.alpha_com = nn.Parameter(torch.ones(1))
        self.encoded_nodes = None
        self.k = None
        self.v = None
        self.single_head_key = None
        self.log_scale = None

    def set_decoder_method(self, strategy):
        self.params["eval_type"] = strategy

    def pre_forward(self, reset_state):
        depot_demand = torch.zeros(
            reset_state.depot_xy.shape[0], 1, 1, device=reset_state.depot_xy.device)
        depot = torch.cat((reset_state.depot_xy, depot_demand), dim=2)
        nodes = torch.cat(
            (reset_state.node_xy, reset_state.node_demand[..., None]), dim=2)
        self.encoded_nodes = self.embedding(torch.cat((depot, nodes), dim=1))
        self.k = self.key(self.encoded_nodes)
        self.v = self.value(self.encoded_nodes)
        self.single_head_key = self.encoded_nodes
        self.log_scale = reset_state.log_scale

    def forward(self, state):
        indices = state.upper_unvisited_index
        distances = state.upper_cur_dist
        mask = state.upper_cur_ninf_mask
        encoded_last = get_encoding(
            self.encoded_nodes, state.current_node[:, None])
        query = self.query_last(
            torch.cat((encoded_last, state.load[:, None, None]), dim=2))
        keys = get_encoding(self.k, indices)
        values = get_encoding(self.v, indices)
        bias = -self.log_scale * self.alpha_attn * distances
        attention = adaptation_attention_free_module(
            query, keys, values, bias, mask)
        node_keys = get_encoding(self.single_head_key, indices)
        scores = compatibility(
            self.params, attention, node_keys,
            -self.log_scale * self.alpha_com * distances, mask,
        )
        selected, probability = select_next_node(
            scores, self.params["eval_type"])
        full_scores = torch.zeros(
            state.batch_size, state.problem_size + 1, device=scores.device)
        full_scores.scatter_(1, indices, scores)
        true_selected = indices.gather(1, selected[:, None]).squeeze(1)
        return full_scores, true_selected, probability


class LowerModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.params = model_params
        self.device = model_params.get("device", None)
        dim = model_params["embedding_dim"]
        self.embedding = nn.Linear(2, dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(model_params)
             for _ in range(model_params["decoder_layer_num"])]
        )
        self.load_embedding = nn.Linear(1, dim, bias=False)
        self.demand_embedding = nn.Linear(1, dim, bias=False)
        self.query_first = nn.Linear(dim, dim, bias=False)
        self.query_last = nn.Linear(dim, dim, bias=False)
        self.alpha_com = nn.Parameter(torch.ones(1))
        self.padding = nn.Parameter(torch.zeros(1, dim), requires_grad=False)

    def set_decoder_method(self, strategy):
        self.params["eval_type"] = strategy

    def forward(self, state):
        batch_size = state.batch_size
        if state.current_node is None:
            return torch.zeros(batch_size, dtype=torch.long, device=self.padding.device), torch.ones(batch_size, device=self.padding.device)
        neighbor_mask = state.lower_cur_ninf_mask[..., 2:]
        load = state.load[:, None, None].clamp_min(1e-12)
        embeddings = self.embedding(state.lower_xy)
        first = self.query_first(embeddings[:, :1]) + self.load_embedding(load)
        last = self.query_last(embeddings[:, 1:2]) + self.load_embedding(load)
        neighbor = embeddings[:, 2:] + \
            self.demand_embedding((state.lower_demand / load).clamp(0, 1))
        padding_mask = neighbor_mask.transpose(1, 2) < 0
        neighbor = torch.where(
            padding_mask, self.padding.view(1, 1, -1), neighbor)
        output = torch.cat((first, last, neighbor), dim=1)
        scale = -torch.log2(state.neighbors_num_list.float()
                            )[:, None, None] * state.lower_pairwise_dist
        mask = state.lower_cur_ninf_mask.expand(-1, output.shape[1], -1)
        for layer in self.layers:
            output = layer(output, scale, mask)
        query = output[:, :1] + output[:, 1:2]
        scores = compatibility(
            self.params, query, output[:, 2:],
            self.alpha_com * scale[:, 1:2, 2:], neighbor_mask,
        )
        scores = torch.nan_to_num(scores, nan=0.0)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        selected, probability = select_next_node(
            scores, self.params["eval_type"])
        return state.lower_neighbors_index.gather(1, selected[:, None]).squeeze(1), probability
