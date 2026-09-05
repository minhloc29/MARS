from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch

from .env import CandidateReductionCVRPEnv
from .layers import LowerModel, UpperModel


class L2RModel(pl.LightningModule):
    """Lightning adapter for the original L2R CVRP policy."""

    def __init__(
        self,
        env,
        embed_dim: int = 128,
        lower_neighbors_num: int = 50,
        reduction_percentage: float = 0.1,
        problem: str = "cvrp",
        optimizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if problem != "cvrp":
            raise NotImplementedError(
                "The L2R adapter currently supports --env cvrp only.")

        self.problem = problem
        self.lower_neighbors_num = lower_neighbors_num
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.l2r_env_cls = CandidateReductionCVRPEnv

        model_params = {
            "embedding_dim": embed_dim,
            "sqrt_embedding_dim": embed_dim**0.5,
            "logit_clipping": 10.0,
            "decoder_layer_num": 6,
            "ff_hidden_dim": 512,
            "eval_type": "sampling",
        }
        model_params["device"] = self.device
        self.upper_model = UpperModel(model_params)
        self.lower_model = LowerModel(model_params)

        self.save_hyperparameters(logger=False, ignore=["env"])

    def _make_dataset(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"locs", "depot", "demand", "capacity"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(
                f"L2R CVRP batch is missing keys: {sorted(missing)}")

        capacity = batch["capacity"].reshape(-1,
                                             1).to(dtype=batch["demand"].dtype)
        demand = batch["demand"].to(dtype=torch.float32)
        if demand.max() > 1.0 + 1e-6:
            demand = demand / capacity
        return {
            "depot_xy": batch["depot"].to(dtype=torch.float32).unsqueeze(1),
            "node_xy": batch["locs"].to(dtype=torch.float32),
            "node_demand": demand,
        }

    def _rollout(self, batch: dict[str, torch.Tensor], sampling: bool):
        dataset = self._make_dataset(batch)
        device = dataset["node_xy"].device
        env = self.l2r_env_cls(
            device=device,
            problem_size=dataset["node_xy"].shape[1],
            lower_neighbors_num=self.lower_neighbors_num,
            reduction_percentage=self.hparams.reduction_percentage,
        )
        env.load_problems(dataset)
        reset_state, _, _ = env.reset()
        if sampling:
            self.upper_model.train()
            self.lower_model.train()
        else:
            self.upper_model.eval()
            self.lower_model.eval()
        self.upper_model.set_decoder_method(
            "sampling" if sampling else "greedy")
        self.lower_model.set_decoder_method(
            "sampling" if sampling else "greedy")
        self.upper_model.pre_forward(reset_state)

        log_probs = []
        state, reward, done = env.pre_step()
        while not done:
            if state.current_node is not None:
                state = env.get_upper_input()
                upper_scores, _, upper_prob = self.upper_model(state)
                env.update_cur_scores(upper_scores)
            state = env.get_lower_transformed_neighbors()
            lower_selected, lower_prob = self.lower_model(state)
            if state.current_node is None:
                upper_prob = torch.ones(
                    dataset["node_xy"].shape[0], device=device)
            elif upper_prob is None:
                upper_prob = torch.ones(
                    dataset["node_xy"].shape[0], device=device)
            if lower_prob is None:
                lower_prob = torch.ones(
                    dataset["node_xy"].shape[0], device=device)
            state, reward, done = env.step(lower_selected)
            log_probs.append((upper_prob.clamp_min(1e-12),
                             lower_prob.clamp_min(1e-12)))

        upper_log_prob = torch.stack([item[0].log()
                                     for item in log_probs], dim=1).sum(1)
        lower_log_prob = torch.stack([item[1].log()
                                     for item in log_probs], dim=1).sum(1)
        return reward, upper_log_prob + lower_log_prob

    def _shared_step(self, batch, phase: str):
        batch = {key: value.to(self.device) for key, value in batch.items()}
        sampling = phase == "train"
        if sampling:
            reward, log_prob = self._rollout(batch, sampling=True)
            with torch.no_grad():
                baseline_reward, _ = self._rollout(batch, sampling=False)
            advantage = reward - baseline_reward
            loss = -(advantage.detach() * log_prob).mean()
        else:
            with torch.no_grad():
                reward, _ = self._rollout(batch, sampling=False)
            loss = -reward.mean()

        self.log(f"{phase}/reward", reward.mean(),
                 prog_bar=True, on_epoch=True)
        self.log(f"{phase}/loss", loss, prog_bar=phase ==
                 "train", on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)
