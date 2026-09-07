"""Self-improved learning on the same fixed CVRP splits used by MARS."""

import copy

import lightning.pytorch as pl
import torch

from tensordict import TensorDict

from .policy import SILPolicy
from .solution import (
    augment_routes,
    feasible,
    insertion_labels,
    order_routes_by_centroid,
    route_length,
    sample_subpaths,
)


class SIL(pl.LightningModule):
    """Insertion labels -> random reconstruction -> subpath imitation.

    Like SIL/CVRP/Train/Trainer.py, update the optimizer at each teacher-forced
    decoding step and improve persistent labels between blocks of epochs.
    Labels are keyed by dataset row, never by the shuffled batch position.
    """

    def __init__(
        self,
        env,
        embed_dim=128,
        num_layers=6,
        num_heads=8,
        feedforward_hidden=512,
        repair_budget=5,
        improve_every=20,
        max_subtour_length=64,
        parallel_reconstruction=True,
        update_mode="batch",
        optimizer_kwargs=None,
    ):
        super().__init__()
        if repair_budget < 0 or improve_every < 1 or max_subtour_length < 4:
            raise ValueError(
                "SIL needs repair_budget >= 0, improve_every >= 1, max_subtour_length >= 4"
            )
        if update_mode not in {"batch", "node"}:
            raise ValueError("SIL update_mode must be 'batch' or 'node'")
        self.save_hyperparameters(ignore=["env"])
        self.env = env
        self.policy = SILPolicy(embed_dim, num_layers, num_heads, feedforward_hidden)
        self.automatic_optimization = False
        self.labels = {}
        self.label_rounds = {}
        self.dataset_signature = None
        self.best_policy_state = None
        self.best_validation_reward = float("-inf")
        self.repair_policy_state = None
        # A frozen teacher is deliberately outside registered modules so it
        # cannot be optimized or accidentally replace the student checkpoint.
        object.__setattr__(self, "_repair_policy", None)

    def on_train_epoch_start(self):
        if self.current_epoch > 0 and self.current_epoch % self.hparams.improve_every == 0:
            self.repair_policy_state = self.best_policy_state or {
                k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()
            }
            object.__setattr__(self, "_repair_policy", None)
        if self.repair_policy_state is not None and self._repair_policy is None:
            teacher = copy.deepcopy(self.policy).eval().requires_grad_(False)
            teacher.load_state_dict(self.repair_policy_state)
            object.__setattr__(self, "_repair_policy", teacher)

    def on_validation_epoch_start(self):
        self._validation_reward_sum = 0.0
        self._validation_count = 0

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self._validation_count:
            return
        reward = self._validation_reward_sum / self._validation_count
        if reward > self.best_validation_reward:
            self.best_validation_reward = reward
            self.best_policy_state = {
                k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()
            }

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.policy.parameters(), **(self.hparams.optimizer_kwargs or {"lr": 1e-4})
        )

    def _reset(self, batch):
        data = {key: batch[key] for key in ("locs", "depot", "demand")}
        demand = data["demand"]
        if not torch.isfinite(demand).all() or (demand <= 0).any() or (demand > 1).any():
            raise ValueError("SIL requires positive, normalized CVRP demands in (0, 1]")
        return self.env.reset(TensorDict(data, batch_size=[demand.size(0)], device=demand.device))

    @torch.no_grad()
    def improve_labels(self, locs, demand, labels):
        """Accept only feasible, strictly shorter reconstructed complete tours."""
        was_training = self.policy.training
        self.policy.eval()
        policy = self._repair_policy or self.policy
        for _ in range(self.hparams.repair_budget):
            labels = augment_routes(order_routes_by_centroid(locs, labels))
            length = int(
                torch.randint(
                    4, min(demand.size(1), self.hparams.max_subtour_length) + 1, ()
                ).item()
            )
            xy, dem, teacher, remaining, owner, pos, mapping = sample_subpaths(
                locs, demand, labels, length, self.hparams.parallel_reconstruction
            )
            repaired = policy.decode(xy, dem, teacher[:, 0], remaining)
            # Each partial path starts at a fixed customer and ends at a depot;
            # compare the same endpoint convention on both sides.
            better = route_length(xy, repaired) < route_length(xy, teacher)
            repaired[:, :, 0] = mapping.gather(1, repaired[:, :, 0] - 1)
            candidate = labels.clone()
            candidate[owner[better, None], pos[better]] = repaired[better]
            accept = feasible(demand, candidate)
            accept &= route_length(locs, candidate) < route_length(locs, labels)
            labels = torch.where(accept[:, None, None], candidate, labels)
        self.policy.train(was_training)
        return labels

    @torch.no_grad()
    def _batch_labels(self, batch, td):
        if "instance_id" not in batch:
            raise ValueError("SIL training batches require stable instance_id values")
        ids = batch["instance_id"].detach().cpu().tolist()
        missing = [i for i, key in enumerate(ids) if key not in self.labels]
        if missing:
            initial = insertion_labels(td["locs"][missing], td["demand"][missing])
            for i, label in zip(missing, initial):
                # Customer IDs at all supported sizes fit in int16. Keeping
                # persistent labels compact reduces N=1000 checkpoints from
                # about 800 MB to 200 MB for 50,000 instances.
                self.labels[ids[i]] = label.to(device="cpu", dtype=torch.int16)
                self.label_rounds[ids[i]] = 0
        labels = torch.stack([self.labels[key] for key in ids]).to(
            device=self.device, dtype=torch.long
        )
        round_id = self.current_epoch // self.hparams.improve_every
        stale = [i for i, key in enumerate(ids) if self.label_rounds[key] < round_id]
        if stale:
            improved = self.improve_labels(td["locs"][stale], td["demand"][stale], labels[stale])
            labels[stale] = improved
            for i, label in zip(stale, improved):
                self.labels[ids[i]] = label.to(device="cpu", dtype=torch.int16)
                self.label_rounds[ids[i]] = round_id
        return labels

    def training_step(self, batch, batch_idx):
        td = self._reset(batch)
        labels = self._batch_labels(batch, td)
        length = int(
            torch.randint(
                4, min(td["demand"].size(1), self.hparams.max_subtour_length) + 1, ()
            ).item()
        )
        xy, demand, teacher, remaining, *_ = sample_subpaths(
            td["locs"], td["demand"], augment_routes(labels), length
        )
        selected = teacher[:, :1, 0]
        remaining, _ = self.policy.advance(demand, remaining, teacher[:, 0, 0], teacher[:, 0, 1])
        optimizer = self.optimizers()
        losses = []
        if self.hparams.update_mode == "batch":
            optimizer.zero_grad()
        for step in range(1, length):
            problems = self.policy.problems(xy, demand, remaining)
            probs = self.policy.probabilities(problems, selected)
            node, flag = teacher[:, step].unbind(-1)
            target = node - 1 + flag * length
            loss = -probs.gather(1, target[:, None]).clamp_min(1e-12).log().mean()
            if self.hparams.update_mode == "node":
                # Upstream SIL behavior: every decoded customer is a separate
                # optimizer update. This is available for strict reproduction.
                optimizer.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(
                    optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                )
                optimizer.step()
            else:
                # Comparable mode: average all teacher-forced decisions into
                # one batch gradient, matching MARS's one update per batch.
                self.manual_backward(loss / (length - 1))
            losses.append(loss.detach())
            remaining, _ = self.policy.advance(demand, remaining, node, flag)
            selected = torch.cat((selected, node[:, None]), 1)
        if self.hparams.update_mode == "batch":
            self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer.step()
        loss = torch.stack(losses).mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=len(labels))
        self.log(
            "train/pseudo_reward",
            -route_length(td["locs"], labels).mean(),
            on_step=False,
            on_epoch=True,
            batch_size=len(labels),
        )
        self.log(
            "train/subtour_length",
            float(length),
            on_step=True,
            on_epoch=True,
            batch_size=len(labels),
        )
        self.log(
            "train/optimizer_updates",
            1.0 if self.hparams.update_mode == "batch" else float(length - 1),
            on_step=True,
            on_epoch=True,
            batch_size=len(labels),
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self.policy(self._reset(batch), self.env, phase="val")
        self._validation_reward_sum += out["reward"].sum().item()
        self._validation_count += len(out["reward"])
        self.log(
            "val/reward",
            out["reward"].mean(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(out["reward"]),
        )
        return out

    def test_step(self, batch, batch_idx):
        out = self.policy(self._reset(batch), self.env)
        self.log("test/reward", out["reward"].mean(), batch_size=len(out["reward"]))
        return out

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sil_labels"] = self.labels
        checkpoint["sil_label_rounds"] = self.label_rounds
        checkpoint["sil_dataset_signature"] = self.dataset_signature
        checkpoint["sil_best_policy_state"] = self.best_policy_state
        checkpoint["sil_best_validation_reward"] = self.best_validation_reward
        checkpoint["sil_repair_policy_state"] = self.repair_policy_state

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint.get("sil_dataset_signature")
        if self.dataset_signature is not None and saved != self.dataset_signature:
            raise ValueError("SIL checkpoint pseudo-labels belong to a different training dataset")
        self.dataset_signature = saved
        self.labels = {
            key: value.to(device="cpu", dtype=torch.int16)
            for key, value in checkpoint.get("sil_labels", {}).items()
        }
        self.label_rounds = checkpoint.get("sil_label_rounds", {})
        self.best_policy_state = checkpoint.get("sil_best_policy_state")
        self.best_validation_reward = checkpoint.get("sil_best_validation_reward", float("-inf"))
        self.repair_policy_state = checkpoint.get("sil_repair_policy_state")
