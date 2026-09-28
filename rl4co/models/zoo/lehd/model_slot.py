"""LEHD-style supervised training with POMOSlot's metric-aware slot core grafted in.

This fuses two paradigms that currently live apart in `train.py`:

  * The **LEHD supervised / imitation loop** (`LEHDModel`) — teacher-forcing
    against HGS-optimal sub-path labels, per-step cross-entropy loss, and a
    heavy multi-layer decoder that outputs P(visit customer | via depot).

  * The **POMOSlot slot core** (`pomo_slot/`) — SlotAttention over the encoder
    output, additive slot-context injection back into node embeddings, and the
    metric-aware auxiliary losses (slot entropy `SlotEntropyLoss` + metric
    preservation `MetricPreservationLoss`) that make the slots "metric aware".

To integrate them we keep LEHD's supervised loop and heavy decoder *as is*,
and replace only the light LEHD encoder with a slot-injecting variant (the same
idea as `SlotInjectingEncoder` in `pomo_slot/policy.py`, but wrapped around
LEHD's encoder and consuming LEHD's raw `(B, V+1, 4)` problem tensor rather than
an RL4CO `TensorDict`).  The metric/entropy aux terms are added to the per-step
cross-entropy loss, computed once per instance on the slots / assignment matrix.

Key differences from a naive copy of `LEHDModel.training_step`:

  * We **encode once per instance** (not once per decode step).  The original
    LEHD re-encodes at every step to enable per-step backprop through the one-
    layer encoder; that is unnecessary here and — critically — SlotAttention is
    iterative and expensive, so re-running it at every step is wasteful and
    numerically noisy.  We hold a single slot-injected encoding for the whole
    trajectory and reuse it across decode steps.

  * The CE policy loss and the slot aux losses are accumulated over the batch,
    and optimised jointly (single backward).  The metric AUX loss is metadata-
    aware but does not depend on WHICH node is chosen at which step, so it is
    naturally per-instance.

Because LEHD's data files carry no explicit slot target, `metric_variant` is
computed **on-the-fly** from the instance coordinates exactly as POMOSlot does
for Variant D when the dataset lacks cached `d_ins` (see `computing
compute_sparse_insertion_cost` in `rl4co/data/insertion_cost.py`).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from rl4co.models.zoo.lehd._network import LEHDVRPModel, LEHDEncoder
from rl4co.models.zoo.lehd._env import LEHDVRPEnv
from rl4co.models.zoo.lehd.model import LEHDModel, CAPACITY_MAP
from rl4co.models.nn.slot_attention import SlotAttention
from rl4co.models.nn.metric_loss import (
    MetricPreservationLoss,
    ProjectionHead,
    SlotEntropyLoss,
)
from rl4co.data.insertion_cost import compute_sparse_insertion_cost


class SlotInjectingLEHDEncoder(nn.Module):
    """LEHD light encoder wrapped with SlotAttention + slot-context injection.

    Mirrors `pomo_slot.policy.SlotInjectingEncoder` but consumes LEHD's raw
    `(B, V+1, 4)` problem tensor and writes the slot context back into LEHD's
    node embeddings:

        hidden[:, 1:, :] += A_ik @ slots     (customer nodes get slot context)
        hidden[:, 0,  :] unchanged            (depot embedding unchanged)

    Side-channel (read by `LEHDSlotModel.training_step` after each forward):
        last_slots: (B, K, d)  — slot embeddings z_k
        last_A_ik:  (B, N, K)  — soft assignment matrix (nodes x slots)
    """

    def __init__(self, base_encoder: nn.Module, slot_attn: nn.Module) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.slot_attn = slot_attn

        # Side-channel: populated after every forward, read by the training step.
        self.last_slots: Optional[torch.Tensor] = None
        self.last_A_ik: Optional[torch.Tensor] = None

    def forward(self, data: torch.Tensor, capacity: float) -> torch.Tensor:
        # LEHD encoder: (B, V+1, 4) -> (B, V+1, d).  Customer nodes are 1..V.
        hidden = self.base_encoder(data, capacity)

        # Slot Attention on customer nodes only (exclude depot at index 0).
        node_embs = hidden[:, 1:, :]                      # (B, V, d)
        slots, A_ik = self.slot_attn(node_embs)           # (B, K, d), (B, V, K)

        # Additive slot-context injection; depot embedding unchanged.
        slot_ctx = torch.bmm(A_ik, slots)                 # (B, V, d)
        pad_depot = torch.zeros_like(hidden[:, :1, :])    # (B, 1, d)
        hidden = hidden + torch.cat([pad_depot, slot_ctx], dim=1)  # (B, V+1, d)

        # Expose side-channel for the aux losses.
        self.last_slots = slots
        self.last_A_ik = A_ik

        return hidden


class LEHDSlotModel(LEHDModel):
    """LEHD supervised (imitation) training with POMOSlot's slot core.

    Subclasses :class:`LEHDModel` so validation and the optimizer/RL-free
    training loop are inherited unchanged; we override :meth:`training_step`
    to (a) encode once per instance through the slot-injecting encoder and
    (b) add the metric-aware auxiliary losses to the CE policy loss.

    The supervised CE loss is exactly LEHD's:
        L = -E[ log p_teacher(step) ]
    summed/averaged over every decoding step of the teacher-forced trajectory.
    To this we add:
        L_total = L + alpha_metric * L_metric + beta_entropy * L_entropy

    Args:
        All :class:`LEHDModel` args, plus the POMOSlot slot/hyper-parameters:
            num_slots (int): K — number of slot/region embeddings.
            metric_variant (str): "none"/"A"/"B"/"C"/"D" — matches POMOSlot.
                "D" is computed on-the-fly from locations. "A" is reconstruction
                (implemented), "C"/"D" use MetricPreservationLoss, "none"/"B"
                only add the entropy regulariser (or nothing, if disable_slots).
            disable_slots (bool): run as plain LEHD (no slots/aux).
            alpha_metric (float): weight for metric/reconstruction loss.
            beta_entropy (float): weight for slot entropy regulariser.
            slot_iters (int): SlotAttention refinement iterations.
            proj_dim (int): projection dim for phi(z_k) in metric loss.
            lambda_init (float), lr_dual (float): metric dual-ascent params.
            k_neighbors (int): k for on-the-fly d_ins neighbour count.
            ins_method (str): insertion-cost definition for on-the-fly Variant D.
            normalize_target / symmetrize_target: metric target aggregation.
    """

    def __init__(
        self,
        data_path: str,
        val_data_path: Optional[str] = None,
        num_loc: int = 100,
        embed_dim: int = 128,
        decoder_layer_num: int = 6,
        head_num: int = 8,
        qkv_dim: int = 16,
        ff_hidden: int = 512,
        n_train_episodes: int = 50_000,
        n_val_episodes: int = 500,
        optimizer_kwargs: Optional[dict] = None,
        # ---- POMOSlot slot args ----
        num_slots: int = 8,
        metric_variant: str = "D",
        alpha_metric: float = 0.1,
        beta_entropy: float = 0.01,
        slot_iters: int = 3,
        proj_dim: int = 64,
        lambda_init: float = 1.0,
        lr_dual: float = 1e-3,
        k_neighbors: int = 15,
        ins_method: str = "construction",
        disable_slots: bool = False,
        normalize_target: bool = True,
        symmetrize_target: bool = True,
    ) -> None:
        # Build the LEHD base model (holds .model whose encoder we'll wrap).
        super().__init__(
            data_path=data_path,
            val_data_path=val_data_path,
            num_loc=num_loc,
            embed_dim=embed_dim,
            decoder_layer_num=decoder_layer_num,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ff_hidden=ff_hidden,
            n_train_episodes=n_train_episodes,
            n_val_episodes=n_val_episodes,
            optimizer_kwargs=optimizer_kwargs,
        )
        # save_hyperparameters already ran in LEHDModel.__init__; add ours so the
        # checkpoint records them.
        self.save_hyperparameters(
            {
                "num_slots": num_slots,
                "metric_variant": metric_variant,
                "alpha_metric": alpha_metric,
                "beta_entropy": beta_entropy,
                "slot_iters": slot_iters,
                "proj_dim": proj_dim,
                "lambda_init": lambda_init,
                "lr_dual": lr_dual,
                "k_neighbors": k_neighbors,
                "ins_method": ins_method,
                "disable_slots": disable_slots,
                "normalize_target": normalize_target,
                "symmetrize_target": symmetrize_target,
            }
        )

        self.num_slots = num_slots
        self.metric_variant = metric_variant
        self.alpha_metric = alpha_metric
        self.beta_entropy = beta_entropy
        self.disable_slots = disable_slots
        self.k_neighbors = k_neighbors
        self.ins_method = ins_method

        # ---- Swap the light encoder for a slot-injecting one ----
        if not disable_slots:
            slot_attn = SlotAttention(num_slots=num_slots, dim=embed_dim, iters=slot_iters)
            self.model.encoder = SlotInjectingLEHDEncoder(self.model.encoder, slot_attn)
            self.slot_attn = slot_attn
        else:
            self.slot_attn = None

        # ---- Aux losses (skipped when slots are disabled) ----
        self.slot_entropy_loss = None if disable_slots else SlotEntropyLoss()

        self.metric_loss_fn: Optional[MetricPreservationLoss] = None
        if not disable_slots and metric_variant in ("C", "D"):
            proj_head = ProjectionHead(input_dim=embed_dim, proj_dim=proj_dim)
            self.metric_loss_fn = MetricPreservationLoss(
                proj_head=proj_head,
                variant=metric_variant,
                lambda_init=lambda_init,
                lr_dual=lr_dual,
                normalize_target=normalize_target,
                symmetrize_target=symmetrize_target,
            )

    # ------------------------------------------------------------------
    # Optimizer: separate dual-param group for log_lambda (dual ascent)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Like POMOSlot: dual ascent on log_lambda with its own lr_dual.

        Base LEHDModel puts everything in one Adam(+StepLR).  When a metric
        loss is present we instead split `log_lambda` into its own group with
        lr=lr_dual and no scheduler, mirroring `POMOSlot.configure_optimizers`.
        The main group keeps LEHD's StepLR schedule.
        """
        from rl4co.models.zoo.lehd.model import LEHDModel as _Base

        if self.metric_loss_fn is None:
            return _Base.configure_optimizers(self)

        lr_dual = self.hparams.get("lr_dual", 1e-3)
        log_lambda_id = id(self.metric_loss_fn.log_lambda)
        main_params = [p for p in self.parameters() if id(p) != log_lambda_id]
        dual_params = [self.metric_loss_fn.log_lambda]

        opt_kw = self.hparams.optimizer_kwargs or {"lr": 5e-5}
        param_groups = [
            {"params": main_params, **opt_kw},       # LEHD lr + StepLR schedule
            {"params": dual_params, "lr": lr_dual},  # dual ascent, unscheduled
        ]
        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lambda epoch: 0.97**epoch, lambda epoch: 1.0],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    # Aux-loss machinery (mirrors POMOSlot.shared_step's tail)
    # ------------------------------------------------------------------

    def _aux_losses(
        self,
        slotted: bool,
        slots: Optional[torch.Tensor],
        A_ik: Optional[torch.Tensor],
        problems: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the metric-aware auxiliary losses once per instance.

        Returns (aux_loss_scalar, log_dict).  slotted is True when the slots
        ran; slots / A_ik are (B, K, d) and (B, N, K) from the side-channel.
        """
        aux_loss = torch.tensor(0.0, device=device)
        log_dict: dict = {}
        if not slotted or slots is None or A_ik is None:
            return aux_loss, log_dict

        # Slot entropy regulariser (all variants).
        ent_loss = self.slot_entropy_loss(A_ik)
        aux_loss = aux_loss + self.beta_entropy * ent_loss
        log_dict["slot_entropy_loss"] = ent_loss.detach()

        # Customer coordinates: problems[:, 1:, :2] — node 0 is the depot.
        locs = problems[:, 1:, :2].contiguous()           # (B, V, 2)

        if self.metric_variant == "A":
            # Reconstruction: slot centroids vs customer coordinates.
            A_norm = A_ik / (A_ik.sum(dim=1, keepdim=True) + 1e-8)
            centroids = torch.einsum("bvk,bvc->bkc", A_norm, locs)
            recon = torch.einsum("bvk,bkc->bvc", A_ik, centroids)
            recon_loss = nn.functional.mse_loss(recon, locs)
            aux_loss = aux_loss + self.alpha_metric * recon_loss
            log_dict["recon_loss"] = recon_loss.detach()
        elif self.metric_loss_fn is not None:
            # Variant C (Euclidean) / D (on-the-fly insertion cost).
            d_ins_idx = d_ins_val = None
            if self.metric_variant == "D":
                customers = locs                       # (B, V, 2)
                depot = problems[:, :1, :2]             # (B, 1, 2)
                d_ins_idx, d_ins_val = compute_sparse_insertion_cost(
                    customers, k_neighbors=self.k_neighbors,
                    depot_loc=depot, method=self.ins_method,
                )  # (B, V, k) int16, (B, V, k) float32
            metric_loss, metric_info = self.metric_loss_fn(
                slots=slots,
                A_ik=A_ik,
                locs=locs,
                d_ins_idx=d_ins_idx,
                d_ins_val=d_ins_val,
            )
            aux_loss = aux_loss + self.alpha_metric * metric_loss
            log_dict.update(metric_info)

        return aux_loss, log_dict

    # ------------------------------------------------------------------
    # Training: teacher-forcing + slot aux losses
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """One supervised epoch-batch.

        Batch is LEHD's index metadata (see `make_lehd_dataloaders`); the real
        data lives in `self._train_env`.  Encodes the instance through the slot-
        injecting encoder ONCE, then teacher-forces the heavy decoder over every
        step, accumulating CE loss; the slot aux losses are added once per batch.
        """
        optimizer = self.optimizers()
        hp = self.hparams
        episode_start = batch["episode_start"].item()
        batch_size = batch["batch_size"].item()

        env = self._train_env
        env.load_problems(episode_start, batch_size)

        dev = self.device
        env.problems = env.problems.to(dev)
        env.solution = env.solution.to(dev)

        env.reset("train")
        state, _, _, done = env.pre_step()
        capacity = float(env.raw_data_capacity[0].item())

        # ---- Encode once (through slots) and cache the side-channel ----
        if not self.disable_slots:
            encoded_graph = self.model.encoder(env.problems, capacity)
            slots = self.model.encoder.last_slots      # (B, K, d)
            A_ik = self.model.encoder.last_A_ik        # (B, V, K)
        else:
            encoded_graph = self.model.encoder(env.problems, capacity)
            slots = A_ik = None

        # Backpropagate each heavy-decoder step immediately so its attention
        # activations can be released.  Decoder gradients accumulate normally
        # while the leaf gradient is bridged through the encoder once below.
        # This is mathematically equivalent to backpropagating the averaged CE
        # loss at the end, but avoids retaining an entire route of decoder
        # graphs (which is prohibitive for the intended batch_size=64).
        encoded_decoder = encoded_graph.detach().requires_grad_(True)
        self.model._encoded = encoded_decoder
        num_loss_steps = max(env.solution.shape[1] - 1, 1)
        optimizer.zero_grad()

        loss_sum = torch.tensor(0.0, device=dev)
        step = 0
        while not done:
            if step == 0:
                selected_teacher = env.solution[:, 0, 0]
                selected_flag_teacher = env.solution[:, 0, 1]
                selected_student = selected_teacher.clone()
                selected_flag_student = selected_flag_teacher.clone()
                step += 1
                state, _, _, done = env.step(
                    selected_teacher, selected_student,
                    selected_flag_teacher, selected_flag_student,
                )
                continue

            remaining_cap = state.problems[:, 0, 3]

            # Decode using the cached slot-injected encoding.
            probs = self.model.decoder(
                self.model._encoded, env.selected_node_list,
                capacity, remaining_cap,
            )  # (B, 2*V)

            # Teacher target: (node_1indexed, flag) -> (direct_idx, via_idx).
            B = probs.shape[0]
            V = state.problems.shape[1] - 1
            target_node = env.solution[:, step, 0]    # 1-indexed
            target_flag = env.solution[:, step, 1]    # 0 or 1
            target_direct = (target_node - 1).long()
            target_via = target_direct + V
            target = torch.where(target_flag.bool(), target_via, target_direct)
            target = target.clamp(0, 2 * V - 1)

            prob_teacher = probs.gather(1, target[:, None]).clamp_min(1e-9)
            loss = -prob_teacher.log().mean()
            self.manual_backward(loss / num_loss_steps)
            loss_sum = loss_sum + loss.detach()

            # Greedy student (for the env step; unused in loss).
            with torch.no_grad():
                student_flat = probs.argmax(dim=1)
                is_via_student = student_flat >= V
                sel_student = torch.where(
                    is_via_student, student_flat - V + 1, student_flat + 1
                ).long()
                flag_student = is_via_student.long()

            step += 1
            state, _, _, done = env.step(
                target_node.long(), sel_student,
                target_flag.long(), flag_student,
            )

        # ---- Backpropagate accumulated CE into the encoder plus aux losses ----
        ce_loss = loss_sum / num_loss_steps
        aux_loss, log_dict = self._aux_losses(
            slotted=not self.disable_slots,
            slots=slots, A_ik=A_ik,
            problems=env.problems, device=dev,
        )
        if encoded_decoder.grad is None:
            raise RuntimeError("LEHD decoder produced no gradient for its encoding")
        encoder_bridge = (
            encoded_graph * encoded_decoder.grad.detach()
        ).sum()
        self.manual_backward(encoder_bridge + aux_loss)

        optimizer.step()
        total = ce_loss + aux_loss.detach()

        self.log("train/loss", ce_loss.item(), on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train/total_loss", total.item(), on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        for k, v in log_dict.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, sync_dist=True)

        return total

    def validation_step(self, batch, batch_idx):
        """Greedy validation with one stable slot encoding per instance.

        The base LEHD validation path calls ``decode_step``, which re-encodes
        the instance at every decoding step.  SlotAttention samples its initial
        slots on each forward, so that path would use a different slot context
        at every step and would not match this model's cached-encoding training
        semantics.
        """
        episode_start = batch["episode_start"].item()
        requested_batch_size = batch["batch_size"].item()

        env = self._val_env
        env.load_problems(episode_start, requested_batch_size)
        batch_size = env.problems.shape[0]
        dev = self.device
        env.problems = env.problems.to(dev)
        env.solution = env.solution.to(dev)

        env.reset("test")
        state, _, _, done = env.pre_step()
        capacity = float(env.raw_data_capacity[0].item())
        self.model._encoded = self.model.encoder(env.problems, capacity)

        step = 0
        r_stud = None
        with torch.no_grad():
            while not done:
                if step == 0:
                    selected = env.solution[:, 0, 0].long()
                    sel_flag = env.solution[:, 0, 1].long()
                    step += 1
                    state, _, _, done = env.step(
                        selected, selected, sel_flag, sel_flag
                    )
                    continue

                remaining_cap = state.problems[:, 0, 3]
                probs = self.model.decoder(
                    self.model._encoded,
                    env.selected_node_list,
                    capacity,
                    remaining_cap,
                )
                V = state.problems.shape[1] - 1
                student_flat = probs.argmax(dim=1)
                is_via = student_flat >= V
                sel_node = torch.where(
                    is_via, student_flat - V + 1, student_flat + 1
                ).long()
                sel_flag = is_via.long()
                step += 1
                state, _, r_stud, done = env.step(
                    sel_node, sel_node, sel_flag, sel_flag
                )

        if r_stud is not None:
            self._val_reward_sum += r_stud.mean().item() * batch_size
            self._val_count += batch_size


# ---------------------------------------------------------------------------
# Minimal registry helpers
# ---------------------------------------------------------------------------

def make_lehd_slot_dataloaders(*args, **kwargs):
    """Alias — LEHDSlotModel reuses LEHD's index-metadata dataloaders."""
    from rl4co.models.zoo.lehd.model import make_lehd_dataloaders
    return make_lehd_dataloaders(*args, **kwargs)
