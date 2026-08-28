from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.models.nn.slot_attention import SlotAttention
from rl4co.data.insertion_cost import compute_sparse_insertion_cost
from rl4co.models.nn.metric_loss import (
    MetricPreservationLoss,
    ProjectionHead,
    SlotEntropyLoss,
)
from rl4co.models.rl.reinforce.baselines import SharedBaseline
from rl4co.models.zoo.pomo_slot.policy import SlotInjectingEncoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SingleSharedBaseline(SharedBaseline):
    """Shared baseline safe for single-start decode.

    Stock :class:`SharedBaseline` reduces over ``dim=1``, which assumes POMO's
    multi-start reward layout ``[B, n_start]``. Single-start AM produces a
    1-D reward ``[B]``, so ``dim=1`` is out of range. Averaging over the last
    dim (``on_dim=-1``) is correct for BOTH layouts: it means over ``n_start``
    for POMO and over the batch for single-start AM.
    """

    def eval(self, td, reward, env=None, on_dim=-1):  # e.g. [batch] or [batch, n_start]
        return reward.mean(dim=on_dim, keepdims=True), 0


class AMSlot(AttentionModel):
    """
    Attention Model extended with Metric-Aware Slot Abstraction.

    Lightweight, single-start (rollout-baseline) alternative to POMOSlot for
    fast iteration. Same slot module and metric loss; only the REINFORCE
    training loop differs (AM rollout baseline + single-start decode).

    Args:
        env: RL4CO environment (e.g. CVRP).
        embed_dim (int): Encoder/decoder embedding dimension. Default: 128.
        num_slots (int): K — number of slot/region embeddings.
        metric_variant (str): Ablation variant — one of "none"/"B", "A", "C", "D".
        alpha_metric (float): Weight for metric preservation/reconstruction loss.
        beta_entropy (float): Weight for slot entropy regulariser.
        slot_iters (int): Number of SlotAttention refinement iterations.
        proj_dim (int): Projection dimension for phi(z_k) in metric loss.
        lambda_init (float): Initial Lagrange multiplier for MetricPreservationLoss.
        lr_dual (float): Learning rate for dual ascent on lambda (log_lambda param group).
        k_neighbors (int): kNN sparsification for on-the-fly d_ins computation (Variant D).
        ins_method (str): d_ins insertion-cost definition for the on-the-fly
            Variant D fallback: "savings" | "construction" | "insertion".
        baseline (str): REINFORCE baseline — "rollout" (default, true AM) or "shared".
        **am_kwargs: Remaining kwargs forwarded to AttentionModel base class.
    """

    # "E" excluded: future-regret target not yet implemented
    METRIC_VARIANTS = {"none", "B", "A", "C", "D"}

    def __init__(
        self,
        env: RL4COEnvBase,
        embed_dim: int = 128,
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
        baseline: str = "rollout",
        disable_slots: bool = False,
        **am_kwargs,
    ) -> None:
        assert metric_variant in self.METRIC_VARIANTS, (
            f"metric_variant must be one of {self.METRIC_VARIANTS}, got '{metric_variant}'"
        )

        # Base encoder used directly when disable_slots (plain AM — no slots).
        # AM default is 3 encoder layers (lighter than POMO's 6). Overridable.
        base_encoder = AttentionModelEncoder(
            embed_dim=embed_dim,
            num_heads=am_kwargs.pop("num_heads", 8),
            num_layers=am_kwargs.pop("num_encoder_layers", 3),
            env_name=env.name,
            normalization=am_kwargs.pop("normalization", "instance"),
            feedforward_hidden=am_kwargs.pop("feedforward_hidden", 512),
        )

        if disable_slots:
            # Plain AM — no slot attention or injection.
            policy = AttentionModelPolicy(
                encoder=base_encoder,
                embed_dim=embed_dim,
                env_name=env.name,
                use_graph_context=am_kwargs.pop("use_graph_context", False),
            )
            slot_attn = None
        else:
            # Wrap the base encoder with slot injection.
            slot_attn = SlotAttention(
                num_slots=num_slots,
                dim=embed_dim,
                iters=slot_iters,
            )
            slot_encoder = SlotInjectingEncoder(base_encoder, slot_attn)
            policy = AttentionModelPolicy(
                encoder=slot_encoder,
                embed_dim=embed_dim,
                env_name=env.name,
                use_graph_context=am_kwargs.pop("use_graph_context", False),
            )

        # Init AM (REINFORCE single-start)
        # The stock "shared" baseline reduces over dim=1 (POMO multi-start
        # layout) and breaks on single-start rewards [B]. Route it through
        # the single-start-safe subclass. Other strings (e.g. "rollout") are
        # handled by rl4co as usual.
        if isinstance(baseline, str) and baseline == "shared":
            baseline = SingleSharedBaseline()
        super().__init__(env, policy=policy, baseline=baseline, **am_kwargs)
        self.save_hyperparameters(logger=False, ignore=["env", "policy"])

        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.metric_variant = metric_variant
        self.alpha_metric = alpha_metric
        self.beta_entropy = beta_entropy
        self.k_neighbors = k_neighbors
        self.ins_method = ins_method
        self.disable_slots = disable_slots

        # Reference for display (lives inside policy.encoder)
        self.slot_attn = slot_attn

        # Auxiliary losses (skipped when slots are disabled)
        self.slot_entropy_loss = None if disable_slots else SlotEntropyLoss()
        self.metric_loss_fn = (
            None if disable_slots else self._build_metric_loss(metric_variant, embed_dim, proj_dim, lambda_init, lr_dual)
        )

        log.info(
            f"AMSlot: embed_dim={embed_dim}, K={num_slots}, "
            f"variant={metric_variant}, alpha={alpha_metric}, beta={beta_entropy}, "
            f"baseline={baseline}"
        )

    def _build_metric_loss(self, variant, embed_dim, proj_dim, lambda_init, lr_dual):
        """Construct the metric-preservation loss for metric variants (C/D)."""
        if variant in ("none", "B", "A"):
            return None
        proj_head = ProjectionHead(input_dim=embed_dim, proj_dim=proj_dim)
        return MetricPreservationLoss(
            proj_head=proj_head,
            variant=variant,
            lambda_init=lambda_init,
            lr_dual=lr_dual,
        )

    # configure_optimizers: separate dual param group for log_lambda (dual ascent),
    # delegating to base to keep the AM LR scheduler.
    def configure_optimizers(self):
        if self.metric_loss_fn is None:
            # No dual parameter — use base class as-is
            return super().configure_optimizers()

        lr_dual = self.hparams.get("lr_dual", 1e-3)
        log_lambda_id = id(self.metric_loss_fn.log_lambda)

        main_params = [p for p in self.parameters() if id(p) != log_lambda_id]
        dual_params = [self.metric_loss_fn.log_lambda]

        param_groups = [
            {"params": main_params},                      # uses base lr + scheduler
            {"params": dual_params, "lr": lr_dual},       # dual ascent, no scheduler
        ]
        return super().configure_optimizers(parameters=param_groups)

    # on_after_optimizer_step: clamp log_lambda (dual ascent). Same safety net
    # as POMOSlot (see model.py for rationale).
    def on_after_optimizer_step(self, optimizer):
        if self.metric_loss_fn is None:
            return
        lambda_max = getattr(self, "lambda_max", 50.0)
        with torch.no_grad():
            self.metric_loss_fn.log_lambda.data.clamp_(max=math.log(lambda_max))

    # shared_step: extract d_ins, run AM (single-start), read slot side-channel,
    # add aux loss. Mirrors POMOSlot.shared_step except it calls the base AM
    # (REINFORCE) step — already single-start, so no multistart reshape needed.
    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # Extract sparse d_ins keys from batch BEFORE passing to base
        d_ins_idx = None
        d_ins_val = None

        if isinstance(batch, dict):
            d_ins_idx = batch.pop("d_ins_idx", None)
            d_ins_val = batch.pop("d_ins_val", None)
            from tensordict import TensorDict as TD
            B = next(iter(batch.values())).shape[0]
            batch = TD(batch, batch_size=[B])
        elif hasattr(batch, "keys"):
            keys = list(batch.keys())
            if "d_ins_idx" in keys:
                d_ins_idx = batch.get("d_ins_idx")
                d_ins_val = batch.get("d_ins_val")
                batch = batch.exclude("d_ins_idx", "d_ins_val")

        # Move to model device
        device = self.device
        batch = batch.to(device)
        if d_ins_idx is not None:
            d_ins_idx = d_ins_idx.to(device)  # keep as int16 for transfer, cast in loss fn
        if d_ins_val is not None:
            d_ins_val = d_ins_val.to(device)
        # Read locs BEFORE base shared_step mutates the batch
        locs_customers: torch.Tensor | None = None
        if self.metric_variant not in ("none", "B"):
            raw_locs = batch.get("locs") if hasattr(batch, "get") else batch["locs"]
            locs_customers = raw_locs.to(device)  # (B, N, 2) — customers only, no depot

        # Fallback: compute d_ins on-the-fly if Variant D needs it, but the
        # dataset didn't cache it (mirrors POMOSlot — see model.py for indexing notes)
        if (
            self.metric_variant == "D"
            and d_ins_idx is None
            and d_ins_val is None
            and locs_customers is not None
            and locs_customers.shape[1] > 1
        ):
            customers = locs_customers[:, 1:, :]   # (B, N_cust, 2) — matches A_ik
            depot = locs_customers[:, :1, :]        # (B, 1, 2) — node 0 as pseudo-depot
            d_ins_idx, d_ins_val = compute_sparse_insertion_cost(
                customers, k_neighbors=self.k_neighbors, depot_loc=depot,
                method=self.ins_method,
            )  # (B, N_cust, k) int16, (B, N_cust, k) float32

        # Run standard AM (REINFORCE) step — single-start. Slot encoder runs
        # inside policy.forward(); slots/A_ik available via side-channel after.
        out = super().shared_step(batch, batch_idx, phase, dataloader_idx)

        # Skip aux losses during val/test, or when slots are disabled.
        if phase != "train" or self.metric_variant in ("none", "B") or self.disable_slots:
            return out

        # Read slot side-channel
        slots = self.policy.encoder.last_slots  # (B, K, d)
        A_ik  = self.policy.encoder.last_A_ik   # (B, N, K)

        if slots is None or A_ik is None:
            log.warning("SlotInjectingEncoder side-channel is None — skipping aux loss.")
            return out

        # Compute auxiliary losses
        aux_loss = torch.tensor(0.0, device=device)
        log_dict: dict = {}

        # Slot entropy regulariser (all variants except "none")
        ent_loss = self.slot_entropy_loss(A_ik)
        aux_loss = aux_loss + self.beta_entropy * ent_loss
        log_dict["slot_entropy_loss"] = ent_loss.detach()

        # Variant A: reconstruction loss (slot centroids vs node coords)
        if self.metric_variant == "A":
            locs = locs_customers  # (B, N, 2) customers only
            A_norm = A_ik / (A_ik.sum(dim=1, keepdim=True) + 1e-8)
            centroids = torch.einsum("bnk,bnc->bkc", A_norm, locs)
            recon = torch.einsum("bnk,bkc->bnc", A_ik, centroids)
            recon_loss = F.mse_loss(recon, locs)
            aux_loss = aux_loss + self.alpha_metric * recon_loss
            log_dict["recon_loss"] = recon_loss.detach()

        # Variants C / D: metric preservation loss
        elif self.metric_loss_fn is not None:
            locs = locs_customers  # (B, N, 2) customers only
            metric_loss, metric_info = self.metric_loss_fn(
                slots=slots,
                A_ik=A_ik,
                locs=locs,
                d_ins_idx=d_ins_idx,
                d_ins_val=d_ins_val,
            )
            aux_loss = aux_loss + self.alpha_metric * metric_loss
            log_dict.update(metric_info)

        # Merge auxiliary loss into policy loss
        policy_loss = out.get("loss", None)
        if policy_loss is not None and aux_loss.requires_grad:
            out["loss"] = policy_loss + aux_loss
            log_dict["aux_loss"]    = aux_loss.detach()
            log_dict["policy_loss"] = policy_loss.detach()

        # Log to Lightning
        for k, v in log_dict.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=True)

        return out
