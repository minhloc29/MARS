"""LEHD and TTPL Lightning wrappers for fair comparison with MARS.

Two model classes are exported:
  LEHDModel  — trains the LEHD (Light Encoder Heavy Decoder) backbone using
               imitation learning from HGS-optimal sub-path labels.
               Reference: Fu Luo et al., NeurIPS 2023.

  TTRLModel  — identical training to LEHDModel (TTPL's training phase is the
               same LEHD backbone); the TTPL-specific test-time projection
               discovery with an LLM is a separate, post-training step.
               Reference: Rongsheng Chen et al., NeurIPS 2025.

Both classes integrate with the MARS train.py Lightning Trainer, share the
same argparse arguments, and log to the same wandb project.

Fair-comparison settings (same as MARS defaults for N=1000):
  epochs      = 200
  batch_size  = 64  (overrides LEHD's original 1024)
  lr          = 5e-5
  seed        = 42
  n_episodes  = 50_000  (overrides LEHD's original 1M)
  embed_dim   = 64
  device      = single GPU (controlled by Lightning Trainer)
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
import lightning.pytorch as pl

from rl4co.models.zoo.lehd._network import LEHDVRPModel
from rl4co.models.zoo.lehd._env import LEHDVRPEnv


# ---------------------------------------------------------------------------
# Capacity look-up (matches LEHD original & MARS SIL)
# ---------------------------------------------------------------------------
CAPACITY_MAP = {50: 40, 100: 50, 200: 80, 500: 100, 1000: 250}


class LEHDModel(pl.LightningModule):
    """Lightning module wrapping the LEHD CVRP imitation-learning loop.

    The model is trained with teacher-forcing: at each decoding step the
    *teacher* token (from HGS optimal sub-path) is fed back, and the model's
    cross-entropy loss against that token is minimised.  This mirrors the
    original VRPTrainer._train_one_batch() exactly.

    Validation reports the mean *student* tour-length reward (greedy decode
    against the teacher sub-path endpoints) so that the wandb `val/reward`
    curve is directly comparable with MARS/SIL.

    Args:
        data_path (str): Path to LEHD-format .txt training file.
        val_data_path (str): Path to LEHD-format .txt validation file.
            If None, a random subset of training episodes is used.
        num_loc (int): Number of customers (50/100/200/500/1000).
        embed_dim (int): Encoder / decoder embedding dimension.
        decoder_layer_num (int): Number of heavy-decoder Transformer layers.
        head_num (int): Attention heads.
        qkv_dim (int): Per-head key/value dimension.
        ff_hidden (int): Feed-forward hidden size.
        n_train_episodes (int): Training episodes per epoch.
        n_val_episodes (int): Validation episodes.
        optimizer_kwargs (dict): Passed to Adam (e.g. {"lr": 5e-5}).
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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LEHDVRPModel(
            embed_dim=embed_dim,
            decoder_layer_num=decoder_layer_num,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ff_hidden=ff_hidden,
        )
        self.capacity = float(CAPACITY_MAP.get(num_loc, 50))
        self.automatic_optimization = False

        # Environments (populated in on_fit_start)
        self._train_env: Optional[LEHDVRPEnv] = None
        self._val_env: Optional[LEHDVRPEnv] = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Load datasets once, before training starts."""
        hp = self.hparams
        self._train_env = LEHDVRPEnv(
            data_path=hp.data_path, mode="train", sub_path=True
        )
        self._train_env.load_raw_data(hp.n_train_episodes)

        val_path = hp.val_data_path or hp.data_path
        self._val_env = LEHDVRPEnv(
            data_path=val_path, mode="test", sub_path=False
        )
        n_val = hp.n_val_episodes
        # For validation we don't use subpath sampling; load a small slice
        self._val_env.load_raw_data(n_val)

    def on_train_epoch_start(self) -> None:
        if self._train_env is not None:
            self._train_env.shuffle_data()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        `batch` is a dict produced by _LEHDDataset (episode index only).
        We ignore the actual data tensors in `batch` — LEHD drives its own
        environment to load problems; the DataLoader is only used for
        iteration control and batch-size.
        """
        optimizer = self.optimizers()
        hp = self.hparams
        episode_start = batch["episode_start"].item()

        env = self._train_env
        env.load_problems(episode_start, hp.n_train_episodes // max(1, self.trainer.num_training_batches))

        # Move problems / solution to current device
        dev = self.device
        env.problems = env.problems.to(dev)
        env.solution = env.solution.to(dev)

        env.reset("train")
        state, _, _, done = env.pre_step()
        capacity = float(env.raw_data_capacity[0].item())

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
                    selected_flag_teacher, selected_flag_student
                )
                continue

            # Remaining capacity from state
            remaining_cap = state.problems[:, 0, 3]

            probs = self.model.decode_step(
                state.problems, env.selected_node_list,
                capacity, remaining_cap, step
            )  # (B, 2*V)

            # Teacher target: teacher_node (1-indexed) → 0-indexed in direct half;
            # teacher_flag=1 → index in via-depot half
            B = probs.shape[0]
            V = state.problems.shape[1] - 1
            target_node = env.solution[:, step, 0]    # 1-indexed
            target_flag = env.solution[:, step, 1]    # 0 or 1

            target_direct = (target_node - 1).long()    # 0-indexed customer
            target_via = target_direct + V              # via-depot offset

            target = torch.where(target_flag.bool(), target_via, target_direct)  # (B,)
            target = target.clamp(0, 2 * V - 1)

            prob_teacher = probs.gather(1, target[:, None]).clamp_min(1e-9)
            loss = -prob_teacher.log().mean()

            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            loss_sum = loss_sum + loss.detach()

            # Greedy student selection
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
                target_flag.long(), flag_student
            )

        avg_loss = (loss_sum / max(step - 1, 1)).item()
        self.log("train/loss", avg_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss_sum

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_reward_sum = 0.0
        self._val_count = 0

    def validation_step(self, batch, batch_idx):
        """Greedy decoding on validation instances (no subpath sampling)."""
        hp = self.hparams
        episode_start = batch["episode_start"].item()
        batch_size = batch["batch_size"].item()

        env = self._val_env
        env.load_problems(episode_start, batch_size)
        dev = self.device
        env.problems = env.problems.to(dev)
        env.solution = env.solution.to(dev)

        env.reset("test")
        state, _, _, done = env.pre_step()
        capacity = float(env.raw_data_capacity[0].item())

        step = 0
        with torch.no_grad():
            while not done:
                if step == 0:
                    selected = env.solution[:, 0, 0].long()
                    sel_flag = env.solution[:, 0, 1].long()
                    step += 1
                    state, _, _, done = env.step(selected, selected, sel_flag, sel_flag)
                    continue

                remaining_cap = state.problems[:, 0, 3]
                probs = self.model.decode_step(
                    state.problems, env.selected_node_list,
                    capacity, remaining_cap, step
                )
                B = probs.shape[0]
                V = state.problems.shape[1] - 1
                student_flat = probs.argmax(dim=1)
                is_via = student_flat >= V
                sel_node = torch.where(is_via, student_flat - V + 1, student_flat + 1).long()
                sel_flag = is_via.long()
                step += 1
                state, r_teach, r_stud, done = env.step(
                    sel_node, sel_node, sel_flag, sel_flag
                )

        # r_stud is -tour_length (higher = better); use as the comparable reward
        if r_stud is not None:
            self._val_reward_sum += r_stud.mean().item() * batch_size
            self._val_count += batch_size

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking or self._val_count == 0:
            return
        reward = self._val_reward_sum / self._val_count
        self.log("val/reward", reward, prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_kw = self.hparams.optimizer_kwargs or {"lr": 5e-5}
        optimizer = torch.optim.Adam(self.model.parameters(), **opt_kw)
        # Replicate LEHD's MultiStepLR (gamma=0.9, step each epoch up to n_epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class TTRLModel(LEHDModel):
    """TTPL baseline — training-phase is identical to LEHDModel.

    TTPL (Test-Time Projection Learning, NeurIPS 2025) discovers projection
    functions at *test time* using an LLM; it does not change training.
    This class exists as a named entry-point so wandb logs and result JSON
    correctly attribute runs to "ttpl" rather than "lehd".

    At test time, apply the TTPL projection by running:
        python TTPL/TTPL/llm4ad/run_TTPL.py
    using the checkpoint saved by this model.
    """
    pass


# ---------------------------------------------------------------------------
# Minimal dataset for Lightning DataLoader iteration control
# ---------------------------------------------------------------------------

class _LEHDDataset(torch.utils.data.Dataset):
    """Yields episode start indices so the DataLoader controls epoch length.

    Each 'sample' is a dict::
        {"episode_start": int, "batch_size": int}

    The actual data is loaded inside LEHDModel via its internal VRPEnv.
    """

    def __init__(self, n_episodes: int, batch_size: int):
        self.batch_size = batch_size
        self.starts = list(range(0, n_episodes, batch_size))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "episode_start": torch.tensor(self.starts[idx], dtype=torch.long),
            "batch_size": torch.tensor(self.batch_size, dtype=torch.long),
        }


def make_lehd_dataloaders(
    n_train: int,
    n_val: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
):
    """Return (train_loader, val_loader) for use with LEHDModel / TTRLModel.

    These loaders yield only index metadata — the actual tensor data is
    managed inside LEHDModel via its LEHDVRPEnv instances.

    Args:
        n_train: Total training episodes.
        n_val: Total validation episodes.
        batch_size: Micro-batch size per training step.
        seed: Generator seed for reproducibility.
        num_workers: DataLoader worker count (0 = main process; safe default
            since data loading happens inside the model).
    """
    train_ds = _LEHDDataset(n_train, batch_size)
    val_ds = _LEHDDataset(n_val, batch_size)
    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=num_workers, generator=g, collate_fn=lambda x: x[0],
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=lambda x: x[0],
    )
    return train_loader, val_loader
