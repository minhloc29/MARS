"""Reader for the cached slot-NCO datasets produced by generate_slot_dataset.py.

The generator writes a dict to disk per split with keys:

    locs (B, N, 2), depot (B, 2), demand (B, N)  [demand pre-normalized by capacity]
    capacity (B, 1), d_ins_idx (B, N, k) int16, d_ins_val (B, N, k) float32
    format_version ("sparse_v2"), method

:class:`SlotDataset` loads one such file and wraps it as a
:class:`torch.utils.data.Dataset`, and :func:`make_dataloader` builds the
training/validation :class:`DataLoader` that POMOSlot/AMSlot consume.

Keeping this beside the generator (rather than inlined in train.py) gives every
consumer — train.py, test.py, eval scripts — a single shared reader, so the
on-disk schema only has to be defined once.

The ``variant`` argument mirrors the POMOSlot/AMSlot ablation switch. d_ins is
needed only for Variant D, so it is loaded conditionally to save memory on the
other variants (which still re-read the file's format header).
"""
from __future__ import annotations

import hashlib

from pathlib import Path

import torch


class SlotDataset(torch.utils.data.Dataset):
    """Wraps cached .pt files from generate_slot_dataset.py (sparse_v2)."""

    def __init__(self, filepath: str | Path, variant: str = "D",
                 max_instances: int | None = None,
                 include_instance_id: bool = False):
        data = torch.load(filepath, map_location="cpu", weights_only=False)

        # Format version sanity check (reject old dense d_ins)
        fmt = data.get("format_version", None)
        if fmt is None:
            if "d_ins" in data:
                raise RuntimeError(
                    f"Old dense d_ins format detected in {filepath}.\n"
                    "Please regenerate datasets using the updated generate_slot_dataset.py "
                    "which produces sparse_v2 format (d_ins_idx + d_ins_val).\n"
                    "Command: python -m rl4co.data.generate_slot_dataset --num_locs N --dist DIST ..."
                )
        elif fmt != "sparse_v2":
            raise RuntimeError(
                f"Unknown dataset format_version: '{fmt}' in {filepath}")

        self.locs = data["locs"]     # (N_inst, N, 2)
        self.depot = data["depot"]    # (N_inst, 2)
        self.demand = data["demand"]   # (N_inst, N)
        self.capacity = data.get("capacity", None)
        # d_ins cost-method tag stamped by the generator; None for legacy datasets.
        self.method: str | None = data.get("method", None)

        # Sparse d_ins only needed for Variant D; load it conditionally to keep
        # the other ablation variants memory-light.
        needs_dins = variant == "D"
        self.d_ins_idx = data.get(
            "d_ins_idx", None) if needs_dins else None  # (N_inst,N,k) int16
        self.d_ins_val = data.get(
            "d_ins_val", None) if needs_dins else None  # (N_inst,N,k) float32
        self.variant = variant
        self.include_instance_id = include_instance_id

        if max_instances is not None:
            self.locs = self.locs[:max_instances]
            self.depot = self.depot[:max_instances]
            self.demand = self.demand[:max_instances]
            if self.capacity is not None:
                self.capacity = self.capacity[:max_instances]
            if self.d_ins_idx is not None:
                self.d_ins_idx = self.d_ins_idx[:max_instances]
            if self.d_ins_val is not None:
                self.d_ins_val = self.d_ins_val[:max_instances]

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        item = {
            "locs":   self.locs[idx],    # (N, 2)
            "depot":  self.depot[idx],   # (2,)
            "demand": self.demand[idx],  # (N,)
        }
        if self.capacity is not None:
            item["capacity"] = self.capacity[idx]   # (1,)
        if self.d_ins_idx is not None:
            item["d_ins_idx"] = self.d_ins_idx[idx]  # (N, k) int16
        if self.d_ins_val is not None:
            item["d_ins_val"] = self.d_ins_val[idx]  # (N, k) float32
        if self.include_instance_id:
            item["instance_id"] = torch.tensor(idx, dtype=torch.long)
        return item

    def signature(self):
        """Identify the actual cached CVRP inputs before reusing SIL labels."""
        digest = hashlib.sha256()
        for tensor in (self.locs, self.depot, self.demand):
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(memoryview(tensor.contiguous().numpy()).cast("B"))
        return digest.hexdigest()


def collate_fn(batch: list[dict]) -> dict:
    """Collate dicts -> batched dict; shared_step converts to TensorDict internally."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def make_dataloader(
    filepath: str | Path,
    batch_size: int,
    shuffle: bool,
    max_instances: int | None = None,
    variant: str = "D",
    include_instance_id: bool = False,
    seed: int = 42,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader over one cached split .pt file."""
    ds = SlotDataset(filepath, variant=variant, max_instances=max_instances,
                     include_instance_id=include_instance_id)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_fn,
    )
