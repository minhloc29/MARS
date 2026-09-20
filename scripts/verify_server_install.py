"""Verify the imports needed by MARS and its integrated CVRP baselines."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import lightning
import tensordict
import torch
import torchrl
import wandb


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_dir))

    from rl4co.envs import CVRPEnv  # noqa: F401
    from rl4co.models.zoo.invit import INViT  # noqa: F401
    from rl4co.models.zoo.lehd import LEHDModel, TTRLModel  # noqa: F401
    from rl4co.models.zoo.pomo_slot import AMSlot, POMOSlot  # noqa: F401
    from rl4co.models.zoo.sil import SIL  # noqa: F401

    subprocess.run(
        [sys.executable, str(repo_dir / "train.py"), "--help"],
        cwd=repo_dir,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    print("MARS installation check passed")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  TorchRL:    {torchrl.__version__}")
    print(f"  TensorDict: {tensordict.__version__}")
    print(f"  Lightning:  {lightning.__version__}")
    print(f"  W&B:        {wandb.__version__}")
    print(f"  CUDA ready: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
