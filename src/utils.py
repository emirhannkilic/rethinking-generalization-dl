"""
utils.py — Shared utilities for the rethinking-generalization reproduction.

Provides device selection, reproducibility seeding, and JSON I/O helpers.
All path handling uses pathlib.Path throughout.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: CUDA → MPS → CPU.

    Prints the selected device so experiment logs are self-contained.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[utils] Selected device: {device}")
    return device


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all RNG seeds for full reproducibility.

    Sets:
      - Python built-in random
      - NumPy global RNG
      - PyTorch CPU and CUDA RNGs
      - CUDA deterministic / benchmark flags (if CUDA is available)

    Args:
        seed: Integer seed value. Should be unique per experimental run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Force deterministic CUDA ops at the cost of some performance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_results(path: str | Path, data: Any) -> None:
    """Serialise *data* to JSON at *path*, creating parent directories.

    Args:
        path: Destination file path (created if it doesn't exist).
        data: JSON-serialisable object (dict, list, …).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_results(path: str | Path) -> Any:
    """Deserialise JSON from *path*.

    Args:
        path: Source file path.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
