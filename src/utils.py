"""Utility functions for reproducible experiments."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int, num_threads: int | None = None) -> None:
    """Set random seeds and optionally limit PyTorch CPU threads.

    Small tabular transformer experiments can be slower when PyTorch uses too
    many CPU threads. Limiting the thread count improves reproducibility and
    makes the smoke test fast on ordinary review machines.
    """
    random.seed(seed)
    np.random.seed(seed)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic mode is preferred for review reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_device(device_config: str = "auto") -> torch.device:
    """Select CPU or CUDA device."""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as formatted JSON."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def to_serializable(value: Any) -> Any:
    """Convert NumPy/PyTorch scalar values to JSON-serializable Python objects."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    return value
