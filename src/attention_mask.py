"""Physics-regularized attention-mask construction.

The feature order is assumed to be:

X1, X2, X3, X4, X5, X6, X7,
X8, X9, X10, X11,
sigma1_mpa, delta_mm, Ap

The mask implements four domain-guided connection rules:
1. quantitative indicators can attend to quantitative indicators;
2. qualitative indicators can attend to qualitative indicators;
3. simulation features can attend to all variables and all variables can attend
   to simulation features;
4. the safety index X8 is explicitly coupled with maximum principal stress sigma1.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def build_feature_types(feature_order: List[str]) -> Dict[str, str]:
    """Return feature-type mapping for the 14-dimensional input vector."""
    types = {}
    for name in feature_order:
        if name.startswith("X"):
            idx = int(name[1:])
            if 1 <= idx <= 7:
                types[name] = "quantitative"
            elif 8 <= idx <= 11:
                types[name] = "qualitative"
            else:
                raise ValueError(f"Unsupported feature name: {name}")
        elif name in {"sigma1_mpa", "delta_mm", "Ap"}:
            types[name] = "simulation"
        else:
            raise ValueError(f"Unsupported feature name: {name}")
    return types


def build_allowed_attention_matrix(feature_order: List[str]) -> np.ndarray:
    """Build a binary allowed-attention matrix.

    Returns
    -------
    allowed : np.ndarray, shape [n_features, n_features]
        allowed[i, j] = 1 means token i is allowed to attend to token j.
    """
    n = len(feature_order)
    if n != 14:
        raise ValueError("The manuscript implementation expects 14 input features.")

    feature_types = build_feature_types(feature_order)
    allowed = np.zeros((n, n), dtype=np.int64)

    # Always allow self-attention.
    np.fill_diagonal(allowed, 1)

    for i, fi in enumerate(feature_order):
        for j, fj in enumerate(feature_order):
            ti = feature_types[fi]
            tj = feature_types[fj]

            # Intra-type quantitative and qualitative blocks.
            if ti == tj and ti in {"quantitative", "qualitative"}:
                allowed[i, j] = 1

            # Simulation-to-all and all-to-simulation paths.
            if ti == "simulation" or tj == "simulation":
                allowed[i, j] = 1

    # Explicit safety--stress coupling X8 <-> sigma1_mpa.
    i_x8 = feature_order.index("X8")
    i_s1 = feature_order.index("sigma1_mpa")
    allowed[i_x8, i_s1] = 1
    allowed[i_s1, i_x8] = 1

    return allowed


def build_torch_attention_mask(feature_order: List[str], device=None) -> torch.Tensor:
    """Return a PyTorch boolean attention mask.

    PyTorch attention convention used here:
    - True means the position is blocked.
    - False means the position is allowed.
    """
    allowed = build_allowed_attention_matrix(feature_order)
    blocked = allowed == 0
    mask = torch.tensor(blocked, dtype=torch.bool, device=device)
    return mask


def describe_allowed_connections(feature_order: List[str]) -> List[Tuple[str, str]]:
    """Return a readable list of allowed directed connections."""
    allowed = build_allowed_attention_matrix(feature_order)
    pairs = []
    for i, fi in enumerate(feature_order):
        for j, fj in enumerate(feature_order):
            if allowed[i, j] == 1:
                pairs.append((fi, fj))
    return pairs
