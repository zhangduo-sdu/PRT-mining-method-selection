"""Compound loss for physics-regularized transformer training."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def attention_entropy_loss(attention: Optional[torch.Tensor]) -> torch.Tensor:
    """Compute normalized attention entropy.

    Parameters
    ----------
    attention:
        Attention weights with shape [batch, heads, n_tokens, n_tokens] or
        [batch, n_tokens, n_tokens]. Masked entries should already be near zero.

    Returns
    -------
    torch.Tensor
        Scalar entropy regularization term.
    """
    if attention is None:
        return torch.tensor(0.0)

    eps = 1e-8
    p = torch.clamp(attention, min=eps, max=1.0)
    entropy = -(p * torch.log(p)).sum(dim=-1)

    # Normalize by log(number of keys) to make the scale more stable.
    n_keys = attention.shape[-1]
    entropy = entropy / torch.log(torch.tensor(float(n_keys), device=attention.device))
    return entropy.mean()


def mechanical_feasibility_penalty(
    y_pred: torch.Tensor,
    raw_features: torch.Tensor,
    feature_order,
    loss_cfg: Dict,
) -> torch.Tensor:
    """Penalize high predicted scores for mechanically unfavorable states.

    This is a soft regularizer, not a strict PDE residual. It discourages
    assigning high superiority scores to samples that exceed engineering limits.

    Parameters
    ----------
    y_pred:
        Predicted superiority scores, shape [batch].
    raw_features:
        Raw, unnormalized input features, shape [batch, n_features].
    feature_order:
        Ordered feature names.
    loss_cfg:
        Loss configuration dictionary.
    """
    idx_sigma = feature_order.index("sigma1_mpa")
    idx_delta = feature_order.index("delta_mm")
    idx_ap = feature_order.index("Ap")

    sigma1 = raw_features[:, idx_sigma]
    delta = raw_features[:, idx_delta]
    ap = raw_features[:, idx_ap]

    stress_limit = float(loss_cfg.get("stress_limit_mpa", 18.19))
    displacement_limit = float(loss_cfg.get("displacement_limit_mm", 40.0))
    plastic_limit = float(loss_cfg.get("plastic_zone_limit", 0.30))

    alpha_mc = float(loss_cfg.get("alpha_mc", 0.5))
    alpha_disp = float(loss_cfg.get("alpha_disp", 0.5))
    alpha_plastic = float(loss_cfg.get("alpha_plastic", 0.2))

    stress_violation = F.relu(sigma1 / stress_limit - 1.0) ** 2
    disp_violation = F.relu(delta / displacement_limit - 1.0) ** 2
    plastic_violation = F.relu(ap / plastic_limit - 1.0) ** 2

    violation = (
        alpha_mc * stress_violation
        + alpha_disp * disp_violation
        + alpha_plastic * plastic_violation
    )

    # High scores on violated states are penalized more strongly.
    return torch.mean((y_pred.view(-1) ** 2) * violation)


def compound_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    raw_features: torch.Tensor,
    attention: Optional[torch.Tensor],
    feature_order,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    """Return total loss and its components."""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    pred_loss = F.mse_loss(y_pred, y_true)

    phy_loss = mechanical_feasibility_penalty(
        y_pred=y_pred,
        raw_features=raw_features,
        feature_order=feature_order,
        loss_cfg=loss_cfg,
    )

    ent_loss = attention_entropy_loss(attention)

    lambda_physics = float(loss_cfg.get("lambda_physics", 0.1))
    lambda_entropy = float(loss_cfg.get("lambda_entropy", 0.05))

    total = pred_loss + lambda_physics * phy_loss + lambda_entropy * ent_loss

    return {
        "total": total,
        "prediction": pred_loss.detach(),
        "physics": phy_loss.detach(),
        "entropy": ent_loss.detach(),
    }
