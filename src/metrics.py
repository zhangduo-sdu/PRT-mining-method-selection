"""Evaluation metrics and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute row-level score-prediction metrics."""
    y = df["target_score"].to_numpy(dtype=float)
    pred = df["pred_score"].to_numpy(dtype=float)

    mae = mean_absolute_error(y, pred)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))

    if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1:
        rho = spearmanr(y, pred).correlation
        tau = kendalltau(y, pred).correlation
    else:
        rho = np.nan
        tau = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "spearman_rho": float(rho) if rho == rho else np.nan,
        "kendall_tau": float(tau) if tau == tau else np.nan,
    }


def case_rank_accuracy(df: pd.DataFrame, case_col: str = "case_id") -> float:
    """Compute case-level rank-order accuracy."""
    correct = 0
    total = 0

    for _, group in df.groupby(case_col):
        true_method = group.loc[group["target_score"].idxmax(), "method_id"]
        pred_method = group.loc[group["pred_score"].idxmax(), "method_id"]
        correct += int(true_method == pred_method)
        total += 1

    return float(correct / total) if total > 0 else float("nan")


def summarize_predictions(df: pd.DataFrame, case_col: str = "case_id") -> Dict[str, float]:
    """Return combined regression and ranking metrics."""
    out = regression_metrics(df)
    out["rank_accuracy"] = case_rank_accuracy(df, case_col=case_col)
    out["n_rows"] = int(len(df))
    out["n_cases"] = int(df[case_col].nunique())
    return out


def bootstrap_metric_ci(
    df: pd.DataFrame,
    case_col: str = "case_id",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Case-level bootstrap confidence intervals for prediction metrics.

    Resampling is performed at the independent mine-case level, not at the row
    level, so the three candidate methods belonging to one case stay together.
    """
    rng = np.random.default_rng(seed)
    case_ids = list(df[case_col].drop_duplicates())
    if len(case_ids) < 2 or n_boot <= 0:
        return {}

    records = []
    for _ in range(n_boot):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        parts = []
        for j, case_id in enumerate(sampled):
            part = df[df[case_col] == case_id].copy()
            part["__bootstrap_case_id"] = f"{case_id}__{j}"
            parts.append(part)
        boot_df = pd.concat(parts, ignore_index=True)
        records.append(summarize_predictions(boot_df, case_col="__bootstrap_case_id"))

    boot = pd.DataFrame(records)
    out: Dict[str, Dict[str, float]] = {}
    for key in ["mae", "rmse", "spearman_rho", "kendall_tau", "rank_accuracy"]:
        values = boot[key].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) == 0:
            continue
        out[key] = {
            "mean": float(values.mean()),
            "ci95_low": float(values.quantile(0.025)),
            "ci95_high": float(values.quantile(0.975)),
        }
    out["n_boot"] = {"value": int(n_boot)}
    return out


def save_parity_plot(df: pd.DataFrame, out_path: str | Path) -> None:
    """Save predicted-vs-ground-truth agreement plot."""
    out_path = Path(out_path)
    y = df["target_score"].to_numpy(dtype=float)
    pred = df["pred_score"].to_numpy(dtype=float)

    plt.figure(figsize=(5.2, 4.6), dpi=300)
    plt.scatter(y, pred, s=28, alpha=0.75)
    lo = min(y.min(), pred.min()) - 0.03
    hi = max(y.max(), pred.max()) + 0.03
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Ground-truth superiority score")
    plt.ylabel("Predicted superiority score")
    plt.title("Predicted versus ground-truth superiority scores")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_attention_heatmap(attn_matrix: np.ndarray, feature_order, out_path: str | Path) -> None:
    """Save mean attention matrix heatmap."""
    out_path = Path(out_path)
    plt.figure(figsize=(7.2, 6.2), dpi=300)
    plt.imshow(attn_matrix, aspect="auto")
    plt.xticks(range(len(feature_order)), feature_order, rotation=90, fontsize=7)
    plt.yticks(range(len(feature_order)), feature_order, fontsize=7)
    plt.colorbar(label="Mean attention weight")
    plt.title("Mean physics-regularized attention matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
