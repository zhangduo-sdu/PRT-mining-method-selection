"""Optional inner-fold hyperparameter search for the PRT model.

The manuscript reports nested cross-validation for model selection. The normal
review-check command uses the selected configuration for speed, while this
module exposes the inner-grid logic for reviewers who want to inspect or rerun
the selection procedure.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Dict, Iterable, List

import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from metrics import summarize_predictions
from trainer import train_single_fold


def iter_grid_configs(base_config: Dict) -> Iterable[Dict]:
    """Yield configuration dictionaries from the hyperparameter grid."""
    grid = base_config.get("hyperparameter_grid", {})
    if not grid:
        yield deepcopy(base_config)
        return

    keys = list(grid)
    for values in product(*[grid[k] for k in keys]):
        cfg = deepcopy(base_config)
        for key, value in zip(keys, values):
            if key in {"d_model", "n_heads", "dropout"}:
                cfg["model"][key] = value
            elif key == "learning_rate":
                cfg["training"]["learning_rate"] = value
            elif key == "batch_size":
                cfg["training"]["batch_size"] = value
            elif key == "lambda_physics":
                cfg["loss"]["lambda_physics"] = value
            elif key == "lambda_entropy":
                cfg["loss"]["lambda_entropy"] = value
            else:
                raise ValueError(f"Unsupported grid key: {key}")
        yield cfg


def run_inner_grid_search(
    train_df: pd.DataFrame,
    base_config: Dict,
    device: torch.device,
    seed: int = 42,
    n_inner_splits: int = 5,
    max_candidates: int | None = None,
) -> tuple[Dict, pd.DataFrame]:
    """Evaluate candidate configurations on case-grouped inner folds.

    Parameters
    ----------
    train_df:
        Outer-training dataframe containing complete candidate-method groups.
    base_config:
        Configuration dictionary with a ``hyperparameter_grid`` section.
    device:
        PyTorch device.
    seed:
        Seed offset used for fold training.
    n_inner_splits:
        Maximum number of grouped inner folds.
    max_candidates:
        Optional cap for quick demonstrations of the grid-search machinery.
    """
    case_col = base_config["columns"]["case_id"]
    groups = train_df[case_col].to_numpy()
    unique_groups = train_df[case_col].nunique()
    n_splits = min(int(n_inner_splits), int(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two independent cases are required for inner CV.")

    splitter = GroupKFold(n_splits=n_splits)
    rows: List[Dict] = []
    best_config = None
    best_mae = float("inf")

    for candidate_id, cfg in enumerate(iter_grid_configs(base_config), start=1):
        if max_candidates is not None and candidate_id > int(max_candidates):
            break

        fold_predictions = []
        for fold_id, (tr_idx, val_idx) in enumerate(splitter.split(train_df, groups=groups), start=1):
            result = train_single_fold(
                outer_train_df=train_df.iloc[tr_idx].reset_index(drop=True),
                outer_test_df=train_df.iloc[val_idx].reset_index(drop=True),
                config=cfg,
                device=device,
                seed=seed + candidate_id * 100 + fold_id,
            )
            pred = result["predictions"].copy()
            pred["inner_fold"] = fold_id
            fold_predictions.append(pred)

        pred_df = pd.concat(fold_predictions, ignore_index=True)
        metrics = summarize_predictions(pred_df, case_col=case_col)
        row = {
            "candidate_id": candidate_id,
            "d_model": cfg["model"]["d_model"],
            "n_heads": cfg["model"]["n_heads"],
            "dropout": cfg["model"]["dropout"],
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["batch_size"],
            "lambda_physics": cfg["loss"]["lambda_physics"],
            "lambda_entropy": cfg["loss"]["lambda_entropy"],
            **metrics,
        }
        rows.append(row)

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_config = cfg

    if best_config is None:
        raise RuntimeError("No grid-search candidates were evaluated.")

    return best_config, pd.DataFrame(rows)
