"""Training and evaluation routines."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from attention_mask import build_torch_attention_mask
from augmentation import augment_training_dataframe
from losses import compound_loss
from model import PhysicsRegularizedTransformer
from preprocessing import TabularPreprocessor


def make_tensors(df: pd.DataFrame, x_norm, feature_order, target_col, device):
    """Create normalized/raw feature and target tensors."""
    x_norm_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)
    raw_tensor = torch.tensor(df[feature_order].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(df[target_col].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    return x_norm_tensor, raw_tensor, y_tensor


def split_inner_train_validation(
    train_df: pd.DataFrame,
    config: Dict,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split outer-training cases into inner training and validation sets."""
    case_col = config["columns"]["case_id"]
    val_fraction = float(config["training"].get("inner_validation_fraction", 0.20))
    groups = train_df[case_col].to_numpy()

    unique_groups = train_df[case_col].nunique()
    if unique_groups < 4:
        # Fallback for very small smoke-test subsets.
        return train_df.copy(), train_df.copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(train_df, groups=groups))
    return train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)


def train_single_fold(
    outer_train_df: pd.DataFrame,
    outer_test_df: pd.DataFrame,
    config: Dict,
    device: torch.device,
    seed: int,
) -> Dict:
    """Train on one outer fold and evaluate on the held-out case."""
    rng = np.random.default_rng(seed)

    feature_order = config["feature_order"]
    target_col = config["columns"]["target"]

    inner_train_df, inner_val_df = split_inner_train_validation(outer_train_df, config, seed=seed)

    # Crucial leakage-prevention step: augmentation is applied only to inner training rows.
    inner_train_aug_df = augment_training_dataframe(inner_train_df, config=config, rng=rng)

    preprocessor = TabularPreprocessor(
        quantitative_cols=config["features"]["quantitative"],
        qualitative_cols=config["features"]["qualitative"],
        simulation_cols=config["features"]["simulation"],
        feature_order=feature_order,
        normalization=config["normalization"],
    )
    x_train = preprocessor.fit_transform(inner_train_aug_df)
    x_val = preprocessor.transform(inner_val_df)
    x_test = preprocessor.transform(outer_test_df)

    x_train_t, raw_train_t, y_train_t = make_tensors(
        inner_train_aug_df, x_train, feature_order, target_col, device
    )
    x_val_t, raw_val_t, y_val_t = make_tensors(
        inner_val_df, x_val, feature_order, target_col, device
    )
    x_test_t, raw_test_t, y_test_t = make_tensors(
        outer_test_df, x_test, feature_order, target_col, device
    )

    train_loader = DataLoader(
        TensorDataset(x_train_t, raw_train_t, y_train_t),
        batch_size=int(config["training"].get("batch_size", 8)),
        shuffle=True,
    )

    model_cfg = config["model"]
    model = PhysicsRegularizedTransformer(
        feature_order=feature_order,
        d_model=int(model_cfg.get("d_model", 64)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        d_ff=int(model_cfg.get("d_ff", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    attn_mask = build_torch_attention_mask(feature_order, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 0.001)),
        weight_decay=float(config["training"].get("weight_decay", 0.001)),
    )

    epochs = int(config["training"].get("epochs", 120))
    patience = int(config["training"].get("patience", 20))
    min_delta = float(config["training"].get("early_stopping_min_delta", 1e-6))

    best_state = None
    best_val = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, rawb, yb in train_loader:
            optimizer.zero_grad()
            pred, attn = model(xb, attn_mask)
            loss_dict = compound_loss(
                y_pred=pred,
                y_true=yb,
                raw_features=rawb,
                attention=attn,
                feature_order=feature_order,
                loss_cfg=config["loss"],
            )
            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred, val_attn = model(x_val_t, attn_mask)
            val_loss_dict = compound_loss(
                y_pred=val_pred,
                y_true=y_val_t,
                raw_features=raw_val_t,
                attention=val_attn,
                feature_order=feature_order,
                loss_cfg=config["loss"],
            )
            val_loss = float(val_loss_dict["total"].detach().cpu())

        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred, test_attn = model(x_test_t, attn_mask)

    pred_df = outer_test_df.copy()
    pred_df["pred_score"] = test_pred.detach().cpu().numpy()

    mean_attn = test_attn.detach().cpu().numpy().mean(axis=(0, 1))

    return {
        "model": model,
        "preprocessor": preprocessor,
        "predictions": pred_df,
        "history": pd.DataFrame(history),
        "mean_attention": mean_attn,
        "best_val_loss": best_val,
        "n_train_rows_after_augmentation": len(inner_train_aug_df),
    }


def train_final_model(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    config: Dict,
    device: torch.device,
    seed: int,
) -> Dict:
    """Train on the real-case pool and evaluate external generalization cases."""
    if external_df is None or len(external_df) == 0:
        return {}

    result = train_single_fold(
        outer_train_df=train_df,
        outer_test_df=external_df,
        config=config,
        device=device,
        seed=seed,
    )
    return result
