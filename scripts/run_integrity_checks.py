"""Repository integrity checks for reviewer-facing PRT code."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from attention_mask import build_allowed_attention_matrix, build_torch_attention_mask
from model import PhysicsRegularizedTransformer
from preprocessing import TabularPreprocessor
from utils import load_yaml, set_seed


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    config = load_yaml(REPO_ROOT / "configs" / "prt_config.yaml")
    feature_order = config["feature_order"]
    assert len(feature_order) == 14, "Expected 14 input features."

    allowed = build_allowed_attention_matrix(feature_order)
    assert allowed.shape == (14, 14), "Attention matrix must be 14 x 14."
    assert int(allowed.sum()) == 140, "Unexpected number of allowed attention links."

    mask = build_torch_attention_mask(feature_order)
    assert mask.shape == (14, 14), "Torch attention mask must be 14 x 14."

    model_cfg = config["model"]
    model = PhysicsRegularizedTransformer(
        feature_order=feature_order,
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(model_cfg["dropout"]),
    )
    assert count_parameters(model) == 36993, "Unexpected trainable parameter count."

    df = pd.read_csv(REPO_ROOT / "data" / "non_confidential_execution_example.csv")
    required = ["case_id", "method_id", "split", "target_score", "is_optimal"] + feature_order
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"Missing required columns: {missing}"
    assert not df[required].isna().any().any(), "Input data contain missing values."

    for case_id, group in df.groupby("case_id"):
        methods = set(group["method_id"].astype(int).tolist())
        assert methods == {1, 2, 3}, f"Case {case_id} does not contain methods 1, 2, and 3."
        assert int(group["is_optimal"].sum()) == 1, f"Case {case_id} must have one optimal method."

    real_df = df[df["split"] == "real_pool"].reset_index(drop=True)
    assert real_df["case_id"].nunique() == 19, "Expected 19 real-pool cases."

    preprocessor = TabularPreprocessor(
        quantitative_cols=config["features"]["quantitative"],
        qualitative_cols=config["features"]["qualitative"],
        simulation_cols=config["features"]["simulation"],
        feature_order=feature_order,
        normalization=config["normalization"],
    )
    x = preprocessor.fit_transform(real_df.head(12))
    set_seed(int(config.get("seed", 42)), num_threads=1)
    with torch.no_grad():
        pred, attention = model(torch.tensor(x, dtype=torch.float32), mask)

    assert pred.shape == (12,), "Prediction shape mismatch."
    assert attention.shape == (12, int(model_cfg["n_heads"]), 14, 14), "Attention shape mismatch."
    assert torch.all((pred >= 0.0) & (pred <= 1.0)), "Predictions must be in [0, 1]."

    print("Integrity checks passed.")
    print("Input features: 14")
    print("Allowed attention connections: 140/196")
    print("Trainable parameters: 36,993")
    print("Real-pool cases: 19")


if __name__ == "__main__":
    main()
