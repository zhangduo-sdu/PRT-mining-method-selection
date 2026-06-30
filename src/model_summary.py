"""Print model size and attention-mask summary."""

from __future__ import annotations

import argparse

import torch

from attention_mask import build_allowed_attention_matrix
from model import PhysicsRegularizedTransformer
from utils import load_yaml


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Summarize PRT model and physics mask.")
    parser.add_argument("--config", type=str, default="configs/prt_config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    feature_order = cfg["feature_order"]
    mcfg = cfg["model"]

    model = PhysicsRegularizedTransformer(
        feature_order=feature_order,
        d_model=int(mcfg.get("d_model", 64)),
        n_heads=int(mcfg.get("n_heads", 4)),
        d_ff=int(mcfg.get("d_ff", 128)),
        dropout=float(mcfg.get("dropout", 0.1)),
    )

    allowed = build_allowed_attention_matrix(feature_order)
    n_allowed = int(allowed.sum())
    n_total = int(allowed.size)

    print("Model: PhysicsRegularizedTransformer")
    print(f"Feature dimension: {len(feature_order)}")
    print(f"d_model: {mcfg.get('d_model', 64)}")
    print(f"n_heads: {mcfg.get('n_heads', 4)}")
    print(f"d_ff: {mcfg.get('d_ff', 128)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Allowed attention connections: {n_allowed}/{n_total}")


if __name__ == "__main__":
    main()
