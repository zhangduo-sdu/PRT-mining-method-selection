"""Run grouped validation and external testing for the PRT model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metrics import (
    bootstrap_metric_ci,
    save_attention_heatmap,
    save_parity_plot,
    summarize_predictions,
)
from trainer import train_final_model, train_single_fold
from utils import ensure_dir, load_yaml, save_json, select_device, set_seed, to_serializable


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRT reproducibility experiment.")
    parser.add_argument("--data", type=str, required=True, help="Input CSV file.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--max-outer-folds", type=int, default=None, help="Run only first N outer folds for smoke testing.")
    parser.add_argument("--bootstrap-iters", type=int, default=1000, help="Case-level bootstrap iterations for metric CIs.")
    parser.add_argument("--no-external", action="store_true", help="Skip external generalization testing.")
    return parser.parse_args()


def validate_input(df: pd.DataFrame, config):
    required = [
        config["columns"]["case_id"],
        config["columns"]["method_id"],
        config["columns"]["target"],
    ] + config["feature_order"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")


def main():
    args = parse_args()
    config = load_yaml(args.config)

    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)

    outdir = ensure_dir(args.outdir)
    set_seed(int(config.get("seed", 42)), num_threads=config.get("training", {}).get("num_threads", 1))
    device = select_device(config["training"].get("device", "auto"))

    df = pd.read_csv(args.data)
    validate_input(df, config)

    case_col = config["columns"]["case_id"]
    split_col = config["columns"].get("split", "split")
    feature_order = config["feature_order"]

    if split_col in df.columns:
        real_df = df[df[split_col] == "real_pool"].reset_index(drop=True)
        external_df = df[df[split_col] == "external_generalization"].reset_index(drop=True)
    else:
        real_df = df.copy().reset_index(drop=True)
        external_df = pd.DataFrame()

    if real_df[case_col].nunique() < 2:
        raise ValueError("At least two independent real cases are required for grouped validation.")

    all_predictions = []
    fold_metrics = []
    attention_matrices = []

    case_ids = list(real_df[case_col].drop_duplicates())
    if args.max_outer_folds is not None:
        case_ids = case_ids[: int(args.max_outer_folds)]

    for fold_id, held_case in enumerate(case_ids, start=1):
        outer_test_df = real_df[real_df[case_col] == held_case].reset_index(drop=True)
        outer_train_df = real_df[real_df[case_col] != held_case].reset_index(drop=True)

        result = train_single_fold(
            outer_train_df=outer_train_df,
            outer_test_df=outer_test_df,
            config=config,
            device=device,
            seed=int(config.get("seed", 42)) + fold_id,
        )

        pred_df = result["predictions"]
        pred_df["outer_fold"] = fold_id
        all_predictions.append(pred_df)
        attention_matrices.append(result["mean_attention"])

        metrics = summarize_predictions(pred_df, case_col=case_col)
        metrics["outer_fold"] = fold_id
        metrics["held_case"] = held_case
        metrics["best_val_loss"] = result["best_val_loss"]
        metrics["n_train_rows_after_augmentation"] = result["n_train_rows_after_augmentation"]
        fold_metrics.append(metrics)

        print(
            f"[Fold {fold_id:02d}] held_case={held_case} "
            f"rank_acc={metrics['rank_accuracy']:.3f} "
            f"mae={metrics['mae']:.4f}"
        )

    loocv_pred = pd.concat(all_predictions, ignore_index=True)
    loocv_pred.to_csv(outdir / "loocv_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(outdir / "fold_metrics.csv", index=False)

    loocv_summary = summarize_predictions(loocv_pred, case_col=case_col)

    if config["outputs"].get("save_parity_plot", True):
        save_parity_plot(loocv_pred, outdir / "parity_plot.png")

    if len(attention_matrices) > 0 and config["outputs"].get("save_attention_matrix", True):
        mean_attention = np.mean(np.stack(attention_matrices, axis=0), axis=0)
        pd.DataFrame(mean_attention, index=feature_order, columns=feature_order).to_csv(
            outdir / "mean_attention_matrix.csv"
        )
        save_attention_heatmap(mean_attention, feature_order, outdir / "mean_attention_matrix.png")

    external_summary = {}
    if (not args.no_external) and external_df is not None and len(external_df) > 0:
        final_result = train_final_model(
            train_df=real_df,
            external_df=external_df,
            config=config,
            device=device,
            seed=int(config.get("seed", 42)) + 1000,
        )
        ext_pred = final_result["predictions"]
        ext_pred.to_csv(outdir / "external_predictions.csv", index=False)
        external_summary = summarize_predictions(ext_pred, case_col=case_col)

    summary = {
        "device": str(device),
        "n_real_cases": int(real_df[case_col].nunique()),
        "n_external_cases": int(external_df[case_col].nunique()) if external_df is not None else 0,
        "loocv_summary": {k: to_serializable(v) for k, v in loocv_summary.items()},
        "loocv_bootstrap_ci": {
            k: to_serializable(v)
            for k, v in bootstrap_metric_ci(
                loocv_pred,
                case_col=case_col,
                n_boot=int(args.bootstrap_iters),
                seed=int(config.get("seed", 42)) + 2000,
            ).items()
        },
        "external_summary": {k: to_serializable(v) for k, v in external_summary.items()},
        "config_epochs": int(config["training"]["epochs"]),
        "note": (
            "Non-confidential execution-example results are for code verification "
            "only and are not intended to reproduce controlled real-mine results exactly."
        ),
    }
    if external_summary:
        summary["external_bootstrap_ci"] = {
            k: to_serializable(v)
            for k, v in bootstrap_metric_ci(
                ext_pred,
                case_col=case_col,
                n_boot=int(args.bootstrap_iters),
                seed=int(config.get("seed", 42)) + 3000,
            ).items()
        }
    save_json(summary, outdir / "summary.json")

    print("\nSummary")
    print(pd.Series(summary["loocv_summary"]).to_string())
    if external_summary:
        print("\nExternal summary")
        print(pd.Series(summary["external_summary"]).to_string())
    print(f"\nOutputs saved to: {outdir}")


if __name__ == "__main__":
    main()
