"""Run a demonstrable inner-fold grid search for PRT hyperparameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grid_search import run_inner_grid_search
from utils import ensure_dir, load_yaml, select_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run optional PRT inner-fold grid search.")
    parser.add_argument("--data", type=str, required=True, help="Input CSV file.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch override for review-time grid checks.")
    parser.add_argument("--inner-folds", type=int, default=5, help="Maximum number of grouped inner folds.")
    parser.add_argument("--max-candidates", type=int, default=6, help="Candidate cap for quick verification.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    config["training"]["epochs"] = int(args.epochs)

    outdir = ensure_dir(args.outdir)
    set_seed(int(config.get("seed", 42)), num_threads=config.get("training", {}).get("num_threads", 1))
    device = select_device(config["training"].get("device", "auto"))

    df = pd.read_csv(args.data)
    split_col = config["columns"].get("split", "split")
    if split_col in df.columns:
        df = df[df[split_col] == "real_pool"].reset_index(drop=True)

    _, search_results = run_inner_grid_search(
        train_df=df,
        base_config=config,
        device=device,
        seed=int(config.get("seed", 42)),
        n_inner_splits=int(args.inner_folds),
        max_candidates=int(args.max_candidates),
    )
    out_path = Path(outdir) / "inner_grid_search_results.csv"
    search_results.to_csv(out_path, index=False)
    print(search_results.sort_values("mae").head().to_string(index=False))
    print(f"\nGrid-search results saved to: {out_path}")


if __name__ == "__main__":
    main()
