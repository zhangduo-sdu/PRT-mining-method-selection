#!/usr/bin/env bash
set -e

python src/run_experiment.py \
  --data data/synthetic_benchmark.csv \
  --config configs/prt_config.yaml \
  --outdir outputs/smoke_test \
  --epochs 5 \
  --max-outer-folds 2
