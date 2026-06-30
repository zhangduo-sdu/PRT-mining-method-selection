#!/usr/bin/env bash
set -e

python scripts/run_integrity_checks.py
python src/model_summary.py --config configs/prt_config.yaml
python src/run_experiment.py \
  --data data/synthetic_benchmark.csv \
  --config configs/prt_config.yaml \
  --outdir outputs/reviewer_check \
  --epochs 5
