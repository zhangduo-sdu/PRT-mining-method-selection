#!/usr/bin/env bash
set -e

python src/run_experiment.py \
  --data data/non_confidential_execution_example.csv \
  --config configs/prt_config.yaml \
  --outdir outputs/full_run
