#!/usr/bin/env bash
set -e

python src/run_experiment.py \
  --data data/synthetic_benchmark.csv \
  --config configs/prt_config.yaml \
  --outdir outputs/full_run
