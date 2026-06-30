# Reviewer Reproducibility Checklist

The repository provides the following items in the current revision.

- [x] Executable Python/PyTorch implementation.
- [x] No hard-coded local workstation paths.
- [x] Non-confidential execution example included.
- [x] Input-data template included.
- [x] Feature descriptions included.
- [x] Lightweight repository integrity checks included.
- [x] Requirements and conda environment files included.
- [x] Physics-regularized attention-mask code included.
- [x] Type-aware Transformer model included.
- [x] Compound-loss code included.
- [x] Training-fold-only constrained augmentation included.
- [x] Case-grouped leave-one-case-out validation included.
- [x] Optional inner-fold grid-search utilities included.
- [x] External generalization testing included.
- [x] Predicted-vs-ground-truth agreement plot generation included.
- [x] Bootstrap confidence-interval output included.
- [x] Attention-matrix output included.
- [x] Example review-check outputs included.

## Minimal Verification Command

```bash
python src/run_experiment.py \
  --data data/non_confidential_execution_example.csv \
  --config configs/prt_config.yaml \
  --outdir outputs/reviewer_check \
  --epochs 5
```

## Model Summary Command

```bash
python src/model_summary.py --config configs/prt_config.yaml
```

## Integrity Check Command

```bash
python scripts/run_integrity_checks.py
```
