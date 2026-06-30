# Manuscript-to-Code Alignment

This file maps the main methodological components in the revised manuscript to
the repository implementation.

| Manuscript component | Repository implementation |
|---|---|
| Fourteen-dimensional input vector `[X1, ..., X11, sigma1, delta, Ap]` | `configs/prt_config.yaml`, `data/non_confidential_execution_example.csv`, `src/preprocessing.py` |
| Quantitative, qualitative, and simulation feature preprocessing | `src/preprocessing.py` |
| FLAC3D-derived model inputs | `sigma1_mpa`, `delta_mm`, and `Ap` columns in the input CSV |
| 14 x 14 physics-regularized attention mask | `src/attention_mask.py` |
| Type-aware embedding | `src/model.py` |
| Physics-regularized Transformer model | `src/model.py` |
| Compound loss | `src/losses.py` |
| Training-fold-only constrained augmentation | `src/augmentation.py`, called only inside training folds by `src/trainer.py` |
| Case-grouped leave-one-case-out validation | `src/run_experiment.py`, `src/trainer.py`, `src/nested_cv.py` |
| Optional inner-fold hyperparameter selection | `src/grid_search.py`, `src/run_grid_search.py` |
| External generalization testing | `src/run_experiment.py` |
| Predicted-vs-ground-truth agreement plot | `src/metrics.py`, output `parity_plot.png` |
| Bootstrap confidence intervals | `src/metrics.py`, output fields in `summary.json` |
| Attention-matrix export and visualization | `src/metrics.py`, outputs `mean_attention_matrix.csv` and `mean_attention_matrix.png` |

## Controlled Data Handling

The code is executable without controlled mine data. The included
`data/non_confidential_execution_example.csv` follows the manuscript feature
schema and workflow logic for software-execution checking only. The same scripts
can be run on real data if the real feature matrix is provided in the same
column format as `data/example_real_case_format.csv`. The manuscript validation
and generalization claims are based on the real cases documented in
Supplementary Material S1, not on the non-confidential execution example.
