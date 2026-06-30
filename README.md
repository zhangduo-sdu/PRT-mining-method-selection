# Physics-Regularized Transformer for Mining Method Selection

This repository provides an executable Python/PyTorch reference implementation
for the manuscript:

**Physics-Regularized Deep Learning for Mining Method Selection in Computational Geomechanics**

The package is intended for peer-review reproducibility in the current revision.
It implements the machine-learning workflow described in the manuscript:

- type-aware preprocessing for 11 conventional indicators and 3 FLAC3D-derived features;
- construction of the 14 x 14 physics-regularized attention mask;
- type-aware Transformer model for superiority-score prediction;
- compound loss with prediction, mechanical-feasibility, and attention-entropy terms;
- training-fold-only constrained augmentation;
- case-grouped leave-one-case-out validation;
- optional inner-fold grid-search utilities for the nested-CV model-selection protocol;
- external generalization testing;
- predicted-vs-ground-truth agreement plots and attention-matrix export.

## Data Availability Note

The complete row-level real-mine feature matrices, original FLAC3D model files,
and trained weights derived from controlled industrial records are governed by
third-party data-use agreements and are therefore not included in the public
repository. This repository therefore includes:

- the full executable PyTorch implementation;
- a deterministic non-confidential execution example with the same column schema as the manuscript;
- an input template for running the same code on real case data;
- review-check outputs generated from the non-confidential execution example.

The non-confidential execution example is for code-executability and workflow
checking only. It is not used for the real-case validation, external
generalization, or domain-boundary claims reported in the manuscript. Those
claims are based on the real cases documented in Supplementary Material S1.

## Repository Structure

```text
PRT_mining_method_selection_code/
|-- README.md
|-- CODE_AVAILABILITY.md
|-- MANUSCRIPT_ALIGNMENT.md
|-- REVIEWER_REPRODUCIBILITY_CHECKLIST.md
|-- requirements.txt
|-- environment.yml
|-- configs/
|   `-- prt_config.yaml
|-- data/
|   |-- non_confidential_execution_example.csv
|   |-- feature_description.csv
|   `-- example_real_case_format.csv
|-- docs/
|   `-- reproducibility_notes.md
|-- scripts/
|   |-- generate_execution_example_data.py
|   |-- run_integrity_checks.py
|   |-- run_review_check.sh
|   |-- run_review_check.ps1
|   |-- run_full_experiment.sh
|   |-- run_smoke_test.sh
|   `-- model_summary.ps1
|-- src/
|   |-- attention_mask.py
|   |-- augmentation.py
|   |-- grid_search.py
|   |-- losses.py
|   |-- metrics.py
|   |-- model.py
|   |-- model_summary.py
|   |-- preprocessing.py
|   |-- run_experiment.py
|   |-- run_grid_search.py
|   |-- trainer.py
|   `-- utils.py
`-- outputs/
    |-- verification_log.txt
    `-- reviewer_check/
```

## Installation

Create a clean Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Or use the conda environment file:

```bash
conda env create -f environment.yml
conda activate prt-mining
```

## Quick Review Check

Run the complete grouped-validation pipeline on the included non-confidential
execution example:

```bash
python src/run_experiment.py \
  --data data/non_confidential_execution_example.csv \
  --config configs/prt_config.yaml \
  --outdir outputs/reviewer_check \
  --epochs 5
```

Expected outputs:

```text
outputs/reviewer_check/fold_metrics.csv
outputs/reviewer_check/loocv_predictions.csv
outputs/reviewer_check/parity_plot.png
outputs/reviewer_check/mean_attention_matrix.csv
outputs/reviewer_check/mean_attention_matrix.png
outputs/reviewer_check/external_predictions.csv
outputs/reviewer_check/summary.json
```

The command above intentionally uses a small epoch count so reviewers can verify
the pipeline quickly. The manuscript configuration in `configs/prt_config.yaml`
uses 200 maximum epochs with early stopping.

## Model Summary

```bash
python src/model_summary.py --config configs/prt_config.yaml
```

The selected manuscript configuration reports:

```text
Trainable parameters: 36,993
Allowed attention connections: 140/196
```

## Integrity Checks

Run lightweight repository checks before training:

```bash
python scripts/run_integrity_checks.py
```

This checks the feature schema, case grouping, attention-mask size and link
count, parameter count, and a forward pass through the model.

## Optional Inner-Grid Check

The selected hyperparameters are in `configs/prt_config.yaml`. The optional
grid-search script demonstrates the nested-CV model-selection logic on a capped
subset of candidate configurations:

```bash
python src/run_grid_search.py \
  --data data/non_confidential_execution_example.csv \
  --config configs/prt_config.yaml \
  --outdir outputs/grid_search_demo \
  --epochs 3 \
  --max-candidates 6
```

Running the full grid is computationally heavier and is not required for the
quick review check.

## Regenerate the Non-confidential Execution Example

```bash
python scripts/generate_execution_example_data.py \
  --out data/non_confidential_execution_example.csv \
  --seed 42
```

## Input Schema

Each row represents one candidate mining method for one independent mine case.
Required columns are:

- `case_id`: independent case or group identifier.
- `method_id`: candidate method identifier.
- `split`: `real_pool` or `external_generalization`.
- `domain`: `in_distribution`, `boundary`, or `out_of_distribution`.
- `RQD`: rock-quality designation.
- `X1` to `X11`: conventional mining-method evaluation indicators.
- `sigma1_mpa`: maximum principal stress from FLAC3D.
- `delta_mm`: maximum roof displacement from FLAC3D.
- `Ap`: normalized plastic-zone area.
- `target_score`: ground-truth or expert-consensus superiority score in [0, 1].
- `is_optimal`: 1 for the best method within each case, 0 otherwise.

The file `data/example_real_case_format.csv` shows the required schema for
running the code on real data.

## Code Availability Statement

Suggested manuscript wording:

```text
The executable Python/PyTorch reference implementation has been made available
for peer review in the current revision. The repository includes preprocessing,
physics-regularized attention-mask construction, model definition, compound-loss
calculation, training-fold-only augmentation, case-grouped validation,
external-generalization testing, predicted-vs-ground-truth plotting, and
attention-matrix export. Complete row-level real-mine feature matrices, original
FLAC3D files, and trained weights derived from controlled industrial records are
governed by data-use agreements; a non-confidential execution example with the
same feature schema is included to verify that the complete computational
workflow is executable. The validation and generalization claims are based on
the real cases documented in Supplementary Material S1.
```
