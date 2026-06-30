# Code Availability Statement

The executable Python/PyTorch reference implementation is provided with the
current revision for peer-review reproducibility. The repository includes:

- feature preprocessing for quantitative, qualitative, and FLAC3D-derived variables;
- construction of the 14 x 14 physics-regularized attention mask;
- type-aware Transformer model implementation;
- compound-loss calculation with prediction, mechanical-feasibility, and attention-entropy terms;
- training-fold-only constrained augmentation;
- case-grouped leave-one-case-out validation;
- optional inner-fold grid-search utilities for the nested-CV model-selection protocol;
- external generalization testing;
- predicted-vs-ground-truth agreement plots;
- attention-matrix export and visualization.

The complete row-level real-mine feature matrices, original FLAC3D files, and
trained model weights derived from controlled industrial records are governed by
third-party data-use agreements and are therefore not included in the public
repository. A deterministic non-confidential execution example with the same
feature schema is included so reviewers can execute the complete workflow end to
end.

The non-confidential execution example is not a substitute for the controlled
real-mine data and is not intended to reproduce the manuscript's real-data
performance values. Validation and generalization claims are based on the real
cases documented in Supplementary Material S1.
