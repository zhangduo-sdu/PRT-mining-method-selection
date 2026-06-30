# Reproducibility Notes

This package is designed to let reviewers verify the executable workflow during
peer review. It covers:

1. feature schema validation;
2. training-partition-only preprocessing statistics;
3. constrained augmentation after fold assignment;
4. physics-regularized attention-mask construction;
5. PyTorch model training;
6. grouped validation and external testing;
7. predicted-vs-ground-truth parity plotting;
8. attention-matrix export;
9. bootstrap confidence-interval summaries.

The following materials are restricted by industrial confidentiality and data-use
agreements:

- full mine-site indicator matrices;
- original FLAC3D model files;
- trained weights obtained from confidential real data.

The included synthetic benchmark has the same feature schema as the manuscript
and is sufficient to verify that the computational pipeline runs end to end. It
does not replace the confidential real-mine validation data.
