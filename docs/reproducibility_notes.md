# Reproducibility Notes

This package is designed to let reviewers verify the executable workflow during
peer review. It covers:

1. feature schema validation;
2. training-partition-only preprocessing statistics;
3. constrained augmentation after fold assignment;
4. physics-regularized attention-mask construction;
5. PyTorch model training;
6. grouped validation and external testing;
7. predicted-vs-ground-truth agreement plotting;
8. attention-matrix export;
9. bootstrap confidence-interval summaries.

The following materials are governed by industrial data-use agreements:

- full mine-site indicator matrices;
- original FLAC3D model files;
- trained weights obtained from controlled real data.

The included non-confidential execution example has the same feature schema as
the manuscript and is sufficient to verify that the computational pipeline runs
end to end. It does not replace the controlled real-mine validation data. The
manuscript validation and generalization claims are based on the real cases
documented in Supplementary Material S1.
