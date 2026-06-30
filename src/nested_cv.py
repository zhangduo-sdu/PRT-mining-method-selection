"""Grouped validation utilities.

This module is intentionally lightweight. The complete fold execution is in
`trainer.py` and `run_experiment.py`; this file exposes reusable helpers for
case-grouped leave-one-out splits used by the manuscript workflow.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


def grouped_leave_one_case_out(
    df: pd.DataFrame,
    case_col: str = "case_id",
) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
    """Yield train/test indices for case-grouped leave-one-out validation."""
    case_ids = list(df[case_col].drop_duplicates())
    for case_id in case_ids:
        test_mask = df[case_col].astype(str).to_numpy() == str(case_id)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx, str(case_id)
