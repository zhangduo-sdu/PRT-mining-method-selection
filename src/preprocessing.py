"""Type-aware feature preprocessing for PRT inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class TabularPreprocessor:
    """Fit-on-training-only preprocessing.

    Quantitative indicators (X1--X7):
        z-score normalization based on training data.

    Qualitative indicators (X8--X11):
        min-max scaling based on training data.

    Simulation features:
        engineering-limit normalization:
            sigma1_mpa / stress_limit_mpa
            delta_mm / displacement_limit_mm
            Ap / plastic_zone_limit
    """

    quantitative_cols: List[str]
    qualitative_cols: List[str]
    simulation_cols: List[str]
    feature_order: List[str]
    normalization: Dict[str, float]

    q_mean_: Dict[str, float] = field(default_factory=dict)
    q_std_: Dict[str, float] = field(default_factory=dict)
    qual_min_: Dict[str, float] = field(default_factory=dict)
    qual_max_: Dict[str, float] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit preprocessing statistics using training data only."""
        self._check_columns(df)

        for col in self.quantitative_cols:
            self.q_mean_[col] = float(df[col].mean())
            std = float(df[col].std(ddof=0))
            self.q_std_[col] = std if std > 1e-12 else 1.0

        for col in self.qualitative_cols:
            cmin = float(df[col].min())
            cmax = float(df[col].max())
            self.qual_min_[col] = cmin
            self.qual_max_[col] = cmax if cmax > cmin else cmin + 1.0

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a dataframe into a normalized feature matrix."""
        if not self.fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        self._check_columns(df)

        out = pd.DataFrame(index=df.index)

        for col in self.quantitative_cols:
            out[col] = (df[col] - self.q_mean_[col]) / self.q_std_[col]

        for col in self.qualitative_cols:
            denom = self.qual_max_[col] - self.qual_min_[col]
            out[col] = (df[col] - self.qual_min_[col]) / denom
            out[col] = out[col].clip(0.0, 1.0)

        stress_limit = float(self.normalization.get("stress_limit_mpa", 18.19))
        displacement_limit = float(self.normalization.get("displacement_limit_mm", 40.0))
        plastic_limit = float(self.normalization.get("plastic_zone_limit", 1.0))

        for col in self.simulation_cols:
            if col == "sigma1_mpa":
                out[col] = df[col] / stress_limit
            elif col == "delta_mm":
                out[col] = df[col] / displacement_limit
            elif col == "Ap":
                out[col] = df[col] / plastic_limit
            else:
                raise ValueError(f"Unknown simulation feature: {col}")

        return out[self.feature_order].to_numpy(dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(df).transform(df)

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_order if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
