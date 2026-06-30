"""Training-fold-only physically constrained data augmentation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def augment_training_dataframe(
    df: pd.DataFrame,
    config: Dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate physically screened local perturbations within the training fold.

    This function must be called only after the outer/inner fold assignment has
    already been made. It never receives validation or test rows.

    Labels are retained because the perturbations are restricted to local
    engineering tolerances and are intended to mimic nearby feasible design
    states, not to create new independent ground-truth decisions.
    """
    aug_cfg = config.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        return df.copy()

    n_aug_per_case = int(aug_cfg.get("n_aug_per_case", 1))
    if n_aug_per_case <= 0:
        return df.copy()

    quantitative_cols: List[str] = config["features"]["quantitative"]
    qualitative_cols: List[str] = config["features"]["qualitative"]
    simulation_cols: List[str] = config["features"]["simulation"]
    case_col = config["columns"]["case_id"]

    q_rel = float(aug_cfg.get("quantitative_relative_noise", 0.10))
    qual_abs = float(aug_cfg.get("qualitative_absolute_noise", 0.05))
    sim_rel = float(aug_cfg.get("simulation_relative_noise", 0.05))

    qual_low = float(aug_cfg.get("clip_qualitative_min", 0.0))
    qual_high = float(aug_cfg.get("clip_qualitative_max", 1.0))

    stress_limit = float(config["loss"].get("stress_limit_mpa", 18.19))
    displacement_limit = float(config["loss"].get("displacement_limit_mm", 40.0))
    plastic_limit = float(config["augmentation"].get("max_plastic_zone_for_keep", 0.30))

    keep_stress_ratio = float(aug_cfg.get("max_stress_ratio_for_keep", 1.0))
    keep_disp_ratio = float(aug_cfg.get("max_displacement_ratio_for_keep", 1.0))

    augmented_groups = []

    for _, group in df.groupby(case_col, sort=False):
        augmented_groups.append(group.copy())

        for k in range(n_aug_per_case):
            g = group.copy()
            original_case_id = str(g[case_col].iloc[0])
            g[case_col] = f"{original_case_id}_aug{k+1}"

            # Quantitative features: multiplicative perturbation within tolerance.
            for col in quantitative_cols:
                factors = rng.uniform(1.0 - q_rel, 1.0 + q_rel, size=len(g))
                g[col] = g[col].astype(float).to_numpy() * factors

            # Qualitative features: discrete local adjustment.
            for col in qualitative_cols:
                steps = rng.choice([-qual_abs, 0.0, qual_abs], size=len(g))
                g[col] = np.clip(g[col].astype(float).to_numpy() + steps, qual_low, qual_high)

            # Simulation features: local physically plausible perturbation.
            for col in simulation_cols:
                factors = rng.uniform(1.0 - sim_rel, 1.0 + sim_rel, size=len(g))
                g[col] = g[col].astype(float).to_numpy() * factors

            # Mechanical feasibility screening.
            feasible = (
                (g["sigma1_mpa"] / stress_limit <= keep_stress_ratio)
                & (g["delta_mm"] / displacement_limit <= keep_disp_ratio)
                & (g["Ap"] <= plastic_limit)
            )

            if feasible.all():
                augmented_groups.append(g)

    out = pd.concat(augmented_groups, ignore_index=True)
    return out
