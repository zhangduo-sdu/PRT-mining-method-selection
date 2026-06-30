"""Generate a deterministic non-confidential execution example with the manuscript feature schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_BASE = {
    1: {
        "X1": 250, "X2": 11.0, "X3": 12.0, "X4": 0.50, "X5": 8.0, "X6": 7.0,
        "X7": 198, "X8": 0.75, "X9": 0.55, "X10": 0.55, "X11": 0.50,
        "sigma1_mpa": 6.89, "delta_mm": 7.40, "Ap": 0.15,
    },
    2: {
        "X1": 250, "X2": 10.0, "X3": 11.5, "X4": 0.52, "X5": 7.5, "X6": 6.5,
        "X7": 190, "X8": 0.86, "X9": 0.40, "X10": 0.50, "X11": 0.72,
        "sigma1_mpa": 2.66, "delta_mm": 5.45, "Ap": 0.11,
    },
    3: {
        "X1": 450, "X2": 8.5, "X3": 20.0, "X4": 0.42, "X5": 9.5, "X6": 9.0,
        "X7": 175, "X8": 0.70, "X9": 0.64, "X10": 0.90, "X11": 0.36,
        "sigma1_mpa": 8.85, "delta_mm": 8.31, "Ap": 0.18,
    },
}


def normalize_between(x, low, high, beneficial=True):
    z = (x - low) / (high - low)
    z = np.clip(z, 0.0, 1.0)
    return z if beneficial else 1.0 - z


def expert_score(row, rqd, domain):
    """Generate an expert-consensus-like score for the execution example."""
    capacity = normalize_between(row["X1"], 80, 500, True)
    dev = normalize_between(row["X2"], 6, 20, False)
    efficiency = normalize_between(row["X3"], 3, 25, True)
    explosive = normalize_between(row["X4"], 0.25, 1.05, False)
    loss = normalize_between(row["X5"], 3, 25, False)
    dilution = normalize_between(row["X6"], 3, 25, False)
    cost = normalize_between(row["X7"], 100, 250, False)
    stress_safe = 1.0 - np.clip(row["sigma1_mpa"] / 18.19, 0, 1)
    disp_safe = 1.0 - np.clip(row["delta_mm"] / 40.0, 0, 1)
    plastic_safe = 1.0 - np.clip(row["Ap"] / 0.30, 0, 1)

    if domain == "out_of_distribution":
        # Hard-rock or non-three-soft conditions: productivity and cost become
        # more dominant, and the safety-first priority structure changes.
        weights = {
            "capacity": 0.24, "dev": 0.08, "efficiency": 0.10, "explosive": 0.04,
            "loss": 0.08, "dilution": 0.08, "cost": 0.20, "safety": 0.08,
            "complexity": 0.04, "mechan": 0.04, "cycle": 0.02,
        }
    else:
        # Three-soft regime: safety and stress response are weighted strongly.
        weights = {
            "capacity": 0.16, "dev": 0.06, "efficiency": 0.06, "explosive": 0.04,
            "loss": 0.08, "dilution": 0.08, "cost": 0.12, "safety": 0.25,
            "complexity": 0.04, "mechan": 0.06, "cycle": 0.05,
        }

    safety_composite = 0.60 * row["X8"] + 0.18 * stress_safe + 0.12 * disp_safe + 0.10 * plastic_safe

    score = (
        weights["capacity"] * capacity
        + weights["dev"] * dev
        + weights["efficiency"] * efficiency
        + weights["explosive"] * explosive
        + weights["loss"] * loss
        + weights["dilution"] * dilution
        + weights["cost"] * cost
        + weights["safety"] * safety_composite
        + weights["complexity"] * row["X9"]
        + weights["mechan"] * row["X10"]
        + weights["cycle"] * row["X11"]
    )

    # Very poor rock quality makes Method 2 more attractive in some cases.
    if rqd < 46 and int(row["method_id"]) == 2:
        score += 0.045
    if rqd < 43 and int(row["method_id"]) == 3:
        score -= 0.060

    return float(np.clip(score, 0.0, 1.0))


def make_case(case_id, rqd, split, domain, rng):
    rows = []
    quality_factor = (55.0 - rqd) / 20.0

    for method_id, base in METHOD_BASE.items():
        row = {"case_id": case_id, "method_id": method_id, "split": split, "domain": domain, "RQD": rqd}
        for key, value in base.items():
            if key.startswith("X") and key not in {"X8", "X9", "X10", "X11"}:
                row[key] = float(value * rng.normal(1.0, 0.06))
            elif key in {"X8", "X9", "X10", "X11"}:
                row[key] = float(np.clip(value + rng.normal(0.0, 0.035), 0.0, 1.0))
            elif key == "sigma1_mpa":
                row[key] = float(value * (1.0 + 0.10 * quality_factor) * rng.normal(1.0, 0.05))
            elif key == "delta_mm":
                row[key] = float(value * (1.0 + 0.12 * quality_factor) * rng.normal(1.0, 0.05))
            elif key == "Ap":
                row[key] = float(np.clip(value * (1.0 + 0.18 * quality_factor) * rng.normal(1.0, 0.05), 0.03, 0.29))
            else:
                row[key] = value

        # Make boundary/OOD sites mechanically and strategically different.
        if domain == "boundary":
            row["X8"] = float(np.clip(row["X8"] - 0.05, 0, 1))
            row["sigma1_mpa"] *= 0.95
            row["Ap"] *= 0.90
        elif domain == "out_of_distribution":
            row["sigma1_mpa"] *= 0.70
            row["delta_mm"] *= 0.65
            row["Ap"] *= 0.70
            if method_id == 1:
                row["X7"] *= 0.88
                row["X8"] += 0.05
            if method_id == 3:
                row["X5"] *= 1.15
                row["X6"] *= 1.15

        row["target_score"] = expert_score(row, rqd, domain)
        rows.append(row)

    # Mark the best method within the case.
    max_idx = int(np.argmax([r["target_score"] for r in rows]))
    for i, row in enumerate(rows):
        row["is_optimal"] = 1 if i == max_idx else 0

    return rows


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # 19 independent real-pool cases for grouped validation.
    rqds = np.linspace(40, 63, 19) + rng.normal(0, 1.2, 19)
    rqds = np.clip(rqds, 40, 63)
    for i, rqd in enumerate(rqds, start=1):
        rows.extend(make_case(f"R{i:02d}", float(rqd), "real_pool", "in_distribution", rng))

    # 10 external generalization cases: 7 ID, 2 boundary, 1 OOD.
    ext_rqds = [43, 45, 48, 50, 52, 55, 63, 67, 69, 78]
    for j, rqd in enumerate(ext_rqds, start=1):
        if j <= 7:
            domain = "in_distribution"
        elif j <= 9:
            domain = "boundary"
        else:
            domain = "out_of_distribution"
        rows.extend(make_case(f"E{j:02d}", float(rqd), "external_generalization", domain, rng))

    df = pd.DataFrame(rows)

    ordered_cols = [
        "case_id", "method_id", "split", "domain", "RQD",
        "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11",
        "sigma1_mpa", "delta_mm", "Ap",
        "target_score", "is_optimal",
    ]
    return df[ordered_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(seed=args.seed)
    df.to_csv(out, index=False)
    print(f"Saved non-confidential execution example to {out} with {len(df)} rows and {df['case_id'].nunique()} cases.")


if __name__ == "__main__":
    main()
