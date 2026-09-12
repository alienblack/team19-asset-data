"""Explainable baseline imputers for missing overhead-conductor material.

The baselines are deliberately simple comparison points for later methods:

* ``global_mode`` predicts the most common material in the development data.
* ``feeder_group_mode`` predicts the most common material for a feeder and
  falls back to the development-set global mode for an unseen feeder.

The experiment masks known values only in the held-out evaluation partition.
This makes the original values a controlled benchmark without allowing the
imputer to learn from the values being evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


DEFAULT_INPUT = Path("data/synthetic/overhead_conductors/material_model_ready_data.csv")
DEFAULT_OUTPUT = Path("outputs/PGS219-99")
DEFAULT_MASK_RATES = (0.10, 0.30, 0.50)
DEFAULT_SEEDS = (101, 202, 303, 404, 505)
LABELS = ("Aluminium", "Copper", "Steel")


def _mode(values: Iterable[object]) -> str:
    """Return a deterministic mode, using alphabetical order to break ties."""
    counts = pd.Series(list(values), dtype="string").value_counts()
    if counts.empty:
        raise ValueError("Cannot calculate a mode from an empty series.")
    top_count = counts.max()
    return sorted(str(value) for value in counts[counts == top_count].index)[0]


def validate_input(frame: pd.DataFrame, group_columns: Sequence[str] = ("feeder_id",)) -> None:
    """Validate the minimum schema required by the material experiment."""
    required = {"segment_id", "material", *group_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input dataset is empty.")
    if frame["material"].isna().any():
        raise ValueError("The complete reference input must not contain missing material values.")
    unexpected = sorted(set(frame["material"].astype(str)) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected material classes: {', '.join(unexpected)}")


def split_indices(n_rows: int, seed: int, test_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible development/evaluation row positions."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if n_rows < 2:
        raise ValueError("At least two rows are required for a split.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_rows)
    n_eval = max(1, int(round(n_rows * test_fraction)))
    return shuffled[:-n_eval], shuffled[-n_eval:]


def select_mask_positions(evaluation_positions: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Select known evaluation values to hide at a requested masking rate."""
    if not 0 < rate <= 1:
        raise ValueError("mask rate must be greater than 0 and at most 1.")
    rng = np.random.default_rng(seed)
    n_masked = max(1, int(round(len(evaluation_positions) * rate)))
    return np.sort(rng.choice(evaluation_positions, size=n_masked, replace=False))


def fit_global_mode(development: pd.DataFrame) -> str:
    """Fit the transparent whole-development-set mode baseline."""
    return _mode(development["material"])


def fit_group_modes(development: pd.DataFrame, group_columns: Sequence[str]) -> dict[tuple[object, ...], str]:
    """Fit one material mode per group using development rows only."""
    if not group_columns:
        raise ValueError("At least one group column is required for group modes.")
    result: dict[tuple[object, ...], str] = {}
    grouping = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in development.groupby(grouping, dropna=False, sort=True):
        normalised_key = key if isinstance(key, tuple) else (key,)
        result[normalised_key] = _mode(group["material"])
    return result


def predict_group_mode(
    rows: pd.DataFrame,
    group_modes: dict[tuple[object, ...], str],
    fallback: str,
    group_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict group modes and flag rows that used the global fallback."""
    predictions: list[str] = []
    fallback_used: list[bool] = []
    for values in rows[list(group_columns)].itertuples(index=False, name=None):
        key = tuple(values)
        prediction = group_modes.get(key, fallback)
        predictions.append(prediction)
        fallback_used.append(key not in group_modes)
    return np.asarray(predictions, dtype=str), np.asarray(fallback_used, dtype=bool)


def evaluate_predictions(y_true: Sequence[object], y_pred: Sequence[object], labels: Sequence[str] = LABELS) -> dict[str, float]:
    """Calculate overall and class-balanced metrics for one method."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
    }


def run_experiment(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    mask_rates: Sequence[float] = DEFAULT_MASK_RATES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    group_columns: Sequence[str] = ("feeder_id",),
    test_fraction: float = 0.20,
) -> pd.DataFrame:
    """Run the baseline comparison and save reproducible evidence files."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    frame = pd.read_csv(input_path)
    validate_input(frame, group_columns)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    sample_masked: pd.DataFrame | None = None
    sample_predictions: pd.DataFrame | None = None

    for seed in seeds:
        development_positions, evaluation_positions = split_indices(len(frame), int(seed), test_fraction)
        development = frame.iloc[development_positions]
        evaluation = frame.iloc[evaluation_positions]
        global_prediction = fit_global_mode(development)
        group_modes = fit_group_modes(development, group_columns)

        for rate in mask_rates:
            mask_seed = int(seed) + int(round(float(rate) * 10_000))
            masked_positions = select_mask_positions(evaluation_positions, float(rate), mask_seed)
            masked_rows = frame.iloc[masked_positions].copy()
            truth = masked_rows["material"].astype(str).to_numpy()
            masked_rows["material"] = pd.NA
            masked_rows["material_was_masked"] = True

            # Use object dtype so class labels such as ``Steel`` are not
            # truncated by NumPy's inferred fixed-width string dtype.
            global_predictions = np.full(len(masked_rows), global_prediction, dtype=object)
            group_predictions, fallback_used = predict_group_mode(
                frame.iloc[masked_positions], group_modes, global_prediction, group_columns
            )

            for method, predictions, fallback_count in (
                ("global_mode", global_predictions, 0),
                ("feeder_group_mode", group_predictions, int(fallback_used.sum())),
            ):
                metrics = evaluate_predictions(truth, predictions)
                metric_rows.append({
                    "seed": int(seed),
                    "mask_rate": float(rate),
                    "method": method,
                    "n_development": int(len(development)),
                    "n_evaluation": int(len(evaluation)),
                    "n_masked": int(len(masked_rows)),
                    "fallback_count": fallback_count,
                    **metrics,
                })
                matrix = confusion_matrix(truth, predictions, labels=list(LABELS))
                for actual_index, actual in enumerate(LABELS):
                    for predicted_index, predicted in enumerate(LABELS):
                        confusion_rows.append({
                            "seed": int(seed),
                            "mask_rate": float(rate),
                            "method": method,
                            "actual_material": actual,
                            "predicted_material": predicted,
                            "count": int(matrix[actual_index, predicted_index]),
                        })

            for segment_id, actual in zip(masked_rows["segment_id"], truth):
                manifest_rows.append({
                    "seed": int(seed),
                    "mask_rate": float(rate),
                    "segment_id": segment_id,
                    "material_true": actual,
                })

            if int(seed) == int(seeds[0]) and float(rate) == 0.30:
                sample_masked = masked_rows[["segment_id", *group_columns, "material", "material_was_masked"]].copy()
                sample_predictions = pd.DataFrame({
                    "segment_id": masked_rows["segment_id"].to_numpy(),
                    "material_true": truth,
                    "global_mode_prediction": global_predictions,
                    "feeder_group_mode_prediction": group_predictions,
                    "feeder_group_fallback_used": fallback_used,
                })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "mask_manifest.csv", index=False)
    if sample_masked is not None and sample_predictions is not None:
        sample_masked.to_csv(output_dir / "masked_material_seed101_rate30.csv", index=False)
        sample_predictions.to_csv(output_dir / "imputed_material_baselines_seed101_rate30.csv", index=False)

    metadata = {
        "input_file": str(input_path),
        "target": "material",
        "methods": ["global_mode", "feeder_group_mode"],
        "group_columns": list(group_columns),
        "mask_rates": [float(rate) for rate in mask_rates],
        "seeds": [int(seed) for seed in seeds],
        "test_fraction": float(test_fraction),
        "truth_handling": "Truth is retained only in evaluation evidence; model-facing masked files contain blank target values.",
        "feature_leakage_control": "conductor_type is not used; the current material-model input intentionally excludes this highly predictive field.",
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explainable overhead-conductor material baselines.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mask-rates", nargs="+", type=float, default=list(DEFAULT_MASK_RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--group-columns", nargs="+", default=["feeder_id"])
    parser.add_argument("--test-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(
        input_path=args.input,
        output_dir=args.output_dir,
        mask_rates=args.mask_rates,
        seeds=args.seeds,
        group_columns=args.group_columns,
        test_fraction=args.test_fraction,
    )
    summary = metrics.groupby(["mask_rate", "method"], as_index=False)[
        ["accuracy", "balanced_accuracy", "macro_f1"]
    ].mean()
    print(summary.to_string(index=False))
    print(f"\nEvidence written to {args.output_dir}")


if __name__ == "__main__":
    main()
