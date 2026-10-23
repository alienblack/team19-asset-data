"""Ground-truth evaluation for controlled imputation experiments.

Works with already-incomplete SAPN-like records. Optional *evaluation* masking
hides only values that are currently observed; pre-existing missing values are
never counted as ground truth.

Mask preparation and metric calculation are separate: saved or externally
prepared masks can be passed directly to ``evaluate_imputation``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)


_METRIC_COLUMNS = [
    "variable",
    "variable_type",
    "n_masked",
    "mae",
    "rmse",
    "median_absolute_error",
    "nrmse",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
]


def prepare_evaluation_holdout(
    X_test: pd.DataFrame,
    columns: Sequence[str],
    rate: float = 0.10,
    seed: int = 319,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a reproducible evaluation mask on CURRENTLY OBSERVED cells.

    ``X_test`` may already contain natural or scenario-generated missing values.
    For each requested target column, randomly hide approximately ``rate`` of
    its *observed* cells, leaving at least one known cell in that column.

    Returns ``(masked_test, test_truth, evaluation_mask)``.  The boolean mask
    identifies **only newly hidden** positions; the original input is not
    modified. The truth and mask must stay outside model fitting and inference.

    This helper is for an independent test partition. If the scenario engine
    already supplies a fixed evaluation mask, use that mask instead so all
    noise levels and model families evaluate exactly the same positions.
    The returned truth contains observed input values, not an external clean
    reference; synthetic noise experiments should score against the original
    clean source when such ground truth is available.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not X_test.index.is_unique:
        raise ValueError("Evaluation record index must uniquely identify rows.")
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not np.isfinite(rate) or not 0 < rate < 1):
        raise ValueError("Evaluation mask rate must be strictly between zero and one.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Evaluation mask seed must be a nonnegative integer.")

    targets = tuple(columns)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("Evaluation target columns must be nonempty and unique.")
    missing_columns = sorted(set(targets) - set(X_test.columns))
    if missing_columns:
        raise ValueError(f"Unknown evaluation target columns: {missing_columns}.")

    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(False, index=X_test.index, columns=X_test.columns)
    for column in targets:
        observed_positions = np.flatnonzero(X_test[column].notna().to_numpy())
        if len(observed_positions) < 2:
            raise ValueError(
                f"Cannot evaluate '{column}': at least two known values "
                "are needed to create a holdout."
            )
        n_hide = max(1, int(round(rate * len(observed_positions))))
        n_hide = min(n_hide, len(observed_positions) - 1)
        selected = rng.choice(observed_positions, size=n_hide, replace=False)
        mask.iloc[selected, mask.columns.get_loc(column)] = True

    truth = X_test.copy(deep=True)
    masked = X_test.copy(deep=True)
    masked[mask] = np.nan
    return masked, truth, mask


def _validate_evaluation_inputs(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """Prevent positional misalignment and scoring unknown/unimputed values."""
    if not isinstance(truth, pd.DataFrame):
        raise TypeError("truth must be a pandas DataFrame.")
    if not truth.index.is_unique or not truth.columns.is_unique:
        raise ValueError("truth must have unique row indices and column names.")

    for name, other in (("imputed", imputed), ("mask", mask)):
        if not isinstance(other, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
        if not truth.index.equals(other.index):
            raise ValueError(f"{name} index must exactly match truth index.")
        if not truth.columns.equals(other.columns):
            raise ValueError(f"{name} columns must exactly match truth columns.")

    unknown_columns = sorted(set(columns) - set(truth.columns))
    if unknown_columns:
        raise ValueError(f"Evaluation columns not found in truth: {unknown_columns}.")

    for column in columns:
        if not pd.api.types.is_bool_dtype(mask[column]) or mask[column].isna().any():
            raise ValueError(f"mask['{column}'] must contain only boolean values.")
        selected = mask[column].to_numpy(dtype=bool)
        if not selected.any():
            continue
        if truth[column].iloc[selected].isna().any():
            raise ValueError(f"Missing ground truth for evaluated cells of '{column}'.")
        if imputed[column].iloc[selected].isna().any():
            raise ValueError(f"Unfilled evaluation cells remain in '{column}'.")


def evaluate_imputation(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """Score only deliberately hidden *known* values, not original missingness.

    Metric definitions are unchanged from the original evaluation module.
    ``mask`` can be generated by ``prepare_evaluation_holdout`` or externally
    by the scenario engine. The scorer never generates or modifies masks.
    """
    _validate_evaluation_inputs(
        truth, imputed, mask, [*numeric_columns, *categorical_columns]
    )
    rows: list[dict[str, object]] = []

    for column in numeric_columns:
        positions = mask[column].to_numpy(dtype=bool)
        if not positions.any():
            continue

        y_true = truth.loc[positions, column].astype(float).to_numpy()
        y_pred = imputed.loc[positions, column].astype(float).to_numpy()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        scale = float(np.std(y_true))

        rows.append(
            {
                "variable": column,
                "variable_type": "numeric",
                "n_masked": int(positions.sum()),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": rmse,
                "median_absolute_error": float(
                    median_absolute_error(y_true, y_pred)
                ),
                "nrmse": rmse / scale if scale > np.finfo(float).eps else np.nan,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "macro_f1": np.nan,
            }
        )

    for column in categorical_columns:
        positions = mask[column].to_numpy(dtype=bool)
        if not positions.any():
            continue

        y_true = truth.loc[positions, column].astype(str).to_numpy()
        y_pred = imputed.loc[positions, column].astype(str).to_numpy()

        rows.append(
            {
                "variable": column,
                "variable_type": "categorical",
                "n_masked": int(positions.sum()),
                "mae": np.nan,
                "rmse": np.nan,
                "median_absolute_error": np.nan,
                "nrmse": np.nan,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, y_pred)
                ),
                "macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
            }
        )

    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)
