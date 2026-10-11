"""Ground-truth evaluation for controlled imputation experiments."""

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


def evaluate_imputation(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """Score only the cells deliberately hidden by the experiment."""
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

    return pd.DataFrame(rows)
