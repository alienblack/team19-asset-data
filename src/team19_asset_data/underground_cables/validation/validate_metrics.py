"""UC-S2-07: hidden-known-value validation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

try:
    from .utils import ValidationInputError, infer_target_type
except ImportError:
    from utils import ValidationInputError, infer_target_type


def calculate_validation_metrics(
    datasets: dict[str, pd.DataFrame], target_column: str, masked_positions: pd.Series
) -> pd.DataFrame:
    """Score both methods on the same artificial-mask positions only."""
    truth = datasets["reference"].loc[masked_positions, target_column]
    target_type = infer_target_type(truth)
    rows = []
    for key, label in (("baseline", "Baseline"), ("alternative", "Alternative")):
        prediction = datasets[key].loc[masked_positions, target_column]
        usable = truth.notna() & prediction.notna()
        if not usable.any():
            raise ValidationInputError(f"{label} has no usable predictions at masked positions.")
        row = {
            "method": label,
            "target_field": target_column,
            "n_masked": int(masked_positions.sum()),
            "n_evaluated": int(usable.sum()),
            "mae": np.nan,
            "rmse": np.nan,
            "accuracy": np.nan,
            "f1": np.nan,
        }
        y_true, y_pred = truth.loc[usable], prediction.loc[usable]
        if target_type == "numeric":
            try:
                y_true = pd.to_numeric(y_true, errors="raise")
                y_pred = pd.to_numeric(y_pred, errors="raise")
            except Exception as exc:
                raise ValidationInputError(f"{label} contains non-numeric target predictions: {exc}") from exc
            row["mae"] = mean_absolute_error(y_true, y_pred)
            row["rmse"] = mean_squared_error(y_true, y_pred) ** 0.5
        else:
            y_true, y_pred = y_true.astype(str), y_pred.astype(str)
            row["accuracy"] = accuracy_score(y_true, y_pred)
            row["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        rows.append(row)
    return pd.DataFrame(rows)


def save_validation_metrics(metrics: pd.DataFrame, output_path) -> None:
    """Save metric results to CSV."""
    metrics.to_csv(output_path, index=False)

