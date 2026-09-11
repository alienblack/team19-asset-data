"""Shared loading, validation, status, and confidence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class ValidationInputError(ValueError):
    """Raised when validation inputs are absent or inconsistent."""


def require_value(value: str | None, description: str) -> str:
    """Return a non-empty configuration value or raise a helpful error."""
    if value is None or not str(value).strip():
        raise ValidationInputError(
            f"{description} is not configured. Provide it in config.py or via the command line."
        )
    return str(value).strip()


def read_csv(path_value: str | Path, label: str) -> pd.DataFrame:
    """Read a CSV file and convert common I/O failures into clear messages."""
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ValidationInputError(f"{label} file does not exist or is not a file: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValidationInputError(f"Could not read {label} file '{path}': {exc}") from exc


def load_datasets(
    reference_path: str | Path,
    masked_path: str | Path,
    baseline_path: str | Path,
    alternative_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load the four datasets required by the framework."""
    return {
        "reference": read_csv(reference_path, "Reference"),
        "masked": read_csv(masked_path, "Masked"),
        "baseline": read_csv(baseline_path, "Baseline"),
        "alternative": read_csv(alternative_path, "Alternative"),
    }


def validate_datasets(
    datasets: dict[str, pd.DataFrame], target_column: str, id_column: str | None = None
) -> pd.Series:
    """Validate dataset alignment and return the shared artificial-mask positions."""
    target_column = require_value(target_column, "Target column")
    lengths = {name: len(frame) for name, frame in datasets.items()}
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={size}" for name, size in lengths.items())
        raise ValidationInputError(f"Dataset row counts must match ({detail}).")

    missing_target = [name for name, frame in datasets.items() if target_column not in frame]
    if missing_target:
        raise ValidationInputError(
            f"Target column '{target_column}' is missing from: {', '.join(missing_target)}."
        )

    if id_column:
        missing_id = [name for name, frame in datasets.items() if id_column not in frame]
        if missing_id:
            raise ValidationInputError(
                f"ID column '{id_column}' is missing from: {', '.join(missing_id)}."
            )
        reference_ids = datasets["reference"][id_column].reset_index(drop=True)
        for name, frame in datasets.items():
            if not reference_ids.equals(frame[id_column].reset_index(drop=True)):
                raise ValidationInputError(
                    f"ID values/order in {name} do not match the reference dataset."
                )

    # A hidden-known value is present in the complete reference and absent from
    # the masked copy. Natural missing values in the reference are not scored.
    masked_positions = (
        datasets["masked"][target_column].isna()
        & datasets["reference"][target_column].notna()
    )
    if not masked_positions.any():
        raise ValidationInputError(
            f"No artificial masked positions were found for target '{target_column}'. "
            "Expected values present in reference and missing in masked."
        )
    for name in ("baseline", "alternative"):
        if datasets[name].loc[masked_positions, target_column].notna().sum() == 0:
            raise ValidationInputError(
                f"{name.title()} predictions are all missing at the validation positions."
            )
    return masked_positions


def infer_target_type(reference_values: pd.Series) -> str:
    """Classify the target as numeric or categorical from its reference dtype."""
    return "numeric" if pd.api.types.is_numeric_dtype(reference_values.dtype) else "categorical"


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create and return all output subdirectories."""
    root = Path(output_dir).expanduser()
    paths = {
        "root": root,
        "metrics": root / "metrics",
        "distributions": root / "distributions",
        "downstream": root / "downstream",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def assign_confidence(
    statuses: Iterable[str], imputed_confidence: str = "TBD"
) -> pd.Series:
    """Assign preliminary confidence labels without inventing certainty.

    Observed values are High. Imputed values default to TBD; callers may supply
    High, Medium, or Low only when experimental evidence supports that choice.
    Derived remains TBD because confidence depends on the future derivation rule.
    This is an internal project convention, not an official SAPN standard.
    """
    allowed = {"High", "Medium", "Low", "TBD"}
    if imputed_confidence not in allowed:
        raise ValidationInputError(
            f"Imputed confidence must be one of {sorted(allowed)}; got '{imputed_confidence}'."
        )
    mapping = {"Observed": "High", "Imputed": imputed_confidence, "Derived": "TBD"}
    result = pd.Series(statuses).map(mapping)
    if result.isna().any():
        bad = sorted(pd.Series(statuses)[result.isna()].astype(str).unique())
        raise ValidationInputError(f"Unsupported data status value(s): {bad}")
    return result


def prepare_status_confidence(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    masked_positions: pd.Series,
    id_column: str | None = None,
    imputed_confidence: str = "TBD",
) -> pd.DataFrame:
    """Build row/method provenance and preliminary confidence records."""
    records: list[pd.DataFrame] = []
    for method in ("baseline", "alternative"):
        status = pd.Series("Observed", index=datasets[method].index, dtype="object")
        filled = masked_positions & datasets[method][target_column].notna()
        status.loc[filled] = "Imputed"
        # A future derived-field integration can set selected rows to Derived.
        frame = pd.DataFrame(
            {
                "row_number": range(len(status)),
                "method": method.title(),
                "target_field": target_column,
                "data_status": status,
                "confidence": assign_confidence(status, imputed_confidence).to_numpy(),
            }
        )
        if id_column:
            frame.insert(1, id_column, datasets[method][id_column].to_numpy())
        records.append(frame)
    return pd.concat(records, ignore_index=True)

