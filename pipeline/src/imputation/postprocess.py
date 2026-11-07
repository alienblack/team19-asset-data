"""Generic post-processing for imputed asset data."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


SUPPORTED_DATA_TYPES = {
    "integer",
    "float",
    "categorical",
}


def _validate_frames(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
) -> None:
    """Validate that original and imputed frames are structurally compatible."""

    if not isinstance(original, pd.DataFrame):
        raise TypeError(
            "Original input must be a pandas DataFrame."
        )

    if not isinstance(imputed, pd.DataFrame):
        raise TypeError(
            "Imputed input must be a pandas DataFrame."
        )

    if len(original) != len(imputed):
        raise ValueError(
            "Original and imputed dataframes must have the same number of rows."
        )

    if not original.index.equals(imputed.index):
        raise ValueError(
            "Original and imputed dataframes must have the same index."
        )

    if list(original.columns) != list(imputed.columns):
        raise ValueError(
            "Original and imputed dataframes must have the same columns "
            "and column order."
        )


def _validate_constraint(
    column: str,
    constraint: Mapping[str, Any],
) -> None:
    """Validate one configured post-processing constraint."""

    if not isinstance(constraint, Mapping):
        raise ValueError(
            f"Constraint for '{column}' must be a mapping."
        )

    data_type = constraint.get("data_type")

    if data_type not in SUPPORTED_DATA_TYPES:
        raise ValueError(
            f"Unsupported data_type '{data_type}' "
            f"for column '{column}'."
        )

    minimum = constraint.get("min")
    maximum = constraint.get("max")

    if (
        minimum is not None
        and maximum is not None
        and minimum > maximum
    ):
        raise ValueError(
            f"Constraint minimum exceeds maximum "
            f"for column '{column}'."
        )


def _numeric_imputed_values(
    series: pd.Series,
    missing_mask: pd.Series,
    column: str,
) -> pd.Series:
    """Return numeric values only for cells that were originally missing."""

    values = pd.to_numeric(
        series.loc[missing_mask],
        errors="coerce",
    )

    invalid_count = int(
        values.isna().sum()
    )

    if invalid_count:
        raise ValueError(
            f"Imputed column '{column}' contains "
            f"{invalid_count} missing or non-numeric value(s) "
            "in originally missing positions."
        )

    return values


def _apply_numeric_constraint(
    result: pd.DataFrame,
    original: pd.DataFrame,
    column: str,
    constraint: Mapping[str, Any],
) -> None:
    """Apply integer/float constraints only to originally missing cells."""

    missing_mask = original[column].isna()

    if not missing_mask.any():
        return

    values = _numeric_imputed_values(
        series=result[column],
        missing_mask=missing_mask,
        column=column,
    )

    minimum = constraint.get("min")
    maximum = constraint.get("max")

    if minimum is not None:
        values = values.clip(
            lower=minimum
        )

    if maximum is not None:
        values = values.clip(
            upper=maximum
        )

    if constraint["data_type"] == "integer":
        values = values.round()

    result.loc[
        missing_mask,
        column,
    ] = values


def _apply_categorical_constraint(
    result: pd.DataFrame,
    original: pd.DataFrame,
    column: str,
) -> None:
    """Validate categorical imputations without altering observed values."""

    missing_mask = original[column].isna()

    if not missing_mask.any():
        return

    imputed_values = result.loc[
        missing_mask,
        column,
    ]

    missing_count = int(
        imputed_values.isna().sum()
    )

    if missing_count:
        raise ValueError(
            f"Imputed categorical column '{column}' still contains "
            f"{missing_count} missing value(s) "
            "in originally missing positions."
        )


def apply_constraints(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    constraints: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Apply configured constraints to imputed cells only.

    Parameters
    ----------
    original:
        Dataset before imputation. Missing positions in this dataframe identify
        which cells are allowed to be post-processed.

    imputed:
        Dataset returned by an imputation method.

    constraints:
        Asset-specific constraints, usually loaded from an imputation YAML
        configuration.

        Example::

            {
                "installation_year": {
                    "data_type": "integer",
                    "min": 1930,
                    "max": 2014,
                },
                "length_km": {
                    "data_type": "float",
                    "min": 0.0,
                },
            }

    Returns
    -------
    pd.DataFrame
        A deep copy of ``imputed`` with configured constraints applied only to
        cells that were missing in ``original``.

    Notes
    -----
    Originally observed values are never deliberately rounded or clipped.
    Validation should still be run afterwards to detect whether an imputation
    method itself modified observed values.
    """

    _validate_frames(
        original=original,
        imputed=imputed,
    )

    if not isinstance(constraints, Mapping):
        raise TypeError(
            "constraints must be a mapping."
        )

    result = imputed.copy(
        deep=True
    )

    for column, constraint in constraints.items():

        if column not in result.columns:
            raise ValueError(
                f"Constraint defined for missing column '{column}'."
            )

        _validate_constraint(
            column=column,
            constraint=constraint,
        )

        data_type = constraint["data_type"]

        if data_type in {
            "integer",
            "float",
        }:
            _apply_numeric_constraint(
                result=result,
                original=original,
                column=column,
                constraint=constraint,
            )

        elif data_type == "categorical":
            _apply_categorical_constraint(
                result=result,
                original=original,
                column=column,
            )

    return result