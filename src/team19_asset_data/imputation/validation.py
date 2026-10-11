"""Generic validation utilities for data-imputation methods."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def _normalise_columns(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalise and validate column-role definitions."""

    numeric = tuple(numeric_columns)
    categorical = tuple(categorical_columns)

    if len(numeric) != len(set(numeric)):
        raise ValueError(
            "numeric_columns contains duplicate column names."
        )

    if len(categorical) != len(set(categorical)):
        raise ValueError(
            "categorical_columns contains duplicate column names."
        )

    overlap = sorted(
        set(numeric).intersection(categorical)
    )

    if overlap:
        raise ValueError(
            "Columns cannot be both numeric and categorical: "
            + ", ".join(overlap)
        )

    if not numeric and not categorical:
        raise ValueError(
            "At least one numeric or categorical model column is required."
        )

    return numeric, categorical


def validate_imputation_input(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> None:
    """Validate a dataframe before running an imputation method.

    Checks
    ------
    - input is a pandas DataFrame;
    - dataframe is not empty;
    - dataframe has no duplicate column names;
    - requested model columns exist;
    - numeric and categorical roles do not overlap;
    - every requested column has at least one observed value;
    - observed values in numeric columns are genuinely numeric.

    Missing values are allowed because they are the values to be imputed.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "Imputation input must be a pandas DataFrame."
        )

    if frame.empty:
        raise ValueError(
            "Imputation input dataframe is empty."
        )

    duplicated_columns = (
        frame.columns[
            frame.columns.duplicated()
        ]
        .astype(str)
        .tolist()
    )

    if duplicated_columns:
        raise ValueError(
            "Input dataframe contains duplicate columns: "
            + ", ".join(sorted(set(duplicated_columns)))
        )

    numeric, categorical = _normalise_columns(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    model_columns = set(
        numeric + categorical
    )

    missing_columns = sorted(
        model_columns.difference(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input is missing requested model columns: "
            + ", ".join(missing_columns)
        )

    for column in numeric:
        series = frame[column]

        if series.notna().sum() == 0:
            raise ValueError(
                f"Numeric column '{column}' has no observed values."
            )

        observed = series.dropna()

        converted = pd.to_numeric(
            observed,
            errors="coerce",
        )

        invalid_count = int(
            converted.isna().sum()
        )

        if invalid_count:
            raise ValueError(
                f"Numeric column '{column}' contains "
                f"{invalid_count} non-numeric observed value(s)."
            )

    for column in categorical:
        if frame[column].notna().sum() == 0:
            raise ValueError(
                f"Categorical column '{column}' has no observed values."
            )


def validate_imputation_result(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> None:
    """Validate the output produced by an imputation method.

    The imputer is expected to fill only previously missing values in the
    requested model columns.

    Checks
    ------
    - result is a pandas DataFrame;
    - row count is unchanged;
    - index is unchanged;
    - column names and order are unchanged;
    - requested columns contain no missing values;
    - originally observed model values were not modified;
    - columns outside the model were not modified.
    """

    validate_imputation_input(
        frame=original,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    if not isinstance(imputed, pd.DataFrame):
        raise TypeError(
            "Imputation result must be a pandas DataFrame."
        )

    numeric, categorical = _normalise_columns(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    model_columns = list(
        numeric + categorical
    )

    if len(imputed) != len(original):
        raise ValueError(
            "Imputation changed the number of rows."
        )

    if not imputed.index.equals(original.index):
        raise ValueError(
            "Imputation changed the dataframe index."
        )

    if list(imputed.columns) != list(original.columns):
        raise ValueError(
            "Imputation changed dataframe columns or column order."
        )

    for column in model_columns:
        remaining_missing = int(
            imputed[column].isna().sum()
        )

        if remaining_missing:
            raise ValueError(
                f"Imputed column '{column}' still contains "
                f"{remaining_missing} missing value(s)."
            )

    # Verify that values which were originally present have not been changed.
    for column in model_columns:
        observed_mask = original[column].notna()

        original_observed = (
            original.loc[
                observed_mask,
                column,
            ]
            .reset_index(drop=True)
        )

        imputed_observed = (
            imputed.loc[
                observed_mask,
                column,
            ]
            .reset_index(drop=True)
        )

        try:
            pd.testing.assert_series_equal(
                original_observed,
                imputed_observed,
                check_names=False,
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(
                f"Imputation modified originally observed values "
                f"in column '{column}'."
            ) from error

    # Everything outside the model-ready columns must pass through unchanged.
    non_model_columns = [
        column
        for column in original.columns
        if column not in model_columns
    ]

    if non_model_columns:
        try:
            pd.testing.assert_frame_equal(
                original[non_model_columns],
                imputed[non_model_columns],
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(
                "Imputation modified one or more non-model columns."
            ) from error


def validate_imputation(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> None:
    """Convenience wrapper for validating a completed imputation."""

    validate_imputation_result(
        original=original,
        imputed=imputed,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )