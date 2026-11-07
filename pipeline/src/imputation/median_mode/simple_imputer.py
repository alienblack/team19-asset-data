from __future__ import annotations

from typing import Sequence

import pandas as pd


SUPPORTED_NUMERIC_STRATEGIES = {"mean"}
SUPPORTED_CATEGORICAL_STRATEGIES = {"mode"}


def deterministic_mode(series: pd.Series) -> object:
    """Return a deterministic mode.

    Missing values are ignored. If several values have the same highest
    frequency, their string representations are sorted and the first value is
    returned so the result is reproducible.
    """
    observed = series.dropna()
    counts = observed.value_counts()

    if counts.empty:
        raise ValueError(
            f"Categorical column '{series.name}' has no observed values."
        )

    max_count = counts.max()

    candidates = counts[counts == max_count].index

    return sorted(
        candidates,
        key=lambda value: str(value),
    )[0]


def _validate_columns(
    incomplete: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate requested imputation columns."""

    numeric = tuple(numeric_columns)
    categorical = tuple(categorical_columns)

    overlap = sorted(
        set(numeric).intersection(categorical)
    )

    if overlap:
        raise ValueError(
            "Columns cannot be both numeric and categorical: "
            + ", ".join(overlap)
        )

    requested = set(numeric).union(categorical)

    missing_columns = sorted(
        requested.difference(incomplete.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input is missing requested columns: "
            + ", ".join(missing_columns)
        )

    return numeric, categorical


def _validate_strategies(
    numeric_strategy: str,
    categorical_strategy: str,
) -> None:
    """Validate configured simple-imputation strategies."""

    if numeric_strategy not in SUPPORTED_NUMERIC_STRATEGIES:
        raise ValueError(
            f"Unsupported numeric strategy '{numeric_strategy}'. "
            f"Supported strategies: "
            f"{', '.join(sorted(SUPPORTED_NUMERIC_STRATEGIES))}."
        )

    if categorical_strategy not in SUPPORTED_CATEGORICAL_STRATEGIES:
        raise ValueError(
            f"Unsupported categorical strategy '{categorical_strategy}'. "
            f"Supported strategies: "
            f"{', '.join(sorted(SUPPORTED_CATEGORICAL_STRATEGIES))}."
        )


def _impute_numeric_mean(
    series: pd.Series,
) -> pd.Series:
    """Fill missing numeric values using the observed column mean."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().sum() == 0:
        raise ValueError(
            f"Numeric column '{series.name}' has no observed values."
        )

    fill_value = float(
        numeric.mean()
    )

    return numeric.fillna(
        fill_value
    )


def _impute_categorical_mode(
    series: pd.Series,
) -> pd.Series:
    """Fill missing categorical values using a deterministic mode."""

    fill_value = deterministic_mode(
        series
    )

    return series.fillna(
        fill_value
    )


def simple_impute(
    incomplete: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    numeric_strategy: str = "mean",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """Impute missing values using simple deterministic strategies.

    Parameters
    ----------
    incomplete:
        Input dataframe containing missing values.

    numeric_columns:
        Columns treated as numeric.

    categorical_columns:
        Columns treated as categorical.

    numeric_strategy:
        Strategy for numeric columns. Currently supports ``mean``.

    categorical_strategy:
        Strategy for categorical columns. Currently supports ``mode``.

    Returns
    -------
    pd.DataFrame
        A deep copy of the input with missing values imputed only in the
        requested columns.

    Notes
    -----
    Columns not explicitly listed as numeric or categorical are left unchanged.
    The input dataframe is never modified in place.
    """

    numeric_columns, categorical_columns = _validate_columns(
        incomplete=incomplete,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    _validate_strategies(
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
    )

    imputed = incomplete.copy(
        deep=True
    )

    if numeric_strategy == "mean":
        for column in numeric_columns:
            imputed[column] = _impute_numeric_mean(
                imputed[column]
            )

    if categorical_strategy == "mode":
        for column in categorical_columns:
            imputed[column] = _impute_categorical_mode(
                imputed[column]
            )

    return imputed