"""
Baseline Imputation

Applies a simple, explainable baseline imputation method to a masked
underground-cable CSV dataset.

Default target columns:
    - asset_age_years
    - groundwater_indicator
    - corrosion_indicator

Imputation rules:
    - Numeric columns: median
    - Non-numeric/categorical columns: mode

The script reads a masked CSV and writes a new imputed CSV.
It does not compare the imputed data against the reference
dataset and does not calculate validation metrics.

Example
-------
python baseline_imputation.py \
    --input data/synthetic/underground_cables/missingness_scenarios/underground_cable_mcar_10.csv \
    --output data/synthetic/underground_cables/imputed/underground_cable_mcar_10_baseline_imputed.csv

If --output is omitted, the output file is created beside the input file
with "_baseline_imputed" appended to the filename.

You can override the default target columns if needed:

python baseline_imputation.py \
    --input masked.csv \
    --columns asset_age_years groundwater_indicator corrosion_indicator
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_TARGET_COLUMNS = (
    "asset_age_years",
    "groundwater_indicator",
    "corrosion_indicator",
)


def validate_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> list[str]:
    """Validate that all requested target columns exist in the dataset."""
    columns = list(columns)

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "The following target columns were not found in the CSV: "
            + ", ".join(missing)
        )

    return columns


def get_imputation_value(series: pd.Series):
    """
    Return the baseline imputation value for a column.

    Numeric columns use the median.
    Non-numeric/categorical columns use the mode.
    """
    observed = series.dropna()

    if observed.empty:
        raise ValueError(
            f"Column '{series.name}' contains no observed values, "
            "so a baseline imputation value cannot be calculated."
        )

    if pd.api.types.is_numeric_dtype(series):
        return observed.median(), "median"

    modes = observed.mode(dropna=True)

    if modes.empty:
        raise ValueError(
            f"Could not calculate a mode for column '{series.name}'."
        )

    # If several values are tied for the mode, pandas returns them in a
    # deterministic order. The first value is used for reproducibility.
    return modes.iloc[0], "mode"


def baseline_impute(
    df: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_TARGET_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """
    Impute missing values in selected columns using simple baseline rules.

    Parameters
    ----------
    df:
        Masked dataset.

    columns:
        Columns to impute.

    Returns
    -------
    imputed_df:
        Copy of the dataset with missing values in the selected columns filled.

    summary:
        Dictionary describing the method, fill value, and number of values
        imputed for each selected column.
    """
    target_columns = validate_columns(df, columns)
    imputed_df = df.copy(deep=True)

    summary: dict[str, dict[str, object]] = {}

    for column in target_columns:
        missing_count = int(imputed_df[column].isna().sum())

        if missing_count == 0:
            summary[column] = {
                "method": "none",
                "fill_value": None,
                "imputed_count": 0,
            }
            continue

        fill_value, method = get_imputation_value(imputed_df[column])

        imputed_df[column] = imputed_df[column].fillna(fill_value)

        summary[column] = {
            "method": method,
            "fill_value": fill_value,
            "imputed_count": missing_count,
        }

    return imputed_df, summary


def default_output_path(input_file: Path) -> Path:
    """Create the default output filename beside the input CSV."""
    return input_file.with_name(
        f"{input_file.stem}_baseline_imputed.csv"
    )


def impute_csv(
    input_file: Path,
    output_file: Path | None = None,
    columns: Iterable[str] = DEFAULT_TARGET_COLUMNS,
) -> Path:
    """Load a masked CSV, apply baseline imputation, and save the result."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Masked CSV not found: {input_file}"
        )

    if input_file.suffix.lower() != ".csv":
        raise ValueError(
            f"Input file must be a CSV file: {input_file}"
        )

    if output_file is None:
        output_file = default_output_path(input_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(input_file)

    imputed_df, summary = baseline_impute(
        df=df,
        columns=columns,
    )

    imputed_df.to_csv(
        output_file,
        index=False,
    )

    print(f"Input CSV:  {input_file}")
    print(f"Output CSV: {output_file}")
    print(f"Rows: {len(df):,}")
    print()

    print("Baseline imputation summary:")

    for column, details in summary.items():
        method = details["method"]
        count = details["imputed_count"]
        fill_value = details["fill_value"]

        if method == "none":
            print(
                f"  {column}: no missing values; nothing imputed"
            )
        else:
            print(
                f"  {column}: {count} value(s) imputed "
                f"using {method} = {fill_value}"
            )

    print()
    print("Finished baseline imputation.")

    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply explainable baseline imputation to a masked "
            "underground-cable CSV."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the masked CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the imputed CSV. If omitted, the output is saved "
            "beside the input with '_baseline_imputed' appended."
        ),
    )

    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_TARGET_COLUMNS),
        help=(
            "Columns to impute. Default: "
            "asset_age_years groundwater_indicator corrosion_indicator"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    impute_csv(
        input_file=args.input,
        output_file=args.output,
        columns=args.columns,
    )


if __name__ == "__main__":
    main()
