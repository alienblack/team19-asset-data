#!/usr/bin/env python3
"""
PGS219-117 - Artificial Missingness Scenario Generator

Generates reproducible MCAR (Missing Completely At Random) masked CSV files
from a complete/reference underground-cable CSV dataset.

Default target columns:
    - asset_age_years
    - groundwater_indicator
    - corrosion_indicator

Default scenarios:
    - 5%
    - 10%
    - 20%
    - 30%

For every scenario this script writes:
    1. A masked dataset CSV
    2. A boolean mask CSV showing exactly which cells were deliberately removed

Existing missing values in the reference dataset are preserved and are NOT
counted as artificial missingness.

Example
-------
python create_missingness.py \
    --input underground_cable_reference_synthetic_data.csv \
    --output-dir missingness_scenarios \
    --columns asset_age_years groundwater_indicator corrosion_indicator
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_RATES = (0.05, 0.10, 0.20, 0.30)
DEFAULT_RANDOM_STATE = 42

DEFAULT_TARGET_COLUMNS = (
    "asset_age_years",
    "groundwater_indicator",
    "corrosion_indicator",
)


def validate_rate(rate: float) -> None:
    """Validate that a missingness rate is between 0 and 1."""
    if not 0 <= rate <= 1:
        raise ValueError(
            f"Invalid missingness rate {rate}. "
            "Rates must be between 0 and 1, e.g. 0.10 for 10%."
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


def inject_mcar(
    df: pd.DataFrame,
    columns: Iterable[str],
    missing_rate: float,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Introduce MCAR artificial missingness into selected columns.

    Parameters
    ----------
    df:
        Reference dataset.

    columns:
        Columns in which known values may be deliberately removed.

    missing_rate:
        Fraction of currently observed values to remove from EACH selected
        column. For example, 0.20 masks 20% of observed values in each column.

    random_state:
        Seed used for reproducibility.

    Returns
    -------
    masked_df:
        Copy of the reference dataset with artificial NaNs inserted.

    artificial_mask:
        Boolean DataFrame. True means that cell was deliberately masked by
        this script. Existing NaNs in the reference dataset remain False.
    """
    validate_rate(missing_rate)
    columns = validate_columns(df, columns)

    masked_df = df.copy(deep=True)

    artificial_mask = pd.DataFrame(
        False,
        index=df.index,
        columns=df.columns,
        dtype=bool,
    )

    rng = np.random.default_rng(random_state)

    for column in columns:
        # Only non-missing reference values are eligible for artificial masking.
        eligible_indices = df.index[df[column].notna()].to_numpy()

        n_to_mask = round(
            len(eligible_indices) * missing_rate
        )

        if n_to_mask == 0:
            continue

        selected_indices = rng.choice(
            eligible_indices,
            size=n_to_mask,
            replace=False,
        )

        masked_df.loc[selected_indices, column] = np.nan
        artificial_mask.loc[selected_indices, column] = True

    return masked_df, artificial_mask


def save_scenario(
    masked_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    output_dir: Path,
    rate: float,
    prefix: str,
) -> tuple[Path, Path]:
    """Save a masked dataset and its corresponding artificial-mask CSV."""
    percentage = int(round(rate * 100))

    data_path = (
        output_dir
        / f"{prefix}_mcar_{percentage:02d}.csv"
    )

    mask_path = (
        output_dir
        / f"{prefix}_mcar_{percentage:02d}_mask.csv"
    )

    masked_df.to_csv(
        data_path,
        index=False,
    )

    mask_df.to_csv(
        mask_path,
        index=False,
    )

    return data_path, mask_path


def generate_scenarios(
    input_file: Path,
    output_dir: Path,
    columns: Iterable[str] = DEFAULT_TARGET_COLUMNS,
    rates: Iterable[float] = DEFAULT_RATES,
    random_state: int = DEFAULT_RANDOM_STATE,
    prefix: str = "underground_cable",
) -> None:
    """Load the reference CSV and generate all requested MCAR scenarios."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Reference CSV not found: {input_file}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(input_file)

    target_columns = validate_columns(
        df,
        columns,
    )

    rates = tuple(rates)

    for rate in rates:
        validate_rate(rate)

    print(f"Reference CSV: {input_file}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        "Target columns: "
        + ", ".join(target_columns)
    )
    print(f"Random seed: {random_state}")
    print()

    for rate in rates:
        masked_df, mask_df = inject_mcar(
            df=df,
            columns=target_columns,
            missing_rate=rate,
            random_state=random_state,
        )

        data_path, mask_path = save_scenario(
            masked_df=masked_df,
            mask_df=mask_df,
            output_dir=output_dir,
            rate=rate,
            prefix=prefix,
        )

        print(f"Scenario: {rate:.0%}")
        print(f"  Masked dataset: {data_path}")
        print(f"  Mask file:      {mask_path}")

        for column in target_columns:
            observed_count = int(
                df[column].notna().sum()
            )

            masked_count = int(
                mask_df[column].sum()
            )

            actual_rate = (
                masked_count / observed_count
                if observed_count > 0
                else 0.0
            )

            print(
                f"    {column}: "
                f"{masked_count}/{observed_count} "
                f"({actual_rate:.1%}) artificially masked"
            )

        print()

    print(
        "Finished generating artificial "
        "missingness scenarios."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible MCAR artificial-missingness CSV files "
            "from an underground-cable reference CSV."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the complete/reference CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where masked CSV files will be written.",
    )

    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_TARGET_COLUMNS),
        help=(
            "Target columns to mask. Default: "
            "asset_age_years groundwater_indicator corrosion_indicator"
        ),
    )

    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=list(DEFAULT_RATES),
        help=(
            "Missingness rates as decimals. "
            "Default: 0.05 0.10 0.20 0.30"
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducibility. Default: 42",
    )

    parser.add_argument(
        "--prefix",
        default="underground_cable",
        help=(
            "Filename prefix for generated files. "
            "Default: underground_cable"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generate_scenarios(
        input_file=args.input,
        output_dir=args.output_dir,
        columns=args.columns,
        rates=args.rates,
        random_state=args.random_state,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
