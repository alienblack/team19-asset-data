"""
Advanced Imputation
MICE-style chained equations with Predictive Mean Matching (PMM).

Targets:
  - asset_age_years: numeric
  - groundwater_indicator: Low / Medium / High
  - corrosion_indicator: Low / Medium / High

Indicators are encoded internally as Low=1, Medium=2, High=3 only for
modelling. PMM copies an observed donor value, and the saved CSV restores
the text labels Low / Medium / High.

Example:

python advanced_imputation.py \
    --input data/synthetic/underground_cables/missingness_scenarios/underground_cable_mcar_10.csv \
    --output data/synthetic/underground_cables/imputed/underground_cable_mcar_10_baseline_imputed.csv

Dependencies: pandas, numpy, scikit-learn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AGE_COLUMN = "asset_age_years"
CATEGORICAL_TARGETS = ("groundwater_indicator", "corrosion_indicator")
TARGET_COLUMNS = (AGE_COLUMN, *CATEGORICAL_TARGETS)

CATEGORY_TO_CODE = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
CODE_TO_CATEGORY = {1.0: "Low", 2.0: "Medium", 3.0: "High"}

DEFAULT_ITERATIONS = 10
DEFAULT_DONORS = 5
DEFAULT_RANDOM_STATE = 42


def normalise_indicator(value):
    """Normalise Low/Medium/High while preserving missing values."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    mapping = {"low": "Low", "medium": "Medium", "high": "High"}

    if text not in mapping:
        raise ValueError(
            f"Unexpected indicator value '{value}'. "
            "Expected Low, Medium, High, or missing."
        )

    return mapping[text]


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError("Required target column(s) not found: " + ", ".join(missing))

    if not pd.api.types.is_numeric_dtype(df[AGE_COLUMN]):
        raise TypeError(f"'{AGE_COLUMN}' must be numeric.")

    for column in CATEGORICAL_TARGETS:
        df[column].map(normalise_indicator)


def encode_targets(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy(deep=True)

    for column in CATEGORICAL_TARGETS:
        normalised = encoded[column].map(normalise_indicator)
        encoded[column] = normalised.map(CATEGORY_TO_CODE)

    return encoded


def build_predictor_matrix(working_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Build numeric predictors, avoiding ID-like high-cardinality text fields."""
    predictors = working_df.drop(columns=[target]).copy()
    usable = []

    for column in predictors.columns:
        series = predictors[column]

        if pd.api.types.is_numeric_dtype(series):
            usable.append(column)
            continue

        unique_count = series.nunique(dropna=True)
        if 1 < unique_count <= 20:
            usable.append(column)

    predictors = predictors[usable].copy()

    if predictors.empty:
        raise ValueError(f"No usable predictors are available for '{target}'.")

    numeric_cols = predictors.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_cols:
        median = predictors[column].median()
        if pd.isna(median):
            predictors = predictors.drop(columns=[column])
        else:
            predictors[column] = predictors[column].fillna(median)

    categorical_cols = predictors.select_dtypes(exclude=[np.number]).columns.tolist()
    for column in categorical_cols:
        predictors[column] = predictors[column].where(
            predictors[column].notna(), "__MISSING__"
        ).astype(str)

    predictors = pd.get_dummies(
        predictors,
        columns=categorical_cols,
        drop_first=False,
        dtype=float,
    )

    if predictors.empty:
        raise ValueError(f"No usable predictors are available for '{target}'.")

    return predictors.astype(float)


def initialise_missing_values(original: pd.DataFrame, working: pd.DataFrame) -> None:
    if working[AGE_COLUMN].isna().any():
        median = original[AGE_COLUMN].median()
        if pd.isna(median):
            raise ValueError(f"'{AGE_COLUMN}' has no observed values.")
        working[AGE_COLUMN] = working[AGE_COLUMN].fillna(median)

    for column in CATEGORICAL_TARGETS:
        if not working[column].isna().any():
            continue

        modes = original[column].dropna().mode()
        if modes.empty:
            raise ValueError(f"'{column}' has no observed values.")

        working[column] = working[column].fillna(float(modes.iloc[0]))


def pmm_update(
    original: pd.DataFrame,
    working: pd.DataFrame,
    target: str,
    donors: int,
    rng: np.random.Generator,
) -> int:
    observed_mask = original[target].notna()
    missing_mask = original[target].isna()
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        return 0

    observed_count = int(observed_mask.sum())
    if observed_count < 2:
        raise ValueError(f"'{target}' has fewer than two observed values.")

    X = build_predictor_matrix(working, target)
    X_observed = X.loc[observed_mask]
    X_missing = X.loc[missing_mask]
    y_observed = original.loc[observed_mask, target].astype(float)

    model = make_pipeline(StandardScaler(), BayesianRidge())
    model.fit(X_observed, y_observed)

    predicted_observed = model.predict(X_observed)
    predicted_missing = model.predict(X_missing)
    observed_values = y_observed.to_numpy()
    k = min(donors, observed_count)

    imputed_values = []
    for prediction in predicted_missing:
        distances = np.abs(predicted_observed - prediction)
        nearest = np.argsort(distances)[:k]
        donor_position = rng.choice(nearest)
        imputed_values.append(observed_values[donor_position])

    working.loc[missing_mask, target] = imputed_values
    return missing_count


def decode_output(original_df: pd.DataFrame, imputed_encoded: pd.DataFrame) -> pd.DataFrame:
    output = original_df.copy(deep=True)
    output[AGE_COLUMN] = imputed_encoded[AGE_COLUMN]

    for column in CATEGORICAL_TARGETS:
        decoded = imputed_encoded[column].map(CODE_TO_CATEGORY)
        if decoded.isna().any():
            raise ValueError(f"Could not decode all values for '{column}'.")
        output[column] = decoded

    return output


def mice_pmm_impute(
    df: pd.DataFrame,
    iterations: int = DEFAULT_ITERATIONS,
    donors: int = DEFAULT_DONORS,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if donors < 1:
        raise ValueError("donors must be at least 1.")

    validate_input(df)

    original_df = df.copy(deep=True)
    original_encoded = encode_targets(original_df)
    working_encoded = original_encoded.copy(deep=True)

    missing_counts = {
        column: int(original_df[column].isna().sum())
        for column in TARGET_COLUMNS
    }

    initialise_missing_values(original_encoded, working_encoded)
    rng = np.random.default_rng(random_state)

    for _ in range(iterations):
        for target in TARGET_COLUMNS:
            pmm_update(
                original=original_encoded,
                working=working_encoded,
                target=target,
                donors=donors,
                rng=rng,
            )

    return decode_output(original_df, working_encoded), missing_counts


def default_output_path(input_file: Path) -> Path:
    return input_file.with_name(f"{input_file.stem}_mice_pmm_imputed.csv")


def impute_csv(
    input_file: Path,
    output_file: Path | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    donors: int = DEFAULT_DONORS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Path:
    if not input_file.exists():
        raise FileNotFoundError(f"Masked CSV not found: {input_file}")
    if input_file.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be CSV: {input_file}")

    if output_file is None:
        output_file = default_output_path(input_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file)

    imputed_df, missing_counts = mice_pmm_impute(
        df=df,
        iterations=iterations,
        donors=donors,
        random_state=random_state,
    )

    imputed_df.to_csv(output_file, index=False)

    print(f"Input CSV:      {input_file}")
    print(f"Output CSV:     {output_file}")
    print(f"Rows:           {len(df):,}")
    print(f"Iterations:     {iterations}")
    print(f"PMM donors:     {donors}")
    print(f"Random seed:    {random_state}")
    print()
    print("MICE + PMM imputation summary:")

    for column in TARGET_COLUMNS:
        print(f"  {column}: {missing_counts[column]} value(s) imputed")

    print()
    print("Categorical output remains: Low / Medium / High")
    print("Finished MICE + PMM imputation.")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PGS219-119: MICE + PMM for asset age and Low/Medium/High indicators."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Masked CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help="MICE iterations. Default: 10",
    )
    parser.add_argument(
        "--donors", type=int, default=DEFAULT_DONORS,
        help="PMM donor candidates. Default: 5",
    )
    parser.add_argument(
        "--random-state", type=int, default=DEFAULT_RANDOM_STATE,
        help="Random seed. Default: 42",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    impute_csv(
        input_file=args.input,
        output_file=args.output,
        iterations=args.iterations,
        donors=args.donors,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
