"""Evaluate imputation reconstruction against clean synthetic ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from team19_asset_data.imputation.config import (
    load_imputation_config,
)


def _load_csv(
    path: Path,
    description: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )

    return pd.read_csv(path)


def _validate_alignment(
    clean: pd.DataFrame,
    simple: pd.DataFrame,
    missforest: pd.DataFrame,
) -> None:
    """Verify all evaluated datasets refer to the same records."""

    if not (
        len(clean)
        == len(simple)
        == len(missforest)
    ):
        raise ValueError(
            "Clean, Simple and MissForest datasets "
            "must have the same number of rows."
        )

    if "segment_id" in clean.columns:
        for name, frame in (
            ("Simple", simple),
            ("MissForest", missforest),
        ):
            if "segment_id" not in frame.columns:
                raise ValueError(
                    f"{name} output is missing segment_id."
                )

            if not (
                clean["segment_id"]
                .astype(str)
                .reset_index(drop=True)
                .equals(
                    frame["segment_id"]
                    .astype(str)
                    .reset_index(drop=True)
                )
            ):
                raise ValueError(
                    f"{name} segment_id order does not "
                    "match the clean source."
                )


def _build_mask_lookup(
    truth: pd.DataFrame,
    clean: pd.DataFrame,
    model_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Build a full-row boolean mask from masking truth metadata."""

    if "segment_id" not in truth.columns:
        raise ValueError(
            "truth dataset must contain segment_id."
        )

    if "segment_id" not in clean.columns:
        raise ValueError(
            "clean dataset must contain segment_id."
        )

    truth_indexed = truth.copy()

    truth_indexed["segment_id"] = (
        truth_indexed["segment_id"]
        .astype(str)
    )

    if truth_indexed["segment_id"].duplicated().any():
        raise ValueError(
            "truth dataset contains duplicate segment_id values."
        )

    clean_ids = (
        clean["segment_id"]
        .astype(str)
    )

    mask_lookup = pd.DataFrame(
        False,
        index=clean.index,
        columns=model_columns,
        dtype=bool,
    )

    truth_indexed = truth_indexed.set_index(
        "segment_id"
    )

    for column in model_columns:
        flag_column = f"{column}__masked"

        if flag_column not in truth_indexed.columns:
            raise ValueError(
                f"Mask flag '{flag_column}' "
                "not found in truth dataset."
            )

        flags = (
            truth_indexed[
                flag_column
            ]
            .astype(bool)
        )

        # mask_lookup[column] = (
        #     clean_ids
        #     .map(flags)
        #     .fillna(False)
        #     .astype(bool)
        #     .to_numpy()
        # )
        mask_lookup[column] = (
            clean_ids
            .map(flags)
            .eq(True)
            .to_numpy(dtype=bool)
        )

    return mask_lookup


def _numeric_metrics(
    truth: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Calculate numeric reconstruction metrics."""

    y_true = pd.to_numeric(
        truth,
        errors="raise",
    ).astype(float)

    y_pred = pd.to_numeric(
        predicted,
        errors="raise",
    ).astype(float)

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    absolute_errors = np.abs(
        y_true.to_numpy()
        - y_pred.to_numpy()
    )

    median_absolute_error = float(
        np.median(
            absolute_errors
        )
    )

    data_range = float(
        y_true.max()
        - y_true.min()
    )

    if data_range > np.finfo(float).eps:
        normalized_mae = (
            mae / data_range
        )

        normalized_rmse = (
            rmse / data_range
        )

    else:
        normalized_mae = float("nan")
        normalized_rmse = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "median_absolute_error": median_absolute_error,
        "normalized_mae_range": normalized_mae,
        "normalized_rmse_range": normalized_rmse,
    }


def _categorical_metrics(
    truth: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Calculate categorical reconstruction metrics."""

    y_true = truth.astype(str)
    y_pred = predicted.astype(str)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _evaluate_method(
    method_name: str,
    predictions: pd.DataFrame,
    clean: pd.DataFrame,
    mask_lookup: pd.DataFrame,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
) -> tuple[
    list[dict[str, Any]],
    dict[str, pd.DataFrame],
]:
    """Evaluate one imputation method."""

    rows: list[
        dict[str, Any]
    ] = []

    confusion_tables: dict[
        str,
        pd.DataFrame
    ] = {}

    for column in numeric_columns:
        mask = mask_lookup[
            column
        ]

        masked_count = int(
            mask.sum()
        )

        if masked_count == 0:
            continue

        truth_values = clean.loc[
            mask,
            column,
        ]

        predicted_values = predictions.loc[
            mask,
            column,
        ]

        metrics = _numeric_metrics(
            truth=truth_values,
            predicted=predicted_values,
        )

        for metric_name, value in metrics.items():
            rows.append(
                {
                    "method": method_name,
                    "column": column,
                    "column_type": "numeric",
                    "masked_count": masked_count,
                    "metric": metric_name,
                    "value": value,
                }
            )

    for column in categorical_columns:
        mask = mask_lookup[
            column
        ]

        masked_count = int(
            mask.sum()
        )

        if masked_count == 0:
            continue

        truth_values = clean.loc[
            mask,
            column,
        ].astype(str)

        predicted_values = predictions.loc[
            mask,
            column,
        ].astype(str)

        metrics = _categorical_metrics(
            truth=truth_values,
            predicted=predicted_values,
        )

        for metric_name, value in metrics.items():
            rows.append(
                {
                    "method": method_name,
                    "column": column,
                    "column_type": "categorical",
                    "masked_count": masked_count,
                    "metric": metric_name,
                    "value": value,
                }
            )

        labels = sorted(
            set(
                truth_values.tolist()
            ).union(
                predicted_values.tolist()
            )
        )

        matrix = confusion_matrix(
            truth_values,
            predicted_values,
            labels=labels,
        )

        confusion_tables[column] = pd.DataFrame(
            matrix,
            index=[
                f"actual::{label}"
                for label in labels
            ],
            columns=[
                f"predicted::{label}"
                for label in labels
            ],
        )

    return rows, confusion_tables


def evaluate_experiment(
    clean_path: Path,
    truth_path: Path,
    simple_path: Path,
    missforest_path: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    """Evaluate Simple and MissForest against clean synthetic truth."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_imputation_config(
        config_path
    )

    clean = _load_csv(
        clean_path,
        "Clean source",
    )

    truth = _load_csv(
        truth_path,
        "Mask truth",
    )

    simple = _load_csv(
        simple_path,
        "Simple output",
    )

    missforest = _load_csv(
        missforest_path,
        "MissForest output",
    )

    _validate_alignment(
        clean=clean,
        simple=simple,
        missforest=missforest,
    )

    mask_lookup = _build_mask_lookup(
        truth=truth,
        clean=clean,
        model_columns=config.model_columns,
    )

    mask_summary = pd.DataFrame(
        {
            "column": list(
                config.model_columns
            ),
            "masked_count": [
                int(
                    mask_lookup[
                        column
                    ].sum()
                )
                for column in config.model_columns
            ],
        }
    )

    mask_summary[
        "masked_rate"
    ] = (
        mask_summary[
            "masked_count"
        ]
        / len(clean)
    )

    mask_summary.to_csv(
        output_dir
        / "evaluation_mask_summary.csv",
        index=False,
    )

    all_rows: list[
        dict[str, Any]
    ] = []

    simple_rows, simple_confusions = (
        _evaluate_method(
            method_name="simple",
            predictions=simple,
            clean=clean,
            mask_lookup=mask_lookup,
            numeric_columns=config.numeric_columns,
            categorical_columns=config.categorical_columns,
        )
    )

    missforest_rows, missforest_confusions = (
        _evaluate_method(
            method_name="missforest",
            predictions=missforest,
            clean=clean,
            mask_lookup=mask_lookup,
            numeric_columns=config.numeric_columns,
            categorical_columns=config.categorical_columns,
        )
    )

    all_rows.extend(
        simple_rows
    )

    all_rows.extend(
        missforest_rows
    )

    metrics = pd.DataFrame(
        all_rows
    )

    metrics.to_csv(
        output_dir
        / "reconstruction_metrics.csv",
        index=False,
    )

    comparison = (
        metrics.pivot_table(
            index=[
                "column",
                "column_type",
                "metric",
                "masked_count",
            ],
            columns="method",
            values="value",
        )
        .reset_index()
    )

    if (
        "simple" in comparison.columns
        and "missforest"
        in comparison.columns
    ):
        comparison[
            "missforest_minus_simple"
        ] = (
            comparison["missforest"]
            - comparison["simple"]
        )

    comparison.to_csv(
        output_dir
        / "method_comparison.csv",
        index=False,
    )

    confusion_dir = (
        output_dir
        / "confusion_matrices"
    )

    confusion_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for column, table in (
        simple_confusions.items()
    ):
        table.to_csv(
            confusion_dir
            / f"simple_{column}.csv"
        )

    for column, table in (
        missforest_confusions.items()
    ):
        table.to_csv(
            confusion_dir
            / f"missforest_{column}.csv"
        )

    metadata = {
        "clean_path": str(
            clean_path
        ),
        "truth_path": str(
            truth_path
        ),
        "simple_path": str(
            simple_path
        ),
        "missforest_path": str(
            missforest_path
        ),
        "config_path": str(
            config_path
        ),
        "row_count": int(
            len(clean)
        ),
        "evaluated_columns": list(
            config.model_columns
        ),
        "evaluation_scope": (
            "Only cells deliberately masked by the masking rule."
        ),
        "ground_truth": (
            "Clean synthetic source values."
        ),
        "numeric_metrics": [
            "mae",
            "rmse",
            "median_absolute_error",
            "normalized_mae_range",
            "normalized_rmse_range",
        ],
        "categorical_metrics": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
    }

    with (
        output_dir
        / "evaluation_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\nImputation evaluation completed successfully."
    )

    print(
        f"Rows: {len(clean)}"
    )

    print(
        f"Metrics: "
        f"{output_dir / 'reconstruction_metrics.csv'}"
    )

    print(
        f"Comparison: "
        f"{output_dir / 'method_comparison.csv'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Simple and MissForest "
            "against clean synthetic ground truth."
        )
    )

    parser.add_argument(
        "--clean",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--truth",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--simple",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--missforest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    evaluate_experiment(
        clean_path=args.clean,
        truth_path=args.truth,
        simple_path=args.simple,
        missforest_path=args.missforest,
        config_path=args.config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()