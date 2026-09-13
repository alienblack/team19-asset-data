"""Similar-asset KNN imputation for overhead-conductor material.

This module extends the material baseline experiment with a transparent
similarity-based classifier.  Preprocessing and neighbour search are fitted on
development rows only.  The hidden evaluation rows are used only for scoring,
so the original material values remain a controlled benchmark.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .material_baseline import (
    DEFAULT_INPUT,
    DEFAULT_MASK_RATES,
    DEFAULT_SEEDS,
    LABELS,
    evaluate_predictions,
    select_mask_positions,
    split_indices,
    validate_input,
)


DEFAULT_OUTPUT = Path("outputs/PGS219-128")
DEFAULT_FEATURES = (
    "installation_year",
    "asset_age",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "distance_to_sea_km",
    "covering",
    "length_km",
)
DEFAULT_NEIGHBORS = 7
DEFAULT_MIN_CONFIDENCE = 0.50


@dataclass
class KNNMaterialModel:
    """Fitted preprocessing and neighbour index for material prediction."""

    preprocessor: ColumnTransformer
    neighbours: NearestNeighbors
    training_labels: np.ndarray
    feature_columns: tuple[str, ...]
    n_neighbors: int


def validate_knn_input(frame: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    """Validate features and reject target/leakage columns explicitly."""
    validate_input(frame)
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing KNN feature columns: {', '.join(missing)}")
    if "material" in feature_columns:
        raise ValueError("The target column 'material' cannot be used as a KNN feature.")
    if "conductor_type" in feature_columns:
        raise ValueError("conductor_type is excluded because it can reveal material directly.")
    protected = {"segment_id", "data_status", "record_version"}
    invalid = sorted(set(feature_columns).intersection(protected))
    if invalid:
        raise ValueError(f"Identifier/status columns cannot be KNN features: {', '.join(invalid)}")


def _feature_types(frame: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[list[str], list[str]]:
    numeric = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = [column for column in feature_columns if column not in numeric]
    return numeric, categorical


def _build_preprocessor(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> ColumnTransformer:
    """Build train-fitted imputation, scaling and categorical encoding steps."""
    numeric, categorical = _feature_types(frame, feature_columns)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )


def fit_knn(
    development: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    n_neighbors: int = DEFAULT_NEIGHBORS,
) -> KNNMaterialModel:
    """Fit preprocessing and a neighbour index from development rows only."""
    validate_knn_input(development, feature_columns)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if len(development) < 1:
        raise ValueError("At least one development row is required.")

    columns = tuple(feature_columns)
    preprocessor = _build_preprocessor(development, columns)
    transformed = preprocessor.fit_transform(development.loc[:, columns])
    effective_neighbors = min(int(n_neighbors), len(development))
    neighbours = NearestNeighbors(n_neighbors=effective_neighbors, metric="euclidean")
    neighbours.fit(transformed)
    return KNNMaterialModel(
        preprocessor=preprocessor,
        neighbours=neighbours,
        training_labels=development["material"].astype(str).to_numpy(),
        feature_columns=columns,
        n_neighbors=effective_neighbors,
    )


def _weighted_vote(labels: np.ndarray, distances: np.ndarray) -> tuple[str, float]:
    """Return a deterministic weighted vote and its agreement score."""
    if len(labels) == 0:
        raise ValueError("Cannot vote without neighbours.")
    weights = 1.0 / np.maximum(distances, 1e-12)
    scores = {label: 0.0 for label in LABELS}
    for label, weight in zip(labels, weights):
        scores[str(label)] = scores.get(str(label), 0.0) + float(weight)
    prediction = max(LABELS, key=lambda label: (scores[label], -LABELS.index(label)))
    agreement = scores[prediction] / float(weights.sum())
    return prediction, float(agreement)


def predict_with_evidence(
    rows: pd.DataFrame,
    model: KNNMaterialModel,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> pd.DataFrame:
    """Predict material and return neighbour, distance and quality evidence.

    ``material_prediction`` is the candidate value for every row.  When the
    weighted neighbour agreement is below ``min_confidence``,
    ``material_imputed`` is left blank and ``abstained`` is set to true.  This
    preserves a useful suggestion without silently treating weak evidence as a
    verified replacement.
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1.")
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "material_prediction",
                "material_imputed",
                "weighted_neighbor_agreement",
                "mean_neighbor_distance",
                "max_neighbor_distance",
                "distance_evidence_score",
                "neighbor_count",
                "quality_flag",
                "abstained",
                "fallback_reason",
            ]
        )

    transformed = model.preprocessor.transform(rows.loc[:, model.feature_columns])
    distances, indices = model.neighbours.kneighbors(transformed)
    evidence: list[dict[str, object]] = []
    for row_distances, row_indices in zip(distances, indices):
        neighbour_labels = model.training_labels[row_indices]
        prediction, agreement = _weighted_vote(neighbour_labels, row_distances)
        mean_distance = float(np.mean(row_distances))
        max_distance = float(np.max(row_distances))
        distance_score = float(1.0 / (1.0 + mean_distance))
        if agreement >= 0.80:
            quality = "High"
        elif agreement >= 0.60:
            quality = "Medium"
        elif agreement >= min_confidence:
            quality = "Low"
        else:
            quality = "Insufficient evidence"
        abstained = agreement < min_confidence
        evidence.append(
            {
                "material_prediction": prediction,
                "material_imputed": prediction if not abstained else pd.NA,
                "weighted_neighbor_agreement": agreement,
                "mean_neighbor_distance": mean_distance,
                "max_neighbor_distance": max_distance,
                "distance_evidence_score": distance_score,
                "neighbor_count": int(len(row_indices)),
                "quality_flag": quality,
                "abstained": abstained,
                "fallback_reason": (
                    "weighted_neighbor_agreement_below_threshold" if abstained else ""
                ),
            }
        )
    return pd.DataFrame(evidence, index=rows.index)


def feature_association_report(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """Create lightweight numeric/categorical leakage diagnostics."""
    rows: list[dict[str, object]] = []
    for column in feature_columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            value = float(abs(series.corr(frame["material"].map({label: i for i, label in enumerate(LABELS)}))))
            association_type = "absolute_pearson_correlation"
            warning = value >= 0.95
        else:
            table = pd.crosstab(series, frame["material"])
            purity = table.div(table.sum(axis=1), axis=0).max(axis=1)
            value = float(purity.max()) if not purity.empty else float("nan")
            association_type = "maximum_category_material_purity"
            warning = bool(value >= 0.98 and table.shape[0] > 1)
        rows.append(
            {
                "feature": column,
                "feature_type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
                "n_unique": int(series.nunique(dropna=False)),
                "association_type": association_type,
                "association_value": value,
                "leakage_warning": warning,
            }
        )
    return pd.DataFrame(rows)


def _metric_row(
    truth: np.ndarray,
    predictions: pd.DataFrame,
    seed: int,
    rate: float,
) -> dict[str, object]:
    candidate = predictions["material_prediction"].astype(str).to_numpy()
    accepted_mask = ~predictions["abstained"].astype(bool).to_numpy()
    accepted = predictions.loc[accepted_mask, "material_prediction"].astype(str).to_numpy()
    accepted_truth = truth[accepted_mask]
    candidate_metrics = evaluate_predictions(truth, candidate)
    accepted_metrics = (
        evaluate_predictions(accepted_truth, accepted)
        if len(accepted)
        else {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan")}
    )
    candidate_matrix = confusion_matrix(truth, candidate, labels=list(LABELS))
    accepted_matrix = (
        confusion_matrix(accepted_truth, accepted, labels=list(LABELS))
        if len(accepted)
        else np.zeros((len(LABELS), len(LABELS)), dtype=int)
    )
    candidate_recall = np.divide(
        np.diag(candidate_matrix),
        candidate_matrix.sum(axis=1),
        out=np.zeros(len(LABELS), dtype=float),
        where=candidate_matrix.sum(axis=1) != 0,
    )
    accepted_recall = np.divide(
        np.diag(accepted_matrix),
        accepted_matrix.sum(axis=1),
        out=np.full(len(LABELS), np.nan, dtype=float),
        where=accepted_matrix.sum(axis=1) != 0,
    )
    return {
        "seed": int(seed),
        "mask_rate": float(rate),
        "n_masked": int(len(truth)),
        "n_accepted": int(accepted_mask.sum()),
        "n_abstained": int((~accepted_mask).sum()),
        "coverage": float(accepted_mask.mean()) if len(accepted_mask) else 0.0,
        "candidate_accuracy": candidate_metrics["accuracy"],
        "candidate_balanced_accuracy": candidate_metrics["balanced_accuracy"],
        "candidate_macro_f1": candidate_metrics["macro_f1"],
        "accepted_accuracy": accepted_metrics["accuracy"],
        "accepted_balanced_accuracy": accepted_metrics["balanced_accuracy"],
        "accepted_macro_f1": accepted_metrics["macro_f1"],
        **{
            f"candidate_recall_{label}": float(score)
            for label, score in zip(LABELS, candidate_recall)
        },
        **{
            f"accepted_recall_{label}": float(score)
            for label, score in zip(LABELS, accepted_recall)
        },
    }


def run_experiment(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    mask_rates: Sequence[float] = DEFAULT_MASK_RATES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    n_neighbors: int = DEFAULT_NEIGHBORS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    test_fraction: float = 0.20,
) -> pd.DataFrame:
    """Run KNN on the same controlled masks as the material baseline."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    frame = pd.read_csv(input_path)
    validate_knn_input(frame, feature_columns)
    output_dir.mkdir(parents=True, exist_ok=True)

    association = feature_association_report(frame, feature_columns)
    association.to_csv(output_dir / "feature_associations.csv", index=False)
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for seed in seeds:
        development_positions, evaluation_positions = split_indices(
            len(frame), int(seed), test_fraction
        )
        development = frame.iloc[development_positions]
        model = fit_knn(development, feature_columns, n_neighbors)
        for rate in mask_rates:
            mask_seed = int(seed) + int(round(float(rate) * 10_000))
            masked_positions = select_mask_positions(evaluation_positions, float(rate), mask_seed)
            masked_rows = frame.iloc[masked_positions].copy()
            truth = masked_rows["material"].astype(str).to_numpy()
            predictions = predict_with_evidence(masked_rows, model, min_confidence)
            predictions.insert(0, "segment_id", masked_rows["segment_id"].to_numpy())
            predictions.insert(0, "mask_rate", float(rate))
            predictions.insert(0, "seed", int(seed))
            predictions.insert(3, "material_true", truth)
            predictions["method"] = "similar_asset_knn"
            prediction_frames.append(predictions)
            metric_rows.append(_metric_row(truth, predictions, int(seed), float(rate)))

            accepted_mask = ~predictions["abstained"].astype(bool).to_numpy()
            scopes = {
                "candidate_predictions": (
                    truth,
                    predictions["material_prediction"].astype(str).to_numpy(),
                ),
                "accepted_predictions": (
                    truth[accepted_mask],
                    predictions.loc[accepted_mask, "material_prediction"].astype(str).to_numpy(),
                ),
            }
            for scope, (scope_truth, scope_predictions) in scopes.items():
                if len(scope_predictions) == 0:
                    continue
                matrix = confusion_matrix(scope_truth, scope_predictions, labels=list(LABELS))
                for actual_index, actual in enumerate(LABELS):
                    for predicted_index, predicted in enumerate(LABELS):
                        confusion_rows.append(
                            {
                                "seed": int(seed),
                                "mask_rate": float(rate),
                                "scope": scope,
                                "actual_material": actual,
                                "predicted_material": predicted,
                                "count": int(matrix[actual_index, predicted_index]),
                            }
                        )
            for segment_id, actual in zip(masked_rows["segment_id"], truth):
                manifest_rows.append(
                    {
                        "seed": int(seed),
                        "mask_rate": float(rate),
                        "segment_id": segment_id,
                        "material_true": actual,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "knn_metrics.csv", index=False)
    metrics.groupby("mask_rate", as_index=False)[
        ["candidate_accuracy", "candidate_balanced_accuracy", "candidate_macro_f1", "coverage"]
    ].mean().to_csv(output_dir / "knn_metrics_summary.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "knn_confusion_matrix.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "mask_manifest.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "knn_predictions.csv", index=False)
    sample = predictions[(predictions["seed"] == int(seeds[0])) & (predictions["mask_rate"] == 0.30)]
    sample.to_csv(output_dir / "knn_imputed_material_seed101_rate30.csv", index=False)

    metadata = {
        "input_file": str(input_path),
        "target": "material",
        "method": "similar_asset_knn",
        "feature_columns": list(feature_columns),
        "excluded_leakage_column": "conductor_type",
        "excluded_identifier_columns": ["segment_id", "data_status", "record_version"],
        "n_neighbors": int(n_neighbors),
        "min_confidence": float(min_confidence),
        "mask_rates": [float(rate) for rate in mask_rates],
        "seeds": [int(seed) for seed in seeds],
        "test_fraction": float(test_fraction),
        "split_and_mask_compatibility": "Reuses material_baseline.split_indices and select_mask_positions with the same seed/rate formula.",
        "confidence_definition": "weighted neighbour agreement; distance evidence is reported separately and is not a calibrated probability",
        "abstention_definition": "material_imputed is blank when weighted neighbour agreement is below min_confidence",
        "truth_handling": "material_true is evaluation evidence only and is never used in preprocessing or neighbour fitting",
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run similar-asset KNN material imputation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mask-rates", nargs="+", type=float, default=list(DEFAULT_MASK_RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(
        input_path=args.input,
        output_dir=args.output_dir,
        mask_rates=args.mask_rates,
        seeds=args.seeds,
        feature_columns=args.features,
        n_neighbors=args.n_neighbors,
        min_confidence=args.min_confidence,
        test_fraction=args.test_fraction,
    )
    summary = metrics.groupby("mask_rate", as_index=False)[
        ["candidate_accuracy", "candidate_balanced_accuracy", "candidate_macro_f1", "coverage"]
    ].mean()
    print(summary.to_string(index=False))
    print(f"\nEvidence written to {args.output_dir}")


if __name__ == "__main__":
    main()
