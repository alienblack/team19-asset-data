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
from sklearn.metrics import confusion_matrix, f1_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

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
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "distance_to_sea_km",
    "covering",
    "length_km",
)
DEFAULT_NEIGHBORS = 7
DEFAULT_CANDIDATE_NEIGHBORS = (3, 5, 7, 9)
DEFAULT_MIN_CONFIDENCE = 0.50
DEFAULT_MIN_CATEGORY_SUPPORT = 25


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
    exact_matches = distances <= 1e-12
    # Treat exact matches explicitly instead of creating an unstable 1/e-12
    # weight. Multiple exact matches share the vote equally.
    weights = (
        exact_matches.astype(float)
        if exact_matches.any()
        else 1.0 / distances
    )
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

    ``material_prediction`` is the candidate value for every row.  The
    confidence score combines weighted neighbour agreement with a bounded
    distance score. When the combined score is below ``min_confidence``,
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
                "confidence_score",
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
        # An exact feature match is stronger evidence than a large-distance
        # neighbour, so do not penalise a unanimous exact match by its other
        # distant neighbours.
        distance_score = (
            1.0
            if (row_distances <= 1e-12).any()
            else float(1.0 / (1.0 + mean_distance))
        )
        confidence_score = float(agreement * distance_score)
        if confidence_score >= 0.65:
            quality = "High"
        elif confidence_score >= 0.50:
            quality = "Medium"
        elif confidence_score >= min_confidence:
            quality = "Low"
        else:
            quality = "Insufficient evidence"
        abstained = confidence_score < min_confidence
        evidence.append(
            {
                "material_prediction": prediction,
                "material_imputed": prediction if not abstained else pd.NA,
                "weighted_neighbor_agreement": agreement,
                "confidence_score": confidence_score,
                "mean_neighbor_distance": mean_distance,
                "max_neighbor_distance": max_distance,
                "distance_evidence_score": distance_score,
                "neighbor_count": int(len(row_indices)),
                "quality_flag": quality,
                "abstained": abstained,
                "fallback_reason": (
                    "confidence_score_below_threshold" if abstained else ""
                ),
            }
        )
    return pd.DataFrame(evidence, index=rows.index)


def feature_association_report(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    min_category_support: int = DEFAULT_MIN_CATEGORY_SUPPORT,
) -> pd.DataFrame:
    """Create multiclass association and support-aware leakage diagnostics."""
    if min_category_support < 1:
        raise ValueError("min_category_support must be at least 1.")
    target_codes = pd.factorize(frame["material"].astype("string"), sort=True)[0]
    rows: list[dict[str, object]] = []
    for column in feature_columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            unique_count = int(series.nunique(dropna=True))
            if unique_count <= 1:
                value = 0.0
            else:
                filled = series.fillna(float(series.median())).to_numpy().reshape(-1, 1)
                n_bins = min(10, unique_count)
                binned = KBinsDiscretizer(
                    n_bins=n_bins,
                    encode="ordinal",
                    strategy="quantile",
                    subsample=None,
                ).fit_transform(filled).ravel()
                value = float(normalized_mutual_info_score(target_codes, binned))
            association_type = "normalized_mutual_information"
            warning = value >= 0.95
        else:
            codes, _ = pd.factorize(series.astype("string").fillna("<MISSING>"), sort=True)
            value = float(normalized_mutual_info_score(target_codes, codes))
            association_type = "normalized_mutual_information"
            table = pd.crosstab(series.astype("string").fillna("<MISSING>"), frame["material"])
            category_support = table.sum(axis=1)
            supported_table = table.loc[category_support >= min_category_support]
            purity = supported_table.div(supported_table.sum(axis=1), axis=0).max(axis=1)
            supported_max_purity = float(purity.max()) if not purity.empty else float("nan")
            warning = bool(value >= 0.95 or supported_max_purity >= 0.98)
        if pd.api.types.is_numeric_dtype(series):
            supported_max_purity = float("nan")
            supported_category_count = 0
        else:
            supported_category_count = int((table.sum(axis=1) >= min_category_support).sum())
        rows.append(
            {
                "feature": column,
                "feature_type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
                "n_unique": int(series.nunique(dropna=False)),
                "association_type": association_type,
                "association_value": value,
                "leakage_warning": warning,
                "min_category_support": int(min_category_support),
                "supported_category_count": supported_category_count,
                "supported_max_category_purity": supported_max_purity,
            }
        )
    return pd.DataFrame(rows)


def select_n_neighbors(
    development: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    candidate_neighbors: Sequence[int] = DEFAULT_CANDIDATE_NEIGHBORS,
    seed: int = 0,
) -> tuple[int, pd.DataFrame]:
    """Select ``k`` using stratified validation inside development data only."""
    candidates = sorted({int(value) for value in candidate_neighbors})
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError("candidate_neighbors must contain positive integers.")
    validate_knn_input(development, feature_columns)
    class_counts = development["material"].value_counts()
    n_splits = min(3, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("At least two examples per material class are required for k selection.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    rows: list[dict[str, object]] = []
    labels = development["material"].astype(str).to_numpy()
    for candidate in candidates:
        fold_scores: list[float] = []
        effective_k = 0
        for train_positions, validation_positions in splitter.split(development, labels):
            effective_k = min(candidate, len(train_positions))
            model = fit_knn(development.iloc[train_positions], feature_columns, effective_k)
            predictions = predict_with_evidence(
                development.iloc[validation_positions], model, min_confidence=0.0
            )
            fold_scores.append(
                float(
                    f1_score(
                        labels[validation_positions],
                        predictions["material_prediction"].astype(str),
                        labels=list(LABELS),
                        average="macro",
                        zero_division=0,
                    )
                )
            )
        rows.append(
            {
                "candidate_k": int(candidate),
                "effective_k": int(effective_k),
                "mean_development_macro_f1": float(np.mean(fold_scores)),
                "std_development_macro_f1": float(np.std(fold_scores)),
                "n_folds": int(n_splits),
            }
        )
    selection = pd.DataFrame(rows)
    selected = int(
        selection.sort_values(
            ["mean_development_macro_f1", "candidate_k"], ascending=[False, True]
        ).iloc[0]["effective_k"]
    )
    return selected, selection


def _rate_token(rate: float) -> str:
    """Format a mask rate for an evidence filename without hard-coding 30%."""
    percentage = rate * 100
    if percentage.is_integer():
        return f"{int(percentage)}pct"
    return f"{percentage:.2f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"


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
    candidate_neighbors: Sequence[int] = DEFAULT_CANDIDATE_NEIGHBORS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    test_fraction: float = 0.20,
) -> pd.DataFrame:
    """Run KNN on the same controlled masks as the material baseline."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not seeds:
        raise ValueError("seeds must contain at least one value.")
    if not mask_rates:
        raise ValueError("mask_rates must contain at least one value.")
    frame = pd.read_csv(input_path)
    validate_knn_input(frame, feature_columns)
    output_dir.mkdir(parents=True, exist_ok=True)

    association = feature_association_report(frame, feature_columns)
    association.to_csv(output_dir / "feature_associations.csv", index=False)
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    tuning_frames: list[pd.DataFrame] = []

    for seed in seeds:
        development_positions, evaluation_positions = split_indices(
            len(frame), int(seed), test_fraction
        )
        development = frame.iloc[development_positions]
        selected_k, tuning = select_n_neighbors(
            development,
            feature_columns=feature_columns,
            candidate_neighbors=candidate_neighbors,
            seed=int(seed),
        )
        tuning.insert(0, "seed", int(seed))
        tuning_frames.append(tuning)
        model = fit_knn(development, feature_columns, selected_k)
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
        [
            "candidate_accuracy",
            "candidate_balanced_accuracy",
            "candidate_macro_f1",
            "accepted_accuracy",
            "accepted_balanced_accuracy",
            "accepted_macro_f1",
            "coverage",
        ]
    ].mean().to_csv(output_dir / "knn_metrics_summary.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "knn_confusion_matrix.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "mask_manifest.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "knn_predictions.csv", index=False)
    tuning_results = pd.concat(tuning_frames, ignore_index=True)
    tuning_results.to_csv(output_dir / "knn_k_selection.csv", index=False)
    sample_seed = int(seeds[0])
    sample_rate = float(mask_rates[0])
    sample = predictions[
        (predictions["seed"] == sample_seed)
        & (predictions["mask_rate"] == sample_rate)
    ]
    sample_name = f"knn_imputed_material_seed{sample_seed}_rate{_rate_token(sample_rate)}.csv"
    sample.to_csv(output_dir / sample_name, index=False)

    metadata = {
        "input_file": str(input_path),
        "target": "material",
        "method": "similar_asset_knn",
        "feature_columns": list(feature_columns),
        "excluded_leakage_column": "conductor_type",
        "excluded_identifier_columns": ["segment_id", "data_status", "record_version"],
        "candidate_neighbors": [int(value) for value in candidate_neighbors],
        "k_selection": "three-fold stratified development-only macro-F1 selection per outer seed",
        "min_confidence": float(min_confidence),
        "mask_rates": [float(rate) for rate in mask_rates],
        "seeds": [int(seed) for seed in seeds],
        "test_fraction": float(test_fraction),
        "split_and_mask_compatibility": "Reuses material_baseline.split_indices and select_mask_positions with the same seed/rate formula.",
        "confidence_definition": "weighted neighbour agreement multiplied by 1/(1+mean transformed neighbour distance); not a calibrated probability",
        "abstention_definition": "material_imputed is blank when combined confidence_score is below min_confidence",
        "truth_handling": "material_true is evaluation evidence only and is never used in preprocessing or neighbour fitting",
        "sample_output": sample_name,
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
    parser.add_argument(
        "--candidate-neighbors",
        nargs="+",
        type=int,
        default=list(DEFAULT_CANDIDATE_NEIGHBORS),
    )
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
        candidate_neighbors=args.candidate_neighbors,
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
