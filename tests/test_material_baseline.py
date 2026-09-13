from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from team19_asset_data.overhead_conductors.material_baseline import (
    evaluate_predictions,
    fit_group_modes,
    fit_global_mode,
    predict_group_mode,
    select_mask_positions,
    split_indices,
    validate_input,
)
from team19_asset_data.overhead_conductors.material_knn import (
    fit_knn,
    predict_with_evidence,
    validate_knn_input,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "feeder_id": ["F1", "F1", "F2", "F2", "F3", "F3"],
            "installation_year": [1980, 1981, 1990, 1991, 2000, 2001],
            "asset_age": [34, 33, 24, 23, 14, 13],
            "voltage_category": ["LV", "LV", "33 kV", "33 kV", "11/7.6 kV", "11/7.6 kV"],
            "location_group": ["Urban", "Urban", "Regional", "Regional", "Rural", "Rural"],
            "corrosion_zone": ["Low", "Low", "Severe", "Severe", "Very Severe", "Very Severe"],
            "distance_to_sea_km": [10.0, 10.5, 50.0, 50.5, 100.0, 100.5],
            "covering": ["Bare", "Bare", "Covered", "Covered", "Bare", "Bare"],
            "length_km": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            "material": ["Steel", "Steel", "Copper", "Copper", "Aluminium", "Steel"],
        }
    )


def test_validate_input_accepts_complete_material_data() -> None:
    validate_input(sample_frame())


def test_validate_input_rejects_missing_material() -> None:
    frame = sample_frame()
    frame.loc[0, "material"] = None
    with pytest.raises(ValueError, match="complete reference"):
        validate_input(frame)


def test_modes_are_fit_from_development_rows() -> None:
    frame = sample_frame()
    development = frame.iloc[[0, 1, 2, 3]]
    assert fit_global_mode(development) == "Copper"
    group_modes = fit_group_modes(development, ["feeder_id"])
    predictions, fallback = predict_group_mode(
        frame.iloc[[0, 2, 4]], group_modes, "Copper", ["feeder_id"]
    )
    assert predictions.tolist() == ["Steel", "Copper", "Copper"]
    assert fallback.tolist() == [False, False, True]


def test_split_and_mask_are_reproducible_and_disjoint() -> None:
    development_a, evaluation_a = split_indices(100, seed=101)
    development_b, evaluation_b = split_indices(100, seed=101)
    masked_a = select_mask_positions(evaluation_a, 0.30, seed=401)
    masked_b = select_mask_positions(evaluation_b, 0.30, seed=401)
    assert np.array_equal(development_a, development_b)
    assert np.array_equal(evaluation_a, evaluation_b)
    assert np.array_equal(masked_a, masked_b)
    assert set(development_a).isdisjoint(set(evaluation_a))
    assert len(masked_a) == 6


def test_metrics_include_balanced_and_macro_scores() -> None:
    result = evaluate_predictions(
        ["Steel", "Copper", "Aluminium"], ["Steel", "Steel", "Aluminium"]
    )
    assert set(result) == {"accuracy", "balanced_accuracy", "macro_f1"}
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_knn_fits_preprocessing_on_training_rows_and_returns_evidence() -> None:
    frame = sample_frame()
    features = ["installation_year", "asset_age", "voltage_category", "location_group"]
    validate_knn_input(frame, features)
    model = fit_knn(frame.iloc[:4], features, n_neighbors=3)
    evidence = predict_with_evidence(frame.iloc[4:], model, min_confidence=0.50)
    assert len(evidence) == 2
    assert {
        "material_prediction",
        "material_imputed",
        "weighted_neighbor_agreement",
        "mean_neighbor_distance",
        "distance_evidence_score",
        "quality_flag",
        "abstained",
        "fallback_reason",
    }.issubset(evidence.columns)
    assert evidence["neighbor_count"].eq(3).all()


def test_knn_rejects_conductor_type_leakage() -> None:
    frame = sample_frame()
    frame["conductor_type"] = ["SC/AC", "SC/AC", "Copper", "Copper", "AAC", "SC/AC"]
    with pytest.raises(ValueError, match="conductor_type"):
        validate_knn_input(frame, ["conductor_type"])
