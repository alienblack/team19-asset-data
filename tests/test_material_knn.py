import numpy as np
import pandas as pd
import pytest

from team19_asset_data.overhead_conductors.material_knn import (
    feature_association_report,
    fit_knn,
    predict_with_evidence,
    select_n_neighbors,
    validate_knn_input,
)


def frame_for_knn() -> pd.DataFrame:
    rows = []
    for index in range(30):
        material = ["Aluminium", "Copper", "Steel"][index % 3]
        rows.append(
            {
                "segment_id": f"S{index:02d}",
                "feeder_id": f"F{index % 5}",
                "installation_year": 1980 + index,
                "asset_age": 34 - index,
                "voltage_category": ["LV", "33 kV", "11/7.6 kV"][index % 3],
                "location_group": ["Urban", "Regional", "Rural"][index % 3],
                "corrosion_zone": ["Low", "Severe", "Very Severe"][index % 3],
                "distance_to_sea_km": float(index + 1),
                "covering": ["Bare", "Covered"][index % 2],
                "length_km": float(index + 1) / 10,
                "material": material,
            }
        )
    return pd.DataFrame(rows)


def test_default_features_do_not_double_weight_age() -> None:
    from team19_asset_data.overhead_conductors.material_knn import DEFAULT_FEATURES

    assert "installation_year" in DEFAULT_FEATURES
    assert "asset_age" not in DEFAULT_FEATURES


def test_preprocessing_statistics_are_fit_from_development_rows() -> None:
    frame = frame_for_knn().iloc[:6].copy()
    frame.loc[4:5, "voltage_category"] = "33 kV"
    features = ["installation_year", "voltage_category"]
    model = fit_knn(frame, features, n_neighbors=3)
    numeric = model.preprocessor.named_transformers_["numeric"]
    categorical = model.preprocessor.named_transformers_["categorical"]
    assert numeric.named_steps["imputer"].statistics_.tolist() == [1982.5]
    assert numeric.named_steps["scaler"].mean_.tolist() == [1982.5]
    assert categorical.named_steps["imputer"].statistics_.tolist() == ["33 kV"]


def test_missing_numeric_and_categorical_predictors_are_preprocessed() -> None:
    frame = frame_for_knn().iloc[:9].copy()
    frame.loc[1, "installation_year"] = np.nan
    frame.loc[2, "voltage_category"] = None
    model = fit_knn(frame, ["installation_year", "voltage_category"], n_neighbors=3)
    query = frame.iloc[[0]].copy()
    query.loc[:, "installation_year"] = np.nan
    query.loc[:, "voltage_category"] = None
    evidence = predict_with_evidence(query, model, min_confidence=0.0)
    assert len(evidence) == 1
    assert evidence["material_prediction"].notna().all()


def test_knn_returns_expected_material_for_similar_asset() -> None:
    frame = frame_for_knn().iloc[:6].copy()
    features = ["installation_year", "voltage_category"]
    model = fit_knn(frame, features, n_neighbors=3)
    query = pd.DataFrame(
        {
            "segment_id": ["Q"],
            "feeder_id": ["FQ"],
            "installation_year": [1981],
            "voltage_category": ["33 kV"],
            "material": ["Copper"],
        }
    )
    evidence = predict_with_evidence(query, model, min_confidence=0.0)
    assert evidence.loc[0, "material_prediction"] == "Copper"


def test_low_agreement_causes_abstention() -> None:
    frame = pd.DataFrame(
        {
            "segment_id": ["A", "C", "S"],
            "feeder_id": ["F1", "F2", "F3"],
            "installation_year": [0, 10, 20],
            "material": ["Aluminium", "Copper", "Steel"],
        }
    )
    model = fit_knn(frame, ["installation_year"], n_neighbors=3)
    query = frame.iloc[[0]].copy()
    query.loc[:, "installation_year"] = 5
    evidence = predict_with_evidence(query, model, min_confidence=0.50)
    assert bool(evidence.loc[0, "abstained"])
    assert pd.isna(evidence.loc[0, "material_imputed"])
    assert evidence.loc[0, "fallback_reason"] == "confidence_score_below_threshold"


def test_exact_matches_are_handled_without_extreme_weights() -> None:
    frame = pd.DataFrame(
        {
            "segment_id": ["C1", "C2", "S1"],
            "feeder_id": ["F1", "F2", "F3"],
            "installation_year": [2000, 2000, 2010],
            "material": ["Copper", "Copper", "Steel"],
        }
    )
    model = fit_knn(frame, ["installation_year"], n_neighbors=3)
    evidence = predict_with_evidence(frame.iloc[[0]], model, min_confidence=0.50)
    assert evidence.loc[0, "material_prediction"] == "Copper"
    assert evidence.loc[0, "weighted_neighbor_agreement"] == pytest.approx(1.0)
    assert evidence.loc[0, "confidence_score"] == pytest.approx(1.0)
    assert not bool(evidence.loc[0, "abstained"])


def test_k_selection_uses_development_rows_and_caps_large_k() -> None:
    frame = frame_for_knn()
    selected, tuning = select_n_neighbors(
        frame,
        ["installation_year", "voltage_category"],
        candidate_neighbors=[1, 3, 100],
        seed=101,
    )
    assert selected in set(tuning["effective_k"])
    assert set(tuning["candidate_k"]) == {1, 3, 100}
    assert (tuning["effective_k"] <= 20).all()


def test_invalid_feature_columns_and_neighbor_values_are_rejected() -> None:
    frame = frame_for_knn()
    with pytest.raises(ValueError, match="material"):
        validate_knn_input(frame, ["material"])
    with pytest.raises(ValueError, match="segment_id"):
        validate_knn_input(frame, ["segment_id"])
    with pytest.raises(ValueError, match="missing"):
        validate_knn_input(frame, ["not_a_column"])
    with pytest.raises(ValueError, match="conductor_type"):
        validate_knn_input(frame.assign(conductor_type="AAC"), ["conductor_type"])
    missing_target = frame.copy()
    missing_target.loc[0, "material"] = None
    with pytest.raises(ValueError, match="complete reference"):
        fit_knn(missing_target, ["installation_year"], n_neighbors=3)
    with pytest.raises(ValueError, match="n_neighbors"):
        fit_knn(frame, ["installation_year"], n_neighbors=0)
    model = fit_knn(frame.iloc[:4], ["installation_year"], n_neighbors=100)
    assert model.n_neighbors == 4


def test_association_report_uses_multiclass_measure_and_support_filter() -> None:
    frame = frame_for_knn()
    report = feature_association_report(
        frame,
        ["installation_year", "voltage_category"],
        min_category_support=10,
    )
    assert set(report["association_type"]) == {"normalized_mutual_information"}
    assert "pearson" not in " ".join(report["association_type"])
    assert report["supported_category_count"].iloc[1] == 3


def test_empty_experiment_inputs_are_rejected(tmp_path) -> None:
    from team19_asset_data.overhead_conductors.material_knn import run_experiment

    source = tmp_path / "data.csv"
    frame_for_knn().to_csv(source, index=False)
    with pytest.raises(ValueError, match="seeds"):
        run_experiment(source, tmp_path / "out", seeds=[])
    with pytest.raises(ValueError, match="mask_rates"):
        run_experiment(source, tmp_path / "out", mask_rates=[])
