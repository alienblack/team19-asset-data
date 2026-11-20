"""Regression tests for the new fitted / reusable PredictiveMissForest API.

Small deterministic synthetic fixtures: assert contracts, not predictive accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from team19_asset_data.imputation.predictive_missforest import (
    PredictiveMissForest,
    PredictiveMissForestConfig,
)

NUMERIC = ("installation_year", "distance_to_sea_km")
CATEGORICAL = (
    "voltage_category", "location_group", "corrosion_zone", "material", "covering"
)
MODEL_COLS = NUMERIC + CATEGORICAL
CONSTRAINTS = {
    "installation_year": {"data_type": "integer", "min": 1930, "max": 2014},
    "distance_to_sea_km": {"data_type": "float", "min": 0.0, "max": 350.0},
}


def make_population(n=90, seed=42):
    """Complete, invented records. No fixture contains real SAPN data."""
    r = np.random.default_rng(seed)
    year = r.integers(1950, 2005, size=n)
    distance = r.uniform(0, 250, size=n)
    voltage = np.where(year < 1975, "SWER", "LV")
    location = np.where(distance > 90, "Rural", "Urban")
    corrosion = np.where(distance < 40, "Very Severe", "Low")
    material = np.where(year < 1975, "Steel", "Aluminium")
    covering = np.where(year > 1992, "Covered", "Bare")
    return pd.DataFrame({
        "segment_id": [f"TEST_{i:05d}" for i in range(n)],
        "installation_year": year.astype(float),
        "distance_to_sea_km": distance,
        "voltage_category": voltage,
        "location_group": location,
        "corrosion_zone": corrosion,
        "material": material,
        "covering": covering,
        "data_status": ["synthetic"] * n,  # Excluded field must survive unchanged.
    })


def mask_some(frame, seed=11):
    result = frame.copy(deep=True)
    r = np.random.default_rng(seed)
    for col in MODEL_COLS:
        result.loc[r.random(len(result)) < 0.22, col] = np.nan
    return result


@pytest.fixture
def fitted():
    config = PredictiveMissForestConfig(
        n_estimators=6, n_passes=2, max_depth=5, min_samples_leaf=2,
        random_state=219, n_jobs=1,
    )
    train = mask_some(make_population(), seed=11)
    model = PredictiveMissForest(NUMERIC, CATEGORICAL, config, CONSTRAINTS)
    model.fit(train)
    return model, train


def test_fit_and_transform_fill_only_missing_values(fitted):
    model, train = fitted
    output = model.transform(train)
    assert output.shape == train.shape
    assert output.index.equals(train.index)
    assert output.columns.tolist() == train.columns.tolist()
    assert not output[list(MODEL_COLS)].isna().any().any()
    for col in MODEL_COLS:
        known = train[col].notna()
        pd.testing.assert_series_equal(
            output.loc[known, col], train.loc[known, col],
            check_dtype=False,
        )
    pd.testing.assert_frame_equal(output[["segment_id", "data_status"]],
                                  train[["segment_id", "data_status"]])
    assert model.fit_report_["n_models"] == len(MODEL_COLS) * model.config.n_passes


def test_replay_on_new_batch_allows_whole_missing_column_and_row(fitted):
    model, _ = fitted
    new = mask_some(make_population(15, seed=802), seed=92)
    new.loc[:, "covering"] = np.nan
    new.loc[new.index[0], list(MODEL_COLS)] = np.nan
    original = new.copy(deep=True)
    predicted = model.transform(new)
    assert not predicted[list(MODEL_COLS)].isna().any().any()
    pd.testing.assert_frame_equal(new, original)  # Model cannot mutate caller input.
    for col in MODEL_COLS:
        known = original[col].notna()
        pd.testing.assert_series_equal(
            predicted.loc[known, col], original.loc[known, col], check_dtype=False
        )
    pd.testing.assert_frame_equal(predicted[["segment_id", "data_status"]],
                                  original[["segment_id", "data_status"]])


def test_model_save_load_same_predictions_and_no_retraining(fitted, tmp_path, monkeypatch):
    model, _ = fitted
    new = mask_some(make_population(16, seed=99), seed=19)
    before = model.transform(new)
    path = tmp_path / "pmf.joblib"
    model.save(path)
    loaded = PredictiveMissForest.load(path)
    assert loaded.is_fitted
    assert len(loaded.models_) == len(model.models_)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("transform() unexpectedly called fit()")

    with monkeypatch.context() as patch:
        patch.setattr(Pipeline, "fit", forbidden_fit)
        replay = loaded.transform(new)
    pd.testing.assert_frame_equal(before, replay)


def test_forest_training_labels_exclude_missing_target_placeholders(monkeypatch):
    train = mask_some(make_population(80, seed=3), seed=7)
    config = PredictiveMissForestConfig(n_estimators=5, n_passes=1, n_jobs=1)
    actual_fit = Pipeline.fit
    captured = []

    def capture(self, X, y=None, **kwargs):
        target = next(col for col in MODEL_COLS if col not in X.columns)
        captured.append((target, X.index.copy(), y.copy()))
        return actual_fit(self, X, y, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Pipeline, "fit", capture)
        PredictiveMissForest(NUMERIC, CATEGORICAL, config, CONSTRAINTS).fit(train)
    assert len(captured) == len(MODEL_COLS)
    for col, indices, labels in captured:
        expected = train.loc[train[col].notna(), col]
        assert indices.equals(expected.index)
        assert labels.index.equals(expected.index)
        assert labels.notna().all()
        if col in NUMERIC:
            np.testing.assert_allclose(labels.astype(float), expected.astype(float))
        else:
            assert labels.astype(str).tolist() == expected.astype(str).tolist()


def test_transform_rejects_missing_required_input_column(fitted):
    model, _ = fitted
    with pytest.raises(ValueError, match="missing requested model columns"):
        model.transform(make_population(10).drop(columns=["covering"]))
