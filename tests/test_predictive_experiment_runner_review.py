"""The predictive runner must score a disjoint held-out set, never fit on it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import run_predictive_imputation_experiment as runner
from team19_asset_data.imputation.evaluation import evaluate_imputation
from team19_asset_data.imputation.predictive_missforest import PredictiveMissForest

MODEL_COLS = (
    "installation_year", "distance_to_sea_km", "voltage_category",
    "location_group", "corrosion_zone", "material", "covering",
)
NUMERIC = MODEL_COLS[:2]
CATEGORICAL = MODEL_COLS[2:]


def complete_fixture(n=100):
    r = np.random.default_rng(19)
    years = r.integers(1940, 2010, size=n)
    d = r.uniform(0, 200, size=n)
    return pd.DataFrame({
        "segment_id": [f"ID_{i:04d}" for i in range(n)],
        "installation_year": years.astype(float),
        "distance_to_sea_km": d,
        "voltage_category": np.where(years > 1970, "LV", "SWER"),
        "location_group": np.where(d > 70, "Rural", "Urban"),
        "corrosion_zone": np.where(d < 50, "Very Severe", "Low"),
        "material": np.where(years > 1970, "Aluminium", "Steel"),
        "covering": np.where(years > 1990, "Covered", "Bare"),
        "data_status": "synthetic",
    })


def config_yaml(path: Path) -> Path:
    path.write_text("""\
version: 2
asset_class: overhead_conductors
columns:
  numeric: [installation_year, distance_to_sea_km]
  categorical: [voltage_category, location_group, corrosion_zone, material, covering]
  excluded: [segment_id, data_status]
constraints:
  installation_year: {data_type: integer, min: 1930, max: 2014}
  distance_to_sea_km: {data_type: float, min: 0, max: 350}
methods:
  simple: {numeric_strategy: mean, categorical_strategy: mode}
  missforest: {n_estimators: 5, max_iter: 3, random_state: 219}
  predictive_missforest:
    n_estimators: 5
    n_passes: 1
    n_jobs: 1
    random_state: 219
    max_depth: 4
    min_samples_leaf: 1
""", encoding="utf-8")
    return path


def test_runner_disjoint_fit_and_heldout_scoring(tmp_path, monkeypatch):
    """One small end-to-end run, with actual forests and test-only metrics."""
    truth = complete_fixture()
    csv = tmp_path / "truth.csv"
    cfg = config_yaml(tmp_path / "config.yaml")
    out = tmp_path / "results"
    truth.to_csv(csv, index=False)

    original_fit = PredictiveMissForest.fit
    fit_indices = []

    def check_fit(self, X):
        fit_indices.append(set(X.index.tolist()))
        return original_fit(self, X)

    monkeypatch.setattr(PredictiveMissForest, "fit", check_fit)
    runner.main(["--input-csv", str(csv), "--config", str(cfg),
                 "--output-dir", str(out), "--mask-rate", "0.30",
                 "--seed", "219"])

    manifest = pd.read_csv(out / "partition_manifest.csv")
    assert manifest["segment_id"].is_unique
    assert set(manifest["segment_id"]) == set(truth["segment_id"])
    assert manifest["partition"].value_counts().to_dict() == {
        "fit": 60, "test": 20, "tune": 10, "calibration": 10
    }
    fitting_rows = set(manifest.loc[manifest.partition == "fit", "source_row"])
    test_rows = set(manifest.loc[manifest.partition == "test", "source_row"])
    assert fitting_rows.isdisjoint(test_rows)
    assert fit_indices == [fitting_rows]  # No fitting on tuning/calibration/test data.

    masked = pd.read_csv(out / "masked_test.csv")
    imputed = pd.read_csv(out / "predictive_imputed_test.csv")
    mask = pd.read_csv(out / "test_hidden_mask.csv")
    metrics = pd.read_csv(out / "metrics_test.csv")
    metadata = json.loads((out / "run_metadata.json").read_text())
    assert metadata["partitions"]["test"] == len(masked) == 20
    assert mask[list(MODEL_COLS)].to_numpy().any()
    assert not imputed[list(MODEL_COLS)].isna().any().any()
    assert masked.segment_id.tolist() == imputed.segment_id.tolist()
    # Input truth is reconstructed by ID in the exact test-file row order.
    heldout_truth = truth.set_index("segment_id").loc[
        masked["segment_id"]
    ].reset_index()

    for col in MODEL_COLS:
        hidden = mask[col].to_numpy(dtype=bool)
        pd.testing.assert_series_equal(
            masked.loc[~hidden, col].reset_index(drop=True),
            imputed.loc[~hidden, col].reset_index(drop=True),
            check_dtype=False,
        )
        if hidden.any():
            assert masked.loc[hidden, col].isna().all()
        # Verify all *unmasked* cells match original truth.
        pd.testing.assert_series_equal(
            masked.loc[~hidden, col].reset_index(drop=True),
            heldout_truth.loc[~hidden, col].reset_index(drop=True),
            check_dtype=False,
        )

    expected = evaluate_imputation(
        truth=heldout_truth, imputed=imputed, mask=mask,
        numeric_columns=NUMERIC, categorical_columns=CATEGORICAL,
    )
    scored = metrics.loc[metrics.method == "predictive_missforest"]
    for _, row in expected.iterrows():
        actual = scored.loc[scored.variable == row.variable].iloc[0]
        assert actual.n_masked == row.n_masked
        for key in ["mae", "rmse", "accuracy", "macro_f1"]:
            if pd.notna(row[key]):
                assert actual[key] == pytest.approx(row[key], rel=1e-9, abs=1e-9)

    # Scores must be computed ONLY on hidden cells. Changing other truth cells
    # must not change the held-out metrics for this particular model output.
    other_truth = heldout_truth.copy()
    for col in NUMERIC:
        other_truth.loc[~mask[col].astype(bool), col] += 100
    revised = evaluate_imputation(other_truth, imputed, mask, NUMERIC, CATEGORICAL)
    pd.testing.assert_frame_equal(expected, revised)


def test_runner_rejects_incomplete_reference_before_training(tmp_path):
    bad = complete_fixture(100)
    bad.loc[0, "material"] = np.nan
    csv = tmp_path / "not_complete.csv"
    bad.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="COMPLETE truth"):
        runner.main([
            "--input-csv", str(csv), "--config", str(config_yaml(tmp_path / "config.yaml")),
            "--output-dir", str(tmp_path / "output"), "--n-estimators", "5",
        ])
