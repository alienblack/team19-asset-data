"""Paired-workflow tests independent of the open predictive-forest PR."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from team19_asset_data.overhead_conductors.common_imputation import (
    load_prepared_inputs,
    run_common_imputation,
)
from team19_asset_data.overhead_conductors.input_preparation import (
    MODEL_FIELDS,
    FrozenPreprocessor,
    load_pilot_config,
    prepare_pilot,
)


ROOT = Path(__file__).resolve().parents[1]


class FakePredictiveForest:
    """Protocol double: tests orchestration, not forest-model correctness."""

    def fit(self, fitting: pd.DataFrame) -> "FakePredictiveForest":
        self.fitting_ids = set(fitting["segment_id"])
        self.statistics = FrozenPreprocessor.fit(fitting)
        return self

    def transform_detailed(self, observed: pd.DataFrame, with_scores: bool = False) -> SimpleNamespace:
        assert not self.fitting_ids.intersection(observed["segment_id"])
        completed = self.statistics.baseline_fill(observed)
        rows = []
        for field in MODEL_FIELDS:
            for position in np.flatnonzero(observed[field].isna().to_numpy()):
                rows.append({
                    "position": int(position), "column": field,
                    "prediction_source": "random_forest",
                    "max_proba": 0.8 if field not in {"installation_year", "distance_to_sea_km"} else np.nan,
                    "tree_std": 1.0 if field in {"installation_year", "distance_to_sea_km"} else np.nan,
                })
        return SimpleNamespace(imputed=completed, scores=pd.DataFrame(rows) if with_scores else None)

    def transform(self, observed: pd.DataFrame) -> pd.DataFrame:
        return self.transform_detailed(observed).imputed

    def save(self, path: Path) -> Path:
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "FakePredictiveForest":
        return joblib.load(path)


class MutatingForest(FakePredictiveForest):
    def transform_detailed(self, observed: pd.DataFrame, with_scores: bool = False) -> SimpleNamespace:
        result = super().transform_detailed(observed, with_scores)
        position = observed.index[observed["material"].notna()][0]
        result.imputed.loc[position, "material"] = "not the observed label"
        return result


class IncompleteDiagnostics(FakePredictiveForest):
    def transform_detailed(self, observed: pd.DataFrame, with_scores: bool = False) -> SimpleNamespace:
        result = super().transform_detailed(observed, with_scores)
        if with_scores:
            result.scores = result.scores.iloc[1:].copy()
        return result


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    reference = pd.read_csv(ROOT / "data/synthetic/overhead_conductors/strength_50/overhead_conductors_complete_reference.csv").head(40)
    config = deepcopy(load_pilot_config(ROOT / "docs/methodology/overhead_conductor_pilot_protocol_v1.yaml"))
    config["pilot_population"]["expected_rows"] = 40
    return prepare_pilot(reference, config, tmp_path / "inputs", "checked-in synthetic sample")


def test_common_workflow_uses_same_test_cells_and_never_reads_truth(prepared: Path, tmp_path: Path, monkeypatch) -> None:
    read_csv = pd.read_csv

    def observed_only(path, *args, **kwargs):
        assert "evaluator_only" not in str(path)
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", observed_only)
    output = run_common_imputation(prepared, tmp_path / "results", forest_builder=FakePredictiveForest)
    candidate = read_csv(output / "candidate_cells.csv")
    assert set(candidate["method"]) == {"median_mode", "predictive_missforest"}
    assert set(candidate["condition"]) == {"N00", "N11"}
    for condition in ("N00", "N11"):
        baseline = candidate.loc[candidate["method"].eq("median_mode") & candidate["condition"].eq(condition)]
        forest = candidate.loc[candidate["method"].eq("predictive_missforest") & candidate["condition"].eq(condition)]
        assert set(zip(baseline.segment_id, baseline.field)) == set(zip(forest.segment_id, forest.field))
        assert len(baseline) == 14  # two masked cells for each of seven fields
        observed = read_csv(prepared / f"observed_inputs/test_{condition}.csv")
        completed = read_csv(output / f"baseline_test_{condition}.csv")
        for field in MODEL_FIELDS:
            available = observed[field].notna()
            pd.testing.assert_series_equal(
                completed.loc[available, field].reset_index(drop=True),
                observed.loc[available, field].reset_index(drop=True),
                check_dtype=False,
            )
        assert (completed["asset_age"] == 2014 - completed["installation_year"]).all()
    for method in ("median_mode", "predictive_missforest"):
        clean = candidate.loc[candidate["method"].eq(method) & candidate["condition"].eq("N00")]
        noisy = candidate.loc[candidate["method"].eq(method) & candidate["condition"].eq("N11")]
        assert set(zip(clean.segment_id, clean.field)) == set(zip(noisy.segment_id, noisy.field))
    assert not candidate["candidate_value"].isna().any()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["evaluation_performed"] is False
    assert manifest["fitting_rows"] == 24 and manifest["test_rows"] == 8
    assert set(manifest["observed_input_sha256"]) == {"fitting_N00", "test_N00", "test_N11"}
    assert set(manifest["source_sha256"]) == {
        "common_imputation", "input_preparation", "forest_implementation"
    }
    assert (output / "predictive_forest.joblib").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        run_common_imputation(prepared, output, forest_builder=FakePredictiveForest)


def test_prepared_input_loader_detects_tampered_mask(prepared: Path) -> None:
    fitting, tests, _, _ = load_prepared_inputs(prepared)
    assert len(fitting) == 24 and len(tests["test_N00"]) == 8
    path = prepared / "observed_inputs/test_N11.csv"
    view = pd.read_csv(path)
    missing_position = view.index[view["material"].isna()][0]
    view.loc[missing_position, "material"] = "Steel"
    view.to_csv(path, index=False)
    with pytest.raises(ValueError, match="mask manifest disagrees"):
        load_prepared_inputs(prepared)


def test_no_shortcut_columns_can_be_loaded(prepared: Path) -> None:
    path = prepared / "observed_inputs/test_N00.csv"
    view = pd.read_csv(path)
    view["conductor_type"] = "SC/AC"
    view.to_csv(path, index=False)
    with pytest.raises(ValueError, match="seven model fields"):
        load_prepared_inputs(prepared)


def test_rejects_forest_that_changes_an_observed_value(prepared: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="modified originally observed values"):
        run_common_imputation(prepared, tmp_path / "bad_result", forest_builder=MutatingForest)


def test_rejects_incomplete_forest_diagnostics(prepared: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match the frozen hidden cells"):
        run_common_imputation(prepared, tmp_path / "incomplete", forest_builder=IncompleteDiagnostics)
