"""Focused tests for the controlled relationship and leakage audit."""

import pandas as pd
import pytest
import json
from pathlib import Path

from team19_asset_data.overhead_conductors import relationship_validation as audit

from team19_asset_data.overhead_conductors.relationship_validation import (
    categorical_association,
    maximum_category_purity,
    schema_and_invariants,
)


def complete_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = pd.DataFrame(
        {
            "segment_id": ["S1", "S2", "S3"],
            "feeder_id": ["F1", "F2", "F3"],
            "installation_year": [2000, 1990, 1980],
            "asset_age": [14, 24, 34],
            "voltage_category": ["LV", "33 kV", "19 kV SWER"],
            "location_group": ["Urban", "Regional", "Rural"],
            "corrosion_zone": ["Low", "Severe", "Very Severe"],
            "distance_to_sea_km": [100.0, 20.0, 2.0],
            "material": ["Aluminium", "Copper", "Steel"],
            "conductor_type": ["AAC", "Copper", "SC/AC"],
            "covering": ["Bare", "Bare", "Covered"],
            "length_km": [1.0, 2.0, 3.0],
            "data_status": ["Synthetic"] * 3,
            "record_version": ["Complete reference"] * 3,
        }
    )
    return complete, complete.drop(columns=["conductor_type"])


def test_multiclass_association_does_not_require_numeric_label_encoding() -> None:
    association = categorical_association(
        pd.Series(["AAC", "Copper", "SC/AC"] * 3),
        pd.Series(["Aluminium", "Copper", "Steel"] * 3),
    )
    assert association["cramers_v"] == pytest.approx(1.0)
    assert association["normalized_mutual_information"] == pytest.approx(1.0)


def test_category_purity_ignores_tiny_categories() -> None:
    result = maximum_category_purity(
        pd.Series(["rare", "common", "common", "common"]),
        pd.Series(["Steel", "Steel", "Copper", "Aluminium"]),
        min_support=2,
    )
    assert result["category"] == "common"
    assert result["support"] == 3


def test_schema_audit_catches_conductor_type_in_model_ready_data() -> None:
    complete, model_ready = complete_frame()
    checks = schema_and_invariants(complete, model_ready, snapshot_year=2014)
    assert checks["passed"]

    invalid_model_ready = model_ready.assign(conductor_type=complete["conductor_type"])
    checks = schema_and_invariants(complete, invalid_model_ready, snapshot_year=2014)
    assert not checks["passed"]
    assert not checks["no_conductor_type_in_model_ready"]


def test_generator_era_boundaries():
    years = pd.Series([1949, 1950, 1977, 1978, 1979, 1980, 1994, 1995, 2014])
    assert audit.voltage_by_era(years).tolist() == [
        "pre_1950", "1950_1977", "1950_1977", "1978_1994", "1978_1994",
        "1978_1994", "1978_1994", "from_1995", "from_1995",
    ]
    assert audit.material_by_era(years).tolist() == [
        "pre_1950", "1950_1979", "1950_1979", "1950_1979", "1950_1979",
        "from_1980", "from_1980", "from_1980", "from_1980",
    ]


def test_profile_metrics_use_distinct_eras():
    complete, _ = complete_frame()
    rows = pd.DataFrame(audit.profile_metrics(complete, "strength_0", 219))
    assert set(rows.loc[rows.relation == "configured_era_to_voltage", "left_feature"]) == {"voltage_era"}
    assert set(rows.loc[rows.relation == "configured_era_to_material", "left_feature"]) == {"material_era"}


def test_model_ready_values_and_row_order_must_match():
    complete, model = complete_frame()
    assert not audit.schema_and_invariants(complete, model.iloc[::-1], 2014)["passed"]
    model.loc[0, "material"] = "Steel"
    assert not audit.schema_and_invariants(complete, model, 2014)["passed"]


@pytest.mark.parametrize("column,value", [("length_km", float("inf")), ("distance_to_sea_km", -1), ("installation_year", 2026), ("covering", "Unknown")])
def test_invalid_reference_values_fail(column, value):
    complete, _ = complete_frame()
    complete.loc[0, column] = value
    assert not audit.schema_and_invariants(complete, complete.drop(columns="conductor_type"), 2014)["passed"]


def test_screens_preserve_profile_scope_and_evaluation_support(monkeypatch):
    complete, _ = complete_frame()
    monkeypatch.setattr(audit, "feature_screen", lambda *args, **kwargs: {"balanced_accuracy": .9, "macro_f1": .8, "evaluation_support": 2})
    rows = pd.DataFrame(audit.leakage_rows(complete, 219, 1, "strength_0"))
    assert set(rows.profile) == {"strength_0"}
    assert set(rows.loc[rows.audit_type == "single_feature_classifier", "support"]) == {2}
    assert set(rows.loc[rows.feature == "voltage_category", "scope"]) == {"profile_dependent_recoverability"}
    assert set(rows.loc[rows.feature == "conductor_type", "scope"]) == {"direct_answer_or_duplicate"}


@pytest.mark.parametrize("seeds,support", [((), 25), ((219, 219), 25), ((-1,), 25), ((219,), 0)])
def test_invalid_run_parameters_rejected(tmp_path, seeds, support):
    with pytest.raises(ValueError):
        audit.run_validation("not_needed.yaml", tmp_path, seeds, support)


def test_run_screens_every_profile_and_seed(tmp_path, monkeypatch):
    complete, model = complete_frame()
    calls = []
    monkeypatch.setattr(audit, "load_or_generate", lambda *args: (complete, model, "test_fixture"))
    original = audit.leakage_rows
    def record(frame, seed, min_support, profile):
        calls.append((profile, seed))
        return original(frame, seed, min_support, profile)
    monkeypatch.setattr(audit, "leakage_rows", record)
    monkeypatch.setattr(audit, "feature_screen", lambda *args, **kwargs: {"balanced_accuracy": .5, "macro_f1": .4, "evaluation_support": 2})
    config = Path(__file__).resolve().parents[1] / "config/overhead_conductors/generation.yaml"
    audit.run_validation(config, tmp_path, seeds=(219, 319), min_support=1)
    expected = {(f"strength_{strength}", seed) for strength in range(0, 101, 10) for seed in (219, 319)}
    assert set(calls) == expected
    rows = pd.read_csv(tmp_path / "leakage_audit.csv")
    assert set(zip(rows.profile, rows.seed)) == expected
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["recoverability_scope"] == "all_profiles_and_seeds"
    # The tiny fixture intentionally violates the configured row/route contract.
    assert metadata["all_invariants_passed"] is False
    assert metadata["source_sha256"]


def test_default_seed_does_not_load_stale_committed_csv(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "config/overhead_conductors/generation.yaml"
    config = audit.Config.from_yaml(config_path)
    strength = audit.StrengthConfig.from_yaml(config_path.parent / "relationship_strength/strength_0.yaml")
    complete, _ = complete_frame()
    calls = []
    def generate(run_config, strengths):
        calls.append(run_config.seed)
        return complete
    monkeypatch.setattr(audit, "generate_dataset", generate)
    monkeypatch.setattr(audit, "calibrate_route_length_marginals", lambda frame, cfg: frame)
    actual, model, source = audit.load_or_generate(config, strength, config.seed, tmp_path)
    assert calls == [config.seed]
    assert source == "generated_in_memory"
    assert actual.equals(complete)
    assert model.equals(complete.drop(columns="conductor_type"))
