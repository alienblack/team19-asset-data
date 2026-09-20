"""Focused tests for the controlled relationship and leakage audit."""

import pandas as pd
import pytest

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
