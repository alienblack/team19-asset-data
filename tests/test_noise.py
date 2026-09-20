"""Tests for protected, reproducible predictor-noise scenarios."""

import pandas as pd
import pytest

from team19_asset_data.overhead_conductors.noise import (
    NoiseScenario,
    apply_noise,
    validate_output,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    voltages = ["LV", "33 kV", "19 kV SWER", "11/7.6 kV"]
    locations = ["Urban", "Regional", "Rural", "Rural"]
    corrosion = ["Low", "Severe", "Very Severe", "Low"]
    materials = ["Aluminium", "Copper", "Steel", "Steel"]
    conductors = ["AAC", "Copper", "SC/AC", "SC/GZ"]
    for index in range(20):
        year = 1990 + index
        rows.append(
            {
                "segment_id": f"SEG_{index:06d}",
                "feeder_id": f"F_{index % 4}",
                "installation_year": year,
                "asset_age": 2014 - year,
                "voltage_category": voltages[index % len(voltages)],
                "location_group": locations[index % len(locations)],
                "corrosion_zone": corrosion[index % len(corrosion)],
                "distance_to_sea_km": float(index + 1),
                "material": materials[index % len(materials)],
                "conductor_type": conductors[index % len(conductors)],
                "covering": "Bare" if index % 3 else "Covered",
                "length_km": float(index + 1) / 10,
                "data_status": "Synthetic",
                "record_version": "Complete reference",
            }
        )
    return pd.DataFrame(rows)


def test_random_numeric_noise_is_reproducible_and_protected() -> None:
    frame = sample_frame()
    scenario = NoiseScenario(
        "N01", "numeric", "random_numeric", ("length_km",), 0.20, 101
    )
    observed_a, metadata_a = apply_noise(frame, scenario)
    observed_b, metadata_b = apply_noise(frame, scenario)
    assert observed_a.equals(observed_b)
    assert metadata_a["observed_sha256"] == metadata_b["observed_sha256"]
    assert observed_a["material"].equals(frame["material"])
    assert observed_a["conductor_type"].equals(frame["conductor_type"])
    assert metadata_a["changed_counts"]["length_km"] == 4


def test_year_noise_keeps_linked_age_consistent() -> None:
    frame = sample_frame()
    scenario = NoiseScenario(
        "N02", "year", "installation_year_error", ("installation_year", "asset_age"), 0.20, 102, 2
    )
    observed, metadata = apply_noise(frame, scenario)
    assert (observed["asset_age"] == 2014 - observed["installation_year"]).all()
    assert metadata["changed_counts"] == {"installation_year": 4, "asset_age": 4}
    validate_output(frame, observed)


def test_categorical_noise_stays_inside_allowed_domain() -> None:
    frame = sample_frame()
    scenario = NoiseScenario(
        "N03", "categorical", "categorical_replacement", ("location_group",), 0.10, 103
    )
    observed, _ = apply_noise(frame, scenario)
    assert set(observed["location_group"]) <= {"Urban", "Regional", "Rural"}
    assert validate_output(frame, observed)["passed"]


def test_systematic_noise_requires_a_valid_group() -> None:
    frame = sample_frame()
    scenario = NoiseScenario(
        "N04", "bad", "systematic_numeric", ("length_km",), 0.10, 104,
        group_column="location_group", group_value="Missing",
    )
    observed, _ = apply_noise(frame, scenario)
    assert observed.equals(frame)


def test_protected_target_cannot_be_used_as_noise_column() -> None:
    frame = sample_frame()
    scenario = NoiseScenario(
        "bad", "bad", "categorical_replacement", ("material",), 0.10, 105
    )
    with pytest.raises(ValueError, match="protected"):
        apply_noise(frame, scenario)
