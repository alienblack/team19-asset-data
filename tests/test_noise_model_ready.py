import pandas as pd
import pytest

from team19_asset_data.overhead_conductors.noise import (
    NoiseScenario,
    apply_noise,
    validate_input,
)


@pytest.fixture
def model_ready_frame():
    return pd.DataFrame(
        {
            "segment_id": ["S1", "S2", "S3", "S4"],
            "feeder_id": ["F1", "F1", "F2", "F2"],
            "installation_year": [1980, 1990, 2000, 2010],
            "asset_age": [34, 24, 14, 4],
            "voltage_category": ["LV", "33 kV", "LV", "66 kV"],
            "location_group": ["Rural", "Urban", "Regional", "Rural"],
            "corrosion_zone": ["Low", "Severe", "Very Severe", "Low"],
            "distance_to_sea_km": [50.0, 20.0, 5.0, 100.0],
            "material": ["Steel", "Copper", "Aluminium", "Steel"],
            "covering": ["Bare", "Covered", "Bare", "Bare"],
            "length_km": [1.0, 2.0, 0.5, 3.0],
            "data_status": ["synthetic"] * 4,
            "record_version": ["v1"] * 4,
        }
    )


def test_noise_accepts_model_ready_without_conductor_type(
    model_ready_frame,
):
    validate_input(model_ready_frame)


def test_noise_preserves_model_ready_schema(
    model_ready_frame,
):
    scenario = NoiseScenario(
        scenario_id="TEST20",
        name="Random numeric noise",
        strategy="random_numeric",
        columns=("length_km", "distance_to_sea_km"),
        noise_level=0.20,
        seed=323,
    )

    observed, metadata = apply_noise(
        model_ready_frame,
        scenario,
    )

    assert list(observed.columns) == list(model_ready_frame.columns)
    assert "conductor_type" not in observed.columns
    assert "conductor_type" not in metadata["protected_columns"]

    pd.testing.assert_series_equal(
        observed["material"],
        model_ready_frame["material"],
    )


def test_noise_preserves_conductor_type_when_present(
    model_ready_frame,
):
    frame = model_ready_frame.copy()
    frame["conductor_type"] = [
        "Type-A",
        "Type-B",
        "Type-C",
        "Type-A",
    ]

    scenario = NoiseScenario(
        scenario_id="TEST20",
        name="Random numeric noise",
        strategy="random_numeric",
        columns=("length_km",),
        noise_level=0.20,
        seed=323,
    )

    observed, metadata = apply_noise(frame, scenario)

    pd.testing.assert_series_equal(
        observed["conductor_type"],
        frame["conductor_type"],
    )

    assert "conductor_type" in metadata["protected_columns"]