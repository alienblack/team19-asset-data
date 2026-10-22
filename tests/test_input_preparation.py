"""Contract tests for the pilot's model-facing and evaluator-only data boundary."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from team19_asset_data.overhead_conductors.input_preparation import (
    MODEL_FIELDS,
    FrozenPreprocessor,
    apply_pilot_noise,
    audit_reference,
    freeze_partitions,
    load_pilot_config,
    prepare_pilot,
)


PROTOCOL = Path(__file__).resolve().parents[1] / "docs/methodology/overhead_conductor_pilot_protocol_v1.yaml"


def reference_rows(count: int = 40) -> pd.DataFrame:
    materials = ("Steel", "Copper", "Aluminium")
    types = {"Steel": "SC/AC", "Copper": "Copper", "Aluminium": "AAC"}
    voltages = ("LV", "11/7.6 kV", "19 kV SWER", "33 kV", "66 kV")
    locations = ("Rural", "Regional", "Urban")
    corrosion = ("Low", "Severe", "Very Severe")
    rows = []
    for index in range(count):
        year = 1930 if index == 0 else 1970 + index % 40
        material = materials[index % 3]
        rows.append({
            "segment_id": f"SEG_{index + 1:06d}",
            "feeder_id": f"FDR_{index % 4}",
            "installation_year": year,
            "asset_age": 2014 - year,
            "voltage_category": voltages[index % 5],
            "location_group": locations[index % 3],
            "corrosion_zone": corrosion[index % 3],
            "distance_to_sea_km": float(1 + index),
            "material": material,
            "conductor_type": types[material],
            "covering": "Bare" if index % 2 else "Covered",
            "length_km": 0.5 + index / 100,
            "data_status": "Synthetic",
            "record_version": "Complete reference",
        })
    return pd.DataFrame(rows)


def small_config() -> dict:
    config = deepcopy(load_pilot_config(PROTOCOL))
    config["pilot_population"]["expected_rows"] = 40
    return config


def test_reference_audit_preserves_rare_valid_rows_and_rejects_bad_schema() -> None:
    reference = reference_rows()
    report = audit_reference(reference)
    assert report["passed"] and report["row_count"] == 40
    assert reference.iloc[0]["installation_year"] == 1930
    duplicate = reference.copy()
    duplicate.loc[1, "segment_id"] = duplicate.loc[0, "segment_id"]
    with pytest.raises(ValueError, match="not unique"):
        audit_reference(duplicate)
    invalid = reference.copy()
    invalid.loc[0, "distance_to_sea_km"] = 351
    with pytest.raises(ValueError, match="distance_to_sea_km"):
        audit_reference(invalid)
    wrong_age = reference.copy()
    wrong_age.loc[0, "asset_age"] = 5
    with pytest.raises(ValueError, match="asset_age"):
        audit_reference(wrong_age)
    unknown = reference.copy()
    unknown.loc[0, "material"] = "Silver"
    with pytest.raises(ValueError, match="material"):
        audit_reference(unknown)
    non_synthetic = reference.copy()
    non_synthetic.loc[0, "data_status"] = "Confidential"
    with pytest.raises(ValueError, match="only synthetic"):
        audit_reference(non_synthetic)


def test_partition_and_noise_are_reproducible_without_changing_truth() -> None:
    reference = reference_rows()
    config = small_config()
    first = freeze_partitions(reference, config)
    pd.testing.assert_frame_equal(first, freeze_partitions(reference, config))
    assert first["partition"].value_counts().to_dict() == {
        "fitting": 24, "tuning": 4, "calibration": 4, "test": 8
    }
    test_rows = reference.loc[reference["segment_id"].isin(first.loc[first["partition"] == "test", "segment_id"])]
    noisy, selected = apply_pilot_noise(test_rows, config)
    noisy_again, selected_again = apply_pilot_noise(test_rows, config)
    pd.testing.assert_frame_equal(noisy, noisy_again)
    assert selected == selected_again
    assert len(selected) == 8  # 20% of eight rows, independently in four fields
    pd.testing.assert_frame_equal(reference, reference_rows())
    assert noisy["material"].equals(test_rows["material"])
    assert noisy["conductor_type"].equals(test_rows["conductor_type"])
    numeric_delta = (noisy["distance_to_sea_km"] / test_rows["distance_to_sea_km"] - 1).abs()
    assert numeric_delta.max() <= 0.30 + 1e-12
    assert noisy["distance_to_sea_km"].between(0, 350).all()


def test_preparation_writes_separate_truth_views_and_common_masks(tmp_path: Path) -> None:
    reference = reference_rows()
    output = prepare_pilot(reference, small_config(), tmp_path / "pilot", "unit-test synthetic data")
    truth = pd.read_csv(output / "evaluator_only/clean_truth.csv")
    pd.testing.assert_frame_equal(truth, reference)
    view_names = ("fitting_N00", "tuning_N00", "calibration_N00", "test_N00", "test_N11")
    views = {name: pd.read_csv(output / f"observed_inputs/{name}.csv") for name in view_names}
    for view in views.values():
        assert set(view.columns) == {"segment_id", *MODEL_FIELDS}
        assert not any("true" in column or "noise" in column for column in view)
        assert view["segment_id"].is_unique
    for field in MODEL_FIELDS:
        assert views["test_N00"][field].isna().equals(views["test_N11"][field].isna())
    assert int(views["fitting_N00"]["material"].isna().sum()) == 5
    assert int(views["test_N00"]["material"].isna().sum()) == 2
    partition = pd.read_csv(output / "audit/partitions.csv")
    assert partition["segment_id"].is_unique
    assert set(partition["segment_id"]) == set(reference["segment_id"])
    ledger = pd.read_csv(output / "evaluator_only/cell_change_ledger.csv")
    assert (ledger["masked"] | ledger["noise_selected"]).all()
    assert ledger.loc[ledger["condition"] == "N00", "corruption_actual"].eq(False).all()
    assert ledger.loc[ledger["condition"] == "N11", "corruption_actual"].any()
    fitting = views["fitting_N00"]
    expected_mean = fitting["distance_to_sea_km"].mean()
    import json
    saved = json.loads((output / "observed_inputs/frozen_preprocessor.json").read_text())
    assert saved["numeric_means"]["distance_to_sea_km"] == pytest.approx(expected_mean)
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_pilot(reference, small_config(), output, "unit-test synthetic data")


def test_preprocessor_fits_only_observed_training_cells_and_reloads() -> None:
    fitting = reference_rows(12)[["segment_id", *MODEL_FIELDS]].copy()
    fitting.loc[0, "distance_to_sea_km"] = np.nan
    fitting.loc[1, "material"] = np.nan
    fitted = FrozenPreprocessor.fit(fitting)
    assert fitted.numeric_means["distance_to_sea_km"] == pytest.approx(fitting["distance_to_sea_km"].mean())
    assert fitted.categorical_modes["material"] == "Aluminium"
    reloaded = FrozenPreprocessor.from_dict(fitted.to_dict())
    application = fitting.copy()
    application.loc[0, "distance_to_sea_km"] = 300
    application.loc[2, "voltage_category"] = "Valid but unseen label"
    forest = reloaded.forest_predictors(application, target="material")
    assert "material" not in forest.columns
    assert not any(column.startswith("material=") for column in forest)
    assert forest.loc[0, "distance_to_sea_km"] == 300
    assert reloaded.numeric_means == fitted.numeric_means
    assert reloaded.unknown_categories(application).to_dict("records") == [{
        "segment_id": application.loc[2, "segment_id"],
        "field": "voltage_category",
        "observed_value": "Valid but unseen label",
    }]
    neural = reloaded.neural_inputs(fitting)
    held_out_neural = reloaded.neural_inputs(application)
    assert held_out_neural.loc[0, "distance_to_sea_km"] == pytest.approx(
        (300 - fitted.numeric_means["distance_to_sea_km"])
        / fitted.numeric_stds["distance_to_sea_km"]
    )
    assert neural.loc[0, "distance_to_sea_km"] == 0
    assert neural.loc[0, "distance_to_sea_km__available"] == 0
    assert neural.loc[1, "material__available"] == 0
    assert neural.loc[2, "material__available"] == 1
    baseline = reloaded.baseline_fill(fitting)
    assert baseline.loc[0, "distance_to_sea_km"] == pytest.approx(fitted.numeric_medians["distance_to_sea_km"])
    assert baseline.loc[1, "material"] == "Aluminium"


def test_model_view_rejects_shortcut_or_truth_columns() -> None:
    view = reference_rows(12)[["segment_id", *MODEL_FIELDS]].copy()
    for forbidden in ("conductor_type", "asset_age", "length_km", "material_true", "clean_value"):
        with pytest.raises(ValueError, match="seven model fields"):
            FrozenPreprocessor.fit(view.assign(**{forbidden: 1}))
