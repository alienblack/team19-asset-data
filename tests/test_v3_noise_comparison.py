import numpy as np
import pandas as pd
import pytest

from team19_asset_data.overhead_conductors.v3_noise_comparison import (
    ARMS,
    changed_input_counts,
    material_by_decade,
    noisy_reference,
    overlay_observed,
    run_comparison,
)
from team19_asset_data.overhead_conductors.v3_pilot import PilotSettings, make_reference, run_pilot


def test_noise_overlay_keeps_ids_and_exact_target_masks():
    clean = make_reference(300)
    arm = next(arm for arm in ARMS if arm.arm_id == "year10_step2")
    noisy, _ = noisy_reference(clean, arm, 101)
    view = clean.loc[:29, ["segment_id", "installation_year", "distance_to_sea_km",
                           "voltage_category", "location_group", "corrosion_zone", "material", "covering"]].copy()
    view.loc[view.index[:5], "material"] = np.nan
    view.loc[view.index[5:10], "installation_year"] = np.nan
    observed = overlay_observed(view, noisy)
    assert observed["segment_id"].tolist() == view["segment_id"].tolist()
    pd.testing.assert_series_equal(observed["material"].isna(), view["material"].isna())
    pd.testing.assert_series_equal(observed["installation_year"].isna(), view["installation_year"].isna())
    assert clean["material"].equals(noisy["material"])
    assert clean["conductor_type"].equals(noisy["conductor_type"])
    assert (noisy["asset_age"] == 2014 - noisy["installation_year"]).all()
    changes = changed_input_counts(view.reset_index(drop=True), observed, "test")
    assert changes.loc[changes["field"].eq("material"), "changed_observed_cells"].iloc[0] == 0


def test_length_negative_control_cannot_reach_model_fields():
    clean = make_reference(300)
    arm = next(arm for arm in ARMS if arm.arm_id == "length20_negative_control")
    noisy, _ = noisy_reference(clean, arm, 101)
    view = clean.loc[:29, ["segment_id", "installation_year", "distance_to_sea_km",
                           "voltage_category", "location_group", "corrosion_zone", "material", "covering"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(view, overlay_observed(view, noisy))
    assert not clean["length_km"].equals(noisy["length_km"])


def test_overlay_rejects_missing_ids():
    clean = make_reference(100)
    view = clean.loc[:3, ["segment_id", "installation_year", "distance_to_sea_km",
                           "voltage_category", "location_group", "corrosion_zone", "material", "covering"]].copy()
    with pytest.raises(ValueError, match="every observed ID"):
        overlay_observed(view, clean.iloc[4:])


def test_small_paired_run_and_reproducibility(tmp_path):
    pilot = run_pilot(tmp_path / "pilot", PilotSettings(seeds=(101,), n_estimators=6, n_passes=1), n_rows=1000)
    arms = tuple(arm for arm in ARMS if arm.arm_id in {"length20_negative_control", "voltage10"})
    first = run_comparison(pilot, tmp_path / "first", arms)
    second = run_comparison(pilot, tmp_path / "second", arms)
    for name in ("cell_metrics.csv", "population_metrics.csv", "paired_cell_metrics.csv",
                 "paired_population_metrics.csv", "observed_changes.csv", "material_by_decade.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    paired = pd.read_csv(first / "paired_cell_metrics.csv")
    assert set(paired["arm"]) == {"length20_negative_control", "voltage10"}
    negative = paired.loc[paired["arm"].eq("length20_negative_control")]
    assert np.allclose(negative["delta_noisy_minus_clean"], 0)
    assert (first / "evaluator_only/review_cases.csv").is_file()
    assert (first / "material_confusion.csv").is_file()
    assert (first / "primary_paired_summary.csv").is_file()
    assert (first / "observed_change_summary.csv").is_file()
    assert (first / "plots/paired_log_loss.png").is_file()
    assert (first / "plots/paired_aged50_population.png").is_file()
    decades = pd.read_csv(first / "material_by_decade.csv")
    assert set(decades["material"]) == {"Steel", "Copper", "Aluminium"}
    with pytest.raises(FileExistsError):
        run_comparison(pilot, first, arms)
