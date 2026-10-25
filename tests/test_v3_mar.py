import json

import numpy as np
import pandas as pd
import pytest

from team19_asset_data.overhead_conductors.v3_mar import (
    ARMS, complete_case_age_metrics, make_mar_view, run_mar_evaluation,
)
from team19_asset_data.overhead_conductors.v3_pilot import PilotSettings, run_pilot


def test_mar_mask_uses_observed_driver_not_hidden_year():
    clean = pd.DataFrame({
        "segment_id": [f"s{i}" for i in range(100)],
        "installation_year": [1950] * 100,
        "voltage_category": ["19 kV SWER"] * 50 + ["LV"] * 50,
        "material": ["Steel"] * 100,
    })
    observed = clean.copy()
    observed.loc[0:9, "material"] = np.nan
    arm = ARMS[0]
    first, first_mask, first_counts = make_mar_view(
        clean, observed, arm, seed=101, arm_index=0, partition_index=3,
    )
    second, second_mask, _ = make_mar_view(
        clean.assign(installation_year=2000), observed, arm,
        seed=101, arm_index=0, partition_index=3,
    )
    assert first_mask.equals(second_mask)
    assert second.loc[second.installation_year.notna(), "installation_year"].eq(2000).all()
    assert first.loc[first.material.isna(), "segment_id"].tolist() == observed.loc[
        observed.material.isna(), "segment_id"
    ].tolist()
    assert first_counts.set_index("driver_value").at["19 kV SWER", "assigned_probability"] == .35
    assert first_counts.set_index("driver_value").at["LV", "assigned_probability"] == .12
    assert first_mask.segment_id.is_unique


def test_material_mar_uses_only_available_material():
    clean = pd.DataFrame({
        "segment_id": ["a", "b", "c"], "installation_year": [1930, 1966, 2000],
        "material": ["Steel", "Copper", "Aluminium"],
    })
    observed = clean.copy()
    observed.loc[1, "material"] = np.nan
    _, _, counts = make_mar_view(clean, observed, ARMS[3], seed=101, arm_index=3, partition_index=3)
    rates = counts.set_index("driver_value")["assigned_probability"]
    assert rates["Steel"] == .19
    assert rates["<unavailable>"] == .30
    assert rates["Aluminium"] == .75
    with pytest.raises(ValueError, match="Unexpected"):
        make_mar_view(clean, observed.assign(material="unknown"), ARMS[3],
                      seed=101, arm_index=3, partition_index=3)


def test_complete_case_error_is_against_full_test_truth():
    truth = pd.DataFrame({"segment_id": ["a", "b", "c"],
                          "installation_year": [1950, 1966, 1966]})
    observed = truth.copy()
    observed.loc[0, "installation_year"] = np.nan
    metrics = {row["measure"]: row["value"] for row in complete_case_age_metrics(observed, truth)}
    assert metrics["aged_50_or_older_share_error"] == pytest.approx(-1 / 3)


def test_mar_end_to_end_preserves_frozen_partitions_and_material_masks(tmp_path):
    pilot = run_pilot(tmp_path / "pilot", PilotSettings(seeds=(101,), n_estimators=6, n_passes=1), n_rows=1000)
    arms = (ARMS[0], ARMS[3])
    output = run_mar_evaluation(pilot, tmp_path / "mar", arms)
    assert json.loads((output / "manifest.json").read_text())["year_support_check"]["truth_outside_pmf_support"] == 0
    metrics = pd.read_csv(output / "population_age_metrics.csv")
    assert set(metrics.arm) == {"voltage_mar", "material_mar"}
    assert set(metrics.method) == {
        "complete_case", "global_floor", "conditional_baseline",
        "predictive_missforest", "generator_reference",
    }
    cell_metrics = pd.read_csv(output / "hidden_year_cell_metrics.csv")
    assert set(cell_metrics.method) == {
        "global_floor", "conditional_baseline", "predictive_missforest"
    }
    assert cell_metrics.year_mae.ge(0).all()
    counts = pd.read_csv(output / "mask_counts.csv")
    assert set(counts.partition) == {"fitting", "tuning", "calibration", "test"}
    assert counts.n_hidden.le(counts.n_rows).all()
    for arm in arms:
        test = pd.read_csv(output / arm.arm_id / "seed_101/observed_test.csv")
        frozen = pd.read_csv(pilot / "seed_101/observed_test.csv")
        assert test.segment_id.equals(frozen.segment_id)
        assert test.material.isna().equals(frozen.material.isna())
        hidden = pd.read_csv(output / arm.arm_id / "seed_101/year_mask_test.csv")
        assert set(hidden.segment_id) == set(test.loc[test.installation_year.isna(), "segment_id"])
        pmfs = pd.read_csv(output / arm.arm_id / "seed_101/predicted_year_pmfs_test.csv.gz")
        assert np.allclose(pmfs.filter(regex="^year_").sum(axis=1), 1)
        assert set(pmfs.method) == {"global_floor", "conditional_baseline", "predictive_missforest"}
        assert (output / arm.arm_id / "seed_101/forest.joblib").exists()
        assert (output / arm.arm_id / "seed_101/candidate_years_test.csv").exists()
        assert (output / arm.arm_id / "seed_101/year_mask_fitting.csv").exists()
    assert (output / "evaluator_only/calibration_cells.csv").exists()
    assert pd.read_csv(output / "test_group_age_metrics.csv").group_field.eq("location_group").any()


def test_arm_selection_keeps_its_fixed_random_stream(tmp_path):
    pilot = run_pilot(tmp_path / "pilot", PilotSettings(seeds=(101,), n_estimators=4, n_passes=1), n_rows=1000)
    alone = run_mar_evaluation(pilot, tmp_path / "alone", (ARMS[3],))
    with_other = run_mar_evaluation(pilot, tmp_path / "with_other", (ARMS[0], ARMS[3]))
    a = pd.read_csv(alone / "material_mar/seed_101/year_mask_test.csv")
    b = pd.read_csv(with_other / "material_mar/seed_101/year_mask_test.csv")
    pd.testing.assert_frame_equal(a, b)
