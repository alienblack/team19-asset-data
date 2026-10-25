import json

import numpy as np
import pandas as pd
import pytest

from team19_asset_data.imputation.predictive_missforest import PredictiveMissForest
from team19_asset_data.overhead_conductors.v3_age_distributions import (
    YEARS,
    calibration_rows,
    empirical_year_pmf,
    leaf_year_pmfs,
    population_age_metrics,
    run_age_distributions,
    validate_year_support,
)
from team19_asset_data.overhead_conductors.v3_noise_comparison import ARMS, run_comparison
from team19_asset_data.overhead_conductors.v3_pilot import PilotSettings, run_pilot


def test_empirical_year_pmf_has_explicit_domain():
    pmf = empirical_year_pmf(np.array([1964, 1964, 1966]))
    assert pmf[1964 - YEARS[0]] == pytest.approx(2 / 3)
    assert pmf[1966 - YEARS[0]] == pytest.approx(1 / 3)
    assert pmf.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="outside"):
        empirical_year_pmf(np.array([2015]))
    with pytest.raises(ValueError, match="integer"):
        empirical_year_pmf(np.array([1964.5]))


def test_clean_truth_support_is_checked_before_evaluation():
    truth = pd.DataFrame({"installation_year": [1930, 1966, 2014]})
    assert validate_year_support(truth) == {
        "truth_min_year": 1930, "truth_max_year": 2014,
        "truth_outside_pmf_support": 0,
    }
    with pytest.raises(ValueError, match="outside PMF support"):
        validate_year_support(pd.DataFrame({"installation_year": [1929, 1966]}))


def test_expected_population_share_preserves_observed_years():
    truth = pd.DataFrame({"segment_id": ["a", "b", "c", "d"],
                          "installation_year": [1966, 1964, 1966, 1964]})
    observed = truth.copy()
    observed.loc[observed.segment_id.eq("c"), "installation_year"] = np.nan
    observed.loc[observed.segment_id.eq("d"), "installation_year"] = np.nan
    pmfs = np.stack([empirical_year_pmf(np.array([1964, 1966]))] * 2)
    metrics = population_age_metrics(observed, pd.Series(["c", "d"]), pmfs, truth)
    values = {row["measure"]: row["value"] for row in metrics}
    assert values["aged_50_or_older_share_error"] == pytest.approx(0)
    assert values["age_band_50-64_share_error"] == pytest.approx(0)
    with pytest.raises(ValueError, match="observed year"):
        population_age_metrics(observed, pd.Series(["b", "d"]), pmfs, truth)


def test_calibration_bins_keep_denominators():
    truth = pd.DataFrame({"segment_id": ["a", "b"], "installation_year": [1964, 1966]})
    pmfs = np.stack([
        empirical_year_pmf(np.array([1964] * 7 + [1966] * 3)),
        empirical_year_pmf(np.array([1964] * 2 + [1966] * 8)),
    ])
    details, bins = calibration_rows(pd.Series(["a", "b"]), pmfs, truth, seed=101, method="test")
    assert [row["p_aged_50_or_older"] for row in details] == pytest.approx([.7, .2])
    assert sum(row["n"] for row in bins) == 2


def test_frozen_forest_leaf_distribution_and_replay(tmp_path):
    pilot = run_pilot(tmp_path / "pilot", PilotSettings(seeds=(101,), n_estimators=6, n_passes=1), n_rows=1000)
    run = pilot / "seed_101"
    fitting = pd.read_csv(run / "observed_fitting.csv")
    test = pd.read_csv(run / "observed_test.csv")
    masks = pd.read_csv(run / "mask_manifest.csv")
    ids = masks.loc[masks.partition.eq("test") & masks.target.eq("installation_year"), "segment_id"]
    model = PredictiveMissForest.load(run / "forest.joblib")
    before = model.transform(test)
    pd.testing.assert_frame_equal(model.transform(fitting), model.fit_imputed_, check_dtype=False)
    pmfs = leaf_year_pmfs(model, fitting, test, ids)
    assert pmfs.shape == (60, len(YEARS))
    assert np.allclose(pmfs.sum(axis=1), 1)
    pd.testing.assert_frame_equal(before, model.transform(test))
    with pytest.raises(ValueError, match="Fitting-year mask"):
        leaf_year_pmfs(model, fitting.assign(installation_year=1950), test, ids)
    altered = fitting.copy()
    altered.loc[altered.index[0], "voltage_category"] = "changed"
    with pytest.raises(ValueError, match="Fitting voltage_category"):
        leaf_year_pmfs(model, altered, test, ids)
    with pytest.raises(ValueError, match="not hidden"):
        observed_id = test.loc[test.installation_year.notna(), "segment_id"].iloc[0]
        leaf_year_pmfs(model, fitting, test, pd.Series([observed_id]))

    age_output = run_age_distributions(pilot, tmp_path / "age")
    population = pd.read_csv(age_output / "population_age_metrics.csv")
    assert set(population.method) == {
        "global_floor", "conditional_baseline", "predictive_missforest", "generator_reference"
    }
    assert population.loc[population.measure.eq("aged_50_or_older_share_error"), "value"].abs().lt(.2).all()
    saved_manifest = json.loads((age_output / "manifest.json").read_text())
    assert saved_manifest["point_predictions_changed"] is False
    assert saved_manifest["year_support_check"]["truth_outside_pmf_support"] == 0
    assert (age_output / "evaluator_only/calibration_cells.csv").exists()
    saved_pmfs = pd.read_csv(age_output / "clean/seed_101/predicted_year_pmfs_test.csv.gz")
    assert len(saved_pmfs) == 4 * len(ids)
    assert np.allclose(saved_pmfs.filter(regex="^year_").sum(axis=1), 1)

    noise = run_comparison(pilot, tmp_path / "noise", (next(a for a in ARMS if a.arm_id == "year20_step2"),))
    paired = run_age_distributions(pilot, tmp_path / "paired", noise, ("year20_step2",))
    paired_population = pd.read_csv(paired / "population_age_metrics.csv")
    assert set(paired_population.condition) == {"clean", "year20_step2"}
    assert not paired_population.loc[paired_population.condition.eq("year20_step2"), "method"].eq("generator_reference").any()
