import numpy as np
import pandas as pd
import pytest

from team19_asset_data.overhead_conductors.v3_pilot import (
    ConditionalBaseline,
    MATERIALS,
    PilotSettings,
    freeze_split,
    make_reference,
    mask_targets,
    material_generator_check,
    material_log_loss,
    exact_year_prior,
    generator_reference_predictions,
    evaluate,
    assert_observed_preserved,
    run_pilot,
    validate_predictions,
)


@pytest.fixture(scope="module")
def reference():
    return make_reference(1000)


def test_probability_order_is_explicit():
    labels = np.array(["Steel", "Copper", "Aluminium"])
    probabilities = np.eye(3) * .8 + .2 / 3
    assert material_log_loss(labels, probabilities) < .15
    assert material_log_loss(labels, probabilities[:, ::-1]) > 1


def test_split_mask_and_learned_era_have_no_answer_copy(reference):
    split = freeze_split(reference, 101)
    assert split["partition"].value_counts().to_dict() == {
        "fitting": 600, "tuning": 100, "calibration": 100, "test": 200
    }
    observed, manifest = mask_targets(reference, split, 101, .30)
    test = observed["test"]
    assert list(test.columns) == ["segment_id", "installation_year", "distance_to_sea_km",
                                  "voltage_category", "location_group", "corrosion_zone", "material", "covering"]
    assert test["material"].isna().sum() == 60
    assert test["installation_year"].isna().sum() == 60
    assert manifest.duplicated(["segment_id", "target"]).sum() == 0
    baseline = ConditionalBaseline().fit(observed["fitting"])
    assert len(baseline.era_edges) > 0
    assert all(1930 <= edge <= 2014 for edge in baseline.era_edges)


def test_material_check_does_not_require_every_group_to_have_steel_plurality(reference):
    check = material_generator_check(reference)
    assert check.groupby(["voltage_category", "material_era", "corrosion_zone"]).ngroups <= 45
    assert check["configured_probability"].between(0, 1).all()
    assert np.allclose(check.groupby(["voltage_category", "material_era", "corrosion_zone"])["configured_probability"].sum(), 1)
    configured = check.pivot(index=["voltage_category", "material_era", "corrosion_zone"],
                             columns="material", values="configured_probability")
    assert configured["Steel"].gt(configured[["Copper", "Aluminium"]].max(axis=1)).all()


def test_generator_reference_has_valid_probabilities_and_years(reference):
    years, prior = exact_year_prior()
    assert years[0] == 1930 and years[-1] == 2014
    assert np.isclose(prior.sum(), 1)
    assert np.isclose(prior[years <= 1963].sum(), .59)
    assert np.isclose(prior[(years >= 1964) & (years <= 1994)].sum(), .39)
    assert np.isclose(prior[years >= 1995].sum(), .02)
    split = freeze_split(reference, 101)
    observed, masks = mask_targets(reference, split, 101, .30)
    test_masks = masks.loc[masks["partition"].eq("test")]
    oracle = generator_reference_predictions(observed["test"], test_masks)
    validate_predictions(oracle, observed["test"], test_masks)
    assert oracle.loc[oracle["target"].eq("material"), "point_prediction"].eq("Steel").all()


def test_integrity_rejects_unmasked_prediction(reference):
    split = freeze_split(reference, 101)
    observed, masks = mask_targets(reference, split, 101, .30)
    test_masks = masks.loc[masks["partition"].eq("test")].copy()
    fitting = observed["fitting"]
    predictions = ConditionalBaseline().fit(fitting).predict(observed["test"], test_masks)
    validate_predictions(predictions, observed["test"], test_masks)
    changed = predictions.copy()
    changed.loc[changed["target"].eq("material"), "p_steel"] = 2.0
    with pytest.raises(ValueError, match="sum to one"):
        validate_predictions(changed, observed["test"], test_masks)


def test_baseline_backoff_ties_and_both_hidden():
    rows = pd.DataFrame({
        "segment_id": [f"S{i}" for i in range(12)],
        "installation_year": [1955] * 12,
        "distance_to_sea_km": [8.0] * 12,
        "voltage_category": ["LV"] * 12,
        "location_group": ["Urban"] * 12,
        "corrosion_zone": ["Moderate"] * 11 + ["Very Severe"],
        "material": ["Steel", "Copper"] * 6,
        "covering": ["Bare"] * 12,
    })
    model = ConditionalBaseline(alpha=5, min_group=3).fit(rows)
    assert model.global_material[MATERIALS.index("Steel")] == model.global_material[MATERIALS.index("Copper")]
    assert model.material(pd.Series({"voltage_category": np.nan}))["prediction"] == "Steel"
    era = model.frame["learned_era"].iloc[0]
    direct = model.material(pd.Series({"voltage_category": "LV", "learned_era": era, "corrosion_zone": "Moderate"}))
    assert direct["prediction"] == "Steel"
    tiny = model.material(pd.Series({"voltage_category": "LV", "learned_era": era, "corrosion_zone": "Very Severe"}))
    assert tiny["fallback_reason"] == "group_below_minimum_support"
    missing = model.material(pd.Series({"voltage_category": "LV", "learned_era": np.nan, "corrosion_zone": "Moderate"}))
    assert missing["fallback_reason"] == "grouping_field_missing"
    unknown = model.material(pd.Series({"voltage_category": "33/66 kV", "learned_era": era, "corrosion_zone": "Moderate"}))
    assert unknown["fallback_reason"] == "group_unavailable"
    masked = rows.iloc[[0]].copy()
    masked["material"] = np.nan
    masked["installation_year"] = np.nan
    masks = pd.DataFrame({"segment_id": ["S0", "S0"], "target": ["material", "installation_year"]})
    result = model.predict(masked, masks)
    assert len(result) == 2
    assert result.loc[result["target"].eq("installation_year"), "fallback_reason"].iloc[0] == "grouping_field_missing"
    assert model.material(model.frame.iloc[0])["prediction"] == "Steel"


def test_fit_only_era_and_length_weighted_scoring(reference):
    split = freeze_split(reference, 101)
    observed, manifest = mask_targets(reference, split, 101, .30)
    fitting = observed["fitting"]
    model = ConditionalBaseline().fit(fitting)
    original_edges = model.era_edges.copy()
    perturbed_tuning = observed["tuning"].copy()
    perturbed_tuning.loc[perturbed_tuning["installation_year"].notna(), "installation_year"] = 2014
    assert np.array_equal(model.era_edges, original_edges)
    assert not np.allclose(np.quantile(perturbed_tuning["installation_year"].dropna(), [1/3, 2/3]), original_edges)
    test_masks = manifest.loc[manifest["partition"].eq("test")]
    prediction = model.predict(observed["test"], test_masks)
    cell, population = evaluate(prediction, reference, observed["test"], test_masks)
    assert "raw_q10_q90_empirical_coverage" in set(cell["measure"])
    assert "macro_f1" in set(cell["measure"])
    assert "fill_coverage" in set(cell["measure"])
    assert "length_weighted_aged_50_or_older_share_error" in set(population["measure"])
    assert "length_weighted_year_wasserstein" in set(population["measure"])
    assert "year_std_error" in set(population["measure"])
    assert "length_weighted_material_expected_share_error_steel" in set(population["measure"])


def test_observed_value_change_is_rejected(reference):
    split = freeze_split(reference, 101)
    observed, _ = mask_targets(reference, split, 101, .30)
    completed = observed["test"].copy()
    known = completed["voltage_category"].first_valid_index()
    completed.at[known, "voltage_category"] = "altered"
    with pytest.raises(AssertionError):
        assert_observed_preserved(observed["test"], completed)


def test_small_end_to_end_smoke(tmp_path):
    settings = PilotSettings(seeds=(101,), n_estimators=6, n_passes=1)
    root = run_pilot(tmp_path / "pilot", settings, n_rows=1000)
    scores = pd.read_csv(root / "cell_metrics.csv")
    assert set(scores["method"]) == {"global_floor", "conditional_baseline", "predictive_missforest", "generator_reference"}
    candidates = pd.read_csv(root / "seed_101/candidate_cells.csv")
    assert set(candidates["condition_id"]) == {"strength100_MCAR30_clean"}
    assert set(candidates["seed"]) == {101}
    forest_year = candidates.loc[candidates["method"].eq("predictive_missforest") & candidates["target"].eq("installation_year")]
    assert forest_year["year_q50"].notna().all()
    assert forest_year["tree_std"].notna().all()
    forest_material = candidates.loc[candidates["method"].eq("predictive_missforest") & candidates["target"].eq("material")]
    assert forest_material["max_vote_share"].notna().all()
    assert (root / "evaluator_only/material_generator_check.csv").is_file()
    assert (root / "evaluator_only/confusion_seed_101.csv").is_file()
    assert (root / "gap_closed.csv").is_file()
    assert (root / "forest_validation_curve.csv").is_file()
    assert (root / "forest_learning_curve.csv").is_file()
    assert (root / "subgroup_diagnostics.csv").is_file()
    assert (root / "subgroup_confusion.csv").is_file()
    assert (root / "evidence_backoff_counts.csv").is_file()
    assert (root / "seed_101/missingness_flags_test.csv").is_file()
    assert (root / "integrity_checks.json").is_file()
    assert (root / "plots/learning_curve.png").is_file()
    assert (root / "seed_101/forest_tuning.json").is_file()
    duplicate = run_pilot(tmp_path / "second", settings, n_rows=1000)
    assert (root / "seed_101/candidate_cells.csv").read_bytes() == (duplicate / "seed_101/candidate_cells.csv").read_bytes()
    assert (root / "cell_metrics.csv").read_bytes() == (duplicate / "cell_metrics.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_pilot(root, settings, n_rows=1000)
