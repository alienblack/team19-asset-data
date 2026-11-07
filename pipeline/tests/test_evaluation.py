"""Holdouts, metrics, integrity, distribution and confidence diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    check_integrity,
    compare_distributions,
    evaluate_imputation,
    hide_cells,
    joint_distribution_shift,
    load_or_create_manifest,
    low_confidence_cells,
    make_partitions,
    per_class_report,
    prepare_holdout,
    rank_confidence,
    repeated_holdouts,
    split_by_manifest,
    validate_alignment,
)
from src.imputation import build_imputer, load_imputation_config
from src.ingestion import load_schema
from src.preprocessing import select_model_frame
from sample_frames import make_conductor_frame, make_masked_frame

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = load_schema(ROOT / "config" / "overhead_conductors" / "data_schema.yaml")
CONFIG = load_imputation_config(ROOT / "config" / "overhead_conductors" / "imputation.yaml", SCHEMA)
NUMERIC, CATEGORICAL = CONFIG.numeric_columns, CONFIG.categorical_columns


# --- partitions ---------------------------------------------------------------------


def test_row_partitions_match_fractions_and_are_reproducible():
    frame = make_conductor_frame(203, seed=1)
    fractions = {"train": 0.6, "validation": 0.1, "calibration": 0.1, "test": 0.2}
    a = make_partitions(frame, fractions, 7, "segment_id")
    b = make_partitions(frame, fractions, 7, "segment_id")
    assert a.equals(b)
    counts = a["partition"].value_counts().to_dict()
    assert sum(counts.values()) == 203 and counts["train"] in (121, 122) and counts["test"] in (40, 41)
    parts = split_by_manifest(frame, a, "segment_id")
    assert sum(len(p) for p in parts.values()) == 203 and set(parts) == set(fractions)


def test_group_partitions_keep_groups_together():
    frame = make_conductor_frame(300, seed=2, n_feeders=15)
    manifest = make_partitions(frame, {"train": 0.7, "test": 0.3}, 3, "segment_id", strategy="group", group_column="feeder_id")
    joined = frame[["segment_id", "feeder_id"]].merge(manifest, on="segment_id")
    assert joined.groupby("feeder_id")["partition"].nunique().eq(1).all()
    assert 0.5 < (joined["partition"] == "train").mean() < 0.9


def test_manifest_is_reused_and_checked(tmp_path):
    frame = make_conductor_frame(50, seed=4)
    path = tmp_path / "manifest.csv"
    first = load_or_create_manifest(frame, path, {"train": 0.8, "test": 0.2}, 1, "segment_id")
    again = load_or_create_manifest(frame, path, {"train": 0.5, "test": 0.5}, 99, "segment_id")
    assert first.equals(again)
    with pytest.raises(ValueError, match="does not cover"):
        load_or_create_manifest(frame.iloc[:40], path, {"train": 0.8, "test": 0.2}, 1, "segment_id")
    with pytest.raises(ValueError, match="sum to one"):
        make_partitions(frame, {"train": 0.5, "test": 0.4}, 1, "segment_id")


# --- holdouts -----------------------------------------------------------------------------


def test_holdout_hides_only_observed_cells_at_the_requested_rate():
    frame = make_masked_frame(400, seed=3)
    masked, truth, mask = prepare_holdout(frame, ["installation_year", "material"], rate=0.2, seed=5)
    assert truth.equals(frame)
    for column in ("installation_year", "material"):
        n_obs = frame[column].notna().sum()
        assert abs(mask[column].sum() - round(0.2 * n_obs)) <= 1
        assert frame.loc[mask[column], column].notna().all()
        assert masked.loc[mask[column], column].isna().all()
        assert masked.loc[~mask[column], column].astype(str).equals(frame.loc[~mask[column], column].astype(str))
    assert not mask["distance_to_sea_km"].any()
    validate_alignment(masked, truth, mask, ["installation_year", "material"])


def test_stratified_holdout_represents_every_class():
    frame = make_masked_frame(600, seed=8)
    _, _, mask = prepare_holdout(frame, ["material", "covering"], rate=0.1, seed=1, stratify_columns=["material", "covering"])
    hidden = frame.loc[mask["covering"], "covering"].value_counts()
    assert set(hidden.index) == set(frame["covering"].dropna().unique())  # minority 'Covered' present
    hidden_material = frame.loc[mask["material"], "material"].value_counts(normalize=True)
    observed_material = frame["material"].value_counts(normalize=True)
    assert (hidden_material.reindex(observed_material.index).fillna(0) - observed_material).abs().max() < 0.05
    seeds = [1, 2, 3]
    repeats = repeated_holdouts(frame, ["material"], 0.1, seeds)
    assert len(repeats) == 3 and not repeats[0][2]["material"].equals(repeats[1][2]["material"])


def test_alignment_validation_catches_leaks():
    frame = make_masked_frame(50, seed=1)
    masked, truth, mask = prepare_holdout(frame, ["material"], rate=0.2, seed=1)
    leaky = masked.copy()
    leaky.loc[mask["material"], "material"] = truth.loc[mask["material"], "material"]
    with pytest.raises(ValueError, match="still visible"):
        validate_alignment(leaky, truth, mask, ["material"])
    with pytest.raises(ValueError, match="rate"):
        prepare_holdout(frame, ["material"], rate=1.5, seed=1)


# --- metrics --------------------------------------------------------------------------------------


def test_metrics_signed_bias_and_per_class():
    truth = pd.DataFrame({"y": [10.0, 20.0, 30.0, 40.0], "c": ["a", "b", "a", "b"]})
    imputed = pd.DataFrame({"y": [12.0, 22.0, 30.0, 40.0], "c": ["a", "a", "a", "b"]})
    mask = pd.DataFrame({"y": [True, True, False, False], "c": [True, True, True, True]})
    table = evaluate_imputation(truth, imputed, mask, ["y"], ["c"]).set_index("variable")
    assert table.loc["y", "mae"] == 2.0 and table.loc["y", "signed_bias"] == 2.0 and table.loc["y", "n_masked"] == 2
    assert table.loc["c", "accuracy"] == 0.75 and table.loc["c", "n_classes_truth"] == 2
    report = per_class_report(truth, imputed, mask, "c").set_index("class")
    assert report.loc["a", "recall"] == 1.0 and report.loc["b", "recall"] == 0.5 and report.loc["a", "precision"] == 2 / 3
    with pytest.raises(ValueError, match="Unfilled"):
        evaluate_imputation(truth, imputed.assign(y=[np.nan, 22.0, 30.0, 40.0]), mask, ["y"], ["c"])


# --- integrity ---------------------------------------------------------------------------------------


def _impute(frame: pd.DataFrame) -> pd.DataFrame:
    return build_imputer("median_mode", CONFIG).fit_transform(frame).imputed


def test_integrity_passes_for_a_correct_output_and_fails_for_tampering():
    original = select_model_frame(make_masked_frame(120, seed=6), SCHEMA, CONFIG).frame
    imputed = _impute(original)
    result = check_integrity(original, imputed, SCHEMA, NUMERIC, CATEGORICAL, CONFIG.constraints)
    assert result.passed, result.checks[result.checks["status"] == "fail"]

    tampered = imputed.copy()
    observed_pos = int(np.flatnonzero(original["material"].notna().to_numpy())[0])
    tampered.iloc[observed_pos, tampered.columns.get_loc("material")] = "Copper" if tampered.iloc[observed_pos]["material"] != "Copper" else "Steel"
    failed = check_integrity(original, tampered, SCHEMA, NUMERIC, CATEGORICAL, CONFIG.constraints)
    assert not failed.passed and set(failed.checks.loc[failed.checks["status"] == "fail", "check"]) == {"observed_values_unchanged"}

    unfilled = imputed.copy()
    missing_pos = int(np.flatnonzero(original["installation_year"].isna().to_numpy())[0])
    unfilled.iloc[missing_pos, unfilled.columns.get_loc("installation_year")] = np.nan
    assert "no_missing_after_imputation" in set(check_integrity(original, unfilled, SCHEMA, NUMERIC, CATEGORICAL, CONFIG.constraints).checks.query("status == 'fail'")["check"])

    out_of_range = imputed.copy()
    out_of_range.iloc[missing_pos, out_of_range.columns.get_loc("installation_year")] = 1850.0
    assert "constraint_range" in set(check_integrity(original, out_of_range, SCHEMA, NUMERIC, CATEGORICAL, CONFIG.constraints).checks.query("status == 'fail'")["check"])


def test_integrity_allows_derived_cells_recomputed_from_imputed_source():
    original = select_model_frame(make_masked_frame(100, seed=12), SCHEMA, CONFIG).frame
    original.loc[original["installation_year"].isna(), "asset_age"] = 5.0  # stale age with unknown year
    imputed = _impute(original)
    imputed["asset_age"] = 2014 - imputed["installation_year"]
    result = check_integrity(original, imputed, SCHEMA, NUMERIC, CATEGORICAL, CONFIG.constraints)
    assert result.passed


# --- distributions ----------------------------------------------------------------------------------------


def test_distribution_diagnostics_expose_median_and_majority_fill():
    original = select_model_frame(make_masked_frame(300, seed=9), SCHEMA, CONFIG).frame
    imputed = _impute(original)
    table = compare_distributions(original, imputed, NUMERIC, CATEGORICAL).set_index("variable")
    assert table.loc["installation_year", "std_ratio"] == 0.0
    assert table.loc["installation_year", "n_imputed"] == original["installation_year"].isna().sum()
    assert table.loc["material", "majority_share_imputed"] == 1.0 and 0 < table.loc["material", "total_variation"] < 1
    joint = joint_distribution_shift(original, imputed, "installation_year", "material", bin_width=10)
    assert 0 <= joint.attrs["total_variation"] <= 1 and {"bin", "class", "difference"} <= set(joint.columns)


# --- confidence --------------------------------------------------------------------------------------------


def test_confidence_ranking_and_low_confidence_listing():
    diagnostics = pd.DataFrame(
        {
            "row": range(6),
            "column": ["material"] * 3 + ["installation_year"] * 3,
            "kind": ["categorical"] * 3 + ["numeric"] * 3,
            "imputed": ["Steel", "Copper", "Steel", 1960.0, 1970.0, 1980.0],
            "prediction_source": ["random_forest", "random_forest", "training_statistic_fallback", "random_forest", "random_forest", "random_forest"],
            "max_proba": [0.9, 0.4, np.nan, np.nan, np.nan, np.nan],
            "margin": [0.8, 0.05, np.nan, np.nan, np.nan, np.nan],
            "tree_std": [np.nan, np.nan, np.nan, 2.0, 15.0, 7.0],
        }
    )
    ranked = rank_confidence(diagnostics).set_index("row")
    assert ranked.loc[0, "confidence_rank"] > ranked.loc[1, "confidence_rank"]
    assert ranked.loc[3, "confidence_rank"] > ranked.loc[5, "confidence_rank"] > ranked.loc[4, "confidence_rank"]
    result = low_confidence_cells(diagnostics, fraction=0.34, min_rows=2)
    listed = set(result.low_confidence["row"])
    assert 1 in listed and 4 in listed and 2 in listed  # least confident of each kind plus the fallback cell
    assert 0 not in listed and result.summary["fallback_cells"] == 1

    plain = diagnostics.assign(max_proba=np.nan, margin=np.nan, tree_std=np.nan)
    empty = low_confidence_cells(plain)
    assert empty.low_confidence.empty and "does not report" in empty.summary["note"]


def test_hide_cells_handles_numeric_and_object_columns():
    frame = pd.DataFrame({"n": [1.0, 2.0], "c": ["a", "b"]})
    mask = pd.DataFrame({"n": [True, False], "c": [False, True]})
    hidden = hide_cells(frame, mask)
    assert np.isnan(hidden.loc[0, "n"]) and hidden.loc[1, "n"] == 2.0 and pd.isna(hidden.loc[1, "c"]) and hidden.loc[0, "c"] == "a"


def test_linked_columns_are_hidden_with_their_source():
    frame = make_masked_frame(200, seed=14)
    frame["asset_age"] = 2014 - frame["installation_year"]  # observed wherever the year is
    masked, truth, mask = prepare_holdout(frame, ["installation_year"], rate=0.2, seed=2, linked_columns=SCHEMA.linked_columns())
    hidden_years = mask["installation_year"]
    assert hidden_years.sum() > 0
    assert (mask.loc[hidden_years, "asset_age"]).all()  # every hidden year hides its age
    assert masked.loc[hidden_years, "asset_age"].isna().all()
    assert not mask.loc[~hidden_years, "asset_age"].any()
    # without links the age would reveal the year
    _, _, plain = prepare_holdout(frame, ["installation_year"], rate=0.2, seed=2)
    assert not plain["asset_age"].any()


def test_cell_provenance_reconciles_with_the_output():
    from src.postprocessing import finalise_output
    from src.preprocessing import prepare_for_imputation

    before = make_masked_frame(150, seed=17)
    before.loc[:5, "installation_year"] = np.nan  # recoverable where asset_age is observed
    before.loc[:9, "length_km"] = np.nan  # passthrough column: stays unresolved
    prepared = prepare_for_imputation(before, SCHEMA, CONFIG)
    output = build_imputer("median_mode", CONFIG).fit_transform(prepared.frame)
    final = finalise_output(
        before, output.imputed, SCHEMA, model_columns=prepared.model_frame.model_columns,
        numeric_columns=NUMERIC, categorical_columns=CATEGORICAL, constraints=CONFIG.constraints,
        method="median_mode", recovered_cells=prepared.recovered_cells, diagnostics=output.diagnostics, prepared=prepared.frame,
    )
    assert final.integrity.passed
    prov = final.provenance
    totals = prov.totals
    assert totals["observed"] + totals["derived"] + totals["imputed"] + totals["unresolved"] == totals["n_cells"] == before.size
    missing_before = int(before.isna().sum().sum())
    assert totals["derived"] + totals["imputed"] + totals["unresolved"] >= missing_before
    assert len(prov.cells) == totals["derived"] + totals["imputed"] + totals["unresolved"]
    recovered = prov.cells[(prov.cells["column"] == "installation_year") & (prov.cells["method"] == "recover_source")]
    assert len(recovered) == int(prepared.recovered_cells["installation_year"].sum()) >= 4
    assert set(prov.unresolved["column"]) == {"length_km"} and prov.unresolved["reason"].iloc[0] == "not a model attribute"
    assert final.frame["length_km"].isna().sum() == int(prov.unresolved["n_cells"].iloc[0]) == before["length_km"].isna().sum()
    assert (prov.cells["status"].isin(["derived", "imputed", "unresolved"])).all()
    assert final.frame[list(NUMERIC) + list(CATEGORICAL)].isna().sum().sum() == 0
