"""Profiling tables/figures, the leakage guard and derived fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.imputation.config import load_imputation_config
from src.ingestion import load_schema
from src.preprocessing import FeatureSelectionError, LeakageError, assert_no_leakage, check_config_against_schema, derive_fields, select_model_frame
from src.profiling import cramers_v, eta_squared, profile_records, render_imputation_figures, render_profile_figures
from sample_frames import make_conductor_frame, make_masked_frame

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = load_schema(ROOT / "config" / "overhead_conductors" / "data_schema.yaml")
CONFIG = load_imputation_config(ROOT / "config" / "overhead_conductors" / "imputation.yaml", SCHEMA)


def test_profile_tables_have_expected_content():
    frame = make_masked_frame(250, seed=7)
    result = profile_records(frame, SCHEMA)
    tables = result.tables
    assert set(tables) == {"missingness", "co_missingness", "missing_patterns", "numeric_summary", "categorical_summary", "missingness_by_attribute", "relationships", "numeric_by_category"}
    miss = tables["missingness"].set_index("column")
    assert miss.loc["segment_id", "n_missing"] == 0
    assert abs(miss.loc["material", "missing_rate"] - frame["material"].isna().mean()) < 1e-12
    co = tables["co_missingness"].set_index("column")
    assert abs(co.loc["material", "material"] - frame["material"].isna().mean()) < 1e-12
    assert tables["missing_patterns"]["share"].sum() <= 1.0 + 1e-12
    assert set(tables["numeric_summary"]["column"]) == {"installation_year", "distance_to_sea_km"}
    rel = tables["relationships"]
    assert len(rel) == 21 and rel["value"].between(-1, 1).all()
    mar = tables["missingness_by_attribute"]
    assert set(mar["target"]) == {"installation_year", "material"}
    assert result.summary["targets"] == ["installation_year", "material"]


def test_association_statistics():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.choice(["x", "y", "z"], 500))
    assert cramers_v(a, a) > 0.95  # identical categoricals
    assert cramers_v(a, pd.Series(rng.choice(["p", "q"], 500))) < 0.15  # independent
    groups = pd.Series(rng.choice(["g1", "g2"], 500))
    values = pd.Series(np.where(groups == "g1", 10.0, 20.0) + rng.normal(0, 1, 500))
    assert eta_squared(values, groups) > 0.9
    assert eta_squared(pd.Series(rng.normal(size=500)), groups) < 0.05


def test_profile_and_imputation_figures_are_written(tmp_path):
    frame = make_masked_frame(150, seed=2)
    figures = render_profile_figures(frame, SCHEMA, tmp_path / "profile")
    assert {"missingness", "co_missingness", "categorical_frequencies", "distribution_installation_year", "missingness_by_attribute_material"} <= set(figures)
    assert all(path.is_file() and path.stat().st_size > 1000 for path in figures.values())
    filled = frame.copy()
    for column in ("installation_year", "distance_to_sea_km"):
        filled[column] = filled[column].fillna(filled[column].median())
    for column in ("voltage_category", "location_group", "corrosion_zone", "material", "covering"):
        filled[column] = filled[column].fillna(filled[column].mode().iloc[0])
    out = render_imputation_figures(frame, filled, SCHEMA, tmp_path / "val", ("installation_year", "distance_to_sea_km"), ("material", "covering"))
    assert {"imputed_vs_observed_installation_year", "imputed_vs_observed_material"} <= set(out)


def test_leakage_guard_rejects_forbidden_and_unapproved_columns():
    with pytest.raises(LeakageError, match="asset_age"):
        assert_no_leakage(["installation_year", "asset_age"], SCHEMA)
    with pytest.raises(LeakageError, match="outside the approved"):
        assert_no_leakage(["installation_year", "mystery_feature"], SCHEMA)
    assert check_config_against_schema(CONFIG, SCHEMA) == []


def test_select_model_frame_orders_and_types_columns():
    frame = make_masked_frame(60, seed=4)
    frame["installation_year"] = frame["installation_year"].astype(object)
    model = select_model_frame(frame, SCHEMA, CONFIG)
    assert list(model.frame.columns[:8]) == ["segment_id", "installation_year", "distance_to_sea_km", "voltage_category", "location_group", "corrosion_zone", "material", "covering"]
    assert model.frame["installation_year"].dtype == float
    assert "asset_age" in model.passthrough_columns and "feeder_id" in model.passthrough_columns

    class BadConfig:
        numeric_columns = ("installation_year", "distance_to_sea_km", "asset_age")
        categorical_columns = CONFIG.categorical_columns
        excluded_columns = tuple(c for c in CONFIG.excluded_columns if c != "asset_age")
        constraints = CONFIG.constraints

    with pytest.raises(FeatureSelectionError, match="forbidden predictor 'asset_age'"):
        select_model_frame(frame, SCHEMA, BadConfig())


def test_derive_fields_fills_and_recomputes_only_where_source_was_imputed():
    complete = make_conductor_frame(50, seed=9)
    original = complete.copy()
    original.loc[:9, "installation_year"] = np.nan  # year imputed later; asset_age still present (stale)
    original.loc[10:19, "asset_age"] = np.nan  # age missing, year present
    original.loc[20, "asset_age"] = 999.0  # inconsistent but both observed -> kept and counted
    imputed = original.copy()
    imputed.loc[:9, "installation_year"] = 1999.0
    result = derive_fields(imputed, SCHEMA, original=original)
    out = result.frame
    assert (out.loc[:9, "asset_age"] == 2014 - 1999).all()  # recomputed from imputed source
    assert (out.loc[10:19, "asset_age"] == 2014 - out.loc[10:19, "installation_year"]).all()  # filled
    assert out.loc[20, "asset_age"] == 999.0
    summary = result.summary["asset_age"]
    assert summary["n_filled"] == 20 and summary["n_recomputed_after_source_imputed"] == 10 and summary["n_inconsistent_existing"] == 1 and summary["n_still_missing"] == 0
    everything = derive_fields(imputed, SCHEMA, overwrite_existing=True).frame
    assert everything.loc[20, "asset_age"] == 2014 - imputed.loc[20, "installation_year"]


def test_recover_sources_restores_year_from_observed_age_within_range():
    from src.preprocessing import prepare_for_imputation, recover_sources

    frame = make_conductor_frame(40, seed=21)
    frame.loc[:4, "installation_year"] = np.nan          # recoverable: asset_age observed
    frame.loc[5:7, ["installation_year", "asset_age"]] = np.nan  # not recoverable
    frame.loc[8, "installation_year"] = np.nan
    frame.loc[8, "asset_age"] = 500.0                      # out of range -> rejected
    result = recover_sources(frame, SCHEMA)
    assert (result.frame.loc[:4, "installation_year"] == 2014 - frame.loc[:4, "asset_age"]).all()
    assert result.frame.loc[5:8, "installation_year"].isna().all()
    assert result.summary["installation_year"] == {"n_recovered": 5, "n_rejected_out_of_range": 1, "n_still_missing": 4}
    assert result.cells["installation_year"].sum() == 5 and not result.cells["asset_age"].any()

    prepared = prepare_for_imputation(frame, SCHEMA, CONFIG)
    assert list(prepared.frame.columns) == list(frame.columns)
    assert prepared.recovered_cells["installation_year"].sum() == 5
    assert prepared.model_frame.model_columns == CONFIG.numeric_columns + CONFIG.categorical_columns


def test_derive_fields_reports_written_cells():
    original = make_conductor_frame(30, seed=22)
    original.loc[:9, "asset_age"] = np.nan
    result = derive_fields(original, SCHEMA)
    assert result.cells["asset_age"].sum() == 10 and result.frame["asset_age"].notna().all()
