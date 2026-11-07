"""Schema parsing and record loading."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.ingestion import IngestionError, SchemaError, apply_schema_types, finalise_output_types, load_records, load_schema, parse_schema
from sample_frames import make_masked_frame, write_csv

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "config" / "overhead_conductors" / "data_schema.yaml"


def _raw_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# --- schema ----------------------------------------------------------------------


def test_repository_schema_loads_with_expected_roles():
    schema = load_schema(SCHEMA_PATH)
    assert schema.id_column == "segment_id"
    assert schema.model_columns == ("installation_year", "distance_to_sea_km", "voltage_category", "location_group", "corrosion_zone", "material", "covering")
    assert schema.numeric_model_columns == ("installation_year", "distance_to_sea_km")
    assert schema.targets == ("installation_year", "material")
    assert schema.derived_columns == ("asset_age",)
    assert schema.group_column == "feeder_id"
    assert "asset_age" in schema.forbidden_predictors and "length_km" in schema.forbidden_predictors
    assert set(schema.allowed_predictors) == set(schema.model_columns)


def test_schema_rejects_forbidden_predictor_with_model_role():
    raw = copy.deepcopy(_raw_schema())
    raw["columns"]["asset_age"]["role"] = "predictor"
    raw["columns"]["asset_age"].pop("derive")
    with pytest.raises(SchemaError, match="model_attributes|Forbidden"):
        parse_schema(raw)


def test_schema_requires_every_non_model_column_to_be_forbidden():
    raw = copy.deepcopy(_raw_schema())
    raw["forbidden_predictors"].remove("length_km")
    with pytest.raises(SchemaError, match="length_km"):
        parse_schema(raw)


def test_schema_requires_allowed_values_for_category_model_columns():
    raw = copy.deepcopy(_raw_schema())
    raw["columns"]["material"].pop("allowed")
    with pytest.raises(SchemaError, match="allowed"):
        parse_schema(raw)


def test_schema_rejects_mapping_onto_existing_column():
    raw = copy.deepcopy(_raw_schema())
    raw["column_mapping"] = {"feeder_id": "material"}
    with pytest.raises(SchemaError, match="column_mapping"):
        parse_schema(raw)


# --- loader ----------------------------------------------------------------------


def test_load_records_reads_text_applies_mapping_and_orders_columns(tmp_path):
    schema = load_schema(SCHEMA_PATH)
    raw = copy.deepcopy(_raw_schema())
    raw["column_mapping"] = {"Install Year": "installation_year", "MATERIAL": "material"}
    mapped_schema = parse_schema(raw)
    frame = make_masked_frame(40, seed=3)
    frame = frame.rename(columns={"installation_year": "Install Year", "material": "MATERIAL"})
    frame["extra_note"] = "x"
    path = write_csv(frame, tmp_path / "in.csv")

    loaded = load_records(path, mapped_schema)
    assert loaded.renamed == {"Install Year": "installation_year", "MATERIAL": "material"}
    assert loaded.unknown_columns == ("extra_note",)
    assert loaded.missing_required == ()
    assert list(loaded.frame.columns)[:3] == ["segment_id", "feeder_id", "installation_year"]
    assert loaded.frame.columns[-1] == "extra_note"
    # text in, missing as NaN
    assert loaded.frame["installation_year"].dtype == object
    assert loaded.frame["installation_year"].isna().sum() == frame["Install Year"].isna().sum()
    assert loaded.frame["material"].dropna().map(type).eq(str).all()
    # plain schema reports the unmapped names as missing required columns rather than raising
    plain = load_records(path, schema)
    assert set(plain.missing_required) == {"installation_year", "material"}


def test_load_records_rejects_duplicate_columns(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text("segment_id,material,material\nA,Steel,Copper\n", encoding="utf-8")
    with pytest.raises(IngestionError, match="duplicate column"):
        load_records(path, load_schema(SCHEMA_PATH))


def test_apply_schema_types_coerces_leniently():
    schema = load_schema(SCHEMA_PATH)
    frame = pd.DataFrame(
        {
            "segment_id": ["A", "B", "C"],
            "installation_year": ["1973", "abc", "NA"],
            "distance_to_sea_km": ["1.5", None, "unknown"],
            "material": [" Steel", "N/A", None],
        },
        dtype=object,
    )
    typed = apply_schema_types(frame, schema)
    assert typed["installation_year"].tolist()[0] == 1973.0 and np.isnan(typed["installation_year"].iloc[1]) and np.isnan(typed["installation_year"].iloc[2])
    assert typed["distance_to_sea_km"].dtype == float and typed["distance_to_sea_km"].isna().sum() == 2
    assert typed["material"].tolist()[0] == "Steel" and typed["material"].isna().sum() == 2


def test_finalise_output_types_casts_complete_integer_columns():
    schema = load_schema(SCHEMA_PATH)
    frame = pd.DataFrame({"segment_id": ["A", "B"], "installation_year": [1973.0, 1980.0], "asset_age": [41.0, np.nan]})
    out = finalise_output_types(frame, schema)
    assert out["installation_year"].dtype == "int64"
    assert out["asset_age"].dtype == float  # still has a missing value


def test_schema_binds_installation_year_maximum_to_snapshot_year():
    schema = load_schema(SCHEMA_PATH)
    assert schema.column("installation_year").max == schema.snapshot_year
    assert schema.constraints()["installation_year"] == {"data_type": "integer", "min": 1900, "max": schema.snapshot_year}
    raw = copy.deepcopy(_raw_schema())
    raw["columns"]["installation_year"]["max"] = raw["snapshot_year"] + 12
    with pytest.raises(SchemaError, match="must not exceed snapshot_year"):
        parse_schema(raw)
    raw["columns"]["installation_year"]["max"] = "snapshot_year"
    assert parse_schema(raw).column("installation_year").max == raw["snapshot_year"]


def test_schema_accepts_new_and_old_spellings_but_not_both():
    raw = copy.deepcopy(_raw_schema())
    assert "primary_targets" in raw and "model_attributes" in raw
    schema = parse_schema(raw)
    assert schema.primary_targets == schema.targets == ("installation_year", "material")
    assert schema.model_attributes == schema.allowed_predictors
    old = copy.deepcopy(raw)
    old["targets"] = old.pop("primary_targets")
    old["allowed_predictors"] = old.pop("model_attributes")
    assert parse_schema(old).primary_targets == schema.primary_targets
    both = copy.deepcopy(raw)
    both["targets"] = both["primary_targets"]
    with pytest.raises(SchemaError, match="not both"):
        parse_schema(both)
