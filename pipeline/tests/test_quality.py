"""Data quality assessment and rule-driven cleaning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion import load_records, load_schema
from src.quality import CleaningError, CleaningRules, clean_records, load_cleaning_rules, parse_cleaning_rules, plan_changes, validate_records
from sample_frames import make_masked_frame, write_csv

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = load_schema(ROOT / "config" / "overhead_conductors" / "data_schema.yaml")
RULES = load_cleaning_rules(ROOT / "config" / "overhead_conductors" / "cleaning_rules.yaml")


def _dirty_frame(tmp_path: Path) -> pd.DataFrame:
    frame = make_masked_frame(120, seed=5)
    frame = frame.astype(object)
    frame.loc[0, "material"] = " steel "
    frame.loc[1, "material"] = "CU"
    frame.loc[2, "material"] = "Titanium"
    frame.loc[3, "material"] = "NA"
    frame.loc[4, "voltage_category"] = "11kV"
    frame.loc[5, "installation_year"] = 9999
    frame.loc[6, "installation_year"] = 1850
    frame.loc[7, "installation_year"] = 1973.4
    frame.loc[8, "distance_to_sea_km"] = "abc"
    frame.loc[9, "distance_to_sea_km"] = -5
    frame.loc[10, "asset_age"] = 3  # disagrees with installation_year
    frame = pd.concat([frame, frame.iloc[[11]]], ignore_index=True)  # exact duplicate row
    path = write_csv(frame, tmp_path / "dirty.csv")
    return load_records(path, SCHEMA).frame


def test_validator_reports_each_kind_of_issue(tmp_path):
    report = validate_records(_dirty_frame(tmp_path), SCHEMA)
    checks = report.issues.set_index(["check", "column"])["count"].to_dict()
    assert checks[("label_variant", "material")] == 1
    assert checks[("unknown_category", "material")] == 2  # CU (synonym, not allowed as-is) and Titanium
    assert checks[("missing_token", "material")] == 1
    assert checks[("sentinel_value", "installation_year")] == 1
    assert checks[("out_of_range", "installation_year")] == 1
    assert checks[("non_integer_value", "installation_year")] == 1
    assert checks[("non_numeric_value", "distance_to_sea_km")] == 1
    assert checks[("out_of_range", "distance_to_sea_km")] == 1
    assert checks[("exact_duplicate_row", "")] == 1
    assert checks[("derived_field_mismatch", "asset_age")] >= 1
    assert ("identifier_duplicate", "segment_id") not in checks  # exact duplicates are not identifier conflicts
    assert report.passed and report.n_errors == 0 and report.n_warnings >= 6


def test_validator_flags_blocking_errors():
    frame = pd.DataFrame({"segment_id": ["A", "A", None], "installation_year": ["1970", "1980", "1990"], "material": [None, None, None]}, dtype=object)
    report = validate_records(frame, SCHEMA)
    errors = set(report.issues.loc[report.issues["severity"] == "error", "check"])
    assert {"missing_required_column", "identifier_missing", "identifier_duplicate", "insufficient_observed_target"} <= errors
    assert not report.passed


def test_cleaner_applies_rules_and_logs_every_change(tmp_path):
    raw = _dirty_frame(tmp_path)
    result = clean_records(raw, SCHEMA, RULES)
    cleaned, log = result.cleaned, result.log
    assert len(cleaned) == len(raw) - 1  # exact duplicate dropped
    assert cleaned.loc[0, "material"] == "Steel" and cleaned.loc[1, "material"] == "Copper"
    assert pd.isna(cleaned.loc[2, "material"])  # unknown label removed (approved policy) and logged for mapping
    assert pd.isna(cleaned.loc[3, "material"])
    assert cleaned.loc[4, "voltage_category"] == "11/7.6 kV"
    assert np.isnan(cleaned.loc[5, "installation_year"]) and np.isnan(cleaned.loc[6, "installation_year"])
    assert cleaned.loc[7, "installation_year"] == 1973.0
    assert np.isnan(cleaned.loc[8, "distance_to_sea_km"]) and np.isnan(cleaned.loc[9, "distance_to_sea_km"])
    rules_fired = set(log["rule"])
    assert {"whitespace", "label_case", "spelling_variant", "unknown_category", "missing_token", "sentinel_value", "out_of_range", "non_integer_rounded", "non_numeric", "exact_duplicate_row"} <= rules_fired
    assert "code_mapping" not in rules_fired and result.summary["inactive_families"] == ["code_mappings"]
    # every logged change names the record and the column
    assert (log["record_id"].astype(str).str.len() > 0).all()
    assert cleaned["installation_year"].dtype == float and cleaned["material"].dtype == object
    assert result.summary["rows_dropped"] == 1 and result.summary["cells_changed"] == len(log[log["column"] != "*"])


def test_cleaner_never_fills_missing_and_never_touches_unruled_values(tmp_path):
    raw = _dirty_frame(tmp_path)
    result = clean_records(raw, SCHEMA, RULES)
    raw_no_dup = raw.iloc[: len(result.cleaned)]
    changed = set(zip(result.log["row"], result.log["column"]))
    for column in ["voltage_category", "location_group", "corrosion_zone", "covering"]:
        before = raw_no_dup[column]
        after = result.cleaned[column]
        for pos in range(len(before)):
            if (pos, column) in changed:
                continue
            b, a = before.iloc[pos], after.iloc[pos]
            assert (pd.isna(b) and pd.isna(a)) or b == a
    assert result.cleaned["material"].isna().sum() >= raw_no_dup["material"].isna().sum()


def test_code_mappings_apply_only_when_approved():
    frame = pd.DataFrame({"segment_id": ["A", "B", "C"], "material": ["ACSR", "aluminum", "HDC"]}, dtype=object)
    off = clean_records(frame, SCHEMA, RULES)
    assert off.cleaned["material"].tolist()[1] == "Aluminium"  # spelling variant always applies
    assert pd.isna(off.cleaned["material"].iloc[0]) and pd.isna(off.cleaned["material"].iloc[2])  # codes unapproved -> unknown -> missing
    assert off.summary["pending_code_mappings"] == {"material": 2}
    on = parse_cleaning_rules(
        {
            "categorical": {"code_mappings": {"material": {"acsr": "Aluminium", "hdc": "Copper"}}},
            "approval": {"code_mappings": {"requires_domain_approval": True, "approved": True, "approved_by": "data owner"}},
        }
    )
    applied = clean_records(frame, SCHEMA, on)
    assert applied.cleaned["material"].tolist() == ["Aluminium", "aluminum", "Copper"]
    assert (applied.log["rule"] == "code_mapping").sum() == 2
    with pytest.raises(CleaningError, match="approved_by"):
        parse_cleaning_rules({"approval": {"code_mappings": {"requires_domain_approval": True, "approved": True}}})


def test_plan_changes_matches_the_real_run_and_leaves_input_untouched(tmp_path):
    raw = _dirty_frame(tmp_path)
    before = raw.copy(deep=True)
    plan = plan_changes(raw, SCHEMA, RULES)
    applied = clean_records(raw, SCHEMA, RULES)
    assert plan.log.equals(applied.log) and plan.summary == applied.summary
    assert raw.equals(before)


def test_cleaner_duplicate_identifier_policy():
    frame = pd.DataFrame({"segment_id": ["A", "A"], "installation_year": ["1970", "1980"], "material": ["Steel", "Copper"]}, dtype=object)
    with pytest.raises(CleaningError, match="share an identifier"):
        clean_records(frame, SCHEMA, RULES)
    lenient = parse_cleaning_rules({"duplicates": {"duplicate_id_action": "keep_first"}})
    result = clean_records(frame, SCHEMA, lenient)
    assert len(result.cleaned) == 1 and result.cleaned.loc[0, "material"] == "Steel"
    assert (result.log["rule"] == "duplicate_identifier").sum() == 1


def test_cleaning_rules_validation():
    with pytest.raises(CleaningError):
        parse_cleaning_rules({"numeric": {"non_numeric_action": "ignore"}})
    rules = CleaningRules(unknown_category_action="missing")
    frame = pd.DataFrame({"segment_id": ["A"], "material": ["Titanium"]}, dtype=object)
    cleaned = clean_records(frame, SCHEMA, rules).cleaned
    assert cleaned["material"].isna().all()
