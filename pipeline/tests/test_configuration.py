"""Configuration manager: validation, cross-file checks, approvals and the release gate."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from src.cli import main
from src.configuration import (
    APPROVED,
    CHANGED,
    UNAPPROVED,
    ApprovalError,
    ApprovalStore,
    ConfigurationError,
    ConfigurationManager,
    load_settings,
)
from sample_frames import make_masked_frame, write_csv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = ("data_schema.yaml", "cleaning_rules.yaml", "imputation.yaml", "pipeline.yaml")


def _config_copy(tmp_path: Path, *, with_approvals: bool = False) -> Path:
    """A private copy of config/ so tests can edit and approve without touching the repository."""
    target = tmp_path / "config"
    target.mkdir()
    for name in CONFIG_FILES:
        shutil.copy(ROOT / "config" / "overhead_conductors" / name, target / name)
    if with_approvals:
        shutil.copy(ROOT / "config" / "overhead_conductors" / "approvals.json", target / "approvals.json")
    # pipeline.yaml paths are repository-relative; point them at the copy
    raw = yaml.safe_load((target / "pipeline.yaml").read_text())
    raw["paths"] = {"data_schema": str(target / "data_schema.yaml"), "cleaning_rules": str(target / "cleaning_rules.yaml"), "imputation": str(target / "imputation.yaml")}
    raw["execution"] = {"mode": "release"}
    (target / "pipeline.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return target / "pipeline.yaml"


def test_repository_configuration_is_valid_but_requires_reapproval_after_refactor():
    check = ConfigurationManager(load_settings(ROOT / "config" / "overhead_conductors" / "pipeline.yaml")).check()
    assert check.is_valid, check.findings_table()
    assert not check.all_approved, check.states_table()
    assert not check.ready
    assert set(check.states) == set(CONFIG_FILES)


def test_states_follow_the_file_hash(tmp_path):
    pipeline_yaml = _config_copy(tmp_path)
    manager = ConfigurationManager(load_settings(pipeline_yaml))
    assert all(s.status == UNAPPROVED for s in manager.states().values())
    with pytest.raises(ConfigurationError, match="unapproved"):
        manager.require("release")
    manager.require("development")  # records, does not refuse

    state = manager.approve("data_schema.yaml", by="tester", note="unit test")
    assert state.status == APPROVED and state.record.approved_by == "tester"
    assert manager.states()["data_schema.yaml"].status == APPROVED
    schema_path = manager.files["data_schema.yaml"]
    schema_path.write_text(schema_path.read_text() + "\n# edited\n")
    assert manager.states()["data_schema.yaml"].status == CHANGED
    assert manager.revoke("data_schema.yaml") and manager.states()["data_schema.yaml"].status == UNAPPROVED

    store = ApprovalStore(manager.store.path)
    with pytest.raises(ApprovalError, match="approved_by"):
        store.approve("x", schema_path, "")


def test_invalid_file_cannot_be_approved_and_blocks_release(tmp_path):
    pipeline_yaml = _config_copy(tmp_path)
    manager = ConfigurationManager(load_settings(pipeline_yaml))
    schema_path = manager.files["data_schema.yaml"]
    raw = yaml.safe_load(schema_path.read_text())
    raw["columns"]["installation_year"]["max"] = raw["snapshot_year"] + 5
    schema_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    check = manager.check()
    assert not check.is_valid and any("snapshot_year" in f.message for f in check.loaded.errors)
    with pytest.raises(ApprovalError, match="cannot be approved"):
        manager.approve("data_schema.yaml", by="tester")
    with pytest.raises(ConfigurationError):
        manager.require("release")


def test_cross_file_findings(tmp_path):
    pipeline_yaml = _config_copy(tmp_path)
    manager = ConfigurationManager(load_settings(pipeline_yaml))
    # constraints duplicated in imputation.yaml must equal the schema
    imputation_path = manager.files["imputation.yaml"]
    raw = yaml.safe_load(imputation_path.read_text())
    raw["constraints"] = {"installation_year": {"data_type": "integer", "min": 1900, "max": 2030}}
    imputation_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    findings = manager.check().findings_table()
    assert findings[(findings["level"] == "error") & (findings["file"] == "imputation.yaml")]["message"].str.contains("disagree").any()
    # a cleaning mapping onto a label outside the allowed list is an error
    rules_path = manager.files["cleaning_rules.yaml"]
    rules = yaml.safe_load(rules_path.read_text())
    rules["categorical"]["spelling_variants"]["material"]["ti"] = "Titanium"
    rules_path.write_text(yaml.safe_dump(rules, sort_keys=False))
    findings = manager.check().findings_table()
    assert findings["message"].str.contains("outside the allowed list").any()
    # unapproved code mappings are a warning, reference == method a warning
    assert findings["message"].str.contains("code_mappings").any()


def test_summaries_cover_every_file():
    manager = ConfigurationManager(load_settings(ROOT / "config" / "overhead_conductors" / "pipeline.yaml"))
    tables = manager.summaries()
    assert {"schema_columns", "schema_summary", "cleaning_families", "cleaning_policies", "imputation_methods", "imputation_columns", "pipeline_settings"} <= set(tables)
    assert (tables["imputation_methods"]["selected"]).sum() == 1
    families = tables["cleaning_families"].set_index("family")
    assert bool(families.loc["code_mappings", "active"]) is False


def test_release_gate_in_the_pipeline_and_the_development_override(tmp_path):
    pipeline_yaml = _config_copy(tmp_path)
    source = write_csv(make_masked_frame(60, seed=3), tmp_path / "records.csv")
    out = tmp_path / "run"
    assert main(["clean", "--input", str(source), "--out", str(out), "--pipeline-config", str(pipeline_yaml)]) == 2
    assert not (out / "cleaned.csv").exists()
    assert main(["clean", "--input", str(source), "--out", str(out), "--pipeline-config", str(pipeline_yaml), "--allow-unapproved"]) == 0
    assert (out / "cleaned.csv").is_file()
    assert main(["config", "check", "--pipeline-config", str(pipeline_yaml)]) == 1
    assert main(["config", "approve", "--pipeline-config", str(pipeline_yaml), "--file", "all", "--by", "tester"]) == 0
    assert main(["config", "check", "--pipeline-config", str(pipeline_yaml)]) == 0
    assert main(["clean", "--input", str(source), "--out", str(tmp_path / "run2"), "--pipeline-config", str(pipeline_yaml)]) == 0
    assert main(["config", "show", "--pipeline-config", str(pipeline_yaml)]) == 0
