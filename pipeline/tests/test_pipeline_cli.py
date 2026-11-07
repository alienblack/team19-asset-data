"""End-to-end pipeline and command-line interface on a small synthetic file."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.cli import main
from src.settings import PipelineSettings, SettingsError, load_settings, parse_settings
from sample_frames import make_masked_frame, write_csv

ROOT = Path(__file__).resolve().parents[1]
FAST = ["--set", "n_estimators=6", "--set", "n_passes=2"]


def _dirty_input(tmp_path: Path, n: int = 160) -> Path:
    frame = make_masked_frame(n, seed=31).astype(object)
    frame.loc[0, "material"] = "steel"
    frame.loc[1, "installation_year"] = 9999
    frame.loc[2, "voltage_category"] = "11kV"
    frame = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
    return write_csv(frame, tmp_path / "records.csv")


def test_settings_defaults_and_validation():
    settings = load_settings(ROOT / "config" / "overhead_conductors" / "pipeline.yaml")
    assert settings.method == "predictive_missforest" and settings.reference_method == "median_mode"
    assert settings.data_schema.is_file() and settings.imputation.is_file()
    assert parse_settings({}).stages["reporting"] is True
    with pytest.raises(SettingsError):
        parse_settings({"stages": {"unknown": True}})
    with pytest.raises(SettingsError):
        parse_settings({"validation": {"reconstruction": {"mask_rate": 1.5}}})
    assert isinstance(PipelineSettings().with_updates(seed=None, method="missforest"), PipelineSettings)


def test_all_stages_run_and_produce_the_contracted_files(tmp_path):
    source = _dirty_input(tmp_path)
    out = tmp_path / "run"
    assert main(["all", "--input", str(source), "--out", str(out), *FAST, "--repeats", "1"]) == 0
    expected = [
        "run.log", "run_metadata.json", "source_records.csv", "quality_report.csv", "cleaned.csv", "cleaning_log.csv",
        "dropped_rows.csv", "profile/missingness.csv", "profile/missingness.png", "imputation_input.csv", "final_imputed.csv",
        "diagnostics.csv", "reference_imputed.csv", "convergence.csv", "derived_fields_log.csv", "model.joblib",
        "model.joblib.provenance.json", "validation/integrity_checks.csv", "validation/distribution_checks.csv",
        "validation/reconstruction_metrics.csv", "validation/reconstruction_summary.csv", "validation/low_confidence_cells.csv",
        "validation/figures/reconstruction_metrics.png", "report.html",
    ]
    missing = [name for name in expected if not (out / name).is_file()]
    assert not missing, missing

    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["imputation"]["method"] == "predictive_missforest" and meta["imputation"]["hyperparameters"]["n_estimators"] == 6
    assert meta["validation"]["integrity_passed"] is True
    assert set(meta["validation"]["reconstruction"]["headline"]) == {"predictive_missforest", "median_mode"}
    assert meta["cleaning"]["rows_dropped"] == 1 and meta["cleaning"]["cells_changed"] >= 3

    imputed = pd.read_csv(out / "final_imputed.csv")
    cleaned = pd.read_csv(out / "cleaned.csv")
    assert len(imputed) == len(cleaned) == 160
    assert imputed[["installation_year", "material", "asset_age"]].isna().sum().sum() == 0
    assert imputed["installation_year"].dtype.kind == "i"
    assert (imputed["asset_age"] == 2014 - imputed["installation_year"]).all()
    assert list(imputed.columns) == list(cleaned.columns)
    diagnostics = pd.read_csv(out / "diagnostics.csv")
    assert {"segment_id", "column", "imputed", "max_proba", "tree_std"} <= set(diagnostics.columns)
    assert len(diagnostics) == meta["imputation"]["cells_imputed"]
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Same-table reconstruction of hidden known values" in html and "data:image/png;base64" in html

    # a run with the legacy method through the same contract
    out2 = tmp_path / "run_legacy"
    assert main(["all", "--input", str(source), "--out", str(out2), "--method", "missforest", "--set", "n_estimators=6", "--set", "max_iter=2", "--no-model", "--repeats", "1"]) == 0
    meta2 = json.loads((out2 / "run_metadata.json").read_text())
    assert meta2["imputation"]["method"] == "missforest" and meta2["imputation"]["model_path"] is None
    assert not (out2 / "model.joblib").exists() and (out2 / "report.html").is_file()


def test_fit_then_apply_with_provenance_enforcement(tmp_path):
    frame = make_masked_frame(200, seed=41)
    train = write_csv(frame.iloc[:150], tmp_path / "train.csv")
    new = frame.iloc[150:].copy()
    new["segment_id"] = "NEW_" + new["segment_id"]
    new_path = write_csv(new, tmp_path / "new.csv")
    model = tmp_path / "model.joblib"
    assert main(["fit", "--input", str(train), "--model", str(model), *FAST]) == 0
    assert model.is_file() and model.with_name("model.joblib.provenance.json").is_file()

    out = tmp_path / "applied"
    assert main(["apply", "--input", str(new_path), "--model", str(model), "--out", str(out), "--require-independent"]) == 0
    imputed = pd.read_csv(out / "final_imputed.csv")
    assert len(imputed) == 50 and imputed["material"].isna().sum() == 0
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["imputation"]["mode"] == "apply" and meta["imputation"]["fitting_overlap"] == "none"

    # records that were used for fitting are refused when independence is required ...
    assert main(["apply", "--input", str(train), "--model", str(model), "--out", str(tmp_path / "overlap"), "--require-independent"]) == 2
    # ... and accepted (with the overlap recorded) when it is not
    assert main(["apply", "--input", str(train), "--model", str(model), "--out", str(tmp_path / "overlap_ok")]) == 0
    meta_overlap = json.loads((tmp_path / "overlap_ok" / "run_metadata.json").read_text())
    assert "used to fit" in meta_overlap["imputation"]["fitting_overlap"]

    # validate and report work on an applied run (reconstruction is skipped there)
    assert main(["validate", "--run", str(out)]) == 0
    applied_meta = json.loads((out / "run_metadata.json").read_text())
    assert applied_meta["validation"]["integrity_passed"] is True and applied_meta["validation"]["reconstruction"] == {"skipped": "apply run"}
    assert main(["report", "--run", str(out)]) == 0 and (out / "report.html").is_file()

    # transductive methods cannot be fitted for later use
    assert main(["fit", "--input", str(train), "--model", str(tmp_path / "legacy.joblib"), "--method", "missforest"]) == 2


def test_quality_errors_stop_the_pipeline(tmp_path):
    frame = make_masked_frame(30, seed=2)
    frame.loc[0, "segment_id"] = frame.loc[1, "segment_id"]  # conflicting duplicate identifier
    source = write_csv(frame, tmp_path / "bad.csv")
    out = tmp_path / "run"
    assert main(["all", "--input", str(source), "--out", str(out), *FAST]) == 2
    report = pd.read_csv(out / "quality_report.csv")
    assert (report["check"] == "identifier_duplicate").any() and not (out / "final_imputed.csv").exists()


def test_profile_and_clean_commands(tmp_path):
    source = _dirty_input(tmp_path, n=80)
    out = tmp_path / "stage"
    assert main(["clean", "--input", str(source), "--out", str(out)]) == 0
    assert (out / "cleaned.csv").is_file() and (out / "cleaning_log.csv").is_file() and (out / "quality_report.csv").is_file()
    assert main(["profile", "--input", str(out / "cleaned.csv"), "--out", str(out)]) == 0
    assert (out / "profile" / "relationships.csv").is_file()
    assert main(["impute", "--input", str(out / "cleaned.csv"), "--out", str(out), "--method", "median_mode"]) == 0
    assert (out / "final_imputed.csv").is_file() and (out / "model.joblib").is_file()  # median/mode is persistable
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["imputation"]["method"] == "median_mode" and meta["imputation"]["reference_method"] is None
