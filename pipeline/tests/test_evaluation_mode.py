"""Evaluation mode: record-disjoint partitions, masks within partitions, truth scoring, no leakage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cli import main
from src.evaluation.protocol import EvaluationSettings, build_evaluation_inputs, run_protocols
from src.imputation import load_imputation_config
from src.ingestion import apply_schema_types, load_schema
from sample_frames import inject_missing, make_conductor_frame, write_csv

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = load_schema(ROOT / "config" / "overhead_conductors" / "data_schema.yaml")
CONFIG = load_imputation_config(ROOT / "config" / "overhead_conductors" / "imputation.yaml", SCHEMA)
MODEL = list(CONFIG.numeric_columns) + list(CONFIG.categorical_columns)
FAST = EvaluationSettings(mask_rate=0.2, repeats=2, protocols=("unseen_records", "transductive"))


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    return apply_schema_types(frame.astype(object).where(frame.notna(), None), SCHEMA)


def test_partitions_are_record_and_group_disjoint_and_only_test_is_scored():
    truth = _typed(make_conductor_frame(400, seed=51, n_feeders=20))
    inputs = build_evaluation_inputs(SCHEMA, CONFIG, FAST, truth=truth)
    manifest = inputs.manifest
    assert set(manifest["partition"]) == {"fit", "validation", "test"}
    ids = {name: set(part["segment_id"]) for name, part in inputs.parts.items()}
    assert not (ids["fit"] & ids["test"]) and not (ids["validation"] & ids["test"]) and not (ids["fit"] & ids["validation"])
    joined = truth[["segment_id", "feeder_id"]].merge(manifest, on="segment_id")
    assert joined.groupby("feeder_id")["partition"].nunique().eq(1).all()
    assert len(inputs.masks) == 2 and not inputs.masks[0][MODEL].equals(inputs.masks[1][MODEL])
    # masks hide only observed cells of model columns, and link asset_age to the year
    mask = inputs.masks[0]
    for column in MODEL:
        assert truth.loc[mask[column], column].notna().all()
    assert (mask.loc[mask["installation_year"], "asset_age"]).all()
    # the masked frame handed to the methods has the hidden cells missing and the fit partition carries no test truth
    assert inputs.masked.loc[mask["material"], "material"].isna().all()


def test_run_protocols_scores_test_cells_against_truth_and_records_everything():
    truth = _typed(make_conductor_frame(300, seed=52, n_feeders=15))
    inputs = build_evaluation_inputs(SCHEMA, CONFIG, EvaluationSettings(mask_rate=0.2, repeats=1), truth=truth)
    config = _fast_config()
    output = run_protocols(inputs, SCHEMA, config, EvaluationSettings(mask_rate=0.2, repeats=1), methods=("median_mode", "predictive_missforest", "missforest"))
    summary = output.summary
    assert set(summary["protocol"]) == {"unseen_records", "transductive"}
    assert set(summary.loc[summary["protocol"] == "unseen_records", "method"]) == {"median_mode", "predictive_missforest"}  # transductive-only method excluded
    assert set(summary.loc[summary["protocol"] == "transductive", "method"]) == {"median_mode", "predictive_missforest", "missforest"}
    test_ids = set(inputs.parts["test"]["segment_id"].astype(str))
    assert set(output.scored_cells["record_id"]) <= test_ids
    n_test_hidden = int(inputs.masks[0].loc[inputs.parts["test"].index, MODEL].to_numpy().sum())
    per_method = output.scored_cells.groupby(["protocol", "method"]).size()
    assert (per_method == n_test_hidden).all()
    # truth in scored_cells equals the reference values
    sample = output.scored_cells.iloc[0]
    assert str(truth.set_index("segment_id").loc[sample["record_id"], sample["column"]]) == str(sample["truth"])
    year = summary[(summary["variable"] == "installation_year") & (summary["method"] == "median_mode")]
    assert year["within_5_years"].between(0, 1).all()
    assert output.metadata["partitions"]["test"] > 0 and output.metadata["mask_source"] == "drawn_in_evaluation"


def test_masked_input_mode_scores_exactly_the_masked_cells_with_truth(tmp_path):
    complete = make_conductor_frame(250, seed=53, n_feeders=10)
    masked = inject_missing(complete, MODEL, rate=0.25, seed=5)
    inputs = build_evaluation_inputs(SCHEMA, CONFIG, EvaluationSettings(), truth=_typed(complete), masked=_typed(masked))
    assert inputs.mask_source == "masked_input" and len(inputs.masks) == 1
    mask = inputs.masks[0]
    for column in MODEL:
        expected = masked[column].isna().to_numpy()
        if column == "installation_year":  # recovery restored years where asset_age survived
            expected = expected & inputs.masked[column].isna().to_numpy()
        assert (mask[column].to_numpy() == expected).all()
    with pytest.raises(ValueError, match="absent from the truth"):
        build_evaluation_inputs(SCHEMA, CONFIG, EvaluationSettings(), truth=_typed(complete.iloc[:100]), masked=_typed(masked))


def test_cli_evaluate_writes_outputs_and_reuses_the_manifest(tmp_path):
    complete = make_conductor_frame(220, seed=54, n_feeders=10)
    truth_csv = write_csv(complete, tmp_path / "complete.csv")
    masked_csv = write_csv(inject_missing(complete, MODEL, rate=0.2, seed=9), tmp_path / "masked.csv")
    out = tmp_path / "run"
    fast = ["--set", "n_estimators=6", "--set", "n_passes=1", "--set", "max_iter=2"]
    assert main(["evaluate", "--truth", str(truth_csv), "--input", str(masked_csv), "--out", str(out), *fast]) == 0
    ev = out / "evaluation"
    for name in ("partition_manifest.csv", "evaluation_mask_repeat0.csv", "metrics.csv", "per_class.csv", "summary.csv", "scored_cells.csv"):
        assert (ev / name).is_file(), name
    meta = json.loads((out / "run_metadata.json").read_text())["evaluation"]
    assert meta["mask_source"] == "masked_input" and meta["partition_strategy"] == "group"
    assert set(meta["headline"]) == {"unseen_records", "transductive"}
    manifest_before = pd.read_csv(ev / "partition_manifest.csv")
    # a second run without a masked input draws its own masks but reuses the manifest
    out2 = tmp_path / "run2"
    assert main(["evaluate", "--truth", str(truth_csv), "--out", str(out2), "--manifest", str(ev / "partition_manifest.csv"), *fast]) == 0
    assert pd.read_csv(out2 / "evaluation" / "partition_manifest.csv").equals(manifest_before)
    meta2 = json.loads((out2 / "run_metadata.json").read_text())["evaluation"]
    assert meta2["mask_source"] == "drawn_in_evaluation" and meta2["mask_seed"] is not None


def _fast_config():
    from dataclasses import replace

    selected = {k: dict(v) for k, v in CONFIG.selected_hyperparameters.items()}
    selected.setdefault("predictive_missforest", {}).update({"n_estimators": 6, "n_passes": 1})
    selected.setdefault("missforest", {}).update({"n_estimators": 6, "max_iter": 2})
    return replace(CONFIG, selected_hyperparameters=selected)
