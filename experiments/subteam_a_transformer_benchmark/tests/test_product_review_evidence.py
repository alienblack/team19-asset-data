"""Reject corrupted recorded values and prediction ledgers in the demo evidence."""
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_product_review import verify


@pytest.fixture
def evidence(tmp_path):
    source = ROOT / "docs" / "product_review" / "evidence" / "live_default"
    target = tmp_path / "evidence"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.png", "README.md"))
    return target


def test_checked_demo_evidence_recomputes(evidence):
    result = verify(evidence)
    assert result["status"] == "PASS"
    assert result["field_metrics_checked"] == 120
    assert result["completed_model_conditions"] == 8


def test_changed_observation_is_rejected(evidence):
    measured = pd.read_csv(evidence / "measured_test.csv.gz", float_precision="round_trip")
    record = measured.loc[measured.condition.eq("Scattered") & measured.x.notna()].iloc[0]
    path = evidence / "completed_test.csv.gz"
    completed = pd.read_csv(path, float_precision="round_trip")
    selected = completed.condition.eq(record.condition) & completed.asset_id.eq(record.asset_id)
    completed.loc[selected, "x"] += 1
    completed.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="recorded observation"):
        verify(evidence)


def test_changed_hidden_prediction_is_rejected(evidence):
    path = evidence / "hidden_predictions.csv.gz"
    ledger = pd.read_csv(path, float_precision="round_trip")
    selected = ledger.index[ledger.method.eq("mixture") & ledger.attribute.eq("install_date")][0]
    ledger.loc[selected, "fill"] += 1000
    ledger.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="Completed fills disagree"):
        verify(evidence)
