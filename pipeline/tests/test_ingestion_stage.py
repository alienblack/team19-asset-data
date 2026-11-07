"""Schema-free ingestion stage (ingest_csv / inspect_csv)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli import main
from src.ingestion import IngestionResult, calculate_file_hash, ingest_csv, inspect_csv


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_ingest_preserves_text_blanks_leading_zeros_and_tokens(tmp_path):
    source = _write(
        tmp_path / "in.csv",
        "segment_id,installation_year,material,note\n"
        "00123, 1973 ,Steel,NA\n"
        "00124,,N/A,\"a, quoted\"\n",
    )
    result = ingest_csv(source, output_dir=tmp_path / "out")
    assert isinstance(result, IngestionResult)
    frame = result.frame
    assert frame.shape == (2, 4)
    assert frame.iloc[0]["segment_id"] == "00123"  # leading zero kept
    assert frame.iloc[0]["installation_year"] == " 1973 "  # whitespace untouched
    assert frame.iloc[0]["note"] == "NA" and frame.iloc[1]["material"] == "N/A"  # tokens survive
    assert frame.iloc[1]["installation_year"] == ""  # blank stays a literal empty string
    assert frame.iloc[1]["note"] == "a, quoted"
    assert all(frame[c].map(type).eq(str).all() for c in frame.columns)

    report = result.report
    assert report["rows"] == 2 and report["columns"] == 4 and report["duplicate_columns"] == []
    assert report["sha256"] == calculate_file_hash(source) and len(report["sha256"]) == 64
    assert result.report_json == tmp_path / "out" / "ingestion_report.json"
    assert json.loads(result.report_json.read_text())["source_file"] == "in.csv"
    assert not (tmp_path / "out" / "ingested.csv").exists()  # no duplicate copy


def test_inspect_reports_duplicate_headers_and_ragged_rows(tmp_path):
    source = _write(tmp_path / "bad.csv", "a,b,a\n1,2,3\n4,5\n")
    info = inspect_csv(source)
    assert info["duplicate_columns"] == ["a"] and info["inconsistent_row_count"] == 1
    with pytest.raises(ValueError, match="inconsistent column counts"):
        ingest_csv(source)
    with pytest.raises(ValueError, match="empty"):
        ingest_csv(_write(tmp_path / "empty.csv", ""))


def test_ingest_refuses_to_overwrite_a_report_unless_asked(tmp_path):
    source = _write(tmp_path / "in.csv", "a,b\n1,2\n")
    ingest_csv(source, output_dir=tmp_path / "out")
    with pytest.raises(FileExistsError):
        ingest_csv(source, output_dir=tmp_path / "out")
    assert ingest_csv(source, output_dir=tmp_path / "out", overwrite=True).report["rows"] == 1


def test_cli_ingest_command(tmp_path, capsys):
    source = _write(tmp_path / "in.csv", "segment_id,material\nA,Steel\n")
    assert main(["ingest", "--input", str(source), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "ingestion_report.json").is_file()
    assert main(["ingest", "--input", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "out2")]) == 2
