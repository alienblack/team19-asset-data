"""Schema-independent CSV ingestion."""

import csv
import json
import hashlib
import pandas as pd

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestionResult:
    frame: pd.DataFrame
    report: dict
    report_json: Path | None = None

def calculate_file_hash(path):
    """Calculate SHA-256 checksum for source integrity."""
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def inspect_csv(path, encoding="utf-8-sig"):
    """
    Inspect CSV structure without modifying data
    or requiring a predefined schema.
    """
    with open(path, "r", encoding=encoding, newline="") as file:
        reader = csv.reader(file, strict=True)
        columns = next(reader, None)

        if not columns:
            raise ValueError("CSV file is empty or has no header.")

        row_count = 0
        inconsistent_rows = []

        for row in reader:
            row_count += 1

            if len(row) != len(columns):
                inconsistent_rows.append(reader.line_num)

    duplicates = [
        name for name, count in Counter(columns).items()
        if count > 1
    ]

    return {
        "rows": row_count,
        "columns": len(columns),
        "column_names": columns,
        "duplicate_columns": duplicates,
        "empty_column_names": sum(not c.strip() for c in columns),
        "inconsistent_row_count": len(inconsistent_rows),
        "inconsistent_row_lines": inconsistent_rows[:20],
    }

def ingest_csv(
    input_csv,
    output_dir=None,
    overwrite=False,
    encoding="utf-8-sig",
):
    """
    Load CSV data without requiring a schema.

    - Reads original data into memory.
    - Preserves original column names and string values.
    - Does not clean, convert, or impute data.
    - Generates an ingestion report.
    - Optionally saves the JSON report.
    - Does not create a duplicate CSV.
    """

    input_csv = Path(input_csv).expanduser().resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_csv.name}"
        )

    # Inspect CSV structure
    inspection = inspect_csv(
        input_csv,
        encoding=encoding,
    )

    # Malformed rows cannot safely form a rectangular DataFrame
    if inspection["inconsistent_row_count"] > 0:
        raise ValueError(
            "CSV contains rows with inconsistent column counts. "
            f"Affected rows: {inspection['inconsistent_row_lines']}"
        )

    # Read original values without automatic type conversion
    with open(input_csv, "r", encoding=encoding, newline="") as file:
        reader = csv.reader(file, strict=True)
        columns = next(reader)
        records = list(reader)

    frame = pd.DataFrame(
        records,
        columns=columns,
    )

    # Generate ingestion metadata
    report = {
        "source_file": input_csv.name,
        "file_size_bytes": input_csv.stat().st_size,
        "sha256": calculate_file_hash(input_csv),
        "encoding_used": encoding,
        **inspection,
    }

    report_json = None

    # Save report only if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        report_json = output_dir / "ingestion_report.json"

        if report_json.exists() and not overwrite:
            raise FileExistsError(
                "Ingestion report already exists. "
                "Set overwrite=True to replace it."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        with open(report_json, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

    return IngestionResult(
        frame=frame,
        report=report,
        report_json=report_json,
    )