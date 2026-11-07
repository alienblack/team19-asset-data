"""Data quality assessment.

Runs schema, identifier, type, range and vocabulary checks on the records as
read from the source (text) and reports every finding as a row of
``quality_report.csv``. The validator never modifies data; the cleaner does
that, with its own log.

Severity levels
    error    the pipeline cannot proceed safely (missing required column,
             duplicate identifiers, no observed values for a target)
    warning  values the cleaner will change or that need a decision
    info     descriptive findings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..ingestion.schema import DataSchema

REPORT_COLUMNS = ("check", "column", "severity", "count", "rate", "detail")
MAX_EXAMPLES = 8


@dataclass
class QualityReport:
    """Findings of :func:`validate_records`."""

    issues: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not (self.issues["severity"] == "error").any()

    @property
    def n_errors(self) -> int:
        return int((self.issues["severity"] == "error").sum())

    @property
    def n_warnings(self) -> int:
        return int((self.issues["severity"] == "warning").sum())


def _text(series: pd.Series) -> pd.Series:
    """Stripped text with missing values preserved as NaN."""
    return series.map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())


def effective_missing_mask(series: pd.Series, schema: DataSchema) -> pd.Series:
    """True where a cell is empty or holds one of the schema's missing tokens."""
    text = _text(series)
    tokens = schema.missing_token_set()
    return text.isna() | text.map(lambda v: isinstance(v, str) and v.lower() in tokens).astype(bool)


def _examples(values: pd.Series, limit: int = MAX_EXAMPLES) -> str:
    counts = values.value_counts()
    parts = [f"{value!r} x{count}" for value, count in counts.head(limit).items()]
    more = len(counts) - limit
    if more > 0:
        parts.append(f"... +{more} more distinct")
    return "; ".join(parts)


def _numeric_parse(text: pd.Series) -> pd.Series:
    return pd.to_numeric(text, errors="coerce")


def validate_records(frame: pd.DataFrame, schema: DataSchema) -> QualityReport:
    """Assess a text frame (as returned by :func:`ingestion.loader.load_records`)."""
    n_rows = len(frame)
    rows: list[dict[str, Any]] = []

    def add(check: str, column: str | None, severity: str, count: int, detail: str = "") -> None:
        rows.append(
            {
                "check": check,
                "column": column if column is not None else "",
                "severity": severity,
                "count": int(count),
                "rate": (float(count) / n_rows) if n_rows else np.nan,
                "detail": detail,
            }
        )

    # --- structure ---------------------------------------------------------
    for name in schema.required_columns:
        if name not in frame.columns:
            add("missing_required_column", name, "error", n_rows, "Required column not found in input.")
    for name in frame.columns:
        if name not in schema.columns:
            add("unknown_column", name, "info", n_rows, "Column is not declared in the schema; carried through unchanged.")
    if n_rows == 0:
        add("empty_input", None, "error", 0, "Input contains no records.")

    # --- identifiers ---------------------------------------------------------
    exact_dup = frame.duplicated(keep="first")
    if exact_dup.any():
        add("exact_duplicate_row", None, "warning", exact_dup.sum(), "Rows identical in every column (first occurrence kept by the cleaner).")

    id_column = schema.id_column
    if id_column in frame.columns and n_rows:
        ids = _text(frame[id_column])
        id_missing = effective_missing_mask(frame[id_column], schema)
        if id_missing.any():
            add("identifier_missing", id_column, "error", id_missing.sum(), "Records without an identifier cannot be traced.")
        # Exact duplicate rows are handled by the cleaner; only identifiers shared
        # by rows that differ elsewhere are conflicts that need a decision.
        present = ids[~id_missing & ~exact_dup]
        dup_mask = present.duplicated(keep=False)
        if dup_mask.any():
            add(
                "identifier_duplicate",
                id_column,
                "error",
                dup_mask.sum(),
                "Identifier shared by differing records: " + _examples(present[dup_mask]),
            )

    # --- per column ----------------------------------------------------------
    tokens = schema.missing_token_set()
    for name, spec in schema.columns.items():
        if name not in frame.columns or n_rows == 0:
            continue
        column = frame[name]
        text = _text(column)
        missing = effective_missing_mask(column, schema)
        token_mask = text.map(lambda v: isinstance(v, str) and v.lower() in tokens).astype(bool)
        if missing.any():
            add("missing_values", name, "info", missing.sum(), "Empty cells plus missing tokens.")
        if token_mask.any():
            add("missing_token", name, "info", token_mask.sum(), "Text standing for missing: " + _examples(text[token_mask]))
        if spec.required and spec.is_model_column and missing.all():
            add("all_missing_column", name, "warning", n_rows, "No observed values; the attribute cannot be learned from this file.")

        observed = text[~missing]
        if spec.is_numeric:
            parsed = _numeric_parse(observed)
            bad = parsed.isna()
            if bad.any():
                add("non_numeric_value", name, "warning", bad.sum(), _examples(observed[bad]))
            values = parsed[~bad]
            if spec.sentinel_values:
                sentinel = values.isin(spec.sentinel_values)
                if sentinel.any():
                    add("sentinel_value", name, "warning", sentinel.sum(), _examples(values[sentinel].astype(object)))
                values = values[~sentinel]
            out_of_range = pd.Series(False, index=values.index)
            if spec.min is not None:
                out_of_range |= values < spec.min
            if spec.max is not None:
                out_of_range |= values > spec.max
            if out_of_range.any():
                add(
                    "out_of_range",
                    name,
                    "warning",
                    out_of_range.sum(),
                    f"Outside [{spec.min}, {spec.max}]: " + _examples(values[out_of_range].astype(object)),
                )
            if spec.dtype == "integer":
                non_int = values.notna() & (values != np.round(values))
                if non_int.any():
                    add("non_integer_value", name, "warning", non_int.sum(), _examples(values[non_int].astype(object)))
        elif spec.is_categorical and spec.allowed:
            allowed = set(spec.allowed)
            folded = {label.lower(): label for label in spec.allowed}
            exact = observed.isin(allowed)
            variant = ~exact & observed.map(lambda v: isinstance(v, str) and v.lower() in folded).astype(bool)
            unknown = ~exact & ~variant
            if variant.any():
                add("label_variant", name, "info", variant.sum(), "Case/whitespace variants of allowed labels: " + _examples(observed[variant]))
            if unknown.any():
                add("unknown_category", name, "warning", unknown.sum(), "Not in the allowed list: " + _examples(observed[unknown]))

        if spec.role == "target":
            n_observed = int((~missing).sum())
            if spec.is_numeric:
                n_observed = int(_numeric_parse(observed).notna().sum())
            if n_observed < 2:
                add("insufficient_observed_target", name, "error", n_rows - n_observed, "Fewer than two observed values; the target cannot be reconstructed.")

    # Derived fields that disagree with their source in the SOURCE data ---------
    for name in schema.derived_columns:
        spec = schema.column(name)
        source = spec.derive["source"]  # type: ignore[index]
        if name not in frame.columns or source not in frame.columns or n_rows == 0:
            continue
        derived_values = _numeric_parse(_text(frame[name]))
        source_values = _numeric_parse(_text(frame[source]))
        expected = schema.snapshot_year - source_values
        both = derived_values.notna() & source_values.notna()
        mismatch = both & ~np.isclose(derived_values.fillna(0), expected.fillna(0))
        if mismatch.any():
            add(
                "derived_field_mismatch",
                name,
                "warning",
                mismatch.sum(),
                f"{name} != {schema.snapshot_year} - {source} (snapshot_year in data_schema.yaml may not match the extract date)",
            )
        orphan = derived_values.notna() & source_values.isna()
        if orphan.any():
            add("derived_without_source", name, "info", orphan.sum(), f"{name} present while {source} is missing; recomputed after {source} is imputed")

    issues = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    severity_order = {"error": 0, "warning": 1, "info": 2}
    issues = issues.sort_values(
        by=["severity", "check", "column"],
        key=lambda s: s.map(severity_order) if s.name == "severity" else s,
        kind="mergesort",
    ).reset_index(drop=True)

    summary = {
        "n_rows": int(n_rows),
        "n_columns": int(frame.shape[1]),
        "n_errors": int((issues["severity"] == "error").sum()),
        "n_warnings": int((issues["severity"] == "warning").sum()),
        "n_info": int((issues["severity"] == "info").sum()),
        "missing_required_columns": [n for n in schema.required_columns if n not in frame.columns],
        "unknown_columns": [n for n in frame.columns if n not in schema.columns],
    }
    return QualityReport(issues=issues, summary=summary)
