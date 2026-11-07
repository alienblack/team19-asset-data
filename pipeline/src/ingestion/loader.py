"""Record ingestion.

Reads a CSV extract, applies the schema's column mapping and returns the
records as text so that the quality and cleaning stages see exactly what the
source contained. Type coercion is done by the cleaner (with an audit log) or,
for inputs that are already clean, by :func:`apply_schema_types`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import DataSchema


class IngestionError(ValueError):
    """Raised when the input cannot be read into the schema."""


@dataclass
class LoadResult:
    """Outcome of :func:`load_records`."""

    frame: pd.DataFrame
    source_path: Path
    renamed: dict[str, str] = field(default_factory=dict)
    unknown_columns: tuple[str, ...] = ()
    missing_required: tuple[str, ...] = ()

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    @property
    def n_columns(self) -> int:
        return int(self.frame.shape[1])


def _order_columns(frame: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    declared = [name for name in schema.column_names if name in frame.columns]
    extra = [name for name in frame.columns if name not in schema.columns]
    return frame.loc[:, declared + extra]


def load_records(path: str | Path, schema: DataSchema) -> LoadResult:
    """Read a CSV extract as text and apply the schema's column mapping.

    * every cell is read as a string; empty cells become missing (NaN);
    * leading/trailing whitespace is removed from column names only (cell
      text is left untouched so the cleaner can log what it changed);
    * ``schema.column_mapping`` renames partner column names to canonical names;
    * declared columns are ordered as in the schema, extra columns follow.

    Missing required columns are reported in the result rather than raised, so
    the quality stage can list them alongside other issues.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[], encoding="utf-8-sig")
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError, csv.Error) as exc:
        raise IngestionError(f"Could not read '{path}': {exc}") from exc

    header = [str(c).strip() for c in header]
    duplicated = sorted({c for c in header if header.count(c) > 1})
    if duplicated:
        raise IngestionError(f"Input has duplicate column names: {duplicated}.")
    result = align_to_schema(frame, schema)
    result.source_path = path.resolve()
    return result


def align_to_schema(frame: pd.DataFrame, schema: DataSchema) -> LoadResult:
    """Map a text frame (every cell a string, blanks as ``""`` or NaN) onto the schema.

    This is the second half of :func:`load_records`, exposed so a frame read by
    the schema-free stage (:func:`ingestion.stage.ingest_csv`) can enter the
    schema-aware stages without re-reading the file: column mapping applied,
    blank cells made missing, declared columns ordered first, a fresh
    ``RangeIndex``. Cell text is otherwise untouched.
    """
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.columns.duplicated().any():
        duplicated = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise IngestionError(f"Input has duplicate column names: {duplicated}.")

    renamed = {src: dst for src, dst in schema.column_mapping.items() if src in frame.columns}
    collisions = sorted(set(renamed.values()) & (set(frame.columns) - set(renamed)))
    if collisions:
        raise IngestionError(
            f"column_mapping would overwrite existing columns: {collisions}. "
            "Remove the mapping or rename the source columns."
        )
    frame = frame.rename(columns=renamed)

    if schema.id_generation == "row_number" and schema.id_column not in frame.columns:
        # Only appropriate for paired synthetic files with identical row order.
        frame[schema.id_column] = [str(i) for i in range(len(frame))]

    # Empty cells are missing; other missing tokens are handled by the cleaner
    # (logged) or apply_schema_types (lenient).
    frame = frame.astype(object)
    blank = frame.map(lambda v: isinstance(v, str) and v == "")
    frame = frame.mask(blank, np.nan)
    frame = _order_columns(frame, schema)
    frame.index = pd.RangeIndex(len(frame))

    unknown = tuple(name for name in frame.columns if name not in schema.columns)
    missing_required = tuple(name for name in schema.required_columns if name not in frame.columns)
    return LoadResult(
        frame=frame,
        source_path=Path("<in-memory frame>"),
        renamed=renamed,
        unknown_columns=unknown,
        missing_required=missing_required,
    )


def apply_schema_types(frame: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    """Lenient, unlogged coercion to schema types for already-clean inputs.

    Numeric columns are parsed with ``errors="coerce"`` (unparseable values
    become missing); integer columns stay float64 while they contain missing
    values; missing tokens become NaN in every declared column. Use the
    cleaner when the changes must be audited.
    """
    out = frame.copy(deep=True)
    tokens = schema.missing_token_set()
    for name, spec in schema.columns.items():
        if name not in out.columns:
            continue
        column = out[name]
        if spec.is_numeric:
            as_text = column.astype(object).where(column.notna(), None)
            as_text = as_text.map(lambda v: None if v is None or str(v).strip().lower() in tokens else v)
            out[name] = pd.to_numeric(as_text, errors="coerce").astype(float)
        else:
            text = column.astype(object).where(column.notna(), None)
            text = text.map(
                lambda v: np.nan if v is None or str(v).strip().lower() in tokens else str(v).strip()
            )
            out[name] = text.astype(object)
    return out


def finalise_output_types(frame: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    """Cast integer columns without missing values to int64 for export."""
    out = frame.copy(deep=True)
    for name, spec in schema.columns.items():
        if name in out.columns and spec.dtype == "integer":
            numeric = pd.to_numeric(out[name], errors="coerce")
            if numeric.notna().all() and np.allclose(numeric, np.round(numeric)):
                out[name] = numeric.round().astype("int64")
            else:
                out[name] = numeric.astype(float)
    return out
