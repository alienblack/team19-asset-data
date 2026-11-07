"""Record ingestion.

Two entry points:

* :func:`ingest_csv` reads a CSV **without a schema**, preserving every value
  as text (blank cells stay ``""``), and returns an :class:`IngestionResult`
  with a provenance report (row/column inventory, duplicate headers, SHA-256).
* :func:`load_records` reads a CSV **against the approved schema** (column
  mapping, declared order, blank cells as missing) for the quality, cleaning
  and imputation stages. :func:`apply_schema_types` and
  :func:`finalise_output_types` coerce typed frames for those stages.
"""

from .loader import IngestionError, LoadResult, align_to_schema, apply_schema_types, finalise_output_types, load_records
from .schema import ColumnSpec, DataSchema, SchemaError, load_schema, parse_schema
from .stage import IngestionResult, calculate_file_hash, ingest_csv, inspect_csv

__all__ = [
    "ColumnSpec",
    "DataSchema",
    "IngestionError",
    "IngestionResult",
    "LoadResult",
    "SchemaError",
    "align_to_schema",
    "apply_schema_types",
    "calculate_file_hash",
    "finalise_output_types",
    "ingest_csv",
    "inspect_csv",
    "load_records",
    "load_schema",
    "parse_schema",
]
