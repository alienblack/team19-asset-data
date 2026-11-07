"""Cell-level provenance of a completed dataset.

Every cell of the final output falls into exactly one class:

``observed``
    the value was present in the cleaned input and was not touched;
``derived``
    the value was computed deterministically from other observed or imputed
    values (a source recovered from its derived column before imputation, or
    a derived column filled/recomputed after imputation);
``imputed``
    the value was produced by the imputation method;
``unresolved``
    the cell is still missing (the column is not a model attribute, or the
    method could not fill it).

Observed cells are not listed in ``cell_provenance.csv``: their absence is
the statement that they were left alone, which the integrity check
``observed_values_unchanged`` enforces for every column. The other three
classes are listed one row per cell, with the method or rule that produced
the value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..ingestion.schema import DataSchema

PROVENANCE_COLUMNS = ("record_id", "column", "status", "method", "value", "detail")
STATUSES = ("observed", "derived", "imputed", "unresolved")


@dataclass
class ProvenanceResult:
    cells: pd.DataFrame
    summary: pd.DataFrame
    totals: dict[str, int] = field(default_factory=dict)
    unresolved: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "totals": self.totals,
            "by_column": self.summary.set_index("column").to_dict(orient="index") if len(self.summary) else {},
            "unresolved_columns": {str(r.column): {"n_cells": int(r.n_cells), "reason": r.reason} for r in self.unresolved.itertuples()} if len(self.unresolved) else {},
        }


def _value_at(frame: pd.DataFrame, positions: np.ndarray, column: str) -> np.ndarray:
    return frame[column].to_numpy()[positions]


def build_cell_provenance(
    before: pd.DataFrame,
    final: pd.DataFrame,
    schema: DataSchema,
    *,
    model_columns: Sequence[str],
    method: str,
    recovered_cells: pd.DataFrame | None = None,
    derived_cells: pd.DataFrame | None = None,
    diagnostics: pd.DataFrame | None = None,
) -> ProvenanceResult:
    """Classify every non-observed cell of ``final`` relative to ``before``.

    ``before`` is the cleaned, typed frame **before** recovery and imputation;
    ``final`` is the completed output with the same shape. ``recovered_cells``
    and ``derived_cells`` are boolean frames marking what the derivation steps
    wrote; ``diagnostics`` (one row per imputed cell) supplies the method's
    per-cell detail when available.
    """
    if list(before.columns) != list(final.columns) or len(before) != len(final):
        raise ValueError("before/final frames must share shape and column order.")
    ids = before[schema.id_column].astype(str).to_numpy()
    n = len(before)
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    recovered = recovered_cells.reindex(columns=before.columns, fill_value=False) if recovered_cells is not None else None
    derived = derived_cells.reindex(columns=before.columns, fill_value=False) if derived_cells is not None else None
    detail_lookup: dict[tuple[int, str], str] = {}
    if diagnostics is not None and len(diagnostics) and {"row", "column"} <= set(diagnostics.columns):
        for r in diagnostics.itertuples(index=False):
            parts = []
            for key in ("prediction_source", "max_proba", "margin", "tree_std"):
                value = getattr(r, key, None)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    parts.append(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}")
            detail_lookup[(int(r.row), str(r.column))] = "; ".join(parts)

    for column in before.columns:
        was_missing = before[column].isna().to_numpy()
        now_missing = final[column].isna().to_numpy()
        col_counts = {"observed": int((~was_missing).sum()), "derived": 0, "imputed": 0, "unresolved": 0}
        spec = schema.columns.get(column)
        for pos in np.flatnonzero(was_missing):
            if now_missing[pos]:
                status, how, detail = "unresolved", "", (
                    "not a model attribute" if column not in model_columns else "method left the cell empty"
                )
            elif recovered is not None and bool(recovered.iat[pos, recovered.columns.get_loc(column)]):
                src = schema.recoverable_sources().get(column, "")
                status, how, detail = "derived", "recover_source", f"{column} = {schema.snapshot_year} - {src}"
            elif derived is not None and bool(derived.iat[pos, derived.columns.get_loc(column)]):
                src = spec.derive["source"] if spec is not None and spec.derive else ""  # type: ignore[index]
                status, how, detail = "derived", "snapshot_year_minus", f"{column} = {schema.snapshot_year} - {src}"
            elif column in model_columns:
                status, how, detail = "imputed", method, detail_lookup.get((pos, column), "")
            else:
                status, how, detail = "derived", "post-imputation", ""
            col_counts[status] += 1
            rows.append({"record_id": ids[pos], "column": column, "status": status, "method": how, "value": final[column].to_numpy()[pos], "detail": detail})
        # observed cells recomputed by a derivation (source was imputed) are derived, not observed
        if derived is not None:
            recomputed = derived[column].to_numpy() & ~was_missing
            for pos in np.flatnonzero(recomputed):
                src = spec.derive["source"] if spec is not None and spec.derive else ""  # type: ignore[index]
                col_counts["observed"] -= 1
                col_counts["derived"] += 1
                rows.append({"record_id": ids[pos], "column": column, "status": "derived", "method": "snapshot_year_minus", "value": final[column].to_numpy()[pos], "detail": f"recomputed: {src} was imputed"})
        counts[column] = col_counts

    cells = pd.DataFrame(rows, columns=list(PROVENANCE_COLUMNS))
    summary = pd.DataFrame(
        [{"column": c, **v, "n_cells": n} for c, v in counts.items()],
        columns=["column", "observed", "derived", "imputed", "unresolved", "n_cells"],
    )
    totals = {status: int(summary[status].sum()) for status in STATUSES}
    totals["n_cells"] = int(n * len(before.columns))
    unresolved = summary[summary["unresolved"] > 0][["column", "unresolved"]].rename(columns={"unresolved": "n_cells"}).copy()
    unresolved["reason"] = ["not a model attribute" if c not in model_columns else "method left the cell empty" for c in unresolved["column"]]
    return ProvenanceResult(cells=cells, summary=summary, totals=totals, unresolved=unresolved.reset_index(drop=True))
