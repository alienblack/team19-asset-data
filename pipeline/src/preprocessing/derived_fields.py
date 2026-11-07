"""Post-imputation derived fields.

Derived columns are deterministic functions of other columns (today only
``asset_age = snapshot_year - installation_year``). They are computed AFTER
imputation so that a target can never be reconstructed from its own derivative
during modelling. The schema declares them with role ``derived``, and the
leakage guard keeps them out of every imputer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..ingestion.schema import DataSchema


@dataclass
class DerivationResult:
    frame: pd.DataFrame
    log: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    #: Boolean frame (same index as ``frame``) marking cells this step wrote; None when not tracked.
    cells: pd.DataFrame | None = None


def recover_sources(frame: pd.DataFrame, schema: DataSchema, snapshot_year: int | None = None) -> DerivationResult:
    """Restore a missing source from its observed derived column (deterministic recovery).

    For every derived column declared with ``recover_source: true`` and method
    ``snapshot_year_minus``: where the source is missing and the derived value
    is observed, ``source = snapshot_year - derived``. A recovered value outside
    the source column's declared range is rejected and counted. Runs BEFORE
    imputation so the imputer sees the recovered value as observed; the cells
    are reported with provenance ``derived``, never ``imputed``.
    """
    snapshot = int(snapshot_year if snapshot_year is not None else schema.snapshot_year)
    out = frame.copy(deep=True)
    cells = pd.DataFrame(False, index=out.index, columns=out.columns)
    rows = []
    for source, derived_name in schema.recoverable_sources().items():
        if source not in out.columns or derived_name not in out.columns:
            rows.append({"column": source, "from": derived_name, "snapshot_year": snapshot, "n_recovered": 0, "n_rejected_out_of_range": 0, "n_still_missing": int(out[source].isna().sum()) if source in out.columns else int(len(out)), "note": "column absent"})
            continue
        spec = schema.column(source)
        source_values = pd.to_numeric(out[source], errors="coerce").astype(float)
        derived_values = pd.to_numeric(out[derived_name], errors="coerce").astype(float)
        candidate = snapshot - derived_values
        eligible = source_values.isna() & derived_values.notna()
        in_range = pd.Series(True, index=out.index)
        if spec.min is not None:
            in_range &= candidate >= spec.min
        if spec.max is not None:
            in_range &= candidate <= spec.max
        if spec.dtype == "integer":
            candidate = candidate.round()
        accept = eligible & in_range
        rejected = eligible & ~in_range
        new = source_values.where(~accept, candidate)
        out[source] = new
        cells.loc[accept, source] = True
        rows.append(
            {
                "column": source,
                "from": derived_name,
                "snapshot_year": snapshot,
                "n_recovered": int(accept.sum()),
                "n_rejected_out_of_range": int(rejected.sum()),
                "n_still_missing": int(new.isna().sum()),
                "note": f"{source} = {snapshot} - {derived_name} where {source} was missing and {derived_name} observed",
            }
        )
    log = pd.DataFrame(rows, columns=["column", "from", "snapshot_year", "n_recovered", "n_rejected_out_of_range", "n_still_missing", "note"])
    summary = {r["column"]: {k: r[k] for k in ("n_recovered", "n_rejected_out_of_range", "n_still_missing")} for r in rows}
    return DerivationResult(frame=out, log=log, summary=summary, cells=cells)


def derive_fields(
    frame: pd.DataFrame,
    schema: DataSchema,
    snapshot_year: int | None = None,
    *,
    original: pd.DataFrame | None = None,
    overwrite_existing: bool = False,
) -> DerivationResult:
    """Fill derived columns from their sources.

    Missing derived cells are always filled. When ``original`` (the frame
    before imputation) is given, derived cells whose SOURCE was imputed are
    recomputed as well, because a value derived from a source that was unknown
    cannot be trusted more than the imputed source. Other existing values are
    kept; those that disagree with the derived value are counted as
    ``n_inconsistent_existing`` so the report can flag them. With
    ``overwrite_existing=True`` every cell is recomputed.
    """
    snapshot = int(snapshot_year if snapshot_year is not None else schema.snapshot_year)
    out = frame.copy(deep=True)
    cells = pd.DataFrame(False, index=out.index, columns=out.columns)
    rows = []
    for name in schema.derived_columns:
        spec = schema.column(name)
        method = spec.derive["method"]  # type: ignore[index]
        source = spec.derive["source"]  # type: ignore[index]
        if source not in out.columns:
            rows.append({"column": name, "source": source, "method": method, "snapshot_year": snapshot, "n_filled": 0, "n_recomputed_after_source_imputed": 0, "n_inconsistent_existing": 0, "n_still_missing": int(len(out)), "note": "source column absent"})
            continue
        source_values = pd.to_numeric(out[source], errors="coerce").astype(float)
        if method == "snapshot_year_minus":
            derived = snapshot - source_values
        else:  # pragma: no cover - schema parser forbids other methods
            raise ValueError(f"Unsupported derive method '{method}'.")

        if name in out.columns:
            existing = pd.to_numeric(out[name], errors="coerce").astype(float)
        else:
            existing = pd.Series(np.nan, index=out.index, dtype=float)

        both = existing.notna() & derived.notna()
        inconsistent = both & ~np.isclose(existing, derived)
        source_imputed = pd.Series(False, index=out.index)
        if original is not None and source in original.columns:
            source_imputed = pd.Series(original[source].isna().to_numpy(), index=out.index)
        if overwrite_existing:
            fill_mask = derived.notna()
        else:
            fill_mask = (existing.isna() | source_imputed) & derived.notna()
        recomputed = int((source_imputed & existing.notna() & derived.notna()).sum()) if not overwrite_existing else int(both.sum())
        inconsistent = inconsistent & ~fill_mask
        new = existing.where(~fill_mask, derived)
        out[name] = new
        if name not in cells.columns:
            cells[name] = False
        cells.loc[fill_mask, name] = True
        rows.append(
            {
                "column": name,
                "source": source,
                "method": method,
                "snapshot_year": snapshot,
                "n_filled": int(fill_mask.sum()),
                "n_recomputed_after_source_imputed": recomputed,
                "n_inconsistent_existing": int(inconsistent.sum()),
                "n_still_missing": int(new.isna().sum()),
                "note": "all values recomputed" if overwrite_existing else "existing values kept unless their source was imputed",
            }
        )
    log = pd.DataFrame(rows, columns=["column", "source", "method", "snapshot_year", "n_filled", "n_recomputed_after_source_imputed", "n_inconsistent_existing", "n_still_missing", "note"])
    summary = {r["column"]: {k: r[k] for k in ("n_filled", "n_recomputed_after_source_imputed", "n_inconsistent_existing", "n_still_missing")} for r in rows}
    return DerivationResult(frame=out, log=log, summary=summary, cells=cells)
