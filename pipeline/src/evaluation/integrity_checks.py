"""Integrity checks on a completed dataset.

These are pass/fail checks that any imputation output must satisfy regardless
of method quality: the imputer filled what it was asked to fill, touched
nothing else, respected the declared constraints and vocabularies, and kept
the records in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..ingestion.schema import DataSchema

CHECK_COLUMNS = ("check", "column", "status", "count", "detail")


@dataclass
class IntegrityResult:
    checks: pd.DataFrame

    @property
    def passed(self) -> bool:
        return not (self.checks["status"] == "fail").any()

    @property
    def n_failed(self) -> int:
        return int((self.checks["status"] == "fail").sum())


def _equal_values(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Element-wise equality treating NaN == NaN as equal and comparing numerically when possible."""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
        return (a == b) | (np.isnan(a) & np.isnan(b))
    a = left.astype(object).to_numpy()
    b = right.astype(object).to_numpy()
    out = np.empty(len(a), dtype=bool)
    for i, (x, y) in enumerate(zip(a, b)):
        x_missing = x is None or (isinstance(x, float) and np.isnan(x))
        y_missing = y is None or (isinstance(y, float) and np.isnan(y))
        if x_missing or y_missing:
            out[i] = x_missing and y_missing
        else:
            try:
                out[i] = (x == y) or (float(x) == float(y))
            except (TypeError, ValueError):
                out[i] = str(x) == str(y)
    return out


def check_integrity(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    schema: DataSchema,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    constraints: Mapping[str, Mapping[str, Any]] | None = None,
) -> IntegrityResult:
    """Run every integrity check and return a table with one row per check."""
    constraints = dict(constraints or {})
    model_columns = list(numeric_columns) + list(categorical_columns)
    rows: list[dict[str, Any]] = []

    def add(check: str, column: str, ok: bool, count: int = 0, detail: str = "") -> None:
        rows.append({"check": check, "column": column, "status": "pass" if ok else "fail", "count": int(count), "detail": detail})

    # Shape and order ----------------------------------------------------------
    add("row_count_unchanged", "", len(original) == len(imputed), abs(len(original) - len(imputed)), f"{len(original)} in, {len(imputed)} out")
    add("columns_unchanged", "", list(original.columns) == list(imputed.columns), 0, "" if list(original.columns) == list(imputed.columns) else "column names or order differ")
    if len(original) != len(imputed) or list(original.columns) != list(imputed.columns):
        return IntegrityResult(pd.DataFrame(rows, columns=list(CHECK_COLUMNS)))

    id_column = schema.id_column
    if id_column in original.columns:
        same_ids = _equal_values(original[id_column], imputed[id_column]).all()
        add("identifier_order_unchanged", id_column, bool(same_ids), 0 if same_ids else int((~_equal_values(original[id_column], imputed[id_column])).sum()))

    # Model columns filled -----------------------------------------------------------
    for column in model_columns:
        remaining = int(imputed[column].isna().sum())
        add("no_missing_after_imputation", column, remaining == 0, remaining, "" if remaining == 0 else f"{remaining} cells still missing")

    # Observed cells untouched (every column) -----------------------------------------------
    # Derived columns are allowed to change where their source was imputed
    # (they are recomputed from the imputed source); everywhere else they must
    # be untouched like any other column.
    derived_sources = {
        name: schema.column(name).derive["source"]  # type: ignore[index]
        for name in schema.derived_columns
    }
    for column in original.columns:
        was_observed = original[column].notna().to_numpy()
        note = ""
        if column in derived_sources and derived_sources[column] in original.columns:
            source_imputed = original[derived_sources[column]].isna().to_numpy()
            was_observed = was_observed & ~source_imputed
            note = f"excluding {int(source_imputed.sum())} cells recomputed from imputed {derived_sources[column]}"
        if not was_observed.any():
            add("observed_values_unchanged", column, True, 0, "no observed values")
            continue
        equal = _equal_values(original[column], imputed[column])
        changed = int((~equal & was_observed).sum())
        add("observed_values_unchanged", column, changed == 0, changed, note if changed == 0 else f"{changed} observed cells were modified")

    # Constraints on imputed cells ------------------------------------------------------------
    for column, rule in constraints.items():
        if column not in imputed.columns:
            continue
        was_missing = original[column].isna().to_numpy()
        values = pd.to_numeric(imputed.loc[was_missing, column], errors="coerce") if rule.get("data_type") in ("integer", "float") else None
        if values is not None and len(values):
            if rule.get("data_type") == "integer":
                non_int = int((values.notna() & (values != np.round(values))).sum())
                add("constraint_integer", column, non_int == 0, non_int, "" if non_int == 0 else "imputed values with a fractional part")
            lo, hi = rule.get("min"), rule.get("max")
            out = pd.Series(False, index=values.index)
            if lo is not None:
                out |= values < lo
            if hi is not None:
                out |= values > hi
            add("constraint_range", column, int(out.sum()) == 0, int(out.sum()), f"[{lo}, {hi}]")
        elif rule.get("data_type") == "categorical" and "allowed" in rule:
            was_missing_cells = imputed.loc[was_missing, column].astype(str)
            bad = int((~was_missing_cells.isin([str(v) for v in rule["allowed"]])).sum())
            add("constraint_categorical", column, bad == 0, bad)

    # Vocabulary: imputed categorical values must come from the observed/allowed set ---------------
    for column in categorical_columns:
        was_missing = original[column].isna().to_numpy()
        filled = imputed.loc[was_missing, column].dropna().astype(str)
        if filled.empty:
            add("imputed_in_vocabulary", column, True, 0, "nothing imputed")
            continue
        spec = schema.columns.get(column)
        vocabulary = set(original[column].dropna().astype(str))
        if spec is not None and spec.allowed:
            vocabulary |= set(spec.allowed)
        bad = int((~filled.isin(vocabulary)).sum())
        add("imputed_in_vocabulary", column, bad == 0, bad, "" if bad == 0 else f"values outside the observed/allowed vocabulary: {sorted(set(filled[~filled.isin(vocabulary)]))[:5]}")

    # Derived fields consistent with their sources ------------------------------------------------
    for name in schema.derived_columns:
        spec = schema.column(name)
        source = spec.derive["source"]  # type: ignore[index]
        if name not in imputed.columns or source not in imputed.columns:
            continue
        derived = schema.snapshot_year - pd.to_numeric(imputed[source], errors="coerce")
        existing = pd.to_numeric(imputed[name], errors="coerce")
        produced = original[source].isna() | original[name].isna()
        both = derived.notna() & existing.notna() & produced
        bad = int((both & ~np.isclose(derived, existing)).sum())
        add("derived_field_consistent", name, bad == 0, bad, f"{name} = {schema.snapshot_year} - {source} on cells the pipeline filled or recomputed")

    return IntegrityResult(pd.DataFrame(rows, columns=list(CHECK_COLUMNS)))
