"""Exploratory data profiling and legacy schema-aware model profiling.

``profile_dataset`` operates on a raw, schema-free ingestion DataFrame and
never changes its values. ``profile_records`` is retained for older callers
that already have an approved, typed schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ProfileResult:
    """Named tables produced by :func:`profile_records`."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def write(self, out_dir: str | Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, table in self.tables.items():
            path = out_dir / f"{name}.csv"
            table.to_csv(path, index=False)
            written.append(path)
        return written


# Candidate tokens are reported for human review, not treated as confirmed
# missing values or automatically converted to NaN.
POSSIBLE_MISSING_TOKENS = frozenset({
    "na", "n/a", "null", "none", "unknown", "missing", "not known", "?", "-"
})


def _blank_mask(series: pd.Series) -> pd.Series:
    """Detect actual nulls and blank/whitespace-only strings for analysis."""
    return series.isna() | series.astype("string").str.strip().eq("").fillna(False)


def _suggest_type(series: pd.Series, name: str) -> str:
    """Suggest a type without ever converting the input data."""
    observed = series.loc[~_blank_mask(series)].astype(str).str.strip()
    # Consider common placeholder tokens when guessing a type, while keeping
    # them in the source and listing them separately for human review.
    observed = observed.loc[~observed.str.casefold().isin(POSSIBLE_MISSING_TOKENS)]
    if observed.empty:
        return "unknown"

    label = name.lower()
    is_identifier = any(word in label for word in ("_id", "id_", "identifier", "serial", "uuid"))
    if is_identifier:
        return "identifier"

    numeric = pd.to_numeric(observed, errors="coerce")
    if numeric.notna().mean() >= 0.98:
        # A code such as 00123 must not be mistaken for a continuous quantity.
        has_leading_zero = observed.str.fullmatch(r"0\d+").fillna(False).any()
        return "identifier" if has_leading_zero else "numeric"

    date_like = observed.str.match(
        r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})"
    )
    if date_like.mean() >= 0.90:
        parsed = pd.to_datetime(observed, errors="coerce")
        if parsed.notna().mean() >= 0.90:
            return "date"

    distinct = observed.nunique(dropna=True)
    if distinct <= min(30, max(5, int(len(observed) * 0.10))):
        return "categorical"
    return "text"


def profile_dataset(frame: pd.DataFrame, max_patterns: int = 15) -> ProfileResult:
    """Discover properties of any ingested DataFrame, without a schema.

    Blank strings and actual nulls count as missing for *reporting only*.
    Strings such as ``Unknown`` are listed as possible missing tokens, not
    silently treated as missing. All source values remain unchanged.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("profile_dataset expects a pandas DataFrame")

    n, width = frame.shape
    names = [str(c) for c in frame.columns]
    duplicates = sorted({c for c in names if names.count(c) > 1})
    column_rows, numeric_rows, category_rows, outlier_rows, token_rows = [], [], [], [], []
    missing_masks: list[pd.Series] = []
    numerical: dict[str, pd.Series] = {}
    categorical: dict[str, pd.Series] = {}

    for position, name in enumerate(names):
        s = frame.iloc[:, position]
        missing = _blank_mask(s)
        missing_masks.append(missing.rename(f"{name} [{position}]"))
        observed = s.loc[~missing]
        kind = _suggest_type(s, name)
        column_rows.append({
            "position": position, "column": name, "suggested_type": kind,
            "n_rows": n, "n_observed": int((~missing).sum()),
            "n_missing": int(missing.sum()),
            "missing_rate": float(missing.mean()) if n else np.nan,
            "n_unique_observed": int(observed.nunique()),
            "sample_values": ", ".join(map(str, observed.astype(str).unique()[:3])),
        })
        stripped = observed.astype(str).str.strip().str.casefold()
        for token, count in stripped[stripped.isin(POSSIBLE_MISSING_TOKENS)].value_counts().items():
            token_rows.append({"column": name, "token": token, "count": int(count)})

        # Duplicated column labels are profiled by position, but excluded from
        # relationship plots until a data owner resolves the ambiguity.
        if name in duplicates:
            continue
        if kind == "numeric":
            vals = pd.to_numeric(observed, errors="coerce").dropna()
            numerical[name] = vals
            q1, q3 = (vals.quantile(0.25), vals.quantile(0.75)) if len(vals) else (np.nan, np.nan)
            numeric_rows.append({
                "column": name, "count": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "std": float(vals.std()) if len(vals) > 1 else np.nan,
                "min": float(vals.min()) if len(vals) else np.nan,
                "q1": float(q1), "median": float(vals.median()) if len(vals) else np.nan,
                "q3": float(q3), "max": float(vals.max()) if len(vals) else np.nan,
            })
            iqr = q3 - q1
            count = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum()) if len(vals) >= 5 and iqr > 0 else 0
            outlier_rows.append({"column": name, "method": "1.5 x IQR (candidate only)", "count": count})
        elif kind == "categorical":
            categorical[name] = observed
            for value, count in observed.value_counts(dropna=False).head(30).items():
                category_rows.append({
                    "column": name, "value": str(value), "count": int(count),
                    "share_of_observed": float(count / len(observed)) if len(observed) else np.nan,
                })

    miss = pd.concat(missing_masks, axis=1) if missing_masks else pd.DataFrame(index=frame.index)
    missing_rate = float(miss.to_numpy(dtype=float).mean()) if n and width else np.nan
    patterns = []
    if n and width:
        pattern_counts = miss.astype(int).astype(str).agg("".join, axis=1).value_counts().head(max_patterns)
        for pattern, count in pattern_counts.items():
            missing_names = [f"{names[i]} [{i}]" for i, flag in enumerate(pattern) if flag == "1"]
            patterns.append({
                "pattern": pattern, "missing_columns": ", ".join(missing_names) or "(complete)",
                "n_rows": int(count), "share": float(count / n),
            })

    relationships = []
    numeric_names = list(numerical)
    for i, left in enumerate(numeric_names):
        for right in numeric_names[i + 1:]:
            a, b = pd.to_numeric(frame[left], errors="coerce"), pd.to_numeric(frame[right], errors="coerce")
            both = pd.concat([a, b], axis=1).dropna()
            relationships.append({
                "column_a": left, "column_b": right, "statistic": "spearman_rho",
                "value": float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")) if len(both) >= 3 else np.nan,
                "n_pairwise_complete": int(len(both)),
            })

    category_names = list(categorical)[:12]
    for i, left in enumerate(category_names):
        for right in category_names[i + 1:]:
            a = frame[left].mask(_blank_mask(frame[left]))
            b = frame[right].mask(_blank_mask(frame[right]))
            n_complete = int((a.notna() & b.notna()).sum())
            relationships.append({
                "column_a": left, "column_b": right, "statistic": "cramers_v",
                "value": cramers_v(a, b), "n_pairwise_complete": n_complete,
            })
    for left in numeric_names[:12]:
        for right in category_names:
            a = pd.to_numeric(frame[left], errors="coerce")
            b = frame[right].mask(_blank_mask(frame[right]))
            relationships.append({
                "column_a": left, "column_b": right, "statistic": "eta_squared",
                "value": eta_squared(a, b),
                "n_pairwise_complete": int((a.notna() & b.notna()).sum()),
            })

    tables = {
        "column_profile": pd.DataFrame(column_rows, columns=["position", "column", "suggested_type", "n_rows", "n_observed", "n_missing", "missing_rate", "n_unique_observed", "sample_values"]),
        "missingness": pd.DataFrame(column_rows)[["column", "n_missing", "missing_rate"]] if column_rows else pd.DataFrame(columns=["column", "n_missing", "missing_rate"]),
        "potential_missing_tokens": pd.DataFrame(token_rows, columns=["column", "token", "count"]),
        "numeric_summary": pd.DataFrame(numeric_rows, columns=["column", "count", "mean", "std", "min", "q1", "median", "q3", "max"]),
        "categorical_summary": pd.DataFrame(category_rows, columns=["column", "value", "count", "share_of_observed"]),
        "outliers": pd.DataFrame(outlier_rows, columns=["column", "method", "count"]),
        "missing_patterns": pd.DataFrame(patterns, columns=["pattern", "missing_columns", "n_rows", "share"]),
        "relationships": pd.DataFrame(relationships, columns=["column_a", "column_b", "statistic", "value", "n_pairwise_complete"]),
    }
    if width and width <= 30:
        joint = (miss.astype(float).T @ miss.astype(float)) / n if n else miss.astype(float).T @ miss.astype(float)
        joint.index.name = "column"
        tables["co_missingness"] = joint.reset_index()

    return ProfileResult(tables=tables, summary={
        "n_rows": int(n), "n_columns": int(width),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_column_names": duplicates,
        "missing_cells": int(miss.to_numpy(dtype=int).sum()) if width else 0,
        "missing_rate": missing_rate,
        "numeric_columns": numeric_names,
        "categorical_columns": list(categorical),
        "note": "Suggested types, outliers, and missing tokens require review; no input values were changed.",
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categorical series (pairwise complete)."""
    both = pd.concat([left, right], axis=1).dropna()
    if len(both) < 2:
        return float("nan")
    table = pd.crosstab(both.iloc[:, 0], both.iloc[:, 1])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)
    r, k = observed.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denominator)) if denominator > 0 else 0.0


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Share of a numeric column's variance explained by a categorical grouping."""
    both = pd.concat([values, groups], axis=1).dropna()
    if len(both) < 3:
        return float("nan")
    y = both.iloc[:, 0].astype(float)
    g = both.iloc[:, 1]
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    means = y.groupby(g, observed=True).transform("mean")
    ss_between = float(((means - grand) ** 2).sum())
    return ss_between / ss_total


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def profile_records(frame: pd.DataFrame, schema: Any, max_patterns: int = 15) -> ProfileResult:
    """Profile a typed frame. All model columns present in ``frame`` are used."""
    model_columns = [c for c in schema.model_columns if c in frame.columns]
    numeric = [c for c in schema.numeric_model_columns if c in frame.columns]
    categorical = [c for c in schema.categorical_model_columns if c in frame.columns]
    declared = [c for c in schema.column_names if c in frame.columns]
    n = len(frame)
    tables: dict[str, pd.DataFrame] = {}

    # --- missingness per column ------------------------------------------------
    rows = []
    for name in declared:
        spec = schema.column(name)
        missing = int(frame[name].isna().sum())
        rows.append(
            {
                "column": name,
                "role": spec.role,
                "dtype": spec.dtype,
                "n_rows": n,
                "n_observed": n - missing,
                "n_missing": missing,
                "missing_rate": missing / n if n else np.nan,
            }
        )
    tables["missingness"] = pd.DataFrame(rows)

    # --- co-missingness (joint missing rate among model columns) -------------------
    miss = frame[model_columns].isna().astype(float)
    joint = (miss.T @ miss) / n if n else miss.T @ miss
    co = joint.round(4)
    co.index.name = "column"
    tables["co_missingness"] = co.reset_index()

    # --- missingness patterns -----------------------------------------------------------
    pattern = miss.astype(int).astype(str).agg("".join, axis=1)
    counts = pattern.value_counts().head(max_patterns)
    pattern_rows = []
    for key, count in counts.items():
        missing_cols = [c for c, flag in zip(model_columns, key) if flag == "1"]
        pattern_rows.append(
            {
                "pattern": key,
                "missing_columns": ", ".join(missing_cols) if missing_cols else "(complete)",
                "n_missing_columns": len(missing_cols),
                "n_rows": int(count),
                "share": count / n if n else np.nan,
            }
        )
    tables["missing_patterns"] = pd.DataFrame(
        pattern_rows, columns=["pattern", "missing_columns", "n_missing_columns", "n_rows", "share"]
    )
    tables["missing_patterns"].attrs["columns_order"] = model_columns

    # --- numeric summary ---------------------------------------------------------------------
    num_rows = []
    for name in numeric:
        s = pd.to_numeric(frame[name], errors="coerce").dropna()
        if s.empty:
            num_rows.append({"column": name, "n_observed": 0})
            continue
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        num_rows.append(
            {
                "column": name,
                "n_observed": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                "min": float(s.min()),
                "p05": float(q[0.05]),
                "p25": float(q[0.25]),
                "median": float(q[0.5]),
                "p75": float(q[0.75]),
                "p95": float(q[0.95]),
                "max": float(s.max()),
                "iqr": float(q[0.75] - q[0.25]),
                "skew": float(s.skew()) if len(s) > 2 else np.nan,
            }
        )
    tables["numeric_summary"] = pd.DataFrame(num_rows)

    # --- categorical summary -----------------------------------------------------------------
    cat_rows = []
    for name in categorical:
        s = frame[name]
        observed = s.dropna()
        counts = observed.value_counts()
        for value, count in counts.items():
            cat_rows.append(
                {
                    "column": name,
                    "value": value,
                    "n": int(count),
                    "share_of_observed": count / len(observed) if len(observed) else np.nan,
                    "n_classes": int(len(counts)),
                }
            )
    tables["categorical_summary"] = pd.DataFrame(cat_rows, columns=["column", "value", "n", "share_of_observed", "n_classes"])

    # --- missingness of targets by other attributes (MAR evidence) -------------------------------
    mar_rows = []
    for target in [t for t in schema.targets if t in frame.columns]:
        target_missing = frame[target].isna()
        overall = float(target_missing.mean()) if n else np.nan
        for name in categorical:
            if name == target:
                continue
            groups = frame[name]
            for value, idx in groups.dropna().groupby(groups.dropna(), observed=True).groups.items():
                rate = float(target_missing.loc[idx].mean())
                mar_rows.append(
                    {
                        "target": target,
                        "by_column": name,
                        "by_value": value,
                        "n_rows": int(len(idx)),
                        "target_missing_rate": rate,
                        "overall_missing_rate": overall,
                        "ratio_to_overall": rate / overall if overall else np.nan,
                    }
                )
        for name in numeric:
            if name == target:
                continue
            values = pd.to_numeric(frame[name], errors="coerce")
            observed_mask = values.notna()
            if observed_mask.sum() < 20:
                continue
            bins = pd.qcut(values[observed_mask], q=4, duplicates="drop")
            for value, idx in bins.groupby(bins, observed=True).groups.items():
                rate = float(target_missing.loc[idx].mean())
                mar_rows.append(
                    {
                        "target": target,
                        "by_column": name,
                        "by_value": str(value),
                        "n_rows": int(len(idx)),
                        "target_missing_rate": rate,
                        "overall_missing_rate": overall,
                        "ratio_to_overall": rate / overall if overall else np.nan,
                    }
                )
    tables["missingness_by_attribute"] = pd.DataFrame(
        mar_rows,
        columns=["target", "by_column", "by_value", "n_rows", "target_missing_rate", "overall_missing_rate", "ratio_to_overall"],
    )

    # --- relationships between model attributes ------------------------------------------------------
    rel_rows = []
    for i, a in enumerate(model_columns):
        for b in model_columns[i + 1 :]:
            a_num, b_num = a in numeric, b in numeric
            both = frame[[a, b]].dropna()
            if a_num and b_num:
                stat, name_ = (float(both[a].astype(float).corr(both[b].astype(float), method="spearman")) if len(both) > 2 else np.nan), "spearman_rho"
            elif not a_num and not b_num:
                stat, name_ = cramers_v(both[a], both[b]), "cramers_v"
            else:
                num_col, cat_col = (a, b) if a_num else (b, a)
                stat, name_ = eta_squared(both[num_col].astype(float), both[cat_col]), "eta_squared"
            rel_rows.append({"column_a": a, "column_b": b, "statistic": name_, "value": stat, "n_pairwise_complete": int(len(both))})
    tables["relationships"] = pd.DataFrame(rel_rows, columns=["column_a", "column_b", "statistic", "value", "n_pairwise_complete"])

    # --- numeric by category (for the report's tables) -------------------------------------------------
    grouped_rows = []
    for num_col in numeric:
        for cat_col in categorical:
            both = frame[[num_col, cat_col]].dropna()
            if both.empty:
                continue
            agg = both.groupby(cat_col, observed=True)[num_col].agg(["count", "mean", "median", "std"]).reset_index()
            for _, r in agg.iterrows():
                grouped_rows.append(
                    {
                        "numeric_column": num_col,
                        "by_column": cat_col,
                        "by_value": r[cat_col],
                        "n": int(r["count"]),
                        "mean": float(r["mean"]),
                        "median": float(r["median"]),
                        "std": float(r["std"]) if pd.notna(r["std"]) else np.nan,
                    }
                )
    tables["numeric_by_category"] = pd.DataFrame(
        grouped_rows, columns=["numeric_column", "by_column", "by_value", "n", "mean", "median", "std"]
    )

    complete_rows = int((~miss.any(axis=1)).sum()) if model_columns else n
    summary = {
        "n_rows": int(n),
        "n_model_columns": len(model_columns),
        "complete_model_rows": complete_rows,
        "complete_model_rows_share": complete_rows / n if n else np.nan,
        "overall_model_missing_rate": float(miss.to_numpy().mean()) if n and model_columns else np.nan,
        "targets": list(schema.targets),
        "target_missing_rates": {t: float(frame[t].isna().mean()) for t in schema.targets if t in frame.columns},
    }
    return ProfileResult(tables=tables, summary=summary)
