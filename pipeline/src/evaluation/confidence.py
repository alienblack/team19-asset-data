"""Confidence ranking of imputed cells.

The predictive method reports, for every imputed cell, how decisive its forest
was: for categorical targets the winning vote share (``max_proba``), the gap
to the runner-up (``margin``) and the vote entropy; for numeric targets the
spread of the individual tree predictions (``tree_std``, ``tree_p10``,
``tree_p90``). These are *rankings*, not calibrated probabilities, so this
module converts them into a within-column percentile rank in [0, 1] and lists
the least confident cells for review. Cells filled by a fallback statistic
(too few observed values to fit a forest) are always listed.

Methods without confidence fields (median/mode, the original MissForest)
produce an empty list and a note saying why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ConfidenceResult:
    ranked: pd.DataFrame
    low_confidence: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


def rank_confidence(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Add ``confidence_signal`` and ``confidence_rank`` (percentile within column)."""
    out = diagnostics.copy()
    if out.empty:
        out["confidence_signal"] = pd.Series(dtype=float)
        out["confidence_rank"] = pd.Series(dtype=float)
        return out
    signal = pd.Series(np.nan, index=out.index, dtype=float)
    is_cat = out["kind"].astype(str) == "categorical"
    if "margin" in out.columns:
        signal[is_cat] = pd.to_numeric(out.loc[is_cat, "margin"], errors="coerce")
    if "max_proba" in out.columns:
        missing_margin = is_cat & signal.isna()
        signal[missing_margin] = pd.to_numeric(out.loc[missing_margin, "max_proba"], errors="coerce")
    if "tree_std" in out.columns:
        spread = pd.to_numeric(out.loc[~is_cat, "tree_std"], errors="coerce")
        signal[~is_cat] = -spread  # smaller spread = more confident
    out["confidence_signal"] = signal
    out["confidence_rank"] = (
        out.groupby("column", sort=False)["confidence_signal"].rank(method="average", pct=True)
    )
    return out


def low_confidence_cells(
    diagnostics: pd.DataFrame,
    fraction: float = 0.10,
    min_rows: int = 20,
) -> ConfidenceResult:
    """Return the ranked diagnostics and the lowest-confidence cells per column.

    For each column with at least ``min_rows`` ranked cells, the bottom
    ``fraction`` by ``confidence_rank`` are listed; columns with fewer cells
    list every cell whose rank is below ``fraction``. Fallback-filled cells
    (``prediction_source`` other than ``random_forest``) are included
    regardless of rank.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between zero and one.")
    ranked = rank_confidence(diagnostics)
    if ranked.empty or ranked["confidence_signal"].notna().sum() == 0:
        note = (
            "The imputation method does not report per-cell confidence; "
            "no low-confidence list can be produced."
        )
        return ConfidenceResult(
            ranked=ranked,
            low_confidence=ranked.iloc[0:0],
            summary={"n_cells": int(len(ranked)), "n_listed": 0, "note": note},
        )
    parts = []
    for column, group in ranked.groupby("column", sort=False):
        scored = group[group["confidence_rank"].notna()]
        if len(scored) >= min_rows:
            n_list = max(1, int(np.ceil(fraction * len(scored))))
            lowest = scored.nsmallest(n_list, "confidence_rank", keep="all")
        else:
            lowest = scored[scored["confidence_rank"] <= fraction]
        fallback = group[group["prediction_source"].astype(str) != "random_forest"] if "prediction_source" in group else group.iloc[0:0]
        combined = pd.concat([lowest, fallback])
        parts.append(combined[~combined.index.duplicated(keep="first")])
    low = pd.concat(parts) if parts else ranked.iloc[0:0]
    low = low.sort_values(["column", "confidence_rank"], kind="mergesort")
    summary = {
        "n_cells": int(len(ranked)),
        "n_scored": int(ranked["confidence_rank"].notna().sum()),
        "n_listed": int(len(low)),
        "fraction": fraction,
        "listed_by_column": {k: int(v) for k, v in low["column"].value_counts().sort_index().items()},
        "fallback_cells": int((ranked["prediction_source"].astype(str) != "random_forest").sum()) if "prediction_source" in ranked else 0,
    }
    return ConfidenceResult(ranked=ranked, low_confidence=low, summary=summary)
