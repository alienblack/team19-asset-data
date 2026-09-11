"""UC-S2-09: preliminary downstream ranking stability validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from .utils import ValidationInputError, infer_target_type
except ImportError:
    from utils import ValidationInputError, infer_target_type


def calculate_risk_proxy(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Return target values as a prototype ranking proxy.

    This is a placeholder/prototype and is not an official SAPN PoF model.
    Replace or wrap this function only when a documented project model exists.
    """
    return df[target_column]


def calculate_ranking_stability(
    datasets: dict[str, pd.DataFrame], target_column: str
) -> pd.DataFrame:
    """Compare reference and imputed target rankings using Spearman correlation."""
    if infer_target_type(datasets["reference"][target_column]) != "numeric":
        raise ValidationInputError(
            "Ranking stability requires a numeric target. Categorical distribution and metrics outputs are still supported."
        )
    reference = calculate_risk_proxy(datasets["reference"], target_column)
    rows = []
    for key, label in (("baseline", "Baseline"), ("alternative", "Alternative")):
        candidate = calculate_risk_proxy(datasets[key], target_column)
        usable = reference.notna() & candidate.notna()
        n_records = int(usable.sum())
        correlation = np.nan
        if n_records >= 2:
            correlation = spearmanr(reference.loc[usable], candidate.loc[usable]).statistic
        rows.append(
            {
                "comparison": f"Reference vs {label}",
                "spearman_correlation": correlation,
                "n_records": n_records,
            }
        )
    return pd.DataFrame(rows)

