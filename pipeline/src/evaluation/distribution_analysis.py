"""Distribution comparison: imputed cells vs observed cells.

An imputer that returns the median for every gap keeps the mean right and
destroys the spread; a classifier that always predicts the majority class
inflates it. These diagnostics make such effects visible by comparing the
distribution of the imputed cells with the distribution of the observed cells
of the same attribute, and the completed column with the observed column.

They are diagnostics, not pass/fail tests: when values are missing at random
the imputed cells *should* look like the observed ones, but when missingness
depends on other attributes a difference can be correct.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

DISTRIBUTION_COLUMNS = (
    "variable",
    "variable_type",
    "n_observed",
    "n_imputed",
    "observed_mean",
    "imputed_mean",
    "observed_std",
    "imputed_std",
    "std_ratio",
    "wasserstein_1",
    "ks_statistic",
    "ks_pvalue",
    "total_variation",
    "majority_share_observed",
    "majority_share_imputed",
    "completed_vs_observed_w1",
    "completed_vs_observed_tv",
)


def wasserstein_1(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.wasserstein_distance(a, b)) if len(a) and len(b) else np.nan


def total_variation(a: pd.Series, b: pd.Series) -> float:
    p = a.value_counts(normalize=True)
    q = b.value_counts(normalize=True)
    classes = p.index.union(q.index)
    return float(0.5 * np.abs(p.reindex(classes, fill_value=0.0) - q.reindex(classes, fill_value=0.0)).sum())


def compare_distributions(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """One row per model column comparing observed, imputed and completed values."""
    rows = []
    for column in numeric_columns:
        was_missing = original[column].isna().to_numpy()
        observed = pd.to_numeric(original.loc[~was_missing, column], errors="coerce").dropna().to_numpy(dtype=float)
        filled = pd.to_numeric(imputed.loc[was_missing, column], errors="coerce").dropna().to_numpy(dtype=float)
        completed = pd.to_numeric(imputed[column], errors="coerce").dropna().to_numpy(dtype=float)
        row = {"variable": column, "variable_type": "numeric", "n_observed": int(len(observed)), "n_imputed": int(len(filled))}
        if len(observed) and len(filled):
            ks = stats.ks_2samp(observed, filled) if len(observed) > 1 and len(filled) > 1 else None
            obs_std = float(np.std(observed, ddof=1)) if len(observed) > 1 else np.nan
            imp_std = float(np.std(filled, ddof=1)) if len(filled) > 1 else np.nan
            row.update(
                {
                    "observed_mean": float(np.mean(observed)),
                    "imputed_mean": float(np.mean(filled)),
                    "observed_std": obs_std,
                    "imputed_std": imp_std,
                    "std_ratio": imp_std / obs_std if obs_std and np.isfinite(obs_std) and obs_std > 0 else np.nan,
                    "wasserstein_1": wasserstein_1(observed, filled),
                    "ks_statistic": float(ks.statistic) if ks is not None else np.nan,
                    "ks_pvalue": float(ks.pvalue) if ks is not None else np.nan,
                    "completed_vs_observed_w1": wasserstein_1(observed, completed),
                }
            )
        rows.append(row)

    for column in categorical_columns:
        was_missing = original[column].isna().to_numpy()
        observed = original.loc[~was_missing, column].dropna().astype(str)
        filled = imputed.loc[was_missing, column].dropna().astype(str)
        completed = imputed[column].dropna().astype(str)
        row = {"variable": column, "variable_type": "categorical", "n_observed": int(len(observed)), "n_imputed": int(len(filled))}
        if len(observed) and len(filled):
            row.update(
                {
                    "total_variation": total_variation(observed, filled),
                    "majority_share_observed": float(observed.value_counts(normalize=True).iloc[0]),
                    "majority_share_imputed": float(filled.value_counts(normalize=True).iloc[0]),
                    "completed_vs_observed_tv": total_variation(observed, completed),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=list(DISTRIBUTION_COLUMNS))


def joint_distribution_shift(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    numeric_column: str,
    categorical_column: str,
    bin_width: float = 10.0,
) -> pd.DataFrame:
    """Joint shares of (binned numeric, categorical) for rows complete in both
    columns before imputation vs rows where at least one was imputed.

    Returns a long table with ``bin, class, share_observed_rows, share_imputed_rows, difference``
    and the total variation distance in ``attrs["total_variation"]``.
    """
    if numeric_column not in original.columns or categorical_column not in original.columns:
        return pd.DataFrame(columns=["bin", "class", "share_observed_rows", "share_imputed_rows", "difference"])
    numeric = pd.to_numeric(imputed[numeric_column], errors="coerce")
    categorical = imputed[categorical_column].astype(object)
    either_missing = original[numeric_column].isna() | original[categorical_column].isna()
    valid = numeric.notna() & categorical.notna()
    if valid.sum() == 0:
        return pd.DataFrame(columns=["bin", "class", "share_observed_rows", "share_imputed_rows", "difference"])
    lower = np.floor(numeric[valid].min() / bin_width) * bin_width
    bins = (np.floor((numeric - lower) / bin_width) * bin_width + lower).astype("Int64")
    table = pd.DataFrame({"bin": bins, "class": categorical.astype(str), "imputed_rows": either_missing})[valid]
    observed_rows = table[~table["imputed_rows"]]
    imputed_rows = table[table["imputed_rows"]]
    p = observed_rows.groupby(["bin", "class"]).size() / max(len(observed_rows), 1)
    q = imputed_rows.groupby(["bin", "class"]).size() / max(len(imputed_rows), 1)
    index = p.index.union(q.index)
    p, q = p.reindex(index, fill_value=0.0), q.reindex(index, fill_value=0.0)
    out = pd.DataFrame(
        {
            "bin": [f"{int(b)}-{int(b) + int(bin_width) - 1}" for b, _ in index],
            "class": [c for _, c in index],
            "share_observed_rows": p.to_numpy(),
            "share_imputed_rows": q.to_numpy(),
            "difference": (q - p).to_numpy(),
        }
    )
    out.attrs["total_variation"] = float(0.5 * np.abs(q - p).sum()) if len(imputed_rows) and len(observed_rows) else np.nan
    out.attrs["n_observed_rows"] = int(len(observed_rows))
    out.attrs["n_imputed_rows"] = int(len(imputed_rows))
    return out
