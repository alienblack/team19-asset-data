"""Validation-only tuning and diagnostics for PredictiveMissForest.

Supports two validation inputs:

* Already-incomplete SAPN-like validation records: the tuner temporarily hides
  a fixed fraction of still-observed year and material values, once before the
  hyperparameter search, and retains them as validation truth.
* Explicit benchmark masks: callers may supply a pre-masked ``X_tune`` plus
  aligned ``tune_truth`` and ``tune_mask``; the tuner never adds another mask.

``X_fit`` is always a separate fitting partition and may contain existing
missing values. The tuner does not use validation truth to fit forests; it is
read only after the candidate model has transformed the masked validation data.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluation import evaluate_imputation
from .predictive_missforest import (
    PredictiveMissForest,
    PredictiveMissForestConfig,
    validate_predictive_config,
)


DEFAULT_GRID: dict[str, Sequence[Any]] = {
    "max_depth": (4, 8, 16, None),
    "min_samples_leaf": (1, 2, 10),
}


def expand_grid(grid: Mapping[str, Sequence[Any]] | None = None) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid as a list of override dicts."""
    grid = dict(grid or DEFAULT_GRID)
    keys = list(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def make_mcar_mask(
    X: pd.DataFrame,
    columns: Sequence[str],
    rate: float,
    seed: int,
) -> pd.DataFrame:
    """Development-only MCAR mask over *observed* cells of ``columns``.

    The real benchmark masks come from the scenario engine; this helper lets
    the tuner and tests run without it.
    """
    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    for column in columns:
        observed = X[column].notna().to_numpy()
        draw = rng.random(len(X)) < rate
        mask[column] = observed & draw
    return mask


def hide_cells(X: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with masked cells set to NaN."""
    hidden = X.copy(deep=True)
    for column in mask.columns:
        if column in hidden.columns and mask[column].any():
            hidden.loc[mask[column].to_numpy(), column] = np.nan
    return hidden


def prepare_tuning_holdout(
    X_validation: pd.DataFrame,
    columns: Sequence[str],
    rate: float = 0.10,
    seed: int = 219,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hide a fixed fraction of CURRENTLY OBSERVED validation values.

    ``X_validation`` may already have natural or scenario missingness. Those
    existing gaps are not scored: only additional, deliberately hidden cells
    are True in the returned boolean mask.

    Returns ``(masked_validation, validation_truth, holdout_mask)``.
    The input is never mutated; the same holdout is reused across all grid
    candidates. This helper is for validation records only, not training rows.
    """
    if not isinstance(X_validation, pd.DataFrame):
        raise TypeError("X_validation must be a DataFrame.")
    if not X_validation.index.is_unique:
        raise ValueError("Validation record index must uniquely identify rows.")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not np.isfinite(rate)
        or not 0 < rate < 1
    ):
        raise ValueError("Tuning mask rate must be strictly between zero and one.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Tuning mask seed must be a nonnegative integer.")

    targets = tuple(columns)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("Tuning target columns must be nonempty and unique.")
    unknown = sorted(set(targets) - set(X_validation.columns))
    if unknown:
        raise ValueError(f"Unknown tuning target columns: {unknown}.")

    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(False, index=X_validation.index, columns=X_validation.columns)
    for column in targets:
        observed_indices = np.flatnonzero(X_validation[column].notna().to_numpy())
        if len(observed_indices) < 2:
            raise ValueError(
                f"Cannot tune '{column}': at least two known validation values "
                "are needed to create a holdout."
            )
        n_hide = max(1, int(round(rate * len(observed_indices))))
        n_hide = min(n_hide, len(observed_indices) - 1)
        selected = rng.choice(observed_indices, size=n_hide, replace=False)
        mask.iloc[selected, mask.columns.get_loc(column)] = True

    truth = X_validation.copy(deep=True)
    masked = hide_cells(X_validation, mask)
    return masked, truth, mask


@dataclass(frozen=True)
class TuningObjective:
    """Lower-is-better selection objective proposed in Plan v2.0.

    ``score = w * (year_MAE / year_scale) + (1 - w) * (1 - material_macro_F1)``

    ``year_scale`` is the IQR of originally observed fitting years. The rank
    average across configurations is reported alongside because the two
    terms live on different scales (see the review notes).
    """

    year_column: str = "installation_year"
    material_column: str = "material"
    year_weight: float = 0.5

    def __call__(self, year_mae: float, year_scale: float, macro_f1: float) -> float:
        if not np.isfinite(year_scale) or year_scale <= 0:
            raise ValueError("year_scale must be a positive finite number.")
        return (
            self.year_weight * (year_mae / year_scale)
            + (1.0 - self.year_weight) * (1.0 - macro_f1)
        )


def _year_scale(X_fit: pd.DataFrame, year_column: str) -> float:
    years = pd.to_numeric(X_fit[year_column], errors="coerce").dropna()
    q75, q25 = np.percentile(years, [75, 25])
    return float(q75 - q25)


def _distribution_diagnostics(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
    year_column: str,
    material_column: str,
) -> dict[str, float]:
    """Cheap population-shape diagnostics on hidden cells only.

    These are reported, not optimised, so the tuner can show what the
    accuracy objective did to distribution shape.
    """
    out: dict[str, float] = {}

    positions = mask[year_column].to_numpy(dtype=bool)
    if positions.any():
        y_true = truth.loc[positions, year_column].astype(float).to_numpy()
        y_pred = imputed.loc[positions, year_column].astype(float).to_numpy()
        a, b = np.sort(y_true), np.sort(y_pred)
        out["year_w1"] = float(np.mean(np.abs(a - b)))
        out["year_signed_bias"] = float(np.mean(y_pred - y_true))
        sd_true = float(np.std(y_true))
        # Standard-deviation ratio, *not* a variance ratio.
        out["year_std_ratio"] = (
            float(np.std(y_pred) / sd_true) if sd_true > 0 else np.nan
        )
        out["year_variance_ratio"] = (
            float(np.var(y_pred) / np.var(y_true)) if sd_true > 0 else np.nan
        )

    positions = mask[material_column].to_numpy(dtype=bool)
    if positions.any():
        c_true = truth.loc[positions, material_column].astype(str)
        c_pred = imputed.loc[positions, material_column].astype(str)
        p = c_true.value_counts(normalize=True)
        q = c_pred.value_counts(normalize=True)
        classes = p.index.union(q.index)
        out["material_tv"] = float(
            0.5 * np.abs(p.reindex(classes, fill_value=0.0)
                         - q.reindex(classes, fill_value=0.0)).sum()
        )

    return out


def validate_tuning_alignment(
    X_tune: pd.DataFrame,
    tune_truth: pd.DataFrame,
    tune_mask: pd.DataFrame,
    model_columns: Sequence[str],
) -> None:
    """Reject positional misalignment and accidentally exposed hidden truth."""
    for name, other in (("tune_truth", tune_truth), ("tune_mask", tune_mask)):
        if not isinstance(other, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame.")
        if not X_tune.index.equals(other.index):
            raise ValueError(f"{name} index is not aligned with X_tune.")
        if not X_tune.columns.equals(other.columns):
            raise ValueError(f"{name} columns are not aligned with X_tune.")
    if not X_tune.index.is_unique:
        raise ValueError("Tuning record index must uniquely identify rows.")
    if not all(pd.api.types.is_bool_dtype(tune_mask[col]) for col in model_columns):
        raise TypeError("tune_mask must contain boolean flags for model columns.")
    for col in model_columns:
        positions = tune_mask[col].to_numpy(dtype=bool)
        if positions.any():
            if not X_tune[col].isna().to_numpy()[positions].all():
                raise ValueError(f"Hidden tune cells are visible in X_tune['{col}'].")
            if not tune_truth[col].notna().to_numpy()[positions].all():
                raise ValueError(f"Hidden tune cells lack truth in '{col}'.")


def score_configuration(
    base: PredictiveMissForest,
    overrides: Mapping[str, Any],
    X_fit: pd.DataFrame,
    X_tune: pd.DataFrame,
    tune_truth: pd.DataFrame,
    tune_mask: pd.DataFrame,
    objective: TuningObjective,
) -> dict[str, Any]:
    """Fit one configuration on X_fit, impute X_tune, score hidden cells."""

    validate_tuning_alignment(X_tune, tune_truth, tune_mask, base.model_columns)
    config = base.config.replace(**overrides)
    model = PredictiveMissForest(
        numeric_columns=base.numeric_columns,
        categorical_columns=base.categorical_columns,
        config=config,
        constraints=base.constraints,
    )

    started = perf_counter()
    model.fit(X_fit)
    fit_seconds = perf_counter() - started

    started = perf_counter()
    imputed = model.transform(X_tune)
    transform_seconds = perf_counter() - started

    metrics = evaluate_imputation(
        truth=tune_truth,
        imputed=imputed,
        mask=tune_mask,
        numeric_columns=base.numeric_columns,
        categorical_columns=base.categorical_columns,
    ).set_index("variable")

    year_mae = float(metrics.loc[objective.year_column, "mae"])
    macro_f1 = float(metrics.loc[objective.material_column, "macro_f1"])
    year_scale = _year_scale(X_fit, objective.year_column)

    row: dict[str, Any] = dict(overrides)
    row.update(
        {
            "year_mae": year_mae,
            "year_rmse": float(metrics.loc[objective.year_column, "rmse"]),
            "material_macro_f1": macro_f1,
            "material_balanced_accuracy": float(
                metrics.loc[objective.material_column, "balanced_accuracy"]
            ),
            "year_scale_iqr": year_scale,
            "objective": objective(year_mae, year_scale, macro_f1),
            "fit_seconds": fit_seconds,
            "transform_seconds": transform_seconds,
            "total_tree_nodes": model.fit_report_["total_tree_nodes"],
        }
    )
    row.update(
        _distribution_diagnostics(
            tune_truth, imputed, tune_mask,
            objective.year_column, objective.material_column,
        )
    )
    return row


def tune_predictive_missforest(
    base: PredictiveMissForest,
    X_fit: pd.DataFrame,
    X_tune: pd.DataFrame,
    tune_truth: pd.DataFrame | None = None,
    tune_mask: pd.DataFrame | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
    objective: TuningObjective | None = None,
    *,
    mask_rate: float = 0.10,
    mask_seed: int = 219,
) -> tuple[pd.DataFrame, PredictiveMissForestConfig]:
    """Validation-only grid search on already-incomplete records.

    If ``tune_truth`` and ``tune_mask`` are both absent, ``X_tune`` is treated
    as the ORIGINAL incomplete validation partition. Known values of the two
    objective targets are then hidden internally once and used for scoring.

    If both are present, ``X_tune`` must already be masked and no additional
    masking occurs (backward-compatible with the original runner).

    Returns the full score table and best hyperparameter configuration.
    Never pass the independent final-test partition here.
    """
    objective = objective or TuningObjective()

    if (tune_truth is None) != (tune_mask is None):
        raise ValueError("Supply both tune_truth and tune_mask, or neither.")
    if tune_truth is None:
        X_tune, tune_truth, tune_mask = prepare_tuning_holdout(
            X_tune,
            columns=[objective.year_column, objective.material_column],
            rate=mask_rate,
            seed=mask_seed,
        )
    validate_tuning_alignment(X_tune, tune_truth, tune_mask, base.model_columns)
    for column in (objective.year_column, objective.material_column):
        if column not in tune_mask or not tune_mask[column].any():
            raise ValueError(
                f"No deliberately masked validation cells for '{column}'. "
                "The combined tuning objective requires both targets."
            )

    candidates = expand_grid(grid)
    rows = []
    for grid_index, overrides in enumerate(candidates):
        row = score_configuration(
            base, overrides, X_fit, X_tune, tune_truth, tune_mask, objective
        )
        row["grid_index"] = grid_index
        rows.append(row)

    results = pd.DataFrame(rows)
    results["rank_average"] = (
        results["year_mae"].rank(method="average")
        + (-results["material_macro_f1"]).rank(method="average")
    ) / 2.0
    results = results.sort_values(
        ["objective", "rank_average", "grid_index"], kind="mergesort"
    ).reset_index(drop=True)

    # Take the winning overrides from the original grid objects so that
    # types survive (a DataFrame would turn ``None`` into NaN and ints into
    # floats, which the config validator rightly rejects).
    best_overrides = candidates[int(results.iloc[0]["grid_index"])]
    best_config = base.config.replace(**best_overrides)
    validate_predictive_config(best_config)
    return results, best_config


def column_permutation_importance(
    model: PredictiveMissForest,
    X_tune: pd.DataFrame,
    tune_truth: pd.DataFrame,
    tune_mask: pd.DataFrame,
    target: str,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Column-level permutation importance measured through the whole imputer.

    For each predictor column, the observed values of that column in
    ``X_tune`` are permuted among the rows where it is observed (the
    missingness pattern is kept), the fitted imputer is replayed, and the
    change in the target's hidden-cell metric is recorded (MAE increase for a
    numeric target, macro-F1 decrease for a categorical one).

    This is the defensible alternative to impurity importance on one-hot
    dummies: it operates on original columns, measures held-out effect, and
    accounts for the iterative interaction between predictors.
    Research diagnostic — not required for correctness.
    """
    rng = np.random.default_rng(seed)
    numeric = model.numeric_columns
    categorical = model.categorical_columns
    is_numeric_target = target in numeric
    metric = "mae" if is_numeric_target else "macro_f1"

    def score(frame: pd.DataFrame) -> float:
        imputed = model.transform(frame)
        table = evaluate_imputation(
            tune_truth, imputed, tune_mask, numeric, categorical
        ).set_index("variable")
        return float(table.loc[target, metric])

    baseline = score(X_tune)
    rows = []
    for column in model.features_[target]:
        deltas = []
        for _ in range(n_repeats):
            permuted = X_tune.copy(deep=True)
            observed = permuted[column].notna().to_numpy()
            values = permuted[column].to_numpy(dtype=object, copy=True)
            values[observed] = rng.permutation(values[observed])
            permuted[column] = pd.Series(values, index=permuted.index)
            if column in numeric:
                permuted[column] = pd.to_numeric(permuted[column])
            delta = score(permuted) - baseline
            deltas.append(delta if is_numeric_target else -delta)
        rows.append(
            {
                "predictor": column,
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas)),
                "baseline_" + metric: baseline,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)
