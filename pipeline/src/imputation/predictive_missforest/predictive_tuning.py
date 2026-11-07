"""Validation-only tuning and diagnostics for PredictiveMissForest.

Hyperparameters are chosen on a validation partition only: forests are fitted
on ``X_fit`` and scored on cells deliberately hidden in ``X_tune``. Because a
single 10% holdout of a small validation partition is noisy, the search can
draw several independent holdouts (``n_repeats``) and select on the mean
objective, preferring the simplest configuration (fewest tree nodes) among
those within one standard deviation of the best. A training median/mode
baseline scored on the identical hidden cells is reported as the noise floor.

Supports two validation inputs:

* Already-incomplete SAPN-like validation records: the tuner temporarily hides
  a fixed fraction of currently observed target values once before the search.
  In ``year_material`` mode, it hides year and material; in ``all_attributes``
  mode, it hides every modelled attribute.
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
import warnings

import numpy as np
import pandas as pd

from ...evaluation.holdout import hide_cells as _hide_cells
from ...evaluation.holdout import prepare_holdout, validate_alignment
from ...evaluation.metrics import evaluate_imputation
from .predictive_missforest import (
    PredictiveMissForest,
    PredictiveMissForestConfig,
    validate_predictive_config,
)
from ..median_mode.simple_imputer import deterministic_mode


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
    return _hide_cells(X, mask)


def prepare_tuning_holdout(
    X_validation: pd.DataFrame,
    columns: Sequence[str],
    rate: float = 0.10,
    seed: int = 219,
    *,
    stratify_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hide a fixed fraction of CURRENTLY OBSERVED validation values.

    ``X_validation`` may already have natural or scenario missingness. Those
    existing gaps are not scored: only additional, deliberately hidden cells
    are True in the returned boolean mask. Categorical columns listed in
    ``stratify_columns`` are hidden class by class so minority classes are
    represented.

    Returns ``(masked_validation, validation_truth, holdout_mask)``.
    The input is never mutated; the same holdout is reused across all grid
    candidates. This helper is for validation records only, not training rows.
    """
    if not isinstance(X_validation, pd.DataFrame):
        raise TypeError("X_validation must be a DataFrame.")
    try:
        return prepare_holdout(X_validation, columns, rate=rate, seed=seed, stratify_columns=stratify_columns)
    except ValueError as exc:
        message = str(exc)
        if "rate" in message:
            raise ValueError("Tuning mask rate must be strictly between zero and one.") from exc
        if "seed" in message:
            raise ValueError("Tuning mask seed must be a nonnegative integer.") from exc
        if "columns" in message and "Unknown" in message:
            raise ValueError(message.replace("holdout", "tuning target")) from exc
        if "non-empty and unique" in message:
            raise ValueError("Tuning target columns must be nonempty and unique.") from exc
        if "at least two observed" in message:
            raise ValueError(message.replace("hold out", "tune").replace("observed values", "known validation values")) from exc
        raise


@dataclass(frozen=True)
class TuningObjective:
    """Validation-only selection objective; lower scores are better.

    ``mode='year_material'`` preserves the original objective:
        year_weight * year_MAE / training_year_IQR
        + (1 - year_weight) * (1 - material_macro_F1).

    ``mode='all_attributes'`` gives equal TOTAL weight to the numerical and
    categorical groups (controlled by ``numeric_weight``). Within each group,
    each attribute receives equal weight:
        numeric_weight * mean(MAE_j / training_IQR_j)
        + (1 - numeric_weight) * mean(1 - macro_F1_k).

    Macro-F1 is calculated using the deliberately hidden validation targets;
    diagnostics report their class counts so single-class holdouts are visible.
    """

    year_column: str = "installation_year"
    material_column: str = "material"
    year_weight: float = 0.5
    mode: str = "year_material"
    numeric_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in {"year_material", "all_attributes"}:
            raise ValueError("mode must be 'year_material' or 'all_attributes'.")
        for name, weight in (("year_weight", self.year_weight),
                             ("numeric_weight", self.numeric_weight)):
            if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                    or not np.isfinite(weight) or not 0 <= weight <= 1):
                raise ValueError(f"{name} must be a number between zero and one.")

    def target_columns(self, base: PredictiveMissForest) -> tuple[str, ...]:
        """Columns that must receive an additional validation mask."""
        if self.mode == "all_attributes":
            # Hide the original two targets FIRST so their holdout positions
            # remain identical to year_material mode for the same seed.
            all_targets = (self.year_column, self.material_column,
                           *base.numeric_columns, *base.categorical_columns)
            return tuple(dict.fromkeys(all_targets))
        return (self.year_column, self.material_column)

    def __call__(self, year_mae: float, year_scale: float, macro_f1: float) -> float:
        """Backward-compatible year/material objective."""
        if not np.isfinite(year_scale) or year_scale <= 0:
            raise ValueError("year_scale must be a positive finite number.")
        return (
            self.year_weight * (year_mae / year_scale)
            + (1.0 - self.year_weight) * (1.0 - macro_f1)
        )


def _observed_iqr(X_fit: pd.DataFrame, column: str) -> float:
    observed = pd.to_numeric(X_fit[column], errors="raise").dropna()
    if observed.empty:
        raise ValueError(f"No observed fitting values to scale '{column}'.")
    q75, q25 = np.percentile(observed.to_numpy(dtype=float), [75, 25])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"Training IQR for '{column}' is zero or invalid. An all-attribute "
            "objective cannot normalise its MAE; review this variable before "
            "choosing an explicit fallback scale or excluding the variable."
        )
    return scale


def _year_scale(X_fit: pd.DataFrame, year_column: str) -> float:
    """Keep the original helper name for backward compatibility."""
    return _observed_iqr(X_fit, year_column)


def _all_attribute_score(
    metrics: pd.DataFrame,
    X_fit: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    numeric_weight: float,
) -> tuple[float, dict[str, float]]:
    """Calculate a balanced numerical/categorical objective, using TRAIN scales.

    No statistic is fitted or inferred from the validation target truth, apart
    from the post-prediction performance metrics that the tuner must score.
    """
    if not numeric_columns or not categorical_columns:
        raise ValueError("All-attribute objective requires both numeric and categorical targets.")

    components: dict[str, float] = {}
    numeric_losses = []
    for column in numeric_columns:
        scale = _observed_iqr(X_fit, column)
        mae = float(metrics.loc[column, "mae"])
        if not np.isfinite(mae):
            raise ValueError(f"Non-finite validation MAE for '{column}'.")
        normalised = mae / scale
        components[f"scale_iqr__{column}"] = scale
        components[f"normalized_mae__{column}"] = normalised
        numeric_losses.append(normalised)

    categorical_losses = []
    for column in categorical_columns:
        f1 = float(metrics.loc[column, "macro_f1"])
        if not np.isfinite(f1):
            raise ValueError(f"Non-finite validation Macro-F1 for '{column}'.")
        loss = 1.0 - f1
        components[f"classification_loss__{column}"] = loss
        categorical_losses.append(loss)

    components["numeric_group_loss"] = float(np.mean(numeric_losses))
    components["categorical_group_loss"] = float(np.mean(categorical_losses))
    score = (numeric_weight * components["numeric_group_loss"]
             + (1.0 - numeric_weight) * components["categorical_group_loss"])
    return float(score), components


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
    try:
        validate_alignment(X_tune, tune_truth, tune_mask, model_columns)
    except TypeError as exc:
        raise TypeError(str(exc).replace("must be boolean", "must contain boolean flags for model columns")) from exc
    except ValueError as exc:
        message = str(exc)
        if "not aligned" in message:
            raise ValueError(message.replace("the masked frame", "X_tune")) from exc
        if "Record index" in message:
            raise ValueError("Tuning record index must uniquely identify rows.") from exc
        if "still visible" in message:
            raise ValueError(message.replace("masked[", "X_tune[").replace("Hidden cells", "Hidden tune cells")) from exc
        if "lack truth" in message:
            raise ValueError(message.replace("Hidden cells", "Hidden tune cells")) from exc
        raise


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
    ).set_index("variable").reindex(base.model_columns)

    year_mae = float(metrics.loc[objective.year_column, "mae"])
    macro_f1 = float(metrics.loc[objective.material_column, "macro_f1"])
    year_scale = _year_scale(X_fit, objective.year_column)

    # Preserve every legacy diagnostic field, including both original targets.
    row: dict[str, Any] = dict(overrides)
    row.update(
        {
            "objective_mode": objective.mode,
            "year_mae": year_mae,
            "year_rmse": float(metrics.loc[objective.year_column, "rmse"]),
            "material_macro_f1": macro_f1,
            "material_balanced_accuracy": float(
                metrics.loc[objective.material_column, "balanced_accuracy"]
            ),
            "year_scale_iqr": year_scale,
            "fit_seconds": fit_seconds,
            "transform_seconds": transform_seconds,
            "total_tree_nodes": model.fit_report_["total_tree_nodes"],
        }
    )

    # Record the validation metrics of EVERY modelled attribute, not only the
    # two legacy objective targets. In legacy mode some are intentionally NaN
    # because those columns were not additionally masked.
    for column in base.numeric_columns:
        row[f"n_masked__{column}"] = int(tune_mask[column].sum())
        row[f"mae__{column}"] = float(metrics.loc[column, "mae"])
        row[f"rmse__{column}"] = float(metrics.loc[column, "rmse"])
    for column in base.categorical_columns:
        row[f"n_masked__{column}"] = int(tune_mask[column].sum())
        row[f"accuracy__{column}"] = float(metrics.loc[column, "accuracy"])
        row[f"balanced_accuracy__{column}"] = float(
            metrics.loc[column, "balanced_accuracy"]
        )
        row[f"macro_f1__{column}"] = float(metrics.loc[column, "macro_f1"])
        hidden = tune_truth.loc[tune_mask[column].to_numpy(dtype=bool), column]
        class_counts = hidden.value_counts()
        row[f"n_truth_classes__{column}"] = int(len(class_counts))
        row[f"min_truth_class_support__{column}"] = (
            int(class_counts.min()) if len(class_counts) else 0
        )

    if objective.mode == "year_material":
        row["objective"] = objective(year_mae, year_scale, macro_f1)
    else:
        score, components = _all_attribute_score(
            metrics, X_fit, base.numeric_columns, base.categorical_columns,
            objective.numeric_weight,
        )
        row.update(components)
        row["objective"] = score

    row.update(
        _distribution_diagnostics(
            tune_truth, imputed, tune_mask,
            objective.year_column, objective.material_column,
        )
    )
    return row


def _baseline_scores(
    base: PredictiveMissForest,
    X_fit: pd.DataFrame,
    X_tune: pd.DataFrame,
    tune_truth: pd.DataFrame,
    tune_mask: pd.DataFrame,
    objective: TuningObjective,
) -> dict[str, Any]:
    """Training median/mode scored on the identical hidden cells (noise floor)."""
    filled = X_tune.copy(deep=True)
    for column in base.numeric_columns:
        observed = pd.to_numeric(X_fit[column], errors="raise").dropna()
        filled[column] = pd.to_numeric(filled[column], errors="raise").astype(float).fillna(float(observed.median()))
    for column in base.categorical_columns:
        mode = deterministic_mode(X_fit[column].astype(object))
        values = filled[column].astype(object)
        filled[column] = values.where(values.notna(), mode)
    metrics = evaluate_imputation(
        tune_truth, filled, tune_mask, base.numeric_columns, base.categorical_columns
    ).set_index("variable").reindex(base.model_columns)
    year_mae = float(metrics.loc[objective.year_column, "mae"])
    macro_f1 = float(metrics.loc[objective.material_column, "macro_f1"])
    year_scale = _year_scale(X_fit, objective.year_column)
    row: dict[str, Any] = {
        "candidate": "median_mode_baseline",
        "objective_mode": objective.mode,
        "year_mae": year_mae,
        "year_rmse": float(metrics.loc[objective.year_column, "rmse"]),
        "material_macro_f1": macro_f1,
        "material_balanced_accuracy": float(metrics.loc[objective.material_column, "balanced_accuracy"]),
        "year_scale_iqr": year_scale,
    }
    for column in base.numeric_columns:
        row[f"mae__{column}"] = float(metrics.loc[column, "mae"])
    for column in base.categorical_columns:
        row[f"macro_f1__{column}"] = float(metrics.loc[column, "macro_f1"])
    if objective.mode == "year_material":
        row["objective"] = objective(year_mae, year_scale, macro_f1)
    else:
        score, components = _all_attribute_score(
            metrics, X_fit, base.numeric_columns, base.categorical_columns, objective.numeric_weight
        )
        row.update(components)
        row["objective"] = score
    return row


def _candidate_label(overrides: Mapping[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in overrides.items())


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
    n_repeats: int = 1,
    stratify_categorical: bool = True,
    include_baseline: bool = True,
    simplicity_tolerance: float = 1.0,
) -> tuple[pd.DataFrame, PredictiveMissForestConfig]:
    """Validation-only grid search on already-incomplete records.

    If ``tune_truth`` and ``tune_mask`` are both absent, ``X_tune`` is treated
    as the ORIGINAL incomplete validation partition and ``n_repeats``
    independent holdouts are drawn (seeds ``mask_seed, mask_seed+1, ...``).
    In ``year_material`` mode known year/material values are hidden; in
    ``all_attributes`` mode known values in EVERY modelled attribute are
    hidden. Categorical holdouts are stratified by class when
    ``stratify_categorical`` is true.

    If both are present, ``X_tune`` must already be masked, no additional
    masking occurs and ``n_repeats`` is ignored (backward-compatible with
    explicit benchmark masks).

    Selection: candidates are ranked by mean objective (lower is better).
    When ``n_repeats > 1`` and ``simplicity_tolerance > 0``, every candidate
    whose mean objective lies within ``simplicity_tolerance`` standard
    deviations (of the best candidate's repeats) of the best mean is treated
    as statistically indistinguishable, and the one with the fewest tree
    nodes is chosen.

    Returns the score table (one row per candidate; with repeats, the means
    with ``_std`` columns and the per-repeat rows in ``results.attrs["per_repeat"]``)
    and the best configuration. The training median/mode baseline on the
    same hidden cells is stored in ``results.attrs["baseline"]`` when
    ``include_baseline`` is true. Never pass the independent final-test
    partition here.
    """
    objective = objective or TuningObjective()
    if type(n_repeats) is not int or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer.")
    if (tune_truth is None) != (tune_mask is None):
        raise ValueError("Supply both tune_truth and tune_mask, or neither.")

    target_columns = objective.target_columns(base)
    stratify = tuple(c for c in base.categorical_columns if c in target_columns) if stratify_categorical else ()
    if tune_truth is None:
        holdouts = [
            prepare_tuning_holdout(
                X_tune, columns=target_columns, rate=mask_rate, seed=mask_seed + r, stratify_columns=stratify
            )
            for r in range(n_repeats)
        ]
    else:
        holdouts = [(X_tune, tune_truth, tune_mask)]
        n_repeats = 1

    for X_masked, truth, mask in holdouts:
        validate_tuning_alignment(X_masked, truth, mask, base.model_columns)
        for column in target_columns:
            if column not in mask or not mask[column].any():
                raise ValueError(
                    f"No deliberately masked validation cells for '{column}'. "
                    f"The '{objective.mode}' objective requires this target; "
                    "ensure the runner uses the same objective when preparing its mask."
                )
        if objective.mode == "all_attributes":
            for column in base.categorical_columns:
                held_out = truth.loc[mask[column].to_numpy(dtype=bool), column]
                n_classes = int(held_out.nunique(dropna=True))
                if n_classes < 2:
                    warnings.warn(
                        f"Only {n_classes} class(es) for '{column}' in the validation "
                        "holdout. Its Macro-F1 contributes to all-attribute tuning, "
                        "but this is not a reliable measure of minority-class quality. "
                        "Consider stratified/repeated holdouts before final selection.",
                        RuntimeWarning, stacklevel=2,
                    )

    candidates = expand_grid(grid)
    per_repeat_rows = []
    baseline_rows = []
    for repeat, (X_masked, truth, mask) in enumerate(holdouts):
        seed = mask_seed + repeat if tune_truth is None else mask_seed
        for grid_index, overrides in enumerate(candidates):
            row = score_configuration(base, overrides, X_fit, X_masked, truth, mask, objective)
            row["grid_index"] = grid_index
            row["candidate"] = _candidate_label(overrides)
            row["repeat"] = repeat
            row["mask_seed"] = seed
            per_repeat_rows.append(row)
        if include_baseline:
            baseline = _baseline_scores(base, X_fit, X_masked, truth, mask, objective)
            baseline["repeat"] = repeat
            baseline["mask_seed"] = seed
            baseline_rows.append(baseline)

    per_repeat = pd.DataFrame(per_repeat_rows)
    grid_keys = list(candidates[0]) if candidates and candidates[0] else []

    if n_repeats == 1:
        results = per_repeat.drop(columns=["repeat"]).copy()
        results["objective_mean"] = results["objective"]
        results["objective_std"] = 0.0
        results["n_repeats"] = 1
    else:
        numeric_cols = [
            c for c in per_repeat.columns
            if c not in {"grid_index", "candidate", "repeat", "mask_seed", "objective_mode", *grid_keys}
            and pd.api.types.is_numeric_dtype(per_repeat[c])
        ]
        grouped = per_repeat.groupby("grid_index", sort=True)
        means = grouped[numeric_cols].mean()
        stds = grouped[["objective", "year_mae", "material_macro_f1"]].std(ddof=1).add_suffix("_std")
        first = grouped[["candidate", "objective_mode", *grid_keys]].first()
        results = pd.concat([first, means, stds], axis=1).reset_index()
        results["objective_mean"] = results["objective"]
        results["objective_std"] = results["objective_std"].fillna(0.0)
        results["n_repeats"] = n_repeats

    if objective.mode == "year_material":
        results["rank_average"] = (
            results["year_mae"].rank(method="average")
            + (-results["material_macro_f1"]).rank(method="average")
        ) / 2.0
    else:
        ranks = [results[f"mae__{column}"].rank(method="average") for column in base.numeric_columns]
        ranks += [(-results[f"macro_f1__{column}"]).rank(method="average") for column in base.categorical_columns]
        results["rank_average"] = pd.concat(ranks, axis=1).mean(axis=1)
    results = results.sort_values(
        ["objective_mean", "rank_average", "grid_index"], kind="mergesort"
    ).reset_index(drop=True)

    # Selection with a simplicity tie-break when repeats make the noise visible.
    best = results.iloc[0]
    chosen_index = int(best["grid_index"])
    if n_repeats > 1 and simplicity_tolerance > 0:
        tolerance = float(best["objective_std"]) * simplicity_tolerance
        within = results[results["objective_mean"] <= float(best["objective_mean"]) + tolerance]
        if len(within) > 1:
            simplest = within.sort_values(["total_tree_nodes", "objective_mean", "grid_index"], kind="mergesort").iloc[0]
            chosen_index = int(simplest["grid_index"])
    results["selected"] = results["grid_index"] == chosen_index

    results.attrs["per_repeat"] = per_repeat
    results.attrs["baseline"] = pd.DataFrame(baseline_rows)
    results.attrs["selected_grid_index"] = chosen_index

    # Take the winning overrides from the original grid objects so that
    # types survive (a DataFrame would turn ``None`` into NaN and ints into
    # floats, which the config validator rightly rejects).
    best_overrides = candidates[chosen_index]
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
