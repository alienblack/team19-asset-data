"""Predictive (inductive) mixed-type MissForest with frozen sequential replay.

This module adds a fit/transform imputer next to the existing whole-table
``missforest_impute`` function without modifying it.

Fit
---
1. Initial fill statistics (numeric mean or median, deterministic categorical
   mode) are computed from originally observed fitting cells only.
2. A per-column vocabulary of observed categories is frozen.
3. The target processing order is frozen: every model column becomes a target,
   ordered by ascending training missingness and then by configuration order.
4. Exactly ``n_passes`` sequential passes are run. In each pass, for each
   target in the frozen order, a fresh Pipeline(ColumnTransformer -> forest)
   is fitted on rows whose target was originally observed, using the current
   working state of the other columns as predictors, and originally missing
   cells of that target are overwritten with its predictions before the next
   target is processed (Gauss-Seidel / sequential update, as in the existing
   implementation and in the original missForest algorithm).
5. Every fitted pipeline is stored under ``(pass, target)``.

Transform (frozen)
------------------
The new batch is initialised with the stored fill statistics, then the stored
pipelines are replayed in exactly the fit-time pass and target order, updating
only originally missing cells. No estimator is fitted and no statistic is
computed from the new batch. Observed input values are restored verbatim and
declared constraints are applied to imputed cells only.

Why replay reproduces fit-time state
------------------------------------
When pipeline ``(p, j)`` was fitted, the predictor state of every row was:
columns earlier than ``j`` in the frozen order already updated by pass ``p``
pipelines, columns later than ``j`` still holding pass ``p-1`` values (or the
initial fill when ``p == 1``). ``transform`` rebuilds exactly that sequence of
states for a new batch because the initial fill, the pipelines, the order and
the pass count are all frozen. On the fitting data itself the replay is
therefore an exact reproduction of the fit-time completed table (tested).

Documented fit/transform asymmetry
----------------------------------
During fit, the predictor values of a training row that were themselves
imputed came from pipelines fitted on rows including that row (in-sample).
During transform, imputed predictor values of a new row come from the same
pipelines applied out-of-sample. This is inherent to any stored-model
MissForest (the R package ``missForestPredict`` shares it) and is not a bug.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .postprocess import apply_constraints
from .simple_imputer import deterministic_mode

FORMAT_VERSION = 1

SUPPORTED_NUMERIC_INIT = {"mean", "median"}
SUPPORTED_UNKNOWN_POLICIES = {"ignore", "mode", "error"}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PredictiveMissForestConfig:
    """Configuration for the predictive (fit/transform) MissForest.

    Forest hyperparameters follow the same rules as the legacy implementation,
    but are validated locally so this module does not depend on legacy code.
    ``n_passes`` fixes the number of sequential passes for frozen replay.

    Parameters
    ----------
    numeric_init:
        ``"mean"`` (existing whole-table behaviour) or ``"median"``.
    unknown_category_policy:
        What to do with an observed categorical value at transform time that
        was never observed during fit.
        ``"ignore"`` keeps the value in the output and lets the one-hot
        encoder represent it as an all-zero row (no category information);
        ``"mode"`` keeps the value in the output but substitutes the training
        mode for prediction purposes; ``"error"`` raises.
    min_observed_target:
        Minimum number of originally observed fitting cells a column needs for
        a forest to be fitted for it. Columns below the threshold are filled
        with the initial statistic only and listed in ``unmodelled_targets_``.
    n_jobs:
        Defaults to 1 because scikit-learn forests accumulate parallel
        predictions in thread-completion order, which is not bit-for-bit
        reproducible. Use -1 for speed during tuning if ulp-level differences
        are acceptable.
    """

    n_estimators: int = 100
    n_passes: int = 5
    random_state: int = 219
    n_jobs: int = 1

    max_features: str | int | float | None = "sqrt"
    max_depth: int | None = None
    min_samples_leaf: int = 1
    class_weight: str | dict | None = None

    numeric_init: str = "mean"
    unknown_category_policy: str = "ignore"
    min_observed_target: int = 2

    def replace(self, **changes: Any) -> "PredictiveMissForestConfig":
        """Return a copy with selected fields changed (validated on use)."""
        return replace(self, **changes)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> "PredictiveMissForestConfig":
        """Build from a YAML ``methods.predictive_missforest`` section."""
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(
                "Unknown predictive_missforest configuration keys: "
                + ", ".join(unknown)
            )
        config = cls(**dict(mapping))
        validate_predictive_config(config)
        return config


def validate_predictive_config(config: PredictiveMissForestConfig) -> None:
    """Validate predictive settings without importing legacy MissForest."""

    _validate_predictive_forest_config(config)

    if config.numeric_init not in SUPPORTED_NUMERIC_INIT:
        raise ValueError(
            "numeric_init must be one of: "
            + ", ".join(sorted(SUPPORTED_NUMERIC_INIT))
        )

    if config.unknown_category_policy not in SUPPORTED_UNKNOWN_POLICIES:
        raise ValueError(
            "unknown_category_policy must be one of: "
            + ", ".join(sorted(SUPPORTED_UNKNOWN_POLICIES))
        )

    if (
        not isinstance(config.min_observed_target, int)
        or isinstance(config.min_observed_target, bool)
        or config.min_observed_target < 1
    ):
        raise ValueError("min_observed_target must be a positive integer.")


# --------------------------------------------------------------------------- #
# Local forest validation and convergence helpers.
# Keep legacy missforest.py unchanged: these functions deliberately implement
# the same checks and convergence formulas without importing legacy internals.
# --------------------------------------------------------------------------- #


def _validate_predictive_forest_config(
    config: PredictiveMissForestConfig,
) -> None:
    """Validate this module's forest hyperparameters and pass count."""

    if (
        not isinstance(config.n_estimators, int)
        or isinstance(config.n_estimators, bool)
        or config.n_estimators < 1
    ):
        raise ValueError(
            "n_estimators must be a positive integer."
        )

    if (
        not isinstance(config.n_passes, int)
        or isinstance(config.n_passes, bool)
        or config.n_passes < 1
    ):
        raise ValueError(
            "n_passes must be a positive integer."
        )

    if (
        not isinstance(config.random_state, int)
        or isinstance(config.random_state, bool)
    ):
        raise ValueError(
            "random_state must be an integer."
        )

    if (
        not isinstance(config.n_jobs, int)
        or isinstance(config.n_jobs, bool)
        or config.n_jobs == 0
    ):
        raise ValueError(
            "n_jobs must be a non-zero integer."
        )

    if config.max_depth is not None:
        if (
            not isinstance(config.max_depth, int)
            or isinstance(config.max_depth, bool)
            or config.max_depth < 1
        ):
            raise ValueError(
                "max_depth must be None or a positive integer."
            )

    if (
        not isinstance(config.min_samples_leaf, int)
        or isinstance(config.min_samples_leaf, bool)
        or config.min_samples_leaf < 1
    ):
        raise ValueError(
            "min_samples_leaf must be a positive integer."
        )

    max_features = config.max_features

    if isinstance(max_features, str):
        if max_features not in {
            "sqrt",
            "log2",
        }:
            raise ValueError(
                "max_features string value must be "
                "'sqrt' or 'log2'."
            )

    elif isinstance(max_features, int):
        if isinstance(max_features, bool) or max_features < 1:
            raise ValueError(
                "Integer max_features must be at least 1."
            )

    elif isinstance(max_features, float):
        if not 0.0 < max_features <= 1.0:
            raise ValueError(
                "Float max_features must be in the range (0, 1]."
            )

    elif max_features is not None:
        raise ValueError(
            "max_features must be 'sqrt', 'log2', "
            "a positive integer, a float in (0, 1], or None."
        )

    if isinstance(config.class_weight, str):
        if config.class_weight not in {
            "balanced",
            "balanced_subsample",
        }:
            raise ValueError(
                "class_weight string value must be "
                "'balanced' or 'balanced_subsample'."
            )


def _numeric_change(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    missing_mask: pd.DataFrame,
    numeric_columns: Sequence[str],
) -> float | None:
    """Measure relative change in imputed numeric values."""

    numerator = 0.0
    denominator = 0.0
    count = 0

    for column in numeric_columns:
        positions = missing_mask[
            column
        ]

        if not positions.any():
            continue

        old = (
            previous.loc[
                positions,
                column,
            ]
            .astype(float)
            .to_numpy()
        )

        new = (
            current.loc[
                positions,
                column,
            ]
            .astype(float)
            .to_numpy()
        )

        numerator += float(
            np.square(
                new - old
            ).sum()
        )

        denominator += float(
            np.square(
                new
            ).sum()
        )

        count += int(
            positions.sum()
        )

    if count == 0:
        return None

    epsilon = np.finfo(
        float
    ).eps

    if denominator <= epsilon:
        if numerator <= epsilon:
            return 0.0

        return float(
            "inf"
        )

    return numerator / denominator


def _categorical_change(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    missing_mask: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> float | None:
    """Measure fraction of imputed categorical values that changed."""

    changed = 0
    total = 0

    for column in categorical_columns:
        positions = missing_mask[
            column
        ]

        if not positions.any():
            continue

        old = (
            previous.loc[
                positions,
                column,
            ]
            .astype(str)
            .to_numpy()
        )

        new = (
            current.loc[
                positions,
                column,
            ]
            .astype(str)
            .to_numpy()
        )

        changed += int(
            np.not_equal(
                old,
                new,
            ).sum()
        )

        total += int(
            positions.sum()
        )

    if total == 0:
        return None

    return changed / total


# --------------------------------------------------------------------------- #
# Validation for fitting and for prediction (different rules)
# --------------------------------------------------------------------------- #


def _basic_frame_checks(
    frame: pd.DataFrame,
    model_columns: Sequence[str],
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("Input dataframe is empty.")

    duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicated:
        raise ValueError(
            "Input dataframe contains duplicate columns: "
            + ", ".join(sorted(set(duplicated)))
        )

    missing_columns = sorted(set(model_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Input is missing requested model columns: "
            + ", ".join(missing_columns)
        )


def _check_numeric_observed(frame: pd.DataFrame, column: str) -> None:
    observed = frame[column].dropna()
    converted = pd.to_numeric(observed, errors="coerce")
    invalid = int(converted.isna().sum())
    if invalid:
        raise ValueError(
            f"Numeric column '{column}' contains {invalid} "
            "non-numeric observed value(s)."
        )


def validate_fit_input(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> None:
    """Rules for fitting: every model column needs at least one observed value
    (to compute an initial fill) and categorical values must share one type."""

    model_columns = list(numeric_columns) + list(categorical_columns)
    _basic_frame_checks(frame, model_columns)

    if len(model_columns) < 2:
        raise ValueError(
            "PredictiveMissForest requires at least two model columns."
        )

    for column in numeric_columns:
        if frame[column].notna().sum() == 0:
            raise ValueError(
                f"Numeric column '{column}' has no observed fitting values."
            )
        _check_numeric_observed(frame, column)

    for column in categorical_columns:
        observed = frame[column].dropna()
        if len(observed) == 0:
            raise ValueError(
                f"Categorical column '{column}' has no observed fitting values."
            )
        types = {type(value) for value in observed}
        if len(types) > 1:
            raise ValueError(
                f"Categorical column '{column}' mixes value types "
                f"({', '.join(sorted(t.__name__ for t in types))}); "
                "use one type (strings recommended)."
            )


def validate_transform_input(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> None:
    """Rules for prediction: columns must exist and observed numeric values must
    be numeric. Entirely missing columns and fully missing rows are allowed."""

    model_columns = list(numeric_columns) + list(categorical_columns)
    _basic_frame_checks(frame, model_columns)

    for column in numeric_columns:
        _check_numeric_observed(frame, column)


def validate_transform_result(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> None:
    """Checks on a transform output: shape, index, columns, completeness,
    untouched observed values, untouched non-model columns."""

    model_columns = list(numeric_columns) + list(categorical_columns)

    if len(imputed) != len(original):
        raise ValueError("Transform changed the number of rows.")

    if not imputed.index.equals(original.index):
        raise ValueError("Transform changed the dataframe index.")

    if list(imputed.columns) != list(original.columns):
        raise ValueError("Transform changed dataframe columns or column order.")

    for column in model_columns:
        remaining = int(imputed[column].isna().sum())
        if remaining:
            raise ValueError(
                f"Imputed column '{column}' still contains {remaining} "
                "missing value(s)."
            )

        observed = original[column].notna().to_numpy()
        before = original[column].to_numpy(dtype=object)[observed]
        after = imputed[column].to_numpy(dtype=object)[observed]
        if not _object_arrays_equal(before, after):
            raise ValueError(
                f"Transform modified originally observed values in '{column}'."
            )

    non_model = [c for c in original.columns if c not in model_columns]
    if non_model:
        try:
            pd.testing.assert_frame_equal(
                original[non_model],
                imputed[non_model],
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(
                "Transform modified one or more non-model columns."
            ) from error


def _object_arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    for x, y in zip(a, b):
        if x is y or x == y:
            continue
        try:
            if np.isnan(x) and np.isnan(y):  # both NaN floats
                continue
        except TypeError:
            pass
        return False
    return True


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class TransformResult:
    """Output of :meth:`PredictiveMissForest.transform_detailed`."""

    imputed: pd.DataFrame
    missing_mask: pd.DataFrame
    unknown_category_counts: dict[str, int]
    runtime_seconds: float
    scores: pd.DataFrame | None = None
    final_pass_features: dict[str, pd.DataFrame] | None = None


# --------------------------------------------------------------------------- #
# Pipeline construction (modified copy of missforest._build_model)
# --------------------------------------------------------------------------- #


def _build_pipeline(
    target: str,
    features: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    vocabularies: Mapping[str, list],
    config: PredictiveMissForestConfig,
    seed: int,
) -> Pipeline:
    """Build Pipeline(preprocessor -> forest) for one target.

    Differences from ``missforest._build_model``: the feature list is supplied
    (frozen at fit) and the OneHotEncoder receives the frozen per-column
    vocabulary so every stored pipeline has an identical encoded layout and
    unknown categories are well defined. The encoder is still fitted and
    persisted inside the Pipeline; no separate encoder object exists.
    """

    numeric_features = [c for c in features if c in numeric_columns]
    categorical_features = [c for c in features if c in categorical_columns]

    transformers: list[tuple] = []

    if numeric_features:
        transformers.append(("numeric", "passthrough", numeric_features))

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    categories=[
                        list(vocabularies[c]) for c in categorical_features
                    ],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )

    if target in categorical_columns:
        forest: Any = RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=seed,
            n_jobs=config.n_jobs,
            max_features=config.max_features,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight=config.class_weight,
        )
    else:
        forest = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=seed,
            n_jobs=config.n_jobs,
            max_features=config.max_features,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("forest", forest)])


def _sorted_vocabulary(values: pd.Series) -> list:
    unique = list(pd.unique(values.dropna()))
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=str)


# --------------------------------------------------------------------------- #
# The imputer
# --------------------------------------------------------------------------- #


class PredictiveMissForest:
    """Mixed-type iterative Random Forest imputer with frozen replay.

    Parameters
    ----------
    numeric_columns, categorical_columns:
        Model columns. Every other column in the input is carried through
        unchanged and never used for prediction.
    config:
        :class:`PredictiveMissForestConfig`.
    constraints:
        Optional ``{column: {"data_type": ..., "min": ..., "max": ...}}`` as
        in the YAML ``constraints`` section; applied to imputed cells only.
    """

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        config: PredictiveMissForestConfig | None = None,
        constraints: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.numeric_columns = tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.config = config or PredictiveMissForestConfig()
        validate_predictive_config(self.config)

        overlap = sorted(set(self.numeric_columns) & set(self.categorical_columns))
        if overlap:
            raise ValueError(
                "Columns cannot be both numeric and categorical: "
                + ", ".join(overlap)
            )
        if len(set(self.numeric_columns)) != len(self.numeric_columns):
            raise ValueError("numeric_columns contains duplicates.")
        if len(set(self.categorical_columns)) != len(self.categorical_columns):
            raise ValueError("categorical_columns contains duplicates.")

        constraints = dict(constraints or {})
        model = set(self.model_columns)
        foreign = sorted(set(constraints) - model)
        if foreign:
            raise ValueError(
                "Constraints refer to non-model columns: " + ", ".join(foreign)
            )
        self.constraints = {k: dict(v) for k, v in constraints.items()}

        self._fitted = False

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_imputation_config(
        cls,
        imputation_config: Any,
        overrides: Mapping[str, Any] | None = None,
    ) -> "PredictiveMissForest":
        """Build from :class:`imputation.config.ImputationConfig`.

        Uses ``methods.predictive_missforest`` when present, otherwise falls
        back to ``methods.missforest`` with ``max_iter`` mapped to
        ``n_passes`` (explicitly, with a warning).
        """
        section = getattr(imputation_config, "predictive_missforest_config", None)
        if section is None:
            legacy = dict(imputation_config.missforest_config)
            warnings.warn(
                "No methods.predictive_missforest section; deriving settings "
                "from methods.missforest (max_iter -> n_passes).",
                stacklevel=2,
            )
            section = {
                "n_estimators": legacy.get("n_estimators", 100),
                "n_passes": legacy.get("max_iter", 5),
                "random_state": legacy.get("random_state", 219),
                "n_jobs": legacy.get("n_jobs", 1),
                "max_features": legacy.get("max_features", "sqrt"),
                "max_depth": legacy.get("max_depth"),
                "min_samples_leaf": legacy.get("min_samples_leaf", 1),
                "class_weight": legacy.get("class_weight"),
            }
        section = dict(section)
        if overrides:
            section.update(overrides)
        config = PredictiveMissForestConfig.from_mapping(section)
        return cls(
            numeric_columns=imputation_config.numeric_columns,
            categorical_columns=imputation_config.categorical_columns,
            config=config,
            constraints=imputation_config.constraints,
        )

    @property
    def model_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PredictiveMissForest is not fitted; call fit().")

    # ------------------------------------------------------------------ #
    # Working-frame preparation (shared by fit and transform)
    # ------------------------------------------------------------------ #

    def _coerce_model_frame(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (raw model-column frame with RangeIndex, missing mask)."""
        frame = X.loc[:, list(self.model_columns)].reset_index(drop=True)
        missing = frame.isna()
        return frame, missing

    def _initialise_with_stats(
        self,
        frame: pd.DataFrame,
        missing: pd.DataFrame,
        init_values: Mapping[str, Any],
        vocabularies: Mapping[str, list],
        policy: str,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Fill the working frame from stored statistics only."""

        working = frame.copy(deep=True)
        unknown_counts: dict[str, int] = {}

        for column in self.numeric_columns:
            values = pd.to_numeric(working[column], errors="coerce")
            working[column] = values.fillna(float(init_values[column])).astype(float)

        for column in self.categorical_columns:
            values = working[column].astype(object)
            observed = ~missing[column].to_numpy()
            vocabulary = set(vocabularies[column])
            is_unknown = observed & ~values.isin(vocabulary).to_numpy()
            n_unknown = int(is_unknown.sum())
            if n_unknown:
                unknown_counts[column] = n_unknown
                if policy == "error":
                    raise ValueError(
                        f"Column '{column}' contains {n_unknown} observed "
                        "categorical value(s) never seen during fit."
                    )
                if policy == "mode":
                    values = values.copy()
                    values.iloc[np.flatnonzero(is_unknown)] = init_values[column]
                # policy == "ignore": leave as is; encoder emits all zeros.
            fill = init_values[column]
            values = values.where(~missing[column].to_numpy(), other=fill)
            working[column] = values

        return working, unknown_counts

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X_fit: pd.DataFrame) -> "PredictiveMissForest":
        """Fit forests for every model column over ``n_passes`` sequential passes."""

        started = perf_counter()
        config = self.config
        numeric = self.numeric_columns
        categorical = self.categorical_columns
        model_columns = list(self.model_columns)

        validate_fit_input(X_fit, numeric, categorical)

        frame, missing = self._coerce_model_frame(X_fit)

        # 1. initial-fill statistics from originally observed cells only
        init_values: dict[str, Any] = {}
        for column in numeric:
            observed = pd.to_numeric(frame[column], errors="coerce").dropna()
            init_values[column] = (
                float(observed.mean())
                if config.numeric_init == "mean"
                else float(observed.median())
            )
        for column in categorical:
            init_values[column] = deterministic_mode(frame[column].astype(object))

        # 2. frozen vocabularies
        vocabularies = {
            column: _sorted_vocabulary(frame[column].astype(object))
            for column in categorical
        }

        # 3. frozen target order: all model columns, ascending training
        #    missingness, then configuration order (same rule as missforest.py)
        column_index = {c: i for i, c in enumerate(model_columns)}
        missing_counts = {c: int(missing[c].sum()) for c in model_columns}
        target_order = sorted(
            model_columns,
            key=lambda c: (missing_counts[c], column_index[c]),
        )

        observed_counts = {c: len(frame) - missing_counts[c] for c in model_columns}
        unmodelled = [
            c for c in target_order
            if observed_counts[c] < config.min_observed_target
        ]
        if unmodelled:
            warnings.warn(
                "Columns with fewer than min_observed_target observed fitting "
                "values receive the initial fill only: " + ", ".join(unmodelled),
                stacklevel=2,
            )

        features = {
            target: [c for c in model_columns if c != target]
            for target in model_columns
        }

        # Seeds tied to column identity (as in missforest.py), fixed across passes.
        seeds = {c: int(config.random_state) + column_index[c] for c in model_columns}

        # str(label) -> original label, per categorical column (see fit loop).
        label_maps = {
            column: {str(value): value for value in vocabularies[column]}
            for column in categorical
        }

        working, unknown_counts = self._initialise_with_stats(
            frame, missing, init_values, vocabularies, policy="ignore"
        )
        if unknown_counts:  # cannot happen: vocabulary built from this frame
            raise RuntimeError("Internal error: unknown categories at fit time.")

        # Supervised targets are the originally observed values, taken from
        # the raw frame so placeholders can never become labels.
        observed_targets: dict[str, pd.Series] = {}
        for column in numeric:
            observed_targets[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in categorical:
            observed_targets[column] = frame[column].astype(object)

        models: dict[tuple[int, str], Pipeline] = {}
        convergence_rows: list[dict[str, Any]] = []
        n_train_rows: dict[tuple[int, str], int] = {}

        for pass_number in range(1, config.n_passes + 1):
            pass_started = perf_counter()
            before = working.copy(deep=True)

            for target in target_order:
                if target in unmodelled:
                    continue

                target_missing = missing[target].to_numpy()
                target_observed = ~target_missing
                feature_list = features[target]

                pipeline = _build_pipeline(
                    target=target,
                    features=feature_list,
                    numeric_columns=numeric,
                    categorical_columns=categorical,
                    vocabularies=vocabularies,
                    config=config,
                    seed=seeds[target],
                )

                X_train = working.loc[target_observed, feature_list]
                y_train = observed_targets[target][target_observed]
                if target in categorical:
                    # Classifiers are trained on string labels and predictions
                    # are mapped back through the frozen vocabulary, so the
                    # original Python type of a category (str or int) is
                    # preserved exactly in the output.
                    y_train = y_train.astype(str)

                pipeline.fit(X_train, y_train)
                models[(pass_number, target)] = pipeline
                n_train_rows[(pass_number, target)] = int(target_observed.sum())

                if target_missing.any():
                    predictions = self._predict_labels(
                        pipeline, working.loc[target_missing, feature_list],
                        target, label_maps,
                    )
                    working.loc[target_missing, target] = predictions

            numeric_change = _numeric_change(before, working, missing, numeric)
            categorical_change = _categorical_change(
                before, working, missing, categorical
            )
            convergence_rows.append(
                {
                    "pass": pass_number,
                    "numeric_change": numeric_change,
                    "categorical_change": categorical_change,
                    "runtime_seconds": perf_counter() - pass_started,
                }
            )

        # Freeze fitted state.
        self.init_values_ = init_values
        self.vocabularies_ = vocabularies
        self.target_order_ = tuple(target_order)
        self.features_ = {k: tuple(v) for k, v in features.items()}
        self.seeds_ = seeds
        self.label_maps_ = label_maps
        self.unmodelled_targets_ = tuple(unmodelled)
        self.models_ = models
        self.fit_missing_mask_ = missing.copy()
        self.fit_missing_mask_.index = X_fit.index
        self.convergence_ = pd.DataFrame(convergence_rows)
        self._fitted = True

        self.fit_imputed_ = self._assemble_output(X_fit, working)

        total_nodes = sum(
            int(t.tree_.node_count)
            for pipeline in models.values()
            for t in pipeline.named_steps["forest"].estimators_
        )
        self.fit_report_ = {
            "n_rows": int(len(frame)),
            "n_models": len(models),
            "n_passes": config.n_passes,
            "target_order": list(target_order),
            "unmodelled_targets": list(unmodelled),
            "missing_counts": missing_counts,
            "observed_counts": observed_counts,
            "n_train_rows": {f"{p}:{t}": n for (p, t), n in n_train_rows.items()},
            "total_tree_nodes": total_nodes,
            "runtime_seconds": perf_counter() - started,
            "sklearn_version": sklearn.__version__,
        }

        # Reuse the strict whole-table result validator on the fitting output.
        validate_transform_result(X_fit, self.fit_imputed_, numeric, categorical)
        return self

    # ------------------------------------------------------------------ #
    # Transform (frozen replay)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _predict_labels(
        pipeline: Pipeline,
        X_rows: pd.DataFrame,
        target: str,
        label_maps: Mapping[str, Mapping[str, Any]],
    ) -> np.ndarray:
        """Predict and, for categorical targets, map string classes back to
        the original label objects."""
        predictions = pipeline.predict(X_rows)
        if target in label_maps:
            mapping = label_maps[target]
            return np.array([mapping[str(p)] for p in predictions], dtype=object)
        return predictions

    def _assemble_output(self, X: pd.DataFrame, working: pd.DataFrame) -> pd.DataFrame:
        """Merge imputed cells into a copy of X (positional), restore observed
        values verbatim, apply constraints to imputed cells, restore index."""

        positional = X.reset_index(drop=True)
        output = positional.copy(deep=True)
        missing = positional.loc[:, list(self.model_columns)].isna()

        for column in self.model_columns:
            original_values = positional[column].to_numpy(dtype=object, copy=True)
            imputed_values = working[column].to_numpy(dtype=object)
            positions = missing[column].to_numpy()
            original_values[positions] = imputed_values[positions]
            if column in self.numeric_columns:
                try:
                    output[column] = pd.to_numeric(pd.Series(original_values))
                except (ValueError, TypeError):
                    output[column] = pd.Series(original_values, dtype=object)
            else:
                series = pd.Series(original_values, dtype=object)
                try:  # keep the caller's categorical dtype (object, str, category)
                    series = series.astype(positional[column].dtype)
                except (ValueError, TypeError):
                    pass
                output[column] = series

        if self.constraints:
            output = apply_constraints(
                original=positional,
                imputed=output,
                constraints=self.constraints,
            )

        output.index = X.index
        return output

    def _replay(
        self,
        X_new: pd.DataFrame,
        capture_scores: bool,
        feature_targets: tuple[str, ...] = (),
    ) -> TransformResult:
        self._require_fitted()
        started = perf_counter()
        numeric = self.numeric_columns
        categorical = self.categorical_columns
        config = self.config

        validate_transform_input(X_new, numeric, categorical)

        frame, missing = self._coerce_model_frame(X_new)
        working, unknown_counts = self._initialise_with_stats(
            frame,
            missing,
            self.init_values_,
            self.vocabularies_,
            policy=config.unknown_category_policy,
        )

        score_rows: list[dict[str, Any]] = []
        feature_views: dict[str, pd.DataFrame] = {}
        index_values = X_new.index.to_numpy()

        for pass_number in range(1, config.n_passes + 1):
            for target in self.target_order_:
                if target in self.unmodelled_targets_:
                    continue

                if pass_number == config.n_passes and target in feature_targets:
                    feature_views[target] = working.loc[:, list(self.features_[target])].copy()

                rows = missing[target].to_numpy()
                if not rows.any():
                    continue

                pipeline = self.models_[(pass_number, target)]
                feature_list = list(self.features_[target])
                X_rows = working.loc[rows, feature_list]
                predictions = self._predict_labels(
                    pipeline, X_rows, target, self.label_maps_
                )

                if capture_scores and pass_number == config.n_passes:
                    score_rows.extend(
                        self._score_rows(
                            pipeline, X_rows, predictions, target,
                            np.flatnonzero(rows), index_values,
                        )
                    )

                working.loc[rows, target] = predictions

        imputed = self._assemble_output(X_new, working)
        validate_transform_result(X_new, imputed, numeric, categorical)

        mask = missing.copy()
        mask.index = X_new.index

        scores = None
        if capture_scores:
            columns = [
                "row", "position", "column", "kind", "imputed",
                "max_proba", "margin", "entropy", "tree_std", "tree_p10", "tree_p50", "tree_p90",
                "prediction_source", "raw_prediction", "material_probabilities",
            ]
            # A rare target with too few observed fitting examples has no forest.
            # Its imputed cells must still be accounted for in diagnostics.
            for target in self.unmodelled_targets_:
                for pos in np.flatnonzero(missing[target].to_numpy()):
                    score_rows.append({
                        "row": index_values[pos], "position": int(pos),
                        "column": target,
                        "kind": "numeric" if target in numeric else "categorical",
                        "imputed": self.init_values_[target],
                        "prediction_source": "training_statistic_fallback",
                        "raw_prediction": self.init_values_[target],
                    })
            for row in score_rows:
                if "prediction_source" not in row:
                    row["prediction_source"] = "random_forest"
                    row["raw_prediction"] = row["imputed"]
                # Report what transform actually outputs, including any
                # rounding/clipping.  Raw forest prediction is separately kept.
                row["imputed"] = imputed.iloc[row["position"]][row["column"]]
            scores = pd.DataFrame(score_rows, columns=columns)

        return TransformResult(
            imputed=imputed,
            missing_mask=mask,
            unknown_category_counts=unknown_counts,
            runtime_seconds=perf_counter() - started,
            scores=scores,
            final_pass_features=feature_views if feature_targets else None,
        )

    def final_pass_features(self, X_new: pd.DataFrame, target: str) -> pd.DataFrame:
        """Return the exact pre-update predictors used by a saved target forest.

        Replays fitted models without refitting. Rows retain positional order and
        include both observed and missing target rows for leaf-support audits.
        """
        self._require_fitted()
        if target not in self.features_ or target in self.unmodelled_targets_:
            raise ValueError(f"No fitted target forest for {target}")
        result = self._replay(X_new, capture_scores=False, feature_targets=(target,))
        if result.final_pass_features is None or target not in result.final_pass_features:
            raise AssertionError(f"Final-pass predictors unavailable for {target}")
        return result.final_pass_features[target]

    def _score_rows(
        self,
        pipeline: Pipeline,
        X_rows: pd.DataFrame,
        predictions: np.ndarray,
        target: str,
        positions: np.ndarray,
        index_values: np.ndarray,
    ) -> list[dict[str, Any]]:
        forest = pipeline.named_steps["forest"]
        rows: list[dict[str, Any]] = []

        if target in self.categorical_columns:
            proba = pipeline.predict_proba(X_rows)
            sorted_proba = np.sort(proba, axis=1)
            top = sorted_proba[:, -1]
            second = sorted_proba[:, -2] if proba.shape[1] > 1 else np.zeros_like(top)
            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = -np.sum(
                    np.where(proba > 0, proba * np.log(proba), 0.0), axis=1
                )
            for i, pos in enumerate(positions):
                rows.append(
                    {
                        "row": index_values[pos],
                        "position": int(pos),
                        "column": target,
                        "kind": "categorical",
                        "imputed": predictions[i],
                        "max_proba": float(top[i]),
                        "margin": float(top[i] - second[i]),
                        "entropy": float(entropy[i]),
                        "tree_std": np.nan,
                        "tree_p10": np.nan,
                        "tree_p50": np.nan,
                        "tree_p90": np.nan,
                        "material_probabilities": (
                            {str(label): float(probability)
                             for label, probability in zip(forest.classes_, proba[i])}
                            if target == "material" else None
                        ),
                    }
                )
        else:
            encoded = pipeline.named_steps["preprocessor"].transform(X_rows)
            per_tree = np.stack([t.predict(encoded) for t in forest.estimators_])
            std = per_tree.std(axis=0)
            p10 = np.percentile(per_tree, 10, axis=0)
            p50 = np.percentile(per_tree, 50, axis=0)
            p90 = np.percentile(per_tree, 90, axis=0)
            for i, pos in enumerate(positions):
                rows.append(
                    {
                        "row": index_values[pos],
                        "position": int(pos),
                        "column": target,
                        "kind": "numeric",
                        "imputed": float(predictions[i]),
                        "max_proba": np.nan,
                        "margin": np.nan,
                        "entropy": np.nan,
                        "tree_std": float(std[i]),
                        "tree_p10": float(p10[i]),
                        "tree_p50": float(p50[i]),
                        "tree_p90": float(p90[i]),
                    }
                )
        return rows

    def transform(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Impute missing model-column cells of unseen records (no fitting)."""
        return self._replay(X_new, capture_scores=False).imputed

    def transform_detailed(
        self,
        X_new: pd.DataFrame,
        with_scores: bool = False,
    ) -> TransformResult:
        """Impute and return the mask, unknown-category counts and, optionally,
        per-cell confidence scores from the final pass."""
        return self._replay(X_new, capture_scores=with_scores)

    def predict_scores(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Per-imputed-cell confidence diagnostics from the final-pass forests.

        Columns: ``row`` (input index label), ``position``, ``column``,
        ``kind``, ``imputed``; categorical: ``max_proba``, ``margin``
        (top-1 minus top-2 vote share), ``entropy``; numeric: ``tree_std``,
        ``tree_p10``, ``tree_p50``, ``tree_p90`` across trees. These are *rankings* of
        confidence, not calibrated probabilities. ``raw_prediction`` is the
        unmodified forest output; ``imputed`` is the final constrained output.
        Low-observation fallbacks are explicitly identified in
        ``prediction_source``. ``material_probabilities`` lists uncalibrated
        per-class vote probabilities when material is the target.
        """
        result = self._replay(X_new, capture_scores=True)
        assert result.scores is not None
        return result.scores

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> Path:
        """Persist the fitted imputer with joblib (compressed)."""
        self._require_fitted()
        path = Path(path)
        payload = {
            "format_version": FORMAT_VERSION,
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "config": asdict(self.config),
            "object": self,
        }
        joblib.dump(payload, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PredictiveMissForest":
        """Load an imputer saved with :meth:`save`."""
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict) or "object" not in payload:
            raise ValueError("File is not a saved PredictiveMissForest.")
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported saved format version {payload.get('format_version')}."
            )
        if payload.get("sklearn_version") != sklearn.__version__:
            warnings.warn(
                "Model saved with scikit-learn "
                f"{payload.get('sklearn_version')} but loaded with "
                f"{sklearn.__version__}; re-run the replay test.",
                stacklevel=2,
            )
        model = payload["object"]
        if not isinstance(model, cls):
            raise ValueError("Saved object is not a PredictiveMissForest.")
        return model

    # ------------------------------------------------------------------ #
    # Introspection helpers for the subgroup's explainability work
    # ------------------------------------------------------------------ #

    def get_pipeline(self, pass_number: int, target: str) -> Pipeline:
        self._require_fitted()
        return self.models_[(pass_number, target)]

    def impurity_importance_by_column(
        self,
        target: str,
        pass_number: int | None = None,
    ) -> pd.Series:
        """Impurity-based importance aggregated back to original columns.

        One-hot dummies of one categorical predictor are summed. Caveat: this
        importance is biased toward continuous and high-cardinality predictors
        and reflects the *training* fit, not generalisation. Prefer the
        column-level permutation importance in ``predictive_tuning`` for
        anything reported to the partner.
        """
        self._require_fitted()
        pass_number = pass_number or self.config.n_passes
        pipeline = self.models_[(pass_number, target)]
        forest = pipeline.named_steps["forest"]
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.Series(forest.feature_importances_, index=names)
        by_column: dict[str, float] = {}
        for name, value in importance.items():
            prefix, _, rest = name.partition("__")
            if prefix == "categorical":
                column = next(
                    (c for c in self.categorical_columns if rest.startswith(c + "_")),
                    rest,
                )
            else:
                column = rest
            by_column[column] = by_column.get(column, 0.0) + float(value)
        return pd.Series(by_column).sort_values(ascending=False)
