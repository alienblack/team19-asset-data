"""Mixed-type iterative Random Forest imputation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .simple_imputer import deterministic_mode
from .validation import (
    validate_imputation_input,
    validate_imputation_result,
)


@dataclass(frozen=True)
class MissForestConfig:
    """Configuration for generic MissForest-style imputation."""

    n_estimators: int = 100
    max_iter: int = 10
    random_state: int = 219
    n_jobs: int = -1

    max_features: str | int | float | None = "sqrt"
    max_depth: int | None = None
    min_samples_leaf: int = 1

    class_weight: str | dict | None = None


@dataclass
class MissForestResult:
    """Result returned by MissForest imputation."""

    imputed: pd.DataFrame
    convergence: pd.DataFrame
    selected_iteration: int
    converged: bool


def _validate_config(
    config: MissForestConfig,
) -> None:
    """Validate MissForest configuration values."""

    if (
        not isinstance(config.n_estimators, int)
        or isinstance(config.n_estimators, bool)
        or config.n_estimators < 1
    ):
        raise ValueError(
            "n_estimators must be a positive integer."
        )

    if (
        not isinstance(config.max_iter, int)
        or isinstance(config.max_iter, bool)
        or config.max_iter < 1
    ):
        raise ValueError(
            "max_iter must be a positive integer."
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


def _initialise(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """Create the initial complete working dataframe.

    Numeric missing values are initialised using the observed mean.

    Categorical missing values are initialised using the deterministic mode.

    These values are only starting estimates. MissForest subsequently replaces
    originally missing cells using Random Forest predictions.
    """

    working = frame.copy(
        deep=True
    )

    for column in numeric_columns:
        values = pd.to_numeric(
            working[column],
            errors="coerce",
        )

        if values.notna().sum() == 0:
            raise ValueError(
                f"Numeric column '{column}' has no observed values."
            )

        fill_value = float(
            values.mean()
        )

        working[column] = values.fillna(
            fill_value
        )

    for column in categorical_columns:
        values = working[column].astype(
            "object"
        )

        fill_value = deterministic_mode(
            values
        )

        working[column] = values.fillna(
            fill_value
        )

    return working


def _build_model(
    target: str,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    config: MissForestConfig,
    seed: int,
) -> tuple[Pipeline, list[str]]:
    """Build the Random Forest model for one target column."""

    all_columns = (
        list(numeric_columns)
        + list(categorical_columns)
    )

    features = [
        column
        for column in all_columns
        if column != target
    ]

    if not features:
        raise ValueError(
            f"MissForest target '{target}' has no predictor columns."
        )

    numeric_features = [
        column
        for column in features
        if column in numeric_columns
    ]

    categorical_features = [
        column
        for column in features
        if column in categorical_columns
    ]

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                "passthrough",
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
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
        forest = RandomForestClassifier(
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

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "forest",
                forest,
            ),
        ]
    )

    return model, features


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


def _restore_observed_values(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    model_columns: Sequence[str],
) -> pd.DataFrame:
    """Restore originally observed values exactly as supplied.

    MissForest is allowed to replace only cells that were originally missing.
    """

    restored = imputed.copy(
        deep=True
    )

    for column in model_columns:
        observed_mask = ~missing_mask[
            column
        ]

        restored.loc[
            observed_mask,
            column,
        ] = original.loc[
            observed_mask,
            column,
        ]

    return restored


def missforest_impute(
    incomplete: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    config: MissForestConfig | None = None,
) -> MissForestResult:
    """Run generic mixed-type MissForest-style iterative imputation.

    Only explicitly supplied model columns participate in prediction.
    Columns outside the model are copied through unchanged.

    Missing numeric values are initially filled using the mean and missing
    categorical values using the deterministic mode.

    Random Forest models are then fitted iteratively for each incomplete
    target column. Targets with fewer missing values are processed first.

    A fixed target-specific random seed is used across iterations. This keeps
    Random Forest randomness stable while the imputed predictor values evolve.

    After every complete iteration, numeric and categorical changes are
    measured only on cells that were missing in the original input.

    An iteration is accepted when at least one active convergence measure
    improves. When no active measure improves, the most recent accepted
    iteration is returned.
    """

    config = config or MissForestConfig()

    _validate_config(
        config
    )

    numeric_columns = tuple(
        numeric_columns
    )

    categorical_columns = tuple(
        categorical_columns
    )

    validate_imputation_input(
        frame=incomplete,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    model_columns = list(
        numeric_columns
        + categorical_columns
    )

    missing_mask = incomplete.loc[
        :,
        model_columns,
    ].isna()

    incomplete_targets = [
        column
        for column in model_columns
        if bool(
            missing_mask[
                column
            ].any()
        )
    ]

    # Nothing needs to be imputed.
    if not incomplete_targets:
        result = incomplete.copy(
            deep=True
        )

        validate_imputation_result(
            original=incomplete,
            imputed=result,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )

        return MissForestResult(
            imputed=result,
            convergence=pd.DataFrame(),
            selected_iteration=0,
            converged=True,
        )

    if len(model_columns) < 2:
        raise ValueError(
            "MissForest requires at least two model columns "
            "when missing values are present."
        )

    # Keep seeds tied to model-column identity/order rather than iteration.
    # This prevents convergence measurements from being affected merely by
    # changing Random Forest seeds between passes.
    target_seeds = {
        column: (
            int(config.random_state)
            + index
        )
        for index, column in enumerate(
            model_columns
        )
    }

    column_order = {
        column: index
        for index, column in enumerate(
            model_columns
        )
    }

    incomplete_targets.sort(
        key=lambda column: (
            int(
                missing_mask[
                    column
                ].sum()
            ),
            column_order[
                column
            ],
        )
    )

    working = _initialise(
        frame=incomplete,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    convergence_rows: list[
        dict[str, object]
    ] = []

    previous_numeric_change = float(
        "inf"
    )

    previous_categorical_change = float(
        "inf"
    )

    # Initial mean/mode fill is iteration zero.
    selected = working.copy(
        deep=True
    )

    selected_iteration = 0
    converged = False

    for iteration in range(
        1,
        config.max_iter + 1,
    ):
        started = perf_counter()

        before_iteration = working.copy(
            deep=True
        )

        for target in incomplete_targets:
            target_missing = (
                missing_mask[
                    target
                ]
            )

            target_observed = (
                ~target_missing
            )

            if not target_missing.any():
                continue

            if not target_observed.any():
                raise ValueError(
                    f"Target '{target}' has no observed training values."
                )

            model, features = _build_model(
                target=target,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                config=config,
                seed=target_seeds[
                    target
                ],
            )

            X_train = working.loc[
                target_observed,
                features,
            ]

            y_train = working.loc[
                target_observed,
                target,
            ]

            X_missing = working.loc[
                target_missing,
                features,
            ]

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_missing
            )

            working.loc[
                target_missing,
                target,
            ] = predictions

        numeric_change = _numeric_change(
            previous=before_iteration,
            current=working,
            missing_mask=missing_mask,
            numeric_columns=numeric_columns,
        )

        categorical_change = _categorical_change(
            previous=before_iteration,
            current=working,
            missing_mask=missing_mask,
            categorical_columns=categorical_columns,
        )

        improvements: list[
            bool
        ] = []

        if numeric_change is not None:
            improvements.append(
                numeric_change
                < previous_numeric_change
            )

        if categorical_change is not None:
            improvements.append(
                categorical_change
                < previous_categorical_change
            )

        # The first forest iteration is always accepted.
        accepted = (
            iteration == 1
            or any(
                improvements
            )
        )

        convergence_rows.append(
            {
                "iteration": iteration,
                "numeric_change": numeric_change,
                "categorical_change": categorical_change,
                "runtime_seconds": (
                    perf_counter()
                    - started
                ),
                "accepted": accepted,
            }
        )

        if not accepted:
            # Do not return the newly worsened solution.
            # Keep the previous accepted imputation.
            working = selected.copy(
                deep=True
            )

            converged = True
            break

        selected = working.copy(
            deep=True
        )

        selected_iteration = iteration

        if numeric_change is not None:
            previous_numeric_change = (
                numeric_change
            )

        if categorical_change is not None:
            previous_categorical_change = (
                categorical_change
            )

    final_imputed = _restore_observed_values(
        original=incomplete,
        imputed=selected,
        missing_mask=missing_mask,
        model_columns=model_columns,
    )

    validate_imputation_result(
        original=incomplete,
        imputed=final_imputed,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    return MissForestResult(
        imputed=final_imputed,
        convergence=pd.DataFrame(
            convergence_rows
        ),
        selected_iteration=selected_iteration,
        converged=converged,
    )