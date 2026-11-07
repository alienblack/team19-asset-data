"""Method registry: one interface for every imputation method.

The pipeline, the experiments and the tests build methods through
:func:`build_imputer` and talk to them only through :class:`BaseImputer`, so a
new method is one class plus one registry entry. Every method returns an
:class:`ImputationOutput` with the same three parts:

``imputed``
    the full input frame with model-column gaps filled (same index, same
    columns, non-model columns untouched);
``diagnostics``
    one row per imputed cell with the method's confidence fields where the
    method has them (NaN otherwise);
``metadata``
    a JSON-serialisable dictionary describing what ran.

Methods
    ``median_mode``           training median / deterministic mode (reference)
    ``missforest``            original transductive MissForest (whole-table)
    ``predictive_missforest`` fit/transform adaptation with frozen replay
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from importlib.util import find_spec
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pkgutil import iter_modules
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ImputationConfig
from .input_validation import validate_imputation_input, validate_imputation_result
from .missforest.missforest import MissForestConfig, missforest_impute
from .postprocess import apply_constraints
from .predictive_missforest.predictive_missforest import PredictiveMissForest, PredictiveMissForestConfig
from .median_mode.simple_imputer import deterministic_mode

DIAGNOSTIC_COLUMNS = (
    "row",
    "column",
    "kind",
    "imputed",
    "prediction_source",
    "raw_prediction",
    "max_proba",
    "margin",
    "entropy",
    "tree_std",
    "tree_p10",
    "tree_p90",
    "material_probabilities",
)


@dataclass
class ImputationOutput:
    """Common result of every method."""

    imputed: pd.DataFrame
    diagnostics: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseImputer(ABC):
    """Interface shared by all methods."""

    name: str = "base"
    #: True when a fitted instance can be saved and later applied to new records.
    supports_persistence: bool = False
    #: True when the method must see the records it fills during fitting
    #: (transductive). Such methods cannot be applied to unseen records.
    transductive: bool = False

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        constraints: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.numeric_columns = tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.constraints = {k: dict(v) for k, v in (constraints or {}).items()}
        self._fitted = False

    @property
    def model_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseImputer":
        """Learn whatever the method needs from ``X``."""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> ImputationOutput:
        """Fill model-column gaps in ``X``."""

    def fit_transform(self, X: pd.DataFrame) -> ImputationOutput:
        return self.fit(X).transform(X)

    def describe(self) -> dict[str, Any]:
        """Configuration summary for run metadata."""
        return {
            "method": self.name,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "supports_persistence": self.supports_persistence,
            "transductive": self.transductive,
        }

    # ------------------------------------------------------------------ #
    # Helpers shared by the concrete methods
    # ------------------------------------------------------------------ #

    def _diagnostics_frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=list(DIAGNOSTIC_COLUMNS))
        return frame

    def _plain_diagnostics(self, X: pd.DataFrame, imputed: pd.DataFrame, source: str) -> pd.DataFrame:
        """Diagnostics for methods without per-cell confidence fields."""
        rows = []
        for column in self.model_columns:
            kind = "numeric" if column in self.numeric_columns else "categorical"
            positions = np.flatnonzero(X[column].isna().to_numpy())
            values = imputed[column].to_numpy()
            for pos in positions:
                rows.append(
                    {
                        "row": X.index[pos],
                        "column": column,
                        "kind": kind,
                        "imputed": values[pos],
                        "prediction_source": source,
                        "raw_prediction": values[pos],
                        "max_proba": np.nan,
                        "margin": np.nan,
                        "entropy": np.nan,
                        "tree_std": np.nan,
                        "tree_p10": np.nan,
                        "tree_p90": np.nan,
                        "material_probabilities": np.nan,
                    }
                )
        return self._diagnostics_frame(rows)


# ---------------------------------------------------------------------------
# median / mode reference
# ---------------------------------------------------------------------------


class MedianModeImputer(BaseImputer):
    """Training median (or mean) for numeric and deterministic mode for categorical columns."""

    name = "median_mode"
    supports_persistence = True
    transductive = False

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        constraints: Mapping[str, Mapping[str, Any]] | None = None,
        numeric_strategy: str = "median",
    ) -> None:
        super().__init__(numeric_columns, categorical_columns, constraints)
        if numeric_strategy not in ("median", "mean"):
            raise ValueError("numeric_strategy must be 'median' or 'mean'.")
        self.numeric_strategy = numeric_strategy
        self.statistics_: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame) -> "MedianModeImputer":
        validate_imputation_input(X, self.numeric_columns, self.categorical_columns)
        stats: dict[str, Any] = {}
        for column in self.numeric_columns:
            observed = pd.to_numeric(X[column], errors="raise").dropna()
            if observed.empty:
                raise ValueError(f"No observed fitting values for '{column}'.")
            stats[column] = float(observed.median() if self.numeric_strategy == "median" else observed.mean())
        for column in self.categorical_columns:
            if X[column].notna().sum() == 0:
                raise ValueError(f"No observed fitting values for '{column}'.")
            stats[column] = deterministic_mode(X[column].astype(object))
        self.statistics_ = stats
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> ImputationOutput:
        if not self._fitted:
            raise RuntimeError("Fit the imputer before transforming.")
        validate_imputation_input(X, self.numeric_columns, self.categorical_columns)
        started = perf_counter()
        out = X.copy(deep=True)
        for column, value in self.statistics_.items():
            if column in self.numeric_columns:
                out[column] = pd.to_numeric(out[column], errors="raise").astype(float).fillna(value)
            else:
                filled = out[column].astype(object)
                out[column] = filled.where(filled.notna(), value)
        out = apply_constraints(X, out, self.constraints)
        validate_imputation_result(X, out, self.numeric_columns, self.categorical_columns)
        diagnostics = self._plain_diagnostics(X, out, "training_statistic")
        metadata = {
            **self.describe(),
            "numeric_strategy": self.numeric_strategy,
            "statistics": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in self.statistics_.items()},
            "runtime_seconds": perf_counter() - started,
        }
        return ImputationOutput(imputed=out, diagnostics=diagnostics, metadata=metadata)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "numeric_strategy": self.numeric_strategy}


# ---------------------------------------------------------------------------
# original (transductive) MissForest
# ---------------------------------------------------------------------------


class LegacyMissForestImputer(BaseImputer):
    """The original whole-table MissForest. Fits and fills the same table."""

    name = "missforest"
    supports_persistence = False
    transductive = True

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        constraints: Mapping[str, Mapping[str, Any]] | None = None,
        config: MissForestConfig | None = None,
    ) -> None:
        super().__init__(numeric_columns, categorical_columns, constraints)
        self.config = config or MissForestConfig()
        self.last_result_ = None

    def fit(self, X: pd.DataFrame) -> "LegacyMissForestImputer":
        # Transductive: there is nothing to learn ahead of transform; the
        # forests are fitted on the table being filled. Validating here gives
        # callers the same error surface as the other methods.
        validate_imputation_input(X, self.numeric_columns, self.categorical_columns)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> ImputationOutput:
        started = perf_counter()
        result = missforest_impute(X, self.numeric_columns, self.categorical_columns, self.config)
        out = apply_constraints(X, result.imputed, self.constraints)
        validate_imputation_result(X, out, self.numeric_columns, self.categorical_columns)
        self.last_result_ = result
        self._fitted = True
        diagnostics = self._plain_diagnostics(X, out, "random_forest_transductive")
        metadata = {
            **self.describe(),
            "config": asdict(self.config),
            "selected_iteration": int(result.selected_iteration),
            "converged": bool(result.converged),
            "convergence": result.convergence.to_dict(orient="records"),
            "runtime_seconds": perf_counter() - started,
        }
        return ImputationOutput(imputed=out, diagnostics=diagnostics, metadata=metadata)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "config": asdict(self.config)}


# ---------------------------------------------------------------------------
# predictive (inductive) MissForest
# ---------------------------------------------------------------------------


class PredictiveMissForestImputer(BaseImputer):
    """Fit once on a reference table, apply to the same or to new records."""

    name = "predictive_missforest"
    supports_persistence = True
    transductive = False

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        constraints: Mapping[str, Mapping[str, Any]] | None = None,
        config: PredictiveMissForestConfig | None = None,
    ) -> None:
        super().__init__(numeric_columns, categorical_columns, constraints)
        self.config = config or PredictiveMissForestConfig()
        self.model = PredictiveMissForest(
            numeric_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
            config=self.config,
            constraints=self.constraints,
        )

    def fit(self, X: pd.DataFrame) -> "PredictiveMissForestImputer":
        self.model.fit(X)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> ImputationOutput:
        if not self._fitted:
            raise RuntimeError("Fit the imputer before transforming.")
        detail = self.model.transform_detailed(X, with_scores=True)
        scores = detail.scores if detail.scores is not None else pd.DataFrame(columns=list(DIAGNOSTIC_COLUMNS))
        diagnostics = scores.reindex(columns=list(DIAGNOSTIC_COLUMNS))
        metadata = {
            **self.describe(),
            "fit_report": _jsonable(self.model.fit_report_),
            "unknown_category_counts": {k: int(v) for k, v in detail.unknown_category_counts.items()},
            "runtime_seconds": float(detail.runtime_seconds),
        }
        return ImputationOutput(imputed=detail.imputed, diagnostics=diagnostics, metadata=metadata)

    @property
    def convergence_(self) -> pd.DataFrame:
        return self.model.convergence_

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "config": asdict(self.config)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHODS: dict[str, type[BaseImputer]] = {
    MedianModeImputer.name: MedianModeImputer,
    LegacyMissForestImputer.name: LegacyMissForestImputer,
    PredictiveMissForestImputer.name: PredictiveMissForestImputer,
}

# Teammates may add ``imputation/<method>/plugin.py`` to register a method,
# keeping the shared evaluation and pipeline code unchanged.
METHOD_FACTORIES: dict[str, Callable[[ImputationConfig, Mapping[str, Any]], BaseImputer]] = {}
_PLUGINS_DISCOVERED = False


def register_method(name: str, factory: Callable[[ImputationConfig, Mapping[str, Any]], BaseImputer]) -> None:
    """Register a teammate's factory; its result must implement ``BaseImputer``."""
    if not name or name in METHODS or name in METHOD_FACTORIES:
        raise ValueError(f"Invalid or duplicate imputation method: {name!r}")
    METHOD_FACTORIES[name] = factory


def discover_plugins() -> None:
    """Import optional method-specific plugin.py modules once per process."""
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED:
        return
    _PLUGINS_DISCOVERED = True
    package = __package__
    for module in iter_modules([str(Path(__file__).parent)]):
        if not module.ispkg:
            continue
        dotted = f"{package}.{module.name}.plugin"
        if find_spec(dotted) is not None:
            import_module(dotted)


def available_methods() -> tuple[str, ...]:
    discover_plugins()
    return tuple(METHODS) + tuple(METHOD_FACTORIES)


def resolve_hyperparameters(
    config: ImputationConfig,
    method: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Defaults from ``methods.<method>`` + ``selected_hyperparameters`` + call-time overrides."""
    if method == "missforest":
        section = dict(config.missforest_config)
    elif method == "predictive_missforest":
        section = dict(config.predictive_missforest_config or {})
        if not section:
            legacy = dict(config.missforest_config)
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
    elif method == "median_mode":
        section = dict(config.median_mode_config)
    elif method in METHOD_FACTORIES or method in available_methods():
        section = dict(config.method_configs.get(method, {}))
    else:
        raise ValueError(f"Unknown method '{method}'. Available: {', '.join(available_methods())}.")
    section.update(config.selected_hyperparameters.get(method, {}))
    if overrides:
        section.update(dict(overrides))
    return section


def build_imputer(
    method: str,
    config: ImputationConfig,
    overrides: Mapping[str, Any] | None = None,
) -> BaseImputer:
    """Instantiate a method from an :class:`ImputationConfig`."""
    discover_plugins()
    if method not in METHODS and method not in METHOD_FACTORIES:
        raise ValueError(f"Unknown method '{method}'. Available: {', '.join(available_methods())}.")
    params = resolve_hyperparameters(config, method, overrides)
    if method in METHOD_FACTORIES:
        imputer = METHOD_FACTORIES[method](config, params)
        if not isinstance(imputer, BaseImputer):
            raise TypeError(f"Plugin '{method}' did not return a BaseImputer instance.")
        return imputer
    numeric, categorical, constraints = config.numeric_columns, config.categorical_columns, config.constraints
    if method == "median_mode":
        return MedianModeImputer(numeric, categorical, constraints, numeric_strategy=params.get("numeric_strategy", "median"))
    if method == "missforest":
        allowed = set(MissForestConfig.__dataclass_fields__)
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(f"Unknown missforest configuration keys: {unknown}")
        return LegacyMissForestImputer(numeric, categorical, constraints, config=MissForestConfig(**params))
    return PredictiveMissForestImputer(
        numeric, categorical, constraints, config=PredictiveMissForestConfig.from_mapping(params)
    )
