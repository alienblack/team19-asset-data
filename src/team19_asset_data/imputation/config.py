"""Generic configuration loader for asset-data imputation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml




@dataclass(frozen=True)
class DataSplitConfig:
    """Record partition proportions; used only by tuning/test workflows."""

    train: float
    validation: float
    calibration: float
    test: float
    random_seed: int
    strategy: str = "row"  # "row" or "group"
    group_column: str | None = None


@dataclass(frozen=True)
class WorkflowConfig:
    """Workflow settings, independent of the PredictiveMissForest estimator."""

    execution_mode: str
    model_path: str
    tuning_enabled: bool
    tuning_mask_rate: float
    tuning_mask_seed: int
    tuning_grid: dict[str, list[Any]]
    evaluation_enabled: bool
    evaluation_mask_rate: float
    evaluation_mask_seed: int
    data_split: DataSplitConfig
    # Default retains original two-target selection for older callers.
    tuning_objective: str = "year_material"


def _strict_bool(section: dict[str, Any], name: str, default: bool) -> bool:
    value = section.get(name, default)
    if type(value) is not bool:
        raise ValueError(f"{name} must be true or false (boolean).")
    return value


def _strict_seed(section: dict[str, Any], name: str, default: int) -> int:
    value = section.get(name, default)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _strict_rate(section: dict[str, Any], name: str, default: float) -> float:
    value = section.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between zero and one.")
    value = float(value)
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between zero and one.")
    return value


def _optional_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return value


def _parse_workflow(raw: dict[str, Any]) -> WorkflowConfig | None:
    """Parse new workflow options only when present; older YAML remains valid."""
    keys = ("execution", "tuning", "evaluation", "data_split")
    if not any(k in raw for k in keys):
        return None
    execution = _optional_mapping(raw, "execution")
    tuning = _optional_mapping(raw, "tuning")
    evaluation = _optional_mapping(raw, "evaluation")
    split = _optional_mapping(raw, "data_split")

    mode = execution.get("mode", "fit_transform")
    if mode not in {"fit_only", "fit_transform", "transform_only", "evaluate_only"}:
        raise ValueError("execution.mode must be one of fit_only, fit_transform, "
                         "transform_only, evaluate_only.")
    model_path = execution.get("model_path", "outputs/models/predictive_missforest.joblib")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("execution.model_path must be a nonempty string.")

    tune_enabled = _strict_bool(tuning, "enabled", False)
    eval_enabled = _strict_bool(evaluation, "enabled", False)
    tuning_objective = tuning.get("objective", "year_material")
    if not isinstance(tuning_objective, str) or tuning_objective not in {
        "year_material", "all_attributes"
    }:
        raise ValueError(
            "tuning.objective must be 'year_material' or 'all_attributes'."
        )
    if mode in {"transform_only", "evaluate_only"} and tune_enabled:
        raise ValueError(f"Tuning cannot run in execution.mode={mode}: it requires fit().")
    if mode == "evaluate_only" and not eval_enabled:
        raise ValueError("execution.mode=evaluate_only requires evaluation.enabled: true.")

    grid = tuning.get("grid", {"max_depth": [4, 8, 16, None], "min_samples_leaf": [1, 2, 10]})
    if not isinstance(grid, dict) or set(grid) != {"max_depth", "min_samples_leaf"}:
        raise ValueError("tuning.grid must contain max_depth and min_samples_leaf lists.")
    depths = grid["max_depth"]
    leaves = grid["min_samples_leaf"]
    if (not isinstance(depths, list) or not depths or
        any((d is not None and (type(d) is not int or d < 1)) for d in depths)):
        raise ValueError("tuning.grid.max_depth must contain positive integers or null.")
    if (not isinstance(leaves, list) or not leaves or
        any(type(leaf) is not int or leaf < 1 for leaf in leaves)):
        raise ValueError("tuning.grid.min_samples_leaf must contain positive integers.")

    ratios = {}
    for name, default in (("train", 0.60), ("validation", 0.10),
                          ("calibration", 0.10), ("test", 0.20)):
        value = split.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"data_split.{name} must be a finite number in [0, 1].")
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError(f"data_split.{name} must be in [0, 1].")
        ratios[name] = value
    if abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("data_split proportions must total 1.0.")
    if mode in {"fit_only", "fit_transform"}:
        if tune_enabled and (not ratios["train"] or not ratios["validation"]):
            raise ValueError("Tuning requires nonzero train and validation splits.")
        if eval_enabled and (not ratios["train"] or not ratios["test"]):
            raise ValueError("Independent test evaluation requires nonzero train and test splits.")

    strategy = split.get("strategy", "row")
    group_column = split.get("group_column", None)
    if strategy not in {"row", "group"}:
        raise ValueError("data_split.strategy must be row or group.")
    if strategy == "group" and (not isinstance(group_column, str) or not group_column.strip()):
        raise ValueError("data_split.group_column is required for group strategy.")
    if group_column is not None and (not isinstance(group_column, str) or not group_column.strip()):
        raise ValueError("data_split.group_column must be a nonempty string or null.")

    return WorkflowConfig(
        execution_mode=mode,
        model_path=model_path,
        tuning_enabled=tune_enabled,
        tuning_mask_rate=_strict_rate(tuning, "mask_rate", 0.10),
        tuning_mask_seed=_strict_seed(tuning, "mask_seed", 219),
        tuning_grid={"max_depth": list(depths), "min_samples_leaf": list(leaves)},
        tuning_objective=tuning_objective,
        evaluation_enabled=eval_enabled,
        evaluation_mask_rate=_strict_rate(evaluation, "mask_rate", 0.10),
        evaluation_mask_seed=_strict_seed(evaluation, "mask_seed", 319),
        data_split=DataSplitConfig(
            **ratios,
            random_seed=_strict_seed(split, "random_seed", 219),
            strategy=strategy,
            group_column=group_column,
        ),
    )


@dataclass(frozen=True)
class ImputationConfig:
    """Validated configuration for one asset class."""

    version: int
    asset_class: str

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]

    constraints: dict[str, dict[str, Any]]

    simple_config: dict[str, Any]
    missforest_config: dict[str, Any]

    # Optional ``methods.predictive_missforest`` section. ``None`` when the
    # YAML has no such section (all historical configurations), so existing
    # callers and positional construction are unaffected.
    predictive_missforest_config: dict[str, Any] | None = None
    workflow_config: WorkflowConfig | None = None

    @property
    def model_columns(self) -> tuple[str, ...]:
        """All columns available to imputation methods."""
        return self.numeric_columns + self.categorical_columns


def _require_mapping(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return a required dictionary section."""

    value = config.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"Configuration section '{key}' must be a mapping."
        )

    return value


def _require_string_list(
    config: dict[str, Any],
    key: str,
) -> tuple[str, ...]:
    """Return a configuration value as a validated tuple of strings."""

    value = config.get(key)

    if not isinstance(value, list):
        raise ValueError(
            f"Configuration field '{key}' must be a list."
        )

    if not all(
        isinstance(item, str) and item.strip()
        for item in value
    ):
        raise ValueError(
            f"Configuration field '{key}' must contain only non-empty strings."
        )

    if len(value) != len(set(value)):
        raise ValueError(
            f"Configuration field '{key}' contains duplicate columns."
        )

    return tuple(value)


def _validate_column_groups(
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    excluded_columns: tuple[str, ...],
) -> None:
    """Ensure a column does not have conflicting roles."""

    numeric = set(numeric_columns)
    categorical = set(categorical_columns)
    excluded = set(excluded_columns)

    overlap_numeric_categorical = sorted(
        numeric.intersection(categorical)
    )

    if overlap_numeric_categorical:
        raise ValueError(
            "Columns cannot be both numeric and categorical: "
            + ", ".join(overlap_numeric_categorical)
        )

    model_columns = numeric.union(categorical)

    overlap_excluded = sorted(
        model_columns.intersection(excluded)
    )

    if overlap_excluded:
        raise ValueError(
            "Model columns cannot also be excluded: "
            + ", ".join(overlap_excluded)
        )

    if not model_columns:
        raise ValueError(
            "At least one numeric or categorical model column is required."
        )


def _validate_constraints(
    constraints: dict[str, Any],
    model_columns: set[str],
) -> None:
    """Validate generic post-processing constraints."""

    supported_types = {
        "integer",
        "float",
        "categorical",
    }

    for column, rule in constraints.items():

        if column not in model_columns:
            raise ValueError(
                f"Constraint defined for unknown model column '{column}'."
            )

        if not isinstance(rule, dict):
            raise ValueError(
                f"Constraint for '{column}' must be a mapping."
            )

        data_type = rule.get("data_type")

        if data_type not in supported_types:
            raise ValueError(
                f"Unsupported data_type '{data_type}' "
                f"for column '{column}'."
            )

        minimum = rule.get("min")
        maximum = rule.get("max")

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError(
                f"Constraint minimum exceeds maximum "
                f"for column '{column}'."
            )


def _validate_methods(
    methods: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate method-specific configuration sections."""

    simple = methods.get("simple")
    missforest = methods.get("missforest")

    if not isinstance(simple, dict):
        raise ValueError(
            "Configuration must contain methods.simple."
        )

    if not isinstance(missforest, dict):
        raise ValueError(
            "Configuration must contain methods.missforest."
        )

    numeric_strategy = simple.get(
        "numeric_strategy"
    )

    categorical_strategy = simple.get(
        "categorical_strategy"
    )

    if numeric_strategy != "mean":
        raise ValueError(
            "Currently supported simple numeric strategy is 'mean'."
        )

    if categorical_strategy != "mode":
        raise ValueError(
            "Currently supported simple categorical strategy is 'mode'."
        )

    n_estimators = missforest.get("n_estimators")

    if (
        not isinstance(n_estimators, int)
        or n_estimators < 1
    ):
        raise ValueError(
            "missforest.n_estimators must be a positive integer."
        )

    max_iter = missforest.get("max_iter")

    if (
        not isinstance(max_iter, int)
        or max_iter < 1
    ):
        raise ValueError(
            "missforest.max_iter must be a positive integer."
        )

    return dict(simple), dict(missforest)


def _validate_predictive_section(
    methods: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate the optional ``methods.predictive_missforest`` section.

    Only structural checks live here; value validation is done by
    ``imputation.predictive_missforest.validate_predictive_config`` so the two
    cannot drift apart.
    """

    section = methods.get("predictive_missforest")

    if section is None:
        return None

    if not isinstance(section, dict):
        raise ValueError(
            "methods.predictive_missforest must be a mapping when present."
        )

    n_passes = section.get("n_passes")

    if (
        not isinstance(n_passes, int)
        or isinstance(n_passes, bool)
        or n_passes < 1
    ):
        raise ValueError(
            "predictive_missforest.n_passes must be a positive integer."
        )

    return dict(section)


def load_imputation_config(
    path: str | Path,
) -> ImputationConfig:
    """Load and validate an asset-specific imputation YAML file."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Imputation configuration not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(
            "Imputation configuration must contain a YAML mapping."
        )

    version = raw.get("version")

    if not isinstance(version, int) or version < 1:
        raise ValueError(
            "Configuration 'version' must be a positive integer."
        )

    asset_class = raw.get("asset_class")

    if (
        not isinstance(asset_class, str)
        or not asset_class.strip()
    ):
        raise ValueError(
            "Configuration 'asset_class' must be a non-empty string."
        )

    columns = _require_mapping(
        raw,
        "columns",
    )

    numeric_columns = _require_string_list(
        columns,
        "numeric",
    )

    categorical_columns = _require_string_list(
        columns,
        "categorical",
    )

    excluded_columns = _require_string_list(
        columns,
        "excluded",
    )

    _validate_column_groups(
        numeric_columns,
        categorical_columns,
        excluded_columns,
    )

    constraints = raw.get(
        "constraints",
        {},
    )

    if not isinstance(constraints, dict):
        raise ValueError(
            "Configuration section 'constraints' must be a mapping."
        )

    _validate_constraints(
        constraints,
        set(numeric_columns).union(
            categorical_columns
        ),
    )

    methods = _require_mapping(
        raw,
        "methods",
    )

    simple_config, missforest_config = (
        _validate_methods(methods)
    )

    predictive_missforest_config = _validate_predictive_section(
        methods
    )

    return ImputationConfig(
        version=version,
        asset_class=asset_class.strip(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        excluded_columns=excluded_columns,
        constraints={
            key: dict(value)
            for key, value in constraints.items()
        },
        simple_config=simple_config,
        missforest_config=missforest_config,
        predictive_missforest_config=predictive_missforest_config,
        workflow_config=_parse_workflow(raw),
    )