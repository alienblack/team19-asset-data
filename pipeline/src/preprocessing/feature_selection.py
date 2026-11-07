"""Feature selection and the leakage guard.

The imputation methods receive the whole record frame and work only on the
model columns declared in ``imputation.yaml``. This module makes sure that
declaration agrees with ``data_schema.yaml``: only the approved attributes can
be model columns, every forbidden column is explicitly excluded, and the
targets are present. Nothing here engineers new features; derived fields are
computed only after imputation (see :mod:`derived_fields`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..ingestion.schema import DataSchema


class LeakageError(ValueError):
    """Raised when a forbidden column could reach an imputer as a predictor."""


class FeatureSelectionError(ValueError):
    """Raised when the imputation configuration disagrees with the schema."""


@dataclass
class ModelFrame:
    """A frame ready for an imputer, with the roles made explicit."""

    frame: pd.DataFrame
    id_column: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    passthrough_columns: tuple[str, ...]

    @property
    def model_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns


def check_config_against_schema(imputation_config: Any, schema: DataSchema) -> list[str]:
    """Return a list of problems (empty when the configuration is consistent)."""
    problems: list[str] = []
    numeric = tuple(imputation_config.numeric_columns)
    categorical = tuple(imputation_config.categorical_columns)
    excluded = tuple(imputation_config.excluded_columns)
    model = numeric + categorical

    for name in model:
        if name not in schema.columns:
            problems.append(f"model column '{name}' is not declared in the schema")
            continue
        if name not in schema.allowed_predictors:
            problems.append(f"model column '{name}' is not an approved predictor")
        spec = schema.column(name)
        if name in numeric and not spec.is_numeric:
            problems.append(f"'{name}' is numeric in imputation.yaml but {spec.dtype} in the schema")
        if name in categorical and not spec.is_categorical:
            problems.append(f"'{name}' is categorical in imputation.yaml but {spec.dtype} in the schema")

    for name in schema.allowed_predictors:
        if name not in model:
            problems.append(f"approved attribute '{name}' is not a model column; it will neither be used nor filled")

    for name in schema.targets:
        if name not in model:
            problems.append(f"target '{name}' is not a model column")

    for name in schema.forbidden_predictors:
        if name in model:
            problems.append(f"forbidden predictor '{name}' is a model column")
        elif name not in excluded:
            problems.append(f"forbidden predictor '{name}' is not listed under columns.excluded in imputation.yaml")

    return problems


def assert_no_leakage(model_columns: Sequence[str], schema: DataSchema) -> None:
    """Raise :class:`LeakageError` if any forbidden column is among ``model_columns``."""
    leaked = [c for c in model_columns if c in schema.forbidden_predictors]
    if leaked:
        raise LeakageError(
            "Forbidden columns would be used as predictors: "
            + ", ".join(leaked)
            + ". See forbidden_predictors in data_schema.yaml."
        )
    not_approved = [c for c in model_columns if c not in schema.allowed_predictors]
    if not_approved:
        raise LeakageError(
            "Columns outside the approved attribute list would be used as predictors: " + ", ".join(not_approved)
        )


def select_model_frame(frame: pd.DataFrame, schema: DataSchema, imputation_config: Any) -> ModelFrame:
    """Validate roles and return a typed frame ordered id, model columns, rest.

    Raises :class:`FeatureSelectionError` when the configuration and schema
    disagree and :class:`LeakageError` when a forbidden column would be used.
    """
    problems = check_config_against_schema(imputation_config, schema)
    hard = [p for p in problems if "forbidden" in p or "not an approved" in p or "not declared" in p or "target" in p or "is numeric" in p or "is categorical" in p]
    if hard:
        raise FeatureSelectionError("imputation.yaml disagrees with data_schema.yaml: " + "; ".join(hard))

    numeric = tuple(imputation_config.numeric_columns)
    categorical = tuple(imputation_config.categorical_columns)
    assert_no_leakage(numeric + categorical, schema)

    missing = [c for c in numeric + categorical if c not in frame.columns]
    if missing:
        raise FeatureSelectionError(f"Model columns missing from the data: {missing}")
    if schema.id_column not in frame.columns:
        raise FeatureSelectionError(f"Identifier column '{schema.id_column}' missing from the data.")

    out = frame.copy(deep=True)
    for name in numeric:
        out[name] = pd.to_numeric(out[name], errors="coerce").astype(float)
    for name in categorical:
        values = out[name].astype(object)
        out[name] = values.where(values.notna(), np.nan).map(lambda v: v if (isinstance(v, float) and np.isnan(v)) else str(v)).astype(object)

    ordered = [schema.id_column, *numeric, *categorical]
    rest = [c for c in out.columns if c not in ordered]
    out = out.loc[:, ordered + rest]
    return ModelFrame(
        frame=out,
        id_column=schema.id_column,
        numeric_columns=numeric,
        categorical_columns=categorical,
        passthrough_columns=tuple(rest),
    )
