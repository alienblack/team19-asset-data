"""Data schema model loaded from ``config/<asset>/data_schema.yaml``.

The schema is the single description of one asset class's records. Every other
stage reads column roles, types, allowed values and the leakage lists from
here rather than hard-coding column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

VALID_DTYPES = ("string", "integer", "float", "category")
VALID_ROLES = ("identifier", "group", "target", "predictor", "derived", "passthrough")
MODEL_ROLES = ("target", "predictor")
DERIVE_METHODS = ("snapshot_year_minus",)


class SchemaError(ValueError):
    """Raised when data_schema.yaml is malformed or internally inconsistent."""


@dataclass(frozen=True)
class ColumnSpec:
    """Declaration of one column."""

    name: str
    dtype: str
    role: str
    required: bool = True
    allowed: tuple[str, ...] | None = None
    min: float | None = None
    max: float | None = None
    sentinel_values: tuple[float, ...] = ()
    derive: Mapping[str, str] | None = None

    @property
    def is_numeric(self) -> bool:
        return self.dtype in ("integer", "float")

    @property
    def is_categorical(self) -> bool:
        return self.dtype == "category"

    @property
    def is_model_column(self) -> bool:
        return self.role in MODEL_ROLES


@dataclass(frozen=True)
class DataSchema:
    """Validated schema for one asset class."""

    version: int
    asset_class: str
    id_column: str
    snapshot_year: int
    missing_tokens: tuple[str, ...]
    column_mapping: dict[str, str]
    columns: dict[str, ColumnSpec]
    targets: tuple[str, ...]
    allowed_predictors: tuple[str, ...]
    forbidden_predictors: tuple[str, ...]
    source_path: Path | None = field(default=None, compare=False)
    id_generation: str = "existing"
    id_alignment_columns: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    # Convenience views
    # ------------------------------------------------------------------ #

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns.values() if c.required)

    @property
    def model_columns(self) -> tuple[str, ...]:
        """Approved model attributes in schema order (targets + predictors)."""
        return tuple(c.name for c in self.columns.values() if c.is_model_column)

    @property
    def numeric_model_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns.values() if c.is_model_column and c.is_numeric)

    @property
    def categorical_model_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns.values() if c.is_model_column and c.is_categorical)

    @property
    def derived_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns.values() if c.role == "derived")

    @property
    def group_column(self) -> str | None:
        for spec in self.columns.values():
            if spec.role == "group":
                return spec.name
        return None

    @property
    def primary_targets(self) -> tuple[str, ...]:
        """The attributes the project is judged on (headline metrics, tuning weights)."""
        return self.targets

    @property
    def model_attributes(self) -> tuple[str, ...]:
        """The approved attributes every imputer may use and must fill."""
        return self.allowed_predictors

    def linked_columns(self) -> dict[str, tuple[str, ...]]:
        """``{source: (derived columns computed from it, ...)}``.

        Evaluation masks hide the linked derived cells together with a hidden
        source cell, so a derived value can never reveal a hidden truth.
        """
        links: dict[str, list[str]] = {}
        for spec in self.columns.values():
            if spec.role == "derived":
                links.setdefault(spec.derive["source"], []).append(spec.name)  # type: ignore[index]
        return {source: tuple(targets) for source, targets in links.items()}

    def recoverable_sources(self) -> dict[str, str]:
        """``{source: derived}`` for derived columns declared with ``recover_source: true``."""
        return {
            spec.derive["source"]: spec.name  # type: ignore[index]
            for spec in self.columns.values()
            if spec.role == "derived" and spec.derive and spec.derive.get("recover_source")
        }

    def constraints(self) -> dict[str, dict[str, Any]]:
        """Post-imputation constraints for numeric model attributes, derived from
        the column declarations: ``{column: {"data_type", "min", "max"}}``.

        This is the single source of the ranges applied to imputed cells;
        ``imputation.yaml`` may repeat them only for older callers and must
        then agree (``imputation.config.load_imputation_config`` checks).
        """
        out: dict[str, dict[str, Any]] = {}
        for spec in self.columns.values():
            if spec.is_model_column and spec.is_numeric:
                rule: dict[str, Any] = {"data_type": spec.dtype}
                if spec.min is not None:
                    rule["min"] = int(spec.min) if spec.dtype == "integer" and float(spec.min).is_integer() else spec.min
                if spec.max is not None:
                    rule["max"] = int(spec.max) if spec.dtype == "integer" and float(spec.max).is_integer() else spec.max
                out[spec.name] = rule
        return out

    def column(self, name: str) -> ColumnSpec:
        try:
            return self.columns[name]
        except KeyError as exc:
            raise SchemaError(f"Column '{name}' is not declared in the schema.") from exc

    def missing_token_set(self) -> frozenset[str]:
        return frozenset(token.strip().lower() for token in self.missing_tokens)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_column(name: str, raw: Mapping[str, Any], snapshot_year: int | None = None) -> ColumnSpec:
    if not isinstance(raw, Mapping):
        raise SchemaError(f"Column '{name}' must be a mapping.")
    dtype = raw.get("dtype")
    if dtype not in VALID_DTYPES:
        raise SchemaError(f"Column '{name}': dtype must be one of {VALID_DTYPES}, got {dtype!r}.")
    role = raw.get("role")
    if role not in VALID_ROLES:
        raise SchemaError(f"Column '{name}': role must be one of {VALID_ROLES}, got {role!r}.")
    required = raw.get("required", True)
    if type(required) is not bool:
        raise SchemaError(f"Column '{name}': required must be true or false.")

    allowed = raw.get("allowed")
    if allowed is not None:
        if dtype != "category":
            raise SchemaError(f"Column '{name}': 'allowed' is only valid for category columns.")
        if not isinstance(allowed, list) or not allowed or not all(isinstance(v, str) and v.strip() for v in allowed):
            raise SchemaError(f"Column '{name}': 'allowed' must be a non-empty list of strings.")
        if len(set(allowed)) != len(allowed):
            raise SchemaError(f"Column '{name}': 'allowed' contains duplicates.")
        allowed = tuple(allowed)
    elif dtype == "category" and role in MODEL_ROLES:
        raise SchemaError(f"Column '{name}': category model columns must declare 'allowed' values.")

    minimum, maximum = raw.get("min"), raw.get("max")
    if isinstance(maximum, str):
        if maximum.strip() != "snapshot_year":
            raise SchemaError(f"Column '{name}': 'max' must be a number or the sentinel 'snapshot_year'.")
        if snapshot_year is None:
            raise SchemaError(f"Column '{name}': 'max: snapshot_year' needs a snapshot_year in the schema.")
        maximum = snapshot_year
    for label, value in (("min", minimum), ("max", maximum)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise SchemaError(f"Column '{name}': '{label}' must be a number.")
        if value is not None and dtype not in ("integer", "float"):
            raise SchemaError(f"Column '{name}': '{label}' is only valid for numeric columns.")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise SchemaError(f"Column '{name}': min exceeds max.")

    sentinels = raw.get("sentinel_values", [])
    if not isinstance(sentinels, list) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in sentinels
    ):
        raise SchemaError(f"Column '{name}': sentinel_values must be a list of numbers.")
    if sentinels and dtype not in ("integer", "float"):
        raise SchemaError(f"Column '{name}': sentinel_values are only valid for numeric columns.")

    derive = raw.get("derive")
    if role == "derived":
        if not isinstance(derive, Mapping) or derive.get("method") not in DERIVE_METHODS:
            raise SchemaError(
                f"Column '{name}': derived columns need derive.method in {DERIVE_METHODS}."
            )
        if not isinstance(derive.get("source"), str):
            raise SchemaError(f"Column '{name}': derive.source must name a column.")
        recover = derive.get("recover_source", False)
        if type(recover) is not bool:
            raise SchemaError(f"Column '{name}': derive.recover_source must be true or false.")
        derive = {"method": derive["method"], "source": derive["source"], "recover_source": recover}
    elif derive is not None:
        raise SchemaError(f"Column '{name}': only derived columns may define 'derive'.")

    return ColumnSpec(
        name=name,
        dtype=dtype,
        role=role,
        required=required,
        allowed=allowed,
        min=None if minimum is None else float(minimum),
        max=None if maximum is None else float(maximum),
        sentinel_values=tuple(float(v) for v in sentinels),
        derive=derive,
    )


def _string_list(raw: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    value = raw.get(key, [] if allow_empty else None)
    if not isinstance(value, list) or (not value and not allow_empty):
        raise SchemaError(f"'{key}' must be a {'' if allow_empty else 'non-empty '}list.")
    if not all(isinstance(v, str) and v.strip() for v in value):
        raise SchemaError(f"'{key}' must contain only non-empty strings.")
    if len(set(value)) != len(value):
        raise SchemaError(f"'{key}' contains duplicates.")
    return tuple(value)


def parse_schema(raw: Mapping[str, Any], source_path: Path | None = None) -> DataSchema:
    """Build and cross-check a :class:`DataSchema` from a YAML mapping."""
    if not isinstance(raw, Mapping):
        raise SchemaError("Schema file must contain a YAML mapping.")

    version = raw.get("version")
    if type(version) is not int or version < 1:
        raise SchemaError("'version' must be a positive integer.")
    asset_class = raw.get("asset_class")
    if not isinstance(asset_class, str) or not asset_class.strip():
        raise SchemaError("'asset_class' must be a non-empty string.")
    id_column = raw.get("id_column")
    if not isinstance(id_column, str) or not id_column.strip():
        raise SchemaError("'id_column' must be a non-empty string.")
    id_generation = raw.get("id_generation", "existing")
    if id_generation not in ("existing", "row_number"):
        raise SchemaError("'id_generation' must be 'existing' or 'row_number'.")
    alignment_columns = _string_list(raw, "id_alignment_columns", allow_empty=True)
    if alignment_columns and id_generation != "row_number":
        raise SchemaError("id_alignment_columns is only valid with id_generation: row_number.")
    snapshot_year = raw.get("snapshot_year")
    if type(snapshot_year) is not int or not 1800 <= snapshot_year <= 2200:
        raise SchemaError("'snapshot_year' must be an integer year.")

    missing_tokens = raw.get("missing_tokens", [""])
    if not isinstance(missing_tokens, list) or not all(isinstance(t, str) for t in missing_tokens):
        raise SchemaError("'missing_tokens' must be a list of strings.")

    mapping = raw.get("column_mapping") or {}
    if not isinstance(mapping, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
    ):
        raise SchemaError("'column_mapping' must map source names to canonical names.")
    if len(set(mapping.values())) != len(mapping):
        raise SchemaError("'column_mapping' maps two source columns to the same canonical name.")

    columns_raw = raw.get("columns")
    if not isinstance(columns_raw, Mapping) or not columns_raw:
        raise SchemaError("'columns' must be a non-empty mapping.")
    columns = {name: _parse_column(name, spec, snapshot_year) for name, spec in columns_raw.items()}

    if id_column not in columns or columns[id_column].role != "identifier":
        raise SchemaError(f"id_column '{id_column}' must be declared with role 'identifier'.")
    if sum(spec.role == "identifier" for spec in columns.values()) != 1:
        raise SchemaError("Exactly one column must have role 'identifier'.")
    if sum(spec.role == "group" for spec in columns.values()) > 1:
        raise SchemaError("At most one column may have role 'group'.")
    for column in alignment_columns:
        if column not in columns or column == id_column:
            raise SchemaError(f"Alignment column '{column}' must be declared and cannot be the identifier.")

    for spec in columns.values():
        if spec.role == "derived":
            source = spec.derive["source"]  # type: ignore[index]
            if source not in columns:
                raise SchemaError(f"Derived column '{spec.name}' refers to unknown source '{source}'.")
            if not columns[source].is_numeric or not spec.is_numeric:
                raise SchemaError(f"Derived column '{spec.name}' and its source must be numeric.")
            if spec.derive["method"] == "snapshot_year_minus":  # type: ignore[index]
                source_max = columns[source].max
                if source_max is None or source_max > snapshot_year:
                    raise SchemaError(
                        f"Column '{source}' is the source of '{spec.name}' = snapshot_year - {source}; "
                        f"its 'max' must be set and must not exceed snapshot_year ({snapshot_year}). "
                        "Write 'max: snapshot_year' to bind them."
                    )

    for target_name, canonical in mapping.items():
        if canonical not in columns:
            raise SchemaError(f"column_mapping target '{canonical}' is not a declared column.")
        if target_name in columns and target_name != canonical:
            raise SchemaError(
                f"column_mapping source '{target_name}' is itself a declared column; mapping would overwrite it."
            )

    def _either(new_key: str, old_key: str) -> tuple[str, ...]:
        if new_key in raw and old_key in raw:
            raise SchemaError(f"Use either '{new_key}' or its older spelling '{old_key}', not both.")
        return _string_list(raw, new_key if new_key in raw else old_key)

    targets = _either("primary_targets", "targets")
    allowed = _either("model_attributes", "allowed_predictors")
    forbidden = _string_list(raw, "forbidden_predictors", allow_empty=True)

    model_columns = {name for name, spec in columns.items() if spec.is_model_column}
    for name in targets:
        if name not in columns or columns[name].role != "target":
            raise SchemaError(f"Target '{name}' must be a declared column with role 'target'.")
    if set(allowed) != model_columns:
        raise SchemaError(
            "model_attributes must list exactly the columns with role target/predictor; "
            f"schema roles give {sorted(model_columns)}, model_attributes gives {sorted(allowed)}."
        )
    overlap = sorted(set(allowed) & set(forbidden))
    if overlap:
        raise SchemaError(f"Columns cannot be both allowed and forbidden predictors: {overlap}.")
    for name in forbidden:
        if name in columns and columns[name].is_model_column:
            raise SchemaError(f"Forbidden predictor '{name}' has a model role in 'columns'.")
    non_model = {name for name, spec in columns.items() if not spec.is_model_column}
    unlisted = sorted(non_model - set(forbidden))
    if unlisted:
        raise SchemaError(
            "Every non-model column must be listed under forbidden_predictors so the "
            f"leakage guard is explicit; missing: {unlisted}."
        )

    return DataSchema(
        version=version,
        asset_class=asset_class.strip(),
        id_column=id_column,
        snapshot_year=snapshot_year,
        missing_tokens=tuple(missing_tokens),
        column_mapping=dict(mapping),
        columns=columns,
        targets=targets,
        allowed_predictors=allowed,
        forbidden_predictors=forbidden,
        id_generation=id_generation,
        id_alignment_columns=alignment_columns,
        source_path=source_path,
    )


def load_schema(path: str | Path) -> DataSchema:
    """Load ``data_schema.yaml``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_schema(raw, source_path=path.resolve())
