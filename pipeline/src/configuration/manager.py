"""Configuration manager: load, validate, cross-check and gate the four YAML files.

The manager wraps the existing parsers (``parse_schema``,
``parse_cleaning_rules``, ``load_imputation_config``, ``load_settings``) and
adds what they cannot see alone: cross-file consistency, the approval state of
each file (``approval.ApprovalStore``) and the release gate that refuses to run
a stage on invalid or unapproved configuration.

Typical use::

    manager = ConfigurationManager.from_pipeline_yaml("config/pipeline.yaml")
    check = manager.check()              # states + findings, nothing changes
    manager.approve("data_schema.yaml", by="J. Smith", note="snapshot year confirmed")
    manager.require("release")           # raises ConfigurationError if not clean
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..imputation.config import ImputationConfig, load_imputation_config
from ..imputation.registry import available_methods
from ..ingestion.schema import DataSchema, SchemaError, load_schema
from ..preprocessing.feature_selection import check_config_against_schema
from ..quality.cleaner import APPROVAL_FAMILIES, CleaningError, CleaningRules, load_cleaning_rules
from ..settings import PipelineSettings, SettingsError, load_settings
from .approval import APPROVED, ApprovalError, ApprovalState, ApprovalStore

ERROR, WARNING, INFO = "error", "warning", "info"


class ConfigurationError(RuntimeError):
    """Raised by :meth:`ConfigurationManager.require` when a stage must not run."""


@dataclass(frozen=True)
class Finding:
    level: str
    file: str
    message: str


@dataclass
class LoadedConfiguration:
    settings: PipelineSettings
    schema: DataSchema | None = None
    cleaning_rules: CleaningRules | None = None
    imputation: ImputationConfig | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == ERROR]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass
class ConfigurationCheck:
    loaded: LoadedConfiguration
    states: dict[str, ApprovalState]

    @property
    def findings(self) -> list[Finding]:
        return self.loaded.findings

    @property
    def is_valid(self) -> bool:
        return self.loaded.is_valid

    @property
    def all_approved(self) -> bool:
        return all(state.status == APPROVED for state in self.states.values())

    @property
    def ready(self) -> bool:
        """Valid and approved: a release-mode stage may run."""
        return self.is_valid and self.all_approved

    def states_table(self) -> pd.DataFrame:
        rows = []
        for name, state in self.states.items():
            rows.append(
                {
                    "file": name,
                    "status": state.status,
                    "sha256": (state.sha256 or "")[:12],
                    "approved_by": state.record.approved_by if state.record else "",
                    "approved_at": state.record.approved_at if state.record else "",
                    "note": state.record.note if state.record else "",
                }
            )
        return pd.DataFrame(rows, columns=["file", "status", "sha256", "approved_by", "approved_at", "note"])

    def findings_table(self) -> pd.DataFrame:
        rows = [{"level": f.level, "file": f.file, "message": f.message} for f in self.findings]
        return pd.DataFrame(rows, columns=["level", "file", "message"])

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "valid": self.is_valid,
            "all_approved": self.all_approved,
            "ready": self.ready,
            "files": {
                name: {"status": s.status, "sha256": s.sha256, "approved_by": s.record.approved_by if s.record else None, "approved_at": s.record.approved_at if s.record else None}
                for name, s in self.states.items()
            },
            "findings": [{"level": f.level, "file": f.file, "message": f.message} for f in self.findings],
        }


class ConfigurationManager:
    """Loads, validates, cross-checks and gates the configuration files."""

    def __init__(self, settings: PipelineSettings, approvals_path: str | Path | None = None) -> None:
        self.settings = settings
        self.store = ApprovalStore(Path(approvals_path) if approvals_path is not None else settings.approvals_path)

    @classmethod
    def from_pipeline_yaml(cls, path: str | Path | None = None, approvals_path: str | Path | None = None) -> "ConfigurationManager":
        return cls(load_settings(path), approvals_path)

    # ------------------------------------------------------------------ #
    # Loading and validation
    # ------------------------------------------------------------------ #

    @property
    def files(self) -> dict[str, Path]:
        return self.settings.config_files

    def load(self) -> LoadedConfiguration:
        """Parse every file and run the cross-file checks; never raises on content."""
        loaded = LoadedConfiguration(settings=self.settings)
        findings = loaded.findings

        for name, path in self.files.items():
            if not path.is_file():
                findings.append(Finding(ERROR, name, f"file not found: {path}"))

        try:
            loaded.schema = load_schema(self.settings.data_schema)
        except (SchemaError, FileNotFoundError, ValueError) as exc:
            findings.append(Finding(ERROR, "data_schema.yaml", str(exc)))

        try:
            loaded.cleaning_rules = load_cleaning_rules(self.settings.cleaning_rules)
        except (CleaningError, FileNotFoundError, ValueError) as exc:
            findings.append(Finding(ERROR, "cleaning_rules.yaml", str(exc)))

        try:
            loaded.imputation = load_imputation_config(self.settings.imputation, loaded.schema)
        except (FileNotFoundError, ValueError) as exc:
            findings.append(Finding(ERROR, "imputation.yaml", str(exc)))

        self._cross_checks(loaded)
        return loaded

    def _cross_checks(self, loaded: LoadedConfiguration) -> None:
        findings = loaded.findings
        settings, schema, rules, imputation = loaded.settings, loaded.schema, loaded.cleaning_rules, loaded.imputation

        if schema is not None and imputation is not None:
            for problem in check_config_against_schema(imputation, schema):
                level = WARNING if problem.startswith("approved attribute") else ERROR
                findings.append(Finding(level, "imputation.yaml", problem))
            for column in set(imputation.numeric_columns) | set(imputation.categorical_columns):
                spec = schema.columns.get(column)
                if spec is not None and spec.is_numeric and spec.max is None:
                    findings.append(Finding(WARNING, "data_schema.yaml", f"numeric model attribute '{column}' has no max; imputed values are unbounded above"))

        if schema is not None and rules is not None:
            for table_name, table in (("spelling_variants", rules.synonyms), ("code_mappings", rules.code_mappings)):
                for column, mapping in table.items():
                    spec = schema.columns.get(column)
                    if spec is None:
                        findings.append(Finding(ERROR, "cleaning_rules.yaml", f"{table_name}.{column}: column not in the schema"))
                        continue
                    if spec.allowed:
                        bad = sorted({v for v in mapping.values() if v not in spec.allowed})
                        if bad:
                            findings.append(Finding(ERROR, "cleaning_rules.yaml", f"{table_name}.{column} maps to labels outside the allowed list: {bad}"))
            for family in rules.inactive_families():
                entry = rules.approval.get(family)
                note = f" ({entry.note})" if entry and entry.note else ""
                findings.append(Finding(WARNING, "cleaning_rules.yaml", f"rule family '{family}' needs domain approval and is not approved: it will be skipped{note}"))
            for family, entry in rules.approval.items():
                if entry.approved and entry.requires_domain_approval and entry.approved_by and "synthetic" in entry.approved_by.lower():
                    findings.append(Finding(INFO, "cleaning_rules.yaml", f"rule family '{family}' is approved for synthetic data only ({entry.approved_by}); re-approve before real records"))

        methods = available_methods()
        if settings.method not in methods:
            findings.append(Finding(ERROR, "pipeline.yaml", f"imputation.method '{settings.method}' is not one of {methods}"))
        if settings.reference_method is not None and settings.reference_method not in methods:
            findings.append(Finding(ERROR, "pipeline.yaml", f"imputation.reference_method '{settings.reference_method}' is not one of {methods}"))
        if settings.reference_method == settings.method:
            findings.append(Finding(WARNING, "pipeline.yaml", "reference_method equals method; the report will compare the method with itself"))
        if settings.stages.get("validation") and not settings.stages.get("imputation"):
            findings.append(Finding(WARNING, "pipeline.yaml", "validation is enabled without imputation; it will be skipped"))
        if imputation is not None and settings.method in imputation.selected_hyperparameters:
            findings.append(Finding(INFO, "imputation.yaml", f"selected_hyperparameters override the defaults for {settings.method}: {imputation.selected_hyperparameters[settings.method]}"))

    # ------------------------------------------------------------------ #
    # Approval state
    # ------------------------------------------------------------------ #

    def states(self) -> dict[str, ApprovalState]:
        return {name: self.store.state(name, path) for name, path in self.files.items()}

    def check(self) -> ConfigurationCheck:
        return ConfigurationCheck(loaded=self.load(), states=self.states())

    def approve(self, name: str, by: str, note: str = "") -> ApprovalState:
        """Record approval of one file at its current content.

        Refused when that file has error-level findings, so an invalid file
        cannot be approved into a release run.
        """
        if name not in self.files:
            raise ApprovalError(f"'{name}' is not a configuration file (expected one of {sorted(self.files)}).")
        check = self.check()
        blocking = [f for f in check.loaded.errors if f.file == name]
        if blocking:
            raise ApprovalError(f"'{name}' has errors and cannot be approved: " + "; ".join(f.message for f in blocking))
        self.store.approve(name, self.files[name], by, note)
        return self.store.state(name, self.files[name])

    def approve_all(self, by: str, note: str = "") -> dict[str, ApprovalState]:
        return {name: self.approve(name, by, note) for name in self.files}

    def revoke(self, name: str) -> bool:
        return self.store.revoke(name)

    def require(self, mode: str | None = None) -> ConfigurationCheck:
        """Gate for stages: raise in release mode unless valid and approved."""
        mode = mode or self.settings.execution_mode
        check = self.check()
        if mode == "release":
            problems = [f"{f.file}: {f.message}" for f in check.loaded.errors]
            problems += [f"{name}: {state.status}" for name, state in check.states.items() if state.status != APPROVED]
            if problems:
                raise ConfigurationError(
                    "Release mode refuses to run on this configuration: "
                    + "; ".join(problems)
                    + ". Run `oc-pipeline config check`, fix the findings, then `oc-pipeline config approve`; "
                    "or set execution.mode: development / pass --allow-unapproved for a development run."
                )
        return check

    # ------------------------------------------------------------------ #
    # Display helpers (notebook)
    # ------------------------------------------------------------------ #

    def summaries(self, loaded: LoadedConfiguration | None = None) -> dict[str, pd.DataFrame]:
        """Tables describing the configuration for review."""
        loaded = loaded or self.load()
        tables: dict[str, pd.DataFrame] = {}
        schema, rules, imputation, settings = loaded.schema, loaded.cleaning_rules, loaded.imputation, loaded.settings
        if schema is not None:
            rows = []
            for spec in schema.columns.values():
                rows.append(
                    {
                        "column": spec.name,
                        "role": spec.role,
                        "dtype": spec.dtype,
                        "required": spec.required,
                        "allowed": ", ".join(spec.allowed) if spec.allowed else "",
                        "min": spec.min if spec.min is not None else "",
                        "max": spec.max if spec.max is not None else "",
                        "sentinels": ", ".join(str(int(v)) if float(v).is_integer() else str(v) for v in spec.sentinel_values),
                        "derived_from": spec.derive["source"] if spec.derive else "",
                    }
                )
            tables["schema_columns"] = pd.DataFrame(rows)
            tables["schema_summary"] = pd.DataFrame(
                [
                    {"setting": "asset_class", "value": schema.asset_class},
                    {"setting": "id_column", "value": schema.id_column},
                    {"setting": "group_column", "value": schema.group_column or ""},
                    {"setting": "snapshot_year", "value": schema.snapshot_year},
                    {"setting": "primary_targets", "value": ", ".join(schema.primary_targets)},
                    {"setting": "model_attributes", "value": ", ".join(schema.model_attributes)},
                    {"setting": "forbidden_predictors", "value": ", ".join(schema.forbidden_predictors)},
                    {"setting": "missing_tokens", "value": ", ".join(repr(t) for t in schema.missing_tokens)},
                    {"setting": "column_mapping", "value": str(schema.column_mapping) if schema.column_mapping else "(none)"},
                ]
            )
        if rules is not None:
            fam_rows = []
            for family in APPROVAL_FAMILIES:
                entry = rules.approval.get(family)
                fam_rows.append(
                    {
                        "family": family,
                        "requires_domain_approval": entry.requires_domain_approval if entry else False,
                        "approved": entry.approved if entry else True,
                        "approved_by": (entry.approved_by or "") if entry else "",
                        "active": rules.family_active(family),
                        "note": entry.note if entry else "",
                    }
                )
            tables["cleaning_families"] = pd.DataFrame(fam_rows)
            policy_rows = [
                {"policy": "unknown_category_action", "value": rules.unknown_category_action},
                {"policy": "non_numeric_action", "value": rules.non_numeric_action},
                {"policy": "out_of_range_action", "value": rules.out_of_range_action},
                {"policy": "sentinel_action", "value": rules.sentinel_action},
                {"policy": "non_integer_action", "value": rules.non_integer_action},
                {"policy": "exact_duplicate_action", "value": rules.exact_duplicate_action},
                {"policy": "duplicate_id_action", "value": rules.duplicate_id_action},
                {"policy": "spelling_variants", "value": f"{sum(len(m) for m in rules.synonyms.values())} entries over {len(rules.synonyms)} columns"},
                {"policy": "code_mappings", "value": f"{sum(len(m) for m in rules.code_mappings.values())} entries over {len(rules.code_mappings)} columns"},
            ]
            tables["cleaning_policies"] = pd.DataFrame(policy_rows)
        if imputation is not None:
            from ..imputation.registry import resolve_hyperparameters

            method_rows = []
            for method in available_methods():
                params = resolve_hyperparameters(imputation, method)
                method_rows.append({"method": method, "selected": method == settings.method, "reference": method == settings.reference_method, "hyperparameters": str(params)})
            tables["imputation_methods"] = pd.DataFrame(method_rows)
            tables["imputation_columns"] = pd.DataFrame(
                [{"role": "numeric", "columns": ", ".join(imputation.numeric_columns)}, {"role": "categorical", "columns": ", ".join(imputation.categorical_columns)}, {"role": "excluded", "columns": ", ".join(imputation.excluded_columns)}, {"role": "constraints", "columns": str(imputation.constraints)}]
            )
        tables["pipeline_settings"] = pd.DataFrame(
            [
                {"setting": "execution_mode", "value": settings.execution_mode},
                {"setting": "method", "value": settings.method},
                {"setting": "reference_method", "value": settings.reference_method or ""},
                {"setting": "stages", "value": ", ".join(k for k, v in settings.stages.items() if v)},
                {"setting": "reconstruction", "value": f"enabled={settings.reconstruction.enabled}, mask_rate={settings.reconstruction.mask_rate}, repeats={settings.reconstruction.repeats}, seed={settings.reconstruction.mask_seed}"},
                {"setting": "seed", "value": settings.seed},
                {"setting": "save_model", "value": settings.save_model},
            ]
        )
        return tables
