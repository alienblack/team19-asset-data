"""Stage orchestration.

Each ``run_*`` function is one pipeline stage. Stages communicate through
files in the run directory so that any stage can be executed on its own
(``oc-pipeline validate --run <dir>``) and so that every number in the report
is traceable to a CSV next to it.

Run directory layout::

    run.log                       what happened, with timings
    run_metadata.json             settings, versions, per-stage summaries
    source_records.csv            byte-for-byte copy of the input
    quality_report.csv            findings of the quality stage
    cleaned.csv / cleaning_log.csv / dropped_rows.csv
    profile/                      profiling tables (*.csv) and figures (*.png)
    imputation_input.csv          the typed frame handed to the imputer (after deterministic recovery)
    recovery_log.csv              sources restored from their derived columns before imputation
    final_imputed.csv             completed records (+ derived fields), source column order
    cell_provenance.csv           one row per derived / imputed / unresolved cell
    provenance_summary.csv        observed / derived / imputed / unresolved counts per column
    diagnostics.csv               one row per imputed cell (method diagnostics)
    reference_imputed.csv         reference method output (median/mode)
    convergence.csv               per-pass/iteration changes (forest methods)
    derived_fields_log.csv
    model.joblib (+ .provenance.json)
    validation/                   integrity, distribution, reconstruction, confidence (operational checks)
    evaluation/                   evaluation MODE only: partitions, masks, metrics against known truth
    report.html
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .evaluation.confidence import low_confidence_cells
from .evaluation.distribution_analysis import compare_distributions, joint_distribution_shift
from .evaluation.holdout import prepare_holdout
from .evaluation.integrity_checks import check_integrity
from .evaluation.metrics import evaluate_imputation, per_class_report
from .evaluation.protocol import EvaluationSettings, build_evaluation_inputs, run_protocols
from .imputation.config import ImputationConfig, load_imputation_config
from .imputation.persistence import check_compatibility, check_provenance, load_imputer, read_provenance, save_imputer
from .imputation.registry import BaseImputer, ImputationOutput, build_imputer, resolve_hyperparameters
from .ingestion.loader import apply_schema_types, finalise_output_types, load_records
from .ingestion.schema import DataSchema, load_schema
from .preprocessing.feature_selection import select_model_frame
from .preprocessing.prepare import prepare_for_imputation
from .postprocessing import derive_fields, finalise_output
from .profiling.profiler import profile_records
from .profiling.visualization import render_imputation_figures, render_profile_figures, render_reconstruction_figure
from .quality.cleaner import clean_records, load_cleaning_rules
from .quality.validator import validate_records
from .settings import PipelineSettings

LOGGER_NAME = "overhead_conductors"


class PipelineError(RuntimeError):
    """Raised when a stage cannot proceed."""


# ---------------------------------------------------------------------------
# Run directory and bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def log(self) -> Path:
        return self.root / "run.log"

    @property
    def metadata(self) -> Path:
        return self.root / "run_metadata.json"

    @property
    def source_records(self) -> Path:
        return self.root / "source_records.csv"

    @property
    def quality_report(self) -> Path:
        return self.root / "quality_report.csv"

    @property
    def cleaned(self) -> Path:
        return self.root / "cleaned.csv"

    @property
    def cleaning_log(self) -> Path:
        return self.root / "cleaning_log.csv"

    @property
    def dropped_rows(self) -> Path:
        return self.root / "dropped_rows.csv"

    @property
    def profile_dir(self) -> Path:
        return self.root / "profile"

    @property
    def imputation_input(self) -> Path:
        return self.root / "imputation_input.csv"

    @property
    def imputed(self) -> Path:
        return self.root / "final_imputed.csv"

    @property
    def cell_provenance(self) -> Path:
        return self.root / "cell_provenance.csv"

    @property
    def provenance_summary(self) -> Path:
        return self.root / "provenance_summary.csv"

    @property
    def diagnostics(self) -> Path:
        return self.root / "diagnostics.csv"

    @property
    def reference_imputed(self) -> Path:
        return self.root / "reference_imputed.csv"

    @property
    def convergence(self) -> Path:
        return self.root / "convergence.csv"

    @property
    def derived_log(self) -> Path:
        return self.root / "derived_fields_log.csv"

    @property
    def recovery_log(self) -> Path:
        return self.root / "recovery_log.csv"

    @property
    def model(self) -> Path:
        return self.root / "model.joblib"

    @property
    def validation_dir(self) -> Path:
        return self.root / "validation"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    @property
    def report(self) -> Path:
        return self.root / "report.html"


def get_logger(run: RunPaths | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_oc_console", False) for h in logger.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        console._oc_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    if run is not None:
        run.root.mkdir(parents=True, exist_ok=True)
        target = str(run.log.resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == target for h in logger.handlers):
            handle = logging.FileHandler(run.log, encoding="utf-8")
            handle.setFormatter(formatter)
            logger.addHandler(handle)
    return logger


def read_metadata(run: RunPaths) -> dict[str, Any]:
    if run.metadata.is_file():
        return json.loads(run.metadata.read_text(encoding="utf-8"))
    return {}


def update_metadata(run: RunPaths, section: str, payload: dict[str, Any]) -> dict[str, Any]:
    run.root.mkdir(parents=True, exist_ok=True)
    meta = read_metadata(run)
    meta.setdefault("package_version", __version__)
    meta.setdefault("python_version", platform.python_version())
    meta.setdefault("sklearn_version", sklearn.__version__)
    meta.setdefault("pandas_version", pd.__version__)
    meta.setdefault("created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta[section] = _jsonable(payload)
    meta["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run.metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def _settings_payload(settings: PipelineSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["reconstruction"] = asdict(settings.reconstruction)
    return payload


def _load_typed(path: Path, schema: DataSchema) -> pd.DataFrame:
    """Load a CSV that is expected to be clean and coerce it to schema types."""
    loaded = load_records(path, schema)
    if loaded.missing_required:
        raise PipelineError(f"Input '{path}' lacks required columns: {list(loaded.missing_required)}")
    return apply_schema_types(loaded.frame, schema)


def check_configuration(settings: PipelineSettings, run: RunPaths | None = None) -> Any:
    """Validate and, in release mode, require approved configuration before a stage runs.

    The check result (file states, findings) is written to the run's
    ``configuration`` metadata section so every output records exactly which
    configuration content it was produced with.
    """
    from .configuration.manager import ConfigurationError, ConfigurationManager

    manager = ConfigurationManager(settings)
    try:
        check = manager.require(settings.execution_mode)
    except ConfigurationError as exc:
        if run is not None:
            update_metadata(run, "configuration", {**manager.check().to_jsonable(), "execution_mode": settings.execution_mode, "refused": True})
        raise PipelineError(str(exc)) from exc
    if run is not None:
        update_metadata(run, "configuration", {**check.to_jsonable(), "execution_mode": settings.execution_mode, "refused": False})
    logger = get_logger(run)
    if not check.all_approved:
        logger.warning("configuration: development run on unapproved configuration (%s)", ", ".join(f"{k}={v.status}" for k, v in check.states.items() if v.status != "approved"))
    return check


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def run_quality(input_csv: Path, run: RunPaths, settings: PipelineSettings) -> tuple[pd.DataFrame, Any]:
    """Load the source records, keep a copy, and write quality_report.csv."""
    logger = get_logger(run)
    check_configuration(settings, run)
    started = perf_counter()
    schema = load_schema(settings.data_schema)
    loaded = load_records(input_csv, schema)
    run.root.mkdir(parents=True, exist_ok=True)
    if input_csv.resolve() != run.source_records.resolve():
        shutil.copyfile(input_csv, run.source_records)
    report = validate_records(loaded.frame, schema)
    report.issues.to_csv(run.quality_report, index=False)
    logger.info(
        "quality: %d rows, %d columns, %d errors, %d warnings (%.1fs)",
        loaded.n_rows, loaded.n_columns, report.n_errors, report.n_warnings, perf_counter() - started,
    )
    update_metadata(
        run,
        "quality",
        {
            "input": str(input_csv.resolve()),
            "schema": str(settings.data_schema),
            "renamed_columns": loaded.renamed,
            "unknown_columns": list(loaded.unknown_columns),
            "missing_required": list(loaded.missing_required),
            **report.summary,
            "passed": report.passed,
            "seconds": perf_counter() - started,
        },
    )
    if not report.passed and settings.quality_fail_on_error:
        errors = report.issues[report.issues["severity"] == "error"]
        raise PipelineError(
            "Quality stage found blocking issues: "
            + "; ".join(f"{r.check}({r.column}) x{r.count}" for r in errors.itertuples())
            + f". See {run.quality_report}."
        )
    return loaded.frame, report


def run_clean(input_csv: Path, run: RunPaths, settings: PipelineSettings, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply cleaning rules and write cleaned.csv + cleaning_log.csv."""
    logger = get_logger(run)
    check_configuration(settings, run)
    started = perf_counter()
    schema = load_schema(settings.data_schema)
    rules = load_cleaning_rules(settings.cleaning_rules)
    if frame is None:
        loaded = load_records(input_csv, schema)
        if loaded.missing_required:
            raise PipelineError(f"Input lacks required columns: {list(loaded.missing_required)}")
        frame = loaded.frame
    result = clean_records(frame, schema, rules)
    run.root.mkdir(parents=True, exist_ok=True)
    finalise_output_types(result.cleaned, schema).to_csv(run.cleaned, index=False)
    result.log.to_csv(run.cleaning_log, index=False)
    if len(result.dropped_rows):
        result.dropped_rows.to_csv(run.dropped_rows, index=False)
    logger.info(
        "cleaning: %d -> %d rows, %d cells changed, %d rows dropped (%.1fs)",
        result.summary["rows_in"], result.summary["rows_out"], result.summary["cells_changed"], result.summary["rows_dropped"], perf_counter() - started,
    )
    update_metadata(run, "cleaning", {**result.summary, "rules": str(settings.cleaning_rules), "seconds": perf_counter() - started})
    return result.cleaned


def run_profile(input_csv: Path, run: RunPaths, settings: PipelineSettings, frame: pd.DataFrame | None = None) -> Any:
    """Write profiling tables and figures to profile/."""
    logger = get_logger(run)
    check_configuration(settings, run)
    started = perf_counter()
    schema = load_schema(settings.data_schema)
    if frame is None:
        frame = _load_typed(input_csv, schema)
    result = profile_records(frame, schema)
    result.write(run.profile_dir)
    figures = render_profile_figures(frame, schema, run.profile_dir)
    logger.info("profiling: %d tables, %d figures (%.1fs)", len(result.tables), len(figures), perf_counter() - started)
    update_metadata(
        run,
        "profiling",
        {**result.summary, "tables": list(result.tables), "figures": {k: str(v.name) for k, v in figures.items()}, "seconds": perf_counter() - started},
    )
    return result


def _build_methods(settings: PipelineSettings, config: ImputationConfig, method: str | None, overrides: dict[str, Any] | None) -> tuple[BaseImputer, BaseImputer | None, dict[str, Any]]:
    method = method or settings.method
    merged = {**settings.overrides, **(overrides or {})}
    imputer = build_imputer(method, config, merged)
    resolved = resolve_hyperparameters(config, method, merged)
    reference = None
    if settings.reference_method and settings.reference_method != method:
        reference = build_imputer(settings.reference_method, config)
    return imputer, reference, resolved


def run_impute(
    input_csv: Path,
    run: RunPaths,
    settings: PipelineSettings,
    method: str | None = None,
    overrides: dict[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
    model_path: Path | None = None,
) -> ImputationOutput:
    """Fit the configured method on the input and fill its gaps (fit_transform)."""
    logger = get_logger(run)
    check_configuration(settings, run)
    started = perf_counter()
    schema = load_schema(settings.data_schema)
    config = load_imputation_config(settings.imputation, schema)
    if frame is None:
        frame = _load_typed(input_csv, schema)
    prepared = prepare_for_imputation(frame, schema, config)
    model_frame = prepared.model_frame
    # Keep the caller's column order in every file; the model-column ordering
    # is internal to feature selection.
    X = prepared.frame
    run.root.mkdir(parents=True, exist_ok=True)
    finalise_output_types(X, schema).to_csv(run.imputation_input, index=False)
    prepared.recovery.log.to_csv(run.recovery_log, index=False)
    recovered_cells = prepared.recovered_cells
    if int(recovered_cells.to_numpy().sum()):
        logger.info("preparation: recovered %s", {k: v["n_recovered"] for k, v in prepared.recovery.summary.items()})

    imputer, reference, resolved = _build_methods(settings, config, method, overrides)
    logger.info("imputation: method=%s hyperparameters=%s on %d rows", imputer.name, resolved, len(X))
    fit_started = perf_counter()
    output = imputer.fit_transform(X)
    logger.info("imputation: %s filled %d cells (%.1fs)", imputer.name, len(output.diagnostics), perf_counter() - fit_started)

    final = finalise_output(
        frame, output.imputed, schema,
        model_columns=model_frame.model_columns, numeric_columns=model_frame.numeric_columns,
        categorical_columns=model_frame.categorical_columns, constraints=config.constraints,
        method=imputer.name, recovered_cells=recovered_cells, diagnostics=output.diagnostics, prepared=X,
    )
    derived = final.derivation
    final.frame.to_csv(run.imputed, index=False)
    final.provenance.cells.to_csv(run.cell_provenance, index=False)
    final.provenance.summary.to_csv(run.provenance_summary, index=False)
    diagnostics = output.diagnostics.copy()
    diagnostics.insert(1, schema.id_column, X[schema.id_column].reindex(diagnostics["row"]).to_numpy() if len(diagnostics) else [])
    diagnostics.to_csv(run.diagnostics, index=False)
    derived.log.to_csv(run.derived_log, index=False)
    logger.info("postprocessing: %s; integrity %s", {k: v for k, v in final.provenance.totals.items() if k != "n_cells"}, "passed" if final.integrity.passed else "FAILED")
    convergence = getattr(imputer, "convergence_", None)
    if convergence is None and "convergence" in output.metadata:
        convergence = pd.DataFrame(output.metadata["convergence"])
    if isinstance(convergence, pd.DataFrame) and len(convergence):
        convergence.to_csv(run.convergence, index=False)

    reference_meta: dict[str, Any] | None = None
    if reference is not None:
        ref_out = reference.fit_transform(X)
        finalise_output_types(derive_fields(ref_out.imputed, schema, original=X).frame, schema).to_csv(run.reference_imputed, index=False)
        reference_meta = {k: v for k, v in ref_out.metadata.items() if k != "statistics"}

    saved_model: str | None = None
    target_model = model_path or (run.model if settings.save_model else None)
    if target_model is not None and imputer.supports_persistence:
        save_imputer(imputer, target_model, X, schema.id_column, extra={"input": str(input_csv.resolve())})
        saved_model = str(Path(target_model).resolve())
    elif target_model is not None:
        logger.info("imputation: method %s is transductive; no model saved", imputer.name)

    metadata = {
        "input": str(input_csv.resolve()),
        "asset": schema.asset_class,
        "id_column": schema.id_column,
        "method": imputer.name,
        "hyperparameters": resolved,
        "model_columns": list(model_frame.model_columns),
        "numeric_columns": list(model_frame.numeric_columns),
        "categorical_columns": list(model_frame.categorical_columns),
        "passthrough_columns": list(model_frame.passthrough_columns),
        "n_rows": int(len(X)),
        "cells_imputed": int(len(output.diagnostics)),
        "cells_imputed_by_column": {k: int(v) for k, v in output.diagnostics["column"].value_counts().sort_index().items()} if len(output.diagnostics) else {},
        "method_metadata": {k: v for k, v in output.metadata.items() if k not in ("convergence", "statistics")},
        "reference_method": reference.name if reference is not None else None,
        "reference_metadata": reference_meta,
        "derived_fields": derived.summary,
        "recovery": prepared.recovery.summary,
        "provenance": final.provenance.to_jsonable(),
        "integrity_passed": final.integrity.passed,
        "model_path": saved_model,
        "seconds": perf_counter() - started,
    }
    update_metadata(run, "imputation", metadata)
    if not final.integrity.passed:
        failed = final.integrity.checks[final.integrity.checks["status"] == "fail"]
        raise PipelineError("Postprocessing integrity checks failed: " + "; ".join(f"{r.check}({r.column}) x{r.count}" for r in failed.itertuples()))
    return output


def run_fit(input_csv: Path, model_path: Path, settings: PipelineSettings, method: str | None = None, overrides: dict[str, Any] | None = None) -> BaseImputer:
    """Fit a persistable method and save it with provenance (no imputation output)."""
    logger = get_logger()
    check_configuration(settings)
    schema = load_schema(settings.data_schema)
    config = load_imputation_config(settings.imputation, schema)
    frame = _load_typed(input_csv, schema)
    model_frame = select_model_frame(frame, schema, config)
    imputer, _, resolved = _build_methods(settings, config, method, overrides)
    if not imputer.supports_persistence:
        raise PipelineError(f"Method '{imputer.name}' is transductive and cannot be fitted for later use; run `impute` instead.")
    started = perf_counter()
    imputer.fit(model_frame.frame)
    save_imputer(imputer, model_path, model_frame.frame, schema.id_column, extra={"input": str(input_csv.resolve()), "hyperparameters": resolved})
    logger.info("fit: %s on %d rows saved to %s (%.1fs)", imputer.name, len(model_frame.frame), model_path, perf_counter() - started)
    return imputer


def run_apply(
    input_csv: Path,
    model_path: Path,
    run: RunPaths,
    settings: PipelineSettings,
    require_independent: bool = False,
) -> ImputationOutput:
    """Apply a saved model to records (transform only)."""
    logger = get_logger(run)
    check_configuration(settings, run)
    started = perf_counter()
    schema = load_schema(settings.data_schema)
    config = load_imputation_config(settings.imputation, schema)
    frame = _load_typed(input_csv, schema)
    prepared = prepare_for_imputation(frame, schema, config)
    model_frame = prepared.model_frame
    X = prepared.frame
    imputer = load_imputer(model_path)
    provenance = read_provenance(model_path)
    compatibility = check_compatibility(
        model_path, imputer, numeric_columns=model_frame.numeric_columns, categorical_columns=model_frame.categorical_columns,
        constraints=config.constraints, frame=X,
    )
    for message in compatibility["warnings"]:
        logger.warning("apply: compatibility: %s", message)
    if compatibility["errors"]:
        raise PipelineError("Saved model is not compatible with this configuration/input: " + "; ".join(compatibility["errors"]))
    overlap_note = "none"
    try:
        check_provenance(model_path, X, schema.id_column)
    except Exception as exc:  # ProvenanceError
        overlap_note = str(exc)
        if require_independent:
            raise PipelineError(str(exc)) from exc
        logger.warning("apply: %s", exc)
    run.root.mkdir(parents=True, exist_ok=True)
    finalise_output_types(X, schema).to_csv(run.imputation_input, index=False)
    prepared.recovery.log.to_csv(run.recovery_log, index=False)
    output = imputer.transform(X)
    final = finalise_output(
        frame, output.imputed, schema,
        model_columns=model_frame.model_columns, numeric_columns=model_frame.numeric_columns,
        categorical_columns=model_frame.categorical_columns, constraints=config.constraints,
        method=imputer.name, recovered_cells=prepared.recovered_cells, diagnostics=output.diagnostics, prepared=X,
    )
    derived = final.derivation
    final.frame.to_csv(run.imputed, index=False)
    final.provenance.cells.to_csv(run.cell_provenance, index=False)
    final.provenance.summary.to_csv(run.provenance_summary, index=False)
    diagnostics = output.diagnostics.copy()
    diagnostics.insert(1, schema.id_column, X[schema.id_column].reindex(diagnostics["row"]).to_numpy() if len(diagnostics) else [])
    diagnostics.to_csv(run.diagnostics, index=False)
    derived.log.to_csv(run.derived_log, index=False)
    logger.info("apply: %s filled %d cells in %d rows; integrity %s (%.1fs)", imputer.name, len(diagnostics), len(X), "passed" if final.integrity.passed else "FAILED", perf_counter() - started)
    update_metadata(
        run,
        "imputation",
        {
            "input": str(input_csv.resolve()),
            "mode": "apply",
            "method": imputer.name,
            "hyperparameters": getattr(imputer, "config", None) and asdict(imputer.config),
            "model_path": str(Path(model_path).resolve()),
            "model_created_utc": provenance.get("created_utc"),
            "model_fitting_rows": provenance.get("n_fitting_rows"),
            "fitting_overlap": overlap_note,
            "model_columns": list(model_frame.model_columns),
            "numeric_columns": list(model_frame.numeric_columns),
            "categorical_columns": list(model_frame.categorical_columns),
            "n_rows": int(len(X)),
            "cells_imputed": int(len(diagnostics)),
            "cells_imputed_by_column": {k: int(v) for k, v in diagnostics["column"].value_counts().sort_index().items()} if len(diagnostics) else {},
            "method_metadata": {k: v for k, v in output.metadata.items() if k not in ("convergence", "statistics")},
            "reference_method": None,
            "derived_fields": derived.summary,
            "recovery": prepared.recovery.summary,
            "provenance": final.provenance.to_jsonable(),
            "compatibility_warnings": compatibility["warnings"],
            "integrity_passed": final.integrity.passed,
            "seconds": perf_counter() - started,
        },
    )
    if not final.integrity.passed:
        failed = final.integrity.checks[final.integrity.checks["status"] == "fail"]
        raise PipelineError("Postprocessing integrity checks failed: " + "; ".join(f"{r.check}({r.column}) x{r.count}" for r in failed.itertuples()))
    return output


def run_validate(run: RunPaths, settings: PipelineSettings) -> dict[str, Any]:
    """Integrity, distribution, reconstruction and confidence checks for a run."""
    logger = get_logger(run)
    check_configuration(settings, run)
    started = perf_counter()
    schema = load_schema(settings.data_schema)
    config = load_imputation_config(settings.imputation, schema)
    meta = read_metadata(run)
    imputation_meta = meta.get("imputation")
    if not imputation_meta or not run.imputed.is_file() or not run.imputation_input.is_file():
        raise PipelineError(f"No imputation output (final_imputed.csv) found in {run.root}; run `impute` first.")
    numeric = tuple(imputation_meta["numeric_columns"])
    categorical = tuple(imputation_meta["categorical_columns"])
    original = _load_typed(run.imputation_input, schema)
    imputed = _load_typed(run.imputed, schema)
    diagnostics = pd.read_csv(run.diagnostics) if run.diagnostics.is_file() else pd.DataFrame()
    out_dir = run.validation_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    integrity = check_integrity(original, imputed, schema, numeric, categorical, config.constraints)
    integrity.checks.to_csv(out_dir / "integrity_checks.csv", index=False)
    summary["integrity_passed"] = integrity.passed
    summary["integrity_failed_checks"] = integrity.n_failed
    logger.info("validation: integrity %s (%d failed checks)", "passed" if integrity.passed else "FAILED", integrity.n_failed)

    if settings.distribution_enabled:
        dist = compare_distributions(original, imputed, numeric, categorical)
        dist.to_csv(out_dir / "distribution_checks.csv", index=False)
        figures = render_imputation_figures(original, imputed, schema, out_dir / "figures", numeric, categorical)
        joint_notes = {}
        year_like = [c for c in numeric if c in schema.targets]
        for num_col in year_like or numeric[:1]:
            for cat_col in [c for c in categorical if c in schema.targets]:
                joint = joint_distribution_shift(original, imputed, num_col, cat_col, bin_width=float(settings.year_bin_width))
                joint.to_csv(out_dir / f"joint_shift_{num_col}_{cat_col}.csv", index=False)
                joint_notes[f"{num_col}_x_{cat_col}"] = {k: joint.attrs.get(k) for k in ("total_variation", "n_observed_rows", "n_imputed_rows")}
        summary["distribution"] = dist.set_index("variable")[["std_ratio", "wasserstein_1", "total_variation"]].to_dict(orient="index")
        summary["joint_shift"] = joint_notes
        summary["figures"] = {k: v.name for k, v in figures.items()}

    conf = low_confidence_cells(diagnostics, settings.low_confidence_fraction, settings.low_confidence_min_rows) if len(diagnostics) else None
    if conf is not None:
        conf.low_confidence.to_csv(out_dir / "low_confidence_cells.csv", index=False)
        conf.ranked.to_csv(out_dir / "diagnostics_ranked.csv", index=False)
        summary["confidence"] = conf.summary

    recon = settings.reconstruction
    if recon.enabled:
        method = imputation_meta["method"]
        hyperparameters = imputation_meta.get("hyperparameters") or {}
        if imputation_meta.get("mode") == "apply":
            logger.info("validation: reconstruction check skipped for an `apply` run (model was not fitted here)")
            summary["reconstruction"] = {"skipped": "apply run"}
        else:
            summary["reconstruction"] = _reconstruction_check(
                original, schema, config, method, hyperparameters, settings, numeric, categorical, out_dir, logger
            )

    summary["seconds"] = perf_counter() - started
    update_metadata(run, "validation", summary)
    return summary


def _reconstruction_check(
    original: pd.DataFrame,
    schema: DataSchema,
    config: ImputationConfig,
    method: str,
    hyperparameters: dict[str, Any],
    settings: PipelineSettings,
    numeric: tuple[str, ...],
    categorical: tuple[str, ...],
    out_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    recon = settings.reconstruction
    model_columns = list(numeric) + list(categorical)
    eligible = [c for c in model_columns if original[c].notna().sum() >= 2]
    stratify = tuple(c for c in categorical if c in eligible) if recon.stratify_categorical else ()
    metric_rows = []
    class_rows = []
    for repeat in range(recon.repeats):
        seed = recon.mask_seed + repeat
        masked, truth, mask = prepare_holdout(
            original, eligible, rate=recon.mask_rate, seed=seed, stratify_columns=stratify, linked_columns=schema.linked_columns()
        )
        methods: list[tuple[str, BaseImputer]] = [(method, build_imputer(method, config, hyperparameters))]
        if recon.include_reference and settings.reference_method and settings.reference_method != method:
            methods.append((settings.reference_method, build_imputer(settings.reference_method, config)))
        for name, imputer in methods:
            started = perf_counter()
            filled = imputer.fit_transform(masked).imputed
            table = evaluate_imputation(truth, filled, mask, numeric, categorical)
            table.insert(0, "method", name)
            table.insert(1, "repeat", repeat)
            table["mask_seed"] = seed
            table["fit_seconds"] = perf_counter() - started
            metric_rows.append(table)
            for column in categorical:
                report = per_class_report(truth, filled, mask, column)
                report.insert(0, "method", name)
                report.insert(1, "repeat", repeat)
                class_rows.append(report)
            logger.info("validation: reconstruction repeat %d method %s (%.1fs)", repeat, name, perf_counter() - started)
        linked = [c for links in schema.linked_columns().values() for c in links if c in mask.columns]
        mask_out = mask.loc[:, model_columns + linked].copy()
        mask_out.insert(0, schema.id_column, original[schema.id_column].to_numpy())
        mask_out.to_csv(out_dir / f"reconstruction_mask_repeat{repeat}.csv", index=False)
    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(out_dir / "reconstruction_metrics.csv", index=False)
    per_class = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
    per_class.to_csv(out_dir / "reconstruction_per_class.csv", index=False)
    value_columns = ["n_masked", "mae", "rmse", "median_absolute_error", "nrmse", "signed_bias", "accuracy", "balanced_accuracy", "macro_f1"]
    summary = (
        metrics.groupby(["method", "variable", "variable_type"], sort=False)[value_columns]
        .mean()
        .reset_index()
    )
    stds = metrics.groupby(["method", "variable"], sort=False)[["mae", "macro_f1"]].std(ddof=1).add_suffix("_std").reset_index()
    summary = summary.merge(stds, on=["method", "variable"], how="left")
    summary["n_repeats"] = recon.repeats
    summary["n_masked"] = summary["n_masked"].round().astype(int)
    summary.to_csv(out_dir / "reconstruction_summary.csv", index=False)
    render_reconstruction_figure(summary, out_dir / "figures")
    headline = {}
    for row in summary.itertuples():
        headline.setdefault(row.method, {})[row.variable] = {
            "mae": None if pd.isna(row.mae) else float(row.mae),
            "macro_f1": None if pd.isna(row.macro_f1) else float(row.macro_f1),
            "n_masked": float(row.n_masked),
        }
    return {
        "protocol": "hide observed cells, fit_transform on the masked table, score hidden cells",
        "mask_rate": recon.mask_rate,
        "repeats": recon.repeats,
        "stratified_categorical": bool(stratify),
        "columns": eligible,
        "headline": headline,
    }


def run_evaluate(
    run: RunPaths,
    settings: PipelineSettings,
    *,
    truth_csv: Path,
    input_csv: Path | None = None,
    manifest_path: Path | None = None,
    methods: Sequence[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare imputers on missing measured cells against paired ground truth.

    The observed measured values may be noisy. Truth is restricted to this
    evaluation stage and never supplied to the imputers. Truth-only synthetic
    masking remains available for legacy experiments.
    """
    from .profiling.visualization import render_reconstruction_figure

    logger = get_logger(run)
    check_configuration(settings, run)
    started = perf_counter()
    schema = load_schema(settings.data_schema)
    config = load_imputation_config(settings.imputation, schema)
    ev = settings.evaluation
    eval_settings = EvaluationSettings(
        fractions=dict(ev.fractions), split_seed=ev.split_seed, strategy=ev.strategy,
        group_column=ev.group_column,
        mask_rate=ev.mask_rate, mask_seed=ev.mask_seed, repeats=ev.repeats, protocols=ev.protocols,
    )
    if overrides:
        config = _config_with_overrides(config, overrides)
    truth = _load_typed(truth_csv, schema)
    masked = _load_typed(input_csv, schema) if input_csv is not None else None
    model_columns = list(config.numeric_columns) + list(config.categorical_columns)
    if truth[model_columns].isna().any().any():
        raise PipelineError("Ground-truth CSV must have no missing model-attribute values.")
    out_dir = run.evaluation_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_evaluation_inputs(schema, config, eval_settings, truth=truth, masked=masked, manifest_path=manifest_path or (out_dir / "partition_manifest.csv"))
    inputs.manifest.to_csv(out_dir / "partition_manifest.csv", index=False)
    observed_summary_path = None
    if masked is not None:
        from .evaluation.protocol import observed_difference_summary

        audit = observed_difference_summary(masked, inputs.truth, schema)
        observed_summary_path = out_dir / "observed_difference_summary.csv"
        audit.to_csv(observed_summary_path, index=False)
    for repeat, mask in enumerate(inputs.masks):
        mask_out = mask.loc[:, model_columns].copy()
        mask_out.insert(0, schema.id_column, inputs.masked[schema.id_column].astype(str).to_numpy())
        mask_out.insert(1, "partition", inputs.masked[schema.id_column].astype(str).map(inputs.manifest.set_index(inputs.manifest[schema.id_column].astype(str))["partition"]).to_numpy())
        mask_out.to_csv(out_dir / f"evaluation_mask_repeat{repeat}.csv", index=False)

    chosen = tuple(methods) if methods else (ev.methods or tuple(dict.fromkeys(m for m in (settings.method, settings.reference_method) if m)))
    logger.info("evaluation: mask source %s; partitions %s; methods %s", inputs.mask_source, {k: len(v) for k, v in inputs.parts.items()}, list(chosen))
    output = run_protocols(inputs, schema, config, eval_settings, methods=chosen, logger=logger)
    output.metrics.to_csv(out_dir / "metrics.csv", index=False)
    output.per_class.to_csv(out_dir / "per_class.csv", index=False)
    output.summary.to_csv(out_dir / "summary.csv", index=False)
    output.scored_cells.to_csv(out_dir / "scored_cells.csv", index=False)
    for protocol, table in output.summary.groupby("protocol", sort=False):
        render_reconstruction_figure(
            table, out_dir / "figures",
            title=f"Evaluation mode, {protocol}: hidden TEST cells scored against the complete reference",
            filename=f"{protocol}_metrics.png",
        )
    metadata = {
        **output.metadata,
        "truth": str(truth_csv.resolve()),
        "measured_input": str(input_csv.resolve()) if input_csv is not None else None,
        "masked_input": str(input_csv.resolve()) if input_csv is not None else None,
        "observed_difference_summary": str(observed_summary_path) if observed_summary_path else None,
        "asset": settings.asset,
        "manifest": str((manifest_path or out_dir / "partition_manifest.csv").resolve()),
        # ``config`` already carries the overrides each method accepts (see _config_with_overrides)
        "hyperparameters": {m: resolve_hyperparameters(config, m) for m in chosen},
        "headline": _evaluation_headline(output.summary, schema),
        "seconds": perf_counter() - started,
    }
    update_metadata(run, "evaluation", metadata)
    logger.info("evaluation complete (%.1fs): %s", perf_counter() - started, out_dir)
    return metadata


def _config_with_overrides(config: ImputationConfig, overrides: dict[str, Any]) -> ImputationConfig:
    """Development-only hyperparameter overrides applied through selected_hyperparameters."""
    from dataclasses import replace

    selected = {k: dict(v) for k, v in config.selected_hyperparameters.items()}
    for method in ("missforest", "predictive_missforest"):
        allowed = {"missforest": {"n_estimators", "max_iter", "max_features", "max_depth", "min_samples_leaf", "class_weight", "n_jobs"}, "predictive_missforest": {"n_estimators", "n_passes", "max_features", "max_depth", "min_samples_leaf", "class_weight", "numeric_init", "n_jobs"}}[method]
        selected.setdefault(method, {}).update({k: v for k, v in overrides.items() if k in allowed})
    return replace(config, selected_hyperparameters=selected)


def _evaluation_headline(summary: pd.DataFrame, schema: DataSchema) -> dict[str, Any]:
    headline: dict[str, Any] = {}
    if summary is None or summary.empty:
        return headline
    for row in summary.itertuples():
        if row.variable not in schema.primary_targets:
            continue
        entry = headline.setdefault(row.protocol, {}).setdefault(row.method, {})
        entry[row.variable] = {
            "mae": None if pd.isna(row.mae) else float(row.mae),
            "macro_f1": None if pd.isna(row.macro_f1) else float(row.macro_f1),
            "within_5_years": None if pd.isna(getattr(row, "within_5_years", np.nan)) else float(row.within_5_years),
            "n_masked": int(row.n_masked),
        }
    return headline


def run_report(run: RunPaths, settings: PipelineSettings) -> Path:
    from .reporting.report_generator import generate_report

    logger = get_logger(run)
    started = perf_counter()
    path = generate_report(run.root, settings)
    logger.info("report: %s (%.1fs)", path, perf_counter() - started)
    update_metadata(run, "report", {"path": str(path), "seconds": perf_counter() - started})
    return path


def run_all(
    input_csv: Path, run: RunPaths, settings: PipelineSettings,
    method: str | None = None, overrides: dict[str, Any] | None = None,
    truth_csv: Path | None = None,
) -> dict[str, Any]:
    """Quality -> cleaning -> profiling -> imputation -> validation -> optional paired evaluation -> report."""
    logger = get_logger(run)
    started = perf_counter()
    run.root.mkdir(parents=True, exist_ok=True)
    update_metadata(run, "settings", {**_settings_payload(settings), "input": str(input_csv.resolve()),
                                     "truth": str(truth_csv.resolve()) if truth_csv else None,
                                     "asset": settings.asset, "method_override": method, "overrides": overrides or {}})
    schema = load_schema(settings.data_schema)
    stages = settings.stages

    if stages["quality"]:
        raw_frame, _ = run_quality(input_csv, run, settings)
    else:
        loaded = load_records(input_csv, schema)
        if loaded.missing_required:
            raise PipelineError(f"Input lacks required columns: {list(loaded.missing_required)}")
        raw_frame = loaded.frame
        if input_csv.resolve() != run.source_records.resolve():
            shutil.copyfile(input_csv, run.source_records)

    if stages["cleaning"]:
        typed = run_clean(input_csv, run, settings, frame=raw_frame)
        typed = apply_schema_types(typed, schema)
        impute_input = run.cleaned
    else:
        typed = apply_schema_types(raw_frame, schema)
        impute_input = input_csv

    if stages["profiling"]:
        run_profile(impute_input, run, settings, frame=typed)

    if stages["imputation"]:
        run_impute(impute_input, run, settings, method=method, overrides=overrides, frame=typed)
        if stages["validation"]:
            run_validate(run, settings)
    if truth_csv is not None:
        if not stages["imputation"]:
            raise PipelineError("Paired ground-truth evaluation requires stages.imputation to be enabled.")
        chosen = tuple(dict.fromkeys(m for m in (method or settings.method, settings.reference_method) if m))
        run_evaluate(run, settings, truth_csv=truth_csv, input_csv=impute_input,
                     methods=chosen, overrides=overrides)
    if stages["reporting"]:
        run_report(run, settings)

    logger.info("pipeline complete in %.1fs: %s", perf_counter() - started, run.root)
    return read_metadata(run)
