"""Static HTML report assembled from a run directory.

The report never computes anything: every table and figure is read from the
files the stages wrote, so a reader can open the CSV behind any number. Figures
are embedded as base64 so the HTML is a single file that opens anywhere.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import __version__
from ..settings import PipelineSettings

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _table(frame: pd.DataFrame | None, max_rows: int | None = None, float_format: str = "{:,.3f}", columns: list[str] | None = None) -> str | None:
    if frame is None or frame.empty:
        return None
    view = frame if columns is None else frame.loc[:, [c for c in columns if c in frame.columns]]
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_html(
        index=False,
        classes="table",
        border=0,
        na_rep="",
        float_format=lambda v: float_format.format(v),
        escape=True,
    )


def _figure(path: Path, embed: bool, root: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    if embed:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"src": f"data:image/png;base64,{data}", "name": path.name}
    return {"src": str(path.relative_to(root)).replace("\\", "/"), "name": path.name}


def _figures_in(directory: Path, embed: bool, root: Path, names: list[str] | None = None) -> list[dict[str, str]]:
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("*.png"))
    if names:
        order = {name: i for i, name in enumerate(names)}
        paths = sorted(paths, key=lambda p: order.get(p.stem, len(order)))
    figures = [_figure(p, embed, root) for p in paths]
    return [f for f in figures if f]


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value: Any, digits: int = 2) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def build_context(run_dir: Path, settings: PipelineSettings) -> dict[str, Any]:
    run_dir = Path(run_dir)
    meta: dict[str, Any] = {}
    if (run_dir / "run_metadata.json").is_file():
        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    embed = settings.embed_figures
    quality = meta.get("quality", {})
    cleaning = meta.get("cleaning", {})
    profiling = meta.get("profiling", {})
    imputation = meta.get("imputation", {})
    id_column = imputation.get("id_column", "segment_id")
    validation = meta.get("validation", {})

    configuration = meta.get("configuration", {})
    evaluation = meta.get("evaluation", {})
    quality_issues = _read_csv(run_dir / "quality_report.csv")
    provenance_summary = _read_csv(run_dir / "provenance_summary.csv")
    recovery_log = _read_csv(run_dir / "recovery_log.csv")
    evaluation_dir = run_dir / "evaluation"
    eval_summary = _read_csv(evaluation_dir / "summary.csv")
    observed_differences = _read_csv(evaluation_dir / "observed_difference_summary.csv")
    eval_per_class = _read_csv(evaluation_dir / "per_class.csv")
    cleaning_log = _read_csv(run_dir / "cleaning_log.csv")
    profile_dir = run_dir / "profile"
    validation_dir = run_dir / "validation"
    integrity = _read_csv(validation_dir / "integrity_checks.csv")
    distribution = _read_csv(validation_dir / "distribution_checks.csv")
    recon_summary = _read_csv(validation_dir / "reconstruction_summary.csv")
    recon_per_class = _read_csv(validation_dir / "reconstruction_per_class.csv")
    low_confidence = _read_csv(validation_dir / "low_confidence_cells.csv")
    convergence = _read_csv(run_dir / "convergence.csv")
    derived_log = _read_csv(run_dir / "derived_fields_log.csv")
    diagnostics = _read_csv(run_dir / "diagnostics.csv")

    # Headline numbers -------------------------------------------------------------
    n_rows = imputation.get("n_rows") or cleaning.get("rows_out") or quality.get("n_rows")
    cells_imputed = imputation.get("cells_imputed")
    integrity_passed = validation.get("integrity_passed")
    headline = (validation.get("reconstruction") or {}).get("headline") or {}
    method = imputation.get("method")
    reference = imputation.get("reference_method")
    targets = profiling.get("targets") or []
    recon_cards = []
    for target in targets:
        main = (headline.get(method) or {}).get(target) if method else None
        ref = (headline.get(reference) or {}).get(target) if reference else None
        if not main:
            continue
        if main.get("mae") is not None:
            recon_cards.append({"label": f"{target}: MAE on hidden known values", "value": _fmt_num(main["mae"], 2), "reference": f"{reference}: {_fmt_num(ref['mae'], 2)}" if ref else "", "note": "years; lower is better"})
        if main.get("macro_f1") is not None:
            recon_cards.append({"label": f"{target}: macro-F1 on hidden known values", "value": _fmt_num(main["macro_f1"], 3), "reference": f"{reference}: {_fmt_num(ref['macro_f1'], 3)}" if ref else "", "note": "higher is better"})

    config_states = None
    if configuration.get("files"):
        config_states = pd.DataFrame(
            [{"file": name, "status": info.get("status"), "sha256": (info.get("sha256") or "")[:12], "approved_by": info.get("approved_by") or "", "approved_at": info.get("approved_at") or ""} for name, info in configuration["files"].items()]
        )
    config_findings = pd.DataFrame(configuration.get("findings") or [], columns=["level", "file", "message"])

    eval_cards = []
    for protocol, by_method in (evaluation.get("headline") or {}).items():
        for method_name, targets in by_method.items():
            for target, values in targets.items():
                if values.get("mae") is not None:
                    note = f"{values['within_5_years']:.0%} within 5 years" if values.get("within_5_years") is not None else "lower is better"
                    eval_cards.append({"label": f"{protocol} · {method_name} · {target} MAE", "value": _fmt_num(values["mae"], 2), "reference": f"n={int(values['n_masked'])}", "note": note})
                if values.get("macro_f1") is not None:
                    eval_cards.append({"label": f"{protocol} · {method_name} · {target} macro-F1", "value": _fmt_num(values["macro_f1"], 3), "reference": f"n={int(values['n_masked'])}", "note": "higher is better"})

    source_counts = None
    if diagnostics is not None and not diagnostics.empty and "prediction_source" in diagnostics:
        source_counts = diagnostics.groupby(["column", "prediction_source"]).size().reset_index(name="cells")

    cleaning_by_rule = None
    if cleaning.get("changes_by_rule"):
        cleaning_by_rule = pd.DataFrame(
            [{"rule": k, "changes": v} for k, v in cleaning["changes_by_rule"].items()]
        )

    missingness = _read_csv(profile_dir / "missingness.csv")
    relationships = _read_csv(profile_dir / "relationships.csv")
    patterns = _read_csv(profile_dir / "missing_patterns.csv")
    numeric_summary = _read_csv(profile_dir / "numeric_summary.csv")
    categorical_summary = _read_csv(profile_dir / "categorical_summary.csv")
    mar = _read_csv(profile_dir / "missingness_by_attribute.csv")
    if mar is not None and not mar.empty:
        mar = mar.assign(deviation=(mar["ratio_to_overall"] - 1).abs()).sort_values("deviation", ascending=False).drop(columns="deviation").head(12)

    files = sorted(str(p.relative_to(run_dir)).replace("\\", "/") for p in run_dir.rglob("*") if p.is_file() and p.name != "report.html")

    return {
        "title": settings.report_title,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "package_version": meta.get("package_version", __version__),
        "versions": {k: meta.get(k) for k in ("python_version", "sklearn_version", "pandas_version")},
        "run_dir": str(run_dir),
        "input": quality.get("input") or imputation.get("input"),
        "meta": meta,
        "method": method,
        "reference": reference,
        "hyperparameters": imputation.get("hyperparameters") or {},
        "method_metadata": imputation.get("method_metadata") or {},
        "cards": [
            {"label": "records", "value": f"{int(n_rows):,}" if n_rows else "n/a"},
            {"label": "model attributes", "value": str(profiling.get("n_model_columns") or len(imputation.get("model_columns") or []) or "n/a")},
            {"label": "complete records before imputation", "value": _fmt_pct(profiling.get("complete_model_rows_share")) if profiling else "n/a"},
            {"label": "cells imputed", "value": f"{int(cells_imputed):,}" if cells_imputed is not None else "n/a"},
            {"label": "integrity checks", "value": "passed" if integrity_passed else ("FAILED" if integrity_passed is not None else "not run"), "status": "good" if integrity_passed else ("bad" if integrity_passed is not None else "")},
            {"label": "quality findings", "value": f"{quality.get('n_errors', 0)} errors / {quality.get('n_warnings', 0)} warnings" if quality else "not run"},
        ],
        "recon_cards": recon_cards,
        "quality": quality,
        "quality_table": _table(quality_issues, columns=["severity", "check", "column", "count", "rate", "detail"], float_format="{:.4f}"),
        "cleaning": cleaning,
        "cleaning_by_rule": _table(cleaning_by_rule),
        "cleaning_log_sample": _table(cleaning_log, max_rows=25),
        "cleaning_log_rows": 0 if cleaning_log is None else len(cleaning_log),
        "profiling": profiling,
        "missingness_table": _table(missingness, float_format="{:.3f}"),
        "patterns_table": _table(patterns, max_rows=10, float_format="{:.3f}"),
        "numeric_summary_table": _table(numeric_summary, float_format="{:,.2f}"),
        "categorical_summary_table": _table(categorical_summary, float_format="{:.3f}"),
        "relationships_table": _table(relationships, float_format="{:.3f}"),
        "mar_table": _table(mar, float_format="{:.3f}"),
        "profile_figures": _figures_in(profile_dir, embed, run_dir, ["missingness", "co_missingness", "distribution_installation_year", "distribution_distance_to_sea_km", "categorical_frequencies", "installation_year_by_category", "distance_to_sea_km_by_category", "missingness_by_attribute_installation_year", "missingness_by_attribute_material"]),
        "imputation": imputation,
        "cells_by_column": _table(pd.DataFrame([{"column": k, "cells_imputed": v} for k, v in (imputation.get("cells_imputed_by_column") or {}).items()])),
        "source_counts": _table(source_counts),
        "convergence_table": _table(convergence, float_format="{:.5f}"),
        "derived_table": _table(derived_log),
        "validation": validation,
        "integrity_table": _table(integrity),
        "integrity_failed": None if integrity is None else int((integrity["status"] == "fail").sum()),
        "distribution_table": _table(distribution, columns=["variable", "variable_type", "n_observed", "n_imputed", "observed_mean", "imputed_mean", "std_ratio", "wasserstein_1", "ks_statistic", "total_variation", "majority_share_observed", "majority_share_imputed"], float_format="{:,.3f}"),
        "joint_shift": (validation.get("joint_shift") or {}),
        "validation_figures": _figures_in(validation_dir / "figures", embed, run_dir),
        "reconstruction": validation.get("reconstruction") or {},
        "recon_table": _table(recon_summary, columns=["method", "variable", "variable_type", "n_masked", "mae", "mae_std", "rmse", "signed_bias", "accuracy", "balanced_accuracy", "macro_f1", "macro_f1_std"], float_format="{:,.3f}"),
        "recon_per_class_table": _table(recon_per_class.groupby(["method", "variable", "class"], sort=False)[["support", "predicted", "recall", "precision", "f1"]].mean().reset_index() if recon_per_class is not None and not recon_per_class.empty else None, float_format="{:,.3f}"),
        "confidence": validation.get("confidence") or {},
        "low_confidence_table": _table(low_confidence, max_rows=40, columns=[c for c in ([id_column, "row", "column", "imputed", "prediction_source", "max_proba", "margin", "tree_std", "tree_p10", "tree_p90", "confidence_rank"]) if low_confidence is not None and c in low_confidence.columns], float_format="{:,.3f}"),
        "low_confidence_rows": 0 if low_confidence is None else len(low_confidence),
        "configuration": configuration,
        "config_states_table": _table(config_states),
        "config_findings_table": _table(config_findings) if len(config_findings) else None,
        "provenance": imputation.get("provenance") or {},
        "provenance_table": _table(provenance_summary),
        "recovery_table": _table(recovery_log, columns=["column", "from", "snapshot_year", "n_recovered", "n_rejected_out_of_range", "n_still_missing"]),
        "compatibility_warnings": imputation.get("compatibility_warnings") or [],
        "cleaning_inactive": cleaning.get("inactive_families") or [],
        "cleaning_pending_codes": cleaning.get("pending_code_mappings") or {},
        "evaluation": evaluation,
        "eval_cards": eval_cards,
        "eval_summary_table": _table(eval_summary, columns=["protocol", "method", "variable", "variable_type", "n_masked", "mae", "mae_std", "signed_bias", "within_5_years", "accuracy", "balanced_accuracy", "macro_f1", "macro_f1_std", "fit_seconds"], float_format="{:,.3f}"),
        "observed_difference_table": _table(observed_differences, columns=["column", "role", "missing_measured", "observed_compared", "observed_different", "difference_rate", "observed_mae"], float_format="{:,.4f}"),
        "eval_per_class_table": _table(eval_per_class.groupby(["protocol", "method", "variable", "class"], sort=False)[["support", "predicted", "recall", "precision", "f1"]].mean().reset_index() if eval_per_class is not None and not eval_per_class.empty else None, float_format="{:,.3f}"),
        "eval_figures": _figures_in(evaluation_dir / "figures", embed, run_dir),
        "files": files,
    }


def generate_report(run_dir: str | Path, settings: PipelineSettings, output: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))
    template = env.get_template("report.html.j2")
    html = template.render(**build_context(run_dir, settings))
    output = output or (run_dir / "report.html")
    output.write_text(html, encoding="utf-8")
    return output
