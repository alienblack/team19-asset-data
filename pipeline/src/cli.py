"""Command-line interface: ``oc-pipeline <command> ...``.

Commands
    ingest    preserve and audit a raw CSV, without cleaning or modeling
    all       quality -> cleaning -> profiling -> imputation -> validation -> report
    profile   profiling tables and figures for a (clean) file
    clean     quality report + cleaning with audit log
    impute    fit the configured method on a file and fill its gaps
    fit       fit a persistable method and save it (no imputation output)
    apply     fill gaps in new records with a saved model
    evaluate  evaluation mode: score methods on hidden cells of a record-disjoint test partition
    validate  integrity / distribution / reconstruction / confidence checks for a run
    report    assemble report.html for a run
    config    check, show, approve or revoke the configuration files (release gate)

Run ``oc-pipeline <command> --help`` for the options of each command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .settings import PipelineSettings, load_settings


def _parse_overrides(values: Sequence[str] | None) -> dict[str, Any]:
    """``key=value`` pairs; values are parsed as JSON when possible (``max_depth=null``)."""
    out: dict[str, Any] = {}
    for item in values or ():
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"override '{item}' must look like key=value")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def _settings(args: argparse.Namespace) -> PipelineSettings:
    settings = load_settings(args.pipeline_config, asset=getattr(args, "asset", "overhead_conductors"))
    updates: dict[str, Any] = {}
    if getattr(args, "schema", None):
        updates["data_schema"] = Path(args.schema).resolve()
    if getattr(args, "rules", None):
        updates["cleaning_rules"] = Path(args.rules).resolve()
    if getattr(args, "config", None):
        updates["imputation"] = Path(args.config).resolve()
    if getattr(args, "reference", None) is not None:
        updates["reference_method"] = None if args.reference in ("none", "null", "") else args.reference
    if getattr(args, "seed", None) is not None:
        updates["seed"] = args.seed
    if getattr(args, "no_model", False):
        updates["save_model"] = False
    if getattr(args, "allow_unapproved", False):
        updates["execution_mode"] = "development"
    if getattr(args, "mask_rate", None) is not None or getattr(args, "repeats", None) is not None:
        from dataclasses import replace

        recon = settings.reconstruction
        if args.mask_rate is not None:
            recon = replace(recon, mask_rate=args.mask_rate)
        if args.repeats is not None:
            recon = replace(recon, repeats=args.repeats)
        updates["reconstruction"] = recon
    return settings.with_updates(**updates)


def _add_common(parser: argparse.ArgumentParser, *, input_required: bool = True, out_required: bool = True) -> None:
    if input_required:
        parser.add_argument("--measured", "--input", dest="input", required=True, type=Path,
                            help="measured CSV for the selected asset (missing/noisy values allowed)")
    if out_required:
        parser.add_argument("--out", required=True, type=Path, help="run directory for outputs")
    parser.add_argument("--asset", default="overhead_conductors", help="asset profile under config/ (default: overhead_conductors)")
    parser.add_argument("--pipeline-config", type=Path, default=None, help="override config/<asset>/pipeline.yaml")
    parser.add_argument("--schema", type=Path, default=None, help="override data_schema.yaml")
    parser.add_argument("--allow-unapproved", action="store_true", help="development run: continue on invalid or unapproved configuration")


def _add_method_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", default=None, help="imputation method (default from pipeline.yaml)")
    parser.add_argument("--config", type=Path, default=None, help="override imputation.yaml")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE", help="hyperparameter override, e.g. --set n_estimators=50 (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oc-pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"oc-pipeline {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="ingest source CSV as text and write a provenance report")
    _add_common(p)
    p.add_argument("--overwrite", action="store_true", help="replace existing ingestion output files")

    p = sub.add_parser("all", help="run every stage on a source file")
    _add_common(p)
    p.add_argument("--truth", type=Path, default=None, help="paired ground-truth CSV; enables comparable evaluation within this run")
    _add_method_options(p)
    p.add_argument("--rules", type=Path, default=None, help="override cleaning_rules.yaml")
    p.add_argument("--reference", default=None, help="reference method shown next to the main one, or 'none'")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mask-rate", type=float, default=None, help="reconstruction holdout rate")
    p.add_argument("--repeats", type=int, default=None, help="reconstruction holdout repeats")
    p.add_argument("--no-model", action="store_true", help="do not save the fitted model")

    p = sub.add_parser("profile", help="profiling tables and figures")
    _add_common(p)

    p = sub.add_parser("clean", help="quality report and cleaning with audit log")
    _add_common(p)
    p.add_argument("--rules", type=Path, default=None, help="override cleaning_rules.yaml")
    p.add_argument("--skip-quality", action="store_true", help="skip the quality report")

    p = sub.add_parser("impute", help="fit the method on the input and fill its gaps")
    _add_common(p)
    _add_method_options(p)
    p.add_argument("--reference", default=None, help="reference method, or 'none'")
    p.add_argument("--model-out", type=Path, default=None, help="where to save the fitted model (predictive only)")
    p.add_argument("--no-model", action="store_true", help="do not save the fitted model")

    p = sub.add_parser("fit", help="fit a persistable method and save it")
    _add_common(p, out_required=False)
    _add_method_options(p)
    p.add_argument("--model", required=True, type=Path, help="output model path (.joblib)")

    p = sub.add_parser("apply", help="fill gaps in new records with a saved model")
    _add_common(p)
    p.add_argument("--model", required=True, type=Path, help="saved model (.joblib)")
    p.add_argument("--config", type=Path, default=None, help="override imputation.yaml")
    p.add_argument("--require-independent", action="store_true", help="refuse records that were used to fit the model")

    p = sub.add_parser("evaluate", help="evaluation mode: score methods on hidden cells of a record-disjoint test partition against known truth")
    p.add_argument("--truth", required=True, type=Path, help="complete reference CSV (every value known)")
    p.add_argument("--measured", "--input", dest="input", type=Path, default=None,
                   help="paired measured CSV with missing/noisy cells (legacy: omit for artificial holdout)")
    p.add_argument("--out", required=True, type=Path, help="run directory; results go to <out>/evaluation/")
    p.add_argument("--manifest", type=Path, default=None, help="partition manifest to reuse (default: <out>/evaluation/partition_manifest.csv)")
    p.add_argument("--methods", default=None, help="comma list of methods (default: the configured method and its reference)")
    p.add_argument("--pipeline-config", type=Path, default=None)
    p.add_argument("--asset", default="overhead_conductors")
    p.add_argument("--schema", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE", help="forest hyperparameter override (development only)")
    p.add_argument("--allow-unapproved", action="store_true")

    p = sub.add_parser("validate", help="integrity, distribution, reconstruction and confidence checks")
    p.add_argument("--run", required=True, type=Path, help="run directory produced by impute/all")
    p.add_argument("--pipeline-config", type=Path, default=None)
    p.add_argument("--asset", default="overhead_conductors")
    p.add_argument("--allow-unapproved", action="store_true")
    p.add_argument("--schema", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--reference", default=None)
    p.add_argument("--mask-rate", type=float, default=None)
    p.add_argument("--repeats", type=int, default=None)

    p = sub.add_parser("report", help="assemble report.html for a run")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--pipeline-config", type=Path, default=None)
    p.add_argument("--asset", default="overhead_conductors")
    p.add_argument("--schema", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)

    p = sub.add_parser("config", help="check, show, approve or revoke configuration approvals")
    config_sub = p.add_subparsers(dest="config_command", required=True)
    c = config_sub.add_parser("check", help="validate the four files, cross-check them and show approval states")
    c.add_argument("--pipeline-config", type=Path, default=None)
    c.add_argument("--asset", default="overhead_conductors")
    c = config_sub.add_parser("show", help="print configuration summaries")
    c.add_argument("--pipeline-config", type=Path, default=None)
    c.add_argument("--asset", default="overhead_conductors")
    c = config_sub.add_parser("approve", help="record approval of a file at its current content")
    c.add_argument("--pipeline-config", type=Path, default=None)
    c.add_argument("--asset", default="overhead_conductors")
    c.add_argument("--file", action="append", required=True, help="file name as listed by `config check`, repeatable, or 'all'")
    c.add_argument("--by", required=True, help="who approves")
    c.add_argument("--note", default="", help="why / for which dataset")
    c = config_sub.add_parser("revoke", help="remove the approval of a file")
    c.add_argument("--pipeline-config", type=Path, default=None)
    c.add_argument("--asset", default="overhead_conductors")
    c.add_argument("--file", action="append", required=True)
    return parser


def _config_command(args: argparse.Namespace) -> int:
    import pandas as pd

    from .configuration.approval import ApprovalError
    from .configuration.manager import ConfigurationManager
    from .settings import load_settings

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 110)
    try:
        manager = ConfigurationManager(load_settings(args.pipeline_config, asset=getattr(args, "asset", "overhead_conductors")))
        if args.config_command == "check":
            check = manager.check()
            print(check.states_table().to_string(index=False))
            print()
            findings = check.findings_table()
            print(findings.to_string(index=False) if len(findings) else "no findings")
            print(f"\nvalid: {check.is_valid}  approved: {check.all_approved}  ready for release runs: {check.ready}")
            return 0 if check.ready else 1
        if args.config_command == "show":
            for name, table in manager.summaries().items():
                print(f"== {name} ==")
                print(table.to_string(index=False))
                print()
            return 0
        if args.config_command == "approve":
            names = list(manager.files) if "all" in args.file else args.file
            for name in names:
                state = manager.approve(name, by=args.by, note=args.note)
                print(f"approved {name} ({state.sha256[:12]}) by {args.by}")
            return 0
        if args.config_command == "revoke":
            for name in args.file:
                print(f"{'revoked' if manager.revoke(name) else 'no approval for'} {name}")
            return 0
        return 2
    except (ApprovalError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Ingestion deliberately works even when model/report dependencies are absent.
    # Keep the stage-specific imports below out of the ingestion code path.
    if args.command == "config":
        return _config_command(args)

    if args.command == "ingest":
        try:
            from .ingestion.stage import ingest_csv

            # Schema-free: the schema is not read here; --schema is accepted for
            # command-line symmetry and ignored.
            result = ingest_csv(args.input, output_dir=args.out, overwrite=args.overwrite)
            report = result.report
            print(f"Ingested {report['rows']:,} rows and {report['columns']} columns from {report['source_file']}")
            print(f"SHA-256: {report['sha256']}")
            if report["duplicate_columns"]:
                print(f"Duplicate column names: {report['duplicate_columns']}")
            print(f"Ingestion report: {result.report_json}")
            return 0
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    from .imputation.registry import available_methods
    from .pipeline import (
        PipelineError,
        RunPaths,
        get_logger,
        run_all,
        run_apply,
        run_clean,
        run_evaluate,
        run_fit,
        run_impute,
        run_profile,
        run_quality,
        run_report,
        run_validate,
        update_metadata,
    )

    if getattr(args, "method", None) is not None and args.method not in available_methods():
        parser.error(f"argument --method: invalid choice: {args.method!r} (choose from {', '.join(available_methods())})")

    try:
        settings = _settings(args)
        overrides = _parse_overrides(getattr(args, "overrides", None))
        if args.command == "all":
            run = RunPaths(args.out.resolve())
            run_all(args.input.resolve(), run, settings, method=args.method, overrides=overrides,
                    truth_csv=args.truth.resolve() if args.truth else None)
            print(f"Pipeline complete: {run.report if settings.stages['reporting'] else run.root}")
        elif args.command == "profile":
            run = RunPaths(args.out.resolve())
            get_logger(run)
            update_metadata(run, "settings", {"input": str(args.input.resolve()), "command": "profile"})
            run_profile(args.input.resolve(), run, settings)
            print(f"Profile written to {run.profile_dir}")
        elif args.command == "clean":
            run = RunPaths(args.out.resolve())
            get_logger(run)
            update_metadata(run, "settings", {"input": str(args.input.resolve()), "command": "clean"})
            frame = None
            if not args.skip_quality:
                frame, _ = run_quality(args.input.resolve(), run, settings)
            run_clean(args.input.resolve(), run, settings, frame=frame)
            print(f"Cleaned records: {run.cleaned}\nAudit log: {run.cleaning_log}")
        elif args.command == "impute":
            run = RunPaths(args.out.resolve())
            get_logger(run)
            update_metadata(run, "settings", {"input": str(args.input.resolve()), "command": "impute", "method": args.method or settings.method, "overrides": overrides})
            run_impute(args.input.resolve(), run, settings, method=args.method, overrides=overrides, model_path=args.model_out)
            print(f"Imputed records: {run.imputed}\nDiagnostics: {run.diagnostics}")
        elif args.command == "fit":
            run_fit(args.input.resolve(), args.model.resolve(), settings, method=args.method, overrides=overrides)
            print(f"Model saved: {args.model.resolve()}")
        elif args.command == "apply":
            run = RunPaths(args.out.resolve())
            get_logger(run)
            update_metadata(run, "settings", {"input": str(args.input.resolve()), "command": "apply", "model": str(args.model.resolve())})
            run_apply(args.input.resolve(), args.model.resolve(), run, settings, require_independent=args.require_independent)
            print(f"Imputed records: {run.imputed}")
        elif args.command == "evaluate":
            run = RunPaths(args.out.resolve())
            get_logger(run)
            update_metadata(run, "settings", {"command": "evaluate", "truth": str(args.truth.resolve()), "input": str(args.input.resolve()) if args.input else None})
            methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
            meta = run_evaluate(run, settings, truth_csv=args.truth.resolve(), input_csv=args.input.resolve() if args.input else None, manifest_path=args.manifest.resolve() if args.manifest else None, methods=methods, overrides=overrides)
            print(f"Evaluation written to {run.evaluation_dir} (test partition: {meta['partitions'].get('test')} records)")
        elif args.command == "validate":
            run = RunPaths(args.run.resolve())
            summary = run_validate(run, settings)
            print(f"Validation written to {run.validation_dir} (integrity {'passed' if summary.get('integrity_passed') else 'FAILED'})")
        elif args.command == "report":
            run = RunPaths(args.run.resolve())
            path = run_report(run, settings)
            print(f"Report: {path}")
        return 0
    except (PipelineError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
