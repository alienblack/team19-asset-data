"""Command-line entry point for all Member 3 validation tasks."""

from __future__ import annotations

import argparse
import sys

try:
    from . import config
    from .utils import (
        ValidationInputError, ensure_output_dirs, load_datasets,
        prepare_status_confidence, require_value, validate_datasets,
    )
    from .validate_distribution import compare_distributions
    from .validate_downstream import calculate_ranking_stability
    from .validate_metrics import calculate_validation_metrics, save_validation_metrics
except ImportError:
    import config
    from utils import (
        ValidationInputError, ensure_output_dirs, load_datasets,
        prepare_status_confidence, require_value, validate_datasets,
    )
    from validate_distribution import compare_distributions
    from validate_downstream import calculate_ranking_stability
    from validate_metrics import calculate_validation_metrics, save_validation_metrics


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser; config.py values act as overridable defaults."""
    parser = argparse.ArgumentParser(description="Run Sprint 2 Member 3 validation.")
    parser.add_argument("--reference", default=config.REFERENCE_FILE)
    parser.add_argument("--masked", default=config.MASKED_FILE)
    parser.add_argument("--baseline", default=config.BASELINE_FILE)
    parser.add_argument("--alternative", default=config.ALTERNATIVE_FILE)
    parser.add_argument("--target", default=config.TARGET_COLUMN)
    parser.add_argument("--id", default=config.ID_COLUMN, help="Optional ID column")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument(
        "--imputed-confidence", default=config.DEFAULT_IMPUTED_CONFIDENCE,
        choices=["High", "Medium", "Low", "TBD"],
        help="Preliminary internal flag; use evidence before overriding TBD.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute validation and save all applicable outputs."""
    reference = require_value(args.reference, "Reference file")
    masked = require_value(args.masked, "Masked file")
    baseline = require_value(args.baseline, "Baseline file")
    alternative = require_value(args.alternative, "Alternative file")
    target = require_value(args.target, "Target column")
    output_dir = require_value(args.output_dir, "Output directory")

    datasets = load_datasets(reference, masked, baseline, alternative)
    masked_positions = validate_datasets(datasets, target, args.id or None)
    paths = ensure_output_dirs(output_dir)

    metrics = calculate_validation_metrics(datasets, target, masked_positions)
    save_validation_metrics(metrics, paths["metrics"] / "validation_metrics.csv")
    target_type, _ = compare_distributions(datasets, target, paths["distributions"])
    status = prepare_status_confidence(
        datasets, target, masked_positions, args.id or None, args.imputed_confidence
    )
    status.to_csv(paths["distributions"] / "data_status_confidence.csv", index=False)

    ranking = None
    if target_type == "numeric":
        ranking = calculate_ranking_stability(datasets, target)
        ranking.to_csv(paths["downstream"] / "ranking_stability.csv", index=False)
    return {"metrics": metrics, "ranking": ranking, "masked": int(masked_positions.sum()), "target": target, "output": paths["root"]}


def print_summary(result: dict[str, object]) -> None:
    """Print a compact terminal summary of generated results."""
    print("\nValidation completed.\n")
    print(f"Target: {result['target']}")
    print(f"Masked values: {result['masked']}")
    metrics = result["metrics"]
    for _, row in metrics.iterrows():
        print(f"\n{row['method']}:")
        for key in ("mae", "rmse", "accuracy", "f1"):
            if row[key] == row[key]:
                print(f"{key.upper()}: {row[key]:.6g}")
    ranking = result["ranking"]
    if ranking is not None:
        print("\nRanking stability:")
        for _, row in ranking.iterrows():
            label = row["comparison"].replace("Reference vs ", "")
            value = row["spearman_correlation"]
            print(f"{label}: {value:.6g}" if value == value else f"{label}: N/A")
    else:
        print("\nRanking stability: skipped (numeric target required)")
    print(f"\nOutputs saved to:\n{result['output']}")


def main() -> int:
    """Parse arguments, run validation, and return a process exit code."""
    try:
        result = run(build_parser().parse_args())
        print_summary(result)
        return 0
    except ValidationInputError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected validation failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

