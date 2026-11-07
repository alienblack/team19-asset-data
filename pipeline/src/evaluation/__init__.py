"""Evaluation: holdouts, metrics, integrity, distribution and confidence diagnostics."""

from .confidence import ConfidenceResult, low_confidence_cells, rank_confidence
from .distribution_analysis import compare_distributions, joint_distribution_shift, total_variation, wasserstein_1
from .holdout import (
    PARTITIONS,
    hide_cells,
    load_or_create_manifest,
    make_partitions,
    prepare_holdout,
    repeated_holdouts,
    split_by_manifest,
    validate_alignment,
)
from .integrity_checks import IntegrityResult, check_integrity
from .metrics import evaluate_imputation, per_class_report, prepare_evaluation_holdout

__all__ = [
    "PARTITIONS",
    "ConfidenceResult",
    "IntegrityResult",
    "check_integrity",
    "compare_distributions",
    "evaluate_imputation",
    "hide_cells",
    "joint_distribution_shift",
    "load_or_create_manifest",
    "low_confidence_cells",
    "make_partitions",
    "per_class_report",
    "prepare_evaluation_holdout",
    "prepare_holdout",
    "rank_confidence",
    "repeated_holdouts",
    "split_by_manifest",
    "total_variation",
    "validate_alignment",
    "wasserstein_1",
]
