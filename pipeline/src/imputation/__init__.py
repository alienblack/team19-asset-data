"""Imputation methods and their configuration.

Public entry points:

* :func:`load_imputation_config` - parse ``imputation.yaml``;
* :func:`build_imputer` / :data:`METHODS` - construct a method by name;
* :class:`PredictiveMissForest` - the fit/transform imputer itself;
* :func:`tune_predictive_missforest` - validation-only hyperparameter search;
* :func:`save_imputer` / :func:`load_imputer` / :func:`check_provenance` - persistence.
"""

from .config import DataSplitConfig, ImputationConfig, WorkflowConfig, load_imputation_config
from .missforest.missforest import MissForestConfig, MissForestResult, missforest_impute
from .persistence import CompatibilityError, ProvenanceError, check_compatibility, check_provenance, load_imputer, model_fingerprint, read_provenance, save_imputer
from .postprocess import apply_constraints
from .predictive_missforest.predictive_missforest import PredictiveMissForest, PredictiveMissForestConfig, TransformResult
from .predictive_missforest.predictive_tuning import TuningObjective, expand_grid, tune_predictive_missforest
from .registry import (
    METHODS,
    BaseImputer,
    ImputationOutput,
    LegacyMissForestImputer,
    MedianModeImputer,
    PredictiveMissForestImputer,
    available_methods,
    build_imputer,
    register_method,
    resolve_hyperparameters,
)
from .median_mode.simple_imputer import deterministic_mode, simple_impute

__all__ = [
    "METHODS",
    "BaseImputer",
    "CompatibilityError",
    "DataSplitConfig",
    "ImputationConfig",
    "ImputationOutput",
    "LegacyMissForestImputer",
    "MedianModeImputer",
    "MissForestConfig",
    "MissForestResult",
    "PredictiveMissForest",
    "PredictiveMissForestConfig",
    "PredictiveMissForestImputer",
    "ProvenanceError",
    "TransformResult",
    "TuningObjective",
    "WorkflowConfig",
    "apply_constraints",
    "available_methods",
    "build_imputer",
    "register_method",
    "check_compatibility",
    "check_provenance",
    "deterministic_mode",
    "expand_grid",
    "load_imputation_config",
    "load_imputer",
    "missforest_impute",
    "model_fingerprint",
    "read_provenance",
    "resolve_hyperparameters",
    "save_imputer",
    "simple_impute",
    "tune_predictive_missforest",
]
