"""Schema-free discovery and optional schema-aware diagnostic utilities."""

from .profiler import ProfileResult, cramers_v, eta_squared, profile_dataset, profile_records
from .visualization import render_discovery_figures, render_imputation_figures, render_profile_figures, render_reconstruction_figure

__all__ = [
    "ProfileResult",
    "cramers_v",
    "eta_squared",
    "profile_dataset",
    "profile_records",
    "render_discovery_figures",
    "render_imputation_figures",
    "render_profile_figures",
    "render_reconstruction_figure",
]
