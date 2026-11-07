"""Static HTML report assembled from a run directory."""

from .report_generator import build_context, generate_report

__all__ = ["build_context", "generate_report"]
