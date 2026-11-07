"""Display helpers for the two controllers in ``notebook/``.

The notebook is a controller: every processing step is a call into the
package (``pipeline.run_*``, ``configuration.ConfigurationManager``, the
ingestion and profiling functions). This module holds only what the
notebook needs to *show* results - path shortening, table truncation,
figure display - so no transformation logic lives in a cell.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def display_path(path: str | Path, root: str | Path) -> str:
    """Project-relative path for display; external files are shown by name only."""
    path = Path(path).expanduser().resolve()
    root = Path(root).resolve()
    try:
        return f"{root.name}/{path.relative_to(root).as_posix()}"
    except ValueError:
        return f"[external]/{path.name}"


def notebook_paths(root: Path, defaults: Mapping[str, str]) -> dict[str, Path]:
    """Resolve the notebook's input/output paths, allowing environment overrides.

    ``defaults`` maps a name (``input``, ``truth``, ``run_dir``) to a
    project-relative default; an environment variable
    ``OC_NOTEBOOK_<NAME>`` overrides it, which is how the automated test runs
    the notebook on a small file without editing it.
    """
    resolved: dict[str, Path] = {}
    for name, default in defaults.items():
        override = os.environ.get(f"OC_NOTEBOOK_{name.upper()}")
        value = Path(override) if override else root / default
        resolved[name] = value if value.is_absolute() else (root / value)
    return resolved


def table(frame: pd.DataFrame | None, max_rows: int = 20, max_colwidth: int = 90) -> pd.DataFrame:
    """A truncated copy for display (the full table stays in the run directory)."""
    if frame is None:
        return pd.DataFrame()
    pd.set_option("display.max_colwidth", max_colwidth)
    pd.set_option("display.width", 200)
    return frame.head(max_rows).copy()


def key_values(mapping: Mapping[str, Any], keys: Iterable[str] | None = None) -> pd.DataFrame:
    """A two-column table from a mapping (for metadata sections)."""
    keys = list(keys) if keys is not None else list(mapping)
    return pd.DataFrame([{"setting": k, "value": mapping.get(k)} for k in keys])


def missingness_before_after(before: pd.DataFrame, after: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Per-column missing counts before and after imputation."""
    rows = []
    for column in columns:
        rows.append(
            {
                "column": column,
                "missing_before": int(before[column].isna().sum()) if column in before else None,
                "missing_after": int(after[column].isna().sum()) if column in after else None,
            }
        )
    out = pd.DataFrame(rows)
    out["filled"] = out["missing_before"] - out["missing_after"]
    return out


def show_figures(figures: Mapping[str, Any]) -> None:
    """Display in-memory matplotlib figures in a notebook (no-op elsewhere) and close them."""
    try:
        from IPython.display import display
    except ImportError:  # pragma: no cover - plain Python
        display = None
    import matplotlib.pyplot as plt

    for figure in figures.values():
        if display is not None:
            display(figure)
        plt.close(figure)


def show_images(paths: Iterable[Path], width: int = 900) -> None:
    """Display saved PNG figures in a notebook (no-op elsewhere)."""
    try:
        from IPython.display import Image, display
    except ImportError:  # pragma: no cover - plain Python
        return
    for path in paths:
        path = Path(path)
        if path.is_file():
            display(Image(filename=str(path), width=width))


def list_outputs(run_dir: Path, root: Path) -> pd.DataFrame:
    """Every file in a run directory with its size, as project-relative paths."""
    rows = []
    for path in sorted(Path(run_dir).rglob("*")):
        if path.is_file():
            rows.append({"file": display_path(path, root), "size_kb": round(path.stat().st_size / 1024, 1)})
    return pd.DataFrame(rows, columns=["file", "size_kb"])
