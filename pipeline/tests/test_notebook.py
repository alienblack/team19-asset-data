"""pipeline.ipynb runs from a fresh kernel, cell by cell, and produces the deliverables.

The notebook is a controller: its code cells call the package and display
results. This test executes every code cell in order in one fresh namespace,
on small synthetic inputs selected through the ``OC_NOTEBOOK_*`` environment
variables the notebook honours, and checks the run directory afterwards.

When ``nbclient`` is installed (the team's Jupyter environment) the notebook
is executed in a real kernel; otherwise the cells are executed in-process with
the IPython display helpers stubbed, which is the same fresh-kernel sequence
without rich output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from sample_frames import MODEL_COLUMNS, inject_missing, make_conductor_frame, write_csv

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook" / "pipeline_overhead_conductors.ipynb"


def _code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        cells.append("".join(source) if isinstance(source, list) else source)
    return cells


def _execute_in_process(path: Path, cwd: Path) -> dict:
    """Run the code cells in order in one namespace (fresh-kernel semantics)."""
    import matplotlib

    matplotlib.use("Agg")
    stubbed: dict[str, types.ModuleType | None] = {}
    try:
        import IPython.display  # noqa: F401
    except ImportError:
        ipython = types.ModuleType("IPython")
        display = types.ModuleType("IPython.display")
        display.display = lambda *args, **kwargs: None
        display.Markdown = lambda text: text
        display.HTML = lambda text: text
        display.Image = lambda *args, **kwargs: None
        ipython.display = display  # type: ignore[attr-defined]
        # matplotlib inspects these when an IPython module is importable
        ipython.get_ipython = lambda: None  # type: ignore[attr-defined]
        ipython.version_info = (99, 0, 0, "")  # type: ignore[attr-defined]
        for name, module in (("IPython", ipython), ("IPython.display", display)):
            stubbed[name] = sys.modules.get(name)
            sys.modules[name] = module
    namespace: dict = {"__name__": "__main__"}
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        for index, source in enumerate(_code_cells(path)):
            exec(compile(source, f"<{path.name} cell {index}>", "exec"), namespace)
    finally:
        os.chdir(previous)
        for name, module in stubbed.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return namespace


def _execute_with_nbclient(path: Path, cwd: Path) -> None:
    import nbclient
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(cwd)}})
    client.execute()


def _inputs(tmp_path: Path) -> dict[str, Path]:
    complete = make_conductor_frame(240, seed=61, n_feeders=12)
    masked = inject_missing(complete, MODEL_COLUMNS, rate=0.2, seed=7)
    dirty = masked.copy()
    # Documented defects the cleaning stage must log: whitespace, label case, spelling variants.
    dirty.loc[dirty.index[:3], "material"] = [" aluminum ", "steel", "Cu"]
    dirty.loc[dirty.index[3:5], "location_group"] = ["urb", "RURAL"]
    dirty.loc[dirty.index[5], "corrosion_zone"] = "vs"
    # a few records keep asset_age while installation_year is missing, so deterministic recovery has work
    recoverable = dirty.index[dirty["installation_year"].isna()][:10]
    dirty.loc[recoverable, "asset_age"] = complete.loc[recoverable, "asset_age"]
    return {
        "input": write_csv(dirty, tmp_path / "dirty.csv"),
        "truth": write_csv(complete, tmp_path / "complete.csv"),
        "run_dir": tmp_path / "notebook_run",
    }


def test_code_cells_are_controllers_not_implementations():
    """No cell defines functions or classes: logic lives in the package, cells call it."""
    cells = _code_cells(NOTEBOOK)
    assert len(cells) >= 15
    for index, source in enumerate(cells):
        assert not re.search(r"^\s*(def|class)\s", source, flags=re.MULTILINE), f"cell {index} defines logic:\n{source}"
    joined = "\n".join(cells)
    assert "MASKED_CSV" not in joined, "notebooks must accept only measured data and truth"
    for stage in ("run_quality", "run_clean", "run_profile", "run_impute", "run_validate", "run_evaluate", "run_report", "ingest_csv", "profile_dataset", "plan_changes", "ConfigurationManager"):
        assert stage in joined, stage


def test_notebook_executes_from_a_fresh_kernel(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    for name, path in paths.items():
        monkeypatch.setenv(f"OC_NOTEBOOK_{name.upper()}", str(path))
    monkeypatch.setenv("OC_NOTEBOOK_OVERRIDES", json.dumps({"n_estimators": 8, "n_passes": 1}))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    approvals_before = hashlib.sha256((ROOT / "config" / "overhead_conductors" / "approvals.json").read_bytes()).hexdigest()

    try:
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401
    except ImportError:
        namespace = _execute_in_process(NOTEBOOK, ROOT)
        assert namespace["PROJECT_ROOT"] == ROOT
        assert namespace["INPUT_CSV"] == paths["input"]
    else:
        _execute_with_nbclient(NOTEBOOK, ROOT)

    run_dir = paths["run_dir"]
    for name in (
        "ingestion/ingestion_report.json", "source_records.csv", "quality_report.csv", "cleaned.csv", "cleaning_log.csv",
        "imputation_input.csv", "recovery_log.csv", "final_imputed.csv", "cell_provenance.csv", "provenance_summary.csv",
        "diagnostics.csv", "model.joblib", "validation/integrity_checks.csv", "validation/reconstruction_summary.csv",
        "evaluation/partition_manifest.csv", "evaluation/summary.csv", "evaluation/scored_cells.csv", "report.html", "run_metadata.json",
    ):
        assert (run_dir / name).is_file(), name
    assert not list(run_dir.glob("imputed.csv")), "legacy output name must not come back"

    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    for section in ("configuration", "quality", "cleaning", "profiling", "imputation", "validation", "evaluation", "report"):
        assert section in meta, section
    assert meta["configuration"]["refused"] is False
    assert meta["imputation"]["integrity_passed"] is True
    assert meta["imputation"]["hyperparameters"]["n_estimators"] == 8  # the override was recorded, not hidden
    assert meta["validation"]["integrity_passed"] is True
    assert meta["evaluation"]["mask_source"] == "masked_input"
    assert set(meta["evaluation"]["headline"]) == {"unseen_records", "transductive"}

    final = pd.read_csv(run_dir / "final_imputed.csv")
    assert len(final) == 240 and final[MODEL_COLUMNS].notna().all().all()
    cleaning_log = pd.read_csv(run_dir / "cleaning_log.csv")
    assert {"whitespace", "label_case", "spelling_variant"} <= set(cleaning_log["rule"])
    provenance = pd.read_csv(run_dir / "provenance_summary.csv").set_index("column")
    assert provenance.loc["installation_year", "derived"] > 0  # recovered from asset_age before imputation
    assert provenance.loc["material", "imputed"] > 0
    summary = pd.read_csv(run_dir / "evaluation" / "summary.csv")
    assert set(summary["protocol"]) == {"unseen_records", "transductive"}
    assert {"installation_year", "material"} <= set(summary["variable"])

    # the notebook must not touch the approval record unless asked to
    assert hashlib.sha256((ROOT / "config" / "overhead_conductors" / "approvals.json").read_bytes()).hexdigest() == approvals_before
