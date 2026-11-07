"""Import boundaries: leaf stages must not pull in orchestration or model code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _modules_loaded_after(statement: str) -> set[str]:
    code = f"import sys; {statement}; print('\\n'.join(sorted(m for m in sys.modules if m.startswith('src'))))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_ingestion_stage_imports_no_model_or_orchestration_code():
    loaded = _modules_loaded_after("import src.ingestion.stage")
    assert not any(m.startswith(("src.imputation", "src.pipeline", "src.reporting")) for m in loaded)


def test_cli_module_import_is_lazy_about_the_registry():
    loaded = _modules_loaded_after("import src.cli")
    assert "src.imputation.registry" not in loaded
    assert "src.pipeline" not in loaded


def test_profiling_does_not_import_imputation():
    loaded = _modules_loaded_after("import src.profiling")
    assert not any(m.startswith("src.imputation") for m in loaded)
