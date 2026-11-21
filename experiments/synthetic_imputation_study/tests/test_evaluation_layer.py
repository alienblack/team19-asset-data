"""Checks for the evaluation layer: shown names, main models, saved evaluation tables and the notebook itself."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from synthetic_imputation import failure as fl
from synthetic_imputation.models import NUMERIC_METHODS as FITTED_NUMERIC_METHODS
from synthetic_imputation.names import CATEGORY_METHODS, NUMERIC_METHODS, shown
from synthetic_imputation.runner import MODEL_SOURCES
from synthetic_imputation.study import EVALUATION_SOURCES, MODEL_FAMILIES, SavedTables, _by_model

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "imputation.ipynb"


def test_shown_names_rename_only_the_mixture_model():
    assert shown("Gaussian EM") == "GMM"
    assert shown("Random Forest") == "Random Forest"
    assert len(NUMERIC_METHODS) == len(FITTED_NUMERIC_METHODS)
    assert "GMM" in NUMERIC_METHODS and "GMM" in CATEGORY_METHODS
    assert "Gaussian EM" not in NUMERIC_METHODS + CATEGORY_METHODS


def test_shown_names_stay_unique_across_model_sections():
    names = [shown(method) for methods in MODEL_FAMILIES.values() for method in methods]
    assert len(names) == len(set(names))


def test_main_models_are_one_tree_based_and_one_joint_model():
    assert fl.MAIN_METHODS == ("Random Forest", "GMM")
    assert fl.LEAD_METHOD == "Random Forest"
    assert fl.GROUP_OF["Random Forest"] == "chained, selected settings"
    assert fl.GROUP_OF["GMM"] == "joint models"


def test_every_model_belongs_to_exactly_one_group():
    grouped = [method for methods in fl.MODEL_GROUPS.values() for method in methods]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(fl.SHOWN_METHODS)


def test_evaluation_code_is_outside_the_run_contract():
    source = ROOT / "src" / "synthetic_imputation"
    for name in EVALUATION_SOURCES:
        assert (source / name).is_file()
    # columns.py describes the data for both layers, so it is part of the contract and of the table key.
    assert set(EVALUATION_SOURCES) & set(MODEL_SOURCES) == {"columns.py"}


@pytest.fixture
def store(tmp_path):
    (tmp_path / "contract.json").write_text("{}", encoding="utf-8")
    return SavedTables(tmp_path)


def test_saved_tables_are_reused_only_under_the_same_key(store):
    calls = []

    def build():
        calls.append(1)
        return {"value": len(calls)}

    first, reused_first = store.get("table", "key-1", build)
    second, reused_second = store.get("table", "key-1", build)
    third, reused_third = store.get("table", "key-2", build)
    assert (reused_first, reused_second, reused_third) == (False, True, False)
    assert first == second == {"value": 1} and third == {"value": 2}
    assert len(calls) == 2


def test_table_key_follows_the_finished_tasks_and_not_their_order(store):
    log = pd.DataFrame({"task_id": ["a", "b", "c"], "status": ["success"] * 3})
    failed = log.assign(status=["success", "failed", "success"])
    assert store.key(log) == store.key(log.iloc[::-1])
    assert store.key(log) != store.key(failed)
    assert store.key(log) != store.key(log.iloc[:2])


def test_table_key_changes_with_the_run_contract(tmp_path):
    log = pd.DataFrame({"task_id": ["a"], "status": ["success"]})
    keys = []
    for contract in ("{}", '{"config": 1}'):
        (tmp_path / "contract.json").write_text(contract, encoding="utf-8")
        keys.append(SavedTables(tmp_path).key(log))
    assert keys[0] != keys[1]


def test_stacked_tables_keep_the_model_and_the_row_labels():
    first = pd.DataFrame({"skill": [0.5, 0.6]}, index=pd.Index(["x", "y"], name="target"))
    second = pd.DataFrame({"skill": [0.1]}, index=pd.Index(["x"], name="target"))
    stacked = _by_model({"Random Forest": first, "GMM": second})
    assert list(stacked.columns) == ["target", "skill", "model"]
    assert stacked.model.tolist() == ["Random Forest", "Random Forest", "GMM"]


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_runs_without_errors_and_every_code_cell_has_run():
    cells = [c for c in _notebook()["cells"] if c["cell_type"] == "code"]
    assert all(c.get("execution_count") for c in cells)
    assert not [o for c in cells for o in c.get("outputs", []) if o["output_type"] == "error"]


def test_notebook_cells_have_unique_ids():
    ids = [c.get("id") for c in _notebook()["cells"]]
    assert all(ids) and len(ids) == len(set(ids))


def test_notebook_text_uses_the_shared_model_names_and_neutral_wording():
    text = "\n".join("".join(c["source"]) for c in _notebook()["cells"])
    for term in ("Gaussian EM", "Subteam", "team pipeline", "this study's"):
        assert term not in text, term
    assert not re.search(r"\b(our|ours|we)\b", text, re.IGNORECASE)


def test_notebook_headings_name_their_content_without_question_words():
    headings = [line for c in _notebook()["cells"] if c["cell_type"] == "markdown"
                for line in "".join(c["source"]).splitlines() if line.startswith("#")]
    assert len(headings) > 60
    for heading in headings:
        assert not re.search(r"\b(when|how|why|what|whether)\b", heading, re.IGNORECASE), heading


LABEL = r"(?:[0-9]+\.[0-9]+[a-z]?|A\.[0-9][a-z]?)"


def _captions():
    """Labels of the numbered captions shown in the outputs, in order."""
    labels = []
    for cell in _notebook()["cells"]:
        for output in cell.get("outputs", []):
            text = "".join(output.get("data", {}).get("text/markdown", ""))
            match = re.match(rf"\*\*((?:Table|Figure) {LABEL})\.\*\* \S", text)
            if match:
                labels.append(match.group(1))
    return labels


def test_tables_and_figures_carry_unique_numbered_captions():
    labels = _captions()
    assert len(labels) > 100
    assert len(labels) == len(set(labels)), [label for label in labels if labels.count(label) > 1]


def test_every_table_and_figure_cited_in_the_text_is_shown():
    shown_labels = set(_captions())
    text = "\n".join("".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "markdown")
    cited = set()
    for kind, group in re.findall(rf"\b(Table|Figure)s? ({LABEL}(?:(?:,| and| to) {LABEL})*)", text):
        cited.update(f"{kind} {label}" for label in re.findall(LABEL, group))
    assert cited, "the text cites no table or figure"
    assert not cited - shown_labels, sorted(cited - shown_labels)
