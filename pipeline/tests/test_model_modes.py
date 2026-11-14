"""Regression checks for configurable training, saved inference and tuning.

Run without pytest: python -m unittest discover -s tests -p test_model_modes.py
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.imputation.config import load_imputation_config
from src.pipeline import PipelineError, RunPaths, read_metadata, run_all, run_impute
from src.settings import load_settings


ROOT = Path(__file__).resolve().parents[1]


class ModelModesTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        source = ROOT / "data" / "transformers"
        self.measured = self.root / "measured.csv"
        self.truth = self.root / "truth.csv"
        selected_rows = None
        for filename, destination in (
            ("completion0.9-noise0.05-measured.csv", self.measured),
            ("completion0.9-noise0.05-truth.csv", self.truth),
        ):
            frame = pd.read_csv(source / filename)
            if selected_rows is None:
                selected_rows = frame.sample(n=230, random_state=219).index
            frame.loc[selected_rows].to_csv(destination, index=False)
        self.config_path = self.root / "imputation.yaml"
        self.config = yaml.safe_load((ROOT / "config/transformers/imputation.yaml").read_text())
        self.config["methods"]["predictive_missforest"].update(n_estimators=3, n_passes=1)
        self.config["model"]["path"] = str(self.root / "saved.joblib")
        self._write_config()
        self.settings = load_settings(asset="transformers").with_updates(
            imputation=self.config_path, method="predictive_missforest", reference_method=None,
            save_model=True, execution_mode="development",
        )

    def _write_config(self) -> None:
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def test_train_then_apply_saved_model_and_score(self) -> None:
        first = RunPaths(self.root / "train")
        run_impute(self.measured, first, self.settings)
        path = self.root / "saved.joblib"
        self.assertTrue(path.is_file())
        self.assertEqual(read_metadata(first)["imputation"]["mode"], "train_new")

        self.config["model"]["mode"] = "use_saved"
        self._write_config()
        second = RunPaths(self.root / "apply")
        has_jinja2 = importlib.util.find_spec("jinja2") is not None
        enabled = {"imputation", "validation"} | ({"reporting"} if has_jinja2 else set())
        stages = {name: name in enabled for name in self.settings.stages}
        run_all(self.measured, second, self.settings.with_updates(stages=stages),
                truth_csv=self.truth)
        self.assertEqual(read_metadata(second)["imputation"]["mode"], "apply")
        result = read_metadata(second)["evaluation"]
        self.assertGreater(result["n_scored_cells"], 0)
        self.assertEqual(result["protocol"], "saved_model_application")
        if has_jinja2:
            self.assertTrue(second.report.is_file())

    def test_missing_saved_model_is_clear_error(self) -> None:
        self.config["model"]["mode"] = "use_saved"
        self._write_config()
        with self.assertRaisesRegex(PipelineError, "does not exist"):
            run_impute(self.measured, RunPaths(self.root / "apply"), self.settings)

    def test_tuning_diagnostics_without_updating_selected(self) -> None:
        self.config["tuning"].update(
            enabled=True, update_selected=False, repeats=1,
            grid={"max_depth": [4], "min_samples_leaf": [1]},
        )
        self._write_config()
        run = RunPaths(self.root / "tune_only")
        run_impute(self.measured, run, self.settings)
        self.assertTrue((run.evaluation_dir / "tuning_candidates.csv").is_file())
        self.assertEqual(read_metadata(run)["tuning"]["updated_selected"], False)
        self.assertNotIn("selected_hyperparameters", yaml.safe_load(self.config_path.read_text()))

    def test_tuning_can_commit_parameters_and_provenance(self) -> None:
        self.config["tuning"].update(
            enabled=True, update_selected=True, repeats=1,
            grid={"max_depth": [4], "min_samples_leaf": [1]},
        )
        self._write_config()
        run = RunPaths(self.root / "tune_and_save")
        run_impute(self.measured, run, self.settings)
        updated = yaml.safe_load(self.config_path.read_text())
        self.assertEqual(updated["selected_hyperparameters"]["predictive_missforest"],
                         {"max_depth": 4, "min_samples_leaf": 1})
        self.assertIn("tuning_provenance", updated)
        self.assertTrue((run.evaluation_dir / "imputation_before_tuning.yaml").is_file())
        self.assertGreater(read_metadata(run)["tuning"]["test_rows_excluded"], 0)

    def test_saved_mode_cannot_tune(self) -> None:
        self.config["model"]["mode"] = "use_saved"
        self.config["tuning"]["enabled"] = True
        self._write_config()
        with self.assertRaisesRegex(ValueError, "must be false"):
            load_imputation_config(self.config_path)


if __name__ == "__main__":
    unittest.main()
