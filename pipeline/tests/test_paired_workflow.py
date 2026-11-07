"""Dependency-light regression checks for the refactored multi-asset workflow.

Run without pytest via ``python -m unittest discover -s tests -p test_paired_workflow.py``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src.imputation.config import load_imputation_config
from src.imputation.registry import METHOD_FACTORIES, MedianModeImputer, available_methods, build_imputer, register_method
from src.ingestion.schema import load_schema
from src.pipeline import RunPaths, run_all, run_evaluate
from src.settings import load_settings


ROOT = Path(__file__).resolve().parents[1]


class PairedInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.tmp = Path(self.scratch.name)
        sample = ROOT / "data" / "transformers"
        self.measured = self.tmp / "measured.csv"
        self.truth = self.tmp / "truth.csv"
        pd.read_csv(sample / "completion0.9-noise0.05-measured.csv", dtype=str).head(220).to_csv(self.measured, index=False)
        pd.read_csv(sample / "completion0.9-noise0.05-truth.csv", dtype=str).head(220).to_csv(self.truth, index=False)
        self.settings = load_settings(asset="transformers")

    def test_all_with_exactly_two_source_files(self) -> None:
        run = RunPaths(self.tmp / "run")
        stages = {**self.settings.stages, "reporting": False}
        result = run_all(self.measured, run, self.settings.with_updates(stages=stages),
                         method="median_mode", truth_csv=self.truth)
        self.assertIn("evaluation", result)
        self.assertTrue(run.imputed.is_file())
        self.assertTrue((run.evaluation_dir / "summary.csv").is_file())
        self.assertTrue((run.evaluation_dir / "observed_difference_summary.csv").is_file())
        scored = pd.read_csv(run.evaluation_dir / "scored_cells.csv")
        self.assertGreater(len(scored), 0)
        observed = pd.read_csv(run.evaluation_dir / "observed_difference_summary.csv")
        self.assertGreater(int(observed.observed_different.sum()), 0)
        self.assertEqual(result["evaluation"]["mask_source"], "masked_input")

    def test_shuffled_truth_is_rejected(self) -> None:
        shifted = self.tmp / "shuffled.csv"
        pd.read_csv(self.truth, dtype=str).iloc[::-1].to_csv(shifted, index=False)
        with self.assertRaisesRegex(ValueError, "row alignment failed"):
            run_evaluate(RunPaths(self.tmp / "bad"), self.settings,
                         truth_csv=shifted, input_csv=self.measured, methods=["median_mode"])

    def test_new_method_can_register_without_editing_shared_scorer(self) -> None:
        schema = load_schema(self.settings.data_schema)
        config = load_imputation_config(self.settings.imputation, schema)
        name = "unit_test_plugin"
        register_method(name, lambda conf, params: MedianModeImputer(
            conf.numeric_columns, conf.categorical_columns, conf.constraints))
        try:
            self.assertIn(name, available_methods())
            self.assertIsInstance(build_imputer(name, config), MedianModeImputer)
        finally:
            METHOD_FACTORIES.pop(name, None)


if __name__ == "__main__":
    unittest.main()
