"""Fast regression tests: run pytest -q experiment/transformer/scripts/test_model_comparison.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from comparison_models import ChainedConditionalImputer, MedianMode, ModelSpec, TabPFNExperimentalImputer
from run_model_comparison import _score, _validate_output


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(17)
    n = 60
    x = np.arange(n, dtype=float)
    y = 2*x + rng.normal(0, 0.1, n)
    cat = np.where(x % 2 == 0, "0", "1").astype(object)
    source = pd.DataFrame({"x": x, "y": y, "model_id": cat})
    train = source.iloc[:45].copy()
    test = source.iloc[45:].copy()
    train.loc[train.index[::6], "x"] = np.nan
    train.loc[train.index[::5], "y"] = np.nan
    train.loc[train.index[::7], "model_id"] = None
    test.loc[test.index[::3], "x"] = np.nan
    test.loc[test.index[::4], "y"] = np.nan
    test.loc[test.index[::5], "model_id"] = None
    spec = ModelSpec(("x", "y"), ("model_id",), seed=17)
    return train, test, source.iloc[45:], spec


@pytest.mark.parametrize("family", ["ridge", "lightgbm"])
def test_chained_models_are_inductive_and_preserve_observations(synthetic, family):
    train, test, _, spec = synthetic
    model = ChainedConditionalImputer(spec, family=family, passes=2, lgbm_estimators=8).fit(train)
    first = model.transform(test)
    _validate_output(test, first, spec)
    assert len(model.models_) == 2*len(spec.columns)
    # No fitting is done by transform, and test targets are never supplied.
    model_ids_before = [id(est) for _, est in model.models_]
    second = model.transform(test)
    assert model_ids_before == [id(est) for _, est in model.models_]
    pd.testing.assert_frame_equal(first, second)


def test_baseline_matches_training_statistics_only(synthetic):
    train, test, _, spec = synthetic
    model = MedianMode(spec).fit(train)
    result = model.transform(test)
    _validate_output(test, result, spec)
    assert model.statistics_["x"] == train["x"].median()
    assert result.loc[test.x.isna(), "x"].eq(train.x.median()).all()


def test_score_includes_only_missing_test_cells(synthetic):
    train, test, truth, spec = synthetic
    baseline = MedianMode(spec).fit(train).transform(test)
    split = pd.DataFrame({"row_position": test.index,
                          "x_tile": 0, "y_tile": 0}, index=test.index)
    metrics, cells = _score(
        test, baseline, truth, split, spec, method="median_mode",
        completion=.8, noise=.05, seed=617, fit_seconds=0,
        transform_seconds=0, baseline_filled=baseline, export_cells=True)
    assert int(metrics.n_masked.sum()) == int(test.isna().sum().sum())
    assert len(cells) == int(test.isna().sum().sum())
    assert metrics.loc[metrics.variable_type=="numeric", "numeric_skill"].eq(0).all()


def test_tabpfn_is_explicitly_optional(synthetic, monkeypatch):
    train, _, _, spec = synthetic
    import sys
    monkeypatch.setitem(sys.modules, "tabpfn", None)
    model = TabPFNExperimentalImputer(spec, max_context_rows=30)
    with pytest.raises(ImportError, match="optional"):
        model.fit(train)


def test_tabpfn_output_mapping_and_observed_preservation(synthetic):
    # Unit-test the adapter's decoding without pretending to run TabPFN weights.
    train, test, _, spec = synthetic
    model = TabPFNExperimentalImputer(spec, max_context_rows=30)
    MedianMode.fit(model, train)
    model.vocab_ = {"model_id": ["0", "1"]}
    class FakeTabPFN:
        def impute(self, matrix, n_permutations=1):
            result = matrix.copy()
            for j in range(result.shape[1]):
                result[np.isnan(result[:, j]), j] = 0 if j==2 else 42.0
            return result
    model.model_ = FakeTabPFN()
    completed = model.transform(test)
    _validate_output(test, completed, spec)
    assert set(completed.model_id.unique()).issubset({"0", "1"})
    assert completed.loc[test.x.isna(), "x"].eq(42.0).all()
