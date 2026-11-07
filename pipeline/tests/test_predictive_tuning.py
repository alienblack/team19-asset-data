"""Tests for the validation-only tuning helpers (fast settings)."""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from src.imputation.predictive_missforest import PredictiveMissForest, PredictiveMissForestConfig
from src.imputation.predictive_missforest.predictive_tuning import (
    TuningObjective,
    column_permutation_importance,
    expand_grid,
    hide_cells,
    make_mcar_mask,
    tune_predictive_missforest,
)

from fixtures import CATEGORICAL, CONSTRAINTS, NUMERIC, make_complete_population, make_fit_frame

FAST = PredictiveMissForestConfig(n_estimators=10, n_passes=2, n_jobs=1, random_state=5)


def _tune_partition():
    truth = make_complete_population(120, seed=11)
    mask = make_mcar_mask(truth, ["installation_year", "material"], rate=0.3, seed=2)
    return hide_cells(truth, mask), truth, mask


def test_expand_grid_sizes():
    assert len(expand_grid()) == 12
    assert len(expand_grid({"max_depth": [4, None]})) == 2


def test_objective_formula():
    objective = TuningObjective(year_weight=0.5)
    assert abs(objective(10.0, 20.0, 0.6) - (0.5 * 0.5 + 0.5 * 0.4)) < 1e-12


def test_tuning_never_touches_truth_before_transform_and_returns_best():
    X_fit = make_fit_frame(200, seed=3)
    X_tune, truth, mask = _tune_partition()
    base = PredictiveMissForest(NUMERIC, CATEGORICAL, FAST, CONSTRAINTS)
    results, best = tune_predictive_missforest(
        base, X_fit, X_tune, truth, mask, grid={"max_depth": [4, None], "min_samples_leaf": [1]}
    )
    assert len(results) == 2
    assert results["objective"].is_monotonic_increasing
    assert {"year_mae", "material_macro_f1", "year_variance_ratio", "material_tv", "rank_average"} <= set(results.columns)
    assert best.max_depth is None or (isinstance(best.max_depth, int) and best.max_depth == 4)
    assert best.n_passes == FAST.n_passes
    # the winning config must be usable as-is (types preserved, validated)
    PredictiveMissForest(NUMERIC, CATEGORICAL, best, CONSTRAINTS)


def test_permutation_importance_runs_and_is_column_level():
    X_fit = make_fit_frame(200, seed=3)
    X_tune, truth, mask = _tune_partition()
    model = PredictiveMissForest(NUMERIC, CATEGORICAL, FAST, CONSTRAINTS).fit(X_fit)
    table = column_permutation_importance(model, X_tune, truth, mask, target="material", n_repeats=2)
    assert set(table["predictor"]) == set(model.features_["material"])
    assert np.isfinite(table["importance_mean"]).all()
    agg = model.impurity_importance_by_column("material")
    assert set(agg.index) == set(model.features_["material"])
    assert abs(agg.sum() - 1.0) < 1e-6


def test_repeated_holdouts_aggregate_and_prefer_simpler_candidates():
    fit_frame = make_fit_frame(600, seed=1)
    tune_frame = make_fit_frame(300, seed=2)
    base = PredictiveMissForest(NUMERIC, CATEGORICAL, PredictiveMissForestConfig(n_estimators=10, n_passes=2), CONSTRAINTS)
    grid = {"max_depth": [4, None], "min_samples_leaf": [1, 10]}
    results, best = tune_predictive_missforest(
        base, fit_frame, tune_frame, grid=grid, objective=TuningObjective(mode="all_attributes"),
        mask_rate=0.2, mask_seed=3, n_repeats=3,
    )
    assert len(results) == 4 and results["n_repeats"].eq(3).all()
    assert {"objective_mean", "objective_std", "selected", "candidate", "rank_average"} <= set(results.columns)
    assert results["selected"].sum() == 1
    per_repeat = results.attrs["per_repeat"]
    assert len(per_repeat) == 12 and set(per_repeat["repeat"]) == {0, 1, 2}
    assert len(set(per_repeat["mask_seed"])) == 3
    baseline = results.attrs["baseline"]
    assert len(baseline) == 3 and (baseline["candidate"] == "median_mode_baseline").all()
    # the chosen candidate is the simplest among those within one std of the best mean
    best_row = results.iloc[0]
    tolerated = results[results["objective_mean"] <= best_row["objective_mean"] + best_row["objective_std"]]
    simplest = tolerated.sort_values(["total_tree_nodes", "objective_mean", "grid_index"]).iloc[0]
    assert int(results.loc[results["selected"], "grid_index"].iloc[0]) == int(simplest["grid_index"])
    assert isinstance(best.min_samples_leaf, int) and (best.max_depth is None or isinstance(best.max_depth, int))


def test_single_repeat_keeps_backward_compatible_columns():
    fit_frame = make_fit_frame(350, seed=1)
    tune_frame = make_fit_frame(150, seed=2)
    base = PredictiveMissForest(NUMERIC, CATEGORICAL, PredictiveMissForestConfig(n_estimators=8, n_passes=2), CONSTRAINTS)
    results, best = tune_predictive_missforest(base, fit_frame, tune_frame, grid={"max_depth": [4, 8]}, mask_rate=0.2, mask_seed=1)
    assert {"objective", "year_mae", "material_macro_f1", "rank_average", "grid_index", "objective_mean"} <= set(results.columns)
    assert results["objective_std"].eq(0.0).all() and results["n_repeats"].eq(1).all()
    assert best.max_depth in (4, 8)
    with pytest.raises(ValueError, match="n_repeats"):
        tune_predictive_missforest(base, fit_frame, tune_frame, grid={"max_depth": [4]}, n_repeats=0)
