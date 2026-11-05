"""Checks for the imputation study code, using small frames and no dataset files."""

import numpy as np
import pandas as pd
import pytest

from team19_asset_data.imputation_study.columns import (
    BY_NAME, COMPANIONS, GENERATOR_LINKS, NAMES, TARGETS, finalise,
)
from team19_asset_data.imputation_study.data import FILE_PATTERN, relationship_residuals
from team19_asset_data.imputation_study.evaluation import (
    class_scores, default_comparison, numeric_scores, visible_group,
)
from team19_asset_data.imputation_study.models import (
    MODEL_CLASSES, BaselineModel, ChainedLinearImputer, Features, MiceforestModel, make_model,
)
from team19_asset_data.imputation_study.study import MODEL_FAMILIES
from team19_asset_data.imputation_study.runner import (
    ALL_NUMERIC, ITERATIVE, Task, build_plan, build_tuning_plan, candidates, default_parameters,
    fold_ids, quick_config, settings_for,
)


def small_frame(rows=200, seed=0):
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "signal": generator.normal(size=rows),
        "model_id": generator.integers(0, 3, rows).astype(float),
    })
    frame["target"] = 3 * frame.signal
    # With one numeric predictor, a gap leaves nan_euclidean nothing to compare,
    # so a complete column keeps the neighbour search meaningful, as in the real data.
    frame["complete"] = frame.signal + generator.normal(scale=0.1, size=rows)
    frame.loc[::7, "signal"] = np.nan
    frame.loc[::5, "target"] = np.nan
    frame.loc[::9, "model_id"] = np.nan
    return frame


def test_every_column_is_a_target_and_is_described_once():
    assert len(NAMES) == 15 == len(set(NAMES)) == len(TARGETS)
    assert set(COMPANIONS) == set(GENERATOR_LINKS) == set(TARGETS) == set(NAMES)


def test_only_measured_readings_are_noisy():
    noisy = {name for name in NAMES if BY_NAME[name].noisy}
    assert noisy == {"measured_humid", "measured_salt", "measured_overgrowth", "random_placebo",
                     "measured_gas_a", "measured_gas_b", "measured_gas_c"}


def test_finalise_applies_the_column_definition():
    assert finalise("x", [-2.0, 3.4, 120.0]).tolist() == [0.0, 3.0, 99.0]
    assert finalise("measured_humid", [-0.1, 0.25]).tolist() == [0.0, 0.25]
    assert finalise("altitude", [90.0, 150.4, 400.0]).tolist() == [100.0, 150.0, 298.0]
    assert finalise("life", [120_000.0]).tolist() == [100_000.0]


def test_interval_columns_use_absolute_tolerance():
    for name in ("x", "y", "install_date", "altitude", "life"):
        assert BY_NAME[name].tolerance_kind == "absolute"


def test_tolerance_excludes_true_zero_from_relative_denominator():
    scores = numeric_scores("measured_overgrowth", [0.0, 1.0, 1.0], [0.5, 1.05, 2.0])
    assert scores["tolerance cells"] == 2
    assert scores["within tight"] == pytest.approx(0.5)


def test_failed_predictions_count_as_misses():
    scores = numeric_scores("x", [1.0, 2.0], [1.0, np.nan])
    assert scores["coverage"] == 0.5
    assert scores["within tight"] == 0.5


def test_class_scores_report_accuracy():
    assert class_scores([0, 1, 1, 2], [0, 1, 0, 2])["accuracy"] == 0.75


def test_features_keep_gaps_and_treat_category_as_nominal():
    frame = small_frame()
    matrix = Features(scale=False).fit(frame[["signal", "model_id"]]).transform(frame[["signal", "model_id"]])
    assert np.isnan(matrix[0, 0])
    assert matrix.shape[1] == 1 + 3 + 1
    assert matrix[:, 1:].sum(axis=1).tolist() == [1.0] * len(frame)


@pytest.mark.parametrize("method", ["KNN imputer", "Random Forest", "HistGradientBoosting",
                                    "CatBoost", "miceforest direct"])
def test_models_handle_missing_predictors_and_learn_a_simple_signal(method):
    config = quick_config()
    frame = small_frame(600)
    train, test = frame.iloc[:500], frame.iloc[500:]
    hidden = test.target.isna() & test.signal.notna()
    settings = settings_for(method, "target", config)
    prediction = make_model(method, "target", config, settings, 0).fit(train).predict(test.loc[hidden])
    error = np.abs(prediction - 3 * test.loc[hidden, "signal"]).mean()
    assert np.isfinite(prediction).all()
    assert error < 0.8 * np.abs(3 * test.loc[hidden, "signal"]).mean()


def test_baseline_returns_the_observed_median():
    frame = small_frame()
    model = BaselineModel("target", quick_config(), {}, 0).fit(frame)
    assert model.predict(frame.iloc[:3]).tolist() == [frame.target.median()] * 3


def test_classifiers_return_class_probabilities_with_missing_predictors():
    config = quick_config()
    frame = small_frame(300).assign(model_id=lambda f: (f.complete > 0).astype(float))
    for method in ("KNN", "Random Forest", "HistGradientBoosting", "CatBoost", "miceforest direct"):
        model = make_model(method, "model_id", config, settings_for(method, "model_id", config), 0).fit(frame)
        probabilities = model.predict_proba(frame.iloc[:5])
        assert probabilities.shape == (5, 2) and np.allclose(probabilities.sum(axis=1), 1)


def test_pmm_returns_one_column_per_draw():
    config = quick_config()
    frame = small_frame(300).drop(columns="model_id")
    settings = settings_for("miceforest direct", "target", config)
    model = MiceforestModel("target", config, settings, 0, draws=config.pmm_draws).fit(frame)
    hidden = frame[frame.target.isna()]
    assert model.draw(hidden).shape == (len(hidden), config.pmm_draws)


def test_every_planned_method_belongs_to_one_notebook_section():
    config = quick_config()
    planned = {task.method for task in build_plan(list(config.scenarios), config)}
    planned |= {task.method for task in build_tuning_plan(config)}
    sections = [method for methods in MODEL_FAMILIES.values() for method in methods]
    assert planned == set(sections) and len(sections) == len(set(sections))
    assert set(MODEL_CLASSES) <= set(sections)


def test_folds_are_a_partition_and_repeatable():
    first, second = fold_ids(103, 5, 219), fold_ids(103, 5, 219)
    assert (first == second).all() and set(first) == set(range(5))


def test_chained_imputer_recovers_a_linear_relation_in_original_units():
    frame = small_frame(600)
    frame = frame.assign(target=3 * frame.complete).drop(columns=["model_id", "signal"])
    frame.loc[::5, "target"] = np.nan
    train, test = frame.iloc[:500], frame.iloc[500:]
    filled = ChainedLinearImputer(default_parameters()[ITERATIVE], 0).fit(train).predict(test)
    hidden = test.target.isna()
    assert np.allclose(filled.loc[hidden, "target"], 3 * test.loc[hidden, "complete"], atol=0.05)
    assert np.allclose(filled.loc[~hidden, "target"], test.loc[~hidden, "target"])


def test_plan_has_unique_tasks_and_skips_targets_inside_a_removed_block():
    config = quick_config()
    plan = build_plan(list(config.scenarios), config)
    assert len({task.task_id for task in plan}) == len(plan)
    assert all(isinstance(task, Task) for task in plan)
    for task in plan:
        if task.experiment == "ablation":
            assert task.target not in config.blocks[task.variant]


def test_iterative_imputer_is_one_task_per_fold():
    config = quick_config()
    plan = build_plan(list(config.scenarios), config)
    tasks = [t for t in plan if t.experiment == "main" and t.method == ITERATIVE]
    assert {t.target for t in tasks} == {ALL_NUMERIC}
    assert len(tasks) == len(config.scenarios) * config.folds


def test_tuning_always_includes_the_default_settings():
    config = quick_config()
    assert candidates("Random Forest", config)[0] == {}
    assert candidates("Random Forest", config) == candidates("Random Forest", config)
    assert {t.experiment for t in build_tuning_plan(config)} == {"tuning"}


def test_tuned_settings_apply_only_outside_the_default_experiment():
    config = quick_config(tuned={"CatBoost|x": {"depth": 4}})
    assert settings_for("CatBoost", "x", config)["depth"] == 4
    assert settings_for("CatBoost", "x", config, tuned=False)["depth"] == 6
    assert settings_for("CatBoost", "y", config)["depth"] == 6


def test_default_comparison_handles_a_perfect_default():
    config = quick_config()
    numeric = pd.DataFrame({"scenario": config.representative, "seed": config.seed, "target": "x",
                            "method": "CatBoost", "experiment": ["main", "defaults"], "MAE": [1.0, 2.0]})
    category = pd.DataFrame({"scenario": config.representative, "seed": config.seed, "target": "model_id",
                             "method": "CatBoost", "experiment": ["main", "defaults"], "accuracy": [1.0, 1.0]})
    table = default_comparison(numeric, category, config).set_index("target")
    assert table.loc["x", "relative change"] == pytest.approx(-0.5)
    assert table.loc["model_id", "relative change"] == 0.0


def test_relationship_residuals_detect_a_broken_definition():
    truth = pd.DataFrame({
        "x": [1.0], "y": [2.0], "altitude": [103.0], "install_date": [10.0], "model_id": [3.0],
        "measured_humid": [0.5], "measured_salt": [0.4], "measured_overgrowth": [0.6],
        "measured_gas_a": [10.0], "measured_gas_b": [10.0], "measured_gas_c": [10.0], "total_gas": [30.0],
    })
    truth["life"] = 100000 - 25000 * (truth.total_gas / 300000) - 10 * 0.6
    table = relationship_residuals(truth).set_index("definition")["largest residual"]
    assert table.max() == pytest.approx(0.0, abs=1e-9)
    broken = truth.assign(altitude=104.0)
    assert relationship_residuals(broken).iloc[0]["largest residual"] == 1.0


def test_file_pattern_reads_both_controls():
    match = FILE_PATTERN.match("completion0.8-noise0.05-measured.csv")
    assert (match["completion"], match["noise"], match["kind"]) == ("0.8", "0.05", "measured")


def test_groups_use_visibility_of_companions():
    frame = pd.DataFrame({"x": [1.0, np.nan], "y": [np.nan, np.nan]})
    assert visible_group("altitude", frame).tolist() == ["x seen · y missing", "x missing · y missing"]
