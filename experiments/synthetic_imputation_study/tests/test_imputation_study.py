"""Checks for the imputation study code, using small frames and no dataset files."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PolynomialFeatures

from team19_asset_data.imputation_study.columns import (
    BY_NAME, COMPANIONS, GENERATOR_LINKS, NAMES, TARGETS, finalise,
)
from team19_asset_data.imputation_study.data import FILE_PATTERN, relationship_residuals
from team19_asset_data.imputation_study.evaluation import (
    class_scores, cost_table, numeric_scores, visible_group,
)
from team19_asset_data.imputation_study.models import (
    CHAINED_CLASSES, MODEL_CLASSES, BaselineModel, BlockRidge, ChainedRidgeImputer, Features, JointModel,
    MiceforestModel, make_model,
)
from team19_asset_data.imputation_study import relationships as rel
from team19_asset_data.imputation_study.study import MODEL_FAMILIES
from team19_asset_data.imputation_study.runner import (
    ALL_NUMERIC, CHAIN_SETTINGS, ITERATIVE, MODEL_SOURCES, Task, build_plan, default_parameters, fold_ids,
    quick_config,
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


@pytest.mark.parametrize("method", ["Random Forest", "CatBoost"])
def test_single_target_models_handle_missing_predictors_and_learn_a_simple_signal(method):
    config = quick_config()
    frame = small_frame(600)
    train, test = frame.iloc[:500], frame.iloc[500:]
    hidden = test.target.isna() & test.signal.notna()
    settings = config.explanation[method]
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
    for method in ("Random Forest", "CatBoost"):
        settings = config.explanation[method]
        model = make_model(method, "model_id", config, settings, 0).fit(frame)
        probabilities = model.predict_proba(frame.iloc[:5])
        assert probabilities.shape == (5, 2) and np.allclose(probabilities.sum(axis=1), 1)


def test_pmm_returns_one_column_per_draw():
    config = quick_config()
    frame = small_frame(300).drop(columns="model_id")
    model = MiceforestModel("target", config, config.parameters["miceforest PMM"], 0, config.pmm_draws).fit(frame)
    hidden = frame[frame.target.isna()]
    assert model.draw(hidden).shape == (len(hidden), config.pmm_draws)


def test_miceforest_runs_as_gradient_boosting():
    config = quick_config()
    model = CHAINED_CLASSES["miceforest direct"](config.parameters["miceforest direct"], 0).fit(chained_frame())
    text = model.kernel.get_model("total", 0).model_to_string()
    assert "[boosting: gbdt]" in text


def test_joint_model_returns_completed_copies_with_observed_cells_unchanged():
    config = quick_config()
    frame = chained_frame()
    train, test = frame.iloc[:400], frame.iloc[400:]
    copies = JointModel(config, config.parameters["Joint miceforest PMM"], 0).fit(train).predict(test)
    assert len(copies) == config.pmm_draws
    for copy in copies:
        assert copy.notna().all().all()
        assert np.allclose(copy.where(test.notna()).fillna(0), test.fillna(0))


def test_every_planned_method_belongs_to_one_notebook_section():
    config = quick_config()
    planned = {task.method for task in build_plan(list(config.scenarios), config)}
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
    settings = {**default_parameters()[ITERATIVE], "linear_alpha": 0.01}
    filled = ChainedRidgeImputer(settings, 0).fit(train).predict(test)
    hidden = test.target.isna()
    assert np.allclose(filled.loc[hidden, "target"], 3 * test.loc[hidden, "complete"], atol=0.05)
    assert np.allclose(filled.loc[~hidden, "target"], test.loc[~hidden, "target"])


def test_block_ridge_fits_a_product_that_a_line_cannot():
    generator = np.random.default_rng(0)
    matrix = generator.normal(size=(800, 3))
    labels = matrix[:, 0] * matrix[:, 1]
    predicted = BlockRidge(1.0, 1.0).fit(matrix, labels).predict(matrix)
    assert np.abs(predicted - labels).mean() < 0.05


def test_block_ridge_applies_a_separate_penalty_to_each_block():
    generator = np.random.default_rng(1)
    matrix = generator.normal(size=(400, 2))
    labels = matrix[:, 0] + matrix[:, 0] * matrix[:, 1]
    linear, curve = 5.0, 50.0
    model = BlockRidge(linear, curve).fit(matrix, labels)
    terms = PolynomialFeatures(2, include_bias=False).fit_transform(matrix)
    centred, target = terms - terms.mean(axis=0), labels - labels.mean()
    penalty = np.diag([linear] * 2 + [curve] * 3)
    expected = centred @ np.linalg.solve(centred.T @ centred + penalty, centred.T @ target) + labels.mean()
    assert np.allclose(model.predict(matrix), expected)


def chained_frame(rows=500):
    generator = np.random.default_rng(3)
    frame = pd.DataFrame({"a": generator.normal(size=rows), "b": generator.normal(size=rows)})
    frame["model_id"] = (frame.a > 0).astype(float) + 2 * (frame.b > 0)
    frame["total"] = frame.a + 2 * frame.b
    for name, step in (("a", 7), ("b", 5), ("model_id", 6), ("total", 4)):
        frame.loc[frame.index[::step], name] = np.nan
    return frame


@pytest.mark.parametrize("method", list(CHAINED_CLASSES))
def test_chained_models_fill_every_numeric_column_and_give_class_probabilities(method):
    config = quick_config()
    frame = chained_frame()
    train, test = frame.iloc[:400], frame.iloc[400:]
    model = CHAINED_CLASSES[method](config.parameters[method], 0).fit(train)
    numeric, shares = model.predict(test)
    assert list(numeric.index) == list(test.index) and numeric.notna().all().all()
    assert "model_id" not in numeric and set(numeric) == {"a", "b", "total"}
    assert shares.shape == (100, len(model.classes_)) and np.allclose(shares.sum(axis=1), 1)


@pytest.mark.parametrize("method", list(CHAINED_CLASSES))
def test_chained_estimate_of_a_row_does_not_depend_on_other_test_rows(method):
    config = quick_config()
    frame = chained_frame()
    train, test = frame.iloc[:400], frame.iloc[400:]
    model = CHAINED_CLASSES[method](config.parameters[method], 0).fit(train)
    together, shares = model.predict(test)
    alone, alone_shares = model.predict(test.iloc[:7])
    assert np.allclose(together.iloc[:7], alone) and np.allclose(shares[:7], alone_shares)


def test_chains_in_ablation_drop_the_block_and_keep_model_id():
    config = quick_config()
    frame = chained_frame()
    train, test = frame.iloc[:400], frame.iloc[400:]
    model = CHAINED_CLASSES["Random Forest"](config.parameters["Random Forest"], 0, ("total",)).fit(train)
    numeric, shares = model.predict(test)
    assert "total" not in numeric and shares.shape[0] == len(test)


def test_cost_shares_a_chained_task_between_the_columns_it_covers():
    log = pd.DataFrame({
        "status": "success", "experiment": "main", "task_id": ["a", "b", "c"], "fit_seconds": 30.0,
        "predict_seconds": 0.0, "method": ["HistGradientBoosting", "IterativeImputer", "Median"],
        "target": ["all columns", "all numeric", "x"],
    })
    cost = cost_table(log).set_index("method")["seconds per target and fold"]
    assert cost.to_dict() == {"HistGradientBoosting": 2.0, "IterativeImputer": 30 / 14, "Median": 30.0}


def test_neighbours_fall_back_for_a_row_with_no_shared_cells():
    config = quick_config()
    frame = chained_frame()
    train = frame.iloc[:400]
    empty = frame.iloc[400:402].copy()
    empty.loc[:, :] = np.nan
    model = CHAINED_CLASSES["KNN imputer"](config.parameters["KNN imputer"], 0).fit(train)
    numeric, shares = model.predict(empty)
    assert numeric.notna().all().all() and np.allclose(shares.sum(axis=1), 1)


def test_neighbours_give_the_same_estimates_in_blocks_as_in_one_call():
    config = quick_config()
    frame = chained_frame()
    train, test = frame.iloc[:400], frame.iloc[400:]
    model = CHAINED_CLASSES["KNN imputer"](config.parameters["KNN imputer"], 0).fit(train)
    together, _ = model.predict(test)
    for _, _, column_model in model.sequence:
        (column_model.local if hasattr(column_model, "local") else column_model).block = 7
    blocks, _ = model.predict(test)
    assert np.allclose(together, blocks)


def test_chained_methods_are_one_task_per_fold():
    config = quick_config()
    plan = build_plan(list(config.scenarios), config)
    for method in CHAINED_CLASSES:
        tasks = [t for t in plan if t.experiment == "main" and t.method == method]
        assert len(tasks) == len(config.scenarios) * config.folds
        assert {t.target for t in tasks} == {"all columns"}


def test_chained_estimates_stay_inside_the_observed_range():
    frame = small_frame(400)
    frame = frame.assign(target=frame.complete ** 3)
    frame.loc[::4, "target"] = np.nan
    model = ChainedRidgeImputer(default_parameters()[ITERATIVE], 0).fit(frame)
    filled = model.predict(frame)
    assert filled.target.max() <= frame.target.max() + 1e-6
    assert filled.target.min() >= frame.target.min() - 1e-6


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


def test_only_prediction_code_decides_whether_a_run_can_be_reused():
    assert set(MODEL_SOURCES) == {"columns.py", "data.py", "models.py", "runner.py"}


def test_calibration_uses_the_forest_of_the_raw_result():
    settings = {k: v for k, v in default_parameters()["Random Forest"].items() if k not in CHAIN_SETTINGS}
    assert not set(settings) & set(CHAIN_SETTINGS) and "n_estimators" in settings


def test_total_gas_groups_count_the_visible_gas_readings():
    frame = pd.DataFrame({"measured_gas_a": [1.0, np.nan], "measured_gas_b": [1.0, np.nan], "measured_gas_c": [1.0, 2.0]})
    assert visible_group("total_gas", frame).tolist() == ["3 of 3 gas readings seen", "1 of 3 gas readings seen"]


def test_life_definition_matches_each_model():
    row = {"install_date": 10.0, "measured_humid": 0.1, "measured_salt": 0.2, "measured_overgrowth": 0.3, "total_gas": 30000.0}
    frame = pd.DataFrame([{**row, "model_id": float(m)} for m in range(4)])
    assert np.allclose(rel._life(frame), [100000 - 10 * h - 2500 for h in (0.6, 0.4, 0.5, 0.3)])


def test_leakage_range_contains_the_true_value():
    day, leakage = 500.0, 1.1
    frame = pd.DataFrame([{
        "install_date": day, "model_id": 0.0, "measured_gas_a": 10000 - day * leakage,
        "measured_gas_b": 10000 - 0.0005 * day**2 * leakage,
        "measured_gas_c": 10000 - (day + 0.00025 * day**2) * leakage,
    }])
    assert abs(rel._leakage(frame)[0] - leakage) < 0.4


def relation_inputs():
    measured = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 1.0, np.nan, 1.0], "altitude": [np.nan, np.nan, np.nan, 105.0]})
    for name in TARGETS:
        if name not in measured:
            measured[name] = np.nan
    scored = pd.DataFrame({
        "experiment": "main", "seed": 219, "scenario": "s", "target": "altitude", "method": ["Median"] * 3 + ["A"] * 3,
        "row_id": [0, 1, 2] * 2, "prediction": [150.0, 150.0, 150.0, 102.0, 104.0, 100.0],
        "truth": [102.0, 103.0, 104.0] * 2,
    })
    return scored, measured


def test_relation_table_uses_only_cells_with_every_input_visible():
    scored, measured = relation_inputs()
    config = quick_config(noise_series=("s",))
    table = rel.relation_table(scored, {"s": measured}, config)
    altitude = table[table.relation.eq("altitude from x and y")].set_index("method")
    assert altitude.n.iloc[0] == 2
    assert altitude.MAE["A"] == pytest.approx(0.5)
    assert altitude.MAE[rel.FORMULA] == 0


def test_links_are_sorted_into_direct_two_steps_and_unrelated():
    importance = pd.DataFrame({
        "method": "M", "scenario": "s", "target": "altitude", "predictor": ["x", "install_date", "random_placebo"],
        "importance": [0.5, 0.1, 0.0],
    })
    table = rel.classify_links(importance).set_index("predictor")
    assert table.direct["x"] and table["two steps"]["install_date"] and table.unrelated["random_placebo"]
    row = rel.link_table(importance).iloc[0]
    assert row["direct links found"] == 1 and row["relied on: direct"] == 0.5


def test_perfect_imputation_keeps_every_correlation():
    generator = np.random.default_rng(0)
    truth = pd.DataFrame(generator.normal(size=(200, len(TARGETS))), columns=list(TARGETS))
    measured = truth.mask(generator.random(truth.shape) < 0.2)
    rows = []
    for target in truth:
        hidden = measured.index[measured[target].isna()]
        rows.append(pd.DataFrame({"row_id": hidden, "target": target, "prediction": truth.loc[hidden, target].to_numpy()}))
    scored = pd.concat(rows, ignore_index=True).assign(scenario="s", experiment="main", seed=219, method="A")
    config = quick_config(representative="s")
    table = rel.correlation_table(scored, pd.DataFrame(), measured, truth, config)
    assert table["mean Pearson gap"].iloc[0] == pytest.approx(0, abs=1e-12)
