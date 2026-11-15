"""Checks for the imputation study code, using small frames and no dataset files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PolynomialFeatures

from synthetic_imputation.columns import (
    BY_NAME, COMPANIONS, GENERATOR_LINKS, NAMES, TARGETS, finalise,
)
from synthetic_imputation.data import FILE_PATTERN, relationship_residuals
from synthetic_imputation.evaluation import (
    class_scores, cost_table, numeric_scores,
)
from synthetic_imputation.models import (
    CHAINED_CLASSES, MODEL_CLASSES, BaselineModel, BlockRidge, ChainedRidgeImputer, Features, make_model,
)
from synthetic_imputation import relationships as rel
from synthetic_imputation.study import MODEL_FAMILIES
from synthetic_imputation.runner import (
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


def test_miceforest_runs_as_gradient_boosting():
    config = quick_config()
    model = CHAINED_CLASSES["miceforest direct"](config.parameters["miceforest direct"], 0).fit(chained_frame())
    text = model.kernel.get_model("total", 0).model_to_string()
    assert "[boosting: gbdt]" in text


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


def test_only_prediction_code_decides_whether_a_run_can_be_reused():
    sources = set(MODEL_SOURCES)
    assert {"columns.py", "data.py", "models.py", "runner.py", "models_external.py"} <= sources
    assert not sources & {"evaluation.py", "failure.py", "relationships.py", "figures.py", "study.py"}
    assert {s for s in sources if s.startswith("vendor/")} == {
        f"vendor/{p.parent.name}/{p.name}" for p in (Path(__file__).parents[1] / "src/synthetic_imputation/vendor").glob("*/*.py")
        if p.name != "__init__.py"}


def test_calibration_uses_the_forest_of_the_raw_result():
    settings = {k: v for k, v in default_parameters()["Random Forest"].items() if k not in CHAIN_SETTINGS}
    assert not set(settings) & set(CHAIN_SETTINGS) and "n_estimators" in settings


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


def test_mechanism_files_hide_the_same_share_in_a_different_pattern(tmp_path):
    from synthetic_imputation.mechanisms import write_mechanism_files
    generator = np.random.default_rng(0)
    truth = pd.DataFrame(generator.random((4000, len(NAMES))), columns=list(NAMES))
    truth["install_date"] = np.arange(4000.0)
    truth["model_id"] = generator.integers(0, 4, 4000).astype(float)
    folder = tmp_path / "data/datasets"
    folder.mkdir(parents=True)
    for completion in (0.8, 0.9):
        measured = truth.mask(generator.random(truth.shape) > completion)
        measured.to_csv(folder / f"completion{completion}-noise0.05-measured.csv", index=False)
        truth.to_csv(folder / f"completion{completion}-noise0.05-truth.csv", index=False)
    config = quick_config()
    scenarios = write_mechanism_files(tmp_path, config)
    first = {name: s.measured.read_bytes() for name, s in scenarios.items()}
    assert {name: s.measured.read_bytes() for name, s in write_mechanism_files(tmp_path, config).items()} == first
    base = pd.read_csv(folder / "completion0.9-noise0.05-measured.csv").isna()
    age = pd.read_csv(scenarios["completion0.8-noise0.05-age"].measured).isna()
    value = pd.read_csv(scenarios["completion0.8-noise0.05-value"].measured).isna()
    for hidden in (age, value):
        assert (hidden | ~base).all().all()  # nothing hidden in the base file is revealed
        assert abs(hidden.to_numpy().mean() - 0.2) < 0.01
    assert age.iloc[:1000].to_numpy().mean() > age.iloc[-1000:].to_numpy().mean() + 0.1
    high = truth.x > truth.x.quantile(0.8)
    assert value.x[high].mean() > value.x[~high].mean() + 0.05
    assert scenarios["completion0.8-noise0.05-age"].mechanism == "age"


def test_a_chain_without_the_ridge_base_uses_only_the_local_model():
    from synthetic_imputation.models import ChainedForest, RidgeThenLocal
    frame = chained_frame(200)
    settings = {"passes": 2, "linear_alpha": 30.0, "curve_alpha": 300.0, "n_estimators": 5}
    plain = ChainedForest({**settings, "base": "none"}, 0).fit(frame)
    based = ChainedForest(settings, 0).fit(frame)
    assert not any(isinstance(model, RidgeThenLocal) for _, _, model in plain.sequence)
    assert any(isinstance(model, RidgeThenLocal) for _, _, model in based.sequence)


def test_plan_covers_mechanisms_structure_and_every_importance_fold():
    config = quick_config()
    plan = pd.DataFrame([vars(t) for t in build_plan(["completion0.8-noise0.05"], config, ["completion0.8-noise0.05-age"])])
    assert set(plan.experiment) == {"main", "stability", "budget", "structure", "calibration", "ablation", "importance",
                                    "mechanism", "spatial"}
    assert plan[plan.experiment.eq("importance")].fold.nunique() == config.folds
    assert set(plan[plan.experiment.eq("mechanism")].scenario) == {"completion0.8-noise0.05-age"}
    assert not plan.method.isin(["miceforest PMM", "Joint miceforest PMM"]).any()


def failure_inputs():
    """Two methods and the baseline on four hidden altitude cells, with x and y partly visible."""
    from synthetic_imputation import failure as fl
    measured = pd.DataFrame(np.nan, index=range(4), columns=list(NAMES))
    measured["x"] = [1.0, 2.0, np.nan, np.nan]
    measured["y"] = [1.0, np.nan, 3.0, np.nan]
    truth = pd.DataFrame(1.0, index=range(4), columns=list(NAMES))
    truth["altitude"] = [102.0, 104.0, 106.0, 108.0]
    rows = []
    for method, predictions in (("Median", [105.0] * 4), ("A", [102.0, 104.0, 107.0, 105.0])):
        for row, value in enumerate(predictions):
            rows.append({"scenario": "s", "experiment": "main", "variant": "all predictors", "seed": 219, "fold": 0,
                         "method": method, "target": "altitude", "row_id": row, "raw": value, "prediction": value,
                         "truth": truth.altitude[row]})
    return fl, fl.cell_table(pd.DataFrame(rows), {"s": measured}, {"s": truth})


def test_cells_carry_the_baseline_error_and_the_visible_links():
    fl, cells = failure_inputs()
    model = cells[cells.method.eq("A")].set_index("row_id")
    assert model.base_error.tolist() == [3.0, 1.0, 1.0, 3.0]
    assert model.seen.tolist() == [2, 1, 1, 0] and model.links.eq(3).all()  # x, y and measured_humid
    assert model.error.tolist() == [0.0, 0.0, 1.0, 3.0]


def test_skill_is_measured_against_the_baseline_on_the_same_cells():
    fl, cells = failure_inputs()
    table = fl.skill_by(cells, ["method"], minimum=1).set_index("method")
    assert table.skill["Median"] == 0
    assert table.skill["A"] == pytest.approx(1 - 4 / 8)
    by_seen = fl.skill_by(cells[cells.method.eq("A")], ["seen"], minimum=1).set_index("seen").skill
    assert by_seen[2] == 1 and by_seen[0] == 0


def test_failure_standard_has_three_states():
    from synthetic_imputation import failure as fl
    assert [fl.state(v) for v in (0.9, 0.5, 0.3, 0.05, -0.2, np.nan)] == ["works", "works", "weak", "fails", "fails", "not measured"]


def test_links_follow_the_generator_definition():
    from synthetic_imputation import failure as fl
    assert fl.link_class("altitude", "x") == fl.DIRECT
    assert fl.link_class("x", "y") == fl.TOGETHER  # they define altitude together
    assert fl.link_class("altitude", "install_date") == fl.THROUGH  # both are computed from x and y
    assert fl.link_class("leakage", "x") == fl.STAND_IN  # x is a near copy of install_date
    assert fl.link_class("altitude", "random_placebo") == fl.NO_LINK
    assert fl.link_class("leakage", "measured_humid") == fl.NO_LINK


def test_downstream_values_of_the_truth_have_no_error():
    from synthetic_imputation import failure as fl
    generator = np.random.default_rng(0)
    truth = pd.DataFrame(generator.random((300, len(NAMES))), columns=list(NAMES))
    truth["model_id"] = generator.integers(0, 4, 300).astype(float)
    measured = truth.mask(generator.random(truth.shape) < 0.2)
    rows = []
    for target in truth:
        hidden = measured.index[measured[target].isna()]
        method = "Mode" if target == "model_id" else "Median"
        for name in (method, "A"):
            rows.append(pd.DataFrame({"row_id": hidden, "target": target, "method": name,
                                      "prediction": truth.loc[hidden, target].to_numpy()}))
    scored = pd.concat(rows, ignore_index=True).assign(scenario="s", experiment="main", variant="all predictors", seed=219,
                                                         fold=0, raw=np.nan)
    scored["truth"] = scored.prediction
    config = quick_config(representative="s")
    cells = fl.cell_table(scored, {"s": measured}, {"s": truth})
    table = fl.downstream_table(cells, {"s": measured}, {"s": truth}, config)
    perfect = table[table.source.eq("A")]
    assert perfect["relative error"].abs().max() < 1e-12
    assert table[table.source.eq("rows with no missing cell")]["rows used"].iloc[0] < 300


def test_scenario_effects_do_not_depend_on_the_order_of_rows():
    from synthetic_imputation.data import Scenario
    from synthetic_imputation.evaluation import scenario_effects
    generator = np.random.default_rng(0)
    scenarios = {name: Scenario(name, 0.8, noise, None, None) for name, noise in (("completion0.8-noise0.0", 0.0), ("completion0.8-noise0.1", 0.1))}
    rows = []
    for name in scenarios:
        truth = generator.normal(size=200)
        rows.append(pd.DataFrame({"scenario": name, "experiment": "main", "method": "A", "target": "altitude",
                                  "row_id": np.arange(200), "truth": truth, "prediction": truth + generator.normal(size=200)}))
    scored = pd.concat(rows, ignore_index=True)
    config = quick_config()
    first = scenario_effects(scored, scenarios, config)
    second = scenario_effects(scored.sample(frac=1, random_state=1), scenarios, config)
    assert first[["low", "high"]].to_numpy() == pytest.approx(second[["low", "high"]].to_numpy(), abs=1e-12)



@pytest.mark.parametrize("method", ["Gaussian EM", "Masked MLP", "LightGBM chained", "Predictive MissForest", "MICE-PMM"])
def test_external_models_fill_every_numeric_column_and_give_class_probabilities(method):
    from synthetic_imputation.models_external import EXTERNAL_CLASSES
    from synthetic_imputation.runner import quick_config
    generator = np.random.default_rng(4)
    frame = pd.DataFrame(generator.random((300, len(NAMES))), columns=list(NAMES))
    frame["model_id"] = generator.integers(0, 4, 300).astype(float)
    frame = frame.mask(generator.random(frame.shape) < 0.2)
    train, test = frame.iloc[:200], frame.iloc[200:]
    model = EXTERNAL_CLASSES[method](quick_config().parameters[method], 0).fit(train)
    numeric, shares = model.predict(test)
    assert list(numeric.columns) == [c for c in NAMES if c != "model_id"]
    assert numeric.notna().all().all() and numeric.index.equals(test.index)
    assert shares.shape == (100, 4) and np.allclose(shares.sum(axis=1), 1)
    observed = test.notna()
    assert np.allclose(numeric[observed.drop(columns="model_id")].stack(), test.drop(columns="model_id")[observed.drop(columns="model_id")].stack())


def test_external_models_accept_a_removed_predictor_block():
    from synthetic_imputation.models_external import GaussianEM
    generator = np.random.default_rng(5)
    frame = pd.DataFrame(generator.random((200, len(NAMES))), columns=list(NAMES))
    frame["model_id"] = generator.integers(0, 4, 200).astype(float)
    frame = frame.mask(generator.random(frame.shape) < 0.2)
    model = GaussianEM({"clusters": 2, "steps": 3, "reg": 0.01, "class_weight": 1.8}, 0, drop=("x", "y")).fit(frame.iloc[:150])
    numeric, _ = model.predict(frame.iloc[150:])
    assert "x" not in numeric and "y" not in numeric


def test_tile_folds_keep_whole_blocks_together():
    from synthetic_imputation.runner import tile_fold_ids
    folds = tile_fold_ids(np.arange(10000), 5, 219)
    blocks = (np.arange(10000) // 100 // 10) * 10 + (np.arange(10000) % 100) // 10
    assert all(len(set(folds[blocks == b])) == 1 for b in range(100))
    assert np.bincount(folds).tolist() == [2000] * 5


def test_errors_scenario_changes_only_visible_dates_and_models():
    from synthetic_imputation.mechanisms import _with_errors
    generator = np.random.default_rng(0)
    frame = pd.DataFrame(generator.random((2000, len(NAMES))), columns=list(NAMES))
    frame["install_date"] = generator.integers(0, 2000, 2000).astype(float)
    frame["model_id"] = generator.integers(0, 4, 2000).astype(float)
    frame = frame.mask(generator.random(frame.shape) < 0.2)
    wrong = _with_errors(frame, np.random.default_rng(1))
    assert wrong.isna().equals(frame.isna())
    assert wrong.drop(columns=["install_date", "model_id"]).equals(frame.drop(columns=["install_date", "model_id"]))
    moved = (wrong.install_date - frame.install_date).dropna()
    assert set(moved.abs().unique()) <= {0, 60} and (wrong.install_date.dropna() >= 0).all()
    swapped = (wrong.model_id != frame.model_id) & frame.model_id.notna()
    assert 0.005 < swapped.mean() < 0.04


def test_mice_pmm_draws_test_gaps_only_from_training_rows():
    from synthetic_imputation.models_external import MICEPMM
    generator = np.random.default_rng(6)
    frame = pd.DataFrame(generator.random((240, len(NAMES))), columns=list(NAMES))
    frame["model_id"] = generator.integers(0, 4, 240).astype(float)
    frame["life"] = np.round(frame["life"], 2) + 1000  # training values
    frame = frame.mask(generator.random(frame.shape) < 0.2)
    train, test = frame.iloc[:160].copy(), frame.iloc[160:].copy()
    test["life"] = test["life"] + 5000  # visible test values that no donor may supply
    settings = {"chains": 1, "sweeps": 2, "donors": 5, "ridge": 1e-4, "logistic_C": 1.0, "logistic_max_iter": 200}
    numeric, _ = MICEPMM(settings, 0).fit(train).predict(test)
    filled = numeric.loc[test["life"].isna(), "life"]
    assert len(filled) and filled.isin(train["life"].dropna().to_numpy()).all()



def test_chain_estimates_stay_inside_the_training_range():
    from synthetic_imputation.models import ChainedForest
    frame = chained_frame(300)
    model = ChainedForest({"passes": 3, "linear_alpha": 30.0, "curve_alpha": 300.0, "n_estimators": 5}, 0).fit(frame)
    shifted = frame.copy()
    shifted[["a", "b"]] = shifted[["a", "b"]] * 50  # inputs far outside the training range
    numeric, _ = model.predict(shifted)
    for column in numeric:
        assert numeric[column].between(frame[column].min(), frame[column].max()).all()
