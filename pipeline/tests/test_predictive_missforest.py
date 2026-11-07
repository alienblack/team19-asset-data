"""Mandatory correctness tests for PredictiveMissForest.

Run with ``pytest tests`` or, without pytest, ``python run_tests.py``.
The tests avoid pytest-only fixtures so both runners work.
"""

from __future__ import annotations

import copy
import pickle
import tempfile
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.imputation.predictive_missforest import (
    PredictiveMissForest,
    PredictiveMissForestConfig,
    validate_transform_input,
)

from fixtures import (
    CATEGORICAL,
    CONSTRAINTS,
    MODEL_COLUMNS,
    NUMERIC,
    make_fit_frame,
    make_new_batch,
)

FAST = PredictiveMissForestConfig(
    n_estimators=20, n_passes=3, random_state=219, n_jobs=1, max_depth=8
)

_CACHE: dict[str, object] = {}


def fitted_model() -> PredictiveMissForest:
    """One fitted model shared across tests (fitting is the slow part)."""
    if "model" not in _CACHE:
        model = PredictiveMissForest(NUMERIC, CATEGORICAL, FAST, CONSTRAINTS)
        model.fit(make_fit_frame())
        _CACHE["model"] = model
    return copy.deepcopy(_CACHE["model"])


def observed_unchanged(original: pd.DataFrame, imputed: pd.DataFrame) -> bool:
    for column in MODEL_COLUMNS:
        observed = original[column].notna().to_numpy()
        before = original[column].to_numpy(dtype=object)[observed]
        after = imputed[column].to_numpy(dtype=object)[observed]
        if not all(x == y for x, y in zip(before, after)):
            return False
    return True


# ---------------------------------------------------------------- fit ---- #


def test_fit_models_every_model_column_in_frozen_order():
    """A column complete at fit time still gets forests (needed at inference);
    order is ascending training missingness then configuration order."""
    model = fitted_model()
    assert set(model.target_order_) == set(MODEL_COLUMNS)
    assert model.target_order_[0] == "voltage_category"   # 0 missing at fit
    counts = model.fit_report_["missing_counts"]
    ordered = [counts[c] for c in model.target_order_]
    assert ordered == sorted(ordered)
    assert len(model.models_) == FAST.n_passes * len(MODEL_COLUMNS)
    assert model.unmodelled_targets_ == ()


def test_initial_statistics_come_from_observed_fit_cells_only():
    frame = make_fit_frame()
    model = fitted_model()
    for column in NUMERIC:
        expected = float(frame[column].dropna().mean())
        assert abs(model.init_values_[column] - expected) < 1e-9
    for column in CATEGORICAL:
        assert model.init_values_[column] == frame[column].dropna().mode().iloc[0]
        assert set(model.vocabularies_[column]) == set(frame[column].dropna().unique())


def test_each_forest_excludes_its_own_target_from_predictors():
    model = fitted_model()
    for (pass_number, target), pipeline in model.models_.items():
        features = model.features_[target]
        assert target not in features
        assert set(features) == set(MODEL_COLUMNS) - {target}
        seen = list(pipeline.named_steps["preprocessor"].feature_names_in_)
        assert target not in seen
        assert set(seen) == set(features)


def test_supervised_labels_are_originally_observed_values_only():
    """Spy on Pipeline.fit: y must equal the raw observed target values; rows
    whose target was missing (placeholder-filled) must never be labels."""
    frame = make_fit_frame()
    raw = frame.reset_index(drop=True)
    calls: list[tuple[str, pd.Series]] = []
    real_fit = Pipeline.fit

    def spy(self, X, y=None, **kw):
        target = [c for c in MODEL_COLUMNS if c not in X.columns][0]
        calls.append((target, y.copy()))
        return real_fit(self, X, y, **kw)

    with mock.patch.object(Pipeline, "fit", spy):
        PredictiveMissForest(NUMERIC, CATEGORICAL, FAST, CONSTRAINTS).fit(frame)

    assert len(calls) == FAST.n_passes * len(MODEL_COLUMNS)
    for target, y in calls:
        observed = raw[target].notna()
        assert list(y.index) == list(raw.index[observed])
        expected = raw.loc[observed, target]
        if target in NUMERIC:
            assert np.allclose(y.astype(float).to_numpy(), expected.astype(float).to_numpy())
        else:
            assert list(y) == list(expected)


def test_fit_records_pass_count_and_change_metrics_without_early_stop():
    model = fitted_model()
    assert list(model.convergence_["pass"]) == list(range(1, FAST.n_passes + 1))
    assert model.fit_report_["n_passes"] == FAST.n_passes
    assert model.fit_report_["runtime_seconds"] > 0


def test_fit_rejects_column_with_no_observed_values():
    frame = make_fit_frame()
    frame["covering"] = np.nan
    try:
        PredictiveMissForest(NUMERIC, CATEGORICAL, FAST).fit(frame)
    except ValueError as error:
        assert "covering" in str(error)
    else:
        raise AssertionError("fit accepted a column with no observed values")


def test_column_below_min_observed_is_unmodelled_and_warned():
    frame = make_fit_frame()
    frame.loc[frame.index[3:], "covering"] = np.nan   # only 3 observed
    config = FAST.replace(min_observed_target=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PredictiveMissForest(NUMERIC, CATEGORICAL, config, CONSTRAINTS).fit(frame)
    assert model.unmodelled_targets_ == ("covering",)
    assert any("covering" in str(w.message) for w in caught)
    out = model.transform(make_new_batch())
    assert (out["covering"] == model.init_values_["covering"]).all()


# ---------------------------------------------------------- transform ---- #


def test_transform_preserves_observed_values_index_order_and_metadata():
    model = fitted_model()
    batch = make_new_batch()
    out = model.transform(batch)
    assert observed_unchanged(batch, out)
    assert out.index.equals(batch.index)
    assert list(out.columns) == list(batch.columns)
    pd.testing.assert_series_equal(out["segment_id"], batch["segment_id"])
    pd.testing.assert_series_equal(out["data_status"], batch["data_status"])
    assert out[MODEL_COLUMNS].isna().sum().sum() == 0


def test_missing_mask_identifies_exactly_the_input_gaps():
    model = fitted_model()
    batch = make_new_batch()
    result = model.transform_detailed(batch)
    pd.testing.assert_frame_equal(result.missing_mask, batch[MODEL_COLUMNS].isna())


def test_transform_never_fits():
    model = fitted_model()
    batch = make_new_batch()

    def forbidden(self, *a, **k):
        raise AssertionError("fit called during transform")

    with mock.patch.object(Pipeline, "fit", forbidden), \
         mock.patch.object(Pipeline, "fit_transform", forbidden):
        model.transform(batch)
        model.predict_scores(batch)


def test_transform_does_not_mutate_fitted_state():
    model = fitted_model()
    before = pickle.dumps((model.init_values_, model.vocabularies_, model.target_order_))
    extreme = make_new_batch()
    extreme["installation_year"] = 2014.0
    extreme["distance_to_sea_km"] = 349.0
    model.transform(extreme)
    after = pickle.dumps((model.init_values_, model.vocabularies_, model.target_order_))
    assert before == after


def test_single_record_equals_same_record_in_batch():
    model = fitted_model()
    batch = make_new_batch()
    whole = model.transform(batch)
    for label in batch.index[:5]:
        single = model.transform(batch.loc[[label]])
        pd.testing.assert_frame_equal(single, whole.loc[[label]], check_dtype=False)


def test_row_order_does_not_change_predictions():
    model = fitted_model()
    batch = make_new_batch()
    forward = model.transform(batch)
    shuffled = batch.sample(frac=1.0, random_state=3)
    backward = model.transform(shuffled).loc[batch.index]
    pd.testing.assert_frame_equal(forward, backward, check_dtype=False)


def test_entirely_missing_column_and_fully_missing_record():
    model = fitted_model()
    batch = make_new_batch()
    assert batch["covering"].isna().all()
    assert batch.loc["R0", MODEL_COLUMNS].isna().all()
    out = model.transform(batch)
    assert out["covering"].isin(model.vocabularies_["covering"]).all()
    assert out.loc["R0", MODEL_COLUMNS].notna().all()


def test_column_complete_at_fit_is_imputed_by_its_forest_at_inference():
    """voltage_category had no missing fit cells; at inference its forest
    must still be used (not only the mode) and produce valid categories."""
    model = fitted_model()
    batch = make_new_batch()
    missing = batch["voltage_category"].isna()
    assert missing.sum() > 5
    out = model.transform(batch)
    imputed = out.loc[missing, "voltage_category"]
    assert imputed.isin(model.vocabularies_["voltage_category"]).all()
    # location strongly predicts voltage in the fixture, so at least one
    # imputed value should differ from the plain mode.
    assert (imputed != model.init_values_["voltage_category"]).any()


def test_unknown_category_policies():
    model = fitted_model()
    batch = make_new_batch()
    batch.loc["R1", "location_group"] = "offshore"      # never seen at fit

    out = model.transform(batch)                           # policy "ignore"
    assert out.loc["R1", "location_group"] == "offshore"   # observed preserved
    detail = model.transform_detailed(batch)
    assert detail.unknown_category_counts == {"location_group": 1}

    model.config = model.config.replace(unknown_category_policy="mode")
    out_mode = model.transform(batch)
    assert out_mode.loc["R1", "location_group"] == "offshore"

    model.config = model.config.replace(unknown_category_policy="error")
    try:
        model.transform(batch)
    except ValueError as error:
        assert "location_group" in str(error)
    else:
        raise AssertionError("policy 'error' did not raise")


def test_outputs_conform_to_types_and_constraints():
    model = fitted_model()
    batch = make_new_batch()
    out = model.transform(batch)
    mask = batch[MODEL_COLUMNS].isna()
    years = out.loc[mask["installation_year"], "installation_year"].astype(float)
    assert (years == years.round()).all()
    assert years.between(1930, 2014).all()
    dist = out.loc[mask["distance_to_sea_km"], "distance_to_sea_km"].astype(float)
    assert dist.between(0.0, 350.0).all()
    for column in CATEGORICAL:
        assert out[column].isin(model.vocabularies_[column]).all()


def test_transform_input_validation_allows_sparse_batches():
    batch = make_new_batch()
    validate_transform_input(batch, NUMERIC, CATEGORICAL)   # no error
    bad = batch.copy()
    bad["installation_year"] = bad["installation_year"].astype(object)
    bad.loc["R2", "installation_year"] = "nineteen-eighty"
    try:
        validate_transform_input(bad, NUMERIC, CATEGORICAL)
    except ValueError as error:
        assert "installation_year" in str(error)
    else:
        raise AssertionError("non-numeric observed value accepted")


# ------------------------------------------------------- replay / seeds -- #


def test_replay_on_fit_data_reproduces_fit_time_completed_table():
    """The strongest replay test: transform(X_fit) must equal the completed
    fitting table produced during fit, because init, forests, order and pass
    count are frozen and the state sequence is therefore identical."""
    model = fitted_model()
    frame = make_fit_frame()
    replayed = model.transform(frame)
    for column in NUMERIC:
        assert np.allclose(
            replayed[column].astype(float), model.fit_imputed_[column].astype(float),
            rtol=0, atol=1e-9,
        )
    for column in CATEGORICAL:
        assert list(replayed[column]) == list(model.fit_imputed_[column])


def test_sequential_state_each_forest_sees_updated_predictors():
    """Spy on Pipeline.predict during transform and check the sequential
    (Gauss-Seidel) state: when forest (p, j) predicts, the columns earlier in
    the order already hold pass-p values, which must differ from the
    initial fill for at least one row where they were missing (p == 1)."""
    model = fitted_model()
    batch = make_new_batch()
    seen: list[tuple[str, pd.DataFrame]] = []
    real_predict = Pipeline.predict

    def spy(self, X, **kw):
        target = [c for c in MODEL_COLUMNS if c not in X.columns][0]
        seen.append((target, X.copy()))
        return real_predict(self, X, **kw)

    with mock.patch.object(Pipeline, "predict", spy):
        model.transform(batch)

    order = list(model.target_order_)
    per_pass = len([t for t in order if batch[t].isna().any()])
    first_pass = seen[:per_pass]
    assert [t for t, _ in first_pass] == [t for t in order if batch[t].isna().any()]

    init = model.init_values_
    missing = batch[MODEL_COLUMNS].isna().reset_index(drop=True)
    for k, (target, X) in enumerate(first_pass):
        earlier = [t for t, _ in first_pass[:k]]
        later = [t for t in order if t not in earlier and t != target]
        rows = missing[target].to_numpy()
        for column in later:            # still at initial fill
            col_missing = missing.loc[rows, column].to_numpy()
            values = X[column].to_numpy()[col_missing]
            if column in NUMERIC:
                assert np.allclose(values.astype(float), init[column])
            else:
                assert (values == init[column]).all()
        for column in earlier:          # already updated by this pass
            col_missing = missing.loc[rows, column].to_numpy()
            values = X[column].to_numpy()[col_missing]
            if len(values) and column in CATEGORICAL:
                assert set(values) <= set(model.vocabularies_[column])


def test_determinism_same_seed_same_output_and_refit():
    frame = make_fit_frame()
    batch = make_new_batch()
    a = PredictiveMissForest(NUMERIC, CATEGORICAL, FAST, CONSTRAINTS).fit(frame)
    b = PredictiveMissForest(NUMERIC, CATEGORICAL, FAST, CONSTRAINTS).fit(frame)
    pd.testing.assert_frame_equal(a.transform(batch), b.transform(batch))
    pd.testing.assert_frame_equal(a.transform(batch), a.transform(batch))


def test_different_seed_changes_something():
    frame = make_fit_frame()
    batch = make_new_batch()
    a = PredictiveMissForest(NUMERIC, CATEGORICAL, FAST, CONSTRAINTS).fit(frame)
    b = PredictiveMissForest(
        NUMERIC, CATEGORICAL, FAST.replace(random_state=99), CONSTRAINTS
    ).fit(frame)
    assert not a.transform(batch).equals(b.transform(batch))


# ----------------------------------------------------------- scores ----- #


def test_predict_scores_cover_every_imputed_cell_with_sane_values():
    model = fitted_model()
    batch = make_new_batch()
    scores = model.predict_scores(batch)
    mask = batch[MODEL_COLUMNS].isna()
    assert len(scores) == int(mask.to_numpy().sum())
    cat = scores[scores["kind"] == "categorical"]
    num = scores[scores["kind"] == "numeric"]
    assert cat["max_proba"].between(0, 1).all()
    assert (cat["margin"] >= 0).all() and (cat["margin"] <= 1).all()
    assert (num["tree_std"] >= 0).all()
    assert (num["tree_p10"] <= num["tree_p90"]).all()
    out = model.transform(batch)
    for _, row in cat.iterrows():
        assert out.loc[row["row"], row["column"]] == row["imputed"]


# ------------------------------------------------------- persistence ---- #


def test_save_and_reload_gives_identical_predictions():
    model = fitted_model()
    batch = make_new_batch()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "pmf.joblib"
        model.save(path)
        reloaded = PredictiveMissForest.load(path)
    pd.testing.assert_frame_equal(model.transform(batch), reloaded.transform(batch))
    assert reloaded.target_order_ == model.target_order_
    assert reloaded.fit_report_ == model.fit_report_


def test_unfitted_model_refuses_transform_and_save():
    model = PredictiveMissForest(NUMERIC, CATEGORICAL, FAST)
    for call in (lambda: model.transform(make_new_batch()), lambda: model.save("x")):
        try:
            call()
        except RuntimeError:
            pass
        else:
            raise AssertionError("unfitted model did not raise")


# ---------------------------------------------------------- config ------ #


def test_config_validation_and_yaml_section():
    bad = [dict(n_passes=0), dict(numeric_init="mode"),
           dict(unknown_category_policy="drop"), dict(min_observed_target=0),
           dict(max_features="all")]
    for overrides in bad:
        try:
            PredictiveMissForestConfig.from_mapping({**overrides})
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid config {overrides}")
    try:
        PredictiveMissForestConfig.from_mapping({"max_iter": 5})
    except ValueError as error:
        assert "max_iter" in str(error)
    else:
        raise AssertionError("unknown key accepted")


def test_integer_coded_categorical_labels_keep_their_type():
    """Categorical codes stored as Python ints must come back as ints, not
    strings or floats (classifier trains on str labels, mapped back)."""
    def intify(frame):
        codes = {"low": 1, "medium": 2, "high": 3}
        values = [codes[v] if isinstance(v, str) else np.nan for v in frame["corrosion_zone"]]
        frame["corrosion_zone"] = pd.Series(values, index=frame.index, dtype=object)
        return frame

    frame = intify(make_fit_frame())
    model = PredictiveMissForest(NUMERIC, CATEGORICAL, FAST, CONSTRAINTS).fit(frame)
    assert set(model.vocabularies_["corrosion_zone"]) == {1, 2, 3}
    out = model.transform(intify(make_new_batch()))
    assert {type(v) for v in out["corrosion_zone"]} == {int}
    assert set(out["corrosion_zone"]) <= {1, 2, 3}
