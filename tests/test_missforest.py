import numpy as np
import pandas as pd
import pytest

import team19_asset_data.imputation.missforest as missforest_module

from team19_asset_data.imputation.missforest import (
    MissForestConfig,
    _build_model,
    _initialise,
    missforest_impute,
)


TEST_CONFIG = MissForestConfig(
    n_estimators=15,
    max_iter=3,
    random_state=219,
    n_jobs=1,
    max_features="sqrt",
)


def mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [
                10.0,
                20.0,
                np.nan,
                40.0,
                50.0,
                60.0,
            ],
            "distance": [
                5.0,
                np.nan,
                15.0,
                20.0,
                25.0,
                30.0,
            ],
            "material": [
                "Steel",
                "Steel",
                "Copper",
                np.nan,
                "Copper",
                "Steel",
            ],
            "location": [
                "Rural",
                "Urban",
                "Rural",
                "Urban",
                np.nan,
                "Rural",
            ],
            "asset_id": [
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
                "A6",
            ],
        }
    )


def test_missforest_imputes_mixed_data():
    frame = mixed_frame()

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    model_columns = [
        "age",
        "distance",
        "material",
        "location",
    ]

    assert not result.imputed[
        model_columns
    ].isna().any().any()


def test_missforest_preserves_shape_and_columns():
    frame = mixed_frame()

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    assert result.imputed.shape == frame.shape
    assert list(result.imputed.columns) == list(frame.columns)
    assert result.imputed.index.equals(frame.index)


def test_missforest_preserves_observed_values():
    frame = mixed_frame()

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    for column in [
        "age",
        "distance",
        "material",
        "location",
    ]:
        observed = frame[column].notna()

        pd.testing.assert_series_equal(
            result.imputed.loc[
                observed,
                column,
            ],
            frame.loc[
                observed,
                column,
            ],
            check_names=False,
            check_dtype=False,
        )


def test_missforest_preserves_non_model_columns():
    frame = mixed_frame()

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    pd.testing.assert_series_equal(
        result.imputed["asset_id"],
        frame["asset_id"],
    )


def test_missforest_does_not_modify_input_dataframe():
    frame = mixed_frame()

    original = frame.copy(
        deep=True
    )

    missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    pd.testing.assert_frame_equal(
        frame,
        original,
    )


def test_missforest_is_deterministic_with_same_seed():
    frame = mixed_frame()

    result_one = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    result_two = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    pd.testing.assert_frame_equal(
        result_one.imputed,
        result_two.imputed,
    )

    pd.testing.assert_frame_equal(
        result_one.convergence.drop(
            columns=["runtime_seconds"]
        ),
        result_two.convergence.drop(
            columns=["runtime_seconds"]
        ),
    )


def test_initialise_uses_numeric_mean():
    frame = pd.DataFrame(
        {
            "age": [
                10.0,
                20.0,
                np.nan,
            ],
            "material": [
                "Steel",
                "Copper",
                "Steel",
            ],
        }
    )

    result = _initialise(
        frame=frame,
        numeric_columns=["age"],
        categorical_columns=["material"],
    )

    assert result.loc[2, "age"] == 15.0


def test_initialise_uses_deterministic_categorical_mode():
    frame = pd.DataFrame(
        {
            "age": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ],
            "material": [
                "Steel",
                "Copper",
                "Steel",
                "Copper",
                np.nan,
            ],
        }
    )

    result = _initialise(
        frame=frame,
        numeric_columns=["age"],
        categorical_columns=["material"],
    )

    # Copper and Steel have equal frequency.
    # Deterministic string-order tie break chooses Copper.
    assert result.loc[4, "material"] == "Copper"


def test_numeric_only_imputation():
    frame = pd.DataFrame(
        {
            "age": [
                10.0,
                20.0,
                np.nan,
                40.0,
                50.0,
            ],
            "distance": [
                5.0,
                np.nan,
                15.0,
                20.0,
                25.0,
            ],
        }
    )

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[],
        config=TEST_CONFIG,
    )

    assert not result.imputed.isna().any().any()


def test_categorical_only_imputation():
    frame = pd.DataFrame(
        {
            "material": [
                "Steel",
                "Steel",
                "Copper",
                np.nan,
                "Copper",
                "Steel",
            ],
            "location": [
                "Rural",
                "Urban",
                "Rural",
                "Urban",
                np.nan,
                "Rural",
            ],
        }
    )

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    assert not result.imputed.isna().any().any()


def test_no_missing_values_returns_immediately():
    frame = pd.DataFrame(
        {
            "age": [
                10.0,
                20.0,
                30.0,
            ],
            "material": [
                "Steel",
                "Copper",
                "Steel",
            ],
        }
    )

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=["age"],
        categorical_columns=["material"],
        config=TEST_CONFIG,
    )

    pd.testing.assert_frame_equal(
        result.imputed,
        frame,
    )

    assert result.selected_iteration == 0
    assert result.converged is True
    assert result.convergence.empty


def test_convergence_dataframe_has_expected_columns():
    frame = mixed_frame()

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    assert list(
        result.convergence.columns
    ) == [
        "iteration",
        "numeric_change",
        "categorical_change",
        "runtime_seconds",
        "accepted",
    ]


def test_iteration_limit_is_respected():
    frame = mixed_frame()

    config = MissForestConfig(
        n_estimators=10,
        max_iter=2,
        random_state=219,
        n_jobs=1,
    )

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=config,
    )

    assert len(
        result.convergence
    ) <= 2

    assert (
        result.convergence[
            "iteration"
        ].max()
        <= 2
    )


def test_selected_iteration_is_valid():
    frame = mixed_frame()

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=TEST_CONFIG,
    )

    assert (
        0
        <= result.selected_iteration
        <= TEST_CONFIG.max_iter
    )

    if result.selected_iteration > 0:
        selected_row = result.convergence.loc[
            result.convergence["iteration"]
            == result.selected_iteration
        ]

        assert not selected_row.empty
        assert bool(
            selected_row.iloc[0][
                "accepted"
            ]
        )


def test_build_model_passes_regressor_configuration():
    config = MissForestConfig(
        n_estimators=25,
        max_iter=5,
        random_state=219,
        n_jobs=1,
        max_features="sqrt",
        max_depth=7,
        min_samples_leaf=2,
    )

    model, features = _build_model(
        target="age",
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[
            "material",
        ],
        config=config,
        seed=500,
    )

    forest = model.named_steps[
        "forest"
    ]

    assert features == [
        "distance",
        "material",
    ]

    assert forest.n_estimators == 25
    assert forest.random_state == 500
    assert forest.n_jobs == 1
    assert forest.max_features == "sqrt"
    assert forest.max_depth == 7
    assert forest.min_samples_leaf == 2


def test_build_model_passes_classifier_configuration():
    config = MissForestConfig(
        n_estimators=30,
        random_state=219,
        n_jobs=1,
        max_features="log2",
        class_weight="balanced",
    )

    model, _ = _build_model(
        target="material",
        numeric_columns=[
            "age",
        ],
        categorical_columns=[
            "material",
            "location",
        ],
        config=config,
        seed=600,
    )

    forest = model.named_steps[
        "forest"
    ]

    assert forest.n_estimators == 30
    assert forest.random_state == 600
    assert forest.max_features == "log2"
    assert forest.class_weight == "balanced"


@pytest.mark.parametrize(
    "config,error_message",
    [
        (
            MissForestConfig(
                n_estimators=0
            ),
            "n_estimators",
        ),
        (
            MissForestConfig(
                max_iter=0
            ),
            "max_iter",
        ),
        (
            MissForestConfig(
                n_jobs=0
            ),
            "n_jobs",
        ),
        (
            MissForestConfig(
                max_depth=0
            ),
            "max_depth",
        ),
        (
            MissForestConfig(
                min_samples_leaf=0
            ),
            "min_samples_leaf",
        ),
        (
            MissForestConfig(
                max_features="invalid"
            ),
            "max_features",
        ),
        (
            MissForestConfig(
                max_features=0
            ),
            "max_features",
        ),
        (
            MissForestConfig(
                max_features=1.5
            ),
            "max_features",
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    config,
    error_message,
):
    frame = mixed_frame()

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        missforest_impute(
            incomplete=frame,
            numeric_columns=[
                "age",
                "distance",
            ],
            categorical_columns=[
                "material",
                "location",
            ],
            config=config,
        )


def test_missing_requested_column_is_rejected():
    frame = pd.DataFrame(
        {
            "age": [
                10.0,
                np.nan,
                30.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing requested model columns",
    ):
        missforest_impute(
            incomplete=frame,
            numeric_columns=[
                "age",
                "distance",
            ],
            categorical_columns=[],
            config=TEST_CONFIG,
        )


def test_numeric_categorical_overlap_is_rejected():
    frame = pd.DataFrame(
        {
            "age": [
                10.0,
                np.nan,
                30.0,
            ],
            "distance": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="both numeric and categorical",
    ):
        missforest_impute(
            incomplete=frame,
            numeric_columns=[
                "age",
                "distance",
            ],
            categorical_columns=[
                "age",
            ],
            config=TEST_CONFIG,
        )


def test_all_missing_target_is_rejected():
    frame = pd.DataFrame(
        {
            "age": [
                np.nan,
                np.nan,
                np.nan,
            ],
            "distance": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Numeric column 'age' has no observed values",
    ):
        missforest_impute(
            incomplete=frame,
            numeric_columns=[
                "age",
                "distance",
            ],
            categorical_columns=[],
            config=TEST_CONFIG,
        )


def test_single_incomplete_model_column_is_rejected():
    frame = pd.DataFrame(
        {
            "age": [
                10.0,
                np.nan,
                30.0,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="requires at least two model columns",
    ):
        missforest_impute(
            incomplete=frame,
            numeric_columns=["age"],
            categorical_columns=[],
            config=TEST_CONFIG,
        )


def test_previous_accepted_iteration_is_kept_when_change_worsens(
    monkeypatch,
):
    frame = pd.DataFrame(
        {
            "age": [
                10.0,
                20.0,
                np.nan,
                40.0,
                50.0,
            ],
            "distance": [
                5.0,
                10.0,
                15.0,
                20.0,
                25.0,
            ],
        }
    )

    changes = iter(
        [
            0.20,
            0.30,
        ]
    )

    def fake_numeric_change(
        previous,
        current,
        missing_mask,
        numeric_columns,
    ):
        return next(changes)

    monkeypatch.setattr(
        missforest_module,
        "_numeric_change",
        fake_numeric_change,
    )

    config = MissForestConfig(
        n_estimators=10,
        max_iter=3,
        random_state=219,
        n_jobs=1,
    )

    result = missforest_impute(
        incomplete=frame,
        numeric_columns=[
            "age",
            "distance",
        ],
        categorical_columns=[],
        config=config,
    )

    assert result.converged is True

    # Iteration 1 is accepted.
    # Iteration 2 has a worse numeric-change value and is rejected.
    assert result.selected_iteration == 1

    assert len(
        result.convergence
    ) == 2

    assert list(
        result.convergence[
            "accepted"
        ]
    ) == [
        True,
        False,
    ]