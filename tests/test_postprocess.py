import numpy as np
import pandas as pd
import pytest

from team19_asset_data.imputation.postprocess import (
    apply_constraints,
)


def test_integer_constraint_rounds_imputed_value():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
                2000.0,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1992.7,
                2000.0,
            ]
        }
    )

    result = apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "year": {
                "data_type": "integer",
            }
        },
    )

    assert result.loc[1, "year"] == 1993.0


def test_integer_constraint_clips_below_minimum():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1900.0,
            ]
        }
    )

    result = apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "year": {
                "data_type": "integer",
                "min": 1930,
                "max": 2014,
            }
        },
    )

    assert result.loc[1, "year"] == 1930.0


def test_integer_constraint_clips_above_maximum():
    original = pd.DataFrame(
        {
            "year": [
                np.nan,
                2000.0,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                2050.0,
                2000.0,
            ]
        }
    )

    result = apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "year": {
                "data_type": "integer",
                "min": 1930,
                "max": 2014,
            }
        },
    )

    assert result.loc[0, "year"] == 2014.0


def test_float_constraint_clips_below_minimum():
    original = pd.DataFrame(
        {
            "distance": [
                10.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "distance": [
                10.0,
                -5.5,
            ]
        }
    )

    result = apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "distance": {
                "data_type": "float",
                "min": 0.0,
            }
        },
    )

    assert result.loc[1, "distance"] == 0.0


def test_float_constraint_clips_above_maximum():
    original = pd.DataFrame(
        {
            "distance": [
                np.nan,
                100.0,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "distance": [
                400.0,
                100.0,
            ]
        }
    )

    result = apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "distance": {
                "data_type": "float",
                "min": 0.0,
                "max": 350.0,
            }
        },
    )

    assert result.loc[0, "distance"] == 350.0


def test_postprocess_does_not_modify_observed_value():
    original = pd.DataFrame(
        {
            "year": [
                1920.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1920.0,
                1950.4,
            ]
        }
    )

    result = apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "year": {
                "data_type": "integer",
                "min": 1930,
                "max": 2014,
            }
        },
    )

    # Existing value is outside the configured imputation range,
    # but post-processing must not change known source data.
    assert result.loc[0, "year"] == 1920.0

    assert result.loc[1, "year"] == 1950.0


def test_categorical_constraint_leaves_valid_imputation_unchanged():
    original = pd.DataFrame(
        {
            "material": [
                "Steel",
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "material": [
                "Steel",
                "Copper",
            ]
        }
    )

    result = apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "material": {
                "data_type": "categorical",
            }
        },
    )

    assert result.loc[1, "material"] == "Copper"


def test_categorical_constraint_rejects_remaining_missing_value():
    original = pd.DataFrame(
        {
            "material": [
                "Steel",
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "material": [
                "Steel",
                np.nan,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="still contains",
    ):
        apply_constraints(
            original=original,
            imputed=imputed,
            constraints={
                "material": {
                    "data_type": "categorical",
                }
            },
        )


def test_numeric_constraint_rejects_missing_imputed_value():
    original = pd.DataFrame(
        {
            "distance": [
                10.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "distance": [
                10.0,
                np.nan,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="missing or non-numeric",
    ):
        apply_constraints(
            original=original,
            imputed=imputed,
            constraints={
                "distance": {
                    "data_type": "float",
                }
            },
        )


def test_numeric_constraint_rejects_non_numeric_imputed_value():
    original = pd.DataFrame(
        {
            "distance": [
                10.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "distance": [
                10.0,
                "invalid",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="missing or non-numeric",
    ):
        apply_constraints(
            original=original,
            imputed=imputed,
            constraints={
                "distance": {
                    "data_type": "float",
                }
            },
        )


def test_unconfigured_column_is_unchanged():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ],
            "description": [
                "A",
                "B",
            ],
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1990.8,
            ],
            "description": [
                "A",
                "B",
            ],
        }
    )

    result = apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "year": {
                "data_type": "integer",
            }
        },
    )

    pd.testing.assert_series_equal(
        result["description"],
        imputed["description"],
    )


def test_postprocess_does_not_modify_original_dataframe():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1990.8,
            ]
        }
    )

    original_copy = original.copy(
        deep=True
    )

    apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "year": {
                "data_type": "integer",
            }
        },
    )

    pd.testing.assert_frame_equal(
        original,
        original_copy,
    )


def test_postprocess_does_not_modify_imputed_dataframe():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1990.8,
            ]
        }
    )

    imputed_copy = imputed.copy(
        deep=True
    )

    apply_constraints(
        original=original,
        imputed=imputed,
        constraints={
            "year": {
                "data_type": "integer",
            }
        },
    )

    pd.testing.assert_frame_equal(
        imputed,
        imputed_copy,
    )


def test_constraint_for_missing_column_raises():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1990.0,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Constraint defined for missing column",
    ):
        apply_constraints(
            original=original,
            imputed=imputed,
            constraints={
                "distance": {
                    "data_type": "float",
                }
            },
        )


def test_unsupported_data_type_raises():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1990.0,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Unsupported data_type",
    ):
        apply_constraints(
            original=original,
            imputed=imputed,
            constraints={
                "year": {
                    "data_type": "date",
                }
            },
        )


def test_invalid_min_max_range_raises():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1990.0,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="minimum exceeds maximum",
    ):
        apply_constraints(
            original=original,
            imputed=imputed,
            constraints={
                "year": {
                    "data_type": "integer",
                    "min": 2020,
                    "max": 1900,
                }
            },
        )


def test_changed_row_count_raises():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ]
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1990.0,
                2000.0,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        apply_constraints(
            original=original,
            imputed=imputed,
            constraints={},
        )


def test_changed_index_raises():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ]
        },
        index=[
            10,
            20,
        ],
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1990.0,
            ]
        },
        index=[
            0,
            1,
        ],
    )

    with pytest.raises(
        ValueError,
        match="same index",
    ):
        apply_constraints(
            original=original,
            imputed=imputed,
            constraints={},
        )


def test_changed_columns_raise():
    original = pd.DataFrame(
        {
            "year": [
                1980.0,
                np.nan,
            ],
            "asset_id": [
                "A1",
                "A2",
            ],
        }
    )

    imputed = pd.DataFrame(
        {
            "year": [
                1980.0,
                1990.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="same columns",
    ):
        apply_constraints(
            original=original,
            imputed=imputed,
            constraints={},
        )