import numpy as np
import pandas as pd
import pytest

from team19_asset_data.imputation.simple_imputer import (
    deterministic_mode,
    simple_impute,
)


def test_deterministic_mode_returns_most_frequent_value():
    series = pd.Series(
        ["Steel", "Steel", "Copper", "Aluminium"]
    )

    result = deterministic_mode(series)

    assert result == "Steel"


def test_deterministic_mode_breaks_ties_deterministically():
    series = pd.Series(
        ["Steel", "Copper", "Steel", "Copper"]
    )

    result = deterministic_mode(series)

    # Alphabetical/string-order tie break.
    assert result == "Copper"


def test_deterministic_mode_ignores_missing_values():
    series = pd.Series(
        ["Steel", np.nan, "Steel", "Copper", np.nan]
    )

    result = deterministic_mode(series)

    assert result == "Steel"


def test_deterministic_mode_raises_when_all_values_missing():
    series = pd.Series(
        [np.nan, np.nan, np.nan],
        name="material",
    )

    with pytest.raises(
        ValueError,
        match="Categorical column 'material' has no observed values",
    ):
        deterministic_mode(series)


def test_simple_impute_fills_numeric_missing_values_with_mean():
    frame = pd.DataFrame(
        {
            "installation_year": [
                1980.0,
                2000.0,
                np.nan,
            ]
        }
    )

    result = simple_impute(
        incomplete=frame,
        numeric_columns=["installation_year"],
        categorical_columns=[],
    )

    # mean = (1980 + 2000) / 2 = 1990
    assert result.loc[2, "installation_year"] == 1990.0


def test_simple_impute_fills_categorical_missing_values_with_mode():
    frame = pd.DataFrame(
        {
            "material": [
                "Steel",
                "Steel",
                "Copper",
                np.nan,
            ]
        }
    )

    result = simple_impute(
        incomplete=frame,
        numeric_columns=[],
        categorical_columns=["material"],
    )

    assert result.loc[3, "material"] == "Steel"


def test_simple_impute_handles_mixed_numeric_and_categorical_columns():
    frame = pd.DataFrame(
        {
            "installation_year": [
                1980.0,
                2000.0,
                np.nan,
            ],
            "distance_to_sea_km": [
                10.0,
                np.nan,
                30.0,
            ],
            "material": [
                "Steel",
                "Steel",
                np.nan,
            ],
            "location_group": [
                "Rural",
                np.nan,
                "Rural",
            ],
        }
    )

    result = simple_impute(
        incomplete=frame,
        numeric_columns=[
            "installation_year",
            "distance_to_sea_km",
        ],
        categorical_columns=[
            "material",
            "location_group",
        ],
    )

    assert result.loc[2, "installation_year"] == 1990.0
    assert result.loc[1, "distance_to_sea_km"] == 20.0
    assert result.loc[2, "material"] == "Steel"
    assert result.loc[1, "location_group"] == "Rural"


def test_simple_impute_does_not_change_observed_values():
    frame = pd.DataFrame(
        {
            "installation_year": [
                1980.0,
                2000.0,
                np.nan,
            ],
            "material": [
                "Steel",
                "Copper",
                np.nan,
            ],
        }
    )

    result = simple_impute(
        incomplete=frame,
        numeric_columns=["installation_year"],
        categorical_columns=["material"],
    )

    assert result.loc[0, "installation_year"] == 1980.0
    assert result.loc[1, "installation_year"] == 2000.0

    assert result.loc[0, "material"] == "Steel"
    assert result.loc[1, "material"] == "Copper"


def test_simple_impute_does_not_modify_input_dataframe():
    frame = pd.DataFrame(
        {
            "installation_year": [
                1980.0,
                np.nan,
            ],
            "material": [
                "Steel",
                np.nan,
            ],
        }
    )

    original = frame.copy(deep=True)

    simple_impute(
        incomplete=frame,
        numeric_columns=["installation_year"],
        categorical_columns=["material"],
    )

    pd.testing.assert_frame_equal(
        frame,
        original,
    )


def test_simple_impute_removes_missing_values_from_requested_columns():
    frame = pd.DataFrame(
        {
            "installation_year": [
                1980.0,
                np.nan,
                2000.0,
            ],
            "material": [
                np.nan,
                "Steel",
                "Steel",
            ],
        }
    )

    result = simple_impute(
        incomplete=frame,
        numeric_columns=["installation_year"],
        categorical_columns=["material"],
    )

    assert not result["installation_year"].isna().any()
    assert not result["material"].isna().any()


def test_simple_impute_leaves_unrequested_columns_unchanged():
    frame = pd.DataFrame(
        {
            "installation_year": [
                1980.0,
                np.nan,
            ],
            "asset_age": [
                34.0,
                np.nan,
            ],
            "material": [
                "Steel",
                np.nan,
            ],
        }
    )

    result = simple_impute(
        incomplete=frame,
        numeric_columns=["installation_year"],
        categorical_columns=["material"],
    )

    # asset_age was not supplied to the imputer.
    assert result.loc[0, "asset_age"] == 34.0
    assert pd.isna(result.loc[1, "asset_age"])


def test_simple_impute_raises_for_missing_requested_column():
    frame = pd.DataFrame(
        {
            "material": [
                "Steel",
                np.nan,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Input is missing requested columns",
    ):
        simple_impute(
            incomplete=frame,
            numeric_columns=["installation_year"],
            categorical_columns=["material"],
        )


def test_simple_impute_rejects_column_in_both_type_lists():
    frame = pd.DataFrame(
        {
            "material": [
                "Steel",
                np.nan,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="both numeric and categorical",
    ):
        simple_impute(
            incomplete=frame,
            numeric_columns=["material"],
            categorical_columns=["material"],
        )


def test_simple_impute_raises_when_numeric_column_has_no_observed_values():
    frame = pd.DataFrame(
        {
            "distance": [
                np.nan,
                np.nan,
                np.nan,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Numeric column 'distance' has no observed values",
    ):
        simple_impute(
            incomplete=frame,
            numeric_columns=["distance"],
            categorical_columns=[],
        )


def test_simple_impute_raises_when_categorical_column_has_no_observed_values():
    frame = pd.DataFrame(
        {
            "material": [
                np.nan,
                np.nan,
                np.nan,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Categorical column 'material' has no observed values",
    ):
        simple_impute(
            incomplete=frame,
            numeric_columns=[],
            categorical_columns=["material"],
        )


def test_simple_impute_rejects_unsupported_numeric_strategy():
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
        match="Unsupported numeric strategy 'median'",
    ):
        simple_impute(
            incomplete=frame,
            numeric_columns=["age"],
            categorical_columns=[],
            numeric_strategy="median",
        )


def test_simple_impute_rejects_unsupported_categorical_strategy():
    frame = pd.DataFrame(
        {
            "material": [
                "Steel",
                np.nan,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Unsupported categorical strategy",
    ):
        simple_impute(
            incomplete=frame,
            numeric_columns=[],
            categorical_columns=["material"],
            categorical_strategy="most_frequent",
        )


def test_simple_impute_accepts_explicit_supported_strategies():
    frame = pd.DataFrame(
        {
            "age": [
                10.0,
                np.nan,
                30.0,
            ],
            "material": [
                "Steel",
                "Steel",
                np.nan,
            ],
        }
    )

    result = simple_impute(
        incomplete=frame,
        numeric_columns=["age"],
        categorical_columns=["material"],
        numeric_strategy="mean",
        categorical_strategy="mode",
    )

    assert result.loc[1, "age"] == 20.0
    assert result.loc[2, "material"] == "Steel"