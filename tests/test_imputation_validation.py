import numpy as np
import pandas as pd
import pytest

from team19_asset_data.imputation.validation import (
    validate_imputation_input,
    validate_imputation_result,
)


def test_validate_input_accepts_valid_mixed_dataframe():
    frame = pd.DataFrame(
        {
            "age": [10.0, np.nan, 30.0],
            "material": ["Steel", "Copper", np.nan],
        }
    )

    validate_imputation_input(
        frame=frame,
        numeric_columns=["age"],
        categorical_columns=["material"],
    )


def test_validate_input_rejects_non_dataframe():
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        validate_imputation_input(
            frame=[1, 2, 3],
            numeric_columns=["age"],
            categorical_columns=[],
        )


def test_validate_input_rejects_empty_dataframe():
    frame = pd.DataFrame(
        columns=["age"]
    )

    with pytest.raises(
        ValueError,
        match="dataframe is empty",
    ):
        validate_imputation_input(
            frame=frame,
            numeric_columns=["age"],
            categorical_columns=[],
        )


def test_validate_input_rejects_missing_model_column():
    frame = pd.DataFrame(
        {
            "material": ["Steel", "Copper"]
        }
    )

    with pytest.raises(
        ValueError,
        match="missing requested model columns",
    ):
        validate_imputation_input(
            frame=frame,
            numeric_columns=["age"],
            categorical_columns=["material"],
        )


def test_validate_input_rejects_numeric_categorical_overlap():
    frame = pd.DataFrame(
        {
            "age": [10.0, 20.0]
        }
    )

    with pytest.raises(
        ValueError,
        match="both numeric and categorical",
    ):
        validate_imputation_input(
            frame=frame,
            numeric_columns=["age"],
            categorical_columns=["age"],
        )


def test_validate_input_rejects_duplicate_numeric_columns():
    frame = pd.DataFrame(
        {
            "age": [10.0, 20.0]
        }
    )

    with pytest.raises(
        ValueError,
        match="numeric_columns contains duplicate",
    ):
        validate_imputation_input(
            frame=frame,
            numeric_columns=["age", "age"],
            categorical_columns=[],
        )


def test_validate_input_rejects_all_missing_numeric_column():
    frame = pd.DataFrame(
        {
            "age": [np.nan, np.nan]
        }
    )

    with pytest.raises(
        ValueError,
        match="Numeric column 'age' has no observed values",
    ):
        validate_imputation_input(
            frame=frame,
            numeric_columns=["age"],
            categorical_columns=[],
        )


def test_validate_input_rejects_all_missing_categorical_column():
    frame = pd.DataFrame(
        {
            "material": [np.nan, np.nan]
        }
    )

    with pytest.raises(
        ValueError,
        match="Categorical column 'material' has no observed values",
    ):
        validate_imputation_input(
            frame=frame,
            numeric_columns=[],
            categorical_columns=["material"],
        )


def test_validate_input_rejects_non_numeric_value_in_numeric_column():
    frame = pd.DataFrame(
        {
            "age": [10.0, "invalid", np.nan]
        }
    )

    with pytest.raises(
        ValueError,
        match="non-numeric observed value",
    ):
        validate_imputation_input(
            frame=frame,
            numeric_columns=["age"],
            categorical_columns=[],
        )


def test_validate_result_accepts_valid_imputation():
    original = pd.DataFrame(
        {
            "age": [10.0, np.nan, 30.0],
            "material": ["Steel", np.nan, "Copper"],
            "asset_id": ["A1", "A2", "A3"],
        }
    )

    imputed = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0],
            "material": ["Steel", "Steel", "Copper"],
            "asset_id": ["A1", "A2", "A3"],
        }
    )

    validate_imputation_result(
        original=original,
        imputed=imputed,
        numeric_columns=["age"],
        categorical_columns=["material"],
    )


def test_validate_result_rejects_remaining_missing_values():
    original = pd.DataFrame(
        {
            "age": [10.0, np.nan]
        }
    )

    imputed = pd.DataFrame(
        {
            "age": [10.0, np.nan]
        }
    )

    with pytest.raises(
        ValueError,
        match="still contains",
    ):
        validate_imputation_result(
            original=original,
            imputed=imputed,
            numeric_columns=["age"],
            categorical_columns=[],
        )


def test_validate_result_rejects_changed_row_count():
    original = pd.DataFrame(
        {
            "age": [10.0, np.nan]
        }
    )

    imputed = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0]
        }
    )

    with pytest.raises(
        ValueError,
        match="number of rows",
    ):
        validate_imputation_result(
            original=original,
            imputed=imputed,
            numeric_columns=["age"],
            categorical_columns=[],
        )


def test_validate_result_rejects_changed_index():
    original = pd.DataFrame(
        {
            "age": [10.0, np.nan]
        },
        index=[10, 20],
    )

    imputed = pd.DataFrame(
        {
            "age": [10.0, 20.0]
        },
        index=[0, 1],
    )

    with pytest.raises(
        ValueError,
        match="dataframe index",
    ):
        validate_imputation_result(
            original=original,
            imputed=imputed,
            numeric_columns=["age"],
            categorical_columns=[],
        )


def test_validate_result_rejects_changed_columns():
    original = pd.DataFrame(
        {
            "age": [10.0, np.nan],
            "asset_id": ["A1", "A2"],
        }
    )

    imputed = pd.DataFrame(
        {
            "age": [10.0, 20.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="changed dataframe columns",
    ):
        validate_imputation_result(
            original=original,
            imputed=imputed,
            numeric_columns=["age"],
            categorical_columns=[],
        )


def test_validate_result_rejects_modified_observed_value():
    original = pd.DataFrame(
        {
            "age": [10.0, np.nan, 30.0]
        }
    )

    imputed = pd.DataFrame(
        {
            "age": [11.0, 20.0, 30.0]
        }
    )

    with pytest.raises(
        ValueError,
        match="modified originally observed values",
    ):
        validate_imputation_result(
            original=original,
            imputed=imputed,
            numeric_columns=["age"],
            categorical_columns=[],
        )


def test_validate_result_rejects_modified_non_model_column():
    original = pd.DataFrame(
        {
            "age": [10.0, np.nan],
            "asset_id": ["A1", "A2"],
        }
    )

    imputed = pd.DataFrame(
        {
            "age": [10.0, 20.0],
            "asset_id": ["A1", "CHANGED"],
        }
    )

    with pytest.raises(
        ValueError,
        match="non-model columns",
    ):
        validate_imputation_result(
            original=original,
            imputed=imputed,
            numeric_columns=["age"],
            categorical_columns=[],
        )