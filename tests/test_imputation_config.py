from pathlib import Path

import pytest

from team19_asset_data.imputation.config import (
    load_imputation_config,
)


def write_config(
    tmp_path: Path,
    content: str,
) -> Path:

    path = tmp_path / "config.yaml"

    path.write_text(
        content,
        encoding="utf-8",
    )

    return path


VALID_CONFIG = """
version: 1
asset_class: test_assets

columns:
  numeric:
    - age
    - length

  categorical:
    - material
    - location

  excluded:
    - asset_id

constraints:
  age:
    data_type: integer
    min: 0
    max: 100

  length:
    data_type: float
    min: 0.0

methods:
  simple:
    numeric_strategy: mean
    categorical_strategy: mode

  missforest:
    n_estimators: 100
    max_iter: 10
    random_state: 219
    n_jobs: -1
    max_features: sqrt
    max_depth: null
    min_samples_leaf: 1
    class_weight: null
"""


def test_load_valid_configuration(
    tmp_path,
):
    path = write_config(
        tmp_path,
        VALID_CONFIG,
    )

    config = load_imputation_config(
        path
    )

    assert config.version == 1

    assert (
        config.asset_class
        == "test_assets"
    )

    assert config.numeric_columns == (
        "age",
        "length",
    )

    assert config.categorical_columns == (
        "material",
        "location",
    )

    assert config.excluded_columns == (
        "asset_id",
    )

    assert config.model_columns == (
        "age",
        "length",
        "material",
        "location",
    )

    assert (
        config.missforest_config[
            "n_estimators"
        ]
        == 100
    )


def test_missing_config_file_raises():
    with pytest.raises(
        FileNotFoundError
    ):
        load_imputation_config(
            "does_not_exist.yaml"
        )


def test_numeric_and_categorical_overlap_raises(
    tmp_path,
):
    config_text = VALID_CONFIG.replace(
        "  categorical:\n"
        "    - material\n"
        "    - location",
        "  categorical:\n"
        "    - age\n"
        "    - location",
    )

    path = write_config(
        tmp_path,
        config_text,
    )

    with pytest.raises(
        ValueError,
        match="both numeric and categorical",
    ):
        load_imputation_config(path)


def test_model_and_excluded_overlap_raises(
    tmp_path,
):
    config_text = VALID_CONFIG.replace(
        "  excluded:\n"
        "    - asset_id",
        "  excluded:\n"
        "    - age",
    )

    path = write_config(
        tmp_path,
        config_text,
    )

    with pytest.raises(
        ValueError,
        match="also be excluded",
    ):
        load_imputation_config(path)


def test_constraint_for_unknown_column_raises(
    tmp_path,
):
    config_text = VALID_CONFIG.replace(
        "constraints:\n",
        "constraints:\n"
        "  unknown_column:\n"
        "    data_type: float\n"
        "    min: 0\n\n",
        1,
    )

    path = write_config(
        tmp_path,
        config_text,
    )

    with pytest.raises(
        ValueError,
        match="unknown model column",
    ):
        load_imputation_config(path)

def test_invalid_constraint_range_raises(
    tmp_path,
):
    config_text = VALID_CONFIG.replace(
        "    min: 0\n"
        "    max: 100",
        "    min: 100\n"
        "    max: 0",
    )

    path = write_config(
        tmp_path,
        config_text,
    )

    with pytest.raises(
        ValueError,
        match="minimum exceeds maximum",
    ):
        load_imputation_config(path)


def test_duplicate_columns_raise(
    tmp_path,
):
    config_text = VALID_CONFIG.replace(
        "  numeric:\n"
        "    - age\n"
        "    - length",
        "  numeric:\n"
        "    - age\n"
        "    - age",
    )

    path = write_config(
        tmp_path,
        config_text,
    )

    with pytest.raises(
        ValueError,
        match="duplicate columns",
    ):
        load_imputation_config(path)


def test_invalid_missforest_estimators_raise(
    tmp_path,
):
    config_text = VALID_CONFIG.replace(
        "n_estimators: 100",
        "n_estimators: 0",
    )

    path = write_config(
        tmp_path,
        config_text,
    )

    with pytest.raises(
        ValueError,
        match="n_estimators",
    ):
        load_imputation_config(path)