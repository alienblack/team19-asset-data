from pathlib import Path

import numpy as np
import pandas as pd

from team19_asset_data.imputation.config import (
    load_imputation_config,
)
from team19_asset_data.imputation.missforest import (
    MissForestConfig,
    missforest_impute,
)
from team19_asset_data.imputation.postprocess import (
    apply_constraints,
)
from team19_asset_data.imputation.simple_imputer import (
    simple_impute,
)
from team19_asset_data.imputation.validation import (
    validate_imputation_input,
    validate_imputation_result,
)


def test_full_imputation_pipeline_with_simple_and_missforest(
    tmp_path: Path,
):
    config_path = tmp_path / "test_assets.yaml"

    config_path.write_text(
        """
version: 1

asset_class: test_assets

columns:
  numeric:
    - installation_year
    - distance

  categorical:
    - material
    - location

  excluded:
    - asset_id

constraints:
  installation_year:
    data_type: integer
    min: 1930
    max: 2014

  distance:
    data_type: float
    min: 0.0
    max: 350.0

  material:
    data_type: categorical

  location:
    data_type: categorical

methods:
  simple:
    numeric_strategy: mean
    categorical_strategy: mode

  missforest:
    n_estimators: 15
    max_iter: 3
    random_state: 219
    n_jobs: 1
    max_features: sqrt
    max_depth: null
    min_samples_leaf: 1
    class_weight: null
""",
        encoding="utf-8",
    )

    config = load_imputation_config(
        config_path
    )

    frame = pd.DataFrame(
        {
            "installation_year": [
                1980.0,
                1990.0,
                np.nan,
                2000.0,
                2010.0,
                1970.0,
            ],
            "distance": [
                5.0,
                np.nan,
                20.0,
                30.0,
                40.0,
                15.0,
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

    validate_imputation_input(
        frame=frame,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
    )

    # ---------------------------------------------------------
    # Simple imputation path
    # ---------------------------------------------------------

    simple_result = simple_impute(
        incomplete=frame,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
        numeric_strategy=config.simple_config[
            "numeric_strategy"
        ],
        categorical_strategy=config.simple_config[
            "categorical_strategy"
        ],
    )

    simple_processed = apply_constraints(
        original=frame,
        imputed=simple_result,
        constraints=config.constraints,
    )

    validate_imputation_result(
        original=frame,
        imputed=simple_processed,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
    )

    assert not simple_processed[
        list(config.model_columns)
    ].isna().any().any()

    pd.testing.assert_series_equal(
        simple_processed["asset_id"],
        frame["asset_id"],
    )

    # installation_year must respect integer post-processing.
    missing_year = frame[
        "installation_year"
    ].isna()

    assert all(
        float(value).is_integer()
        for value in simple_processed.loc[
            missing_year,
            "installation_year",
        ]
    )

    # ---------------------------------------------------------
    # MissForest path
    # ---------------------------------------------------------

    missforest_config = MissForestConfig(
        **config.missforest_config
    )

    missforest_result = missforest_impute(
        incomplete=frame,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
        config=missforest_config,
    )

    missforest_processed = apply_constraints(
        original=frame,
        imputed=missforest_result.imputed,
        constraints=config.constraints,
    )

    validate_imputation_result(
        original=frame,
        imputed=missforest_processed,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
    )

    assert not missforest_processed[
        list(config.model_columns)
    ].isna().any().any()

    pd.testing.assert_series_equal(
        missforest_processed["asset_id"],
        frame["asset_id"],
    )

    assert (
        0
        <= missforest_result.selected_iteration
        <= missforest_config.max_iter
    )

    assert not missforest_result.convergence.empty