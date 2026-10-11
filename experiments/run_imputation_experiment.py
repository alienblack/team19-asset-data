"""Run a controlled imputation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

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


def _load_json(
    path: Path | None,
) -> dict[str, Any] | None:
    """Load optional JSON metadata."""

    if path is None:
        return None

    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object in metadata file: {path}"
        )

    return data


def _verify_clean_source(
    masked: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    """Verify that masked data corresponds to the supplied clean source."""

    if len(masked) != len(clean):
        raise ValueError(
            "Masked dataset and clean source have different row counts."
        )

    if "segment_id" in masked.columns and "segment_id" in clean.columns:
        masked_ids = (
            masked["segment_id"]
            .astype(str)
            .reset_index(drop=True)
        )

        clean_ids = (
            clean["segment_id"]
            .astype(str)
            .reset_index(drop=True)
        )

        if not masked_ids.equals(clean_ids):
            raise ValueError(
                "segment_id order differs between masked dataset "
                "and clean source."
            )


def _missingness_summary(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Summarise missingness for configured model columns."""

    rows = []

    total_rows = len(frame)

    for column in model_columns:
        missing_count = int(
            frame[column].isna().sum()
        )

        rows.append(
            {
                "column": column,
                "rows": total_rows,
                "missing_count": missing_count,
                "missing_rate": (
                    missing_count / total_rows
                    if total_rows
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def run_experiment(
    input_path: Path,
    clean_source_path: Path,
    config_path: Path,
    output_dir: Path,
    mask_metadata_path: Path | None = None,
    relationship_strength: int | float | None = None,
    noise_level: int | float = 0,
    mask_rule: str | None = None,
) -> None:
    """Run Simple and MissForest on the same incomplete dataset."""

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Masked input not found: {input_path}"
        )

    if not clean_source_path.is_file():
        raise FileNotFoundError(
            f"Clean source not found: {clean_source_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_imputation_config(
        config_path
    )

    incomplete = pd.read_csv(
        input_path
    )

    clean_source = pd.read_csv(
        clean_source_path
    )

    _verify_clean_source(
        masked=incomplete,
        clean=clean_source,
    )

    validate_imputation_input(
        frame=incomplete,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
    )

    missingness = _missingness_summary(
        frame=incomplete,
        model_columns=config.model_columns,
    )

    missingness.to_csv(
        output_dir / "input_missingness.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Simple mean/mode baseline
    # ---------------------------------------------------------

    simple_started = perf_counter()

    simple_raw = simple_impute(
        incomplete=incomplete,
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
        original=incomplete,
        imputed=simple_raw,
        constraints=config.constraints,
    )

    validate_imputation_result(
        original=incomplete,
        imputed=simple_processed,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
    )

    simple_runtime = (
        perf_counter()
        - simple_started
    )

    simple_processed.to_csv(
        output_dir / "simple_imputed.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # MissForest
    # ---------------------------------------------------------

    missforest_config = MissForestConfig(
        **config.missforest_config
    )

    missforest_started = perf_counter()

    missforest_result = missforest_impute(
        incomplete=incomplete,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
        config=missforest_config,
    )

    missforest_processed = apply_constraints(
        original=incomplete,
        imputed=missforest_result.imputed,
        constraints=config.constraints,
    )

    validate_imputation_result(
        original=incomplete,
        imputed=missforest_processed,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
    )

    missforest_runtime = (
        perf_counter()
        - missforest_started
    )

    missforest_processed.to_csv(
        output_dir / "missforest_imputed.csv",
        index=False,
    )

    missforest_result.convergence.to_csv(
        output_dir / "missforest_convergence.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    masking_metadata = _load_json(
        mask_metadata_path
    )

    run_metadata = {
        "asset_class": config.asset_class,
        "relationship_strength": relationship_strength,
        "noise_level": noise_level,
        "mask_rule": mask_rule,
        "input_path": str(input_path),
        "clean_source_path": str(clean_source_path),
        "config_path": str(config_path),
        "mask_metadata_path": (
            str(mask_metadata_path)
            if mask_metadata_path is not None
            else None
        ),
        "row_count": int(
            len(incomplete)
        ),
        "model_columns": list(
            config.model_columns
        ),
        "numeric_columns": list(
            config.numeric_columns
        ),
        "categorical_columns": list(
            config.categorical_columns
        ),
        "simple": {
            "config": config.simple_config,
            "runtime_seconds": simple_runtime,
        },
        "missforest": {
            "config": config.missforest_config,
            "runtime_seconds": missforest_runtime,
            "selected_iteration": (
                missforest_result.selected_iteration
            ),
            "converged": (
                missforest_result.converged
            ),
        },
        "masking_metadata": masking_metadata,
    }

    with (
        output_dir / "run_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_metadata,
            file,
            indent=2,
        )

    print(
        "\nImputation experiment completed successfully."
    )

    print(
        f"Asset class: {config.asset_class}"
    )

    print(
        f"Relationship strength: {relationship_strength}"
    )

    print(
        f"Noise level: {noise_level}"
    )

    print(
        f"Mask rule: {mask_rule}"
    )

    print(
        f"Rows: {len(incomplete)}"
    )

    print(
        f"Simple runtime: {simple_runtime:.3f}s"
    )

    print(
        f"MissForest runtime: {missforest_runtime:.3f}s"
    )

    print(
        "MissForest selected iteration: "
        f"{missforest_result.selected_iteration}"
    )

    print(
        "MissForest converged: "
        f"{missforest_result.converged}"
    )

    print(
        f"Outputs: {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Simple and MissForest imputation "
            "on the same controlled dataset."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Masked/incomplete input CSV.",
    )

    parser.add_argument(
        "--clean-source",
        required=True,
        type=Path,
        help="Clean synthetic source CSV used before masking/noise.",
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Asset-specific imputation YAML.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for experiment outputs.",
    )

    parser.add_argument(
        "--mask-metadata",
        type=Path,
        default=None,
        help="Optional masking metadata JSON.",
    )

    parser.add_argument(
        "--relationship-strength",
        type=float,
        default=None,
        help="Synthetic relationship strength for experiment metadata.",
    )

    parser.add_argument(
        "--noise-level",
        type=float,
        default=0,
        help="Noise level for experiment metadata.",
    )

    parser.add_argument(
        "--mask-rule",
        default=None,
        help="Masking rule identifier.",
    )

    args = parser.parse_args()

    run_experiment(
        input_path=args.input,
        clean_source_path=args.clean_source,
        config_path=args.config,
        output_dir=args.output_dir,
        mask_metadata_path=args.mask_metadata,
        relationship_strength=args.relationship_strength,
        noise_level=args.noise_level,
        mask_rule=args.mask_rule,
    )


if __name__ == "__main__":
    main()