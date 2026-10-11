"""
Command-line runner for overhead-conductor masking experiments.

Examples
--------
Run one rule:
    python -m team19_asset_data.overhead_conductors.run_masking --rules M02

Run several rules sequentially:
    python -m team19_asset_data.overhead_conductors.run_masking --rules M02 Y02

Use explicit paths:
    python -m team19_asset_data.overhead_conductors.run_masking \
        --rules R04 \
        --input path/to/complete_reference.csv \
        --config path/to/masking_rules.yaml \
        --output-dir path/to/output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .masking import MaskingEngine
except ImportError:
    # Allows direct execution of this file as well as `python -m ...`.
    from masking import MaskingEngine


ROOT_DIR = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG = (
    ROOT_DIR
    / "config"
    / "overhead_conductor_masking_rules.yaml"
)

DEFAULT_INPUT = (
    ROOT_DIR 
    / "data" 
    / "synthetic"
    / "overhead_conductors"
    / "material_model_ready_data.csv"
   
)

DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "data" 
    / "synthetic"
    / "overhead_conductors"
    / "masked"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply configured masking rules to the complete overhead-conductor dataset."
    )

    parser.add_argument(
        "--rules",
        nargs="+",
        required=True,
        help="One or more rule IDs, e.g. M02 or M02 Y02.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input complete-reference CSV. Default: {DEFAULT_INPUT}",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Masking-rules YAML. Default: {DEFAULT_CONFIG}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for masked dataset and metadata. Default: {DEFAULT_OUTPUT_DIR}",
    )

    return parser.parse_args()

def project_path(path: Path) -> str:
    relative_path = path.resolve().relative_to(ROOT_DIR)
    return str(Path(ROOT_DIR.name) / relative_path)

def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    if not args.config.exists():
        raise FileNotFoundError(f"Masking config not found: {args.config}")

    df = pd.read_csv(args.input)
    engine = MaskingEngine.from_yaml(args.config)

    masked_df, metadata = engine.apply_rules(df, args.rules)

    experiment_name = "_".join(args.rules)

    experiment_dir = args.output_dir
    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = experiment_dir / f"masked_{experiment_name}.csv"
    metadata_path = experiment_dir / f"metadata_{experiment_name}.json"

    # Keep hidden truth OUT of the model-facing masked dataset.
    # This prevents accidental leakage during imputation/model training.
    truth_columns = [c for c in masked_df.columns if c.endswith("_true")]
    mask_flag_columns = [c for c in masked_df.columns if c.endswith("__masked")]

    id_columns = ["segment_id"] if "segment_id" in masked_df.columns else []

    truth_path = experiment_dir / f"truth_{experiment_name}.csv"

    if truth_columns:
        truth_df = masked_df[id_columns + truth_columns + mask_flag_columns].copy()

        # Keep only rows that were intentionally masked by at least one rule.
        if mask_flag_columns:
            truth_df = truth_df[truth_df[mask_flag_columns].any(axis=1)]

        truth_df.to_csv(truth_path, index=False)

    model_facing_df = masked_df.drop(
        columns=truth_columns + mask_flag_columns
    )
    model_facing_df.to_csv(dataset_path, index=False)

    metadata_payload = {
        # "input_file": str(args.input),
        # "config_file": str(args.config),
        "input_file": project_path(args.input),
        "config_file": project_path(args.config),
        "rules_applied": list(args.rules),
        # "masked_output_file": str(dataset_path),
        "masked_output_file": project_path(dataset_path),
        # "truth_output_file": str(truth_path) if truth_columns else None,
        "truth_output_file": (
            project_path(truth_path)
            if truth_columns
            else None
        ),
        "rule_metadata": metadata,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata_payload, file, indent=2)

    print("\nMasking completed successfully.")
    print(f"Rules applied: {', '.join(args.rules)}")
    # print(f"Masked dataset: {dataset_path}")
    print(f"Masked dataset: {project_path(dataset_path)}")
    if truth_columns:
        # print(f"Truth dataset:  {truth_path}")
        print(f"Truth dataset:  {project_path(truth_path)}")
    # print(f"Metadata:       {metadata_path}")
    print(f"Metadata:       {project_path(metadata_path)}")

    for item in metadata:
        print(f"\n{item['rule_id']} - {item['name']}")
        print(f"  Strategy: {item['strategy']}")

        if item.get("selected_attributes"):
            print(
                "  Selected attributes: "
                + ", ".join(item["selected_attributes"])
            )

        for column, count in item.get("masked_counts", {}).items():
            print(f"  {column}: {count} values masked")

        for column, count in item.get("linked_masked_counts", {}).items():
            print(f"  linked {column}: {count} values masked")


if __name__ == "__main__":
    main()
