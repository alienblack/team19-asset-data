"""Development-only Predictive MissForest benchmark on COMPLETE synthetic truth.

This runner deliberately does not reuse experiments/run_imputation_experiment.py:
that script fits legacy MissForest to the whole supplied masked table.

Protocol:
  * Row-disjoint 60% fit / 10% tune / 10% reserved calibration / 20% final test.
  * Hide a specified fraction of *observed* entries independently in the seven
    approved model columns, AFTER the row split.
  * Fit a training-only Median/Mode baseline and Predictive MissForest using
    only the masked fitting rows. Neither ever receives tune/test ground truth.
  * Optional hyperparameter search evaluates only tuning-partition hidden cells.
  * Test is evaluated once after selection, on the SAME mask for both methods.
  * Calibration partition is RESERVED and not used by this runner.

THIS IS A DEVELOPMENT SMOKE/PRIMARY-SYNTHETIC EXPERIMENT ONLY. It does not
replace the team's scenario engine, repeated-seed protocol, or real SAPN
validation.  The input must contain complete ground truth for all model
columns; never use this runner directly on an incomplete real asset register.

Example:
    python experiments/run_predictive_imputation_experiment.py \\
        --input-csv /path/to/material_model_ready_data.csv \\
        --output-dir outputs/synthetic_pilot \\
        --sample-rows 350 --n-estimators 20

Full tuned experiment (substantially slower):
    python experiments/run_predictive_imputation_experiment.py \\
        --input-csv /path/to/material_model_ready_data.csv \\
        --output-dir outputs/full_tuned --tune --grid full
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from team19_asset_data.imputation.config import load_imputation_config
from team19_asset_data.imputation.evaluation import evaluate_imputation
from team19_asset_data.imputation.postprocess import apply_constraints
from team19_asset_data.imputation.predictive_missforest import PredictiveMissForest
from team19_asset_data.imputation.predictive_tuning import (
    hide_cells, make_mcar_mask, tune_predictive_missforest,
)
from team19_asset_data.imputation.simple_imputer import deterministic_mode


class TrainingMedianMode:
    """Training-fitted Median/Mode baseline; never learns from the test set."""

    def __init__(self, numeric_columns: tuple[str, ...],
                 categorical_columns: tuple[str, ...],
                 constraints: dict[str, dict[str, Any]]):
        self.numeric_columns = tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.constraints = constraints
        self.statistics: dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame) -> "TrainingMedianMode":
        for column in self.numeric_columns:
            observed = pd.to_numeric(frame[column], errors="raise").dropna()
            if observed.empty:
                raise ValueError(f"No observed fitting values for {column}")
            self.statistics[column] = float(observed.median())
        for column in self.categorical_columns:
            self.statistics[column] = deterministic_mode(frame[column])
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.statistics:
            raise RuntimeError("Fit TrainingMedianMode before calling transform.")
        result = frame.copy(deep=True)
        for column in self.numeric_columns + self.categorical_columns:
            result[column] = result[column].fillna(self.statistics[column])
        return apply_constraints(frame, result, self.constraints)


def split_rows(
    complete: pd.DataFrame, seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Disjoint partitions, indices preserved to detect accidental mixing."""
    n = len(complete)
    if n < 80:
        raise ValueError("At least 80 complete rows are required for this pilot.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_fit = int(n * 0.60)
    n_tune = int(n * 0.10)
    n_calibration = int(n * 0.10)
    spans = {
        "fit": order[:n_fit],
        "tune": order[n_fit:n_fit+n_tune],
        "calibration": order[n_fit+n_tune:n_fit+n_tune+n_calibration],
        "test": order[n_fit+n_tune+n_calibration:],
    }
    parts = {key: complete.iloc[ids].copy() for key, ids in spans.items()}
    manifest = pd.DataFrame({"source_row": complete.index, "partition": ""})
    for key, ids in spans.items():
        manifest.iloc[ids, manifest.columns.get_loc("partition")] = key
    if "segment_id" in complete.columns:
        manifest.insert(1, "segment_id", complete["segment_id"].to_numpy())
    assert sum(len(p) for p in parts.values()) == n
    assert not any(parts[a].index.intersection(parts[b].index).size
                   for a in parts for b in parts if a < b)
    return parts, manifest


def require_complete_truth(frame: pd.DataFrame, model_columns: tuple[str, ...]) -> None:
    missing_columns = [c for c in model_columns if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing model columns in complete reference: {missing_columns}")
    incomplete_truth = {c: int(frame[c].isna().sum()) for c in model_columns
                        if frame[c].isna().any()}
    if incomplete_truth:
        raise ValueError(
            "This synthetic pilot requires COMPLETE truth for model columns. "
            f"Found gaps: {incomplete_truth}. Use the team's masking and truth "
            "protocol for incomplete real records."
        )
    if "segment_id" in frame and not frame["segment_id"].is_unique:
        raise ValueError("segment_id must be unique for a traceable experiment.")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="Complete SYNTHETIC truth with seven model fields")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" /
                        "overhead_conductors_prototype_v2.yaml")
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="First N synthetic rows for a fast development smoke test")
    parser.add_argument("--seed", type=int, default=219)
    parser.add_argument("--mask-rate", type=float, default=0.30)
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="Development override; full configuration uses 100")
    parser.add_argument("--n-passes", type=int, default=None,
                        help="Development override; full configuration uses 5")
    parser.add_argument("--tune", action="store_true",
                        help="Search forest hyperparameters on tune split only")
    parser.add_argument("--grid", choices=("quick", "full"), default="full",
                        help="quick=2 configurations for smoke checks; full=12 planned")
    args = parser.parse_args(argv)

    if not 0 < args.mask_rate < 1:
        parser.error("--mask-rate must be strictly between 0 and 1")
    if args.sample_rows is not None and args.sample_rows < 80:
        parser.error("--sample-rows must be at least 80")
    if args.n_estimators is not None and args.n_estimators < 1:
        parser.error("--n-estimators must be positive")
    if args.n_passes is not None and args.n_passes < 1:
        parser.error("--n-passes must be positive")

    config = load_imputation_config(args.config)
    full = pd.read_csv(args.input_csv)
    if args.sample_rows is not None:
        full = full.head(args.sample_rows).copy()
    require_complete_truth(full, config.model_columns)

    parts, manifest = split_rows(full, args.seed)
    masks = {
        name: make_mcar_mask(frame, config.model_columns,
                             rate=args.mask_rate, seed=args.seed + i * 1009)
        for i, (name, frame) in enumerate(parts.items(), start=1)
        if name != "calibration"
    }
    X_fit = hide_cells(parts["fit"], masks["fit"])
    X_tune = hide_cells(parts["tune"], masks["tune"])
    X_test = hide_cells(parts["test"], masks["test"])
    model = PredictiveMissForest.from_imputation_config(config)
    overrides = {}
    if args.n_estimators is not None:
        overrides["n_estimators"] = args.n_estimators
    if args.n_passes is not None:
        overrides["n_passes"] = args.n_passes
    if overrides:
        model = PredictiveMissForest(
            config.numeric_columns, config.categorical_columns,
            model.config.replace(**overrides), config.constraints,
        )

    baseline = TrainingMedianMode(
        config.numeric_columns, config.categorical_columns, config.constraints,
    ).fit(X_fit)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_table = None
    if args.tune:
        grid = (
            {"max_depth": [4, None], "min_samples_leaf": [1]}
            if args.grid == "quick" else None
        )
        tuning_table, winner = tune_predictive_missforest(
            model, X_fit, X_tune, parts["tune"], masks["tune"], grid=grid,
        )
        tuning_table.to_csv(output_dir / "tuning_validation.csv", index=False)
        model = PredictiveMissForest(
            config.numeric_columns, config.categorical_columns,
            winner, config.constraints,
        )

    model.fit(X_fit)
    model.save(output_dir / "predictive_missforest.joblib")
    detailed = model.transform_detailed(X_test, with_scores=True)
    predictive = detailed.imputed
    baseline_output = baseline.transform(X_test)
    # The same held-out test cells are used for all methods.
    metric_tables = []
    for method, imputed in (
        ("median_mode", baseline_output),
        ("predictive_missforest", predictive),
    ):
        score = evaluate_imputation(
            truth=parts["test"], imputed=imputed, mask=masks["test"],
            numeric_columns=config.numeric_columns,
            categorical_columns=config.categorical_columns,
        )
        score.insert(0, "method", method)
        metric_tables.append(score)
    metrics = pd.concat(metric_tables, ignore_index=True)

    # Verify the saved model works and gives identical imputed values.
    reloaded = PredictiveMissForest.load(
        output_dir / "predictive_missforest.joblib"
    )
    pd.testing.assert_frame_equal(reloaded.transform(X_test), predictive)

    manifest.to_csv(output_dir / "partition_manifest.csv", index=False)
    X_test.to_csv(output_dir / "masked_test.csv", index=False)
    masks["test"].to_csv(output_dir / "test_hidden_mask.csv", index=False)
    predictive.to_csv(output_dir / "predictive_imputed_test.csv", index=False)
    baseline_output.to_csv(output_dir / "baseline_imputed_test.csv", index=False)
    detailed.scores.to_csv(output_dir / "predictive_diagnostics_test.csv", index=False)
    model.convergence_.to_csv(output_dir / "convergence_training.csv", index=False)
    metrics.to_csv(output_dir / "metrics_test.csv", index=False)

    meta = {
        "scope": "synthetic development experiment; NOT real SAPN validation",
        "input_reference": str(args.input_csv.resolve()),
        "source_rows": int(len(full)),
        "partitions": {key: int(len(value)) for key, value in parts.items()},
        "calibration": "reserved, not used for fitting, tuning, or scoring",
        "seed": args.seed,
        "mask_rate": args.mask_rate,
        "masked_cell_counts": {
            key: {c: int(mask[c].sum()) for c in config.model_columns}
            for key, mask in masks.items()
        },
        "baseline": "training-only median/mode; constraints on imputed cells only",
        "baseline_statistics": {c: (v.item() if isinstance(v, np.generic) else v)
                                for c, v in baseline.statistics.items()},
        "predictive_config": asdict(model.config),
        "fit_report": model.fit_report_,
        "unknown_category_counts_test": detailed.unknown_category_counts,
        "tuned": args.tune,
        "tuning_candidates": 0 if tuning_table is None else len(tuning_table),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Synthetic pilot complete. Rows: {meta['partitions']}")
    print(f"Trained forest models: {model.fit_report_['n_models']}")
    print(metrics.loc[
        metrics["variable"].isin(["installation_year", "material"]),
        ["method", "variable", "n_masked", "mae", "macro_f1"],
    ].to_string(index=False))
    print(f"Output folder: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
