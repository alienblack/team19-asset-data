"""Controlled transformer masking x measurement-noise benchmark.

This experiment uses Team 19's existing imputer implementations and paired
synthetic transformer files. It does not regenerate, clean, or modify source
data. In particular, the complete truth is never passed to an imputer.

Two protocols are explicit: a joint-table comparison where all three models
fit/complete the same measured rows, and an unseen-record comparison where
Median/Mode and Predictive MissForest fit only on training rows. Original
MissForest cannot use the unseen-record protocol. Truth is held out from all
model fitting, and only the same test hidden cells are scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml


SCENARIOS = [(completion, noise) for completion in (0.9, 0.8, 0.7) for noise in (0.0, 0.05, 0.1)]
NUMERIC = (
    "x", "y", "install_date", "altitude", "measured_humid", "measured_salt",
    "measured_overgrowth", "life", "random_placebo", "measured_gas_a",
    "measured_gas_b", "measured_gas_c", "total_gas", "leakage",
)
CATEGORICAL = ("model_id",)
COLUMNS = (
    "x", "y", "install_date", "model_id", "altitude", "measured_humid",
    "measured_salt", "measured_overgrowth", "life", "random_placebo",
    "measured_gas_a", "measured_gas_b", "measured_gas_c", "total_gas", "leakage",
)
NOISY = (
    "measured_humid", "measured_salt", "measured_overgrowth",
    "random_placebo", "measured_gas_a", "measured_gas_b", "measured_gas_c",
)
METHODS = ("median_mode", "missforest", "predictive_missforest")


def load_experiment_config(path: Path) -> dict:
    """Resolve explicit transformer columns and method-specific YAML overrides."""
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    schema = raw.get("columns", {})
    numeric = tuple(schema.get("numeric", []))
    categorical = tuple(schema.get("categorical", []))
    excluded = tuple(schema.get("excluded", []))
    modeled = numeric + categorical
    if not numeric or not categorical or len(set(modeled)) != len(modeled):
        raise ValueError("Model attributes must be nonempty and unique")
    if set(modeled) != set(COLUMNS) or excluded:
        raise ValueError("This all-attribute transformer experiment requires all 15 original columns and no exclusions")
    if set(numeric) != set(NUMERIC) or set(categorical) != set(CATEGORICAL):
        raise ValueError("Transformer numerical/categorical types differ from the source schema")

    methods = raw.get("methods", {})
    baseline = dict(methods.get("median_mode", {}))
    if baseline.get("numeric_strategy") != "median" or baseline.get("categorical_strategy") != "mode":
        raise ValueError("Median/Mode baseline requires numeric median and categorical mode")
    effective = {"median_mode": baseline}
    overrides = raw.get("selected_hyperparameters", {}) or {}
    for method in ("missforest", "predictive_missforest"):
        parameters = dict(methods.get(method, {}))
        selected = dict(overrides.get(method, {}))
        unknown = set(selected) - set(parameters)
        if not parameters or unknown:
            raise ValueError(f"Missing {method} defaults or unknown selected overrides: {sorted(unknown)}")
        parameters.update(selected)
        effective[method] = parameters
    return {
        "source_config": str(path.resolve()),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "effective_methods": effective,
        "selected_hyperparameters": overrides,
        "tuning_provenance": raw.get("tuning_provenance", {}),
        "experiment_notes": raw.get("experiment_notes", {}),
    }


def filename(completion: float, noise: float, kind: str) -> str:
    return f"completion{completion:.1f}-noise{noise}-{kind}.csv"


def read_pair(data_dir: Path, completion: float, noise: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    measured = pd.read_csv(data_dir / filename(completion, noise, "measured"))
    truth = pd.read_csv(data_dir / filename(completion, noise, "truth"))
    for name, frame in (("measured", measured), ("truth", truth)):
        if tuple(frame.columns) != COLUMNS or len(frame) != 10000:
            raise ValueError(f"Unexpected {name} shape or schema for completion={completion}, noise={noise}")
        # Classification labels are nominal, not an ordered numerical predictor.
        frame["model_id"] = frame["model_id"].map(
            lambda value: str(int(value)) if pd.notna(value) else None
        )
    if truth.isna().any().any():
        raise ValueError("Ground truth contains missing values")
    return measured, truth


def audit_source_files(data_dir: Path) -> pd.DataFrame:
    """Verify paired source integrity and the generator's shared masks/noise draws."""
    summary: list[dict] = []
    reference_truth: pd.DataFrame | None = None
    masks: dict[tuple[float, float], pd.DataFrame] = {}
    measured_frames: dict[tuple[float, float], pd.DataFrame] = {}

    for completion, noise in SCENARIOS:
        measured, truth = read_pair(data_dir, completion, noise)
        if reference_truth is None:
            reference_truth = truth
            positions = np.arange(len(truth))
            if not (
                np.array_equal(truth["x"].to_numpy(), positions // 100)
                and np.array_equal(truth["y"].to_numpy(), positions % 100)
            ):
                raise ValueError("Grid-coordinate/row alignment differs from documented generator")
            if not np.allclose(
                truth["altitude"], truth["x"] + truth["y"] + 100, rtol=0, atol=1e-9
            ):
                raise ValueError("Truth altitude identity violated")
        elif not truth.equals(reference_truth):
            raise ValueError("Truth records differ between scenarios")

        mask = measured.isna()
        masks[(completion, noise)] = mask
        measured_frames[(completion, noise)] = measured
        n_observed = int((~mask).sum().sum())
        changed = 0
        for column in NUMERIC:
            visible = ~mask[column]
            delta = np.abs(
                measured.loc[visible, column].to_numpy(dtype=float)
                - truth.loc[visible, column].to_numpy(dtype=float)
            )
            different = ~np.isclose(delta, 0.0, rtol=0, atol=1e-9)
            if column not in NOISY and different.any():
                raise ValueError(f"Unexpected measurement noise in {column}")
            changed += int(different.sum())
        visible_cat = ~mask["model_id"]
        if not measured.loc[visible_cat, "model_id"].equals(truth.loc[visible_cat, "model_id"]):
            raise ValueError("Unexpected categorical corruption in model_id")
        summary.append({
            "completion": completion,
            "mask_rate_target": round(1 - completion, 2),
            "noise_amplitude": noise,
            "rows": len(measured),
            "model_columns": len(COLUMNS),
            "masked_cells": int(mask.sum().sum()),
            "mask_rate_actual": float(mask.to_numpy().mean()),
            "observed_cells": n_observed,
            "changed_observed_numeric": changed,
            "changed_observed_rate_all": changed / n_observed,
        })

    for completion in (0.9, 0.8, 0.7):
        base = masks[(completion, 0.0)]
        for noise in (0.05, 0.1):
            if not masks[(completion, noise)].equals(base):
                raise ValueError("Noise scenarios do not have identical masking")
        if completion != 0.9:
            less_missing = masks[(round(completion + 0.1, 1), 0.0)]
            if not (base | ~less_missing).to_numpy().all():
                raise ValueError("Masks are not nested across completion levels")

        for column in NOISY:
            idx = ~base[column]
            true = reference_truth.loc[idx, column].to_numpy(dtype=float)
            err05 = measured_frames[(completion, 0.05)].loc[idx, column].to_numpy(dtype=float) - true
            err10 = measured_frames[(completion, 0.1)].loc[idx, column].to_numpy(dtype=float) - true
            if not np.allclose(err10, 2 * err05, rtol=1e-8, atol=1e-8):
                raise ValueError(f"Noise draws not aligned for {column}")
    return pd.DataFrame(summary)


def make_spatial_split(truth: pd.DataFrame, seed: int, pilot: bool) -> pd.DataFrame:
    """Select disjoint 10x10 grid tiles and hold out tiles in each x-band."""
    x_block = (truth["x"].to_numpy(dtype=int) // 10)
    y_block = (truth["y"].to_numpy(dtype=int) // 10)
    rng = np.random.default_rng(seed)
    tile_labels: dict[tuple[int, int], str] = {}
    for x_band in range(10):
        y_blocks = rng.permutation(10)
        if pilot:
            kept = y_blocks[:2]
            heldout = set(int(v) for v in kept[:1])
        else:
            kept = y_blocks
            heldout = set(int(v) for v in kept[:2])
        for y_band in kept:
            tile_labels[(x_band, int(y_band))] = "test" if int(y_band) in heldout else "train"
    partition = [tile_labels.get((int(x), int(y)), "unused") for x, y in zip(x_block, y_block)]
    manifest = pd.DataFrame({
        "row_position": np.arange(len(truth)),
        "x_tile": x_block,
        "y_tile": y_block,
        "partition": partition,
        "split_seed": seed,
    })
    train_tiles = set(map(tuple, manifest.loc[manifest.partition == "train", ["x_tile", "y_tile"]].values))
    test_tiles = set(map(tuple, manifest.loc[manifest.partition == "test", ["x_tile", "y_tile"]].values))
    if train_tiles & test_tiles:
        raise RuntimeError("Spatial split shares tiles")
    return manifest


def scenario_metrics(
    data_dir: Path,
    completion: float,
    noise: float,
    manifest: pd.DataFrame,
    *,
    experiment_config: dict,
    split_seed: int,
    pipeline_dir: Path,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    sys.path.insert(0, str(pipeline_dir.resolve()))
    from src.evaluation.metrics import evaluate_imputation
    from src.imputation.registry import (
        LegacyMissForestImputer, MedianModeImputer, PredictiveMissForestImputer,
    )
    from src.imputation.missforest.missforest import MissForestConfig
    from src.imputation.predictive_missforest.predictive_missforest import PredictiveMissForestConfig

    numeric = experiment_config["numeric_columns"]
    categorical = experiment_config["categorical_columns"]
    model_columns = list(numeric + categorical)
    effective = experiment_config["effective_methods"]
    measured, truth = read_pair(data_dir, completion, noise)
    train_idx = manifest.index[manifest.partition == "train"]
    test_idx = manifest.index[manifest.partition == "test"]
    train = measured.loc[train_idx, model_columns].copy()
    test = measured.loc[test_idx, model_columns].copy()
    reference = truth.loc[test_idx, model_columns].copy()
    mask = test.isna()
    outcomes: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    visible = manifest.partition != "unused"
    working = measured.loc[visible, model_columns].copy()

    for protocol in ("joint_table_transductive", "unseen_records"):
        methods = METHODS if protocol == "joint_table_transductive" else (
            "median_mode", "predictive_missforest"
        )
        for method in methods:
            started = perf_counter()
            selected_iteration: int | None = None
            converged: bool | None = None
            if method == "median_mode":
                imputer = MedianModeImputer(numeric, categorical, numeric_strategy=effective[method]["numeric_strategy"])
            elif method == "predictive_missforest":
                config = PredictiveMissForestConfig.from_mapping(effective[method])
                imputer = PredictiveMissForestImputer(numeric, categorical, config=config)
            else:
                config = MissForestConfig(**effective[method])
                imputer = LegacyMissForestImputer(numeric, categorical, config=config)

            if protocol == "joint_table_transductive":
                completed_all = imputer.fit_transform(working).imputed
                completed = completed_all.loc[test_idx, model_columns]
            else:
                imputer.fit(train)
                completed = imputer.transform(test).imputed
            if method == "missforest":
                selected_iteration = int(imputer.last_result_.selected_iteration)
                converged = bool(imputer.last_result_.converged)
                convergence_history = imputer.last_result_.convergence.copy().rename(columns={"iteration": "step"})
            elif method == "predictive_missforest":
                convergence_history = imputer.convergence_.copy().rename(columns={"pass": "step"})
            else:
                convergence_history = pd.DataFrame()
            if not convergence_history.empty:
                for label, value in (
                    ("method", method), ("protocol", protocol), ("completion", completion),
                    ("noise", noise), ("split_seed", split_seed),
                ):
                    convergence_history[label] = value
                histories.append(convergence_history)

            if completed.isna().any().any():
                raise RuntimeError(f"{method} left missing values")
            for column in model_columns:
                observed = ~test[column].isna()
                if not np.array_equal(
                    completed.loc[observed, column].to_numpy(),
                    test.loc[observed, column].to_numpy(),
                ):
                    raise RuntimeError(f"{method} changed observed input values in {column}")

            elapsed = perf_counter() - started
            scored = evaluate_imputation(reference, completed, mask, numeric, categorical)
            scored.insert(0, "method", method)
            scored.insert(1, "protocol", protocol)
            scored.insert(2, "completion", completion)
            scored.insert(3, "noise", noise)
            scored.insert(4, "split_seed", split_seed)
            scored["runtime_seconds"] = elapsed
            scored["selected_iteration"] = selected_iteration
            scored["converged"] = converged
            outcomes.append(scored)
            print(f"  {method} ({protocol}): {elapsed:.1f}s, "
                  f"test hidden cells {int(mask.sum().sum()):,}", flush=True)
    return outcomes, histories


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate baseline-relative numerical skill without masking target differences."""
    baseline = metrics.loc[metrics.method == "median_mode", [
        "completion", "noise", "split_seed", "protocol", "variable", "mae", "macro_f1"
    ]].rename(columns={"mae": "baseline_mae", "macro_f1": "baseline_macro_f1"})
    joined = metrics.merge(baseline, on=["completion", "noise", "split_seed", "protocol", "variable"], validate="many_to_one")
    joined["numeric_skill"] = np.where(
        (joined.variable_type == "numeric") & (joined.baseline_mae > 0),
        1 - joined.mae / joined.baseline_mae,
        np.nan,
    )
    grouped = joined.groupby(["completion", "noise", "split_seed", "method", "protocol"], dropna=False)
    summary = grouped.apply(lambda frame: pd.Series({
        "mean_numeric_skill_ex_placebo": frame.loc[
            (frame.variable_type == "numeric") & (frame.variable != "random_placebo"), "numeric_skill"
        ].mean(),
        "placebo_skill": frame.loc[frame.variable == "random_placebo", "numeric_skill"].mean(),
        "model_id_macro_f1": frame.loc[frame.variable == "model_id", "macro_f1"].mean(),
        "runtime_seconds": frame.runtime_seconds.iloc[0],
    }), include_groups=False).reset_index()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Use 20 spatial tiles, keeping all configured model hyperparameters")
    parser.add_argument("--cases", default="all", help="Comma-separated completion:noise pairs, e.g. 0.8:0,0.8:0.05")
    parser.add_argument("--split-seeds", default="617", help="Comma-separated held-out split seeds")
    args = parser.parse_args()
    if not args.pipeline_dir.joinpath("src", "imputation", "registry.py").exists():
        parser.error("--pipeline-dir must contain src/imputation/registry.py")
    config = load_experiment_config(args.config)
    print("Effective MissForest configuration:", config["effective_methods"]["missforest"], flush=True)
    print("Effective Predictive MissForest configuration:", config["effective_methods"]["predictive_missforest"], flush=True)
    cases = SCENARIOS if args.cases == "all" else [tuple(map(float, token.split(":"))) for token in args.cases.split(",")]
    if not cases or any(case not in SCENARIOS for case in cases):
        parser.error("Cases must be among the nine supplied completion:noise pairs")
    split_seeds = [int(token) for token in args.split_seeds.split(",")]
    if not split_seeds:
        parser.error("At least one split seed is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_source_files(args.data_dir)
    audit.to_csv(args.output_dir / "scenario_audit.csv", index=False)
    print(audit[["completion", "noise_amplitude", "mask_rate_actual", "changed_observed_numeric"]].to_string(index=False), flush=True)

    truth = read_pair(args.data_dir, 0.9, 0.0)[1]
    all_metrics: list[pd.DataFrame] = []
    all_histories: list[pd.DataFrame] = []
    manifest_records: list[pd.DataFrame] = []
    for split_seed in split_seeds:
        manifest = make_spatial_split(truth, split_seed, args.pilot)
        manifest_records.append(manifest)
        print(f"Split {split_seed}: {int((manifest.partition == 'train').sum()):,} train, "
              f"{int((manifest.partition == 'test').sum()):,} test, "
              f"{int((manifest.partition == 'unused').sum()):,} unused", flush=True)
        for completion, noise in cases:
            print(f"Scenario mask={1-completion:.0%}, noise=±{noise:.0%}", flush=True)
            results, histories = scenario_metrics(
                args.data_dir, completion, noise, manifest,
                experiment_config=config,
                split_seed=split_seed,
                pipeline_dir=args.pipeline_dir,
            )
            all_metrics.extend(results)
            all_histories.extend(histories)
    pd.concat(manifest_records).to_csv(args.output_dir / "spatial_split_manifest.csv", index=False)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(args.output_dir / "per_target_metrics.csv", index=False)
    if all_histories:
        pd.concat(all_histories, ignore_index=True).to_csv(args.output_dir / "convergence_history.csv", index=False)
    summary = summarise(metrics)
    summary.to_csv(args.output_dir / "scenario_summary.csv", index=False)
    manifest_info = {
        "source": str(args.data_dir.resolve()),
        "pipeline": str(args.pipeline_dir.resolve()),
        "pilot": bool(args.pilot),
        "cases": [{"completion": c, "noise": n} for c, n in cases],
        "split_seeds": split_seeds,
        "numeric_columns": list(config["numeric_columns"]),
        "categorical_columns": list(config["categorical_columns"]),
        "methods": list(METHODS),
        "n_estimators": config["effective_methods"]["missforest"]["n_estimators"],
        "passes": config["effective_methods"]["predictive_missforest"]["n_passes"],
        "max_iter": config["effective_methods"]["missforest"]["max_iter"],
        "model_seed": config["effective_methods"]["missforest"]["random_state"],
        "effective_methods": config["effective_methods"],
        "selected_hyperparameters": config["selected_hyperparameters"],
        "tuning_provenance": config["tuning_provenance"],
        "experiment_notes": config["experiment_notes"],
        "experiment_config_path": config["source_config"],
        "note": "Single supplied generator seed. CI over split seeds alone is not generator uncertainty.",
        "protocols": {
            "joint_table_transductive": "all three models fitted/completed on the same measured rows",
            "unseen_records": "Median/Mode and Predictive MissForest fitted on training rows and applied to held-out rows",
        },
        "input_policy": "all 15 original attributes; no derived-field reconstruction or cleaning",
    }
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest_info, indent=2), encoding="utf-8")
    print("\nScenario summary\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
