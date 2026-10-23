"""Predictive MissForest workflow for incomplete SAPN-like or synthetic scenario data.

The input is ALREADY incomplete (natural missingness and/or the team's separate
noise and masking scripts). This runner never creates the main scenario mask.
For validation/test only, it temporarily hides additional CURRENTLY OBSERVED
cells, scores those cells, and leaves the original input unchanged for the
final fit/transform. This is NOT evaluation of naturally missing values.

YAML controls execution.mode, tuning.enabled, evaluation.enabled and data_split.
Only evaluation/tuning uses partitions. Final fitting uses all eligible input.
For a genuinely independent evaluation with transform_only/evaluate_only, pass
an independent input dataset whose records were not used to fit/tune the model.

Examples (run from project root):
    PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \\
      --input-csv data/sapn/observed.csv \\
      --config config/imputation/overhead_conductors_prototype_v2.yaml \\
      --output-dir outputs/imputation

    PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \\
      --mode evaluate_only --input-csv data/evaluation/independent_test.csv \\
      --config config/imputation/overhead_conductors_prototype_v2.yaml \\
      --output-dir outputs/independent_test

For evaluate_only, set evaluation.enabled: true in YAML and select a model
fitted only on OTHER records (for instance evaluation_model.joblib).

Optional --truth-csv refers to external CLEAN truth for synthetic scoring;
if omitted, only values observed in the supplied (possibly noisy) input are
used as truth for the additional evaluation mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from team19_asset_data.imputation.config import DataSplitConfig, load_imputation_config
from team19_asset_data.imputation.evaluation import (
    evaluate_imputation,
    prepare_evaluation_holdout,
)
from team19_asset_data.imputation.postprocess import apply_constraints
from team19_asset_data.imputation.predictive_missforest import PredictiveMissForest
from team19_asset_data.imputation.predictive_tuning import (
    TuningObjective,
    tune_predictive_missforest,
)
from team19_asset_data.imputation.simple_imputer import deterministic_mode

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "validation", "calibration", "test")


class TrainingMedianMode:
    """Fit statistics from training records only, never evaluation records."""

    def __init__(self, numeric_columns, categorical_columns, constraints):
        self.numeric_columns = tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.constraints = constraints
        self.statistics: dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame) -> "TrainingMedianMode":
        for column in self.numeric_columns:
            observed = pd.to_numeric(frame[column], errors="raise").dropna()
            if observed.empty:
                raise ValueError(f"No observed fitting values for '{column}'.")
            self.statistics[column] = float(observed.median())
        for column in self.categorical_columns:
            self.statistics[column] = deterministic_mode(frame[column])
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.statistics:
            raise RuntimeError("Fit baseline before transforming.")
        out = frame.copy(deep=True)
        for column, value in self.statistics.items():
            out[column] = out[column].fillna(value)
        return apply_constraints(frame, out, self.constraints)


def _ids(frame: pd.DataFrame, id_column: str) -> pd.Series:
    if id_column not in frame:
        raise ValueError(f"'{id_column}' is required for traceable partitions/evaluation.")
    ids = frame[id_column].astype("string")
    if ids.isna().any() or not ids.is_unique or (ids.str.strip() == "").any():
        raise ValueError(f"'{id_column}' must contain unique nonmissing identifiers.")
    return ids.astype(str)


def _allocated_counts(n: int, split: DataSplitConfig) -> dict[str, int]:
    targets = np.array([getattr(split, p) for p in PARTITIONS], dtype=float) * n
    counts = np.floor(targets).astype(int)
    for i in np.argsort(-(targets - counts), kind="stable")[:n - int(counts.sum())]:
        counts[i] += 1
    return dict(zip(PARTITIONS, counts.tolist()))


def _create_manifest(frame: pd.DataFrame, split: DataSplitConfig,
                     id_column: str) -> pd.DataFrame:
    rng = np.random.default_rng(split.random_seed)
    counts = _allocated_counts(len(frame), split)
    chosen = np.empty(len(frame), dtype=object)
    if split.strategy == "row":
        order = rng.permutation(len(frame))
        at = 0
        for name in PARTITIONS:
            chosen[order[at:at + counts[name]]] = name
            at += counts[name]
    else:
        group_col = split.group_column
        if group_col not in frame:
            raise ValueError(f"Group-aware split requires '{group_col}'.")
        if frame[group_col].isna().any():
            raise ValueError(f"Group column '{group_col}' contains missing values.")
        sizes = frame.groupby(group_col, sort=False).size().to_dict()
        groups = list(sizes)
        rng.shuffle(groups)
        active = [name for name in PARTITIONS if getattr(split, name) > 0]
        if len(groups) < len(active):
            raise ValueError("Not enough groups for the requested partitions.")
        assigned, actual = {}, {name: 0 for name in PARTITIONS}
        for name, group in zip(active, groups[:len(active)]):
            assigned[group] = name
            actual[name] += sizes[group]
        for group in groups[len(active):]:
            name = max(active, key=lambda p: (counts[p] - actual[p], -actual[p]))
            assigned[group] = name
            actual[name] += sizes[group]
        chosen = frame[group_col].map(assigned).to_numpy()
    return pd.DataFrame({id_column: _ids(frame, id_column).to_numpy(),
                         "partition": chosen})


def load_or_create_partitions(frame: pd.DataFrame, split: DataSplitConfig,
                              manifest_path: Path, id_column: str = "segment_id"):
    """Stable ID-based splits, independent of row ordering and noise values."""
    ids = _ids(frame, id_column)
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype={id_column: str})
        if not {id_column, "partition"}.issubset(manifest):
            raise ValueError("Partition manifest requires ID and partition fields.")
        if manifest[id_column].isna().any() or not manifest[id_column].is_unique:
            raise ValueError("Partition manifest has duplicate/absent identifiers.")
        if set(manifest["partition"]) - set(PARTITIONS):
            raise ValueError("Partition manifest has unknown partition names.")
        lookup = manifest.set_index(id_column)["partition"]
        if set(lookup.index) != set(ids):
            raise ValueError("Input record identifiers do not match partition manifest.")
        assignment = ids.map(lookup)
        manifest = pd.DataFrame({id_column: ids.to_numpy(),
                                 "partition": assignment.to_numpy()})
        if split.strategy == "group":
            if split.group_column not in frame:
                raise ValueError(f"Missing group column '{split.group_column}'.")
            check = pd.DataFrame({"group": frame[split.group_column].to_numpy(),
                                  "partition": assignment.to_numpy()})
            if check.groupby("group", dropna=False)["partition"].nunique().gt(1).any():
                raise ValueError("Saved manifest splits a group across partitions.")
    else:
        manifest = _create_manifest(frame, split, id_column)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
    parts = {name: frame.iloc[np.flatnonzero(manifest.partition.to_numpy() == name)].copy()
             for name in PARTITIONS}
    return parts, manifest


def _align_reference(source: Path, data: pd.DataFrame, id_column: str,
                     required_columns: tuple[str, ...]) -> pd.DataFrame:
    """Align a clean reference (or saved mask) by stable IDs, not row order.

    The reference may contain additional rows (e.g. full scenario truth while
    evaluating only the test partition), but must cover all requested IDs.
    """
    ref = pd.read_csv(source, dtype={id_column: str})
    source_ids, ref_ids = _ids(data, id_column), _ids(ref, id_column)
    if not set(source_ids).issubset(set(ref_ids)):
        raise ValueError(f"Reference '{source}' lacks some input record IDs.")
    missing_cols = sorted(set(required_columns) - set(ref.columns))
    if missing_cols:
        raise ValueError(f"Reference '{source}' lacks columns: {missing_cols}.")
    reference = ref.set_index(ref_ids).loc[source_ids].reset_index(drop=True)
    reference.index = data.index
    return reference


def _external_eval_mask(path: Path, data: pd.DataFrame, id_column: str,
                        model_columns: tuple[str, ...]) -> pd.DataFrame:
    """Read a saved ADDITIONAL evaluation mask (not the original scenario mask)."""
    raw = _align_reference(path, data, id_column, model_columns)
    result = pd.DataFrame(False, index=data.index, columns=data.columns)
    for column in model_columns:
        values = raw[column].map(lambda x: str(x).strip().lower())
        if not values.isin({"true", "false", "1", "0"}).all():
            raise ValueError(f"Evaluation mask '{column}' must contain booleans.")
        result[column] = values.isin({"true", "1"}).to_numpy()
    return result


def _prepare_eval(data: pd.DataFrame, model_columns: tuple[str, ...],
                  rate: float, seed: int, id_column: str,
                  supplied_mask: Path | None):
    # Hold out all eligible model columns, while allowing some sparsely observed
    # columns to have no valid score. Target metrics require sufficient support.
    eligible = [col for col in model_columns if data[col].notna().sum() >= 2]
    if not eligible:
        raise ValueError("No model column contains two known values for evaluation.")
    if supplied_mask is not None:
        mask = _external_eval_mask(supplied_mask, data, id_column, model_columns)
        if mask.loc[:, model_columns].to_numpy().sum() == 0:
            raise ValueError("Supplied evaluation mask hides no model values.")
        bad = mask.loc[:, model_columns] & data.loc[:, model_columns].isna()
        if bad.to_numpy().any():
            raise ValueError("Evaluation mask selects values that are already missing.")
        masked = data.copy(deep=True)
        masked.loc[:, model_columns] = masked.loc[:, model_columns].mask(
            mask.loc[:, model_columns]
        )
        return masked, data.copy(deep=True), mask
    return prepare_evaluation_holdout(data, eligible, rate=rate, seed=seed)


def _evaluate(model: PredictiveMissForest, source: pd.DataFrame,
              reference: pd.DataFrame, model_columns: tuple[str, ...],
              numeric: tuple[str, ...], categorical: tuple[str, ...],
              mask_rate: float, seed: int, output_dir: Path,
              id_column: str, supplied_mask: Path | None,
              *, include_baseline: bool, baseline_fit: pd.DataFrame | None):
    masked, observed_truth, new_mask = _prepare_eval(
        source, model_columns, mask_rate, seed, id_column, supplied_mask
    )
    # For real SAPN data truth=the originally observed input. Synthetic noise
    # benchmarks can provide the clean ground truth as an EXTERNAL reference.
    truth = reference if reference is not None else observed_truth
    detailed = model.transform_detailed(masked, with_scores=True)
    predictions = detailed.imputed
    table = evaluate_imputation(truth, predictions, new_mask, numeric, categorical)
    table.insert(0, "method", "predictive_missforest")
    if include_baseline:
        base = TrainingMedianMode(numeric, categorical, model.constraints).fit(baseline_fit)
        base_preds = base.transform(masked)
        score = evaluate_imputation(truth, base_preds, new_mask, numeric, categorical)
        score.insert(0, "method", "median_mode")
        table = pd.concat([score, table], ignore_index=True)
        base_preds.to_csv(output_dir / "baseline_imputed_test.csv", index=False)
    masked.to_csv(output_dir / "masked_test.csv", index=False)
    saved_mask = new_mask.copy(deep=True)
    saved_mask[id_column] = source[id_column].to_numpy()
    saved_mask.to_csv(output_dir / "test_hidden_mask.csv", index=False)
    predictions.to_csv(output_dir / "predictive_imputed_test.csv", index=False)
    detailed.scores.to_csv(output_dir / "predictive_diagnostics_test.csv", index=False)
    table.to_csv(output_dir / "metrics_test.csv", index=False)
    return table


def _fingerprints(data: pd.DataFrame, id_column: str) -> set[str]:
    return {hashlib.sha256(x.encode("utf-8")).hexdigest()
            for x in _ids(data, id_column)}


def _write_provenance(path: Path, fitted_rows: pd.DataFrame, id_column: str):
    """Record fitting IDs so later independent evaluations can detect overlap."""
    provenance = {"id_column": id_column,
                  "fitting_id_sha256": sorted(_fingerprints(fitted_rows, id_column))}
    path.with_suffix(path.suffix + ".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def _check_provenance(path: Path, test: pd.DataFrame, id_column: str):
    meta_path = path.with_suffix(path.suffix + ".provenance.json")
    if not meta_path.exists():
        raise ValueError(
            "No saved fitting-ID provenance for this model. Independent evaluation "
            "cannot be verified; evaluate using a model saved by this runner."
        )
    info = json.loads(meta_path.read_text(encoding="utf-8"))
    if info.get("id_column") != id_column:
        raise ValueError("Model ID column differs from evaluation ID column.")
    overlap = _fingerprints(test, id_column).intersection(info["fitting_id_sha256"])
    if overlap:
        raise ValueError(
            f"Evaluation records overlap with the model fitting data ({len(overlap)} IDs)."
        )


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="Already-incomplete SAPN data or prepared synthetic scenario input")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/imputation/overhead_conductors_prototype_v2.yaml")
    parser.add_argument("--mode", choices=("fit_only", "fit_transform", "transform_only", "evaluate_only"),
                        help="Override execution.mode in YAML")
    parser.add_argument("--model-path", type=Path,
                        help="Override execution.model_path in YAML (relative paths are repo-root relative)")
    parser.add_argument("--partition-manifest", type=Path,
                        help="Shared partition manifest to create once/reuse across scenarios")
    parser.add_argument("--evaluation-mask-csv", type=Path,
                        help="Reusable ADDITIONAL evaluation mask, keyed by --id-column")
    parser.add_argument("--truth-csv", type=Path,
                        help="External CLEAN truth for scoring synthetic scenarios only")
    parser.add_argument("--id-column", default="segment_id")
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="Development-only: first N records; not for final reporting")
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="Development-only model override")
    parser.add_argument("--n-passes", type=int, default=None,
                        help="Development-only model override (fixed passes remain unchanged)")
    args = parser.parse_args(argv)

    config = load_imputation_config(args.config)
    wf = config.workflow_config
    if wf is None:
        raise ValueError("This workflow runner requires YAML execution/tuning/evaluation sections.")
    mode = args.mode or wf.execution_mode
    if mode in {"transform_only", "evaluate_only"} and wf.tuning_enabled:
        raise ValueError(f"tuning.enabled must be false for {mode}.")
    if mode == "evaluate_only" and not wf.evaluation_enabled:
        raise ValueError("evaluate_only requires evaluation.enabled: true.")
    if args.n_estimators is not None and args.n_estimators < 1:
        parser.error("--n-estimators must be positive")
    if args.n_passes is not None and args.n_passes < 1:
        parser.error("--n-passes must be positive")

    input_data = pd.read_csv(args.input_csv)
    if args.sample_rows is not None:
        if args.sample_rows < 2:
            parser.error("--sample-rows must be >= 2")
        input_data = input_data.head(args.sample_rows).copy()
    if not input_data.index.is_unique or not input_data.columns.is_unique:
        raise ValueError("Input row index and columns must be unique.")
    missing = set(config.model_columns) - set(input_data.columns)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")
    _ids(input_data, args.id_column)
    model_columns = config.model_columns
    numeric, categorical = config.numeric_columns, config.categorical_columns
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or Path(wf.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model_path = model_path.resolve()

    if args.evaluation_mask_csv and not (wf.evaluation_enabled or mode == "evaluate_only"):
        raise ValueError("--evaluation-mask-csv requires evaluation to be enabled.")
    if args.truth_csv and not (wf.evaluation_enabled or mode == "evaluate_only"):
        raise ValueError("--truth-csv requires evaluation to be enabled.")
    truth = (_align_reference(args.truth_csv, input_data, args.id_column, model_columns)
             if args.truth_csv else None)

    meta = {"execution_mode": mode, "input_reference": str(args.input_csv.resolve()),
            "input_rows": int(len(input_data)), "tuning_enabled": wf.tuning_enabled,
            "evaluation_enabled": wf.evaluation_enabled, "model_path": str(model_path),
            "data_split": asdict(wf.data_split),
            "clean_reference": str(args.truth_csv.resolve()) if args.truth_csv else None,
            "reused_evaluation_mask": (str(args.evaluation_mask_csv.resolve())
                                       if args.evaluation_mask_csv else None)}
    needs_fit = mode in {"fit_only", "fit_transform"}
    do_test = wf.evaluation_enabled or mode == "evaluate_only"
    needs_partitions = needs_fit and (wf.tuning_enabled or do_test)

    if needs_partitions:
        manifest_path = args.partition_manifest or (output_dir / "partition_manifest.csv")
        parts, manifest = load_or_create_partitions(input_data, wf.data_split,
                                                    manifest_path, args.id_column)
        manifest.to_csv(output_dir / "partition_manifest.csv", index=False)
        meta["partitions"] = {key: len(value) for key, value in parts.items()}
        if parts["train"].empty:
            raise ValueError("Empty training partition.")
        if wf.tuning_enabled and parts["validation"].empty:
            raise ValueError("Empty validation partition.")
        if do_test and parts["test"].empty:
            raise ValueError("Empty test partition.")
    elif args.partition_manifest and mode == "evaluate_only":
        parts, manifest = load_or_create_partitions(input_data, wf.data_split,
                                                    args.partition_manifest, args.id_column)
        input_data = parts["test"]
        if input_data.empty:
            raise ValueError("Manifest selects an empty evaluation partition.")
        if truth is not None:
            truth = truth.loc[input_data.index]
    elif args.partition_manifest:
        raise ValueError("--partition-manifest is only supported with tuning/evaluation or evaluate_only.")

    if needs_fit:
        base_model = PredictiveMissForest.from_imputation_config(config)
        overrides = {}
        if args.n_estimators is not None:
            overrides["n_estimators"] = args.n_estimators
        if args.n_passes is not None:
            overrides["n_passes"] = args.n_passes
        if overrides:
            base_model = PredictiveMissForest(
                numeric, categorical, base_model.config.replace(**overrides), config.constraints
            )
        selected_config = base_model.config
        if wf.tuning_enabled:
            results, selected_config = tune_predictive_missforest(
                base=base_model, X_fit=parts["train"], X_tune=parts["validation"],
                grid=wf.tuning_grid,
                objective=TuningObjective(mode=wf.tuning_objective),
                mask_rate=wf.tuning_mask_rate,
                mask_seed=wf.tuning_mask_seed,
            )
            results.to_csv(output_dir / "tuning_validation.csv", index=False)
            meta["tuning_objective"] = wf.tuning_objective
            meta["selected_hyperparameters"] = asdict(selected_config)

        if do_test:
            # Temporary model: fit on train only, score held-out test cells.
            evaluation_model = PredictiveMissForest(
                numeric, categorical, selected_config, config.constraints
            )
            evaluation_model.fit(parts["train"])
            # Save the held-out model separately. The final full-data model below
            # is NOT eligible for independent evaluation on these same records.
            eval_model_path = output_dir / "evaluation_model.joblib"
            evaluation_model.save(eval_model_path)
            _write_provenance(eval_model_path, parts["train"], args.id_column)
            meta["evaluation_model_path"] = str(eval_model_path)
            test_truth = truth.loc[parts["test"].index] if truth is not None else None
            _evaluate(
                evaluation_model, parts["test"], test_truth,
                model_columns, numeric, categorical, wf.evaluation_mask_rate,
                wf.evaluation_mask_seed, output_dir, args.id_column,
                args.evaluation_mask_csv, include_baseline=True,
                baseline_fit=parts["train"],
            )
            meta["test_model_fit_rows"] = len(parts["train"])

        # Final fitting is separate: fit on ALL eligible input records.
        final_model = PredictiveMissForest(
            numeric, categorical, selected_config, config.constraints
        )
        final_model.fit(input_data)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(model_path)
        _write_provenance(model_path, input_data, args.id_column)
        final_model.convergence_.to_csv(output_dir / "convergence_training.csv", index=False)
        meta["final_fit_rows"] = len(input_data)
        meta["fit_report"] = final_model.fit_report_
        if mode == "fit_transform":
            detail = final_model.transform_detailed(input_data, with_scores=True)
            detail.imputed.to_csv(output_dir / "predictive_imputed_full.csv", index=False)
            detail.scores.to_csv(output_dir / "predictive_diagnostics_full.csv", index=False)
    else:
        if not model_path.is_file():
            raise FileNotFoundError(f"No fitted model found: {model_path}")
        final_model = PredictiveMissForest.load(model_path)
        if do_test:
            _check_provenance(model_path, input_data, args.id_column)
            _evaluate(
                final_model, input_data, truth, model_columns, numeric, categorical,
                wf.evaluation_mask_rate, wf.evaluation_mask_seed, output_dir,
                args.id_column, args.evaluation_mask_csv,
                include_baseline=False, baseline_fit=None,
            )
        if mode == "transform_only":
            detail = final_model.transform_detailed(input_data, with_scores=True)
            detail.imputed.to_csv(output_dir / "predictive_imputed_full.csv", index=False)
            detail.scores.to_csv(output_dir / "predictive_diagnostics_full.csv", index=False)

    (output_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Predictive MissForest workflow complete: {mode}; outputs: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
