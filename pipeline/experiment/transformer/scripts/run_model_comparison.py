"""T2 extension: common held-out transformer imputation model comparison.

Uses only the supplied measured/truth pairs, original T2 spatial split,
original T2 evaluation cells, and training-only imputers. Existing T2/T3
files are NEVER overwritten. Ground truth is only read by the scorer.

Run from any working directory:
  python experiment/transformer/scripts/run_model_comparison.py --methods median_mode,chained_ridge,lightgbm_chained
  python experiment/transformer/scripts/run_model_comparison.py --quick --methods median_mode,chained_ridge,lightgbm_chained

TabPFN is an optional separate method and is not installed by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
sys.path.insert(0, str(HERE))
from comparison_models import ModelSpec, make_candidate
from transformer_noise_masking import CATEGORICAL, NUMERIC, read_pair, make_spatial_split

ALLOWED = ("median_mode", "chained_ridge", "lightgbm_chained", "tabpfn_unsupervised")
REFERENCE_PROTOCOL = "unseen_records"
REFERENCE = EXPERIMENT_DIR / "outputs" / "notebook_full_20261111_092108" / "per_target_metrics.csv"


def _load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Expected model_comparison/config.yaml version 1")
    return data


def _validate_output(original: pd.DataFrame, filled: pd.DataFrame, spec: ModelSpec) -> None:
    if not original.index.equals(filled.index) or list(original.columns) != list(filled.columns):
        raise AssertionError("Imputation changed record IDs or columns")
    if filled[list(spec.columns)].isna().any().any():
        raise AssertionError("Imputer left unfilled model columns")
    for c in spec.columns:
        observed = original[c].notna()
        left = original.loc[observed, c].to_numpy()
        right = filled.loc[observed, c].to_numpy()
        if not np.array_equal(left, right):
            raise AssertionError(f"Imputation changed observed values in {c}")


def _score(
    original: pd.DataFrame,
    filled: pd.DataFrame,
    truth: pd.DataFrame,
    split: pd.DataFrame,
    spec: ModelSpec,
    *,
    method: str,
    completion: float,
    noise: float,
    seed: int,
    fit_seconds: float,
    transform_seconds: float,
    baseline_filled: pd.DataFrame,
    export_cells: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cells = [], []
    for c in spec.columns:
        missing = original[c].isna()
        ix = original.index[missing]
        if len(ix) == 0:
            continue
        y_true = truth.loc[ix, c]
        y_pred = filled.loc[ix, c]
        y_baseline = baseline_filled.loc[ix, c]
        row = dict(
            method=method, protocol=REFERENCE_PROTOCOL, completion=completion,
            noise=noise, split_seed=seed, variable=c,
            variable_type="categorical" if c in spec.categorical else "numeric",
            n_masked=len(ix), fit_seconds=fit_seconds,
            transform_seconds=transform_seconds,
        )
        if c in spec.numeric:
            t = y_true.to_numpy(dtype=float)
            p = y_pred.to_numpy(dtype=float)
            b = y_baseline.to_numpy(dtype=float)
            err = p - t
            base_mae = float(np.abs(b - t).mean())
            row.update(
                mae=float(np.abs(err).mean()),
                rmse=float(np.sqrt(np.square(err).mean())),
                median_absolute_error=float(np.median(np.abs(err))),
                signed_bias=float(np.mean(err)),
                baseline_mae=base_mae,
                numeric_skill=(1-float(np.abs(err).mean())/base_mae) if base_mae > 0 else np.nan,
                accuracy=np.nan, balanced_accuracy=np.nan, macro_f1=np.nan,
                n_classes_truth=np.nan,
            )
        else:
            t = y_true.astype(str).to_numpy()
            p = y_pred.astype(str).to_numpy()
            labels = sorted(set(t) | set(p))
            row.update(
                mae=np.nan, rmse=np.nan, median_absolute_error=np.nan,
                signed_bias=np.nan, baseline_mae=np.nan, numeric_skill=np.nan,
                accuracy=float(accuracy_score(t, p)),
                balanced_accuracy=float(balanced_accuracy_score(t, p)),
                macro_f1=float(f1_score(t, p, average="macro", labels=labels, zero_division=0)),
                n_classes_truth=len(set(t)),
            )
        rows.append(row)
        if export_cells:
            subset = split.loc[ix, ["row_position", "x_tile", "y_tile"]].copy()
            subset["method"] = method
            subset["protocol"] = REFERENCE_PROTOCOL
            subset["completion"] = completion
            subset["noise"] = noise
            subset["split_seed"] = seed
            subset["variable"] = c
            subset["variable_type"] = row["variable_type"]
            subset["truth"] = y_true.astype(str).to_numpy()
            subset["prediction"] = y_pred.astype(str).to_numpy()
            subset["baseline_prediction"] = y_baseline.astype(str).to_numpy()
            if c in spec.numeric:
                subset["absolute_error"] = np.abs(y_pred.to_numpy(dtype=float) - y_true.to_numpy(dtype=float))
                subset["baseline_absolute_error"] = np.abs(y_baseline.to_numpy(dtype=float) - y_true.to_numpy(dtype=float))
            else:
                subset["absolute_error"] = (y_pred.astype(str).to_numpy() != y_true.astype(str).to_numpy()).astype(float)
                subset["baseline_absolute_error"] = (y_baseline.astype(str).to_numpy() != y_true.astype(str).to_numpy()).astype(float)
            cells.append(subset.reset_index(drop=True))
    return pd.DataFrame(rows), pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()


def _compare_t2_baseline(scores: pd.DataFrame, *, tolerance: float = 1e-8) -> None:
    if not REFERENCE.is_file():
        print("No saved T2 result table found; baseline check skipped", flush=True)
        return
    source = pd.read_csv(REFERENCE)
    ref = source.loc[
        (source.method == "median_mode") & (source.protocol == REFERENCE_PROTOCOL) &
        (source.completion == 0.8) & (source.noise == 0.05) & (source.split_seed == 617),
        ["variable", "n_masked", "mae", "macro_f1"],
    ]
    cur = scores.loc[scores.method == "median_mode", ["variable", "n_masked", "mae", "macro_f1"]]
    paired = cur.merge(ref, on="variable", suffixes=("_new", "_t2"), validate="one_to_one")
    if len(paired) != len(NUMERIC) + len(CATEGORICAL):
        raise AssertionError("Reference baseline has different target columns")
    if not np.array_equal(paired.n_masked_new, paired.n_masked_t2):
        raise AssertionError("Baseline scoring mask differs from saved T2")
    for metric in ("mae", "macro_f1"):
        new = paired[f"{metric}_new"].to_numpy(dtype=float)
        old = paired[f"{metric}_t2"].to_numpy(dtype=float)
        valid = np.isfinite(new) & np.isfinite(old)
        max_error = float(np.max(abs(new[valid] - old[valid]))) if valid.any() else 0
        print(f"T2 baseline check {metric}: max absolute difference {max_error:.6g}")
        if max_error > tolerance:
            raise AssertionError(
                f"Median/Mode does not reproduce original T2 metric {metric}; "
                "do not claim matched comparability until this is fixed"
            )


def _merge_reference(results: pd.DataFrame, *, completion: float, noise: float, split_seed: int) -> pd.DataFrame:
    """Annotate existing T2 scores without passing them through a new implementation."""
    if not REFERENCE.is_file():
        return results
    source = pd.read_csv(REFERENCE)
    existing = source.loc[
        (source.protocol == REFERENCE_PROTOCOL) &
        (source.completion == completion) & (source.noise == noise) &
        (source.split_seed == split_seed) &
        (source.method == "predictive_missforest")
    ].copy()
    if existing.empty:
        return results
    base = results.loc[results.method == "median_mode", ["variable", "baseline_mae"]]
    existing = existing.merge(base, on="variable", how="left", validate="one_to_one")
    existing["numeric_skill"] = np.where(
        existing.baseline_mae > 0, 1 - existing.mae / existing.baseline_mae, np.nan
    )
    existing["fit_seconds"] = np.nan  # T2 stores combined fit+transform time
    existing["transform_seconds"] = np.nan
    return pd.concat([results, existing.reindex(columns=results.columns)], ignore_index=True)


def run(
    *, methods: list[str], quick: bool, cases: list[tuple[float, float]],
    output_dir: Path, config_path: Path, split_seed: int = 617,
    export_cells: bool = True,
) -> pd.DataFrame:
    if len(set(methods)) != len(methods) or any(method not in ALLOWED for method in methods):
        raise ValueError(f"Methods must be unique and in {ALLOWED}")
    if "median_mode" not in methods:
        methods = ["median_mode", *methods]
    config = _load_config(config_path)
    spec = ModelSpec(numeric=NUMERIC, categorical=CATEGORICAL, seed=int(config.get("seed", 219)))
    config_methods = config.get("methods", {})
    output_dir.mkdir(parents=True, exist_ok=False)
    output_rows = []
    run_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "experiment/transformer/dataset (original uploaded files)",
        "protocol": REFERENCE_PROTOCOL,
        "quick": quick,
        "split_seed": split_seed,
        "methods": methods,
        "config": config,
        "cases": cases,
        "ground_truth_policy": "Truth only used after fit+transform to score originally missing held-out cells",
        "baseline_rule": "Median/Mode fit only on measured training records",
        "important": "Original T2/T3 untouched; reference PMF scores read from saved T2 CSV if available",
    }
    for completion, noise in cases:
        measured, truth = read_pair(EXPERIMENT_DIR / "dataset", completion, noise)
        split = make_spatial_split(truth, split_seed, quick)
        train_idx = split.index[split.partition == "train"]
        test_idx = split.index[split.partition == "test"]
        if set(train_idx) & set(test_idx):
            raise AssertionError("Train/test leakage")
        train = measured.loc[train_idx, spec.columns].copy()
        test = measured.loc[test_idx, spec.columns].copy()
        test_truth = truth.loc[test_idx, spec.columns].copy()
        print(f"Scenario {completion=}, {noise=}, {len(train)=}, {len(test)=}", flush=True)
        baseline = make_candidate("median_mode", spec, config_methods).fit(train)
        baseline_filled = baseline.transform(test)
        _validate_output(test, baseline_filled, spec)
        for name in methods:
            imputer = make_candidate(name, spec, config_methods)
            if quick:
                from comparison_models import ChainedConditionalImputer, TabPFNExperimentalImputer
                if isinstance(imputer, ChainedConditionalImputer):
                    imputer.passes = 1
                    imputer.lgbm_estimators = min(25, imputer.lgbm_estimators)
                elif isinstance(imputer, TabPFNExperimentalImputer):
                    imputer.max_context_rows = min(200, imputer.max_context_rows)
            started = perf_counter()
            imputer.fit(train)
            fit_seconds = perf_counter() - started
            started = perf_counter()
            filled = imputer.transform(test)
            transform_seconds = perf_counter() - started
            _validate_output(test, filled, spec)
            metrics, individual = _score(
                test, filled, test_truth, split, spec,
                method=name, completion=completion, noise=noise, seed=split_seed,
                fit_seconds=fit_seconds, transform_seconds=transform_seconds,
                baseline_filled=baseline_filled, export_cells=export_cells,
            )
            output_rows.append(metrics)
            if export_cells:
                label = f"{name}_completion{completion:.1f}_noise{noise:.2f}"
                individual.to_csv(output_dir / f"cells_{label}.csv", index=False)
            print(f" {name}: fit={fit_seconds:.2f}s, transform={transform_seconds:.2f}s, "
                  f"skill={metrics.loc[(metrics.variable_type=='numeric') & (metrics.variable!='random_placebo'), 'numeric_skill'].mean():.3f}",
                  flush=True)
            if name == "tabpfn_unsupervised":
                run_metadata["tabpfn_context_rows"] = int(imputer.context_rows_)
                run_metadata["tabpfn_model_version"] = imputer.model_version
                run_metadata["tabpfn_context_caveat"] = (
                    "Reduced context is not strictly equal training-data exposure to T2 or other candidate methods"
                )
    scores = pd.concat(output_rows, ignore_index=True)
    if not quick and cases == [(0.8, 0.05)] and split_seed == 617:
        _compare_t2_baseline(scores)
        if config.get("include_saved_t2_pmf", True):
            scores = _merge_reference(scores, completion=0.8, noise=0.05, split_seed=617)
    scores.to_csv(output_dir / "per_target_metrics.csv", index=False)
    overall = scores.groupby(["method", "protocol", "completion", "noise", "split_seed"]).apply(
        lambda sub: pd.Series({
            "mean_numeric_skill_ex_placebo": sub.loc[(sub.variable_type == "numeric") & (sub.variable != "random_placebo"), "numeric_skill"].mean(),
            "placebo_skill": sub.loc[sub.variable == "random_placebo", "numeric_skill"].mean(),
            "model_id_macro_f1": sub.loc[sub.variable == "model_id", "macro_f1"].mean(),
            "fit_seconds": sub.fit_seconds.iloc[0],
            "transform_seconds": sub.transform_seconds.iloc[0],
        }), include_groups=False
    ).reset_index()
    overall.to_csv(output_dir / "method_summary.csv", index=False)
    run_metadata["evaluation_rows"] = len(scores)
    run_metadata["notes"] = [
        "Per-target results are primary; average numerical skill is secondary.",
        "All methods were evaluated on the same originally missing cells.",
        "No confidence intervals or independent generator replications are claimed.",
        "Reference predictive_missforest rows were imported from T2 and were not rerun.",
        "A TabPFN run is not directly comparable if its fitting context is subsampled.",
    ]
    (output_dir / "run_manifest.json").write_text(json.dumps(run_metadata, indent=2, default=str))
    print("\n" + overall.to_string(index=False), flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="median_mode,chained_ridge,lightgbm_chained")
    parser.add_argument("--quick", action="store_true", help="Reduced spatial tiles, one chained pass and up to 25 boosting trees. NOT report-ready")
    parser.add_argument("--cases", default="0.8:0.05", help="Comma-separated completion:noise pairs from the supplied nine scenarios")
    parser.add_argument("--split-seed", type=int, default=617)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "model_comparison" / "config.yaml")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory. Existing directories are never overwritten")
    parser.add_argument("--no-cell-export", action="store_true")
    args = parser.parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    cases = [tuple(map(float, x.strip().split(":"))) for x in args.cases.split(",")]
    supported = {(c, n) for c in (0.9, 0.8, 0.7) for n in (0.0, 0.05, 0.1)}
    if any(x not in supported for x in cases):
        parser.error("Choose only the nine supplied completion:noise combinations")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or EXPERIMENT_DIR / "outputs" / f"model_comparison_{'pilot' if args.quick else 'full'}_{stamp}"
    run(
        methods=methods, quick=args.quick, cases=cases, output_dir=output,
        config_path=args.config, split_seed=args.split_seed,
        export_cells=not args.no_cell_export,
    )
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
