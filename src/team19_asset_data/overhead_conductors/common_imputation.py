"""Apply frozen imputation methods to the same prepared overhead-conductor views.

This stage deliberately never reads ``evaluator_only``. Reconstruction and
population metrics belong to the following evaluation ticket.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn

from team19_asset_data.imputation.postprocess import apply_constraints
from team19_asset_data.imputation.validation import validate_imputation_result

from .input_preparation import (
    ALLOWED_CATEGORIES,
    CATEGORICAL_FIELDS,
    MODEL_FIELDS,
    NUMERIC_FIELDS,
    FrozenPreprocessor,
)


CONSTRAINTS = {
    "installation_year": {"data_type": "integer", "min": 1930, "max": 2014},
    "distance_to_sea_km": {"data_type": "float", "min": 0.0, "max": 350.0},
}
TEST_VIEWS = ("test_N00", "test_N11")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_view(path: Path) -> pd.DataFrame:
    view = pd.read_csv(path, dtype={"segment_id": str})
    expected = ["segment_id", *MODEL_FIELDS]
    if list(view.columns) != expected:
        raise ValueError(f"{path.name} must contain only segment_id and the seven model fields")
    if view.empty or view["segment_id"].isna().any() or not view["segment_id"].is_unique:
        raise ValueError(f"{path.name} has missing or duplicate segment IDs")
    for field in NUMERIC_FIELDS:
        values = pd.to_numeric(view[field], errors="coerce")
        if values.isna().sum() != view[field].isna().sum():
            raise ValueError(f"{path.name} has nonnumeric observed {field} values")
    for field in CATEGORICAL_FIELDS:
        unexpected = set(view[field].dropna().astype(str)) - ALLOWED_CATEGORIES[field]
        if unexpected:
            raise ValueError(f"{path.name} has invalid {field} labels: {sorted(unexpected)}")
    return view


def load_prepared_inputs(pilot_dir: str | Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], FrozenPreprocessor, dict[str, Any]]:
    """Verify shared IDs and masks without opening evaluator-only answers."""
    root = Path(pilot_dir).expanduser().resolve()
    fitting = _load_view(root / "observed_inputs/fitting_N00.csv")
    tests = {name: _load_view(root / f"observed_inputs/{name}.csv") for name in TEST_VIEWS}
    partitions = pd.read_csv(root / "audit/partitions.csv", dtype={"segment_id": str})
    masks = pd.read_csv(root / "audit/mask_manifest.csv", dtype={"segment_id": str})
    manifest = json.loads((root / "audit/run_manifest.json").read_text(encoding="utf-8"))
    if partitions["segment_id"].isna().any() or not partitions["segment_id"].is_unique:
        raise ValueError("Partition manifest must have unique non-null IDs")
    if set(partitions["partition"]) != {"fitting", "tuning", "calibration", "test"}:
        raise ValueError("Partition manifest must contain all four partitions")
    expected_fitting = set(partitions.loc[partitions["partition"].eq("fitting"), "segment_id"])
    expected_test = set(partitions.loc[partitions["partition"].eq("test"), "segment_id"])
    if set(fitting["segment_id"]) != expected_fitting or set(tests["test_N00"]["segment_id"]) != expected_test:
        raise ValueError("Prepared views do not match the frozen partition IDs")
    if not tests["test_N00"]["segment_id"].equals(tests["test_N11"]["segment_id"]):
        raise ValueError("N00 and N11 test IDs or row order differ")
    if masks.duplicated(["segment_id", "partition", "field"]).any():
        raise ValueError("Mask manifest has duplicate cell keys")
    for name, view in (("fitting", fitting), ("test", tests["test_N00"]), ("test", tests["test_N11"])):
        for field in MODEL_FIELDS:
            expected_mask = set(masks.loc[masks["partition"].eq(name) & masks["field"].eq(field), "segment_id"])
            actual_mask = set(view.loc[view[field].isna(), "segment_id"])
            if actual_mask != expected_mask:
                raise ValueError(f"{name} mask manifest disagrees with observed {field}")
    saved = json.loads((root / "observed_inputs/frozen_preprocessor.json").read_text(encoding="utf-8"))
    preprocessor = FrozenPreprocessor.from_dict(saved)
    if preprocessor.to_dict() != FrozenPreprocessor.fit(fitting).to_dict():
        raise ValueError("Saved preprocessing statistics do not match observed fitting rows")
    return fitting, tests, preprocessor, manifest


def _validate_completed(observed: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    constrained = apply_constraints(observed, candidate, CONSTRAINTS)
    validate_imputation_result(observed, constrained, NUMERIC_FIELDS, CATEGORICAL_FIELDS)
    for field in CATEGORICAL_FIELDS:
        unexpected = set(constrained[field].dropna().astype(str)) - ALLOWED_CATEGORIES[field]
        if unexpected:
            raise ValueError(f"Imputed {field} has invalid labels: {sorted(unexpected)}")
    constrained["asset_age"] = 2014 - constrained["installation_year"].astype(int)
    if not constrained["asset_age"].between(0, 84).all():
        raise ValueError("Derived asset_age is outside the synthetic snapshot range")
    return constrained


def _candidate_rows(
    observed: pd.DataFrame,
    completed: pd.DataFrame,
    method: str,
    condition: str,
    diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    detail = diagnostics if diagnostics is not None else pd.DataFrame()
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not detail.empty:
        if not {"segment_id", "column"}.issubset(detail.columns):
            raise ValueError("Forest diagnostics lack segment ID or target column")
        if detail.duplicated(["segment_id", "column"]).any():
            raise ValueError("Forest diagnostics contain duplicate cell keys")
        lookup = {
            (str(row["segment_id"]), str(row["column"])): row.to_dict()
            for _, row in detail.iterrows()
        }
    if method == "predictive_missforest":
        expected = {
            (str(observed.iloc[position]["segment_id"]), field)
            for field in MODEL_FIELDS
            for position in np.flatnonzero(observed[field].isna().to_numpy())
        }
        if set(lookup) != expected:
            raise ValueError("Forest diagnostics do not match the frozen hidden cells")
    for field in MODEL_FIELDS:
        for position in np.flatnonzero(observed[field].isna().to_numpy()):
            segment = str(observed.iloc[position]["segment_id"])
            item = lookup.get((segment, field), {})
            source = str(item.get("prediction_source", "training_median_or_mode"))
            records.append({
                "segment_id": segment,
                "partition": "test",
                "condition": condition.removeprefix("test_"),
                "field": field,
                "method": method,
                "candidate_value": completed.iloc[position][field],
                "prediction_source": source,
                "fallback": source == "training_statistic_fallback",
                "quality_flag": "uncalibrated_diagnostic" if method == "predictive_missforest" else "no_probability",
                "max_vote_share": item.get("max_proba", np.nan),
                "tree_spread": item.get("tree_std", np.nan),
            })
    return pd.DataFrame(records)


def _forest_builder(estimators: int, passes: int, seed: int) -> Any:
    try:
        from team19_asset_data.imputation.predictive_missforest import (
            PredictiveMissForest,
            PredictiveMissForestConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PredictiveMissForest is not in this checkout. Merge/review Zisan's "
            "predictive-forest PR before running the shared pilot."
        ) from exc
    return PredictiveMissForest(
        numeric_columns=NUMERIC_FIELDS,
        categorical_columns=CATEGORICAL_FIELDS,
        config=PredictiveMissForestConfig(
            n_estimators=estimators, n_passes=passes, random_state=seed, n_jobs=1
        ),
        constraints=CONSTRAINTS,
    )


def run_common_imputation(
    pilot_dir: str | Path,
    output_dir: str | Path,
    *,
    estimators: int = 24,
    passes: int = 2,
    seed: int = 219,
    forest_builder: Callable[[], Any] | None = None,
) -> Path:
    """Fit once on fitting rows and replay both methods on paired test views."""
    if estimators < 1 or passes < 1:
        raise ValueError("estimators and passes must be positive")
    pilot_root = Path(pilot_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty imputation output: {output}")
    fitting, tests, baseline, input_manifest = load_prepared_inputs(pilot_root)
    builder = forest_builder or (lambda: _forest_builder(estimators, passes, seed))
    forest = builder()
    forest.fit(fitting.copy(deep=True))
    output.mkdir(parents=True, exist_ok=True)
    model_path = forest.save(output / "predictive_forest.joblib")
    reloaded = type(forest).load(model_path)
    candidate_tables: list[pd.DataFrame] = []
    diagnostic_tables: list[pd.DataFrame] = []
    for name, observed in tests.items():
        original = observed.copy(deep=True)
        baseline_completed = _validate_completed(original, baseline.baseline_fill(original))
        detailed = reloaded.transform_detailed(original.copy(deep=True), with_scores=True)
        forest_completed = _validate_completed(original, detailed.imputed)
        replay = reloaded.transform(original.copy(deep=True))
        pd.testing.assert_frame_equal(detailed.imputed, replay, check_dtype=False)
        pd.testing.assert_frame_equal(observed, original)
        score = detailed.scores.copy() if detailed.scores is not None else pd.DataFrame()
        if not score.empty:
            positions = score["position"].to_numpy(dtype=int)
            if (positions < 0).any() or (positions >= len(original)).any():
                raise ValueError("Forest diagnostics contain invalid row positions")
            score.insert(0, "segment_id", original.iloc[positions]["segment_id"].to_numpy())
            score.insert(1, "condition", name.removeprefix("test_"))
            diagnostic_tables.append(score)
        baseline_completed.to_csv(output / f"baseline_{name}.csv", index=False)
        forest_completed.to_csv(output / f"predictive_forest_{name}.csv", index=False)
        candidate_tables.append(_candidate_rows(original, baseline_completed, "median_mode", name))
        candidate_tables.append(_candidate_rows(original, forest_completed, "predictive_missforest", name, score))
    candidates = pd.concat(candidate_tables, ignore_index=True)
    candidates.to_csv(output / "candidate_cells.csv", index=False)
    if diagnostic_tables:
        pd.concat(diagnostic_tables, ignore_index=True).to_csv(output / "forest_diagnostics.csv", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "input_protocol_sha256": input_manifest["protocol_sha256"],
        "input_truth_sha256": input_manifest["truth_sha256"],
        "fitting_rows": len(fitting),
        "test_rows": len(tests["test_N00"]),
        "candidate_counts": candidates.groupby(["condition", "method", "field"]).size().rename_axis(["condition", "method", "field"]).reset_index(name="count").to_dict("records"),
        "methods": ["median_mode", "predictive_missforest"],
        "forest_parameters": {"n_estimators": estimators, "n_passes": passes, "random_state": seed, "n_jobs": 1},
        "observed_input_sha256": {
            name: _sha256(pilot_root / "observed_inputs" / f"{name}.csv")
            for name in ("fitting_N00", *TEST_VIEWS)
        },
        "source_sha256": {
            "common_imputation": _sha256(Path(__file__)),
            "input_preparation": _sha256(Path(inspect.getfile(FrozenPreprocessor))),
            "forest_implementation": _sha256(Path(inspect.getfile(type(forest)))),
        },
        "library_versions": {
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__
        },
        "fitted_model_sha256": _sha256(Path(model_path)),
        "confidence_status": "uncalibrated_diagnostics_only",
        "evaluation_performed": False,
        "notes": "No clean answers were loaded; metrics and accepted-fill decisions belong to the evaluation ticket",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
