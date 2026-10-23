"""Prepare leakage-controlled overhead-conductor inputs for the pilot.

Only ``observed_inputs`` and the frozen preprocessor belong in model code.
``evaluator_only`` contains answers and simulated-error labels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .generate_synthetic_data import REQUIRED_OUTPUT_COLUMNS
from .noise import CATEGORICAL_VALUES, NUMERIC_BOUNDS


MODEL_FIELDS = (
    "installation_year",
    "distance_to_sea_km",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "material",
    "covering",
)
NUMERIC_FIELDS = ("installation_year", "distance_to_sea_km")
CATEGORICAL_FIELDS = tuple(field for field in MODEL_FIELDS if field not in NUMERIC_FIELDS)
ALLOWED_CATEGORIES = {
    **CATEGORICAL_VALUES,
    "material": {"Steel", "Copper", "Aluminium"},
}
PARTITIONS = ("fitting", "tuning", "calibration", "test")


def load_pilot_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the small pilot contract, not a model CLI config."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Pilot configuration must be a YAML mapping")
    scope = config["scope"]
    fields = tuple(scope["numeric_model_fields"] + scope["categorical_model_fields"])
    if set(fields) != set(MODEL_FIELDS) or len(fields) != len(MODEL_FIELDS):
        raise ValueError("Pilot configuration must contain exactly the seven model fields")
    if set(scope["primary_targets"]) != {"material", "installation_year"}:
        raise ValueError("Pilot targets must be material and installation_year")
    if int(scope["snapshot_year"]) != 2014:
        raise ValueError("The current generator uses the 2014 snapshot")
    parts = config["partitions"]
    if any(float(parts[name]) <= 0 for name in PARTITIONS) or not np.isclose(
        sum(float(parts[name]) for name in PARTITIONS), 1.0
    ):
        raise ValueError("Partition shares must be positive and sum to one")
    for name in ("fitting", "tuning", "test"):
        rate = float(config["observations"][name]["per_field_missingness"])
        if not 0 < rate < 1:
            raise ValueError(f"{name} missingness must be between zero and one")
    noise = config["observations"]["test"]
    numeric = noise["numeric_error"]
    categorical = noise["categorical_error"]
    if numeric["field"] != "distance_to_sea_km":
        raise ValueError("Pilot numeric error is defined for sea distance only")
    if set(categorical["fields"]) != {
        "voltage_category", "location_group", "corrosion_zone"
    }:
        raise ValueError("Pilot categorical error fields do not match the protocol")
    for value in (numeric["affected_fraction"], categorical["affected_fraction_per_field"]):
        if not 0 <= float(value) <= 1:
            raise ValueError("Noise prevalence must be between zero and one")
    if not 0 <= float(numeric["relative_uniform_bound"]) <= 1:
        raise ValueError("Numeric noise magnitude must be between zero and one")
    return config


def audit_reference(reference: pd.DataFrame, snapshot_year: int = 2014) -> dict[str, Any]:
    """Reject invalid reference records without dropping or repairing rows."""
    errors: list[str] = []
    missing_columns = sorted(set(REQUIRED_OUTPUT_COLUMNS) - set(reference.columns))
    if missing_columns:
        raise ValueError(f"Reference is missing columns: {missing_columns}")
    if reference.empty:
        errors.append("reference is empty")
    if reference[REQUIRED_OUTPUT_COLUMNS].isna().any().any():
        errors.append("reference contains null cells")
    for field in ("segment_id", "feeder_id"):
        if reference[field].astype(str).str.strip().eq("").any():
            errors.append(f"{field} contains blank IDs")
    if not reference["segment_id"].is_unique:
        errors.append("segment_id is not unique")
    for field, (low, high) in NUMERIC_BOUNDS.items():
        numeric = pd.to_numeric(reference[field], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all() or not numeric.between(low, high).all():
            errors.append(f"{field} has nonnumeric or out-of-range values in declared units")
    for field in ("installation_year", "asset_age"):
        numeric = pd.to_numeric(reference[field], errors="coerce")
        if not numeric.dropna().mod(1).eq(0).all():
            errors.append(f"{field} must contain whole years")
    if not (
        pd.to_numeric(reference["asset_age"], errors="coerce")
        == snapshot_year - pd.to_numeric(reference["installation_year"], errors="coerce")
    ).all():
        errors.append("asset_age does not equal snapshot_year - installation_year")
    for field, allowed in ALLOWED_CATEGORIES.items():
        unexpected = set(reference[field].dropna().astype(str)) - allowed
        if unexpected:
            errors.append(f"{field} has unsupported labels: {sorted(unexpected)}")
    if reference["conductor_type"].astype(str).str.strip().eq("").any():
        errors.append("conductor_type contains blank values")
    if not reference["data_status"].eq("Synthetic").all():
        errors.append("only synthetic references are permitted in this pilot")
    if not reference["record_version"].eq("Complete reference").all():
        errors.append("record_version must identify a complete reference")
    if errors:
        raise ValueError("Reference schema audit failed: " + "; ".join(errors))
    return {
        "passed": True,
        "row_count": int(len(reference)),
        "unique_segment_ids": int(reference["segment_id"].nunique()),
        "snapshot_year": snapshot_year,
        "numeric_units": {
            "installation_year": "calendar year",
            "asset_age": "years at 2014 snapshot",
            "distance_to_sea_km": "km",
            "length_km": "km",
        },
        "checked_categories": {key: sorted(value) for key, value in ALLOWED_CATEGORIES.items()},
        "rare_valid_records_removed": 0,
        "automatic_corrections": 0,
    }


def freeze_partitions(reference: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Assign each synthetic segment to one fixed row-level partition."""
    if len(reference) < 10:
        raise ValueError("At least ten rows are needed for four nonempty partitions")
    shares = [float(config["partitions"][name]) for name in PARTITIONS]
    counts = [int(round(len(reference) * share)) for share in shares[:3]]
    counts.append(len(reference) - sum(counts))
    if min(counts) < 1:
        raise ValueError("Partition settings produce an empty partition")
    assignments = np.empty(len(reference), dtype=object)
    indices = np.random.default_rng(int(config["partitions"]["split_seed"])).permutation(len(reference))
    start = 0
    for name, count in zip(PARTITIONS, counts):
        assignments[indices[start : start + count]] = name
        start += count
    return pd.DataFrame({"segment_id": reference["segment_id"].to_numpy(), "partition": assignments})


def select_masks(rows: pd.DataFrame, rate: float, seed: int) -> dict[str, set[str]]:
    """Independently mask the same fixed fraction of originally known cells per field."""
    selected: dict[str, set[str]] = {}
    for index, field in enumerate(MODEL_FIELDS):
        eligible = rows.loc[rows[field].notna(), "segment_id"].astype(str).to_numpy()
        count = int(round(len(eligible) * rate))
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        selected[field] = set(rng.choice(eligible, size=count, replace=False).tolist())
    return selected


def apply_pilot_noise(
    rows: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Make the N11 pre-mask view; prevalence and magnitude are independent."""
    settings = config["observations"]["test"]
    numeric = settings["numeric_error"]
    categorical = settings["categorical_error"]
    base_seed = int(settings["noise_seed"])
    noisy = rows.copy(deep=True)
    selected: set[tuple[str, str]] = set()
    field = numeric["field"]
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, 0]))
    count = int(round(len(rows) * float(numeric["affected_fraction"])))
    positions = rng.choice(len(rows), size=count, replace=False)
    magnitude = float(numeric["relative_uniform_bound"])
    multipliers = 1 + rng.uniform(-magnitude, magnitude, size=count)
    lower, upper = [float(x) for x in numeric["allowed_range_km"]]
    noisy.iloc[positions, noisy.columns.get_loc(field)] = np.clip(
        noisy.iloc[positions][field].to_numpy(dtype=float) * multipliers, lower, upper
    )
    selected.update((str(rows.iloc[pos]["segment_id"]), field) for pos in positions)
    for index, field in enumerate(categorical["fields"], start=1):
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, index]))
        count = int(round(len(rows) * float(categorical["affected_fraction_per_field"])))
        positions = rng.choice(len(rows), size=count, replace=False)
        allowed = sorted(ALLOWED_CATEGORIES[field])
        column = noisy.columns.get_loc(field)
        for pos in positions:
            current = str(noisy.iat[pos, column])
            alternatives = [label for label in allowed if label != current]
            noisy.iat[pos, column] = rng.choice(alternatives)
            selected.add((str(rows.iloc[pos]["segment_id"]), field))
    return noisy, selected


def apply_masks(rows: pd.DataFrame, masks: Mapping[str, set[str]]) -> pd.DataFrame:
    """Return the only seven-field view that model code may consume."""
    observed = rows.loc[:, ["segment_id", *MODEL_FIELDS]].copy(deep=True)
    for field in MODEL_FIELDS:
        observed.loc[observed["segment_id"].isin(masks[field]), field] = np.nan
    return observed


def change_ledger(
    truth: pd.DataFrame,
    pre_mask: pd.DataFrame,
    observed: pd.DataFrame,
    masks: Mapping[str, set[str]],
    noise_selected: set[tuple[str, str]],
    partition: str,
    condition: str,
) -> pd.DataFrame:
    """Store clean answers only in the evaluator's change record."""
    records: list[dict[str, Any]] = []
    truth_by_id = truth.set_index("segment_id")
    pre_by_id = pre_mask.set_index("segment_id")
    obs_by_id = observed.set_index("segment_id")
    for field in MODEL_FIELDS:
        affected_ids = masks[field] | {segment for segment, changed_field in noise_selected if changed_field == field}
        for segment in sorted(affected_ids):
            clean = truth_by_id.at[segment, field]
            before_mask = pre_by_id.at[segment, field]
            after_mask = obs_by_id.at[segment, field]
            records.append({
                "segment_id": segment,
                "partition": partition,
                "condition": condition,
                "field": field,
                "clean_value": clean,
                "pre_mask_value": before_mask,
                "observed_value": None if pd.isna(after_mask) else after_mask,
                "noise_selected": (segment, field) in noise_selected,
                "corruption_actual": bool(clean != before_mask),
                "masked": segment in masks[field],
            })
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class FrozenPreprocessor:
    """Fitting-observation statistics; no truth or held-out rows are accepted at fit."""

    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    categorical_modes: dict[str, str]
    categorical_vocabularies: dict[str, list[str]]

    @classmethod
    def fit(cls, fitting_observed: pd.DataFrame) -> "FrozenPreprocessor":
        _validate_model_view(fitting_observed)
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        modes: dict[str, str] = {}
        vocabularies: dict[str, list[str]] = {}
        for field in NUMERIC_FIELDS:
            values = pd.to_numeric(fitting_observed[field], errors="coerce").dropna()
            if values.empty:
                raise ValueError(f"No observed fitting values for {field}")
            medians[field] = float(values.median())
            means[field] = float(values.mean())
            deviation = float(values.std(ddof=0))
            stds[field] = deviation if deviation > 0 else 1.0
        for field in CATEGORICAL_FIELDS:
            values = fitting_observed[field].dropna().astype(str)
            if values.empty:
                raise ValueError(f"No observed fitting values for {field}")
            counts = values.value_counts()
            modes[field] = sorted(counts[counts == counts.max()].index)[0]
            vocabularies[field] = sorted(values.unique().tolist())
        return cls(medians, means, stds, modes, vocabularies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "categorical_modes": self.categorical_modes,
            "categorical_vocabularies": self.categorical_vocabularies,
            "fit_scope": "fitting_observed_rows_only",
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FrozenPreprocessor":
        return cls(*(
            dict(values[name]) for name in (
                "numeric_medians", "numeric_means", "numeric_stds",
                "categorical_modes", "categorical_vocabularies",
            )
        ))

    def baseline_fill(self, observed: pd.DataFrame) -> pd.DataFrame:
        _validate_model_view(observed)
        result = observed.copy(deep=True)
        for field in NUMERIC_FIELDS:
            result[field] = result[field].fillna(self.numeric_medians[field])
        for field in CATEGORICAL_FIELDS:
            result[field] = result[field].fillna(self.categorical_modes[field])
        return result

    def forest_predictors(self, observed: pd.DataFrame, target: str) -> pd.DataFrame:
        _validate_model_view(observed)
        if target not in MODEL_FIELDS:
            raise ValueError(f"Unknown target: {target}")
        columns: dict[str, Any] = {}
        for field in MODEL_FIELDS:
            if field == target:
                continue
            if field in NUMERIC_FIELDS:
                columns[field] = observed[field].fillna(self.numeric_means[field]).to_numpy(dtype=float)
            else:
                values = observed[field]
                for category in self.categorical_vocabularies[field]:
                    columns[f"{field}={category}"] = values.eq(category).to_numpy(dtype=int)
        return pd.DataFrame(columns, index=observed.index)

    def neural_inputs(self, observed: pd.DataFrame) -> pd.DataFrame:
        _validate_model_view(observed)
        columns: dict[str, Any] = {}
        for field in MODEL_FIELDS:
            values = observed[field]
            if field in NUMERIC_FIELDS:
                columns[field] = (
                    (values.fillna(self.numeric_means[field]) - self.numeric_means[field])
                    / self.numeric_stds[field]
                ).to_numpy(dtype=float)
            else:
                for category in self.categorical_vocabularies[field]:
                    columns[f"{field}={category}"] = values.eq(category).to_numpy(dtype=int)
            columns[f"{field}__available"] = values.notna().to_numpy(dtype=int)
        return pd.DataFrame(columns, index=observed.index)

    def unknown_categories(self, observed: pd.DataFrame) -> pd.DataFrame:
        _validate_model_view(observed)
        records: list[dict[str, str]] = []
        for field in CATEGORICAL_FIELDS:
            unknown = observed[field].notna() & ~observed[field].isin(self.categorical_vocabularies[field])
            for segment, value in zip(observed.loc[unknown, "segment_id"], observed.loc[unknown, field]):
                records.append({"segment_id": str(segment), "field": field, "observed_value": str(value)})
        return pd.DataFrame(records, columns=["segment_id", "field", "observed_value"])


def _validate_model_view(observed: pd.DataFrame) -> None:
    expected = {"segment_id", *MODEL_FIELDS}
    if set(observed.columns) != expected:
        raise ValueError(f"Model view must have only segment_id and seven model fields; found {sorted(observed.columns)}")
    if observed["segment_id"].isna().any() or not observed["segment_id"].is_unique:
        raise ValueError("Model view requires unique non-null segment IDs")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_pilot(
    reference: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: str | Path,
    source: str,
) -> Path:
    """Materialize one frozen pilot's input artefacts without fitting a model."""
    audit = audit_reference(reference, snapshot_year=int(config["scope"]["snapshot_year"]))
    expected = int(config["pilot_population"]["expected_rows"])
    if len(reference) != expected:
        raise ValueError(f"Expected {expected} pilot rows, got {len(reference)}")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty pilot output: {output}")
    evaluator = output / "evaluator_only"
    model_inputs = output / "observed_inputs"
    audit_dir = output / "audit"
    for folder in (evaluator, model_inputs, audit_dir):
        folder.mkdir(parents=True, exist_ok=True)
    truth_path = evaluator / "clean_truth.csv"
    reference.to_csv(truth_path, index=False)
    _write_json(audit_dir / "schema_audit.json", audit)
    partitions = freeze_partitions(reference, config)
    partitions.to_csv(audit_dir / "partitions.csv", index=False)
    reference_with_split = reference.merge(partitions, on="segment_id", validate="one_to_one")
    reference_with_split[["segment_id", "feeder_id", "data_status", "record_version", "partition"]].to_csv(
        audit_dir / "tracking.csv", index=False
    )

    mask_records: list[dict[str, str]] = []
    ledgers: list[pd.DataFrame] = []
    views: dict[str, pd.DataFrame] = {}
    test_masks: dict[str, set[str]] | None = None
    for partition in PARTITIONS:
        rows = reference_with_split.loc[reference_with_split["partition"] == partition, reference.columns]
        observation = config["observations"][partition]
        masks = select_masks(rows, float(observation["per_field_missingness"]), int(observation["mask_seed"]))
        if partition == "test":
            test_masks = masks
        for field, segments in masks.items():
            mask_records.extend({"segment_id": segment, "partition": partition, "field": field} for segment in sorted(segments))
        conditions = ("N00", "N11") if partition == "test" else ("N00",)
        for condition in conditions:
            pre_mask, selected = (
                apply_pilot_noise(rows, config) if condition == "N11" else (rows.copy(deep=True), set())
            )
            observed = apply_masks(pre_mask, masks)
            name = f"{partition}_{condition}"
            observed.to_csv(model_inputs / f"{name}.csv", index=False)
            views[name] = observed
            ledgers.append(change_ledger(rows, pre_mask, observed, masks, selected, partition, condition))
            if not observed["segment_id"].equals(rows["segment_id"]):
                raise AssertionError("Observed view changed segment IDs")
    assert test_masks is not None
    mask_manifest = pd.DataFrame(mask_records, columns=["segment_id", "partition", "field"])
    mask_manifest.to_csv(audit_dir / "mask_manifest.csv", index=False)
    ledger = pd.concat(ledgers, ignore_index=True)
    ledger.to_csv(evaluator / "cell_change_ledger.csv", index=False)
    preprocessor = FrozenPreprocessor.fit(views["fitting_N00"])
    _write_json(model_inputs / "frozen_preprocessor.json", preprocessor.to_dict())
    unknown = []
    for name, view in views.items():
        flagged = preprocessor.unknown_categories(view)
        flagged.insert(0, "view", name)
        unknown.append(flagged)
    pd.concat(unknown, ignore_index=True).to_csv(audit_dir / "unknown_categories.csv", index=False)
    test_n00 = views["test_N00"].set_index("segment_id")
    test_n11 = views["test_N11"].set_index("segment_id")
    for field in MODEL_FIELDS:
        if not test_n00[field].isna().equals(test_n11[field].isna()):
            raise AssertionError(f"Test masks differ between N00 and N11 for {field}")
    _write_json(audit_dir / "run_manifest.json", {
        "protocol_version": config["protocol_version"],
        "protocol_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "source": source,
        "truth_sha256": _sha256(truth_path),
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "partition_counts": partitions["partition"].value_counts().to_dict(),
        "mask_counts": mask_manifest.groupby(["partition", "field"]).size().rename_axis(["partition", "field"]).reset_index(name="count").to_dict("records"),
        "actual_noise_changes_on_test": ledger.loc[ledger["condition"].eq("N11") & ledger["corruption_actual"]].groupby("field").size().to_dict(),
        "seeds": {
            "generation": config["pilot_population"]["generation_seed"],
            "split": config["partitions"]["split_seed"],
            "fitting_mask": config["observations"]["fitting"]["mask_seed"],
            "tuning_mask": config["observations"]["tuning"]["mask_seed"],
            "calibration_mask": config["observations"]["calibration"]["mask_seed"],
            "test_mask": config["observations"]["test"]["mask_seed"],
            "test_noise": config["observations"]["test"]["noise_seed"],
        },
        "model_fields": list(MODEL_FIELDS),
        "methods_fitted": [],
        "notes": "Synthetic input preparation only; no imputation, calibration or PoF results",
    })
    return output
