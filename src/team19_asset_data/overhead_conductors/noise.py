"""Controlled predictor-noise scenarios for overhead-conductor experiments.

The complete synthetic reference is never modified. Each scenario creates a
separate noisy observed view, before the common MF01 missingness mask is applied
by the existing masking pipeline.

Random numeric noise: ``noise_level`` controls BOTH (1) the share of source
cells selected and (2) the maximum absolute relative perturbation. Thus 0.30
means approximately 30% of eligible entries receive an error in [-30%, +30%],
subject to column bounds. At 100% this is an extreme stress test.

Random categorical noise: ``noise_level`` is the fraction of eligible entries
replaced with another valid category.

For the same seed, strategy family and column, selection order and underlying
random draws are stable across levels. Consequently 30% selection is a subset
of 50% selection, which is a subset of 100% selection. To obtain repeated
independent trials, vary the seed at the scenario-family level.

Important: material is a PROTECTED evaluation target, not corrupted directly.
conductor_type is also protected (it would leak the material target). The
separate ``installation_year_error`` strategy preserves asset-age consistency.
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


REQUIRED_COLUMNS = [
    "segment_id",
    "feeder_id",
    "installation_year",
    "asset_age",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "distance_to_sea_km",
    "material",
    # "conductor_type",
    "covering",
    "length_km",
    "data_status",
    "record_version",
]
PROTECTED_COLUMNS = {
    "segment_id",
    "feeder_id",
    "material",
    "conductor_type",
    "data_status",
    "record_version",
}
NUMERIC_BOUNDS = {
    "installation_year": (1930, 2014),
    "asset_age": (0, 84),
    "distance_to_sea_km": (0.0, 350.0),
    "length_km": (0.01, 10.0),
}
CATEGORICAL_VALUES = {
    "voltage_category": {"66 kV", "33 kV", "19 kV SWER", "11/7.6 kV", "LV"},
    "location_group": {"Rural", "Regional", "Urban"},
    "corrosion_zone": {"Low", "Severe", "Very Severe"},
    "covering": {"Bare", "Covered"},
}


@dataclass(frozen=True)
class NoiseScenario:
    """Validated configuration for one noise experiment."""

    scenario_id: str
    name: str
    strategy: str
    columns: tuple[str, ...]
    noise_level: float
    seed: int
    year_step: int = 1
    group_column: str | None = None
    group_value: str | None = None
    group_multiplier: Mapping[str, float] | None = None

    @classmethod
    def from_mapping(cls, scenario_id: str, raw: Mapping[str, Any]) -> "NoiseScenario":
        columns = tuple(str(value) for value in raw.get("columns", []))
        level = float(raw.get("noise_level", 0.0))
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"{scenario_id}.noise_level must be between 0 and 1")
        strategy = str(raw.get("strategy", "")).strip()
        if not strategy:
            raise ValueError(f"{scenario_id}.strategy is required")
        year_step = int(raw.get("year_step", 1))
        if year_step not in {1, 2}:
            raise ValueError(f"{scenario_id}.year_step must be 1 or 2")
        multiplier = raw.get("group_multiplier")
        if multiplier is not None:
            multiplier = {str(key): float(value) for key, value in multiplier.items()}
        return cls(
            scenario_id=scenario_id,
            name=str(raw.get("name", scenario_id)),
            strategy=strategy,
            columns=columns,
            noise_level=level,
            seed=int(raw.get("seed", 220)),
            year_step=year_step,
            group_column=raw.get("group_column"),
            group_value=raw.get("group_value"),
            group_multiplier=multiplier,
        )


def load_scenarios(config_path: str | Path) -> tuple[dict[str, Any], list[NoiseScenario]]:
    """Load and validate the YAML noise configuration."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict) or not isinstance(raw.get("scenarios"), dict):
        raise ValueError("Noise configuration must contain a scenarios mapping")
    settings = dict(raw.get("settings", {}))
    scenarios = [NoiseScenario.from_mapping(key, value) for key, value in raw["scenarios"].items()]
    if not scenarios:
        raise ValueError("Noise configuration must contain at least one scenario")
    return settings, scenarios


def validate_input(df: pd.DataFrame) -> None:
    """Validate the clean complete reference before applying noise."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Complete reference is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Complete reference must contain at least one row")
    if not df["segment_id"].is_unique:
        raise ValueError("segment_id must be unique")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Complete reference must not contain missing values")
    if set(df["material"].astype(str).unique()) - {"Steel", "Copper", "Aluminium"}:
        raise ValueError("Complete reference contains unsupported material classes")


def _sample_indices(rng: np.random.Generator, index: pd.Index, rate: float) -> pd.Index:
    count = int(round(len(index) * rate))
    if rate > 0 and count == 0 and len(index) > 0:
        count = 1
    if count == 0:
        return index[:0]
    return pd.Index(rng.choice(index.to_numpy(), size=min(count, len(index)), replace=False))


def _stable_column_rng(seed: int, family: str, column: str) -> np.random.Generator:
    """Stable independent stream per column, unaffected by other columns/rates."""
    tag = f"{int(seed)}|{family}|{column}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(tag).digest()[:8], "big")
    return np.random.default_rng(derived_seed)


def _nested_sample_indices(
    rng: np.random.Generator, eligible: pd.Index, rate: float
) -> pd.Index:
    """Select the first n positions of a full seeded permutation."""
    count = int(round(len(eligible) * rate))
    if rate > 0 and count == 0 and len(eligible):
        count = 1
    permutation = rng.permutation(len(eligible))
    return eligible.take(permutation[:count])


def _clip_numeric(column: str, values: pd.Series) -> pd.Series:
    low, high = NUMERIC_BOUNDS[column]
    return values.clip(lower=low, upper=high)


def _replace_categories(
    values: pd.Series,
    indices: pd.Index,
    rng: np.random.Generator,
) -> None:
    allowed = sorted(CATEGORICAL_VALUES[values.name])
    for index in indices:
        current = str(values.loc[index])
        alternatives = [value for value in allowed if value != current]
        values.loc[index] = rng.choice(alternatives)


def _apply_random_numeric(
    observed: pd.DataFrame,
    scenario: NoiseScenario,
    changed: dict[str, int],
    columns: tuple[str, ...] | None = None,
) -> None:
    """Use stable nested samples and per-record relative-error draws."""
    for column in (scenario.columns if columns is None else columns):
        if column not in {"length_km", "distance_to_sea_km"}:
            raise ValueError(
                f"Unsupported random numeric noise column '{column}'. "
                "For installation_year use installation_year_error so asset_age stays consistent."
            )
        if column not in observed.columns:
            raise ValueError(f"Unknown numeric noise column: {column}")
        rng = _stable_column_rng(scenario.seed, "random_numeric", column)
        eligible = observed.index[observed[column].notna()]
        indices = _nested_sample_indices(rng, eligible, scenario.noise_level)
        # Generate one full-length set of signed coefficients independent of rate.
        # Draws for a selected row are then stable across scenarios with same seed.
        coefficients = pd.Series(
            rng.uniform(-1.0, 1.0, size=len(eligible)),
            index=eligible,
        )
        relative_error = coefficients.loc[indices].to_numpy() * scenario.noise_level
        observed.loc[indices, column] = _clip_numeric(
            column,
            observed.loc[indices, column].astype(float) * (1.0 + relative_error),
        )
        changed[column] = int(len(indices))


def _apply_systematic_numeric(
    observed: pd.DataFrame,
    scenario: NoiseScenario,
    changed: dict[str, int],
) -> None:
    if not scenario.group_column or scenario.group_value is None:
        raise ValueError("Systematic noise requires group_column and group_value")
    if scenario.group_column not in observed.columns:
        raise ValueError(f"Unknown systematic-noise group column: {scenario.group_column}")
    selected = observed.index[observed[scenario.group_column].astype(str) == str(scenario.group_value)]
    for column in scenario.columns:
        if column not in NUMERIC_BOUNDS or column in PROTECTED_COLUMNS:
            raise ValueError(f"Unsupported or protected systematic noise column: {column}")
        observed.loc[selected, column] = _clip_numeric(
            column,
            observed.loc[selected, column].astype(float) * (1.0 + scenario.noise_level),
        )
        changed[column] = int(len(selected))


def _apply_year_error(
    observed: pd.DataFrame,
    scenario: NoiseScenario,
    rng: np.random.Generator,
    changed: dict[str, int],
    snapshot_year: int,
) -> None:
    indices = _sample_indices(
        rng,
        observed.index[observed["installation_year"].notna()],
        scenario.noise_level,
    )
    signs = rng.choice(np.array([-1, 1]), size=len(indices))
    observed.loc[indices, "installation_year"] = _clip_numeric(
        "installation_year",
        observed.loc[indices, "installation_year"].astype(int)
        + signs * scenario.year_step,
    ).round().astype(int)
    # asset_age is linked to installation_year in the source design.  It must
    # be recomputed in the observed view instead of independently corrupted.
    observed["asset_age"] = snapshot_year - observed["installation_year"]
    changed["installation_year"] = int(len(indices))
    changed["asset_age"] = int(len(indices))


def _apply_categorical_replacement(
    observed: pd.DataFrame,
    scenario: NoiseScenario,
    changed: dict[str, int],
    columns: tuple[str, ...] | None = None,
) -> None:
    """Replace a nested, reproducible subset with alternative valid categories."""
    for column in (scenario.columns if columns is None else columns):
        if column not in CATEGORICAL_VALUES or column in PROTECTED_COLUMNS:
            raise ValueError(f"Unsupported or protected categorical noise column: {column}")
        if column not in observed.columns:
            raise ValueError(f"Unknown categorical noise column: {column}")
        rng = _stable_column_rng(scenario.seed, "categorical_replacement", column)
        eligible = observed.index[observed[column].notna()]
        indices = _nested_sample_indices(rng, eligible, scenario.noise_level)
        values = observed[column].copy()
        _replace_categories(values, indices, rng)
        observed.loc[:, column] = values
        changed[column] = int(len(indices))


def _apply_random_mixed(
    observed: pd.DataFrame,
    scenario: NoiseScenario,
    changed: dict[str, int],
) -> None:
    """Apply independent numeric and categorical corruption at the same rate."""
    numeric_columns = tuple(c for c in scenario.columns if c in {"length_km", "distance_to_sea_km"})
    categorical_columns = tuple(c for c in scenario.columns if c in CATEGORICAL_VALUES)
    unsupported = set(scenario.columns) - set(numeric_columns) - set(categorical_columns)
    if unsupported:
        raise ValueError(f"Unsupported mixed-noise columns: {sorted(unsupported)}")
    if not numeric_columns or not categorical_columns:
        raise ValueError("random_mixed needs at least one numeric and one categorical column")
    _apply_random_numeric(observed, scenario, changed, numeric_columns)
    _apply_categorical_replacement(observed, scenario, changed, categorical_columns)


def _apply_group_numeric(
    observed: pd.DataFrame,
    scenario: NoiseScenario,
    rng: np.random.Generator,
    changed: dict[str, int],
) -> None:
    if not scenario.group_column or not scenario.group_multiplier:
        raise ValueError("Group-dependent noise requires group_column and group_multiplier")
    if scenario.group_column not in observed.columns:
        raise ValueError(f"Unknown group-dependent noise column: {scenario.group_column}")
    for column in scenario.columns:
        if column not in NUMERIC_BOUNDS or column in PROTECTED_COLUMNS:
            raise ValueError(f"Unsupported or protected group-noise column: {column}")
        total_changed = 0
        for group, multiplier in scenario.group_multiplier.items():
            group_index = observed.index[observed[scenario.group_column].astype(str) == group]
            rate = min(1.0, scenario.noise_level * multiplier)
            indices = _sample_indices(rng, group_index, rate)
            relative_error = rng.uniform(-scenario.noise_level, scenario.noise_level, size=len(indices))
            observed.loc[indices, column] = _clip_numeric(
                column,
                observed.loc[indices, column].astype(float) * (1.0 + relative_error),
            )
            total_changed += len(indices)
        changed[column] = int(total_changed)


def apply_noise(
    df: pd.DataFrame,
    scenario: NoiseScenario,
    snapshot_year: int = 2014,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one scenario and return the observed view plus audit metadata."""
    validate_input(df)
    observed = df.copy(deep=True)
    rng = np.random.default_rng(scenario.seed)
    changed: dict[str, int] = {}

    if scenario.strategy == "clean":
        pass
    elif scenario.strategy == "random_numeric":
        _apply_random_numeric(observed, scenario, changed)
    elif scenario.strategy == "random_mixed":
        _apply_random_mixed(observed, scenario, changed)
    elif scenario.strategy == "systematic_numeric":
        _apply_systematic_numeric(observed, scenario, changed)
    elif scenario.strategy == "installation_year_error":
        _apply_year_error(observed, scenario, rng, changed, snapshot_year)
    elif scenario.strategy == "categorical_replacement":
        _apply_categorical_replacement(observed, scenario, changed)
    elif scenario.strategy == "group_numeric":
        _apply_group_numeric(observed, scenario, rng, changed)
    else:
        raise ValueError(f"Unsupported noise strategy: {scenario.strategy}")

    actual_changed = {
        column: int((df[column].astype(str) != observed[column].astype(str)).sum())
        for column in changed
    }
    metadata = {
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "strategy": scenario.strategy,
        "seed": scenario.seed,
        "noise_level": scenario.noise_level,
        "noise_level_meaning": (
            "selected_fraction_and_max_relative_error"
            if scenario.strategy in {"random_numeric", "random_mixed"}
            else "selected_fraction_for_categorical_replacement_or_strategy_specific"
        ),
        "nested_selection": scenario.strategy in {
            "random_numeric", "categorical_replacement", "random_mixed"
        },
        "columns": list(scenario.columns),
        "snapshot_year": snapshot_year,
        "selected_counts": changed,
        "changed_counts": actual_changed,
        "realized_rates": {column: count / len(df) for column, count in actual_changed.items()},
        # "protected_columns": sorted(PROTECTED_COLUMNS),
        "protected_columns": sorted(PROTECTED_COLUMNS.intersection(df.columns)),
        "truth_sha256": dataframe_hash(df),
        "observed_sha256": dataframe_hash(observed),
    }
    validate_output(df, observed, snapshot_year=snapshot_year)
    return observed, metadata


def dataframe_hash(df: pd.DataFrame) -> str:
    """Create a stable hash for metadata and reproducibility checks."""
    payload = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def validate_output(
    truth: pd.DataFrame,
    observed: pd.DataFrame,
    snapshot_year: int = 2014,
) -> dict[str, Any]:
    """Verify that a noisy observed view remains safe and structurally valid."""
    validate_input(truth)
    if list(truth.columns) != list(observed.columns):
        raise ValueError("Truth and observed schemas must match")
    if len(truth) != len(observed):
        raise ValueError("Truth and observed row counts must match")
    if not truth["segment_id"].equals(observed["segment_id"]):
        raise ValueError("segment_id values must not change")
    # for column in PROTECTED_COLUMNS:
    #     if not truth[column].equals(observed[column]):
    #         raise ValueError(f"Protected column changed: {column}")
    for column in PROTECTED_COLUMNS.intersection(truth.columns):
        if not truth[column].equals(observed[column]):
            raise ValueError(
                f"Protected column changed: {column}"
            )
    if not (observed["asset_age"] == snapshot_year - observed["installation_year"]).all():
        raise ValueError("Observed installation_year and asset_age are inconsistent")
    for column, allowed in CATEGORICAL_VALUES.items():
        invalid = set(observed[column].astype(str).unique()) - allowed
        if invalid:
            raise ValueError(f"Invalid values in {column}: {sorted(invalid)}")
    for column, (low, high) in NUMERIC_BOUNDS.items():
        if not observed[column].between(low, high).all():
            raise ValueError(f"Values in {column} exceed valid bounds")
    return {
        "passed": True,
        "changed_cells": int((truth != observed).sum().sum()),
        "changed_columns": [column for column in truth.columns if not truth[column].equals(observed[column])],
    }


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Calculate nominal association without encoding classes as numbers."""
    table = pd.crosstab(left.astype(str), right.astype(str))
    if table.empty or min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0))
    return float(np.sqrt((chi2 / observed.sum()) / max(1, min(table.shape) - 1)))


def _material_association(values: pd.Series, material: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(values):
        bins = pd.qcut(values, q=10, duplicates="drop")
        return _cramers_v(bins, material)
    return _cramers_v(values, material)


def distribution_diagnostics(
    truth: pd.DataFrame,
    observed: pd.DataFrame,
    scenario: NoiseScenario,
) -> pd.DataFrame:
    """Summarise distribution and material-association drift per changed field."""
    rows: list[dict[str, Any]] = []
    changed_columns = [column for column in truth.columns if not truth[column].equals(observed[column])]
    for column in changed_columns:
        truth_values = truth[column]
        observed_values = observed[column]
        changed_mask = truth_values.astype(str) != observed_values.astype(str)
        row: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "strategy": scenario.strategy,
            "seed": scenario.seed,
            "column": column,
            "changed_count": int(changed_mask.sum()),
            "changed_rate": float(changed_mask.mean()),
            "truth_material_association_cramers_v": _material_association(truth_values, truth["material"]),
            "observed_material_association_cramers_v": _material_association(observed_values, observed["material"]),
        }
        row["material_association_delta"] = (
            row["observed_material_association_cramers_v"]
            - row["truth_material_association_cramers_v"]
        )
        if pd.api.types.is_numeric_dtype(truth_values):
            row.update(
                {
                    "truth_mean": float(truth_values.mean()),
                    "observed_mean": float(observed_values.mean()),
                    "truth_std": float(truth_values.std()),
                    "observed_std": float(observed_values.std()),
                    "mean_absolute_change": float((truth_values - observed_values).abs().mean()),
                    "truth_observed_correlation": float(truth_values.corr(observed_values)),
                    "distribution_l1": _numeric_distribution_l1(truth_values, observed_values),
                }
            )
        else:
            truth_distribution = truth_values.astype(str).value_counts(normalize=True)
            observed_distribution = observed_values.astype(str).value_counts(normalize=True)
            categories = truth_distribution.index.union(observed_distribution.index)
            row.update(
                {
                    "truth_mean": np.nan,
                    "observed_mean": np.nan,
                    "truth_std": np.nan,
                    "observed_std": np.nan,
                    "mean_absolute_change": np.nan,
                    "truth_observed_correlation": np.nan,
                    "distribution_l1": float(
                        0.5
                        * sum(
                            abs(
                                truth_distribution.get(category, 0.0)
                                - observed_distribution.get(category, 0.0)
                            )
                            for category in categories
                        )
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _numeric_distribution_l1(truth: pd.Series, observed: pd.Series) -> float:
    edges = np.unique(np.quantile(truth.astype(float), np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    truth_hist, _ = np.histogram(truth, bins=edges)
    observed_hist, _ = np.histogram(observed, bins=edges)
    truth_share = truth_hist / max(1, truth_hist.sum())
    observed_share = observed_hist / max(1, observed_hist.sum())
    return float(0.5 * np.abs(truth_share - observed_share).sum())


def _write_markdown_report(
    output: Path,
    quality: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> Path:
    lines = [
        "# Controlled Predictor Noise Report",
        "",
        "This report describes synthetic observed-data views created from the clean overhead-conductor reference. It does not estimate SAPN data-quality rates.",
        "",
        "## Scenario results",
        "",
        "| Scenario | Strategy | Column | Changed rate | Validation |",
        "|---|---|---|---:|---|",
    ]
    for row in quality.itertuples(index=False):
        lines.append(
            f"| {row.scenario_id} | {row.strategy} | {row.column or 'none'} | {row.realized_rate:.3f} | {'passed' if row.validation_passed else 'failed'} |"
        )
    lines += [
        "",
        "## Distribution and relationship drift",
        "",
        "`distribution_l1` is zero when the binned numeric or categorical distribution is unchanged. `material_association_delta` records the change in nominal association with the material target; it is diagnostic only.",
        "",
        "| Scenario | Column | Distribution L1 | Material association delta | Truth-observed correlation |",
        "|---|---|---:|---:|---:|",
    ]
    for row in diagnostics.itertuples(index=False):
        correlation = "-" if pd.isna(row.truth_observed_correlation) else f"{row.truth_observed_correlation:.3f}"
        lines.append(
            f"| {row.scenario_id} | {row.column} | {row.distribution_l1:.3f} | {row.material_association_delta:+.3f} | {correlation} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Random numerical levels specify both the selected fraction and maximum relative error; categorical levels specify the selected fraction.",
        "- Same-seed random scenarios of the same family use nested corruption selections across increasing levels.",
        "- Apply the same original MF01 missingness mask downstream; this generator alone does not add missingness.",
        "- Do not treat a 100% categorical replacement scenario as a plausible real-world error rate; it is an extreme stress test.",
        "- Systematic and group-dependent scenarios represent structured data-quality problems and should not be averaged together with random noise.",
        "- Installation-year noise is propagated to asset age to preserve the source data contract.",
        "- `material` and `conductor_type` remain unchanged in every scenario.",
        "- Method comparisons are intentionally deferred to the next ticket.",
    ]
    report = output / "noise_quality_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_scenarios(
    input_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    scenario_ids: list[str] | None = None,
) -> Path:
    """Run configured scenarios and write datasets, metadata and a report."""
    input_file = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    truth = pd.read_csv(input_file)
    settings, scenarios = load_scenarios(config_path)
    snapshot_year = int(settings.get("snapshot_year", 2014))
    selected = set(scenario_ids) if scenario_ids else None
    report_rows: list[dict[str, Any]] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for scenario in scenarios:
        if selected is not None and scenario.scenario_id not in selected:
            continue
        observed, metadata = apply_noise(truth, scenario, snapshot_year=snapshot_year)
        scenario_dir = output / scenario.scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        observed_path = scenario_dir / "observed.csv"
        truth_path = scenario_dir / "truth.csv"
        metadata_path = scenario_dir / "metadata.json"
        observed.to_csv(observed_path, index=False)
        truth.to_csv(truth_path, index=False)
        validation = validate_output(truth, observed, snapshot_year=snapshot_year)
        diagnostic_frames.append(distribution_diagnostics(truth, observed, scenario))
        metadata.update(
            {
                "input_file": str(input_file),
                "observed_file": str(observed_path),
                "truth_file": str(truth_path),
                "validation": validation,
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for column, count in metadata["changed_counts"].items():
            report_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "strategy": scenario.strategy,
                    "seed": scenario.seed,
                    "column": column,
                    "requested_noise_level": scenario.noise_level,
                    "changed_count": count,
                    "realized_rate": count / len(truth),
                    "validation_passed": validation["passed"],
                }
            )
        if not metadata["changed_counts"]:
            report_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "strategy": scenario.strategy,
                    "seed": scenario.seed,
                    "column": "",
                    "requested_noise_level": scenario.noise_level,
                    "changed_count": 0,
                    "realized_rate": 0.0,
                    "validation_passed": validation["passed"],
                }
            )
    if not report_rows:
        raise ValueError("No scenarios selected")
    report_path = output / "noise_quality_report.csv"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    diagnostics.to_csv(output / "noise_distribution_report.csv", index=False)
    _write_markdown_report(output, pd.DataFrame(report_rows), diagnostics)
    return report_path
