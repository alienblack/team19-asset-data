"""Validate controlled relationships and leakage in the overhead-conductor data.

This module is deliberately separate from the imputation implementations.  It
audits the data-generating assumptions first, so an apparent imputation result
cannot be mistaken for evidence about SAPN's real asset register.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .generate_synthetic_data import (
    EXPECTED_CORROSION_ZONES,
    EXPECTED_LOCATION_GROUPS,
    EXPECTED_MATERIAL_CLASSES,
    REQUIRED_OUTPUT_COLUMNS,
    EXPECTED_VOLTAGE_CATEGORIES,
    Config,
    StrengthConfig,
    calibrate_route_length_marginals,
    generate_dataset,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_SEEDS = (219, 319, 419)
DEFAULT_MIN_SUPPORT = 25
AGE_BANDS = ("under_20", "20_to_50", "over_50")
MODEL_READY_COLUMNS = set(REQUIRED_OUTPUT_COLUMNS) - {"conductor_type"}


def cramers_v(left: Iterable[Any], right: Iterable[Any]) -> float:
    """Return bias-uncorrected Cramer's V for two categorical variables."""
    table = pd.crosstab(pd.Series(left), pd.Series(right))
    if table.empty or min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0))
    n = observed.sum()
    phi2 = chi2 / n
    return float(np.sqrt(phi2 / max(1, min(table.shape) - 1)))


def categorical_association(left: pd.Series, right: pd.Series) -> dict[str, float]:
    """Return multiclass association metrics without arbitrary label encoding."""
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if frame.empty:
        return {"n_rows": 0, "cramers_v": 0.0, "normalized_mutual_information": 0.0}
    return {
        "n_rows": int(len(frame)),
        "cramers_v": cramers_v(frame["left"], frame["right"]),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(frame["left"].astype(str), frame["right"].astype(str))
        ),
    }


def maximum_category_purity(
    feature: pd.Series,
    target: pd.Series,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> dict[str, float]:
    """Measure target purity while ignoring tiny, misleading categories."""
    frame = pd.DataFrame({"feature": feature, "target": target}).dropna()
    counts = frame.groupby("feature", observed=False).size()
    eligible = counts[counts >= min_support].index
    if len(eligible) == 0:
        return {"max_purity": 0.0, "category": "", "support": 0}
    proportions = (
        frame[frame["feature"].isin(eligible)]
        .groupby("feature", observed=False)["target"]
        .value_counts(normalize=True)
    )
    best_index = proportions.groupby(level=0).max().idxmax()
    return {
        "max_purity": float(proportions.groupby(level=0).max().loc[best_index]),
        "category": str(best_index),
        "support": int(counts.loc[best_index]),
    }


def material_by_era(years: pd.Series) -> pd.Series:
    return pd.cut(
        years,
        bins=[-np.inf, 1994, 2014],
        labels=["pre_1995", "from_1995"],
        right=True,
    ).astype(str)


def age_band(ages: pd.Series) -> pd.Series:
    return pd.cut(
        ages,
        bins=[-np.inf, 19, 50, np.inf],
        labels=list(AGE_BANDS),
        right=True,
    ).astype(str)


def schema_and_invariants(
    complete: pd.DataFrame,
    model_ready: pd.DataFrame,
    snapshot_year: int,
) -> dict[str, Any]:
    """Check the contract shared by generation and modelling layers."""
    checks: dict[str, Any] = {
        "complete_columns_match": list(complete.columns) == list(REQUIRED_OUTPUT_COLUMNS),
        "model_ready_columns_match": set(model_ready.columns) == MODEL_READY_COLUMNS,
        "no_missing_complete_cells": int(complete.isna().sum().sum()) == 0,
        "unique_segment_id": bool(complete["segment_id"].is_unique),
        "age_equation_holds": bool(
            (complete["asset_age"] == snapshot_year - complete["installation_year"]).all()
        ),
        "valid_material_classes": set(complete["material"].unique()) <= EXPECTED_MATERIAL_CLASSES,
        "valid_voltage_categories": set(complete["voltage_category"].unique()) <= EXPECTED_VOLTAGE_CATEGORIES,
        "valid_location_groups": set(complete["location_group"].unique()) <= EXPECTED_LOCATION_GROUPS,
        "valid_corrosion_zones": set(complete["corrosion_zone"].unique()) <= EXPECTED_CORROSION_ZONES,
        "positive_lengths": bool((complete["length_km"] > 0).all()),
        "nonnegative_distance": bool((complete["distance_to_sea_km"] >= 0).all()),
        "no_conductor_type_in_model_ready": "conductor_type" not in model_ready.columns,
    }
    checks["passed"] = all(checks.values())
    return checks


def material_predictor_frame(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return intentionally auditable material predictors and derived checks."""
    return {
        "installation_year": df["installation_year"],
        "asset_age": df["asset_age"],
        "voltage_category": df["voltage_category"],
        "location_group": df["location_group"],
        "corrosion_zone": df["corrosion_zone"],
        "distance_to_sea_km": df["distance_to_sea_km"],
        "covering": df["covering"],
        "length_km": df["length_km"],
        "feeder_id": df["feeder_id"],
        "conductor_type": df["conductor_type"],
    }


def feature_screen(df: pd.DataFrame, feature: str, random_state: int = 219) -> dict[str, float]:
    """Estimate single-feature material recoverability using a train-only fit."""
    target = df["material"].astype(str)
    values = df[[feature]].copy()
    train_x, test_x, train_y, test_y = train_test_split(
        values, target, test_size=0.20, stratify=target, random_state=random_state
    )
    numeric = pd.api.types.is_numeric_dtype(values[feature])
    if numeric:
        transformer = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
    else:
        transformer = Pipeline(
            [("imputer", SimpleImputer(strategy="most_frequent")),
             ("encode", OneHotEncoder(handle_unknown="ignore"))]
        )
    model = Pipeline(
        [("preprocess", ColumnTransformer([("feature", transformer, [feature])])),
         ("classifier", LogisticRegression(max_iter=500, class_weight="balanced"))]
    )
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "macro_f1": float(f1_score(test_y, prediction, average="macro")),
    }


def profile_metrics(df: pd.DataFrame, profile: str, seed: int) -> list[dict[str, Any]]:
    """Calculate association statistics for one generated profile and seed."""
    frame = df.copy()
    frame["installation_era"] = material_by_era(frame["installation_year"])
    frame["age_band"] = age_band(frame["asset_age"])
    relations = [
        ("installation_era", "voltage_category", "configured_era_to_voltage"),
        ("voltage_category", "location_group", "configured_voltage_to_location"),
        ("voltage_category", "corrosion_zone", "configured_voltage_to_corrosion"),
        ("location_group", "corrosion_zone", "configured_location_to_corrosion"),
        ("installation_era", "material", "configured_era_to_material"),
        ("voltage_category", "material", "configured_voltage_to_material"),
        ("corrosion_zone", "material", "configured_corrosion_to_material"),
        ("age_band", "material", "age_band_to_material"),
        ("feeder_id", "material", "feeder_to_material"),
        ("conductor_type", "material", "direct_leakage_conductor_type"),
    ]
    rows: list[dict[str, Any]] = []
    for left, right, relation in relations:
        association = categorical_association(frame[left], frame[right])
        rows.extend(
            {
                "profile": profile,
                "seed": seed,
                "relation": relation,
                "left_feature": left,
                "right_feature": right,
                "statistic": statistic,
                "value": value,
                "n_rows": association["n_rows"],
            }
            for statistic, value in association.items()
            if statistic != "n_rows"
        )

    for feature in ("installation_year", "asset_age", "distance_to_sea_km", "length_km"):
        association = categorical_association(
            pd.qcut(frame[feature], q=10, duplicates="drop"), frame["material"]
        )
        rows.append(
            {
                "profile": profile,
                "seed": seed,
                "relation": f"binned_{feature}_to_material",
                "left_feature": feature,
                "right_feature": "material",
                "statistic": "normalized_mutual_information",
                "value": association["normalized_mutual_information"],
                "n_rows": association["n_rows"],
            }
        )
    return rows


def leakage_rows(df: pd.DataFrame, seed: int, min_support: int) -> list[dict[str, Any]]:
    """Produce explicit leakage, duplicate-evidence and model-screening rows."""
    rows: list[dict[str, Any]] = []
    target = df["material"]
    for feature in ("conductor_type", "feeder_id", "voltage_category", "location_group", "corrosion_zone", "covering"):
        association = categorical_association(df[feature], target)
        purity = maximum_category_purity(df[feature], target, min_support=min_support)
        for statistic, value in (
            ("cramers_v", association["cramers_v"]),
            ("normalized_mutual_information", association["normalized_mutual_information"]),
            ("max_category_purity", purity["max_purity"]),
        ):
            rows.append(
                {
                    "feature": feature,
                    "audit_type": "categorical_leakage_screen",
                    "statistic": statistic,
                    "value": float(value),
                    "support": purity["support"],
                    "threshold": 0.95 if statistic == "max_category_purity" else 0.80,
                    "flag": bool(value >= (0.95 if statistic == "max_category_purity" else 0.80)),
                    "interpretation": "Direct target-revealing evidence" if feature == "conductor_type" else "Review for unintended recoverability",
                    "seed": seed,
                }
            )

    screen_features = (
        "conductor_type", "feeder_id", "installation_year", "asset_age",
        "voltage_category", "location_group", "corrosion_zone", "distance_to_sea_km",
        "covering", "length_km",
    )
    for feature in screen_features:
        metrics = feature_screen(df, feature, random_state=seed)
        for statistic, value in metrics.items():
            rows.append(
                {
                    "feature": feature,
                    "audit_type": "single_feature_classifier",
                    "statistic": statistic,
                    "value": value,
                    "support": len(df),
                    "threshold": 0.80,
                    "flag": bool(value >= 0.80),
                    "interpretation": "Potential leakage or overly informative feature" if value >= 0.80 else "No standalone high-performance signal detected",
                    "seed": seed,
                }
            )

    duplicate = float(df["installation_year"].corr(df["asset_age"]))
    rows.append(
        {
            "feature": "installation_year + asset_age",
            "audit_type": "duplicate_timing_evidence",
            "statistic": "pearson_correlation",
            "value": duplicate,
            "support": len(df),
            "threshold": 0.99,
            "flag": abs(duplicate) >= 0.99,
            "interpretation": "Deterministic duplicate evidence; use one timing field per model",
            "seed": seed,
        }
    )
    return rows


def direction_checks(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare low/high strength endpoints and estimate monotonic direction."""
    relation_map = {
        "configured_era_to_voltage": "increase",
        "configured_voltage_to_location": "increase",
        "configured_voltage_to_corrosion": "increase",
        "configured_location_to_corrosion": "increase",
        "configured_era_to_material": "increase",
        "configured_voltage_to_material": "increase",
        "configured_corrosion_to_material": "increase",
    }
    rows: list[dict[str, Any]] = []
    filtered = metrics[metrics["statistic"] == "normalized_mutual_information"]
    for relation, expected in relation_map.items():
        values = filtered[filtered["relation"] == relation].groupby("profile")["value"].mean()
        if values.empty:
            continue
        numeric_profile = values.index.to_series().str.extract(r"(\d+)")[0].astype(float)
        ordered = pd.Series(values.to_numpy(), index=numeric_profile).sort_index()
        low = float(ordered.iloc[0])
        high = float(ordered.iloc[-1])
        slope = float(np.polyfit(ordered.index.to_numpy(), ordered.to_numpy(), 1)[0]) if len(ordered) > 1 else 0.0
        delta = high - low
        rows.append(
            {
                "relation": relation,
                "statistic": "nmi_low_to_high_strength",
                "low_strength_value": low,
                "high_strength_value": high,
                "delta": delta,
                "slope": slope,
                "expected_direction": expected,
                "status": "visible" if delta > 0.01 else "weak_or_indirect",
            }
        )
    return pd.DataFrame(rows)


def load_or_generate(
    config: Config,
    strengths: StrengthConfig,
    seed: int,
    profile_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load the committed seed-219 artefact or generate another seed in memory."""
    if seed == config.seed:
        directory = profile_root / strengths.profile_name
        complete_path = directory / "overhead_conductors_complete_reference.csv"
        model_path = directory / "material_model_ready_data.csv"
        if complete_path.is_file() and model_path.is_file():
            return pd.read_csv(complete_path), pd.read_csv(model_path), str(complete_path)

    run_config = replace(config, seed=seed)
    complete = calibrate_route_length_marginals(
        generate_dataset(run_config, strengths), run_config
    )
    model_ready = complete.drop(columns=["conductor_type"])
    return complete, model_ready, "generated_in_memory"


def write_heatmap(metrics: pd.DataFrame, output_dir: Path) -> None:
    subset = metrics[
        (metrics["statistic"] == "normalized_mutual_information")
        & (metrics["right_feature"] == "material")
    ].copy()
    pivot = subset.pivot_table(index="relation", columns="profile", values="value", aggfunc="mean")
    if pivot.empty:
        return
    figure, axis = plt.subplots(figsize=(12, 5.5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.01, float(pivot.to_numpy().max())))
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Material association across relationship-strength profiles")
    axis.set_xlabel("Generator profile")
    axis.set_ylabel("Association audit")
    figure.colorbar(image, ax=axis, label="Normalised mutual information")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "material_association_heatmap.png", dpi=160)
    plt.close(figure)


def write_report(
    output_dir: Path,
    profile_metrics: pd.DataFrame,
    direction: pd.DataFrame,
    leakage: pd.DataFrame,
    invariant_records: list[dict[str, Any]],
    seeds: tuple[int, ...],
    stability: pd.DataFrame,
) -> None:
    invariant_pass = all(item["checks"]["passed"] for item in invariant_records)
    direct = leakage[
        (leakage["feature"] == "conductor_type")
        & (leakage["statistic"] == "max_category_purity")
    ]
    duplicate = leakage[leakage["audit_type"] == "duplicate_timing_evidence"]
    high_screen = leakage[(leakage["audit_type"] == "single_feature_classifier") & leakage["flag"]]
    lines = [
        "# Overhead-conductor relationship and leakage validation",
        "",
        "## Purpose",
        "",
        "This report validates the controlled synthetic dataset before it is used for imputation. It measures recoverability under documented synthetic assumptions; it is not evidence about the confidential SAPN asset register.",
        "",
        "## Run configuration",
        "",
        f"- Seeds: `{', '.join(map(str, seeds))}`",
        f"- Profiles: `{profile_metrics['profile'].nunique()}` relationship-strength profiles",
        f"- Association rows: `{len(profile_metrics):,}`",
        f"- Cross-seed stability rows: `{len(stability):,}`",
        "- Categorical association: Cramer's V and normalised mutual information; no arbitrary numeric encoding of material.",
        "- The direct `conductor_type` field is retained only for the leakage audit and excluded from material-model-ready data.",
        "",
        "## Results",
        "",
        f"- Schema and invariant checks: **{'passed' if invariant_pass else 'not all passed'}** across `{len(invariant_records)}` profile-seed runs.",
        f"- `conductor_type` leakage audit rows: `{len(direct)}`; high-purity flag observed: **{bool(direct['flag'].any())}**.",
        f"- Deterministic timing duplicate (`installation_year` and `asset_age`) detected: **{bool(duplicate['flag'].any())}**.",
        f"- Single-feature classifier screens flagged for review: `{len(high_screen)}` rows. These are review warnings, not proof that every flagged feature is invalid.",
        "",
        "### Direction checks",
        "",
        "| Relationship | Low-strength NMI | High-strength NMI | Change | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for row in direction.itertuples(index=False):
        lines.append(
            f"| {row.relation} | {row.low_strength_value:.3f} | {row.high_strength_value:.3f} | {row.delta:+.3f} | {row.status} |"
        )
    lines += [
        "",
        "## Interpretation for the next experiment",
        "",
        "1. Freeze the material-model feature list without `conductor_type`.",
        "2. Use only one of `installation_year` or `asset_age` in a given distance/model experiment; they are deterministic duplicates.",
        "3. Keep the clean complete reference as ground truth and create missingness/noise as separate observed views.",
        "4. Run baseline and KNN comparisons only after recording the same seeds, masks, predictors and metrics.",
        "5. Treat downstream PoF/Weibull stability as a later validation design until SAPN confirms its reference calculation and acceptable deviation.",
        "",
        "## Limitations",
        "",
        "- The generator uses explicit synthetic conditional probabilities and aggregate route-length calibration; observed associations are not SAPN record-level estimates.",
        "- Some associations at strength zero can remain because of upstream dependencies and route-length calibration.",
        "- This ticket validates data-generation assumptions and leakage controls. It does not claim a final imputation method or downstream model stability.",
        "",
        "## Reproducibility",
        "",
        "Run from the repository root with:",
        "",
        "```bash",
        "PYTHONPATH=src python -m team19_asset_data.overhead_conductors.relationship_validation \\",
        "  --config config/overhead_conductors/generation.yaml \\",
        "  --output-dir outputs/PGS219-relationship-validation \\",
        "  --seeds 219,319,419",
        "```",
        "",
        "The machine-readable outputs in this directory are the evidence for the Jira ticket.",
    ]
    (output_dir / "relationship_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    config_path: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> Path:
    """Run E0-E3 validation and write all evidence files."""
    config = Config.from_yaml(config_path)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_root = config.output_dir
    profile_paths = sorted(
        (Path(config_path).resolve().parent / "relationship_strength").glob("strength_*.yaml"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not profile_paths:
        raise FileNotFoundError("No strength_*.yaml profiles found")

    metric_rows: list[dict[str, Any]] = []
    leakage_rows_all: list[dict[str, Any]] = []
    invariant_records: list[dict[str, Any]] = []
    for profile_path in profile_paths:
        strengths = StrengthConfig.from_yaml(profile_path)
        for seed in seeds:
            LOGGER.info("Validating %s with seed %s", strengths.profile_name, seed)
            complete, model_ready, source = load_or_generate(config, strengths, seed, profile_root)
            checks = schema_and_invariants(complete, model_ready, config.snapshot_year)
            invariant_records.append(
                {
                    "profile": strengths.profile_name,
                    "seed": seed,
                    "source": source,
                    "rows": len(complete),
                    "checks": checks,
                }
            )
            metric_rows.extend(profile_metrics(complete, strengths.profile_name, seed))
            # Leakage is invariant to relationship profile, so collect it for
            # the strongest profile only to avoid duplicating the same audit.
            if strengths.profile_name == "strength_100":
                leakage_rows_all.extend(leakage_rows(complete, seed, min_support))

    metrics = pd.DataFrame(metric_rows)
    leakage = pd.DataFrame(leakage_rows_all)
    direction = direction_checks(metrics)
    stability = (
        metrics.groupby(["profile", "relation", "statistic"], as_index=False)["value"]
        .agg(["mean", "std", "min", "max"])
        .rename(
            columns={
                "std": "standard_deviation",
                "min": "minimum",
                "max": "maximum",
            }
        )
        .fillna(0.0)
        .reset_index()
    )
    metrics.to_csv(output / "relationship_profile_metrics.csv", index=False)
    direction.to_csv(output / "relationship_direction_checks.csv", index=False)
    stability.to_csv(output / "relationship_seed_stability.csv", index=False)
    leakage.to_csv(output / "leakage_audit.csv", index=False)
    (output / "schema_and_invariant_checks.json").write_text(
        json.dumps(invariant_records, indent=2), encoding="utf-8"
    )
    metadata = {
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(output),
        "seeds": list(seeds),
        "profile_count": len(profile_paths),
        "profiles": [path.stem for path in profile_paths],
        "min_category_support": min_support,
        "material_model_excluded_columns": ["conductor_type"],
        "duplicate_timing_fields": ["installation_year", "asset_age"],
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_heatmap(metrics, output / "relationship_heatmaps")
    write_report(output, metrics, direction, leakage, invariant_records, seeds, stability)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="219,319,419", help="Comma-separated generation seeds")
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise SystemExit("At least one seed is required")
    run_validation(args.config, args.output_dir, seeds=seeds, min_support=args.min_support)


if __name__ == "__main__":
    main()
