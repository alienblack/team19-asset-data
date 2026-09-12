
"""
Synthetic Overhead Conductor Dataset Generator
==============================================

Purpose
-------
Generate a controlled, SAPN-inspired synthetic overhead-conductor dataset for
Sprint 2 material-imputation experiments.

Important
---------
This script DOES NOT claim to reproduce SAPN's real asset register.

It uses:
1. Public SAPN aggregate evidence from the 2014 Overhead Conductor AMP.
2. Explicit synthetic conditional rules where the AMP does not provide exact
   record-level joint probabilities.
3. A complete reference dataset with no missing values.
4. A three-class material target for the current pilot:
      Aluminium, Copper, Steel
   ACSR is intentionally excluded from this pilot target because it is a
   composite conductor group and cannot be cleanly mapped to one of those
   three material labels.

Snapshot year
-------------
The AMP's age statistics describe the network in 2014, so age is calculated as:

    asset_age = 2014 - installation_year

This lets validation compare generated age statistics with the published 2014
profile without mixing time periods.

Dependencies
------------
installation_year -> asset_age
installation_year -> voltage_category
voltage_category -> corrosion_zone
installation_year + voltage_category + corrosion_zone -> material
material -> conductor_type
location_group -> feeder_group
location_group + voltage_category -> length_km
location_group -> distance_to_sea_km
covering is generated weakly and independently enough to avoid making it an
unrealistically strong material predictor.

Target leakage warning
----------------------
conductor_type is kept in the complete synthetic asset register because it is
realistic, but it should usually be EXCLUDED from a material-imputation model
because codes such as AAC, AAAC, SC/AC and SC/GZ reveal material directly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT_DIR / "data" / "synthetic" / "overhead_conductors"

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    n_rows: int = 10_000
    seed: int = 219
    snapshot_year: int = 2014
    output_dir: str = OUTPUT_DIR

    # Published SAPN route-length constraints used as validation targets.
    voltage_target: Dict[str, float] = None
    corrosion_target: Dict[str, float] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "voltage_target",
            {
                "66 kV": 0.020,
                "33 kV": 0.056,
                "19 kV SWER": 0.409,
                "11/7.6 kV": 0.249,
                "LV": 0.266,
            },
        )
        object.__setattr__(
            self,
            "corrosion_target",
            {
                "Low": 0.53,
                "Severe": 0.35,
                "Very Severe": 0.11,
            },
        )


# ---------------------------------------------------------------------------
# 2. Small reusable helpers
# ---------------------------------------------------------------------------

def weighted_choice(rng: np.random.Generator, labels, probs, size=None):
    """Choose values using explicit probabilities."""
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(labels, size=size, p=probs)


def clipped_lognormal(
    rng: np.random.Generator,
    median: float,
    sigma: float,
    low: float,
    high: float,
) -> float:
    """Generate a positive right-skewed value and keep it in a sensible range."""
    value = rng.lognormal(mean=np.log(median), sigma=sigma)
    return float(np.clip(value, low, high))


# ---------------------------------------------------------------------------
# 3. Installation year / age
# ---------------------------------------------------------------------------

def sample_installation_year(rng: np.random.Generator) -> int:
    """
    Generate installation year from three age bands calibrated to the 2014 AMP.

    Published 2014 constraints:
    - about 59% of overhead conductors were older than 50 years
    - about 2% were younger than 20 years
    - strong historical installation peaks in 1955, 1956, 1958 and 1966

    The remaining 39% are placed in the 20-50 year band.

    Within each band, exact year-level probabilities are synthetic choices used
    to preserve the published shape while still generating varied records.
    """

    age_band = weighted_choice(
        rng,
        labels=["older_than_50", "age_20_to_50", "younger_than_20"],
        probs=[0.59, 0.39, 0.02],
    )

    if age_band == "older_than_50":
        # Years before 1964 => age > 50 in 2014.
        # Put extra weight on the published peaks 1955, 1956 and 1958.
        if rng.random() < 0.48:
            return int(
                weighted_choice(
                    rng,
                    labels=[1955, 1956, 1958],
                    probs=[0.36, 0.29, 0.35],
                )
            )

        # Most remaining older assets are placed in the post-war build-out era,
        # with a smaller tail of earlier installations.
        if rng.random() < 0.78:
            return int(rng.integers(1948, 1964))
        return int(rng.integers(1930, 1948))

    if age_band == "age_20_to_50":
        # 1964-1994 inclusive. Give 1966 a visible historical peak.
        if rng.random() < 0.28:
            return 1966
        return int(rng.integers(1964, 1995))

    # Younger than 20 in the 2014 snapshot.
    return int(rng.integers(1995, 2015))


# ---------------------------------------------------------------------------
# 4. Voltage
# ---------------------------------------------------------------------------

def sample_voltage(
    rng: np.random.Generator,
    installation_year: int,
) -> str:
    """
    Generate voltage conditionally on installation era.

    Evidence-backed direction:
    - 1930-1949 installations were predominantly 33 kV.
    - 1955/56/58/66 installations were predominantly SWER and 11 kV.
    - LV, 33 kV and 66 kV also occurred during 1950-1979.

    Exact probabilities are transparent synthetic assumptions.
    """

    labels = ["66 kV", "33 kV", "19 kV SWER", "11/7.6 kV", "LV"]

    if installation_year <= 1949:
        probs = [0.05, 0.50, 0.12, 0.15, 0.18]
    elif installation_year <= 1977:
        probs = [0.02, 0.05, 0.46, 0.29, 0.18]
    elif installation_year <= 1994:
        probs = [0.02, 0.05, 0.30, 0.25, 0.38]
    else:
        probs = [0.01, 0.04, 0.16, 0.25, 0.54]

    return str(weighted_choice(rng, labels, probs))


# ---------------------------------------------------------------------------
# 5. Location, corrosion and feeder
# ---------------------------------------------------------------------------

def sample_location_group(
    rng: np.random.Generator,
    voltage: str,
) -> str:
    """
    Generate a simple operational location group.

    The AMP explicitly uses rural/urban in its multi-variable defect model,
    but does not publish a complete population split. These probabilities are
    therefore synthetic assumptions.

    SWER is made strongly rural because it is a rural distribution technology.
    """

    if voltage == "19 kV SWER":
        return str(weighted_choice(rng, ["Rural", "Regional", "Urban"], [0.80, 0.18, 0.02]))

    if voltage in {"33 kV", "66 kV"}:
        return str(weighted_choice(rng, ["Rural", "Regional", "Urban"], [0.35, 0.35, 0.30]))

    if voltage == "LV":
        return str(weighted_choice(rng, ["Rural", "Regional", "Urban"], [0.18, 0.27, 0.55]))

    return str(weighted_choice(rng, ["Rural", "Regional", "Urban"], [0.45, 0.30, 0.25]))


def sample_corrosion_zone(
    rng: np.random.Generator,
    voltage: str,
    location_group: str,
) -> str:
    """
    Generate corrosion zone using voltage/location context.

    Published overall route-length target:
        Low ~53%, Severe ~35%, Very Severe ~11%

    The exact conditional probabilities below are synthetic choices designed
    to create realistic overlap rather than deterministic rules.
    """

    # Start from the published marginal mix.
    base = {
        "Low": 0.535,
        "Severe": 0.355,
        "Very Severe": 0.110,
    }

    # Small context shifts.
    if location_group == "Urban":
        base["Low"] -= 0.08
        base["Severe"] += 0.05
        base["Very Severe"] += 0.03
    elif location_group == "Rural":
        base["Low"] += 0.06
        base["Severe"] -= 0.04
        base["Very Severe"] -= 0.02

    if voltage == "19 kV SWER":
        base["Low"] += 0.04
        base["Severe"] -= 0.03
        base["Very Severe"] -= 0.01

    probs = np.array([base["Low"], base["Severe"], base["Very Severe"]], dtype=float)
    probs = np.clip(probs, 0.01, None)
    probs = probs / probs.sum()

    return str(weighted_choice(rng, ["Low", "Severe", "Very Severe"], probs))


def make_feeder_map(rng: np.random.Generator) -> Dict[str, list]:
    """Create synthetic feeder groups by location."""
    return {
        "Urban": [f"U_FDR_{i:03d}" for i in range(1, 31)],
        "Regional": [f"G_FDR_{i:03d}" for i in range(1, 26)],
        "Rural": [f"R_FDR_{i:03d}" for i in range(1, 46)],
    }


def sample_feeder(
    rng: np.random.Generator,
    location_group: str,
    feeder_map: Dict[str, list],
) -> str:
    return str(rng.choice(feeder_map[location_group]))


# ---------------------------------------------------------------------------
# 6. Distance to sea
# ---------------------------------------------------------------------------

def sample_distance_to_sea(
    rng: np.random.Generator,
    location_group: str,
    corrosion_zone: str,
) -> float:
    """
    Generate a plausible enrichment variable.

    IMPORTANT:
    The AMP supports the concept that coastal saline exposure affects corrosion,
    but does NOT provide exact distance thresholds. Therefore these values are
    synthetic assumptions and must not be described as SAPN thresholds.
    """

    zone_ranges: Dict[str, Tuple[float, float]] = {
        "Very Severe": (0.2, 15.0),
        "Severe": (2.0, 70.0),
        "Low": (20.0, 300.0),
    }

    low, high = zone_ranges[corrosion_zone]

    # Slight location adjustment while preserving overlap.
    if location_group == "Urban":
        high = min(high, 100.0)
    elif location_group == "Rural" and corrosion_zone == "Low":
        high = 350.0

    # Beta gives more density near the lower end without using a hard rule.
    x = rng.beta(1.6, 2.8)
    return round(low + x * (high - low), 2)


# ---------------------------------------------------------------------------
# 7. Material
# ---------------------------------------------------------------------------

def sample_material(
    rng: np.random.Generator,
    installation_year: int,
    voltage: str,
    corrosion_zone: str,
) -> str:
    """
    Generate the three-class material target.

    Pilot classes:
        Steel, Copper, Aluminium

    Overall known non-ACSR SAPN route-length mix, after excluding ACSR and
    Unknown, is approximately:
        Steel      74.0%
        Copper     15.9%
        Aluminium  10.1%

    This function does NOT simply sample those marginals. It modifies them using
    year, voltage and corrosion context so the imputation problem contains
    learnable but non-deterministic structure.

    Exact conditional probabilities are synthetic modelling assumptions.
    """

    p = {
        "Steel": 0.74,
        "Copper": 0.159,
        "Aluminium": 0.101,
    }

    # Voltage effect: SWER strongly associated with steel in the AMP.
    if voltage == "19 kV SWER":
        p["Steel"] += 0.16
        p["Copper"] -= 0.09
        p["Aluminium"] -= 0.07

    elif voltage in {"33 kV", "66 kV"}:
        p["Steel"] -= 0.12
        p["Copper"] += 0.07
        p["Aluminium"] += 0.05

    elif voltage == "LV":
        p["Steel"] -= 0.10
        p["Copper"] += 0.04
        p["Aluminium"] += 0.06

    # Historical-era effect.
    if installation_year <= 1949:
        p["Steel"] -= 0.16
        p["Copper"] += 0.12
        p["Aluminium"] += 0.04

    elif installation_year >= 1980:
        p["Steel"] -= 0.18
        p["Copper"] -= 0.02
        p["Aluminium"] += 0.20

    # Corrosion-context effect kept intentionally mild.
    if corrosion_zone == "Very Severe":
        p["Steel"] -= 0.05
        p["Copper"] += 0.02
        p["Aluminium"] += 0.03
    elif corrosion_zone == "Low":
        p["Steel"] += 0.03
        p["Copper"] += 0.01
        p["Aluminium"] -= 0.04

    labels = ["Steel", "Copper", "Aluminium"]
    probs = np.array([p[x] for x in labels], dtype=float)
    probs = np.clip(probs, 0.02, None)
    probs = probs / probs.sum()

    return str(weighted_choice(rng, labels, probs))


# ---------------------------------------------------------------------------
# 8. Conductor type
# ---------------------------------------------------------------------------

def sample_conductor_type(
    rng: np.random.Generator,
    material: str,
) -> str:
    """
    Generate realistic type labels from published SAPN management groups.

    Published route-length breakdown:
    Aluminium group:
        AAC       1,409 km
        AAAC        367 km
        Aluminium 3,069 km
    Steel group:
        SC/AC     7,587 km
        SC/GZ    27,769 km

    Copper maps to Copper.

    WARNING: this field directly reveals material and should normally be removed
    from the feature matrix when predicting material.
    """

    if material == "Aluminium":
        return str(
            weighted_choice(
                rng,
                ["AAC", "AAAC", "Aluminium"],
                [1409, 367, 3069],
            )
        )

    if material == "Steel":
        return str(
            weighted_choice(
                rng,
                ["SC/AC", "SC/GZ"],
                [7587, 27769],
            )
        )

    return "Copper"


# ---------------------------------------------------------------------------
# 9. Covering
# ---------------------------------------------------------------------------

def sample_covering(
    rng: np.random.Generator,
    installation_year: int,
) -> str:
    """
    Published SAPN evidence says covered conductor was only about 2.2% of the
    overhead network.

    We keep the relationship weak. Newer assets are slightly more likely to be
    covered, but the overall rate remains small.
    """

    if installation_year >= 1995:
        p_covered = 0.08
    elif installation_year >= 1980:
        p_covered = 0.035
    else:
        p_covered = 0.008

    return "Covered" if rng.random() < p_covered else "Bare"


# ---------------------------------------------------------------------------
# 10. Segment length
# ---------------------------------------------------------------------------

def sample_length_km(
    rng: np.random.Generator,
    location_group: str,
    voltage: str,
) -> float:
    """
    Generate segment length.

    The SAPN AMP provides total route length, not individual segment-length
    distributions. Therefore this is an explicit synthetic assumption.

    Rural/SWER segments are allowed to be longer on average than urban/LV
    segments, with substantial overlap.
    """

    if location_group == "Urban":
        median = 0.18
    elif location_group == "Regional":
        median = 0.35
    else:
        median = 0.55

    if voltage == "19 kV SWER":
        median *= 1.35
    elif voltage == "LV":
        median *= 0.70

    return round(
        clipped_lognormal(
            rng,
            median=median,
            sigma=0.65,
            low=0.03,
            high=5.0,
        ),
        3,
    )


# ---------------------------------------------------------------------------
# 11. Generate one complete dataset
# ---------------------------------------------------------------------------

def generate_dataset(config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    feeder_map = make_feeder_map(rng)

    rows = []

    for i in range(1, config.n_rows + 1):
        installation_year = sample_installation_year(rng)
        asset_age = config.snapshot_year - installation_year

        voltage = sample_voltage(rng, installation_year)
        location = sample_location_group(rng, voltage)
        corrosion = sample_corrosion_zone(rng, voltage, location)
        feeder = sample_feeder(rng, location, feeder_map)

        distance_to_sea = sample_distance_to_sea(
            rng,
            location_group=location,
            corrosion_zone=corrosion,
        )

        material = sample_material(
            rng,
            installation_year=installation_year,
            voltage=voltage,
            corrosion_zone=corrosion,
        )

        conductor_type = sample_conductor_type(rng, material)
        covering = sample_covering(rng, installation_year)
        length_km = sample_length_km(rng, location, voltage)

        rows.append(
            {
                "segment_id": f"SEG_{i:06d}",
                "feeder_id": feeder,
                "installation_year": installation_year,
                "asset_age": asset_age,
                "voltage_category": voltage,
                "location_group": location,
                "corrosion_zone": corrosion,
                "distance_to_sea_km": distance_to_sea,
                "material": material,
                "conductor_type": conductor_type,
                "covering": covering,
                "length_km": length_km,
                "data_status": "Synthetic",
                "record_version": "Complete reference",
            }
        )

    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# 12. Route-length calibration
# ---------------------------------------------------------------------------

def calibrate_route_length_marginals(
    df: pd.DataFrame,
    config: Config,
    iterations: int = 12,
) -> pd.DataFrame:
    """
    Calibrate synthetic segment lengths so aggregate route-length shares align
    with the published SAPN marginals.

    Why this is needed:
    SAPN's published percentages are mostly percentages of ROUTE LENGTH, not
    percentages of database rows. Because synthetic segment length varies by
    location and voltage, matching row counts alone is not sufficient.

    We iteratively rescale lengths within:
      1. voltage categories
      2. corrosion zones
      3. the three-class pilot material groups

    This is a synthetic calibration step. It preserves category membership and
    only adjusts generated segment lengths.

    The material target is the normalized non-ACSR, known-conductor route-length
    mix derived from SAPN Table 2:
        Steel ~74.0%, Copper ~15.9%, Aluminium ~10.1%.
    """

    out = df.copy()

    material_target = {
        "Steel": 0.740,
        "Copper": 0.159,
        "Aluminium": 0.101,
    }

    targets = [
        ("voltage_category", config.voltage_target),
        ("corrosion_zone", config.corrosion_target),
        ("material", material_target),
    ]

    for _ in range(iterations):
        total_length = out["length_km"].sum()

        for column, target in targets:
            current = (
                out.groupby(column, observed=False)["length_km"].sum()
                / total_length
            )

            factors = {}
            for category, desired_share in target.items():
                current_share = float(current.get(category, 0.0))
                if current_share > 0:
                    factors[category] = desired_share / current_share
                else:
                    factors[category] = 1.0

            out["length_km"] = (
                out["length_km"]
                * out[column].map(factors).fillna(1.0)
            )

            # Keep the overall total stable after each marginal adjustment.
            new_total = out["length_km"].sum()
            if new_total > 0:
                out["length_km"] *= total_length / new_total

    out["length_km"] = out["length_km"].clip(lower=0.01).round(4)
    return out


# ---------------------------------------------------------------------------
# 13. Validation helpers
# ---------------------------------------------------------------------------

def row_share(df: pd.DataFrame, column: str) -> Dict[str, float]:
    return (
        df[column]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
        .to_dict()
    )


def route_length_share(df: pd.DataFrame, column: str) -> Dict[str, float]:
    grouped = df.groupby(column, observed=False)["length_km"].sum()
    shares = grouped / grouped.sum()
    return shares.sort_index().round(4).to_dict()


def validate_dataset(df: pd.DataFrame, config: Config) -> Dict:
    """
    Produce transparent validation statistics.

    NOTE:
    Published SAPN percentages are mostly ROUTE-LENGTH shares, not record shares.
    Therefore both are reported.
    """

    summary = {
        "seed": config.seed,
        "rows": len(df),
        "snapshot_year": config.snapshot_year,
        "missing_cells": int(df.isna().sum().sum()),
        "mean_age": round(float(df["asset_age"].mean()), 2),
        "share_age_over_50": round(float((df["asset_age"] > 50).mean()), 4),
        "share_age_under_20": round(float((df["asset_age"] < 20).mean()), 4),
        "voltage_row_share": row_share(df, "voltage_category"),
        "voltage_route_length_share": route_length_share(df, "voltage_category"),
        "corrosion_row_share": row_share(df, "corrosion_zone"),
        "corrosion_route_length_share": route_length_share(df, "corrosion_zone"),
        "material_row_share": row_share(df, "material"),
        "material_route_length_share": route_length_share(df, "material"),
        "covering_row_share": row_share(df, "covering"),
        "total_route_length_km_in_synthetic_sample": round(float(df["length_km"].sum()), 2),
    }

    return summary


# ---------------------------------------------------------------------------
# 14. Assumptions / provenance log
# ---------------------------------------------------------------------------

def assumptions_table() -> pd.DataFrame:
    rows = [
        {
            "item": "snapshot_year",
            "value": "2014",
            "basis": "SAPN AMP publication/network snapshot",
            "status": "SAPN grounded",
        },
        {
            "item": "installation_year_profile",
            "value": "Strong 1955-1977 expansion; extra peaks 1955, 1956, 1958, 1966",
            "basis": "SAPN AMP age profile and text",
            "status": "SAPN grounded shape; exact probabilities synthetic",
        },
        {
            "item": "asset_age",
            "value": "2014 - installation_year",
            "basis": "Deterministic derivation",
            "status": "Derived",
        },
        {
            "item": "voltage_categories",
            "value": "66 kV, 33 kV, 19 kV SWER, 11/7.6 kV, LV",
            "basis": "SAPN AMP Table 1",
            "status": "SAPN grounded",
        },
        {
            "item": "voltage_by_era",
            "value": "Older pre-1950 biased to 33kV; 1955-1977 biased to SWER/11kV",
            "basis": "SAPN AMP text",
            "status": "SAPN grounded direction; exact probabilities synthetic",
        },
        {
            "item": "corrosion_marginal",
            "value": "Low ~53%, Severe ~35%, Very Severe ~11%",
            "basis": "SAPN AMP corrosion-zone section",
            "status": "SAPN grounded",
        },
        {
            "item": "material_target",
            "value": "Steel, Copper, Aluminium only",
            "basis": "Sprint 2 matrix + SAPN conductor groups",
            "status": "Project scope",
        },
        {
            "item": "material_marginal_reference",
            "value": "Steel ~74.0%, Copper ~15.9%, Aluminium ~10.1%",
            "basis": "Normalized known non-ACSR route lengths from SAPN Table 2",
            "status": "SAPN-derived reference, not a direct material-field distribution",
        },
        {
            "item": "ACSR",
            "value": "Excluded from current 3-class target",
            "basis": "Composite conductor group; cannot cleanly map to one material class",
            "status": "Project modelling decision",
        },
        {
            "item": "conductor_type",
            "value": "AAC, AAAC, Aluminium, Copper, SC/AC, SC/GZ",
            "basis": "SAPN AMP Table 2",
            "status": "SAPN grounded",
        },
        {
            "item": "conductor_type_as_predictor",
            "value": "Do not use directly for material prediction",
            "basis": "Would reveal target and create leakage",
            "status": "Modelling safeguard",
        },
        {
            "item": "covering",
            "value": "Bare / Covered; low overall covered share",
            "basis": "SAPN AMP says covered conductor ~2.2% of overhead network",
            "status": "SAPN grounded marginal; exact age relationship synthetic",
        },
        {
            "item": "location_group",
            "value": "Urban / Regional / Rural",
            "basis": "AMP uses rural/urban but does not publish population proportions",
            "status": "Synthetic assumption",
        },
        {
            "item": "distance_to_sea_km",
            "value": "Conditional on corrosion/location",
            "basis": "AMP supports saline/coastal corrosion relationship but no km thresholds",
            "status": "Synthetic enrichment assumption",
        },
        {
            "item": "segment_length",
            "value": "Lognormal, context dependent",
            "basis": "AMP gives total route length but no segment-length distribution",
            "status": "Synthetic assumption / needs external calibration",
        },
        {
            "item": "route_length_calibration",
            "value": "Iterative calibration to voltage, corrosion and 3-class material route-length marginals",
            "basis": "SAPN percentages are route-length shares, not row shares",
            "status": "Synthetic calibration step using SAPN aggregate targets",
        },
        {
            "item": "missing_values",
            "value": "None in complete reference dataset",
            "basis": "Masking is a separate validation step",
            "status": "Project design",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 15. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    summary: Dict,
    config: Config,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir / "overhead_conductors_complete_reference.csv",
        index=False,
    )

    assumptions_table().to_csv(
        output_dir / "generation_assumptions.csv",
        index=False,
    )

    with open(
        output_dir / "generation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    # A modelling-ready copy with the direct target-leakage field removed.
    model_df = df.drop(columns=["conductor_type"]).copy()
    model_df.to_csv(
        output_dir / "material_model_ready_data.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# 16. Main
# ---------------------------------------------------------------------------

def main():
    config = Config(
        n_rows=10_000,
        seed=219,
        snapshot_year=2014,
        output_dir=OUTPUT_DIR,
    )

    df = generate_dataset(config)
    df = calibrate_route_length_marginals(df, config)
    summary = validate_dataset(df, config)
    save_outputs(df, summary, config)

    print("\nSynthetic dataset generated successfully.\n")
    print(f"Rows: {len(df):,}")
    print(f"Missing cells: {df.isna().sum().sum()}")
    print(f"Mean age: {summary['mean_age']}")
    print(f"Share age > 50: {summary['share_age_over_50']}")
    print(f"Share age < 20: {summary['share_age_under_20']}")

    print("\nMaterial row shares:")
    for k, v in summary["material_row_share"].items():
        print(f"  {k}: {v:.4f}")

    print("\nVoltage route-length shares:")
    for k, v in summary["voltage_route_length_share"].items():
        print(f"  {k}: {v:.4f}")

    print("\nCorrosion route-length shares:")
    for k, v in summary["corrosion_route_length_share"].items():
        print(f"  {k}: {v:.4f}")

    print("\nFiles written to:")
    print(f"  {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
