"""
Generate a complete synthetic underground-cable asset dataset for imputation
and validation experiments.

The schema is aligned with the current project CSV and includes five synthetic
environmental proxy fields:
    groundwater_depth_m
    groundwater_exposure_score
    groundwater_indicator
    corrosion_exposure_score
    corrosion_indicator

IMPORTANT:
- All records are artificial.
- Environmental fields are synthetic proxies for method testing and are NOT
  measured SA Power Networks or South Australian environmental observations.
- The output is intentionally complete so that known values can later be
  masked and recovered during imputation validation.

Example:
    python generate_underground_cable_synthetic.py
    python generate_underground_cable_synthetic.py --rows 5000 --seed 19 \
        --output underground_cable_synthetic_data.csv
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_YEAR = 2026

# Approximate South Australian location mixture used only to create plausible
# synthetic spatial structure. Coordinates do not represent real cable assets.
LOCATION_CENTRES = [
    # name, latitude, longitude, weight, latitude SD, longitude SD
    ("Adelaide",     -34.9285, 138.6007, 0.82, 0.085, 0.110),
    ("MurrayBridge", -35.1199, 139.2735, 0.04, 0.070, 0.090),
    ("PortAugusta",  -32.4952, 137.7894, 0.03, 0.060, 0.080),
    ("Whyalla",      -33.0333, 137.5833, 0.03, 0.060, 0.080),
    ("PortLincoln",  -34.7200, 135.8600, 0.03, 0.060, 0.080),
    ("MountGambier", -37.8284, 140.7804, 0.05, 0.070, 0.090),
]


def stable_rng(asset_id: str, label: str, seed: int) -> random.Random:
    """Return a deterministic per-row RNG."""
    raw = f"{asset_id}|{label}|{seed}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()[:8]
    return random.Random(int.from_bytes(digest, "big"))


def weighted_choice(rng: np.random.Generator, values, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return rng.choice(values, p=probabilities)


def generate_installation_year(rng: np.random.Generator) -> int:
    """
    Generate a plausible installation year from 1950 to 2026.

    A beta distribution gives more recent assets greater probability while
    retaining a meaningful older population.
    """
    fraction = rng.beta(2.35, 1.45)
    year = int(round(1950 + fraction * (CURRENT_YEAR - 1950)))
    return int(np.clip(year, 1950, CURRENT_YEAR))


def generate_insulation(rng: np.random.Generator, installation_year: int) -> str:
    """Older assets are more likely to use PILC; newer assets are mostly XLPE."""
    if installation_year < 1975:
        p_pilc = 0.88
    elif installation_year < 1990:
        p_pilc = 0.62
    elif installation_year < 2000:
        p_pilc = 0.28
    elif installation_year < 2005:
        p_pilc = 0.10
    else:
        p_pilc = 0.02
    return "PILC" if rng.random() < p_pilc else "XLPE"


def generate_material(
    rng: np.random.Generator,
    insulation_type: str,
    installation_year: int,
) -> str:
    """PILC/older assets favour copper; XLPE/newer assets favour aluminium."""
    if insulation_type == "PILC":
        p_copper = 0.88
    else:
        p_copper = 0.18

    if installation_year < 1980:
        p_copper = min(0.95, p_copper + 0.08)
    elif installation_year >= 2010:
        p_copper = max(0.08, p_copper - 0.06)

    return "Copper" if rng.random() < p_copper else "Aluminium"


def generate_conductor_size(
    rng: np.random.Generator,
    material: str,
    rated_voltage_kv: int,
) -> int:
    """
    Generate conductor size using material-specific categories similar to the
    project dataset.
    """
    if material == "Aluminium":
        values = [50, 95, 185, 300, 630]
        probs =  [0.02, 0.44, 0.02, 0.34, 0.18]
    else:
        values = [70, 95, 120, 185, 240, 630]
        probs =  [0.15, 0.20, 0.24, 0.19, 0.16, 0.06]

    size = int(weighted_choice(rng, values, probs))

    # In the current project dataset, 33 kV cables favour medium/large sizes
    # and 630 mm2 is mainly represented at 11 kV.
    if rated_voltage_kv == 33 and size == 630:
        size = int(weighted_choice(rng, [95, 120, 240, 300], [0.20, 0.20, 0.25, 0.35]))

    return size


def generate_location(rng: np.random.Generator) -> tuple[float, float]:
    weights = [x[3] for x in LOCATION_CENTRES]
    idx = int(rng.choice(len(LOCATION_CENTRES), p=np.array(weights) / np.sum(weights)))
    _, lat_c, lon_c, _, lat_sd, lon_sd = LOCATION_CENTRES[idx]

    lat = rng.normal(lat_c, lat_sd)
    lon = rng.normal(lon_c, lon_sd)

    # Broad bounds chosen only to keep synthetic points in a plausible SA range.
    lat = float(np.clip(lat, -38.05, -32.30))
    lon = float(np.clip(lon, 135.70, 141.00))
    return lat, lon


def route_linestring(
    lat: float,
    lon: float,
    length_m: float,
    rng: np.random.Generator,
) -> str:
    """
    Create a simple two-point WKT route centred on the asset coordinate.

    This is not a surveyed cable route; it is only synthetic geometry.
    """
    bearing = rng.uniform(0, 2 * math.pi)
    half_length = length_m / 2.0

    # Approximate metre-to-degree conversions.
    dlat = (half_length * math.cos(bearing)) / 111_320.0
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    dlon = (half_length * math.sin(bearing)) / (111_320.0 * cos_lat)

    lat1, lon1 = lat - dlat, lon - dlon
    lat2, lon2 = lat + dlat, lon + dlon

    return (
        f"LINESTRING ({lon1:.6f} {lat1:.6f}, "
        f"{lon2:.6f} {lat2:.6f})"
    )


def environmental_fields(
    asset_id: str,
    lat: float,
    lon: float,
    installation_method: str,
    seed: int,
) -> tuple[float, float, str, float, str]:
    """
    Generate synthetic groundwater and corrosion proxy fields.

    Groundwater depth has spatial structure plus controlled noise.
    Shallower groundwater produces a higher groundwater exposure score.
    Corrosion exposure is related mainly to groundwater exposure, with a
    modest installation-method effect and additional noise.

    These are method-testing proxies, not real environmental observations.
    """
    # Normalise coordinates against broad SA-like synthetic bounds.
    lat_s = (lat - (-38.05)) / (-32.30 - (-38.05))
    lon_s = (lon - 135.70) / (141.00 - 135.70)

    r_gw = stable_rng(asset_id, "groundwater_depth", seed)
    spatial_component = (
        0.45 * math.sin(2 * math.pi * lat_s)
        + 0.35 * math.cos(2 * math.pi * lon_s)
        + 0.20 * math.sin(2 * math.pi * (lat_s + lon_s))
    )

    groundwater_depth = 6.0 - 2.8 * spatial_component + r_gw.gauss(0, 1.15)
    groundwater_depth = max(0.5, min(15.0, groundwater_depth))

    r_gw_score = stable_rng(asset_id, "groundwater_score", seed)
    groundwater_score = 1.0 - (groundwater_depth - 0.5) / (15.0 - 0.5)
    groundwater_score += r_gw_score.gauss(0, 0.045)
    groundwater_score = max(0.0, min(1.0, groundwater_score))

    if groundwater_score < 0.40:
        groundwater_indicator = "Low"
    elif groundwater_score < 0.70:
        groundwater_indicator = "Medium"
    else:
        groundwater_indicator = "High"

    r_corr = stable_rng(asset_id, "corrosion", seed)
    corrosion_score = 0.12 + 0.72 * groundwater_score

    if installation_method == "Direct Buried":
        corrosion_score += 0.08
    elif installation_method == "Duct/Conduit":
        corrosion_score += 0.02

    corrosion_score += r_corr.gauss(0, 0.075)
    corrosion_score = max(0.0, min(1.0, corrosion_score))

    if corrosion_score < 0.40:
        corrosion_indicator = "Low"
    elif corrosion_score < 0.70:
        corrosion_indicator = "Medium"
    else:
        corrosion_indicator = "High"

    return (
        round(groundwater_depth, 2),
        round(groundwater_score, 3),
        groundwater_indicator,
        round(corrosion_score, 3),
        corrosion_indicator,
    )


def build_cable_type_raw(
    conductor_size_mm2: int,
    core_count: int,
    conductor_material: str,
    insulation_type: str,
    rated_voltage_kv: int,
) -> str:
    material_code = "AL" if conductor_material == "Aluminium" else "CU"
    insulation_label = "XLPE/HDPE" if insulation_type == "XLPE" else "PILC"

    return (
        f"{conductor_size_mm2}MM2 {core_count}C {material_code} "
        f"{insulation_label} {rated_voltage_kv}KV UNDERGROUND CABLE"
    )


def generate_dataset(rows: int = 5000, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Create non-uniform feeder probabilities so feeder populations are not
    # perfectly balanced.
    feeder_ids = [f"FDR_{i:03d}" for i in range(1, 61)]
    feeder_weights = rng.dirichlet(np.full(60, 4.0))

    records = []

    for i in range(1, rows + 1):
        asset_id = f"UGC_{i:06d}"
        feeder_id = str(rng.choice(feeder_ids, p=feeder_weights))

        rated_voltage_kv = int(rng.choice([11, 33], p=[0.77, 0.23]))

        installation_year = generate_installation_year(rng)
        asset_age_years = CURRENT_YEAR - installation_year

        insulation_type = generate_insulation(rng, installation_year)
        conductor_material = generate_material(
            rng, insulation_type, installation_year
        )
        conductor_size_mm2 = generate_conductor_size(
            rng, conductor_material, rated_voltage_kv
        )

        p_one_core = 0.58 if rated_voltage_kv == 11 else 0.74
        core_count = 1 if rng.random() < p_one_core else 3

        # Right-skewed length distribution; clipping prevents unrealistic tails.
        cable_length_m = float(
            np.clip(rng.lognormal(mean=math.log(165), sigma=0.65), 15, 1800)
        )
        cable_length_m = round(cable_length_m, 2)

        installation_method = str(
            rng.choice(["Duct/Conduit", "Direct Buried"], p=[0.77, 0.23])
        )

        lat, lon = generate_location(rng)

        # Longer cable sections tend to have more joints.
        joint_lambda = max(0.05, cable_length_m / 185.0)
        joint_count = int(min(rng.poisson(joint_lambda), 12))

        route_geometry_wkt = route_linestring(
            lat, lon, cable_length_m, rng
        )

        cable_type_raw = build_cable_type_raw(
            conductor_size_mm2,
            core_count,
            conductor_material,
            insulation_type,
            rated_voltage_kv,
        )

        (
            groundwater_depth_m,
            groundwater_exposure_score,
            groundwater_indicator,
            corrosion_exposure_score,
            corrosion_indicator,
        ) = environmental_fields(
            asset_id,
            lat,
            lon,
            installation_method,
            seed,
        )

        records.append(
            {
                "asset_id": asset_id,
                "feeder_id": feeder_id,
                "rated_voltage_kv": rated_voltage_kv,
                "installation_year": installation_year,
                "asset_age_years": asset_age_years,
                "cable_length_m": cable_length_m,
                "insulation_type": insulation_type,
                "conductor_material": conductor_material,
                "conductor_size_mm2": conductor_size_mm2,
                "core_count": core_count,
                "installation_method": installation_method,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "joint_count": joint_count,
                "route_geometry_wkt": route_geometry_wkt,
                "cable_type_raw": cable_type_raw,
                "groundwater_depth_m": groundwater_depth_m,
                "groundwater_exposure_score": groundwater_exposure_score,
                "groundwater_indicator": groundwater_indicator,
                "corrosion_exposure_score": corrosion_exposure_score,
                "corrosion_indicator": corrosion_indicator,
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic underground-cable dataset."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5000,
        help="Number of synthetic cable records to generate (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=19,
        help="Random seed for reproducibility (default: 19).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Underground_Cable_Synthetic_data.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError("--rows must be greater than zero.")

    df = generate_dataset(rows=args.rows, seed=args.seed)
    df.to_csv(args.output, index=False)

    print(f"Created: {args.output}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nKey distributions:")
    print("Voltage (kV):")
    print(df["rated_voltage_kv"].value_counts().sort_index())
    print("\nInsulation:")
    print(df["insulation_type"].value_counts())
    print("\nGroundwater indicator:")
    print(df["groundwater_indicator"].value_counts())
    print("\nCorrosion indicator:")
    print(df["corrosion_indicator"].value_counts())


if __name__ == "__main__":
    main()
