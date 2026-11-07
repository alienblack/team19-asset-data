"""Schema-conformant synthetic frames for the pipeline tests (small and fast)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_YEAR = 2014
VOLTAGES = ["LV", "11/7.6 kV", "19 kV SWER", "33 kV", "66 kV"]
LOCATIONS = ["Urban", "Regional", "Rural"]
CORROSION = ["Low", "Severe", "Very Severe"]
MATERIALS = ["Aluminium", "Copper", "Steel"]
COVERINGS = ["Bare", "Covered"]


def make_conductor_frame(n: int = 300, seed: int = 0, n_feeders: int = 12) -> pd.DataFrame:
    """Complete records with simple dependencies between the model attributes."""
    rng = np.random.default_rng(seed)
    location = rng.choice(LOCATIONS, size=n, p=[0.25, 0.25, 0.5])
    distance = np.where(location == "Urban", rng.gamma(2.0, 6.0, n), np.where(location == "Regional", rng.gamma(3.0, 20.0, n), rng.gamma(4.0, 35.0, n)))
    distance = np.clip(distance, 0.0, 400.0).round(2)
    corrosion = np.where(distance < 15, rng.choice(CORROSION, n, p=[0.2, 0.4, 0.4]), np.where(distance < 60, rng.choice(CORROSION, n, p=[0.5, 0.4, 0.1]), rng.choice(CORROSION, n, p=[0.85, 0.13, 0.02])))
    voltage = np.where(location == "Urban", rng.choice(VOLTAGES, n, p=[0.5, 0.35, 0.0, 0.1, 0.05]), np.where(location == "Regional", rng.choice(VOLTAGES, n, p=[0.25, 0.35, 0.2, 0.15, 0.05]), rng.choice(VOLTAGES, n, p=[0.1, 0.2, 0.55, 0.1, 0.05])))
    year = np.where(voltage == "19 kV SWER", rng.normal(1962, 8, n), np.where(voltage == "LV", rng.normal(1975, 15, n), rng.normal(1968, 12, n)))
    year = np.clip(year, 1930, SNAPSHOT_YEAR).round()
    material = np.where(year < 1955, rng.choice(MATERIALS, n, p=[0.05, 0.6, 0.35]), np.where(voltage == "19 kV SWER", rng.choice(MATERIALS, n, p=[0.05, 0.05, 0.9]), rng.choice(MATERIALS, n, p=[0.45, 0.15, 0.4])))
    covering = np.where((voltage == "LV") & (year > 1990), rng.choice(COVERINGS, n, p=[0.3, 0.7]), rng.choice(COVERINGS, n, p=[0.97, 0.03]))
    feeders = [f"FDR_{i:03d}" for i in rng.integers(0, n_feeders, n)]
    return pd.DataFrame(
        {
            "segment_id": [f"SEG_{i:06d}" for i in range(1, n + 1)],
            "feeder_id": feeders,
            "installation_year": year.astype(float),
            "asset_age": (SNAPSHOT_YEAR - year).astype(float),
            "voltage_category": voltage.astype(object),
            "location_group": location.astype(object),
            "corrosion_zone": corrosion.astype(object),
            "distance_to_sea_km": distance.astype(float),
            "material": material.astype(object),
            "covering": covering.astype(object),
            "length_km": rng.gamma(2.0, 0.3, n).round(4),
            "data_status": "Synthetic",
            "record_version": "test",
        }
    )


def inject_missing(frame: pd.DataFrame, columns, rate: float = 0.2, seed: int = 1, link_age: bool = True) -> pd.DataFrame:
    """Hide ``rate`` of each listed column independently (MCAR); keep asset_age linked to year."""
    rng = np.random.default_rng(seed)
    out = frame.copy(deep=True)
    for column in columns:
        hide = rng.random(len(out)) < rate
        if pd.api.types.is_numeric_dtype(out[column]):
            out.loc[hide, column] = np.nan
        else:
            values = out[column].astype(object).to_numpy(copy=True)
            values[hide] = np.nan
            out[column] = pd.Series(values, index=out.index, dtype=object)
        if column == "installation_year" and link_age and "asset_age" in out.columns:
            out.loc[hide, "asset_age"] = np.nan
    return out


MODEL_COLUMNS = ["installation_year", "distance_to_sea_km", "voltage_category", "location_group", "corrosion_zone", "material", "covering"]


def make_masked_frame(n: int = 300, seed: int = 0, rate: float = 0.2) -> pd.DataFrame:
    return inject_missing(make_conductor_frame(n, seed), MODEL_COLUMNS, rate=rate, seed=seed + 1)


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write as the partner would deliver: integers without '.0', blanks for missing."""
    out = frame.copy()
    for column in ("installation_year", "asset_age"):
        if column in out.columns:
            out[column] = out[column].map(lambda v: "" if (not isinstance(v, str) and pd.isna(v)) else (str(int(v)) if isinstance(v, (int, float)) and float(v) == int(v) else str(v)))
    out.to_csv(path, index=False)
    return path
