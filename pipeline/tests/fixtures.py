"""Tiny self-contained synthetic fixture for imputation tests.

Not SAPN data. Relationships are deliberately simple so tests can check
*behaviour* (replay, leakage, order) rather than accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["installation_year", "distance_to_sea_km"]
CATEGORICAL = [
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "material",
    "covering",
]
MODEL_COLUMNS = NUMERIC + CATEGORICAL

CONSTRAINTS = {
    "installation_year": {"data_type": "integer", "min": 1930, "max": 2014},
    "distance_to_sea_km": {"data_type": "float", "min": 0.0, "max": 350.0},
}


def make_complete_population(n: int, seed: int) -> pd.DataFrame:
    """Complete synthetic conductor records with simple dependencies."""
    rng = np.random.default_rng(seed)

    location = rng.choice(["urban", "rural"], size=n, p=[0.4, 0.6])
    # voltage depends on location (so a fully observed column is predictable)
    voltage = np.where(
        location == "urban",
        rng.choice(["LV", "HV"], size=n, p=[0.7, 0.3]),
        rng.choice(["HV", "SWER"], size=n, p=[0.5, 0.5]),
    )
    distance = np.clip(
        np.where(location == "urban",
                 rng.normal(15, 10, size=n),
                 rng.normal(120, 60, size=n)),
        0.0, 350.0,
    )
    corrosion = np.where(distance < 10, "high",
                         np.where(distance < 50, "medium", "low"))
    year = np.clip(
        np.where(voltage == "SWER",
                 rng.normal(1965, 10, size=n),
                 rng.normal(1985, 15, size=n)),
        1930, 2014,
    ).round()
    material = np.where(
        year < 1960, "Cu",
        np.where(voltage == "SWER", "Steel",
                 rng.choice(["Al", "Steel"], size=n, p=[0.7, 0.3])),
    )
    covering = np.where(
        (voltage == "LV") & (year > 1990),
        rng.choice(["covered", "bare"], size=n, p=[0.8, 0.2]),
        "bare",
    )

    frame = pd.DataFrame(
        {
            "segment_id": [f"S{i:05d}" for i in range(n)],
            "installation_year": year.astype(float),
            "distance_to_sea_km": distance.astype(float),
            "voltage_category": voltage.astype(object),
            "location_group": location.astype(object),
            "corrosion_zone": corrosion.astype(object),
            "material": material.astype(object),
            "covering": covering.astype(object),
            "data_status": "synthetic",
        }
    )
    return frame


def inject_missing(
    frame: pd.DataFrame,
    rate: float,
    seed: int,
    keep_complete: tuple[str, ...] = ("voltage_category",),
) -> pd.DataFrame:
    """MCAR missingness in model columns except ``keep_complete``."""
    rng = np.random.default_rng(seed)
    out = frame.copy(deep=True)
    for column in MODEL_COLUMNS:
        if column in keep_complete:
            continue
        hide = rng.random(len(out)) < rate
        out.loc[hide, column] = np.nan
    return out


def make_fit_frame(n: int = 300, seed: int = 1) -> pd.DataFrame:
    return inject_missing(make_complete_population(n, seed), rate=0.2, seed=seed + 100)


def make_new_batch(n: int = 40, seed: int = 7) -> pd.DataFrame:
    """Unseen records: heavier missingness, one entirely missing column,
    one record with every model column missing, voltage missing although it
    was complete during fitting."""
    batch = inject_missing(
        make_complete_population(n, seed), rate=0.35, seed=seed + 100,
        keep_complete=(),
    )
    batch["covering"] = np.nan               # entirely missing column
    batch.loc[batch.index[0], MODEL_COLUMNS] = np.nan   # fully missing record
    batch.index = [f"R{i}" for i in range(n)]            # non-default index
    return batch
